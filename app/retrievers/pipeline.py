"""The retrieval pipeline: vector search → threshold → MMR → rerank → compress.

Funnel design (each stage narrows the candidate set):

1. **Vector search** — fetch ``fetch_k`` candidates with cosine relevance
   scores, optionally constrained by a ChromaDB metadata filter.
2. **Score threshold** — discard candidates below ``score_threshold`` so a
   query about an unknown topic yields *nothing* instead of noise (critical
   for the "say so when information is unavailable" requirement).
3. **MMR** (optional) — re-select for diversity so five near-duplicate chunks
   of the same paragraph don't crowd out complementary context.
4. **Cross-encoder rerank** (optional) — joint (query, chunk) scoring; keeps
   the best ``rerank_top_k``.
5. **Compression** (optional) — trim query-irrelevant sentences.

Only stage 1 touches the vector database; everything downstream is pure
in-process computation, which keeps the pipeline easy to unit test.
"""

import re
from typing import Any, Dict, List, Optional

from app.config.settings import RetrievalSettings
from app.database.chroma import ChromaManager
from app.retrievers.compression import SentenceCompressor
from app.retrievers.models import RetrievalResult, RetrievedChunk
from app.retrievers.reranker import BaseReranker
from app.utils.exceptions import RetrievalError
from app.utils.logging import get_logger
from app.utils.timing import Timer

logger = get_logger(__name__)

#: When a query names one document, return up to this many of its chunks so the
#: whole record is in context. Bounds the prompt for a large file.
_SCOPED_MAX_CHUNKS = 25


class RetrievalPipeline:
    """Composable retrieval over the internal document collection."""

    def __init__(
        self,
        manager: ChromaManager,
        settings: RetrievalSettings,
        reranker: Optional[BaseReranker] = None,
        compressor: Optional[SentenceCompressor] = None,
    ) -> None:
        self._manager = manager
        self._settings = settings
        # Collaborators are injected; the factory in app.agents decides which
        # concrete ones to build based on configuration.
        self._reranker = reranker if settings.rerank_enabled else None
        self._compressor = compressor if settings.compression_enabled else None

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """Run the full funnel and return only highly relevant chunks."""
        settings = self._settings
        final_k = top_k or settings.top_k
        result = RetrievalResult()

        # An explicit document reference ("ticket 1160") is an EXACT lookup.
        # Embeddings can't tell 1160 from 1161 — the vectors are nearly
        # identical — so scope the search to that file instead of hoping
        # similarity picks the right one.
        scoped_filter = self._document_filter(query) if metadata_filter is None else None
        if scoped_filter is not None:
            metadata_filter = scoped_filter
            # The named document IS the answer scope, so hand the LLM all of it
            # rather than the top few chunks — the detail asked for (a drop
            # address on page 1) is often not the chunk that ranks highest for
            # the question's wording.
            final_k = max(final_k, _SCOPED_MAX_CHUNKS)

        with Timer() as timer:
            try:
                scored = self._search_with_scores(query, metadata_filter)
            except Exception as exc:  # noqa: BLE001 — normalize backend errors
                raise RetrievalError(f"Vector search failed: {exc}") from exc

            result.candidates = len(scored)

            # ---- Stage 2: relevance floor -------------------------------- #
            scored = [
                (doc, score) for doc, score in scored if score >= settings.score_threshold
            ]
            result.after_threshold = len(scored)

            # ---- Stage 3: MMR diversity re-selection --------------------- #
            if settings.search_type == "mmr" and scored:
                scored = self._apply_mmr(query, scored, metadata_filter)

            # ---- Stage 4: cross-encoder rerank --------------------------- #
            if self._reranker is not None and scored:
                similarity_by_id = {
                    doc.metadata.get("chunk_id"): score for doc, score in scored
                }
                ranked = self._reranker.rerank(query, [doc for doc, _ in scored])
                # Rerank orders the chunks; only trim when we are NOT answering
                # from one explicitly-named document.
                ranked = ranked[: max(settings.rerank_top_k, final_k)]
                chunks = [
                    RetrievedChunk(
                        document=doc,
                        similarity_score=similarity_by_id.get(doc.metadata.get("chunk_id")),
                        rerank_score=score,
                    )
                    for doc, score in ranked
                ]
            else:
                chunks = [
                    RetrievedChunk(document=doc, similarity_score=score)
                    for doc, score in scored
                ]
            result.after_rerank = len(chunks)

            # ---- Final cut + optional compression ------------------------ #
            chunks = chunks[:final_k]
            if self._compressor is not None and chunks:
                compressed = self._compressor.compress(query, [c.document for c in chunks])
                for chunk, document in zip(chunks, compressed):
                    chunk.document = document

            result.chunks = chunks

        result.latency_ms = timer.elapsed_ms
        logger.info(
            "retrieval_complete",
            candidates=result.candidates,
            after_threshold=result.after_threshold,
            after_rerank=result.after_rerank,
            returned=len(result.chunks),
            latency_ms=round(result.latency_ms, 1),
        )
        return result

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _document_filter(self, query: str) -> Optional[Dict[str, Any]]:
        """Scope the search to a document the query names explicitly.

        Matches an identifier in the query (``ticket 1160``, ``Ticket-1160``,
        ``invoice #4471``) against the indexed filenames. Returns ``None`` when
        the query names nothing, or names something not in the index — in which
        case normal semantic search runs and the answer honestly reports what it
        found.
        """
        numbers = set(re.findall(r"\d{3,}", query))
        if not numbers:
            return None
        try:
            filenames = list(self._manager.stats().get("documents_by_file", {}))
        except Exception as exc:  # noqa: BLE001 — never let this break retrieval
            logger.debug("document_filter_stats_failed", error=str(exc))
            return None

        matches = sorted(
            {name for name in filenames if any(number in name for number in numbers)}
        )
        if not matches:
            return None
        logger.info("retrieval_scoped_to_documents", files=matches)
        # Chroma wants $in for a set, a bare value for one.
        return {"filename": matches[0] if len(matches) == 1 else {"$in": matches}}

    def _search_with_scores(
        self, query: str, metadata_filter: Optional[Dict[str, Any]]
    ) -> List[tuple]:
        """Stage 1: fetch candidates with cosine relevance scores in [0, 1]."""
        store = self._manager.vector_store()
        return store.similarity_search_with_relevance_scores(
            query, k=self._settings.fetch_k, filter=metadata_filter
        )

    def _apply_mmr(
        self,
        query: str,
        scored: List[tuple],
        metadata_filter: Optional[Dict[str, Any]],
    ) -> List[tuple]:
        """Stage 3: swap the candidate order/set for an MMR-diverse selection.

        MMR search doesn't return scores, so results are joined back to the
        similarity scores already collected in stage 1 via ``chunk_id``.
        Chunks MMR surfaces that fell below the threshold in stage 1 are
        dropped (the relevance floor always wins over diversity).
        """
        store = self._manager.vector_store()
        score_by_id = {doc.metadata.get("chunk_id"): score for doc, score in scored}

        diverse = store.max_marginal_relevance_search(
            query,
            k=min(len(scored), self._settings.fetch_k),
            fetch_k=self._settings.fetch_k,
            lambda_mult=self._settings.mmr_lambda,
            filter=metadata_filter,
        )
        return [
            (doc, score_by_id[doc.metadata.get("chunk_id")])
            for doc in diverse
            if doc.metadata.get("chunk_id") in score_by_id
        ]
