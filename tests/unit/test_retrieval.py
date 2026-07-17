"""Unit tests for the retrieval pipeline (real ChromaDB, fake embeddings,
fake reranker — fully offline and deterministic)."""

from pathlib import Path
from typing import List, Sequence, Tuple

import pytest
from langchain_core.documents import Document

from app.config.settings import RetrievalSettings, Settings
from app.database.chroma import ChromaManager
from app.retrievers.compression import SentenceCompressor
from app.retrievers.models import RetrievedChunk
from app.retrievers.pipeline import RetrievalPipeline
from app.retrievers.reranker import BaseReranker
from tests.conftest import FakeEmbeddings


class KeywordReranker(BaseReranker):
    """Deterministic reranker: score = shared word count with the query."""

    def rerank(
        self, query: str, documents: Sequence[Document]
    ) -> List[Tuple[Document, float]]:
        query_words = set(query.lower().split())
        scored = [
            (doc, float(len(query_words & set(doc.page_content.lower().split()))))
            for doc in documents
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)


def make_chunk(text: str, chunk_id: str, **metadata: object) -> Document:
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "filename": f"{chunk_id}.txt", "doc_type": "text", **metadata},
    )


@pytest.fixture()
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ChromaManager:
    monkeypatch.setenv("VECTOR_STORE__PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    settings = Settings(_env_file=None)
    manager = ChromaManager(settings, FakeEmbeddings())
    manager.add_chunks(
        [
            make_chunk("Air cargo cargo cargo capacity varies by aircraft.", "cargo1"),
            make_chunk("Cargo pallets and cargo containers are unit load devices.", "cargo2"),
            make_chunk("Customs clearance requires an invoice at customs.", "customs1", page=4),
            make_chunk("Weather in Dubai is hot; weather forecasts change.", "weather1"),
        ]
    )
    return manager


def make_pipeline(manager: ChromaManager, **overrides: object) -> RetrievalPipeline:
    defaults = dict(
        search_type="similarity",
        top_k=3,
        fetch_k=10,
        score_threshold=0.30,
        rerank_enabled=False,
        compression_enabled=False,
    )
    defaults.update(overrides)
    reranker = overrides.pop("reranker", KeywordReranker()) if "reranker" in overrides else KeywordReranker()
    settings = RetrievalSettings(**defaults)
    return RetrievalPipeline(manager, settings, reranker=reranker)


class TestRetrievalPipeline:
    def test_similarity_returns_relevant_chunks_with_scores(
        self, manager: ChromaManager
    ) -> None:
        result = make_pipeline(manager).retrieve("cargo aircraft")
        assert not result.is_empty
        assert "cargo" in result.chunks[0].document.page_content.lower()
        assert all(c.similarity_score is not None for c in result.chunks)
        assert result.latency_ms > 0

    def test_threshold_filters_irrelevant_topics(self, manager: ChromaManager) -> None:
        # A query about an unindexed topic must return nothing, not noise.
        result = make_pipeline(manager, score_threshold=0.9).retrieve(
            "quantum blockchain cooking recipes"
        )
        assert result.is_empty
        assert result.candidates > 0  # candidates existed but were rejected

    def test_metadata_filter_restricts_results(self, manager: ChromaManager) -> None:
        result = make_pipeline(manager).retrieve(
            "customs invoice", metadata_filter={"filename": "customs1.txt"}
        )
        assert not result.is_empty
        assert all(
            c.document.metadata["filename"] == "customs1.txt" for c in result.chunks
        )

    def test_top_k_is_respected(self, manager: ChromaManager) -> None:
        result = make_pipeline(manager, top_k=1).retrieve("cargo")
        assert len(result.chunks) == 1

    def test_rerank_orders_by_cross_score_and_records_it(
        self, manager: ChromaManager
    ) -> None:
        result = make_pipeline(manager, rerank_enabled=True, rerank_top_k=3).retrieve(
            "customs clearance invoice"
        )
        assert result.chunks[0].document.metadata["chunk_id"] == "customs1"
        assert result.chunks[0].rerank_score is not None
        assert result.chunks[0].similarity_score is not None  # both preserved

    def test_mmr_mode_returns_scored_diverse_results(self, manager: ChromaManager) -> None:
        result = make_pipeline(manager, search_type="mmr", top_k=3).retrieve("cargo")
        assert not result.is_empty
        # MMR results keep their stage-1 similarity scores via chunk_id join.
        assert all(c.similarity_score is not None for c in result.chunks)

    def test_citation_view_exposes_provenance(self, manager: ChromaManager) -> None:
        result = make_pipeline(manager).retrieve("customs invoice")
        citation = result.chunks[0].citation
        assert citation["filename"] == "customs1.txt"
        assert citation["page"] == 4


class TestSentenceCompressor:
    def test_removes_irrelevant_sentences(self, fake_embeddings: FakeEmbeddings) -> None:
        # FakeEmbeddings' constant component gives unrelated sentences a
        # baseline cosine of ~0.71; 0.8 separates keyword matches from it.
        compressor = SentenceCompressor(fake_embeddings, similarity_threshold=0.8)
        doc = Document(
            page_content=(
                "Cargo loading follows the manifest. "
                "The cafeteria menu changes weekly. "
                "Cargo weight must be balanced."
            )
        )
        [compressed] = compressor.compress("cargo handling", [doc])
        assert "cafeteria" not in compressed.page_content
        assert "Cargo" in compressed.page_content

    def test_fails_open_when_everything_would_be_removed(
        self, fake_embeddings: FakeEmbeddings
    ) -> None:
        compressor = SentenceCompressor(fake_embeddings, similarity_threshold=0.99)
        doc = Document(page_content="One sentence. Two sentence. Three sentence.")
        [compressed] = compressor.compress("unrelated query", [doc])
        assert compressed.page_content == doc.page_content
