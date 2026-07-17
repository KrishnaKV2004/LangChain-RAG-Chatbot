"""Embedding-based contextual compression (LLM-free).

After the final chunk set is chosen, each chunk can still contain sentences
irrelevant to the query (boilerplate, neighboring topics). The compressor
splits a chunk into sentences, embeds them, and keeps only sentences whose
cosine similarity to the query clears a threshold — shrinking the prompt
without an extra LLM call.

Guard rails: if compression would remove everything, the original chunk is
kept — losing context is worse than sending a few extra tokens.
"""

import math
import re
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.utils.logging import get_logger

logger = get_logger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class SentenceCompressor:
    """Keeps only query-relevant sentences within each retrieved chunk."""

    def __init__(self, embeddings: Embeddings, similarity_threshold: float = 0.25) -> None:
        self._embeddings = embeddings
        self._threshold = similarity_threshold

    def compress(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return documents

        query_vector = self._embeddings.embed_query(query)
        compressed: List[Document] = []

        for document in documents:
            sentences = [
                s.strip() for s in _SENTENCE_SPLIT.split(document.page_content) if s.strip()
            ]
            if len(sentences) <= 2:
                compressed.append(document)  # nothing meaningful to trim
                continue

            vectors = self._embeddings.embed_documents(sentences)
            kept = [
                sentence
                for sentence, vector in zip(sentences, vectors)
                if _cosine(query_vector, vector) >= self._threshold
            ]
            # Fail open: an over-aggressive threshold must not delete context.
            if not kept:
                compressed.append(document)
                continue
            compressed.append(
                Document(page_content=" ".join(kept), metadata=dict(document.metadata))
            )

        original = sum(len(d.page_content) for d in documents)
        reduced = sum(len(d.page_content) for d in compressed)
        logger.debug("compression_done", chars_before=original, chars_after=reduced)
        return compressed
