"""Cross-encoder reranking.

Bi-encoder (vector) search is fast but lossy: query and document are embedded
independently. A cross-encoder reads the (query, chunk) pair *jointly* and
scores actual relevance, which reliably pushes near-miss chunks out of the
final context. It runs only on the small candidate set, so the extra latency
is bounded.

The abstract base exists so tests (and future providers like Cohere Rerank)
can swap implementations without touching the pipeline.
"""

import abc
from typing import List, Sequence, Tuple

from langchain_core.documents import Document

from app.utils.exceptions import ConfigurationError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class BaseReranker(abc.ABC):
    """Scores (query, document) pairs; higher means more relevant."""

    @abc.abstractmethod
    def rerank(
        self, query: str, documents: Sequence[Document]
    ) -> List[Tuple[Document, float]]:
        """Return (document, score) pairs sorted by descending relevance."""


class CrossEncoderReranker(BaseReranker):
    """Local cross-encoder via sentence-transformers (no API calls)."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ConfigurationError(
                "RETRIEVAL__RERANK_ENABLED=true requires the "
                "'sentence-transformers' package: pip install sentence-transformers"
            ) from exc

        # Loaded once at construction; ~90 MB download on first ever use,
        # cached under ~/.cache afterwards.
        self._model = CrossEncoder(model_name)
        self._model_name = model_name
        logger.info("reranker_initialized", model=model_name)

    def rerank(
        self, query: str, documents: Sequence[Document]
    ) -> List[Tuple[Document, float]]:
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda item: item[1], reverse=True)
        return [(doc, float(score)) for doc, score in ranked]
