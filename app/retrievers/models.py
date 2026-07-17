"""Data structures returned by the retrieval pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


@dataclass
class RetrievedChunk:
    """One retrieved chunk with its relevance evidence.

    Attributes:
        document: The chunk (text + metadata) as stored in ChromaDB.
        similarity_score: Cosine relevance in [0, 1] from vector search.
        rerank_score: Cross-encoder score (unbounded logit); ``None`` when
            reranking is disabled.
    """

    document: Document
    similarity_score: Optional[float] = None
    rerank_score: Optional[float] = None

    @property
    def citation(self) -> Dict[str, Any]:
        """Citation-ready view of this chunk's provenance."""
        meta = self.document.metadata
        citation: Dict[str, Any] = {"filename": meta.get("filename", "unknown")}
        for key in ("page", "section_title", "rows", "doc_type"):
            if key in meta:
                citation[key] = meta[key]
        return citation


@dataclass
class RetrievalResult:
    """Full outcome of one retrieval run, including funnel counts for logs."""

    chunks: List[RetrievedChunk] = field(default_factory=list)
    #: How many candidates each stage saw — for observability/tuning.
    candidates: int = 0
    after_threshold: int = 0
    after_rerank: int = 0
    latency_ms: float = 0.0

    @property
    def documents(self) -> List[Document]:
        return [c.document for c in self.chunks]

    @property
    def is_empty(self) -> bool:
        return not self.chunks
