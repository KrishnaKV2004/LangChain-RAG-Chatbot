"""ChromaDB vector-store management.

One :class:`ChromaManager` owns the persistent client and its collections:

* ``documents`` — the internal knowledge base (permanent),
* ``web_cache`` — transient web content (TTL-governed, see ``app.cache``),
* anything else — future extensions get a collection through the same API.

The manager exposes LangChain ``Chroma`` vector stores for retrieval while
keeping raw-collection operations (stats, reset, source listing) in one place.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config.settings import Settings
from app.utils.exceptions import VectorStoreError
from app.utils.logging import get_logger
from app.utils.timing import Timer

logger = get_logger(__name__)

# ChromaDB rejects payloads above ~5461 items; stay safely below.
_MAX_BATCH = 1000


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce metadata to the scalar types ChromaDB accepts.

    ``None`` values are dropped; everything non-scalar is stringified.
    """
    clean: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        clean[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return clean


class ChromaManager:
    """Owns the persistent ChromaDB client and collection lifecycle."""

    def __init__(self, settings: Settings, embeddings: Embeddings) -> None:
        import chromadb

        self._settings = settings
        self._embeddings = embeddings
        persist_dir = Path(settings.vector_store.persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=chromadb.Settings(anonymized_telemetry=False, allow_reset=False),
        )
        self._stores: Dict[str, Chroma] = {}
        logger.info("chroma_initialized", path=str(persist_dir))

    # ------------------------------------------------------------------ #
    # Collection / vector-store access
    # ------------------------------------------------------------------ #

    @property
    def documents_collection_name(self) -> str:
        return self._settings.vector_store.documents_collection

    def vector_store(self, collection_name: Optional[str] = None) -> Chroma:
        """LangChain vector store bound to a collection (cached per name)."""
        name = collection_name or self.documents_collection_name
        if name not in self._stores:
            self._stores[name] = Chroma(
                client=self._client,
                collection_name=name,
                embedding_function=self._embeddings,
                collection_metadata={
                    "hnsw:space": self._settings.vector_store.distance_metric
                },
            )
        return self._stores[name]

    # ------------------------------------------------------------------ #
    # Write path
    # ------------------------------------------------------------------ #

    def add_chunks(
        self, chunks: List[Document], collection_name: Optional[str] = None
    ) -> int:
        """Upsert chunks into a collection using their deterministic chunk IDs.

        Because IDs are content-derived (see ``DocumentChunker``), re-adding
        an unchanged file is a no-op upsert — safe to call repeatedly.
        """
        if not chunks:
            return 0

        name = collection_name or self.documents_collection_name
        store = self.vector_store(name)
        with Timer() as timer:
            try:
                for start in range(0, len(chunks), _MAX_BATCH):
                    batch = chunks[start : start + _MAX_BATCH]
                    ids = [c.metadata["chunk_id"] for c in batch]
                    documents = [
                        Document(
                            page_content=c.page_content,
                            metadata=sanitize_metadata(c.metadata),
                        )
                        for c in batch
                    ]
                    store.add_documents(documents=documents, ids=ids)
            except Exception as exc:  # noqa: BLE001 — normalize backend errors
                raise VectorStoreError(f"Failed to add chunks to '{name}': {exc}") from exc

        logger.info(
            "chunks_added",
            collection=name,
            chunks=len(chunks),
            latency_ms=round(timer.elapsed_ms, 1),
        )
        return len(chunks)

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #

    def reset_collection(self, collection_name: Optional[str] = None) -> None:
        """Drop and recreate a collection (used by full index rebuilds)."""
        name = collection_name or self.documents_collection_name
        try:
            self._client.delete_collection(name)
        except Exception:  # noqa: BLE001 — collection may not exist yet
            pass
        self._stores.pop(name, None)  # force re-creation on next access
        self.vector_store(name)  # recreate immediately so stats() sees it
        logger.info("collection_reset", collection=name)

    def delete_by_source(self, filename: str, collection_name: Optional[str] = None) -> int:
        """Remove every chunk originating from ``filename``."""
        name = collection_name or self.documents_collection_name
        collection = self._client.get_or_create_collection(name)
        existing = collection.get(where={"filename": filename}, include=[])
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
        logger.info("source_deleted", collection=name, file=filename, chunks=len(existing["ids"]))
        return len(existing["ids"])

    # ------------------------------------------------------------------ #
    # Introspection (powers /health, the UI sidebar and /cache endpoints)
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict[str, Any]:
        """Chunk counts per collection plus a per-file breakdown of documents."""
        collections = {c.name: c for c in self._client.list_collections()}
        per_collection = {name: c.count() for name, c in collections.items()}

        sources: Dict[str, int] = {}
        documents = collections.get(self.documents_collection_name)
        if documents is not None and documents.count() > 0:
            metadatas = documents.get(include=["metadatas"])["metadatas"] or []
            sources = dict(Counter(m.get("filename", "unknown") for m in metadatas))

        return {
            "persist_directory": str(self._settings.vector_store.persist_directory),
            "collections": per_collection,
            "documents_by_file": sources,
            "total_chunks": sum(per_collection.values()),
        }
