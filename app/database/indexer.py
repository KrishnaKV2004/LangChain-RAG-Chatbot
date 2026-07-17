"""Indexing service — glues ingestion (files → chunks) to storage (ChromaDB).

This is the single entry point used by:

* initial bulk loading (CLI / startup),
* the ``/upload`` endpoint (index one uploaded file),
* the ``/reindex`` endpoint (full rebuild from the documents directory).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from langchain_core.documents import Document

from app.database.chroma import ChromaManager
from app.loaders.ingestion import IngestionService
from app.utils.logging import get_logger
from app.utils.timing import Timer

logger = get_logger(__name__)


@dataclass
class IndexReport:
    """Summary of an indexing run, safe to serialize straight into JSON."""

    files_processed: List[str] = field(default_factory=list)
    files_skipped: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)
    chunks_indexed: int = 0
    duration_ms: float = 0.0
    rebuilt: bool = False


class IndexingService:
    """End-to-end indexing: parse, chunk, embed and persist documents."""

    def __init__(
        self,
        ingestion: IngestionService,
        manager: ChromaManager,
        documents_dir: Path,
        chunk_processor: Optional[Callable[[List[Document]], List[Document]]] = None,
    ) -> None:
        self._ingestion = ingestion
        self._manager = manager
        self._documents_dir = Path(documents_dir)
        # Hook applied to chunks before storage — the security layer uses it
        # to stamp sensitivity metadata (Layer 2) at ingestion time.
        self._process = chunk_processor or (lambda chunks: chunks)

    def index_file(self, path: Path) -> IndexReport:
        """Index a single file, replacing any previously indexed version.

        Deleting by source first means a re-uploaded file whose content
        shrank doesn't leave orphaned chunks from the old version behind.
        """
        path = Path(path)
        report = IndexReport()
        with Timer() as timer:
            self._manager.delete_by_source(path.name)
            chunks = self._process(self._ingestion.ingest_file(path))
            report.chunks_indexed = self._manager.add_chunks(chunks)
        report.files_processed.append(path.name)
        report.duration_ms = timer.elapsed_ms
        return report

    def index_directory(self, directory: Optional[Path] = None) -> IndexReport:
        """Index every supported file in a directory (incremental upsert)."""
        target = Path(directory) if directory else self._documents_dir
        report = IndexReport()
        with Timer() as timer:
            result = self._ingestion.ingest_directory(target)
            report.chunks_indexed = self._manager.add_chunks(self._process(result.chunks))
        report.files_processed = result.files_processed
        report.files_skipped = result.files_skipped
        report.errors = result.errors
        report.duration_ms = timer.elapsed_ms
        return report

    def rebuild(self) -> IndexReport:
        """Drop the documents collection and re-index from scratch.

        The reset happens *before* re-ingestion so deleted source files
        disappear from the index — a plain upsert could never remove them.
        """
        logger.info("rebuild_started", directory=str(self._documents_dir))
        self._manager.reset_collection()
        report = self.index_directory()
        report.rebuilt = True
        logger.info(
            "rebuild_complete",
            files=len(report.files_processed),
            chunks=report.chunks_indexed,
            latency_ms=round(report.duration_ms, 1),
        )
        return report
