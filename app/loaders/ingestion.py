"""Ingestion service: files → loaded documents → metadata-rich chunks.

This module is deliberately storage-agnostic — it produces chunks; the
database layer (``app.database``) persists them.  That separation lets the
same service power the initial bulk load, the ``/upload`` endpoint, and full
index rebuilds.

Error philosophy: one corrupt file must never abort a batch.  Per-file
failures are collected into the result object and logged; the rest of the
directory continues to ingest.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

from app.config.settings import ChunkingSettings
from app.loaders.chunking import DocumentChunker
from app.loaders.registry import LoaderRegistry, default_registry
from app.utils.exceptions import DocumentLoadError, IngestionError
from app.utils.logging import get_logger
from app.utils.timing import Timer

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """Outcome of an ingestion run — everything the caller needs to report."""

    chunks: List[Document] = field(default_factory=list)
    files_processed: List[str] = field(default_factory=list)
    files_skipped: List[str] = field(default_factory=list)
    #: filename → error message for files that failed to parse.
    errors: Dict[str, str] = field(default_factory=dict)
    duration_ms: float = 0.0

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


class IngestionService:
    """Coordinates loading and chunking for single files and directories."""

    def __init__(
        self,
        chunking_settings: ChunkingSettings,
        registry: Optional[LoaderRegistry] = None,
    ) -> None:
        self._registry = registry or default_registry
        self._chunker = DocumentChunker(chunking_settings)

    def ingest_file(self, path: Path) -> List[Document]:
        """Load and chunk a single file.

        Raises:
            DocumentLoadError: if the file cannot be parsed or is unsupported.
        """
        path = Path(path)
        loader = self._registry.loader_for(path)
        with Timer() as timer:
            documents = loader.load(path)
            chunks = self._chunker.chunk(documents)
        logger.info(
            "file_ingested",
            file=path.name,
            documents=len(documents),
            chunks=len(chunks),
            latency_ms=round(timer.elapsed_ms, 1),
        )
        return chunks

    def ingest_directory(self, directory: Path, recursive: bool = True) -> IngestionResult:
        """Ingest every supported file under ``directory``.

        Unsupported extensions are skipped (recorded, not errors); parse
        failures are captured per-file so the batch always completes.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise IngestionError(f"Not a directory: {directory}")

        result = IngestionResult()
        pattern = "**/*" if recursive else "*"

        with Timer() as timer:
            for path in sorted(directory.glob(pattern)):
                if not path.is_file() or path.name.startswith("."):
                    continue
                if not self._registry.supports(path):
                    result.files_skipped.append(path.name)
                    continue
                try:
                    result.chunks.extend(self.ingest_file(path))
                    result.files_processed.append(path.name)
                except DocumentLoadError as exc:
                    result.errors[path.name] = exc.message
                    logger.warning("file_ingest_failed", file=path.name, error=exc.message)

        result.duration_ms = timer.elapsed_ms
        logger.info(
            "directory_ingested",
            directory=str(directory),
            files_processed=len(result.files_processed),
            files_skipped=len(result.files_skipped),
            errors=len(result.errors),
            chunks=result.chunk_count,
            latency_ms=round(result.duration_ms, 1),
        )
        return result
