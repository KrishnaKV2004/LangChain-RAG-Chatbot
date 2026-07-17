"""Loader base class and shared metadata extraction.

Every format-specific loader derives from :class:`BaseDocumentLoader`, which
handles the concerns common to all formats:

* file validation (exists, readable, non-empty),
* base metadata every chunk must carry (filename, source, document type,
  creation date, ingestion timestamp),
* uniform error wrapping into :class:`DocumentLoadError`.

Subclasses only implement ``_load()`` and enrich metadata with format-specific
fields (page number, section title, sheet name, ...).

Note: metadata values must stay ChromaDB-compatible (str / int / float / bool).
Missing values are omitted entirely rather than stored as ``None``.
"""

import abc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

from app.utils.exceptions import DocumentLoadError
from app.utils.logging import get_logger

logger = get_logger(__name__)


def file_created_at(path: Path) -> str:
    """Best-effort file creation date as an ISO-8601 string.

    macOS exposes true birth time via ``st_birthtime``; on other platforms we
    fall back to the modification time, which is the closest portable proxy.
    """
    stat = path.stat()
    timestamp = getattr(stat, "st_birthtime", None) or stat.st_mtime
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


class BaseDocumentLoader(abc.ABC):
    """Abstract loader: one subclass per document format."""

    #: File extensions (lowercase, with dot) this loader accepts.
    suffixes: Tuple[str, ...] = ()
    #: Value stored in the ``doc_type`` metadata field.
    doc_type: str = "unknown"

    def load(self, path: Path) -> List[Document]:
        """Load ``path`` into one or more LangChain Documents.

        Wraps all parser failures in :class:`DocumentLoadError` so ingestion
        can skip a corrupt file and continue with the rest of the batch.
        """
        path = Path(path)
        if not path.is_file():
            raise DocumentLoadError(f"File not found: {path}")
        if path.stat().st_size == 0:
            raise DocumentLoadError(f"File is empty: {path}")

        try:
            documents = self._load(path)
        except DocumentLoadError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalize any parser error
            raise DocumentLoadError(
                f"Failed to parse {path.name}: {exc}",
                details={"path": str(path), "loader": type(self).__name__},
            ) from exc

        # Stamp base metadata on every produced document. Format-specific
        # fields set by the subclass (page, section_title...) are preserved.
        base = self._base_metadata(path)
        for doc in documents:
            doc.metadata = {**base, **doc.metadata}

        # Drop documents whose extracted text is pure whitespace.
        documents = [d for d in documents if d.page_content.strip()]
        if not documents:
            raise DocumentLoadError(f"No extractable text in {path.name}")

        logger.debug("file_loaded", file=path.name, documents=len(documents))
        return documents

    @abc.abstractmethod
    def _load(self, path: Path) -> List[Document]:
        """Parse the file. Subclasses set format-specific metadata only."""

    def _base_metadata(self, path: Path) -> Dict[str, Any]:
        """Metadata common to every chunk regardless of format."""
        return {
            "filename": path.name,
            "source": str(path.resolve()),
            "doc_type": self.doc_type,
            "created_at": file_created_at(path),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }


def require(module_name: str, pip_name: str) -> Any:
    """Import an optional parser dependency with an actionable error.

    Keeps heavyweight parsers as lazy imports so the application starts even
    when a format's library isn't installed — only using that format requires
    it.
    """
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise DocumentLoadError(
            f"Missing dependency '{pip_name}' — install it with: pip install {pip_name}"
        ) from exc
