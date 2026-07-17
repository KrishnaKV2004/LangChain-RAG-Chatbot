"""PDF loader built on ``pypdf``.

Produces one Document per page so page numbers survive chunking and can be
cited in answers ("Shipping Manual.pdf, page 12"). If the PDF declares an
intrinsic creation date in its XMP/DocInfo metadata, it overrides the
filesystem date.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from app.loaders.base import BaseDocumentLoader


class PDFLoader(BaseDocumentLoader):
    """Page-aware PDF text extraction."""

    suffixes = (".pdf",)
    doc_type = "pdf"

    def _load(self, path: Path) -> List[Document]:
        from pypdf import PdfReader  # bundled dependency, cheap import

        reader = PdfReader(str(path))
        created = self._document_creation_date(reader)

        documents: List[Document] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue  # skip image-only / blank pages
            metadata = {"page": page_number, "total_pages": len(reader.pages)}
            if created:
                metadata["created_at"] = created
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    @staticmethod
    def _document_creation_date(reader: "object") -> Optional[str]:
        """Extract the PDF's own creation date, if present and parseable."""
        try:
            raw = reader.metadata.creation_date  # type: ignore[attr-defined]
            if isinstance(raw, datetime):
                if raw.tzinfo is None:
                    raw = raw.replace(tzinfo=timezone.utc)
                return raw.isoformat()
        except Exception:  # noqa: BLE001 — malformed DocInfo is common
            pass
        return None
