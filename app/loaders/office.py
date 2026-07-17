"""Loaders for Microsoft Office formats: Word (.docx) and PowerPoint (.pptx).

Both parsers are lazy-imported (see :func:`app.loaders.base.require`) so the
application runs without them until a file of that type is actually ingested.
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from app.loaders.base import BaseDocumentLoader, require


class DocxLoader(BaseDocumentLoader):
    """Word loader that splits on headings to preserve section titles.

    Paragraphs styled as "Heading N" open a new section; every emitted
    Document carries the heading it belongs to as ``section_title``, which
    later appears in citations.  Tables are serialized as tab-separated rows
    and appended to the section they appear in.
    """

    suffixes = (".docx",)
    doc_type = "docx"

    def _load(self, path: Path) -> List[Document]:
        docx = require("docx", "python-docx")
        word_doc = docx.Document(str(path))

        created = self._core_created(word_doc)
        sections: List[Document] = []
        current_title: Optional[str] = None
        buffer: List[str] = []

        def flush() -> None:
            """Emit the accumulated paragraphs as one section Document."""
            text = "\n".join(buffer).strip()
            if not text:
                return
            metadata = {}
            if current_title:
                metadata["section_title"] = current_title
            if created:
                metadata["created_at"] = created
            sections.append(Document(page_content=text, metadata=metadata))

        for paragraph in word_doc.paragraphs:
            style = (paragraph.style.name or "") if paragraph.style else ""
            if style.startswith("Heading"):
                flush()
                buffer = []
                current_title = paragraph.text.strip() or current_title
                if current_title:
                    buffer.append(current_title)  # keep heading in the text
            elif paragraph.text.strip():
                buffer.append(paragraph.text)

        # Tables are not interleaved with paragraphs by python-docx, so they
        # are appended after the prose as their own block.
        for table in word_doc.tables:
            rows = ["\t".join(cell.text.strip() for cell in row.cells) for row in table.rows]
            table_text = "\n".join(r for r in rows if r.strip())
            if table_text:
                buffer.append(table_text)

        flush()
        return sections

    @staticmethod
    def _core_created(word_doc: "object") -> Optional[str]:
        """Creation date from the document's core properties, if set."""
        try:
            created = word_doc.core_properties.created  # type: ignore[attr-defined]
            return created.isoformat() if created else None
        except Exception:  # noqa: BLE001
            return None


class PptxLoader(BaseDocumentLoader):
    """PowerPoint loader: one Document per slide.

    The slide number maps onto the ``page`` metadata field so PPTX citations
    read like PDF ones ("Deck.pptx, page 4"); the slide title becomes
    ``section_title``. Speaker notes are included — SOPs often live there.
    """

    suffixes = (".pptx",)
    doc_type = "pptx"

    def _load(self, path: Path) -> List[Document]:
        pptx = require("pptx", "python-pptx")
        presentation = pptx.Presentation(str(path))

        documents: List[Document] = []
        for slide_number, slide in enumerate(presentation.slides, start=1):
            parts: List[str] = []
            title: Optional[str] = None

            for shape in slide.shapes:
                if not getattr(shape, "has_text_frame", False):
                    continue
                text = shape.text_frame.text.strip()
                if not text:
                    continue
                # The first title-placeholder shape names the slide.
                if title is None and shape == getattr(slide.shapes, "title", None):
                    title = text
                parts.append(text)

            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    parts.append(f"[Speaker notes] {notes}")

            if not parts:
                continue
            metadata = {"page": slide_number}
            if title:
                metadata["section_title"] = title
            documents.append(Document(page_content="\n".join(parts), metadata=metadata))
        return documents
