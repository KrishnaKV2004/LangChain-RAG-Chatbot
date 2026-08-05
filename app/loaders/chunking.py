"""Document chunking.

Two strategies, chosen automatically per document:

* **Markdown** — a header-aware split first (so every chunk knows the heading
  it lives under → ``section_title`` metadata), then a size-bounded recursive
  split within each section.
* **Everything else** — recursive character splitting on paragraph/sentence
  boundaries.

Every chunk receives:

* ``chunk_index`` — position within its source document,
* ``chunk_id``    — deterministic UUID5 of (source, index, content hash), so
  re-ingesting an unchanged file produces identical IDs and ChromaDB upserts
  stay idempotent (this is what makes "rebuild index" safe).
"""

import hashlib
import re
import uuid
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config.settings import ChunkingSettings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Headers the markdown splitter tracks; the deepest one present becomes the
# chunk's section title.
_MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


class DocumentChunker:
    """Splits loaded Documents into embedding-sized chunks with rich metadata."""

    def __init__(self, settings: ChunkingSettings) -> None:
        self._settings = settings
        self._recursive = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            # Prefer paragraph, then line, then sentence boundaries.
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._markdown = MarkdownHeaderTextSplitter(
            headers_to_split_on=_MD_HEADERS,
            strip_headers=False,  # keep headings inside the chunk text
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split documents, stamp chunk metadata, drop noise fragments."""
        chunks: List[Document] = []
        for document in documents:
            if document.metadata.get("doc_type") == "markdown":
                chunks.extend(self._split_markdown(document))
            else:
                chunks.extend(self._recursive.split_documents([document]))

        # Filter fragments too short to carry meaning (page headers, bullets).
        chunks = [
            c for c in chunks if len(c.page_content.strip()) >= self._settings.min_chunk_chars
        ]

        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_id"] = self._chunk_id(chunk, index)
            chunk.page_content = self._with_source_header(chunk)

        logger.debug("chunking_complete", input_docs=len(documents), chunks=len(chunks))
        return chunks

    def _split_markdown(self, document: Document) -> List[Document]:
        """Header-aware markdown split; deepest heading → section_title."""
        sections = self._markdown.split_text(document.page_content)
        result: List[Document] = []
        for section in sections:
            # Merge source-file metadata with the splitter's header metadata.
            headers = section.metadata
            title = headers.get("h3") or headers.get("h2") or headers.get("h1")
            merged = dict(document.metadata)
            if title:
                merged["section_title"] = title
            section_doc = Document(page_content=section.page_content, metadata=merged)
            result.extend(self._recursive.split_documents([section_doc]))
        return result

    @staticmethod
    def _with_source_header(chunk: Document) -> str:
        """Prefix the chunk with the document it came from.

        Identifiers frequently live only in the FILENAME — "Ticket-1160.pdf"
        never repeats "1160" in its body, it uses an internal Zoho id. Because
        only ``page_content`` is embedded, a question like "source and
        destination for ticket 1160" had nothing to match and retrieved a
        different ticket. Embedding the filename fixes that for every document
        whose name carries meaning (tickets, invoices, BOLs, dated reports).
        """
        meta = chunk.metadata
        filename = meta.get("filename")
        if not filename:
            return chunk.page_content
        header = f"[Source: {filename}"
        # The stem alone ("Ticket-1160" → "Ticket 1160") so a spaced or
        # punctuation-free phrasing of the identifier matches too.
        stem = re.sub(r"\.[A-Za-z0-9]+$", "", str(filename))
        spaced = re.sub(r"[-_]+", " ", stem)
        if spaced != stem:
            header += f" | {spaced}"
        if meta.get("page") is not None:
            header += f" | page {meta['page']}"
        if meta.get("section_title"):
            header += f" | {meta['section_title']}"
        return f"{header}]\n{chunk.page_content}"

    @staticmethod
    def _chunk_id(chunk: Document, index: int) -> str:
        """Deterministic chunk identity: same content in same place → same ID."""
        content_hash = hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()
        source = chunk.metadata.get("source", "unknown")
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{index}:{content_hash}"))
