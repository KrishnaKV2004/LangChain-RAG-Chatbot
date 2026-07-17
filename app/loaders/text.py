"""Loaders for text-native formats: TXT, Markdown, HTML and JSON."""

import json
from pathlib import Path
from typing import Any, List

from langchain_core.documents import Document

from app.loaders.base import BaseDocumentLoader, require
from app.utils.exceptions import DocumentLoadError


def _read_text(path: Path) -> str:
    """Read a file as UTF-8, tolerating stray bytes from legacy encodings."""
    return path.read_text(encoding="utf-8", errors="replace")


class TextLoader(BaseDocumentLoader):
    """Plain-text loader — a single Document; the chunker does the splitting."""

    suffixes = (".txt",)
    doc_type = "text"

    def _load(self, path: Path) -> List[Document]:
        return [Document(page_content=_read_text(path))]


class MarkdownLoader(BaseDocumentLoader):
    """Markdown loader.

    Returns the raw markdown as one Document; the chunker detects
    ``doc_type == "markdown"`` and applies a header-aware splitter so each
    chunk inherits its section title from the nearest heading.
    """

    suffixes = (".md", ".markdown")
    doc_type = "markdown"

    def _load(self, path: Path) -> List[Document]:
        return [Document(page_content=_read_text(path))]


class HTMLLoader(BaseDocumentLoader):
    """HTML loader: strips markup/scripts, keeps <title> as section_title."""

    suffixes = (".html", ".htm")
    doc_type = "html"

    def _load(self, path: Path) -> List[Document]:
        bs4 = require("bs4", "beautifulsoup4")
        soup = bs4.BeautifulSoup(_read_text(path), "lxml")

        # Remove non-content elements before extracting text.
        for tag in soup(["script", "style", "noscript", "template"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        # Collapse the blank-line noise HTML extraction produces.
        lines = [line.strip() for line in text.splitlines()]
        cleaned = "\n".join(line for line in lines if line)

        metadata = {}
        if soup.title and soup.title.string:
            metadata["section_title"] = soup.title.string.strip()
        return [Document(page_content=cleaned, metadata=metadata)]


class JSONLoader(BaseDocumentLoader):
    """JSON loader: flattens nested structures into 'key.path: value' lines.

    A top-level array yields one Document per item (typical for FAQ dumps or
    record exports); a top-level object yields a single Document.
    """

    suffixes = (".json",)
    doc_type = "json"

    def _load(self, path: Path) -> List[Document]:
        try:
            data = json.loads(_read_text(path))
        except json.JSONDecodeError as exc:
            raise DocumentLoadError(f"Invalid JSON in {path.name}: {exc}") from exc

        if isinstance(data, list):
            documents = []
            for index, item in enumerate(data, start=1):
                text = self._flatten(item)
                if text.strip():
                    documents.append(
                        Document(page_content=text, metadata={"item_index": index})
                    )
            return documents
        return [Document(page_content=self._flatten(data))]

    @classmethod
    def _flatten(cls, value: Any, prefix: str = "") -> str:
        """Recursively render JSON as readable 'dotted.path: value' lines."""
        if isinstance(value, dict):
            parts = [
                cls._flatten(child, f"{prefix}.{key}" if prefix else str(key))
                for key, child in value.items()
            ]
            return "\n".join(p for p in parts if p)
        if isinstance(value, list):
            parts = [
                cls._flatten(child, f"{prefix}[{i}]") for i, child in enumerate(value)
            ]
            return "\n".join(p for p in parts if p)
        return f"{prefix}: {value}" if prefix else str(value)
