"""Extension → loader registry.

New formats are added by writing a :class:`BaseDocumentLoader` subclass and
registering it here — nothing else in the pipeline changes (open/closed
principle).
"""

from pathlib import Path
from typing import Dict, List, Type

from app.loaders.base import BaseDocumentLoader
from app.loaders.office import DocxLoader, PptxLoader
from app.loaders.pdf import PDFLoader
from app.loaders.tabular import CSVLoader, ExcelLoader
from app.loaders.text import HTMLLoader, JSONLoader, MarkdownLoader, TextLoader
from app.utils.exceptions import DocumentLoadError

_LOADER_CLASSES: List[Type[BaseDocumentLoader]] = [
    PDFLoader,
    DocxLoader,
    PptxLoader,
    CSVLoader,
    ExcelLoader,
    TextLoader,
    MarkdownLoader,
    HTMLLoader,
    JSONLoader,
]


class LoaderRegistry:
    """Maps file extensions to loader instances (one instance per class)."""

    def __init__(self) -> None:
        self._by_suffix: Dict[str, BaseDocumentLoader] = {}
        for loader_class in _LOADER_CLASSES:
            self.register(loader_class())

    def register(self, loader: BaseDocumentLoader) -> None:
        """Register (or override) a loader for its declared suffixes."""
        for suffix in loader.suffixes:
            self._by_suffix[suffix.lower()] = loader

    def supported_suffixes(self) -> List[str]:
        return sorted(self._by_suffix)

    def supports(self, path: Path) -> bool:
        return Path(path).suffix.lower() in self._by_suffix

    def loader_for(self, path: Path) -> BaseDocumentLoader:
        suffix = Path(path).suffix.lower()
        loader = self._by_suffix.get(suffix)
        if loader is None:
            raise DocumentLoadError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {', '.join(self.supported_suffixes())}"
            )
        return loader


#: Default shared registry — modules that need custom loaders build their own.
default_registry = LoaderRegistry()
