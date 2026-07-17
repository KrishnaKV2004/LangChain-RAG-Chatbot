"""Layer 2 — sensitive-document detection.

Assigns every chunk a :class:`SensitivityLevel` from three signals, strongest
first:

1. **Explicit metadata** — a ``confidential`` flag stamped at ingestion (or a
   pre-existing ``sensitivity`` value) always wins.
2. **Filename conventions** — ``confidential``, ``internal_only``, ``hr_``,
   ``salary`` ... in the filename.
3. **Content markers** — classification banners like "CONFIDENTIAL",
   "internal use only", "do not distribute" inside the text.

Documents with no signal default to INTERNAL: this is a company knowledge
base, so "unknown" must not mean "public" (fail closed). Truly public
documents can be marked ``sensitivity: public`` in metadata at ingestion.
"""

import re
from typing import List, Pattern, Tuple

from langchain_core.documents import Document

from app.security.models import SensitivityLevel
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Filename fragments that imply confidentiality.
_FILENAME_CONFIDENTIAL: Pattern[str] = re.compile(
    r"confidential|secret|restricted|internal[-_ ]only|do[-_ ]not[-_ ]share|"
    r"\bhr[-_]|salary|salaries|payroll|compensation|nda\b|m&a|acquisition",
    re.I,
)

#: In-content classification banners. Word-boundary anchored so a sentence
#: *about* confidentiality ("mark the AWB as confidential cargo") in running
#: prose is weighed by banner style, not topic.
_CONTENT_CONFIDENTIAL: List[Pattern[str]] = [
    re.compile(r"\bstrictly\s+confidential\b", re.I),
    re.compile(r"^\s*confidential\b", re.I | re.M),           # banner line
    re.compile(r"\binternal\s+use\s+only\b", re.I),
    re.compile(r"\bdo\s+not\s+(distribute|share|forward)\b", re.I),
    re.compile(r"\bproprietary\s+(and|&)\s+confidential\b", re.I),
    re.compile(r"\bclassification\s*:\s*(confidential|secret|restricted)\b", re.I),
    re.compile(r"\btrade\s+secret\b", re.I),
]

#: Metadata key set by ingestion / this classifier.
SENSITIVITY_KEY = "sensitivity"
CONFIDENTIAL_FLAG_KEY = "confidential"


class SensitivityClassifier:
    """Classifies chunks and stamps their sensitivity into metadata."""

    def classify(self, document: Document) -> SensitivityLevel:
        """Determine a chunk's sensitivity level (metadata > filename > content)."""
        meta = document.metadata

        # 1) Explicit signals always win.
        if str(meta.get(CONFIDENTIAL_FLAG_KEY, "")).lower() in ("true", "1", "yes"):
            return SensitivityLevel.CONFIDENTIAL
        explicit = str(meta.get(SENSITIVITY_KEY, "")).lower()
        if explicit in SensitivityLevel.__members__.keys() or explicit in (
            "public", "internal", "confidential",
        ):
            return SensitivityLevel[explicit.upper()]

        # 2) Filename conventions.
        if _FILENAME_CONFIDENTIAL.search(meta.get("filename", "")):
            return SensitivityLevel.CONFIDENTIAL

        # 3) Content banners.
        for pattern in _CONTENT_CONFIDENTIAL:
            if pattern.search(document.page_content):
                return SensitivityLevel.CONFIDENTIAL

        # Default for company documents: internal, never public (fail closed).
        return SensitivityLevel.INTERNAL

    def tag(self, documents: List[Document]) -> List[Document]:
        """Stamp ``sensitivity`` metadata on chunks (called at ingestion).

        Storing the level in ChromaDB means Layer 3 can later filter at
        *retrieval* time via metadata filters, not just post-hoc.
        """
        confidential = 0
        for document in documents:
            level = self.classify(document)
            document.metadata[SENSITIVITY_KEY] = level.name.lower()
            if level is SensitivityLevel.CONFIDENTIAL:
                confidential += 1
        if confidential:
            logger.info("sensitive_chunks_tagged", confidential=confidential, total=len(documents))
        return documents
