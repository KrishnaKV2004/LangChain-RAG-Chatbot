"""Layer 3 — permission validation.

Compares the requesting user's clearance (from their role) against each
chunk's sensitivity level (from Layer 2) and partitions retrieved chunks into
allowed and denied sets.

The denied *texts* are kept (in memory, for this request only) because
Layer 4 uses them to verify the final answer doesn't contain leaked fragments
of documents the user was never allowed to see.
"""

from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document

from app.security.models import SensitivityLevel, UserContext
from app.security.sensitivity import SENSITIVITY_KEY, SensitivityClassifier
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PermissionDecision:
    """Result of filtering retrieved chunks against a user's clearance."""

    allowed: List[Document] = field(default_factory=list)
    #: Texts of denied chunks — consumed by Layer 4's leak check.
    denied_texts: List[str] = field(default_factory=list)
    denied_count: int = 0


class PermissionValidator:
    """Enforces clearance >= sensitivity for every chunk."""

    def __init__(self, classifier: SensitivityClassifier) -> None:
        # The classifier is the fallback for chunks indexed before tagging
        # existed (or added through a path that skipped Layer 2).
        self._classifier = classifier

    def filter_chunks(
        self, user: UserContext, documents: List[Document]
    ) -> PermissionDecision:
        """Split chunks into what this user may and may not see."""
        decision = PermissionDecision()
        for document in documents:
            level = self._level_of(document)
            if user.clearance >= level:
                decision.allowed.append(document)
            else:
                decision.denied_texts.append(document.page_content)
                decision.denied_count += 1

        if decision.denied_count:
            logger.warning(
                "chunks_denied_by_permission",
                user=user.user_id,
                role=user.role.value,
                denied=decision.denied_count,
                allowed=len(decision.allowed),
            )
        return decision

    def _level_of(self, document: Document) -> SensitivityLevel:
        """Read the stored sensitivity; classify on the fly if absent."""
        stored = str(document.metadata.get(SENSITIVITY_KEY, "")).upper()
        if stored in SensitivityLevel.__members__:
            return SensitivityLevel[stored]
        return self._classifier.classify(document)
