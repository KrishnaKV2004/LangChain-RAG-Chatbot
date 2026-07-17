"""SecurityGuard — the facade the rest of the system talks to.

Three checkpoints map onto the request lifecycle:

* :meth:`validate_query`      — before any retrieval (Layer 1 + input caps),
* :meth:`filter_context`      — after retrieval, before the LLM
                                (Layers 2 + 3 + indirect-injection scan),
* :meth:`validate_response`   — after generation, before the user
                                (Layer 4 + Layer 5 redaction).

Every layer can be toggled via ``SecuritySettings``; a layer that *crashes*
blocks the request (fail closed) — a security exception must never become an
open door.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from langchain_core.documents import Document

from app.config.settings import SecuritySettings
from app.security.injection import InjectionDetector
from app.security.models import (
    REFUSAL_MESSAGE,
    SecurityVerdict,
    ThreatType,
    UserContext,
)
from app.security.permissions import PermissionValidator
from app.security.pii import PIIDetector
from app.security.response_scan import ResponseScanner
from app.security.sensitivity import SensitivityClassifier
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ContextDecision:
    """Outcome of the retrieval-context checkpoint."""

    documents: List[Document] = field(default_factory=list)
    denied_texts: List[str] = field(default_factory=list)
    dropped_injected: int = 0
    denied_by_permission: int = 0


@dataclass
class ResponseDecision:
    """Outcome of the response checkpoint."""

    answer: str = ""
    blocked: bool = False
    pii_redactions: int = 0


class SecurityGuard:
    """Coordinates the five security layers around the RAG pipeline."""

    def __init__(
        self,
        settings: SecuritySettings,
        injection: Optional[InjectionDetector] = None,
        classifier: Optional[SensitivityClassifier] = None,
        permissions: Optional[PermissionValidator] = None,
        pii: Optional[PIIDetector] = None,
        scanner: Optional[ResponseScanner] = None,
        protected_markers: Sequence[str] = (),
    ) -> None:
        self._settings = settings
        self._injection = injection or InjectionDetector()
        self._classifier = classifier or SensitivityClassifier()
        self._permissions = permissions or PermissionValidator(self._classifier)
        self._pii = pii or PIIDetector()
        self._scanner = scanner or ResponseScanner(self._pii, protected_markers)

    # ------------------------------------------------------------------ #
    # Checkpoint 1: the incoming query
    # ------------------------------------------------------------------ #

    def validate_query(self, query: str) -> SecurityVerdict:
        """Layer 1 + structural input limits. Blocked = refuse immediately."""
        try:
            if len(query) > self._settings.max_query_chars:
                logger.warning("query_too_long", length=len(query))
                return SecurityVerdict(
                    allowed=False,
                    layer="input_validation",
                    threat_type=ThreatType.OVERSIZED_INPUT,
                    reasons=[f"query exceeds {self._settings.max_query_chars} chars"],
                )
            if self._settings.prompt_injection_detection:
                return self._injection.scan_query(query)
            return SecurityVerdict.ok("prompt_injection")
        except Exception as exc:  # noqa: BLE001 — fail closed
            logger.error("security_layer_error", layer="validate_query", error=str(exc))
            return SecurityVerdict(
                allowed=False, layer="input_validation", reasons=["internal error"]
            )

    # ------------------------------------------------------------------ #
    # Checkpoint 2: retrieved context (documents AND web content)
    # ------------------------------------------------------------------ #

    def filter_context(
        self, user: UserContext, documents: List[Document]
    ) -> ContextDecision:
        """Layers 2+3 plus the indirect-injection scan on retrieved content."""
        decision = ContextDecision()
        try:
            candidates = documents

            # Indirect injection: content that tries to instruct the LLM is
            # dropped entirely — it is data that has proven itself hostile.
            if self._settings.prompt_injection_detection:
                safe: List[Document] = []
                for document in candidates:
                    origin = document.metadata.get("filename") or document.metadata.get(
                        "url", "unknown"
                    )
                    verdict = self._injection.scan_untrusted(document.page_content, origin)
                    if verdict.allowed:
                        safe.append(document)
                    else:
                        decision.dropped_injected += 1
                candidates = safe

            # Sensitivity + permission filtering.
            if self._settings.sensitive_document_detection and self._settings.permission_validation:
                permission = self._permissions.filter_chunks(user, candidates)
                decision.documents = permission.allowed
                decision.denied_texts = permission.denied_texts
                decision.denied_by_permission = permission.denied_count
            else:
                decision.documents = candidates

            return decision
        except Exception as exc:  # noqa: BLE001 — fail closed: empty context
            logger.error("security_layer_error", layer="filter_context", error=str(exc))
            return ContextDecision(documents=[])

    # ------------------------------------------------------------------ #
    # Checkpoint 3: the outgoing answer
    # ------------------------------------------------------------------ #

    def validate_response(
        self, answer: str, denied_texts: Sequence[str] = ()
    ) -> ResponseDecision:
        """Layer 4 scan, then Layer 5 PII redaction on the surviving answer."""
        try:
            if self._settings.response_scanning:
                verdict = self._scanner.scan(answer, denied_texts)
                if not verdict.allowed:
                    return ResponseDecision(answer=REFUSAL_MESSAGE, blocked=True)

            if self._settings.pii_detection and self._settings.redact_pii_in_responses:
                redacted, findings = self._pii.redact(answer)
                return ResponseDecision(answer=redacted, pii_redactions=len(findings))

            return ResponseDecision(answer=answer)
        except Exception as exc:  # noqa: BLE001 — fail closed
            logger.error("security_layer_error", layer="validate_response", error=str(exc))
            return ResponseDecision(answer=REFUSAL_MESSAGE, blocked=True)

    # ------------------------------------------------------------------ #
    # Ingestion hook (Layer 2 tagging at write time)
    # ------------------------------------------------------------------ #

    def tag_sensitivity(self, documents: List[Document]) -> List[Document]:
        """Stamp sensitivity metadata on chunks before they are indexed."""
        if self._settings.sensitive_document_detection:
            return self._classifier.tag(documents)
        return documents

    @property
    def refusal_message(self) -> str:
        return REFUSAL_MESSAGE
