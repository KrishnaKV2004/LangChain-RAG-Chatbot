"""Layer 4 — response scanning.

The last gate before an answer leaves the system. Three checks:

1. **Denied-content leakage** — n-gram shingle overlap between the answer and
   the text of chunks Layer 3 denied for this request. Even if the LLM was
   somehow influenced by content it shouldn't reveal, verbatim fragments are
   caught here.
2. **System-prompt leakage** — distinctive marker phrases from our own prompt
   templates must never appear in output (extraction defense in depth).
3. **Credential leakage** — API-key/token patterns shared with Layer 5.

Any hit blocks the response outright (fail closed); PII redaction (Layer 5)
is handled separately by the guard because it modifies rather than blocks.
"""

import re
from typing import List, Sequence, Set

from app.security.models import SecurityVerdict
from app.security.pii import PIIDetector
from app.utils.logging import get_logger

logger = get_logger(__name__)

_LAYER = "response_scan"
#: Words per shingle for the leak check. 8 consecutive shared words is far
#: beyond coincidental phrasing overlap.
_SHINGLE_SIZE = 8


def _normalize(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _shingles(words: Sequence[str], size: int) -> Set[str]:
    if len(words) < size:
        # Short texts fall back to one shingle covering the whole text —
        # a tiny confidential note must still be protected.
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


class ResponseScanner:
    """Validates outgoing answers against leakage of protected content."""

    def __init__(
        self,
        pii_detector: PIIDetector,
        protected_markers: Sequence[str] = (),
    ) -> None:
        self._pii = pii_detector
        # Marker phrases from prompt templates (wired in by the agent layer).
        self._markers = [m.lower() for m in protected_markers if m]

    def scan(self, answer: str, denied_texts: Sequence[str] = ()) -> SecurityVerdict:
        """Return a verdict; ``allowed=False`` means suppress the answer."""
        reasons: List[str] = []
        answer_words = _normalize(answer)
        answer_shingles = _shingles(answer_words, _SHINGLE_SIZE)
        answer_lower = answer.lower()

        # 1) Leaked fragments of permission-denied chunks.
        for denied in denied_texts:
            overlap = _shingles(_normalize(denied), _SHINGLE_SIZE) & answer_shingles
            if overlap:
                reasons.append("denied_content_leak")
                break

        # 2) Our own prompt template text must never be echoed.
        for marker in self._markers:
            if marker in answer_lower:
                reasons.append("system_prompt_leak")
                break

        # 3) Credentials of any kind.
        if any(f.kind == "api_key" for f in self._pii.detect(answer)):
            reasons.append("credential_leak")

        if reasons:
            logger.warning("response_blocked", reasons=reasons)
            return SecurityVerdict(allowed=False, layer=_LAYER, reasons=reasons)
        return SecurityVerdict.ok(_LAYER)
