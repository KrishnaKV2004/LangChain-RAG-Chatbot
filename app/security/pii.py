"""Layer 5 — PII detection and redaction.

Regex-based detectors with validity checks where they exist (Luhn for card
numbers) to keep false positives down. Findings in *responses* are redacted
in place — blocking an otherwise good answer over an email address would be
worse than masking it.

Credentials (API keys, tokens) are treated as PII of the most severe kind:
they are always redacted and additionally reported to Layer 4.
"""

import re
from dataclasses import dataclass
from typing import List, Pattern, Tuple

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PIIFinding:
    """One detected piece of personally identifiable information."""

    kind: str
    value: str
    start: int
    end: int


def _luhn_valid(number: str) -> bool:
    """Luhn checksum — filters random digit runs from real card numbers."""
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


# (kind, pattern, needs_validation)
_DETECTORS: List[Tuple[str, Pattern[str]]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    # End lookahead permits '.' so sentence-final numbers still match.
    ("phone", re.compile(r"(?<![\d.\w])\+?\d{1,3}[ .-]?\(?\d{2,4}\)?[ .-]?\d{3,4}[ .-]?\d{3,4}(?!\w)")),
    ("api_key", re.compile(
        r"\b(sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|"
        r"xox[bpars]-[A-Za-z0-9-]{10,}|tvly-[A-Za-z0-9-]{16,})\b")),
]

#: Minimum digits for a phone candidate (rules out "extension 1234" etc.).
_PHONE_MIN_DIGITS = 9


class PIIDetector:
    """Finds and redacts PII in text."""

    def detect(self, text: str) -> List[PIIFinding]:
        # Card-number *candidates* (even Luhn-invalid ones) suppress phone
        # matches inside the same digit run — a 16-digit sequence is never a
        # phone number, and overlapping spans would corrupt redaction.
        card_spans = [
            (m.start(), m.end())
            for m in dict(_DETECTORS)["credit_card"].finditer(text)
        ]

        findings: List[PIIFinding] = []
        for kind, pattern in _DETECTORS:
            for match in pattern.finditer(text):
                value = match.group(0)
                if kind == "credit_card" and not _luhn_valid(value):
                    continue
                if kind == "phone":
                    if len(re.sub(r"\D", "", value)) < _PHONE_MIN_DIGITS:
                        continue
                    if any(s < match.end() and match.start() < e for s, e in card_spans):
                        continue
                findings.append(
                    PIIFinding(kind=kind, value=value, start=match.start(), end=match.end())
                )
        return findings

    def redact(self, text: str) -> Tuple[str, List[PIIFinding]]:
        """Replace each finding with a ``[REDACTED-KIND]`` placeholder.

        Replacement runs right-to-left so earlier offsets stay valid.
        """
        findings = self.detect(text)
        redacted = text
        for finding in sorted(findings, key=lambda f: f.start, reverse=True):
            placeholder = f"[REDACTED-{finding.kind.upper().replace('_', '-')}]"
            redacted = redacted[: finding.start] + placeholder + redacted[finding.end :]
        if findings:
            logger.info(
                "pii_redacted",
                kinds=sorted({f.kind for f in findings}),
                count=len(findings),
            )
        return redacted, findings
