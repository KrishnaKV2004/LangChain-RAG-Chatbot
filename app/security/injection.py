"""Layer 1 — prompt-injection detection.

Weighted pattern scoring over four attack families (role override, prompt
extraction, data exfiltration, jailbreak) plus structural heuristics
(chat-template tokens, zero-width characters). A single strong pattern match
is enough to block; weaker structural signals must accumulate.

The same detector doubles as the **indirect-injection** scanner: retrieved
documents and webpages are scanned with :meth:`scan_untrusted` before they
enter the prompt, so instructions planted inside a poisoned document or a
malicious webpage are caught (context-poisoning defense).

Patterns are intentionally conservative about qualifiers ("your", "system",
"hidden", "every", "all") so legitimate logistics questions like "show the
instructions for packing lithium batteries" pass untouched — verified by
tests.
"""

import re
from typing import List, Pattern, Tuple

from app.security.models import SecurityVerdict, ThreatType
from app.utils.logging import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Pattern catalog. Weight 1.0 = blocks on its own (threshold is 1.0);
# weight 0.5 = structural signal, two needed to block.
# --------------------------------------------------------------------------- #

_PATTERNS: List[Tuple[ThreatType, float, Pattern[str]]] = [
    # ---- Role override / instruction hijacking --------------------------- #
    (ThreatType.ROLE_OVERRIDE, 1.0, re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|any|your)\b"
        r".{0,30}\b(instructions?|prompts?|rules?|guidelines?|messages?|training)\b", re.I | re.S)),
    (ThreatType.ROLE_OVERRIDE, 1.0, re.compile(
        r"\byou\s+are\s+(now|no\s+longer)\b", re.I)),
    (ThreatType.ROLE_OVERRIDE, 1.0, re.compile(
        r"\b(pretend|act\s+as\s+if|imagine)\b.{0,40}\b(no\s+(rules|restrictions|guidelines)|"
        r"unrestricted|different\s+(ai|assistant|system))\b", re.I | re.S)),
    (ThreatType.ROLE_OVERRIDE, 1.0, re.compile(
        r"\bnew\s+(system\s+)?(instructions?|persona|role)\s*:", re.I)),
    # ---- System-prompt extraction ----------------------------------------- #
    (ThreatType.PROMPT_EXTRACTION, 1.0, re.compile(
        r"\b(reveal|show|print|display|output|repeat|expose|tell\s+me|share|leak)\b.{0,30}"
        r"\b(your|the)\b.{0,20}\b(system|internal|hidden|secret|initial|original)\b.{0,20}"
        r"\b(prompt|prompts|instructions?|context|rules|message)\b", re.I | re.S)),
    (ThreatType.PROMPT_EXTRACTION, 1.0, re.compile(
        r"\b(your)\s+(system\s+prompt|instructions|initial\s+prompt|original\s+prompt|"
        r"hidden\s+(context|rules)|internal\s+prompts?)\b", re.I)),
    (ThreatType.PROMPT_EXTRACTION, 1.0, re.compile(
        r"\brepeat\b.{0,20}\b(everything|the\s+text|all\s+text|words?)\b.{0,20}\babove\b", re.I)),
    (ThreatType.PROMPT_EXTRACTION, 1.0, re.compile(
        r"\bwhat\s+(are|were)\s+your\s+(exact\s+)?(instructions|rules|prompts?)\b", re.I)),
    (ThreatType.PROMPT_EXTRACTION, 1.0, re.compile(
        r"\b(hidden|secret)\s+context\b", re.I)),
    # "prompts" is our vocabulary, not logistics vocabulary — any talk of
    # internal/system prompts is an extraction attempt.
    (ThreatType.PROMPT_EXTRACTION, 1.0, re.compile(
        r"\b(internal|hidden|secret|system)\s+prompts?\b", re.I)),
    # ---- Data exfiltration ------------------------------------------------- #
    (ThreatType.DATA_EXFILTRATION, 1.0, re.compile(
        r"\b(show|give|list|dump|export|send|print|display)\b.{0,20}"
        r"\b(every|all)\b.{0,20}\b(documents?|files?|records?|data|customers?|clients?|chunks?)\b",
        re.I | re.S)),
    (ThreatType.DATA_EXFILTRATION, 1.0, re.compile(
        r"\bdump\b.{0,20}\b(database|db|vector\s*store|index|collection|memory)\b", re.I)),
    (ThreatType.DATA_EXFILTRATION, 1.0, re.compile(
        r"\b(list|show|give|reveal|name)\b.{0,20}\bconfidential\b", re.I)),
    (ThreatType.DATA_EXFILTRATION, 1.0, re.compile(
        r"\b(entire|whole|complete|full)\s+(database|knowledge\s*base|document\s*store|"
        r"vector\s*store|index)\b", re.I)),
    # ---- Jailbreak --------------------------------------------------------- #
    (ThreatType.JAILBREAK, 1.0, re.compile(
        r"\b(jailbreak|jail\s*break)\b|\bDAN\b|\bdo\s+anything\s+now\b", re.I)),
    (ThreatType.JAILBREAK, 1.0, re.compile(
        r"\b(developer|god|sudo|root|unrestricted|unfiltered)\s+mode\b", re.I)),
    (ThreatType.JAILBREAK, 1.0, re.compile(
        r"\b(without|no|bypass(ing)?|disable[sd]?|remove)\b.{0,20}"
        r"\b(restrictions?|limitations?|filters?|guardrails?|safety|censorship)\b", re.I | re.S)),
]

# Structural signals — chat-template tokens and markers that have no business
# appearing in a genuine user question or a legitimate document.
_STRUCTURAL: List[Tuple[float, Pattern[str]]] = [
    (0.5, re.compile(r"<\|[a-z_]+\|>", re.I)),                # <|im_start|> etc.
    (0.5, re.compile(r"^\s*###?\s*(system|instruction)\b", re.I | re.M)),
    (0.5, re.compile(r"\[\s*(system|assistant)\s*\]\s*:", re.I)),
    (0.5, re.compile(r"[​-‏⁠﻿]")),        # zero-width chars
]

_BLOCK_THRESHOLD = 1.0
_LAYER = "prompt_injection"


class InjectionDetector:
    """Detects direct (user query) and indirect (document) prompt injection."""

    def scan_query(self, query: str) -> SecurityVerdict:
        """Layer-1 check on the raw user input."""
        return self._scan(query, layer=_LAYER)

    def scan_untrusted(self, text: str, origin: str) -> SecurityVerdict:
        """Indirect-injection check on retrieved/document/web content.

        ``origin`` (filename or URL) is recorded in logs so poisoned sources
        can be traced and removed.
        """
        verdict = self._scan(text, layer="document_injection")
        if not verdict.allowed:
            verdict.threat_type = ThreatType.CONTEXT_POISONING
            logger.warning(
                "untrusted_content_flagged", origin=origin, reasons=verdict.reasons
            )
        return verdict

    # ------------------------------------------------------------------ #

    def _scan(self, text: str, layer: str) -> SecurityVerdict:
        score = 0.0
        threat: ThreatType = ThreatType.ROLE_OVERRIDE
        reasons: List[str] = []

        for threat_type, weight, pattern in _PATTERNS:
            match = pattern.search(text)
            if match:
                score += weight
                if not reasons:  # first (strongest) hit names the threat
                    threat = threat_type
                reasons.append(f"{threat_type.value}:{pattern.pattern[:40]}")

        for weight, pattern in _STRUCTURAL:
            if pattern.search(text):
                score += weight
                reasons.append(f"structural:{pattern.pattern[:40]}")

        if score >= _BLOCK_THRESHOLD:
            logger.warning(
                "injection_detected", layer=layer, score=score, threat=threat.value
            )
            return SecurityVerdict(
                allowed=False, layer=layer, threat_type=threat, reasons=reasons, score=score
            )
        return SecurityVerdict(allowed=True, layer=layer, score=score, reasons=reasons)
