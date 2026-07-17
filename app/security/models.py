"""Shared security data model: verdicts, roles, sensitivity levels.

The clearance model is deliberately simple and explicit:

* every document chunk has a **sensitivity level** (public < internal <
  confidential), assigned by Layer 2;
* every request carries a **user context** whose role maps to a clearance;
* Layer 3 allows a chunk only when ``clearance >= sensitivity``.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ThreatType(str, Enum):
    """Categories of attack the input layers detect."""

    ROLE_OVERRIDE = "role_override"
    PROMPT_EXTRACTION = "prompt_extraction"
    DATA_EXFILTRATION = "data_exfiltration"
    JAILBREAK = "jailbreak"
    CONTEXT_POISONING = "context_poisoning"
    OVERSIZED_INPUT = "oversized_input"


class SensitivityLevel(int, Enum):
    """Document classification. Higher = more restricted."""

    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2


class Role(str, Enum):
    """User roles known to the system."""

    GUEST = "guest"
    EMPLOYEE = "employee"
    ADMIN = "admin"


#: Highest sensitivity level each role may read.
ROLE_CLEARANCE = {
    Role.GUEST: SensitivityLevel.PUBLIC,
    Role.EMPLOYEE: SensitivityLevel.INTERNAL,
    Role.ADMIN: SensitivityLevel.CONFIDENTIAL,
}


@dataclass
class UserContext:
    """Identity attached to every request (API auth populates this)."""

    user_id: str = "anonymous"
    role: Role = Role.EMPLOYEE

    @property
    def clearance(self) -> SensitivityLevel:
        return ROLE_CLEARANCE[self.role]


@dataclass
class SecurityVerdict:
    """Outcome of a security check.

    ``allowed=False`` means the request/response must be refused. ``reasons``
    are for structured logs only — they are never shown to the end user
    (explaining which pattern fired would teach attackers to evade it).
    """

    allowed: bool
    layer: str
    threat_type: Optional[ThreatType] = None
    reasons: List[str] = field(default_factory=list)
    score: float = 0.0

    @classmethod
    def ok(cls, layer: str) -> "SecurityVerdict":
        return cls(allowed=True, layer=layer)


#: The polite, uninformative refusal shown for any blocked request.
REFUSAL_MESSAGE = (
    "I'm sorry, but I can't help with that request. I'm designed to protect "
    "confidential company information and my own operating instructions. "
    "If you believe you should have access to this information, please "
    "contact your administrator. Is there something else I can help you with?"
)
