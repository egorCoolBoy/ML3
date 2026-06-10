from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RecallPolicy = Literal["active", "passive_only"]


@dataclass
class Fact:
    id: str
    user_id: str
    text: str
    subject: str
    predicate: str
    object: str
    recall_policy: RecallPolicy
    source_quote: str
    created_at: str
    session_id: str | None = None


@dataclass
class Turn:
    id: str
    user_id: str
    role: str
    content: str
    persona: str
    created_at: str
    session_id: str | None = None
