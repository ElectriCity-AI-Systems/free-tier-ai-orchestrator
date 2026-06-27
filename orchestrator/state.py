"""Shared dataclasses describing the run state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# Subtask lifecycle:
#   pending -> done      (critic accepted within attempt budget)
#           -> partial   (best-effort kept after attempts exhausted)
#           -> failed    (could not produce any output at all)
STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"


@dataclass
class SubTask:
    id: str
    description: str
    role: str
    acceptance: str
    depends_on: List[str] = field(default_factory=list)

    status: str = STATUS_PENDING
    result: str = ""
    attempts: int = 0
    model_used: str = ""
    score: int = 0
