"""Speculation cache (Definition 4 in Kumar et al. 2026)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass(frozen=True)
class Outcome:
    """A verification outcome v^T = (k_accepted, bonus_token_id)."""

    k_accepted: int
    bonus: int


@dataclass
class CachedSpeculation:
    """K speculated tokens + the draft probabilities used to generate them.

    The probabilities are kept so the next round's acceptance step can compute
    α = min(1, p_target / p_draft) without re-running the draft.
    """

    tokens: list[int]
    draft_probs: torch.Tensor  # (K, V)
    kv_after: object           # draft KV covering [outcome prefix + tokens[:-1]]


@dataclass
class SpeculationCache:
    """Dict-like mapping from Outcome → CachedSpeculation."""

    entries: dict[Outcome, CachedSpeculation] = field(default_factory=dict)

    def put(self, outcome: Outcome, value: CachedSpeculation) -> None:
        self.entries[outcome] = value

    def get(self, outcome: Outcome) -> Optional[CachedSpeculation]:
        return self.entries.get(outcome)

    def __contains__(self, outcome: Outcome) -> bool:
        return outcome in self.entries

    def __len__(self) -> int:
        return len(self.entries)
