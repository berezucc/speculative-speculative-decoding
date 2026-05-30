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
    """K speculated tokens with everything the next round needs to verify them.

    Stores both the biased draft probs (used in α = min(1, p_target/p_draft)) and
    the raw logits (used to predict outcomes for the round after that).
    """

    tokens: list[int]
    draft_probs: torch.Tensor   # (K+1, V) biased — for verification
    raw_logits: torch.Tensor    # (K+1, V) — for next round's outcome prediction
    snapshots: list             # K+1 KV snapshots covering positions 0..K of the cached spec


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
