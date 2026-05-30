"""Speculative Speculative Decoding (Kumar, Dao & May 2026), single-device study.

On real SSD hardware, the speculator runs on a separate device in parallel with
the verifier. Here both run sequentially on one device — output is identical and
cache-hit dynamics are preserved exactly, but wall-clock is slower.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch

from .cache import CachedSpeculation, Outcome, SpeculationCache
from .models import LM, truncate_kv
from .outcomes import FanOutFn, predict_outcomes, uniform_fan_out
from .sampling import logits_to_probs, sample, sample_residual, saguaro_probs


@dataclass
class SSDStats:
    rounds: int = 0
    cache_hits: int = 0
    total_proposed: int = 0
    total_accepted: int = 0
    cache_sizes: list[int] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / self.rounds if self.rounds else 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted / self.total_proposed if self.total_proposed else 0.0


def _prefill(lm: LM, ids: torch.Tensor):
    if ids.shape[1] <= 1:
        return None
    return lm.forward(ids[:, :-1]).past_key_values


def _draft_K_from(
    draft: LM,
    kv_init,
    first_token: int,
    K: int,
    temperature: float,
    fan_out: list[int] | None = None,
    saguaro_c: float = 1.0,
):
    """K+1 draft passes from `kv_init`, starting by feeding `first_token`.

    Returns:
        spec_tokens   (K,)         spec[k] sampled from biased distribution at position k
        biased_probs  (K+1, V)     distribution used for sampling (and for α in verification)
        raw_logits    (K+1, V)     unbiased logits (used for outcome prediction)
        snapshots     (K+1)        snap[k] = KV covering (kv_init prefix) + first_token + spec[:k]
    """
    snapshots: list = []
    spec_tokens: list[torch.Tensor] = []
    biased: list[torch.Tensor] = []
    raw: list[torch.Tensor] = []

    kv = kv_init
    prev = torch.tensor([[first_token]], device=draft.device)
    for j in range(K):
        out = draft.forward(prev, past_kv=kv)
        kv = out.past_key_values
        snapshots.append(copy.deepcopy(kv))
        z = out.logits[0, -1, :]
        raw.append(z)
        F = fan_out[j] if fan_out is not None else 0
        p = saguaro_probs(z, temperature, F, saguaro_c)
        t = sample(p)
        biased.append(p)
        spec_tokens.append(t)
        prev = t.view(1, 1)

    out = draft.forward(prev, past_kv=kv)
    snapshots.append(copy.deepcopy(out.past_key_values))
    z = out.logits[0, -1, :]
    raw.append(z)
    F_K = fan_out[K] if fan_out is not None else 0
    biased.append(saguaro_probs(z, temperature, F_K, saguaro_c))

    return torch.stack(spec_tokens), torch.stack(biased), torch.stack(raw), snapshots


def _build_cache(
    draft: LM,
    outcomes: list[Outcome],
    snapshots: list,
    k: int,
    temperature: float,
    fan_out: list[int] | None,
    saguaro_c: float,
) -> SpeculationCache:
    cache = SpeculationCache()
    for outcome in outcomes:
        toks, biased, raw, snaps = _draft_K_from(
            draft,
            copy.deepcopy(snapshots[outcome.k_accepted]),
            outcome.bonus,
            k,
            temperature,
            fan_out=fan_out,
            saguaro_c=saguaro_c,
        )
        cache.put(
            outcome,
            CachedSpeculation(
                tokens=toks.tolist(),
                draft_probs=biased,
                raw_logits=raw,
                snapshots=snaps,
            ),
        )
    return cache


def _verify(
    verifier: LM,
    last_token: int,
    spec_tokens: torch.Tensor,
    spec_probs: torch.Tensor,
    v_kv,
    temperature: float,
):
    """One verifier forward pass on [last_token, spec[0..K-1]]; accept/reject + sample bonus."""
    K = spec_tokens.shape[0]
    v_input = torch.cat(
        [torch.tensor([[last_token]], device=verifier.device), spec_tokens.view(1, -1)], dim=1
    )
    out = verifier.forward(v_input, past_kv=v_kv)
    v_kv = out.past_key_values
    v_probs = logits_to_probs(out.logits[0], temperature)  # (K+1, V)

    new_tokens: list[int] = []
    rejected = False
    for i in range(K):
        tok = int(spec_tokens[i].item())
        alpha = min(1.0, (v_probs[i, tok] / (spec_probs[i, tok] + 1e-10)).item())
        if torch.rand(1).item() < alpha:
            new_tokens.append(tok)
        else:
            new_tokens.append(sample_residual(v_probs[i], spec_probs[i]))
            rejected = True
            break

    accepted = len(new_tokens) - (1 if rejected else 0)
    if not rejected:
        new_tokens.append(int(sample(v_probs[K]).item()))

    outcome = Outcome(k_accepted=accepted, bonus=new_tokens[-1])
    return outcome, new_tokens, accepted, v_kv


def ssd_decode(
    draft: LM,
    verifier: LM,
    prompt: str,
    n_tokens: int,
    k: int = 4,
    budget: int = 8,
    temperature: float = 0.0,
    fan_out_fn: FanOutFn | None = None,
    saguaro_c: float = 1.0,
) -> tuple[list[int], SSDStats]:
    """SSD with pluggable fan-out, optional Saguaro sampling, sync-draft fallback.

    Args:
        fan_out_fn:  callable(K, B) → list of K+1 per-position cache budgets.
                     Defaults to uniform. See `outcomes.geometric_fan_out` for §4.1.
        saguaro_c:   Saguaro downweight C ∈ [0, 1]. 1.0 disables (vanilla sampling).
                     Lower values increase cache hit rate at the cost of acceptance.
    """
    if fan_out_fn is None:
        fan_out_fn = uniform_fan_out

    ids = verifier.tokenizer.encode(prompt, return_tensors="pt").to(verifier.device)
    L0 = ids.shape[1]
    current = ids.clone()

    v_kv = _prefill(verifier, ids)
    d_kv = _prefill(draft, ids)
    stats = SSDStats()

    fan_out = fan_out_fn(k, budget)
    last = int(current[0, -1].item())
    spec_tokens, spec_probs, raw_logits, snapshots = _draft_K_from(
        draft, d_kv, last, k, temperature, fan_out=fan_out, saguaro_c=saguaro_c
    )

    while current.shape[1] - L0 < n_tokens:
        T = current.shape[1]

        outcome, new_tokens, accepted, v_kv = _verify(
            verifier, last, spec_tokens, spec_probs[:k], v_kv, temperature
        )

        predicted = predict_outcomes(
            [raw_logits[i] for i in range(k + 1)], spec_tokens.tolist(), fan_out
        )
        cache = _build_cache(draft, predicted, snapshots, k, temperature, fan_out, saguaro_c)
        stats.cache_sizes.append(len(cache))

        if accepted < k:
            v_kv = truncate_kv(v_kv, T + accepted)
        current = torch.cat(
            [current, torch.tensor([new_tokens], dtype=current.dtype, device=current.device)],
            dim=1,
        )
        stats.rounds += 1
        stats.total_proposed += k
        stats.total_accepted += accepted

        if current.shape[1] - L0 >= n_tokens:
            break

        last = int(current[0, -1].item())
        if outcome in cache:
            stats.cache_hits += 1
            cached = cache.get(outcome)
            spec_tokens = torch.tensor(cached.tokens, device=draft.device)
            spec_probs = cached.draft_probs
            raw_logits = cached.raw_logits
            snapshots = cached.snapshots
        else:
            spec_tokens, spec_probs, raw_logits, snapshots = _draft_K_from(
                draft, snapshots[accepted], last, k, temperature,
                fan_out=fan_out, saguaro_c=saguaro_c,
            )

    return current[0, L0 : L0 + n_tokens].tolist(), stats
