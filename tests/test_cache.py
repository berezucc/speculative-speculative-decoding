import torch

from src.cache import CachedSpeculation, Outcome, SpeculationCache
from src.outcomes import predict_outcomes, uniform_fan_out


def test_uniform_fan_out_sums_to_budget():
    assert sum(uniform_fan_out(4, 8)) == 8
    assert sum(uniform_fan_out(4, 10)) == 10
    assert sum(uniform_fan_out(4, 0)) == 0


def test_uniform_fan_out_spreads_evenly():
    # K=4 → 5 positions, B=10 → [2,2,2,2,2]
    assert uniform_fan_out(4, 10) == [2, 2, 2, 2, 2]
    # B=11 → one extra to position 0
    assert uniform_fan_out(4, 11) == [3, 2, 2, 2, 2]


def test_cache_basic():
    cache = SpeculationCache()
    outcome = Outcome(k_accepted=2, bonus=42)
    cache.put(
        outcome,
        CachedSpeculation(
            tokens=[1, 2, 3, 4],
            draft_probs=torch.zeros(5, 10),
            raw_logits=torch.zeros(5, 10),
            snapshots=[],
        ),
    )
    assert outcome in cache
    assert len(cache) == 1
    assert cache.get(outcome).tokens == [1, 2, 3, 4]
    assert cache.get(Outcome(k_accepted=2, bonus=43)) is None


def test_predict_outcomes_excludes_sampled_token():
    # 3 positions, vocab size 5
    # logits favor token 0; spec_tokens[k] = 0 → should be excluded as bonus candidate
    logits = [torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]) for _ in range(3)]
    spec_tokens = [0, 0]  # K=2, so position 2 is the "full accept" with no exclusion
    fan_out = [2, 2, 2]
    outs = predict_outcomes(logits, spec_tokens, fan_out)

    # Position 0: top-2 excluding token 0 → tokens 1, 2
    assert Outcome(0, 1) in outs and Outcome(0, 2) in outs
    assert Outcome(0, 0) not in outs
    # Position 1: same exclusion
    assert Outcome(1, 1) in outs and Outcome(1, 2) in outs
    assert Outcome(1, 0) not in outs
    # Position 2 (k=K=2): no exclusion, top-2 = tokens 0, 1
    assert Outcome(2, 0) in outs and Outcome(2, 1) in outs


def test_predict_outcomes_respects_budget_per_position():
    logits = [torch.arange(10.0, 0.0, -1.0) for _ in range(3)]
    fan_out = [1, 0, 3]
    outs = predict_outcomes(logits, [9, 9], fan_out)
    by_pos = {k: 0 for k in range(3)}
    for o in outs:
        by_pos[o.k_accepted] += 1
    assert by_pos == {0: 1, 1: 0, 2: 3}
