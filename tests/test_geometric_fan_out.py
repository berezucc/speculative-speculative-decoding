"""Geometric fan-out (Theorem 12) properties."""
import pytest

from functools import partial

from src.models import LM, assert_shared_vocab
from src.outcomes import geometric_fan_out
from src.sd import greedy_decode
from src.ssd import ssd_decode
from src.utils import get_device, set_seed


def test_sums_to_budget():
    for B in [0, 1, 8, 17, 100]:
        for K in [1, 4, 8]:
            for a_p in [0.1, 0.5, 0.9]:
                f = geometric_fan_out(K, B, a_p)
                assert sum(f) == B, f"K={K} B={B} a_p={a_p}: sum={sum(f)}"
                assert len(f) == K + 1
                assert all(x >= 0 for x in f)


def test_decreasing_in_position():
    # Among positions 0..K-1, F_k should be monotonically non-increasing.
    f = geometric_fan_out(K=8, B=100, acceptance_rate=0.5, r=1.0)
    for k in range(7):
        assert f[k] >= f[k + 1] - 1, f"position {k}: {f[k]} should not be << f[{k+1}]={f[k+1]}"


def test_extreme_acceptance_rates():
    # a_p = 0: all budget on position 0 (can't accept anything, so bonus from prefix)
    assert geometric_fan_out(K=4, B=10, acceptance_rate=0.0) == [10, 0, 0, 0, 0]
    # a_p = 1: all budget on position K (always accept everything)
    assert geometric_fan_out(K=4, B=10, acceptance_rate=1.0) == [0, 0, 0, 0, 10]


def test_zero_budget_is_zero():
    assert geometric_fan_out(K=4, B=0, acceptance_rate=0.5) == [0, 0, 0, 0, 0]


@pytest.fixture(scope="module")
def models():
    device = get_device()
    draft = LM.load("distilgpt2", device)
    verifier = LM.load("distilgpt2", device)
    assert_shared_vocab(draft, verifier)
    return draft, verifier


def test_ssd_lossless_with_geometric(models):
    draft, verifier = models
    fn = partial(geometric_fan_out, acceptance_rate=0.7, r=1.0)
    set_seed()
    ref = greedy_decode(verifier, "The transformer architecture", 15)
    set_seed()
    spec, stats = ssd_decode(
        draft, verifier, "The transformer architecture", 15,
        k=4, budget=8, temperature=0.0, fan_out_fn=fn,
    )
    assert spec == ref
    assert stats.cache_hits > 0
