"""SSD at temperature 0 must produce token-identical output to verifier-only greedy."""
import pytest

from src.models import LM, assert_shared_vocab
from src.sd import greedy_decode
from src.ssd import ssd_decode
from src.utils import get_device, set_seed


@pytest.fixture(scope="module")
def models():
    device = get_device()
    draft = LM.load("distilgpt2", device)
    verifier = LM.load("distilgpt2", device)
    assert_shared_vocab(draft, verifier)
    return draft, verifier


@pytest.mark.parametrize("k,budget", [(2, 4), (4, 8), (4, 0)])
def test_ssd_matches_greedy_at_temp_zero(models, k, budget):
    draft, verifier = models
    prompt = "The transformer architecture"
    n = 15

    set_seed()
    ref = greedy_decode(verifier, prompt, n)
    set_seed()
    spec, stats = ssd_decode(draft, verifier, prompt, n, k=k, budget=budget, temperature=0.0)
    assert spec == ref, f"K={k}, B={budget}: SSD diverges from greedy"
    assert stats.rounds > 0
    if budget == 0:
        assert stats.cache_hits == 0
    else:
        # Self-speculation at temp=0: full-accept outcomes are deterministic, so
        # any round past the first should hit cache (max hits = rounds - 1).
        assert stats.cache_hits > 0
