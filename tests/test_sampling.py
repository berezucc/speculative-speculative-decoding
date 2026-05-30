import torch

from src.sampling import logits_to_probs, sample_residual


def test_logits_to_probs_greedy_is_one_hot():
    logits = torch.tensor([1.0, 3.0, 2.0])
    p = logits_to_probs(logits, temperature=0.0)
    assert p.tolist() == [0.0, 1.0, 0.0]


def test_logits_to_probs_softmax_sums_to_one():
    logits = torch.randn(50)
    p = logits_to_probs(logits, temperature=0.7)
    assert abs(p.sum().item() - 1.0) < 1e-5


def test_residual_falls_back_to_target_when_empty():
    # p_draft ≥ p_target everywhere → residual is all zero; fall back to argmax(target)
    p_t = torch.tensor([0.2, 0.5, 0.3])
    p_d = torch.tensor([0.4, 0.4, 0.2])
    # p_t - p_d = [-0.2, 0.1, 0.1]; positive on idx 1 and 2
    samples = [sample_residual(p_t, p_d) for _ in range(200)]
    assert set(samples).issubset({1, 2})


def test_residual_sampling_distribution():
    torch.manual_seed(0)
    p_t = torch.tensor([0.5, 0.3, 0.2])
    p_d = torch.tensor([0.1, 0.1, 0.1])
    # residual ∝ [0.4, 0.2, 0.1] → normalized [4/7, 2/7, 1/7]
    counts = [0, 0, 0]
    n = 5000
    for _ in range(n):
        counts[sample_residual(p_t, p_d)] += 1
    freqs = [c / n for c in counts]
    expected = [4 / 7, 2 / 7, 1 / 7]
    for f, e in zip(freqs, expected):
        assert abs(f - e) < 0.03
