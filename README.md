# Speculative Speculative Decoding

An implementation study of [Kumar, Dao & May 2026](https://arxiv.org/abs/2603.03251),
extending the ordinary speculative-decoding baseline in
[`berezucc/speculative-decoding`](https://github.com/berezucc/speculative-decoding).

**Scope of this repo.** Single-device (Apple M2 Max). The point of SSD is hiding
draft latency on a *separate* device, which this hardware cannot demonstrate.
This implementation studies the **algorithmic** behavior — cache hit rates,
fan-out strategies, sampling tradeoffs — not wall-clock speedup.

## What's here

- A clean implementation of ordinary speculative decoding (SD) as the baseline.
- The SSD framework: speculation cache, verification-outcome prediction, lossless fallback.
- The two key Saguaro optimizations from the paper:
  - Geometric fan-out for cache allocation (§4.1, Theorem 12).
  - Saguaro sampling for biasing the residual onto cache tokens (§4.2).
- Reproductions of the paper's algorithmic figures (cache hit rate vs fan-out, etc.).

## Quick start

```bash
pip install -r requirements.txt

# Lossless correctness checks (must pass before any benchmark is trustworthy)
pytest tests/

# Run benchmarks
python benchmarks/cache_hit_rate.py
python benchmarks/fan_out_sweep.py
python benchmarks/acceptance_hit_pareto.py
```

## Models

Configurable via CLI flags; defaults match the baseline repo.

| Role | Default | Override |
|---|---|---|
| Draft | `distilgpt2` | `--draft <hf-id>` |
| Verifier | `gpt2-medium` | `--verifier <hf-id>` |

A shared tokenizer is required.

## Status

In progress. See commit log for piece-by-piece build.
