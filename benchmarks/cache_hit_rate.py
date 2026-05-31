"""Cache hit rate as a function of fan-out budget B.

Reproduces the spirit of Figure 3 in Kumar et al. 2026, scaled down: sweep B
across a small range, average hit rate over a handful of prompts and seeds.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from _common import PROMPTS, aggregate, base_parser, load, reset_seed_per_run, write_csv
from src.ssd import ssd_decode


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--budgets", type=int, nargs="+", default=[0, 2, 4, 8, 16, 32])
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    draft, verifier = load(args.draft, args.verifier)

    rows = []
    for B in args.budgets:
        hit_rates, acc_rates = [], []
        for prompt in PROMPTS[: args.prompts]:
            reset_seed_per_run()
            _, stats = ssd_decode(
                draft, verifier, prompt, n_tokens=args.n_tokens,
                k=args.k, budget=B, temperature=args.temperature,
            )
            hit_rates.append(stats.hit_rate)
            acc_rates.append(stats.acceptance_rate)
        row = {
            "budget": B,
            "hit_rate": aggregate(hit_rates),
            "acceptance_rate": aggregate(acc_rates),
        }
        rows.append(row)
        print(f"B={B:3d}  hit_rate={row['hit_rate']:.2%}  α={row['acceptance_rate']:.2%}")

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "cache_hit_rate.csv", rows, ["budget", "hit_rate", "acceptance_rate"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["budget"] for r in rows], [r["hit_rate"] for r in rows], marker="o")
    ax.set_xscale("symlog")
    ax.set_xlabel("fan-out budget B")
    ax.set_ylabel("cache hit rate")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.set_title(f"hit rate vs B  (K={args.k}, T={args.temperature}, {args.draft} → {args.verifier})")
    fig.tight_layout()
    fig.savefig(args.out / "cache_hit_rate.png", dpi=150)
    print(f"wrote {args.out / 'cache_hit_rate.png'}")


if __name__ == "__main__":
    main()
