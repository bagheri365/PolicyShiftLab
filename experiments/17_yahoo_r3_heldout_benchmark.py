"""Run leakage-safe Yahoo! R3 logged-vs-randomized evaluation."""

from pathlib import Path

from policyshiftlab.yahoo_r3_audit import load_yahoo_r3_triplets
from policyshiftlab.yahoo_r3_empirical import (
    run_yahoo_r3_heldout_benchmark,
)


def _resolve_paths(base: Path) -> tuple[Path, Path]:
    candidates = (
        (
            base / "ydata-ymusic-rating-study-v1_0-train.txt",
            base / "ydata-ymusic-rating-study-v1_0-test.txt",
        ),
        (base / "user.txt", base / "random.txt"),
    )
    for train_path, randomized_path in candidates:
        if train_path.exists() and randomized_path.exists():
            return train_path, randomized_path
    raise SystemExit("Yahoo! R3 train/randomized files not found")


def main() -> None:
    base = Path("data/yahoo_r3")
    train_path, randomized_path = _resolve_paths(base)

    result = run_yahoo_r3_heldout_benchmark(
        load_yahoo_r3_triplets(train_path),
        load_yahoo_r3_triplets(randomized_path),
        positive_threshold=4,
        prior_strength=10.0,
        holdout_fraction=1.0 / 3.0,
        seed=2026,
    )

    print("Yahoo! R3 leakage-safe benchmark")
    print(f"target randomized users: {result.n_target_users}")
    print(f"randomized items:        {result.n_items_randomized}")
    print(f"logged cohort total:     {result.n_logged_cohort_total}")
    print(f"logged fit:              {result.n_logged_fit}")
    print(f"logged evaluation:       {result.n_logged_evaluation}")
    print(f"randomized observations: {result.n_randomized}")
    print(
        f"holdout fraction:        {result.holdout_fraction:.3f} "
        f"(seed={result.seed})"
    )
    print(f"binary event:            rating >= {result.positive_threshold}")
    print()
    print("model                 randomized   heldout-log      gap")
    for row in result.scores:
        gap = row.heldout_logged_brier - row.randomized_brier
        print(
            f"{row.model:20s} "
            f"{row.randomized_brier:10.6f} "
            f"{row.heldout_logged_brier:12.6f} "
            f"{gap:+10.6f}"
        )

    print()
    print("ranking agreement with randomized benchmark")
    for row in result.rankings:
        print(
            f"{row.method:22s} "
            f"Spearman={row.spearman_vs_randomized:6.3f} "
            f"Kendall={row.kendall_vs_randomized:6.3f} "
            f"reversals={len(row.reversals_vs_randomized)} "
            f"order={' > '.join(row.order)}"
        )

    print()
    print(
        "Models are fitted only on logged-fit ratings from the 5,400 users "
        "who appear in randomized evaluation. Held-out logged ratings and "
        "randomized ratings are evaluation-only. No propensity correction is "
        "reported in this first Yahoo! R3 replication because no oracle "
        "exposure propensity is assumed."
    )


if __name__ == "__main__":
    main()
