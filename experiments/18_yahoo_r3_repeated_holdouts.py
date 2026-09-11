"""Run repeated leakage-safe Yahoo! R3 holdouts."""

from pathlib import Path

from policyshiftlab.yahoo_r3_audit import load_yahoo_r3_triplets
from policyshiftlab.yahoo_r3_repeated import (
    bootstrap_yahoo_randomized_pair_difference,
    run_repeated_yahoo_r3_holdouts,
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
    train = load_yahoo_r3_triplets(train_path)
    randomized = load_yahoo_r3_triplets(randomized_path)

    repeated = run_repeated_yahoo_r3_holdouts(
        train,
        randomized,
        n_splits=200,
        first_seed=0,
        holdout_fraction=1.0 / 3.0,
        positive_threshold=4,
        prior_strength=10.0,
        pair=("user_smoothed", "user_item_blend"),
    )

    row = repeated.method_summary

    print(
        f"repeated Yahoo! R3 holdouts: {repeated.n_splits} splits "
        f"(seeds {repeated.first_seed}-"
        f"{repeated.first_seed + repeated.n_splits - 1})"
    )
    print(f"holdout fraction: {repeated.holdout_fraction:.3f}")
    print()
    print("metric-level gap: held-out logged Brier - randomized Brier")
    print(f"mean signed gap: {row.mean_signed_brier_gap:+.6f}")
    print(f"mean absolute gap: {row.mean_absolute_brier_gap:.6f}")
    print(f"RMSE gap: {row.rmse_brier_gap:.6f}")
    print()
    print("ranking stability relative to randomized benchmark")
    print(f"mean Spearman:        {row.mean_spearman:.3f}")
    print(f"median Spearman:      {row.median_spearman:.3f}")
    print(f"mean Kendall:         {row.mean_kendall:.3f}")
    print(f"median Kendall:       {row.median_kendall:.3f}")
    print(f"any-reversal rate:    {row.any_reversal_rate:.3f}")
    print(f"mean reversal count:  {row.mean_reversal_count:.3f}")
    print(f"exact-order recovery: {row.exact_order_recovery_rate:.3f}")
    print()
    pair = repeated.pair_reversal
    print(
        f"specific reversal {pair.model_a} vs {pair.model_b}: "
        f"{pair.reversal_rate:.3f}"
    )

    bootstrap = bootstrap_yahoo_randomized_pair_difference(
        train,
        randomized,
        fit_seed=2026,
        holdout_fraction=1.0 / 3.0,
        positive_threshold=4,
        prior_strength=10.0,
        model_a="user_item_blend",
        model_b="item_smoothed",
        n_bootstrap=5000,
        seed=271828,
    )

    print()
    print("randomized benchmark top-pair user-cluster bootstrap")
    print(
        f"difference = Brier({bootstrap.model_a}) - "
        f"Brier({bootstrap.model_b})"
    )
    print(f"observed difference: {bootstrap.observed_difference:+.6f}")
    print(
        "pointwise 95% percentile interval: "
        f"[{bootstrap.lower_95:+.6f}, {bootstrap.upper_95:+.6f}]"
    )
    print(
        f"bootstrap P({bootstrap.model_a} better): "
        f"{bootstrap.probability_a_better:.3f}"
    )
    print()
    print(
        "Repeated holdouts measure split sensitivity conditional on this "
        "observed Yahoo! R3 dataset; they are not independent dataset "
        "replications. The bootstrap resamples randomized users as clusters "
        "and is descriptive uncertainty rather than a formal test."
    )


if __name__ == "__main__":
    main()
