"""Run repeated leakage-safe Coat holdouts and bootstrap uncertainty."""

from pathlib import Path

from policyshiftlab.coat_audit import load_coat_matrix
from policyshiftlab.coat_propensity_audit import load_coat_propensities
from policyshiftlab.coat_repeated import (
    bootstrap_randomized_pair_difference,
    run_repeated_coat_holdouts,
)


def main() -> None:
    base = Path("data/coat")
    train = load_coat_matrix(base / "train.ascii")
    randomized = load_coat_matrix(base / "test.ascii")
    propensities = load_coat_propensities(base / "propensities.ascii")

    repeated = run_repeated_coat_holdouts(
        train,
        randomized,
        propensities,
        n_splits=200,
        first_seed=0,
        holdout_fraction=1.0 / 3.0,
        positive_threshold=4,
        prior_strength=10.0,
        pair=("user_smoothed", "item_smoothed"),
    )

    print(
        f"repeated Coat holdouts: {repeated.n_splits} splits "
        f"(seeds {repeated.first_seed}-"
        f"{repeated.first_seed + repeated.n_splits - 1})"
    )
    print(f"holdout fraction: {repeated.holdout_fraction:.3f}")
    print()
    print("metric-level error relative to randomized Brier")
    print("method                   mean-error   mean-abs     RMSE")
    for row in repeated.method_summaries:
        print(
            f"{row.method:24s} "
            f"{row.mean_signed_brier_error:+10.6f} "
            f"{row.mean_absolute_brier_error:10.6f} "
            f"{row.rmse_brier_error:10.6f}"
        )

    print()
    print("ranking stability relative to randomized benchmark")
    print(
        "method                   mean-sp  med-sp  mean-tau med-tau "
        "any-rev  mean-rev exact"
    )
    for row in repeated.method_summaries:
        print(
            f"{row.method:24s} "
            f"{row.mean_spearman:7.3f} "
            f"{row.median_spearman:7.3f} "
            f"{row.mean_kendall:8.3f} "
            f"{row.median_kendall:7.3f} "
            f"{row.any_reversal_rate:7.3f} "
            f"{row.mean_reversal_count:8.3f} "
            f"{row.exact_order_recovery_rate:6.3f}"
        )

    print()
    print("specific pair reversal frequency")
    for row in repeated.pair_reversal_summaries:
        print(
            f"{row.method:24s} "
            f"{row.model_a} vs {row.model_b}: "
            f"{row.reversal_rate:.3f}"
        )

    bootstrap = bootstrap_randomized_pair_difference(
        train,
        randomized,
        fit_seed=2026,
        holdout_fraction=1.0 / 3.0,
        positive_threshold=4,
        prior_strength=10.0,
        model_a="user_smoothed",
        model_b="item_smoothed",
        n_bootstrap=2000,
        seed=314159,
    )

    print()
    print("randomized benchmark paired user-cluster bootstrap")
    print(
        f"difference = Brier({bootstrap.model_a}) - "
        f"Brier({bootstrap.model_b})"
    )
    print(f"observed difference: {bootstrap.observed_difference:+.6f}")
    print(
        f"pointwise 95% percentile interval: "
        f"[{bootstrap.lower_95:+.6f}, {bootstrap.upper_95:+.6f}]"
    )
    print(
        f"bootstrap P({bootstrap.model_a} better): "
        f"{bootstrap.probability_a_better:.3f}"
    )
    print()
    print(
        "Repeated holdouts quantify sensitivity to the logged split "
        "conditional on this observed dataset; they are not independent "
        "dataset replications. The bootstrap resamples users as clusters "
        "and is descriptive uncertainty rather than a formal test."
    )


if __name__ == "__main__":
    main()
