"""Audit the learned propensity matrix distributed with Coat."""

from pathlib import Path

from policyshiftlab.coat_audit import load_coat_matrix
from policyshiftlab.coat_propensity_audit import (
    audit_coat_propensities,
    load_coat_propensities,
)


def main() -> None:
    base = Path("data/coat")
    train_path = base / "train.ascii"
    randomized_path = base / "test.ascii"
    propensity_path = base / "propensities.ascii"

    missing = [
        str(path)
        for path in (train_path, randomized_path, propensity_path)
        if not path.exists()
    ]
    if missing:
        raise SystemExit(
            "Missing Coat files: " + ", ".join(missing)
        )

    train = load_coat_matrix(train_path)
    randomized = load_coat_matrix(randomized_path)
    prop = load_coat_propensities(propensity_path)
    result = audit_coat_propensities(train, randomized, prop)

    print(
        f"propensity shape: {result.n_users} users x "
        f"{result.n_items} items"
    )
    print(
        "propensity min/q01/q05/median/q95/q99/max: "
        f"{result.min_propensity:.6f}/"
        f"{result.q01_propensity:.6f}/"
        f"{result.q05_propensity:.6f}/"
        f"{result.median_propensity:.6f}/"
        f"{result.q95_propensity:.6f}/"
        f"{result.q99_propensity:.6f}/"
        f"{result.max_propensity:.6f}"
    )
    print(f"mean propensity: {result.mean_propensity:.6f}")
    print()
    print(
        "propensity row-sum min/median/max: "
        f"{result.row_sum_min:.3f}/"
        f"{result.row_sum_median:.3f}/"
        f"{result.row_sum_max:.3f}"
    )
    print(
        "mean row-sum (expected observations/user if Bernoulli): "
        f"{result.expected_row_observations:.3f}"
    )
    print()
    print(
        "mean supplied propensity on logged observations: "
        f"{result.train_observed_mean_propensity:.6f}"
    )
    print(
        "mean supplied propensity on logged-unobserved pairs: "
        f"{result.train_unobserved_mean_propensity:.6f}"
    )
    print(
        "mean supplied propensity on randomized observations: "
        f"{result.randomized_observed_mean_propensity:.6f}"
    )
    print()
    print(
        "logged inverse-weight min/median/max: "
        f"{result.train_observed_weight_min:.2f}/"
        f"{result.train_observed_weight_median:.2f}/"
        f"{result.train_observed_weight_max:.2f}"
    )
    print(
        "logged inverse-weight Kish ESS: "
        f"{result.train_observed_weight_ess:.1f}"
    )
    print()
    print(
        "Interpretation: propensities.ascii contains learned, not oracle, "
        "selection probabilities. Treat any propensity-weighted empirical "
        "estimate as an estimated-propensity analysis rather than an "
        "identification guarantee."
    )


if __name__ == "__main__":
    main()
