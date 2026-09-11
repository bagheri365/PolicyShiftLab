"""Compare oracle and misspecified propensity correction."""

from policyshiftlab.propensity_misspecification import (
    run_propensity_misspecification_experiment,
)


def _print_scenario(summary) -> None:
    print(summary.label)
    print(f"  Brier estimate:          {summary.brier_estimate:.6f}")
    print(f"  Brier error:             {summary.brier_error:+.6f}")
    print(f"  mean Spearman:           {summary.mean_spearman:.3f}")
    print(f"  mean Kendall:            {summary.mean_kendall:.3f}")
    print(f"  any-reversal rate:       {summary.any_reversal_rate:.3f}")
    print(
        "  exact-order recovery:    "
        f"{summary.exact_order_recovery_rate:.3f}"
    )


def main() -> None:
    result = run_propensity_misspecification_experiment(
        n_users=200,
        n_items=80,
        n_replications=300,
        seed=0,
    )

    print("selection is always redrawn from the true logging propensity")
    print(f"target Brier:       {result.target_brier:.6f}")
    print(f"naive logged Brier: {result.naive_logged_brier:.6f}")
    print()

    _print_scenario(result.oracle)
    print()
    _print_scenario(result.mild_misspecified)
    print()
    _print_scenario(result.severe_misspecified)


if __name__ == "__main__":
    main()
