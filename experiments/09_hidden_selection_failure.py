"""Demonstrate failure under an unobserved selection driver."""

from policyshiftlab.hidden_selection import (
    generate_hidden_selection_data,
    run_hidden_selection_monte_carlo,
)


def _print_summary(summary) -> None:
    print(summary.estimator)
    print(f"  mean estimate: {summary.mean_estimate:.6f}")
    print(f"  bias:          {summary.bias:+.6f}")
    print(f"  variance:      {summary.variance:.8f}")
    print(f"  RMSE:          {summary.rmse:.6f}")


def main() -> None:
    data = generate_hidden_selection_data(
        n_users=200,
        n_items=80,
        outcome_hidden_strength=1.5,
        selection_hidden_strength=2.0,
        exposure_strength=1.5,
        min_propensity=0.05,
        max_propensity=0.95,
        seed=0,
    )

    result = run_hidden_selection_monte_carlo(
        data,
        n_replications=500,
        seed=123,
    )

    print("hidden Z affects both Y and selection")
    print("observed-only propensity is exact P(S=1 | X)")
    print(f"target Brier: {result.target_brier:.6f}")
    print()

    _print_summary(result.naive)
    print()
    _print_summary(result.full_ipw)
    print()
    _print_summary(result.observed_only_ipw)


if __name__ == "__main__":
    main()
