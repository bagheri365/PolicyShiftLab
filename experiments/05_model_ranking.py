"""Compare model rankings under target, logged, and IPW evaluation."""

from policyshiftlab.ranking import (
    compare_model_rankings,
    fixed_candidate_predictions,
)
from policyshiftlab.synthetic import generate_synthetic_recommender


def _print_scores(
    title: str,
    scores: dict[str, float],
    order: tuple[str, ...],
) -> None:
    print(title)
    for rank, name in enumerate(order, start=1):
        print(f"  {rank}. {name:<20} {scores[name]:.6f}")


def main() -> None:
    data = generate_synthetic_recommender(
        n_users=500,
        n_items=150,
        logging_policy="popularity",
        popularity_strength=3.0,
        exposure_strength=2.0,
        min_propensity=0.05,
        max_propensity=0.95,
        seed=0,
    )

    predictions = fixed_candidate_predictions(
        data.true_outcome_prob,
        data.logging_score,
    )

    result = compare_model_rankings(
        data.outcome,
        predictions,
        data.selected,
        data.propensity,
    )

    _print_scores(
        "target Brier ranking",
        result.target_scores,
        result.target_order,
    )
    print()
    _print_scores(
        "naive logged Brier ranking",
        result.logged_scores,
        result.logged_order,
    )
    print()
    _print_scores(
        "oracle IPW Brier ranking",
        result.oracle_ipw_scores,
        result.oracle_ipw_order,
    )

    print()
    print(f"logged Spearman vs target: {result.logged_spearman:.3f}")
    print(f"logged Kendall vs target:  {result.logged_kendall:.3f}")
    print(
        "oracle IPW Spearman vs target: "
        f"{result.oracle_ipw_spearman:.3f}"
    )
    print(
        "oracle IPW Kendall vs target:  "
        f"{result.oracle_ipw_kendall:.3f}"
    )

    print()
    print(f"logged reversals: {len(result.logged_reversals)}")
    for reversal in result.logged_reversals:
        print(
            "  "
            f"{reversal.model_a} vs {reversal.model_b}: "
            f"target prefers {reversal.target_prefers}, "
            f"logged prefers {reversal.comparison_prefers}"
        )

    print(f"oracle IPW reversals: {len(result.oracle_ipw_reversals)}")
    for reversal in result.oracle_ipw_reversals:
        print(
            "  "
            f"{reversal.model_a} vs {reversal.model_b}: "
            f"target prefers {reversal.target_prefers}, "
            f"IPW prefers {reversal.comparison_prefers}"
        )


if __name__ == "__main__":
    main()
