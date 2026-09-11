import numpy as np
import pytest

from policyshiftlab.monte_carlo import run_selection_monte_carlo
from policyshiftlab.synthetic import generate_synthetic_recommender


def test_monte_carlo_returns_all_estimators() -> None:
    data = generate_synthetic_recommender(
        n_users=40,
        n_items=30,
        seed=7,
    )

    result = run_selection_monte_carlo(
        data.outcome,
        data.true_outcome_prob,
        data.propensity,
        n_replications=50,
        seed=11,
    )

    assert set(result) == {"naive", "oracle_ipw", "stratified"}
    for summary in result.values():
        assert summary.n_replications > 1
        assert np.isfinite(summary.mean_estimate)
        assert np.isfinite(summary.bias)
        assert summary.variance >= 0.0
        assert summary.rmse >= 0.0


def test_oracle_ipw_bias_is_small_under_strong_overlap() -> None:
    data = generate_synthetic_recommender(
        n_users=80,
        n_items=60,
        logging_policy="popularity",
        popularity_strength=2.0,
        exposure_strength=1.2,
        min_propensity=0.15,
        max_propensity=0.85,
        seed=123,
    )

    result = run_selection_monte_carlo(
        data.outcome,
        data.true_outcome_prob,
        data.propensity,
        n_replications=600,
        seed=999,
    )

    assert abs(result["oracle_ipw"].bias) < 0.002


def test_oracle_ipw_coverage_is_reasonable_under_strong_overlap() -> None:
    data = generate_synthetic_recommender(
        n_users=70,
        n_items=50,
        logging_policy="aligned",
        exposure_strength=1.0,
        min_propensity=0.2,
        max_propensity=0.8,
        seed=321,
    )

    result = run_selection_monte_carlo(
        data.outcome,
        data.true_outcome_prob,
        data.propensity,
        n_replications=500,
        seed=456,
    )

    coverage = result["oracle_ipw"].coverage_95
    assert 0.88 <= coverage <= 0.99


def test_stratified_coverage_is_nan_until_interval_is_implemented() -> None:
    data = generate_synthetic_recommender(
        n_users=30,
        n_items=20,
        seed=2,
    )

    result = run_selection_monte_carlo(
        data.outcome,
        data.true_outcome_prob,
        data.propensity,
        n_replications=40,
        seed=3,
    )

    assert np.isnan(result["stratified"].coverage_95)
    assert np.isnan(result["stratified"].mean_interval_width)


@pytest.mark.parametrize(
    "n_replications",
    [1, 0],
)
def test_invalid_replication_count_is_rejected(n_replications) -> None:
    y = np.array([0.0, 1.0])
    p = np.array([0.2, 0.8])
    e = np.array([0.5, 0.5])

    with pytest.raises(ValueError):
        run_selection_monte_carlo(
            y,
            p,
            e,
            n_replications=n_replications,
        )
