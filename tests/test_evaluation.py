import numpy as np
import pytest

from policyshiftlab.evaluation import (
    evaluate_brier_under_selection,
    oracle_ipw_brier_score,
    propensity_stratified_brier_score,
)
from policyshiftlab.synthetic import generate_synthetic_recommender


def test_oracle_ipw_matches_horvitz_thompson_calculation() -> None:
    y = np.array([0, 1, 0, 1], dtype=float)
    p = np.array([0.1, 0.8, 0.3, 0.6], dtype=float)
    selected = np.array([True, True, False, True])
    propensity = np.array([0.5, 0.25, 0.8, 0.5])

    observed_loss = (p[selected] - y[selected]) ** 2
    expected = np.sum(observed_loss / propensity[selected]) / y.size

    assert oracle_ipw_brier_score(
        y,
        p,
        selected,
        propensity,
    ) == pytest.approx(expected)


def test_oracle_ipw_equals_target_when_everything_is_observed() -> None:
    y = np.array([0, 1, 0, 1], dtype=float)
    p = np.array([0.1, 0.8, 0.3, 0.6], dtype=float)
    selected = np.ones(4, dtype=bool)
    propensity = np.ones(4)

    target = np.mean((p - y) ** 2)

    assert oracle_ipw_brier_score(
        y,
        p,
        selected,
        propensity,
    ) == pytest.approx(target)


def test_stratified_estimate_equals_target_when_everything_is_observed() -> None:
    y = np.array([0, 1, 0, 1], dtype=float)
    p = np.array([0.2, 0.8, 0.4, 0.6], dtype=float)
    selected = np.ones(4, dtype=bool)
    propensity = np.array([0.1, 0.2, 0.8, 0.9])

    expected = np.mean((p - y) ** 2)

    assert propensity_stratified_brier_score(
        y,
        p,
        selected,
        propensity,
        n_strata=2,
    ) == pytest.approx(expected)


def test_evaluation_returns_expected_fields() -> None:
    data = generate_synthetic_recommender(
        n_users=40,
        n_items=30,
        seed=17,
    )

    result = evaluate_brier_under_selection(
        data.outcome,
        data.true_outcome_prob,
        data.selected,
        data.propensity,
    )

    assert 0.0 <= result.target <= 1.0
    assert 0.0 <= result.logged_naive <= 1.0
    assert result.logged_oracle_ipw >= 0.0
    assert 0.0 <= result.logged_stratified <= 1.0
    assert 0.0 < result.ipw_effective_sample_size <= data.n_selected


def test_oracle_ipw_is_unbiased_over_repeated_selection_draws() -> None:
    # Hold the finite population and losses fixed. Only redraw selection.
    data = generate_synthetic_recommender(
        n_users=50,
        n_items=40,
        logging_policy="popularity",
        popularity_strength=2.5,
        exposure_strength=1.7,
        seed=123,
    )
    target = np.mean(
        (data.true_outcome_prob - data.outcome) ** 2
    )

    rng = np.random.default_rng(999)
    estimates = []

    for _ in range(1500):
        selected = rng.random(data.n_pairs) < data.propensity
        estimates.append(
            oracle_ipw_brier_score(
                data.outcome,
                data.true_outcome_prob,
                selected,
                data.propensity,
            )
        )

    # Monte Carlo standard error is small enough that 0.002 is conservative.
    assert abs(np.mean(estimates) - target) < 0.002


def test_stratified_estimate_is_finite_on_synthetic_data() -> None:
    data = generate_synthetic_recommender(
        n_users=80,
        n_items=50,
        logging_policy="popularity",
        popularity_strength=2.5,
        seed=13,
    )

    estimate = propensity_stratified_brier_score(
        data.outcome,
        data.true_outcome_prob,
        data.selected,
        data.propensity,
        n_strata=5,
    )

    assert np.isfinite(estimate)
    assert 0.0 <= estimate <= 1.0


@pytest.mark.parametrize(
    "propensity",
    [
        np.array([0.0, 0.5]),
        np.array([-0.1, 0.5]),
        np.array([0.5, 1.1]),
    ],
)
def test_invalid_propensity_is_rejected(propensity) -> None:
    with pytest.raises(ValueError):
        oracle_ipw_brier_score(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            np.array([True, True]),
            propensity,
        )


def test_empty_selected_sample_is_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate_brier_under_selection(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            np.array([False, False]),
            np.array([0.5, 0.5]),
        )
