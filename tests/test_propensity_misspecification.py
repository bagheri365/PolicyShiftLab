import numpy as np
import pytest

from policyshiftlab.propensity_misspecification import (
    _ranking_summary_under_misspecification,
    make_misspecified_propensities,
    run_propensity_misspecification_experiment,
)


def test_misspecified_propensities_are_valid() -> None:
    e = np.linspace(0.05, 0.95, 100)
    score = np.linspace(-3.0, 3.0, 100)

    scenarios = make_misspecified_propensities(e, score)

    assert set(scenarios) == {"oracle", "mild", "severe"}
    np.testing.assert_allclose(scenarios["oracle"], e)

    for values in scenarios.values():
        assert np.all((values > 0.0) & (values <= 1.0))


def test_mild_specification_is_closer_to_truth_than_severe() -> None:
    e = np.linspace(0.05, 0.95, 200)
    score = np.linspace(-4.0, 4.0, 200)

    scenarios = make_misspecified_propensities(e, score)

    mild_error = np.mean(np.abs(scenarios["mild"] - e))
    severe_error = np.mean(np.abs(scenarios["severe"] - e))

    assert mild_error < severe_error


def test_ranking_summary_holds_true_selection_policy_fixed() -> None:
    y = np.array([0, 0, 1, 1, 0, 1], dtype=float)
    predictions = {
        "a": np.array([0.1, 0.2, 0.8, 0.9, 0.2, 0.8]),
        "b": np.array([0.2, 0.3, 0.7, 0.8, 0.3, 0.7]),
    }
    true_e = np.array([0.2, 0.3, 0.8, 0.9, 0.25, 0.85])
    supplied = np.full(6, 0.5)

    first = _ranking_summary_under_misspecification(
        y,
        predictions,
        true_e,
        supplied,
        n_replications=50,
        seed=9,
    )
    second = _ranking_summary_under_misspecification(
        y,
        predictions,
        true_e,
        supplied,
        n_replications=50,
        seed=9,
    )

    assert first == second


def test_experiment_returns_all_scenarios() -> None:
    result = run_propensity_misspecification_experiment(
        n_users=50,
        n_items=20,
        n_replications=40,
        seed=7,
    )

    assert np.isfinite(result.target_brier)
    assert np.isfinite(result.naive_logged_brier)

    for summary in (
        result.oracle,
        result.mild_misspecified,
        result.severe_misspecified,
    ):
        assert np.isfinite(summary.brier_estimate)
        assert np.isfinite(summary.brier_error)
        assert -1.0 <= summary.mean_spearman <= 1.0
        assert -1.0 <= summary.mean_kendall <= 1.0
        assert 0.0 <= summary.any_reversal_rate <= 1.0
        assert 0.0 <= summary.exact_order_recovery_rate <= 1.0


def test_fixed_seed_is_reproducible() -> None:
    first = run_propensity_misspecification_experiment(
        n_users=40,
        n_items=20,
        n_replications=30,
        seed=11,
    )
    second = run_propensity_misspecification_experiment(
        n_users=40,
        n_items=20,
        n_replications=30,
        seed=11,
    )

    assert first == second


def test_invalid_propensity_shape_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_misspecified_propensities(
            np.array([0.2, 0.8]),
            np.array([0.0]),
        )
