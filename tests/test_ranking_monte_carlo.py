import numpy as np
import pytest

from policyshiftlab.ranking_monte_carlo import run_ranking_monte_carlo


def _fixture():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=float)
    propensity = np.array([0.2, 0.3, 0.8, 0.9, 0.25, 0.85, 0.4, 0.75])
    predictions = {
        "a": np.array([0.1, 0.2, 0.8, 0.9, 0.2, 0.8, 0.3, 0.7]),
        "b": np.array([0.2, 0.3, 0.7, 0.8, 0.3, 0.7, 0.4, 0.6]),
        "c": np.full(8, 0.5),
    }
    return y, predictions, propensity


def test_ranking_monte_carlo_returns_both_methods() -> None:
    y, predictions, propensity = _fixture()

    result = run_ranking_monte_carlo(
        y,
        predictions,
        propensity,
        n_replications=50,
        seed=3,
    )

    assert result.logged.method == "logged"
    assert result.oracle_ipw.method == "oracle_ipw"
    assert result.logged.n_replications == 50
    assert result.oracle_ipw.n_replications == 50


def test_rates_and_correlations_are_in_valid_ranges() -> None:
    y, predictions, propensity = _fixture()

    result = run_ranking_monte_carlo(
        y,
        predictions,
        propensity,
        n_replications=60,
        seed=4,
    )

    for summary in (result.logged, result.oracle_ipw):
        assert -1.0 <= summary.mean_spearman <= 1.0
        assert -1.0 <= summary.mean_kendall <= 1.0
        assert 0.0 <= summary.any_reversal_rate <= 1.0
        assert summary.mean_reversal_count >= 0.0
        assert 0.0 <= summary.exact_order_recovery_rate <= 1.0


def test_fixed_seed_is_reproducible() -> None:
    y, predictions, propensity = _fixture()

    first = run_ranking_monte_carlo(
        y,
        predictions,
        propensity,
        n_replications=40,
        seed=10,
    )
    second = run_ranking_monte_carlo(
        y,
        predictions,
        propensity,
        n_replications=40,
        seed=10,
    )

    assert first == second


def test_full_observation_recovers_target_order_every_time() -> None:
    y, predictions, _ = _fixture()
    propensity = np.ones_like(y)

    result = run_ranking_monte_carlo(
        y,
        predictions,
        propensity,
        n_replications=20,
        seed=1,
    )

    assert result.logged.any_reversal_rate == pytest.approx(0.0)
    assert result.logged.exact_order_recovery_rate == pytest.approx(1.0)
    assert result.oracle_ipw.any_reversal_rate == pytest.approx(0.0)
    assert result.oracle_ipw.exact_order_recovery_rate == pytest.approx(1.0)


@pytest.mark.parametrize("n_replications", [0, 1])
def test_invalid_replication_count_is_rejected(n_replications) -> None:
    y, predictions, propensity = _fixture()

    with pytest.raises(ValueError):
        run_ranking_monte_carlo(
            y,
            predictions,
            propensity,
            n_replications=n_replications,
        )
