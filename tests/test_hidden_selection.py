import numpy as np
import pytest

from policyshiftlab.hidden_selection import (
    generate_hidden_selection_data,
    run_hidden_selection_monte_carlo,
)


def test_hidden_selection_data_has_valid_propensities() -> None:
    data = generate_hidden_selection_data(
        n_users=40,
        n_items=20,
        seed=2,
    )

    assert data.outcome.shape == data.observed_prediction.shape
    assert data.full_propensity.shape == data.outcome.shape
    assert data.observed_only_propensity.shape == data.outcome.shape
    assert np.all((data.full_propensity > 0.0) & (data.full_propensity <= 1.0))
    assert np.all(
        (data.observed_only_propensity > 0.0)
        & (data.observed_only_propensity <= 1.0)
    )


def test_hidden_variable_changes_full_propensity_beyond_observed_propensity() -> None:
    data = generate_hidden_selection_data(
        n_users=60,
        n_items=20,
        selection_hidden_strength=2.0,
        seed=4,
    )

    difference = np.abs(
        data.full_propensity - data.observed_only_propensity
    )

    assert float(np.mean(difference)) > 0.02


def test_full_ipw_is_nearly_unbiased_over_repeated_selection() -> None:
    data = generate_hidden_selection_data(
        n_users=100,
        n_items=40,
        outcome_hidden_strength=1.2,
        selection_hidden_strength=1.8,
        seed=5,
    )

    result = run_hidden_selection_monte_carlo(
        data,
        n_replications=500,
        seed=99,
    )

    assert abs(result.full_ipw.bias) < 0.003


def test_observed_only_ipw_remains_biased_in_failure_case() -> None:
    data = generate_hidden_selection_data(
        n_users=120,
        n_items=40,
        outcome_hidden_strength=1.5,
        selection_hidden_strength=2.0,
        seed=7,
    )

    result = run_hidden_selection_monte_carlo(
        data,
        n_replications=500,
        seed=101,
    )

    assert abs(result.observed_only_ipw.bias) > abs(result.full_ipw.bias)
    assert abs(result.observed_only_ipw.bias) > 0.005


def test_fixed_seed_is_reproducible() -> None:
    data = generate_hidden_selection_data(
        n_users=40,
        n_items=20,
        seed=8,
    )

    first = run_hidden_selection_monte_carlo(
        data,
        n_replications=50,
        seed=12,
    )
    second = run_hidden_selection_monte_carlo(
        data,
        n_replications=50,
        seed=12,
    )

    assert first == second


@pytest.mark.parametrize(
    "kwargs",
    [
        {"outcome_hidden_strength": 0.0},
        {"selection_hidden_strength": 0.0},
        {"exposure_strength": 0.0},
        {"min_propensity": 0.9, "max_propensity": 0.8},
    ],
)
def test_invalid_configuration_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        generate_hidden_selection_data(
            n_users=20,
            n_items=10,
            **kwargs,
        )
