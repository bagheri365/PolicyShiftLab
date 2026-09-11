import numpy as np
import pytest

from policyshiftlab.overlap import (
    _expected_ipw_effective_sample_size,
    run_overlap_stress_test,
)


def test_expected_ess_equals_n_for_uniform_propensity() -> None:
    propensity = np.full(100, 0.5)

    ess = _expected_ipw_effective_sample_size(propensity)

    assert ess == pytest.approx(50.0)


def test_expected_ess_drops_with_more_extreme_propensities() -> None:
    moderate = np.full(100, 0.5)
    extreme = np.concatenate(
        [np.full(50, 0.05), np.full(50, 0.95)]
    )

    assert (
        _expected_ipw_effective_sample_size(extreme)
        < _expected_ipw_effective_sample_size(moderate)
    )


def test_overlap_stress_test_returns_requested_rows() -> None:
    results = run_overlap_stress_test(
        min_propensities=(0.2, 0.05),
        exposure_strengths=(1.0, 2.0),
        n_users=40,
        n_items=20,
        n_replications=40,
        seed=7,
    )

    assert len(results) == 2

    for result in results:
        assert 0.0 < result.mean_propensity < 1.0
        assert result.min_observed_propensity > 0.0
        assert result.oracle_ipw_rmse >= 0.0
        assert result.naive_rmse >= 0.0
        assert result.stratified_rmse >= 0.0
        assert result.mean_ipw_effective_sample_size > 0.0
        assert result.expected_selected > 0.0


def test_weaker_overlap_reduces_expected_ess_in_controlled_pair() -> None:
    results = run_overlap_stress_test(
        min_propensities=(0.20, 0.02),
        exposure_strengths=(0.8, 3.0),
        n_users=80,
        n_items=40,
        n_replications=80,
        seed=11,
    )

    strong, weak = results

    assert (
        weak.mean_ipw_effective_sample_size
        < strong.mean_ipw_effective_sample_size
    )


@pytest.mark.parametrize(
    ("min_propensities", "exposure_strengths"),
    [
        ((0.1,), (1.0, 2.0)),
        ((), ()),
        ((0.0,), (1.0,)),
        ((1.0,), (1.0,)),
        ((0.1,), (0.0,)),
    ],
)
def test_invalid_overlap_configuration_is_rejected(
    min_propensities,
    exposure_strengths,
) -> None:
    with pytest.raises(ValueError):
        run_overlap_stress_test(
            min_propensities=min_propensities,
            exposure_strengths=exposure_strengths,
            n_users=20,
            n_items=10,
            n_replications=10,
        )
