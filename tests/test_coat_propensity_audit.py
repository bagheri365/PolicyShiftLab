import numpy as np
import pytest

from policyshiftlab.coat_propensity_audit import (
    audit_coat_propensities,
    load_coat_propensities,
)


def _inputs():
    train = np.array(
        [
            [5, 0, 4],
            [0, 2, 0],
        ]
    )
    randomized = np.array(
        [
            [0, 3, 0],
            [5, 0, 4],
        ]
    )
    prop = np.array(
        [
            [0.5, 0.1, 0.4],
            [0.2, 0.3, 0.5],
        ]
    )
    return train, randomized, prop


def test_propensity_audit_reports_selection_concentration() -> None:
    train, randomized, prop = _inputs()

    result = audit_coat_propensities(train, randomized, prop)

    assert result.n_users == 2
    assert result.n_items == 3
    assert result.mean_propensity == pytest.approx(1 / 3)
    assert result.expected_row_observations == pytest.approx(1.0)
    assert result.train_observed_mean_propensity == pytest.approx(0.4)
    assert result.train_unobserved_mean_propensity == pytest.approx(
        (0.1 + 0.2 + 0.5) / 3
    )
    assert result.randomized_observed_mean_propensity == pytest.approx(
        (0.1 + 0.2 + 0.5) / 3
    )


def test_inverse_weight_summary_is_finite() -> None:
    train, randomized, prop = _inputs()

    result = audit_coat_propensities(train, randomized, prop)

    assert result.train_observed_weight_min == pytest.approx(2.0)
    assert result.train_observed_weight_max == pytest.approx(10 / 3)
    assert 0 < result.train_observed_weight_ess <= 3


def test_load_propensities_round_trip(tmp_path) -> None:
    prop = np.array([[0.1, 0.2], [0.3, 0.4]])
    path = tmp_path / "propensities.ascii"
    np.savetxt(path, prop)

    loaded = load_coat_propensities(path)

    np.testing.assert_allclose(loaded, prop)


@pytest.mark.parametrize(
    "prop",
    [
        np.array([[0.0, 0.2]]),
        np.array([[1.1, 0.2]]),
        np.array([[np.nan, 0.2]]),
    ],
)
def test_invalid_propensities_are_rejected(prop) -> None:
    train = np.ones_like(prop)
    randomized = np.ones_like(prop)

    with pytest.raises(ValueError):
        audit_coat_propensities(train, randomized, prop)


def test_shape_mismatch_is_rejected() -> None:
    train = np.ones((2, 2))
    randomized = np.ones((2, 2))
    prop = np.ones((2, 3)) * 0.5

    with pytest.raises(ValueError):
        audit_coat_propensities(train, randomized, prop)
