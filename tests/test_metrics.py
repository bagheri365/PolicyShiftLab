import numpy as np
import pytest

from policyshiftlab.metrics import brier_score, effective_sample_size


def test_brier_score_matches_manual_calculation() -> None:
    y_true = np.array([0, 1, 1, 0], dtype=float)
    y_prob = np.array([0.1, 0.8, 0.6, 0.3], dtype=float)

    expected = np.mean((y_prob - y_true) ** 2)

    assert brier_score(y_true, y_prob) == pytest.approx(expected)


def test_weighted_brier_score_matches_manual_calculation() -> None:
    y_true = np.array([0, 1], dtype=float)
    y_prob = np.array([0.2, 0.7], dtype=float)
    weights = np.array([1.0, 3.0])

    expected = np.sum(weights * (y_prob - y_true) ** 2) / np.sum(weights)

    assert brier_score(
        y_true,
        y_prob,
        sample_weight=weights,
    ) == pytest.approx(expected)


def test_effective_sample_size_for_equal_weights() -> None:
    weights = np.ones(8)

    assert effective_sample_size(weights) == pytest.approx(8.0)


def test_effective_sample_size_decreases_for_unequal_weights() -> None:
    equal = np.ones(4)
    unequal = np.array([1.0, 1.0, 1.0, 10.0])

    assert effective_sample_size(unequal) < effective_sample_size(equal)


@pytest.mark.parametrize(
    ("y_true", "y_prob"),
    [
        ([0, 1], [0.2]),
        ([0, 2], [0.2, 0.8]),
        ([0, 1], [-0.1, 0.8]),
        ([0, 1], [0.2, 1.1]),
    ],
)
def test_brier_score_rejects_invalid_inputs(y_true, y_prob) -> None:
    with pytest.raises(ValueError):
        brier_score(y_true, y_prob)
