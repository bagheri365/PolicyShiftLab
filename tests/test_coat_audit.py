import numpy as np
import pytest

from policyshiftlab.coat_audit import (
    audit_coat_matrices,
    load_coat_matrix,
)


def _matrices():
    train = np.array(
        [
            [5, 0, 3, 0],
            [4, 2, 0, 0],
            [0, 1, 5, 4],
        ],
        dtype=int,
    )
    randomized = np.array(
        [
            [0, 4, 0, 2],
            [5, 0, 3, 0],
            [2, 0, 0, 5],
        ],
        dtype=int,
    )
    return train, randomized


def test_audit_counts_support_and_overlap() -> None:
    train, randomized = _matrices()

    result = audit_coat_matrices(train, randomized)

    assert result.n_users == 3
    assert result.n_items == 4
    assert result.train_observed == 7
    assert result.randomized_observed == 6
    assert result.exact_pair_overlap == 2
    assert result.train_supported_users == 3
    assert result.train_supported_items == 4
    assert result.randomized_supported_items == 4
    assert result.randomized_pairs_with_train_user_item_support == 6
    assert result.randomized_support_fraction == pytest.approx(1.0)


def test_binary_prevalence_uses_configured_threshold() -> None:
    train, randomized = _matrices()

    result = audit_coat_matrices(
        train,
        randomized,
        positive_threshold=4,
    )

    assert result.train_positive_rate == pytest.approx(4 / 7)
    assert result.randomized_positive_rate == pytest.approx(3 / 6)


def test_randomized_equal_items_per_user_is_detected() -> None:
    train, randomized = _matrices()

    result = audit_coat_matrices(train, randomized)

    assert result.randomized_user_count_is_constant
    assert result.randomized_items_per_active_user == 2


def test_rating_support_is_reported() -> None:
    train, randomized = _matrices()

    result = audit_coat_matrices(train, randomized)

    assert result.train_rating_values == (1, 2, 3, 4, 5)
    assert result.randomized_rating_values == (2, 3, 4, 5)


def test_load_matrix_round_trip(tmp_path) -> None:
    matrix = np.array([[5, 0], [0, 3]], dtype=int)
    path = tmp_path / "train.ascii"
    np.savetxt(path, matrix, fmt="%d")

    loaded = load_coat_matrix(path)

    np.testing.assert_array_equal(loaded, matrix)


@pytest.mark.parametrize(
    "train, randomized",
    [
        (np.ones((2, 2)), np.ones((2, 3))),
        (np.ones(3), np.ones(3)),
        (np.array([[1, -1]]), np.array([[1, 0]])),
    ],
)
def test_invalid_matrices_are_rejected(train, randomized) -> None:
    with pytest.raises(ValueError):
        audit_coat_matrices(train, randomized)
