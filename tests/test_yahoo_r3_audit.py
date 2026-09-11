import numpy as np
import pytest

from policyshiftlab.yahoo_r3_audit import (
    audit_yahoo_r3,
    load_yahoo_r3_triplets,
)


def _data():
    train = np.array(
        [
            [0, 10, 5],
            [0, 11, 3],
            [1, 10, 4],
            [1, 12, 2],
            [2, 13, 5],
        ],
        dtype=int,
    )
    randomized = np.array(
        [
            [0, 12, 4],
            [0, 13, 2],
            [1, 11, 5],
            [1, 14, 1],
            [3, 10, 3],
            [3, 13, 4],
        ],
        dtype=int,
    )
    return train, randomized


def test_support_and_overlap_counts() -> None:
    train, randomized = _data()
    result = audit_yahoo_r3(train, randomized)

    assert result.train_observations == 5
    assert result.randomized_observations == 6
    assert result.train_users == 3
    assert result.randomized_users == 3
    assert result.shared_users == 2
    assert result.train_items == 4
    assert result.randomized_items == 5
    assert result.shared_items == 4
    assert result.exact_pair_overlap == 0
    assert result.randomized_pairs_with_train_user_item_support == 3
    assert result.randomized_support_fraction == pytest.approx(0.5)


def test_rating_support_and_binary_prevalence() -> None:
    train, randomized = _data()
    result = audit_yahoo_r3(train, randomized, positive_threshold=4)

    assert result.train_rating_values == (2, 3, 4, 5)
    assert result.randomized_rating_values == (1, 2, 3, 4, 5)
    assert result.train_positive_rate == pytest.approx(3 / 5)
    assert result.randomized_positive_rate == pytest.approx(3 / 6)


def test_constant_randomized_items_per_user_is_detected() -> None:
    train, randomized = _data()
    result = audit_yahoo_r3(train, randomized)

    assert result.randomized_user_count_is_constant
    assert result.randomized_items_per_active_user == 2


def test_loader_accepts_whitespace_one_based_copy(tmp_path) -> None:
    data = np.array([[1, 2, 5], [2, 3, 4]], dtype=int)
    path = tmp_path / "train.txt"
    np.savetxt(path, data, fmt="%d")

    loaded = load_yahoo_r3_triplets(path)

    np.testing.assert_array_equal(loaded, data)


def test_loader_accepts_comma_zero_based_copy(tmp_path) -> None:
    data = np.array([[0, 2, 5], [1, 3, 4]], dtype=int)
    path = tmp_path / "user.txt"
    np.savetxt(path, data, fmt="%d", delimiter=",")

    loaded = load_yahoo_r3_triplets(path)

    np.testing.assert_array_equal(loaded, data)


def test_loader_accepts_single_row(tmp_path) -> None:
    path = tmp_path / "one.txt"
    path.write_text("0,2,5\n", encoding="utf-8")

    loaded = load_yahoo_r3_triplets(path)

    assert loaded.shape == (1, 3)


@pytest.mark.parametrize(
    "values",
    [
        np.array([[1, 2], [2, 3]]),
        np.array([[-1, 2, 5]]),
        np.array([[1, 2, 0]]),
        np.array([[1, 2, 5], [1, 2, 4]]),
    ],
)
def test_invalid_inputs_are_rejected(values) -> None:
    train, _ = _data()

    with pytest.raises(ValueError):
        audit_yahoo_r3(values, train)
