"""Support and composition audit for the Coat explicit-rating dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CoatAuditSummary:
    """Dataset support/composition diagnostics."""

    n_users: int
    n_items: int
    train_observed: int
    randomized_observed: int
    exact_pair_overlap: int
    train_supported_users: int
    train_supported_items: int
    randomized_supported_users: int
    randomized_supported_items: int
    randomized_pairs_with_train_user_item_support: int
    randomized_support_fraction: float
    train_rating_values: tuple[int, ...]
    randomized_rating_values: tuple[int, ...]
    train_positive_rate: float
    randomized_positive_rate: float
    train_user_activity_min: int
    train_user_activity_median: float
    train_user_activity_max: int
    randomized_user_activity_min: int
    randomized_user_activity_median: float
    randomized_user_activity_max: int
    train_item_activity_min: int
    train_item_activity_median: float
    train_item_activity_max: int
    randomized_item_activity_min: int
    randomized_item_activity_median: float
    randomized_item_activity_max: int
    randomized_user_count_is_constant: bool
    randomized_items_per_active_user: int | None
    randomized_observation_fraction: float


def load_coat_matrix(path: str | Path) -> np.ndarray:
    """Load one Coat dense rating matrix.

    The original Coat files use positive integers for observed ratings and zero
    for unobserved user-item pairs.
    """
    matrix = np.loadtxt(Path(path), dtype=int)

    if matrix.ndim != 2:
        raise ValueError("Coat matrix must be two-dimensional")
    if matrix.size == 0:
        raise ValueError("Coat matrix must be non-empty")
    if np.any(matrix < 0):
        raise ValueError("Coat ratings/missingness codes must be non-negative")

    return matrix


def _observed_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[matrix > 0]


def _activity_stats(mask: np.ndarray, axis: int) -> tuple[int, float, int]:
    counts = np.sum(mask, axis=axis)
    active = counts[counts > 0]
    if active.size == 0:
        return 0, 0.0, 0
    return (
        int(np.min(active)),
        float(np.median(active)),
        int(np.max(active)),
    )


def _positive_rate(values: np.ndarray, threshold: int) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values >= threshold))


def audit_coat_matrices(
    train: np.ndarray,
    randomized: np.ndarray,
    *,
    positive_threshold: int = 4,
) -> CoatAuditSummary:
    """Audit support and composition of Coat biased/randomized matrices."""
    train = np.asarray(train, dtype=int)
    randomized = np.asarray(randomized, dtype=int)

    if train.ndim != 2 or randomized.ndim != 2:
        raise ValueError("train and randomized matrices must be 2D")
    if train.shape != randomized.shape:
        raise ValueError("train and randomized matrices must have same shape")
    if train.size == 0:
        raise ValueError("matrices must be non-empty")
    if np.any(train < 0) or np.any(randomized < 0):
        raise ValueError("matrix entries must be non-negative")
    if positive_threshold <= 0:
        raise ValueError("positive_threshold must be positive")

    train_mask = train > 0
    randomized_mask = randomized > 0
    train_values = _observed_values(train)
    randomized_values = _observed_values(randomized)

    train_user_activity = np.sum(train_mask, axis=1)
    train_item_activity = np.sum(train_mask, axis=0)
    randomized_user_activity = np.sum(randomized_mask, axis=1)

    train_user_supported = train_user_activity > 0
    train_item_supported = train_item_activity > 0
    randomized_user_supported = randomized_user_activity > 0
    randomized_item_supported = np.sum(randomized_mask, axis=0) > 0

    supported_pair_mask = (
        train_user_supported[:, None] & train_item_supported[None, :]
    )
    randomized_supported_pairs = int(
        np.sum(randomized_mask & supported_pair_mask)
    )
    randomized_observed = int(np.sum(randomized_mask))
    support_fraction = (
        float(randomized_supported_pairs / randomized_observed)
        if randomized_observed
        else float("nan")
    )

    active_randomized_counts = randomized_user_activity[
        randomized_user_activity > 0
    ]
    count_is_constant = bool(
        active_randomized_counts.size > 0
        and np.all(active_randomized_counts == active_randomized_counts[0])
    )
    items_per_active_user = (
        int(active_randomized_counts[0])
        if count_is_constant
        else None
    )

    train_user_stats = _activity_stats(train_mask, axis=1)
    randomized_user_stats = _activity_stats(randomized_mask, axis=1)
    train_item_stats = _activity_stats(train_mask, axis=0)
    randomized_item_stats = _activity_stats(randomized_mask, axis=0)

    n_users, n_items = train.shape

    return CoatAuditSummary(
        n_users=n_users,
        n_items=n_items,
        train_observed=int(np.sum(train_mask)),
        randomized_observed=randomized_observed,
        exact_pair_overlap=int(np.sum(train_mask & randomized_mask)),
        train_supported_users=int(np.sum(train_user_supported)),
        train_supported_items=int(np.sum(train_item_supported)),
        randomized_supported_users=int(np.sum(randomized_user_supported)),
        randomized_supported_items=int(np.sum(randomized_item_supported)),
        randomized_pairs_with_train_user_item_support=randomized_supported_pairs,
        randomized_support_fraction=support_fraction,
        train_rating_values=tuple(int(x) for x in np.unique(train_values)),
        randomized_rating_values=tuple(
            int(x) for x in np.unique(randomized_values)
        ),
        train_positive_rate=_positive_rate(
            train_values, positive_threshold
        ),
        randomized_positive_rate=_positive_rate(
            randomized_values, positive_threshold
        ),
        train_user_activity_min=train_user_stats[0],
        train_user_activity_median=train_user_stats[1],
        train_user_activity_max=train_user_stats[2],
        randomized_user_activity_min=randomized_user_stats[0],
        randomized_user_activity_median=randomized_user_stats[1],
        randomized_user_activity_max=randomized_user_stats[2],
        train_item_activity_min=train_item_stats[0],
        train_item_activity_median=train_item_stats[1],
        train_item_activity_max=train_item_stats[2],
        randomized_item_activity_min=randomized_item_stats[0],
        randomized_item_activity_median=randomized_item_stats[1],
        randomized_item_activity_max=randomized_item_stats[2],
        randomized_user_count_is_constant=count_is_constant,
        randomized_items_per_active_user=items_per_active_user,
        randomized_observation_fraction=float(
            randomized_observed / (n_users * n_items)
        ),
    )


def audit_coat_files(
    train_path: str | Path,
    randomized_path: str | Path,
    *,
    positive_threshold: int = 4,
) -> CoatAuditSummary:
    """Load and audit the two original Coat rating matrices."""
    train = load_coat_matrix(train_path)
    randomized = load_coat_matrix(randomized_path)
    return audit_coat_matrices(
        train,
        randomized,
        positive_threshold=positive_threshold,
    )
