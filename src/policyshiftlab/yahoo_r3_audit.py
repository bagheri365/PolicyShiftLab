"""Support and composition audit for the Yahoo! R3 dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class YahooR3AuditSummary:
    train_observations: int
    randomized_observations: int
    train_users: int
    randomized_users: int
    shared_users: int
    train_items: int
    randomized_items: int
    shared_items: int
    exact_pair_overlap: int
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


def _detect_delimiter(path: Path) -> str | None:
    """Return comma delimiter for CSV-like copies, else whitespace parsing."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                return "," if "," in stripped else None
    raise ValueError("Yahoo! R3 data must be non-empty")


def load_yahoo_r3_triplets(path: str | Path) -> np.ndarray:
    """Load Yahoo! R3 ``user item rating`` triplets.

    Both the original whitespace-separated Webscope files and common
    comma-separated research copies are supported. User/item identifiers may
    be zero- or one-based, but must be non-negative.
    """
    path = Path(path)
    delimiter = _detect_delimiter(path)
    values = np.loadtxt(path, dtype=int, delimiter=delimiter)

    if values.ndim == 1:
        if values.size != 3:
            raise ValueError("Yahoo! R3 rows must contain exactly three columns")
        values = values.reshape(1, 3)

    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Yahoo! R3 data must have exactly three columns")
    if values.shape[0] == 0:
        raise ValueError("Yahoo! R3 data must be non-empty")
    if np.any(values[:, :2] < 0):
        raise ValueError("user and item ids must be non-negative")
    if np.any(values[:, 2] <= 0):
        raise ValueError("ratings must be positive")

    pairs = values[:, :2]
    if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
        raise ValueError("duplicate user-item pairs are not supported")

    return values


def _count_by_id(ids: np.ndarray) -> np.ndarray:
    _, counts = np.unique(ids, return_counts=True)
    return counts


def _activity_stats(counts: np.ndarray) -> tuple[int, float, int]:
    if counts.size == 0:
        return 0, 0.0, 0
    return int(np.min(counts)), float(np.median(counts)), int(np.max(counts))


def _pair_set(data: np.ndarray) -> set[tuple[int, int]]:
    return {(int(user), int(item)) for user, item in data[:, :2]}


def audit_yahoo_r3(
    train: np.ndarray,
    randomized: np.ndarray,
    *,
    positive_threshold: int = 4,
) -> YahooR3AuditSummary:
    """Audit support and composition of Yahoo! R3 logged/randomized triplets."""
    train = np.asarray(train, dtype=int)
    randomized = np.asarray(randomized, dtype=int)

    for name, values in (("train", train), ("randomized", randomized)):
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"{name} must have shape (n, 3)")
        if values.shape[0] == 0:
            raise ValueError(f"{name} must be non-empty")
        if np.any(values[:, :2] < 0) or np.any(values[:, 2] <= 0):
            raise ValueError(
                f"{name} user/item ids must be non-negative and ratings positive"
            )
        if np.unique(values[:, :2], axis=0).shape[0] != values.shape[0]:
            raise ValueError(f"{name} contains duplicate user-item pairs")

    if positive_threshold <= 0:
        raise ValueError("positive_threshold must be positive")

    train_users = set(int(x) for x in np.unique(train[:, 0]))
    random_users = set(int(x) for x in np.unique(randomized[:, 0]))
    train_items = set(int(x) for x in np.unique(train[:, 1]))
    random_items = set(int(x) for x in np.unique(randomized[:, 1]))

    train_pairs = _pair_set(train)
    random_pairs = _pair_set(randomized)

    supported_random_pairs = sum(
        1
        for user, item in random_pairs
        if user in train_users and item in train_items
    )

    random_user_counts = _count_by_id(randomized[:, 0])
    count_is_constant = bool(
        random_user_counts.size
        and np.all(random_user_counts == random_user_counts[0])
    )
    items_per_user = int(random_user_counts[0]) if count_is_constant else None

    train_user_stats = _activity_stats(_count_by_id(train[:, 0]))
    random_user_stats = _activity_stats(random_user_counts)
    train_item_stats = _activity_stats(_count_by_id(train[:, 1]))
    random_item_stats = _activity_stats(_count_by_id(randomized[:, 1]))

    return YahooR3AuditSummary(
        train_observations=int(train.shape[0]),
        randomized_observations=int(randomized.shape[0]),
        train_users=len(train_users),
        randomized_users=len(random_users),
        shared_users=len(train_users & random_users),
        train_items=len(train_items),
        randomized_items=len(random_items),
        shared_items=len(train_items & random_items),
        exact_pair_overlap=len(train_pairs & random_pairs),
        randomized_pairs_with_train_user_item_support=supported_random_pairs,
        randomized_support_fraction=float(
            supported_random_pairs / randomized.shape[0]
        ),
        train_rating_values=tuple(int(x) for x in np.unique(train[:, 2])),
        randomized_rating_values=tuple(
            int(x) for x in np.unique(randomized[:, 2])
        ),
        train_positive_rate=float(
            np.mean(train[:, 2] >= positive_threshold)
        ),
        randomized_positive_rate=float(
            np.mean(randomized[:, 2] >= positive_threshold)
        ),
        train_user_activity_min=train_user_stats[0],
        train_user_activity_median=train_user_stats[1],
        train_user_activity_max=train_user_stats[2],
        randomized_user_activity_min=random_user_stats[0],
        randomized_user_activity_median=random_user_stats[1],
        randomized_user_activity_max=random_user_stats[2],
        train_item_activity_min=train_item_stats[0],
        train_item_activity_median=train_item_stats[1],
        train_item_activity_max=train_item_stats[2],
        randomized_item_activity_min=random_item_stats[0],
        randomized_item_activity_median=random_item_stats[1],
        randomized_item_activity_max=random_item_stats[2],
        randomized_user_count_is_constant=count_is_constant,
        randomized_items_per_active_user=items_per_user,
    )


def audit_yahoo_r3_files(
    train_path: str | Path,
    randomized_path: str | Path,
    *,
    positive_threshold: int = 4,
) -> YahooR3AuditSummary:
    """Load and audit Yahoo! R3 train/randomized files."""
    return audit_yahoo_r3(
        load_yahoo_r3_triplets(train_path),
        load_yahoo_r3_triplets(randomized_path),
        positive_threshold=positive_threshold,
    )
