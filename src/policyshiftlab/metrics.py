"""Probability-quality metrics used throughout PolicyShiftLab."""

from __future__ import annotations

import numpy as np


def _as_float_array(values: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("inputs must be one-dimensional")
    return array


def _validate_probability_inputs(
    y_true: np.ndarray | list[float],
    y_prob: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    y = _as_float_array(y_true)
    p = _as_float_array(y_prob)

    if y.shape != p.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if y.size == 0:
        raise ValueError("inputs must be non-empty")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("y_true must contain only 0/1 labels")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("y_prob values must be in [0, 1]")

    return y, p


def brier_score(
    y_true: np.ndarray | list[float],
    y_prob: np.ndarray | list[float],
    *,
    sample_weight: np.ndarray | list[float] | None = None,
) -> float:
    """Return the optionally weighted Brier score."""
    y, p = _validate_probability_inputs(y_true, y_prob)
    loss = np.square(p - y)

    if sample_weight is None:
        return float(np.mean(loss))

    w = _as_float_array(sample_weight)
    if w.shape != y.shape:
        raise ValueError("sample_weight must have the same shape as y_true")
    if np.any(w < 0.0):
        raise ValueError("sample_weight must be non-negative")

    total_weight = float(np.sum(w))
    if total_weight <= 0.0:
        raise ValueError("sample_weight must have positive total weight")

    return float(np.sum(w * loss) / total_weight)


def effective_sample_size(
    sample_weight: np.ndarray | list[float],
) -> float:
    """Return Kish effective sample size for non-negative weights."""
    w = _as_float_array(sample_weight)
    if w.size == 0:
        raise ValueError("sample_weight must be non-empty")
    if np.any(w < 0.0):
        raise ValueError("sample_weight must be non-negative")

    numerator = float(np.sum(w) ** 2)
    denominator = float(np.sum(np.square(w)))

    if denominator <= 0.0:
        raise ValueError("sample_weight must contain positive mass")

    return numerator / denominator
