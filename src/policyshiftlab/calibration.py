"""Calibration and reliability-diagram utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationBins:
    """Aggregated calibration statistics."""

    mean_predicted: np.ndarray
    fraction_positive: np.ndarray
    bin_weight: np.ndarray
    bin_count: np.ndarray


def quantile_bin_edges(
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> np.ndarray:
    """Return monotone quantile bin edges for a reference score distribution."""
    p = np.asarray(y_prob, dtype=float)

    if p.ndim != 1:
        raise ValueError("y_prob must be one-dimensional")
    if p.size == 0:
        raise ValueError("y_prob must be non-empty")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("y_prob values must be in [0, 1]")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = 0.0
    edges[-1] = 1.0

    if np.any(np.diff(edges) < 0.0):
        raise RuntimeError("quantile edges must be monotone")

    return edges


def calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    n_bins: int = 10,
    strategy: str = "quantile",
    bin_edges: np.ndarray | None = None,
) -> CalibrationBins:
    """Compute weighted or unweighted calibration-bin summaries.

    When ``bin_edges`` is supplied, those exact edges are reused. This is the
    preferred mode for comparing target, logged, and reweighted populations,
    because it avoids changing score intervals across curves.

    Weighted within-bin positive rates are ratio estimators:
    ``sum(w*y) / sum(w)``. They need not be closer to target rates in every
    finite sample.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)

    if y.ndim != 1 or p.ndim != 1:
        raise ValueError("y_true and y_prob must be one-dimensional")
    if y.shape != p.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if y.size == 0:
        raise ValueError("inputs must be non-empty")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("y_true must contain only 0/1 labels")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("y_prob values must be in [0, 1]")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if strategy not in {"quantile", "uniform"}:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    if sample_weight is None:
        w = np.ones_like(y, dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float)
        if w.ndim != 1 or w.shape != y.shape:
            raise ValueError("sample_weight must match y_true shape")
        if np.any(w < 0.0):
            raise ValueError("sample_weight must be non-negative")
        if np.sum(w) <= 0.0:
            raise ValueError("sample_weight must have positive total weight")

    if bin_edges is not None:
        edges = np.asarray(bin_edges, dtype=float)
        if edges.ndim != 1 or edges.size < 3:
            raise ValueError("bin_edges must contain at least three values")
        if np.any(np.diff(edges) < 0.0):
            raise ValueError("bin_edges must be monotone non-decreasing")
        if edges[0] > 0.0 or edges[-1] < 1.0:
            raise ValueError("bin_edges must cover the full [0, 1] range")
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = quantile_bin_edges(p, n_bins=n_bins)

    mean_predicted = []
    fraction_positive = []
    bin_weight = []
    bin_count = []

    for idx in range(edges.size - 1):
        left = edges[idx]
        right = edges[idx + 1]

        if idx == edges.size - 2:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)

        if not np.any(mask):
            continue

        local_w = w[mask]
        weight_sum = float(np.sum(local_w))
        if weight_sum <= 0.0:
            continue

        mean_predicted.append(
            float(np.sum(local_w * p[mask]) / weight_sum)
        )
        fraction_positive.append(
            float(np.sum(local_w * y[mask]) / weight_sum)
        )
        bin_weight.append(weight_sum)
        bin_count.append(int(np.sum(mask)))

    return CalibrationBins(
        mean_predicted=np.asarray(mean_predicted, dtype=float),
        fraction_positive=np.asarray(fraction_positive, dtype=float),
        bin_weight=np.asarray(bin_weight, dtype=float),
        bin_count=np.asarray(bin_count, dtype=int),
    )


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    n_bins: int = 10,
    strategy: str = "quantile",
    bin_edges: np.ndarray | None = None,
) -> float:
    """Return weighted expected calibration error.

    ECE is a descriptive diagnostic in PolicyShiftLab, not the primary metric.
    """
    bins = calibration_bins(
        y_true,
        y_prob,
        sample_weight=sample_weight,
        n_bins=n_bins,
        strategy=strategy,
        bin_edges=bin_edges,
    )

    total_weight = float(np.sum(bins.bin_weight))
    gaps = np.abs(bins.fraction_positive - bins.mean_predicted)

    return float(np.sum((bins.bin_weight / total_weight) * gaps))
