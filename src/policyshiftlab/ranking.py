"""Model-ranking stability under policy-induced selective observation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policyshiftlab.evaluation import oracle_ipw_brier_score
from policyshiftlab.metrics import brier_score


@dataclass(frozen=True)
class PairwiseReversal:
    """A pair whose ordering differs between target and comparison evaluation."""

    model_a: str
    model_b: str
    target_prefers: str
    comparison_prefers: str


@dataclass(frozen=True)
class RankingComparison:
    """Target, logged, and oracle-IPW model-selection summaries."""

    target_scores: dict[str, float]
    logged_scores: dict[str, float]
    oracle_ipw_scores: dict[str, float]
    target_order: tuple[str, ...]
    logged_order: tuple[str, ...]
    oracle_ipw_order: tuple[str, ...]
    logged_spearman: float
    logged_kendall: float
    oracle_ipw_spearman: float
    oracle_ipw_kendall: float
    logged_reversals: tuple[PairwiseReversal, ...]
    oracle_ipw_reversals: tuple[PairwiseReversal, ...]


def _validate_predictions(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    y = np.asarray(y_true, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y_true must be a non-empty 1D array")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("y_true must contain only 0/1 labels")
    if len(predictions) < 2:
        raise ValueError("at least two candidate models are required")

    converted: dict[str, np.ndarray] = {}
    for name, values in predictions.items():
        if not name:
            raise ValueError("model names must be non-empty")
        p = np.asarray(values, dtype=float)
        if p.ndim != 1 or p.shape != y.shape:
            raise ValueError(f"predictions for {name!r} must match y_true")
        if np.any((p < 0.0) | (p > 1.0)):
            raise ValueError(f"predictions for {name!r} must lie in [0, 1]")
        converted[name] = p

    return y, converted


def _ordered_names(scores: dict[str, float]) -> tuple[str, ...]:
    return tuple(sorted(scores, key=lambda name: (scores[name], name)))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)

    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1

        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def _spearman_from_scores(
    reference: dict[str, float],
    comparison: dict[str, float],
) -> float:
    names = tuple(sorted(reference))
    x = _average_ranks(np.array([reference[name] for name in names]))
    y = _average_ranks(np.array([comparison[name] for name in names]))

    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = float(
        np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    )
    if denominator == 0.0:
        return float("nan")

    return float(np.sum(x_centered * y_centered) / denominator)


def _kendall_from_scores(
    reference: dict[str, float],
    comparison: dict[str, float],
) -> float:
    """Return Kendall tau-b over all model pairs."""
    names = tuple(sorted(reference))
    concordant = 0
    discordant = 0
    tied_reference = 0
    tied_comparison = 0

    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            ref_diff = reference[name_a] - reference[name_b]
            cmp_diff = comparison[name_a] - comparison[name_b]

            if ref_diff == 0.0 and cmp_diff == 0.0:
                continue
            if ref_diff == 0.0:
                tied_reference += 1
                continue
            if cmp_diff == 0.0:
                tied_comparison += 1
                continue

            if ref_diff * cmp_diff > 0.0:
                concordant += 1
            else:
                discordant += 1

    numerator = concordant - discordant
    denominator = np.sqrt(
        (concordant + discordant + tied_reference)
        * (concordant + discordant + tied_comparison)
    )
    if denominator == 0.0:
        return float("nan")

    return float(numerator / denominator)


def pairwise_reversals(
    target_scores: dict[str, float],
    comparison_scores: dict[str, float],
) -> tuple[PairwiseReversal, ...]:
    """Return pairs whose strict ordering reverses relative to target."""
    if set(target_scores) != set(comparison_scores):
        raise ValueError("score dictionaries must contain the same model names")

    names = tuple(sorted(target_scores))
    reversals: list[PairwiseReversal] = []

    for i, model_a in enumerate(names):
        for model_b in names[i + 1 :]:
            target_diff = target_scores[model_a] - target_scores[model_b]
            comparison_diff = (
                comparison_scores[model_a] - comparison_scores[model_b]
            )

            if target_diff * comparison_diff >= 0.0:
                continue

            reversals.append(
                PairwiseReversal(
                    model_a=model_a,
                    model_b=model_b,
                    target_prefers=(
                        model_a if target_diff < 0.0 else model_b
                    ),
                    comparison_prefers=(
                        model_a if comparison_diff < 0.0 else model_b
                    ),
                )
            )

    return tuple(reversals)


def compare_model_rankings(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    selected: np.ndarray,
    propensity: np.ndarray,
) -> RankingComparison:
    """Compare target, naive logged, and oracle-IPW Brier rankings."""
    y, preds = _validate_predictions(y_true, predictions)
    s = np.asarray(selected, dtype=bool)
    e = np.asarray(propensity, dtype=float)

    if s.ndim != 1 or s.shape != y.shape:
        raise ValueError("selected must match y_true")
    if e.ndim != 1 or e.shape != y.shape:
        raise ValueError("propensity must match y_true")
    if not np.any(s):
        raise ValueError("at least one observation must be selected")
    if np.any((e <= 0.0) | (e > 1.0)):
        raise ValueError("propensity values must lie in (0, 1]")

    target_scores = {
        name: brier_score(y, p)
        for name, p in preds.items()
    }
    logged_scores = {
        name: brier_score(y[s], p[s])
        for name, p in preds.items()
    }
    oracle_ipw_scores = {
        name: oracle_ipw_brier_score(y, p, s, e)
        for name, p in preds.items()
    }

    return RankingComparison(
        target_scores=target_scores,
        logged_scores=logged_scores,
        oracle_ipw_scores=oracle_ipw_scores,
        target_order=_ordered_names(target_scores),
        logged_order=_ordered_names(logged_scores),
        oracle_ipw_order=_ordered_names(oracle_ipw_scores),
        logged_spearman=_spearman_from_scores(target_scores, logged_scores),
        logged_kendall=_kendall_from_scores(target_scores, logged_scores),
        oracle_ipw_spearman=_spearman_from_scores(
            target_scores, oracle_ipw_scores
        ),
        oracle_ipw_kendall=_kendall_from_scores(
            target_scores, oracle_ipw_scores
        ),
        logged_reversals=pairwise_reversals(
            target_scores, logged_scores
        ),
        oracle_ipw_reversals=pairwise_reversals(
            target_scores, oracle_ipw_scores
        ),
    )


def fixed_candidate_predictions(
    true_outcome_prob: np.ndarray,
    logging_score: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return a predeclared deterministic candidate-model set.

    The candidates have different error profiles across logging-score regions.
    Selective observation can therefore alter aggregate model rankings without
    changing the conditional outcome mechanism P(Y|X).
    """
    p = np.asarray(true_outcome_prob, dtype=float)
    score = np.asarray(logging_score, dtype=float)

    if p.ndim != 1 or score.ndim != 1 or p.shape != score.shape:
        raise ValueError(
            "true_outcome_prob and logging_score must be matching 1D arrays"
        )
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("true_outcome_prob must lie in [0, 1]")

    median_score = float(np.median(score))
    high = score >= median_score
    low = ~high

    def clip(values: np.ndarray) -> np.ndarray:
        return np.clip(values, 1e-6, 1.0 - 1e-6)

    high_region_robust = p.copy()
    high_region_robust[low] = 0.5 + 0.55 * (p[low] - 0.5)

    low_region_robust = p.copy()
    low_region_robust[high] = 0.5 + 0.55 * (p[high] - 0.5)

    return {
        "oracle": clip(p),
        "underconfident": clip(0.5 + 0.70 * (p - 0.5)),
        "overconfident": clip(0.5 + 1.20 * (p - 0.5)),
        "high_region_robust": clip(high_region_robust),
        "low_region_robust": clip(low_region_robust),
    }
