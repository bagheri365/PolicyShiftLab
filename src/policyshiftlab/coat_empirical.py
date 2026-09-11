"""First empirical Coat benchmark: logged vs randomized evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policyshiftlab.metrics import brier_score
from policyshiftlab.ranking import PairwiseReversal, pairwise_reversals


@dataclass(frozen=True)
class CoatModelScores:
    model: str
    randomized_brier: float
    logged_brier: float
    estimated_ht_brier: float
    estimated_snips_brier: float


@dataclass(frozen=True)
class CoatRankingSummary:
    method: str
    order: tuple[str, ...]
    spearman_vs_randomized: float
    kendall_vs_randomized: float
    reversals_vs_randomized: tuple[PairwiseReversal, ...]


@dataclass(frozen=True)
class CoatClipSensitivity:
    propensity_floor: float
    model: str
    estimated_ht_brier: float
    estimated_snips_brier: float


@dataclass(frozen=True)
class CoatEmpiricalBenchmark:
    positive_threshold: int
    n_users: int
    n_items: int
    n_logged: int
    n_randomized: int
    scores: tuple[CoatModelScores, ...]
    rankings: tuple[CoatRankingSummary, ...]
    clipping: tuple[CoatClipSensitivity, ...]


def _validate_matrix(name: str, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D matrix")
    return matrix


def _smoothed_rate(
    positive: np.ndarray,
    observed: np.ndarray,
    *,
    axis: int,
    prior_mean: float,
    prior_strength: float,
) -> np.ndarray:
    successes = np.sum(positive & observed, axis=axis, dtype=float)
    counts = np.sum(observed, axis=axis, dtype=float)
    return (
        successes + prior_strength * prior_mean
    ) / (counts + prior_strength)


def fit_logged_probability_models(
    train: np.ndarray,
    *,
    positive_threshold: int = 4,
    prior_strength: float = 10.0,
) -> dict[str, np.ndarray]:
    """Fit fixed probability estimators using logged ratings only.

    The returned matrices contain predictions for every user-item pair.
    Randomized ratings are not accepted by this function, preventing evaluation
    labels from entering model fitting.
    """
    train = _validate_matrix("train", train)
    if positive_threshold <= 0:
        raise ValueError("positive_threshold must be positive")
    if prior_strength <= 0.0:
        raise ValueError("prior_strength must be positive")

    observed = train > 0
    if not np.any(observed):
        raise ValueError("train must contain at least one observed rating")

    positive = train >= positive_threshold
    global_rate = float(np.mean(positive[observed]))

    user_rate = _smoothed_rate(
        positive,
        observed,
        axis=1,
        prior_mean=global_rate,
        prior_strength=prior_strength,
    )
    item_rate = _smoothed_rate(
        positive,
        observed,
        axis=0,
        prior_mean=global_rate,
        prior_strength=prior_strength,
    )

    n_users, n_items = train.shape
    global_pred = np.full((n_users, n_items), global_rate, dtype=float)
    user_pred = np.repeat(user_rate[:, None], n_items, axis=1)
    item_pred = np.repeat(item_rate[None, :], n_users, axis=0)

    # Fixed equal-weight blend. This is intentionally simple and predeclared;
    # the randomized test matrix is never used to tune this coefficient.
    user_item_blend = 0.5 * user_pred + 0.5 * item_pred

    return {
        "global": global_pred,
        "user_smoothed": user_pred,
        "item_smoothed": item_pred,
        "user_item_blend": user_item_blend,
    }


def _estimated_ht_brier(
    y_logged: np.ndarray,
    p_logged: np.ndarray,
    propensity_logged: np.ndarray,
    *,
    target_size: int,
) -> float:
    loss = np.square(p_logged - y_logged)
    return float(np.sum(loss / propensity_logged) / target_size)


def _estimated_snips_brier(
    y_logged: np.ndarray,
    p_logged: np.ndarray,
    propensity_logged: np.ndarray,
) -> float:
    weights = 1.0 / propensity_logged
    loss = np.square(p_logged - y_logged)
    return float(np.sum(weights * loss) / np.sum(weights))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks


def _spearman(
    reference: dict[str, float],
    comparison: dict[str, float],
) -> float:
    names = tuple(sorted(reference))
    x = _average_ranks(np.array([reference[name] for name in names]))
    y = _average_ranks(np.array([comparison[name] for name in names]))
    x -= np.mean(x)
    y -= np.mean(y)
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(x * y) / denominator)


def _kendall(
    reference: dict[str, float],
    comparison: dict[str, float],
) -> float:
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
    denominator = float(
        np.sqrt(
            (concordant + discordant + tied_reference)
            * (concordant + discordant + tied_comparison)
        )
    )
    if denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)


def _ranking_summary(
    randomized: dict[str, float],
    comparison: dict[str, float],
    *,
    method: str,
) -> CoatRankingSummary:
    order = tuple(sorted(comparison, key=lambda name: (comparison[name], name)))
    return CoatRankingSummary(
        method=method,
        order=order,
        spearman_vs_randomized=_spearman(randomized, comparison),
        kendall_vs_randomized=_kendall(randomized, comparison),
        reversals_vs_randomized=pairwise_reversals(
            randomized,
            comparison,
        ),
    )


def run_coat_empirical_benchmark(
    train: np.ndarray,
    randomized: np.ndarray,
    propensities: np.ndarray,
    *,
    positive_threshold: int = 4,
    prior_strength: float = 10.0,
    propensity_floors: tuple[float, ...] = (0.01, 0.02, 0.05),
) -> CoatEmpiricalBenchmark:
    """Evaluate logged-trained models against the randomized Coat benchmark."""
    train = _validate_matrix("train", train)
    randomized = _validate_matrix("randomized", randomized)
    propensity = np.asarray(propensities, dtype=float)

    if train.shape != randomized.shape or train.shape != propensity.shape:
        raise ValueError("train, randomized, and propensities must align")
    if np.any(~np.isfinite(propensity)):
        raise ValueError("propensities must be finite")
    if np.any((propensity <= 0.0) | (propensity > 1.0)):
        raise ValueError("propensities must lie in (0, 1]")
    if not propensity_floors:
        raise ValueError("at least one propensity floor is required")
    if any(floor <= 0.0 or floor > 1.0 for floor in propensity_floors):
        raise ValueError("propensity floors must lie in (0, 1]")

    logged_mask = train > 0
    randomized_mask = randomized > 0
    if not np.any(logged_mask) or not np.any(randomized_mask):
        raise ValueError("both matrices must contain observed ratings")

    predictions = fit_logged_probability_models(
        train,
        positive_threshold=positive_threshold,
        prior_strength=prior_strength,
    )
    y_logged = (train[logged_mask] >= positive_threshold).astype(float)
    y_randomized = (
        randomized[randomized_mask] >= positive_threshold
    ).astype(float)
    e_logged = propensity[logged_mask]
    target_size = train.size

    randomized_scores: dict[str, float] = {}
    logged_scores: dict[str, float] = {}
    ht_scores: dict[str, float] = {}
    snips_scores: dict[str, float] = {}
    score_rows: list[CoatModelScores] = []

    for name, prediction in predictions.items():
        p_logged = prediction[logged_mask]
        p_randomized = prediction[randomized_mask]

        randomized_score = brier_score(y_randomized, p_randomized)
        logged_score = brier_score(y_logged, p_logged)
        ht_score = _estimated_ht_brier(
            y_logged,
            p_logged,
            e_logged,
            target_size=target_size,
        )
        snips_score = _estimated_snips_brier(
            y_logged,
            p_logged,
            e_logged,
        )

        randomized_scores[name] = randomized_score
        logged_scores[name] = logged_score
        ht_scores[name] = ht_score
        snips_scores[name] = snips_score
        score_rows.append(
            CoatModelScores(
                model=name,
                randomized_brier=randomized_score,
                logged_brier=logged_score,
                estimated_ht_brier=ht_score,
                estimated_snips_brier=snips_score,
            )
        )

    rankings = (
        _ranking_summary(
            randomized_scores,
            randomized_scores,
            method="randomized",
        ),
        _ranking_summary(
            randomized_scores,
            logged_scores,
            method="logged_naive",
        ),
        _ranking_summary(
            randomized_scores,
            ht_scores,
            method="estimated_ht",
        ),
        _ranking_summary(
            randomized_scores,
            snips_scores,
            method="estimated_snips",
        ),
    )

    clip_rows: list[CoatClipSensitivity] = []
    for floor in propensity_floors:
        clipped = np.maximum(e_logged, floor)
        for name, prediction in predictions.items():
            p_logged = prediction[logged_mask]
            clip_rows.append(
                CoatClipSensitivity(
                    propensity_floor=float(floor),
                    model=name,
                    estimated_ht_brier=_estimated_ht_brier(
                        y_logged,
                        p_logged,
                        clipped,
                        target_size=target_size,
                    ),
                    estimated_snips_brier=_estimated_snips_brier(
                        y_logged,
                        p_logged,
                        clipped,
                    ),
                )
            )

    return CoatEmpiricalBenchmark(
        positive_threshold=positive_threshold,
        n_users=train.shape[0],
        n_items=train.shape[1],
        n_logged=int(np.sum(logged_mask)),
        n_randomized=int(np.sum(randomized_mask)),
        scores=tuple(score_rows),
        rankings=rankings,
        clipping=tuple(clip_rows),
    )
