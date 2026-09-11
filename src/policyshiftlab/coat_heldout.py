"""Leakage-safe Coat benchmark using held-out logged evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policyshiftlab.coat_empirical import fit_logged_probability_models
from policyshiftlab.metrics import brier_score
from policyshiftlab.ranking import PairwiseReversal, pairwise_reversals


@dataclass(frozen=True)
class LoggedHoldoutSplit:
    fit_matrix: np.ndarray
    evaluation_mask: np.ndarray
    evaluation_probability: np.ndarray
    n_fit: int
    n_evaluation: int


@dataclass(frozen=True)
class HeldoutModelScores:
    model: str
    randomized_brier: float
    heldout_logged_brier: float
    estimated_ht_brier: float
    estimated_snips_brier: float


@dataclass(frozen=True)
class HeldoutRankingSummary:
    method: str
    order: tuple[str, ...]
    spearman_vs_randomized: float
    kendall_vs_randomized: float
    reversals_vs_randomized: tuple[PairwiseReversal, ...]


@dataclass(frozen=True)
class HeldoutClipSensitivity:
    propensity_floor: float
    model: str
    estimated_ht_brier: float
    estimated_snips_brier: float


@dataclass(frozen=True)
class CoatHeldoutBenchmark:
    positive_threshold: int
    seed: int
    holdout_fraction: float
    n_users: int
    n_items: int
    n_logged_total: int
    n_logged_fit: int
    n_logged_evaluation: int
    n_randomized: int
    scores: tuple[HeldoutModelScores, ...]
    rankings: tuple[HeldoutRankingSummary, ...]
    clipping: tuple[HeldoutClipSensitivity, ...]


def split_logged_ratings_by_user(
    train: np.ndarray,
    *,
    holdout_fraction: float = 1.0 / 3.0,
    seed: int = 2026,
) -> LoggedHoldoutSplit:
    """Randomly hold out logged ratings within each user.

    For a user with ``n`` observed logged ratings, exactly ``h`` are sampled
    without replacement for evaluation, where ``h`` is the rounded requested
    fraction constrained to leave at least one fit observation when possible.

    Each observed pair for that user therefore has known conditional evaluation
    probability ``h / n`` given that it was logged.
    """
    train = np.asarray(train)
    if train.ndim != 2 or train.size == 0:
        raise ValueError("train must be a non-empty 2D matrix")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must lie in (0, 1)")

    observed = train > 0
    if not np.any(observed):
        raise ValueError("train must contain observed ratings")

    rng = np.random.default_rng(seed)
    evaluation_mask = np.zeros(train.shape, dtype=bool)
    evaluation_probability = np.zeros(train.shape, dtype=float)

    for user in range(train.shape[0]):
        items = np.flatnonzero(observed[user])
        n_observed = items.size
        if n_observed == 0:
            continue
        if n_observed == 1:
            # Keep the only rating in fitting. This user contributes no held-out
            # logged evaluation observations.
            continue

        n_holdout = int(np.rint(holdout_fraction * n_observed))
        n_holdout = min(max(n_holdout, 1), n_observed - 1)
        chosen = rng.choice(items, size=n_holdout, replace=False)
        evaluation_mask[user, chosen] = True
        evaluation_probability[user, items] = n_holdout / n_observed

    fit_matrix = np.array(train, copy=True)
    fit_matrix[evaluation_mask] = 0

    return LoggedHoldoutSplit(
        fit_matrix=fit_matrix,
        evaluation_mask=evaluation_mask,
        evaluation_probability=evaluation_probability,
        n_fit=int(np.sum(fit_matrix > 0)),
        n_evaluation=int(np.sum(evaluation_mask)),
    )


def _estimated_ht(
    y: np.ndarray,
    p: np.ndarray,
    inclusion_probability: np.ndarray,
    *,
    target_size: int,
) -> float:
    loss = np.square(p - y)
    return float(np.sum(loss / inclusion_probability) / target_size)


def _estimated_snips(
    y: np.ndarray,
    p: np.ndarray,
    inclusion_probability: np.ndarray,
) -> float:
    weights = 1.0 / inclusion_probability
    loss = np.square(p - y)
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
) -> HeldoutRankingSummary:
    order = tuple(sorted(comparison, key=lambda name: (comparison[name], name)))
    return HeldoutRankingSummary(
        method=method,
        order=order,
        spearman_vs_randomized=_spearman(randomized, comparison),
        kendall_vs_randomized=_kendall(randomized, comparison),
        reversals_vs_randomized=pairwise_reversals(
            randomized,
            comparison,
        ),
    )


def run_coat_heldout_benchmark(
    train: np.ndarray,
    randomized: np.ndarray,
    propensities: np.ndarray,
    *,
    positive_threshold: int = 4,
    prior_strength: float = 10.0,
    holdout_fraction: float = 1.0 / 3.0,
    seed: int = 2026,
    propensity_floors: tuple[float, ...] = (0.01, 0.02, 0.05),
) -> CoatHeldoutBenchmark:
    """Compare held-out logged and randomized evaluation for the same models."""
    train = np.asarray(train)
    randomized = np.asarray(randomized)
    propensity = np.asarray(propensities, dtype=float)

    if train.ndim != 2 or randomized.ndim != 2 or propensity.ndim != 2:
        raise ValueError("all inputs must be 2D matrices")
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

    split = split_logged_ratings_by_user(
        train,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    if split.n_evaluation == 0:
        raise ValueError("logged split produced no evaluation observations")

    randomized_mask = randomized > 0
    if not np.any(randomized_mask):
        raise ValueError("randomized matrix must contain observed ratings")

    predictions = fit_logged_probability_models(
        split.fit_matrix,
        positive_threshold=positive_threshold,
        prior_strength=prior_strength,
    )

    eval_mask = split.evaluation_mask
    y_eval = (train[eval_mask] >= positive_threshold).astype(float)
    y_randomized = (
        randomized[randomized_mask] >= positive_threshold
    ).astype(float)

    e_eval = propensity[eval_mask]
    q_eval = split.evaluation_probability[eval_mask]
    inclusion_eval = e_eval * q_eval
    target_size = train.size

    randomized_scores: dict[str, float] = {}
    logged_scores: dict[str, float] = {}
    ht_scores: dict[str, float] = {}
    snips_scores: dict[str, float] = {}
    score_rows: list[HeldoutModelScores] = []

    for name, prediction in predictions.items():
        p_eval = prediction[eval_mask]
        p_randomized = prediction[randomized_mask]

        randomized_score = brier_score(y_randomized, p_randomized)
        logged_score = brier_score(y_eval, p_eval)
        ht_score = _estimated_ht(
            y_eval,
            p_eval,
            inclusion_eval,
            target_size=target_size,
        )
        snips_score = _estimated_snips(
            y_eval,
            p_eval,
            inclusion_eval,
        )

        randomized_scores[name] = randomized_score
        logged_scores[name] = logged_score
        ht_scores[name] = ht_score
        snips_scores[name] = snips_score

        score_rows.append(
            HeldoutModelScores(
                model=name,
                randomized_brier=randomized_score,
                heldout_logged_brier=logged_score,
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
            method="heldout_logged_naive",
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

    clip_rows: list[HeldoutClipSensitivity] = []
    for floor in propensity_floors:
        clipped_exposure = np.maximum(e_eval, floor)
        clipped_inclusion = clipped_exposure * q_eval

        for name, prediction in predictions.items():
            p_eval = prediction[eval_mask]
            clip_rows.append(
                HeldoutClipSensitivity(
                    propensity_floor=float(floor),
                    model=name,
                    estimated_ht_brier=_estimated_ht(
                        y_eval,
                        p_eval,
                        clipped_inclusion,
                        target_size=target_size,
                    ),
                    estimated_snips_brier=_estimated_snips(
                        y_eval,
                        p_eval,
                        clipped_inclusion,
                    ),
                )
            )

    return CoatHeldoutBenchmark(
        positive_threshold=positive_threshold,
        seed=seed,
        holdout_fraction=holdout_fraction,
        n_users=train.shape[0],
        n_items=train.shape[1],
        n_logged_total=int(np.sum(train > 0)),
        n_logged_fit=split.n_fit,
        n_logged_evaluation=split.n_evaluation,
        n_randomized=int(np.sum(randomized_mask)),
        scores=tuple(score_rows),
        rankings=rankings,
        clipping=tuple(clip_rows),
    )
