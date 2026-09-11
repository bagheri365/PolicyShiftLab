"""Leakage-safe Yahoo! R3 benchmark within the randomized-user cohort."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policyshiftlab.metrics import brier_score
from policyshiftlab.ranking import PairwiseReversal, pairwise_reversals


@dataclass(frozen=True)
class YahooLoggedSplit:
    fit_triplets: np.ndarray
    evaluation_triplets: np.ndarray
    n_fit: int
    n_evaluation: int


@dataclass(frozen=True)
class YahooModelScores:
    model: str
    randomized_brier: float
    heldout_logged_brier: float


@dataclass(frozen=True)
class YahooRankingSummary:
    method: str
    order: tuple[str, ...]
    spearman_vs_randomized: float
    kendall_vs_randomized: float
    reversals_vs_randomized: tuple[PairwiseReversal, ...]


@dataclass(frozen=True)
class YahooHeldoutBenchmark:
    positive_threshold: int
    seed: int
    holdout_fraction: float
    n_target_users: int
    n_items_randomized: int
    n_logged_cohort_total: int
    n_logged_fit: int
    n_logged_evaluation: int
    n_randomized: int
    scores: tuple[YahooModelScores, ...]
    rankings: tuple[YahooRankingSummary, ...]


def restrict_logged_to_randomized_users(
    train: np.ndarray,
    randomized: np.ndarray,
) -> np.ndarray:
    """Restrict logged triplets to users appearing in randomized evaluation."""
    train = np.asarray(train, dtype=int)
    randomized = np.asarray(randomized, dtype=int)

    if train.ndim != 2 or train.shape[1] != 3:
        raise ValueError("train must have shape (n, 3)")
    if randomized.ndim != 2 or randomized.shape[1] != 3:
        raise ValueError("randomized must have shape (n, 3)")
    if train.shape[0] == 0 or randomized.shape[0] == 0:
        raise ValueError("train and randomized must be non-empty")

    target_users = np.unique(randomized[:, 0])
    mask = np.isin(train[:, 0], target_users)
    restricted = train[mask]

    if restricted.shape[0] == 0:
        raise ValueError("no logged observations for randomized users")

    return restricted


def split_logged_triplets_by_user(
    train: np.ndarray,
    *,
    holdout_fraction: float = 1.0 / 3.0,
    seed: int = 2026,
) -> YahooLoggedSplit:
    """Split logged triplets within user into fit and held-out evaluation rows."""
    train = np.asarray(train, dtype=int)

    if train.ndim != 2 or train.shape[1] != 3 or train.shape[0] == 0:
        raise ValueError("train must be a non-empty (n, 3) matrix")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must lie in (0, 1)")

    rng = np.random.default_rng(seed)
    evaluation = np.zeros(train.shape[0], dtype=bool)

    for user in np.unique(train[:, 0]):
        rows = np.flatnonzero(train[:, 0] == user)
        n_rows = rows.size
        if n_rows <= 1:
            continue

        n_holdout = int(np.rint(holdout_fraction * n_rows))
        n_holdout = min(max(n_holdout, 1), n_rows - 1)
        chosen = rng.choice(rows, size=n_holdout, replace=False)
        evaluation[chosen] = True

    fit = train[~evaluation]
    heldout = train[evaluation]

    if heldout.shape[0] == 0:
        raise ValueError("split produced no held-out observations")

    return YahooLoggedSplit(
        fit_triplets=fit,
        evaluation_triplets=heldout,
        n_fit=int(fit.shape[0]),
        n_evaluation=int(heldout.shape[0]),
    )


def fit_yahoo_probability_models(
    fit_triplets: np.ndarray,
    *,
    positive_threshold: int = 4,
    prior_strength: float = 10.0,
) -> dict[str, object]:
    """Fit simple fixed probability estimators from logged-fit triplets only."""
    data = np.asarray(fit_triplets, dtype=int)

    if data.ndim != 2 or data.shape[1] != 3 or data.shape[0] == 0:
        raise ValueError("fit_triplets must be a non-empty (n, 3) matrix")
    if positive_threshold <= 0:
        raise ValueError("positive_threshold must be positive")
    if prior_strength <= 0.0:
        raise ValueError("prior_strength must be positive")

    y = (data[:, 2] >= positive_threshold).astype(float)
    global_rate = float(np.mean(y))

    users, user_inverse = np.unique(data[:, 0], return_inverse=True)
    items, item_inverse = np.unique(data[:, 1], return_inverse=True)

    user_count = np.bincount(user_inverse).astype(float)
    user_success = np.bincount(user_inverse, weights=y).astype(float)
    item_count = np.bincount(item_inverse).astype(float)
    item_success = np.bincount(item_inverse, weights=y).astype(float)

    user_rate = (
        user_success + prior_strength * global_rate
    ) / (user_count + prior_strength)
    item_rate = (
        item_success + prior_strength * global_rate
    ) / (item_count + prior_strength)

    return {
        "global_rate": global_rate,
        "users": users,
        "user_rate": user_rate,
        "items": items,
        "item_rate": item_rate,
    }


def _lookup_smoothed(
    ids: np.ndarray,
    known_ids: np.ndarray,
    known_rates: np.ndarray,
    fallback: float,
) -> np.ndarray:
    positions = np.searchsorted(known_ids, ids)
    valid = (
        (positions < known_ids.size)
        & (known_ids[np.minimum(positions, known_ids.size - 1)] == ids)
    )

    result = np.full(ids.shape, fallback, dtype=float)
    if np.any(valid):
        result[valid] = known_rates[positions[valid]]
    return result


def predict_yahoo_probability_models(
    fitted: dict[str, object],
    triplets: np.ndarray,
) -> dict[str, np.ndarray]:
    """Predict event probabilities for arbitrary Yahoo! R3 triplets."""
    data = np.asarray(triplets, dtype=int)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("triplets must have shape (n, 3)")

    global_rate = float(fitted["global_rate"])
    users = np.asarray(fitted["users"], dtype=int)
    user_rate = np.asarray(fitted["user_rate"], dtype=float)
    items = np.asarray(fitted["items"], dtype=int)
    item_rate = np.asarray(fitted["item_rate"], dtype=float)

    user_pred = _lookup_smoothed(
        data[:, 0], users, user_rate, global_rate
    )
    item_pred = _lookup_smoothed(
        data[:, 1], items, item_rate, global_rate
    )
    global_pred = np.full(data.shape[0], global_rate, dtype=float)

    return {
        "global": global_pred,
        "user_smoothed": user_pred,
        "item_smoothed": item_pred,
        "user_item_blend": 0.5 * user_pred + 0.5 * item_pred,
    }


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
    randomized_scores: dict[str, float],
    comparison_scores: dict[str, float],
    *,
    method: str,
) -> YahooRankingSummary:
    order = tuple(
        sorted(
            comparison_scores,
            key=lambda name: (comparison_scores[name], name),
        )
    )
    return YahooRankingSummary(
        method=method,
        order=order,
        spearman_vs_randomized=_spearman(
            randomized_scores, comparison_scores
        ),
        kendall_vs_randomized=_kendall(
            randomized_scores, comparison_scores
        ),
        reversals_vs_randomized=pairwise_reversals(
            randomized_scores,
            comparison_scores,
        ),
    )


def run_yahoo_r3_heldout_benchmark(
    train: np.ndarray,
    randomized: np.ndarray,
    *,
    positive_threshold: int = 4,
    prior_strength: float = 10.0,
    holdout_fraction: float = 1.0 / 3.0,
    seed: int = 2026,
) -> YahooHeldoutBenchmark:
    """Run leakage-safe Yahoo! R3 evaluation on the randomized-user cohort."""
    train = np.asarray(train, dtype=int)
    randomized = np.asarray(randomized, dtype=int)

    cohort_train = restrict_logged_to_randomized_users(train, randomized)
    split = split_logged_triplets_by_user(
        cohort_train,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    fitted = fit_yahoo_probability_models(
        split.fit_triplets,
        positive_threshold=positive_threshold,
        prior_strength=prior_strength,
    )

    heldout_pred = predict_yahoo_probability_models(
        fitted, split.evaluation_triplets
    )
    randomized_pred = predict_yahoo_probability_models(
        fitted, randomized
    )

    y_heldout = (
        split.evaluation_triplets[:, 2] >= positive_threshold
    ).astype(float)
    y_randomized = (
        randomized[:, 2] >= positive_threshold
    ).astype(float)

    randomized_scores: dict[str, float] = {}
    heldout_scores: dict[str, float] = {}
    score_rows: list[YahooModelScores] = []

    for name in randomized_pred:
        randomized_score = brier_score(
            y_randomized, randomized_pred[name]
        )
        heldout_score = brier_score(
            y_heldout, heldout_pred[name]
        )
        randomized_scores[name] = randomized_score
        heldout_scores[name] = heldout_score
        score_rows.append(
            YahooModelScores(
                model=name,
                randomized_brier=randomized_score,
                heldout_logged_brier=heldout_score,
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
            heldout_scores,
            method="heldout_logged_naive",
        ),
    )

    return YahooHeldoutBenchmark(
        positive_threshold=positive_threshold,
        seed=seed,
        holdout_fraction=holdout_fraction,
        n_target_users=int(np.unique(randomized[:, 0]).size),
        n_items_randomized=int(np.unique(randomized[:, 1]).size),
        n_logged_cohort_total=int(cohort_train.shape[0]),
        n_logged_fit=split.n_fit,
        n_logged_evaluation=split.n_evaluation,
        n_randomized=int(randomized.shape[0]),
        scores=tuple(score_rows),
        rankings=rankings,
    )
