"""Repeated leakage-safe Coat evaluation and randomized uncertainty."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from policyshiftlab.coat_empirical import fit_logged_probability_models
from policyshiftlab.coat_heldout import (
    run_coat_heldout_benchmark,
    split_logged_ratings_by_user,
)


@dataclass(frozen=True)
class CoatRepeatedMethodSummary:
    method: str
    mean_signed_brier_error: float
    mean_absolute_brier_error: float
    rmse_brier_error: float
    mean_spearman: float
    median_spearman: float
    mean_kendall: float
    median_kendall: float
    any_reversal_rate: float
    mean_reversal_count: float
    exact_order_recovery_rate: float
    n_splits: int


@dataclass(frozen=True)
class CoatPairReversalSummary:
    method: str
    model_a: str
    model_b: str
    reversal_rate: float
    n_splits: int


@dataclass(frozen=True)
class RandomizedPairBootstrap:
    model_a: str
    model_b: str
    observed_difference: float
    lower_95: float
    upper_95: float
    probability_a_better: float
    n_bootstrap: int
    seed: int


@dataclass(frozen=True)
class CoatRepeatedHoldoutResult:
    n_splits: int
    first_seed: int
    holdout_fraction: float
    method_summaries: tuple[CoatRepeatedMethodSummary, ...]
    pair_reversal_summaries: tuple[CoatPairReversalSummary, ...]


def _score_map(result, attribute: str) -> dict[str, float]:
    return {row.model: float(getattr(row, attribute)) for row in result.scores}


def _pair_reversed(
    randomized: dict[str, float],
    comparison: dict[str, float],
    model_a: str,
    model_b: str,
) -> bool:
    target_diff = randomized[model_a] - randomized[model_b]
    comparison_diff = comparison[model_a] - comparison[model_b]
    return bool(target_diff * comparison_diff < 0.0)


def run_repeated_coat_holdouts(
    train: np.ndarray,
    randomized: np.ndarray,
    propensities: np.ndarray,
    *,
    n_splits: int = 200,
    first_seed: int = 0,
    holdout_fraction: float = 1.0 / 3.0,
    positive_threshold: int = 4,
    prior_strength: float = 10.0,
    pair: tuple[str, str] = ("user_smoothed", "item_smoothed"),
) -> CoatRepeatedHoldoutResult:
    """Repeat leakage-safe Coat splits across deterministic seeds.

    This measures sensitivity to the logged fit/evaluation split conditional on
    the observed Coat dataset. It is not replication over new datasets.
    """
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
    if first_seed < 0:
        raise ValueError("first_seed must be non-negative")

    methods = {
        "heldout_logged_naive": "heldout_logged_brier",
        "estimated_ht": "estimated_ht_brier",
        "estimated_snips": "estimated_snips_brier",
    }
    errors = {name: [] for name in methods}
    spearman = {name: [] for name in methods}
    kendall = {name: [] for name in methods}
    reversal_counts = {name: [] for name in methods}
    exact_recovery = {name: [] for name in methods}
    pair_reversed = {name: [] for name in methods}

    model_a, model_b = pair

    for seed in range(first_seed, first_seed + n_splits):
        result = run_coat_heldout_benchmark(
            train,
            randomized,
            propensities,
            positive_threshold=positive_threshold,
            prior_strength=prior_strength,
            holdout_fraction=holdout_fraction,
            seed=seed,
            propensity_floors=(0.01,),
        )
        randomized_scores = _score_map(result, "randomized_brier")
        if model_a not in randomized_scores or model_b not in randomized_scores:
            raise ValueError("requested pair is not present in candidate models")

        ranking_by_method = {row.method: row for row in result.rankings}

        for method, attr in methods.items():
            comparison_scores = _score_map(result, attr)
            errors[method].extend(
                comparison_scores[name] - randomized_scores[name]
                for name in randomized_scores
            )
            ranking = ranking_by_method[method]
            spearman[method].append(ranking.spearman_vs_randomized)
            kendall[method].append(ranking.kendall_vs_randomized)
            count = len(ranking.reversals_vs_randomized)
            reversal_counts[method].append(count)
            exact_recovery[method].append(float(count == 0))
            pair_reversed[method].append(
                float(
                    _pair_reversed(
                        randomized_scores,
                        comparison_scores,
                        model_a,
                        model_b,
                    )
                )
            )

    method_summaries = []
    pair_summaries = []

    for method in methods:
        err = np.asarray(errors[method], dtype=float)
        sp = np.asarray(spearman[method], dtype=float)
        tau = np.asarray(kendall[method], dtype=float)
        rev = np.asarray(reversal_counts[method], dtype=float)
        exact = np.asarray(exact_recovery[method], dtype=float)
        pair_rev = np.asarray(pair_reversed[method], dtype=float)

        method_summaries.append(
            CoatRepeatedMethodSummary(
                method=method,
                mean_signed_brier_error=float(np.mean(err)),
                mean_absolute_brier_error=float(np.mean(np.abs(err))),
                rmse_brier_error=float(np.sqrt(np.mean(np.square(err)))),
                mean_spearman=float(np.nanmean(sp)),
                median_spearman=float(np.nanmedian(sp)),
                mean_kendall=float(np.nanmean(tau)),
                median_kendall=float(np.nanmedian(tau)),
                any_reversal_rate=float(np.mean(rev > 0)),
                mean_reversal_count=float(np.mean(rev)),
                exact_order_recovery_rate=float(np.mean(exact)),
                n_splits=n_splits,
            )
        )
        pair_summaries.append(
            CoatPairReversalSummary(
                method=method,
                model_a=model_a,
                model_b=model_b,
                reversal_rate=float(np.mean(pair_rev)),
                n_splits=n_splits,
            )
        )

    return CoatRepeatedHoldoutResult(
        n_splits=n_splits,
        first_seed=first_seed,
        holdout_fraction=holdout_fraction,
        method_summaries=tuple(method_summaries),
        pair_reversal_summaries=tuple(pair_summaries),
    )


def bootstrap_randomized_pair_difference(
    train: np.ndarray,
    randomized: np.ndarray,
    *,
    fit_seed: int = 2026,
    holdout_fraction: float = 1.0 / 3.0,
    positive_threshold: int = 4,
    prior_strength: float = 10.0,
    model_a: str = "user_smoothed",
    model_b: str = "item_smoothed",
    n_bootstrap: int = 2000,
    seed: int = 314159,
) -> RandomizedPairBootstrap:
    """Paired user-cluster bootstrap for randomized Brier difference.

    Difference is Brier(A) - Brier(B), so negative values favor model A.
    """
    train = np.asarray(train)
    randomized = np.asarray(randomized)
    if train.ndim != 2 or randomized.ndim != 2:
        raise ValueError("train and randomized must be 2D")
    if train.shape != randomized.shape:
        raise ValueError("train and randomized must align")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    split = split_logged_ratings_by_user(
        train,
        holdout_fraction=holdout_fraction,
        seed=fit_seed,
    )
    predictions = fit_logged_probability_models(
        split.fit_matrix,
        positive_threshold=positive_threshold,
        prior_strength=prior_strength,
    )
    if model_a not in predictions or model_b not in predictions:
        raise ValueError("requested model is not present in candidate set")

    observed = randomized > 0
    y = (randomized >= positive_threshold).astype(float)

    user_loss_diff = []
    for user in range(randomized.shape[0]):
        mask = observed[user]
        if not np.any(mask):
            continue
        loss_a = np.square(predictions[model_a][user, mask] - y[user, mask])
        loss_b = np.square(predictions[model_b][user, mask] - y[user, mask])
        user_loss_diff.append(loss_a - loss_b)

    if not user_loss_diff:
        raise ValueError("randomized matrix has no observed ratings")

    observed_difference = float(np.mean(np.concatenate(user_loss_diff)))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    n_users = len(user_loss_diff)

    for index in range(n_bootstrap):
        sampled = rng.integers(0, n_users, size=n_users)
        boot[index] = float(
            np.mean(
                np.concatenate([user_loss_diff[i] for i in sampled])
            )
        )

    lower, upper = np.quantile(boot, [0.025, 0.975])
    return RandomizedPairBootstrap(
        model_a=model_a,
        model_b=model_b,
        observed_difference=observed_difference,
        lower_95=float(lower),
        upper_95=float(upper),
        probability_a_better=float(np.mean(boot < 0.0)),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
