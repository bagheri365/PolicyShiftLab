import numpy as np
import pytest

from policyshiftlab.ranking_overlap import (
    expected_ipw_effective_sample_size,
    run_ranking_overlap_sweep,
)


def test_expected_ess_is_valid() -> None:
    e = np.array([0.5, 0.5, 0.5, 0.5])

    ess = expected_ipw_effective_sample_size(e)

    assert ess == pytest.approx(2.0)


def test_overlap_sweep_returns_requested_regimes() -> None:
    results = run_ranking_overlap_sweep(
        regimes=(
            ("strong", 0.20, 1.0),
            ("weak", 0.02, 3.0),
        ),
        n_users=40,
        n_items=20,
        n_replications=40,
        seed=7,
    )

    assert [result.label for result in results] == ["strong", "weak"]

    for result in results:
        assert result.expected_ipw_effective_sample_size > 0.0
        assert result.expected_selected > 0.0
        assert -1.0 <= result.logged_mean_spearman <= 1.0
        assert -1.0 <= result.ipw_mean_spearman <= 1.0
        assert 0.0 <= result.logged_any_reversal_rate <= 1.0
        assert 0.0 <= result.ipw_any_reversal_rate <= 1.0


def test_weaker_overlap_reduces_expected_ipw_ess() -> None:
    strong, weak = run_ranking_overlap_sweep(
        regimes=(
            ("strong", 0.20, 0.8),
            ("weak", 0.02, 3.0),
        ),
        n_users=80,
        n_items=40,
        n_replications=60,
        seed=11,
    )

    assert (
        weak.expected_ipw_effective_sample_size
        < strong.expected_ipw_effective_sample_size
    )


def test_fixed_population_is_preserved_across_regimes() -> None:
    # This exercises the internal equality checks; any unintended target-DGP
    # change across regimes would raise before the function returns.
    results = run_ranking_overlap_sweep(
        regimes=(
            ("a", 0.20, 1.0),
            ("b", 0.05, 2.0),
        ),
        n_users=30,
        n_items=15,
        n_replications=20,
        seed=3,
    )

    assert len(results) == 2


@pytest.mark.parametrize(
    "regimes",
    [
        (),
        (("", 0.1, 1.0),),
        (("bad", 0.0, 1.0),),
        (("bad", 1.0, 1.0),),
        (("bad", 0.1, 0.0),),
    ],
)
def test_invalid_regimes_are_rejected(regimes) -> None:
    with pytest.raises(ValueError):
        run_ranking_overlap_sweep(
            regimes=regimes,
            n_users=20,
            n_items=10,
            n_replications=10,
        )
