import numpy as np
import pytest

from atlas_posthoc.diagnostics.rhat import (
    RHatError,
    classic_r_hat,
    gelman_rubin_by_rank,
    rank_interpretation,
    rank_normalized_split_r_hat,
    summarize_gelman_rubin_by_rank,
)


def test_classic_r_hat_constant_chains_returns_one():
    chains = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    assert classic_r_hat(chains) == 1.0


def test_classic_r_hat_shifted_chains_is_finite():
    chains = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [1.5, 2.5, 3.5, 4.5],
        ]
    )

    rhat = classic_r_hat(chains)

    assert np.isfinite(rhat)
    assert rhat > 0


def test_rank_normalized_split_r_hat_constant_chains_returns_one():
    chains = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    assert rank_normalized_split_r_hat(chains) == 1.0


def test_rank_normalized_split_r_hat_is_finite():
    chains = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.2, 2.2, 3.2, 4.2, 5.2, 6.2],
            [0.8, 1.8, 2.8, 3.8, 4.8, 5.8],
            [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
        ]
    )

    rhat = rank_normalized_split_r_hat(chains)

    assert np.isfinite(rhat)
    assert rhat > 0


def test_rank_interpretation_vote_share():
    assert rank_interpretation(1, 15, rank_type="vote_share") == "most Republican"
    assert rank_interpretation(2, 15, rank_type="vote_share") == "2nd most Republican"
    assert rank_interpretation(3, 15, rank_type="vote_share") == "3rd most Republican"
    assert rank_interpretation(4, 15, rank_type="vote_share") == "4th most Republican"
    assert rank_interpretation(15, 15, rank_type="vote_share") == "least Republican"

def test_rank_interpretation_compactness():
    assert (
        rank_interpretation(1, 15, rank_type="compactness")
        == "most compact / lowest score"
    )
    assert (
        rank_interpretation(2, 15, rank_type="compactness")
        == "2nd lowest compactness score"
    )
    assert (
        rank_interpretation(15, 15, rank_type="compactness")
        == "least compact / highest score"
    )


def test_gelman_rubin_by_rank_list_input():
    run_a = np.array(
        [
            [0.30, 0.50, 0.70],
            [0.31, 0.51, 0.71],
            [0.32, 0.52, 0.72],
            [0.33, 0.53, 0.73],
            [0.34, 0.54, 0.74],
            [0.35, 0.55, 0.75],
        ]
    )

    run_b = run_a + 0.01
    run_c = run_a - 0.01
    run_d = run_a + 0.005

    rows, min_draws, num_ranks = gelman_rubin_by_rank(
        [run_a, run_b, run_c, run_d]
    )

    assert min_draws == 6
    assert num_ranks == 3
    assert len(rows) == 3
    assert rows[0]["rank"] == 1
    assert rows[0]["interpretation"] == "most Republican"
    assert rows[-1]["interpretation"] == "least Republican"

    for row in rows:
        assert np.isfinite(row["classic_r_hat"])
        assert np.isfinite(row["rank_normalized_split_r_hat"])


def test_gelman_rubin_by_rank_dict_input():
    run = np.array(
        [
            [0.30, 0.70],
            [0.31, 0.71],
            [0.32, 0.72],
            [0.33, 0.73],
        ]
    )

    rows, min_draws, num_ranks = gelman_rubin_by_rank(
        {
            "run9": run,
            "run10": run + 0.01,
            "run11": run - 0.01,
            "run12": run + 0.02,
        }
    )

    assert min_draws == 4
    assert num_ranks == 2
    assert len(rows) == 2


def test_summarize_gelman_rubin_by_rank():
    run = np.array(
        [
            [0.30, 0.70],
            [0.31, 0.71],
            [0.32, 0.72],
            [0.33, 0.73],
        ]
    )

    summary = summarize_gelman_rubin_by_rank(
        [run, run + 0.01, run - 0.01, run + 0.02]
    )

    assert summary.min_draws == 4
    assert summary.num_ranks == 2
    assert summary.max_classic_r_hat is not None
    assert summary.max_rank_normalized_split_r_hat is not None
    assert len(summary.rows) == 2


def test_too_few_chains_raises():
    run = np.array(
        [
            [0.30, 0.70],
            [0.31, 0.71],
            [0.32, 0.72],
            [0.33, 0.73],
        ]
    )

    with pytest.raises(RHatError):
        gelman_rubin_by_rank([run])


def test_too_few_draws_raises():
    run = np.array(
        [
            [0.30, 0.70],
            [0.31, 0.71],
            [0.32, 0.72],
        ]
    )

    with pytest.raises(RHatError):
        gelman_rubin_by_rank([run, run, run, run])