import numpy as np
import pytest

from atlas_posthoc.diagnostics.total_variation import (
    TotalVariationError,
    compute_rankwise_pairwise_tv,
    compute_tv,
    get_tv_convergence,
    make_vote_share_bins,
    max_average_pairwise_tv,
)


def test_compute_tv_identical_distributions():
    bins = np.array([0.0, 0.5, 1.0])

    tv = compute_tv(
        [0.1, 0.2, 0.8],
        [0.1, 0.2, 0.8],
        bins,
    )

    assert tv == 0.0


def test_compute_tv_disjoint_distributions():
    bins = np.array([0.0, 0.5, 1.0])

    tv = compute_tv(
        [0.1, 0.2],
        [0.8, 0.9],
        bins,
    )

    assert tv == 1.0


def test_compute_tv_partial_difference():
    bins = np.array([0.0, 0.5, 1.0])

    tv = compute_tv(
        [0.1, 0.8],
        [0.1, 0.2],
        bins,
    )

    assert tv == 0.5


def test_compute_tv_empty_raises():
    bins = np.array([0.0, 0.5, 1.0])

    with pytest.raises(TotalVariationError):
        compute_tv([], [0.1, 0.2], bins)


def test_compute_rankwise_pairwise_tv_identical_runs():
    bins = np.array([0.0, 0.5, 1.0])

    run_a = np.array(
        [
            [0.1, 0.8],
            [0.2, 0.7],
        ]
    )

    run_b = run_a.copy()

    result = compute_rankwise_pairwise_tv(
        run_a,
        run_b,
        bins,
        chain_a="run_a",
        chain_b="run_b",
    )

    assert result.chain_a == "run_a"
    assert result.chain_b == "run_b"
    assert result.tv_by_rank == [0.0, 0.0]
    assert result.average_tv == 0.0


def test_compute_rankwise_pairwise_tv_nonidentical_runs():
    bins = np.array([0.0, 0.5, 1.0])

    run_a = np.array(
        [
            [0.1, 0.8],
            [0.2, 0.7],
        ]
    )

    run_b = np.array(
        [
            [0.1, 0.8],
            [0.8, 0.2],
        ]
    )

    result = compute_rankwise_pairwise_tv(run_a, run_b, bins)

    assert result.tv_by_rank == [0.5, 0.5]
    assert result.average_tv == 0.5


def test_max_average_pairwise_tv_two_chains():
    bins = np.array([0.0, 0.5, 1.0])

    run_a = np.array(
        [
            [0.1, 0.8],
            [0.2, 0.7],
        ]
    )

    run_b = np.array(
        [
            [0.1, 0.8],
            [0.8, 0.2],
        ]
    )

    max_tv, pairwise = max_average_pairwise_tv(
        [run_a, run_b],
        bins,
        chain_ids=["a", "b"],
    )

    assert max_tv == 0.5
    assert len(pairwise) == 1
    assert pairwise[0].chain_a == "a"
    assert pairwise[0].chain_b == "b"


def test_get_tv_convergence():
    bins = np.array([0.0, 0.5, 1.0])

    run_a = np.array(
        [
            [0.1, 0.8],
            [0.2, 0.7],
        ]
    )

    run_b = np.array(
        [
            [0.1, 0.8],
            [0.8, 0.2],
        ]
    )

    result = get_tv_convergence(
        [run_a, run_b],
        bins,
        output_freq=5,
        snapshots=[1, 2],
        chain_ids=["a", "b"],
    )

    assert result.snapshots == [1, 2]
    assert result.proposals == [5, 10]
    assert result.max_tvs == [0.0, 0.5]
    assert result.details[1].max_average_tv == 0.5


def test_get_tv_convergence_accepts_mapping():
    bins = np.array([0.0, 0.5, 1.0])

    runs = {
        "run9": np.array([[0.1, 0.8], [0.2, 0.7]]),
        "run10": np.array([[0.1, 0.8], [0.8, 0.2]]),
    }

    result = get_tv_convergence(
        runs,
        bins,
        output_freq=10000,
        snapshots=[2],
    )

    assert result.proposals == [20000]
    assert result.max_tvs == [0.5]
    assert result.details[0].pairwise[0].chain_a == "run9"
    assert result.details[0].pairwise[0].chain_b == "run10"


def test_make_vote_share_bins_has_edge_at_half():
    bins = make_vote_share_bins(bin_width=0.002)

    assert np.any(np.isclose(bins, 0.5))
    assert bins[0] <= 0.0
    assert bins[-1] >= 1.0