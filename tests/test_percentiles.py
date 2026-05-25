import numpy as np
import pytest

from atlas_posthoc.audit.percentiles import (
    PercentileAuditError,
    compare_compactness_reference_to_ensemble,
    compare_reference_to_ensemble,
    compare_vote_share_reference_to_ensemble,
    empirical_percentile,
    empirical_tail_probabilities,
    pooled_rank_samples,
)


def test_empirical_tail_probabilities_midrank():
    samples = [1.0, 2.0, 2.0, 3.0]

    lower, upper, two_sided = empirical_tail_probabilities(
        samples,
        2.0,
        method="midrank",
    )

    assert lower == pytest.approx(0.5)
    assert upper == pytest.approx(0.5)
    assert two_sided == pytest.approx(1.0)


def test_empirical_percentile_weak():
    samples = [1.0, 2.0, 3.0]

    assert empirical_percentile(samples, 2.0, method="weak") == pytest.approx(2 / 3)


def test_pooled_rank_samples():
    runs = {
        "run1": np.array([[0.1, 0.6], [0.2, 0.7]]),
        "run2": np.array([[0.3, 0.8]]),
    }

    pooled = pooled_rank_samples(runs)

    assert pooled.shape == (3, 2)
    assert np.allclose(pooled[:, 0], [0.1, 0.2, 0.3])


def test_compare_reference_to_ensemble_generic():
    runs = {
        "run1": np.array([[0.1, 0.6], [0.2, 0.7]]),
        "run2": np.array([[0.3, 0.8], [0.4, 0.9]]),
    }

    result = compare_reference_to_ensemble(
        runs,
        reference_values=[0.25, 0.75],
        rank_type="generic",
    )

    assert result.n_ranks == 2
    assert result.n_chains == 2
    assert result.n_total_samples_per_rank == 4
    assert len(result.rows) == 2

    assert result.rows[0].rank == 1
    assert result.rows[0].interpretation == "rank 1"
    assert result.rows[0].percentile == pytest.approx(0.5)
    assert result.rows[1].percentile == pytest.approx(0.5)


def test_compare_vote_share_reference_to_ensemble_has_vote_labels():
    runs = {
        "run1": np.array([[0.1, 0.6], [0.2, 0.7]]),
        "run2": np.array([[0.3, 0.8], [0.4, 0.9]]),
    }

    result = compare_vote_share_reference_to_ensemble(
        runs,
        reference_values=[0.25, 0.75],
    )

    assert result.rows[0].interpretation == "most Republican"
    assert result.rows[1].interpretation == "least Republican"


def test_compare_compactness_reference_to_ensemble_has_compactness_labels():
    runs = {
        "run1": np.array([[20.0, 40.0], [21.0, 41.0]]),
        "run2": np.array([[22.0, 42.0], [23.0, 43.0]]),
    }

    result = compare_compactness_reference_to_ensemble(
        runs,
        reference_values=[21.5, 41.5],
    )

    assert result.rows[0].interpretation == "most compact / lowest score"
    assert result.rows[1].interpretation == "least compact / highest score"


def test_reference_length_mismatch_raises():
    runs = {
        "run1": np.array([[0.1, 0.6], [0.2, 0.7]]),
    }

    with pytest.raises(PercentileAuditError):
        compare_reference_to_ensemble(
            runs,
            reference_values=[0.25],
        )


def test_empty_samples_raise():
    with pytest.raises(PercentileAuditError):
        empirical_percentile([], 0.5)