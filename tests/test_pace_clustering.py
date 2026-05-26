import numpy as np
import pytest

from atlas_posthoc.diagnostics.pace_clustering import (
    PACEClusteringError,
    assign_to_medoids,
    cluster_distance_matrix,
    cluster_distance_matrix_by_threshold,
    farthest_first_medoids,
    pace_from_distance_matrix,
    pam_refine_medoids,
    validate_distance_matrix,
)


def make_two_cluster_distance_matrix():
    # Points 0,1 are close. Points 2,3 are close. Cross-cluster distances large.
    return np.array(
        [
            [0.0, 1.0, 10.0, 11.0],
            [1.0, 0.0, 9.0, 10.0],
            [10.0, 9.0, 0.0, 1.0],
            [11.0, 10.0, 1.0, 0.0],
        ]
    )


def test_validate_distance_matrix():
    D = make_two_cluster_distance_matrix()
    out = validate_distance_matrix(D)

    assert out.shape == (4, 4)


def test_validate_distance_matrix_rejects_non_square():
    with pytest.raises(PACEClusteringError):
        validate_distance_matrix(np.zeros((2, 3)))


def test_validate_distance_matrix_rejects_nonsymmetric():
    D = np.array(
        [
            [0.0, 1.0],
            [2.0, 0.0],
        ]
    )

    with pytest.raises(PACEClusteringError):
        validate_distance_matrix(D)


def test_farthest_first_medoids():
    D = make_two_cluster_distance_matrix()

    medoids = farthest_first_medoids(
        D,
        n_clusters=2,
        first_medoid=0,
    )

    assert medoids[0] == 0
    assert len(medoids) == 2
    assert medoids[1] in {2, 3}


def test_assign_to_medoids():
    D = make_two_cluster_distance_matrix()

    labels, objective, max_dist = assign_to_medoids(D, [0, 2])

    assert labels.tolist() == [0, 0, 1, 1]
    assert objective == pytest.approx(2.0)
    assert max_dist == pytest.approx(1.0)


def test_pam_refine_medoids():
    D = make_two_cluster_distance_matrix()

    medoids, objective = pam_refine_medoids(
        D,
        [0, 2],
        max_iter=5,
    )

    assert len(medoids) == 2
    assert objective <= 2.0


def test_cluster_distance_matrix_farthest_first():
    D = make_two_cluster_distance_matrix()

    result = cluster_distance_matrix(
        D,
        method="farthest_first",
        n_clusters=2,
    )

    assert result.n_clusters == 2
    assert result.n_points == 4
    assert len(result.labels) == 4
    assert len(result.medoid_indices) == 2


def test_cluster_distance_matrix_threshold():
    D = make_two_cluster_distance_matrix()

    result = cluster_distance_matrix_by_threshold(
        D,
        threshold=2.0,
    )

    assert result.n_clusters == 2
    assert result.labels == [0, 0, 1, 1]
    assert len(result.medoid_indices) == 2


def test_cluster_distance_matrix_threshold_singletons():
    D = make_two_cluster_distance_matrix()

    result = cluster_distance_matrix_by_threshold(
        D,
        threshold=0.0,
    )

    assert result.n_clusters == 4
    assert result.labels == [0, 1, 2, 3]


def test_pace_from_distance_matrix():
    D = make_two_cluster_distance_matrix()

    chain_ids = ["run1", "run1", "run2", "run2"]

    result = pace_from_distance_matrix(
        D,
        chain_ids,
        clustering_method="threshold",
        threshold=2.0,
        pace_threshold=0.1,
        apply_correction=False,
    )

    assert result.clustering.n_clusters == 2
    assert result.pace.n_chains == 2
    assert result.pace.n_cells == 2

    # Each chain occupies a different cluster, so PACE should not converge.
    assert result.pace.delta == pytest.approx(0.5)
    assert result.pace.converged is False


def test_pace_from_distance_matrix_balanced_chains():
    D = make_two_cluster_distance_matrix()

    # Both chains have one point in each cluster.
    chain_ids = ["run1", "run2", "run1", "run2"]

    result = pace_from_distance_matrix(
        D,
        chain_ids,
        clustering_method="threshold",
        threshold=2.0,
        pace_threshold=0.1,
        apply_correction=False,
    )

    assert result.pace.delta == pytest.approx(0.0)
    assert result.pace.converged is True


def test_pace_from_distance_matrix_rejects_bad_chain_length():
    D = make_two_cluster_distance_matrix()

    with pytest.raises(PACEClusteringError):
        pace_from_distance_matrix(
            D,
            ["run1", "run2"],
            clustering_method="threshold",
            threshold=2.0,
        )