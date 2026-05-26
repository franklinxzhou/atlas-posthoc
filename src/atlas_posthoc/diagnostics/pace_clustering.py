from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np

from atlas_posthoc.diagnostics.pace import PACEResult, pace_from_labels


ClusteringMethod = Literal["farthest_first", "pam", "threshold"]


class PACEClusteringError(Exception):
    """Base exception for PACE clustering helpers."""


@dataclass
class PACEClusteringResult:
    labels: list[int]
    medoid_indices: list[int]
    method: str
    n_clusters: int
    n_points: int
    objective: float | None
    max_assigned_distance: float | None
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PACEDistanceClusteringResult:
    clustering: PACEClusteringResult
    pace: PACEResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "clustering": self.clustering.to_dict(),
            "pace": self.pace.to_dict(),
        }


def validate_distance_matrix(
    distance_matrix: Any,
    *,
    require_symmetric: bool = True,
    atol: float = 1e-8,
) -> np.ndarray:
    """
    Validate and coerce a pairwise distance matrix.

    The expected input is an NxN matrix where entry (i, j) is the distance
    between sampled plan i and sampled plan j.
    """
    D = np.asarray(distance_matrix, dtype=float)

    if D.ndim != 2:
        raise PACEClusteringError(f"distance matrix must be 2D; got shape {D.shape}")

    if D.shape[0] != D.shape[1]:
        raise PACEClusteringError(
            f"distance matrix must be square; got shape {D.shape}"
        )

    if D.shape[0] == 0:
        raise PACEClusteringError("distance matrix is empty")

    if not np.all(np.isfinite(D)):
        raise PACEClusteringError("distance matrix contains non-finite values")

    if np.any(D < -atol):
        raise PACEClusteringError("distance matrix contains negative values")

    # Clean tiny negative numerical noise.
    D = np.maximum(D, 0.0)

    if require_symmetric and not np.allclose(D, D.T, atol=atol):
        raise PACEClusteringError("distance matrix is not symmetric")

    if not np.allclose(np.diag(D), 0.0, atol=atol):
        raise PACEClusteringError("distance matrix diagonal must be zero")

    return D


def _warn_large_distance_matrix(n_points: int) -> None:
    if n_points >= 1000:
        warnings.warn(
            f"PACE clustering received {n_points} points. The distance matrix "
            f"contains about {n_points * (n_points - 1) // 2:,} pairwise "
            "distances. This is already expensive; Wasserplan distance "
            "construction is usually the bottleneck.",
            RuntimeWarning,
            stacklevel=2,
        )


def farthest_first_medoids(
    distance_matrix: Any,
    *,
    n_clusters: int,
    first_medoid: int = 0,
) -> list[int]:
    """
    Greedy farthest-first medoid initialization.

    This is deterministic and fast. It is a good default for producing
    preliminary PACE cells from a distance matrix.
    """
    D = validate_distance_matrix(distance_matrix)
    n = D.shape[0]

    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")

    if n_clusters > n:
        raise ValueError("n_clusters cannot exceed number of points")

    if not (0 <= first_medoid < n):
        raise ValueError("first_medoid is out of range")

    medoids = [int(first_medoid)]

    while len(medoids) < n_clusters:
        dist_to_nearest = np.min(D[:, medoids], axis=1)
        dist_to_nearest[medoids] = -np.inf
        next_medoid = int(np.argmax(dist_to_nearest))
        medoids.append(next_medoid)

    return medoids


def assign_to_medoids(
    distance_matrix: Any,
    medoid_indices: Sequence[int],
) -> tuple[np.ndarray, float, float]:
    """
    Assign each point to its nearest medoid.

    Returns
    -------
    labels:
        Cluster label for each point.

    objective:
        Sum of distances from each point to its assigned medoid.

    max_assigned_distance:
        Maximum assigned point-to-medoid distance.
    """
    D = validate_distance_matrix(distance_matrix)
    n = D.shape[0]

    medoids = [int(x) for x in medoid_indices]

    if not medoids:
        raise PACEClusteringError("medoid_indices cannot be empty")

    if len(set(medoids)) != len(medoids):
        raise PACEClusteringError("medoid_indices must be unique")

    if any(m < 0 or m >= n for m in medoids):
        raise PACEClusteringError("medoid index out of range")

    distances_to_medoids = D[:, medoids]
    labels = np.argmin(distances_to_medoids, axis=1).astype(int)
    assigned_distances = distances_to_medoids[np.arange(n), labels]

    return (
        labels,
        float(np.sum(assigned_distances)),
        float(np.max(assigned_distances)),
    )


def pam_refine_medoids(
    distance_matrix: Any,
    medoid_indices: Sequence[int],
    *,
    max_iter: int = 20,
) -> tuple[list[int], float]:
    """
    Lightweight PAM-style medoid refinement.

    This tries all single swaps between current medoids and non-medoids and
    accepts the best improving swap. It is deterministic but can be expensive
    for large N and large K.

    Computational warning
    ---------------------
    This is O(iterations * K * (N-K) * N). Use it for small to moderate
    thinned samples, not full chains.
    """
    D = validate_distance_matrix(distance_matrix)
    n = D.shape[0]

    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    medoids = [int(x) for x in medoid_indices]

    _, current_objective, _ = assign_to_medoids(D, medoids)

    for _ in range(max_iter):
        best_objective = current_objective
        best_medoids = medoids

        medoid_set = set(medoids)
        non_medoids = [i for i in range(n) if i not in medoid_set]

        for old in medoids:
            for new in non_medoids:
                candidate = [new if m == old else m for m in medoids]
                _, obj, _ = assign_to_medoids(D, candidate)

                if obj < best_objective:
                    best_objective = obj
                    best_medoids = candidate

        if best_objective < current_objective:
            medoids = best_medoids
            current_objective = best_objective
        else:
            break

    return medoids, float(current_objective)


def cluster_distance_matrix_by_medoids(
    distance_matrix: Any,
    *,
    n_clusters: int,
    method: Literal["farthest_first", "pam"] = "farthest_first",
    first_medoid: int = 0,
    initial_medoids: Sequence[int] | None = None,
    pam_max_iter: int = 20,
) -> PACEClusteringResult:
    """
    Cluster a distance matrix using medoid-based cells.

    Methods
    -------
    farthest_first:
        Greedy deterministic medoid selection.

    pam:
        Farthest-first initialization followed by PAM-style swap refinement.
    """
    D = validate_distance_matrix(distance_matrix)
    _warn_large_distance_matrix(D.shape[0])

    if initial_medoids is None:
        medoids = farthest_first_medoids(
            D,
            n_clusters=n_clusters,
            first_medoid=first_medoid,
        )
    else:
        medoids = [int(x) for x in initial_medoids]
        if len(medoids) != n_clusters:
            raise PACEClusteringError(
                "initial_medoids length must equal n_clusters"
            )

    if method == "pam":
        medoids, objective = pam_refine_medoids(
            D,
            medoids,
            max_iter=pam_max_iter,
        )
    elif method == "farthest_first":
        _, objective, _ = assign_to_medoids(D, medoids)
    else:
        raise ValueError(f"unknown medoid clustering method: {method!r}")

    labels, objective, max_dist = assign_to_medoids(D, medoids)

    return PACEClusteringResult(
        labels=labels.tolist(),
        medoid_indices=medoids,
        method=method,
        n_clusters=int(n_clusters),
        n_points=int(D.shape[0]),
        objective=float(objective),
        max_assigned_distance=float(max_dist),
        threshold=None,
    )


def cluster_distance_matrix_by_threshold(
    distance_matrix: Any,
    *,
    threshold: float,
) -> PACEClusteringResult:
    """
    Cluster points by connected components under a distance threshold.

    Points i and j are connected if D[i, j] <= threshold. The resulting
    connected components are used as PACE cells.

    This is useful when the threshold has a substantive meaning, for example
    "plans within this Wasserplan distance are treated as the same region of
    state space."
    """
    D = validate_distance_matrix(distance_matrix)
    _warn_large_distance_matrix(D.shape[0])

    if threshold < 0:
        raise ValueError("threshold must be nonnegative")

    n = D.shape[0]
    visited = np.zeros(n, dtype=bool)
    labels = np.full(n, -1, dtype=int)

    cluster_id = 0

    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        labels[start] = cluster_id

        while stack:
            i = stack.pop()
            neighbors = np.where(D[i, :] <= threshold)[0]

            for j in neighbors:
                if not visited[j]:
                    visited[j] = True
                    labels[j] = cluster_id
                    stack.append(int(j))

        cluster_id += 1

    medoids: list[int] = []

    for k in range(cluster_id):
        members = np.where(labels == k)[0]

        if members.size == 1:
            medoids.append(int(members[0]))
            continue

        sub = D[np.ix_(members, members)]
        within_sum = sub.sum(axis=1)
        medoids.append(int(members[int(np.argmin(within_sum))]))

    # For threshold clusters, objective is still useful as medoid assignment
    # distance, but labels are component labels, not nearest-medoid labels.
    max_assigned_distance = 0.0
    objective = 0.0

    for k, medoid in enumerate(medoids):
        members = np.where(labels == k)[0]
        distances = D[members, medoid]
        objective += float(distances.sum())
        max_assigned_distance = max(max_assigned_distance, float(distances.max()))

    return PACEClusteringResult(
        labels=labels.tolist(),
        medoid_indices=medoids,
        method="threshold",
        n_clusters=int(cluster_id),
        n_points=int(n),
        objective=float(objective),
        max_assigned_distance=float(max_assigned_distance),
        threshold=float(threshold),
    )


def cluster_distance_matrix(
    distance_matrix: Any,
    *,
    method: ClusteringMethod = "farthest_first",
    n_clusters: int | None = None,
    threshold: float | None = None,
    first_medoid: int = 0,
    initial_medoids: Sequence[int] | None = None,
    pam_max_iter: int = 20,
) -> PACEClusteringResult:
    """
    General distance-matrix clustering entry point.

    For `method in {"farthest_first", "pam"}`, provide `n_clusters`.
    For `method="threshold"`, provide `threshold`.
    """
    if method in {"farthest_first", "pam"}:
        if n_clusters is None:
            raise PACEClusteringError(
                f"method={method!r} requires n_clusters"
            )

        return cluster_distance_matrix_by_medoids(
            distance_matrix,
            n_clusters=n_clusters,
            method=method,
            first_medoid=first_medoid,
            initial_medoids=initial_medoids,
            pam_max_iter=pam_max_iter,
        )

    if method == "threshold":
        if threshold is None:
            raise PACEClusteringError("method='threshold' requires threshold")

        return cluster_distance_matrix_by_threshold(
            distance_matrix,
            threshold=threshold,
        )

    raise ValueError(f"unknown clustering method: {method!r}")


def pace_from_distance_matrix(
    distance_matrix: Any,
    chain_ids: Sequence[Any],
    *,
    clustering_method: ClusteringMethod = "farthest_first",
    n_clusters: int | None = None,
    threshold: float | None = None,
    first_medoid: int = 0,
    initial_medoids: Sequence[int] | None = None,
    pam_max_iter: int = 20,
    pace_threshold: float | None = 0.1,
    apply_correction: bool = True,
    n_bootstrap: int = 0,
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
) -> PACEDistanceClusteringResult:
    """
    Cluster a distance matrix and run PACE on the resulting cell labels.

    This is the main bridge:

        distance matrix
            -> cluster labels
            -> PACE

    The distance matrix may come from Wasserplan, transfer distance, or any
    other plan distance.
    """
    D = validate_distance_matrix(distance_matrix)
    chain_arr = np.asarray(chain_ids)

    if chain_arr.ndim != 1:
        raise PACEClusteringError("chain_ids must be one-dimensional")

    if chain_arr.size != D.shape[0]:
        raise PACEClusteringError(
            f"chain_ids length {chain_arr.size} does not match distance matrix "
            f"size {D.shape[0]}"
        )

    clustering = cluster_distance_matrix(
        D,
        method=clustering_method,
        n_clusters=n_clusters,
        threshold=threshold,
        first_medoid=first_medoid,
        initial_medoids=initial_medoids,
        pam_max_iter=pam_max_iter,
    )

    pace = pace_from_labels(
        clustering.labels,
        chain_arr,
        threshold=pace_threshold,
        apply_correction=apply_correction,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )

    return PACEDistanceClusteringResult(
        clustering=clustering,
        pace=pace,
    )