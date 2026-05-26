from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from atlas_posthoc.diagnostics.partition_metrics import (
    PairwiseDistanceMatrix,
    PartitionMetricName,
    pairwise_partition_distance_matrix,
)
from atlas_posthoc.diagnostics.pace_clustering import (
    ClusteringMethod,
    PACEDistanceClusteringResult,
    pace_from_distance_matrix,
)
from atlas_posthoc.pipeline.chain_partitions import (
    MultiChainPartitionResult,
    build_multi_chain_partitions,
)


class PACEWorkflowError(Exception):
    """Base exception for end-to-end PACE workflow helpers."""


@dataclass
class PACEWorkflowResult:
    """
    End-to-end PACE workflow result.

    This object intentionally does not store the GerryChain Partition objects,
    because those can be large. It stores the distance matrix, clustering result,
    PACE result, and lightweight metadata.
    """

    distance_matrix: PairwiseDistanceMatrix
    distance_clustering: PACEDistanceClusteringResult
    sample_chain_ids: list[str]
    sample_labels: list[str]
    chain_sizes: dict[str, int]
    partition_summaries: dict[str, dict[str, Any]] | None = None

    @property
    def clustering(self) -> Any:
        return self.distance_clustering.clustering

    @property
    def pace(self) -> Any:
        return self.distance_clustering.pace

    @property
    def n_samples(self) -> int:
        return len(self.sample_chain_ids)

    @property
    def n_chains(self) -> int:
        return len(self.chain_sizes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "distance_matrix": self.distance_matrix.to_dict(),
            "distance_clustering": self.distance_clustering.to_dict(),
            "sample_chain_ids": self.sample_chain_ids,
            "sample_labels": self.sample_labels,
            "chain_sizes": self.chain_sizes,
            "partition_summaries": self.partition_summaries,
            "n_samples": self.n_samples,
            "n_chains": self.n_chains,
        }


def _warn_expensive_pace_distance_matrix(
    *,
    n_samples: int,
    metric: str,
) -> None:
    n_pairs = n_samples * (n_samples - 1) // 2

    if metric in {"wasserplan", "optimal_transport"}:
        warnings.warn(
            f"Computing a full pairwise Wasserplan distance matrix for "
            f"{n_samples} partitions requires approximately {n_pairs:,} "
            "distance computations. This can be very expensive. For a smoke "
            "test, use a small `max_maps_per_chain`, larger `thin_step`, or "
            "try `unit_transfer` / `population_transfer` first.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif n_samples >= 1000:
        warnings.warn(
            f"Computing a full pairwise distance matrix for {n_samples} "
            f"partitions requires approximately {n_pairs:,} distances. "
            "This may be expensive even for simple metrics.",
            RuntimeWarning,
            stacklevel=2,
        )


def _normalize_partitions_by_chain(
    partitions: Mapping[str, Sequence[Any]] | MultiChainPartitionResult,
) -> tuple[list[Any], list[str], list[str], dict[str, int], dict[str, dict[str, Any]] | None]:
    """
    Normalize either a MultiChainPartitionResult or a plain mapping into pooled
    partitions, sample chain ids, and sample labels.
    """
    partition_summaries = None

    if isinstance(partitions, MultiChainPartitionResult):
        partitions_by_chain = partitions.partitions_by_chain
        partition_summaries = partitions.summaries()
    elif hasattr(partitions, "partitions_by_chain"):
        partitions_by_chain = partitions.partitions_by_chain
        if hasattr(partitions, "summaries"):
            partition_summaries = partitions.summaries()
    else:
        partitions_by_chain = partitions

    if not partitions_by_chain:
        raise PACEWorkflowError("no partitions were provided")

    pooled_partitions: list[Any] = []
    sample_chain_ids: list[str] = []
    sample_labels: list[str] = []
    chain_sizes: dict[str, int] = {}

    for chain_id, chain_partitions in partitions_by_chain.items():
        cid = str(chain_id)
        parts = list(chain_partitions)

        if not parts:
            raise PACEWorkflowError(f"chain {cid!r} has no partitions")

        chain_sizes[cid] = len(parts)

        for idx, part in enumerate(parts):
            pooled_partitions.append(part)
            sample_chain_ids.append(cid)
            sample_labels.append(f"{cid}:{idx}")

    if len(set(sample_chain_ids)) < 2:
        raise PACEWorkflowError("PACE workflow requires at least two chains")

    return (
        pooled_partitions,
        sample_chain_ids,
        sample_labels,
        chain_sizes,
        partition_summaries,
    )


def run_pace_workflow_from_partitions(
    partitions: Mapping[str, Sequence[Any]] | MultiChainPartitionResult,
    *,
    metric: PartitionMetricName | Any = "population_transfer",
    pop_col: str = "POP20",
    weight_col: str | None = None,
    clustering_method: ClusteringMethod = "farthest_first",
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
    first_medoid: int = 0,
    initial_medoids: Sequence[int] | None = None,
    pam_max_iter: int = 20,
    pace_threshold: float | None = 0.1,
    apply_correction: bool = True,
    n_bootstrap: int = 0,
    confidence_level: float = 0.95,
    random_state: int | Any | None = None,
) -> PACEWorkflowResult:
    """
    Run the PACE workflow from already-built GerryChain partitions.

    Computational warning
    ---------------------
    This function computes a full pairwise distance matrix over the pooled
    partitions. If there are N pooled partitions, this requires roughly

        N(N - 1) / 2

    pairwise distance calls.

    For Wasserplan / optimal transport distances, this can be very expensive.
    Start with small thinned samples, for example 25--100 maps per chain.

    Parameters
    ----------
    partitions:
        Either a `MultiChainPartitionResult` or a mapping:

            {"run5": [Partition, ...], "run6": [Partition, ...], ...}

    metric:
        Partition distance metric. Useful options:
          - `"unit_transfer"` for a fast smoke test;
          - `"population_transfer"` for a weighted smoke test;
          - `"wasserplan"` or `"optimal_transport"` for the serious metric.

    clustering_method:
        How to convert the distance matrix into PACE cells:
          - `"farthest_first"` needs `n_clusters`;
          - `"pam"` needs `n_clusters`;
          - `"threshold"` needs `distance_threshold`.

    Returns
    -------
    PACEWorkflowResult
        Contains the distance matrix, clustering result, and PACE result.
    """
    (
        pooled_partitions,
        sample_chain_ids,
        sample_labels,
        chain_sizes,
        partition_summaries,
    ) = _normalize_partitions_by_chain(partitions)

    metric_name = metric if isinstance(metric, str) else getattr(metric, "__name__", "callable")

    _warn_expensive_pace_distance_matrix(
        n_samples=len(pooled_partitions),
        metric=str(metric_name),
    )

    distance_matrix = pairwise_partition_distance_matrix(
        pooled_partitions,
        metric=metric,
        labels=sample_labels,
        pop_col=pop_col,
        weight_col=weight_col,
    )

    distance_clustering = pace_from_distance_matrix(
        distance_matrix.matrix,
        sample_chain_ids,
        clustering_method=clustering_method,
        n_clusters=n_clusters,
        threshold=distance_threshold,
        first_medoid=first_medoid,
        initial_medoids=initial_medoids,
        pam_max_iter=pam_max_iter,
        pace_threshold=pace_threshold,
        apply_correction=apply_correction,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )

    return PACEWorkflowResult(
        distance_matrix=distance_matrix,
        distance_clustering=distance_clustering,
        sample_chain_ids=sample_chain_ids,
        sample_labels=sample_labels,
        chain_sizes=chain_sizes,
        partition_summaries=partition_summaries,
    )


def run_pace_workflow_from_chains(
    chain_files: Sequence[str | Path],
    graph: str | Path | Any,
    *,
    gerrychain_graph: Any | None = None,
    chain_ids: Sequence[str] | None = None,
    key_cols: Sequence[str] | None = None,
    node_id_col: str | None = "id",
    name_col: str = "NAME",
    thin_step: int = 1,
    max_maps_per_chain: int | None = 100,
    skip_bad_lines: bool = True,
    output_frequency: int | None = None,
    require_perfect_key_scheme: bool = True,
    missing_policy: str = "error",
    ambiguous_policy: str = "error",
    require_complete_assignment: bool = True,
    metric: PartitionMetricName | Any = "population_transfer",
    pop_col: str = "POP20",
    weight_col: str | None = None,
    clustering_method: ClusteringMethod = "farthest_first",
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
    first_medoid: int = 0,
    initial_medoids: Sequence[int] | None = None,
    pam_max_iter: int = 20,
    pace_threshold: float | None = 0.1,
    apply_correction: bool = True,
    n_bootstrap: int = 0,
    confidence_level: float = 0.95,
    random_state: int | Any | None = None,
) -> PACEWorkflowResult:
    """
    End-to-end PACE workflow from raw AtlasIO chains.

    This function does all of the following:

        AtlasIO chains
        -> GerryChain partitions
        -> pairwise partition distance matrix
        -> distance-matrix clustering
        -> PACE diagnostic

    Computational warning
    ---------------------
    This is potentially the most expensive workflow in atlas-posthoc.

    If you request four chains with `max_maps_per_chain=250`, the pooled sample
    has up to 1000 partitions, so the distance matrix has about 499,500
    pairwise distances. That is likely too expensive for Wasserplan as a first
    run.

    Recommended first run
    ---------------------
    Start with something like:

        metric="unit_transfer"
        thin_step=10
        max_maps_per_chain=25 or 50
        clustering_method="threshold" or "farthest_first"

    Then scale up once the workflow works.
    """
    partitions = build_multi_chain_partitions(
        chain_files,
        graph,
        gerrychain_graph=gerrychain_graph,
        chain_ids=chain_ids,
        key_cols=key_cols,
        node_id_col=node_id_col,
        name_col=name_col,
        thin_step=thin_step,
        max_maps_per_chain=max_maps_per_chain,
        skip_bad_lines=skip_bad_lines,
        output_frequency=output_frequency,
        require_perfect_key_scheme=require_perfect_key_scheme,
        missing_policy=missing_policy,
        ambiguous_policy=ambiguous_policy,
        require_complete_assignment=require_complete_assignment,
    )

    return run_pace_workflow_from_partitions(
        partitions,
        metric=metric,
        pop_col=pop_col,
        weight_col=weight_col,
        clustering_method=clustering_method,
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        first_medoid=first_medoid,
        initial_medoids=initial_medoids,
        pam_max_iter=pam_max_iter,
        pace_threshold=pace_threshold,
        apply_correction=apply_correction,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )


# Friendly alias.
run_pace_workflow = run_pace_workflow_from_partitions