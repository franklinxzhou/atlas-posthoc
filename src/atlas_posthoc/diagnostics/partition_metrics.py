from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from wasserplan import Pair


PartitionMetricName = Literal[
    "transfer",
    "unit_transfer",
    "population_transfer",
    "wasserplan",
    "optimal_transport",
]


class PartitionMetricError(Exception):
    """Base exception for partition-distance diagnostics."""


@dataclass
class PartitionOverlap:
    districts_a: list[Any]
    districts_b: list[Any]
    overlap_matrix: np.ndarray
    total_weight: float
    n_nodes: int
    weight_col: str | None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["overlap_matrix"] = self.overlap_matrix.tolist()
        return out


@dataclass
class TransferDistanceResult:
    distance: float
    normalized_distance: float | None
    max_overlap: float
    total_weight: float
    n_nodes: int
    n_districts_a: int
    n_districts_b: int
    weight_col: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PairwiseDistanceMatrix:
    labels: list[str]
    matrix: np.ndarray
    metric: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": self.labels,
            "matrix": self.matrix.tolist(),
            "metric": self.metric,
        }


def _assignment_dict(partition: Any) -> Mapping[Any, Any]:
    if not hasattr(partition, "assignment"):
        raise PartitionMetricError("partition object must have an `.assignment` attribute")

    assignment = partition.assignment

    if isinstance(assignment, Mapping):
        return assignment

    try:
        return dict(assignment)
    except Exception as exc:
        raise PartitionMetricError("could not coerce partition.assignment to a dict") from exc


def _graph_nodes(partition: Any) -> set[Any]:
    if not hasattr(partition, "graph"):
        raise PartitionMetricError("partition object must have a `.graph` attribute")

    try:
        return set(partition.graph.nodes)
    except Exception as exc:
        raise PartitionMetricError("could not read partition.graph.nodes") from exc


def _node_weight(partition: Any, node: Any, weight_col: str | None) -> float:
    if weight_col is None:
        return 1.0

    try:
        value = partition.graph.nodes[node].get(weight_col, 1.0)
    except Exception as exc:
        raise PartitionMetricError(
            f"could not read node attribute {weight_col!r} for node {node!r}"
        ) from exc

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PartitionMetricError(
            f"node {node!r} has non-numeric {weight_col}={value!r}"
        ) from exc


def _sorted_labels(values: set[Any]) -> list[Any]:
    return sorted(values, key=lambda x: (str(type(x)), str(x)))


def partition_overlap(
    part_a: Any,
    part_b: Any,
    *,
    weight_col: str | None = None,
    require_same_nodes: bool = True,
) -> PartitionOverlap:
    """
    Compute the district-overlap matrix between two partitions.

    Rows correspond to districts in `part_a`; columns correspond to districts in
    `part_b`. Entry `(i, j)` is the total node weight assigned to district `i`
    in `part_a` and district `j` in `part_b`.

    This is the core object behind label-invariant transfer distance.
    """
    assign_a = _assignment_dict(part_a)
    assign_b = _assignment_dict(part_b)

    nodes_a = _graph_nodes(part_a)
    nodes_b = _graph_nodes(part_b)

    if require_same_nodes and nodes_a != nodes_b:
        raise PartitionMetricError(
            "partitions have different graph node sets: "
            f"|A \\ B|={len(nodes_a - nodes_b)}, |B \\ A|={len(nodes_b - nodes_a)}"
        )

    nodes = nodes_a & nodes_b

    if not nodes:
        raise PartitionMetricError("partitions have no common graph nodes")

    missing_a = [node for node in nodes if node not in assign_a]
    missing_b = [node for node in nodes if node not in assign_b]

    if missing_a or missing_b:
        raise PartitionMetricError(
            "assignments are missing common graph nodes: "
            f"missing_in_a={missing_a[:5]}, missing_in_b={missing_b[:5]}"
        )

    districts_a = _sorted_labels({assign_a[node] for node in nodes})
    districts_b = _sorted_labels({assign_b[node] for node in nodes})

    idx_a = {district: i for i, district in enumerate(districts_a)}
    idx_b = {district: j for j, district in enumerate(districts_b)}

    overlap = np.zeros((len(districts_a), len(districts_b)), dtype=float)
    total_weight = 0.0

    for node in nodes:
        weight = _node_weight(part_a, node, weight_col)

        if weight < 0:
            raise PartitionMetricError(
                f"node {node!r} has negative weight {weight}"
            )

        total_weight += weight
        overlap[idx_a[assign_a[node]], idx_b[assign_b[node]]] += weight

    return PartitionOverlap(
        districts_a=districts_a,
        districts_b=districts_b,
        overlap_matrix=overlap,
        total_weight=float(total_weight),
        n_nodes=len(nodes),
        weight_col=weight_col,
    )


def transfer_distance_result(
    part_a: Any,
    part_b: Any,
    *,
    weight_col: str | None = None,
    normalize: bool = False,
    require_same_nodes: bool = True,
) -> TransferDistanceResult:
    """
    Compute label-invariant transfer distance between two partitions.

    This is the cleaned package version of the old
    `unconstrained_transfer_distance` helper.

    The distance is:

        total_weight - maximum_weighted_overlap_after_relabeling

    The maximizing relabeling is found with the Hungarian algorithm.
    """
    overlap = partition_overlap(
        part_a,
        part_b,
        weight_col=weight_col,
        require_same_nodes=require_same_nodes,
    )

    if overlap.total_weight <= 0:
        raise PartitionMetricError("total node weight is zero")

    # Hungarian solves a minimization problem. We maximize overlap by minimizing
    # negative overlap.
    cost_matrix = -overlap.overlap_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    max_overlap = float(overlap.overlap_matrix[row_ind, col_ind].sum())
    distance = float(overlap.total_weight - max_overlap)

    normalized_distance = None
    if normalize:
        normalized_distance = distance / overlap.total_weight

    return TransferDistanceResult(
        distance=distance,
        normalized_distance=normalized_distance,
        max_overlap=max_overlap,
        total_weight=float(overlap.total_weight),
        n_nodes=overlap.n_nodes,
        n_districts_a=len(overlap.districts_a),
        n_districts_b=len(overlap.districts_b),
        weight_col=weight_col,
    )


def unconstrained_transfer_distance(
    part_a: Any,
    part_b: Any,
    *,
    weight_col: str | None = None,
    require_same_nodes: bool = True,
) -> float:
    """
    Minimum node/weight transfer distance up to district-label permutation.

    If `weight_col=None`, this is a raw node-count transfer distance.
    If `weight_col="POP20"`, this is a population-weighted transfer distance.
    """
    return transfer_distance_result(
        part_a,
        part_b,
        weight_col=weight_col,
        normalize=False,
        require_same_nodes=require_same_nodes,
    ).distance


def normalized_transfer_distance(
    part_a: Any,
    part_b: Any,
    *,
    weight_col: str | None = None,
    require_same_nodes: bool = True,
) -> float:
    """
    Transfer distance divided by total node weight.
    """
    result = transfer_distance_result(
        part_a,
        part_b,
        weight_col=weight_col,
        normalize=True,
        require_same_nodes=require_same_nodes,
    )

    assert result.normalized_distance is not None
    return result.normalized_distance


def unit_transfer_distance(part_a: Any, part_b: Any) -> float:
    """
    Convenience wrapper for unweighted transfer distance.
    """
    return unconstrained_transfer_distance(part_a, part_b, weight_col=None)


def population_transfer_distance(
    part_a: Any,
    part_b: Any,
    *,
    pop_col: str = "POP20",
) -> float:
    """
    Convenience wrapper for population-weighted transfer distance.
    """
    return unconstrained_transfer_distance(part_a, part_b, weight_col=pop_col)

def optimal_transport_distance(
    part_a: Any,
    part_b: Any,
    *,
    pop_col: str = "POP20",
    indicator: str = "population",
    **pair_kwargs: Any,
) -> float:
    """
    Compute Wasserplan optimal transport distance between two GerryChain partitions.

    This wraps Wasserplan's `Pair(part_a, part_b, indicator="population",
    pop_col=...)` interface, matching the prototype in `atlas-to-gerrychain`.

    Parameters
    ----------
    part_a, part_b:
        GerryChain Partition-like objects.

    pop_col:
        Node population attribute.

    indicator:
        Wasserplan indicator argument. Default is `"population"`.

    pair_kwargs:
        Additional keyword arguments passed to `wasserplan.Pair`.

    Returns
    -------
    float
        Wasserplan distance.
    """
    Pair = _import_wasserplan_pair()

    pair = Pair(
        part_a,
        part_b,
        indicator=indicator,
        pop_col=pop_col,
        **pair_kwargs,
    )

    distance = getattr(pair, "distance", None)

    if distance is None:
        raise PartitionMetricError("Wasserplan Pair object has no `distance` attribute")

    if callable(distance):
        distance = distance()

    try:
        return float(distance)
    except (TypeError, ValueError) as exc:
        raise PartitionMetricError(
            f"could not convert Wasserplan distance {distance!r} to float"
        ) from exc


def get_partition_metric(
    metric: PartitionMetricName | Callable[[Any, Any], float],
    *,
    pop_col: str = "POP20",
    weight_col: str | None = None,
) -> Callable[[Any, Any], float]:
    """
    Resolve a metric name or callable into a pairwise distance function.
    """
    if callable(metric):
        return metric

    if metric in {"wasserplan", "optimal_transport"}:
        return lambda a, b: optimal_transport_distance(a, b, pop_col=pop_col)

    if metric == "unit_transfer":
        return lambda a, b: unconstrained_transfer_distance(a, b, weight_col=None)

    if metric == "population_transfer":
        return lambda a, b: unconstrained_transfer_distance(a, b, weight_col=pop_col)

    if metric == "transfer":
        return lambda a, b: unconstrained_transfer_distance(a, b, weight_col=weight_col)

    raise ValueError(f"unknown partition metric: {metric!r}")


def pairwise_partition_distance_matrix(
    partitions: Sequence[Any],
    *,
    metric: PartitionMetricName | Callable[[Any, Any], float] = "transfer",
    labels: Sequence[str] | None = None,
    pop_col: str = "POP20",
    weight_col: str | None = None,
) -> PairwiseDistanceMatrix:
    """
    Compute a symmetric pairwise distance matrix for partitions.

    This will be useful for PACE clustering, medoid selection, and sanity checks.
    """
    parts = list(partitions)

    if not parts:
        raise PartitionMetricError("no partitions were provided")

    if labels is None:
        labels_list = [str(i) for i in range(len(parts))]
    else:
        labels_list = [str(x) for x in labels]

    if len(labels_list) != len(parts):
        raise PartitionMetricError("labels length must match number of partitions")

    distance_fn = get_partition_metric(
        metric,
        pop_col=pop_col,
        weight_col=weight_col,
    )

    matrix = np.zeros((len(parts), len(parts)), dtype=float)

    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            d = float(distance_fn(parts[i], parts[j]))
            matrix[i, j] = d
            matrix[j, i] = d

    metric_name = metric if isinstance(metric, str) else getattr(metric, "__name__", "callable")

    return PairwiseDistanceMatrix(
        labels=labels_list,
        matrix=matrix,
        metric=str(metric_name),
    )