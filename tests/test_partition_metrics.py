import networkx as nx
import numpy as np
import pytest

from atlas_posthoc.diagnostics.partition_metrics import (
    PartitionMetricError,
    normalized_transfer_distance,
    pairwise_partition_distance_matrix,
    partition_overlap,
    population_transfer_distance,
    transfer_distance_result,
    unconstrained_transfer_distance,
    unit_transfer_distance,
)


class FakePartition:
    def __init__(self, graph, assignment):
        self.graph = graph
        self.assignment = assignment


def make_graph():
    graph = nx.path_graph(4)

    pops = {
        0: 10,
        1: 20,
        2: 30,
        3: 40,
    }

    for node, pop in pops.items():
        graph.nodes[node]["POP20"] = pop

    return graph


def test_partition_overlap_unweighted():
    graph = make_graph()

    part_a = FakePartition(graph, {0: 1, 1: 1, 2: 2, 3: 2})
    part_b = FakePartition(graph, {0: 7, 1: 7, 2: 8, 3: 8})

    overlap = partition_overlap(part_a, part_b)

    assert overlap.total_weight == 4
    assert overlap.n_nodes == 4
    assert overlap.overlap_matrix.shape == (2, 2)
    assert overlap.overlap_matrix.sum() == 4


def test_unconstrained_transfer_distance_label_invariant_zero():
    graph = make_graph()

    part_a = FakePartition(graph, {0: 1, 1: 1, 2: 2, 3: 2})
    part_b = FakePartition(graph, {0: 7, 1: 7, 2: 8, 3: 8})

    assert unconstrained_transfer_distance(part_a, part_b) == 0
    assert unit_transfer_distance(part_a, part_b) == 0


def test_unconstrained_transfer_distance_one_node_changed():
    graph = make_graph()

    part_a = FakePartition(graph, {0: 1, 1: 1, 2: 2, 3: 2})
    part_b = FakePartition(graph, {0: 1, 1: 2, 2: 2, 3: 2})

    result = transfer_distance_result(part_a, part_b)

    assert result.distance == 1
    assert result.max_overlap == 3
    assert result.total_weight == 4
    assert result.n_districts_a == 2
    assert result.n_districts_b == 2


def test_population_transfer_distance():
    graph = make_graph()

    part_a = FakePartition(graph, {0: 1, 1: 1, 2: 2, 3: 2})
    part_b = FakePartition(graph, {0: 1, 1: 2, 2: 2, 3: 2})

    # Node 1, population 20, is the minimum unmatched transfer.
    assert population_transfer_distance(part_a, part_b, pop_col="POP20") == 20


def test_normalized_transfer_distance():
    graph = make_graph()

    part_a = FakePartition(graph, {0: 1, 1: 1, 2: 2, 3: 2})
    part_b = FakePartition(graph, {0: 1, 1: 2, 2: 2, 3: 2})

    assert normalized_transfer_distance(part_a, part_b) == pytest.approx(1 / 4)


def test_pairwise_partition_distance_matrix():
    graph = make_graph()

    p0 = FakePartition(graph, {0: 1, 1: 1, 2: 2, 3: 2})
    p1 = FakePartition(graph, {0: 7, 1: 7, 2: 8, 3: 8})
    p2 = FakePartition(graph, {0: 1, 1: 2, 2: 2, 3: 2})

    result = pairwise_partition_distance_matrix(
        [p0, p1, p2],
        labels=["p0", "p1", "p2"],
        metric="unit_transfer",
    )

    assert result.labels == ["p0", "p1", "p2"]
    assert result.matrix.shape == (3, 3)
    assert np.allclose(np.diag(result.matrix), 0)
    assert result.matrix[0, 1] == 0
    assert result.matrix[0, 2] == 1
    assert result.matrix[1, 2] == 1


def test_missing_assignment_nodes_raise():
    graph = make_graph()

    part_a = FakePartition(graph, {0: 1, 1: 1, 2: 2})
    part_b = FakePartition(graph, {0: 1, 1: 1, 2: 2, 3: 2})

    with pytest.raises(PartitionMetricError):
        unconstrained_transfer_distance(part_a, part_b)


def test_different_graph_node_sets_raise():
    graph_a = make_graph()
    graph_b = nx.path_graph(3)

    for node in graph_b.nodes:
        graph_b.nodes[node]["POP20"] = 1

    part_a = FakePartition(graph_a, {0: 1, 1: 1, 2: 2, 3: 2})
    part_b = FakePartition(graph_b, {0: 1, 1: 1, 2: 2})

    with pytest.raises(PartitionMetricError):
        unconstrained_transfer_distance(part_a, part_b)