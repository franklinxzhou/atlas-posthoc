import networkx as nx
import pytest

from atlas_posthoc.pipeline.pace_workflow import (
    PACEWorkflowError,
    run_pace_workflow_from_partitions,
)


class FakePartition:
    def __init__(self, graph, assignment):
        self.graph = graph
        self.assignment = assignment


def make_graph():
    graph = nx.path_graph(4)
    for node in graph.nodes:
        graph.nodes[node]["POP20"] = 100
    return graph


def make_state_a(graph):
    return FakePartition(graph, {0: 1, 1: 1, 2: 2, 3: 2})


def make_state_a_relabelled(graph):
    return FakePartition(graph, {0: 7, 1: 7, 2: 8, 3: 8})


def make_state_b(graph):
    return FakePartition(graph, {0: 1, 1: 2, 2: 1, 3: 2})


def test_run_pace_workflow_balanced_exact_transfer_clusters():
    graph = make_graph()

    partitions = {
        "run1": [make_state_a(graph), make_state_b(graph)],
        "run2": [make_state_a_relabelled(graph), make_state_b(graph)],
    }

    result = run_pace_workflow_from_partitions(
        partitions,
        metric="unit_transfer",
        clustering_method="threshold",
        distance_threshold=0.0,
        pace_threshold=0.1,
        apply_correction=False,
    )

    assert result.n_samples == 4
    assert result.n_chains == 2
    assert result.chain_sizes == {"run1": 2, "run2": 2}
    assert result.distance_matrix.matrix.shape == (4, 4)

    # Both chains have one sample in each exact transfer-distance cluster.
    assert result.pace.delta == pytest.approx(0.0)
    assert result.pace.converged is True


def test_run_pace_workflow_unbalanced_clusters_warns_nonconvergence():
    graph = make_graph()

    partitions = {
        "run1": [make_state_a(graph), make_state_a_relabelled(graph)],
        "run2": [make_state_b(graph), make_state_b(graph)],
    }

    result = run_pace_workflow_from_partitions(
        partitions,
        metric="unit_transfer",
        clustering_method="threshold",
        distance_threshold=0.0,
        pace_threshold=0.1,
        apply_correction=False,
    )

    assert result.pace.delta == pytest.approx(0.5)
    assert result.pace.converged is False


def test_run_pace_workflow_farthest_first():
    graph = make_graph()

    partitions = {
        "run1": [make_state_a(graph), make_state_b(graph)],
        "run2": [make_state_a_relabelled(graph), make_state_b(graph)],
    }

    result = run_pace_workflow_from_partitions(
        partitions,
        metric="unit_transfer",
        clustering_method="farthest_first",
        n_clusters=2,
        pace_threshold=0.1,
        apply_correction=False,
    )

    assert result.clustering.n_clusters == 2
    assert result.pace.n_cells == 2


def test_run_pace_workflow_requires_two_chains():
    graph = make_graph()

    partitions = {
        "run1": [make_state_a(graph), make_state_b(graph)],
    }

    with pytest.raises(PACEWorkflowError):
        run_pace_workflow_from_partitions(
            partitions,
            metric="unit_transfer",
            clustering_method="threshold",
            distance_threshold=0.0,
        )


def test_run_pace_workflow_rejects_empty_chain():
    graph = make_graph()

    partitions = {
        "run1": [make_state_a(graph)],
        "run2": [],
    }

    with pytest.raises(PACEWorkflowError):
        run_pace_workflow_from_partitions(
            partitions,
            metric="unit_transfer",
            clustering_method="threshold",
            distance_threshold=0.0,
        )