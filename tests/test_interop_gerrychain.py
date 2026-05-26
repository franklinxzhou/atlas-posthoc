import pytest


def make_tiny_gerrychain_graph():
    gerrychain = pytest.importorskip("gerrychain")
    nx = pytest.importorskip("networkx")

    graph = nx.path_graph(3)

    for node in graph.nodes:
        graph.nodes[node]["id"] = node
        graph.nodes[node]["NAME"] = f"NODE_{node}"
        graph.nodes[node]["POP20"] = 100

    if not hasattr(gerrychain.Graph, "from_networkx"):
        pytest.skip("GerryChain Graph.from_networkx is unavailable")

    return gerrychain.Graph.from_networkx(graph)


def test_build_graph_node_attribute_crosswalk():
    from atlas_posthoc.interop.gerrychain import (
        build_graph_node_attribute_crosswalk,
    )

    graph = make_tiny_gerrychain_graph()

    crosswalk = build_graph_node_attribute_crosswalk(
        graph,
        attr_col="NAME",
    )

    assert crosswalk == {
        "NODE_0": 0,
        "NODE_1": 1,
        "NODE_2": 2,
    }


def test_partition_from_node_to_district():
    from atlas_posthoc.interop.gerrychain import (
        partition_from_node_to_district,
    )

    graph = make_tiny_gerrychain_graph()

    result = partition_from_node_to_district(
        graph,
        {
            0: 1,
            1: 1,
            2: 2,
        },
    )

    assert result.summary.is_complete
    assert result.summary.n_graph_nodes == 3
    assert result.summary.n_assignment_nodes == 3
    assert dict(result.partition.assignment) == {
        0: 1,
        1: 1,
        2: 2,
    }


def test_partition_from_node_to_district_rejects_incomplete_assignment():
    from atlas_posthoc.interop.gerrychain import (
        GerryChainInteropError,
        partition_from_node_to_district,
    )

    graph = make_tiny_gerrychain_graph()

    with pytest.raises(GerryChainInteropError):
        partition_from_node_to_district(
            graph,
            {
                0: 1,
                1: 1,
            },
            require_complete_assignment=True,
        )


def test_partition_from_record_with_name_keys():
    from atlas_posthoc.interop.gerrychain import partition_from_record

    graph = make_tiny_gerrychain_graph()

    graph_nodes = [
        {"id": 0, "NAME": "NODE_0"},
        {"id": 1, "NAME": "NODE_1"},
        {"id": 2, "NAME": "NODE_2"},
    ]

    class DummyRecord:
        assignment = {
            ("NODE_0",): 1,
            ("NODE_1",): 1,
            ("NODE_2",): 2,
        }

    result = partition_from_record(
        DummyRecord(),
        graph,
        graph_nodes,
        key_cols=["NAME"],
    )

    assert result.summary.is_complete
    assert dict(result.partition.assignment) == {
        0: 1,
        1: 1,
        2: 2,
    }