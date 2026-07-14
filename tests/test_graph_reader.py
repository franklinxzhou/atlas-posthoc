import json

import pytest

from atlas_posthoc.io.graph_reader import (
    GraphReaderError,
    adjacency_edges,
    infer_key_cols_for_multiscale,
    read_graph_json,
    read_graph_nodes,
    require_node_columns,
    sum_numeric_node_column,
)


def write_tiny_graph(path):
    graph = {
        "directed": False,
        "multigraph": False,
        "graph": [],
        "nodes": [
            {
                "id": 0,
                "COUNTY": "FRANKLIN",
                "NAME": "COLUMBUS 01_FRANKLIN",
                "ORIG_NAME": "COLUMBUS 01_FRANKLIN",
                "POP20": 100,
                "G20PREDEM": 60,
                "G20PREREP": 40,
                "PLACE": {"Columbus": 75, "Bexley": 5},
            },
            {
                "id": 1,
                "COUNTY": "FRANKLIN",
                "NAME": "COLUMBUS 02_FRANKLIN",
                "ORIG_NAME": "COLUMBUS 02_FRANKLIN",
                "POP20": 200,
                "G20PREDEM": 80,
                "G20PREREP": 120,
            },
            {
                "id": 2,
                "COUNTY": "CUYAHOGA",
                "NAME": "CLEVELAND 01_CUYAHOGA",
                "ORIG_NAME": "CLEVELAND 01_CUYAHOGA",
                "POP20": 300,
                "G20PREDEM": 180,
                "G20PREREP": 120,
            },
        ],
        "adjacency": [
            [{"id": 1, "length": 10.0}],
            [{"id": 0, "length": 10.0}, {"id": 2, "length": 20.0}],
            [{"id": 1, "length": 20.0}],
        ],
    }

    path.write_text(json.dumps(graph), encoding="utf-8")


def test_read_graph_json(tmp_path):
    path = tmp_path / "tiny_graph.json"
    write_tiny_graph(path)

    graph = read_graph_json(path)

    assert len(graph.nodes) == 3
    assert len(graph.adjacency) == 3
    assert graph.node_by_id[0]["COUNTY"] == "FRANKLIN"
    assert graph.node_by_id[2]["COUNTY"] == "CUYAHOGA"
    assert graph.node_by_id[0]["PLACE"] == {"Columbus": 75, "Bexley": 5}


def test_read_graph_nodes(tmp_path):
    path = tmp_path / "tiny_graph.json"
    write_tiny_graph(path)

    nodes = read_graph_nodes(path)

    assert len(nodes) == 3
    assert nodes[0]["id"] == 0


def test_require_node_columns(tmp_path):
    path = tmp_path / "tiny_graph.json"
    write_tiny_graph(path)

    graph = read_graph_json(path)

    require_node_columns(
        graph.nodes,
        ["id", "COUNTY", "NAME", "POP20", "G20PREDEM", "G20PREREP"],
    )

    with pytest.raises(GraphReaderError):
        require_node_columns(graph.nodes, ["NOT_A_COLUMN"])


def test_sum_numeric_node_column(tmp_path):
    path = tmp_path / "tiny_graph.json"
    write_tiny_graph(path)

    graph = read_graph_json(path)

    assert sum_numeric_node_column(graph.nodes, "POP20") == 600.0
    assert graph.total_population("POP20") == 600.0

    totals = graph.election_totals("G20PREDEM", "G20PREREP")

    assert totals["dem"] == 320.0
    assert totals["rep"] == 280.0
    assert totals["total"] == 600.0
    assert totals["dem_share"] == pytest.approx(320.0 / 600.0)


def test_adjacency_edges_deduplicates_undirected_edges(tmp_path):
    path = tmp_path / "tiny_graph.json"
    write_tiny_graph(path)

    graph = read_graph_json(path)

    edges = adjacency_edges(graph)

    assert len(edges) == 2
    assert (0, 1, {"length": 10.0}) in edges
    assert (1, 2, {"length": 20.0}) in edges


def test_infer_key_cols_for_multiscale(tmp_path):
    path = tmp_path / "tiny_graph.json"
    write_tiny_graph(path)

    graph = read_graph_json(path)

    assert infer_key_cols_for_multiscale(graph) == ["COUNTY", "NAME"]


def test_duplicate_node_id_raises(tmp_path):
    path = tmp_path / "bad_graph.json"
    graph = {
        "nodes": [
            {"id": 0, "COUNTY": "FRANKLIN"},
            {"id": 0, "COUNTY": "CUYAHOGA"},
        ],
        "adjacency": [[], []],
    }
    path.write_text(json.dumps(graph), encoding="utf-8")

    with pytest.raises(GraphReaderError):
        read_graph_json(path)


def test_bad_adjacency_length_raises(tmp_path):
    path = tmp_path / "bad_graph.json"
    graph = {
        "nodes": [
            {"id": 0, "COUNTY": "FRANKLIN"},
            {"id": 1, "COUNTY": "CUYAHOGA"},
        ],
        "adjacency": [[]],
    }
    path.write_text(json.dumps(graph), encoding="utf-8")

    with pytest.raises(GraphReaderError):
        read_graph_json(path)
