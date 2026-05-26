import json

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


def write_tiny_graph_json(path):
    graph = {
        "directed": False,
        "multigraph": False,
        "graph": [],
        "nodes": [
            {"id": 0, "NAME": "NODE_0", "POP20": 100},
            {"id": 1, "NAME": "NODE_1", "POP20": 100},
            {"id": 2, "NAME": "NODE_2", "POP20": 100},
        ],
        "adjacency": [
            [{"id": 1, "length": 1.0}],
            [{"id": 0, "length": 1.0}, {"id": 2, "length": 1.0}],
            [{"id": 1, "length": 1.0}],
        ],
    }

    path.write_text(json.dumps(graph), encoding="utf-8")


def write_tiny_atlas(path, records):
    lines = [
        json.dumps("This is an Atlas for Redistricting Maps"),
        json.dumps(
            {
                "description": "tiny AtlasIO partition test",
                "date": "2026-01-01T00:00:00",
                "atlasParamType": "Dict{String, Any}",
                "mapParamType": "Dict{String, Any}",
            }
        ),
        json.dumps({"districts": 2}),
    ]

    for record in records:
        lines.append(json.dumps(record))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def atlas_record(name, d0, d1, d2):
    return {
        "name": name,
        "weight": 1,
        "data": {},
        "districting": [
            {'["NODE_0"]': d0},
            {'["NODE_1"]': d1},
            {'["NODE_2"]': d2},
        ],
    }


def test_build_chain_partitions(tmp_path):
    from atlas_posthoc.pipeline.chain_partitions import build_chain_partitions

    graph_path = tmp_path / "graph.json"
    chain_path = tmp_path / "chain.jsonl"

    write_tiny_graph_json(graph_path)

    write_tiny_atlas(
        chain_path,
        [
            atlas_record("step0", 1, 1, 2),
            atlas_record("step10", 1, 2, 2),
        ],
    )

    gc_graph = make_tiny_gerrychain_graph()

    result = build_chain_partitions(
        chain_path,
        graph_path,
        gerrychain_graph=gc_graph,
        chain_id="run_a",
        key_cols=["NAME"],
        output_frequency=10,
        max_maps_per_chain=2,
    )

    assert result.chain_id == "run_a"
    assert result.n_partitions == 2
    assert result.steps == [0, 10]
    assert result.selected_key_cols == ["NAME"]
    assert result.key_diagnostic.is_perfect_match

    assert dict(result.partitions[0].assignment) == {0: 1, 1: 1, 2: 2}
    assert dict(result.partitions[1].assignment) == {0: 1, 1: 2, 2: 2}


def test_build_chain_partitions_respects_max_maps(tmp_path):
    from atlas_posthoc.pipeline.chain_partitions import build_chain_partitions

    graph_path = tmp_path / "graph.json"
    chain_path = tmp_path / "chain.jsonl"

    write_tiny_graph_json(graph_path)

    write_tiny_atlas(
        chain_path,
        [
            atlas_record("step0", 1, 1, 2),
            atlas_record("step10", 1, 2, 2),
            atlas_record("step20", 2, 2, 1),
        ],
    )

    gc_graph = make_tiny_gerrychain_graph()

    result = build_chain_partitions(
        chain_path,
        graph_path,
        gerrychain_graph=gc_graph,
        chain_id="run_a",
        key_cols=["NAME"],
        max_maps_per_chain=1,
    )

    assert result.n_partitions == 1
    assert result.names == ["step0"]


def test_build_multi_chain_partitions(tmp_path):
    from atlas_posthoc.pipeline.chain_partitions import build_multi_chain_partitions

    graph_path = tmp_path / "graph.json"
    chain_a = tmp_path / "chain_a.jsonl"
    chain_b = tmp_path / "chain_b.jsonl"

    write_tiny_graph_json(graph_path)

    write_tiny_atlas(
        chain_a,
        [
            atlas_record("step0", 1, 1, 2),
            atlas_record("step10", 1, 2, 2),
        ],
    )

    write_tiny_atlas(
        chain_b,
        [
            atlas_record("step0", 1, 1, 2),
            atlas_record("step10", 2, 2, 1),
        ],
    )

    gc_graph = make_tiny_gerrychain_graph()

    result = build_multi_chain_partitions(
        [chain_a, chain_b],
        graph_path,
        gerrychain_graph=gc_graph,
        chain_ids=["run_a", "run_b"],
        key_cols=["NAME"],
        max_maps_per_chain=2,
    )

    assert set(result.chains.keys()) == {"run_a", "run_b"}
    assert len(result.partitions_by_chain["run_a"]) == 2
    assert len(result.partitions_by_chain["run_b"]) == 2
    assert len(result.pooled_partitions) == 4
    assert result.pooled_chain_ids == ["run_a", "run_a", "run_b", "run_b"]

    summaries = result.summaries()
    assert summaries["run_a"]["n_partitions"] == 2
    assert summaries["run_b"]["n_partitions"] == 2


def test_build_multi_chain_partitions_rejects_bad_chain_ids(tmp_path):
    from atlas_posthoc.pipeline.chain_partitions import (
        ChainPartitionError,
        build_multi_chain_partitions,
    )

    graph_path = tmp_path / "graph.json"
    chain_a = tmp_path / "chain_a.jsonl"

    write_tiny_graph_json(graph_path)
    write_tiny_atlas(chain_a, [atlas_record("step0", 1, 1, 2)])

    gc_graph = make_tiny_gerrychain_graph()

    with pytest.raises(ChainPartitionError):
        build_multi_chain_partitions(
            [chain_a],
            graph_path,
            gerrychain_graph=gc_graph,
            chain_ids=["run_a", "extra"],
            key_cols=["NAME"],
        )