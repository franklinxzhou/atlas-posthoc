import json

import numpy as np
import pytest

from atlas_posthoc.diagnostics.rhat import summarize_gelman_rubin_by_rank
from atlas_posthoc.diagnostics.total_variation import (
    get_tv_convergence,
    make_vote_share_bins,
)
from atlas_posthoc.pipeline.chain_observables import (
    ChainObservableError,
    build_chain_observables,
    build_multi_chain_observables,
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
                "POP20": 100,
                "G20PREDEM": 60,
                "G20PREREP": 40,
            },
            {
                "id": 1,
                "COUNTY": "FRANKLIN",
                "NAME": "COLUMBUS 02_FRANKLIN",
                "POP20": 100,
                "G20PREDEM": 30,
                "G20PREREP": 70,
            },
            {
                "id": 2,
                "COUNTY": "CUYAHOGA",
                "NAME": "CLEVELAND 01_CUYAHOGA",
                "POP20": 100,
                "G20PREDEM": 80,
                "G20PREREP": 20,
            },
            {
                "id": 3,
                "COUNTY": "HAMILTON",
                "NAME": "CINCINNATI 01_HAMILTON",
                "POP20": 100,
                "G20PREDEM": 20,
                "G20PREREP": 80,
            },
        ],
        "adjacency": [
            [{"id": 1}],
            [{"id": 0}, {"id": 2}],
            [{"id": 1}, {"id": 3}],
            [{"id": 2}],
        ],
    }

    path.write_text(json.dumps(graph), encoding="utf-8")


def write_tiny_atlas(path, records):
    lines = [
        json.dumps("This is an Atlas for Redistricting Maps"),
        json.dumps(
            {
                "description": "tiny AtlasIO test",
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


def atlas_record(name, d0, d1, d2, d3, compactness):
    return {
        "name": name,
        "weight": 1,
        "data": {
            "get_isoperimetric_scores": compactness,
        },
        "districting": [
            {'["COLUMBUS 01_FRANKLIN"]': d0},
            {'["COLUMBUS 02_FRANKLIN"]': d1},
            {'["CLEVELAND 01_CUYAHOGA"]': d2},
            {'["CINCINNATI 01_HAMILTON"]': d3},
        ],
    }


def test_build_chain_observables(tmp_path):
    graph_path = tmp_path / "tiny_graph.json"
    chain_path = tmp_path / "chain.jsonl"

    write_tiny_graph(graph_path)

    write_tiny_atlas(
        chain_path,
        [
            atlas_record("step0", 1, 1, 2, 2, [10.0, 20.0]),
            atlas_record("step10", 1, 2, 2, 1, [11.0, 19.0]),
        ],
    )

    result = build_chain_observables(
        chain_path,
        graph_path,
        chain_id="run_a",
        key_cols=["NAME"],
        dem_col="G20PREDEM",
        rep_col="G20PREREP",
        n_districts=2,
        require_compactness=True,
    )

    assert result.chain_id == "run_a"
    assert result.selected_key_cols == ["NAME"]
    assert result.steps == [0, 10]
    assert result.vote_shares.shape == (2, 2)
    assert result.compactness_scores.shape == (2, 2)

    # step0:
    # district 1 = nodes 0,1: D=90, R=110, share=.45
    # district 2 = nodes 2,3: D=100, R=100, share=.50
    assert np.allclose(result.vote_shares[0], [0.45, 0.50])

    # compactness sorted ascending
    assert np.allclose(result.compactness_scores[0], [10.0, 20.0])


def test_build_chain_observables_auto_key_scheme(tmp_path):
    graph_path = tmp_path / "tiny_graph.json"
    chain_path = tmp_path / "chain.jsonl"

    write_tiny_graph(graph_path)

    write_tiny_atlas(
        chain_path,
        [
            atlas_record("step0", 1, 1, 2, 2, [10.0, 20.0]),
        ],
    )

    result = build_chain_observables(
        chain_path,
        graph_path,
        chain_id="run_a",
        key_cols=None,
        dem_col="G20PREDEM",
        rep_col="G20PREREP",
        n_districts=2,
        require_compactness=True,
    )

    assert result.selected_key_cols == ["NAME"]
    assert result.key_diagnostic.is_perfect_match


def test_build_multi_chain_observables_feeds_tv_and_rhat(tmp_path):
    graph_path = tmp_path / "tiny_graph.json"
    chain_a_path = tmp_path / "chain_a.jsonl"
    chain_b_path = tmp_path / "chain_b.jsonl"

    write_tiny_graph(graph_path)

    write_tiny_atlas(
        chain_a_path,
        [
            atlas_record("step0", 1, 1, 2, 2, [10.0, 20.0]),
            atlas_record("step10", 1, 2, 2, 1, [11.0, 19.0]),
            atlas_record("step20", 1, 1, 2, 2, [10.5, 20.5]),
            atlas_record("step30", 1, 2, 2, 1, [11.5, 19.5]),
        ],
    )

    write_tiny_atlas(
        chain_b_path,
        [
            atlas_record("step0", 1, 1, 2, 2, [10.1, 20.1]),
            atlas_record("step10", 1, 2, 2, 1, [11.1, 19.1]),
            atlas_record("step20", 1, 1, 2, 2, [10.6, 20.6]),
            atlas_record("step30", 1, 2, 2, 1, [11.6, 19.6]),
        ],
    )

    result = build_multi_chain_observables(
        [chain_a_path, chain_b_path],
        graph_path,
        chain_ids=["run_a", "run_b"],
        key_cols=["NAME"],
        dem_col="G20PREDEM",
        rep_col="G20PREREP",
        n_districts=2,
        require_compactness=True,
    )

    assert set(result.vote_runs.keys()) == {"run_a", "run_b"}
    assert result.vote_runs["run_a"].shape == (4, 2)
    assert result.compactness_runs["run_b"].shape == (4, 2)

    bins = make_vote_share_bins(bin_width=0.1)
    tv = get_tv_convergence(
        result.vote_runs,
        bins,
        output_freq=10,
        snapshots=[2, 4],
    )

    assert tv.snapshots == [2, 4]

    # R-hat needs at least two chains and enough draws.
    rhat_summary = summarize_gelman_rubin_by_rank(result.vote_runs)

    assert rhat_summary.num_ranks == 2
    assert len(rhat_summary.rows) == 2


def test_missing_compactness_required_raises(tmp_path):
    graph_path = tmp_path / "tiny_graph.json"
    chain_path = tmp_path / "chain.jsonl"

    write_tiny_graph(graph_path)

    record = atlas_record("step0", 1, 1, 2, 2, [10.0, 20.0])
    record["data"] = {}

    write_tiny_atlas(chain_path, [record])

    with pytest.raises(ChainObservableError):
        build_chain_observables(
            chain_path,
            graph_path,
            chain_id="run_a",
            key_cols=["NAME"],
            dem_col="G20PREDEM",
            rep_col="G20PREREP",
            n_districts=2,
            require_compactness=True,
        )