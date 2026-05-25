import json

import numpy as np
import pytest

from atlas_posthoc.audit.enacted import (
    EnactedAuditError,
    compute_reference_audit,
    compute_reference_compactness,
    compute_reference_vote_share,
    node_to_district_from_column,
)


def write_tiny_graph(path):
    graph = {
        "directed": False,
        "multigraph": False,
        "graph": [],
        "nodes": [
            {
                "id": 0,
                "DIST": 1,
                "POP20": 100,
                "D": 60,
                "R": 40,
                "area": 10.0,
                "border_length": 2.0,
            },
            {
                "id": 1,
                "DIST": 1,
                "POP20": 100,
                "D": 30,
                "R": 70,
                "area": 10.0,
                "border_length": 2.0,
            },
            {
                "id": 2,
                "DIST": 2,
                "POP20": 100,
                "D": 80,
                "R": 20,
                "area": 10.0,
                "border_length": 2.0,
            },
            {
                "id": 3,
                "DIST": 2,
                "POP20": 100,
                "D": 20,
                "R": 80,
                "area": 10.0,
                "border_length": 2.0,
            },
        ],
        "adjacency": [
            [{"id": 1, "length": 1.0}],
            [{"id": 0, "length": 1.0}, {"id": 2, "length": 3.0}],
            [{"id": 1, "length": 3.0}, {"id": 3, "length": 1.0}],
            [{"id": 2, "length": 1.0}],
        ],
    }

    path.write_text(json.dumps(graph), encoding="utf-8")


def test_node_to_district_from_column():
    nodes = [
        {"id": 10, "DIST": "1"},
        {"id": 11, "DIST": 2},
    ]

    mapping = node_to_district_from_column(
        nodes,
        district_col="DIST",
        node_id_col="id",
    )

    assert mapping == {10: 1, 11: 2}


def test_compute_reference_vote_share_from_district_column(tmp_path):
    path = tmp_path / "graph.json"
    write_tiny_graph(path)

    result = compute_reference_vote_share(
        path,
        district_col="DIST",
        dem_col="D",
        rep_col="R",
        n_districts=2,
    )

    # District 1: D=90, R=110 -> 0.45
    # District 2: D=100, R=100 -> 0.50
    assert result.ordered_vote_shares == [0.45, 0.50]
    assert result.missing_districts == []


def test_compute_reference_vote_share_from_mapping(tmp_path):
    path = tmp_path / "graph.json"
    write_tiny_graph(path)

    mapping = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
    }

    result = compute_reference_vote_share(
        path,
        node_to_district=mapping,
        dem_col="D",
        rep_col="R",
        n_districts=2,
    )

    assert result.ordered_vote_shares == [0.45, 0.50]


def test_compute_reference_compactness_from_graph(tmp_path):
    path = tmp_path / "graph.json"
    write_tiny_graph(path)

    result = compute_reference_compactness(
        path,
        district_col="DIST",
        n_districts=2,
        edge_length_col="length",
    )

    assert len(result.ordered_scores) == 2
    assert result.edge_length_col == "length"

    # District 1:
    # area = 20
    # perimeter = node borders 2+2 + crossing edge 1-2 length 3 = 7
    # score = 49/20
    #
    # District 2 is symmetric.
    assert result.ordered_scores == [pytest.approx(49 / 20), pytest.approx(49 / 20)]


def test_compute_reference_audit(tmp_path):
    path = tmp_path / "graph.json"
    write_tiny_graph(path)

    audit = compute_reference_audit(
        path,
        district_col="DIST",
        dem_col="D",
        rep_col="R",
        n_districts=2,
        edge_length_col="length",
    )

    assert audit.vote_share is not None
    assert audit.compactness is not None
    assert audit.vote_share.ordered_vote_shares == [0.45, 0.50]
    assert len(audit.compactness.ordered_scores) == 2


def test_missing_assignment_source_raises(tmp_path):
    path = tmp_path / "graph.json"
    write_tiny_graph(path)

    with pytest.raises(EnactedAuditError):
        compute_reference_vote_share(
            path,
            dem_col="D",
            rep_col="R",
        )


def test_missing_district_column_raises(tmp_path):
    path = tmp_path / "graph.json"
    write_tiny_graph(path)

    with pytest.raises(EnactedAuditError):
        compute_reference_vote_share(
            path,
            district_col="NOT_A_COLUMN",
            dem_col="D",
            rep_col="R",
        )