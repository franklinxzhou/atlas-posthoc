import pytest

from atlas_posthoc.assignments.multiscale import (
    AmbiguousAssignmentError,
    MissingAssignmentError,
    build_node_key,
    expand_multiscale_assignment,
    key_prefixes,
    normalize_multiscale_assignment,
)


def test_build_node_key():
    node = {"id": 1, "county": "FRANKLIN", "prec_id": "001"}
    assert build_node_key(node, ["county", "prec_id"]) == ("FRANKLIN", "001")


def test_key_prefixes():
    assert key_prefixes(("A", "B", "C")) == [
        ("A",),
        ("A", "B"),
        ("A", "B", "C"),
    ]


def test_county_level_assignment_expands_to_precinct_nodes():
    assignment = {
        ("FRANKLIN",): 3,
    }

    nodes = [
        {"id": 1, "county": "FRANKLIN", "prec_id": "001"},
        {"id": 2, "county": "FRANKLIN", "prec_id": "002"},
    ]

    result = expand_multiscale_assignment(
        assignment,
        nodes,
        key_cols=["county", "prec_id"],
    )

    assert result.node_to_district == {1: 3, 2: 3}
    assert result.node_to_assignment_key == {
        1: ("FRANKLIN",),
        2: ("FRANKLIN",),
    }
    assert result.summary.n_nodes == 2
    assert result.summary.n_assigned == 2
    assert result.summary.n_missing == 0
    assert result.summary.n_ambiguous == 0


def test_precinct_level_assignment_expands_to_matching_nodes():
    assignment = {
        ("CUYAHOGA", "CLE-01-A"): 11,
        ("CUYAHOGA", "CLE-01-B"): 11,
    }

    nodes = [
        {"id": 1, "county": "CUYAHOGA", "prec_id": "CLE-01-A"},
        {"id": 2, "county": "CUYAHOGA", "prec_id": "CLE-01-B"},
    ]

    result = expand_multiscale_assignment(
        assignment,
        nodes,
        key_cols=["county", "prec_id"],
    )

    assert result.node_to_district == {1: 11, 2: 11}
    assert result.node_to_assignment_key == {
        1: ("CUYAHOGA", "CLE-01-A"),
        2: ("CUYAHOGA", "CLE-01-B"),
    }


def test_mixed_county_and_precinct_assignment():
    assignment = {
        ("FRANKLIN",): 3,
        ("CUYAHOGA", "CLE-01-A"): 11,
        ("CUYAHOGA", "CLE-01-B"): 11,
        ("HAMILTON", "CIN-01-A"): 1,
    }

    nodes = [
        {"id": 1, "county": "FRANKLIN", "prec_id": "COL-01-A"},
        {"id": 2, "county": "CUYAHOGA", "prec_id": "CLE-01-A"},
        {"id": 3, "county": "CUYAHOGA", "prec_id": "CLE-01-B"},
        {"id": 4, "county": "HAMILTON", "prec_id": "CIN-01-A"},
    ]

    result = expand_multiscale_assignment(
        assignment,
        nodes,
        key_cols=["county", "prec_id"],
    )

    assert result.node_to_district == {
        1: 3,
        2: 11,
        3: 11,
        4: 1,
    }
    assert result.node_to_assignment_key == {
        1: ("FRANKLIN",),
        2: ("CUYAHOGA", "CLE-01-A"),
        3: ("CUYAHOGA", "CLE-01-B"),
        4: ("HAMILTON", "CIN-01-A"),
    }
    assert result.summary.unmatched_assignment_keys == []


def test_raw_atlas_string_keys_are_normalized():
    assignment = {
        '["FRANKLIN"]': "3",
        '["CUYAHOGA", "CLE-01-A"]': "11",
    }

    normalized = normalize_multiscale_assignment(assignment)

    assert normalized == {
        ("FRANKLIN",): 3,
        ("CUYAHOGA", "CLE-01-A"): 11,
    }


def test_missing_node_raises_by_default():
    assignment = {
        ("FRANKLIN",): 3,
    }

    nodes = [
        {"id": 1, "county": "SUMMIT", "prec_id": "AKR-01-A"},
    ]

    with pytest.raises(MissingAssignmentError):
        expand_multiscale_assignment(
            assignment,
            nodes,
            key_cols=["county", "prec_id"],
        )


def test_missing_policy_none_keeps_node_with_none():
    assignment = {
        ("FRANKLIN",): 3,
    }

    nodes = [
        {"id": 1, "county": "SUMMIT", "prec_id": "AKR-01-A"},
    ]

    result = expand_multiscale_assignment(
        assignment,
        nodes,
        key_cols=["county", "prec_id"],
        missing_policy="none",
    )

    assert result.node_to_district == {1: None}
    assert result.node_to_assignment_key == {1: None}
    assert result.summary.n_missing == 1


def test_missing_policy_drop_omits_node():
    assignment = {
        ("FRANKLIN",): 3,
    }

    nodes = [
        {"id": 1, "county": "SUMMIT", "prec_id": "AKR-01-A"},
    ]

    result = expand_multiscale_assignment(
        assignment,
        nodes,
        key_cols=["county", "prec_id"],
        missing_policy="drop",
    )

    assert result.node_to_district == {}
    assert result.node_to_assignment_key == {}
    assert result.summary.n_missing == 1


def test_ambiguous_county_and_precinct_match_raises_by_default():
    assignment = {
        ("MONTGOMERY",): 10,
        ("MONTGOMERY", "DAY-01-A"): 4,
    }

    nodes = [
        {"id": 1, "county": "MONTGOMERY", "prec_id": "DAY-01-A"},
    ]

    with pytest.raises(AmbiguousAssignmentError):
        expand_multiscale_assignment(
            assignment,
            nodes,
            key_cols=["county", "prec_id"],
        )


def test_ambiguous_policy_finest():
    assignment = {
        ("MONTGOMERY",): 10,
        ("MONTGOMERY", "DAY-01-A"): 4,
    }

    nodes = [
        {"id": 1, "county": "MONTGOMERY", "prec_id": "DAY-01-A"},
    ]

    result = expand_multiscale_assignment(
        assignment,
        nodes,
        key_cols=["county", "prec_id"],
        ambiguous_policy="finest",
    )

    assert result.node_to_district == {1: 4}
    assert result.node_to_assignment_key == {1: ("MONTGOMERY", "DAY-01-A")}


def test_ambiguous_policy_coarsest():
    assignment = {
        ("MONTGOMERY",): 10,
        ("MONTGOMERY", "DAY-01-A"): 4,
    }

    nodes = [
        {"id": 1, "county": "MONTGOMERY", "prec_id": "DAY-01-A"},
    ]

    result = expand_multiscale_assignment(
        assignment,
        nodes,
        key_cols=["county", "prec_id"],
        ambiguous_policy="coarsest",
    )

    assert result.node_to_district == {1: 10}
    assert result.node_to_assignment_key == {1: ("MONTGOMERY",)}