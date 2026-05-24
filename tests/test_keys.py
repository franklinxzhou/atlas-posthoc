import pytest

from atlas_posthoc.assignments.keys import (
    AssignmentKeyError,
    choose_best_key_scheme,
    diagnose_key_scheme,
    diagnose_key_schemes,
    format_key_scheme_diagnostics,
)


def ohio_nodes():
    return [
        {
            "id": 1,
            "COUNTY": "FRANKLIN",
            "NAME": "COLUMBUS 01_FRANKLIN",
            "ORIG_NAME": "COLUMBUS 01_FRANKLIN",
            "MCD": "3904900001",
        },
        {
            "id": 2,
            "COUNTY": "FRANKLIN",
            "NAME": "COLUMBUS 02_FRANKLIN",
            "ORIG_NAME": "COLUMBUS 02_FRANKLIN",
            "MCD": "3904900002",
        },
        {
            "id": 3,
            "COUNTY": "CUYAHOGA",
            "NAME": "CLEVELAND 01_CUYAHOGA",
            "ORIG_NAME": "CLEVELAND 01_CUYAHOGA",
            "MCD": "3903500001",
        },
    ]


def test_diagnose_county_level_scheme():
    assignment = {
        ("FRANKLIN",): 3,
        ("CUYAHOGA",): 11,
    }

    diagnostic = diagnose_key_scheme(
        assignment,
        ohio_nodes(),
        ["COUNTY", "NAME"],
    )

    assert diagnostic.n_nodes == 3
    assert diagnostic.n_assigned == 3
    assert diagnostic.n_missing == 0
    assert diagnostic.n_ambiguous == 0
    assert diagnostic.n_assignment_keys == 2
    assert diagnostic.n_used_assignment_keys == 2
    assert diagnostic.n_unmatched_assignment_keys == 0
    assert diagnostic.is_perfect_match


def test_diagnose_name_level_scheme():
    assignment = {
        ("COLUMBUS 01_FRANKLIN",): 3,
        ("COLUMBUS 02_FRANKLIN",): 3,
        ("CLEVELAND 01_CUYAHOGA",): 11,
    }

    diagnostic = diagnose_key_scheme(
        assignment,
        ohio_nodes(),
        ["NAME"],
    )

    assert diagnostic.n_assigned == 3
    assert diagnostic.n_missing == 0
    assert diagnostic.n_ambiguous == 0
    assert diagnostic.n_unmatched_assignment_keys == 0
    assert diagnostic.is_perfect_match


def test_diagnose_missing_nodes():
    assignment = {
        ("FRANKLIN",): 3,
    }

    diagnostic = diagnose_key_scheme(
        assignment,
        ohio_nodes(),
        ["COUNTY", "NAME"],
    )

    assert diagnostic.n_assigned == 2
    assert diagnostic.n_missing == 1
    assert diagnostic.n_ambiguous == 0
    assert not diagnostic.is_perfect_match
    assert diagnostic.example_missing_nodes == [3]


def test_diagnose_ambiguous_nodes():
    assignment = {
        ("FRANKLIN",): 3,
        ("FRANKLIN", "COLUMBUS 01_FRANKLIN"): 4,
        ("CUYAHOGA",): 11,
    }

    diagnostic = diagnose_key_scheme(
        assignment,
        ohio_nodes(),
        ["COUNTY", "NAME"],
    )

    assert diagnostic.n_assigned == 2
    assert diagnostic.n_missing == 0
    assert diagnostic.n_ambiguous == 1
    assert not diagnostic.is_perfect_match
    assert diagnostic.example_ambiguous_nodes == [1]


def test_diagnose_unmatched_assignment_key():
    assignment = {
        ("FRANKLIN",): 3,
        ("CUYAHOGA",): 11,
        ("HAMILTON",): 1,
    }

    diagnostic = diagnose_key_scheme(
        assignment,
        ohio_nodes(),
        ["COUNTY", "NAME"],
    )

    assert diagnostic.n_assigned == 3
    assert diagnostic.n_missing == 0
    assert diagnostic.n_ambiguous == 0
    assert diagnostic.n_unmatched_assignment_keys == 1
    assert diagnostic.example_unmatched_assignment_keys == [("HAMILTON",)]
    assert not diagnostic.is_perfect_match


def test_diagnose_key_schemes_sorts_best_first():
    assignment = {
        ("FRANKLIN",): 3,
        ("CUYAHOGA",): 11,
    }

    diagnostics = diagnose_key_schemes(
        assignment,
        ohio_nodes(),
        candidate_key_cols=[
            ["NAME"],
            ["COUNTY", "NAME"],
            ["COUNTY"],
        ],
    )

    assert diagnostics[0].n_assigned == 3
    assert diagnostics[0].n_missing == 0


def test_choose_best_key_scheme():
    assignment = {
        ("FRANKLIN",): 3,
        ("CUYAHOGA",): 11,
    }

    best = choose_best_key_scheme(
        assignment,
        ohio_nodes(),
        candidate_key_cols=[
            ["NAME"],
            ["COUNTY", "NAME"],
            ["COUNTY"],
        ],
    )

    assert best.is_perfect_match
    assert best.n_assigned == 3


def test_choose_best_key_scheme_raises_when_no_perfect_match():
    assignment = {
        ("FRANKLIN",): 3,
    }

    with pytest.raises(AssignmentKeyError):
        choose_best_key_scheme(
            assignment,
            ohio_nodes(),
            candidate_key_cols=[
                ["COUNTY", "NAME"],
                ["NAME"],
            ],
            require_perfect=True,
        )


def test_format_key_scheme_diagnostics():
    assignment = {
        ("FRANKLIN",): 3,
        ("CUYAHOGA",): 11,
    }

    diagnostics = diagnose_key_schemes(
        assignment,
        ohio_nodes(),
        candidate_key_cols=[
            ["COUNTY", "NAME"],
            ["NAME"],
        ],
    )

    text = format_key_scheme_diagnostics(diagnostics)

    assert "key_cols" in text
    assert "assigned" in text
    assert "missing" in text