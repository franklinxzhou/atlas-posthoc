import pytest

from atlas_posthoc.observables.vote_share import (
    VoteShareError,
    aggregate_district_votes,
    compute_ordered_dem_vote_shares,
)


def test_aggregate_district_votes():
    nodes = [
        {"id": 1, "G20PREDEM": 60, "G20PREREP": 40},
        {"id": 2, "G20PREDEM": 30, "G20PREREP": 70},
        {"id": 3, "G20PREDEM": 80, "G20PREREP": 20},
    ]

    node_to_district = {
        1: 1,
        2: 1,
        3: 2,
    }

    totals = aggregate_district_votes(
        node_to_district,
        nodes,
        dem_col="G20PREDEM",
        rep_col="G20PREREP",
    )

    assert len(totals) == 2

    d1 = totals[0]
    d2 = totals[1]

    assert d1.district == 1
    assert d1.dem == 90
    assert d1.rep == 110
    assert d1.total == 200
    assert d1.dem_share == 0.45

    assert d2.district == 2
    assert d2.dem == 80
    assert d2.rep == 20
    assert d2.total == 100
    assert d2.dem_share == 0.80


def test_compute_ordered_dem_vote_shares_ascending():
    nodes = [
        {"id": 1, "G20PREDEM": 60, "G20PREREP": 40},
        {"id": 2, "G20PREDEM": 30, "G20PREREP": 70},
        {"id": 3, "G20PREDEM": 80, "G20PREREP": 20},
        {"id": 4, "G20PREDEM": 20, "G20PREREP": 80},
    ]

    node_to_district = {
        1: 1,
        2: 1,
        3: 2,
        4: 3,
    }

    result = compute_ordered_dem_vote_shares(
        node_to_district,
        nodes,
        dem_col="G20PREDEM",
        rep_col="G20PREREP",
        n_districts=3,
    )

    assert result.ordered_vote_shares == [0.20, 0.45, 0.80]
    assert result.n_districts_observed == 3
    assert result.missing_districts == []


def test_compute_ordered_dem_vote_shares_descending():
    nodes = [
        {"id": 1, "D": 10, "R": 90},
        {"id": 2, "D": 80, "R": 20},
    ]

    node_to_district = {
        1: 1,
        2: 2,
    }

    result = compute_ordered_dem_vote_shares(
        node_to_district,
        nodes,
        dem_col="D",
        rep_col="R",
        n_districts=2,
        sort="descending",
    )

    assert result.ordered_vote_shares == [0.80, 0.10]


def test_missing_district_is_reported():
    nodes = [
        {"id": 1, "D": 10, "R": 90},
        {"id": 2, "D": 80, "R": 20},
    ]

    node_to_district = {
        1: 1,
        2: 3,
    }

    result = compute_ordered_dem_vote_shares(
        node_to_district,
        nodes,
        dem_col="D",
        rep_col="R",
        n_districts=3,
    )

    assert result.missing_districts == [2]
    assert result.n_districts_observed == 2


def test_zero_vote_district_raises_by_default():
    nodes = [
        {"id": 1, "D": 0, "R": 0},
    ]

    node_to_district = {
        1: 1,
    }

    with pytest.raises(VoteShareError):
        compute_ordered_dem_vote_shares(
            node_to_district,
            nodes,
            dem_col="D",
            rep_col="R",
            n_districts=1,
        )


def test_zero_vote_district_allowed():
    nodes = [
        {"id": 1, "D": 0, "R": 0},
    ]

    node_to_district = {
        1: 1,
    }

    result = compute_ordered_dem_vote_shares(
        node_to_district,
        nodes,
        dem_col="D",
        rep_col="R",
        n_districts=1,
        allow_zero_vote_districts=True,
    )

    assert result.ordered_vote_shares == [None]
    assert result.zero_vote_districts == [1]


def test_missing_node_assignment_raises():
    nodes = [
        {"id": 1, "D": 10, "R": 90},
        {"id": 2, "D": 80, "R": 20},
    ]

    node_to_district = {
        1: 1,
    }

    with pytest.raises(VoteShareError):
        compute_ordered_dem_vote_shares(
            node_to_district,
            nodes,
            dem_col="D",
            rep_col="R",
        )


def test_drop_unassigned_none_node():
    nodes = [
        {"id": 1, "D": 10, "R": 90},
        {"id": 2, "D": 80, "R": 20},
    ]

    node_to_district = {
        1: 1,
        2: None,
    }

    result = compute_ordered_dem_vote_shares(
        node_to_district,
        nodes,
        dem_col="D",
        rep_col="R",
        drop_unassigned=True,
    )

    assert result.ordered_vote_shares == [0.10]