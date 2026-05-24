from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Mapping


SortOrder = Literal["ascending", "descending", "none"]


class VoteShareError(Exception):
    """Base exception for vote-share observable errors."""


@dataclass
class DistrictVoteTotals:
    district: int
    dem: float
    rep: float
    total: float
    dem_share: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OrderedVoteShareResult:
    ordered_vote_shares: list[float | None]
    district_totals: list[DistrictVoteTotals]
    n_districts_expected: int | None
    n_districts_observed: int
    missing_districts: list[int]
    zero_vote_districts: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordered_vote_shares": self.ordered_vote_shares,
            "district_totals": [x.to_dict() for x in self.district_totals],
            "n_districts_expected": self.n_districts_expected,
            "n_districts_observed": self.n_districts_observed,
            "missing_districts": self.missing_districts,
            "zero_vote_districts": self.zero_vote_districts,
        }


def _as_float(value: Any, *, field_name: str) -> float:
    if value is None:
        return 0.0

    if isinstance(value, float) and math.isnan(value):
        return 0.0

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise VoteShareError(f"could not convert {field_name}={value!r} to float") from exc


def _lookup_node_value(
    node: Mapping[str, Any],
    field: str,
    *,
    case_insensitive_cols: bool = False,
) -> Any:
    if field in node:
        return node[field]

    if case_insensitive_cols:
        target = field.lower()
        for key, value in node.items():
            if str(key).lower() == target:
                return value

    raise VoteShareError(f"node is missing required field {field!r}")


def _get_node_id(
    node: Mapping[str, Any],
    *,
    index: int,
    node_id_col: str | None = "id",
    case_insensitive_cols: bool = False,
) -> Any:
    if node_id_col is None:
        return index

    try:
        value = _lookup_node_value(
            node,
            node_id_col,
            case_insensitive_cols=case_insensitive_cols,
        )
    except VoteShareError:
        return index

    if value is None:
        return index

    if isinstance(value, float) and math.isnan(value):
        return index

    return value


def aggregate_district_votes(
    node_to_district: Mapping[Any, int | None],
    nodes: Iterable[Mapping[str, Any]],
    *,
    dem_col: str,
    rep_col: str,
    node_id_col: str | None = "id",
    case_insensitive_cols: bool = False,
    drop_unassigned: bool = True,
) -> list[DistrictVoteTotals]:
    """
    Aggregate Democratic and Republican votes by district.

    Parameters
    ----------
    node_to_district:
        Mapping from node id to district label.

    nodes:
        Node metadata/election records. Each node must have `dem_col` and
        `rep_col`.

    dem_col, rep_col:
        Column names for Democratic and Republican votes.

    node_id_col:
        Node id field used to match `node_to_district`.

    drop_unassigned:
        If True, nodes assigned to None are skipped. If False, raises.

    Returns
    -------
    list[DistrictVoteTotals]
        One entry per observed district, sorted by district label.
    """
    dem_by_dist: dict[int, float] = defaultdict(float)
    rep_by_dist: dict[int, float] = defaultdict(float)

    for index, node in enumerate(nodes):
        node_id = _get_node_id(
            node,
            index=index,
            node_id_col=node_id_col,
            case_insensitive_cols=case_insensitive_cols,
        )

        if node_id not in node_to_district:
            raise VoteShareError(f"node id {node_id!r} is missing from node_to_district")

        district = node_to_district[node_id]

        if district is None:
            if drop_unassigned:
                continue
            raise VoteShareError(f"node id {node_id!r} has no assigned district")

        district = int(district)

        dem = _as_float(
            _lookup_node_value(
                node,
                dem_col,
                case_insensitive_cols=case_insensitive_cols,
            ),
            field_name=dem_col,
        )
        rep = _as_float(
            _lookup_node_value(
                node,
                rep_col,
                case_insensitive_cols=case_insensitive_cols,
            ),
            field_name=rep_col,
        )

        dem_by_dist[district] += dem
        rep_by_dist[district] += rep

    totals: list[DistrictVoteTotals] = []

    for district in sorted(set(dem_by_dist) | set(rep_by_dist)):
        dem = dem_by_dist[district]
        rep = rep_by_dist[district]
        total = dem + rep
        dem_share = None if total <= 0 else dem / total

        totals.append(
            DistrictVoteTotals(
                district=district,
                dem=dem,
                rep=rep,
                total=total,
                dem_share=dem_share,
            )
        )

    return totals


def ordered_dem_vote_shares_from_totals(
    district_totals: Iterable[DistrictVoteTotals],
    *,
    n_districts: int | None = None,
    sort: SortOrder = "ascending",
    allow_zero_vote_districts: bool = False,
) -> OrderedVoteShareResult:
    """
    Convert district vote totals into an ordered Democratic vote-share vector.

    By default, vote shares are sorted ascending, so rank 1 is the most
    Republican district and the final rank is the most Democratic district.
    """
    totals = list(district_totals)

    observed_districts = sorted(x.district for x in totals)

    if n_districts is None:
        missing_districts: list[int] = []
    else:
        expected = set(range(1, n_districts + 1))
        missing_districts = sorted(expected - set(observed_districts))

    zero_vote_districts = [
        x.district for x in totals if x.dem_share is None
    ]

    if zero_vote_districts and not allow_zero_vote_districts:
        raise VoteShareError(
            f"districts with zero total Dem+Rep votes: {zero_vote_districts}"
        )

    shares: list[float | None] = [x.dem_share for x in totals]

    numeric_shares = [x for x in shares if x is not None]
    none_count = len(shares) - len(numeric_shares)

    if sort == "ascending":
        ordered: list[float | None] = sorted(numeric_shares) + [None] * none_count
    elif sort == "descending":
        ordered = sorted(numeric_shares, reverse=True) + [None] * none_count
    elif sort == "none":
        ordered = shares
    else:
        raise ValueError(f"unknown sort order: {sort!r}")

    return OrderedVoteShareResult(
        ordered_vote_shares=ordered,
        district_totals=totals,
        n_districts_expected=n_districts,
        n_districts_observed=len(observed_districts),
        missing_districts=missing_districts,
        zero_vote_districts=zero_vote_districts,
    )


def compute_ordered_dem_vote_shares(
    node_to_district: Mapping[Any, int | None],
    nodes: Iterable[Mapping[str, Any]],
    *,
    dem_col: str,
    rep_col: str,
    node_id_col: str | None = "id",
    n_districts: int | None = None,
    sort: SortOrder = "ascending",
    allow_zero_vote_districts: bool = False,
    case_insensitive_cols: bool = False,
    drop_unassigned: bool = True,
) -> OrderedVoteShareResult:
    """
    Compute ordered Democratic vote shares from a node-level district assignment.

    This is the main function for CT-style vote-share convergence diagnostics.

    Typical usage:

        result = compute_ordered_dem_vote_shares(
            node_to_district,
            nodes,
            dem_col=\"G20PREDEM\",
            rep_col=\"G20PREREP\",
            n_districts=15,
        )

        result.ordered_vote_shares
    """
    totals = aggregate_district_votes(
        node_to_district,
        nodes,
        dem_col=dem_col,
        rep_col=rep_col,
        node_id_col=node_id_col,
        case_insensitive_cols=case_insensitive_cols,
        drop_unassigned=drop_unassigned,
    )

    return ordered_dem_vote_shares_from_totals(
        totals,
        n_districts=n_districts,
        sort=sort,
        allow_zero_vote_districts=allow_zero_vote_districts,
    )