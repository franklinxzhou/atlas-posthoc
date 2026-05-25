from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from atlas_posthoc.io.graph_reader import GraphData, adjacency_edges, read_graph_json
from atlas_posthoc.observables.vote_share import (
    OrderedVoteShareResult,
    compute_ordered_dem_vote_shares,
)


SortOrder = Literal["ascending", "descending", "none"]


class EnactedAuditError(Exception):
    """Base exception for enacted/reference-plan observable calculations."""


@dataclass
class DistrictCompactnessTotals:
    district: int
    area: float
    perimeter: float
    score: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReferenceCompactnessResult:
    ordered_scores: list[float | None]
    district_totals: list[DistrictCompactnessTotals]
    n_districts_expected: int | None
    n_districts_observed: int
    missing_districts: list[int]
    zero_area_districts: list[int]
    edge_length_col: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordered_scores": self.ordered_scores,
            "district_totals": [x.to_dict() for x in self.district_totals],
            "n_districts_expected": self.n_districts_expected,
            "n_districts_observed": self.n_districts_observed,
            "missing_districts": self.missing_districts,
            "zero_area_districts": self.zero_area_districts,
            "edge_length_col": self.edge_length_col,
        }


@dataclass
class ReferenceAuditResult:
    vote_share: OrderedVoteShareResult | None = None
    compactness: ReferenceCompactnessResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "vote_share": None if self.vote_share is None else self.vote_share.to_dict(),
            "compactness": None if self.compactness is None else self.compactness.to_dict(),
        }


def _as_graph_data(graph: str | Path | GraphData) -> GraphData:
    if isinstance(graph, GraphData):
        return graph
    return read_graph_json(graph)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    return False


def _coerce_district(value: Any) -> int | None:
    if _is_missing(value):
        return None

    if isinstance(value, bool):
        raise EnactedAuditError("boolean values are not valid district labels")

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise EnactedAuditError(f"non-integer float district label: {value!r}")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise EnactedAuditError(
                f"non-integer string district label: {value!r}"
            ) from exc

    raise EnactedAuditError(f"cannot convert district label to int: {value!r}")


def _node_id(node: Mapping[str, Any], *, index: int, node_id_col: str | None) -> Any:
    if node_id_col is None:
        return index

    value = node.get(node_id_col)

    if _is_missing(value):
        return index

    return value


def node_to_district_from_column(
    nodes: Sequence[Mapping[str, Any]],
    *,
    district_col: str,
    node_id_col: str | None = "id",
    drop_missing: bool = False,
) -> dict[Any, int | None]:
    """
    Build a node-to-district mapping from a district column in node metadata.

    This is the usual entry point for enacted/reference plans when the graph
    already contains a district assignment column, for example `CONG_DIST`,
    `ENACTED_UPPER`, or `ENACTED_LOWER`.
    """
    out: dict[Any, int | None] = {}

    for index, node in enumerate(nodes):
        if district_col not in node:
            raise EnactedAuditError(
                f"node at index {index} is missing district column {district_col!r}"
            )

        node_id = _node_id(node, index=index, node_id_col=node_id_col)
        district = _coerce_district(node[district_col])

        if district is None and drop_missing:
            continue

        out[node_id] = district

    return out


def compute_reference_vote_share(
    graph_or_nodes: str | Path | GraphData | Sequence[Mapping[str, Any]],
    *,
    district_col: str | None = None,
    node_to_district: Mapping[Any, int | None] | None = None,
    dem_col: str,
    rep_col: str,
    node_id_col: str | None = "id",
    n_districts: int | None = None,
    sort: SortOrder = "ascending",
    allow_zero_vote_districts: bool = False,
) -> OrderedVoteShareResult:
    """
    Compute the ordered Democratic vote-share vector for a reference/enacted plan.

    The resulting vector has the same interpretation as ensemble vote-share
    vectors:

        rank 1 = most Republican district
        final rank = least Republican / most Democratic district

    You may provide either:
      - `district_col`, if the node metadata contains the reference assignment;
      - `node_to_district`, if the assignment was built externally.
    """
    if isinstance(graph_or_nodes, (str, Path, GraphData)):
        graph = _as_graph_data(graph_or_nodes)
        nodes = graph.nodes
    else:
        nodes = [dict(node) for node in graph_or_nodes]

    if node_to_district is None:
        if district_col is None:
            raise EnactedAuditError(
                "provide either `district_col` or `node_to_district`"
            )

        node_to_district = node_to_district_from_column(
            nodes,
            district_col=district_col,
            node_id_col=node_id_col,
        )

    return compute_ordered_dem_vote_shares(
        node_to_district,
        nodes,
        dem_col=dem_col,
        rep_col=rep_col,
        node_id_col=node_id_col,
        n_districts=n_districts,
        sort=sort,
        allow_zero_vote_districts=allow_zero_vote_districts,
    )


def _lookup_numeric(
    node: Mapping[str, Any],
    column: str,
    *,
    node_id: Any,
) -> float:
    if column not in node:
        raise EnactedAuditError(f"node {node_id!r} is missing column {column!r}")

    value = node[column]

    if _is_missing(value):
        return 0.0

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise EnactedAuditError(
            f"node {node_id!r} has non-numeric {column}={value!r}"
        ) from exc


def _infer_edge_length_col(
    edge_attrs: Sequence[Mapping[str, Any]],
    *,
    candidates: Sequence[str] = ("length", "shared_perim", "shared_perimeter", "border_length"),
) -> str:
    for candidate in candidates:
        if any(candidate in attrs for attrs in edge_attrs):
            return candidate

    available = sorted({str(k) for attrs in edge_attrs for k in attrs.keys()})
    raise EnactedAuditError(
        "could not infer edge length column from graph adjacency attributes. "
        f"Tried {list(candidates)!r}. Available edge attributes: {available!r}"
    )


def compute_district_geometry_totals(
    graph: str | Path | GraphData,
    *,
    district_col: str | None = None,
    node_to_district: Mapping[Any, int | None] | None = None,
    node_id_col: str | None = "id",
    area_col: str = "area",
    border_length_col: str = "border_length",
    edge_length_col: str | None = None,
) -> tuple[list[DistrictCompactnessTotals], str]:
    """
    Compute district area, perimeter, and isoperimetric score from graph geometry.

    Perimeter convention:
      - add each node's external `border_length` to its district perimeter;
      - for each graph edge crossing two districts, add the shared boundary length
        to both districts' perimeters.

    Score convention:
        perimeter^2 / area

    This matches the usual isoperimetric-ratio scale where lower values indicate
    more compact districts.
    """
    graph_data = _as_graph_data(graph)
    nodes = graph_data.nodes

    if node_to_district is None:
        if district_col is None:
            raise EnactedAuditError(
                "provide either `district_col` or `node_to_district`"
            )

        node_to_district = node_to_district_from_column(
            nodes,
            district_col=district_col,
            node_id_col=node_id_col,
        )

    node_lookup = graph_data.node_by_id

    area_by_dist: dict[int, float] = defaultdict(float)
    perimeter_by_dist: dict[int, float] = defaultdict(float)

    for index, node in enumerate(nodes):
        nid = _node_id(node, index=index, node_id_col=node_id_col)

        if nid not in node_to_district:
            raise EnactedAuditError(f"node {nid!r} is missing from node_to_district")

        district = node_to_district[nid]

        if district is None:
            continue

        district = int(district)

        area_by_dist[district] += _lookup_numeric(node, area_col, node_id=nid)
        perimeter_by_dist[district] += _lookup_numeric(
            node,
            border_length_col,
            node_id=nid,
        )

    edges = adjacency_edges(graph_data)

    if not edges:
        raise EnactedAuditError("graph has no adjacency edges")

    if edge_length_col is None:
        edge_length_col = _infer_edge_length_col([attrs for _, _, attrs in edges])

    for source, target, attrs in edges:
        if source not in node_to_district or target not in node_to_district:
            raise EnactedAuditError(
                f"edge ({source!r}, {target!r}) references a node missing "
                "from node_to_district"
            )

        source_dist = node_to_district[source]
        target_dist = node_to_district[target]

        if source_dist is None or target_dist is None:
            continue

        source_dist = int(source_dist)
        target_dist = int(target_dist)

        if source_dist == target_dist:
            continue

        if edge_length_col not in attrs:
            raise EnactedAuditError(
                f"edge ({source!r}, {target!r}) is missing edge length column "
                f"{edge_length_col!r}"
            )

        try:
            length = float(attrs[edge_length_col])
        except (TypeError, ValueError) as exc:
            raise EnactedAuditError(
                f"edge ({source!r}, {target!r}) has non-numeric "
                f"{edge_length_col}={attrs[edge_length_col]!r}"
            ) from exc

        perimeter_by_dist[source_dist] += length
        perimeter_by_dist[target_dist] += length

    district_ids = sorted(set(area_by_dist) | set(perimeter_by_dist))

    totals: list[DistrictCompactnessTotals] = []

    for district in district_ids:
        area = area_by_dist[district]
        perimeter = perimeter_by_dist[district]
        score = None if area <= 0 else (perimeter**2) / area

        totals.append(
            DistrictCompactnessTotals(
                district=district,
                area=area,
                perimeter=perimeter,
                score=score,
            )
        )

    return totals, edge_length_col


def ordered_compactness_from_totals(
    district_totals: Sequence[DistrictCompactnessTotals],
    *,
    n_districts: int | None = None,
    sort: SortOrder = "ascending",
    allow_zero_area_districts: bool = False,
    edge_length_col: str = "length",
) -> ReferenceCompactnessResult:
    """
    Order reference/enacted compactness scores.

    By default scores are sorted ascending, so rank 1 is the most compact /
    lowest score district.
    """
    observed_districts = sorted(x.district for x in district_totals)

    if n_districts is None:
        missing_districts: list[int] = []
    else:
        expected = set(range(1, n_districts + 1))
        missing_districts = sorted(expected - set(observed_districts))

    zero_area_districts = [x.district for x in district_totals if x.score is None]

    if zero_area_districts and not allow_zero_area_districts:
        raise EnactedAuditError(
            f"districts with zero/nonpositive area: {zero_area_districts}"
        )

    raw_scores = [x.score for x in district_totals]
    numeric_scores = [x for x in raw_scores if x is not None]
    none_count = len(raw_scores) - len(numeric_scores)

    if sort == "ascending":
        ordered: list[float | None] = sorted(numeric_scores) + [None] * none_count
    elif sort == "descending":
        ordered = sorted(numeric_scores, reverse=True) + [None] * none_count
    elif sort == "none":
        ordered = raw_scores
    else:
        raise ValueError(f"unknown sort order: {sort!r}")

    return ReferenceCompactnessResult(
        ordered_scores=ordered,
        district_totals=list(district_totals),
        n_districts_expected=n_districts,
        n_districts_observed=len(observed_districts),
        missing_districts=missing_districts,
        zero_area_districts=zero_area_districts,
        edge_length_col=edge_length_col,
    )


def compute_reference_compactness(
    graph: str | Path | GraphData,
    *,
    district_col: str | None = None,
    node_to_district: Mapping[Any, int | None] | None = None,
    node_id_col: str | None = "id",
    area_col: str = "area",
    border_length_col: str = "border_length",
    edge_length_col: str | None = None,
    n_districts: int | None = None,
    sort: SortOrder = "ascending",
    allow_zero_area_districts: bool = False,
) -> ReferenceCompactnessResult:
    """
    Compute ordered reference/enacted compactness scores from graph geometry.
    """
    totals, resolved_edge_length_col = compute_district_geometry_totals(
        graph,
        district_col=district_col,
        node_to_district=node_to_district,
        node_id_col=node_id_col,
        area_col=area_col,
        border_length_col=border_length_col,
        edge_length_col=edge_length_col,
    )

    return ordered_compactness_from_totals(
        totals,
        n_districts=n_districts,
        sort=sort,
        allow_zero_area_districts=allow_zero_area_districts,
        edge_length_col=resolved_edge_length_col,
    )


def compute_reference_audit(
    graph: str | Path | GraphData,
    *,
    district_col: str | None = None,
    node_to_district: Mapping[Any, int | None] | None = None,
    dem_col: str,
    rep_col: str,
    node_id_col: str | None = "id",
    n_districts: int | None = None,
    compute_compactness: bool = True,
    area_col: str = "area",
    border_length_col: str = "border_length",
    edge_length_col: str | None = None,
) -> ReferenceAuditResult:
    """
    Compute reference/enacted vote-share and optional compactness vectors.

    This is the object you use for red-dot overlays in marginal plots.
    """
    graph_data = _as_graph_data(graph)

    if node_to_district is None:
        if district_col is None:
            raise EnactedAuditError(
                "provide either `district_col` or `node_to_district`"
            )

        node_to_district = node_to_district_from_column(
            graph_data.nodes,
            district_col=district_col,
            node_id_col=node_id_col,
        )

    vote_share = compute_reference_vote_share(
        graph_data,
        node_to_district=node_to_district,
        dem_col=dem_col,
        rep_col=rep_col,
        node_id_col=node_id_col,
        n_districts=n_districts,
        sort="ascending",
    )

    compactness = None

    if compute_compactness:
        compactness = compute_reference_compactness(
            graph_data,
            node_to_district=node_to_district,
            node_id_col=node_id_col,
            area_col=area_col,
            border_length_col=border_length_col,
            edge_length_col=edge_length_col,
            n_districts=n_districts,
            sort="ascending",
        )

    return ReferenceAuditResult(
        vote_share=vote_share,
        compactness=compactness,
    )


# Friendly aliases for the common language in the project.
compute_enacted_vote_share = compute_reference_vote_share
compute_enacted_compactness = compute_reference_compactness
compute_enacted_audit = compute_reference_audit