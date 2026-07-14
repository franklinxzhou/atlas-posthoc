from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

import numpy as np


class MunicipalSplitError(Exception):
    """Base exception for municipal-split observables."""


GeographyMode = Literal["COUSUB", "PLACE"]


@dataclass
class MunicipalSplitPlanSummary:
    n_cut_municipalities: int
    ousted_population: float
    ousted_population_share: float
    total_population: float
    n_total_municipalities: int
    max_ousted_single_municipality: float
    mean_ousted_population_among_cut: float
    municipality_cols: list[str]
    pop_col: str
    geography_mode: GeographyMode = "COUSUB"
    population_basis: str = "POP20"
    represented_population: float | None = None
    graph_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MunicipalSplitTrace:
    chain_id: str
    proposals: list[int]
    n_cut_municipalities: list[int]
    ousted_population: list[float]
    ousted_population_share: list[float]
    n_maps: int
    municipality_cols: list[str]
    pop_col: str
    geography_mode: GeographyMode = "COUSUB"
    graph_path: str | None = None
    population_basis: str = "POP20"
    represented_population: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_graph_nodes(graph_nodes: Any) -> list[Mapping[str, Any]]:
    if hasattr(graph_nodes, "nodes"):
        nodes = getattr(graph_nodes, "nodes")
    else:
        nodes = graph_nodes

    if not isinstance(nodes, Sequence):
        raise MunicipalSplitError("graph_nodes must be a sequence of node dictionaries")

    out = list(nodes)

    if not out:
        raise MunicipalSplitError("graph_nodes is empty")

    for idx, node in enumerate(out[:5]):
        if not isinstance(node, Mapping):
            raise MunicipalSplitError(
                f"graph_nodes must contain dictionaries; item {idx} has type {type(node)}"
            )

    return out


def _assignment_lookup(
    assignment: Mapping[Any, Any] | Sequence[Any],
    graph_nodes: Sequence[Mapping[str, Any]],
    *,
    node_id_col: str,
) -> dict[Any, Any]:
    if isinstance(assignment, Mapping):
        return dict(assignment)

    values = list(assignment)

    if len(values) != len(graph_nodes):
        raise MunicipalSplitError(
            "assignment sequence length must match graph_nodes length; "
            f"got {len(values)} and {len(graph_nodes)}"
        )

    out: dict[Any, Any] = {}

    for node, district in zip(graph_nodes, values):
        if node_id_col not in node:
            raise MunicipalSplitError(f"graph node is missing node_id_col={node_id_col!r}")

        out[node[node_id_col]] = district

    return out


def _municipality_key(
    node: Mapping[str, Any],
    *,
    municipality_cols: Sequence[str],
    skip_missing_municipality: bool,
) -> tuple[Any, ...] | None:
    values: list[Any] = []

    for col in municipality_cols:
        value = node.get(col)

        if value is None or value == "":
            if skip_missing_municipality:
                return None

            raise MunicipalSplitError(f"node is missing municipality column {col!r}")

        if isinstance(value, Mapping):
            raise MunicipalSplitError(
                f"municipality column {col!r} is a mapping; "
                "use geography_mode='PLACE' for PLACE population mappings"
            )

        values.append(value)

    return tuple(values)


def _node_population(
    node: Mapping[str, Any],
    *,
    pop_col: str,
) -> float:
    if pop_col not in node:
        raise MunicipalSplitError(f"graph node is missing pop_col={pop_col!r}")

    try:
        pop = float(node[pop_col])
    except (TypeError, ValueError) as exc:
        raise MunicipalSplitError(
            f"node has non-numeric {pop_col}={node[pop_col]!r}"
        ) from exc

    if pop < 0:
        raise MunicipalSplitError(f"node has negative population {pop}")

    return pop


def _normalize_geography_mode(mode: str) -> GeographyMode:
    normalized = str(mode).upper()
    if normalized not in {"COUSUB", "PLACE"}:
        raise MunicipalSplitError(
            f"geography_mode must be 'COUSUB' or 'PLACE'; got {mode!r}"
        )
    return normalized  # type: ignore[return-value]


def _municipality_population_pieces(
    node: Mapping[str, Any],
    *,
    geography_mode: GeographyMode,
    municipality_cols: Sequence[str],
    place_col: str,
    pop_col: str,
    skip_missing_municipality: bool,
) -> Iterator[tuple[tuple[Any, ...], float]]:
    """Yield the municipality/population pieces represented by one graph node."""
    if geography_mode == "COUSUB":
        key = _municipality_key(
            node,
            municipality_cols=municipality_cols,
            skip_missing_municipality=skip_missing_municipality,
        )
        if key is not None:
            # Preserve the legacy scalar algorithm: one key receives full POP20.
            yield key, _node_population(node, pop_col=pop_col)
        return

    if place_col not in node or node[place_col] is None:
        if skip_missing_municipality:
            return
        raise MunicipalSplitError(f"node is missing PLACE mapping column {place_col!r}")

    place_mapping = node[place_col]
    if not isinstance(place_mapping, Mapping):
        raise MunicipalSplitError(
            f"node has non-mapping {place_col}={place_mapping!r}"
        )

    # PLACE keys are names in the current graph schema. Duplicate names are
    # intentionally conflated until the graph carries stable Place GEOIDs.
    for place, population in place_mapping.items():
        if place is None or place == "":
            if skip_missing_municipality:
                continue
            raise MunicipalSplitError(f"node has a missing Place key in {place_col!r}")

        try:
            place_population = float(population)
        except (TypeError, ValueError) as exc:
            raise MunicipalSplitError(
                f"node has non-numeric {place_col}[{place!r}]={population!r}"
            ) from exc

        if place_population < 0:
            raise MunicipalSplitError(
                f"node has negative {place_col}[{place!r}] population "
                f"{place_population}"
            )

        yield (place,), place_population


def compute_municipal_split_statistics_for_assignment(
    assignment: Mapping[Any, Any] | Sequence[Any],
    graph_nodes: Any,
    *,
    municipality_cols: Sequence[str] = ("COUNTY", "MCD"),
    geography_mode: GeographyMode = "COUSUB",
    place_col: str = "PLACE",
    graph_path: str | Path | None = None,
    node_id_col: str = "id",
    pop_col: str = "POP20",
    skip_missing_municipality: bool = True,
    require_complete_assignment: bool = True,
) -> MunicipalSplitPlanSummary:
    """
    Compute municipal split statistics for one districting assignment.

    In legacy ``COUSUB`` mode, one scalar municipality key receives the node's
    full ``pop_col`` population. In ``PLACE`` mode, every entry in the node's
    ``PLACE`` mapping is accumulated independently using its mapped population.
    Empty mappings are excluded from the represented-population denominator.

    A municipality is cut if its positive-population pieces are assigned to
    more than one district.

    Ousted population is computed municipality-by-municipality as:

        pop(municipality) - max_d pop(municipality ∩ district d)

    and then summed over municipalities.
    """
    mode = _normalize_geography_mode(geography_mode)
    nodes = _as_graph_nodes(graph_nodes)
    assignment_by_node = _assignment_lookup(
        assignment,
        nodes,
        node_id_col=node_id_col,
    )

    # municipality -> district -> population
    municipal_district_pop: dict[tuple[Any, ...], dict[Any, float]] = {}
    municipal_total_pop: dict[tuple[Any, ...], float] = {}

    missing_assignment_nodes: list[Any] = []

    for node in nodes:
        if node_id_col not in node:
            raise MunicipalSplitError(f"graph node is missing node_id_col={node_id_col!r}")

        node_id = node[node_id_col]

        if node_id not in assignment_by_node:
            missing_assignment_nodes.append(node_id)
            continue

        district = assignment_by_node[node_id]

        if district is None:
            missing_assignment_nodes.append(node_id)
            continue

        for key, pop in _municipality_population_pieces(
            node,
            geography_mode=mode,
            municipality_cols=municipality_cols,
            place_col=place_col,
            pop_col=pop_col,
            skip_missing_municipality=skip_missing_municipality,
        ):
            if key not in municipal_district_pop:
                municipal_district_pop[key] = {}
                municipal_total_pop[key] = 0.0

            municipal_total_pop[key] += pop
            municipal_district_pop[key][district] = (
                municipal_district_pop[key].get(district, 0.0) + pop
            )

    if require_complete_assignment and missing_assignment_nodes:
        raise MunicipalSplitError(
            f"{len(missing_assignment_nodes)} graph nodes are missing assignments. "
            f"Examples: {missing_assignment_nodes[:5]}"
        )

    if not municipal_district_pop:
        raise MunicipalSplitError("no municipalities were available for split statistics")

    n_cut = 0
    ousted_total = 0.0
    ousted_values_for_cut_municipalities: list[float] = []

    for key, district_pop in municipal_district_pop.items():
        positive_pieces = {
            district: pop
            for district, pop in district_pop.items()
            if pop > 0
        }

        total_pop = municipal_total_pop[key]

        if not positive_pieces:
            continue

        largest_piece = max(positive_pieces.values())
        ousted = max(0.0, total_pop - largest_piece)

        if len(positive_pieces) > 1:
            n_cut += 1
            ousted_values_for_cut_municipalities.append(ousted)

        ousted_total += ousted

    total_population = float(sum(municipal_total_pop.values()))
    ousted_share = ousted_total / total_population if total_population > 0 else float("nan")

    if ousted_values_for_cut_municipalities:
        max_ousted = float(max(ousted_values_for_cut_municipalities))
        mean_ousted = float(np.mean(ousted_values_for_cut_municipalities))
    else:
        max_ousted = 0.0
        mean_ousted = 0.0

    return MunicipalSplitPlanSummary(
        n_cut_municipalities=int(n_cut),
        ousted_population=float(ousted_total),
        ousted_population_share=float(ousted_share),
        total_population=total_population,
        n_total_municipalities=int(len(municipal_district_pop)),
        max_ousted_single_municipality=max_ousted,
        mean_ousted_population_among_cut=mean_ousted,
        municipality_cols=(
            list(municipality_cols) if mode == "COUSUB" else [place_col]
        ),
        pop_col=pop_col,
        geography_mode=mode,
        population_basis=(pop_col if mode == "COUSUB" else f"{place_col} mapping"),
        represented_population=total_population,
        graph_path=None if graph_path is None else str(graph_path),
    )


def compute_municipal_split_statistics_for_assignments(
    assignments: Sequence[Mapping[Any, Any] | Sequence[Any]],
    graph_nodes: Any,
    *,
    municipality_cols: Sequence[str] = ("COUNTY", "MCD"),
    geography_mode: GeographyMode = "COUSUB",
    place_col: str = "PLACE",
    graph_path: str | Path | None = None,
    node_id_col: str = "id",
    pop_col: str = "POP20",
    skip_missing_municipality: bool = True,
    require_complete_assignment: bool = True,
) -> list[MunicipalSplitPlanSummary]:
    """
    Compute municipal split statistics for a sequence of districting assignments.
    """
    return [
        compute_municipal_split_statistics_for_assignment(
            assignment,
            graph_nodes,
            municipality_cols=municipality_cols,
            geography_mode=geography_mode,
            place_col=place_col,
            graph_path=graph_path,
            node_id_col=node_id_col,
            pop_col=pop_col,
            skip_missing_municipality=skip_missing_municipality,
            require_complete_assignment=require_complete_assignment,
        )
        for assignment in assignments
    ]


def _normalize_assignments_by_chain(
    assignments_by_chain: Mapping[str, Sequence[Any]] | Sequence[Sequence[Any]],
    *,
    chain_ids: Sequence[str] | None = None,
) -> tuple[list[str], list[Sequence[Any]]]:
    if isinstance(assignments_by_chain, Mapping):
        ids = [str(k) for k in assignments_by_chain.keys()]
        assignments = [list(v) for v in assignments_by_chain.values()]
    else:
        assignments = [list(v) for v in assignments_by_chain]

        if chain_ids is None:
            ids = [f"chain_{i}" for i in range(len(assignments))]
        else:
            ids = [str(x) for x in chain_ids]

    if not assignments:
        raise MunicipalSplitError("no assignment chains were provided")

    if len(ids) != len(assignments):
        raise MunicipalSplitError("chain_ids length must match number of assignment chains")

    for chain_id, chain_assignments in zip(ids, assignments):
        if not chain_assignments:
            raise MunicipalSplitError(f"chain {chain_id!r} has no assignments")

    return ids, assignments


def make_municipal_split_traces(
    assignments_by_chain: Mapping[str, Sequence[Any]] | Sequence[Sequence[Any]],
    graph_nodes: Any,
    *,
    chain_ids: Sequence[str] | None = None,
    municipality_cols: Sequence[str] = ("COUNTY", "MCD"),
    geography_mode: GeographyMode = "COUSUB",
    place_col: str = "PLACE",
    graph_path: str | Path | None = None,
    node_id_col: str = "id",
    pop_col: str = "POP20",
    output_frequency: int = 1,
    start_proposal: int = 0,
    skip_missing_municipality: bool = True,
    require_complete_assignment: bool = True,
) -> list[MunicipalSplitTrace]:
    """
    Build municipal split traces for multiple chains.

    Input assignments should be node-to-district mappings, or district vectors in
    graph-node order.
    """
    mode = _normalize_geography_mode(geography_mode)

    if output_frequency <= 0:
        raise ValueError("output_frequency must be positive")

    ids, assignment_chains = _normalize_assignments_by_chain(
        assignments_by_chain,
        chain_ids=chain_ids,
    )

    traces: list[MunicipalSplitTrace] = []

    for chain_id, assignments in zip(ids, assignment_chains):
        summaries = compute_municipal_split_statistics_for_assignments(
            assignments,
            graph_nodes,
            municipality_cols=municipality_cols,
            geography_mode=mode,
            place_col=place_col,
            graph_path=graph_path,
            node_id_col=node_id_col,
            pop_col=pop_col,
            skip_missing_municipality=skip_missing_municipality,
            require_complete_assignment=require_complete_assignment,
        )

        proposals = [
            int(start_proposal + i * output_frequency)
            for i in range(len(summaries))
        ]

        represented_populations = {s.represented_population for s in summaries}
        if len(represented_populations) != 1:
            raise MunicipalSplitError(
                f"chain {chain_id!r} has inconsistent represented-population "
                "denominators across assignments"
            )

        traces.append(
            MunicipalSplitTrace(
                chain_id=chain_id,
                proposals=proposals,
                n_cut_municipalities=[
                    s.n_cut_municipalities
                    for s in summaries
                ],
                ousted_population=[
                    s.ousted_population
                    for s in summaries
                ],
                ousted_population_share=[
                    s.ousted_population_share
                    for s in summaries
                ],
                n_maps=len(summaries),
                municipality_cols=(
                    list(municipality_cols) if mode == "COUSUB" else [place_col]
                ),
                pop_col=pop_col,
                geography_mode=mode,
                graph_path=None if graph_path is None else str(graph_path),
                population_basis=(
                    pop_col if mode == "COUSUB" else f"{place_col} mapping"
                ),
                represented_population=summaries[0].represented_population,
            )
        )

    return traces
