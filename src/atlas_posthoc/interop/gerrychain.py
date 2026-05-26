from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class GerryChainInteropError(Exception):
    """Base exception for GerryChain interoperability helpers."""


@dataclass
class GerryChainGraphLoadResult:
    graph: Any
    crosswalk: dict[Any, Any]
    name_col: str
    n_graph_nodes: int
    n_crosswalk_nodes: int
    missing_name_count: int

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["graph"] = repr(self.graph)
        return out


@dataclass
class PartitionConversionSummary:
    n_graph_nodes: int
    n_assignment_nodes: int
    n_missing_graph_nodes: int
    n_extra_assignment_nodes: int
    missing_graph_nodes: list[Any]
    extra_assignment_nodes: list[Any]

    @property
    def is_complete(self) -> bool:
        return self.n_missing_graph_nodes == 0 and self.n_extra_assignment_nodes == 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PartitionConversionResult:
    partition: Any
    node_to_district: dict[Any, int]
    summary: PartitionConversionSummary
    expansion_summary: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": repr(self.partition),
            "node_to_district": self.node_to_district,
            "summary": self.summary.to_dict(),
            "expansion_summary": (
                None
                if self.expansion_summary is None
                else (
                    self.expansion_summary.to_dict()
                    if hasattr(self.expansion_summary, "to_dict")
                    else self.expansion_summary
                )
            ),
        }


def _require_gerrychain() -> tuple[Any, Any]:
    """
    Import GerryChain lazily.

    This keeps atlas-posthoc usable for ordinary TV/R-hat workflows even when
    GerryChain is not installed. PACE / Wasserplan workflows can depend on the
    optional GerryChain stack.
    """
    try:
        from gerrychain import Graph, Partition
    except ImportError as exc:
        raise GerryChainInteropError(
            "GerryChain is required for this function. Install the optional "
            "PACE/GerryChain dependencies before using atlas_posthoc.interop.gerrychain."
        ) from exc

    return Graph, Partition


def load_gerrychain_graph(
    filepath: str | Path,
    *,
    name_col: str = "NAME",
    require_full_crosswalk: bool = True,
) -> GerryChainGraphLoadResult:
    """
    Load a graph JSON as a GerryChain Graph and build an attribute crosswalk.

    This is the package-native version of the old atlas-to-gerrychain
    `load_base_graph` idea: load the base graph once, then cache a node
    attribute crosswalk for fast downstream conversion.

    Parameters
    ----------
    filepath:
        Path to graph JSON, for example `data/graph/OHpct20.json`.

    name_col:
        Node attribute used by Atlas keys. For the Ohio congressional chains we
        diagnosed, this is usually `"NAME"`.

    require_full_crosswalk:
        If True, raise when any graph node is missing `name_col`.

    Returns
    -------
    GerryChainGraphLoadResult
        Contains the GerryChain graph and the attribute-to-node crosswalk.
    """
    Graph, _ = _require_gerrychain()

    graph = Graph.from_json(str(filepath))

    crosswalk, missing_count = build_graph_node_attribute_crosswalk(
        graph,
        attr_col=name_col,
        require_unique=True,
        require_all=False,
        return_missing_count=True,
    )

    if require_full_crosswalk and missing_count > 0:
        raise GerryChainInteropError(
            f"{missing_count} graph nodes are missing attribute {name_col!r}"
        )

    return GerryChainGraphLoadResult(
        graph=graph,
        crosswalk=crosswalk,
        name_col=name_col,
        n_graph_nodes=len(graph.nodes),
        n_crosswalk_nodes=len(crosswalk),
        missing_name_count=missing_count,
    )


def build_graph_node_attribute_crosswalk(
    graph: Any,
    *,
    attr_col: str,
    require_unique: bool = True,
    require_all: bool = False,
    return_missing_count: bool = False,
) -> dict[Any, Any] | tuple[dict[Any, Any], int]:
    """
    Build a crosswalk from a graph node attribute to the graph's node id.

    Example
    -------
    If graph node 123 has `NAME = "COLUMBUS 01_FRANKLIN"`, then the crosswalk
    contains:

        {"COLUMBUS 01_FRANKLIN": 123}

    This generalizes the crosswalk idea from atlas-to-gerrychain.
    """
    crosswalk: dict[Any, Any] = {}
    missing_count = 0

    for graph_node, data in graph.nodes(data=True):
        if attr_col not in data or data[attr_col] is None:
            missing_count += 1
            continue

        attr_value = data[attr_col]

        if require_unique and attr_value in crosswalk:
            raise GerryChainInteropError(
                f"duplicate graph node attribute {attr_col}={attr_value!r}"
            )

        crosswalk[attr_value] = graph_node

    if require_all and missing_count > 0:
        raise GerryChainInteropError(
            f"{missing_count} graph nodes are missing attribute {attr_col!r}"
        )

    if return_missing_count:
        return crosswalk, missing_count

    return crosswalk


def _coerce_district(value: Any) -> int:
    if isinstance(value, bool):
        raise GerryChainInteropError(f"boolean district label is invalid: {value!r}")

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise GerryChainInteropError(f"non-integer district label: {value!r}")

    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError as exc:
            raise GerryChainInteropError(
                f"non-integer string district label: {value!r}"
            ) from exc

    raise GerryChainInteropError(f"cannot coerce district label to int: {value!r}")


def _maybe_translate_node_ids(
    graph: Any,
    node_to_district: Mapping[Any, Any],
    *,
    node_id_col: str | None = "id",
    node_id_to_graph_node: Mapping[Any, Any] | None = None,
) -> dict[Any, int]:
    """
    Translate atlas-posthoc node ids to GerryChain graph node ids when needed.

    Most Ohio workflows use the same integer ids in the graph JSON and in the
    atlas-posthoc node metadata. In that common case, this function is just the
    identity. If the keys do not match, it tries a graph attribute crosswalk
    using `node_id_col`, usually `"id"`.
    """
    graph_nodes = set(graph.nodes)
    assignment_keys = set(node_to_district.keys())

    if node_id_to_graph_node is None and assignment_keys <= graph_nodes:
        return {
            node_id: _coerce_district(dist)
            for node_id, dist in node_to_district.items()
            if dist is not None
        }

    if node_id_to_graph_node is None:
        if node_id_col is None:
            raise GerryChainInteropError(
                "node ids do not match graph nodes, and no node_id_col/crosswalk "
                "was provided"
            )

        node_id_to_graph_node = build_graph_node_attribute_crosswalk(
            graph,
            attr_col=node_id_col,
            require_unique=True,
            require_all=False,
        )

    translated: dict[Any, int] = {}
    missing: list[Any] = []

    for node_id, dist in node_to_district.items():
        if dist is None:
            continue

        if node_id in graph_nodes:
            graph_node = node_id
        elif node_id in node_id_to_graph_node:
            graph_node = node_id_to_graph_node[node_id]
        else:
            missing.append(node_id)
            continue

        translated[graph_node] = _coerce_district(dist)

    if missing:
        raise GerryChainInteropError(
            f"{len(missing)} assignment node ids could not be matched to "
            f"GerryChain graph nodes. Examples: {missing[:5]}"
        )

    return translated


def summarize_partition_assignment(
    graph: Any,
    assignment: Mapping[Any, Any],
    *,
    max_examples: int = 5,
) -> PartitionConversionSummary:
    graph_nodes = set(graph.nodes)
    assignment_nodes = set(assignment.keys())

    missing = sorted(graph_nodes - assignment_nodes, key=str)
    extra = sorted(assignment_nodes - graph_nodes, key=str)

    return PartitionConversionSummary(
        n_graph_nodes=len(graph_nodes),
        n_assignment_nodes=len(assignment_nodes),
        n_missing_graph_nodes=len(missing),
        n_extra_assignment_nodes=len(extra),
        missing_graph_nodes=missing[:max_examples],
        extra_assignment_nodes=extra[:max_examples],
    )


def partition_from_node_to_district(
    graph: Any,
    node_to_district: Mapping[Any, Any],
    *,
    node_id_col: str | None = "id",
    node_id_to_graph_node: Mapping[Any, Any] | None = None,
    require_complete_assignment: bool = True,
) -> PartitionConversionResult:
    """
    Convert a node-to-district assignment into a GerryChain Partition.

    This is the most important interop function. It avoids making PACE depend
    on raw AtlasIO key formats. Upstream code should first use atlas-posthoc's
    multiscale expansion tools to obtain `node_to_district`.

    Parameters
    ----------
    graph:
        GerryChain Graph.

    node_to_district:
        Mapping from node id to district id.

    node_id_col:
        Graph node attribute used to translate node ids if assignment keys do
        not directly match GerryChain graph node ids.

    node_id_to_graph_node:
        Optional explicit crosswalk from assignment node ids to graph node ids.

    require_complete_assignment:
        If True, raise unless every graph node receives exactly one district.

    Returns
    -------
    PartitionConversionResult
        Contains the GerryChain Partition, translated assignment, and summary.
    """
    _, Partition = _require_gerrychain()

    translated = _maybe_translate_node_ids(
        graph,
        node_to_district,
        node_id_col=node_id_col,
        node_id_to_graph_node=node_id_to_graph_node,
    )

    summary = summarize_partition_assignment(graph, translated)

    if require_complete_assignment and not summary.is_complete:
        raise GerryChainInteropError(
            "assignment is not a complete GerryChain partition: "
            f"missing_graph_nodes={summary.n_missing_graph_nodes}, "
            f"extra_assignment_nodes={summary.n_extra_assignment_nodes}, "
            f"missing_examples={summary.missing_graph_nodes}, "
            f"extra_examples={summary.extra_assignment_nodes}"
        )

    partition = Partition(graph, translated)

    return PartitionConversionResult(
        partition=partition,
        node_to_district=translated,
        summary=summary,
    )


def partition_from_record(
    record: Any,
    graph: Any,
    graph_nodes: Sequence[Mapping[str, Any]],
    key_cols: Sequence[str],
    *,
    node_id_col: str | None = "id",
    node_id_to_graph_node: Mapping[Any, Any] | None = None,
    missing_policy: str = "error",
    ambiguous_policy: str = "error",
    require_complete_assignment: bool = True,
) -> PartitionConversionResult:
    """
    Convert an AtlasRecord-like object into a GerryChain Partition.

    This wrapper uses atlas-posthoc's multiscale assignment expansion first:

        AtlasRecord.assignment
            -> expand_multiscale_assignment(...)
            -> node_to_district
            -> GerryChain Partition

    Therefore it supports one-level Ohio `["NAME"]` keys and more general
    multiscale keys such as `["COUNTY"]` or `["COUNTY", "prec_id"]`.
    """
    from atlas_posthoc.assignments.multiscale import expand_multiscale_assignment

    if not hasattr(record, "assignment"):
        raise GerryChainInteropError("record must have an `.assignment` attribute")

    expanded = expand_multiscale_assignment(
        record.assignment,
        graph_nodes,
        key_cols,
        node_id_col=node_id_col,
        missing_policy=missing_policy,  # type: ignore[arg-type]
        ambiguous_policy=ambiguous_policy,  # type: ignore[arg-type]
    )

    result = partition_from_node_to_district(
        graph,
        expanded.node_to_district,
        node_id_col=node_id_col,
        node_id_to_graph_node=node_id_to_graph_node,
        require_complete_assignment=require_complete_assignment,
    )

    result.expansion_summary = expanded.summary
    return result


# Friendly aliases matching the old atlas-to-gerrychain language.
load_base_graph = load_gerrychain_graph
atlas_record_to_partition = partition_from_record