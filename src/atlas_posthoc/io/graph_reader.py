from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


class GraphReaderError(Exception):
    """Base exception for graph JSON reading errors."""


@dataclass
class GraphSummary:
    path: Path
    n_nodes: int
    n_adjacency_rows: int | None
    directed: bool | None
    multigraph: bool | None
    node_id_col: str
    columns: list[str]
    total_population: float | None = None
    county_counts: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["path"] = str(self.path)
        return out


@dataclass
class GraphData:
    path: Path
    raw: dict[str, Any]
    nodes: list[dict[str, Any]]
    adjacency: list[list[dict[str, Any]]] | None = None
    node_id_col: str = "id"
    node_by_id: dict[Any, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_by_id:
            self.node_by_id = build_node_lookup(
                self.nodes,
                node_id_col=self.node_id_col,
            )

    @property
    def columns(self) -> list[str]:
        cols: set[str] = set()
        for node in self.nodes:
            cols.update(str(k) for k in node.keys())
        return sorted(cols)

    @property
    def node_ids(self) -> list[Any]:
        return list(self.node_by_id.keys())

    def require_columns(
        self,
        columns: Iterable[str],
        *,
        case_insensitive: bool = False,
    ) -> None:
        require_node_columns(
            self.nodes,
            columns,
            case_insensitive=case_insensitive,
        )

    def total(self, col: str, *, case_insensitive: bool = False) -> float:
        return sum_numeric_node_column(
            self.nodes,
            col,
            case_insensitive=case_insensitive,
        )

    def total_population(
        self,
        pop_col: str = "POP20",
        *,
        case_insensitive: bool = False,
    ) -> float:
        return self.total(pop_col, case_insensitive=case_insensitive)

    def election_totals(
        self,
        dem_col: str,
        rep_col: str,
        *,
        case_insensitive: bool = False,
    ) -> dict[str, float]:
        dem = self.total(dem_col, case_insensitive=case_insensitive)
        rep = self.total(rep_col, case_insensitive=case_insensitive)
        return {
            "dem": dem,
            "rep": rep,
            "total": dem + rep,
            "dem_share": dem / (dem + rep) if dem + rep > 0 else float("nan"),
        }

    def summary(
        self,
        *,
        pop_col: str | None = "POP20",
        county_col: str | None = "COUNTY",
    ) -> GraphSummary:
        total_pop = None
        if pop_col is not None and has_node_column(self.nodes, pop_col):
            total_pop = self.total_population(pop_col)

        county_counts = None
        if county_col is not None and has_node_column(self.nodes, county_col):
            county_counts = dict(
                Counter(str(node[county_col]) for node in self.nodes)
            )

        return GraphSummary(
            path=self.path,
            n_nodes=len(self.nodes),
            n_adjacency_rows=len(self.adjacency) if self.adjacency is not None else None,
            directed=self.raw.get("directed"),
            multigraph=self.raw.get("multigraph"),
            node_id_col=self.node_id_col,
            columns=self.columns,
            total_population=total_pop,
            county_counts=county_counts,
        )


def read_graph_json(
    path: str | Path,
    *,
    node_id_col: str = "id",
    require_adjacency_length: bool = True,
) -> GraphData:
    """
    Read a redistricting graph JSON file.

    The Ohio file currently used by the project is NetworkX adjacency JSON with:

        {
            "directed": false,
            "multigraph": false,
            "graph": [],
            "nodes": [...],
            "adjacency": [...]
        }

    This function focuses on node metadata because downstream modules need nodes
    for multiscale expansion and vote aggregation.
    """
    p = Path(path)

    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise GraphReaderError(
            f"graph file must contain a JSON object; got {type(raw).__name__}"
        )

    nodes = raw.get("nodes")

    if not isinstance(nodes, list):
        raise GraphReaderError("graph JSON does not contain a list field `nodes`")

    if not all(isinstance(node, dict) for node in nodes):
        raise GraphReaderError("every item in `nodes` must be a dictionary")

    adjacency = raw.get("adjacency")

    if adjacency is not None:
        if not isinstance(adjacency, list):
            raise GraphReaderError("`adjacency` must be a list when present")

        if require_adjacency_length and len(adjacency) != len(nodes):
            raise GraphReaderError(
                f"adjacency length {len(adjacency)} does not match "
                f"number of nodes {len(nodes)}"
            )

    node_by_id = build_node_lookup(nodes, node_id_col=node_id_col)

    return GraphData(
        path=p,
        raw=raw,
        nodes=[dict(node) for node in nodes],
        adjacency=adjacency,
        node_id_col=node_id_col,
        node_by_id=node_by_id,
    )


def read_graph_nodes(
    path: str | Path,
    *,
    node_id_col: str = "id",
) -> list[dict[str, Any]]:
    """
    Convenience function returning only node metadata.
    """
    return read_graph_json(path, node_id_col=node_id_col).nodes


def build_node_lookup(
    nodes: Iterable[Mapping[str, Any]],
    *,
    node_id_col: str = "id",
) -> dict[Any, dict[str, Any]]:
    """
    Build a node-id lookup from node metadata.

    Raises if a node is missing the id field or if ids are duplicated.
    """
    out: dict[Any, dict[str, Any]] = {}

    for idx, node in enumerate(nodes):
        if node_id_col not in node:
            raise GraphReaderError(
                f"node at index {idx} is missing node id column {node_id_col!r}"
            )

        node_id = node[node_id_col]

        if node_id in out:
            raise GraphReaderError(f"duplicate node id: {node_id!r}")

        out[node_id] = dict(node)

    return out


def has_node_column(
    nodes: Iterable[Mapping[str, Any]],
    column: str,
    *,
    case_insensitive: bool = False,
) -> bool:
    """
    Return whether every node has the requested column.
    """
    nodes_list = list(nodes)

    if not nodes_list:
        return False

    if case_insensitive:
        target = column.lower()
        return all(
            any(str(k).lower() == target for k in node.keys())
            for node in nodes_list
        )

    return all(column in node for node in nodes_list)


def require_node_columns(
    nodes: Iterable[Mapping[str, Any]],
    columns: Iterable[str],
    *,
    case_insensitive: bool = False,
) -> None:
    """
    Validate that every node has the requested columns.
    """
    nodes_list = list(nodes)

    if not nodes_list:
        raise GraphReaderError("node list is empty")

    missing: dict[str, int] = {}

    for col in columns:
        count_missing = 0

        for node in nodes_list:
            if case_insensitive:
                target = col.lower()
                has_col = any(str(k).lower() == target for k in node.keys())
            else:
                has_col = col in node

            if not has_col:
                count_missing += 1

        if count_missing:
            missing[col] = count_missing

    if missing:
        raise GraphReaderError(
            "required node columns are missing: "
            + ", ".join(f"{col} missing in {n} nodes" for col, n in missing.items())
        )


def _lookup_node_value(
    node: Mapping[str, Any],
    column: str,
    *,
    case_insensitive: bool = False,
) -> Any:
    if column in node:
        return node[column]

    if case_insensitive:
        target = column.lower()
        for key, value in node.items():
            if str(key).lower() == target:
                return value

    raise GraphReaderError(f"node is missing column {column!r}")


def sum_numeric_node_column(
    nodes: Iterable[Mapping[str, Any]],
    column: str,
    *,
    case_insensitive: bool = False,
) -> float:
    """
    Sum a numeric column over all nodes.
    """
    total = 0.0

    for idx, node in enumerate(nodes):
        value = _lookup_node_value(
            node,
            column,
            case_insensitive=case_insensitive,
        )

        try:
            total += float(value)
        except (TypeError, ValueError) as exc:
            raise GraphReaderError(
                f"node at index {idx} has non-numeric {column}={value!r}"
            ) from exc

    return total


def adjacency_edges(
    graph: GraphData,
    *,
    include_self_loops: bool = False,
    deduplicate_undirected: bool = True,
) -> list[tuple[Any, Any, dict[str, Any]]]:
    """
    Convert NetworkX adjacency JSON rows into an edge list.

    Returns
    -------
    list of (source_id, target_id, edge_attributes)

    Notes
    -----
    The uploaded Ohio graph stores adjacency rows where each neighbor dictionary
    has an `id` field and optional attributes such as `length`.
    """
    if graph.adjacency is None:
        raise GraphReaderError("graph has no adjacency field")

    node_ids = graph.node_ids

    if len(node_ids) != len(graph.adjacency):
        raise GraphReaderError(
            "number of node ids does not match number of adjacency rows"
        )

    edges: list[tuple[Any, Any, dict[str, Any]]] = []
    seen: set[tuple[Any, Any]] = set()

    for row_idx, neighbors in enumerate(graph.adjacency):
        source = node_ids[row_idx]

        if not isinstance(neighbors, list):
            raise GraphReaderError(
                f"adjacency row {row_idx} must be a list; got {type(neighbors).__name__}"
            )

        for neighbor in neighbors:
            if not isinstance(neighbor, dict):
                raise GraphReaderError(
                    f"adjacency entry in row {row_idx} must be a dict"
                )

            if "id" not in neighbor:
                raise GraphReaderError(
                    f"adjacency entry in row {row_idx} is missing `id`"
                )

            target = neighbor["id"]

            if not include_self_loops and source == target:
                continue

            if deduplicate_undirected and not graph.raw.get("directed", False):
                edge_key = tuple(sorted((source, target), key=str))
                if edge_key in seen:
                    continue
                seen.add(edge_key)

            attrs = {k: v for k, v in neighbor.items() if k != "id"}
            edges.append((source, target, attrs))

    return edges


def infer_key_cols_for_multiscale(
    graph: GraphData,
    *,
    preferred: list[list[str]] | None = None,
) -> list[str]:
    """
    Infer plausible key columns for multiscale assignment expansion.

    This is intentionally conservative. For Ohio `OHpct20.json`, useful choices
    are often:

        ["COUNTY", "NAME"]

    or sometimes:

        ["COUNTY", "ORIG_NAME"]

    depending on how the AtlasIO districting keys were written.
    """
    if preferred is None:
        preferred = [
            ["COUNTY", "NAME"],
            ["COUNTY", "ORIG_NAME"],
            ["county", "prec_id"],
            ["COUNTY"],
        ]

    available = set(graph.columns)

    for candidate in preferred:
        if all(col in available for col in candidate):
            return candidate

    raise GraphReaderError(
        "could not infer multiscale key columns. "
        f"Available columns: {graph.columns}"
    )