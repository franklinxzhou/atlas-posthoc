from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from atlas_posthoc.assignments.keys import (
    KeySchemeDiagnostic,
    choose_best_key_scheme,
    diagnose_key_scheme,
)
from atlas_posthoc.interop.gerrychain import (
    GerryChainGraphLoadResult,
    load_gerrychain_graph,
    partition_from_record,
)
from atlas_posthoc.io.atlas_reader import (
    AtlasRecord,
    ParseSummary,
    infer_chain_id,
    iter_atlas_records,
)
from atlas_posthoc.io.graph_reader import GraphData, read_graph_json


class ChainPartitionError(Exception):
    """Base exception for AtlasIO-to-GerryChain partition pipeline errors."""


@dataclass
class ChainPartitionSummary:
    chain_id: str
    path: str
    n_partitions: int
    n_records_seen: int
    selected_key_cols: list[str]
    key_diagnostic: dict[str, Any]
    parse_summary: dict[str, Any]
    max_maps_per_chain: int | None
    thin_step: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChainPartitionResult:
    """
    GerryChain partitions produced from one AtlasIO chain.

    Warning
    -------
    These objects are primarily intended for PACE / Wasserplan workflows.
    Holding many GerryChain Partition objects in memory can be expensive.
    Computing pairwise Wasserplan distances on them can be much more expensive.
    Prefer `thin_step` and `max_maps_per_chain` for exploratory work.
    """

    chain_id: str
    path: Path
    partitions: list[Any]
    steps: list[int]
    map_indices: list[int]
    names: list[str | None]
    selected_key_cols: list[str]
    key_diagnostic: KeySchemeDiagnostic
    parse_summary: ParseSummary
    graph_load_result: GerryChainGraphLoadResult | None
    thin_step: int
    max_maps_per_chain: int | None

    @property
    def n_partitions(self) -> int:
        return len(self.partitions)

    def summary(self) -> ChainPartitionSummary:
        return ChainPartitionSummary(
            chain_id=self.chain_id,
            path=str(self.path),
            n_partitions=self.n_partitions,
            n_records_seen=len(self.map_indices),
            selected_key_cols=self.selected_key_cols,
            key_diagnostic=self.key_diagnostic.to_dict(),
            parse_summary=self.parse_summary.to_dict(),
            max_maps_per_chain=self.max_maps_per_chain,
            thin_step=self.thin_step,
        )


@dataclass
class MultiChainPartitionResult:
    """
    GerryChain partitions produced from multiple AtlasIO chains.

    Warning
    -------
    If each chain has M sampled partitions and there are J chains, pairwise
    distance computations over the pooled sample involve roughly `(J*M)^2 / 2`
    distances. For Wasserplan, this can be very expensive.
    """

    chains: dict[str, ChainPartitionResult]

    @property
    def partitions_by_chain(self) -> dict[str, list[Any]]:
        return {
            chain_id: result.partitions
            for chain_id, result in self.chains.items()
        }

    @property
    def pooled_partitions(self) -> list[Any]:
        out: list[Any] = []
        for result in self.chains.values():
            out.extend(result.partitions)
        return out

    @property
    def pooled_chain_ids(self) -> list[str]:
        out: list[str] = []
        for chain_id, result in self.chains.items():
            out.extend([chain_id] * result.n_partitions)
        return out

    def summaries(self) -> dict[str, dict[str, Any]]:
        return {
            chain_id: result.summary().to_dict()
            for chain_id, result in self.chains.items()
        }


def _as_graph_data(graph: str | Path | GraphData) -> GraphData:
    if isinstance(graph, GraphData):
        return graph

    return read_graph_json(graph)


def _maybe_warn_expensive_partition_build(
    *,
    max_maps_per_chain: int | None,
    n_chains: int = 1,
) -> None:
    if max_maps_per_chain is None:
        warnings.warn(
            "Building GerryChain partitions for all maps in each chain may be "
            "memory-intensive. Downstream pairwise Wasserplan distances scale "
            "quadratically in the pooled number of partitions. Consider setting "
            "`max_maps_per_chain` and/or increasing `thin_step` for exploratory runs.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    pooled = max_maps_per_chain * n_chains
    approx_pairs = pooled * (pooled - 1) // 2

    if pooled >= 1000:
        warnings.warn(
            f"This request may produce up to {pooled} pooled partitions and "
            f"approximately {approx_pairs:,} pairwise distances downstream. "
            "This is likely expensive for Wasserplan. Consider using fewer maps "
            "for an initial PACE smoke test.",
            RuntimeWarning,
            stacklevel=2,
        )


def _record_step(
    record: AtlasRecord,
    *,
    output_frequency: int | None = None,
) -> int:
    if record.step is not None:
        return int(record.step)

    if output_frequency is not None:
        return int(record.map_index * output_frequency)

    return int(record.map_index)


def _load_gerrychain_graph_if_needed(
    graph: str | Path | GraphData,
    *,
    gerrychain_graph: Any | None,
    name_col: str,
) -> tuple[Any, GerryChainGraphLoadResult | None]:
    if gerrychain_graph is not None:
        return gerrychain_graph, None

    if isinstance(graph, (str, Path)):
        loaded = load_gerrychain_graph(graph, name_col=name_col)
        return loaded.graph, loaded

    raise ChainPartitionError(
        "gerrychain_graph must be provided when `graph` is already a GraphData "
        "object. GerryChain Graph loading requires the original graph JSON path."
    )


def _resolve_key_scheme(
    *,
    first_record: AtlasRecord,
    graph_data: GraphData,
    key_cols: Sequence[str] | None,
    node_id_col: str | None,
    require_perfect_key_scheme: bool,
) -> tuple[list[str], KeySchemeDiagnostic]:
    if key_cols is None:
        diagnostic = choose_best_key_scheme(
            first_record.assignment,
            graph_data.nodes,
            node_id_col=node_id_col,
            require_perfect=require_perfect_key_scheme,
        )
        return diagnostic.key_cols, diagnostic

    selected = list(key_cols)

    diagnostic = diagnose_key_scheme(
        first_record.assignment,
        graph_data.nodes,
        selected,
        node_id_col=node_id_col,
    )

    if require_perfect_key_scheme and not diagnostic.is_perfect_match:
        raise ChainPartitionError(
            "provided key_cols do not perfectly match the first record. "
            f"key_cols={selected}, assigned={diagnostic.n_assigned}/{diagnostic.n_nodes}, "
            f"missing={diagnostic.n_missing}, ambiguous={diagnostic.n_ambiguous}, "
            f"unmatched_assignment_keys={diagnostic.n_unmatched_assignment_keys}"
        )

    return selected, diagnostic


def build_chain_partitions(
    chain_file: str | Path,
    graph: str | Path | GraphData,
    *,
    gerrychain_graph: Any | None = None,
    chain_id: str | None = None,
    key_cols: Sequence[str] | None = None,
    node_id_col: str | None = "id",
    name_col: str = "NAME",
    thin_step: int = 1,
    max_maps_per_chain: int | None = 250,
    skip_bad_lines: bool = True,
    output_frequency: int | None = None,
    require_perfect_key_scheme: bool = True,
    missing_policy: str = "error",
    ambiguous_policy: str = "error",
    require_complete_assignment: bool = True,
) -> ChainPartitionResult:
    """
    Convert one AtlasIO chain into GerryChain Partition objects.

    Computational warning
    ---------------------
    This function is intended for PACE / state-space diagnostics. Do not casually
    convert a full long chain unless you really need it. A downstream pairwise
    distance matrix over N partitions requires roughly N(N-1)/2 distance calls.
    For Wasserplan, this can be slow.

    Recommended exploratory settings
    --------------------------------
    Start with something like:

        thin_step=10
        max_maps_per_chain=100 or 250

    Then scale up only after the pipeline works.
    """
    if thin_step <= 0:
        raise ValueError("thin_step must be positive")

    if max_maps_per_chain is not None and max_maps_per_chain <= 0:
        raise ValueError("max_maps_per_chain must be positive or None")

    _maybe_warn_expensive_partition_build(
        max_maps_per_chain=max_maps_per_chain,
        n_chains=1,
    )

    path = Path(chain_file)
    cid = chain_id or infer_chain_id(path)

    graph_data = _as_graph_data(graph)
    gc_graph, graph_load_result = _load_gerrychain_graph_if_needed(
        graph,
        gerrychain_graph=gerrychain_graph,
        name_col=name_col,
    )

    parse_summary = ParseSummary(path=path, chain_id=cid)

    record_iter = iter_atlas_records(
        path,
        chain_id=cid,
        thin_step=thin_step,
        skip_bad_lines=skip_bad_lines,
        summary=parse_summary,
    )

    try:
        first_record = next(record_iter)
    except StopIteration as exc:
        raise ChainPartitionError(f"no valid records found in {path}") from exc

    selected_key_cols, key_diagnostic = _resolve_key_scheme(
        first_record=first_record,
        graph_data=graph_data,
        key_cols=key_cols,
        node_id_col=node_id_col,
        require_perfect_key_scheme=require_perfect_key_scheme,
    )

    partitions: list[Any] = []
    steps: list[int] = []
    map_indices: list[int] = []
    names: list[str | None] = []

    def process_record(record: AtlasRecord) -> None:
        conversion = partition_from_record(
            record,
            gc_graph,
            graph_data.nodes,
            selected_key_cols,
            node_id_col=node_id_col,
            missing_policy=missing_policy,
            ambiguous_policy=ambiguous_policy,
            require_complete_assignment=require_complete_assignment,
        )

        partitions.append(conversion.partition)
        steps.append(_record_step(record, output_frequency=output_frequency))
        map_indices.append(record.map_index)
        names.append(record.name)

    process_record(first_record)

    for record in record_iter:
        if max_maps_per_chain is not None and len(partitions) >= max_maps_per_chain:
            break

        process_record(record)

    if not partitions:
        raise ChainPartitionError(f"no partitions were produced for {path}")

    return ChainPartitionResult(
        chain_id=cid,
        path=path,
        partitions=partitions,
        steps=steps,
        map_indices=map_indices,
        names=names,
        selected_key_cols=selected_key_cols,
        key_diagnostic=key_diagnostic,
        parse_summary=parse_summary,
        graph_load_result=graph_load_result,
        thin_step=thin_step,
        max_maps_per_chain=max_maps_per_chain,
    )


def build_multi_chain_partitions(
    chain_files: Sequence[str | Path],
    graph: str | Path | GraphData,
    *,
    gerrychain_graph: Any | None = None,
    chain_ids: Sequence[str] | None = None,
    key_cols: Sequence[str] | None = None,
    node_id_col: str | None = "id",
    name_col: str = "NAME",
    thin_step: int = 1,
    max_maps_per_chain: int | None = 250,
    skip_bad_lines: bool = True,
    output_frequency: int | None = None,
    require_perfect_key_scheme: bool = True,
    missing_policy: str = "error",
    ambiguous_policy: str = "error",
    require_complete_assignment: bool = True,
) -> MultiChainPartitionResult:
    """
    Convert multiple AtlasIO chains into GerryChain Partition objects.

    Computational warning
    ---------------------
    If you later compute a full pairwise distance matrix on the pooled partitions,
    the number of distances is quadratic in the total number of partitions.

    Example
    -------
    Four chains with `max_maps_per_chain=250` create at most 1000 partitions,
    which implies about 499,500 pairwise distances. That may already be heavy
    for Wasserplan.
    """
    if chain_ids is not None and len(chain_ids) != len(chain_files):
        raise ChainPartitionError("chain_ids must have the same length as chain_files")

    if not chain_files:
        raise ChainPartitionError("no chain files were provided")

    _maybe_warn_expensive_partition_build(
        max_maps_per_chain=max_maps_per_chain,
        n_chains=len(chain_files),
    )

    graph_data = _as_graph_data(graph)

    if gerrychain_graph is None:
        if isinstance(graph, (str, Path)):
            graph_load = load_gerrychain_graph(graph, name_col=name_col)
            gc_graph = graph_load.graph
        else:
            raise ChainPartitionError(
                "gerrychain_graph must be provided when `graph` is already GraphData"
            )
    else:
        gc_graph = gerrychain_graph

    results: dict[str, ChainPartitionResult] = {}

    for idx, chain_file in enumerate(chain_files):
        cid = chain_ids[idx] if chain_ids is not None else None

        result = build_chain_partitions(
            chain_file,
            graph_data,
            gerrychain_graph=gc_graph,
            chain_id=cid,
            key_cols=key_cols,
            node_id_col=node_id_col,
            name_col=name_col,
            thin_step=thin_step,
            max_maps_per_chain=max_maps_per_chain,
            skip_bad_lines=skip_bad_lines,
            output_frequency=output_frequency,
            require_perfect_key_scheme=require_perfect_key_scheme,
            missing_policy=missing_policy,
            ambiguous_policy=ambiguous_policy,
            require_complete_assignment=require_complete_assignment,
        )

        if result.chain_id in results:
            raise ChainPartitionError(f"duplicate chain_id: {result.chain_id!r}")

        results[result.chain_id] = result

    return MultiChainPartitionResult(chains=results)