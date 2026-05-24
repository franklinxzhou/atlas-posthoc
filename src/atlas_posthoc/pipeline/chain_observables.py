from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from atlas_posthoc.assignments.keys import (
    KeySchemeDiagnostic,
    choose_best_key_scheme,
    diagnose_key_scheme,
)
from atlas_posthoc.assignments.multiscale import expand_multiscale_assignment
from atlas_posthoc.io.atlas_reader import (
    AtlasRecord,
    ParseSummary,
    infer_chain_id,
    iter_atlas_records,
    read_atlas_header,
)
from atlas_posthoc.io.graph_reader import GraphData, read_graph_json
from atlas_posthoc.observables.compactness import (
    CompactnessError,
    extract_ordered_compactness_from_record,
)
from atlas_posthoc.observables.vote_share import compute_ordered_dem_vote_shares


class ChainObservableError(Exception):
    """Base exception for chain-observable construction errors."""


@dataclass
class ChainObservableSummary:
    chain_id: str
    path: str
    n_records_processed: int
    n_vote_vectors: int
    n_compactness_vectors: int
    selected_key_cols: list[str]
    key_diagnostic: dict[str, Any]
    parse_summary: dict[str, Any]
    compactness_errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChainObservableResult:
    chain_id: str
    path: Path
    steps: list[int]
    map_indices: list[int]
    names: list[str | None]
    vote_shares: np.ndarray
    compactness_scores: np.ndarray | None
    selected_key_cols: list[str]
    key_diagnostic: KeySchemeDiagnostic
    parse_summary: ParseSummary
    compactness_errors: list[str]

    @property
    def n_maps(self) -> int:
        return int(self.vote_shares.shape[0])

    @property
    def n_ranks(self) -> int:
        return int(self.vote_shares.shape[1])

    def summary(self) -> ChainObservableSummary:
        return ChainObservableSummary(
            chain_id=self.chain_id,
            path=str(self.path),
            n_records_processed=self.n_maps,
            n_vote_vectors=int(self.vote_shares.shape[0]),
            n_compactness_vectors=(
                0 if self.compactness_scores is None else int(self.compactness_scores.shape[0])
            ),
            selected_key_cols=self.selected_key_cols,
            key_diagnostic=self.key_diagnostic.to_dict(),
            parse_summary=self.parse_summary.to_dict(),
            compactness_errors=self.compactness_errors,
        )


@dataclass
class MultiChainObservableResult:
    chains: dict[str, ChainObservableResult]

    @property
    def vote_runs(self) -> dict[str, np.ndarray]:
        return {chain_id: result.vote_shares for chain_id, result in self.chains.items()}

    @property
    def compactness_runs(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}

        for chain_id, result in self.chains.items():
            if result.compactness_scores is not None:
                out[chain_id] = result.compactness_scores

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


def _none_to_nan_vector(values: Sequence[float | None]) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else float(value) for value in values],
        dtype=float,
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


def _resolve_key_scheme(
    *,
    first_record: AtlasRecord,
    graph: GraphData,
    key_cols: Sequence[str] | None,
    node_id_col: str | None,
    require_perfect_key_scheme: bool,
) -> tuple[list[str], KeySchemeDiagnostic]:
    if key_cols is None:
        diagnostic = choose_best_key_scheme(
            first_record.assignment,
            graph.nodes,
            node_id_col=node_id_col,
            require_perfect=require_perfect_key_scheme,
        )
        return diagnostic.key_cols, diagnostic

    selected = list(key_cols)

    diagnostic = diagnose_key_scheme(
        first_record.assignment,
        graph.nodes,
        selected,
        node_id_col=node_id_col,
    )

    if require_perfect_key_scheme and not diagnostic.is_perfect_match:
        raise ChainObservableError(
            "provided key_cols do not perfectly match the first record. "
            f"key_cols={selected}, assigned={diagnostic.n_assigned}/{diagnostic.n_nodes}, "
            f"missing={diagnostic.n_missing}, ambiguous={diagnostic.n_ambiguous}, "
            f"unmatched_assignment_keys={diagnostic.n_unmatched_assignment_keys}"
        )

    return selected, diagnostic


def build_chain_observables(
    chain_file: str | Path,
    graph: str | Path | GraphData,
    *,
    chain_id: str | None = None,
    key_cols: Sequence[str] | None = None,
    node_id_col: str | None = "id",
    dem_col: str,
    rep_col: str,
    n_districts: int | None = None,
    output_frequency: int | None = None,
    thin_step: int = 1,
    skip_bad_lines: bool = True,
    require_perfect_key_scheme: bool = True,
    missing_policy: str = "error",
    ambiguous_policy: str = "error",
    compute_compactness: bool = True,
    require_compactness: bool = False,
    compactness_keys: Sequence[str] | None = None,
    allow_zero_vote_districts: bool = False,
    allow_missing_compactness_scores: bool = False,
) -> ChainObservableResult:
    """
    Build ordered vote-share and compactness matrices for one AtlasIO chain.

    Parameters
    ----------
    chain_file:
        Path to one AtlasIO `.jsonl` or `.jsonl.gz` output file.

    graph:
        Either a `GraphData` object or a path to a graph JSON file.

    key_cols:
        Graph node columns used to match Atlas assignment keys. If None, the
        best scheme is chosen automatically using the first parsed record.

    dem_col, rep_col:
        Historical election vote columns used to compute Democratic vote share.

    n_districts:
        Expected number of districts. If omitted, tries to infer from the Atlas
        header field `districts`.

    output_frequency:
        Used to reconstruct proposal counts when record names do not encode
        steps.

    Returns
    -------
    ChainObservableResult
        Contains ordered vote-share matrix and optional compactness matrix.

    Notes
    -----
    This function intentionally does not compute TV or R-hat. It only constructs
    the per-chain observable matrices that those diagnostics consume.
    """
    path = Path(chain_file)
    graph_data = _as_graph_data(graph)
    cid = chain_id or infer_chain_id(path)

    header = read_atlas_header(path)

    if n_districts is None:
        maybe_districts = header.atlas_param.get("districts")
        if maybe_districts is not None:
            n_districts = int(maybe_districts)

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
        raise ChainObservableError(f"no valid records found in {path}") from exc

    selected_key_cols, key_diagnostic = _resolve_key_scheme(
        first_record=first_record,
        graph=graph_data,
        key_cols=key_cols,
        node_id_col=node_id_col,
        require_perfect_key_scheme=require_perfect_key_scheme,
    )

    vote_vectors: list[np.ndarray] = []
    compactness_vectors: list[np.ndarray] = []
    compactness_errors: list[str] = []

    steps: list[int] = []
    map_indices: list[int] = []
    names: list[str | None] = []

    def process_record(record: AtlasRecord) -> None:
        expanded = expand_multiscale_assignment(
            record.assignment,
            graph_data.nodes,
            selected_key_cols,
            node_id_col=node_id_col,
            missing_policy=missing_policy,  # type: ignore[arg-type]
            ambiguous_policy=ambiguous_policy,  # type: ignore[arg-type]
        )

        vote_result = compute_ordered_dem_vote_shares(
            expanded.node_to_district,
            graph_data.nodes,
            dem_col=dem_col,
            rep_col=rep_col,
            node_id_col=node_id_col,
            n_districts=n_districts,
            sort="ascending",
            allow_zero_vote_districts=allow_zero_vote_districts,
        )

        vote_vectors.append(
            _none_to_nan_vector(vote_result.ordered_vote_shares)
        )

        if compute_compactness:
            try:
                kwargs: dict[str, Any] = {
                    "n_districts": n_districts,
                    "sort": "ascending",
                    "allow_missing_scores": allow_missing_compactness_scores,
                }

                if compactness_keys is not None:
                    kwargs["keys"] = compactness_keys

                compactness_result = extract_ordered_compactness_from_record(
                    record,
                    **kwargs,
                )

                compactness_vectors.append(
                    _none_to_nan_vector(compactness_result.ordered_scores)
                )

            except CompactnessError as exc:
                message = (
                    f"record line {record.line_number}, map_index {record.map_index}: {exc}"
                )
                compactness_errors.append(message)

                if require_compactness:
                    raise ChainObservableError(message) from exc

        steps.append(_record_step(record, output_frequency=output_frequency))
        map_indices.append(record.map_index)
        names.append(record.name)

    process_record(first_record)

    for record in record_iter:
        process_record(record)

    if not vote_vectors:
        raise ChainObservableError(f"no vote vectors were produced for {path}")

    vote_matrix = np.vstack(vote_vectors)

    compactness_matrix: np.ndarray | None

    if not compute_compactness:
        compactness_matrix = None
    elif len(compactness_vectors) == 0:
        compactness_matrix = None
    elif len(compactness_vectors) == len(vote_vectors):
        compactness_matrix = np.vstack(compactness_vectors)
    else:
        raise ChainObservableError(
            "compactness scores were available for only some records. "
            f"vote_vectors={len(vote_vectors)}, compactness_vectors={len(compactness_vectors)}. "
            "Use require_compactness=True to fail earlier, or inspect compactness_errors."
        )

    return ChainObservableResult(
        chain_id=cid,
        path=path,
        steps=steps,
        map_indices=map_indices,
        names=names,
        vote_shares=vote_matrix,
        compactness_scores=compactness_matrix,
        selected_key_cols=selected_key_cols,
        key_diagnostic=key_diagnostic,
        parse_summary=parse_summary,
        compactness_errors=compactness_errors,
    )


def build_multi_chain_observables(
    chain_files: Sequence[str | Path],
    graph: str | Path | GraphData,
    *,
    chain_ids: Sequence[str] | None = None,
    **kwargs: Any,
) -> MultiChainObservableResult:
    """
    Build observable matrices for multiple AtlasIO chains.

    Returns an object whose `.vote_runs` and `.compactness_runs` can be passed
    directly into `total_variation.py` or `rhat.py`.
    """
    if chain_ids is not None and len(chain_ids) != len(chain_files):
        raise ChainObservableError(
            "chain_ids must have the same length as chain_files"
        )

    graph_data = _as_graph_data(graph)

    results: dict[str, ChainObservableResult] = {}

    for idx, chain_file in enumerate(chain_files):
        cid = chain_ids[idx] if chain_ids is not None else None

        result = build_chain_observables(
            chain_file,
            graph_data,
            chain_id=cid,
            **kwargs,
        )

        if result.chain_id in results:
            raise ChainObservableError(f"duplicate chain_id: {result.chain_id!r}")

        results[result.chain_id] = result

    return MultiChainObservableResult(chains=results)