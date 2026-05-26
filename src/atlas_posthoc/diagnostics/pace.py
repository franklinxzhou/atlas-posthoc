from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np


PACEBootstrapMode = Literal["pooled", "within_chain"]


class PACEError(Exception):
    """Base exception for PACE diagnostics."""


@dataclass
class PACECellSummary:
    label: int
    pooled_count: int
    pooled_probability: float
    chain_counts: dict[str, int]
    chain_probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PACEBootstrapResult:
    n_bootstrap: int
    confidence_level: float
    upper_quantile: float
    bootstrap_deltas: list[float]
    mode: PACEBootstrapMode

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PACEResult:
    delta: float
    raw_delta: float
    correction_factor: float
    threshold: float | None
    converged: bool | None
    n_chains: int
    n_cells: int
    n_total_samples: int
    chain_sizes: dict[str, int]
    labels: list[int]
    chain_names: list[str]
    contingency: list[list[int]]
    cell_summaries: list[PACECellSummary]
    bootstrap: PACEBootstrapResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta": self.delta,
            "raw_delta": self.raw_delta,
            "correction_factor": self.correction_factor,
            "threshold": self.threshold,
            "converged": self.converged,
            "n_chains": self.n_chains,
            "n_cells": self.n_cells,
            "n_total_samples": self.n_total_samples,
            "chain_sizes": self.chain_sizes,
            "labels": self.labels,
            "chain_names": self.chain_names,
            "contingency": self.contingency,
            "cell_summaries": [x.to_dict() for x in self.cell_summaries],
            "bootstrap": None if self.bootstrap is None else self.bootstrap.to_dict(),
        }


def _as_1d_array(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x)

    if arr.ndim != 1:
        raise PACEError(f"{name} must be one-dimensional; got shape {arr.shape}")

    if arr.size == 0:
        raise PACEError(f"{name} is empty")

    return arr


def normalize_chain_labels(
    labels: Sequence[Any],
    chain_ids: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """
    Normalize arbitrary cell labels and chain ids to integer arrays.

    Returns
    -------
    label_codes:
        Integer cell labels, shape `(n_samples,)`.

    chain_codes:
        Integer chain codes, shape `(n_samples,)`.

    chain_names:
        Chain names in code order.

    unique_labels:
        Original unique cell labels in code order, coerced to integers when possible.
    """
    label_arr = _as_1d_array(labels, name="labels")
    chain_arr = _as_1d_array(chain_ids, name="chain_ids")

    if label_arr.size != chain_arr.size:
        raise PACEError(
            f"labels and chain_ids must have the same length; "
            f"got {label_arr.size} and {chain_arr.size}"
        )

    chain_names = sorted({str(x) for x in chain_arr})
    chain_to_code = {name: i for i, name in enumerate(chain_names)}
    chain_codes = np.asarray([chain_to_code[str(x)] for x in chain_arr], dtype=int)

    # Use stable string sorting so labels can be ints, tuples, strings, etc.
    unique_raw = sorted(set(label_arr.tolist()), key=lambda x: str(x))
    label_to_code = {label: i for i, label in enumerate(unique_raw)}
    label_codes = np.asarray([label_to_code[x] for x in label_arr.tolist()], dtype=int)

    unique_labels: list[int] = []
    for i, raw in enumerate(unique_raw):
        try:
            unique_labels.append(int(raw))
        except Exception:
            unique_labels.append(i)

    return label_codes, chain_codes, chain_names, unique_labels


def contingency_table_from_codes(
    label_codes: Sequence[int],
    chain_codes: Sequence[int],
    *,
    n_cells: int | None = None,
    n_chains: int | None = None,
) -> np.ndarray:
    """
    Build chain-by-cell contingency table.

    Rows are chains. Columns are partition cells / clusters.
    """
    labels = np.asarray(label_codes, dtype=int)
    chains = np.asarray(chain_codes, dtype=int)

    if labels.ndim != 1 or chains.ndim != 1:
        raise PACEError("label_codes and chain_codes must be one-dimensional")

    if labels.size != chains.size:
        raise PACEError("label_codes and chain_codes must have the same length")

    if labels.size == 0:
        raise PACEError("cannot build contingency table from empty labels")

    if np.any(labels < 0) or np.any(chains < 0):
        raise PACEError("label_codes and chain_codes must be nonnegative")

    if n_cells is None:
        n_cells = int(labels.max()) + 1

    if n_chains is None:
        n_chains = int(chains.max()) + 1

    table = np.zeros((n_chains, n_cells), dtype=int)

    for chain, label in zip(chains, labels):
        table[chain, label] += 1

    return table


def contingency_table(
    labels: Sequence[Any],
    chain_ids: Sequence[Any],
) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Build chain-by-cell contingency table from arbitrary labels and chain ids.

    Returns
    -------
    table:
        Array of shape `(n_chains, n_cells)`.

    chain_names:
        Chain names in row order.

    unique_labels:
        Cell labels in column order.
    """
    label_codes, chain_codes, chain_names, unique_labels = normalize_chain_labels(
        labels,
        chain_ids,
    )

    table = contingency_table_from_codes(
        label_codes,
        chain_codes,
        n_cells=len(unique_labels),
        n_chains=len(chain_names),
    )

    return table, chain_names, unique_labels


def pace_correction_factor(n_chains: int, *, apply: bool = True) -> float:
    """
    Bias-correction factor used by the PACE-style statistic.

    If `apply=False`, returns 1.0.
    """
    if n_chains < 2:
        raise PACEError("PACE requires at least two chains")

    if not apply:
        return 1.0

    return float(np.sqrt(n_chains) / (n_chains - 1))


def pace_delta_from_contingency(
    table: np.ndarray,
    *,
    apply_correction: bool = True,
) -> tuple[float, float, float]:
    """
    Compute PACE discrepancy from a chain-by-cell contingency table.

    The raw discrepancy is:

        max_chain TV(p_chain, p_pooled)

    where TV is `0.5 * L1`.

    The reported PACE delta applies the correction factor:

        sqrt(J) / (J - 1)

    where `J` is the number of chains.

    Returns
    -------
    delta:
        Corrected PACE discrepancy.

    raw_delta:
        Uncorrected maximum chain-vs-pooled TV.

    correction_factor:
        Multiplicative correction factor.
    """
    counts = np.asarray(table, dtype=float)

    if counts.ndim != 2:
        raise PACEError(f"contingency table must be 2D; got shape {counts.shape}")

    n_chains, n_cells = counts.shape

    if n_chains < 2:
        raise PACEError("PACE requires at least two chains")

    if n_cells == 0:
        raise PACEError("PACE requires at least one cell")

    chain_sizes = counts.sum(axis=1)

    if np.any(chain_sizes <= 0):
        raise PACEError("every chain must have at least one sample")

    pooled_size = float(counts.sum())

    if pooled_size <= 0:
        raise PACEError("contingency table has zero total count")

    chain_probs = counts / chain_sizes[:, None]
    pooled_probs = counts.sum(axis=0) / pooled_size

    tv_by_chain = 0.5 * np.sum(np.abs(chain_probs - pooled_probs[None, :]), axis=1)
    raw_delta = float(np.max(tv_by_chain))

    correction = pace_correction_factor(n_chains, apply=apply_correction)
    delta = correction * raw_delta

    return float(delta), raw_delta, correction


def _cell_summaries(
    table: np.ndarray,
    *,
    chain_names: Sequence[str],
    labels: Sequence[int],
) -> list[PACECellSummary]:
    counts = np.asarray(table, dtype=int)
    total = int(counts.sum())
    chain_totals = counts.sum(axis=1)

    summaries: list[PACECellSummary] = []

    for cell_idx, label in enumerate(labels):
        pooled_count = int(counts[:, cell_idx].sum())
        pooled_prob = pooled_count / total if total > 0 else float("nan")

        chain_counts = {
            chain_names[j]: int(counts[j, cell_idx])
            for j in range(len(chain_names))
        }

        chain_probs = {
            chain_names[j]: (
                float(counts[j, cell_idx] / chain_totals[j])
                if chain_totals[j] > 0
                else float("nan")
            )
            for j in range(len(chain_names))
        }

        summaries.append(
            PACECellSummary(
                label=int(label),
                pooled_count=pooled_count,
                pooled_probability=float(pooled_prob),
                chain_counts=chain_counts,
                chain_probabilities=chain_probs,
            )
        )

    return summaries


def bootstrap_pace_delta(
    table: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
    mode: PACEBootstrapMode = "pooled",
    apply_correction: bool = True,
) -> PACEBootstrapResult:
    """
    Bootstrap a null distribution for the PACE discrepancy.

    Modes
    -----
    pooled:
        Resample each chain from the pooled empirical cell distribution. This is
        the usual null model that all chains target the same distribution.

    within_chain:
        Resample each chain from its own empirical distribution. This is mainly
        useful as a stability check, not as a common-target null.
    """
    counts = np.asarray(table, dtype=int)

    if counts.ndim != 2:
        raise PACEError("table must be 2D")

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be in (0, 1)")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    n_chains, n_cells = counts.shape
    chain_sizes = counts.sum(axis=1)

    if np.any(chain_sizes <= 0):
        raise PACEError("every chain must have at least one sample")

    total = counts.sum()

    if total <= 0:
        raise PACEError("table has zero total count")

    pooled_probs = counts.sum(axis=0) / total

    chain_probs = counts / chain_sizes[:, None]

    deltas = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        boot = np.zeros_like(counts)

        for j in range(n_chains):
            if mode == "pooled":
                probs = pooled_probs
            elif mode == "within_chain":
                probs = chain_probs[j]
            else:
                raise ValueError(f"unknown bootstrap mode: {mode!r}")

            boot[j, :] = rng.multinomial(int(chain_sizes[j]), probs)

        deltas[b], _, _ = pace_delta_from_contingency(
            boot,
            apply_correction=apply_correction,
        )

    upper = float(np.quantile(deltas, confidence_level))

    return PACEBootstrapResult(
        n_bootstrap=int(n_bootstrap),
        confidence_level=float(confidence_level),
        upper_quantile=upper,
        bootstrap_deltas=deltas.tolist(),
        mode=mode,
    )


def pace_from_contingency(
    table: np.ndarray,
    *,
    chain_names: Sequence[str] | None = None,
    labels: Sequence[int] | None = None,
    threshold: float | None = 0.1,
    apply_correction: bool = True,
    n_bootstrap: int = 0,
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
    bootstrap_mode: PACEBootstrapMode = "pooled",
) -> PACEResult:
    """
    Compute PACE diagnostic from a chain-by-cell contingency table.
    """
    counts = np.asarray(table, dtype=int)

    if counts.ndim != 2:
        raise PACEError("table must be 2D")

    n_chains, n_cells = counts.shape

    if chain_names is None:
        chain_names = [f"chain_{j}" for j in range(n_chains)]
    else:
        chain_names = [str(x) for x in chain_names]

    if len(chain_names) != n_chains:
        raise PACEError("chain_names length must match table rows")

    if labels is None:
        labels = list(range(n_cells))
    else:
        labels = [int(x) for x in labels]

    if len(labels) != n_cells:
        raise PACEError("labels length must match table columns")

    delta, raw_delta, correction = pace_delta_from_contingency(
        counts,
        apply_correction=apply_correction,
    )

    bootstrap = None
    converged = None

    if n_bootstrap > 0:
        bootstrap = bootstrap_pace_delta(
            counts,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state,
            mode=bootstrap_mode,
            apply_correction=apply_correction,
        )

    if threshold is not None:
        # Conservative rule: if bootstrap is requested, judge by the upper
        # bootstrap quantile; otherwise judge by the observed delta.
        compare_value = bootstrap.upper_quantile if bootstrap is not None else delta
        converged = bool(compare_value <= threshold)

    chain_sizes = {
        chain_names[j]: int(counts[j, :].sum())
        for j in range(n_chains)
    }

    return PACEResult(
        delta=delta,
        raw_delta=raw_delta,
        correction_factor=correction,
        threshold=threshold,
        converged=converged,
        n_chains=n_chains,
        n_cells=n_cells,
        n_total_samples=int(counts.sum()),
        chain_sizes=chain_sizes,
        labels=list(labels),
        chain_names=list(chain_names),
        contingency=counts.tolist(),
        cell_summaries=_cell_summaries(
            counts,
            chain_names=chain_names,
            labels=labels,
        ),
        bootstrap=bootstrap,
    )


def pace_from_labels(
    labels: Sequence[Any],
    chain_ids: Sequence[Any],
    *,
    threshold: float | None = 0.1,
    apply_correction: bool = True,
    n_bootstrap: int = 0,
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
    bootstrap_mode: PACEBootstrapMode = "pooled",
) -> PACEResult:
    """
    Compute PACE diagnostic from cell labels and chain ids.

    This is the main API once an external clustering method, exact-state
    labeling method, or Wasserplan-based partitioning method has assigned each
    sampled plan to a cell.
    """
    table, chain_names, unique_labels = contingency_table(labels, chain_ids)

    return pace_from_contingency(
        table,
        chain_names=chain_names,
        labels=unique_labels,
        threshold=threshold,
        apply_correction=apply_correction,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
        bootstrap_mode=bootstrap_mode,
    )


def canonicalize_assignment_state(
    state: Sequence[Any],
    *,
    relabel: bool = True,
) -> tuple[int, ...]:
    """
    Canonicalize one assignment vector for exact-state PACE.

    If `relabel=True`, district labels are replaced by 1, 2, ... in order of
    first appearance. This makes exact-state labels invariant to arbitrary
    district-label permutations, assuming node order is fixed.
    """
    if not relabel:
        return tuple(int(x) for x in state)

    label_map: dict[Any, int] = {}
    next_label = 1
    out: list[int] = []

    for value in state:
        if value not in label_map:
            label_map[value] = next_label
            next_label += 1
        out.append(label_map[value])

    return tuple(out)


def exact_state_labels_from_matrix(
    states: Sequence[Sequence[Any]],
    *,
    relabel: bool = True,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """
    Assign exact-state cell labels to rows of a state matrix.

    Rows that have identical canonical assignment vectors receive the same label.
    """
    arr = np.asarray(states)

    if arr.ndim != 2:
        raise PACEError(f"states must be a 2D matrix; got shape {arr.shape}")

    if arr.shape[0] == 0:
        raise PACEError("states matrix has no rows")

    state_to_label: dict[tuple[int, ...], int] = {}
    label_to_state: list[tuple[int, ...]] = []
    labels: list[int] = []

    for row in arr:
        key = canonicalize_assignment_state(row.tolist(), relabel=relabel)

        if key not in state_to_label:
            state_to_label[key] = len(state_to_label)
            label_to_state.append(key)

        labels.append(state_to_label[key])

    return np.asarray(labels, dtype=int), label_to_state


def exact_state_labels_from_chain_states(
    chain_states: Mapping[str, Any],
    *,
    relabel: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    """
    Pool chain state matrices and assign exact-state labels.

    Parameters
    ----------
    chain_states:
        Mapping from chain id to matrix of shape `(n_maps, n_nodes)`.

    Returns
    -------
    labels:
        Exact-state labels for all pooled states.

    chain_ids:
        Chain id for each pooled state.

    label_to_state:
        Canonical state key for each label.
    """
    if not chain_states:
        raise PACEError("no chain states were provided")

    matrices: list[np.ndarray] = []
    pooled_chain_ids: list[str] = []

    n_nodes: int | None = None

    for chain_id, matrix in chain_states.items():
        arr = np.asarray(matrix)

        if arr.ndim != 2:
            raise PACEError(
                f"chain {chain_id!r} must have shape (n_maps, n_nodes); "
                f"got shape {arr.shape}"
            )

        if arr.shape[0] == 0:
            raise PACEError(f"chain {chain_id!r} has no states")

        if n_nodes is None:
            n_nodes = arr.shape[1]
        elif arr.shape[1] != n_nodes:
            raise PACEError(
                f"all chains must have the same number of nodes; "
                f"chain {chain_id!r} has {arr.shape[1]}, expected {n_nodes}"
            )

        matrices.append(arr)
        pooled_chain_ids.extend([str(chain_id)] * arr.shape[0])

    pooled_states = np.vstack(matrices)

    labels, label_to_state = exact_state_labels_from_matrix(
        pooled_states,
        relabel=relabel,
    )

    return labels, np.asarray(pooled_chain_ids, dtype=object), label_to_state


def pace_exact_from_chain_states(
    chain_states: Mapping[str, Any],
    *,
    relabel: bool = True,
    threshold: float | None = 0.1,
    apply_correction: bool = True,
    n_bootstrap: int = 0,
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
    bootstrap_mode: PACEBootstrapMode = "pooled",
) -> PACEResult:
    """
    Run exact-state PACE on chain state matrices.

    This is strict: two plans are in the same cell only if their full assignment
    vectors match after optional district-label canonicalization.

    For redistricting chains with few exact repeated plans, Wasserplan/medoid
    clustering will often be more useful; this exact version is still valuable
    for testing and small examples.
    """
    labels, chain_ids, _ = exact_state_labels_from_chain_states(
        chain_states,
        relabel=relabel,
    )

    return pace_from_labels(
        labels,
        chain_ids,
        threshold=threshold,
        apply_correction=apply_correction,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
        bootstrap_mode=bootstrap_mode,
    )