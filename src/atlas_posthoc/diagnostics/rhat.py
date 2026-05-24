from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np


class RHatError(Exception):
    """Base exception for Gelman-Rubin / R-hat diagnostics."""


@dataclass
class RHatRankRow:
    rank: int
    interpretation: str
    classic_r_hat: float
    rank_normalized_split_r_hat: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RHatSummary:
    rows: list[dict[str, Any]]
    min_draws: int
    num_ranks: int
    max_classic_r_hat: float | None
    max_rank_normalized_split_r_hat: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """
    Pure-numpy average ranks, equivalent to scipy.stats.rankdata(..., method='average').

    Ranks are 1-indexed.
    """
    values = np.asarray(values, dtype=float)
    n = values.size

    if n == 0:
        return np.asarray([], dtype=float)

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]

    ranks_sorted = np.empty(n, dtype=float)

    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1

        # 1-indexed average rank for the tie block [start, end)
        avg_rank = 0.5 * ((start + 1) + end)
        ranks_sorted[start:end] = avg_rank
        start = end

    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def classic_r_hat(chains: Sequence[Sequence[float]]) -> float:
    """
    Classic Gelman-Rubin R-hat.

    Parameters
    ----------
    chains:
        Array-like object with shape (num_chains, num_draws).

    Notes
    -----
    This follows the helper used in the earlier Ohio MFR diagnostics.
    """
    chains = np.asarray(chains, dtype=float)

    if chains.ndim != 2:
        raise RHatError("chains must have shape (num_chains, num_draws)")

    m, n = chains.shape

    if m < 2 or n < 2:
        return float("nan")

    chain_means = np.mean(chains, axis=1)
    grand_mean = np.mean(chain_means)

    b = n * np.sum((chain_means - grand_mean) ** 2) / (m - 1)
    w = np.mean(np.var(chains, axis=1, ddof=1))

    if not np.isfinite(w) or w <= 0:
        return 1.0 if np.allclose(chains, chains[0, 0], equal_nan=False) else float("nan")

    var_plus = ((n - 1) / n) * w + b / n
    return float(np.sqrt(var_plus / w))


def rank_normalized_split_r_hat(chains: Sequence[Sequence[float]]) -> float:
    """
    Rank-normalized split R-hat.

    Splits each chain in half, rank-normalizes the pooled draws, then applies
    classic R-hat to the transformed split chains.

    This follows the structure used in the earlier Ohio MFR diagnostics, but avoids
    requiring scipy by using a small internal average-rank implementation and
    Python's NormalDist inverse CDF.
    """
    chains = np.asarray(chains, dtype=float)

    if chains.ndim != 2:
        raise RHatError("chains must have shape (num_chains, num_draws)")

    m, n = chains.shape
    half_n = n // 2

    if m < 2 or half_n < 2:
        return float("nan")

    split_chains = []

    for chain_idx in range(m):
        split_chains.append(chains[chain_idx, :half_n])
        split_chains.append(chains[chain_idx, half_n : 2 * half_n])

    split_chains = np.asarray(split_chains, dtype=float)

    flat = split_chains.ravel()

    if not np.all(np.isfinite(flat)):
        return float("nan")

    ranks = _average_ranks(flat)
    total = len(flat)

    probs = (ranks - 0.375) / (total + 0.25)
    normal = NormalDist()
    z = np.asarray([normal.inv_cdf(float(p)) for p in probs], dtype=float)

    z_chains = z.reshape(split_chains.shape)

    return classic_r_hat(z_chains)


def _normalize_runs(
    runs_list: Sequence[Any] | Mapping[str, Any],
) -> tuple[list[str], list[np.ndarray]]:
    """
    Normalize list/dict chain inputs into chain ids and 2D arrays.
    """
    if isinstance(runs_list, Mapping):
        chain_ids = [str(k) for k in runs_list.keys()]
        runs = [np.asarray(v, dtype=float) for v in runs_list.values()]
    else:
        runs = [np.asarray(run, dtype=float) for run in runs_list]
        chain_ids = [f"chain_{i}" for i in range(len(runs))]

    if len(runs) < 2:
        raise RHatError("Need at least two chains/runs for Gelman-Rubin diagnostics.")

    for chain_id, run in zip(chain_ids, runs):
        if run.ndim != 2:
            raise RHatError(
                f"run {chain_id!r} must have shape (num_draws, num_ranks); "
                f"got shape {run.shape}"
            )

    return chain_ids, runs


def rank_interpretation(rank: int, num_ranks: int) -> str:
    """
    Human-readable interpretation for ordered election-vector ranks.

    Rank 1 is the most Republican district. The final rank is the least
    Republican / most Democratic district.
    """
    if rank == 1:
        return "most Republican"

    if rank == num_ranks:
        return "least Republican"

    suffix = "th"
    if rank % 10 == 1 and rank % 100 != 11:
        suffix = "st"
    elif rank % 10 == 2 and rank % 100 != 12:
        suffix = "nd"
    elif rank % 10 == 3 and rank % 100 != 13:
        suffix = "rd"

    return f"{rank}{suffix} most Republican"


def gelman_rubin_by_rank(
    runs_list: Sequence[Any] | Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Compute both R-hat diagnostics for each rank of the sorted election vector.

    This preserves the helper-function shape from `oh_mfr_diagnostics.py`:

        rows, min_draws, num_ranks = gelman_rubin_by_rank(runs_list)

    Parameters
    ----------
    runs_list:
        Either a list of chain matrices or a dict mapping chain ids to matrices.

        Each chain matrix must have shape:

            (num_saved_maps, num_ordered_ranks)

    Returns
    -------
    rows:
        List of dictionaries with rank-wise R-hat diagnostics.

    min_draws:
        Common number of draws used after truncating chains to equal length.

    num_ranks:
        Number of ordered district ranks used.
    """
    _, runs = _normalize_runs(runs_list)

    min_draws = min(run.shape[0] for run in runs)
    num_ranks = min(run.shape[1] for run in runs)

    if min_draws < 4:
        raise RHatError("Need at least four saved maps per run for split R-hat.")

    if num_ranks == 0:
        raise RHatError("No district-rank columns were found.")

    truncated = np.asarray(
        [run[:min_draws, :num_ranks] for run in runs],
        dtype=float,
    )

    rows: list[dict[str, Any]] = []

    for rank_idx in range(num_ranks):
        chains = truncated[:, :, rank_idx]
        rank = rank_idx + 1

        rows.append(
            {
                "rank": rank,
                "interpretation": rank_interpretation(rank, num_ranks),
                "classic_r_hat": classic_r_hat(chains),
                "rank_normalized_split_r_hat": rank_normalized_split_r_hat(chains),
            }
        )

    return rows, min_draws, num_ranks


def summarize_gelman_rubin_by_rank(
    runs_list: Sequence[Any] | Mapping[str, Any],
) -> RHatSummary:
    """
    Richer wrapper around `gelman_rubin_by_rank`.

    Useful for reports and convergence gates.
    """
    rows, min_draws, num_ranks = gelman_rubin_by_rank(runs_list)

    finite_classic = [
        row["classic_r_hat"]
        for row in rows
        if np.isfinite(row["classic_r_hat"])
    ]
    finite_split = [
        row["rank_normalized_split_r_hat"]
        for row in rows
        if np.isfinite(row["rank_normalized_split_r_hat"])
    ]

    return RHatSummary(
        rows=rows,
        min_draws=min_draws,
        num_ranks=num_ranks,
        max_classic_r_hat=max(finite_classic) if finite_classic else None,
        max_rank_normalized_split_r_hat=max(finite_split) if finite_split else None,
    )