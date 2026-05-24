from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np


class TotalVariationError(Exception):
    """Base exception for total-variation diagnostics."""


@dataclass
class PairwiseTVResult:
    chain_a: str
    chain_b: str
    tv_by_rank: list[float]
    average_tv: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TVSnapshotResult:
    snapshot: int
    proposals: int
    max_average_tv: float
    pairwise: list[PairwiseTVResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot": self.snapshot,
            "proposals": self.proposals,
            "max_average_tv": self.max_average_tv,
            "pairwise": [x.to_dict() for x in self.pairwise],
        }


@dataclass
class TVConvergenceResult:
    snapshots: list[int]
    proposals: list[int]
    max_tvs: list[float]
    details: list[TVSnapshotResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshots": self.snapshots,
            "proposals": self.proposals,
            "max_tvs": self.max_tvs,
            "details": [x.to_dict() for x in self.details],
        }


def compute_tv(
    dist1: Sequence[float],
    dist2: Sequence[float],
    bins: Sequence[float],
    *,
    drop_nan: bool = True,
) -> float:
    """
    Compute histogram-based total variation distance between two samples.

    This is the package version of the helper used in the original diagnostic
    notebook.

    TV(p, q) = 0.5 * sum_b |p_b - q_b|

    Parameters
    ----------
    dist1, dist2:
        One-dimensional samples from two chains for the same ordered rank.

    bins:
        Shared histogram bin edges.

    drop_nan:
        If True, non-finite values are dropped before histogramming.

    Returns
    -------
    float
        Total variation distance between the empirical histograms.
    """
    x = np.asarray(dist1, dtype=float)
    y = np.asarray(dist2, dtype=float)
    bin_edges = np.asarray(bins, dtype=float)

    if bin_edges.ndim != 1 or bin_edges.size < 2:
        raise TotalVariationError("bins must be a one-dimensional array of length at least 2")

    if drop_nan:
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]

    if x.size == 0 or y.size == 0:
        raise TotalVariationError("cannot compute TV distance on empty samples")

    h1, _ = np.histogram(x, bins=bin_edges, density=False)
    h2, _ = np.histogram(y, bins=bin_edges, density=False)

    n1 = h1.sum()
    n2 = h2.sum()

    if n1 == 0 or n2 == 0:
        raise TotalVariationError(
            "all observations fell outside the histogram bins; "
            "use wider bins or check the observable scale"
        )

    p = h1 / n1
    q = h2 / n2

    return float(0.5 * np.sum(np.abs(p - q)))


def _as_2d_array(x: Any, *, name: str = "run") -> np.ndarray:
    arr = np.asarray(x, dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise TotalVariationError(
            f"{name} must be a 2D array of shape (n_maps, n_ranks); got shape {arr.shape}"
        )

    if arr.shape[0] == 0:
        raise TotalVariationError(f"{name} has zero maps")

    if arr.shape[1] == 0:
        raise TotalVariationError(f"{name} has zero ranks/districts")

    return arr


def _normalize_runs(
    runs: Sequence[Any] | Mapping[str, Any],
    *,
    chain_ids: Sequence[str] | None = None,
    trim_to_min_length: bool = True,
) -> tuple[list[str], list[np.ndarray]]:
    """
    Normalize list/dict chain inputs into chain ids and 2D arrays.
    """
    if isinstance(runs, Mapping):
        ids = [str(k) for k in runs.keys()]
        arrays = [_as_2d_array(v, name=f"run {k}") for k, v in runs.items()]
    else:
        arrays = [_as_2d_array(v, name=f"run {i}") for i, v in enumerate(runs)]

        if chain_ids is None:
            ids = [f"chain_{i}" for i in range(len(arrays))]
        else:
            ids = [str(x) for x in chain_ids]

    if len(arrays) < 2:
        raise TotalVariationError("at least two chains are required")

    if len(ids) != len(arrays):
        raise TotalVariationError("chain_ids length must match number of chains")

    n_ranks = arrays[0].shape[1]
    for i, arr in enumerate(arrays):
        if arr.shape[1] != n_ranks:
            raise TotalVariationError(
                f"all chains must have the same number of ranks/districts; "
                f"chain {ids[i]!r} has {arr.shape[1]}, expected {n_ranks}"
            )

    if trim_to_min_length:
        min_len = min(arr.shape[0] for arr in arrays)
        arrays = [arr[:min_len, :] for arr in arrays]
    else:
        n_maps = arrays[0].shape[0]
        for i, arr in enumerate(arrays):
            if arr.shape[0] != n_maps:
                raise TotalVariationError(
                    f"all chains must have the same number of maps; "
                    f"chain {ids[i]!r} has {arr.shape[0]}, expected {n_maps}"
                )

    return ids, arrays


def compute_rankwise_pairwise_tv(
    run_a: Any,
    run_b: Any,
    bins: Sequence[float],
    *,
    chain_a: str = "chain_a",
    chain_b: str = "chain_b",
) -> PairwiseTVResult:
    """
    Compute TV distance rank-by-rank between two chains.

    Each run should have shape:

        (n_maps, n_ordered_ranks)

    For vote share, the columns are ordered district ranks from least Democratic
    to most Democratic. For compactness, the columns are ordered compactness ranks.
    """
    a = _as_2d_array(run_a, name=chain_a)
    b = _as_2d_array(run_b, name=chain_b)

    if a.shape[1] != b.shape[1]:
        raise TotalVariationError(
            f"chains have different numbers of ranks: {a.shape[1]} vs {b.shape[1]}"
        )

    n = min(a.shape[0], b.shape[0])
    a = a[:n, :]
    b = b[:n, :]

    tv_by_rank = [
        compute_tv(a[:, rank], b[:, rank], bins)
        for rank in range(a.shape[1])
    ]

    return PairwiseTVResult(
        chain_a=chain_a,
        chain_b=chain_b,
        tv_by_rank=tv_by_rank,
        average_tv=float(np.mean(tv_by_rank)),
    )


def max_average_pairwise_tv(
    runs_list: Sequence[Any] | Mapping[str, Any],
    bins: Sequence[float],
    *,
    chain_ids: Sequence[str] | None = None,
    trim_to_min_length: bool = True,
) -> tuple[float, list[PairwiseTVResult]]:
    """
    Compute the CT-paper-style max average pairwise TV across chains.

    For each pair of chains:
      1. compute TV for each ordered rank;
      2. average TV across ranks.

    Then take the maximum average TV over all chain pairs.
    """
    ids, arrays = _normalize_runs(
        runs_list,
        chain_ids=chain_ids,
        trim_to_min_length=trim_to_min_length,
    )

    pairwise_results: list[PairwiseTVResult] = []

    for i, j in combinations(range(len(arrays)), 2):
        result = compute_rankwise_pairwise_tv(
            arrays[i],
            arrays[j],
            bins,
            chain_a=ids[i],
            chain_b=ids[j],
        )
        pairwise_results.append(result)

    max_tv = max(x.average_tv for x in pairwise_results)

    return float(max_tv), pairwise_results


def get_tv_convergence(
    runs_list: Sequence[Any] | Mapping[str, Any],
    bins: Sequence[float],
    *,
    output_freq: int = 10000,
    snapshot_step: int = 10,
    min_snapshot: int = 10,
    snapshots: Sequence[int] | None = None,
    chain_ids: Sequence[str] | None = None,
    trim_to_min_length: bool = True,
) -> TVConvergenceResult:
    """
    Compute max average pairwise TV as a function of saved-map prefix length.

    This is the package version of the original helper:

        get_tv_convergence(runs_list, bins, output_freq=10000)

    Parameters
    ----------
    runs_list:
        List or dict of chain matrices. Each matrix must have shape:

            (n_saved_maps, n_ordered_ranks)

    bins:
        Shared histogram bins.

    output_freq:
        Number of MCMC proposals between saved maps.

    snapshot_step:
        Evaluate convergence every `snapshot_step` saved maps.

    min_snapshot:
        First saved-map prefix length to evaluate.

    snapshots:
        Optional explicit list of saved-map prefix lengths.

    Returns
    -------
    TVConvergenceResult
        Contains proposals, max TV values, and pairwise details.
    """
    if output_freq <= 0:
        raise ValueError("output_freq must be positive")

    if snapshot_step <= 0:
        raise ValueError("snapshot_step must be positive")

    ids, arrays = _normalize_runs(
        runs_list,
        chain_ids=chain_ids,
        trim_to_min_length=trim_to_min_length,
    )

    n_saved_maps = min(arr.shape[0] for arr in arrays)

    if snapshots is None:
        if min_snapshot <= 0:
            raise ValueError("min_snapshot must be positive")

        snapshot_arr = np.arange(min_snapshot, n_saved_maps + 1, snapshot_step)

        if snapshot_arr.size == 0 or snapshot_arr[-1] != n_saved_maps:
            snapshot_arr = np.append(snapshot_arr, n_saved_maps)
    else:
        snapshot_arr = np.asarray(snapshots, dtype=int)

    snapshot_arr = snapshot_arr[
        (snapshot_arr > 0) & (snapshot_arr <= n_saved_maps)
    ]

    if snapshot_arr.size == 0:
        raise TotalVariationError("no valid snapshots to evaluate")

    details: list[TVSnapshotResult] = []
    max_tvs: list[float] = []
    proposal_counts: list[int] = []

    for t in snapshot_arr:
        prefix_runs = [arr[:t, :] for arr in arrays]

        max_tv, pairwise = max_average_pairwise_tv(
            prefix_runs,
            bins,
            chain_ids=ids,
            trim_to_min_length=False,
        )

        proposals = int(t * output_freq)

        details.append(
            TVSnapshotResult(
                snapshot=int(t),
                proposals=proposals,
                max_average_tv=max_tv,
                pairwise=pairwise,
            )
        )

        max_tvs.append(max_tv)
        proposal_counts.append(proposals)

    return TVConvergenceResult(
        snapshots=[int(x) for x in snapshot_arr],
        proposals=proposal_counts,
        max_tvs=max_tvs,
        details=details,
    )


def make_histogram_bins(
    *,
    lower: float,
    upper: float,
    bin_width: float,
    anchor: float | None = None,
) -> np.ndarray:
    """
    Make regular histogram bins.

    If `anchor` is provided, one bin edge is aligned to that value. This is useful
    for vote-share bins centered around 50%, matching the CT-paper convention.
    """
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")

    if lower >= upper:
        raise ValueError("lower must be smaller than upper")

    if anchor is None:
        start = lower
    else:
        k = np.floor((lower - anchor) / bin_width)
        start = anchor + k * bin_width

    n_steps = int(np.ceil((upper - start) / bin_width))
    edges = start + bin_width * np.arange(n_steps + 1)

    if edges[-1] < upper:
        edges = np.append(edges, edges[-1] + bin_width)

    return edges


def make_vote_share_bins(
    *,
    bin_width: float = 0.002,
    lower: float = 0.0,
    upper: float = 1.0,
    center: float = 0.5,
) -> np.ndarray:
    """
    Make vote-share bins.

    Default bin width is 0.2 percentage points, with a bin edge aligned at 50%.
    """
    return make_histogram_bins(
        lower=lower,
        upper=upper,
        bin_width=bin_width,
        anchor=center,
    )


def make_compactness_bins(
    runs_list: Sequence[Any] | Mapping[str, Any],
    *,
    bin_width: float = 0.5,
    anchor: float = 0.0,
) -> np.ndarray:
    """
    Make compactness/isoperimetric-score bins from observed chain values.

    Default bin width is 0.5 with a bin edge aligned at 0, matching the CT-paper
    convention for isoperimetric ratios.
    """
    _, arrays = _normalize_runs(runs_list)

    values = np.concatenate([arr.ravel() for arr in arrays])
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise TotalVariationError("cannot build compactness bins from empty/non-finite values")

    lower = float(values.min())
    upper = float(values.max())

    if lower == upper:
        lower -= bin_width
        upper += bin_width

    return make_histogram_bins(
        lower=lower,
        upper=upper,
        bin_width=bin_width,
        anchor=anchor,
    )