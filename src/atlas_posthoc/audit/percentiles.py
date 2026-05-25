from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from atlas_posthoc.diagnostics.rhat import rank_interpretation


PercentileMethod = Literal["weak", "strict", "midrank"]
RankType = Literal["vote_share", "compactness", "generic"]


class PercentileAuditError(Exception):
    """Base exception for enacted/reference percentile comparisons."""


@dataclass
class ReferencePercentileRow:
    rank: int
    interpretation: str
    reference_value: float
    percentile: float
    lower_tail_prob: float
    upper_tail_prob: float
    two_sided_p: float
    ensemble_mean: float
    ensemble_std: float
    ensemble_min: float
    ensemble_max: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReferencePercentileResult:
    rows: list[ReferencePercentileRow]
    n_ranks: int
    n_chains: int
    n_total_samples_per_rank: int
    method: PercentileMethod
    rank_type: RankType

    @property
    def percentiles(self) -> list[float]:
        return [row.percentile for row in self.rows]

    @property
    def two_sided_p_values(self) -> list[float]:
        return [row.two_sided_p for row in self.rows]

    @property
    def min_two_sided_p(self) -> float:
        return min(self.two_sided_p_values) if self.rows else float("nan")

    @property
    def max_reference_extremeness(self) -> float:
        """
        Larger means more extreme relative to the ensemble.

        Defined as:
            max_rank |percentile - 0.5|
        """
        return max(abs(p - 0.5) for p in self.percentiles) if self.rows else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [row.to_dict() for row in self.rows],
            "n_ranks": self.n_ranks,
            "n_chains": self.n_chains,
            "n_total_samples_per_rank": self.n_total_samples_per_rank,
            "method": self.method,
            "rank_type": self.rank_type,
            "percentiles": self.percentiles,
            "two_sided_p_values": self.two_sided_p_values,
            "min_two_sided_p": self.min_two_sided_p,
            "max_reference_extremeness": self.max_reference_extremeness,
        }


def _normalize_runs(
    runs: Sequence[Any] | Mapping[str, Any],
) -> tuple[list[str], list[np.ndarray]]:
    if isinstance(runs, Mapping):
        chain_ids = [str(k) for k in runs.keys()]
        arrays = [np.asarray(v, dtype=float) for v in runs.values()]
    else:
        arrays = [np.asarray(v, dtype=float) for v in runs]
        chain_ids = [f"chain_{i}" for i in range(len(arrays))]

    if not arrays:
        raise PercentileAuditError("no ensemble runs were provided")

    n_ranks = arrays[0].shape[1]

    for chain_id, arr in zip(chain_ids, arrays):
        if arr.ndim != 2:
            raise PercentileAuditError(
                f"run {chain_id!r} must have shape (n_maps, n_ranks); "
                f"got shape {arr.shape}"
            )

        if arr.shape[1] != n_ranks:
            raise PercentileAuditError(
                f"all runs must have the same number of ranks; "
                f"run {chain_id!r} has {arr.shape[1]}, expected {n_ranks}"
            )

    return chain_ids, arrays


def empirical_tail_probabilities(
    samples: Sequence[float],
    reference_value: float,
    *,
    method: PercentileMethod = "midrank",
) -> tuple[float, float, float]:
    """
    Compute empirical lower-tail, upper-tail, and two-sided probabilities.

    Methods
    -------
    weak:
        lower = P(X <= x_ref), upper = P(X >= x_ref)

    strict:
        lower = P(X < x_ref), upper = P(X > x_ref)

    midrank:
        lower = P(X < x_ref) + 0.5 P(X = x_ref)
        upper = P(X > x_ref) + 0.5 P(X = x_ref)

    For continuous observables, ties are rare and these methods are usually
    nearly identical. `midrank` is the safest default for discrete/tied values.
    """
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise PercentileAuditError("cannot compute percentile from empty samples")

    if not np.isfinite(reference_value):
        raise PercentileAuditError(
            f"reference value must be finite; got {reference_value!r}"
        )

    less = float(np.mean(x < reference_value))
    equal = float(np.mean(x == reference_value))
    greater = float(np.mean(x > reference_value))

    if method == "weak":
        lower = less + equal
        upper = greater + equal
    elif method == "strict":
        lower = less
        upper = greater
    elif method == "midrank":
        lower = less + 0.5 * equal
        upper = greater + 0.5 * equal
    else:
        raise ValueError(f"unknown percentile method: {method!r}")

    two_sided = min(1.0, 2.0 * min(lower, upper))

    return float(lower), float(upper), float(two_sided)


def empirical_percentile(
    samples: Sequence[float],
    reference_value: float,
    *,
    method: PercentileMethod = "midrank",
) -> float:
    """
    Empirical percentile of a reference value relative to ensemble samples.

    Returns a value in [0, 1].
    """
    lower, _, _ = empirical_tail_probabilities(
        samples,
        reference_value,
        method=method,
    )
    return lower


def pooled_rank_samples(
    runs: Sequence[Any] | Mapping[str, Any],
) -> np.ndarray:
    """
    Pool chains into one matrix of shape:

        (total_maps_across_chains, n_ranks)

    This is the main ensemble representation for enacted/reference percentile
    comparisons.
    """
    _, arrays = _normalize_runs(runs)

    return np.vstack(arrays)


def compare_reference_to_ensemble(
    runs: Sequence[Any] | Mapping[str, Any],
    reference_values: Sequence[float],
    *,
    method: PercentileMethod = "midrank",
    rank_type: RankType = "generic",
    compactness_lower_is_better: bool = True,
) -> ReferencePercentileResult:
    """
    Compare an ordered reference/enacted vector to an ensemble of ordered vectors.

    Parameters
    ----------
    runs:
        Chain matrices, each with shape (n_maps, n_ranks), or a mapping from
        chain id to such matrices.

    reference_values:
        Ordered reference vector with length n_ranks. For example:
          - enacted ordered vote-share vector;
          - enacted ordered compactness vector.

    Returns
    -------
    ReferencePercentileResult
        Rank-wise empirical percentiles and two-sided tail probabilities.

    Notes
    -----
    This function pools chains. It assumes convergence diagnostics have already
    been checked before interpreting the pooled ensemble substantively.
    """
    chain_ids, arrays = _normalize_runs(runs)
    pooled = np.vstack(arrays)

    reference = np.asarray(reference_values, dtype=float)

    if reference.ndim != 1:
        raise PercentileAuditError(
            f"reference_values must be one-dimensional; got shape {reference.shape}"
        )

    n_ranks = pooled.shape[1]

    if reference.size != n_ranks:
        raise PercentileAuditError(
            f"reference vector has length {reference.size}, expected {n_ranks}"
        )

    rows: list[ReferencePercentileRow] = []

    for rank_idx in range(n_ranks):
        rank = rank_idx + 1
        samples = pooled[:, rank_idx]
        samples = samples[np.isfinite(samples)]

        if samples.size == 0:
            raise PercentileAuditError(
                f"rank {rank} has no finite ensemble samples"
            )

        ref_value = float(reference[rank_idx])

        lower, upper, two_sided = empirical_tail_probabilities(
            samples,
            ref_value,
            method=method,
        )

        rows.append(
            ReferencePercentileRow(
                rank=rank,
                interpretation=rank_interpretation(
                    rank,
                    n_ranks,
                    rank_type=rank_type,
                    compactness_lower_is_better=compactness_lower_is_better,
                ),
                reference_value=ref_value,
                percentile=lower,
                lower_tail_prob=lower,
                upper_tail_prob=upper,
                two_sided_p=two_sided,
                ensemble_mean=float(np.mean(samples)),
                ensemble_std=float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0,
                ensemble_min=float(np.min(samples)),
                ensemble_max=float(np.max(samples)),
                n_samples=int(samples.size),
            )
        )

    return ReferencePercentileResult(
        rows=rows,
        n_ranks=n_ranks,
        n_chains=len(chain_ids),
        n_total_samples_per_rank=int(pooled.shape[0]),
        method=method,
        rank_type=rank_type,
    )


def compare_vote_share_reference_to_ensemble(
    runs: Sequence[Any] | Mapping[str, Any],
    reference_values: Sequence[float],
    *,
    method: PercentileMethod = "midrank",
) -> ReferencePercentileResult:
    """
    Convenience wrapper for enacted/reference vote-share percentiles.
    """
    return compare_reference_to_ensemble(
        runs,
        reference_values,
        method=method,
        rank_type="vote_share",
    )


def compare_compactness_reference_to_ensemble(
    runs: Sequence[Any] | Mapping[str, Any],
    reference_values: Sequence[float],
    *,
    method: PercentileMethod = "midrank",
    compactness_lower_is_better: bool = True,
) -> ReferencePercentileResult:
    """
    Convenience wrapper for enacted/reference compactness percentiles.
    """
    return compare_reference_to_ensemble(
        runs,
        reference_values,
        method=method,
        rank_type="compactness",
        compactness_lower_is_better=compactness_lower_is_better,
    )


# Friendly aliases for common project language.
compare_enacted_to_ensemble = compare_reference_to_ensemble
compare_enacted_vote_share_to_ensemble = compare_vote_share_reference_to_ensemble
compare_enacted_compactness_to_ensemble = compare_compactness_reference_to_ensemble