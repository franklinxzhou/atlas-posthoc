from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np


IsoperimetricConvention = Literal[
    "p2_over_area",
    "normalized_inverse",
    "already_polsby_popper",
]
InvalidPolicy = Literal["nan", "raise"]


class PolsbyPopperError(Exception):
    """Base exception for Polsby-Popper compactness helpers."""


@dataclass
class PolsbyPopperSummary:
    average: float
    minimum: float
    maximum: float
    n_scores: int
    n_finite: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "average": self.average,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "n_scores": self.n_scores,
            "n_finite": self.n_finite,
        }


@dataclass
class PolsbyPopperTrace:
    chain_id: str
    proposals: list[int]
    average_polsby_popper: list[float]
    minimum_polsby_popper: list[float]
    maximum_polsby_popper: list[float]
    n_maps: int
    convention: IsoperimetricConvention

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "proposals": self.proposals,
            "average_polsby_popper": self.average_polsby_popper,
            "minimum_polsby_popper": self.minimum_polsby_popper,
            "maximum_polsby_popper": self.maximum_polsby_popper,
            "n_maps": self.n_maps,
            "convention": self.convention,
        }


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _handle_invalid(
    out: np.ndarray,
    *,
    invalid_policy: InvalidPolicy,
    message: str,
) -> np.ndarray:
    if np.any(~np.isfinite(out)):
        if invalid_policy == "raise":
            raise PolsbyPopperError(message)

        if invalid_policy != "nan":
            raise ValueError(f"unknown invalid_policy: {invalid_policy!r}")

    return out


def polsby_popper_from_area_perimeter(
    area: Any,
    perimeter: Any,
    *,
    invalid_policy: InvalidPolicy = "nan",
    clip: bool = False,
) -> np.ndarray:
    """
    Compute Polsby-Popper compactness from area and perimeter.

        PP = 4π * area / perimeter²

    Larger values indicate more compact districts. A perfect circle has score 1.
    """
    area_arr = _as_float_array(area)
    perimeter_arr = _as_float_array(perimeter)

    with np.errstate(divide="ignore", invalid="ignore"):
        pp = 4.0 * np.pi * area_arr / (perimeter_arr**2)

    pp = np.where((area_arr > 0) & (perimeter_arr > 0), pp, np.nan)

    if clip:
        pp = np.clip(pp, 0.0, 1.0)

    return _handle_invalid(
        pp,
        invalid_policy=invalid_policy,
        message="invalid area/perimeter values produced non-finite Polsby-Popper scores",
    )


def isoperimetric_to_polsby_popper(
    scores: Any,
    *,
    convention: IsoperimetricConvention = "p2_over_area",
    invalid_policy: InvalidPolicy = "nan",
    clip: bool = False,
) -> np.ndarray:
    """
    Convert isoperimetric-style compactness scores to Polsby-Popper scores.

    Conventions
    -----------
    p2_over_area:
        Input score is perimeter² / area.
        Then PP = 4π / score.

    normalized_inverse:
        Input score is perimeter² / (4π area).
        Then PP = 1 / score.

    already_polsby_popper:
        Input is already Polsby-Popper.
        Returned unchanged except for validation/clipping.
    """
    x = _as_float_array(scores)

    with np.errstate(divide="ignore", invalid="ignore"):
        if convention == "p2_over_area":
            pp = 4.0 * np.pi / x
        elif convention == "normalized_inverse":
            pp = 1.0 / x
        elif convention == "already_polsby_popper":
            pp = x.copy()
        else:
            raise ValueError(f"unknown isoperimetric convention: {convention!r}")

    pp = np.where(x > 0, pp, np.nan)

    if convention == "already_polsby_popper":
        pp = np.where(x >= 0, pp, np.nan)

    if clip:
        pp = np.clip(pp, 0.0, 1.0)

    return _handle_invalid(
        pp,
        invalid_policy=invalid_policy,
        message="invalid compactness scores produced non-finite Polsby-Popper scores",
    )


def polsby_popper_to_isoperimetric(
    pp_scores: Any,
    *,
    convention: Literal["p2_over_area", "normalized_inverse"] = "p2_over_area",
    invalid_policy: InvalidPolicy = "nan",
) -> np.ndarray:
    """
    Convert Polsby-Popper scores back to an isoperimetric-style score.
    """
    pp = _as_float_array(pp_scores)

    with np.errstate(divide="ignore", invalid="ignore"):
        if convention == "p2_over_area":
            out = 4.0 * np.pi / pp
        elif convention == "normalized_inverse":
            out = 1.0 / pp
        else:
            raise ValueError(f"unknown convention: {convention!r}")

    out = np.where(pp > 0, out, np.nan)

    return _handle_invalid(
        out,
        invalid_policy=invalid_policy,
        message="invalid Polsby-Popper scores produced non-finite isoperimetric scores",
    )


def summarize_polsby_popper_scores(
    pp_scores: Any,
    *,
    ignore_nan: bool = True,
) -> PolsbyPopperSummary:
    """
    Summarize one plan's district-level Polsby-Popper scores.
    """
    x = _as_float_array(pp_scores).ravel()

    if x.size == 0:
        raise PolsbyPopperError("cannot summarize an empty Polsby-Popper vector")

    finite = x[np.isfinite(x)] if ignore_nan else x

    if finite.size == 0:
        raise PolsbyPopperError("no finite Polsby-Popper scores to summarize")

    if not ignore_nan and np.any(~np.isfinite(finite)):
        raise PolsbyPopperError("non-finite Polsby-Popper scores found")

    return PolsbyPopperSummary(
        average=float(np.mean(finite)),
        minimum=float(np.min(finite)),
        maximum=float(np.max(finite)),
        n_scores=int(x.size),
        n_finite=int(np.sum(np.isfinite(x))),
    )


def summarize_polsby_popper_matrix(
    matrix: Any,
    *,
    convention: IsoperimetricConvention = "p2_over_area",
    input_is_polsby_popper: bool = False,
    invalid_policy: InvalidPolicy = "nan",
) -> dict[str, np.ndarray]:
    """
    Summarize a matrix of compactness scores row-by-row.

    Expected shape:

        (n_maps, n_districts)

    Returns arrays of length n_maps:
      - average_polsby_popper
      - minimum_polsby_popper
      - maximum_polsby_popper
    """
    arr = _as_float_array(matrix)

    if arr.ndim != 2:
        raise PolsbyPopperError(
            f"compactness matrix must have shape (n_maps, n_districts); got {arr.shape}"
        )

    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise PolsbyPopperError("compactness matrix must be non-empty")

    if input_is_polsby_popper:
        pp = isoperimetric_to_polsby_popper(
            arr,
            convention="already_polsby_popper",
            invalid_policy=invalid_policy,
        )
    else:
        pp = isoperimetric_to_polsby_popper(
            arr,
            convention=convention,
            invalid_policy=invalid_policy,
        )

    return {
        "average_polsby_popper": np.nanmean(pp, axis=1),
        "minimum_polsby_popper": np.nanmin(pp, axis=1),
        "maximum_polsby_popper": np.nanmax(pp, axis=1),
    }


def _normalize_runs(
    runs: Mapping[str, Any] | Sequence[Any],
    *,
    chain_ids: Sequence[str] | None = None,
) -> tuple[list[str], list[np.ndarray]]:
    if isinstance(runs, Mapping):
        ids = [str(k) for k in runs.keys()]
        arrays = [np.asarray(v, dtype=float) for v in runs.values()]
    else:
        arrays = [np.asarray(v, dtype=float) for v in runs]
        if chain_ids is None:
            ids = [f"chain_{i}" for i in range(len(arrays))]
        else:
            ids = [str(x) for x in chain_ids]

    if not arrays:
        raise PolsbyPopperError("no runs were provided")

    if len(ids) != len(arrays):
        raise PolsbyPopperError("chain_ids length must match number of runs")

    n_districts = arrays[0].shape[1]

    for chain_id, arr in zip(ids, arrays):
        if arr.ndim != 2:
            raise PolsbyPopperError(
                f"run {chain_id!r} must have shape (n_maps, n_districts); got {arr.shape}"
            )

        if arr.shape[1] != n_districts:
            raise PolsbyPopperError(
                f"all runs must have the same number of districts; "
                f"run {chain_id!r} has {arr.shape[1]}, expected {n_districts}"
            )

    return ids, arrays


def make_polsby_popper_traces(
    runs: Mapping[str, Any] | Sequence[Any],
    *,
    chain_ids: Sequence[str] | None = None,
    output_frequency: int = 1,
    start_proposal: int = 0,
    convention: IsoperimetricConvention = "p2_over_area",
    input_is_polsby_popper: bool = False,
    invalid_policy: InvalidPolicy = "nan",
) -> list[PolsbyPopperTrace]:
    """
    Convert chain compactness matrices into average/min/max Polsby-Popper traces.

    This is the observable layer behind the Polsby-Popper compactness trace plot.
    """
    if output_frequency <= 0:
        raise ValueError("output_frequency must be positive")

    ids, arrays = _normalize_runs(runs, chain_ids=chain_ids)

    traces: list[PolsbyPopperTrace] = []

    for chain_id, arr in zip(ids, arrays):
        summary = summarize_polsby_popper_matrix(
            arr,
            convention=convention,
            input_is_polsby_popper=input_is_polsby_popper,
            invalid_policy=invalid_policy,
        )

        proposals = [
            int(start_proposal + i * output_frequency)
            for i in range(arr.shape[0])
        ]

        traces.append(
            PolsbyPopperTrace(
                chain_id=chain_id,
                proposals=proposals,
                average_polsby_popper=summary["average_polsby_popper"].tolist(),
                minimum_polsby_popper=summary["minimum_polsby_popper"].tolist(),
                maximum_polsby_popper=summary["maximum_polsby_popper"].tolist(),
                n_maps=int(arr.shape[0]),
                convention="already_polsby_popper" if input_is_polsby_popper else convention,
            )
        )

    return traces