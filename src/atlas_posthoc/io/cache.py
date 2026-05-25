from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl


class CacheError(Exception):
    """Base exception for cache read/write errors."""


@dataclass
class CacheManifest:
    output_dir: Path
    files: dict[str, Path]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "files": {k: str(v) for k, v in self.files.items()},
        }


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_json_serializable(obj: Any) -> Any:
    """
    Convert common package objects into JSON-serializable objects.
    """
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return make_json_serializable(obj.to_dict())

    if is_dataclass(obj):
        return make_json_serializable(asdict(obj))

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {
            str(make_json_serializable(k)): make_json_serializable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]

    if isinstance(obj, float):
        if np.isnan(obj):
            return None
        if np.isposinf(obj):
            return "Infinity"
        if np.isneginf(obj):
            return "-Infinity"

    return obj


def write_json(obj: Any, path: str | Path, *, indent: int = 2) -> Path:
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(obj), f, indent=indent)

    return path


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def observable_runs_to_long_frame(
    runs: Mapping[str, Any],
    *,
    value_col: str,
    steps: Mapping[str, list[int]] | None = None,
    names: Mapping[str, list[str | None]] | None = None,
) -> pl.DataFrame:
    """
    Convert chain observable matrices to a long Polars DataFrame.

    Input
    -----
    runs:
        Mapping from chain id to array of shape:

            (n_maps, n_ranks)

    Output columns
    --------------
    chain_id, map_index, rank, value_col

    Optionally:
    step, name
    """
    frames: list[pl.DataFrame] = []

    for chain_id, values in runs.items():
        arr = np.asarray(values, dtype=float)

        if arr.ndim != 2:
            raise CacheError(
                f"run {chain_id!r} must be a 2D matrix; got shape {arr.shape}"
            )

        n_maps, n_ranks = arr.shape

        map_index = np.repeat(np.arange(n_maps, dtype=int), n_ranks)
        rank = np.tile(np.arange(1, n_ranks + 1, dtype=int), n_maps)

        data: dict[str, Any] = {
            "chain_id": [str(chain_id)] * (n_maps * n_ranks),
            "map_index": map_index,
            "rank": rank,
            value_col: arr.reshape(-1),
        }

        if steps is not None:
            chain_steps = steps.get(chain_id)
            if chain_steps is None:
                raise CacheError(f"missing steps for chain {chain_id!r}")
            if len(chain_steps) != n_maps:
                raise CacheError(
                    f"steps for chain {chain_id!r} have length {len(chain_steps)}, "
                    f"expected {n_maps}"
                )
            data["step"] = np.repeat(np.asarray(chain_steps, dtype=int), n_ranks)

        if names is not None:
            chain_names = names.get(chain_id)
            if chain_names is None:
                raise CacheError(f"missing names for chain {chain_id!r}")
            if len(chain_names) != n_maps:
                raise CacheError(
                    f"names for chain {chain_id!r} have length {len(chain_names)}, "
                    f"expected {n_maps}"
                )
            data["name"] = [
                x
                for x in chain_names
                for _ in range(n_ranks)
            ]

        frames.append(pl.DataFrame(data))

    if not frames:
        raise CacheError("no observable runs were provided")

    return pl.concat(frames, how="vertical")


def write_observable_runs(
    runs: Mapping[str, Any],
    path: str | Path,
    *,
    value_col: str,
    steps: Mapping[str, list[int]] | None = None,
    names: Mapping[str, list[str | None]] | None = None,
) -> Path:
    """
    Save observable matrices as long-format Parquet.
    """
    path = Path(path)
    ensure_dir(path.parent)

    df = observable_runs_to_long_frame(
        runs,
        value_col=value_col,
        steps=steps,
        names=names,
    )

    df.write_parquet(path)
    return path


def read_observable_runs(
    path: str | Path,
    *,
    value_col: str,
) -> dict[str, np.ndarray]:
    """
    Read long-format observable Parquet back into chain matrices.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from chain id to matrix of shape:

            (n_maps, n_ranks)
    """
    df = pl.read_parquet(path)

    required = {"chain_id", "map_index", "rank", value_col}
    missing = required - set(df.columns)

    if missing:
        raise CacheError(f"observable cache is missing columns: {sorted(missing)}")

    out: dict[str, np.ndarray] = {}

    chain_ids = (
        df.select("chain_id")
        .unique(maintain_order=True)
        .get_column("chain_id")
        .to_list()
    )

    for chain_id in chain_ids:
        sub = df.filter(pl.col("chain_id") == chain_id)

        map_indices = sorted(sub.get_column("map_index").unique().to_list())
        ranks = sorted(sub.get_column("rank").unique().to_list())

        map_index_to_row = {m: i for i, m in enumerate(map_indices)}
        rank_to_col = {r: j for j, r in enumerate(ranks)}

        arr = np.full((len(map_indices), len(ranks)), np.nan, dtype=float)

        seen: set[tuple[int, int]] = set()

        for row in sub.iter_rows(named=True):
            key = (row["map_index"], row["rank"])

            if key in seen:
                raise CacheError(
                    f"duplicate cache entry for chain={chain_id!r}, "
                    f"map_index={row['map_index']}, rank={row['rank']}"
                )

            seen.add(key)

            arr[
                map_index_to_row[row["map_index"]],
                rank_to_col[row["rank"]],
            ] = row[value_col]

        if len(seen) != len(map_indices) * len(ranks):
            raise CacheError(
                f"incomplete rectangular observable matrix for chain {chain_id!r}"
            )

        out[str(chain_id)] = arr

    return out


def write_chain_metadata(
    chains: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """
    Save per-map metadata from ChainObservableResult objects.

    Expected each value to have:
        - map_indices
        - steps
        - names
    """
    rows: list[dict[str, Any]] = []

    for chain_id, result in chains.items():
        map_indices = list(result.map_indices)
        steps = list(result.steps)
        names = list(result.names)

        if not (len(map_indices) == len(steps) == len(names)):
            raise CacheError(f"inconsistent metadata lengths for chain {chain_id!r}")

        for map_index, step, name in zip(map_indices, steps, names):
            rows.append(
                {
                    "chain_id": str(chain_id),
                    "map_index": int(map_index),
                    "step": int(step),
                    "name": name,
                }
            )

    path = Path(path)
    ensure_dir(path.parent)

    pl.DataFrame(rows).write_parquet(path)
    return path


def read_chain_metadata(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def extract_steps_from_observable_result(obs: Any) -> dict[str, list[int]]:
    return {
        chain_id: list(result.steps)
        for chain_id, result in obs.chains.items()
    }


def extract_names_from_observable_result(obs: Any) -> dict[str, list[str | None]]:
    return {
        chain_id: list(result.names)
        for chain_id, result in obs.chains.items()
    }


def write_pipeline_cache(
    output_dir: str | Path,
    *,
    observables: Any,
    vote_tv: Any | None = None,
    compactness_tv: Any | None = None,
    vote_rhat: Any | None = None,
    compactness_rhat: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> CacheManifest:
    """
    Save the main outputs of the Atlas post-hoc pipeline.

    Writes:
        cache/vote_runs.parquet
        cache/compactness_runs.parquet, if available
        cache/chain_metadata.parquet
        cache/chain_summaries.json
        diagnostics/vote_tv.json, if provided
        diagnostics/compactness_tv.json, if provided
        diagnostics/vote_rhat.json, if provided
        diagnostics/compactness_rhat.json, if provided
        diagnostics/extra.json, if provided
        manifest.json
    """
    output_dir = ensure_dir(output_dir)
    cache_dir = ensure_dir(output_dir / "cache")
    diagnostics_dir = ensure_dir(output_dir / "diagnostics")

    files: dict[str, Path] = {}

    steps = extract_steps_from_observable_result(observables)
    names = extract_names_from_observable_result(observables)

    files["vote_runs"] = write_observable_runs(
        observables.vote_runs,
        cache_dir / "vote_runs.parquet",
        value_col="dem_share",
        steps=steps,
        names=names,
    )

    if observables.compactness_runs:
        files["compactness_runs"] = write_observable_runs(
            observables.compactness_runs,
            cache_dir / "compactness_runs.parquet",
            value_col="compactness",
            steps=steps,
            names=names,
        )

    files["chain_metadata"] = write_chain_metadata(
        observables.chains,
        cache_dir / "chain_metadata.parquet",
    )

    files["chain_summaries"] = write_json(
        observables.summaries(),
        cache_dir / "chain_summaries.json",
    )

    if vote_tv is not None:
        files["vote_tv"] = write_json(
            vote_tv,
            diagnostics_dir / "vote_tv.json",
        )

    if compactness_tv is not None:
        files["compactness_tv"] = write_json(
            compactness_tv,
            diagnostics_dir / "compactness_tv.json",
        )

    if vote_rhat is not None:
        files["vote_rhat"] = write_json(
            vote_rhat,
            diagnostics_dir / "vote_rhat.json",
        )

    if compactness_rhat is not None:
        files["compactness_rhat"] = write_json(
            compactness_rhat,
            diagnostics_dir / "compactness_rhat.json",
        )

    if extra is not None:
        files["extra"] = write_json(
            dict(extra),
            diagnostics_dir / "extra.json",
        )

    manifest = CacheManifest(output_dir=output_dir, files=files)

    files["manifest"] = write_json(
        manifest,
        output_dir / "manifest.json",
    )

    return manifest


def load_pipeline_observable_cache(
    output_dir: str | Path,
) -> dict[str, Any]:
    """
    Load cached observable matrices and metadata.

    This does not reconstruct the full ChainObservableResult object. It returns
    the matrices needed by plotting/diagnostics.
    """
    output_dir = Path(output_dir)
    cache_dir = output_dir / "cache"

    vote_path = cache_dir / "vote_runs.parquet"
    compactness_path = cache_dir / "compactness_runs.parquet"
    metadata_path = cache_dir / "chain_metadata.parquet"
    summaries_path = cache_dir / "chain_summaries.json"

    if not vote_path.exists():
        raise CacheError(f"missing vote cache: {vote_path}")

    out: dict[str, Any] = {
        "vote_runs": read_observable_runs(
            vote_path,
            value_col="dem_share",
        )
    }

    if compactness_path.exists():
        out["compactness_runs"] = read_observable_runs(
            compactness_path,
            value_col="compactness",
        )

    if metadata_path.exists():
        out["metadata"] = read_chain_metadata(metadata_path)

    if summaries_path.exists():
        out["summaries"] = read_json(summaries_path)

    return out