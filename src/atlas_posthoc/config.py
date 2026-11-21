"""Load project settings for the AtlasIO convergence workflow."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """A missing or invalid project configuration value."""


@dataclass(frozen=True)
class ConvergenceOptions:
    vote_bin_width: float
    compactness_bin_width: float
    vote_y_limits: tuple[float, float] | None
    compactness_y_limits: tuple[float, float] | None
    figure_prefix: str
    report_title: str


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    root: Path
    chain_files: tuple[Path, ...]
    chain_ids: tuple[str, ...] | None
    graph_json: Path
    key_cols: tuple[str, ...] | None
    dem_col: str
    rep_col: str
    n_districts: int | None
    output_frequency: int
    thin_step: int
    compute_compactness: bool
    output_root: Path
    paper_mode: bool
    convergence: ConvergenceOptions

    def check_inputs(self) -> None:
        inputs = (self.graph_json, *self.chain_files)
        missing = [str(path) for path in inputs if not path.is_file()]
        if missing:
            raise ConfigError(
                "input files are missing or are not regular files:\n  "
                + "\n  ".join(missing)
            )
        if self.output_root.exists() and not self.output_root.is_dir():
            raise ConfigError(f"outputs.root is not a directory: {self.output_root}")


def _string(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a nonempty string")
    return value


def _strings(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{key} must be a nonempty array of strings")
    return tuple(_string(item, key) for item in value)


def _positive_int(value: Any, key: str) -> int:
    if type(value) is not int or value < 1:
        raise ConfigError(f"{key} must be a positive integer")
    return value


def _bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")
    return value


def _finite_number(value: Any, key: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ConfigError(f"{key} must be a finite number")
    return float(value)


def _bin_width(value: Any, key: str) -> float:
    width = _finite_number(value, key)
    if width <= 0:
        raise ConfigError(f"{key} must be positive")
    return width


def _limits(value: Any, key: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{key} must contain a lower and upper bound")
    lower, upper = (_finite_number(item, key) for item in value)
    if lower >= upper:
        raise ConfigError(f"{key} must be in increasing order")
    return lower, upper


def _resolve_path(root: Path, value: str) -> Path:
    return (root / Path(value).expanduser()).resolve()


def load_config(path: str | Path) -> ProjectConfig:
    """Resolve project.root from the TOML file and all other paths from that root."""
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("rb") as stream:
            data = tomllib.load(stream)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"could not read {config_path}: {exc}") from exc

    def get(dotted_key: str, default: Any = None) -> Any:
        current = data
        parts = dotted_key.split(".")
        for index, part in enumerate(parts[:-1]):
            current = current.get(part, {})
            if not isinstance(current, dict):
                raise ConfigError(f"{'.'.join(parts[: index + 1])} must be a table")
        return current.get(parts[-1], default)

    root = _resolve_path(
        config_path.parent, _string(get("project.root", "."), "project.root")
    )
    if not root.is_dir():
        raise ConfigError(f"project.root is not a directory: {root}")

    chain_files = tuple(
        _resolve_path(root, file)
        for file in _strings(get("chains.files"), "chains.files")
    )
    if len(chain_files) < 2 or len(set(chain_files)) != len(chain_files):
        raise ConfigError("chains.files must contain at least two distinct chain files")
    chain_ids = None
    if (value := get("chains.ids")) is not None:
        chain_ids = _strings(value, "chains.ids")
        if len(chain_ids) != len(chain_files) or len(set(chain_ids)) != len(chain_ids):
            raise ConfigError("chains.ids must contain one unique ID per chain file")

    key_cols = None
    if (value := get("columns.key_cols")) is not None:
        key_cols = _strings(value, "columns.key_cols")
    n_districts = None
    if (value := get("analysis.n_districts")) is not None:
        n_districts = _positive_int(value, "analysis.n_districts")

    prefix = get("workflow.convergence.figure_prefix", "")
    if not isinstance(prefix, str) or "/" in prefix or "\\" in prefix:
        raise ConfigError(
            "workflow.convergence.figure_prefix must be a filename prefix"
        )

    def option(key: str, default: Any, validate: Any) -> Any:
        dotted_key = f"workflow.convergence.{key}"
        return validate(get(dotted_key, default), dotted_key)

    return ProjectConfig(
        path=config_path,
        root=root,
        chain_files=chain_files,
        chain_ids=chain_ids,
        graph_json=_resolve_path(
            root, _string(get("inputs.graph_json"), "inputs.graph_json")
        ),
        key_cols=key_cols,
        dem_col=_string(get("columns.dem_col"), "columns.dem_col"),
        rep_col=_string(get("columns.rep_col"), "columns.rep_col"),
        n_districts=n_districts,
        output_frequency=_positive_int(
            get("analysis.output_frequency", 10000), "analysis.output_frequency"
        ),
        thin_step=_positive_int(get("analysis.thin_step", 1), "analysis.thin_step"),
        compute_compactness=_bool(
            get("analysis.compute_compactness", True), "analysis.compute_compactness"
        ),
        output_root=_resolve_path(
            root, _string(get("outputs.root", "outputs"), "outputs.root")
        ),
        paper_mode=_bool(get("figures.paper_mode", False), "figures.paper_mode"),
        convergence=ConvergenceOptions(
            vote_bin_width=option("vote_bin_width", 0.002, _bin_width),
            compactness_bin_width=option("compactness_bin_width", 0.5, _bin_width),
            vote_y_limits=option("vote_y_limits", None, _limits),
            compactness_y_limits=option("compactness_y_limits", None, _limits),
            figure_prefix=prefix,
            report_title=option(
                "report_title", "Atlas post-hoc diagnostic report", _string
            ),
        ),
    )
