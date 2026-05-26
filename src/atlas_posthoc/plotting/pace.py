from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from atlas_posthoc.plotting.style import (
    PlotStyle,
    PlotStyleName,
    apply_axis_labels,
    get_plot_style,
    plot_style,
    save_figure,
)


PACEHeatmapValue = Literal["probability", "count"]
PACECellSort = Literal["pooled_count", "label", "none"]


class PACEPlotError(Exception):
    """Base exception for PACE plotting helpers."""


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_pace_result(pace_like: Any) -> Any:
    """
    Extract a PACEResult-like object.

    Accepts:
      - PACEResult;
      - PACEDistanceClusteringResult, via `.pace`;
      - dictionary with key `"pace"`;
      - dictionary with direct PACEResult-style keys.
    """
    if hasattr(pace_like, "pace"):
        return pace_like.pace

    if isinstance(pace_like, Mapping) and "pace" in pace_like:
        return pace_like["pace"]

    if _get(pace_like, "contingency") is not None:
        return pace_like

    raise PACEPlotError(
        "expected a PACEResult-like object or PACEDistanceClusteringResult-like object"
    )


def _extract_counts_chain_names_labels(
    pace_like: Any,
) -> tuple[np.ndarray, list[str], list[Any]]:
    pace = _extract_pace_result(pace_like)

    contingency = _get(pace, "contingency")
    if contingency is None:
        raise PACEPlotError("PACE result does not contain `contingency`")

    counts = np.asarray(contingency, dtype=float)

    if counts.ndim != 2:
        raise PACEPlotError(f"contingency must be 2D; got shape {counts.shape}")

    if counts.shape[0] == 0 or counts.shape[1] == 0:
        raise PACEPlotError("contingency table must have at least one row and one column")

    chain_names = _get(pace, "chain_names", None)
    if chain_names is None:
        chain_names = [f"chain_{i}" for i in range(counts.shape[0])]
    chain_names = [str(x) for x in chain_names]

    if len(chain_names) != counts.shape[0]:
        raise PACEPlotError(
            f"chain_names length {len(chain_names)} does not match "
            f"contingency rows {counts.shape[0]}"
        )

    labels = _get(pace, "labels", None)
    if labels is None:
        labels = list(range(counts.shape[1]))

    labels = list(labels)

    if len(labels) != counts.shape[1]:
        raise PACEPlotError(
            f"labels length {len(labels)} does not match contingency columns {counts.shape[1]}"
        )

    return counts, chain_names, labels


def _sort_cells(
    counts: np.ndarray,
    labels: list[Any],
    *,
    sort_cells_by: PACECellSort,
) -> tuple[np.ndarray, list[Any]]:
    if sort_cells_by == "none":
        return counts, labels

    if sort_cells_by == "label":
        order = sorted(range(len(labels)), key=lambda j: str(labels[j]))
    elif sort_cells_by == "pooled_count":
        pooled = counts.sum(axis=0)
        order = list(np.argsort(-pooled))
    else:
        raise ValueError(f"unknown sort_cells_by: {sort_cells_by!r}")

    return counts[:, order], [labels[j] for j in order]


def _collapse_cells(
    counts: np.ndarray,
    labels: list[Any],
    *,
    max_cells: int | None,
    other_label: str = "other",
) -> tuple[np.ndarray, list[Any]]:
    """
    Collapse small cells into an 'other' column if there are too many cells.

    This is important for real PACE runs, where exact-state or fine clustering
    can produce many cells. Heatmaps become unreadable if every tiny cell is
    drawn separately.
    """
    if max_cells is None or counts.shape[1] <= max_cells:
        return counts, labels

    if max_cells < 2:
        raise ValueError("max_cells must be at least 2 when provided")

    keep = max_cells - 1
    kept_counts = counts[:, :keep]
    other_counts = counts[:, keep:].sum(axis=1, keepdims=True)

    collapsed = np.hstack([kept_counts, other_counts])
    collapsed_labels = labels[:keep] + [other_label]

    return collapsed, collapsed_labels


def _values_from_counts(
    counts: np.ndarray,
    *,
    value: PACEHeatmapValue,
) -> np.ndarray:
    if value == "count":
        return counts

    if value == "probability":
        row_sums = counts.sum(axis=1, keepdims=True)
        return np.divide(
            counts,
            row_sums,
            out=np.zeros_like(counts, dtype=float),
            where=row_sums > 0,
        )

    raise ValueError(f"unknown heatmap value: {value!r}")


def _default_title(pace_like: Any, *, value: PACEHeatmapValue) -> str:
    pace = _extract_pace_result(pace_like)

    delta = _get(pace, "delta", None)
    threshold = _get(pace, "threshold", None)
    converged = _get(pace, "converged", None)

    value_text = "cell probabilities" if value == "probability" else "cell counts"

    if delta is None:
        return f"PACE chain-by-cell {value_text}"

    if converged is True:
        status = "PASS"
    elif converged is False:
        status = "WARN"
    else:
        status = "INFO"

    if threshold is None:
        return f"PACE {value_text}: delta={float(delta):.4f} [{status}]"

    return (
        f"PACE {value_text}: delta={float(delta):.4f}, "
        f"threshold={float(threshold):.3f} [{status}]"
    )


def _format_cell_value(x: float, *, value: PACEHeatmapValue) -> str:
    if value == "count":
        return str(int(round(x)))

    return f"{x:.2f}"


def plot_pace_contingency_heatmap(
    pace_like: Any,
    *,
    value: PACEHeatmapValue = "probability",
    sort_cells_by: PACECellSort = "pooled_count",
    max_cells: int | None = 30,
    title: str | None = None,
    xlabel: str = "PACE cell / cluster",
    ylabel: str = "Chain",
    colorbar_label: str | None = None,
    style: PlotStyleName | PlotStyle = "manuscript",
    figsize: tuple[float, float] | None = None,
    cmap: str | None = None,
    annotate: bool | None = None,
    annotation_fontsize: int = 7,
    rotate_xticks: bool = True,
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """
    Plot a chain-by-cell PACE heatmap.

    Rows are chains. Columns are PACE cells or clusters. The default value is
    the within-chain probability of each cell.

    This plot is meant to diagnose state-space convergence:

        similar rows  -> chains visit PACE cells with similar frequencies
        different rows -> chains occupy different regions of state space

    Parameters
    ----------
    pace_like:
        PACEResult, PACEDistanceClusteringResult, or a matching dictionary.

    value:
        `"probability"` plots row-normalized cell probabilities.
        `"count"` plots raw contingency counts.

    max_cells:
        If there are too many cells, keep the largest `max_cells - 1` cells by
        pooled count and collapse the rest into an `"other"` column. Set to
        None to show all cells.
    """
    counts, chain_names, labels = _extract_counts_chain_names_labels(pace_like)

    counts, labels = _sort_cells(
        counts,
        labels,
        sort_cells_by=sort_cells_by,
    )

    counts, labels = _collapse_cells(
        counts,
        labels,
        max_cells=max_cells,
    )

    values = _values_from_counts(counts, value=value)

    s = get_plot_style(style)

    if title is None:
        title = _default_title(pace_like, value=value)

    if colorbar_label is None:
        colorbar_label = "Cell probability" if value == "probability" else "Count"

    if annotate is None:
        annotate = values.size <= 80

    if figsize is None:
        # Wider if many cells, but bounded.
        width = max(s.figsize_wide[0], min(14.0, 0.35 * len(labels) + 3.0))
        height = max(3.0, 0.45 * len(chain_names) + 2.0)
        figsize = (width, height)

    if ax is None:
        with plot_style(s):
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Heatmaps should not inherit the normal line-plot grid.
    ax.grid(False)

    im = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
    )

    ax.set_yticks(np.arange(len(chain_names)))
    ax.set_yticklabels(chain_names)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([str(x) for x in labels])

    if rotate_xticks:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(
                    j,
                    i,
                    _format_cell_value(values[i, j], value=value),
                    ha="center",
                    va="center",
                    fontsize=annotation_fontsize,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)

    apply_axis_labels(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig, ax


def plot_pace_probability_heatmap(
    pace_like: Any,
    *,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """
    Convenience wrapper for row-normalized PACE cell probabilities.
    """
    return plot_pace_contingency_heatmap(
        pace_like,
        value="probability",
        output_path=output_path,
        **kwargs,
    )


def plot_pace_count_heatmap(
    pace_like: Any,
    *,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """
    Convenience wrapper for raw PACE cell counts.
    """
    return plot_pace_contingency_heatmap(
        pace_like,
        value="count",
        output_path=output_path,
        **kwargs,
    )


# Friendly alias.
plot_pace_heatmap = plot_pace_probability_heatmap