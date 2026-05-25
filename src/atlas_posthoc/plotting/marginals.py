from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from atlas_posthoc.plotting.style import (
    PlotStyle,
    PlotStyleName,
    add_zero_one_bounds,
    apply_axis_labels,
    chain_color_cycle,
    despine,
    format_percent_axis,
    get_plot_style,
    plot_style,
    save_figure,
)


ObservableKind = Literal["vote_share", "compactness", "generic"]


class MarginalPlotError(Exception):
    """Base exception for ordered marginal plotting."""


def _normalize_runs(
    runs: Sequence[Any] | Mapping[str, Any],
    *,
    chain_ids: Sequence[str] | None = None,
) -> tuple[list[str], list[np.ndarray]]:
    """
    Normalize list/dict chain inputs into chain ids and 2D arrays.

    Each run must have shape:
        (n_maps, n_ranks)
    """
    if isinstance(runs, Mapping):
        ids = [str(k) for k in runs.keys()]
        arrays = [np.asarray(v, dtype=float) for v in runs.values()]
    else:
        arrays = [np.asarray(v, dtype=float) for v in runs]
        if chain_ids is None:
            ids = [f"Run {i + 1}" for i in range(len(arrays))]
        else:
            ids = [str(x) for x in chain_ids]

    if len(arrays) == 0:
        raise MarginalPlotError("no runs were provided")

    if len(ids) != len(arrays):
        raise MarginalPlotError("chain_ids length must match number of runs")

    n_ranks = arrays[0].shape[1]

    for chain_id, arr in zip(ids, arrays):
        if arr.ndim != 2:
            raise MarginalPlotError(
                f"run {chain_id!r} must be 2D with shape (n_maps, n_ranks); "
                f"got shape {arr.shape}"
            )
        if arr.shape[1] != n_ranks:
            raise MarginalPlotError(
                f"all runs must have the same number of ranks; "
                f"run {chain_id!r} has {arr.shape[1]}, expected {n_ranks}"
            )

    return ids, arrays


def _finite_values(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _silverman_bandwidth(x: np.ndarray) -> float:
    """
    Silverman's rule of thumb for 1D KDE bandwidth.
    """
    x = _finite_values(x)

    if x.size < 2:
        return 1.0

    std = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(std, iqr / 1.34) if iqr > 0 else std

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = std

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    return 0.9 * sigma * (x.size ** (-1 / 5))


def _gaussian_kde_1d(
    samples: np.ndarray,
    grid: np.ndarray,
    *,
    bandwidth: float | None = None,
) -> np.ndarray:
    """
    Lightweight Gaussian KDE without scipy.

    Returns an unnormalized density on `grid`.
    """
    x = _finite_values(samples)

    if x.size == 0:
        return np.zeros_like(grid)

    if x.size == 1:
        bw = 1.0 if bandwidth is None else float(bandwidth)
    else:
        bw = _silverman_bandwidth(x) if bandwidth is None else float(bandwidth)

    if not np.isfinite(bw) or bw <= 0:
        bw = 1.0

    z = (grid[:, None] - x[None, :]) / bw
    density = np.exp(-0.5 * z**2).sum(axis=1)
    density /= (x.size * bw * np.sqrt(2 * np.pi))

    return density


def _auto_value_grid(
    arrays: Sequence[np.ndarray],
    *,
    observable_kind: ObservableKind,
    reference_values: Sequence[float] | None = None,
    n_grid: int = 200,
) -> np.ndarray:
    values = np.concatenate([arr.ravel() for arr in arrays])
    values = values[np.isfinite(values)]

    if reference_values is not None:
        ref = np.asarray(reference_values, dtype=float)
        ref = ref[np.isfinite(ref)]
        if ref.size > 0:
            values = np.concatenate([values, ref])

    if values.size == 0:
        raise MarginalPlotError("all run values are non-finite")

    vmin = float(values.min())
    vmax = float(values.max())

    if observable_kind == "vote_share":
        lower = min(0.0, vmin)
        upper = max(1.0, vmax)
    else:
        if vmin == vmax:
            pad = 1.0
        else:
            pad = 0.05 * (vmax - vmin)
        lower = vmin - pad
        upper = vmax + pad

    if lower == upper:
        lower -= 1.0
        upper += 1.0

    return np.linspace(lower, upper, n_grid)


def _default_labels(
    observable_kind: ObservableKind,
) -> tuple[str, str]:
    if observable_kind == "vote_share":
        return (
            "Districts ordered from most to least Republican",
            "Democratic vote share",
        )

    if observable_kind == "compactness":
        return (
            "Districts ordered from most to least compact",
            "Isoperimetric score",
        )

    return ("Ordered district rank", "Observable value")


def _plot_density_at_rank(
    ax: Any,
    samples: np.ndarray,
    *,
    x0: float,
    grid: np.ndarray,
    width_scale: float,
    color: str,
    alpha: float,
    line_width: float,
    bandwidth: float | None = None,
) -> None:
    """
    Draw one mirrored density shape centered at x0.
    """
    vals = _finite_values(samples)

    if vals.size == 0:
        return

    density = _gaussian_kde_1d(vals, grid, bandwidth=bandwidth)

    max_density = density.max()
    if not np.isfinite(max_density) or max_density <= 0:
        return

    half_width = width_scale * density / max_density

    x_left = x0 - half_width
    x_right = x0 + half_width

    ax.fill_betweenx(
        grid,
        x_left,
        x_right,
        facecolor=color,
        edgecolor="black",
        linewidth=line_width,
        alpha=alpha,
        zorder=2,
    )


def plot_ordered_marginals_by_chain(
    runs: Sequence[Any] | Mapping[str, Any],
    *,
    chain_ids: Sequence[str] | None = None,
    observable_kind: ObservableKind = "generic",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    style: PlotStyleName | PlotStyle = "manuscript",
    reference_values: Sequence[float] | None = None,
    reference_label: str = "Enacted plan",
    reference_color: str = "red",
    percent_axis: bool | None = None,
    y_limits: tuple[float, float] | None = None,
    bandwidth: float | None = None,
    violin_width: float = 0.18,
    rank_gap: float = 1.0,
    show_legend: bool = True,
    legend_loc: str = "upper left",
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """
    Plot ordered rank-wise marginal densities by chain.

    Parameters
    ----------
    runs:
        Mapping or list of matrices with shape (n_maps, n_ranks).

    reference_values:
        Optional ordered vector to overlay as red dots, e.g. enacted plan.

    Returns
    -------
    (fig, ax)
    """
    ids, arrays = _normalize_runs(runs, chain_ids=chain_ids)
    s = get_plot_style(style)

    n_chains = len(arrays)
    n_ranks = arrays[0].shape[1]

    if reference_values is not None and len(reference_values) != n_ranks:
        raise MarginalPlotError(
            f"reference_values has length {len(reference_values)}, expected {n_ranks}"
        )

    if percent_axis is None:
        percent_axis = observable_kind == "vote_share"

    if xlabel is None or ylabel is None:
        default_xlabel, default_ylabel = _default_labels(observable_kind)
        xlabel = xlabel or default_xlabel
        ylabel = ylabel or default_ylabel

    if ax is None:
        with plot_style(s):
            fig, ax = plt.subplots(figsize=s.figsize_wide)
    else:
        fig = ax.figure

    colors = chain_color_cycle()
    if n_chains > len(colors):
        raise MarginalPlotError(
            f"too many chains ({n_chains}); default color cycle supports {len(colors)}"
        )

    grid = _auto_value_grid(
        arrays,
        observable_kind=observable_kind,
        reference_values=reference_values,
    )

    # offsets for side-by-side density shapes at each rank
    if n_chains == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-0.24, 0.24, n_chains)

    rank_positions = np.arange(1, n_ranks + 1, dtype=float) * rank_gap

    # draw densities
    for rank_idx in range(n_ranks):
        x_rank = rank_positions[rank_idx]

        for chain_idx, (chain_id, arr) in enumerate(zip(ids, arrays)):
            samples = arr[:, rank_idx]
            x0 = x_rank + offsets[chain_idx]

            _plot_density_at_rank(
                ax,
                samples,
                x0=x0,
                grid=grid,
                width_scale=violin_width,
                color=colors[chain_idx],
                alpha=s.alpha,
                line_width=0.8,
                bandwidth=bandwidth,
            )

    # overlay reference / enacted points
    if reference_values is not None:
        ref = np.asarray(reference_values, dtype=float)
        ax.scatter(
            rank_positions,
            ref,
            color=reference_color,
            s=28,
            label=reference_label,
            zorder=4,
        )

    # legend handles
    if show_legend:
        from matplotlib.patches import Patch

        handles = [
            Patch(
                facecolor=colors[i],
                edgecolor="black",
                label=ids[i],
                alpha=s.alpha,
            )
            for i in range(n_chains)
        ]

        if reference_values is not None:
            from matplotlib.lines import Line2D

            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    color=reference_color,
                    label=reference_label,
                )
            )

        ax.legend(handles=handles, loc=legend_loc, frameon=True, fancybox=False)

    ax.set_xticks(rank_positions)
    ax.set_xticklabels([str(i) for i in range(1, n_ranks + 1)])

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    elif observable_kind == "vote_share":
        add_zero_one_bounds(ax, axis="y", lower=0.0, upper=1.0)

    if percent_axis:
        format_percent_axis(ax, axis="y")

    apply_axis_labels(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    despine(ax)

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig, ax


def plot_vote_share_marginals(
    runs: Sequence[Any] | Mapping[str, Any],
    *,
    chain_ids: Sequence[str] | None = None,
    reference_values: Sequence[float] | None = None,
    reference_label: str = "Enacted plan",
    style: PlotStyleName | PlotStyle = "manuscript",
    title: str | None = None,
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """
    Convenience wrapper for ordered vote-share marginals.
    """
    return plot_ordered_marginals_by_chain(
        runs,
        chain_ids=chain_ids,
        observable_kind="vote_share",
        title=title,
        style=style,
        reference_values=reference_values,
        reference_label=reference_label,
        ax=ax,
        output_path=output_path,
    )


def plot_compactness_marginals(
    runs: Sequence[Any] | Mapping[str, Any],
    *,
    chain_ids: Sequence[str] | None = None,
    reference_values: Sequence[float] | None = None,
    reference_label: str = "Enacted plan",
    style: PlotStyleName | PlotStyle = "manuscript",
    title: str | None = None,
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """
    Convenience wrapper for ordered compactness marginals.
    """
    return plot_ordered_marginals_by_chain(
        runs,
        chain_ids=chain_ids,
        observable_kind="compactness",
        title=title,
        style=style,
        reference_values=reference_values,
        reference_label=reference_label,
        ax=ax,
        output_path=output_path,
    )