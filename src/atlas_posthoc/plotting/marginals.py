from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
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

def _plot_half_violin_at_rank(
    ax: Any,
    samples: np.ndarray,
    *,
    x0: float,
    color: str,
    alpha: float,
    width: float = 0.08,
    line_width: float = 0.8,
    mean_line_width: float = 1.0,
    min_pad: float = 1e-4,
) -> None:
    """
    Draw one side-by-side half violin at a district rank.

    This intentionally follows the older MFR / CT-style plotting grammar:
    local KDE support, one-sided density, no full-height violin outline, and
    a short black mean marker.
    """
    data = _finite_values(samples)

    if data.size == 0:
        return

    if data.size < 2 or np.ptp(data) < 1e-12:
        y0 = float(np.mean(data))
        ax.plot(
            [x0, x0 + width * 0.5],
            [y0, y0],
            color=color,
            linewidth=2.0,
            zorder=3,
        )
        return

    data_range = float(np.ptp(data))
    pad = max(0.01 * data_range, min_pad)

    y_vals = np.linspace(
        float(np.min(data) - pad),
        float(np.max(data) + pad),
        160,
    )

    try:
        kde = gaussian_kde(data)
        density = kde(y_vals)
    except Exception:
        y0 = float(np.mean(data))
        ax.plot(
            [x0, x0 + width * 0.5],
            [y0, y0],
            color=color,
            linewidth=2.0,
            zorder=3,
        )
        return

    if not np.any(np.isfinite(density)) or np.nanmax(density) <= 0:
        return

    scaled_density = density * (width / np.nanmax(density))
    x_vals = x0 + scaled_density

    ax.fill_betweenx(
        y_vals,
        x0,
        x_vals,
        facecolor=color,
        alpha=alpha,
        linewidth=0,
        zorder=2,
    )

    ax.plot(
        x_vals,
        y_vals,
        color=color,
        linewidth=line_width,
        zorder=3,
    )

    mean_val = float(np.mean(data))

    try:
        mean_width = float(kde(mean_val)[0] * (width / np.nanmax(density)))
    except Exception:
        mean_width = width * 0.5

    ax.hlines(
        mean_val,
        x0,
        x0 + mean_width,
        color="black",
        linewidth=mean_line_width,
        zorder=4,
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
    violin_width: float = 0.08,
    rank_gap: float = 1.0,
    force_vote_share_bounds: bool = False,
    show_legend: bool = True,
    legend_loc: str = "upper left",
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """
    Notice: This function has been overhauled by some old MetropolizedForestRecom plotting aesthetics.

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

    # Side-by-side offsets matching the old MFR / CT-style half-violin grammar.
    if n_chains == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-0.15, 0.15, n_chains)

    rank_positions = np.arange(1, n_ranks + 1, dtype=float) * rank_gap

    # Draw local half violins.
    for rank_idx in range(n_ranks):
        x_rank = rank_positions[rank_idx]

        for chain_idx, (chain_id, arr) in enumerate(zip(ids, arrays)):
            samples = arr[:, rank_idx]
            x0 = x_rank + offsets[chain_idx]

            _plot_half_violin_at_rank(
                ax,
                samples,
                x0=x0,
                color=colors[chain_idx],
                alpha=0.60,
                width=violin_width,
                line_width=0.8,
                mean_line_width=1.0,
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
    ax.set_xlim(0.5, n_ranks + 0.5)

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    elif observable_kind == "vote_share" and force_vote_share_bounds:
        add_zero_one_bounds(ax, axis="y", lower=0.0, upper=1.0)
    else:
        all_values = np.concatenate([arr.ravel() for arr in arrays])
        all_values = all_values[np.isfinite(all_values)]

        if reference_values is not None:
            ref = np.asarray(reference_values, dtype=float)
            ref = ref[np.isfinite(ref)]
            if ref.size > 0:
                all_values = np.concatenate([all_values, ref])

        if all_values.size > 0:
            ymin = float(np.min(all_values))
            ymax = float(np.max(all_values))
            pad = max(0.05 * (ymax - ymin), 1e-4)
            ax.set_ylim(ymin - pad, ymax + pad)

    if percent_axis:
        format_percent_axis(ax, axis="y")

    apply_axis_labels(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(False, which="both")
    despine(ax)

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig, ax

def _apply_ordered_marginal_axes(
    ax: Any,
    arrays: Sequence[np.ndarray],
    *,
    observable_kind: ObservableKind,
    reference_values: Sequence[float] | None,
    rank_positions: np.ndarray,
    title: str | None,
    xlabel: str,
    ylabel: str,
    y_limits: tuple[float, float] | None,
    percent_axis: bool,
    force_vote_share_bounds: bool,
) -> None:
    ax.set_xticks(rank_positions)
    ax.set_xticklabels([str(i) for i in range(1, len(rank_positions) + 1)])
    ax.set_xlim(0.5, len(rank_positions) + 0.5)

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    elif observable_kind == "vote_share" and force_vote_share_bounds:
        add_zero_one_bounds(ax, axis="y", lower=0.0, upper=1.0)
    else:
        all_values = np.concatenate([arr.ravel() for arr in arrays])
        all_values = all_values[np.isfinite(all_values)]

        if reference_values is not None:
            ref = np.asarray(reference_values, dtype=float)
            ref = ref[np.isfinite(ref)]
            if ref.size > 0:
                all_values = np.concatenate([all_values, ref])

        if all_values.size > 0:
            ymin = float(np.min(all_values))
            ymax = float(np.max(all_values))
            pad = max(0.05 * (ymax - ymin), 1e-4)
            ax.set_ylim(ymin - pad, ymax + pad)

    if percent_axis:
        format_percent_axis(ax, axis="y")

    apply_axis_labels(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(False, which="both")
    despine(ax)

def plot_ordered_boxplots_by_chain(
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
    force_vote_share_bounds: bool = False,
    box_width: float = 0.10,
    rank_gap: float = 1.0,
    showfliers: bool = False,
    showmeans: bool = True,
    show_legend: bool = True,
    legend_loc: str = "upper left",
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """
    Plot ordered rank-wise boxplots by chain.

    This is the boxplot companion to `plot_ordered_marginals_by_chain`.
    Each chain gets a side-by-side boxplot at each ordered district rank.
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

    if n_chains == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-0.18, 0.18, n_chains)

    rank_positions = np.arange(1, n_ranks + 1, dtype=float) * rank_gap

    for chain_idx, (chain_id, arr) in enumerate(zip(ids, arrays)):
        data_by_rank = []
        positions = []

        for rank_idx in range(n_ranks):
            data = arr[:, rank_idx]
            data = data[np.isfinite(data)]

            if data.size == 0:
                continue

            data_by_rank.append(data)
            positions.append(rank_positions[rank_idx] + offsets[chain_idx])

        if not data_by_rank:
            continue

        bp = ax.boxplot(
            data_by_rank,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=showfliers,
            showmeans=showmeans,
            manage_ticks=False,
            medianprops={
                "color": "black",
                "linewidth": 1.0,
            },
            meanprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 3,
            },
            boxprops={
                "linewidth": 0.8,
                "edgecolor": "black",
            },
            whiskerprops={
                "linewidth": 0.8,
                "color": "black",
            },
            capprops={
                "linewidth": 0.8,
                "color": "black",
            },
        )

        for patch in bp["boxes"]:
            patch.set_facecolor(colors[chain_idx])
            patch.set_alpha(0.60)

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

    if show_legend:
        from matplotlib.patches import Patch

        handles = [
            Patch(
                facecolor=colors[i],
                edgecolor="black",
                label=ids[i],
                alpha=0.60,
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

    _apply_ordered_marginal_axes(
        ax,
        arrays,
        observable_kind=observable_kind,
        reference_values=reference_values,
        rank_positions=rank_positions,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        y_limits=y_limits,
        percent_axis=percent_axis,
        force_vote_share_bounds=force_vote_share_bounds,
    )

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
    y_limits: tuple[float, float] | None = None,
    force_vote_share_bounds: bool = False,
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    return plot_ordered_marginals_by_chain(
        runs,
        chain_ids=chain_ids,
        observable_kind="vote_share",
        title=title,
        style=style,
        reference_values=reference_values,
        reference_label=reference_label,
        y_limits=y_limits,
        force_vote_share_bounds=force_vote_share_bounds,
        ax=ax,
        output_path=output_path,
    )

def plot_vote_share_boxplots(
    runs: Sequence[Any] | Mapping[str, Any],
    *,
    chain_ids: Sequence[str] | None = None,
    reference_values: Sequence[float] | None = None,
    reference_label: str = "Enacted plan",
    style: PlotStyleName | PlotStyle = "manuscript",
    title: str | None = None,
    y_limits: tuple[float, float] | None = None,
    force_vote_share_bounds: bool = False,
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """
    Convenience wrapper for ordered vote-share boxplots.
    """
    return plot_ordered_boxplots_by_chain(
        runs,
        chain_ids=chain_ids,
        observable_kind="vote_share",
        title=title,
        style=style,
        reference_values=reference_values,
        reference_label=reference_label,
        y_limits=y_limits,
        force_vote_share_bounds=force_vote_share_bounds,
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
    y_limits: tuple[float, float] | None = None,
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    return plot_ordered_marginals_by_chain(
        runs,
        chain_ids=chain_ids,
        observable_kind="compactness",
        title=title,
        style=style,
        reference_values=reference_values,
        reference_label=reference_label,
        y_limits=y_limits,
        ax=ax,
        output_path=output_path,
    )


def plot_compactness_boxplots(
    runs: Sequence[Any] | Mapping[str, Any],
    *,
    chain_ids: Sequence[str] | None = None,
    reference_values: Sequence[float] | None = None,
    reference_label: str = "Enacted plan",
    style: PlotStyleName | PlotStyle = "manuscript",
    title: str | None = None,
    y_limits: tuple[float, float] | None = None,
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """
    Convenience wrapper for ordered compactness boxplots.
    """
    return plot_ordered_boxplots_by_chain(
        runs,
        chain_ids=chain_ids,
        observable_kind="compactness",
        title=title,
        style=style,
        reference_values=reference_values,
        reference_label=reference_label,
        y_limits=y_limits,
        ax=ax,
        output_path=output_path,
    )