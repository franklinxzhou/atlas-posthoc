from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from atlas_posthoc.observables.municipal_splits import MunicipalSplitTrace
from atlas_posthoc.plotting.style import (
    PlotStyle,
    PlotStyleName,
    apply_axis_labels,
    despine,
    format_percent_axis,
    get_plot_style,
    plot_style,
    save_figure,
)


OustedScale = Literal["share", "population"]


class MunicipalSplitPlotError(Exception):
    """Base exception for municipal split plotting helpers."""


def _trace_get(trace: Any, key: str, default: Any = None) -> Any:
    if isinstance(trace, Mapping):
        return trace.get(key, default)
    return getattr(trace, key, default)


def _normalize_traces(traces: Sequence[MunicipalSplitTrace] | Sequence[Mapping[str, Any]]) -> list[Any]:
    out = list(traces)

    if not out:
        raise MunicipalSplitPlotError("no municipal split traces were provided")

    return out


def _chain_colors(n: int) -> list[Any]:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        return [None] * n
    return [colors[i % len(colors)] for i in range(n)]


def _ousted_values(trace: Any, *, scale: OustedScale) -> list[float]:
    if scale == "share":
        return list(_trace_get(trace, "ousted_population_share", []))
    if scale == "population":
        return list(_trace_get(trace, "ousted_population", []))
    raise ValueError(f"unknown ousted scale: {scale!r}")


def _ousted_ylabel(scale: OustedScale) -> str:
    if scale == "share":
        return "Ousted municipal population share"
    if scale == "population":
        return "Ousted municipal population"
    raise ValueError(f"unknown ousted scale: {scale!r}")


def _plot_trace_lines(
    ax: Any,
    traces: Sequence[Any],
    *,
    y_key: str | None = None,
    ousted_scale: OustedScale | None = None,
    linewidth: float = 0.9,
    alpha: float = 0.85,
    show_legend: bool = True,
) -> None:
    colors = _chain_colors(len(traces))

    for idx, trace in enumerate(traces):
        chain_id = str(_trace_get(trace, "chain_id", f"chain_{idx}"))
        proposals = list(_trace_get(trace, "proposals", []))

        if ousted_scale is not None:
            y = _ousted_values(trace, scale=ousted_scale)
        elif y_key is not None:
            y = list(_trace_get(trace, y_key, []))
        else:
            raise MunicipalSplitPlotError("must provide y_key or ousted_scale")

        if len(proposals) != len(y):
            raise MunicipalSplitPlotError(
                f"trace {chain_id!r} has mismatched proposals/y lengths"
            )

        ax.plot(
            proposals,
            y,
            label=chain_id,
            color=colors[idx],
            linewidth=linewidth,
            alpha=alpha,
        )

    ax.grid(True, alpha=0.25)

    if show_legend:
        ax.legend(frameon=True, fancybox=False)


def plot_cut_municipalities_trace(
    traces: Sequence[MunicipalSplitTrace] | Sequence[Mapping[str, Any]],
    *,
    title: str = "Number of cut municipalities over proposals",
    xlabel: str = "Proposals",
    ylabel: str = "Number of cut municipalities",
    style: PlotStyleName | PlotStyle = "manuscript",
    figsize: tuple[float, float] | None = None,
    show_legend: bool = True,
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """
    Plot number of cut municipalities against proposals.
    """
    traces_list = _normalize_traces(traces)
    s = get_plot_style(style)

    if figsize is None:
        figsize = s.figsize_wide

    if ax is None:
        with plot_style(s):
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    _plot_trace_lines(
        ax,
        traces_list,
        y_key="n_cut_municipalities",
        show_legend=show_legend,
    )

    apply_axis_labels(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    despine(ax)

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig, ax


def plot_ousted_population_trace(
    traces: Sequence[MunicipalSplitTrace] | Sequence[Mapping[str, Any]],
    *,
    scale: OustedScale = "share",
    title: str | None = None,
    xlabel: str = "Proposals",
    ylabel: str | None = None,
    style: PlotStyleName | PlotStyle = "manuscript",
    figsize: tuple[float, float] | None = None,
    show_legend: bool = True,
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """
    Plot ousted municipal population, either as raw population or population share.
    """
    traces_list = _normalize_traces(traces)
    s = get_plot_style(style)

    if title is None:
        title = f"{_ousted_ylabel(scale)} over proposals"

    if ylabel is None:
        ylabel = _ousted_ylabel(scale)

    if figsize is None:
        figsize = s.figsize_wide

    if ax is None:
        with plot_style(s):
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    _plot_trace_lines(
        ax,
        traces_list,
        ousted_scale=scale,
        show_legend=show_legend,
    )

    if scale == "share":
        format_percent_axis(ax, axis="y")

    apply_axis_labels(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    despine(ax)

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig, ax


def plot_municipal_split_trace_panel(
    traces: Sequence[MunicipalSplitTrace] | Sequence[Mapping[str, Any]],
    *,
    ousted_scale: OustedScale = "share",
    title: str | None = "Municipal split traces by chain",
    style: PlotStyleName | PlotStyle = "manuscript",
    figsize: tuple[float, float] | None = None,
    show_legend: bool = True,
    output_path: str | Path | None = None,
) -> tuple[Any, Sequence[Any]]:
    """
    Two-panel trace diagnostic:

      1. number of cut municipalities vs proposals;
      2. ousted municipal population/share vs proposals.
    """
    traces_list = _normalize_traces(traces)
    s = get_plot_style(style)

    if figsize is None:
        figsize = (11.0, 4.0)

    with plot_style(s):
        fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False)

    ax0, ax1 = axes.ravel()

    _plot_trace_lines(
        ax0,
        traces_list,
        y_key="n_cut_municipalities",
        show_legend=show_legend,
    )
    apply_axis_labels(
        ax0,
        title="Cut municipalities",
        xlabel="Proposals",
        ylabel="Number of cut municipalities",
    )
    despine(ax0)

    _plot_trace_lines(
        ax1,
        traces_list,
        ousted_scale=ousted_scale,
        show_legend=False,
    )

    if ousted_scale == "share":
        format_percent_axis(ax1, axis="y")

    apply_axis_labels(
        ax1,
        title=_ousted_ylabel(ousted_scale),
        xlabel="Proposals",
        ylabel=_ousted_ylabel(ousted_scale),
    )
    despine(ax1)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig, [ax0, ax1]


def _boxplot_by_chain(
    ax: Any,
    traces: Sequence[Any],
    *,
    y_key: str | None = None,
    ousted_scale: OustedScale | None = None,
    ylabel: str,
    title: str,
    showfliers: bool = False,
) -> None:
    colors = _chain_colors(len(traces))
    data: list[np.ndarray] = []
    labels: list[str] = []

    for idx, trace in enumerate(traces):
        chain_id = str(_trace_get(trace, "chain_id", f"chain_{idx}"))

        if ousted_scale is not None:
            values = _ousted_values(trace, scale=ousted_scale)
        elif y_key is not None:
            values = list(_trace_get(trace, y_key, []))
        else:
            raise MunicipalSplitPlotError("must provide y_key or ousted_scale")

        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            continue

        data.append(arr)
        labels.append(chain_id)

    if not data:
        raise MunicipalSplitPlotError("no finite data available for boxplot")

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showfliers=showfliers,
        medianprops={"color": "black", "linewidth": 1.0},
        boxprops={"linewidth": 0.8, "edgecolor": "black"},
        whiskerprops={"linewidth": 0.8, "color": "black"},
        capprops={"linewidth": 0.8, "color": "black"},
    )

    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[idx])
        patch.set_alpha(0.60)

    ax.grid(True, axis="y", alpha=0.25)
    apply_axis_labels(ax, title=title, xlabel="Chain", ylabel=ylabel)
    despine(ax)


def plot_municipal_split_distribution_panel(
    traces: Sequence[MunicipalSplitTrace] | Sequence[Mapping[str, Any]],
    *,
    ousted_scale: OustedScale = "share",
    title: str | None = "Municipal split distributions by chain",
    style: PlotStyleName | PlotStyle = "manuscript",
    figsize: tuple[float, float] | None = None,
    showfliers: bool = False,
    output_path: str | Path | None = None,
) -> tuple[Any, Sequence[Any]]:
    """
    Two-panel ensemble distribution diagnostic using boxplots by chain.
    """
    traces_list = _normalize_traces(traces)
    s = get_plot_style(style)

    if figsize is None:
        figsize = (11.0, 4.0)

    with plot_style(s):
        fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False)

    ax0, ax1 = axes.ravel()

    _boxplot_by_chain(
        ax0,
        traces_list,
        y_key="n_cut_municipalities",
        ylabel="Number of cut municipalities",
        title="Cut municipalities",
        showfliers=showfliers,
    )

    _boxplot_by_chain(
        ax1,
        traces_list,
        ousted_scale=ousted_scale,
        ylabel=_ousted_ylabel(ousted_scale),
        title=_ousted_ylabel(ousted_scale),
        showfliers=showfliers,
    )

    if ousted_scale == "share":
        format_percent_axis(ax1, axis="y")

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig, [ax0, ax1]


def plot_municipal_split_scatter(
    traces: Sequence[MunicipalSplitTrace] | Sequence[Mapping[str, Any]],
    *,
    ousted_scale: OustedScale = "share",
    title: str = "Municipal split tradeoff",
    xlabel: str = "Number of cut municipalities",
    ylabel: str | None = None,
    style: PlotStyleName | PlotStyle = "manuscript",
    figsize: tuple[float, float] | None = None,
    alpha: float = 0.45,
    s: float = 14.0,
    show_legend: bool = True,
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """
    Scatter plot of cut-count vs ousted population/share.

    Each point is one sampled plan.
    """
    traces_list = _normalize_traces(traces)
    plot_s = get_plot_style(style)

    if ylabel is None:
        ylabel = _ousted_ylabel(ousted_scale)

    if figsize is None:
        figsize = plot_s.figsize_wide

    if ax is None:
        with plot_style(plot_s):
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    colors = _chain_colors(len(traces_list))

    for idx, trace in enumerate(traces_list):
        chain_id = str(_trace_get(trace, "chain_id", f"chain_{idx}"))
        x = np.asarray(_trace_get(trace, "n_cut_municipalities", []), dtype=float)
        y = np.asarray(_ousted_values(trace, scale=ousted_scale), dtype=float)

        if x.size != y.size:
            raise MunicipalSplitPlotError(
                f"trace {chain_id!r} has mismatched cut/ousted lengths"
            )

        mask = np.isfinite(x) & np.isfinite(y)

        ax.scatter(
            x[mask],
            y[mask],
            label=chain_id,
            color=colors[idx],
            alpha=alpha,
            s=s,
        )

    if ousted_scale == "share":
        format_percent_axis(ax, axis="y")

    ax.grid(True, alpha=0.25)

    if show_legend:
        ax.legend(frameon=True, fancybox=False)

    apply_axis_labels(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    despine(ax)

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig, ax