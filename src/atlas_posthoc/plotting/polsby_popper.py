from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from atlas_posthoc.observables.polsby_popper import (
    IsoperimetricConvention,
    PolsbyPopperTrace,
    make_polsby_popper_traces,
)
from atlas_posthoc.plotting.style import (
    PlotStyle,
    PlotStyleName,
    apply_axis_labels,
    get_plot_style,
    plot_style,
    save_figure,
)


PolsbyPopperTraceKind = Literal["average", "minimum", "maximum"]


class PolsbyPopperPlotError(Exception):
    """Base exception for Polsby-Popper plotting helpers."""


def _trace_y(trace: PolsbyPopperTrace, kind: PolsbyPopperTraceKind) -> list[float]:
    if kind == "average":
        return trace.average_polsby_popper
    if kind == "minimum":
        return trace.minimum_polsby_popper
    if kind == "maximum":
        return trace.maximum_polsby_popper
    raise ValueError(f"unknown Polsby-Popper trace kind: {kind!r}")


def _trace_ylabel(kind: PolsbyPopperTraceKind) -> str:
    if kind == "average":
        return "Average Polsby-Popper"
    if kind == "minimum":
        return "Minimum Polsby-Popper"
    if kind == "maximum":
        return "Maximum Polsby-Popper"
    raise ValueError(f"unknown Polsby-Popper trace kind: {kind!r}")


def _normalize_traces(
    traces_or_runs: Sequence[PolsbyPopperTrace] | Mapping[str, Any] | Sequence[Any],
    *,
    chain_ids: Sequence[str] | None = None,
    output_frequency: int = 1,
    convention: IsoperimetricConvention = "p2_over_area",
    input_is_polsby_popper: bool = False,
) -> list[PolsbyPopperTrace]:
    if isinstance(traces_or_runs, Sequence) and not isinstance(traces_or_runs, Mapping):
        if traces_or_runs and all(isinstance(x, PolsbyPopperTrace) for x in traces_or_runs):
            return list(traces_or_runs)

    return make_polsby_popper_traces(
        traces_or_runs,  # type: ignore[arg-type]
        chain_ids=chain_ids,
        output_frequency=output_frequency,
        convention=convention,
        input_is_polsby_popper=input_is_polsby_popper,
    )


def _plot_single_trace_kind(
    ax: Any,
    traces: Sequence[PolsbyPopperTrace],
    *,
    kind: PolsbyPopperTraceKind,
    colors: Sequence[str] | None = None,
    linewidth: float = 0.9,
    alpha: float = 0.85,
    show_legend: bool = True,
) -> None:
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for idx, trace in enumerate(traces):
        color = colors[idx % len(colors)] if colors else None

        ax.plot(
            trace.proposals,
            _trace_y(trace, kind),
            label=trace.chain_id,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )

    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)

    if show_legend:
        ax.legend(frameon=True, fancybox=False)


def plot_polsby_popper_trace(
    traces_or_runs: Sequence[PolsbyPopperTrace] | Mapping[str, Any] | Sequence[Any],
    *,
    kind: PolsbyPopperTraceKind = "average",
    chain_ids: Sequence[str] | None = None,
    output_frequency: int = 1,
    convention: IsoperimetricConvention = "p2_over_area",
    input_is_polsby_popper: bool = False,
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
    Plot one Polsby-Popper compactness trace.

    Typical choices:
      - kind="average": average district compactness per plan;
      - kind="minimum": worst district compactness per plan.
    """
    traces = _normalize_traces(
        traces_or_runs,
        chain_ids=chain_ids,
        output_frequency=output_frequency,
        convention=convention,
        input_is_polsby_popper=input_is_polsby_popper,
    )

    s = get_plot_style(style)

    if figsize is None:
        figsize = s.figsize_wide

    if ylabel is None:
        ylabel = _trace_ylabel(kind)

    if title is None:
        title = f"{ylabel} over proposals"

    if ax is None:
        with plot_style(s):
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    _plot_single_trace_kind(
        ax,
        traces,
        kind=kind,
        show_legend=show_legend,
    )

    apply_axis_labels(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig, ax


def plot_average_polsby_popper_trace(
    traces_or_runs: Sequence[PolsbyPopperTrace] | Mapping[str, Any] | Sequence[Any],
    **kwargs: Any,
) -> tuple[Any, Any]:
    """
    Convenience wrapper for average Polsby-Popper trace.
    """
    return plot_polsby_popper_trace(
        traces_or_runs,
        kind="average",
        **kwargs,
    )


def plot_minimum_polsby_popper_trace(
    traces_or_runs: Sequence[PolsbyPopperTrace] | Mapping[str, Any] | Sequence[Any],
    **kwargs: Any,
) -> tuple[Any, Any]:
    """
    Convenience wrapper for minimum Polsby-Popper trace.
    """
    return plot_polsby_popper_trace(
        traces_or_runs,
        kind="minimum",
        **kwargs,
    )


def plot_polsby_popper_trace_panel(
    traces_or_runs: Sequence[PolsbyPopperTrace] | Mapping[str, Any] | Sequence[Any],
    *,
    chain_ids: Sequence[str] | None = None,
    output_frequency: int = 1,
    convention: IsoperimetricConvention = "p2_over_area",
    input_is_polsby_popper: bool = False,
    kinds: Sequence[PolsbyPopperTraceKind] = ("average", "minimum"),
    title: str | None = "Polsby-Popper compactness traces",
    style: PlotStyleName | PlotStyle = "manuscript",
    figsize: tuple[float, float] | None = None,
    show_legend: bool = True,
    output_path: str | Path | None = None,
) -> tuple[Any, Sequence[Any]]:
    """
    Plot a compactness trace panel.

    By default, this creates the two most useful diagnostics:
      - average Polsby-Popper over districts in each plan;
      - minimum Polsby-Popper over districts in each plan.

    This mirrors the compactness trace idea from the MFR paper while keeping the
    package API independent of sampler internals.
    """
    traces = _normalize_traces(
        traces_or_runs,
        chain_ids=chain_ids,
        output_frequency=output_frequency,
        convention=convention,
        input_is_polsby_popper=input_is_polsby_popper,
    )

    if not kinds:
        raise PolsbyPopperPlotError("kinds must contain at least one trace kind")

    s = get_plot_style(style)

    if figsize is None:
        figsize = (5.2 * len(kinds), 3.6)

    with plot_style(s):
        fig, axes = plt.subplots(
            1,
            len(kinds),
            figsize=figsize,
            squeeze=False,
        )

    axes_list = list(axes.ravel())

    for ax, kind in zip(axes_list, kinds):
        _plot_single_trace_kind(
            ax,
            traces,
            kind=kind,
            show_legend=show_legend and ax is axes_list[0],
        )

        apply_axis_labels(
            ax,
            title=_trace_ylabel(kind),
            xlabel="Proposals",
            ylabel=_trace_ylabel(kind),
        )

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig, axes_list