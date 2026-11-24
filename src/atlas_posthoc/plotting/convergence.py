from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from atlas_posthoc.plotting.style import (
    PlotStyle,
    PlotStyleName,
    apply_axis_labels,
    chain_color_cycle,
    despine,
    get_plot_style,
    plot_style,
    save_figure,
)


RHatMetric = Literal["classic_r_hat", "rank_normalized_split_r_hat"]


class ConvergencePlotError(Exception):
    """Base exception for convergence plotting."""


@dataclass
class PowerLawFit:
    slope: float
    intercept: float
    order: float
    x_fit: list[float]
    y_fit: list[float]
    n_points: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_tv_arrays(tv_result: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract proposal counts and max TV values from a TVConvergenceResult-like object.

    Accepts:
      - package TVConvergenceResult with `.proposals` and `.max_tvs`;
      - dictionary with keys `proposals` and `max_tvs`;
      - tuple/list `(proposals, max_tvs)`.
    """
    if hasattr(tv_result, "proposals") and hasattr(tv_result, "max_tvs"):
        proposals = np.asarray(tv_result.proposals, dtype=float)
        max_tvs = np.asarray(tv_result.max_tvs, dtype=float)
        return proposals, max_tvs

    if isinstance(tv_result, Mapping):
        if "proposals" not in tv_result or "max_tvs" not in tv_result:
            raise ConvergencePlotError(
                "TV result mapping must contain `proposals` and `max_tvs`"
            )
        return (
            np.asarray(tv_result["proposals"], dtype=float),
            np.asarray(tv_result["max_tvs"], dtype=float),
        )

    if isinstance(tv_result, (tuple, list)) and len(tv_result) == 2:
        return (
            np.asarray(tv_result[0], dtype=float),
            np.asarray(tv_result[1], dtype=float),
        )

    raise ConvergencePlotError(
        "could not extract proposals/max_tvs from TV result"
    )


def fit_power_law_decay(
    x: Sequence[float],
    y: Sequence[float],
    *,
    min_points: int = 3,
) -> PowerLawFit:
    """
    Fit a power-law decay model:

        y ≈ C x^{-alpha}

    by linear regression on log(x), log(y).

    Returns
    -------
    PowerLawFit
        `order` is alpha, so larger positive values indicate faster decay.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    mask = (
        np.isfinite(x_arr)
        & np.isfinite(y_arr)
        & (x_arr > 0)
        & (y_arr > 0)
    )

    x_use = x_arr[mask]
    y_use = y_arr[mask]

    if x_use.size < min_points:
        raise ConvergencePlotError(
            f"need at least {min_points} positive finite points for power-law fit"
        )

    log_x = np.log(x_use)
    log_y = np.log(y_use)

    slope, intercept = np.polyfit(log_x, log_y, deg=1)

    x_fit = np.linspace(x_use.min(), x_use.max(), 200)
    y_fit = np.exp(intercept) * x_fit**slope

    return PowerLawFit(
        slope=float(slope),
        intercept=float(intercept),
        order=float(-slope),
        x_fit=x_fit.tolist(),
        y_fit=y_fit.tolist(),
        n_points=int(x_use.size),
    )


def _default_tv_label(observable: str) -> str:
    if observable == "vote_share":
        return "Max. avg. pairwise total variation on vote share"

    if observable == "compactness":
        return "Max. avg. pairwise total variation on iso. score"

    return "Max. avg. pairwise total variation"


def plot_tv_convergence(
    tv_result: Any,
    *,
    observable: Literal["vote_share", "compactness", "generic"] = "generic",
    label: str | None = None,
    title: str | None = None,
    xlabel: str = "Proposals",
    ylabel: str = "Max. avg. total variation",
    style: PlotStyleName | PlotStyle = "manuscript",
    fit_power_law: bool = True,
    fit_label_prefix: str = "Best Fit",
    loglog: bool = True,
    marker: str | None = None,
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any, PowerLawFit | None]:
    """
    Plot max average pairwise TV convergence versus proposals.

    This targets the CT-style convergence plot:
      - x-axis: proposals;
      - y-axis: max average pairwise total variation;
      - optional power-law best-fit line.
    """
    proposals, max_tvs = _extract_tv_arrays(tv_result)

    if proposals.shape != max_tvs.shape:
        raise ConvergencePlotError(
            f"proposals and max_tvs must have same shape; "
            f"got {proposals.shape} and {max_tvs.shape}"
        )

    if proposals.size == 0:
        raise ConvergencePlotError("TV result contains no points")

    s = get_plot_style(style)

    if label is None:
        label = _default_tv_label(observable)

    if ax is None:
        with plot_style(s):
            fig, ax = plt.subplots(figsize=s.figsize_wide)
    else:
        fig = ax.figure

    colors = chain_color_cycle()
    empirical_color = colors[0]
    fit_color = colors[1]

    finite_mask = np.isfinite(proposals) & np.isfinite(max_tvs)

    if loglog:
        positive_x_mask = finite_mask & (proposals > 0)
        positive_y_mask = positive_x_mask & (max_tvs > 0)

        if positive_y_mask.sum() > 0:
            # Standard CT-style log-log plot.
            plot_x = proposals[positive_y_mask]
            plot_y = max_tvs[positive_y_mask]

            ax.set_xscale("log")
            ax.set_yscale("log")
            can_fit_power_law = True
        else:
            # All TV values are zero or nonpositive.
            # A log y-axis is not meaningful, so fall back to log-x / linear-y.
            plot_x = proposals[positive_x_mask]
            plot_y = max_tvs[positive_x_mask]

            if plot_x.size == 0:
                raise ConvergencePlotError(
                    "TV convergence plot has no positive finite proposal counts"
                )

            ax.set_xscale("log")
            can_fit_power_law = False

            # Give the zero line visible breathing room.
            y_pad = 0.05
            ax.set_ylim(-y_pad, y_pad)
    else:
        if finite_mask.sum() == 0:
            raise ConvergencePlotError("TV convergence plot has no finite values")

        plot_x = proposals[finite_mask]
        plot_y = max_tvs[finite_mask]
        can_fit_power_law = True

    ax.plot(
        plot_x,
        plot_y,
        marker=marker,
        color=empirical_color,
        linewidth=s.line_width,
        label=label,
        zorder=3,
    )

    fit: PowerLawFit | None = None

    if fit_power_law and can_fit_power_law:
        try:
            fit = fit_power_law_decay(plot_x, plot_y)
            ax.plot(
                fit.x_fit,
                fit.y_fit,
                color=fit_color,
                linewidth=s.line_width,
                alpha=0.75,
                label=f"{fit_label_prefix}, Order {fit.order:.2f}",
                zorder=2,
            )
        except ConvergencePlotError:
            fit = None

    apply_axis_labels(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    ax.grid(False, which="both")
    despine(ax)
    ax.legend(loc="best", frameon=True, fancybox=False)

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig, ax, fit


def plot_vote_share_tv_convergence(
    tv_result: Any,
    *,
    style: PlotStyleName | PlotStyle = "manuscript",
    title: str | None = None,
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any, PowerLawFit | None]:
    """
    Convenience wrapper for vote-share TV convergence.
    """
    return plot_tv_convergence(
        tv_result,
        observable="vote_share",
        title=title,
        style=style,
        ax=ax,
        output_path=output_path,
    )


def plot_compactness_tv_convergence(
    tv_result: Any,
    *,
    style: PlotStyleName | PlotStyle = "manuscript",
    title: str | None = None,
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any, PowerLawFit | None]:
    """
    Convenience wrapper for compactness / isoperimetric-score TV convergence.
    """
    return plot_tv_convergence(
        tv_result,
        observable="compactness",
        title=title,
        style=style,
        ax=ax,
        output_path=output_path,
    )


def _extract_rhat_rows(rhat_summary: Any) -> list[dict[str, Any]]:
    """
    Extract rank-wise R-hat rows from an RHatSummary-like object or dict.
    """
    if hasattr(rhat_summary, "rows"):
        return [dict(row) for row in rhat_summary.rows]

    if isinstance(rhat_summary, Mapping):
        if "rows" not in rhat_summary:
            raise ConvergencePlotError("R-hat summary mapping must contain `rows`")
        return [dict(row) for row in rhat_summary["rows"]]

    if isinstance(rhat_summary, Sequence) and not isinstance(
        rhat_summary, (str, bytes, bytearray)
    ):
        return [dict(row) for row in rhat_summary]

    raise ConvergencePlotError("could not extract rows from R-hat summary")


def _metric_label(metric: RHatMetric) -> str:
    if metric == "classic_r_hat":
        return "Classic R-hat"

    if metric == "rank_normalized_split_r_hat":
        return "Rank-normalized split R-hat"

    raise ValueError(f"unknown R-hat metric: {metric!r}")


def plot_rankwise_rhat(
    rhat_summary: Any,
    *,
    metric: RHatMetric = "rank_normalized_split_r_hat",
    title: str | None = None,
    xlabel: str = "Ordered district rank",
    ylabel: str | None = None,
    threshold: float | None = 1.05,
    style: PlotStyleName | PlotStyle = "manuscript",
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """
    Plot rank-wise R-hat values.

    This is useful as a compact visual companion to the rank-wise R-hat table.
    """
    rows = _extract_rhat_rows(rhat_summary)

    if not rows:
        raise ConvergencePlotError("R-hat summary has no rows")

    ranks = np.asarray([row["rank"] for row in rows], dtype=int)
    values = np.asarray([row[metric] for row in rows], dtype=float)

    s = get_plot_style(style)

    if ylabel is None:
        ylabel = _metric_label(metric)

    if ax is None:
        with plot_style(s):
            fig, ax = plt.subplots(figsize=s.figsize_wide)
    else:
        fig = ax.figure

    colors = chain_color_cycle()

    ax.plot(
        ranks,
        values,
        marker="o",
        color=colors[0],
        linewidth=s.line_width,
        markersize=s.marker_size,
        label=ylabel,
    )

    if threshold is not None:
        ax.axhline(
            threshold,
            color=colors[1],
            linestyle="--",
            linewidth=1.0,
            label=f"Threshold {threshold:g}",
        )

    ax.set_xticks(ranks)
    ax.set_ylim(bottom=min(0.99, float(np.nanmin(values)) - 0.02))

    apply_axis_labels(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    ax.grid(False, which="both")
    despine(ax)
    ax.legend(loc="best", frameon=True, fancybox=False)

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig, ax


def plot_vote_share_rhat(
    rhat_summary: Any,
    *,
    metric: RHatMetric = "rank_normalized_split_r_hat",
    style: PlotStyleName | PlotStyle = "manuscript",
    title: str | None = "Rank-wise R-hat on ordered vote share",
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    return plot_rankwise_rhat(
        rhat_summary,
        metric=metric,
        title=title,
        xlabel="Districts ordered from most to least Republican",
        style=style,
        ax=ax,
        output_path=output_path,
    )


def plot_compactness_rhat(
    rhat_summary: Any,
    *,
    metric: RHatMetric = "rank_normalized_split_r_hat",
    style: PlotStyleName | PlotStyle = "manuscript",
    title: str | None = "Rank-wise R-hat on ordered compactness",
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    return plot_rankwise_rhat(
        rhat_summary,
        metric=metric,
        title=title,
        xlabel="Districts ordered from most to least compact",
        style=style,
        ax=ax,
        output_path=output_path,
    )