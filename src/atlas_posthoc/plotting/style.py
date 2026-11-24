from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import matplotlib.pyplot as plt

from atlas_posthoc.paper import (
    apply_paper_style,
    save_figure as _save_paper_aware_figure,
)


PlotStyleName = Literal["manuscript", "presentation", "notebook"]


class PlotStyleError(Exception):
    """Base exception for plotting style helpers."""


@dataclass(frozen=True)
class PlotStyle:
    name: PlotStyleName
    figsize_single: tuple[float, float]
    figsize_wide: tuple[float, float]
    figsize_square: tuple[float, float]
    dpi: int
    font_size: int
    title_size: int
    label_size: int
    tick_size: int
    legend_size: int
    line_width: float
    marker_size: float
    alpha: float
    grid_alpha: float


MANUSCRIPT_STYLE = PlotStyle(
    name="manuscript",
    figsize_single=(6.5, 4.2),
    figsize_wide=(8.0, 4.2),
    figsize_square=(6.0, 6.0),
    dpi=300,
    font_size=10,
    title_size=12,
    label_size=10,
    tick_size=9,
    legend_size=9,
    line_width=1.5,
    marker_size=3.0,
    alpha=0.85,
    grid_alpha=0.25,
)

PRESENTATION_STYLE = PlotStyle(
    name="presentation",
    figsize_single=(8.0, 5.0),
    figsize_wide=(10.0, 5.0),
    figsize_square=(7.0, 7.0),
    dpi=200,
    font_size=13,
    title_size=16,
    label_size=13,
    tick_size=11,
    legend_size=11,
    line_width=2.0,
    marker_size=4.0,
    alpha=0.90,
    grid_alpha=0.25,
)

NOTEBOOK_STYLE = PlotStyle(
    name="notebook",
    figsize_single=(7.0, 4.5),
    figsize_wide=(9.0, 4.5),
    figsize_square=(6.5, 6.5),
    dpi=150,
    font_size=11,
    title_size=13,
    label_size=11,
    tick_size=10,
    legend_size=10,
    line_width=1.6,
    marker_size=3.5,
    alpha=0.85,
    grid_alpha=0.25,
)


STYLE_REGISTRY: dict[PlotStyleName, PlotStyle] = {
    "manuscript": MANUSCRIPT_STYLE,
    "presentation": PRESENTATION_STYLE,
    "notebook": NOTEBOOK_STYLE,
}


def get_plot_style(style: PlotStyleName | PlotStyle = "manuscript") -> PlotStyle:
    """
    Resolve a style name or PlotStyle object.
    """
    if isinstance(style, PlotStyle):
        return style

    if style not in STYLE_REGISTRY:
        raise PlotStyleError(
            f"unknown plot style {style!r}; available styles are {sorted(STYLE_REGISTRY)}"
        )

    return STYLE_REGISTRY[style]


def rc_params_for_style(style: PlotStyleName | PlotStyle = "manuscript") -> dict[str, Any]:
    """
    Matplotlib rcParams for atlas-posthoc plots.

    We intentionally avoid hard-coding a full external style sheet. These settings
    are conservative and should work across local machines, CI, and notebooks.
    """
    s = get_plot_style(style)

    return {
        "figure.dpi": s.dpi,
        "savefig.dpi": s.dpi,
        "font.size": s.font_size,
        "axes.titlesize": s.title_size,
        "axes.labelsize": s.label_size,
        "xtick.labelsize": s.tick_size,
        "ytick.labelsize": s.tick_size,
        "legend.fontsize": s.legend_size,
        "lines.linewidth": s.line_width,
        "lines.markersize": s.marker_size,
        "axes.grid": False,
        "grid.alpha": s.grid_alpha,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }


@contextmanager
def plot_style(style: PlotStyleName | PlotStyle = "manuscript") -> Iterator[PlotStyle]:
    """
    Context manager applying atlas-posthoc matplotlib defaults.

    Example
    -------
    with plot_style("manuscript") as style:
        fig, ax = plt.subplots(figsize=style.figsize_single)
        ...
    """
    s = get_plot_style(style)

    with plt.rc_context(rc_params_for_style(s)):
        # This also supports enabling paper mode after atlas_posthoc was imported.
        apply_paper_style()
        yield s


def ensure_parent_dir(path: str | Path) -> Path:
    """
    Ensure the parent directory of an output path exists.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_figure(
    fig: Any,
    path: str | Path,
    *,
    dpi: int | None = None,
    close: bool = True,
    bbox_inches: str = "tight",
    pad_inches: float = 0.05,
) -> Path:
    """
    Save a matplotlib figure with consistent defaults.
    """
    p = _save_paper_aware_figure(
        fig,
        path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
    )

    if close:
        plt.close(fig)

    return p


def apply_axis_labels(
    ax: Any,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """
    Small helper to avoid repeated label boilerplate.
    """
    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)


def despine(ax: Any) -> None:
    """
    Remove top/right spines.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def format_percent_axis(ax: Any, *, axis: Literal["x", "y"] = "y") -> None:
    """
    Format an axis with percentage labels when values are proportions.
    """
    from matplotlib.ticker import PercentFormatter

    formatter = PercentFormatter(xmax=1.0)

    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    elif axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        raise ValueError("axis must be 'x' or 'y'")


def add_zero_one_bounds(
    ax: Any,
    *,
    axis: Literal["x", "y"] = "y",
    lower: float = 0.0,
    upper: float = 1.0,
) -> None:
    """
    Set axis bounds for proportion-scale quantities.
    """
    if axis == "x":
        ax.set_xlim(lower, upper)
    elif axis == "y":
        ax.set_ylim(lower, upper)
    else:
        raise ValueError("axis must be 'x' or 'y'")


def chain_color_cycle() -> list[str]:
    """
    Return a stable color cycle for chain plots.

    We use Matplotlib's default color names rather than custom hex values so
    figures remain visually standard and easy to override.
    """
    return [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
    ]
