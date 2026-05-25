from atlas_posthoc.plotting.style import (
    MANUSCRIPT_STYLE,
    NOTEBOOK_STYLE,
    PRESENTATION_STYLE,
    STYLE_REGISTRY,
    PlotStyle,
    PlotStyleError,
    PlotStyleName,
    add_zero_one_bounds,
    apply_axis_labels,
    chain_color_cycle,
    despine,
    ensure_parent_dir,
    format_percent_axis,
    get_plot_style,
    plot_style,
    rc_params_for_style,
    save_figure,
)

from atlas_posthoc.plotting.marginals import (
    MarginalPlotError,
    plot_compactness_marginals,
    plot_ordered_marginals_by_chain,
    plot_vote_share_marginals,
)

__all__ = [
    # style
    "MANUSCRIPT_STYLE",
    "NOTEBOOK_STYLE",
    "PRESENTATION_STYLE",
    "STYLE_REGISTRY",
    "PlotStyle",
    "PlotStyleError",
    "PlotStyleName",
    "add_zero_one_bounds",
    "apply_axis_labels",
    "chain_color_cycle",
    "despine",
    "ensure_parent_dir",
    "format_percent_axis",
    "get_plot_style",
    "plot_style",
    "rc_params_for_style",
    "save_figure",

    # marginals
    "MarginalPlotError",
    "plot_compactness_marginals",
    "plot_ordered_marginals_by_chain",
    "plot_vote_share_marginals",
]