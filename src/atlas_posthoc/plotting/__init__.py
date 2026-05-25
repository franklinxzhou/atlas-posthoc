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

from atlas_posthoc.plotting.convergence import (
    ConvergencePlotError,
    PowerLawFit,
    fit_power_law_decay,
    plot_compactness_rhat,
    plot_compactness_tv_convergence,
    plot_rankwise_rhat,
    plot_tv_convergence,
    plot_vote_share_rhat,
    plot_vote_share_tv_convergence,
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

    # convergence
    "ConvergencePlotError",
    "PowerLawFit",
    "fit_power_law_decay",
    "plot_compactness_rhat",
    "plot_compactness_tv_convergence",
    "plot_rankwise_rhat",
    "plot_tv_convergence",
    "plot_vote_share_rhat",
    "plot_vote_share_tv_convergence",
]