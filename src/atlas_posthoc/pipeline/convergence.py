"""Run the standard chain convergence analysis and write its outputs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt

from atlas_posthoc.config import ProjectConfig
from atlas_posthoc.diagnostics.rhat import summarize_gelman_rubin_by_rank
from atlas_posthoc.diagnostics.total_variation import (
    get_tv_convergence,
    make_compactness_bins,
    make_vote_share_bins,
)
from atlas_posthoc.io.cache import write_pipeline_cache
from atlas_posthoc.paper import paper_mode
from atlas_posthoc.pipeline.chain_observables import build_multi_chain_observables
from atlas_posthoc.plotting.convergence import (
    plot_compactness_rhat,
    plot_compactness_tv_convergence,
    plot_vote_share_rhat,
    plot_vote_share_tv_convergence,
)
from atlas_posthoc.plotting.marginals import (
    plot_compactness_boxplots,
    plot_compactness_marginals,
    plot_vote_share_boxplots,
    plot_vote_share_marginals,
)
from atlas_posthoc.reports.summary import write_diagnostic_report


def run_convergence(config: ProjectConfig, *, paper: bool | None = None) -> Path:
    """Build chain observables once and use them for diagnostics, figures, and the report."""
    config.check_inputs()
    options = config.convergence
    proposal_interval = config.output_frequency * config.thin_step

    observables = build_multi_chain_observables(
        config.chain_files,
        config.graph_json,
        chain_ids=config.chain_ids,
        key_cols=config.key_cols,
        dem_col=config.dem_col,
        rep_col=config.rep_col,
        n_districts=config.n_districts,
        output_frequency=proposal_interval,
        thin_step=config.thin_step,
        compute_compactness=config.compute_compactness,
        require_compactness=config.compute_compactness,
    )
    vote_runs = observables.vote_runs
    compactness_runs = observables.compactness_runs
    vote_tv = get_tv_convergence(
        vote_runs,
        make_vote_share_bins(bin_width=options.vote_bin_width),
        output_freq=proposal_interval,
    )
    vote_rhat = summarize_gelman_rubin_by_rank(vote_runs, rank_type="vote_share")
    compactness_tv = compactness_rhat = None
    if compactness_runs:
        compactness_tv = get_tv_convergence(
            compactness_runs,
            make_compactness_bins(
                compactness_runs, bin_width=options.compactness_bin_width
            ),
            output_freq=proposal_interval,
        )
        compactness_rhat = summarize_gelman_rubin_by_rank(
            compactness_runs, rank_type="compactness"
        )

    diagnostics = {
        "vote_tv": vote_tv,
        "compactness_tv": compactness_tv,
        "vote_rhat": vote_rhat,
        "compactness_rhat": compactness_rhat,
    }
    use_paper_mode = config.paper_mode if paper is None else paper
    write_pipeline_cache(
        config.output_root,
        observables=observables,
        **diagnostics,
        extra={
            "config": config.path,
            "graph_path": config.graph_json,
            "chains": config.chain_files,
            "chain_ids": list(observables.chains),
            "key_cols": config.key_cols,
            "dem_col": config.dem_col,
            "rep_col": config.rep_col,
            "n_districts": config.n_districts,
            "output_frequency": config.output_frequency,
            "thin_step": config.thin_step,
            "compute_compactness": config.compute_compactness,
            "paper_mode": use_paper_mode,
            "convergence": asdict(options),
        },
    )

    plots = [
        (
            plot_vote_share_marginals,
            vote_runs,
            "vote_share_marginals",
            {
                "title": "Ordered vote-share marginals by chain",
                "y_limits": options.vote_y_limits,
            },
        ),
        (
            plot_vote_share_boxplots,
            vote_runs,
            "vote_share_boxplots",
            {
                "title": "Ordered vote-share boxplots by chain",
                "y_limits": options.vote_y_limits,
            },
        ),
        (
            plot_vote_share_tv_convergence,
            vote_tv,
            "vote_share_tv_convergence",
            {
                "title": "Vote-share TV convergence",
            },
        ),
        (plot_vote_share_rhat, vote_rhat, "vote_share_rhat", {}),
    ]
    if compactness_runs:
        plots.extend(
            [
                (
                    plot_compactness_marginals,
                    compactness_runs,
                    "compactness_marginals",
                    {
                        "title": "Ordered compactness marginals by chain",
                        "y_limits": options.compactness_y_limits,
                    },
                ),
                (
                    plot_compactness_boxplots,
                    compactness_runs,
                    "compactness_boxplots",
                    {
                        "title": "Ordered compactness boxplots by chain",
                        "y_limits": options.compactness_y_limits,
                    },
                ),
                (
                    plot_compactness_tv_convergence,
                    compactness_tv,
                    "compactness_tv_convergence",
                    {
                        "title": "Compactness TV convergence",
                    },
                ),
                (plot_compactness_rhat, compactness_rhat, "compactness_rhat", {}),
            ]
        )

    figure_dir = config.output_root / "figures"
    with paper_mode(use_paper_mode):
        for plot, data, name, kwargs in plots:
            output_path = figure_dir / f"{options.figure_prefix}{name}.png"
            figure, *_ = plot(data, output_path=output_path, **kwargs)
            plt.close(figure)

    return write_diagnostic_report(
        config.output_root / "diagnostic_report.md",
        title=options.report_title,
        chain_summaries=observables.summaries(),
        **diagnostics,
    )
