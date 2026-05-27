# Legacy code before implementing PACE; don't use; won't work!

from __future__ import annotations

from pathlib import Path

import matplotlib

# Safe for non-interactive script runs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from atlas_posthoc.pipeline.chain_observables import build_multi_chain_observables
from atlas_posthoc.diagnostics.total_variation import (
    get_tv_convergence,
    make_vote_share_bins,
    make_compactness_bins,
)
from atlas_posthoc.diagnostics.rhat import summarize_gelman_rubin_by_rank
from atlas_posthoc.io.cache import write_pipeline_cache
from atlas_posthoc.plotting.marginals import (
    plot_vote_share_marginals,
    plot_compactness_marginals,
    plot_vote_share_boxplots,
    plot_compactness_boxplots,
)
from atlas_posthoc.plotting.convergence import (
    plot_vote_share_tv_convergence,
    plot_compactness_tv_convergence,
    plot_vote_share_rhat,
    plot_compactness_rhat,
)
from atlas_posthoc.reports.summary import write_diagnostic_report


BASE = Path(
    "/Users/franklinzhou/Documents/Work/Duke/Gerrymandering/ct-gerrymandering"
)

GRAPH_PATH = BASE / "data/graph/OHpct20.json"

CHAINS = [
    BASE / "outputs/chains/soft_constraint/congressional/run13_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl",
    BASE / "outputs/chains/soft_constraint/congressional/run14_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl",
    BASE / "outputs/chains/soft_constraint/congressional/run15_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl",
    BASE / "outputs/chains/soft_constraint/congressional/run16_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl",
]

CHAIN_IDS = ["run13", "run14", "run15", "run16"]

OUTPUT_DIR = Path("outputs/smoke_full_four_chains_13_16")
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = OUTPUT_DIR / "diagnostic_report.md"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Atlas post-hoc full four-chain smoke test")
    print("=" * 80)

    print("\n[1/6] Building chain observables...")

    obs = build_multi_chain_observables(
        CHAINS,
        GRAPH_PATH,
        chain_ids=CHAIN_IDS,
        key_cols=["NAME"],
        dem_col="G20PREDEM",
        rep_col="G20PREREP",
        n_districts=15,
        output_frequency=10000,
        thin_step=1,
        require_compactness=True,
    )

    print("\nObservable summaries:")
    for chain_id, summary in obs.summaries().items():
        print(
            chain_id,
            {
                "n_vote_vectors": summary["n_vote_vectors"],
                "n_compactness_vectors": summary["n_compactness_vectors"],
                "selected_key_cols": summary["selected_key_cols"],
                "assigned_fraction": summary["key_diagnostic"]["assigned_fraction"],
                "bad_json_lines": summary["parse_summary"]["bad_json_lines"],
                "bad_assignment_lines": summary["parse_summary"]["bad_assignment_lines"],
                "compactness_errors": len(summary["compactness_errors"]),
            },
        )

    print("\nVote run shapes:")
    for chain_id, arr in obs.vote_runs.items():
        print(f"  {chain_id}: {arr.shape}")

    print("\nCompactness run shapes:")
    for chain_id, arr in obs.compactness_runs.items():
        print(f"  {chain_id}: {arr.shape}")

    assert set(obs.vote_runs.keys()) == set(CHAIN_IDS)
    assert set(obs.compactness_runs.keys()) == set(CHAIN_IDS)

    for chain_id in CHAIN_IDS:
        assert obs.vote_runs[chain_id].shape[1] == 15
        assert obs.compactness_runs[chain_id].shape[1] == 15

    print("\n[2/6] Computing TV diagnostics...")

    vote_tv = get_tv_convergence(
        obs.vote_runs,
        make_vote_share_bins(bin_width=0.002),
        output_freq=10000,
    )

    compactness_tv = get_tv_convergence(
        obs.compactness_runs,
        make_compactness_bins(obs.compactness_runs, bin_width=0.5),
        output_freq=10000,
    )

    print("  Vote final TV:", vote_tv.max_tvs[-1])
    print("  Compactness final TV:", compactness_tv.max_tvs[-1])

    print("\n[3/6] Computing R-hat diagnostics...")

    vote_rhat = summarize_gelman_rubin_by_rank(
        obs.vote_runs,
        rank_type="vote_share",
    )

    compactness_rhat = summarize_gelman_rubin_by_rank(
        obs.compactness_runs,
        rank_type="compactness",
        compactness_lower_is_better=True,
    )

    print("  Vote max classic R-hat:", vote_rhat.max_classic_r_hat)
    print("  Vote max split R-hat:", vote_rhat.max_rank_normalized_split_r_hat)
    print("  Compactness max classic R-hat:", compactness_rhat.max_classic_r_hat)
    print(
        "  Compactness max split R-hat:",
        compactness_rhat.max_rank_normalized_split_r_hat,
    )

    print("\n[4/6] Writing cache...")

    manifest = write_pipeline_cache(
        OUTPUT_DIR,
        observables=obs,
        vote_tv=vote_tv,
        compactness_tv=compactness_tv,
        vote_rhat=vote_rhat,
        compactness_rhat=compactness_rhat,
        extra={
            "graph_path": str(GRAPH_PATH),
            "chains": [str(path) for path in CHAINS],
            "chain_ids": CHAIN_IDS,
            "dem_col": "G20PREDEM",
            "rep_col": "G20PREREP",
            "n_districts": 15,
            "output_frequency": 10000,
            "key_cols": ["NAME"],
        },
    )

    print("  Manifest:", manifest.to_dict())

    print("\n[5/6] Writing plots...")

    fig, ax = plot_vote_share_marginals(
        obs.vote_runs,
        title="Ordered vote-share marginals by chain",
        y_limits=(0.20, 0.82),
        output_path=FIGURE_DIR / "oh_vote_share_marginals.png",
    )
    plt.close(fig)

    fig, ax = plot_vote_share_boxplots(
        obs.vote_runs,
        title="Ordered vote-share boxplots by chain",
        y_limits=(0.20, 0.82),
        output_path=FIGURE_DIR / "oh_vote_share_boxplots.png",
    )
    plt.close(fig)

    fig, ax = plot_compactness_marginals(
        obs.compactness_runs,
        title="Ordered compactness marginals by chain",
        y_limits=(15, 55),
        output_path=FIGURE_DIR / "oh_compactness_marginals.png",
    )
    plt.close(fig)

    fig, ax = plot_compactness_boxplots(
        obs.compactness_runs,
        title="Ordered compactness boxplots by chain",
        y_limits=(15, 55),
        output_path=FIGURE_DIR / "oh_compactness_boxplots.png",
    )
    plt.close(fig)

    fig, ax, fit = plot_vote_share_tv_convergence(
        vote_tv,
        title="Vote-share TV convergence",
        output_path=FIGURE_DIR / "oh_vote_share_tv_convergence.png",
    )
    plt.close(fig)

    fig, ax, fit = plot_compactness_tv_convergence(
        compactness_tv,
        title="Compactness TV convergence",
        output_path=FIGURE_DIR / "oh_compactness_tv_convergence.png",
    )
    plt.close(fig)

    fig, ax = plot_vote_share_rhat(
        vote_rhat,
        title="Rank-wise R-hat on ordered vote share",
        output_path=FIGURE_DIR / "oh_vote_share_rhat.png",
    )
    plt.close(fig)

    fig, ax = plot_compactness_rhat(
        compactness_rhat,
        title="Rank-wise R-hat on ordered compactness",
        output_path=FIGURE_DIR / "oh_compactness_rhat.png",
    )
    plt.close(fig)

    print("  Wrote figures to:", FIGURE_DIR)

    print("\n[6/6] Writing markdown report...")

    write_diagnostic_report(
        REPORT_PATH,
        title="Ohio congressional Atlas post-hoc diagnostics: run13-run16",
        chain_summaries=obs.summaries(),
        vote_tv=vote_tv,
        compactness_tv=compactness_tv,
        vote_rhat=vote_rhat,
        compactness_rhat=compactness_rhat,
        pace_result=None,
        notes=(
            "This smoke test was generated from four AtlasIO chains. "
            "Vote-share vectors are sorted from most Republican to least Republican. "
            "Compactness vectors are sorted from most compact / lowest score to "
            "least compact / highest score. TV and R-hat diagnostics are computed "
            "rank-wise on these ordered observables."
        ),
    )

    print("  Wrote report:", REPORT_PATH)

    print("\nSmoke test completed successfully.")
    print("\nExpected outputs:")
    print(f"  {OUTPUT_DIR / 'manifest.json'}")
    print(f"  {OUTPUT_DIR / 'cache' / 'vote_runs.parquet'}")
    print(f"  {OUTPUT_DIR / 'cache' / 'compactness_runs.parquet'}")
    print(f"  {OUTPUT_DIR / 'diagnostics' / 'vote_tv.json'}")
    print(f"  {OUTPUT_DIR / 'diagnostics' / 'compactness_tv.json'}")
    print(f"  {OUTPUT_DIR / 'diagnostics' / 'vote_rhat.json'}")
    print(f"  {OUTPUT_DIR / 'diagnostics' / 'compactness_rhat.json'}")
    print(f"  {FIGURE_DIR / 'oh_vote_share_marginals.png'}")
    print(f"  {FIGURE_DIR / 'oh_vote_share_boxplots.png'}")
    print(f"  {FIGURE_DIR / 'oh_compactness_marginals.png'}")
    print(f"  {FIGURE_DIR / 'oh_compactness_boxplots.png'}")
    print(f"  {FIGURE_DIR / 'oh_vote_share_tv_convergence.png'}")
    print(f"  {FIGURE_DIR / 'oh_compactness_tv_convergence.png'}")
    print(f"  {FIGURE_DIR / 'oh_vote_share_rhat.png'}")
    print(f"  {FIGURE_DIR / 'oh_compactness_rhat.png'}")
    print(f"  {REPORT_PATH}")


if __name__ == "__main__":
    main()