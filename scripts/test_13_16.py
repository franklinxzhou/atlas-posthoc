"""
Call in the terminal: python3 scripts/test_13_16.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

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
    os.environ.get(
        "ATLAS_POSTHOC_BASE",
        "/Users/franklinzhou/Documents/Work/Duke/Gerrymandering/ct-gerrymandering",
    )
)

GRAPH_PATH = Path(
    os.environ.get(
        "ATLAS_POSTHOC_GRAPH_PATH",
        str(BASE / "data/graph/OHpct20.json"),
    )
)
GEOGRAPHY_MODE = os.environ.get("ATLAS_POSTHOC_GEOGRAPHY_MODE", "COUSUB").upper()

CHAINS = [
    BASE / "outputs/chains/soft_constraint/congressional/run37_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl",
    BASE / "outputs/chains/soft_constraint/congressional/run38_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl",
    BASE / "outputs/chains/soft_constraint/congressional/run39_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl",
    BASE / "outputs/chains/soft_constraint/congressional/run40_cntyw20.0_mcdw20.0_isoWeight0.25.jsonl",
]

CHAIN_IDS = ["run37", "run38", "run39", "run40"]

N_DISTRICTS = 15
OUTPUT_FREQUENCY = 10000

OUTPUT_DIR = Path(
    os.environ.get(
        "ATLAS_POSTHOC_OUTPUT_DIR",
        "outputs/smoke_full_four_chains_37_40",
    )
)
FIGURE_DIR = OUTPUT_DIR / "figures"
DIAGNOSTIC_DIR = OUTPUT_DIR / "diagnostics"
REPORT_PATH = OUTPUT_DIR / "diagnostic_report.md"


def safe_to_dict(x: Any) -> Any:
    if hasattr(x, "to_dict"):
        return x.to_dict()
    return x


def validate_observables(obs) -> None:
    """Basic checks that the parser produced usable arrays."""
    assert set(obs.vote_runs.keys()) == set(CHAIN_IDS)
    assert set(obs.compactness_runs.keys()) == set(CHAIN_IDS)

    for chain_id in CHAIN_IDS:
        vote = obs.vote_runs[chain_id]
        comp = obs.compactness_runs[chain_id]

        assert vote.shape[0] > 0, f"{chain_id}: no vote vectors parsed"
        assert comp.shape[0] > 0, f"{chain_id}: no compactness vectors parsed"

        assert vote.shape[1] == N_DISTRICTS, (
            f"{chain_id}: vote vector width is {vote.shape[1]}, "
            f"expected {N_DISTRICTS}"
        )
        assert comp.shape[1] == N_DISTRICTS, (
            f"{chain_id}: compactness vector width is {comp.shape[1]}, "
            f"expected {N_DISTRICTS}"
        )

        assert np.isfinite(vote).all(), f"{chain_id}: non-finite vote values"
        assert np.isfinite(comp).all(), f"{chain_id}: non-finite compactness values"

        assert ((0 <= vote) & (vote <= 1)).all(), (
            f"{chain_id}: vote share outside [0, 1]"
        )


def print_parse_warnings(obs) -> None:
    """Warn about parsing issues without killing the smoke test."""
    print("\nParser quality checks:")

    for chain_id, summary in obs.summaries().items():
        assigned_fraction = summary["key_diagnostic"]["assigned_fraction"]
        bad_json_lines = summary["parse_summary"]["bad_json_lines"]
        bad_assignment_lines = summary["parse_summary"]["bad_assignment_lines"]
        compactness_errors = len(summary["compactness_errors"])

        print(
            f"  {chain_id}: "
            f"assigned_fraction={assigned_fraction}, "
            f"bad_json_lines={bad_json_lines}, "
            f"bad_assignment_lines={bad_assignment_lines}, "
            f"compactness_errors={compactness_errors}"
        )

        if assigned_fraction < 0.99:
            print(f"    WARNING: {chain_id} has assigned_fraction < 0.99")

        if bad_json_lines > 0:
            print(f"    WARNING: {chain_id} has bad JSON lines")

        if bad_assignment_lines > 0:
            print(f"    WARNING: {chain_id} has bad assignment lines")

        if compactness_errors > 0:
            print(f"    WARNING: {chain_id} has compactness errors")




def write_cache(
    *,
    obs,
    vote_tv,
    compactness_tv,
    vote_rhat,
    compactness_rhat,
):
    """Write the observable diagnostics and run metadata."""
    extra = {
        "graph_path": str(GRAPH_PATH),
        "geography_mode": GEOGRAPHY_MODE,
        "chains": [str(path) for path in CHAINS],
        "chain_ids": CHAIN_IDS,
        "dem_col": "G20PREDEM",
        "rep_col": "G20PREREP",
        "n_districts": N_DISTRICTS,
        "output_frequency": OUTPUT_FREQUENCY,
        "key_cols": ["NAME"],
    }

    return write_pipeline_cache(
        OUTPUT_DIR,
        observables=obs,
        vote_tv=vote_tv,
        compactness_tv=compactness_tv,
        vote_rhat=vote_rhat,
        compactness_rhat=compactness_rhat,
        extra=extra,
    )


def assert_expected_outputs_exist(expected_paths: list[Path]) -> None:
    missing = [path for path in expected_paths if not path.exists()]
    if missing:
        missing_str = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing expected output files:\n{missing_str}")


def main() -> None:
    if GEOGRAPHY_MODE not in {"COUSUB", "PLACE"}:
        raise ValueError("ATLAS_POSTHOC_GEOGRAPHY_MODE must be COUSUB or PLACE")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Atlas post-hoc four-chain smoke test")
    print("=" * 80)

    print("\nConfiguration:")
    print(f"  BASE: {BASE}")
    print(f"  GRAPH_PATH: {GRAPH_PATH}")
    print(f"  GEOGRAPHY_MODE: {GEOGRAPHY_MODE}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")

    print("\n[1/6] Building chain observables...")

    obs = build_multi_chain_observables(
        CHAINS,
        GRAPH_PATH,
        chain_ids=CHAIN_IDS,
        key_cols=["NAME"],
        dem_col="G20PREDEM",
        rep_col="G20PREREP",
        n_districts=N_DISTRICTS,
        output_frequency=OUTPUT_FREQUENCY,
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

    validate_observables(obs)
    print_parse_warnings(obs)

    print("\n[2/6] Computing TV diagnostics...")

    vote_tv = get_tv_convergence(
        obs.vote_runs,
        make_vote_share_bins(bin_width=0.002),
        output_freq=OUTPUT_FREQUENCY,
    )

    compactness_tv = get_tv_convergence(
        obs.compactness_runs,
        make_compactness_bins(obs.compactness_runs, bin_width=0.5),
        output_freq=OUTPUT_FREQUENCY,
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

    manifest = write_cache(
        obs=obs,
        vote_tv=vote_tv,
        compactness_tv=compactness_tv,
        vote_rhat=vote_rhat,
        compactness_rhat=compactness_rhat,
    )

    print("  Manifest:", safe_to_dict(manifest))

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
        title="Ohio congressional Atlas post-hoc diagnostics: run37-run40",
        chain_summaries=obs.summaries(),
        vote_tv=vote_tv,
        compactness_tv=compactness_tv,
        vote_rhat=vote_rhat,
        compactness_rhat=compactness_rhat,
        notes=(
            "This smoke test was generated from four AtlasIO chains. "
            "Vote-share vectors are sorted from most Republican to least Republican. "
            "Compactness vectors are sorted from most compact / lowest score to "
            "least compact / highest score. TV and R-hat diagnostics are computed "
            "rank-wise on these ordered observables. "
        ),
    )

    print("  Wrote report:", REPORT_PATH)

    expected_paths = [
        OUTPUT_DIR / "manifest.json",
        OUTPUT_DIR / "cache" / "vote_runs.parquet",
        OUTPUT_DIR / "cache" / "compactness_runs.parquet",
        OUTPUT_DIR / "diagnostics" / "vote_tv.json",
        OUTPUT_DIR / "diagnostics" / "compactness_tv.json",
        OUTPUT_DIR / "diagnostics" / "vote_rhat.json",
        OUTPUT_DIR / "diagnostics" / "compactness_rhat.json",
        FIGURE_DIR / "oh_vote_share_marginals.png",
        FIGURE_DIR / "oh_vote_share_boxplots.png",
        FIGURE_DIR / "oh_compactness_marginals.png",
        FIGURE_DIR / "oh_compactness_boxplots.png",
        FIGURE_DIR / "oh_vote_share_tv_convergence.png",
        FIGURE_DIR / "oh_compactness_tv_convergence.png",
        FIGURE_DIR / "oh_vote_share_rhat.png",
        FIGURE_DIR / "oh_compactness_rhat.png",
        REPORT_PATH,
    ]

    assert_expected_outputs_exist(expected_paths)

    print("\nSmoke test completed successfully.")
    print("\nExpected outputs:")
    for path in expected_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
