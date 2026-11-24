import matplotlib.pyplot as plt
import numpy as np
import pytest

from atlas_posthoc.diagnostics.rhat import summarize_gelman_rubin_by_rank
from atlas_posthoc.diagnostics.total_variation import get_tv_convergence
from atlas_posthoc.plotting.convergence import (
    fit_power_law_decay,
    plot_compactness_rhat,
    plot_compactness_tv_convergence,
    plot_vote_share_rhat,
    plot_vote_share_tv_convergence,
)


def test_fit_power_law_decay():
    x = np.array([10, 100, 1000, 10000], dtype=float)
    y = x ** (-0.5)

    fit = fit_power_law_decay(x, y)

    assert fit.order == pytest.approx(0.5)
    assert fit.n_points == 4


def test_plot_vote_share_tv_convergence(tmp_path):
    runs = {
        "run1": np.array([[0.1, 0.6], [0.2, 0.7], [0.3, 0.8], [0.4, 0.9]]),
        "run2": np.array([[0.1, 0.6], [0.2, 0.7], [0.35, 0.85], [0.4, 0.9]]),
    }

    tv = get_tv_convergence(
        runs,
        bins=np.array([0.0, 0.5, 1.0]),
        output_freq=10,
        snapshots=[2, 3, 4],
    )

    out = tmp_path / "vote_tv.png"

    fig, ax, fit = plot_vote_share_tv_convergence(
        tv,
        output_path=out,
    )

    assert out.exists()
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "linear"
    assert fit is None
    plt.close(fig)


def test_plot_compactness_tv_convergence(tmp_path):
    tv = {
        "proposals": [10, 100, 1000, 10000],
        "max_tvs": [0.8, 0.4, 0.15, 0.05],
    }

    out = tmp_path / "compactness_tv.png"
    fig, ax = plt.subplots()
    ax.minorticks_on()
    ax.grid(True, which="both")

    _, _, fit = plot_compactness_tv_convergence(
        tv,
        ax=ax,
        output_path=out,
    )

    assert out.exists()
    assert fit is not None
    for axis in (ax.xaxis, ax.yaxis):
        for tick in axis.get_major_ticks() + axis.get_minor_ticks():
            assert not tick.gridline.get_visible()
    plt.close(fig)


def test_plot_vote_share_rhat(tmp_path):
    run = np.array(
        [
            [0.30, 0.70],
            [0.31, 0.71],
            [0.32, 0.72],
            [0.33, 0.73],
        ]
    )

    summary = summarize_gelman_rubin_by_rank(
        [run, run + 0.01, run - 0.01, run + 0.02],
        rank_type="vote_share",
    )

    out = tmp_path / "vote_rhat.png"

    fig, ax = plot_vote_share_rhat(
        summary,
        output_path=out,
    )

    assert out.exists()
    assert "Republican" in ax.get_xlabel()
    plt.close(fig)


def test_plot_compactness_rhat(tmp_path):
    run = np.array(
        [
            [20.0, 40.0],
            [21.0, 41.0],
            [22.0, 42.0],
            [23.0, 43.0],
        ]
    )

    summary = summarize_gelman_rubin_by_rank(
        [run, run + 0.5, run - 0.5, run + 1.0],
        rank_type="compactness",
    )

    out = tmp_path / "compactness_rhat.png"

    fig, ax = plot_compactness_rhat(
        summary,
        output_path=out,
    )

    assert out.exists()
    assert "compact" in ax.get_xlabel()
    plt.close(fig)