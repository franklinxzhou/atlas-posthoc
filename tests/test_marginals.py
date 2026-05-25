import matplotlib.pyplot as plt
import numpy as np

from atlas_posthoc.plotting.marginals import (
    plot_compactness_marginals,
    plot_ordered_marginals_by_chain,
    plot_vote_share_marginals,
)

from atlas_posthoc.plotting.marginals import (
    plot_compactness_boxplots,
    plot_ordered_boxplots_by_chain,
    plot_vote_share_boxplots,
)


def make_vote_runs():
    return {
        "Run 1": np.array(
            [
                [0.52, 0.58, 0.60],
                [0.53, 0.57, 0.61],
                [0.51, 0.59, 0.60],
            ]
        ),
        "Run 2": np.array(
            [
                [0.50, 0.56, 0.59],
                [0.51, 0.57, 0.60],
                [0.50, 0.58, 0.59],
            ]
        ),
    }


def make_compactness_runs():
    return {
        "Run 1": np.array(
            [
                [30.0, 35.0, 42.0],
                [31.0, 36.0, 43.0],
                [29.0, 34.0, 41.0],
            ]
        ),
        "Run 2": np.array(
            [
                [28.0, 34.0, 40.0],
                [29.0, 35.0, 41.0],
                [30.0, 35.5, 40.5],
            ]
        ),
    }


def test_plot_vote_share_marginals(tmp_path):
    out = tmp_path / "vote_marginals.png"

    fig, ax = plot_vote_share_marginals(
        make_vote_runs(),
        reference_values=[0.51, 0.575, 0.605],
        output_path=out,
    )

    assert out.exists()
    assert len(ax.get_xticks()) == 3
    plt.close(fig)


def test_plot_compactness_marginals(tmp_path):
    out = tmp_path / "compactness_marginals.png"

    fig, ax = plot_compactness_marginals(
        make_compactness_runs(),
        reference_values=[29.5, 35.2, 41.5],
        output_path=out,
    )

    assert out.exists()
    assert len(ax.get_xticks()) == 3
    plt.close(fig)


def test_plot_ordered_marginals_by_chain_generic(tmp_path):
    out = tmp_path / "generic_marginals.png"

    runs = {
        "Run 1": np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
        "Run 2": np.array([[0.8, 1.8], [0.85, 1.85], [0.9, 1.9]]),
    }

    fig, ax = plot_ordered_marginals_by_chain(
        runs,
        observable_kind="generic",
        xlabel="Rank",
        ylabel="Value",
        output_path=out,
    )

    assert out.exists()
    assert ax.get_xlabel() == "Rank"
    assert ax.get_ylabel() == "Value"
    plt.close(fig)

def test_plot_vote_share_boxplots(tmp_path):
    out = tmp_path / "vote_boxplots.png"

    fig, ax = plot_vote_share_boxplots(
        make_vote_runs(),
        reference_values=[0.51, 0.575, 0.605],
        output_path=out,
    )

    assert out.exists()
    assert len(ax.get_xticks()) == 3
    plt.close(fig)


def test_plot_compactness_boxplots(tmp_path):
    out = tmp_path / "compactness_boxplots.png"

    fig, ax = plot_compactness_boxplots(
        make_compactness_runs(),
        reference_values=[29.5, 35.2, 41.5],
        output_path=out,
    )

    assert out.exists()
    assert len(ax.get_xticks()) == 3
    plt.close(fig)


def test_plot_ordered_boxplots_by_chain_generic(tmp_path):
    out = tmp_path / "generic_boxplots.png"

    runs = {
        "Run 1": np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
        "Run 2": np.array([[0.8, 1.8], [0.85, 1.85], [0.9, 1.9]]),
    }

    fig, ax = plot_ordered_boxplots_by_chain(
        runs,
        observable_kind="generic",
        xlabel="Rank",
        ylabel="Value",
        output_path=out,
    )

    assert out.exists()
    assert ax.get_xlabel() == "Rank"
    assert ax.get_ylabel() == "Value"
    plt.close(fig)