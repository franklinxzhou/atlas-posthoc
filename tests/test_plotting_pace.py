import matplotlib.pyplot as plt
import numpy as np

from atlas_posthoc.diagnostics.pace import pace_from_contingency
from atlas_posthoc.plotting.pace import (
    plot_pace_contingency_heatmap,
    plot_pace_count_heatmap,
    plot_pace_probability_heatmap,
)


def make_pace_result():
    table = np.array(
        [
            [8, 2, 0],
            [7, 3, 0],
            [1, 2, 7],
            [1, 1, 8],
        ]
    )

    return pace_from_contingency(
        table,
        chain_names=["run1", "run2", "run3", "run4"],
        labels=[10, 20, 30],
        threshold=0.1,
        apply_correction=False,
    )


def test_plot_pace_probability_heatmap(tmp_path):
    out = tmp_path / "pace_probability.png"

    fig, ax = plot_pace_probability_heatmap(
        make_pace_result(),
        output_path=out,
    )

    assert out.exists()
    assert ax.get_xlabel() == "PACE cell / cluster"
    assert ax.get_ylabel() == "Chain"

    plt.close(fig)


def test_plot_pace_count_heatmap(tmp_path):
    out = tmp_path / "pace_counts.png"

    fig, ax = plot_pace_count_heatmap(
        make_pace_result(),
        output_path=out,
    )

    assert out.exists()

    plt.close(fig)


def test_plot_pace_contingency_heatmap_from_dict(tmp_path):
    pace = make_pace_result().to_dict()
    out = tmp_path / "pace_from_dict.png"

    fig, ax = plot_pace_contingency_heatmap(
        pace,
        value="probability",
        output_path=out,
    )

    assert out.exists()

    plt.close(fig)


def test_plot_pace_heatmap_collapses_cells(tmp_path):
    table = np.array(
        [
            [10, 8, 6, 4, 2, 1],
            [9, 7, 5, 3, 2, 1],
        ]
    )

    pace = pace_from_contingency(
        table,
        chain_names=["a", "b"],
        labels=[0, 1, 2, 3, 4, 5],
    )

    out = tmp_path / "pace_collapsed.png"

    fig, ax = plot_pace_probability_heatmap(
        pace,
        max_cells=4,
        output_path=out,
    )

    # max_cells=4 means 3 largest cells + other.
    assert len(ax.get_xticklabels()) == 4
    assert out.exists()

    plt.close(fig)