import matplotlib.pyplot as plt
import pytest

from atlas_posthoc.plotting.style import (
    PlotStyleError,
    chain_color_cycle,
    get_plot_style,
    plot_style,
    save_figure,
)


def test_get_plot_style():
    style = get_plot_style("manuscript")

    assert style.name == "manuscript"
    assert style.dpi == 300


def test_get_plot_style_rejects_unknown():
    with pytest.raises(PlotStyleError):
        get_plot_style("bad-style")  # type: ignore[arg-type]


def test_chain_color_cycle_has_multiple_colors():
    colors = chain_color_cycle()

    assert len(colors) >= 4
    assert colors[0] == "C0"


def test_plot_style_context_and_save(tmp_path):
    out = tmp_path / "figures" / "test.png"

    with plot_style("manuscript") as style:
        fig, ax = plt.subplots(figsize=style.figsize_single)
        ax.plot([0, 1], [0, 1])
        saved = save_figure(fig, out)

    assert saved.exists()