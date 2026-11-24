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


@pytest.mark.parametrize("style_name", ["manuscript", "presentation", "notebook"])
def test_plot_style_context_and_save(tmp_path, style_name):
    out = tmp_path / "figures" / "test.png"

    with plt.rc_context({"axes.grid": True}), plot_style(style_name) as style:
        fig, ax = plt.subplots(figsize=style.figsize_single)
        ax.plot([0, 1], [0, 1])
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        assert not any(line.get_visible() for line in gridlines)
        saved = save_figure(fig, out)

    assert saved.exists()