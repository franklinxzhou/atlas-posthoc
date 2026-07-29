from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from atlas_posthoc.paper import (
    PANEL_LABEL_GID,
    PAPER_MODE_ENV,
    paper_mode,
    paper_mode_enabled,
    paper_output_path,
)
from atlas_posthoc.plotting.style import save_figure


def _panel_labels(figure: object) -> list[str]:
    return [
        artist.get_text()
        for artist in figure.findobj()
        if getattr(artist, "get_gid", lambda: None)() == PANEL_LABEL_GID
    ]


def test_paper_mode_saves_pdf_without_titles_and_adds_panel_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PAPER_MODE_ENV, raising=False)

    with paper_mode():
        figure, axes = plt.subplots(1, 2)
        figure.suptitle("Figure title")
        axes[0].set_title("Left title")
        axes[1].set_title("Right title")
        axes[0].plot([0, 1], [0, 1])
        axes[1].plot([0, 1], [1, 0])

        destination = save_figure(
            figure,
            tmp_path / "figure.png",
            dpi=72,
            close=False,
        )

        assert paper_mode_enabled()
        assert destination == tmp_path / "figure.pdf"
        assert destination.read_bytes().startswith(b"%PDF")
        assert figure._suptitle.get_text() == ""
        assert [axis.get_title() for axis in axes] == ["", ""]
        assert _panel_labels(figure) == ["(a)", "(b)"]
        assert mpl.rcParams["font.sans-serif"][0] == "Helvetica Neue"
        plt.close(figure)

    assert not paper_mode_enabled()


def test_paper_mode_does_not_label_three_panel_figure(tmp_path: Path) -> None:
    with paper_mode():
        figure, _ = plt.subplots(1, 3)
        destination = save_figure(
            figure,
            tmp_path / "three-panels.svg",
            close=False,
        )

        assert destination == tmp_path / "three-panels.pdf"
        assert not _panel_labels(figure)
        plt.close(figure)


def test_paper_mode_does_not_label_two_stacked_panels(tmp_path: Path) -> None:
    with paper_mode():
        figure, _ = plt.subplots(2, 1)
        save_figure(figure, tmp_path / "stacked.png", close=False)

        assert not _panel_labels(figure)
        plt.close(figure)


def test_default_save_behavior_remains_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PAPER_MODE_ENV, raising=False)
    figure, axis = plt.subplots()
    axis.set_title("Screen title")

    destination = save_figure(
        figure,
        tmp_path / "screen.png",
        close=False,
    )

    assert destination == tmp_path / "screen.png"
    assert destination.is_file()
    assert axis.get_title() == "Screen title"
    plt.close(figure)


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_false_environment_values_do_not_enable_paper_mode(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PAPER_MODE_ENV, value)

    assert not paper_mode_enabled()
    assert paper_output_path("figure.png") == Path("figure.png")
