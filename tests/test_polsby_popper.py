import matplotlib.pyplot as plt
import numpy as np
import pytest

from atlas_posthoc.observables.polsby_popper import (
    PolsbyPopperError,
    isoperimetric_to_polsby_popper,
    make_polsby_popper_traces,
    polsby_popper_from_area_perimeter,
    polsby_popper_to_isoperimetric,
    summarize_polsby_popper_matrix,
    summarize_polsby_popper_scores,
)
from atlas_posthoc.plotting.polsby_popper import (
    plot_average_polsby_popper_trace,
    plot_minimum_polsby_popper_trace,
    plot_polsby_popper_trace_panel,
)


def test_polsby_popper_from_area_perimeter_circle():
    area = np.pi
    perimeter = 2 * np.pi

    pp = polsby_popper_from_area_perimeter(area, perimeter)

    assert pp == pytest.approx(1.0)


def test_isoperimetric_to_polsby_popper_p2_over_area():
    iso = np.array([4 * np.pi, 8 * np.pi])

    pp = isoperimetric_to_polsby_popper(
        iso,
        convention="p2_over_area",
    )

    assert np.allclose(pp, [1.0, 0.5])


def test_isoperimetric_to_polsby_popper_normalized_inverse():
    iso = np.array([1.0, 2.0, 4.0])

    pp = isoperimetric_to_polsby_popper(
        iso,
        convention="normalized_inverse",
    )

    assert np.allclose(pp, [1.0, 0.5, 0.25])


def test_polsby_popper_to_isoperimetric():
    pp = np.array([1.0, 0.5])

    iso = polsby_popper_to_isoperimetric(
        pp,
        convention="p2_over_area",
    )

    assert np.allclose(iso, [4 * np.pi, 8 * np.pi])


def test_invalid_values_raise():
    with pytest.raises(PolsbyPopperError):
        isoperimetric_to_polsby_popper(
            [0.0, 1.0],
            invalid_policy="raise",
        )


def test_summarize_polsby_popper_scores():
    summary = summarize_polsby_popper_scores([0.2, 0.4, 0.6])

    assert summary.average == pytest.approx(0.4)
    assert summary.minimum == pytest.approx(0.2)
    assert summary.maximum == pytest.approx(0.6)
    assert summary.n_scores == 3
    assert summary.n_finite == 3


def test_summarize_polsby_popper_matrix():
    matrix = np.array(
        [
            [4 * np.pi, 8 * np.pi],
            [16 * np.pi, 4 * np.pi],
        ]
    )

    summary = summarize_polsby_popper_matrix(
        matrix,
        convention="p2_over_area",
    )

    assert np.allclose(summary["average_polsby_popper"], [0.75, 0.625])
    assert np.allclose(summary["minimum_polsby_popper"], [0.5, 0.25])
    assert np.allclose(summary["maximum_polsby_popper"], [1.0, 1.0])


def test_make_polsby_popper_traces_from_mapping():
    runs = {
        "run1": np.array(
            [
                [4 * np.pi, 8 * np.pi],
                [16 * np.pi, 4 * np.pi],
            ]
        ),
        "run2": np.array(
            [
                [8 * np.pi, 8 * np.pi],
                [4 * np.pi, 4 * np.pi],
            ]
        ),
    }

    traces = make_polsby_popper_traces(
        runs,
        output_frequency=100,
        convention="p2_over_area",
    )

    assert len(traces) == 2
    assert traces[0].chain_id == "run1"
    assert traces[0].proposals == [0, 100]
    assert traces[0].average_polsby_popper == pytest.approx([0.75, 0.625])
    assert traces[0].minimum_polsby_popper == pytest.approx([0.5, 0.25])


def test_plot_average_polsby_popper_trace(tmp_path):
    runs = {
        "run1": np.array(
            [
                [4 * np.pi, 8 * np.pi],
                [16 * np.pi, 4 * np.pi],
            ]
        ),
        "run2": np.array(
            [
                [8 * np.pi, 8 * np.pi],
                [4 * np.pi, 4 * np.pi],
            ]
        ),
    }

    out = tmp_path / "average_pp.png"

    fig, ax = plot_average_polsby_popper_trace(
        runs,
        output_frequency=100,
        output_path=out,
    )

    assert out.exists()
    assert ax.get_ylabel() == "Average Polsby-Popper"

    plt.close(fig)


def test_plot_minimum_polsby_popper_trace(tmp_path):
    runs = {
        "run1": np.array(
            [
                [4 * np.pi, 8 * np.pi],
                [16 * np.pi, 4 * np.pi],
            ]
        ),
        "run2": np.array(
            [
                [8 * np.pi, 8 * np.pi],
                [4 * np.pi, 4 * np.pi],
            ]
        ),
    }

    out = tmp_path / "minimum_pp.png"

    fig, ax = plot_minimum_polsby_popper_trace(
        runs,
        output_frequency=100,
        output_path=out,
    )

    assert out.exists()
    assert ax.get_ylabel() == "Minimum Polsby-Popper"

    plt.close(fig)


def test_plot_polsby_popper_trace_panel(tmp_path):
    runs = {
        "run1": np.array(
            [
                [4 * np.pi, 8 * np.pi],
                [16 * np.pi, 4 * np.pi],
            ]
        ),
        "run2": np.array(
            [
                [8 * np.pi, 8 * np.pi],
                [4 * np.pi, 4 * np.pi],
            ]
        ),
    }

    out = tmp_path / "pp_panel.png"

    fig, axes = plot_polsby_popper_trace_panel(
        runs,
        output_frequency=100,
        output_path=out,
    )

    assert out.exists()
    assert len(axes) == 2

    plt.close(fig)