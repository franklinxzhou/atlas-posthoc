import matplotlib.pyplot as plt
import pytest

from atlas_posthoc.observables.municipal_splits import (
    MunicipalSplitError,
    compute_municipal_split_statistics_for_assignment,
    make_municipal_split_traces,
)
from atlas_posthoc.plotting.municipal_splits import (
    plot_cut_municipalities_trace,
    plot_municipal_split_distribution_panel,
    plot_municipal_split_scatter,
    plot_municipal_split_trace_panel,
    plot_ousted_population_trace,
)


def graph_nodes():
    return [
        {"id": 1, "COUNTY": "FRANKLIN", "MCD": "COLUMBUS", "POP20": 100},
        {"id": 2, "COUNTY": "FRANKLIN", "MCD": "COLUMBUS", "POP20": 50},
        {"id": 3, "COUNTY": "FRANKLIN", "MCD": "DUBLIN", "POP20": 30},
        {"id": 4, "COUNTY": "FRANKLIN", "MCD": "DUBLIN", "POP20": 70},
        {"id": 5, "COUNTY": "CUYAHOGA", "MCD": "COLUMBUS", "POP20": 20},
    ]


def test_uncut_assignment():
    assignment = {
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 3,
    }

    summary = compute_municipal_split_statistics_for_assignment(
        assignment,
        graph_nodes(),
    )

    assert summary.n_cut_municipalities == 0
    assert summary.ousted_population == 0
    assert summary.ousted_population_share == 0
    assert summary.n_total_municipalities == 3


def test_simple_cut_assignment():
    assignment = {
        1: 1,
        2: 2,  # COLUMBUS/FRANKLIN split: 100 vs 50, ousted 50
        3: 2,
        4: 2,
        5: 3,
    }

    summary = compute_municipal_split_statistics_for_assignment(
        assignment,
        graph_nodes(),
    )

    assert summary.n_cut_municipalities == 1
    assert summary.ousted_population == pytest.approx(50)
    assert summary.total_population == pytest.approx(270)
    assert summary.ousted_population_share == pytest.approx(50 / 270)
    assert summary.max_ousted_single_municipality == pytest.approx(50)
    assert summary.mean_ousted_population_among_cut == pytest.approx(50)


def test_same_mcd_name_in_different_counties_is_separate():
    assignment = {
        1: 1,
        2: 2,  # FRANKLIN/COLUMBUS cut
        3: 2,
        4: 2,
        5: 3,  # CUYAHOGA/COLUMBUS separate municipality, not cut
    }

    summary = compute_municipal_split_statistics_for_assignment(
        assignment,
        graph_nodes(),
        municipality_cols=("COUNTY", "MCD"),
    )

    assert summary.n_cut_municipalities == 1
    assert summary.n_total_municipalities == 3


def test_missing_assignment_raises():
    assignment = {
        1: 1,
        2: 1,
    }

    with pytest.raises(MunicipalSplitError):
        compute_municipal_split_statistics_for_assignment(
            assignment,
            graph_nodes(),
            require_complete_assignment=True,
        )


def test_sequence_assignment():
    assignment = [1, 1, 2, 2, 3]

    summary = compute_municipal_split_statistics_for_assignment(
        assignment,
        graph_nodes(),
    )

    assert summary.n_cut_municipalities == 0


def test_make_municipal_split_traces():
    assignments_by_chain = {
        "run1": [
            {1: 1, 2: 1, 3: 2, 4: 2, 5: 3},
            {1: 1, 2: 2, 3: 2, 4: 2, 5: 3},
        ],
        "run2": [
            {1: 1, 2: 1, 3: 2, 4: 2, 5: 3},
            {1: 1, 2: 1, 3: 2, 4: 3, 5: 3},
        ],
    }

    traces = make_municipal_split_traces(
        assignments_by_chain,
        graph_nodes(),
        output_frequency=100,
    )

    assert len(traces) == 2
    assert traces[0].chain_id == "run1"
    assert traces[0].proposals == [0, 100]
    assert traces[0].n_cut_municipalities == [0, 1]
    assert traces[0].ousted_population[1] == pytest.approx(50)


def make_traces():
    assignments_by_chain = {
        "run1": [
            {1: 1, 2: 1, 3: 2, 4: 2, 5: 3},
            {1: 1, 2: 2, 3: 2, 4: 2, 5: 3},
            {1: 1, 2: 2, 3: 2, 4: 3, 5: 3},
        ],
        "run2": [
            {1: 1, 2: 1, 3: 2, 4: 2, 5: 3},
            {1: 1, 2: 1, 3: 2, 4: 3, 5: 3},
            {1: 2, 2: 2, 3: 3, 4: 3, 5: 1},
        ],
    }

    return make_municipal_split_traces(
        assignments_by_chain,
        graph_nodes(),
        output_frequency=100,
    )


def test_plot_cut_municipalities_trace(tmp_path):
    out = tmp_path / "cut_trace.png"

    fig, ax = plot_cut_municipalities_trace(
        make_traces(),
        output_path=out,
    )

    assert out.exists()
    assert ax.get_ylabel() == "Number of cut municipalities"

    plt.close(fig)


def test_plot_ousted_population_trace(tmp_path):
    out = tmp_path / "ousted_trace.png"

    fig, ax = plot_ousted_population_trace(
        make_traces(),
        output_path=out,
    )

    assert out.exists()
    assert ax.get_ylabel() == "Ousted municipal population share"

    plt.close(fig)


def test_plot_municipal_split_trace_panel(tmp_path):
    out = tmp_path / "trace_panel.png"

    fig, axes = plot_municipal_split_trace_panel(
        make_traces(),
        output_path=out,
    )

    assert out.exists()
    assert len(axes) == 2

    plt.close(fig)


def test_plot_municipal_split_distribution_panel(tmp_path):
    out = tmp_path / "distribution_panel.png"

    fig, axes = plot_municipal_split_distribution_panel(
        make_traces(),
        output_path=out,
    )

    assert out.exists()
    assert len(axes) == 2

    plt.close(fig)


def test_plot_municipal_split_scatter(tmp_path):
    out = tmp_path / "scatter.png"

    fig, ax = plot_municipal_split_scatter(
        make_traces(),
        output_path=out,
    )

    assert out.exists()
    assert ax.get_xlabel() == "Number of cut municipalities"

    plt.close(fig)