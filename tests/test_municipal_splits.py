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


def place_graph_nodes():
    return [
        {"id": 1, "POP20": 100, "PLACE": {"Alpha": 60, "Beta": 20}},
        {"id": 2, "POP20": 80, "PLACE": {"Alpha": 40}},
        {"id": 3, "POP20": 50, "PLACE": {"Beta": 30}},
        {"id": 4, "POP20": 70, "PLACE": {}},
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
    assert summary.geography_mode == "COUSUB"
    assert summary.population_basis == "POP20"
    assert summary.represented_population == pytest.approx(270)


def test_default_mode_preserves_explicit_legacy_cousub_result():
    assignment = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3}

    default_summary = compute_municipal_split_statistics_for_assignment(
        assignment,
        graph_nodes(),
    )
    explicit_summary = compute_municipal_split_statistics_for_assignment(
        assignment,
        graph_nodes(),
        geography_mode="COUSUB",
        municipality_cols=("COUNTY", "MCD"),
    )

    assert default_summary.to_dict() == explicit_summary.to_dict()


def test_place_mapping_uses_weighted_multi_membership_population():
    summary = compute_municipal_split_statistics_for_assignment(
        {1: 1, 2: 2, 3: 1, 4: 2},
        place_graph_nodes(),
        geography_mode="PLACE",
        graph_path="data/graph/OH_pct20.json",
    )

    assert summary.n_cut_municipalities == 1
    assert summary.n_total_municipalities == 2
    assert summary.ousted_population == pytest.approx(40)
    assert summary.total_population == pytest.approx(150)
    assert summary.ousted_population_share == pytest.approx(40 / 150)
    assert summary.municipality_cols == ["PLACE"]
    assert summary.geography_mode == "PLACE"
    assert summary.population_basis == "PLACE mapping"
    assert summary.represented_population == pytest.approx(150)
    assert summary.graph_path == "data/graph/OH_pct20.json"


def test_place_mode_rejects_non_mapping_place_column():
    nodes = [{"id": 1, "POP20": 100, "PLACE": "Alpha"}]

    with pytest.raises(MunicipalSplitError, match="non-mapping"):
        compute_municipal_split_statistics_for_assignment(
            {1: 1},
            nodes,
            geography_mode="PLACE",
        )


def test_place_dictionary_cannot_be_used_as_scalar_municipality_key():
    with pytest.raises(MunicipalSplitError, match="use geography_mode='PLACE'"):
        compute_municipal_split_statistics_for_assignment(
            {1: 1, 2: 2, 3: 1, 4: 2},
            place_graph_nodes(),
            municipality_cols=("PLACE",),
        )


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


def test_make_place_traces_records_provenance_and_denominator():
    traces = make_municipal_split_traces(
        {"place-run": [{1: 1, 2: 2, 3: 1, 4: 2}]},
        place_graph_nodes(),
        geography_mode="PLACE",
        graph_path="data/graph/OH_pct20.json",
    )

    assert traces[0].geography_mode == "PLACE"
    assert traces[0].graph_path == "data/graph/OH_pct20.json"
    assert traces[0].population_basis == "PLACE mapping"
    assert traces[0].represented_population == pytest.approx(150)


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
    assert ax.get_ylabel() == "Cut county subdivisions"

    plt.close(fig)


def test_plot_ousted_population_trace(tmp_path):
    out = tmp_path / "ousted_trace.png"

    fig, ax = plot_ousted_population_trace(
        make_traces(),
        output_path=out,
    )

    assert out.exists()
    assert ax.get_ylabel() == "Ousted COUSUB population share"

    plt.close(fig)


def test_plot_municipal_split_trace_panel(tmp_path):
    out = tmp_path / "trace_panel.png"

    fig, axes = plot_municipal_split_trace_panel(
        make_traces(),
        output_path=out,
    )

    assert out.exists()
    assert len(axes) == 2
    assert axes[0].get_title() == "Cut county subdivisions"
    assert axes[1].get_ylabel() == "Ousted COUSUB population share"

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
    assert ax.get_xlabel() == "Cut county subdivisions"

    plt.close(fig)


def test_place_plot_labels_and_legacy_json_default():
    place_traces = make_municipal_split_traces(
        {"place-run": [{1: 1, 2: 2, 3: 1, 4: 2}]},
        place_graph_nodes(),
        geography_mode="PLACE",
    )
    fig, axes = plot_municipal_split_trace_panel(place_traces)
    assert axes[0].get_title() == "Cut Places"
    assert axes[1].get_ylabel() == "Ousted Place population share"
    plt.close(fig)

    legacy_trace = make_traces()[0].to_dict()
    for key in (
        "geography_mode",
        "graph_path",
        "population_basis",
        "represented_population",
    ):
        legacy_trace.pop(key)

    fig, ax = plot_ousted_population_trace([legacy_trace])
    assert ax.get_ylabel() == "Ousted COUSUB population share"
    plt.close(fig)
