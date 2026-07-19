from dataclasses import dataclass

from atlas_posthoc.reports.summary import (
    ConvergenceThresholds,
    build_diagnostic_report,
    evaluate_diagnostics,
    final_tv,
    municipal_split_traces_to_markdown,
    rhat_table_to_markdown,
    write_diagnostic_report,
)


@dataclass
class FakeTV:
    proposals: list[int]
    max_tvs: list[float]


@dataclass
class FakeRHat:
    rows: list[dict]
    min_draws: int
    num_ranks: int
    max_classic_r_hat: float
    max_rank_normalized_split_r_hat: float

@dataclass
class FakeMunicipalSplitTrace:
    chain_id: str
    proposals: list[int]
    n_cut_municipalities: list[int]
    ousted_population: list[float]
    ousted_population_share: list[float]
    n_maps: int
    municipality_cols: list[str]
    pop_col: str


def fake_rhat():
    return FakeRHat(
        rows=[
            {
                "rank": 1,
                "interpretation": "most Republican",
                "classic_r_hat": 1.02,
                "rank_normalized_split_r_hat": 1.03,
            },
            {
                "rank": 2,
                "interpretation": "least Republican",
                "classic_r_hat": 1.01,
                "rank_normalized_split_r_hat": 1.02,
            },
        ],
        min_draws=100,
        num_ranks=2,
        max_classic_r_hat=1.02,
        max_rank_normalized_split_r_hat=1.03,
    )


def test_final_tv():
    tv = FakeTV(proposals=[10, 20, 30], max_tvs=[0.5, 0.3, 0.1])

    assert final_tv(tv) == 0.1


def test_evaluate_diagnostics():
    tv = FakeTV(proposals=[10, 20, 30], max_tvs=[0.5, 0.3, 0.1])
    rhat = fake_rhat()

    statuses = evaluate_diagnostics(
        vote_tv=tv,
        vote_rhat=rhat,
        thresholds=ConvergenceThresholds(
            max_rank_normalized_split_rhat=1.05,
            final_tv=0.2,
        ),
    )

    by_name = {s.name: s for s in statuses}

    assert by_name["Vote-share final TV"].passed is True
    assert by_name["Vote-share max rank-normalized split R-hat"].passed is True


def test_rhat_table_to_markdown():
    text = rhat_table_to_markdown(fake_rhat(), title="Vote R-hat")

    assert "### Vote R-hat" in text
    assert "most Republican" in text
    assert "Rank-normalized split R-hat" in text


def test_build_diagnostic_report():
    tv = FakeTV(proposals=[10, 20, 30], max_tvs=[0.5, 0.3, 0.1])
    rhat = fake_rhat()

    report = build_diagnostic_report(
        title="Test report",
        vote_tv=tv,
        vote_rhat=rhat,
        notes="Synthetic test report.",
    )

    assert "# Test report" in report
    assert "## Convergence status" in report
    assert "Vote-share final TV" in report
    assert "Synthetic test report." in report


def test_write_diagnostic_report(tmp_path):
    path = tmp_path / "report.md"

    out = write_diagnostic_report(
        path,
        title="Saved report",
        vote_tv=FakeTV(proposals=[10], max_tvs=[0.2]),
        vote_rhat=fake_rhat(),
    )

    assert out.exists()
    assert "Saved report" in out.read_text(encoding="utf-8")

def fake_municipal_split_traces():
    return [
        FakeMunicipalSplitTrace(
            chain_id="run37",
            proposals=[0, 10000, 20000],
            n_cut_municipalities=[20, 22, 21],
            ousted_population=[100000.0, 120000.0, 110000.0],
            ousted_population_share=[0.0085, 0.0102, 0.0093],
            n_maps=3,
            municipality_cols=["COUNTY", "MCD"],
            pop_col="POP20",
        ),
        FakeMunicipalSplitTrace(
            chain_id="run38",
            proposals=[0, 10000, 20000],
            n_cut_municipalities=[19, 20, 20],
            ousted_population=[90000.0, 95000.0, 98000.0],
            ousted_population_share=[0.0076, 0.0081, 0.0083],
            n_maps=3,
            municipality_cols=["COUNTY", "MCD"],
            pop_col="POP20",
        ),
    ]

def test_municipal_split_traces_to_markdown():
    text = municipal_split_traces_to_markdown(
        fake_municipal_split_traces(),
        tail_n=2,
    )

    assert "Municipal split traces" in text
    assert "Final cuts" in text
    assert "Tail cuts" in text
    assert "Final ousted share" in text
    assert "Tail ousted pop." in text
    assert "run37" in text
    assert "run38" in text
    assert "Mode" in text
    assert "COUSUB" in text
    assert "Denominator" in text
    assert "COUNTY" in text
    assert "MCD" in text


def test_municipal_split_report_records_place_metadata():
    text = municipal_split_traces_to_markdown(
        [
            {
                "chain_id": "place-run",
                "proposals": [0],
                "n_cut_municipalities": [12],
                "ousted_population": [1234],
                "ousted_population_share": [0.001],
                "n_maps": 1,
                "municipality_cols": ["PLACE"],
                "pop_col": "POP20",
                "geography_mode": "PLACE",
                "graph_path": "data/graph/OH_pct20.json",
                "population_basis": "PLACE mapping",
                "represented_population": 8344527,
            }
        ]
    )

    assert "PLACE" in text
    assert "8,344,527" in text
    assert "PLACE mapping" in text
    assert "data/graph/OH_pct20.json" in text


def test_build_diagnostic_report_with_municipal_splits():
    report = build_diagnostic_report(
        title="Municipal report",
        vote_rhat=fake_rhat(),
        municipal_split_traces=fake_municipal_split_traces(),
    )

    assert "# Municipal report" in report
    assert "## Municipal split diagnostics" in report
    assert "Municipal split traces" in report
    assert "Final cuts" in report
