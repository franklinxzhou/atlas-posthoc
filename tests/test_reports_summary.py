from dataclasses import dataclass

from atlas_posthoc.reports.summary import (
    ConvergenceThresholds,
    build_diagnostic_report,
    evaluate_diagnostics,
    final_tv,
    pace_summary_to_markdown,
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
class FakeBootstrap:
    n_bootstrap: int
    confidence_level: float
    upper_quantile: float
    bootstrap_deltas: list[float]
    mode: str


@dataclass
class FakePACE:
    delta: float
    raw_delta: float
    correction_factor: float
    threshold: float
    converged: bool
    n_chains: int
    n_cells: int
    n_total_samples: int
    chain_sizes: dict
    labels: list[int]
    chain_names: list[str]
    contingency: list[list[int]]
    cell_summaries: list
    bootstrap: FakeBootstrap | None = None


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


def fake_pace():
    return FakePACE(
        delta=0.083,
        raw_delta=0.125,
        correction_factor=0.6667,
        threshold=0.1,
        converged=True,
        n_chains=4,
        n_cells=6,
        n_total_samples=400,
        chain_sizes={
            "run5": 100,
            "run6": 100,
            "run7": 100,
            "run8": 100,
        },
        labels=[0, 1, 2, 3, 4, 5],
        chain_names=["run5", "run6", "run7", "run8"],
        contingency=[],
        cell_summaries=[],
        bootstrap=FakeBootstrap(
            n_bootstrap=100,
            confidence_level=0.95,
            upper_quantile=0.092,
            bootstrap_deltas=[],
            mode="pooled",
        ),
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

# Additional tests after adding PACE results to the reports/summary.py
def test_pace_summary_to_markdown():
    text = pace_summary_to_markdown(fake_pace())

    assert "PACE state-space convergence" in text
    assert "Delta: 0.0830" in text
    assert "Raw delta: 0.1250" in text
    assert "Correction factor: 0.6667" in text
    assert "Bootstrap upper 95%: 0.0920" in text
    assert "Threshold: 0.1000" in text
    assert "Status: PASS" in text
    assert "Number of cells: 6" in text
    assert "Number of samples: 400" in text


def test_build_diagnostic_report_with_pace():
    report = build_diagnostic_report(
        title="PACE report",
        vote_rhat=fake_rhat(),
        pace_result=fake_pace(),
    )

    assert "# PACE report" in report
    assert "## PACE diagnostic" in report
    assert "PACE state-space convergence" in report
    assert "Status: PASS" in report