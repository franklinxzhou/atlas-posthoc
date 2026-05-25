from dataclasses import dataclass

from atlas_posthoc.reports.summary import (
    ConvergenceThresholds,
    build_diagnostic_report,
    evaluate_diagnostics,
    final_tv,
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