from __future__ import annotations

import math
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class ReportError(Exception):
    """Base exception for report-generation helpers."""


@dataclass
class ConvergenceThresholds:
    """
    Configurable convergence thresholds for summary reporting.

    These are reporting thresholds only. They do not prove convergence.
    """
    max_rank_normalized_split_rhat: float = 1.05
    max_classic_rhat: float | None = None
    final_tv: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosticStatus:
    name: str
    value: float | None
    threshold: float | None
    passed: bool | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_dict(obj: Any) -> Any:
    if obj is None:
        return None

    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()

    if is_dataclass(obj):
        return asdict(obj)

    if isinstance(obj, Mapping):
        return dict(obj)

    return obj


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default

    if isinstance(obj, Mapping):
        return obj.get(key, default)

    return getattr(obj, key, default)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"

    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)

    if math.isnan(x):
        return "NA"

    if abs(x) >= 1000:
        return f"{x:,.0f}"

    return f"{x:.{digits}f}"


def _status_symbol(passed: bool | None) -> str:
    if passed is True:
        return "PASS"
    if passed is False:
        return "WARN"
    return "INFO"


def final_tv(tv_result: Any) -> float | None:
    """
    Extract final max average pairwise TV from a TVConvergenceResult-like object.
    """
    if tv_result is None:
        return None

    max_tvs = _get(tv_result, "max_tvs")

    if max_tvs is None:
        return None

    if len(max_tvs) == 0:
        return None

    return float(max_tvs[-1])


def final_proposals(tv_result: Any) -> int | None:
    if tv_result is None:
        return None

    proposals = _get(tv_result, "proposals")

    if proposals is None or len(proposals) == 0:
        return None

    return int(proposals[-1])


def rhat_maxes(rhat_summary: Any) -> tuple[float | None, float | None]:
    """
    Return max classic R-hat and max rank-normalized split R-hat.
    """
    if rhat_summary is None:
        return None, None

    classic = _get(rhat_summary, "max_classic_r_hat")
    split = _get(rhat_summary, "max_rank_normalized_split_r_hat")

    return (
        None if classic is None else float(classic),
        None if split is None else float(split),
    )


def _extract_pace_result(pace_like: Any) -> Any:
    """
    Accept either:
      - PACEResult
      - PACEDistanceClusteringResult, via `.pace`
      - dict with key "pace"
      - dict with PACEResult-style fields
    """
    if pace_like is None:
        return None

    if hasattr(pace_like, "pace"):
        return pace_like.pace

    if isinstance(pace_like, Mapping) and "pace" in pace_like:
        return pace_like["pace"]

    return pace_like


def pace_status(pace_like: Any) -> DiagnosticStatus:
    """
    Produce a PASS/WARN/INFO status for a PACE result.

    If bootstrap is available, PACEResult.converged should already reflect the
    bootstrap upper quantile rule used in diagnostics.pace.
    """
    pace = _extract_pace_result(pace_like)

    if pace is None:
        return DiagnosticStatus(
            name="PACE state-space convergence",
            value=None,
            threshold=None,
            passed=None,
            message="not provided",
        )

    delta = _get(pace, "delta")
    threshold = _get(pace, "threshold")
    converged = _get(pace, "converged")

    if converged is True:
        message = "within threshold"
    elif converged is False:
        message = "above threshold"
    else:
        message = "reported without pass/fail decision"

    return DiagnosticStatus(
        name="PACE state-space convergence",
        value=None if delta is None else float(delta),
        threshold=None if threshold is None else float(threshold),
        passed=converged,
        message=message,
    )


def pace_summary_to_markdown(
    pace_like: Any,
    *,
    title: str = "PACE state-space convergence",
    digits: int = 4,
) -> str:
    """
    Convert a PACEResult-like object into a compact markdown summary.

    This is intentionally scalar and advisor/report friendly. The detailed
    chain-by-cell table is better shown as the PACE heatmap.
    """
    pace = _extract_pace_result(pace_like)

    if pace is None:
        return f"### {title}\n\nNot provided.\n"

    bootstrap = _get(pace, "bootstrap")
    confidence_level = None
    bootstrap_upper = None

    if bootstrap is not None:
        confidence_level = _get(bootstrap, "confidence_level")
        bootstrap_upper = _get(bootstrap, "upper_quantile")

    delta = _get(pace, "delta")
    raw_delta = _get(pace, "raw_delta")
    correction_factor = _get(pace, "correction_factor")
    threshold = _get(pace, "threshold")
    converged = _get(pace, "converged")
    n_cells = _get(pace, "n_cells")
    n_samples = _get(pace, "n_total_samples")

    if converged is True:
        status = "PASS"
    elif converged is False:
        status = "WARN"
    else:
        status = "INFO"

    if confidence_level is None:
        bootstrap_label = "Bootstrap upper"
    else:
        bootstrap_label = f"Bootstrap upper {100 * float(confidence_level):.0f}%"

    lines = [
        f"### {title}",
        "",
        f"Delta: {_fmt(delta, digits)}",
        f"Raw delta: {_fmt(raw_delta, digits)}",
        f"Correction factor: {_fmt(correction_factor, digits)}",
        f"{bootstrap_label}: {_fmt(bootstrap_upper, digits)}",
        f"Threshold: {_fmt(threshold, digits)}",
        f"Status: {status}",
        f"Number of cells: {_fmt(n_cells, 0)}",
        f"Number of samples: {_fmt(n_samples, 0)}",
    ]

    return "\n".join(lines)

def evaluate_diagnostics(
    *,
    vote_tv: Any | None = None,
    compactness_tv: Any | None = None,
    vote_rhat: Any | None = None,
    compactness_rhat: Any | None = None,
    pace_result: Any | None = None,
    thresholds: ConvergenceThresholds | None = None,
) -> list[DiagnosticStatus]:
    """
    Produce pass/warn/info statuses for core convergence diagnostics.
    """
    if thresholds is None:
        thresholds = ConvergenceThresholds()

    statuses: list[DiagnosticStatus] = []

    for name, tv in [
        ("Vote-share final TV", vote_tv),
        ("Compactness final TV", compactness_tv),
    ]:
        value = final_tv(tv)
        threshold = thresholds.final_tv

        if value is None:
            passed = None
            message = "not provided"
        elif threshold is None:
            passed = None
            message = "reported without threshold"
        else:
            passed = value <= threshold
            message = f"{'within' if passed else 'above'} threshold {threshold:g}"

        statuses.append(
            DiagnosticStatus(
                name=name,
                value=value,
                threshold=threshold,
                passed=passed,
                message=message,
            )
        )

    for name, rhat in [
        ("Vote-share max rank-normalized split R-hat", vote_rhat),
        ("Compactness max rank-normalized split R-hat", compactness_rhat),
    ]:
        _, split = rhat_maxes(rhat)
        threshold = thresholds.max_rank_normalized_split_rhat

        if split is None:
            passed = None
            message = "not provided"
        else:
            passed = split <= threshold
            message = f"{'within' if passed else 'above'} threshold {threshold:g}"

        statuses.append(
            DiagnosticStatus(
                name=name,
                value=split,
                threshold=threshold,
                passed=passed,
                message=message,
            )
        )

    if thresholds.max_classic_rhat is not None:
        for name, rhat in [
            ("Vote-share max classic R-hat", vote_rhat),
            ("Compactness max classic R-hat", compactness_rhat),
        ]:
            classic, _ = rhat_maxes(rhat)
            threshold = thresholds.max_classic_rhat

            if classic is None:
                passed = None
                message = "not provided"
            else:
                passed = classic <= threshold
                message = f"{'within' if passed else 'above'} threshold {threshold:g}"

            statuses.append(
                DiagnosticStatus(
                    name=name,
                    value=classic,
                    threshold=threshold,
                    passed=passed,
                    message=message,
                )
            )



    if pace_result is not None:
        statuses.append(pace_status(pace_result))

    return statuses


def statuses_to_markdown(statuses: Sequence[DiagnosticStatus]) -> str:
    lines = [
        "| Diagnostic | Value | Threshold | Status | Note |",
        "|---|---:|---:|---|---|",
    ]

    for status in statuses:
        lines.append(
            "| "
            f"{status.name} | "
            f"{_fmt(status.value)} | "
            f"{_fmt(status.threshold)} | "
            f"{_status_symbol(status.passed)} | "
            f"{status.message} |"
        )

    return "\n".join(lines)


def rhat_table_to_markdown(
    rhat_summary: Any,
    *,
    title: str | None = None,
    digits: int = 4,
) -> str:
    """
    Convert an RHatSummary-like object into a markdown table.
    """
    if rhat_summary is None:
        return "" if title is None else f"### {title}\n\nNot provided.\n"

    rows = _get(rhat_summary, "rows")

    if rows is None:
        raise ReportError("R-hat summary does not contain `rows`")

    out: list[str] = []

    if title is not None:
        out.append(f"### {title}")
        out.append("")

    out.extend(
        [
            "| Rank | Interpretation | Classic R-hat | Rank-normalized split R-hat |",
            "|---:|---|---:|---:|",
        ]
    )

    for row in rows:
        out.append(
            "| "
            f"{row['rank']} | "
            f"{row['interpretation']} | "
            f"{_fmt(row['classic_r_hat'], digits)} | "
            f"{_fmt(row['rank_normalized_split_r_hat'], digits)} |"
        )

    return "\n".join(out)


def percentile_table_to_markdown(
    percentile_result: Any,
    *,
    title: str | None = None,
    digits: int = 4,
) -> str:
    """
    Convert a ReferencePercentileResult-like object into a markdown table.
    """
    if percentile_result is None:
        return "" if title is None else f"### {title}\n\nNot provided.\n"

    data = _to_dict(percentile_result)
    rows = data.get("rows")

    if rows is None:
        raise ReportError("percentile result does not contain `rows`")

    out: list[str] = []

    if title is not None:
        out.append(f"### {title}")
        out.append("")

    out.extend(
        [
            "| Rank | Interpretation | Reference | Percentile | Two-sided p | Ensemble mean | Ensemble sd |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )

    for row in rows:
        out.append(
            "| "
            f"{row['rank']} | "
            f"{row['interpretation']} | "
            f"{_fmt(row['reference_value'], digits)} | "
            f"{_fmt(row['percentile'], digits)} | "
            f"{_fmt(row['two_sided_p'], digits)} | "
            f"{_fmt(row['ensemble_mean'], digits)} | "
            f"{_fmt(row['ensemble_std'], digits)} |"
        )

    return "\n".join(out)


def chain_summaries_to_markdown(
    summaries: Mapping[str, Any] | None,
    *,
    title: str = "Chain parsing summary",
) -> str:
    if summaries is None:
        return f"### {title}\n\nNot provided.\n"

    lines = [
        f"### {title}",
        "",
        "| Chain | Records | Vote vectors | Compactness vectors | Key cols | Bad JSON | Bad assignments | Compactness errors |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]

    for chain_id, summary in summaries.items():
        s = _to_dict(summary)
        parse = s.get("parse_summary", {})
        key_cols = s.get("selected_key_cols", [])

        lines.append(
            "| "
            f"{chain_id} | "
            f"{s.get('n_records_processed', 'NA')} | "
            f"{s.get('n_vote_vectors', 'NA')} | "
            f"{s.get('n_compactness_vectors', 'NA')} | "
            f"{key_cols} | "
            f"{parse.get('bad_json_lines', 'NA')} | "
            f"{parse.get('bad_assignment_lines', 'NA')} | "
            f"{len(s.get('compactness_errors', []))} |"
        )

    return "\n".join(lines)


def tv_summary_to_markdown(
    *,
    vote_tv: Any | None = None,
    compactness_tv: Any | None = None,
    title: str = "Total variation summary",
) -> str:
    lines = [
        f"### {title}",
        "",
        "| Observable | Final proposals | Final max avg. TV | Number of snapshots |",
        "|---|---:|---:|---:|",
    ]

    for label, tv in [
        ("Vote share", vote_tv),
        ("Compactness / iso. score", compactness_tv),
    ]:
        if tv is None:
            lines.append(f"| {label} | NA | NA | NA |")
            continue

        proposals = _get(tv, "proposals", [])
        max_tvs = _get(tv, "max_tvs", [])

        lines.append(
            "| "
            f"{label} | "
            f"{_fmt(proposals[-1] if proposals else None, 0)} | "
            f"{_fmt(max_tvs[-1] if max_tvs else None)} | "
            f"{len(max_tvs)} |"
        )

    return "\n".join(lines)


def build_diagnostic_report(
    *,
    title: str = "Atlas post-hoc diagnostic report",
    chain_summaries: Mapping[str, Any] | None = None,
    vote_tv: Any | None = None,
    compactness_tv: Any | None = None,
    vote_rhat: Any | None = None,
    compactness_rhat: Any | None = None,
    pace_result: Any | None = None,
    vote_percentiles: Any | None = None,
    compactness_percentiles: Any | None = None,
    thresholds: ConvergenceThresholds | None = None,
    notes: str | None = None,
) -> str:
    """
    Build a markdown diagnostic report.

    This is meant to be copied into manuscript notes, advisor updates, or a
    cached run directory.
    """
    statuses = evaluate_diagnostics(
        vote_tv=vote_tv,
        compactness_tv=compactness_tv,
        vote_rhat=vote_rhat,
        compactness_rhat=compactness_rhat,
        pace_result=pace_result,
        thresholds=thresholds,
    )

    blocks = [
        f"# {title}",
        "",
        "## Convergence status",
        "",
        statuses_to_markdown(statuses),
        "",
        tv_summary_to_markdown(
            vote_tv=vote_tv,
            compactness_tv=compactness_tv,
        ),
        "",
        chain_summaries_to_markdown(chain_summaries),
        "",
        "## Rank-wise R-hat",
        "",
        rhat_table_to_markdown(vote_rhat, title="Vote-share R-hat"),
        "",
        rhat_table_to_markdown(compactness_rhat, title="Compactness R-hat"),
    ]

    if pace_result is not None:
        blocks.extend(
            [
                "",
                "## PACE diagnostic",
                "",
                pace_summary_to_markdown(pace_result),
            ]
        )

    if vote_percentiles is not None or compactness_percentiles is not None:
        blocks.extend(
            [
                "",
                "## Reference/enacted percentile comparison",
                "",
                percentile_table_to_markdown(
                    vote_percentiles,
                    title="Vote-share reference percentiles",
                ),
                "",
                percentile_table_to_markdown(
                    compactness_percentiles,
                    title="Compactness reference percentiles",
                ),
            ]
        )

    if notes is not None:
        blocks.extend(
            [
                "",
                "## Notes",
                "",
                notes.strip(),
            ]
        )

    return "\n".join(blocks).strip() + "\n"


def write_diagnostic_report(
    path: str | Path,
    *,
    title: str = "Atlas post-hoc diagnostic report",
    chain_summaries: Mapping[str, Any] | None = None,
    vote_tv: Any | None = None,
    compactness_tv: Any | None = None,
    vote_rhat: Any | None = None,
    compactness_rhat: Any | None = None,
    pace_result: Any | None = None,
    vote_percentiles: Any | None = None,
    compactness_percentiles: Any | None = None,
    thresholds: ConvergenceThresholds | None = None,
    notes: str | None = None,
) -> Path:
    """
    Write a markdown diagnostic report to disk.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    report = build_diagnostic_report(
        title=title,
        chain_summaries=chain_summaries,
        vote_tv=vote_tv,
        compactness_tv=compactness_tv,
        vote_rhat=vote_rhat,
        compactness_rhat=compactness_rhat,
        pace_result=pace_result,
        vote_percentiles=vote_percentiles,
        compactness_percentiles=compactness_percentiles,
        thresholds=thresholds,
        notes=notes,
    )

    p.write_text(report, encoding="utf-8")
    return p