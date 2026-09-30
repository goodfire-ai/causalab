"""Derived study reports, eager controls and explicitly authored scalar criteria."""

from __future__ import annotations

from collections.abc import Mapping
import html
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
from typing import Any

from causalab.measurement.collection import file_hash, write_record


def evaluate_acceptance(
    report: Mapping[str, Any], criteria: list[dict[str, Any]]
) -> dict[str, Any]:
    """Missing evidence never satisfies a bound. Paths address JSON keys/indices."""
    outcomes = []
    for criterion in criteria:
        value: Any = report
        try:
            for segment in criterion["path"]:
                if isinstance(value, list) and segment.isdecimal():
                    value = value[int(segment)]
                else:
                    value = value[segment]
        except (KeyError, IndexError, TypeError):
            value = None
        valid = type(value) in (int, float) and math.isfinite(value)
        bound = "minimum" if "minimum" in criterion else "maximum"
        passed = valid and (
            value >= criterion[bound]
            if bound == "minimum"
            else value <= criterion[bound]
        )
        outcomes.append(
            {
                **criterion,
                "value": value if valid else None,
                "status": ("passed" if passed else "failed")
                if valid
                else "insufficient_evidence",
            }
        )
    statuses = {row["status"] for row in outcomes}
    status = (
        "not_evaluated"
        if not outcomes
        else "failed"
        if "failed" in statuses
        else "insufficient_evidence"
        if "insufficient_evidence" in statuses
        else "passed"
    )
    return {
        "status": status,
        "criteria": outcomes,
        "policy": "only authored scalar bounds; no universal numerical tolerance",
    }


def analysis_identity() -> dict[str, Any]:
    """Analysis provenance is independent of the measured source installations."""
    from causalab.measurement.analysis import (
        compare,
        stability,
        summary,
        profiles,
        training,
        single,
        receipts,
        statistics,
    )

    packages = {}
    for name in ("numpy", "torch", "safetensors"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    files = [Path(__file__).resolve()] + [
        Path(module.__file__).resolve()
        for module in (
            compare,
            stability,
            summary,
            profiles,
            training,
        )
        if module.__file__ is not None
    ]
    # Attest shared helpers separately from the report entry points above.
    # Both maps track report provenance, independently of resume exclusions.
    dependencies = [
        Path(module.__file__).resolve() for module in (single, receipts, statistics)
    ]
    return {
        "python": platform.python_version(),
        "packages": packages,
        "dependency_files_sha256": {
            p.relative_to(Path(__file__).resolve().parents[2]).as_posix(): file_hash(p)
            for p in dependencies
        },
        "files_sha256": {
            p.relative_to(Path(__file__).resolve().parents[2]).as_posix(): file_hash(p)
            for p in files
        },
    }


def render_study(report: Mapping[str, Any]) -> str:
    def escape(value):
        return html.escape(str(value), quote=True)

    rows = []
    for case, value in report["cases"].items():
        for name, comparison in value["comparisons"].items():
            links = value["files"][name]
            timing = comparison["timing"]
            intervals = timing["mean_paired_seconds_change_ci"]["interval"]
            speedups = [round(row["median_speedup"], 3) for row in timing["per_seed"]]
            summary = comparison["comparison_summary"]
            caveats = (
                "<ul>"
                + "".join(f"<li>{escape(note)}</li>" for note in summary["caveats"])
                + "</ul>"
            )
            rows.append(
                f"<tr><td>{escape(case)}</td><td>{escape(name)}</td>"
                f"<td>{escape(comparison['scope'])}</td><td>{escape(speedups)}</td>"
                f"<td>{escape(intervals)}</td><td>{escape(summary['status'])}{caveats}</td><td>"
                f'<a href="{escape(links["report"])}">Numerics and traces</a> · '
                f'<a href="{escape(links["summary"])}">JSON</a></td></tr>'
            )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>Measurement study</title><style>body{font:16px system-ui;"
        "margin:2rem;max-width:100rem}table{border-collapse:collapse}td,th{"
        "text-align:left;vertical-align:top;padding:.6rem;border:1px solid #ccc}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        "<h1>Measurement study</h1><p>Paired timing and numerical comparisons. "
        "Speedups list one median ratio per seed; intervals resample seeds. "
        "These descriptive intervals are not calibrated performance regression gates; "
        "few seeds limit precision. Traces are separate instrumented passes.</p>"
        f"<p>Authored acceptance: <strong>{escape(report['acceptance']['status'])}"
        "</strong>. This is not an optimization recommendation.</p>"
        "<p>An eager arm, when declared, provides separate eager-to-before and "
        "eager-to-after comparisons. Its within-seed and across-seed variation "
        "remain separate from paired implementation drift.</p>"
        "<table><tr><th>Case</th><th>Reference → candidate</th><th>Scope</th>"
        "<th>Speedup by seed</th><th>Descriptive 95% bootstrap interval, seconds change</th>"
        "<th>Comparison caveats</th><th>Evidence</th></tr>"
        + "".join(rows)
        + "</table><h2>Acceptance evidence</h2><pre>"
        + escape(json.dumps(report["acceptance"], indent=2, allow_nan=False))
        + "</pre><details><summary>Full study and analysis provenance</summary><pre>"
        + escape(json.dumps(report, indent=2, allow_nan=False))
        + "</pre></details></html>"
    )


def write_reports(
    collections: Mapping[str, Mapping[str, Path]],
    plan: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    if plan.get("mode") == "single":
        from causalab.measurement.analysis.single import write_single_reports

        return write_single_reports(collections, plan, output, analysis_identity())

    from causalab.measurement.analysis.compare import compare, render_report

    output.mkdir(parents=True, exist_ok=True)
    study: dict[str, Any] = {
        "schema_version": 1,
        "analysis": analysis_identity(),
        "plan": plan,
        "cases": {},
    }
    index = {}
    for case, arms in collections.items():
        comparisons, files = {}, {}
        contrasts = [("before_after", "before", "after")]
        if "eager" in arms:
            contrasts.extend(
                [("eager_before", "eager", "before"), ("eager_after", "eager", "after")]
            )
        for name, reference, candidate in contrasts:
            report = compare(
                arms[reference],
                arms[candidate],
                bootstrap_draws=plan["bootstrap_draws"],
                bootstrap_seed=plan["order_seed"],
            )
            report["contrast"] = {"reference": reference, "candidate": candidate}
            report["analysis"] = study["analysis"]
            # Existing per-case links remain stable for the primary contrast.
            stem = case if name == "before_after" else f"{case}.{name}"
            write_record(output / f"{stem}.json", report)
            (output / f"{stem}.html").write_text(render_report(report))
            comparisons[name] = report
            files[name] = {"summary": f"{stem}.json", "report": f"{stem}.html"}
        study["cases"][case] = {"comparisons": comparisons, "files": files}
        index[case] = files["before_after"]
    study["acceptance"] = evaluate_acceptance(study, plan.get("acceptance", []))
    write_record(output / "study.json", study)
    write_record(output / "index.json", index)
    (output / "index.html").write_text(render_study(study))
    return study
