"""Absolute single-source measurements and portable links to diagnostic evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import html
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .profiles import capture_pairs, compare_captures
from .receipts import load_measurement, verified_artifact
from .statistics import describe
from ..collection import write_record


@dataclass
class SingleReportError(ValueError):
    case: str
    reason: str

    def __str__(self) -> str:
        return f"single measurement {self.case}: {self.reason}"


def _link(path: Path, output: Path) -> str:
    return quote(os.path.relpath(path.resolve(), output.resolve()), safe="/")


def _execution(record: Mapping[str, Any]) -> dict[str, Any]:
    worker = record.get("context", {}).get("worker", {})
    probe = worker.get("execution_probe", {}).get("cases", {}).get(record["case"])
    dispatch = probe.get("python_dispatch", []) if probe else []
    return {
        "declared": worker.get("execution", {}),
        "probe_status": "collected" if probe else "not_collected",
        "probe_scope": "separate untimed execution; outside clean samples",
        "graph_replay_observed": any(
            "graph_replay" in name or "CUDAGraph.replay" in name for name in dispatch
        )
        if probe
        else None,
        "python_dispatch": dispatch,
        "native_profile": probe.get("native_profile", {}) if probe else {},
    }


def _output_links(
    samples: Mapping[tuple[int, int], Mapping[str, Any]],
    path: Path,
    output: Path,
    *,
    field: str,
    scope: str,
) -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "repeat": repeat,
            "scope": scope,
            "file": name,
            "link": _link(
                verified_artifact(path.parent, {"file": name, "sha256": sha}), output
            ),
            "sha256": sha,
        }
        for (seed, repeat), sample in sorted(samples.items())
        for name, sha in sample.get(field, {}).items()
    ]


def summarize_single(path: Path, output: Path) -> dict[str, Any]:
    record, samples = load_measurement(path, require_observations=False)
    numerical = record.get("observation_policy", "required") == "required"
    captures = compare_captures(record, path, samples)
    for capture in captures:
        capture["artifact_links"] = [
            _link(Path(p), output) for p in capture.pop("artifact_paths")
        ]
        capture["log_links"] = [
            _link(Path(p), output) for p in capture.pop("log_paths")
        ]
        capture["output_links"] = [
            _link(verified_artifact(path.parent, {"file": name, "sha256": sha}), output)
            for name, sha in capture.get("output_files", {}).items()
        ]
        capture.setdefault("observation_check", {"status": "unavailable"})
    pairs = capture_pairs(record, captures)
    trace = dict(record.get("trace", {"status": "not_requested"}))
    if trace["status"] == "completed":
        ref = {k: trace[k] for k in ("file", "sha256")}
        trace["artifact_link"] = _link(verified_artifact(path.parent, ref), output)
        # Legacy collector traces use the same clean-reference rules as captures.
        legacy = {**trace, "mode": "warm", "backend": "torch", "artifacts": [ref]}
        checked = compare_captures({**record, "captures": [legacy]}, path, samples)[0]
        trace["observation_check"] = checked["observation_check"]
        if "comparison_reference" in checked:
            trace["comparison_reference"] = checked["comparison_reference"]
    elif not numerical:
        trace["observation_check"] = {"status": "not_requested"}
    memory = [
        {"seed": seed, "repeat": repeat, "peak_memory": sample.get("peak_memory")}
        for (seed, repeat), sample in sorted(samples.items())
    ]
    available = sum(row["peak_memory"] is not None for row in memory)
    checks = (
        [
            {
                "seed": seed,
                "repeat": repeat,
                "check": sample.get("observer_check", {"status": "unavailable"}),
            }
            for (seed, repeat), sample in sorted(samples.items())
        ]
        if numerical
        else []
    )
    capture_status = (
        "incomplete"
        if any(pair["status"] != "completed" for pair in pairs)
        or trace["status"] not in {"completed", "not_requested"}
        else "completed"
        if captures or trace["status"] == "completed"
        else "not_requested"
    )
    return {
        "schema_version": 1,
        "mode": "single",
        "case": record["case"],
        "collection_status": record["status"],
        "capture_status": capture_status,
        "observation_check": {
            "status": "collected" if numerical else "not_requested",
            "samples": checks,
        },
        "observation_policy": "required" if numerical else "not_requested",
        "scope": record["scope"],
        "timing": {
            "units": "seconds",
            "per_seed": [
                {
                    "seed": seed,
                    "scope": record["scope"],
                    "seconds": describe(
                        [
                            samples[seed, repeat]["seconds"]
                            for repeat in range(record["plan"]["repeats"])
                        ]
                    ),
                    "samples": [
                        {"repeat": repeat, "seconds": samples[seed, repeat]["seconds"]}
                        for repeat in range(record["plan"]["repeats"])
                    ],
                }
                for seed in record["plan"]["seeds"]
            ],
        },
        "memory": {
            "units": "bytes",
            "scope": record["scope"],
            "status": "available"
            if available == len(memory)
            else "partial"
            if available
            else "unavailable",
            "samples": memory,
            "resident_reference": {
                "scope": "separate resident workflow timing pass; outside the cold-process timer",
                "units": "bytes",
                "status": "available"
                if any(
                    sample.get("resident_peak_memory") is not None
                    for sample in samples.values()
                )
                else "unavailable",
                "samples": [
                    {
                        "seed": seed,
                        "repeat": repeat,
                        "peak_memory": sample.get("resident_peak_memory"),
                    }
                    for (seed, repeat), sample in sorted(samples.items())
                    if "resident_peak_memory" in sample
                ],
            },
        },
        "input_identity": record["input_identity"],
        "reset_policy": record["reset_policy"],
        "plan": record["plan"],
        "passes": record.get("passes"),
        "provenance": record["provenance"],
        "context": record.get("context", {}),
        "execution": _execution(record),
        "cache_policy": record.get("context", {}).get("cache_policy"),
        "receipt": _link(path, output),
        "outputs": _output_links(
            samples, path, output, field="output_files", scope=record["scope"]
        ),
        "resident_outputs": _output_links(
            samples,
            path,
            output,
            field="resident_output_files",
            scope="separate resident workflow pass; excluded from cold timing",
        ),
        "captures": captures,
        "capture_pairs": pairs,
        "trace": trace,
    }


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _details(title: str, value: Any) -> str:
    return f"<details><summary>{_escape(title)}</summary><pre>{_escape(json.dumps(value, indent=2, allow_nan=False))}</pre></details>"


def _anchor(link: str, label: str) -> str:
    return f'<a href="{_escape(link)}">{_escape(label)}</a>'


def _page(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f"<title>{_escape(title)}</title><style>body{{font:16px system-ui;max-width:80rem;margin:2rem auto;padding:0 1rem}}"
        "table{border-collapse:collapse}td,th{padding:.6rem;border:1px solid #ccc;text-align:left}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        f"<h1>{_escape(title)}</h1>{body}</html>"
    )


def render_single(report: Mapping[str, Any]) -> str:
    parts = [
        "<p>Absolute clean timings. Profiling and execution probes run separately from benchmark clocks. "
        "Warmups and repeated samples execute the workflow multiple times.</p>",
        f"<p>Collection: <strong>{_escape(report['collection_status'])}</strong>; captures: "
        f"<strong>{_escape(report['capture_status'])}</strong>; numerical observations: "
        f"<strong>{_escape(report['observation_check']['status'])}</strong>.</p>",
        f"<p>Timing scope: {_escape(report['scope'])}</p>",
        "<h2>Clean timings</h2><p>Seconds, grouped by seed. Variability is unavailable for a single sample.</p>",
        "<table><tr><th>Seed</th><th>Samples</th><th>Mean</th><th>Median</th><th>Standard deviation</th><th>Range</th></tr>",
    ]
    for row in report["timing"]["per_seed"]:
        stats = row["seconds"]
        parts.append(
            "<tr>"
            + "".join(
                f"<td>{_escape(value)}</td>"
                for value in (
                    row["seed"],
                    stats["count"],
                    stats["mean"],
                    stats["median"],
                    stats["standard_deviation"]
                    if stats["standard_deviation"] is not None
                    else "unavailable",
                    f"{stats['min']} – {stats['max']}",
                )
            )
            + "</tr>"
        )
    parts.extend(
        [
            "</table>",
            _details("Individual clean samples", report["timing"]["per_seed"]),
            "<h2>Peak memory</h2><p>Allocated and reserved bytes for the clean timing scope. "
            "Cold-process memory is unavailable; resident diagnostic memory does not describe the cold process.</p>",
            f"<p>Status: {_escape(report['memory']['status'])}</p>",
            _details("Scoped memory samples", report["memory"]),
            "<h2>Execution and provenance</h2>",
            _details(
                "Source, inputs and environment",
                {k: report[k] for k in ("provenance", "input_identity", "context")},
            ),
            _details("Execution settings and observed dispatch", report["execution"]),
            _details(
                "Cache and reset policy",
                {k: report[k] for k in ("cache_policy", "reset_policy", "passes")},
            ),
            "<h2>Workflow outputs</h2><p>"
            + _anchor(report["receipt"], "Measurement receipt")
            + "</p><ul>",
        ]
    )
    parts.extend(
        f"<li>{_anchor(row['link'], row['file'])} (seed {row['seed']}, repeat {row['repeat']})</li>"
        for row in report["outputs"]
    )
    parts.append("</ul>")
    if report["resident_outputs"]:
        parts.append(
            "<h3>Resident reference outputs</h3>"
            "<p>Separate resident workflow pass; excluded from cold timing.</p><ul>"
        )
        parts.extend(
            f"<li>{_anchor(row['link'], row['file'])} (seed {row['seed']}, repeat {row['repeat']})</li>"
            for row in report["resident_outputs"]
        )
        parts.append("</ul>")
    parts.extend(
        [
            "<h2>Profiler captures</h2><p>Open Torch traces in a Chrome-trace viewer and native reports in their profiler. Capture durations are separate diagnostic measurements.</p>",
            _details("Cold/warm coverage", report["capture_pairs"]),
        ]
    )
    for capture in report["captures"]:
        parts.append(
            f"<h3>{_escape(capture['backend'])} / {_escape(capture['mode'])}: {_escape(capture['status'])}</h3>"
        )
        for category in ("artifact_links", "log_links", "output_links"):
            parts.extend(
                "<p>" + _anchor(link, link.split("/")[-1]) + "</p>"
                for link in capture[category]
            )
        parts.append(_details("Tool, options, coverage and output checks", capture))
    trace = report["trace"]
    if trace["status"] != "not_requested":
        parts.append(f"<h3>Torch trace: {_escape(trace['status'])}</h3>")
        if "artifact_link" in trace:
            parts.append(
                "<p>" + _anchor(trace["artifact_link"], "Torch trace") + "</p>"
            )
        parts.append(_details("Trace evidence", trace))
    parts.append(_details("Numerical output checks", report["observation_check"]))
    return _page(f"Single-source measurement: {report['case']}", "".join(parts))


def write_single_reports(
    collections: Mapping[str, Mapping[str, Path]],
    plan: Mapping[str, Any],
    output: Path,
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    study: dict[str, Any] = {
        "schema_version": 1,
        "mode": "single",
        "analysis": analysis,
        "plan": plan,
        "cases": {},
    }
    index = {}
    rows = []
    for case, targets in collections.items():
        if set(targets) != {"source"}:
            raise SingleReportError(case, "requires exactly one source target")
        # Reserved study filenames live in a separate directory, so arbitrary
        # authored case names cannot collide with either the index or each other.
        stem = f"cases/{case}" if case in {"index", "study"} else case
        files = {"summary": f"{stem}.json", "report": f"{stem}.html"}
        report_directory = (output / files["report"]).parent
        report_directory.mkdir(parents=True, exist_ok=True)
        report = summarize_single(targets["source"], report_directory)
        expected_policy = plan.get("observation_policy")
        if expected_policy is None and plan.get("observations"):
            expected_policy = "required"
        if (
            expected_policy is not None
            and report["observation_policy"] != expected_policy
        ):
            raise SingleReportError(
                case, "receipt observation policy differs from the authored plan"
            )
        report["analysis"] = analysis
        write_record(output / files["summary"], report)
        (output / files["report"]).write_text(render_single(report))
        study["cases"][case] = {"measurement": report, "files": files}
        index[case] = files
        rows.append(
            f"<tr><td>{_escape(case)}</td><td>{_escape(report['scope'])}</td>"
            f"<td>{_escape(report['collection_status'])}</td><td>{_escape(report['capture_status'])}</td>"
            f"<td>{_anchor(files['report'], 'Measurements and artifacts')} · {_anchor(files['summary'], 'JSON')}</td></tr>"
        )
    write_record(output / "study.json", study)
    write_record(output / "index.json", index)
    (output / "index.html").write_text(
        _page(
            "Single-source measurement study",
            "<p>Absolute timings and separate profiling artifacts for one source revision.</p>"
            "<table><tr><th>Case</th><th>Timing scope</th><th>Collection</th><th>Captures</th><th>Evidence</th></tr>"
            + "".join(rows)
            + "</table><p>"
            + _anchor("study.json", "Study JSON")
            + "</p>"
            + _details(
                "Plan and analysis provenance", {"plan": plan, "analysis": analysis}
            ),
        )
    )
    return study
