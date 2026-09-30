"""Verify and compare profiler captures without interpreting native trace events."""

from __future__ import annotations

from collections import defaultdict
import html
import json
from pathlib import Path


def validate_captures(record, root, samples):
    from causalab.measurement.analysis.receipts import (
        verified_artifact,
        verify_output_files,
    )

    numerical = record.get("observation_policy", "required") == "required"
    if not numerical and (
        record.get("observation_policy") != "not_requested"
        or record.get("mode") != "single"
    ):
        raise ValueError("invalid capture observation policy")
    captures = record.get("captures", [])
    if not isinstance(captures, list):
        raise ValueError("captures must be a list")
    seen = set()
    paired_samples = {}
    for capture in captures:
        if not isinstance(capture, dict):
            raise ValueError("capture must be an object")
        backend, mode = capture.get("backend"), capture.get("mode")
        if not isinstance(backend, str) or not backend or mode not in {"cold", "warm"}:
            raise ValueError("capture requires backend and cold/warm mode")
        identity = capture.get("id")
        if identity != f"{backend}:{mode}" or identity in seen:
            raise ValueError("duplicate or invalid capture identity")
        seen.add(identity)
        if (
            type(capture.get("seed")) is not int
            or type(capture.get("repeat")) is not int
            or (capture["seed"], capture["repeat"]) not in samples
        ):
            raise ValueError("capture has no corresponding clean sample")
        sample_key = (capture["seed"], capture["repeat"])
        if backend in paired_samples and paired_samples[backend] != sample_key:
            raise ValueError("cold/warm captures must share a seed and repeat")
        paired_samples[backend] = sample_key
        if capture.get("status") not in {
            "completed",
            "failed",
            "unavailable",
            "interrupted",
        }:
            raise ValueError("invalid capture status")
        artifacts = capture.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise ValueError("capture artifacts must be a list")
        for ref in artifacts:
            path = verified_artifact(root, ref)
            if not path.stat().st_size:
                raise ValueError("empty profiler artifact")
        for ref in capture.get("logs", []):
            verified_artifact(root, ref)
        verify_output_files(root, capture.get("output_files", {}))
        if not numerical and (
            capture.get("observation_status") != "not_requested"
            or any(
                "observation" in name and name != "observation_status"
                for name in capture
            )
        ):
            raise ValueError(
                "timing-only capture has inconsistent observation evidence"
            )
        if "observations" in capture:
            verified_artifact(root, capture["observations"])
        if capture["status"] == "completed":
            if not artifacts:
                raise ValueError("completed capture needs native artifacts")
            if numerical and "observations" not in capture:
                raise ValueError("completed capture needs observations")
    plan = record.get("capture_plan")
    if plan is not None:
        if (
            not isinstance(plan, dict)
            or set(plan) != {"backends", "modes"}
            or not isinstance(plan["backends"], list)
            or not plan["backends"]
            or any(not isinstance(b, str) or not b for b in plan["backends"])
            or len(set(plan["backends"])) != len(plan["backends"])
            or plan["modes"] not in (["warm"], ["cold", "warm"])
        ):
            raise ValueError("invalid capture plan")
        expected = {f"{b}:{m}" for b in plan["backends"] for m in plan["modes"]}
        if seen - expected:
            raise ValueError("unexpected capture outside declared plan")


def compare_captures(record, path, samples):
    from causalab.measurement.analysis.receipts import verified_artifact

    results = []
    for original in record.get("captures", []):
        capture = dict(original)
        capture["artifact_paths"] = [
            str(verified_artifact(path.parent, ref))
            for ref in capture.get("artifacts", [])
        ]
        capture["log_paths"] = [
            str(verified_artifact(path.parent, ref)) for ref in capture.get("logs", [])
        ]
        if record.get("observation_policy") == "not_requested":
            capture["observation_check"] = {"status": "not_requested"}
        elif capture["status"] == "completed":
            import torch
            from safetensors.torch import load_file
            from causalab.measurement.analysis.compare import (
                paired_difference,
                observation_spec,
            )

            sample = samples[capture["seed"], capture["repeat"]]
            paired_warm = capture["mode"] == "warm" and "cold" in record.get(
                "capture_plan", {}
            ).get("modes", [])
            if paired_warm and "resident_unobserved_observations" not in sample:
                capture["comparison_reference"] = (
                    "missing clean resident timing-pass outputs"
                )
                capture["observation_check"] = {
                    "status": "missing_clean_warm_reference"
                }
                results.append(capture)
                continue
            reference_field = (
                "resident_unobserved_observations"
                if capture["mode"] == "warm"
                and "resident_unobserved_observations" in sample
                else "unobserved_observations"
            )
            spec_field = reference_field.replace("observations", "observation_specs")
            if reference_field in sample:
                reference = load_file(
                    str(verified_artifact(path.parent, sample[reference_field]))
                )
                reference_specs = {"observation_specs": sample.get(spec_field, {})}
                capture["comparison_reference"] = (
                    "clean resident timing-pass outputs"
                    if reference_field.startswith("resident")
                    else "clean timing-pass outputs"
                )
            else:
                reference, reference_specs = sample["tensors"], sample
                capture["comparison_reference"] = (
                    "sample observations; clean origin unspecified"
                )
            traced = load_file(
                str(verified_artifact(path.parent, capture["observations"]))
            )
            keys = set(traced)
            if (
                not keys
                or keys != set(reference)
                or any(traced[k].shape != reference[k].shape for k in keys)
            ):
                check = {"status": "unaligned"}
            elif any(
                value.is_complex() or not torch.isfinite(value).all()
                for k in keys
                for value in (traced[k], reference[k])
            ):
                check = {"status": "nonfinite_or_complex"}
            else:
                check = {
                    "status": "compared",
                    "not_captured": sorted(set(reference) - keys),
                    "exactly_equal": all(
                        torch.equal(reference[k], traced[k]) for k in keys
                    ),
                    "observation_specs_match": all(
                        observation_spec(record, k, reference_specs)
                        == observation_spec(record, k, capture)
                        for k in keys
                    ),
                    "drift": {
                        k: paired_difference(
                            reference[k].to(torch.float64),
                            traced[k].to(torch.float64),
                            observation_spec(record, k, reference_specs),
                            observation_spec(record, k, capture),
                            observation=k,
                        )
                        for k in sorted(keys)
                    },
                }
            capture["observation_check"] = check
        results.append(capture)
    return results


def capture_pairs(record, captures):
    grouped = defaultdict(dict)
    for capture in captures:
        grouped[capture["backend"]][capture["mode"]] = capture["status"]
    plan = record.get("capture_plan", {})
    for backend in plan.get("backends", []):
        for mode in plan.get("modes", []):
            grouped[backend].setdefault(mode, "missing")
    return [
        {
            "backend": backend,
            "modes": modes,
            "status": "completed"
            if all(v == "completed" for v in modes.values())
            else "incomplete",
        }
        for backend, modes in sorted(grouped.items())
    ]


def render_captures(report):
    if not any(report.get("captures", {}).values()) and not any(
        report.get("capture_pairs", {}).values()
    ):
        return ""
    parts = [
        "<h2>Profiler captures</h2><p>Separate diagnostic executions. "
        "Capture durations and replayed kernel metrics are not clean benchmark timings.</p>"
    ]
    for arm, pairs in report.get("capture_pairs", {}).items():
        for pair in pairs:
            parts.append(
                "<p>"
                + html.escape(
                    f"{arm} / {pair['backend']}: {pair['status']} — {pair['modes']}"
                )
                + "</p>"
            )
    for arm, captures in report.get("captures", {}).items():
        for capture in captures:
            parts.append(
                "<h3>"
                + html.escape(
                    f"{arm} / {capture['backend']} / {capture['mode']}: {capture['status']}"
                )
                + "</h3>"
            )
            for path in capture.get("artifact_paths", []):
                uri = html.escape(Path(path).as_uri(), quote=True)
                label = html.escape(
                    f"{Path(path).name} ({capture.get('artifact_format', 'native profiler artifact')})"
                )
                parts.append(f'<p><a href="{uri}">{label}</a></p>')
            for path in capture.get("log_paths", []):
                uri = html.escape(Path(path).as_uri(), quote=True)
                parts.append(
                    f'<p><a href="{uri}">{html.escape(Path(path).name)}</a> (capture log)</p>'
                )
            details = {
                k: v
                for k, v in capture.items()
                if k not in {"artifact_paths", "log_paths"}
            }
            parts.append(
                "<details><summary>Coverage, configuration and output comparison</summary><pre>"
                + html.escape(json.dumps(details, indent=2))
                + "</pre></details>"
            )
    return "".join(parts)
