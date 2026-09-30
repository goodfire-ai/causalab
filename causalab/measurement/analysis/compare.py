"""Before/after timing, tensor drift and repeated-run variance comparison.

Workflow inputs ``before`` and ``after`` are measurement JSON paths; outputs are
``summary`` (JSON) and ``report`` (HTML). Analysis verifies manifests, tensors and
traces without running models. Heavy imports are function-local.
"""

from __future__ import annotations

from collections.abc import Mapping
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any

from causalab.measurement.collection import file_hash, write_record
from causalab.measurement.analysis.statistics import describe as _stats
from causalab.measurement.analysis.receipts import load_measurement, verified_artifact


def _ratio(after: float | None, before: float | None) -> float | None:
    return after / before if after is not None and before not in (None, 0) else None


def _differences(before: Any, after: Any) -> dict[str, Any]:
    import torch

    delta = after - before
    norm_before = torch.linalg.vector_norm(before)
    norm_after = torch.linalg.vector_norm(after)
    before_rms = float(before.square().mean().sqrt())
    rms = float(delta.square().mean().sqrt())
    cosine = None
    if norm_before > 0 and norm_after > 0:
        similarity = float(torch.sum((before / norm_before) * (after / norm_after)))
        cosine = 1 - min(1.0, max(-1.0, similarity))
    return {
        "max_abs": float(delta.abs().max()),
        "rms": rms,
        "relative_rms": rms / before_rms if before_rms else None,
        "cosine_distance": cosine,
        "signed_mean_change": float(delta.mean()),
    }


def observation_spec(
    record: Mapping[str, Any], key: str, sample: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    specs = {
        **record.get("observation_specs", {}),
        **(sample or {}).get("observation_specs", {}),
    }
    # A gate's entry temperature/provenance cannot be inferred from its base
    # declaration, or from the parent of a ragged row. Refuse missing metadata.
    if key not in specs and any(
        isinstance(spec, dict)
        and spec.get("kind") == "gate"
        and key.startswith(name + "/")
        for name, spec in specs.items()
    ):
        from causalab.measurement.analysis.stability import (
            MissingGateObservationSpecError,
        )

        raise MissingGateObservationSpecError(key)
    spec = specs.get(key, specs.get(key.split("/", 1)[0], {"kind": "tensor"}))
    if not isinstance(spec, dict) or spec.get("kind") not in {
        "tensor",
        "table",
        "subspace",
        "gate",
    }:
        raise ValueError(f"invalid observation specification for {key}")
    if spec["kind"] == "gate":
        from causalab.measurement.analysis.stability import require_sigmoid_gate

        require_sigmoid_gate(key, spec.get("parametrization", "sigmoid"))
    return spec


def paired_difference(
    before: Any,
    after: Any,
    bs: Mapping[str, Any],
    ats: Mapping[str, Any],
    *,
    observation: str,
) -> dict[str, Any]:
    from causalab.measurement.analysis.stability import (
        compare_gates,
        compare_subspaces,
        require_sigmoid_gate,
    )

    if bs["kind"] != ats["kind"]:
        raise ValueError("before/after observation kinds must match")
    if bs["kind"] == "subspace":
        return compare_subspaces(before, after)
    if bs["kind"] == "gate":
        require_sigmoid_gate(observation, bs.get("parametrization", "sigmoid"))
        require_sigmoid_gate(observation, ats.get("parametrization", "sigmoid"))
        return compare_gates(
            before,
            after,
            before_temperature=bs["temperature"],
            after_temperature=ats["temperature"],
        )
    return _differences(before, after)


def _variance(before: Any, after: Any) -> dict[str, Any]:
    """Across runs, then equal weight per aligned tensor coordinate."""
    count = before.shape[0]
    vb = float(before.var(dim=0, correction=1).mean()) if count > 1 else None
    va = float(after.var(dim=0, correction=1).mean()) if count > 1 else None
    scalar = before[0].numel() == 1
    return {
        "count": count,
        "before_mean": float(before.mean()) if scalar else None,
        "after_mean": float(after.mean()) if scalar else None,
        "before_values": before.reshape(-1).tolist() if scalar else None,
        "after_values": after.reshape(-1).tolist() if scalar else None,
        "before_standard_deviation": math.sqrt(vb)
        if scalar and vb is not None
        else None,
        "after_standard_deviation": math.sqrt(va)
        if scalar and va is not None
        else None,
        "before_mean_coordinate_variance": vb,
        "after_mean_coordinate_variance": va,
        "variance_change": va - vb if va is not None and vb is not None else None,
        "variance_ratio": _ratio(va, vb),
        "mean_output_drift": _differences(before.mean(dim=0), after.mean(dim=0)),
    }


def _bootstrap(values: list[float], *, draws: int, seed: int) -> dict[str, Any]:
    """Seed-level percentile bootstrap; never treat repeats as independent seeds."""
    import numpy as np

    if len(values) < 2:
        return {"interval": None, "reason": "fewer than two independent seeds"}
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(rng.choice(values, size=len(values), replace=True)))
        for _ in range(draws)
    ]
    return {
        "interval": np.quantile(means, [0.025, 0.975]).tolist(),
        "method": "percentile bootstrap of mean paired seed effects",
        "confidence": 0.95,
        "resampling_unit": "seed",
        "draws": draws,
        "seed": seed,
    }


def compare(
    before_path: Path,
    after_path: Path,
    *,
    bootstrap_draws: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file
    from causalab.measurement.analysis.summary import comparison_mode
    from causalab.measurement.analysis.training import compare_training

    if type(bootstrap_draws) is not int or bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be an integer >= 100")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    before, bs = load_measurement(before_path)
    after, ats = load_measurement(after_path)
    comparison = comparison_mode(before, after)
    for field in ("case", "input_identity", "scope", "reset_policy", "alignment"):
        if before[field] != after[field]:
            raise ValueError(
                f"incomparable {field}: {before[field]!r} != {after[field]!r}"
            )
    if set(bs) != set(ats):
        raise ValueError("before/after seed and repetition identities must match")
    identities = sorted(bs)
    keys = set(bs[identities[0]]["tensors"])
    shapes = {k: bs[identities[0]]["tensors"][k].shape for k in keys}
    dtypes: dict[str, set[str]] = {"before": set(), "after": set()}
    for arm, samples in (("before", bs), ("after", ats)):
        for sample in samples.values():
            tensors = sample["tensors"]
            if set(tensors) != keys:
                raise ValueError("missing or unexpected logical observation keys")
            for key, tensor in tensors.items():
                if tensor.shape != shapes[key] or not tensor.numel():
                    raise ValueError(f"unaligned tensor shape for {key}")
                if tensor.is_complex() or not torch.isfinite(tensor).all():
                    raise ValueError(
                        f"nonfinite or complex observation for {key}; comparison refused"
                    )
                dtypes[arm].add(str(tensor.dtype))
                tensors[key] = tensor.to(torch.float64)
    seeds = sorted({s for s, _ in identities})
    timings = []
    observations: dict[str, Any] = {}
    for seed in seeds:
        ids = [i for i in identities if i[0] == seed]
        b = [bs[i]["seconds"] for i in ids]
        a = [ats[i]["seconds"] for i in ids]
        bv, av = _stats(b), _stats(a)
        timings.append(
            {
                "seed": seed,
                "before": bv,
                "after": av,
                "paired_seconds_change": statistics.mean([y - x for x, y in zip(b, a)]),
                "median_speedup": bv["median"] / av["median"],
                "runtime_variance_ratio": _ratio(
                    av["sample_variance"], bv["sample_variance"]
                ),
            }
        )
    for key in sorted(keys):
        from causalab.measurement.analysis.stability import (
            gate_probabilities,
            gate_selection_frequency,
            summarize_subspaces,
        )

        specs_b = {i: observation_spec(before, key, bs[i]) for i in identities}
        specs_a = {i: observation_spec(after, key, ats[i]) for i in identities}
        spec_b = specs_b[identities[0]]
        if len({s["kind"] for s in [*specs_b.values(), *specs_a.values()]}) != 1:
            raise ValueError(f"observation kind changed for {key}")
        values_b = {i: bs[i]["tensors"][key] for i in identities}
        values_a = {i: ats[i]["tensors"][key] for i in identities}
        if spec_b["kind"] == "subspace":
            value = summarize_subspaces(values_b, values_a)
            value["mean_paired_projector_distance_ci"] = _bootstrap(
                [
                    statistics.mean(
                        row["projector_frobenius_distance"]
                        for row in value["paired_drift"]
                        if row["seed"] == seed
                    )
                    for seed in seeds
                ],
                draws=bootstrap_draws,
                seed=bootstrap_seed,
            )
            effects = [row["dispersion_change"] for row in value["within_seed"]]
            value["mean_within_seed_dispersion_change_ci"] = (
                _bootstrap(effects, draws=bootstrap_draws, seed=bootstrap_seed)
                if all(effect is not None for effect in effects)
                else {"interval": None, "reason": "fewer than two repeats per seed"}
            )
            observations[key] = value
            continue
        gate = spec_b["kind"] == "gate"
        if gate:
            values_b = {
                i: gate_probabilities(v, specs_b[i]["temperature"])
                for i, v in values_b.items()
            }
            values_a = {
                i: gate_probabilities(v, specs_a[i]["temperature"])
                for i, v in values_a.items()
            }
        per_seed = []
        seed_before, seed_after = [], []
        for seed in seeds:
            ids = [i for i in identities if i[0] == seed]
            b = torch.stack([values_b[i] for i in ids])
            a = torch.stack([values_a[i] for i in ids])
            per_seed.append({"seed": seed, **_variance(b, a)})
            seed_before.append(b.mean(dim=0))
            seed_after.append(a.mean(dim=0))
        variance_effects = [v["variance_change"] for v in per_seed]
        observations[key] = {
            "kind": spec_b["kind"],
            "variance_units": "gate probabilities" if gate else "tensor coordinates",
            "shape": list(shapes[key]),
            "paired_drift": [
                {
                    "seed": s,
                    "repeat": r,
                    **paired_difference(
                        bs[s, r]["tensors"][key],
                        ats[s, r]["tensors"][key],
                        specs_b[s, r],
                        specs_a[s, r],
                        observation=key,
                    ),
                }
                for s, r in identities
            ],
            "within_seed": per_seed,
            "across_seed_means": _variance(
                torch.stack(seed_before), torch.stack(seed_after)
            ),
            "mean_within_seed_variance_change_ci": (
                _bootstrap(variance_effects, draws=bootstrap_draws, seed=bootstrap_seed)
                if all(v is not None for v in variance_effects)
                else {"interval": None, "reason": "fewer than two repeats per seed"}
            ),
            "mean_signed_output_change_ci": _bootstrap(
                [row["mean_output_drift"]["signed_mean_change"] for row in per_seed],
                draws=bootstrap_draws,
                seed=bootstrap_seed,
            ),
        }
        if gate:
            observations[key]["selection_frequency"] = {
                arm: gate_selection_frequency(
                    [samples[i]["tensors"][key] for i in identities]
                )
                for arm, samples in (("before", bs), ("after", ats))
            }
            observations[key]["effectiveness"] = (
                "not inferred; inspect separately declared task metrics"
            )
    from causalab.measurement.analysis.profiles import capture_pairs, compare_captures

    traces, captures, pairs = {}, {}, {}
    for arm, record, path, samples in (
        ("before", before, before_path, bs),
        ("after", after, after_path, ats),
    ):
        trace = dict(record.get("trace", {"status": "not_requested"}))
        if trace["status"] == "completed":
            traced = load_file(
                str(verified_artifact(path.parent, trace["observations"]))
            )
            sample = samples[trace["seed"], trace["repeat"]]
            reference = sample["tensors"]
            reference_specs = sample
            trace["comparison_reference"] = "separate numerical pass"
            if "unobserved_observations" in sample:
                reference = load_file(
                    str(
                        verified_artifact(
                            path.parent, sample["unobserved_observations"]
                        )
                    )
                )
                trace["comparison_reference"] = (
                    "required outputs of the uninstrumented timing pass"
                )
                reference_specs = {
                    "observation_specs": sample["unobserved_observation_specs"]
                }
            captured = set(trace.get("observation_keys", keys))
            if (
                not captured <= keys
                or not captured <= set(reference)
                or set(traced) != captured
                or any(traced[k].shape != shapes[k] for k in captured)
                or any(reference[k].shape != traced[k].shape for k in captured)
            ):
                trace["observation_check"] = {"status": "unaligned"}
            elif any(
                not torch.isfinite(t).all() or t.is_complex() for t in traced.values()
            ) or any(
                not torch.isfinite(reference[k]).all() or reference[k].is_complex()
                for k in captured
            ):
                trace["observation_check"] = {"status": "nonfinite_or_complex"}
            else:
                trace["observation_check"] = {
                    "status": "compared",
                    "not_captured": sorted(keys - captured),
                    "drift": {
                        k: paired_difference(
                            reference[k].to(torch.float64),
                            traced[k].to(torch.float64),
                            observation_spec(record, k, reference_specs),
                            observation_spec(record, k, trace),
                            observation=k,
                        )
                        for k in sorted(captured)
                    },
                }
            trace["absolute_path"] = str((path.parent / trace["file"]).resolve())
        traces[arm] = trace
        captures[arm] = compare_captures(record, path, samples)
        pairs[arm] = capture_pairs(record, captures[arm])
    from causalab.measurement.analysis.summary import comparison_summary

    training = compare_training(bs, ats, identities, comparison=comparison)
    return {
        "schema_version": 1,
        "comparison_summary": comparison_summary(
            before, after, bs, ats, identities, training
        ),
        "case": before["case"],
        "scope": before["scope"],
        "acceptance": "not_evaluated",
        "timing": {
            "per_seed": timings,
            "mean_paired_seconds_change_ci": _bootstrap(
                [v["paired_seconds_change"] for v in timings],
                draws=bootstrap_draws,
                seed=bootstrap_seed,
            ),
        },
        "observations": observations,
        "training": training,
        "memory": {
            "policy": "timing-pass peak allocator bytes; cold-process peaks unavailable",
            "samples": {
                arm: [
                    {
                        "seed": seed,
                        "repeat": repeat,
                        "peak_memory": samples[seed, repeat].get("peak_memory"),
                    }
                    for seed, repeat in identities
                ]
                for arm, samples in (("before", bs), ("after", ats))
            },
        },
        "traces": traces,
        "captures": captures,
        "capture_pairs": pairs,
        "sources": {
            arm: {
                "file": str(path.resolve()),
                "sha256": file_hash(path),
                "provenance": record["provenance"],
                "context": record.get("context", {}),
                "dtypes": sorted(dtypes[arm]),
            }
            for arm, path, record in (
                ("before", before_path, before),
                ("after", after_path, after),
            )
        },
        "policies": {
            "alignment": before["alignment"],
            "nonfinite": "refuse comparison",
            "variance": "unbiased across runs; equal weight per aligned tensor coordinate",
            "zero_denominator": "null; no epsilon",
            "seed_mean_spread": "includes finite-repeat uncertainty",
            "uncertainty": "descriptive percentile intervals; few seeds limit precision",
            "provenance": "differences retained; matched hardware/environment is not attested by comparison",
            "acceptance": "no universal tolerance or optimization recommendation",
        },
    }


def render_report(report: Mapping[str, Any]) -> str:
    """Standalone HTML, retaining the full structured evidence for inspection."""
    from causalab.measurement.analysis.summary import render_summary

    rows = []
    for row in report["timing"]["per_seed"]:
        rows.append(
            f"<tr><td>{row['seed']}</td><td>{row['before']['median']:.6g}</td>"
            f"<td>{row['after']['median']:.6g}</td><td>{row['median_speedup']:.4g}×</td></tr>"
        )
    numerics = []
    for key, value in report["observations"].items():
        for row in value["within_seed"]:
            if value.get("kind") == "subspace":
                numerics.append(
                    f"<tr><td>{html.escape(key)} (projector dispersion)</td><td>{row['seed']}</td>"
                    f"<td>{row['mean_projector_distance']:.6g}</td>"
                    f"<td>{row['before']['mean_squared_projector_distance']}</td>"
                    f"<td>{row['after']['mean_squared_projector_distance']}</td>"
                    f"<td>{row['dispersion_ratio']}</td></tr>"
                )
                continue
            numerics.append(
                f"<tr><td>{html.escape(key)}</td><td>{row['seed']}</td>"
                f"<td>{row['mean_output_drift']['rms']:.6g}</td>"
                f"<td>{row['before_mean_coordinate_variance']}</td>"
                f"<td>{row['after_mean_coordinate_variance']}</td>"
                f"<td>{row['variance_ratio']}</td></tr>"
            )
    across = []
    for key, value in report["observations"].items():
        row = value["across_seed_means"]
        if value.get("kind") == "subspace":
            b, a = (
                row[arm]["mean_squared_projector_distance"]
                for arm in ("before", "after")
            )
            ratio = row["dispersion_ratio"]
            units = "projector dispersion"
        else:
            b, a = (
                row[f"{arm}_mean_coordinate_variance"] for arm in ("before", "after")
            )
            ratio = row["variance_ratio"]
            units = value["variance_units"]
        across.append(
            f"<tr><td>{html.escape(key)}</td><td>{html.escape(units)}</td><td>{b}</td><td>{a}</td><td>{ratio}</td></tr>"
        )
    from causalab.measurement.analysis.profiles import render_captures

    trace_items = []
    for arm, trace in report["traces"].items():
        text = html.escape(f"{arm}: {trace['status']}")
        if trace.get("absolute_path"):
            uri = html.escape(Path(trace["absolute_path"]).as_uri(), quote=True)
            text += f' — <a href="{uri}">{html.escape(trace["absolute_path"])}</a> (open in a Chrome-trace viewer)'
        trace_items.append(f"<li>{text}</li>")
    standalone_traces = (
        "<h2>Traces</h2><ul>" + "".join(trace_items) + "</ul>"
        if any(t["status"] != "not_requested" for t in report["traces"].values())
        or not any(report.get("captures", {}).values())
        else ""
    )
    return """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Before/after measurements</title><style>
body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#202020}
table{border-collapse:collapse;width:100%}th,td{padding:8px;text-align:left;border-bottom:1px solid #ddd}
pre{white-space:pre-wrap;overflow-wrap:anywhere}li{margin:8px 0}</style>""" + (
        f"<h1>{html.escape(str(report['case']))}</h1><p>{html.escape(str(report['scope']))}</p>"
        + (
            f"<p>Reference: {html.escape(report['contrast']['reference'])}; "
            f"candidate: {html.escape(report['contrast']['candidate'])}. "
            "Tables and structured evidence call these before and after, respectively.</p>"
            if "contrast" in report
            else ""
        )
        + "<p>Descriptive comparison. No numerical acceptance threshold has been applied. "
        "Null variance/ratios mean insufficient samples or a zero denominator.</p>"
        + render_summary(report["comparison_summary"])
        + "<h2>Timing</h2><table><tr><th>Seed</th><th>Before median (s)</th>"
        "<th>After median (s)</th><th>Speedup</th></tr>" + "".join(rows) + "</table>"
        "<h2>Within-seed numerical drift and variance</h2><table><tr><th>Observation</th><th>Seed</th>"
        "<th>Mean-output RMS drift</th><th>Before variance</th><th>After variance</th>"
        "<th>Variance ratio</th></tr>" + "".join(numerics) + "</table>"
        "<h2>Across-seed variation</h2><p>Variation of seed means includes finite-repeat uncertainty. "
        "Subspaces use basis-invariant projector dispersion.</p><table><tr><th>Observation</th><th>Units</th>"
        "<th>Before variance / dispersion</th><th>After variance / dispersion</th><th>Ratio</th></tr>"
        + "".join(across)
        + "</table>"
        + standalone_traces
        + render_captures(report)
        + "<h2>Training pairing and observer checks</h2><p>Seed labels alone do not "
        "establish identical initial states or training schedules. Diagnostic state "
        "and output comparisons are retained below; missing evidence is not a match.</p><pre>"
        + html.escape(
            json.dumps(
                report.get("training", {"status": "not_collected"}),
                indent=2,
                allow_nan=False,
            )
        )
        + "</pre>"
        "<details><summary>Raw contrasts, provenance and uncertainty</summary><pre>"
        + html.escape(json.dumps(report, indent=2, allow_nan=False))
        + "</pre></details></html>"
    )


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    if set(inputs) - {
        "before",
        "after",
        "before_sha256",
        "after_sha256",
        "bootstrap_draws",
        "bootstrap_seed",
    }:
        raise ValueError("unknown comparison input")
    if not {"before", "after", "before_sha256", "after_sha256"} <= set(inputs) or set(
        outputs
    ) != {"summary", "report"}:
        raise ValueError(
            "comparison needs before/after inputs with sha256 pins and summary/report outputs"
        )
    for arm in ("before", "after"):
        if file_hash(Path(inputs[arm])) != inputs[f"{arm}_sha256"]:
            raise ValueError(
                f"{arm} receipt hash mismatch; update the authored comparison identity"
            )
    result = compare(
        Path(inputs["before"]),
        Path(inputs["after"]),
        bootstrap_draws=inputs.get("bootstrap_draws", 1000),
        bootstrap_seed=inputs.get("bootstrap_seed", 0),
    )
    write_record(outputs["summary"], result)
    outputs["report"].write_text(render_report(result))
