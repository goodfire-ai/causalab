"""Visible evidence limits, without a scientific acceptance or optimization verdict."""

from collections import Counter
from dataclasses import dataclass
import html
import json
from typing import Any, Literal, Mapping, TypedDict


@dataclass(eq=False)
class ComparisonModeError(ValueError):
    modes: tuple[Any, Any]

    def __str__(self) -> str:
        return "incomparable comparison modes: workers must attest the same mode"


def comparison_mode(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> Literal["code", "workflow"]:
    """Require the same known comparison mode for analysis and direct summaries."""
    modes = (
        before.get("context", {}).get("worker", {}).get("comparison_kind", "code"),
        after.get("context", {}).get("worker", {}).get("comparison_kind", "code"),
    )
    match modes:
        case ("workflow", "workflow"):
            return "workflow"
        case ("code", "code"):
            return "code"
        case _:
            raise ComparisonModeError(modes)


class ArmSourceEvidence(TypedDict):
    benchmark_identity: str | None
    source_commit: str | None
    source_pins: dict[str, dict[str, str]] | None
    shared_pins: dict[str, dict[str, str]] | None


class BenchmarkEvidence(TypedDict):
    shared_identity: str | None
    identity_status: Literal["matched", "missing", "mismatch"]
    arms: dict[Literal["before", "after"], ArmSourceEvidence]


def _benchmark_evidence(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> BenchmarkEvidence:
    arms: dict[Literal["before", "after"], ArmSourceEvidence] = {}
    records: dict[Literal["before", "after"], Mapping[str, Any]] = {
        "before": before,
        "after": after,
    }
    for arm, record in records.items():
        worker = record.get("context", {}).get("worker", {})
        arms[arm] = {
            "benchmark_identity": worker.get("benchmark_identity"),
            "source_commit": worker.get("source_commit"),
            "source_pins": worker.get("source_pins"),
            "shared_pins": worker.get("shared_pins"),
        }
    baseline, candidate = before.get("input_identity"), after.get("input_identity")
    status = (
        "missing"
        if baseline is None or candidate is None
        else "matched"
        if baseline == candidate
        else "mismatch"
    )
    return {
        "shared_identity": baseline if status == "matched" else None,
        "identity_status": status,
        "arms": arms,
    }


def comparison_summary(before, after, bs, ats, identities, training):
    caveats, execution, outputs = [], {}, {}
    cache_policy = {
        arm: record.get("context", {}).get("cache_policy")
        for arm, record in (("before", before), ("after", after))
    }
    workers = [
        record.get("context", {}).get("worker", {}) for record in (before, after)
    ]
    workflow_mode = comparison_mode(before, after) == "workflow"
    configuration = None
    if workflow_mode:
        configuration = {
            "contract": workers[0].get("comparison_contract"),
            "reference_changes": workers[0].get("configuration_changes", []),
            "changes": workers[1].get("configuration_changes", []),
        }
        caveats.append(
            "Workflow configurations differ or are being compared explicitly. Numerical contrasts compare configured outputs; "
            "matching observation names and shapes does not establish equivalent scientific meaning or equal work. "
            "Review configuration differences, data roles and training budgets before interpreting speedups."
        )
    repetitions = Counter(seed for seed, _ in identities)
    for arm, record, samples in (("before", before, bs), ("after", after, ats)):
        worker = record.get("context", {}).get("worker", {})
        declared = worker.get("execution", {})
        probe = worker.get("execution_probe", {}).get("cases", {}).get(record["case"])
        dispatch = probe.get("python_dispatch", []) if probe else []
        replay = any(
            "graph_replay" in name or "CUDAGraph.replay" in name for name in dispatch
        )
        execution[arm] = {
            "declared": declared,
            "probe_status": "collected" if probe else "not_collected",
            "graph_replay_observed": replay if probe else None,
            "python_dispatch": dispatch,
            "native_profile": probe.get("native_profile", {}) if probe else {},
        }
        if not probe:
            caveats.append(f"{arm}: execution probe evidence is missing.")
        elif declared.get("cuda_graphs") and not replay:
            caveats.append(
                f"{arm}: CUDA graphs requested, but graph replay was not observed in the untimed probe."
            )
        origins, checks = Counter(), Counter()
        for identity in identities:
            sample = samples[identity]
            origins[sample.get("observation_origin", "unspecified")] += 1
            check = sample.get("observer_check", {})
            state = (
                "equal"
                if check.get("exactly_equal") is True
                and check.get("observation_specs_match") is True
                else "different"
                if check.get("exactly_equal") is False
                or check.get("observation_specs_match") is False
                else "unavailable"
            )
            checks[state] += 1
        outputs[arm] = {"origins": dict(origins), "diagnostic_checks": dict(checks)}
        if set(origins) - {"timing_pass", "cold_timing_pass"}:
            caveats.append(
                f"{arm}: primary outputs are not attested as saved timing-pass outputs."
            )
        if checks["different"]:
            caveats.append(
                f"{arm}: {checks['different']} diagnostic runs differ from clean outputs or their interpretation; stochastic variation and observer effects are not separated."
            )
        if checks["unavailable"]:
            caveats.append(
                f"{arm}: {checks['unavailable']} clean/diagnostic output checks are unavailable."
            )
    if execution["before"]["declared"] != execution["after"]["declared"]:
        caveats.append(
            "Declared execution settings differ between arms; interpret this as a comparison of those configurations."
        )
    memo_capabilities = [
        (policy or {}).get("host_memos") for policy in cache_policy.values()
    ]
    if (
        all(value is not None for value in memo_capabilities)
        and memo_capabilities[0] != memo_capabilities[1]
    ):
        caveats.append(
            "Observed host memo capabilities differ between arms; timing differences may include implementation support for memoization. Availability does not establish cache hits or warmed occupancy."
        )
    for field in ("hardware", "environment"):
        workers = [
            record.get("context", {}).get("worker", {}) for record in (before, after)
        ]
        if (
            all(field in worker for worker in workers)
            and workers[0][field] != workers[1][field]
        ):
            caveats.append(
                f"Observed {field} differs between arms; inspect provenance before attributing the timing change to code."
            )
    work = []
    unavailable = 0
    for sample in training["per_sample"]:
        if sample["status"] != "compared":
            unavailable += 1
        for fit in sample["fits"]:
            work.append(
                {
                    "seed": sample["seed"],
                    "repeat": sample["repeat"],
                    "fit": fit["before"]["identity"],
                    "before_updates": fit["before"]["optimizer_steps"],
                    "after_updates": fit["after"]["optimizer_steps"],
                    "updates_match": fit["optimizer_steps_match"],
                    "schedule_matches": fit["logical_schedule_matches"],
                    "initial_parameters_match": fit["initial_parameters_match"],
                    "initial_optimizer_match": fit["initial_optimizer_match"],
                    "observed_rng_states_match": fit["observed_rng_states_match"],
                }
            )
    if any(not row["updates_match"] or not row["schedule_matches"] for row in work):
        caveats.append(
            "Diagnostic update counts or logical schedules differ. Wall-time changes may include changed training work or stopping; they do not establish a same-work implementation speedup."
        )
    if any(
        not row["initial_parameters_match"] or not row["initial_optimizer_match"]
        for row in work
    ):
        caveats.append("Diagnostic fitted initial states differ between arms.")
    if any(not row["observed_rng_states_match"] for row in work):
        caveats.append("Observed diagnostic RNG boundary states differ between arms.")
    if unavailable:
        caveats.append(
            f"Paired training diagnostics unavailable for {unavailable} samples (non-fitting operations may have none)."
        )
    return {
        "status": "caveats_present" if caveats else "no_detected_caveats",
        "policy": "Evidence summary, not acceptance. Seed-bootstrap intervals are descriptive, not calibrated performance regression gates; inspect replication and A/A controls. Output origins identify native saves; added common evaluation is untimed replay of those fits. Execution probes and training diagnostics are separate runs; even equal saved outputs do not prove identical timed work. No stopping reason is inferred from update counts.",
        "replication": {
            "seed_count": len(repetitions),
            "repeats_per_seed": [
                {"seed": seed, "repeats": count}
                for seed, count in sorted(repetitions.items())
            ],
        },
        "caveats": caveats,
        "benchmark": _benchmark_evidence(before, after),
        **(
            {"configuration_comparison": configuration}
            if configuration is not None
            else {}
        ),
        "execution": execution,
        "cache_policy": cache_policy,
        "outputs": outputs,
        "diagnostic_work": work,
    }


def render_summary(summary):
    def escape(value):
        return html.escape(str(value), quote=True)

    benchmark = summary["benchmark"]
    configuration = summary.get("configuration_comparison")
    benchmark_label = (
        "Shared benchmark identity (equality enforced)"
        if benchmark["identity_status"] == "matched"
        else "Shared benchmark identity"
    )
    benchmark_value = (
        benchmark["shared_identity"]
        if benchmark["identity_status"] == "matched"
        else "not recorded"
        if benchmark["identity_status"] == "missing"
        else "mismatch"
    )
    if configuration is not None:
        benchmark_label = benchmark_label.replace(
            "Shared benchmark identity", "Shared comparison contract"
        )
    definition_label = (
        "Workflow definition" if configuration is not None else "Benchmark identity"
    )
    introduction = (
        "Workflow comparisons execute independently authored definitions at one code commit. "
        "Each definition and its pins are verified independently; shared inputs remain checked. "
        "Configuration differences below are relative to the study baseline."
        if configuration is not None
        else "Paired studies execute each arm's committed source against one shared benchmark. "
        "Source hashes cover directly referenced modules and may differ; the installation receipt attests the whole package. "
        "Empty pin maps contain no referenced pins."
    )
    configuration_html = (
        "<details><summary>Configuration differences</summary><pre>"
        + escape(json.dumps(configuration, indent=2, sort_keys=True))
        + "</pre></details>"
        if configuration is not None
        else ""
    )
    source_rows = []
    for arm, label in (("before", "Baseline (before)"), ("after", "Candidate (after)")):
        evidence = benchmark["arms"][arm]
        fields = [evidence["benchmark_identity"], evidence["source_commit"]]
        fields.extend(
            json.dumps(evidence[field], sort_keys=True)
            if evidence[field] is not None
            else None
            for field in ("source_pins", "shared_pins")
        )
        source_rows.append(
            f"<tr><td>{label}</td>"
            + "".join(
                f"<td>{escape(value if value is not None else 'not recorded')}</td>"
                for value in fields
            )
            + "</tr>"
        )

    rows = []
    for arm, value in summary["execution"].items():
        outputs = summary["outputs"][arm]
        rows.append(
            f"<tr><td>{escape(arm)}</td><td>{escape(json.dumps(value['declared'], sort_keys=True))}</td>"
            f"<td>{escape(value['graph_replay_observed'])}</td>"
            f"<td>{escape(value['native_profile'].get('status', 'unavailable'))}</td>"
            f"<td>{escape(outputs['origins'])}</td><td>{escape(outputs['diagnostic_checks'])}</td></tr>"
        )
    work = [
        f"<tr><td>{row['seed']}/{row['repeat']}</td><td>{escape(json.dumps(row['fit'], sort_keys=True))}</td>"
        f"<td>{row['before_updates']}</td><td>{row['after_updates']}</td>"
        f"<td>{row['schedule_matches']}</td></tr>"
        for row in summary["diagnostic_work"]
    ]
    cache_rows = []
    for arm, recorded in summary["cache_policy"].items():
        policy = recorded or {}
        memos = policy.get("host_memos")
        cache_rows.append(
            f"<tr><td>{escape(arm)}</td><td>{escape(policy.get('lifetime', 'not recorded'))}</td>"
            f"<td>{escape(policy.get('observed_at', 'not recorded'))}</td>"
            f"<td>{escape(json.dumps(memos, sort_keys=True) if memos is not None else 'not recorded')}</td>"
            f"<td>{escape(policy.get('external_caches', 'not recorded'))}</td></tr>"
        )
    return (
        f"<h2>Comparison evidence: {escape(summary['status'])}</h2><p>{escape(summary['policy'])}</p>"
        f"<h3>Benchmark and source</h3><p>{benchmark_label}: {escape(benchmark_value)}.</p>"
        f"<p>{introduction}</p>{configuration_html}"
        f"<table><tr><th>Arm</th><th>{definition_label}</th><th>Source commit</th>"
        "<th>Source hashes</th><th>Shared input hashes</th></tr>"
        + "".join(source_rows)
        + "</table>"
        f"<p>Independent seeds: {summary['replication']['seed_count']}; repeats per seed: "
        f"{escape(summary['replication']['repeats_per_seed'])}.</p><ul>"
        + "".join(f"<li>{escape(note)}</li>" for note in summary["caveats"])
        + "</ul><table><tr><th>Arm</th><th>Declared execution</th><th>Graph replay in probe</th>"
        "<th>Native probe profiling</th><th>Primary output source</th><th>Clean/diagnostic checks</th></tr>"
        + "".join(rows)
        + "</table><h3>Cache policy</h3>"
        "<p>Host memo attributes describe implementation capability, not cache hits or warmed occupancy. "
        "Discovery covers known attributes in loaded modules and is not exhaustive; module_not_loaded means availability was not inspected.</p>"
        "<table><tr><th>Arm</th><th>Process lifetime</th><th>Observed at</th>"
        "<th>Host memo capabilities</th><th>External caches</th></tr>"
        + "".join(cache_rows)
        + "</table><details><summary>Diagnostic training work</summary>"
        "<table><tr><th>Seed/repeat</th><th>Fit</th><th>Before updates</th><th>After updates</th>"
        "<th>Same logical schedule</th></tr>" + "".join(work) + "</table></details>"
    )
