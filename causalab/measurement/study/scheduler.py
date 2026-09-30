"""Publish complete blocks for every declared source, with verified resume.

Workers execute samples; the controller owns scheduling and publication. Only
complete blocks contribute to reports. Individual workflow steps are not reused.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Protocol
import uuid

from ..collection import file_hash, write_record
from ..analysis.receipts import load_measurement


class Session(Protocol):
    identity: Mapping[str, Any]

    def sample(self, case: str, seed: int, repeat: int, directory: Path) -> Path: ...

    def capture(self, case: str, seed: int, repeat: int, directory: Path) -> Path: ...


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def schedule(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Seed-randomized block order, balanced cyclic arm order across blocks."""
    rng = random.Random(plan["order_seed"])
    cells = [
        (case, seed, repeat)
        for case in plan["cases"]
        for seed in plan["seeds"]
        for repeat in range(plan["repeats"])
    ]
    rng.shuffle(cells)
    arms = list(plan["arms"])
    rng.shuffle(arms)
    return [
        {
            "index": index,
            "case": case,
            "seed": seed,
            "repeat": repeat,
            "arms": arms[index % len(arms) :] + arms[: index % len(arms)],
        }
        for index, (case, seed, repeat) in enumerate(cells)
    ]


def manifest(root: Path) -> dict[str, str]:
    """Hash every published byte, including required outputs and native traces."""
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"measurement artifacts cannot contain symlinks: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = file_hash(path)
    return result


def _verify_block(root: Path, block: Mapping[str, Any]) -> None:
    actual = manifest(root)
    # The publication record hashes its contents, not itself.
    actual.pop("block.json", None)
    if actual != block["files"]:
        raise ValueError(f"completed measurement block changed: {root}")


def _validate_policy(record: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    """The authored policy controls evidence requirements, never the receipt."""
    if record.get("observation_policy", "required") != plan.get(
        "observation_policy", "required"
    ):
        raise ValueError("worker observation policy differs from the authored plan")
    if record.get("mode", "comparison") != plan.get("mode", "comparison"):
        raise ValueError("worker mode differs from the authored plan")


def _load_receipt(
    path: Path, plan: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
    record = json.loads(path.read_text())
    _validate_policy(record, plan)
    return load_measurement(
        path,
        require_observations=plan.get("observation_policy", "required") == "required",
    )


def _aggregate(
    output: Path, plan: Mapping[str, Any], blocks: list[dict[str, Any]]
) -> dict[str, dict[str, Path]]:
    grouped: dict[str, dict[str, Path]] = {}
    for case in plan["cases"]:
        grouped[case] = {}
        for arm in plan["arms"]:
            cells = [b for b in blocks if b["cell"]["case"] == case]
            record = None
            samples, warmups = [], []
            for block in cells:
                path = output / block["directory"] / arm / "measurement.json"
                value = json.loads(path.read_text())
                _validate_policy(value, plan)
                if value["status"] != "completed" or len(value["samples"]) != 1:
                    raise ValueError(
                        "each paired cell must publish exactly one completed sample"
                    )
                if record is None:
                    record = copy.deepcopy(value)
                else:
                    for field in (
                        "case",
                        "scope",
                        "input_identity",
                        "reset_policy",
                        "alignment",
                        "observation_specs",
                        "mode",
                        "observation_policy",
                    ):
                        if value.get(field) != record.get(field):
                            raise ValueError(
                                f"measurement contract changed between cells: {field}"
                            )

                def relocate(ref):
                    return {
                        **ref,
                        "file": (path.parent / ref["file"])
                        .relative_to(output)
                        .as_posix(),
                    }

                sample = copy.deepcopy(value["samples"][0])
                if sample["seed"] != block["cell"]["seed"] or sample["repeat"] != 0:
                    raise ValueError(
                        "worker sample identity disagrees with its scheduled block"
                    )
                sample["repeat"] = block["cell"]["repeat"]
                sample["block"] = block["cell"]["index"]
                if "observations" in sample:
                    sample["observations"] = relocate(sample["observations"])
                sample["output_files"] = {
                    (path.parent / name).relative_to(output).as_posix(): sha
                    for name, sha in sample.get("output_files", {}).items()
                }
                if "resident_output_files" in sample:
                    sample["resident_output_files"] = {
                        (path.parent / name).relative_to(output).as_posix(): sha
                        for name, sha in sample["resident_output_files"].items()
                    }
                for name in (
                    "unobserved_observations",
                    "resident_unobserved_observations",
                    "training_observed_observations",
                    "diagnostic_observations",
                    "resident_diagnostic_observations",
                ):
                    if name in sample:
                        sample[name] = relocate(sample[name])
                if "resident_observations" in sample:
                    sample["resident_observations"] = relocate(
                        sample["resident_observations"]
                    )
                samples.append(sample)
                warmups.extend(
                    {**w, "block": block["cell"]["index"]} for w in value["warmups"]
                )
                if "native_observations" in sample:
                    sample["native_observations"] = relocate(
                        sample["native_observations"]
                    )
                for field in (
                    "workflow_outputs",
                    "diagnostic_workflow_outputs",
                    "resident_diagnostic_workflow_outputs",
                    "timing_directory",
                    "numerics_directory",
                ):
                    if field in sample:
                        sample[field] = (
                            (path.parent / sample[field]).relative_to(output).as_posix()
                        )
                if value["trace"]["status"] != "not_requested" or value.get("captures"):
                    raise ValueError(
                        "clean measurement blocks cannot contain profiler captures"
                    )
            assert record is not None
            record.update(
                plan={
                    "seeds": plan["seeds"],
                    "repeats": plan["repeats"],
                    "warmups": plan["warmups"],
                },
                samples=sorted(samples, key=lambda s: (s["seed"], s["repeat"])),
                warmups=warmups,
                trace={"status": "not_requested"},
                captures=[],
            )
            path = output / f"{case}.{arm}.json"
            write_record(path, record)
            grouped[case][arm] = path
    return grouped


def _collect_profiles(
    plan, output, open_session, identities, study_identity, state, ledger, grouped
):
    """Collect diagnostics after all clean timing blocks are published."""
    from causalab.measurement.analysis.profiles import validate_captures

    state["status"] = "profiling"
    state["profiles"] = []
    write_record(ledger, state)
    for case in plan["profile"]["cases"]:
        for arm in plan["arms"]:
            final = output / "profiles" / case / arm
            expected = {
                "case": case,
                "arm": arm,
                "seed": plan["seeds"][0],
                "repeat": 0,
                "study_identity": study_identity,
            }
            if final.exists():
                publication = json.loads((final / "publication.json").read_text())
                if publication["identity"] != expected:
                    raise ValueError("profile publication belongs to a different study")
                actual = manifest(final)
                actual.pop("publication.json", None)
                if actual != publication["files"]:
                    raise ValueError(f"completed profiler capture changed: {final}")
            else:
                attempt = (
                    output / "profile_attempts" / f"{case}-{arm}-{uuid.uuid4().hex}"
                )
                attempt.parent.mkdir(parents=True, exist_ok=True)
                with open_session(arm) as session:
                    if dict(session.identity) != identities[arm]:
                        raise ValueError(f"{arm} identity changed before profiling")
                    receipt = session.capture(case, plan["seeds"][0], 0, attempt)
                if receipt.resolve() != (attempt / "measurement.json").resolve():
                    raise ValueError("capture published outside its assigned directory")
                capture_record = json.loads(receipt.read_text())
                expected_plan = {
                    "backends": list(plan["profile"]["backends"]),
                    "modes": ["cold", "warm"]
                    if plan["cases"][case].get("cold_process")
                    else ["warm"],
                }
                if capture_record.get("capture_plan") != expected_plan:
                    raise ValueError("capture plan differs from authored profile")
                _validate_policy(capture_record, plan)
                _, clean_samples = _load_receipt(grouped[case][arm], plan)
                validate_captures(capture_record, attempt, clean_samples)
                publication = {"identity": expected, "files": manifest(attempt)}
                write_record(attempt / "publication.json", publication)
                final.parent.mkdir(parents=True, exist_ok=True)
                attempt.rename(final)
            state["profiles"].append(
                {"directory": final.relative_to(output).as_posix(), **publication}
            )
            write_record(ledger, state)
            # Relocate references into the study root without modifying the
            # immutable clean block or profiler publication.
            record = json.loads(grouped[case][arm].read_text())
            captured = json.loads((final / "measurement.json").read_text())
            record["capture_plan"] = captured["capture_plan"]
            record["captures"] = captured["captures"]
            for capture in record["captures"]:
                capture["output_files"] = {
                    (final / name).relative_to(output).as_posix(): sha
                    for name, sha in capture.get("output_files", {}).items()
                }
                refs = [*capture.get("artifacts", []), *capture.get("logs", [])]
                if "observations" in capture:
                    refs.append(capture["observations"])
                for ref in refs:
                    ref["file"] = (final / ref["file"]).relative_to(output).as_posix()
            write_record(grouped[case][arm], record)
            _load_receipt(grouped[case][arm], plan)


def run_schedule(
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    output: Path,
    open_session: Callable[[str], AbstractContextManager[Session]],
    *,
    resume: bool = False,
    finish_block: Callable[[Mapping[str, Any], Path, Mapping[str, Any]], None]
    | None = None,
) -> dict[str, dict[str, Path]]:
    """Run/resume a study under an exclusive process lock.

    Each session must attest its source, inputs, environment, hardware and dispatch
    before returning. All arms are re-attested on resume, before any reuse. A
    session is closed before opening another: full-size models need not coexist.
    """

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=resume)
    with (output / ".controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "another controller owns this measurement directory"
            ) from exc
        ledger = output / "study.json"
        cells = schedule(plan)
        identity = digest({"plan": plan, "contract": contract, "schedule": cells})
        if ledger.exists():
            previous = json.loads(ledger.read_text())
            if not resume or previous["identity"] != identity:
                raise ValueError(
                    "resume refused: authored study or deployment identity changed"
                )
        else:
            if resume and any(
                path.name != ".controller.lock" for path in output.iterdir()
            ):
                raise ValueError("resume requires an existing study receipt")
            previous = None
        state: dict[str, Any] = {
            "schema_version": 1,
            "identity": identity,
            "plan": dict(plan),
            "contract": dict(contract),
            "schedule": cells,
            "status": "attesting",
            "blocks": [],
        }
        identities = {}
        if previous is None:
            state["arms"] = {}
            write_record(ledger, state)
        try:
            for arm in plan["arms"]:
                with open_session(arm) as session:
                    identities[arm] = dict(session.identity)
                old = (previous or {}).get("arms", {}).get(arm)
                if old is not None and identities[arm] != old:
                    raise ValueError(
                        f"resume refused: observed {arm} execution identity changed"
                    )
                if previous is None:
                    state["arms"] = dict(identities)
                    write_record(ledger, state)
        except BaseException as exc:
            # Do not replace a previous valid ledger with partial re-attestation.
            if previous is None:
                state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                write_record(ledger, state)
            raise
        state["arms"] = identities
        shared = [
            identity.get("comparison_identity") for identity in identities.values()
        ]
        if any(value is not None for value in shared) and len(set(shared)) != 1:
            raise ValueError(
                "before/after resolved scientific inputs or checkpoint bytes differ"
            )
        state["status"] = "running"
        write_record(ledger, state)
        attempts = output / "attempts"
        published = output / "blocks"
        attempts.mkdir(exist_ok=True)
        published.mkdir(exist_ok=True)
        try:
            for cell in cells:
                final = published / str(cell["index"])
                if final.exists():
                    block = json.loads((final / "block.json").read_text())
                    if block["cell"] != cell or block["identity"] != identity:
                        raise ValueError(
                            "published block belongs to a different schedule"
                        )
                    _verify_block(final, block)
                else:
                    attempt = attempts / f"{cell['index']}-{uuid.uuid4().hex}"
                    attempt.mkdir()
                    for arm in cell["arms"]:
                        with open_session(arm) as session:
                            if dict(session.identity) != identities[arm]:
                                raise ValueError(
                                    f"{arm} execution identity changed during collection"
                                )
                            receipt = session.sample(
                                cell["case"],
                                cell["seed"],
                                cell["repeat"],
                                attempt / arm,
                            )
                            if (
                                receipt.resolve()
                                != (attempt / arm / "measurement.json").resolve()
                            ):
                                raise ValueError(
                                    "worker published outside its assigned cell"
                                )
                            # Verify observations and traces before publishing the block.
                            _load_receipt(receipt, plan)
                    if finish_block is not None:
                        finish_block(cell, attempt, identities)
                        for arm in cell["arms"]:
                            _load_receipt(attempt / arm / "measurement.json", plan)
                    block = {
                        "cell": cell,
                        "identity": identity,
                        "files": manifest(attempt),
                        "directory": final.relative_to(output).as_posix(),
                    }
                    write_record(attempt / "block.json", block)
                    attempt.rename(final)
                state["blocks"].append(block)
                write_record(ledger, state)
            result = _aggregate(output, plan, state["blocks"])
            if plan["profile"]["cases"]:
                _collect_profiles(
                    plan,
                    output,
                    open_session,
                    identities,
                    identity,
                    state,
                    ledger,
                    result,
                )
            state.update(
                status="completed",
                collections={
                    case: {arm: str(p.relative_to(output)) for arm, p in arms.items()}
                    for case, arms in result.items()
                },
            )
            write_record(ledger, state)
            return result
        except BaseException as exc:
            state.update(
                status="interrupted"
                if isinstance(exc, (KeyboardInterrupt, SystemExit))
                else "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            write_record(ledger, state)
            raise
