"""Receipt and artifact validation shared by reports and execution."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

from ..collection import file_hash


@dataclass
class ReceiptValidationError(ValueError):
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"measurement receipt {self.path}: {self.reason}"


def verify_output_files(root: Path, value: Any, *, receipt: Path | None = None) -> None:
    """Validate a saved workflow-output manifest and verify its file hashes."""
    location = root / "measurement.json" if receipt is None else receipt
    if not isinstance(value, dict):
        raise ReceiptValidationError(
            location, "output_files must be a file-to-sha256 mapping"
        )
    for name, sha in value.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha) is None
        ):
            raise ReceiptValidationError(
                location, "output_files needs nonempty file names and SHA-256 hashes"
            )
        verified_artifact(root, {"file": name, "sha256": sha})


def verified_artifact(root: Path, ref: Any) -> Path:
    if not isinstance(ref, dict) or set(ref) != {"file", "sha256"}:
        raise ValueError("artifact needs exactly file and sha256")
    if not isinstance(ref["file"], str) or Path(ref["file"]).is_absolute():
        raise ValueError("artifact must be relative to its measurement receipt")
    path = (root / ref["file"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing or escaping artifact: {ref['file']}")
    if file_hash(path) != ref["sha256"]:
        raise ValueError(f"artifact hash mismatch: {ref['file']}")
    return path


def load_measurement(
    path: Path,
    *,
    require_observations: bool = True,
) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
    import torch
    from safetensors.torch import load_file

    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise ReceiptValidationError(path, "expected a receipt object")
    policy = record.get("observation_policy", "required")
    if policy not in {"required", "not_requested"}:
        raise ValueError("invalid observation policy")
    numerical = policy == "required"
    if not numerical and (require_observations or record.get("mode") != "single"):
        raise ValueError("numerical comparison requires observations")
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or record.get("status") != "completed"
    ):
        raise ValueError("comparison requires completed schema_version=1 measurements")
    for field in ("case", "input_identity", "scope", "reset_policy", "alignment"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError(f"measurement is missing {field}")
    if not isinstance(record.get("provenance"), dict) or not record["provenance"].get(
        "implementation"
    ):
        raise ValueError("measurement needs implementation provenance")
    plan = record.get("plan", {})
    if not isinstance(plan, dict):
        raise ReceiptValidationError(path, "plan must be an object")
    seeds, repeats = plan.get("seeds"), plan.get("repeats")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(s) is not int or s < 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
        or type(repeats) is not int
        or repeats < 1
    ):
        raise ValueError("invalid seed/repetition plan")
    samples: dict[tuple[int, int], dict[str, Any]] = {}
    if not isinstance(record.get("samples", []), list):
        raise ReceiptValidationError(path, "samples must be a list")
    for sample in record.get("samples", []):
        if not isinstance(sample, dict):
            raise ReceiptValidationError(path, "sample must be an object")
        if type(sample.get("seed")) is not int or type(sample.get("repeat")) is not int:
            raise ValueError("sample identity must use integer seed and repeat")
        key = (sample["seed"], sample["repeat"])
        if key in samples:
            raise ValueError(f"duplicate sample identity {key}")
        elapsed = sample.get("seconds")
        if (
            type(elapsed) not in (int, float)
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            raise ValueError("sample seconds must be finite and positive")
        verify_output_files(path.parent, sample.get("output_files", {}), receipt=path)
        if "resident_output_files" in sample:
            verify_output_files(
                path.parent, sample["resident_output_files"], receipt=path
            )
        if not numerical:
            if sample.get("observation_status") != "not_requested" or any(
                "observation" in name and name != "observation_status"
                for name in sample
            ):
                raise ValueError(
                    "timing-only sample has inconsistent observation evidence"
                )
            samples[key] = {**sample, "seconds": elapsed}
            continue
        if "observations" not in sample:
            raise ValueError("requested observations are missing")
        source = verified_artifact(path.parent, sample["observations"])
        if "resident_observations" in sample:
            verified_artifact(path.parent, sample["resident_observations"])
        if "native_observations" in sample:
            verified_artifact(path.parent, sample["native_observations"])
        for name in (
            "unobserved_observations",
            "resident_unobserved_observations",
            "training_observed_observations",
            "diagnostic_observations",
            "resident_diagnostic_observations",
        ):
            if name in sample:
                verified_artifact(path.parent, sample[name])
        tensors = load_file(str(source))
        if not tensors:
            raise ValueError("empty observation bundle")
        for name, tensor in tensors.items():
            if not tensor.numel():
                raise ReceiptValidationError(path, f"empty observation tensor: {name}")
            if tensor.is_complex() or not torch.isfinite(tensor).all():
                raise ReceiptValidationError(
                    path, f"nonfinite or complex observation: {name}"
                )
        samples[key] = {**sample, "seconds": elapsed, "tensors": tensors}
    expected = {(s, r) for s in seeds for r in range(repeats)}
    if set(samples) != expected:
        raise ValueError("missing or unexpected seed/repeat observations")
    trace = record.get("trace", {"status": "not_requested"})
    if not isinstance(trace, dict) or trace.get("status") not in {
        "not_requested",
        "completed",
        "failed",
        "unavailable",
        "interrupted",
    }:
        raise ReceiptValidationError(path, "invalid trace status")
    if trace["status"] == "completed":
        if (
            type(trace.get("seed")) is not int
            or type(trace.get("repeat")) is not int
            or (trace["seed"], trace["repeat"]) not in samples
        ):
            raise ReceiptValidationError(
                path, "trace has no corresponding clean sample"
            )
        if not {"file", "sha256"} <= trace.keys():
            raise ReceiptValidationError(path, "completed trace needs file and sha256")
        artifact = verified_artifact(
            path.parent, {key: trace[key] for key in ("file", "sha256")}
        )
        if not artifact.stat().st_size:
            raise ReceiptValidationError(path, "empty trace artifact")
        if numerical:
            if "observations" not in trace:
                raise ReceiptValidationError(path, "completed trace needs observations")
            verified_artifact(path.parent, trace["observations"])
        elif trace.get("observation_status") != "not_requested" or any(
            "observation" in name and name != "observation_status" for name in trace
        ):
            raise ReceiptValidationError(
                path, "timing-only trace has inconsistent observation evidence"
            )
    from causalab.measurement.analysis.profiles import validate_captures

    validate_captures(record, path.parent, samples)
    return record, samples
