"""Extract aligned scientific observations from actual protocol save products."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


from ..analysis.stability import require_sigmoid_gate


def observation_specs(
    root: Path, specs: Mapping[str, Any], *, step: str | None = None
) -> dict[str, Any]:
    """Bind each saved gate's confidence temperature to its actual fitted point.

    A gate's final/early-stopped temperature can differ between repetitions.
    Fixed artifacts without diagnostics require an explicit declared temperature.
    Ambiguous multi-featurizer diagnostics require an authored selector.
    """
    from safetensors import safe_open

    from causalab.protocol.lowering import short_coords

    result = {}
    for name, spec in specs.items():
        if step is not None and spec["step"] != step:
            continue
        result[name] = dict(spec)
        if spec["kind"] != "gate":
            continue
        path = root / spec["step"] / spec["file"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("gate observation file escapes workflow output")
        diagnostics = root / spec["step"] / "fit_diagnostics.json"
        rows = json.loads(diagnostics.read_text()) if diagnostics.is_file() else []
        with safe_open(path, framework="pt", device="cpu") as bundle:
            header = bundle.metadata() or {}
            entries = json.loads(header.get("entries", "{}"))
        if not entries:
            raise ValueError("gate temperature needs per-entry protocol provenance")
        for metadata in entries.values():
            # Per-entry identity is authoritative for swept bundles. Unstamped
            # legacy gate artifacts mean sigmoid, as in the protocol loader.
            parametrization = metadata.get(
                "parametrization", header.get("parametrization", "sigmoid")
            )
            require_sigmoid_gate(name, parametrization)
            # A diagnostics row is placed by its coordinates, as every
            # fan-out join is: the bundle entry names no producing point.
            temperatures = [
                value["temperature"]
                for row in rows
                for feature, value in row["featurizers"].items()
                if "temperature" in value
                and ("featurizer" not in spec or feature == spec["featurizer"])
                # TensorFile shortens against the declared featurizer, not
                # the tensor slot. Diagnostics retain full sweep axis names.
                and short_coords(row["coords"], entry=feature)
                == metadata.get("coords", {})
            ]
            if rows and len(temperatures) != 1:
                raise ValueError(
                    f"gate {name} needs one matching fit temperature; "
                    "check provenance and the featurizer selector"
                )
            declared = spec.get("temperature", "fit")
            temperature = next(iter(temperatures)) if temperatures else declared
            if (
                type(temperature) not in (int, float)
                or not math.isfinite(temperature)
                or temperature <= 0
            ):
                raise ValueError(f"gate {name} has no finite positive fit temperature")
            if declared != "fit" and temperature != declared:
                raise ValueError(f"gate {name} declared temperature disagrees with fit")
            label = {"slot": metadata["slot"], "coords": metadata.get("coords", {})}
            key = (
                name
                + "/"
                + json.dumps(
                    label, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
            )
            result[key] = {
                **spec,
                "parametrization": parametrization,
                "temperature": float(temperature),
                "temperature_source": "fit_diagnostics" if temperatures else "declared",
            }
    return result


def observations(
    root: Path, specs: Mapping[str, Any], *, step: str | None = None
) -> dict[str, Any]:
    """Use saved semantic slots/coordinates; gathered ragged reads exclude padding.

    Tensor row order is the protocol's frozen logical dataset order, never a
    physical microbatch index. Table observations explicitly author their row keys.
    Changed eligibility produces different keys, preventing paired analysis.
    """
    import torch
    from safetensors import safe_open

    from causalab.protocol.bundles import RAGGED_SUFFIX

    result = {}
    for name, spec in specs.items():
        if step is not None and spec["step"] != step:
            continue
        path = root / spec["step"] / spec["file"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("observation file escapes workflow output")
        if spec["kind"] == "table":
            rows = json.loads(path.read_text())
            if not isinstance(rows, list):
                raise ValueError("table observation requires an array of rows")
            seen = set()
            for row in rows:
                coords = {key: row[key] for key in spec["row_keys"]}
                key = (
                    name
                    + "/"
                    + json.dumps(
                        coords, sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
                )
                if key in seen:
                    raise ValueError(f"duplicate logical table row: {key}")
                seen.add(key)
                if row.get("eligible", True) is False:
                    continue
                value = row[spec["value"]]
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(
                        f"nonfinite or nonnumeric table observation: {key}"
                    )
                result[key] = torch.tensor([value], dtype=torch.float64)
            continue
        with safe_open(path, framework="pt", device="cpu") as bundle:
            entries = json.loads((bundle.metadata() or {}).get("entries", "{}"))
            if not entries:
                raise ValueError(
                    f"observation needs per-entry protocol provenance: {path}"
                )
            seen_entries = set()
            for entry, metadata in entries.items():
                label = {"slot": metadata["slot"], "coords": metadata.get("coords", {})}
                key = (
                    name
                    + "/"
                    + json.dumps(
                        label, sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
                )
                if key in seen_entries:
                    raise ValueError(f"duplicate semantic tensor slot: {key}")
                seen_entries.add(key)
                tensor = bundle.get_tensor(entry)
                if entry + RAGGED_SUFFIX in bundle.keys():
                    widths = bundle.get_tensor(entry + RAGGED_SUFFIX).tolist()
                    if any(type(w) is not int or w < 0 for w in widths) or sum(
                        widths
                    ) != len(tensor):
                        raise ValueError("invalid saved ragged widths")
                    offset = 0
                    for row, width in enumerate(widths):
                        if width:
                            result[f"{key}/row_{row}"] = tensor[offset : offset + width]
                        offset += width
                else:
                    result[key] = tensor
    if not result:
        raise ValueError("selected observations contain no eligible numeric values")
    return result
