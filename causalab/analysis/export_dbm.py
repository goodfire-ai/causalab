"""Export evaluated DBM masks, metrics, and provenance as portable JSON data.

The manifest lists experiments with ``id``, ``title``, and ``evaluations``.
Each evaluation names ``document``, ``run_dir``, ``data_root``, and
``artifacts_root``. Paths resolve from the manifest's directory.

Only apply documents are accepted. The document is compiled here and its
steps signed, so each point's identity comes from the document and the
artifacts it names, whose canonical form carries each frozen fit's content
digest; the run directory holds only the saved tables. Metric rows are placed
on the signed steps by their coordinate columns. Masks use the same
``theta > 0`` readout as the generator's sigmoid gates. Missing evaluations
are recorded in ``omitted_points``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from causalab.protocol.bundles import entry_selection, select_entry
from causalab.protocol.results import example_labels
from causalab.io.sources import load_text
from causalab.neural.shared.sweep import signed_steps
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.registry import get_model_info
from causalab.protocol.schema import inline_train_saves
from causalab.io.env import (
    FileArtifacts,
    FileDatasets,
    ResolutionEnv,
    entry_table,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def plain(value: Any) -> Any:
    """Serialize a coordinate the way the engine writes it on a metric row."""
    if isinstance(value, (int, float, str, bool)):
        return value
    return json.dumps(value, sort_keys=True)


def coordinate_key(coords: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """The join key between a signed step and its metric rows: the sorted
    (full axis id, serialized value) pairs of the step's coordinates."""
    return tuple(sorted((axis, plain(value)) for axis, value in coords.items()))


def encoded_mask(mask: Any) -> dict[str, Any]:
    """Choose a lossless range or bitset representation of a flat hard mask."""
    import numpy as np

    values = np.asarray(mask, dtype=bool).reshape(-1)
    boundaries = np.flatnonzero(np.diff(np.pad(values.astype(np.int8), (1, 1))))
    ranges = boundaries.reshape(-1, 2).tolist()
    ranged = {"encoding": "ranges", "ranges": ranges}
    packed = {
        "encoding": "bitset",
        "data": base64.b64encode(
            np.packbits(values, bitorder="little").tobytes()
        ).decode(),
        "unit_count": int(values.size),
    }
    return min((ranged, packed), key=lambda value: len(json.dumps(value)))


def metric_summary(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Average eligible per-example records from one measured point."""
    values: list[float] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("metric") != name:
            raise ValueError(f"Metric table for {name} contains another metric")
        identity = str(row["example_id"])
        if identity in seen:
            raise ValueError(f"Metric {name} repeats example {identity}")
        seen.add(identity)
        if row.get("eligible", True) is False:
            continue
        value = row.get("value")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Metric {name} has an invalid eligible value")
        if name == "iia" and value not in (0, 1):
            raise ValueError("IIA records must be binary match outcomes")
        values.append(float(value))
    result: dict[str, Any] = {"value": None, "n": len(values), "total": len(rows)}
    if not values:
        result["reason"] = "No eligible saved evaluations"
        return result
    mean = sum(values) / len(values)
    result["value"] = mean
    if len(values) > 1:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        result["standard_error"] = math.sqrt(variance / len(values))
    return result


def model_manifest(model: dict[str, Any]) -> dict[str, Any]:
    info = get_model_info(model["key"])
    layers = []
    for layer in range(info.num_layers):
        delta = (
            info.layer_types is not None
            and info.layer_types[layer] == "linear_attention"
        )
        layers.append(
            {
                "index": layer,
                "type": "gated_delta_net" if delta else "normal_attention",
                "heads": info.linear_num_value_heads if delta else info.num_heads,
                "head_dim": info.linear_value_head_dim if delta else info.head_dim,
                "experts": info.num_experts or 0,
                "expert_dim": info.moe_intermediate_size or 0,
                "shared_expert_dim": info.shared_expert_intermediate_size or 0,
                "mlp_dim": info.intermediate_size or 0,
            }
        )
    return {
        **model,
        "layers": layers,
    }


def position(value: Any, method: dict[str, Any]) -> int | str:
    """Resolve a scalar token index or the shared all-token position."""
    from causalab.analysis.hypothesis_artifacts import resolve_position

    value = resolve_position(value, method)
    if isinstance(value, dict):
        if set(value) == {"index"}:
            value = value["index"]
        elif value == {"all": True}:
            value = "all"
    if type(value) is int or value == "all":
        return value
    raise ValueError("DBM export requires a scalar token index or all positions")


def resolved_read(read: dict[str, Any], method: dict[str, Any]) -> dict[str, Any]:
    return {
        **read,
        "site": method["sites"][read["site"]],
        "pos": position(read["pos"], method),
    }


def models_of(method: dict[str, Any], read: str) -> list[str]:
    """The models that list ``read`` (§2.9), in declaration order."""
    return [
        name
        for name, entry in method.get("intervened_models", {}).items()
        if read in entry.get("reads", [])
    ]


def _unwritten_on(method: dict[str, Any], model: str, role: str) -> bool:
    entry = method["intervened_models"].get(model)
    return entry is not None and entry.get("input") == role and not entry.get("writes")


def saved_aggregations(method: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The save entries that carry an aggregation, by their file stem — the
    label their table goes by (§2.12). A ``train`` entry counts as the term
    it names."""
    out: dict[str, dict[str, Any]] = {}
    for entry in inline_train_saves(method):
        if "aggregation" in entry:
            stem = str(entry["file_path"]).rsplit("/", 1)[-1].rsplit(".", 1)[0]
            out[stem] = entry
    return out


def report_entries(method: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The two save entries a DBM export reports, by report key: the one
    ``match`` (``iia``) and the one ``logit_diff`` (``logit_diff``)."""
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for entry in saved_aggregations(method).values():
        by_kind.setdefault(str(entry["aggregation"].get("kind")), []).append(entry)
    matches, diffs = by_kind.get("match", []), by_kind.get("logit_diff", [])
    if len(matches) != 1 or len(diffs) != 1:
        raise ValueError(
            "DBM export requires distinct match IIA and logit difference metrics"
        )
    return {"iia": matches[0], "logit_diff": diffs[0]}


def gate_manifest(
    name: str, method: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    writes = [
        write for write in method["writes"].values() if write.get("featurizer") == name
    ]
    if len(writes) != 1:
        raise ValueError(f"Gate {name} must belong to one position and site")
    write = writes[0]
    if set(write) != {"site", "pos", "featurizer", "do"}:
        raise ValueError(f"Gate {name} requires a scalar or all-position swap")
    site = method["sites"][write["site"]]
    target_position = position(write["pos"], method)
    operation = write.get("do", {})
    if set(operation) != {"swap"} or not isinstance(operation["swap"], (str, dict)):
        raise ValueError(f"Gate {name} requires a direct counterfactual swap")
    operand = operation["swap"]
    read_name = operand["read"] if isinstance(operand, dict) else operand
    source = resolved_read(method["reads"][read_name], method)
    if source != {"site": site, "pos": target_position, "featurizer": name}:
        raise ValueError(f"Gate {name} requires an aligned counterfactual swap")
    source_models = models_of(method, read_name)
    if len(source_models) != 1 or not _unwritten_on(
        method, source_models[0], "counterfactual"
    ):
        raise ValueError(
            f"Gate {name} requires its operand read on the un-intervened "
            "counterfactual model"
        )
    if len(site["layers"]) != 1 or "head" in site or "expert" in site:
        raise ValueError(f"Gate {name} requires one complete layer site")
    layer = site["layers"][0]
    architecture = model["layers"][layer]
    component = site["component"]
    gate = method["featurizers"][name]
    family = "mlp"
    if component in ("attention_premix", "delta_premix"):
        family = (
            "gated_delta_net" if component == "delta_premix" else "normal_attention"
        )
        if gate.get("group") == "head":
            kind, shape = "attention_head", [architecture["heads"]]
        else:
            kind, shape = (
                "attention_channel",
                [architecture["heads"], architecture["head_dim"]],
            )
    elif component == "expert_neuron_output" and gate.get("group") == "expert_neuron":
        kind, shape = (
            "routed_expert_neuron",
            [architecture["experts"], architecture["expert_dim"]],
        )
    elif component == "shared_expert_activation":
        kind, shape = "shared_expert_neuron", [architecture["shared_expert_dim"]]
    elif component == "mlp_neuron_output":
        kind, shape = "mlp_neuron", [architecture["mlp_dim"]]
    else:
        raise ValueError(f"Unsupported DBM export site {component!r}")
    if any(not isinstance(size, int) or size <= 0 for size in shape):
        raise ValueError(f"Unknown shape for {name}")
    return {
        "id": name,
        "layer": layer,
        "site_component": component,
        "component": kind,
        "family": family,
        "position": target_position,
        "shape": shape,
        "unit_count": math.prod(shape),
    }


def dbm_aggregations(doc: Any) -> dict[str, Any]:
    """The two saved aggregations a DBM export reports, by report key: the
    one ``match`` (``iia``) and the one ``logit_diff`` (``logit_diff``) over
    the same bound read, agreeing on the gold label (``match.expected ==
    logit_diff.a``). Located by kind through the document's accessors, so a
    document may label them as it likes; refused naming what was found."""
    saved = doc.saved_aggregations()
    by_kind: dict[str, list[Any]] = {}
    for agg in saved:
        by_kind.setdefault(str(agg.spec.kind), []).append(agg)
    matches, diffs = by_kind.get("match", []), by_kind.get("logit_diff", [])
    if len(matches) != 1 or len(diffs) != 1:
        raise ValueError(
            "DBM export requires exactly one saved match (IIA) and one saved "
            f"logit_diff aggregation; found {len(matches)} match and "
            f"{len(diffs)} logit_diff among {[a.label for a in saved]}"
        )
    (iia,), (logit_diff,) = matches, diffs
    if iia.read != logit_diff.read:
        raise ValueError(
            "IIA and logit difference must reduce the same read on the same model: "
            f"{iia.read} vs {logit_diff.read}"
        )
    if iia.spec.fields.get("expected") != logit_diff.spec.fields.get("a"):
        raise ValueError("IIA and logit difference must target the same gold label")
    return {"iia": iia, "logit_diff": logit_diff}


def _save_index(owner: str) -> int:
    assert owner.startswith("save[") and owner.endswith("]"), owner
    return int(owner[len("save[") : -1])


def validate_measurement(method: dict[str, Any], gates: list[dict[str, Any]]) -> None:
    """Require both scores to measure the model with every exported gate active."""
    entries = report_entries(method)
    bound = [(entries[k]["read"], entries[k]["model"]) for k in ("iia", "logit_diff")]
    resolved = [resolved_read(method["reads"][read], method) for read, _ in bound]
    if resolved[0] != resolved[1] or bound[0][1] != bound[1][1]:
        raise ValueError("IIA and logit difference must use the same resolved read")
    measured = method["intervened_models"].get(bound[0][1])
    if (
        measured is None
        or measured["input"] != "base"
        or not measured.get("writes")
        or bound[0][0] not in measured.get("reads", [])
    ):
        raise ValueError("DBM metrics must read the intervened model on its input")
    if not gates:
        raise ValueError("A DBM export requires at least one gate")
    exported = {gate["id"] for gate in gates}
    required = {
        name
        for name, write in method["writes"].items()
        if isinstance(write.get("featurizer"), str) and write["featurizer"] in exported
    }
    if not required.issubset(set(measured["writes"])):
        raise ValueError("Every exported gate must be active in the measured model")
    if set(measured["writes"]) != required:
        raise ValueError(
            "The measured model must contain only the exported gate writes"
        )


def frozen_mask(
    spec: dict[str, Any], coords: dict[str, Any], root: Path
) -> tuple[Any, dict[str, Any]]:
    """Read exactly the fitted tensor named by a recorded sigmoid apply."""
    import numpy as np
    from safetensors import safe_open

    if (
        spec.get("parametrization", "sigmoid") != "sigmoid"
        or "top_k" in spec
        or "pool" in spec
    ):
        raise ValueError("This exporter requires threshold replay of sigmoid DBM gates")
    path = (root / spec["file_path"]).resolve()
    with safe_open(path, framework="numpy") as bundle:
        selection, implicit = entry_selection(spec.get("entry"), coords, "gate")
        key = select_entry(
            bundle.keys(),
            "theta",
            selection,
            what=str(path),
            coords_by_key=entry_table(bundle.metadata()),
            implicit=implicit,
        )
        theta = bundle.get_tensor(key)
        if not np.isfinite(theta).all():
            raise ValueError(f"Non-finite mask parameter in {path}:{key}")
        mask = theta.reshape(-1) > 0
    # the entry's content digest as the compile recorded it (the document's
    # ``content_digest``): what identifies the fitted tensor independently of
    # where the run tree sits (``hypothesis_artifacts.frozen_dbm_identity``)
    return mask, {
        "file_path": str(path),
        "entry": key,
        **({"sha256": spec["content_digest"]} if "content_digest" in spec else {}),
    }


def evaluation(
    item: dict[str, Any],
    base: Path,
    *,
    register_from_hf: bool = False,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    document = (base / item["document"]).resolve()
    run = (base / item["run_dir"]).resolve()
    artifact_root = (base / item["artifacts_root"]).resolve()
    source = load_text(document)
    if register_from_hf:
        from causalab.cli import register_model_key

        register_model_key(source)
    if "train" in source["method"]:
        raise ValueError(f"Use a held-out apply document: {document}")
    datasets = FileDatasets(root=(base / item["data_root"]).resolve())
    env = ResolutionEnv(
        datasets=datasets,
        artifacts=FileArtifacts(root=artifact_root),
    )
    compiled = compile_protocol(source, env=env)
    # the steps as the engine enumerated and signed them: coordinates,
    # canonical forms and digests, in enumeration order — the same signing
    # the run performed, so the run directory needs no receipt
    steps = signed_steps(compiled, env)
    first = steps[0].canonical
    if any(
        step.canonical["model"] != first["model"]
        or step.canonical["data"] != first["data"]
        for step in steps
    ):
        raise ValueError("Model or data changes within one DBM evaluation")
    model = model_manifest(first["model"])
    method = first["method"]
    entries = report_entries(method)
    definitions = {key: dict(entry["aggregation"]) for key, entry in entries.items()}
    if definitions["iia"]["expected"] != definitions["logit_diff"]["a"]:
        raise ValueError("IIA and logit difference must target the same gold label")
    pair_rows = datasets.rows(first["data"]["base"]["dataset"])
    pair_ids = example_labels(pair_rows)
    changed = {
        identity
        for identity, row in zip(pair_ids, pair_rows)
        if row[definitions["logit_diff"]["a"]] != row[definitions["logit_diff"]["b"]]
    }
    gates = [
        gate_manifest(name, method, model)
        for name, spec in method["featurizers"].items()
        if spec["kind"] == "gate"
    ]
    tables: dict[str, list[dict[str, Any]]] = {}
    for metric, agg in dbm_aggregations(compiled.document).items():
        path = run / compiled.document.save[_save_index(agg.owner)].file_path
        tables[metric] = read_json(path) if path.exists() else []
    routing_path = run / "routing_mismatch.json"
    routing = read_json(routing_path) if routing_path.exists() else []
    points, omitted = [], []
    # a metric row belongs to the signed step whose coordinates it carries:
    # every axis of the sweep is a column on every row, keyed by its full id
    coords_of = [dict(step.coords) for step in steps]
    keys = [coordinate_key(coords) for coords in coords_of]
    recorded = set(keys)
    if len(recorded) != len(keys):
        raise ValueError(f"Signed steps repeat coordinates: {document}")
    axes = sorted({axis for coords in coords_of for axis in coords})

    def key_of(row: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
        return tuple((axis, row.get(axis)) for axis in axes)

    if any(key_of(row) not in recorded for rows in tables.values() for row in rows):
        raise ValueError(f"Unrecorded point in evaluation tables: {run}")
    for index, (step, coords, key) in enumerate(zip(steps, coords_of, keys)):
        digest = step.digest
        scores = {}
        for name, rows in tables.items():
            saved = [row for row in rows if key_of(row) == key]
            if saved and {str(row["example_id"]) for row in saved} != set(pair_ids):
                raise ValueError(
                    f"Incomplete or unknown example records: {run}:{index}:{name}"
                )
            if name == "logit_diff":
                saved = [row for row in saved if str(row["example_id"]) in changed]
            scores[name] = metric_summary(saved, name)
        if not any(score["n"] for score in scores.values()):
            omitted.append({"id": digest, "reason": "No eligible saved evaluations"})
            continue
        concrete = step.canonical["method"]
        if {
            key: dict(entry["aggregation"])
            for key, entry in report_entries(concrete).items()
        } != definitions:
            raise ValueError("Metric definitions change within one experiment")
        validate_measurement(concrete, gates)
        masks, fits, selected = {}, {}, 0
        for gate in gates:
            current = gate_manifest(gate["id"], concrete, model)
            if current != gate:
                raise ValueError("Gate layout changes within one experiment")
            mask, fit = frozen_mask(
                concrete["featurizers"][gate["id"]], coords, artifact_root
            )
            if len(mask) != gate["unit_count"]:
                raise ValueError(f"Mask shape mismatch: {gate['id']}")
            masks[gate["id"]] = encoded_mask(mask)
            fits[gate["id"]] = fit
            selected += int(mask.sum())
        eligible = sum(gate["unit_count"] for gate in gates)
        matching = [row for row in routing if row["point"] == digest]
        coverage: dict[str, Any] | None = None
        if matching:
            coverage = {
                "mismatched": sum(row["mismatched"] for row in matching),
                "slots": sum(row["slots"] for row in matching),
                "by_gate": {},
            }
            for name, write in concrete["writes"].items():
                rows = [row for row in matching if row["write"] == name]
                if rows:
                    slots = sum(row["slots"] for row in rows)
                    coverage["by_gate"][write["featurizer"]] = {
                        "slots": slots,
                        "matched": slots - sum(row["mismatched"] for row in rows),
                    }
        points.append(
            {
                "id": digest,
                # the fitted entries alone, independent of the run tree's path
                "fit_id": hashlib.sha256(
                    json.dumps(
                        {
                            name: {"sha256": fit.get("sha256"), "entry": fit["entry"]}
                            for name, fit in fits.items()
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
                "coords": coords,
                "selected_count": selected,
                "eligible_count": eligible,
                "sparsity": 1 - selected / eligible,
                "metrics": scores,
                "masks": masks,
                "routing": coverage,
                "provenance": {
                    "document": str(document),
                    "run_dir": str(run),
                    "dataset": first["data"]["base"],
                    "fits": fits,
                    "metric_definitions": definitions,
                    "comparison": {
                        "data": step.canonical["data"],
                        "metrics": definitions,
                        "readouts": {
                            name: {
                                **resolved_read(
                                    concrete["reads"][entry["read"]], concrete
                                ),
                                "model": entry["model"],
                            }
                            for name, entry in report_entries(concrete).items()
                        },
                    },
                    "logit_difference_population": "answer-changing pairs",
                },
            }
        )
    from causalab.analysis.hypothesis_artifacts import frozen_dbm_identity

    for point in points:
        point["artifact_id"] = frozen_dbm_identity(gates, point)
    return model, gates, points, omitted


def export(manifest_path: Path, *, register_from_hf: bool = False) -> dict[str, Any]:
    """Validate frozen DBM applies and export their measured masks and scores."""
    manifest = read_json(manifest_path)
    result: dict[str, Any] = {
        "schema_version": 1,
        "synthetic": False,
        "experiments": [],
    }
    ids: set[str] = set()
    for experiment in manifest["experiments"]:
        identifier = experiment["id"]
        if identifier in ids:
            raise ValueError(f"Repeated experiment ID: {identifier}")
        ids.add(identifier)
        combined = {
            "id": identifier,
            "title": experiment["title"],
            "points": [],
            "omitted_points": [],
        }
        seen: set[str] = set()
        comparison = None
        for item in experiment["evaluations"]:
            model, gates, points, omitted = evaluation(
                item, manifest_path.parent, register_from_hf=register_from_hf
            )
            if result.setdefault("model", model) != model:
                raise ValueError("Experiments use different model realizations")
            if combined.setdefault("gates", gates) != gates:
                raise ValueError(
                    "Evaluation documents use different component universes"
                )
            for point in points:
                contract = point["provenance"]["comparison"]
                if comparison is not None and comparison != contract:
                    raise ValueError(
                        "Evaluation data, metrics, or readout differ within one curve"
                    )
                comparison = contract
                if point["id"] in seen:
                    raise ValueError(f"Repeated evaluated point: {point['id']}")
                seen.add(point["id"])
                combined["points"].append(point)
            combined["omitted_points"].extend(omitted)
        if not combined["points"]:
            raise ValueError(f"Experiment {identifier} has no saved, evaluated masks")
        combined["unit"] = (
            "heads"
            if all(gate["component"] == "attention_head" for gate in combined["gates"])
            else "neurons"
        )
        combined["position_mode"] = (
            "independent"
            if len({str(gate["position"]) for gate in combined["gates"]}) > 1
            else "shared"
        )
        result["experiments"].append(combined)
    if not result["experiments"]:
        raise ValueError("Manifest has no experiments")
    return result
