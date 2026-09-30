"""Common and crossed evaluation of saved fits, before paired-block publication.

Authored protocol steps specify evaluation data, observables and saved fits.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..collection import file_hash, write_record
from causalab.measurement.study.scheduler import digest, manifest


def finish_evaluations(plan, open_session, cell, attempt: Path, identities) -> None:
    from safetensors.torch import load_file, save_file

    evaluation = plan["evaluation"]
    if evaluation is None or cell["case"] not in evaluation["cases"]:
        return
    steps = evaluation["cases"][cell["case"]]
    evaluators = list(plan["arms"]) if evaluation["crossed"] else [evaluation["arm"]]
    records, fits, hashes, values = {}, {}, {}, {}
    for arm in plan["arms"]:
        path = attempt / arm / "measurement.json"
        record = json.loads(path.read_text())
        sample = record["samples"][0]
        root = (path.parent / sample["workflow_outputs"]).resolve()
        if not root.is_relative_to(path.parent.resolve()) or not root.is_dir():
            raise ValueError("common evaluation needs contained saved workflow outputs")
        records[arm], fits[arm], hashes[arm] = record, root, manifest(root)
        ref = sample["observations"]
        file = path.parent / ref["file"]
        if file_hash(file) != ref["sha256"]:
            raise ValueError("native observations changed before common evaluation")
        values[arm] = load_file(str(file))
        sample["native_observations"] = dict(ref)
        if record["trace"]["status"] == "completed":
            record["trace"]["observation_keys"] = sorted(values[arm])
    for evaluator in evaluators:
        with open_session(evaluator) as session:
            if session.identity != identities[evaluator]:
                raise ValueError("common evaluator execution identity changed")
            for fitted_arm in plan["arms"]:
                destination = attempt / "evaluations" / evaluator / fitted_arm
                result = session.evaluate(
                    steps, cell["seed"], fits[fitted_arm], destination
                )
                file = Path(result["observations"]["file"]).resolve()
                if (
                    not file.is_relative_to(destination.resolve())
                    or file_hash(file) != result["observations"]["sha256"]
                ):
                    raise ValueError(
                        "common evaluator published an invalid observation artifact"
                    )
                if result["identity"] != identities[evaluator]:
                    raise ValueError(
                        "common evaluator result has a different execution identity"
                    )
                prefix = f"evaluation__{evaluator}__"
                extra = {
                    prefix + key: tensor for key, tensor in load_file(str(file)).items()
                }
                if values[fitted_arm].keys() & extra.keys():
                    raise ValueError(
                        "common evaluation observation names collide with native observations"
                    )
                values[fitted_arm].update(extra)
                records[fitted_arm]["samples"][0].setdefault(
                    "observation_specs", {}
                ).update(
                    {
                        prefix + name: spec
                        for name, spec in result["observation_specs"].items()
                    }
                )
    for arm in plan["arms"]:
        if manifest(fits[arm]) != hashes[arm]:
            raise ValueError("evaluation modified the saved fit's input tree")
        record = records[arm]
        artifact = attempt / arm / "comparison.safetensors"
        save_file(values[arm], str(artifact))
        record["samples"][0]["observations"] = {
            "file": artifact.name,
            "sha256": file_hash(artifact),
        }
        record["samples"][0]["evaluated_fit_tree_sha256"] = digest(hashes[arm])
        record["samples"][0]["evaluated_fit_origin"] = record["samples"][0].get(
            "observation_origin", "unspecified"
        )
        record["context"]["evaluation"] = {
            "common_arm": evaluation["arm"],
            "evaluators": {arm: identities[arm] for arm in evaluators},
            "steps": steps,
            "crossed": evaluation["crossed"],
            "scope": "additional replay of authored evaluation steps against each saved fit; outside primary timing",
        }
        write_record(attempt / arm / "measurement.json", record)
