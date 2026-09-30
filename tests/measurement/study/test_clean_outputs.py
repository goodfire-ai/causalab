"""A differing diagnostic fit must never replace the timed fit or its temperature."""

from contextlib import contextmanager
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from causalab.measurement.collection import file_hash, write_record
from causalab.measurement.study.evaluation import finish_evaluations
from causalab.measurement.runtime.observations import observations
from causalab.measurement.runtime.worker import bind_timing_outputs

pytestmark = pytest.mark.unit


def test_clean_fit_is_primary_and_common_evaluators_receive_it(tmp_path):
    specs = {"gate": {"step": "fit", "file": "gate.safetensors", "kind": "gate"}}
    identities = {"before": {"arm": "before"}, "after": {"arm": "after"}}
    for arm in identities:
        directory = tmp_path / arm
        for label, value, temperature in (("timing", 1.0, 0.1), ("numerics", 9.0, 0.9)):
            root = directory / label / "research" / "fit"
            root.mkdir(parents=True)
            save_file(
                {"theta": torch.tensor([value])},
                str(root / "gate.safetensors"),
                metadata={
                    "entries": json.dumps(
                        {
                            "theta": {
                                "slot": "theta",
                                "coords": {},
                                "produced_by": "point",
                            }
                        }
                    )
                },
            )
            (root / "fit_diagnostics.json").write_text(
                json.dumps(
                    [
                        {
                            "point": "point",
                            "coords": {},
                            "featurizers": {"gate": {"temperature": temperature}},
                        }
                    ]
                )
            )
        diagnostic = directory / "diagnostic.safetensors"
        save_file(observations(directory / "numerics/research", specs), str(diagnostic))
        sample = {
            "timing_directory": "timing",
            "numerics_directory": "numerics",
            "observations": {"file": diagnostic.name, "sha256": file_hash(diagnostic)},
        }
        bind_timing_outputs(sample, directory, "research", specs)
        key = next(iter(load_file(str(directory / sample["observations"]["file"]))))
        assert (
            load_file(str(directory / sample["observations"]["file"]))[key].item() == 1
        )
        assert sample["observation_specs"][key]["temperature"] == 0.1
        assert sample["diagnostic_observation_specs"][key]["temperature"] == 0.9
        assert sample["observer_check"]["exactly_equal"] is False
        assert sample["observer_check"]["observation_specs_match"] is False
        write_record(
            directory / "measurement.json",
            {
                "samples": [sample],
                "trace": {"status": "not_requested"},
                "context": {},
            },
        )

    calls = []

    class Evaluator:
        def __init__(self, arm):
            self.identity = identities[arm]

        def evaluate(self, steps, seed, fit_root, destination):
            assert fit_root.parts[-2:] == ("timing", "research")
            assert (
                load_file(str(fit_root / "fit/gate.safetensors"))["theta"].item() == 1
            )
            calls.append((self.identity["arm"], fit_root.parent.parent.name))
            destination.mkdir(parents=True)
            file = destination / "eval.safetensors"
            save_file({"score": torch.tensor([2.0])}, str(file))
            return {
                "identity": self.identity,
                "observations": {"file": str(file), "sha256": file_hash(file)},
                "observation_specs": {"score": {"kind": "tensor"}},
            }

    @contextmanager
    def open_session(arm):
        yield Evaluator(arm)

    finish_evaluations(
        {
            "arms": identities,
            "evaluation": {
                "arm": "before",
                "crossed": True,
                "cases": {"fit": ["eval"]},
            },
        },
        open_session,
        {"case": "fit", "seed": 7},
        tmp_path,
        identities,
    )
    assert set(calls) == {
        (evaluator, fit) for evaluator in identities for fit in identities
    }
    for arm in identities:
        sample = json.loads((tmp_path / arm / "measurement.json").read_text())[
            "samples"
        ][0]
        assert sample["evaluated_fit_origin"] == "timing_pass"
        assert sample["evaluated_fit_tree_sha256"]
        values = load_file(str(tmp_path / arm / sample["observations"]["file"]))
        assert values[key].item() == 1
        assert values["evaluation__before__score"].item() == 2
        assert values["evaluation__after__score"].item() == 2
