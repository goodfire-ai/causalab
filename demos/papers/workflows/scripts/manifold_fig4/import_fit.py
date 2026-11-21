"""Copy the declared fitting artifacts into the steering workflow's run tree."""

import json
from pathlib import Path
from shutil import copyfile
from typing import Any, Mapping

from causalab.io.step_io import StepError


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    contract = json.loads(inputs["contract"].read_text(encoding="utf-8"))
    expected = inputs["expected"]
    differences = sorted(
        key
        for key in contract.keys() | expected.keys()
        if contract.get(key) != expected.get(key)
    )
    if differences:
        raise StepError(
            f"import_fit: fitting artifacts disagree on {', '.join(differences)}"
        )
    for slot, destination in outputs.items():
        copyfile(inputs[slot], destination)
