"""``causalab.analysis.certify_control`` — the certification legs of a
``self_swap`` control (workflow spec §2.2, §8).

```json
"certify": {
  "type": "script", "script": {"module": "causalab.analysis.certify_control"},
  "inputs": {"receiver_original": {"step": "control", "file": "receiver_original.safetensors"},
             "receiver_control":  {"step": "control", "file": "receiver_control.safetensors"},
             "receiver_target":   {"step": "control", "file": "receiver_target.safetensors"},
             "operand":           {"step": "control", "file": "operand.safetensors"},
             "overwritten":       {"step": "control", "file": "overwritten.safetensors"}},
  "outputs": {"controls": "controls.json"}
}
```

A bit-exact no-op is **necessary but insufficient**: a self-swap that writes
a value back at the very address it was read from leaves the receiver
untouched whether or not the intervention it controls does anything at all.
So the control certifies only when three legs hold together, per point:

* **(i) identity** — the receiver read under the self-swap model equals the
  same read under ``original`` **bit for bit** (``torch.equal``);
* **(ii) positive sender effect** — the operand the *target* model writes
  differs from the value it overwrites, ``not allclose(operand, overwritten,
  atol)``;
* **(iii) changed receiver** — the receiver under the target model differs
  from the receiver under ``original``, ``not allclose(…, atol)``.

Legs (ii) and (iii) are the anti-vacuity idiom of the engine's oracle tests
(``assert not torch.allclose(want, clean, atol=1e-4)``) promoted to a recorded
status. The fourth leg of the bar — agreement with an independent oracle — is
**test-side only** (``tests/neural/engines/pytorch_hooks/hook_oracle_lib.py``):
nothing shipped implements a write oracle, and this script does not pretend to.

The five inputs are the saved reads of one control document (IM spec §2.12):
the receiver — an ``lm_head`` read, typically — under ``original``, under the
self-swap model and under the target model; the target's operand read; and
the same address read under ``original`` on the target model's own input. All
five are bundles of the same document, so their entries pair by the
coordinates the header's ``entries`` table records; a bundle whose entries
pair with nothing is refused, not skipped. ``atol`` (default ``1e-4``) may be
authored higher by a campaign that measured its own floor; the runner hands
the control's declaration in under ``control`` (``kind``, ``seam``), which
the rows repeat so the table reads on its own.

One row per point in ``controls.json``: ``coords``, ``label``, ``kind``,
``seam``, ``status`` (``passed`` when all three legs hold, else ``failed``),
the three numbers (``identity_max_abs``, ``sender_effect``,
``receiver_change``), ``identity_exact`` and ``atol``. The runner reads the
statuses back, joins them onto every downstream step's points and holds the
failure rate to the control's ``stop_after_failure_rate`` (workflow spec §8).

Numerics are imported inside [`main`][], and the module imports only
``causalab.io.step_io`` at module level, so its import closure is exactly the
shared protocol core (``tests/workflow/test_closure_census.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_io import StepError

__all__ = ["LEGS", "REQUIRED_INPUTS", "main"]

#: The five bundles a certification reads, by input name.
REQUIRED_INPUTS: tuple[str, ...] = (
    "receiver_original",
    "receiver_control",
    "receiver_target",
    "operand",
    "overwritten",
)

#: The three legs this script decides, in the order the rows carry them.
LEGS: tuple[str, ...] = ("identity", "sender_effect", "receiver_change")

#: The default tolerance of legs (ii) and (iii): the oracle tests' non-vacuity
#: bar.
DEFAULT_ATOL = 1e-4


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    import torch
    from causalab.io.tensor_files import load_file

    from causalab.io.step_io import entry_table, write_table
    from causalab.protocol.bundles import RAGGED_SUFFIX, parse_entry_key
    from causalab.io.env import read_safetensors_metadata

    missing = [name for name in REQUIRED_INPUTS if name not in inputs]
    if missing:
        raise StepError(
            f"certify_control: inputs {missing} are required — the five saved "
            f"reads of the control document ({', '.join(REQUIRED_INPUTS)})"
        )
    atol = inputs.get("atol", DEFAULT_ATOL)
    if isinstance(atol, bool) or not isinstance(atol, (int, float)) or atol < 0:
        raise StepError(
            f"certify_control: 'atol' must be a non-negative number, got {atol!r}"
        )
    declaration = inputs.get("control")
    if declaration is not None and not isinstance(declaration, Mapping):
        raise StepError("certify_control: 'control' is the declaration object")
    kind = str((declaration or {}).get("kind", "self_swap"))
    seam = (declaration or {}).get("seam")

    bundles: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_INPUTS:
        source = inputs[name]
        if not isinstance(source, (str, Path)):
            raise StepError(
                f"certify_control: {name!r} must reference the saved bundle "
                "itself, not a tensor selected out of it — the entries pair "
                "by the header's coordinates. Drop the reference's 'entry'"
            )
        source = Path(source)
        if not source.is_file():
            raise StepError(
                f"certify_control: {name!r} bundle {str(source)!r} does not exist"
            )
        metadata = dict(read_safetensors_metadata(source) or {})
        tensors = load_file(str(source))
        entries = entry_table(metadata)
        by_coords: dict[str, Any] = {}
        for key, tensor in tensors.items():
            if str(key).endswith(RAGGED_SUFFIX):
                raise StepError(
                    f"certify_control: {name!r} holds a ragged entry {key!r} — "
                    "the legs compare dense reads; read one position, not all"
                )
            stored = entries.get(str(key))
            coords = (
                dict(stored["coords"])
                if stored is not None and "coords" in stored
                else parse_entry_key(str(key))[1]
            )
            token = _coords_token(coords)
            if token in by_coords:
                raise StepError(
                    f"certify_control: {name!r} holds two entries at coordinates "
                    f"{coords} — one read, one entry per point"
                )
            by_coords[token] = {"coords": coords, "key": str(key), "tensor": tensor}
        if not by_coords:
            raise StepError(f"certify_control: {name!r} bundle holds no tensors")
        bundles[name] = by_coords

    tokens = sorted(bundles["receiver_original"])
    for name, held in bundles.items():
        if sorted(held) != tokens:
            raise StepError(
                f"certify_control: {name!r} pairs with nothing — its entries sit at "
                f"{[e['coords'] for e in held.values()]}, the receiver's at "
                f"{[bundles['receiver_original'][t]['coords'] for t in tokens]}; "
                "all five reads are saved by one document, so they share its points"
            )

    rows: list[dict[str, Any]] = []
    for token in tokens:
        entry = {name: bundles[name][token] for name in REQUIRED_INPUTS}
        tensors = {name: e["tensor"] for name, e in entry.items()}
        _same_shape(
            tensors, ("receiver_original", "receiver_control", "receiver_target")
        )
        _same_shape(tensors, ("operand", "overwritten"))
        identity = (tensors["receiver_control"] - tensors["receiver_original"]).abs()
        sender = (tensors["operand"] - tensors["overwritten"]).abs()
        receiver = (tensors["receiver_target"] - tensors["receiver_original"]).abs()
        exact = bool(
            torch.equal(tensors["receiver_control"], tensors["receiver_original"])
        )
        positive = not torch.allclose(
            tensors["operand"], tensors["overwritten"], atol=float(atol), rtol=0.0
        )
        changed = not torch.allclose(
            tensors["receiver_target"],
            tensors["receiver_original"],
            atol=float(atol),
            rtol=0.0,
        )
        coords = entry["receiver_original"]["coords"]
        key = entry["receiver_original"]["key"]
        slot = key.split("[", 1)[0]
        rows.append(
            {
                "coords": coords,
                "label": key[len(slot) :],
                "kind": kind,
                "seam": seam,
                "status": "passed" if exact and positive and changed else "failed",
                "identity_exact": exact,
                "identity_max_abs": _max(identity),
                "sender_effect": _max(sender),
                "receiver_change": _max(receiver),
                "atol": float(atol),
            }
        )

    target = Path(outputs["controls"])
    target.parent.mkdir(parents=True, exist_ok=True)
    write_table(target, rows)


def _coords_token(coords: Mapping[str, Any]) -> str:
    import json

    return json.dumps(coords, sort_keys=True)


def _same_shape(tensors: Mapping[str, Any], names: tuple[str, ...]) -> None:
    shapes = {name: tuple(tensors[name].shape) for name in names}
    if len(set(shapes.values())) != 1:
        raise StepError(
            f"certify_control: {' / '.join(names)} differ in shape ({shapes}) — "
            "the legs compare reads of one address under different models"
        )


def _max(difference: Any) -> float:
    return float(difference.max()) if difference.numel() else 0.0
