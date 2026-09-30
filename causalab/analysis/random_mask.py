"""``causalab.analysis.random_mask`` — a size-matched random mask, the DBM control.

```json
"random": {
  "type": "script", "script": {"module": "causalab.analysis.random_mask"},
  "inputs": {"gate": {"step": "fit", "file": "gate.safetensors"},
             "seed": 0, "match_layers": true, "draw": "uniform"},
  "outputs": {"gate": "gate.safetensors"}
}
```

A fitted DBM mask that selects ``m`` of ``d`` units scores something; the
question a control answers is how much of that score *any* ``m`` units would
get. This script draws that control: it reads a fitted gate bundle and writes a
bundle of the same shape whose hard mask has exactly the input's selected
count per tensor, on a uniformly drawn set of units. The hard split is the
fit's own (§2.5 ``parametrization``, read from the header): ``θ > 0`` for a
sigmoid gate, ``θ > ½`` for a ``clamp`` gate, and for a ``hard_concrete`` gate
``θ > logit((½−γ)/(ζ−γ))`` at the ``stretch`` the fit stamped beside its map
(``0`` at the default). ``theta`` is written decisively on either side of that
threshold — ``1`` / ``0`` for a ``clamp`` gate, whose parameter lives on the
unit interval, and one unit either side of the threshold for the others — so
a reloaded gate is the hard split and nothing soft survives to blur the
comparison.

``draw`` picks the null. ``uniform`` (the default) draws the count from **all**
units, the null "any m units". ``complement`` draws it from the units the fit
**dropped** (NeuroSurgeon's ``complement_sampled``): the sharper null when the
fitted set is small in a frame whose leading units matter — a uniform 16-of-64
draw shares ~4 units with the fit's own 16, a complement draw shares none — and
refused when the fit kept more than half, where the complement is too small to
draw from (as NeuroSurgeon refuses it).

The bundle is taken as it comes. A fit writes one ``theta`` entry per sweep
point, each keyed by its coordinates, and every entry is resampled at its own
count (``match_layers`` true, the default). ``top_k`` (optional, a non-negative
integer) replaces that count with a fixed one — the control for a document that
reads the gate out at ``top_k`` (§2.5) rather than at its map's threshold, so
the null is size-matched to the cut actually scored, not to the fit's own split.
A bundle that also holds other
slots — a ``trajectory`` of a fit that trained a rotation beside its gate
photographs ``weight`` and ``theta`` at every step — is read for its ``theta``
entries alone; the control written holds only those, and a document loads the
rotation from the fit's own bundle beside it. With ``match_layers`` false only
the *total* count is kept and redistributed uniformly over the union of all
units, which is the weaker null: same budget, no layer structure. ``entry``
(a coordinate mapping, the shape a protocol ``entry`` selector has) narrows the
output to the entries it matches; the rest are dropped, header table included.

Every header field of the input is copied verbatim — identity, the per-entry
table, and anything a future gate kind records there — so the output is
addressable exactly where the fit was, and an apply document that names the
fit's bundle accepts the control at the same address. Provenance is then the
runner's: it stamps ``engine`` for this step, and inherits
the fit's identity from the tensor input, so nothing here has to claim it.

A grouped gate (``group: head``, ``group: expert_neuron``) or a position gate
(``axis: position``) needs nothing special: its ``theta`` already has one entry
per unit — one per head, the ``(num_experts, d_expert)`` table, or one per
position of the window — so the count matched and the units drawn are heads,
expert neurons or positions, and the ``group`` / ``group_map`` / ``axis`` the
fit stamped come through with the rest of the header.

A ``boundary`` bundle (Boundless DAS, §2.5) is refused: its ``theta`` is one
β over the ordered coordinates of the rotation it sits behind, not one entry
per unit, so there is no set of units to resample. The size-matched control
for a boundary fit is a random basis at the learned rank ``⌈θ · width⌉``
(``hard_mask_size`` in the ``fit_diagnostics.json`` beside the bundle) —
``demos/methods/protocols/random_subspace_control.json`` with ``k`` set to it — and
the refusal names the fraction the bundle holds.

The draw is a pure function of ``seed`` and the entry: each ``theta`` is drawn
through its own generator, seeded from ``(seed, entry key)``, so an entry's
control does not depend on which other entries share the bundle — the control
drawn for ``theta[l1=0.1]`` is the same whether the fit swept one penalty or
ten, or was narrowed by ``entry``. The un-matched draw (``match_layers``
false) is one draw over the union of the entries by construction, and is
seeded from ``seed`` alone.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_io import StepError
from causalab.protocol.schema import (
    FEATURIZER_SLOTS,
    GATE_MAPS,
    hard_concrete_threshold,
)

__all__ = ["main"]

#: The one slot a gate stores; the tensors this script draws over.
(THETA,) = FEATURIZER_SLOTS["gate"]

#: The nulls a control may be drawn from: ``uniform`` over all units, or
#: ``complement`` over the units the fit dropped (NeuroSurgeon's
#: ``complement_sampled``).
DRAWS: tuple[str, ...] = ("uniform", "complement")


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    import torch
    from causalab.io.tensor_files import load_file, save_file

    from causalab.io.step_io import entry_table
    from causalab.io.env import read_safetensors_metadata

    source = inputs["gate"]
    if not isinstance(source, (str, Path)):
        raise StepError(
            "random_mask: 'gate' must reference the fitted bundle itself, not a "
            "tensor selected out of it — the header is what makes the control "
            "addressable where the fit was. Drop the reference's 'entry' and "
            "pass this script's own 'entry' input to narrow the draw"
        )
    source = Path(source)
    if not source.is_file():
        raise StepError(f"random_mask: 'gate' bundle {str(source)!r} does not exist")
    if "seed" not in inputs:
        raise StepError("random_mask: 'seed' is required — a control is reproducible")
    seed = inputs["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise StepError(
            f"random_mask: 'seed' must be an integer, got {seed!r} — a recorded "
            "seed is what reproduces a recorded control, so it is not coerced"
        )
    match_layers = inputs.get("match_layers", True)
    if not isinstance(match_layers, bool):
        raise StepError(
            f"random_mask: 'match_layers' must be true or false, got "
            f"{match_layers!r} — it picks which null is drawn, so it is not coerced"
        )
    draw = inputs.get("draw", "uniform")
    if draw not in DRAWS:
        raise StepError(
            f"random_mask: 'draw' is one of {list(DRAWS)}, got {draw!r} — it picks "
            "which null is drawn, so it is not coerced"
        )
    if draw == "complement" and not match_layers:
        raise StepError(
            "random_mask: 'complement' draws inside each entry's own dropped units; "
            "it has no un-matched form (match_layers false)"
        )

    metadata = dict(read_safetensors_metadata(source) or {})
    tensors = load_file(str(source))
    entries = entry_table(metadata)
    thetas = _theta_entries(tensors, entries)
    if (selection := inputs.get("entry")) is not None:
        thetas = _select(thetas, entries, selection)

    # the fit's own hard split (§2.5 parametrization): θ > 0 for a sigmoid
    # gate, θ > ½ for a clamp gate — the control matches the count the fit's
    # replay would score through, and is written on the same map's poles
    # a bundle FITTED in a pool (its stamp says so): its θ ranks against its
    # co-members, and one bundle cannot know the pooled cut's share — refused
    # before any per-entry work. A pooled READOUT of separately fitted bundles
    # is a document fact no bundle carries, so it cannot be seen here: `top_k`
    # is a per-entry count, and a control for a readout pool is drawn per member
    # at that member's own share of the cut
    pooled = [
        key
        for key in thetas
        if entries.get(key, {}).get("pool", metadata.get("pool")) is not None
    ]
    if pooled:
        raise StepError(
            f"random_mask: {source} holds gates fitted in pool "
            f"({', '.join(pooled)}) — a size-matched control for a pool needs "
            "every member's bundle and the pooled cut, which this step does not "
            "take yet; draw the control per pool outside it"
        )
    maps = {key: _parametrization(metadata, entries, key) for key in thetas}
    boundaries = [
        key for key, m in maps.items() if m in GATE_MAPS and GATE_MAPS[m].indexed
    ]
    if boundaries:
        # one β over an ordered basis (§2.5 `boundary`): no units to resample.
        # The natural control is a random basis at the learned rank, and the
        # refusal says which rank that is
        fractions = ", ".join(
            f"{key!r}: θ = {float(thetas[key].view(-1)[0]):.3f}" for key in boundaries
        )
        raise StepError(
            f"random_mask: {source} holds a boundary gate ({', '.join(boundaries)}) "
            "— its theta is one boundary over the ordered coordinates of the "
            "rotation it sits behind, the kept fraction rather than one entry per "
            "unit, so there is no set of units to resample. The size-matched "
            "control for a boundary fit is a random basis at the learned rank "
            f"⌈θ · width⌉ ({fractions}; the rank is hard_mask_size in the "
            "fit_diagnostics.json beside the bundle): author "
            "demos/methods/protocols/random_subspace_control.json with k set to it"
        )
    top_k = inputs.get("top_k")
    if top_k is None and any(m == "budget" for m in maps.values()):
        raise StepError(
            f"random_mask: {source} holds a budget gate — its theta is a ranking "
            "with no threshold, so the count to match is the consumer's 'top_k' "
            "(§2.5); pass it"
        )
    thresholds = {
        key: (
            0.0
            if maps[key] == "budget"
            else _hard_threshold(metadata, entries, key, source)
        )
        for key in thetas
    }
    counts = {
        key: int((theta > thresholds[key]).sum()) for key, theta in thetas.items()
    }
    if top_k is not None:
        # the consumer cuts the ranking at a count (§2.5 `top_k`): match that
        # count, not the threshold's — a control is size-matched to what was
        # scored
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise StepError(
                f"random_mask: 'top_k' must be a non-negative integer, got {top_k!r}"
            )
        for key, theta in thetas.items():
            if top_k > theta.numel():
                raise StepError(
                    f"random_mask: 'top_k' {top_k} exceeds the {theta.numel()} units "
                    f"of entry {key!r}"
                )
        counts = {key: top_k for key in thetas}
    if draw == "complement":
        # the same count, drawn from the units the fit dropped: one generator
        # per entry as below, permuting the dropped indices instead of all
        drawn = {}
        for key, theta in thetas.items():
            dropped = (theta.view(-1) <= thresholds[key]).nonzero().view(-1)
            if counts[key] > dropped.numel():
                raise StepError(
                    f"random_mask: 'complement' needs at least as many dropped units "
                    f"as kept ones, but {key!r} keeps {counts[key]} of "
                    f"{theta.numel()} — a complement draw has nothing to stand on when "
                    "the fit kept more than half; use draw 'uniform'"
                )
            order = torch.randperm(
                dropped.numel(),
                generator=torch.Generator().manual_seed(_entry_seed(seed, key)),
            )
            drawn[key] = dropped[order[: counts[key]]]
    elif match_layers:
        # one generator per entry, seeded from (seed, key): the draw for an
        # entry is its own, whatever else the bundle holds
        drawn = {
            key: torch.randperm(
                theta.numel(),
                generator=torch.Generator().manual_seed(_entry_seed(seed, key)),
            )[: counts[key]]
            for key, theta in thetas.items()
        }
    else:
        # one draw over the union of every tensor's units, then cut back into
        # the tensors it came from: the budget survives, the layer structure
        # does not — which is what the un-matched null is for
        generator = torch.Generator().manual_seed(seed)
        sizes = [theta.numel() for theta in thetas.values()]
        flat = torch.randperm(sum(sizes), generator=generator)[: sum(counts.values())]
        drawn, start = {}, 0
        for key, size in zip(thetas, sizes):
            inside = flat[(flat >= start) & (flat < start + size)] - start
            drawn[key] = inside
            start += size

    written = {}
    for key, theta in thetas.items():
        off, on = _poles(thresholds[key], maps[key])
        mask = torch.full_like(theta, off)
        mask.view(-1)[drawn[key]] = on
        written[key] = mask
    if entries and len(written) != len(entries):
        metadata["entries"] = json.dumps(
            {key: entries[key] for key in written if key in entries}, sort_keys=True
        )
    target = Path(outputs["gate"])
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(written, str(target), metadata=metadata or None)


def _hard_threshold(
    metadata: Mapping[str, Any],
    entries: Mapping[str, Mapping[str, Any]],
    key: str,
    source: Path,
) -> float:
    """Where the bundle's own map splits ``theta`` into the hard mask (§2.5
    ``parametrization``): ``½`` under ``clamp``; ``0`` under ``sigmoid`` — and
    under a stamp from before the field existed, which is a sigmoid gate; under
    ``hard_concrete`` the θ where the stretched σ(θ) crosses ½,
    ``logit((½ − γ) / (ζ − γ))`` at the stamped ``stretch`` — the same
    arithmetic the gate itself uses (``protocol.schema.hard_concrete_threshold``),
    so the control's count is the fit's own. A hard-concrete stamp with no
    ``stretch`` is refused rather than read at ``0``: every protocol-written
    bundle stamps both (the identity check refuses a missing key), so a bundle
    without it was not written by a fit, and a silently wrong threshold would
    be a plausible-looking wrong null. Read from the entry's record first (a
    swept fit stamps per entry), then the file-level identity."""
    record = entries.get(key, {})
    parametrization = _parametrization(metadata, entries, key)
    if parametrization == "clamp":
        return 0.5
    if parametrization == "hard_concrete":
        stretch = record.get("stretch", metadata.get("stretch"))
        if stretch is None:
            raise StepError(
                f"{source}: entry {key!r} is stamped parametrization 'hard_concrete' "
                "but no 'stretch' — a hard-concrete fit stamps both, and the hard "
                "split depends on the stretch, so this bundle's threshold is "
                "unknown"
            )
        lo, hi = json.loads(stretch) if isinstance(stretch, str) else stretch
        return hard_concrete_threshold((float(lo), float(hi)))
    return 0.0


def _parametrization(
    metadata: Mapping[str, Any], entries: Mapping[str, Mapping[str, Any]], key: str
) -> str:
    """The bundle's map for one entry (§2.5 ``parametrization``): the entry's
    record first (a swept fit stamps per entry), then the file-level identity,
    ``sigmoid`` under a stamp from before the field existed."""
    record = entries.get(key, {})
    return str(
        record.get("parametrization", metadata.get("parametrization", "sigmoid"))
    )


def _poles(threshold: float, parametrization: str) -> tuple[float, float]:
    """The two decisive theta values a control is written on, keyed on the
    map: ``(0, 1)`` for a ``clamp`` gate, whose parameter lives on the unit
    interval and splits at ½; one unit either side of the threshold for the
    others — decisive under their hard map, which is the only map replay
    uses."""
    if parametrization == "clamp":
        return (0.0, 1.0)
    return (threshold - 1.0, threshold + 1.0)


def _entry_seed(seed: int, key: str) -> int:
    """The generator seed one entry's draw runs from: ``seed`` and the entry
    key, hashed together, so the draw is a function of the two alone — the
    same control for ``theta[l1=0.1]`` whether or not ``theta[l1=0.01]``
    shares its bundle. SHA-256 rather than `hash`, whose string hashing
    is salted per process."""
    digest = hashlib.sha256(f"{seed}\x00{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _theta_entries(
    tensors: Mapping[str, Any], entries: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """The bundle's ``theta`` entries in sorted key order, or a refusal naming
    what the bundle holds instead.

    A bundle with no ``theta`` at all — a ``weight`` (a subspace fit) or a
    harvested read — is a different kind of artifact that no random draw of
    ``θ > 0`` units can stand in for. A bundle that holds ``theta`` entries
    *beside* other slots (the trajectory of a fit that trained a rotation with
    its gate) is read for its thetas; the other slots are not the control's to
    rewrite and are left out of what it writes."""
    from causalab.protocol.bundles import parse_entry_key

    slots = {
        key: str(entries.get(key, {}).get("slot", parse_entry_key(key)[0]))
        for key in tensors
    }
    thetas = sorted(key for key, slot in slots.items() if slot == THETA)
    if not thetas:
        others = sorted(set(slots.values()))
        held = f"holds slots {others}" if others else "holds no tensors"
        raise StepError(
            f"random_mask: 'gate' is not a fitted gate bundle — it {held}; a control "
            f"is drawn over {THETA!r} entries (one per fitted point) and there are none"
        )
    return {key: tensors[key] for key in thetas}


def _select(
    thetas: Mapping[str, Any],
    entries: Mapping[str, Mapping[str, Any]],
    selection: Any,
) -> dict[str, Any]:
    """The entries matching every ``(name, value)`` pair of ``selection``.

    Same matching rule as [`causalab.protocol.bundles.select_entry`][] —
    pairs, never the rendered label — minus its uniqueness demand: a control
    may cover several fitted points at once (every seed of one penalty, say),
    so several matches are a result here, not an ambiguity. No match at all is
    still a refusal, listing what was there to match."""
    from causalab.protocol.bundles import SLOT_KEY, parse_entry_key
    from causalab.protocol.lowering import label_value

    if not isinstance(selection, Mapping):
        raise StepError("random_mask: 'entry' maps coordinate names to values")
    if SLOT_KEY in selection and str(selection[SLOT_KEY]) != THETA:
        raise StepError(
            f"random_mask: 'entry' names slot {selection[SLOT_KEY]!r}, but a gate "
            f"bundle holds only {THETA!r}"
        )
    wanted = {
        name: label_value(value)
        for name, value in selection.items()
        if name != SLOT_KEY
    }

    def coords(key: str) -> Mapping[str, Any]:
        stored = entries.get(key)
        return (
            stored["coords"]
            if stored and "coords" in stored
            else parse_entry_key(key)[1]
        )

    kept = {
        key: theta
        for key, theta in thetas.items()
        if all(
            name in coords(key) and label_value(coords(key)[name]) == value
            for name, value in wanted.items()
        )
    }
    if not kept:
        shown = ", ".join(f"{name}={value}" for name, value in sorted(wanted.items()))
        raise StepError(
            f"random_mask: no {THETA!r} entry matches {{{shown}}} "
            f"(has {sorted(thetas)})"
        )
    return kept
