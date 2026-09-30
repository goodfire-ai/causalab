"""Write a joint DBM fit or replay document.

Fit one gate per component and layer under a shared sparsity penalty. Use
``--positions 0 1 2`` for independent gates at aligned prompt positions.
``--position all`` shares each gate across positions. The answer readout
uses ``--readout-position`` and defaults to the final token.

Families:
    heads: whole heads at attention_premix or delta_premix.
    head-channels: individual output channels within those heads.
    neurons: complete dense, routed-expert and shared-expert neuron outputs.
    all-neurons: head channels and MLP neurons fitted together.

The fit sweeps ``--penalties`` and ``--seeds``. Every point saves its fitted
gates. IIA scores the counterfactual answer. Logit difference subtracts the
base-answer logit from the counterfactual-answer logit after intervention.
Validation saves both metrics. Add ``--save-rank`` to save one record per
mask unit in rank.json. This file can be large for neuron masks.

Use ``--apply --penalty 0.01 --seed 0`` to replay one saved fit, or
``--apply-all`` to replay the complete grid. Supply ``--fit-dir`` and
``--confirmation`` with either form. Every gate loads the same saved cell.
Replay uses the saved hard mask.

Documents are validated before writing. Apply validation reads bundle headers
under ``--artifacts-root``. ``--no-validate`` skips validation.
"""

from __future__ import annotations

import copy

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from protocol_authoring import (  # noqa: E402 — the sibling module, found above
    AuthoringError,
    dump_document,
    ensure_site,
    ordered_document,
    layer_stream,
    layer_types,
    load_document,
    method_of,
    model_info,
    moe_layers,
    parse_layer_range,
    reserve_name,
    routes_experts,
)

from causalab.cli import register_model_key  # noqa: E402
from causalab.protocol.rules.errors import ProtocolError  # noqa: E402
from causalab.protocol.pipeline import compile_protocol  # noqa: E402
from causalab.protocol.lowering import point_count  # noqa: E402
from causalab.protocol.registry import ModelInfo, component_shape  # noqa: E402
from causalab.io.env import (  # noqa: E402
    FileArtifacts,
    FileDatasets,
    ResolutionEnv,
)
from causalab.protocol.schema import (  # noqa: E402
    MATCH_MODES,
    PROTOCOL_VERSION,
)

FAMILIES: tuple[str, ...] = ("heads", "head-channels", "neurons", "all-neurons")

#: The shipped DBM document whose ``method.train`` the flags inherit, so every
#: family trains under the one DBM method. Read at parser-build time so
#: ``--help`` shows the real values.
DBM_DOCUMENT = Path(__file__).resolve().parents[1] / "demos/methods/protocols/dbm.json"

#: The one intervened model every write is in force in, and its logits read.
#: The un-intervened model the counterfactual reads are taken on (§2.9).
UNWRITTEN = "original_counterfactual"
MASKED = "masked"
LOGITS = "logits"
LM_HEAD = "lm_head"

#: The fit's coordinate names as an apply document's ``entry`` selector spells
#: them (spec §2.5: short names, ``train`` dropped).
PENALTY_COORD = "objective.sparsity.weight"
SEED_COORD = "seed"


@dataclasses.dataclass(frozen=True)
class GateSpec:
    """One gate the document fits: its site, component, layer and group."""

    name: str
    site: str
    component: str
    layer: int
    group: str | None
    pos: Any = -1

    @property
    def tail(self) -> str:
        """``L{layer}_{unit}`` — the suffix the gate's read and write share."""
        return self.name.removeprefix("gate_")


@dataclasses.dataclass(frozen=True)
class Spec:
    """Everything a fit document is a function of, parsed from the flags."""

    model: str
    revision: str
    dtype: str
    variable: str
    dataset: str
    base_field: str
    cf_field: str
    pos: Any
    independent_positions: tuple[int, ...] | None
    readout_pos: int
    save_rank: bool
    positions: dict[str, Any] | None
    family: str
    layers: str | None
    dense: bool
    penalties: tuple[float, ...]
    seeds: tuple[int, ...]
    base_answer: str
    label: str
    match_mode: str
    anneal: tuple[float, float, float]
    optimizer: dict[str, Any]
    steps: dict[str, int]
    batch_pairs: int
    validation: str | None
    eval_every: int
    eval_metrics: tuple[str, ...]


# --------------------------------------------------------------------------- #
# gates per family
# --------------------------------------------------------------------------- #


def head_gates(
    info: ModelInfo, layers: range, *, channels: bool = False
) -> list[GateSpec]:
    """One head-grouped gate per layer, at the mixer the registry says the
    layer carries. Refuses a model with no per-layer stream pattern and a
    model lacking a stream's component (the registry's own [V4], by name)."""
    if layer_types(info) is None:
        raise AuthoringError(
            f"model {info.key!r} declares no per-layer stream types: its registry "
            "entry carries no layer_types (a dense tower, or a registry with no "
            "such field yet) — the heads family gates attention_premix on a "
            "full_attention layer and delta_premix on a linear_attention layer, "
            "and cannot tell which each layer is"
        )
    gates: list[GateSpec] = []
    unit = "head_channels" if channels else "heads"
    for layer in layers:
        stream = layer_stream(info, layer)
        component = "attention_premix" if stream == "full_attention" else "delta_premix"
        component_shape(info, component)  # [V4] on a model without the stream
        gates.append(
            GateSpec(
                name=f"gate_L{layer}_{unit}",
                site=f"L{layer}_{unit}",
                component=component,
                layer=layer,
                group=None if channels else "head",
            )
        )
    return gates


def neuron_gates(info: ModelInfo, layers: range, *, dense: bool) -> list[GateSpec]:
    """Per MoE layer an expert-neuron gate on the routed interior and a plain
    gate on the shared expert; per dense layer (only with ``dense``) a plain
    gate on ``mlp_neuron_output``. A model with no experts at all is refused
    unless ``dense`` says the MLP gate is what was meant."""
    if not routes_experts(info) and not dense:
        raise AuthoringError(
            f"model {info.key!r} routes no experts, so it has no expert_neuron_output "
            "or shared_expert_activation to gate — pass --dense to place one "
            "plain gate per layer on mlp_neuron_output instead"
        )
    sparse = set(moe_layers(info))
    if sparse:
        component_shape(info, "expert_neuron_output")  # [V4] without top-k / d_expert
        component_shape(info, "shared_expert_activation")
    gates: list[GateSpec] = []
    for layer in layers:
        if layer in sparse:
            gates.append(
                GateSpec(
                    name=f"gate_L{layer}_neurons",
                    site=f"L{layer}_neurons",
                    component="expert_neuron_output",
                    layer=layer,
                    group="expert_neuron",
                )
            )
            gates.append(
                GateSpec(
                    name=f"gate_L{layer}_shared",
                    site=f"L{layer}_shared",
                    component="shared_expert_activation",
                    layer=layer,
                    group=None,
                )
            )
        else:
            gates.append(
                GateSpec(
                    name=f"gate_L{layer}_neurons",
                    site=f"L{layer}_neurons",
                    component="mlp_neuron_output",
                    layer=layer,
                    group=None,
                )
            )
    return gates


def gates_for(spec: Spec, info: ModelInfo) -> list[GateSpec]:
    """Build gates in layer order, then assign their intervention positions."""
    layers = parse_layer_range(spec.layers, info)
    if spec.family == "heads":
        gates = head_gates(info, layers)
    elif spec.family == "head-channels":
        gates = head_gates(info, layers, channels=True)
    elif spec.family == "neurons":
        gates = neuron_gates(info, layers, dense=spec.dense)
    elif spec.family == "all-neurons":
        gates = sorted(
            head_gates(info, layers, channels=True)
            + neuron_gates(info, layers, dense=spec.dense),
            key=lambda gate: gate.layer,
        )
    else:
        raise AuthoringError(f"unknown family {spec.family!r}; one of {FAMILIES}")
    if spec.independent_positions is None:
        return [dataclasses.replace(gate, pos=spec.pos) for gate in gates]
    return [
        dataclasses.replace(
            gate,
            name=f"{gate.name}_P{pos}",
            site=f"{gate.site}_P{pos}",
            pos=pos,
        )
        for gate in gates
        for pos in spec.independent_positions
    ]


# --------------------------------------------------------------------------- #
# the documents
# --------------------------------------------------------------------------- #


def _skeleton(spec: Spec, dataset: str, description: str) -> dict[str, Any]:
    """The four groups both documents share before any gate is added (spec
    §1): the header, the model, data on ``dataset``, and a method holding the
    position table, the ``lm_head`` site, the one intervened model and the
    three metrics over its logits."""
    method: dict[str, Any] = {}
    doc: dict[str, Any] = {
        "header": {"protocol_version": PROTOCOL_VERSION, "description": description},
        "model": {"key": spec.model, "revision": spec.revision, "dtype": spec.dtype},
        "data": {
            "base": {"dataset": dataset, "field": spec.base_field},
            "counterfactual": {"dataset": dataset, "field": spec.cf_field},
        },
        "method": method,
    }
    if spec.positions:
        method["positions"] = dict(spec.positions)
    method["sites"] = {}
    method["featurizers"] = {}
    method["reads"] = {}
    method["writes"] = {}
    method["intervened_models"] = {
        UNWRITTEN: {"input": "counterfactual", "reads": []},
        MASKED: {"input": "base", "reads": [LOGITS], "writes": []},
    }
    aggregations = {
        "iia": {
            "kind": "match",
            "expected": spec.label,
            "mode": spec.match_mode,
        },
        "logit_diff": {
            "kind": "logit_diff",
            "a": spec.label,
            "b": spec.base_answer,
        },
        "ce": {"kind": "cross_entropy", "target": spec.label},
    }
    method["save"] = [
        {
            "read": LOGITS,
            "model": MASKED,
            "aggregation": aggregations[name],
            "file_path": f"{name}.json",
        }
        for name in ("iia", "ce", "logit_diff")
    ]
    if spec.save_rank:
        method["save"].append({"kind": "rank", "file_path": "rank.json"})
    ensure_site(method, LM_HEAD, {"component": LM_HEAD})
    return doc


def _add_gate(
    method: dict[str, Any],
    gate: GateSpec,
    pos: Any,
    *,
    loaded: Mapping[str, Any] | None = None,
) -> None:
    """Declare one gate: its site, the featurizer (trainable, or loaded from
    ``loaded``'s ``file_path``/``entry``), the counterfactual read through
    it, the swap write, the write's place in the intervened model, and — for
    a trainable gate — its bundle in ``save``."""
    ensure_site(
        method, gate.site, {"component": gate.component, "layers": [gate.layer]}
    )
    reserve_name(method, gate.name, "featurizers")
    featurizer: dict[str, Any] = {"kind": "gate"}
    if gate.group is not None:
        featurizer["group"] = gate.group
    if loaded is not None:
        featurizer.update(loaded)
    method["featurizers"][gate.name] = featurizer
    read, write = f"cf_{gate.tail}", f"swap_{gate.tail}"
    reserve_name(method, read, "reads")
    method["reads"][read] = {"site": gate.site, "pos": pos, "featurizer": gate.name}
    method["intervened_models"][UNWRITTEN]["reads"].append(read)
    reserve_name(method, write, "writes")
    method["writes"][write] = {
        "site": gate.site,
        "pos": pos,
        "featurizer": gate.name,
        "do": {"swap": read},
    }
    method["intervened_models"][MASKED]["writes"].append(write)
    if loaded is None:
        method["save"].append(
            {
                "value": gate.name,
                "site": gate.site,
                "file_path": f"{gate.name}.safetensors",
            }
        )


def _add_logits(method: dict[str, Any], pos: Any) -> None:
    """The ``lm_head`` read on the intervened model every metric reduces."""
    reserve_name(method, LOGITS, "reads")
    method["reads"][LOGITS] = {"site": LM_HEAD, "pos": pos}


def _describe(spec: Spec, gates: Sequence[GateSpec], *, apply: bool) -> str:
    """Describe the intervention and identify the score split."""
    patch_positions = spec.independent_positions or spec.pos
    description = (
        f"Joint {spec.family} DBM for {spec.variable!r} on {spec.model}. "
        f"The {len(gates)} gates swap counterfactual values at positions "
        f"{json.dumps(patch_positions)}. Answer logits are read at position "
        f"{spec.readout_pos}. "
    )
    if apply:
        return description + (
            "Each gate loads its saved penalty and seed cell. Confirmation "
            "metrics use the hard mask (theta > 0). "
            "Generated by scripts/joint_dbm.py."
        )
    return description + (
        "One sparsity term covers all gates. Its weight sweeps "
        f"{list(spec.penalties)}, crossed with seeds {list(spec.seeds)}. "
        "iia.json, ce.json and logit_diff.json score the training split. "
        f"train_eval.json scores {spec.validation!r}. Check decisive_fraction "
        "in fit_diagnostics.json before interpreting results. "
        "Generated by scripts/joint_dbm.py."
    )


def _saved_aggregation(method: Mapping[str, Any], label: str) -> dict[str, Any]:
    """The ``{read, model, aggregation}`` of the save entry whose file stem is
    ``label`` — what an eval entry restates (§2.11)."""
    for entry in method.get("save", []):
        if entry.get("file_path") == f"{label}.json" and "aggregation" in entry:
            return {
                key: copy.deepcopy(entry[key])
                for key in ("read", "model", "aggregation")
            }
    raise ValueError(f"no saved aggregation is labelled {label!r}")


def build_fit_document(spec: Spec, info: ModelInfo) -> dict[str, Any]:
    """The fit document ``spec`` describes."""
    if spec.validation is None:
        raise AuthoringError("--validation names the split train.eval scores")
    gates = gates_for(spec, info)
    doc = _skeleton(spec, spec.dataset, _describe(spec, gates, apply=False))
    method = doc["method"]
    for gate in gates:
        _add_gate(method, gate, gate.pos)
    _add_logits(method, spec.readout_pos)
    names = [gate.name for gate in gates]
    start, end, frac = spec.anneal
    method["train"] = {
        "objective": {
            "fit": {"weight": 1.0, **_saved_aggregation(method, "ce")},
            "sparsity": {"weight": {"sweep": list(spec.penalties)}, "l1": names},
        },
        "params": names,
        "optimizer": dict(spec.optimizer),
        "steps": dict(spec.steps),
        "batch": {"pairs": spec.batch_pairs},
        "anneal": {f"{name}.theta.temperature": [start, end, frac] for name in names},
        "precision": {"feature": "fp32", "loss": "fp32"},
        "eval": {
            "every": {"epochs": spec.eval_every},
            "split": spec.validation,
            "aggregations": {
                name: _saved_aggregation(doc["method"], name)
                for name in spec.eval_metrics
            },
        },
        "seed": {"sweep": list(spec.seeds)},
    }
    return doc


def build_apply_document(
    spec: Spec,
    info: ModelInfo,
    *,
    fit_dir: str,
    penalty: float | None,
    seed: int | None,
    confirmation: str,
    all_cells: bool = False,
) -> dict[str, Any]:
    """Replay one saved cell or the complete grid on ``confirmation``."""
    if not all_cells and penalty not in spec.penalties:
        raise AuthoringError(
            f"--penalty {penalty} is not in the fit's grid {list(spec.penalties)} — "
            "the apply document selects one of the fit's own cells"
        )
    if not all_cells and seed not in spec.seeds:
        raise AuthoringError(
            f"--seed {seed} is not among the fit's seeds {list(spec.seeds)}"
        )
    gates = gates_for(spec, info)
    doc = _skeleton(spec, confirmation, _describe(spec, gates, apply=True))
    method = doc["method"]
    prefix = fit_dir.rstrip("/")
    entry: dict[str, Any] = {PENALTY_COORD: penalty, SEED_COORD: seed}
    if all_cells:
        doc["axes"] = {
            "fit_cell": {
                "rows": [
                    {"entry": {PENALTY_COORD: weight, SEED_COORD: fit_seed}}
                    for weight in spec.penalties
                    for fit_seed in spec.seeds
                ]
            }
        }
        entry = {"axis": "fit_cell.entry"}
    for gate in gates:
        _add_gate(
            method,
            gate,
            gate.pos,
            loaded={
                "file_path": f"{prefix}/{gate.name}.safetensors",
                "entry": entry,
            },
        )
    _add_logits(method, spec.readout_pos)
    return doc


# --------------------------------------------------------------------------- #
# flags
# --------------------------------------------------------------------------- #


def _dbm_defaults() -> dict[str, Any]:
    """``demos/methods/protocols/dbm.json``'s ``method.train`` — the family's defaults."""
    return json.loads(DBM_DOCUMENT.read_text(encoding="utf-8"))["method"]["train"]


def _position(
    raw: str, *, name: str, template: Path | None
) -> tuple[Any, dict[str, Any] | None]:
    """The ``pos`` reads and writes carry, and the ``positions`` table entry
    to declare (``None`` for the int / ``all`` sugar). JSON is a spec: an int,
    or a mapping declared under ``name``; anything else is the name of an
    entry in ``template``'s ``positions`` table."""
    if raw == "all":
        return "all", None
    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    if isinstance(value, bool):
        raise AuthoringError(f"--position {raw!r} is not a position")
    if isinstance(value, int):
        return value, None
    if isinstance(value, Mapping):
        return name, {name: dict(value)}
    if not isinstance(value, str):
        raise AuthoringError(
            f"--position {raw!r} is neither an int, a JSON position spec nor a name"
        )
    if template is None:
        raise AuthoringError(
            f"--position {value!r} is a name, so --template must give the document "
            "whose positions table declares it"
        )
    table = method_of(load_document(template)).get("positions", {})
    if not isinstance(table, Mapping) or value not in table:
        raise AuthoringError(
            f"{template} declares no positions.{value} (has "
            f"{sorted(table) if isinstance(table, Mapping) else 'no positions table'})"
        )
    return value, {value: table[value]}


def _distinct(values: Sequence[Any], flag: str) -> None:
    if not values:
        raise AuthoringError(
            f"{flag} is empty — a sweep with no values is a load error [V14], and "
            "a campaign of zero points is a bug in whatever produced the list"
        )
    if len(set(values)) != len(values):
        raise AuthoringError(f"{flag} repeats a value: {list(values)}")


def _build_parser() -> argparse.ArgumentParser:
    defaults = _dbm_defaults()
    optimizer = defaults["optimizer"]
    (anneal,) = defaults["anneal"].values()
    parser = argparse.ArgumentParser(
        prog="joint_dbm",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", required=True, type=Path, help="document to write")
    parser.add_argument(
        "--family",
        required=True,
        choices=FAMILIES,
        help="whole heads, head-output channels, MLP neurons, or all neurons",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="allow neuron gates on a model that routes no experts",
    )
    parser.add_argument(
        "--model", required=True, help="a registered model key (see --register-from-hf)"
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--dtype",
        default="bf16",
        help="model.dtype — part of every bundle's identity, so the apply "
        "document must say the same (default bf16)",
    )
    parser.add_argument(
        "--variable",
        required=True,
        help="the causal variable the counterfactual pairs change (recorded in "
        "the description)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="the counterfactual pair table the fit trains on",
    )
    parser.add_argument("--base-field", default="input")
    parser.add_argument("--cf-field", default="counterfactual_inputs[0]")
    patch = parser.add_mutually_exclusive_group()
    patch.add_argument(
        "--position",
        default="-1",
        help="patch an integer position (default -1), all positions with one "
        "shared mask, or a JSON or named position from --template",
    )
    patch.add_argument(
        "--positions",
        nargs="+",
        type=int,
        help="fit independent gates at these distinct, nonnegative aligned "
        "prompt positions; all examples must contain each position",
    )
    parser.add_argument(
        "--readout-position",
        type=int,
        default=-1,
        help="answer-logit position, independent of the patch (default -1)",
    )
    parser.add_argument(
        "--save-rank",
        action="store_true",
        help="save rank.json with one record per mask unit and evaluated cell",
    )
    parser.add_argument(
        "--position-name",
        default="target_pos",
        help="the name an inline JSON --position is declared under",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="a document whose positions table declares the named --position",
    )
    parser.add_argument(
        "--layers",
        default=None,
        metavar="START:STOP",
        help="half-open layer range (default: every layer of the model)",
    )
    parser.add_argument(
        "--penalties",
        nargs="*",
        type=float,
        default=[],
        metavar="W",
        help="the shared l1 weight's sweep values — the sparsity curve's axis",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[0],
        metavar="S",
        help="train.seed sweep",
    )
    parser.add_argument("--base-answer-column", default="base_answer")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--match-mode",
        default="exact",
        choices=MATCH_MODES,
        help="iia's match mode; first_token credits a multi-token answer's first "
        "piece (§2.10)",
    )
    parser.add_argument(
        "--anneal",
        nargs=3,
        type=float,
        default=anneal,
        metavar=("START", "END", "FRAC"),
        help="every gate's theta.temperature schedule",
    )
    parser.add_argument("--optimizer", default=optimizer["name"])
    parser.add_argument("--lr", type=float, default=optimizer["lr"])
    parser.add_argument(
        "--weight-decay", type=float, default=optimizer.get("weight_decay", 0.0)
    )
    steps = parser.add_mutually_exclusive_group()
    steps.add_argument("--epochs", type=int, default=None)
    steps.add_argument("--updates", type=int, default=None)
    parser.add_argument("--batch-pairs", type=int, default=defaults["batch"]["pairs"])
    parser.add_argument(
        "--validation", default=None, help="train.eval.split (required for a fit)"
    )
    parser.add_argument(
        "--eval-every", type=int, default=1, help="epochs between evals"
    )
    parser.add_argument(
        "--eval-metrics",
        nargs="+",
        default=["iia", "logit_diff"],
        choices=("iia", "logit_diff", "ce"),
    )
    apply = parser.add_argument_group("apply document (--apply)")
    replay = apply.add_mutually_exclusive_group()
    replay.add_argument(
        "--apply",
        action="store_true",
        help="write the apply document for one cell of this fit instead",
    )
    replay.add_argument(
        "--apply-all",
        action="store_true",
        help="replay every saved penalty and seed cell through a shared axis",
    )
    apply.add_argument(
        "--fit-dir",
        default="fit",
        help="where the fit's bundles are, relative to the artifacts root "
        "(the apply's file_path prefix)",
    )
    apply.add_argument("--penalty", type=float, default=None, help="the cell's weight")
    apply.add_argument("--seed", type=int, default=None, help="the cell's seed")
    apply.add_argument(
        "--confirmation", default=None, help="the split the apply scores"
    )
    check = parser.add_argument_group("validation")
    check.add_argument(
        "--no-validate", action="store_true", help="write without loading first"
    )
    check.add_argument("--data-root", type=Path, default=Path("."))
    check.add_argument("--artifacts-root", type=Path, default=Path("."))
    check.add_argument(
        "--register-from-hf",
        action="store_true",
        help="resolve an unregistered model key from its HF config before sizing "
        "the output (the CLI's opt-in; the only thing here that can touch the "
        "network)",
    )
    return parser


def _spec(args: argparse.Namespace) -> Spec:
    pos, positions = _position(
        args.position, name=args.position_name, template=args.template
    )
    if args.positions is not None:
        _distinct(args.positions, "--positions")
        if any(pos < 0 for pos in args.positions):
            raise AuthoringError(
                "--positions requires nonnegative absolute positions so its "
                "independent gates cannot overlap"
            )
    _distinct(args.penalties, "--penalties")
    _distinct(args.seeds, "--seeds")
    if args.epochs is not None:
        steps = {"epochs": args.epochs}
    elif args.updates is not None:
        steps = {"updates": args.updates}
    else:
        steps = dict(_dbm_defaults()["steps"])
    return Spec(
        model=args.model,
        revision=args.revision,
        dtype=args.dtype,
        variable=args.variable,
        dataset=args.dataset,
        base_field=args.base_field,
        cf_field=args.cf_field,
        pos=pos,
        independent_positions=tuple(args.positions) if args.positions else None,
        readout_pos=args.readout_position,
        save_rank=args.save_rank,
        positions=positions,
        family=args.family,
        layers=args.layers,
        dense=args.dense,
        penalties=tuple(args.penalties),
        seeds=tuple(args.seeds),
        base_answer=args.base_answer_column,
        label=args.label_column,
        match_mode=args.match_mode,
        anneal=tuple(args.anneal),
        optimizer={
            "name": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        steps=steps,
        batch_pairs=args.batch_pairs,
        validation=args.validation,
        eval_every=args.eval_every,
        eval_metrics=tuple(args.eval_metrics),
    )


def _document(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """The document the flags describe and its gate count."""
    spec = _spec(args)
    probe = {"model": {"key": spec.model, "revision": spec.revision}}
    if args.register_from_hf:
        register_model_key(probe)
    info = model_info(probe)
    if not (args.apply or args.apply_all):
        doc = build_fit_document(spec, info)
    else:
        if args.apply_all and (args.penalty is not None or args.seed is not None):
            raise AuthoringError(
                "--apply-all uses the complete --penalties and --seeds grid"
            )
        missing = [
            flag
            for flag, value in (
                ("--penalty", args.penalty),
                ("--seed", args.seed),
                ("--confirmation", args.confirmation),
            )
            if value is None and (flag == "--confirmation" or not args.apply_all)
        ]
        if missing:
            raise AuthoringError(f"--apply needs {', '.join(missing)}")
        doc = build_apply_document(
            spec,
            info,
            fit_dir=args.fit_dir,
            penalty=args.penalty,
            seed=args.seed,
            confirmation=args.confirmation,
            all_cells=args.apply_all,
        )
    return doc, len(doc["method"]["featurizers"])


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        doc, n_gates = _document(args)
        # what `dump_document` writes, so the loader validates the tree the
        # file will hold (a `train` block added after `save` would otherwise
        # draw the §5 rule 2 warning)
        doc = ordered_document(doc)
        if args.no_validate:
            train = doc["method"].get("train")
            points = (
                len(train["objective"]["sparsity"]["weight"]["sweep"])
                * len(train["seed"]["sweep"])
                if train
                else len(doc.get("axes", {}).get("fit_cell", {}).get("rows", [None]))
            )
            status = f"{points} point{'s' if points != 1 else ''}, not validated"
        else:
            env = ResolutionEnv(
                datasets=FileDatasets(root=args.data_root),
                artifacts=FileArtifacts(root=args.artifacts_root),
            )
            compiled = compile_protocol(doc, env=env)
            points = point_count(compiled.axes)
            status = (
                f"{points} point{'s' if points != 1 else ''}, "
                f"digest {compiled.digests.document[:16]}…"
            )
    except (AuthoringError, ProtocolError) as err:
        raise SystemExit(f"refused: {err}") from err
    dump_document(doc, args.out)
    print(f"wrote {args.out} ({n_gates} gates, {status})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
