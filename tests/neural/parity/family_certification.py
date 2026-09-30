"""Per-family raw-hook oracle certification for the hybrid tower.

Two tiers already existed and are easy to confuse. ``tests/golden/
test_a3b_engine_parity.py`` compares causalab's two *engines* with each other;
``tests/neural/engines/pytorch_hooks/test_write_oracle.py`` compares causalab
against an *independent* raw-hook oracle. This module is the second kind: it
extends that tier to two more families — ``qwen35moe`` (the hybrid tower on
``tiny-random/qwen3.5-moe``, CPU, fp32) and ``qwen36_a3b`` (the real
``Qwen/Qwen3.6-35B-A3B``, GPU, bf16, ``-m golden``) — and records the
eight fields a certification carries.

**What is compared.** Every case drives the same intervention through the
reference engine (an intervention specification, ``PointExecutor``) and
through the oracle (raw ``register_forward_hook`` / ``register_forward_pre_hook``
plus, at the DeltaNet kernel boundary no hook reaches, a swap of the mixer's
own kernel global — ``hook_oracle_lib``). Reads compare the whole tensor at
every position; writes compare the patched logits **and** a downstream
intermediate component read in the patched model. The cases cover both of the
tower's layer kinds (a Gated DeltaNet layer and a full-attention layer), the
module boundaries, the kernel's argument slots, the recurrence interior
(``delta_state``, ``delta_kv_mem``, ``delta_state_update``) read *and* written,
and the ``lm_head`` logits.

**The oracle matches the model's operation order.** For the recurrence that
means ``hook_oracle_lib.delta_recurrence``: ``torch_recurrent_gated_delta_rule``
transcribed one operation at a time, which is the formulation under which the
engine defines the per-step interior. It is held to the engine at a **bit-exact**
band in fp32 (`BANDS`), and that band is what turns an order-of-operations
mismatch into a mutation check: reassociating one sum in the oracle by one ulp
fails the certification on an intermediate component while the logits alone
would still pass (``test_family_certification.py``, T6). The chunked kernel is
*another* association of the same sums, and the record measures the gap
(``measurements.chunked_vs_stepwise_kernel_output``) rather than absorbing it.

**The engine runs its library path.** The engine can swap optional kernels
into the forward — the single-chunk Gated DeltaNet kernel for short sequences
(``CAUSALAB_GDN_SHORT_SEQ``) and the fused MoE glue (``CAUSALAB_MOE_GLUE``) —
each certified against the library by its own goldens at its own band. This
certification is about the hook machinery, so the engine side of every capture
runs under `tests._helpers.kernel_paths.library_kernel_paths` (both off)
and the oracle sees the same library operations; the record's
``context.kernel_paths`` is read from the environment inside that scope, so it
attests the setting the capture ran under rather than asserting it. A bf16-scale
disagreement here is therefore a placement or ordering fault, never a kernel's
documented rounding.

**The eight fields** (`CERTIFICATION_FIELDS`) live under ``certification`` in
each record, in the goldens' existing shape otherwise (``family``, ``values``,
``tolerance``, ``context`` …). Provenance reuses the landed spellings: the
causalab revision is [`causalab.provenance.runtime_identity`][] (its
``source_kind`` / ``tree_digest``), the model revision is the bundle's
requested revision beside the hub snapshot it resolved to, under
``requested_revision`` / ``resolved_revision``.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
from causalab.neural.shared.encoding import encode
from causalab.protocol.schema import PROTOCOL_VERSION
from causalab.provenance import runtime_identity

from tests._helpers.kernel_paths import kernel_paths_in_force, library_kernel_paths
from tests.neural.engines.pytorch_hooks import hook_oracle_lib as oracle_lib
from tests.neural.engines.pytorch_hooks._drive import base_data_section, executor_for
from tests.neural.engines.pytorch_hooks.conftest import OracleShim
from tests.protocol._docs import saved

__all__ = [
    "BANDS",
    "BASE_TEXT",
    "CERTIFIED_FAMILIES",
    "COUNTERFACTUAL_TEXT",
    "FAMILIES",
    "GOLDENS_DIR",
    "CERTIFICATION_FIELDS",
    "Band",
    "Capture",
    "FamilySpec",
    "capture_family",
    "certification_failures",
    "check_certification_fields",
    "compare_records",
    "load_record",
    "record_path",
    "render_record",
]

GOLDENS_DIR = Path(__file__).resolve().parent / "goldens"

#: The closed field set every certification record's
#: ``certification`` block carries — censused by
#: ``test_family_certification.py::test_the_record_carries_exactly_the_eight_fields``
#: (T5), in the pattern of ``tests/protocol/test_vocabulary_census.py``.
CERTIFICATION_FIELDS: tuple[str, ...] = (
    "model_revision",
    "causalab_revision",
    "attn_implementation",
    "dtype",
    "hook_names",
    "tensor_shapes",
    "max_activation_diff",
    "max_logit_diff",
)

BASE_TEXT = "the quick brown fox jumps"
COUNTERFACTUAL_TEXT = "a slow green turtle sleeps"

#: The step a state interchange lands on. Not the last one: a state edit at
#: step ``t`` feeds step ``t+1``, so a last-step edit would move nothing and
#: the anti-vacuity half of T7 would be unsatisfiable by construction.
STATE_WRITE_STEP = 1

N_PROBES = 8


@dataclasses.dataclass(frozen=True)
class FamilySpec:
    """One certified family: the checkpoint and the realization it is
    certified at."""

    name: str
    key: str
    dtype: str
    device: str
    #: ``"cpu"`` runs under ``-m "not golden"``; ``"golden"`` is the accelerator
    #: tier the nightly runs.
    tier: str


FAMILIES: dict[str, FamilySpec] = {
    "qwen35moe": FamilySpec(
        name="qwen35moe",
        key="tiny-random/qwen3.5-moe",
        dtype="fp32",
        device="cpu",
        tier="cpu",
    ),
    "qwen36_a3b": FamilySpec(
        name="qwen36_a3b",
        key="Qwen/Qwen3.6-35B-A3B",
        dtype="bf16",
        device="cuda",
        tier="golden",
    ),
}

#: The record files this certification owns. The three frozen goldens beside
#: them carry no ``certification`` block and are never regenerated (docs/TESTS.md).
CERTIFIED_FAMILIES: tuple[str, ...] = tuple(FAMILIES)


@dataclasses.dataclass(frozen=True)
class Band:
    """The declared tolerance of one dtype, justified by measurement in the
    record it is written into.

    ``activation`` and ``logits`` bound the engine-vs-oracle comparison (both
    sides computed in the same process on the same device, so nothing but the
    operation order can separate them). ``pins_abs`` / ``pins_rel`` bound the
    replay of the pinned scalars against a fresh capture — a *different*
    process, possibly a different node. ``state_substitution_drift`` bounds the
    one deliberate path-forcing: a state write substitutes the stepwise
    recurrence for the chunked kernel, and even a self-swap then moves the
    logits by the two kernels' association gap.
    """

    activation: float
    logits: float
    pins_abs: float
    pins_rel: float
    state_substitution_drift: float
    justification: str


BANDS: dict[str, Band] = {
    "fp32": Band(
        activation=0.0,
        logits=0.0,
        pins_abs=1e-4,
        pins_rel=0.0,
        state_substitution_drift=1e-6,
        justification=(
            "engine vs oracle measured 0.0 on every hook and on the logits (CPU, "
            "fp32, same process): the oracle reproduces the model's operation "
            "order, so the band is exact and a one-ulp reassociation fails it. "
            "Pins replay at the frozen goldens' 1e-4 (cross-node float text). The "
            "state self-swap drift is the chunked-vs-recurrent association gap, "
            "bounded as tests/neural/engines/pytorch_hooks/"
            "test_sites_round4_deltanet.py bounds it."
        ),
    ),
    "bf16": Band(
        activation=0.0,
        logits=0.0,
        pins_abs=1e-3,
        pins_rel=2**-7,
        state_substitution_drift=1.0,
        justification=(
            "engine vs oracle measured 0.0 on every hook and on the logits "
            "(cuda, bf16 weights, fp32 recurrence, same process): the same eager "
            "kernels over the same bytes in the same order. Pins replay within "
            "one bf16 ulp relative (2^-7) plus a 1e-3 absolute floor for values "
            "near zero; two captures in separate processes on the same 2xH100 "
            "node were byte-identical. The state self-swap drift measured "
            "0.484375 on the real checkpoint: the chunked-vs-recurrent "
            "association gap (5.96e-8 at the kernel output, fp32) amplified "
            "through 40 bf16 layers to ~4 bf16 ulps at the logit magnitude this "
            "model produces (|max| ~18, ulp 0.125); bounded at 8 such ulps, and "
            "required to stay below what the genuine state interchange moves "
            "(1.109 measured), which the tests assert."
        ),
    ),
}


# --------------------------------------------------------------------------- #
# the case table
# --------------------------------------------------------------------------- #

#: A layer role: the first Gated DeltaNet layer or the first full-attention
#: layer of the loaded tower, resolved per family (the fixture and the real
#: A3B run the two block kinds in different orders).
DELTA, FULL = "delta", "full"


@dataclasses.dataclass(frozen=True)
class ReadCase:
    component: str
    role: str | None  # None for lm_head


@dataclasses.dataclass(frozen=True)
class WriteCase:
    component: str
    role: str
    #: the written position — the last token, or the state's step
    pos: int
    #: downstream intermediate components read in the patched model, (component, role)
    downstream: tuple[tuple[str, str], ...]


READS: tuple[ReadCase, ...] = (
    ReadCase("block_output", DELTA),
    ReadCase("attention_output", DELTA),
    ReadCase("mlp_output", DELTA),
    ReadCase("delta_value", DELTA),
    ReadCase("delta_kernel_output", DELTA),
    ReadCase("delta_kv_mem", DELTA),
    ReadCase("delta_state_update", DELTA),
    ReadCase("delta_state", DELTA),
    ReadCase("block_output", FULL),
    ReadCase("attention_output", FULL),
    ReadCase("mlp_output", FULL),
    ReadCase("lm_head", None),
)

WRITES: tuple[WriteCase, ...] = (
    WriteCase("block_output", DELTA, -1, (("block_output", FULL),)),
    WriteCase("attention_output", FULL, -1, (("block_output", FULL),)),
    WriteCase(
        "delta_value",
        DELTA,
        -1,
        (("delta_kernel_output", DELTA), ("block_output", DELTA)),
    ),
    WriteCase(
        "delta_state",
        DELTA,
        STATE_WRITE_STEP,
        (("delta_kernel_output", DELTA), ("block_output", DELTA)),
    ),
)

#: kernel-boundary components, by the slot the oracle serves them from
_KERNEL_SLOTS: dict[str, str] = {
    "delta_value": "value",
    "delta_kernel_output": "out",
    "delta_kv_mem": "kv_mem",
    "delta_state_update": "state_update",
    "delta_state": "state",
}


def _pos_word(pos: int) -> str:
    return "last" if pos == -1 else f"step{pos}"


def _read_id(family: str, case: ReadCase, layer: int | None) -> str:
    suffix = "" if layer is None else f".L{layer}"
    return f"{family}.collect.{case.component}.identity.all{suffix}"


def _write_id(family: str, case: WriteCase, layer: int) -> str:
    return (
        f"{family}.interchange.{case.component}.identity.{_pos_word(case.pos)}.L{layer}"
    )


# --------------------------------------------------------------------------- #
# the engine side — intervention specifications through PointExecutor
# --------------------------------------------------------------------------- #


def _layers(bundle: ModelBundle) -> dict[str, int]:
    streams = bundle.streams
    assert "linear_attention" in streams and "full_attention" in streams, streams
    return {
        DELTA: streams.index("linear_attention"),
        FULL: streams.index("full_attention"),
    }


def _site(component: str, layer: int | None) -> dict[str, Any]:
    return (
        {"component": component}
        if layer is None
        else {
            "component": component,
            "layers": [layer],
        }
    )


#: The un-intervened model on each input, named as `causalab migrate` names
#: them when the network is read un-intervened on both roles (§2.9).
_ORIGINAL = {"base": "original_base", "counterfactual": "original_counterfactual"}


def _doc(with_counterfactual: bool) -> dict[str, Any]:
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=with_counterfactual),
        "method": {"intervened_models": {}, "sites": {}, "reads": {}, "save": []},
    }


def _read(
    doc: dict[str, Any], name: str, model: str, input_role: str, **address: Any
) -> None:
    """Declare read ``name`` at ``address``, list it on ``model`` (declaring the
    model on ``input_role`` if this is its first read) and save its tensor."""
    method = doc["method"]
    method["reads"][name] = dict(address)
    entry = method["intervened_models"].setdefault(model, {"input": input_role})
    assert entry["input"] == input_role, (model, entry["input"], input_role)
    entry.setdefault("reads", []).append(name)
    method["save"].append(saved(name, model, f"{name}.safetensors"))


def _engine(
    doc: dict[str, Any], bundle: ModelBundle, *, with_counterfactual: bool
) -> PointExecutor:
    return executor_for(
        doc,
        bundle,
        base_texts=[BASE_TEXT],
        counterfactual_texts=[COUNTERFACTUAL_TEXT] if with_counterfactual else None,
    )


def _engine_reads(
    bundle: ModelBundle, layers: Mapping[str, int]
) -> dict[str, dict[str, torch.Tensor]]:
    """Every read case at every position, on the base and on the counterfactual
    input: ``{"base": {read_id: tensor}, "counterfactual": {...}}`` — on CPU."""
    doc = _doc(with_counterfactual=True)
    names: dict[str, tuple[str, str]] = {}
    for i, case in enumerate(READS):
        layer = None if case.role is None else layers[case.role]
        site = f"s{i}"
        doc["method"]["sites"][site] = _site(case.component, layer)
        for role in ("base", "counterfactual"):
            name = f"r{i}_{role}"
            _read(doc, name, _ORIGINAL[role], role, site=site, pos="all")
            names[name] = (role, _read_id("family", case, layer))
    executor = _engine(doc, bundle, with_counterfactual=True)
    out: dict[str, dict[str, torch.Tensor]] = {"base": {}, "counterfactual": {}}
    for name, (role, read_id) in names.items():
        out[role][read_id] = executor.read_value(name).detach().to("cpu")
    return out


def _engine_write(
    bundle: ModelBundle,
    layers: Mapping[str, int],
    case: WriteCase,
    *,
    operand_input: str,
) -> dict[str, torch.Tensor]:
    """One interchange: ``swap`` the operand read at the same address from
    ``operand_input`` (``"counterfactual"`` for the genuine write,
    ``"base"`` for the no-op twin). Returns CPU tensors: ``after`` (patched
    last-position logits, ``(1, vocab)``), ``clean`` (the same, unpatched),
    ``logits_all`` (patched, every position) and one ``ds/<component>.L<n>``
    per downstream intermediate read in the patched model."""
    layer = layers[case.role]
    doc = _doc(with_counterfactual=True)
    doc["method"]["sites"]["tap"] = _site(case.component, layer)
    doc["method"]["sites"]["lm_head"] = {"component": "lm_head"}
    pos = {"index": case.pos}
    doc["method"]["intervened_models"]["patched"] = {
        "input": "base",
        "reads": [],
        "writes": ["patch"],
    }
    _read(doc, "v_cf", _ORIGINAL[operand_input], operand_input, site="tap", pos=pos)
    _read(doc, "clean", _ORIGINAL["base"], "base", site="lm_head", pos={"index": -1})
    _read(doc, "after", "patched", "base", site="lm_head", pos={"index": -1})
    _read(doc, "logits_all", "patched", "base", site="lm_head", pos="all")
    ds_names: dict[str, str] = {}
    for j, (component, role) in enumerate(case.downstream):
        site, name = f"ds{j}_site", f"ds{j}"
        doc["method"]["sites"][site] = _site(component, layers[role])
        _read(doc, name, "patched", "base", site=site, pos="all")
        ds_names[name] = f"ds/{component}.L{layers[role]}"
    doc["method"]["writes"] = {
        "patch": {"site": "tap", "pos": pos, "do": {"swap": "v_cf"}}
    }
    executor = _engine(doc, bundle, with_counterfactual=True)
    out = {
        "after": executor.read_value("after")[:, 0, :].detach().to("cpu"),
        "clean": executor.read_value("clean")[:, 0, :].detach().to("cpu"),
        "logits_all": executor.read_value("logits_all").detach().to("cpu"),
    }
    for name, label in ds_names.items():
        out[label] = executor.read_value(name).detach().to("cpu")
    return out


# --------------------------------------------------------------------------- #
# the oracle side — raw hooks, and the kernel boundary
# --------------------------------------------------------------------------- #


def _inputs(bundle: ModelBundle, text: str) -> dict[str, torch.Tensor]:
    batch = encode(bundle.tokenizer, [text])
    device = next(bundle.model.parameters()).device
    return {
        "input_ids": batch.input_ids.to(device),
        "attention_mask": batch.attention_mask.to(device),
    }


def _module_taps(
    shim: OracleShim, layers: Mapping[str, int]
) -> dict[tuple[str, str | None], tuple[torch.nn.Module, str]]:
    """``(component, role) -> (module, kind)`` for every module-boundary read."""
    taps: dict[tuple[str, str | None], tuple[torch.nn.Module, str]] = {}
    for case in READS:
        if case.component in _KERNEL_SLOTS:
            continue
        if case.component == "lm_head":
            taps[(case.component, None)] = (shim.hf_model.lm_head, "out")
            continue
        taps[(case.component, case.role)] = oracle_lib.component_module(
            shim, layers[case.role], case.component
        )
    return taps


@dataclasses.dataclass
class OracleForward:
    """One oracle forward on one input: module captures by ``(component,
    role)``, the DeltaNet layer's kernel call, and the recurrence run on its
    arguments — all on CPU."""

    logits: torch.Tensor
    captures: dict[tuple[str, str | None], torch.Tensor]
    call: oracle_lib.DeltaKernelCall
    recurrence: oracle_lib.DeltaRecurrence


def _oracle_forward(
    shim: OracleShim,
    layers: Mapping[str, int],
    inputs: Mapping[str, torch.Tensor],
    *,
    writes: list[oracle_lib.WriteSpec] | None = None,
    edit_args: oracle_lib.DeltaArgsEdit | None = None,
    substitute: oracle_lib.DeltaSubstitute | None = None,
) -> OracleForward:
    taps = _module_taps(shim, layers)
    named = {f"{c}|{r}": tap for (c, r), tap in taps.items()}
    with oracle_lib.delta_kernel_boundary(
        shim, layers[DELTA], edit_args=edit_args, substitute=substitute
    ) as calls:
        logits, grabbed = oracle_lib.capture_many_with_writes(
            shim, inputs, named, list(writes or [])
        )
    assert len(calls) == 1, (
        f"expected one kernel call at the DeltaNet layer, saw {len(calls)}"
    )
    call = calls[0]
    recurrence = oracle_lib.delta_recurrence(
        call.query,
        call.key,
        call.value,
        call.g,
        call.beta,
        initial_state=call.kwargs.get("initial_state"),
        use_qk_l2norm=bool(call.kwargs.get("use_qk_l2norm_in_kernel", False)),
    )
    captures = {key: grabbed[f"{key[0]}|{key[1]}"].to("cpu") for key in taps}
    return OracleForward(
        logits=logits.to("cpu"), captures=captures, call=call, recurrence=recurrence
    )


def _flat(t: torch.Tensor) -> torch.Tensor:
    """``(b, s, h, d) -> (b, s, h·d)`` — the engine's head-major contract form."""
    return t.reshape(t.shape[0], t.shape[1], -1).to("cpu")


def _oracle_read(fwd: OracleForward, case: ReadCase) -> torch.Tensor:
    """The oracle's value of one read case, in the engine's contract shape."""
    slot = _KERNEL_SLOTS.get(case.component)
    if slot is None:
        return fwd.captures[(case.component, case.role)]
    if slot == "value":
        return _flat(fwd.call.value)
    if slot == "out":
        return _flat(fwd.call.out)
    if slot == "kv_mem":
        return _flat(fwd.recurrence.kv_mems)
    if slot == "state_update":
        return _flat(fwd.recurrence.deltas)
    assert slot == "state"
    return fwd.recurrence.states.to("cpu")


def _hook_name(
    shim: OracleShim, layers: Mapping[str, int], case: ReadCase | tuple[str, str]
) -> str:
    component, role = (
        (case.component, case.role) if isinstance(case, ReadCase) else case
    )
    slot = _KERNEL_SLOTS.get(component)
    if slot is not None:
        return oracle_lib.delta_kernel_hook_name(shim, layers[DELTA], slot)
    if component == "lm_head":
        return "lm_head"
    module, kind = oracle_lib.component_module(shim, layers[role], component)
    return f"{oracle_lib.module_path(shim, module)}.{kind}"


def _oracle_write(
    shim: OracleShim,
    layers: Mapping[str, int],
    case: WriteCase,
    base_inputs: Mapping[str, torch.Tensor],
    operand: OracleForward,
) -> OracleForward:
    """The oracle's realization of one interchange: the operand forward's
    value at the written address lands on the base forward at the same address,
    by the raw mechanism that address has."""
    layer = layers[case.role]
    if case.component in ("block_output", "attention_output"):
        module, kind = oracle_lib.component_module(shim, layer, case.component)
        donor = operand.captures[(case.component, case.role)]
        device = next(shim.hf_model.parameters()).device
        patch = donor[:, case.pos, :].to(device)

        def write(hidden: torch.Tensor) -> None:
            hidden[:, case.pos, :] = patch

        return _oracle_forward(
            shim, layers, base_inputs, writes=[(module, kind, write)]
        )
    if case.component == "delta_value":
        donor_value = operand.call.value

        def edit_args(query, key, value, g, beta):
            value = value.clone()
            value[:, case.pos] = donor_value[:, case.pos].to(value)
            return query, key, value, g, beta

        return _oracle_forward(shim, layers, base_inputs, edit_args=edit_args)
    assert case.component == "delta_state", case
    donor_state = operand.recurrence.states[:, case.pos]

    def edit_state(step: int, state: torch.Tensor) -> torch.Tensor:
        return donor_state.to(state) if step == case.pos else state

    def substitute(query, key, value, g, beta, kwargs):
        run = oracle_lib.delta_recurrence(
            query,
            key,
            value,
            g,
            beta,
            initial_state=kwargs.get("initial_state"),
            use_qk_l2norm=bool(kwargs.get("use_qk_l2norm_in_kernel", False)),
            edit_state=edit_state,
        )
        final = run.final_state if kwargs.get("output_final_state") else None
        return run.out, final

    return _oracle_forward(shim, layers, base_inputs, substitute=substitute)


# --------------------------------------------------------------------------- #
# pins — the frozen goldens' recipe, verbatim
# --------------------------------------------------------------------------- #


def _pin_values(
    case_id: str, value: torch.Tensor, clean: torch.Tensor | None
) -> dict[str, Any]:
    """shape + mean/std/first/last, ``N_PROBES`` seeded probe elements, and —
    for writes — the non-vacuity pin ``clean_delta.max`` (the recipe of
    ``test_parity_goldens.py::_pin_values``, kept identical)."""
    values: dict[str, Any] = {}
    t = value.detach().float().cpu()
    values[f"{case_id}.out.shape"] = list(t.shape)
    flat = t.flatten()
    values[f"{case_id}.out.mean"] = float(t.mean())
    values[f"{case_id}.out.first"] = float(flat[0])
    values[f"{case_id}.out.last"] = float(flat[-1])
    if t.numel() >= 2:
        values[f"{case_id}.out.std"] = float(t.std())
    gen = torch.Generator().manual_seed(0)
    for j, i in enumerate(
        torch.randperm(flat.numel(), generator=gen)[:N_PROBES].tolist()
    ):
        values[f"{case_id}.probe.{j}"] = float(flat[i])
    if clean is not None:
        values[f"{case_id}.clean_delta.max"] = float(
            (value.detach().float().cpu() - clean.detach().float().cpu()).abs().max()
        )
    return values


# --------------------------------------------------------------------------- #
# provenance — the landed spellings
# --------------------------------------------------------------------------- #


def _resolved_model_revision(bundle: ModelBundle) -> str | None:
    """The hub snapshot the bundle's ``(key, revision)`` resolved to in the
    local cache — the sha of the ``snapshots/<sha>/`` directory ``config.json``
    was read from. ``None`` when the cache has no such file (a model built
    from a config rather than loaded, or a path that is not a hub key)."""
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(bundle.key, "config.json", revision=bundle.revision)
    if not isinstance(found, str):
        return None
    return Path(found).parent.name


def _provenance(bundle: ModelBundle) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = runtime_identity()
    causalab_revision = {
        "source_kind": identity.source_kind,
        "tree_digest": identity.tree_digest,
    }
    model_revision = {
        "key": bundle.key,
        "requested_revision": bundle.revision,
        "resolved_revision": _resolved_model_revision(bundle),
    }
    return model_revision, causalab_revision


def _context() -> dict[str, str]:
    """Capture provenance; ``kernel_paths`` is what the environment holds at the
    call (inside `capture_family`'s scope: the library setting)."""
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "kernel_paths": kernel_paths_in_force(),
    }


# --------------------------------------------------------------------------- #
# the capture
# --------------------------------------------------------------------------- #


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    assert tuple(a.shape) == tuple(b.shape), (tuple(a.shape), tuple(b.shape))
    return float((a.detach().double().cpu() - b.detach().double().cpu()).abs().max())


@dataclasses.dataclass
class Capture:
    """One family's certification run: the record to write, and the
    per-comparison differences in forward order (for the first-failure report
    T6 asks for)."""

    record: dict[str, Any]
    #: ``"<case_id>@<hook_name>" -> max abs diff`` (activations)
    activation_diffs: dict[str, float]
    #: ``"<case_id>" -> max abs diff`` (logits)
    logit_diffs: dict[str, float]


def capture_family(bundle: ModelBundle, family: str) -> Capture:
    """Run every case through the engine and the oracle and build the record.

    The bundle must be the family's realization ([`FAMILIES`][causalab.protocol.registry.families.FAMILIES]): eager
    attention, the declared dtype. The comparison is engine vs oracle in this
    process; the pins are the engine's values. The whole capture runs under
    `library_kernel_paths` (module docstring): the oracle reads neither
    variable, so this is the engine's setting alone, and ``context`` records it.
    """
    with library_kernel_paths():
        return _capture_family(bundle, family)


def _capture_family(bundle: ModelBundle, family: str) -> Capture:
    spec = FAMILIES[family]
    assert bundle.model.config._attn_implementation == "eager"
    assert bundle.dtype == spec.dtype, (bundle.dtype, spec.dtype)
    band = BANDS[spec.dtype]
    shim = OracleShim(hf_model=bundle.model)
    layers = _layers(bundle)
    base_inputs = _inputs(bundle, BASE_TEXT)
    cf_inputs = _inputs(bundle, COUNTERFACTUAL_TEXT)

    values: dict[str, Any] = {}
    activation_diffs: dict[str, float] = {}
    logit_diffs: dict[str, float] = {}
    tensor_shapes: dict[str, list[int]] = {}
    hook_names: set[str] = set()

    def compare(
        case_id: str, hook: str, have: torch.Tensor, want: torch.Tensor
    ) -> None:
        hook_names.add(hook)
        tensor_shapes.setdefault(hook, list(have.shape))
        diff = _max_abs(have, want)
        if hook == "lm_head":
            logit_diffs[case_id] = diff
        else:
            activation_diffs[f"{case_id}@{hook}"] = diff

    # --- reads ------------------------------------------------------------ #
    engine_reads = _engine_reads(bundle, layers)
    oracle_base = _oracle_forward(shim, layers, base_inputs)
    oracle_cf = _oracle_forward(shim, layers, cf_inputs)
    for case in READS:
        layer = None if case.role is None else layers[case.role]
        read_id = _read_id(family, case, layer)
        have = engine_reads["base"][_read_id("family", case, layer)]
        want = _oracle_read(oracle_base, case)
        compare(read_id, _hook_name(shim, layers, case), have, want)
        values.update(_pin_values(read_id, have, None))

    # --- writes, and their no-op twins (T7) --------------------------------- #
    noop: dict[str, float] = {}
    for case in WRITES:
        layer = layers[case.role]
        write_id = _write_id(family, case, layer)
        have = _engine_write(bundle, layers, case, operand_input="counterfactual")
        want = _oracle_write(shim, layers, case, base_inputs, oracle_cf)
        compare(write_id, "lm_head", have["logits_all"], want.logits)
        values.update(_pin_values(write_id, have["after"], have["clean"]))
        for component, role in case.downstream:
            ds_id = f"{write_id}.downstream.{component}.L{layers[role]}"
            ds_have = have[f"ds/{component}.L{layers[role]}"]
            ds_want = _oracle_read(want, ReadCase(component, role))
            compare(
                ds_id, _hook_name(shim, layers, (component, role)), ds_have, ds_want
            )
            clean_ds = engine_reads["base"][
                _read_id("family", ReadCase(component, role), layers[role])
            ]
            values.update(_pin_values(ds_id, ds_have, clean_ds))
        twin = _engine_write(bundle, layers, case, operand_input="base")
        noop[write_id] = _max_abs(twin["after"], twin["clean"])

    # --- measurements the tolerance is justified by ------------------------- #
    chunked_vs_stepwise = _max_abs(
        _flat(oracle_base.call.out), _flat(oracle_base.recurrence.out)
    )
    state_id = _write_id(
        family, next(c for c in WRITES if c.component == "delta_state"), layers[DELTA]
    )
    measurements = {
        "chunked_vs_stepwise_kernel_output": chunked_vs_stepwise,
        "noop_swap_logit_drift": noop,
        "state_substitution_drift": noop[state_id],
    }

    model_revision, causalab_revision = _provenance(bundle)
    certification: dict[str, Any] = {
        "model_revision": model_revision,
        "causalab_revision": causalab_revision,
        "attn_implementation": bundle.model.config._attn_implementation,
        "dtype": bundle.dtype,
        "hook_names": sorted(hook_names),
        "tensor_shapes": {name: tensor_shapes[name] for name in sorted(tensor_shapes)},
        "max_activation_diff": {
            "per_hook": dict(activation_diffs),
            "max": max(activation_diffs.values()),
        },
        "max_logit_diff": {
            "per_case": dict(logit_diffs),
            "max": max(logit_diffs.values()),
        },
    }
    assert tuple(certification) == CERTIFICATION_FIELDS
    record: dict[str, Any] = {
        "attn_implementation": bundle.model.config._attn_implementation,
        "captured_from": "hook_oracle",
        "certification": certification,
        "context": _context(),
        "deterministic": True,
        "device": spec.device,
        "family": family,
        "layers": {"delta": layers[DELTA], "full": layers[FULL]},
        "measurements": measurements,
        "model": spec.key,
        "recapture": (
            "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python "
            f"tests/neural/parity/update_family_goldens.py --family {family}"
        ),
        "texts": {"base": BASE_TEXT, "counterfactual": COUNTERFACTUAL_TEXT},
        "tolerance": {
            "default": band.pins_abs,
            "relative": band.pins_rel,
            "certification": {"activation": band.activation, "logits": band.logits},
            "state_substitution_drift": band.state_substitution_drift,
            "justification": band.justification,
        },
        "values": values,
    }
    return Capture(
        record=record, activation_diffs=activation_diffs, logit_diffs=logit_diffs
    )


# --------------------------------------------------------------------------- #
# what a certification asserts
# --------------------------------------------------------------------------- #


def certification_failures(capture: Capture, band: Band) -> list[tuple[str, float]]:
    """Every comparison outside the band, in forward order — activations
    first (the intermediate components), then the logits."""
    failures = [
        (name, diff)
        for name, diff in capture.activation_diffs.items()
        if not diff <= band.activation
    ]
    failures += [
        (f"{case_id}@lm_head", diff)
        for case_id, diff in capture.logit_diffs.items()
        if not diff <= band.logits
    ]
    return failures


def check_certification_fields(
    record: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """``(missing, extra)`` of the record's ``certification`` key set against
    `CERTIFICATION_FIELDS` — the census guard's comparison (T5)."""
    block = record.get("certification")
    if not isinstance(block, Mapping):
        return list(CERTIFICATION_FIELDS), []
    have = set(block)
    want = set(CERTIFICATION_FIELDS)
    return sorted(want - have), sorted(have - want)


def compare_records(
    committed: Mapping[str, Any], fresh: Mapping[str, Any]
) -> list[str]:
    """How a fresh capture disagrees with the committed record: provenance
    that must be equal (model snapshot, attention implementation, dtype, hook
    names, tensor shapes, layers), measured differences that must stay inside
    the committed band, and pins that must replay within tolerance.

    The causalab revision is capture provenance, not a replay condition: a
    later commit replays the same record.
    """
    problems: list[str] = []
    c_cert, f_cert = committed["certification"], fresh["certification"]
    for field in ("attn_implementation", "dtype", "hook_names", "tensor_shapes"):
        if c_cert[field] != f_cert[field]:
            problems.append(
                f"certification.{field}: {f_cert[field]!r} != committed {c_cert[field]!r}"
            )
    if c_cert["model_revision"] != f_cert["model_revision"]:
        problems.append(
            f"certification.model_revision: {f_cert['model_revision']!r} != committed "
            f"{c_cert['model_revision']!r}"
        )
    for field in ("family", "model", "layers", "texts", "attn_implementation"):
        if committed[field] != fresh[field]:
            problems.append(
                f"{field}: {fresh[field]!r} != committed {committed[field]!r}"
            )
    band = committed["tolerance"]["certification"]
    for name, diff in f_cert["max_activation_diff"]["per_hook"].items():
        if not diff <= band["activation"]:
            problems.append(
                f"activation {name}: {diff!r} > band {band['activation']!r}"
            )
    for name, diff in f_cert["max_logit_diff"]["per_case"].items():
        if not diff <= band["logits"]:
            problems.append(f"logits {name}: {diff!r} > band {band['logits']!r}")
    drift = fresh["measurements"]["state_substitution_drift"]
    bound = committed["tolerance"]["state_substitution_drift"]
    if not 0.0 < drift <= bound:
        problems.append(f"state_substitution_drift {drift!r} not in (0, {bound!r}]")
    tol_abs = float(committed["tolerance"]["default"])
    tol_rel = float(committed["tolerance"].get("relative", 0.0))
    want, have = committed["values"], fresh["values"]
    if set(want) != set(have):
        problems.append(
            f"pinned keys diverge (missing from fresh: {sorted(set(want) - set(have))[:5]}, "
            f"unpinned: {sorted(set(have) - set(want))[:5]})"
        )
    for key in sorted(set(want) & set(have)):
        expected, got = want[key], have[key]
        if isinstance(expected, list):
            if list(got) != expected:
                problems.append(f"{key}: shape {got} != pinned {expected}")
            continue
        tol = tol_abs + tol_rel * abs(float(expected))
        if not abs(float(got) - float(expected)) <= tol:
            problems.append(f"{key}: {got!r} != pinned {expected!r} (tol {tol})")
    return problems


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #


def record_path(family: str) -> Path:
    return GOLDENS_DIR / f"{family}.json"


@functools.lru_cache(maxsize=None)
def load_record(family: str) -> dict[str, Any]:
    with record_path(family).open() as f:
        return json.load(f)


def render_record(record: Mapping[str, Any]) -> str:
    """The frozen goldens' serialization: ``indent=2, sort_keys=True`` and a
    trailing newline, so a no-op recapture is an empty diff."""
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def make_bundle(family: str, *, device: str | None = None) -> ModelBundle:
    """Load the family's realization through the engine's own loader."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model

    spec = FAMILIES[family]
    return load_model(spec.key, dtype=spec.dtype, device=device or spec.device)


#: for a test that wants to swap one oracle function (T6)
OracleFunction = Callable[..., Any]
