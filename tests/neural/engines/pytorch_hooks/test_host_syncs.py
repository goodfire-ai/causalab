"""Host reads on the hot path are O(1) per forward — never O(rows × layers).

A ``.item()``, ``.tolist()``, ``int(t)``, ``bool(t)`` or ``.cpu()`` on a device
tensor is a device→host synchronization: the CPU stops issuing kernels until
the GPU has drained. The cohort forward used to make about 270 of them per
call on the A3B DAS step — one ``argmax(mask[row]).item()`` per row per
position resolution, for every read and write at every layer — and spent 5.6 s
of wall on 2.1 s of GPU work while the backward, which makes none, ran at the
GPU's pace. Every one of those reads is a Python-level call on ``torch.Tensor``,
so this suite counts them on the CPU, where the same code runs the same calls,
and pins two invariants:

* a forward's reads do not grow with the rows of the batch or the layers a
  band spans — with grad enabled there are none at all past the encode;
* an optimizer step's reads do not grow with the rows of its minibatch.

What is left on the step is named in `test_a_step_pays_a_fixed_number_of_host_reads`.
"""

from __future__ import annotations

import contextlib
from collections import Counter
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.executor import (
    PROMPT_MASK_TYPES,
    PointExecutor,
    prompt_masks,
)
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.engines.pytorch_hooks.train import (
    run_cohort_training,
    run_training,
)
from causalab.neural.shared.encoding import encode
from causalab.protocol.rules.errors import ProtocolError

from tests.neural.engines.pytorch_hooks._drive import executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA, TINY_QWEN35_MOE
from tests.neural.engines.pytorch_hooks.test_featurizer_groups import (
    MOE_LAYER,
    _expert_dbm_doc,
)
from tests.neural.engines.pytorch_hooks.test_fit_cohort import (
    _campaign,
    _request,
    _train_doc,
)
from tests.neural.engines.pytorch_hooks.test_train import (
    ANSWERS,
    BASES,
    COUNTERFACTUALS,
    das_doc,
)
from tests.protocol._docs import saved, term

pytestmark = pytest.mark.unit

#: every ``torch.Tensor`` method that moves a value to the host
#: (``__format__`` on a scalar goes through ``item``; ``numpy(force=True)``
#: and ``np.asarray`` through ``__array__``)
HOST_READS = (
    "item",
    "tolist",
    "__int__",
    "__float__",
    "__bool__",
    "__index__",
    "nonzero",
    "cpu",
    "numpy",
    "__array__",
    "to",
)


def _moves_to_host(tensor: torch.Tensor, args: tuple[Any, ...], kwargs: Any) -> bool:
    """Whether ``tensor.to(*args, **kwargs)`` copies a device tensor onto the
    host — the spelling of ``.cpu()`` the counter would otherwise miss. A
    ``to`` that stays on its device (a dtype cast, a device→device move, the
    no-op ``to(t.device)``) is not a host read; on the CPU-only suite the
    wrapper therefore counts nothing — every tensor is already on the host,
    so a ``to("cpu")`` regression is caught only when this suite runs on an
    accelerator — and there it counts the same call ``.cpu()`` counts."""
    if tensor.device.type == "cpu":
        return False
    target = kwargs.get("device")
    if target is None:
        for arg in args:
            if isinstance(arg, torch.Tensor):
                target = arg.device
                break
            if isinstance(arg, (str, torch.device)):
                target = arg
                break
    return target is not None and torch.device(target).type == "cpu"


ROWS = BASES + [
    "tiny paper boats float slowly",
    "old copper bells ring softly",
    "three hungry goats climb upward",
    "new glass towers gleam brightly",
]
CFS = COUNTERFACTUALS + [
    "blue frozen lakes shine dimly",
    "sharp wooden arrows fly straight",
    "eight sleepy owls blink twice",
    "dark stone bridges span widely",
]
LABELS = ANSWERS + [" five", " six", " seven", " eight"]


@contextlib.contextmanager
def counting_host_reads() -> Iterator[Counter[str]]:
    """Count every host read made on any tensor while the block runs. The
    methods are wrapped on ``torch.Tensor`` itself — a dunder set on the
    class updates its C slot, so ``int(t)`` and ``if t:`` are counted too —
    and restored afterwards."""
    counts: Counter[str] = Counter()
    saved: dict[str, tuple[Any, bool]] = {}
    for name in HOST_READS:
        original = getattr(torch.Tensor, name)
        saved[name] = (original, name in torch.Tensor.__dict__)

        def wrapped(
            self: torch.Tensor,
            *args: Any,
            _name: str = name,
            _original: Any = original,
            **kwargs: Any,
        ) -> Any:
            if _name != "to" or _moves_to_host(self, args, kwargs):
                counts[_name] += 1
            return _original(self, *args, **kwargs)

        setattr(torch.Tensor, name, wrapped)
    try:
        yield counts
    finally:
        for name, (original, own) in saved.items():
            if own:
                setattr(torch.Tensor, name, original)
            else:
                delattr(torch.Tensor, name)


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    return load_model(TINY_LLAMA)


def _interchange(layers: Sequence[int]) -> dict[str, Any]:
    """The DAS document as a plain interchange over a band: no featurizer
    (a featurizer on a band is refused), no training."""
    raw = das_doc()
    method = raw["method"]
    method["sites"]["tgt"]["layers"] = list(layers)
    method.pop("train")
    method.pop("featurizers")
    method["reads"]["v_cf"].pop("featurizer")
    method["writes"]["patch"].pop("featurizer")
    method["save"] = [method["save"][0]]
    return raw


def _executor(
    raw: dict[str, Any], bundle: ModelBundle, *, rows: int, **kwargs: Any
) -> PointExecutor:
    return executor_for(
        raw,
        bundle,
        base_texts=ROWS[:rows],
        counterfactual_texts=CFS[:rows],
        extra_columns={"label": LABELS[:rows]},
        **kwargs,
    )


def _forward_reads(
    bundle: ModelBundle, *, rows: int, layers: Sequence[int], grad: bool
) -> Counter[str]:
    """Host reads of one intervened forward with its reads finalized, the
    roles encoded beforehand (an encode reads the tokenizer's host tensors,
    once per role per point — not the forward's concern)."""
    executor = _executor(_interchange(layers), bundle, rows=rows, grad_enabled=grad)
    executor._batch("base")
    executor._batch("counterfactual")
    with counting_host_reads() as counts:
        executor.run_all()
        executor.dense_value("logits")
    return counts


def _fit_reads(
    bundle: ModelBundle, *, rows: int, pairs: int, layers: Sequence[int]
) -> Counter[str]:
    """Host reads of a whole one-epoch cohort fit — one DAS member per
    layer, ``rows`` training rows in minibatches of ``pairs``, no eval."""
    raws = [
        _train_doc("das", layer=layer, epochs=1, eval_every=None) for layer in layers
    ]
    for raw in raws:
        raw["method"]["train"]["batch"] = {"pairs": pairs}
    _docs, handles = _campaign(raws)
    executors = [
        _executor(raw, bundle, rows=rows, interning=handle)
        for raw, handle in zip(raws, handles)
    ]
    with counting_host_reads() as counts:
        run_cohort_training([ex.doc for ex in executors], executors, _request())
    return counts


class TestForwardHostReads:
    @pytest.mark.parametrize("layers", [(0,), (0, 1)])
    def test_a_grad_forward_reads_nothing_back_from_the_device(
        self, bundle: ModelBundle, layers: tuple[int, ...]
    ) -> None:
        """Position resolution for every read and write at every layer of
        the band is host arithmetic on the frame's cached first-real
        indices; the training forward makes no round trip at all."""
        for rows in (2, 4, 8):
            assert _forward_reads(bundle, rows=rows, layers=layers, grad=True) == {}

    def test_an_eval_forward_reads_back_each_finalized_value_once(
        self, bundle: ModelBundle
    ) -> None:
        """Without grad a finalized read moves its value to the host — one
        copy per read (``v_cf`` per band member, plus ``logits``), the same
        for 2, 4 and 8 rows, and nothing else."""
        for layers in ((0,), (0, 1)):
            expected = Counter({"cpu": len(layers) + 1})
            for rows in (2, 4, 8):
                assert _forward_reads(bundle, rows=rows, layers=layers, grad=False) == (
                    expected
                )


class TestStepHostReads:
    def test_a_steps_host_reads_do_not_grow_with_its_rows(
        self, bundle: ModelBundle
    ) -> None:
        """Four training rows in minibatches of one pair and eight rows in
        minibatches of two are the same four updates per member; every host
        read of the two fits agrees, so no read is paid per row of a
        minibatch — the cohort forward, the loss and the update included."""
        for layers in ((0,), (0, 1)):
            small = _fit_reads(bundle, rows=4, pairs=1, layers=layers)
            large = _fit_reads(bundle, rows=8, pairs=2, layers=layers)
            assert small == large, (layers, small, large)

    def test_a_step_pays_a_fixed_number_of_host_reads(
        self, bundle: ModelBundle
    ) -> None:
        """What a two-member, four-update fit still reads back, by kind:

        * ``item`` — the optimizer's own step counter (``torch.optim.Adam``
          with ``capturable=False`` keeps ``step`` on the host and reads it
          once per member per update); not a device synchronization;
        * ``tolist`` / ``__float__`` — the encode (once per role per member),
          the per-epoch ``randperm`` order, the subspace's orthonormality
          diagnostic at init and at the end, and — on the CPU only — the
          frame constructor's check of a carried first-real cache against
          its mask (per minibatch selection and per cohort concatenation);
        * nothing else: no ``int(t)``, ``bool(t)``, ``nonzero`` or ``.cpu()``
          on the step, and the loss record is not read until a checkpoint
          asks for it.
        """
        members, updates = 2, 4
        counts = _fit_reads(bundle, rows=8, pairs=2, layers=(0, 1))
        assert counts["item"] <= members * updates
        for kind in ("__int__", "__bool__", "__index__", "nonzero", "cpu", "numpy"):
            assert counts[kind] == 0, (kind, counts)


# --------------------------------------------------------------------------- #
# the hybrid family: a DeltaNet model's forward, and the DBM step on it
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def moe() -> ModelBundle:
    return load_model(TINY_QWEN35_MOE)


def _dbm_doc(pairs: int, *, control: bool) -> dict[str, Any]:
    """The expert-neuron DBM fit on the tiny MoE: the fixture's document
    (a write through an expert-keyed gate at the routed interior and a plain
    gate on the shared expert) with an ``lm_head`` read, a cross-entropy
    objective and an ``l1`` term — and, with ``control``, a PID moving the
    sparsity weight off the two gates' kept-unit counts (§2.11)."""
    raw = _expert_dbm_doc()
    method = raw["method"]
    method["sites"]["lm_head"] = {"component": "lm_head"}
    method["reads"]["logits"] = {"site": "lm_head", "pos": -1}
    method["intervened_models"]["masked"]["reads"].append("logits")
    ce = {"kind": "cross_entropy", "target": "label"}
    method["train"] = {
        "objective": {
            "fit": term("logits", "masked", dict(ce), weight=1.0),
            "sparsity": {"weight": 0.01, "l1": ["routed_gate", "shared_gate"]},
        },
        "params": ["routed_gate", "shared_gate"],
        "optimizer": {"name": "adamw", "lr": 0.01, "weight_decay": 0.0},
        "steps": {"epochs": 1},
        "batch": {"pairs": pairs},
        "seed": 0,
    }
    if control:
        method["train"]["control"] = {
            "train.objective.sparsity.weight": {
                "kind": "pid",
                "signal": {"hard_mask_size": ["routed_gate", "shared_gate"]},
                "setpoint": {"ramp": [16, 0, 1.0]},
                "gains": {"kp": 0.5, "ki": 0.05},
            }
        }
    method["save"] += [
        saved("logits", "masked", "ce.json", dict(ce)),
        {"value": "routed_gate", "site": "routed", "file_path": "routed.safetensors"},
        {"value": "shared_gate", "site": "shared", "file_path": "shared.safetensors"},
    ]
    return raw


def _dbm_reads(
    moe: ModelBundle, *, rows: int, pairs: int, control: bool
) -> tuple[Counter[str], PointExecutor]:
    executor = _executor(_dbm_doc(pairs, control=control), moe, rows=rows)
    with counting_host_reads() as counts:
        run_training(executor.doc, executor, _request())
    return counts, executor


class TestHybridForwardHostReads:
    def test_a_deltanet_forward_decides_its_padding_mask_without_a_round_trip(
        self, moe: ModelBundle
    ) -> None:
        """transformers builds the DeltaNet layers' padding mask per forward
        and asks the device whether the batch is padded at all (``torch.all``
        over the mask, a ``bool(tensor)``); the eager executor hands the
        forward prebuilt masks instead (``prompt_masks``), so a hybrid
        family's grad forward reads nothing back either."""
        for rows in (2, 4):
            assert _forward_reads(moe, rows=rows, layers=(0, 1), grad=True) == {}


class TestPromptMasks:
    """The prebuilt mapping is keyed by the layer types the config declares,
    over the closed [`PROMPT_MASK_TYPES`][causalab.neural.engines.pytorch_hooks.executor.PROMPT_MASK_TYPES]; a family declaring a type the
    helper has no mask for is refused by name before any mask is built,
    never handed a mapping its block loop would ``KeyError`` on."""

    @staticmethod
    def _inputs(moe: ModelBundle) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = encode(
            moe.tokenizer, ["one two three", "four"], device=str(moe.devices.embedding)
        )
        return batch.input_ids, batch.attention_mask, batch.position_ids()

    def test_the_mapping_covers_exactly_the_declared_layer_types(
        self, moe: ModelBundle
    ) -> None:
        ids, mask, positions = self._inputs(moe)
        masks = prompt_masks(moe.model, ids, mask, positions)
        assert set(masks) == set(moe.model.config.layer_types) == PROMPT_MASK_TYPES
        # the padding entry is the caller's mask itself (prepare_batch
        # restages it in place), the attention entry the 4-D causal mask
        assert masks["linear_attention"].data_ptr() == mask.data_ptr()
        assert masks["full_attention"] is not mask

    def test_an_undeclared_layer_type_is_refused_before_any_mask_is_built(
        self, moe: ModelBundle
    ) -> None:
        ids, mask, positions = self._inputs(moe)
        config = SimpleNamespace(
            model_type="hybrid_with_window",
            layer_types=["linear_attention", "sliding_attention", "full_attention"],
        )
        model = SimpleNamespace(config=config)  # no embeddings: never reached
        with pytest.raises(ProtocolError, match=r"P4.*sliding_attention"):
            prompt_masks(model, ids, mask, positions)


class TestDbmStepHostReads:
    def test_a_dbm_steps_host_reads_do_not_grow_with_its_rows(
        self, moe: ModelBundle
    ) -> None:
        small, _ = _dbm_reads(moe, rows=4, pairs=1, control=True)
        large, _ = _dbm_reads(moe, rows=8, pairs=2, control=True)
        assert small == large, (small, large)

    def test_what_a_dbm_step_still_reads_back(self, moe: ModelBundle) -> None:
        """Four updates of one member on the tiny MoE, by kind:

        * the write through the expert-keyed gate leaves its per-example
          mismatch counts on the device — no read per layer per forward; the
          point's ``routing_mismatch`` brings them over when asked;
        * a controller's signal is one ``tolist`` per update for the whole
          step (every gate of every controller of every member), not one
          ``float(hard.sum())`` per gate;
        * ``item`` is the optimizer's host-side step counter, one per
          parameter group per update;
        * no ``bool(tensor)``: the hybrid forward's padding-mask decision is
          made from the prebuilt masks.
        """
        updates, param_groups = 4, 2
        with_control, executor = _dbm_reads(moe, rows=4, pairs=1, control=True)
        without, _ = _dbm_reads(moe, rows=4, pairs=1, control=False)
        assert with_control["__bool__"] == 0 and without["__bool__"] == 0
        assert with_control["__float__"] == without["__float__"]  # diagnostics only
        assert with_control["item"] <= updates * param_groups
        assert with_control["tolist"] - without["tolist"] == updates
        for kind in ("__int__", "__index__", "nonzero", "cpu", "numpy"):
            assert with_control[kind] == 0, (kind, with_control)
        # the deferred record is served on demand, in one read
        with counting_host_reads() as counts:
            executor.run_all()
            mismatch = executor.routing_mismatch
        assert mismatch and set(mismatch) == {
            ("mask_routed", MOE_LAYER, i) for i in range(4)
        }
        assert counts["tolist"] == 1
