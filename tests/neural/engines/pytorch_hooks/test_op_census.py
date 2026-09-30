"""The engine's own op stream around a forward, counted at the dispatcher.

The model's kernels are the model's; what this engine adds around them — the
hook path, the read gather, the write landing — is ours to keep small,
because a solo or eval forward is launch-bound and every op we add is a
launch the GPU waits for. A ``TorchDispatchMode`` attributes each aten op to
the innermost frame that is either ours (``causalab/neural``) or the
library's, and this pins, on the tiny MoE and the locate/control shape (a
swap of a counterfactual read into ``block_output``, a logits read):

* the index tensors a position table gathers with are built once — a cold
  cache shows the construction, the second forward over the same table
  issues none from ``gather.py`` (and the cache's own counters agree);
* no sync (``SYNCS``: a ``.item()``, an explicit ``nonzero``, a
  boolean-mask select seen as ``index`` with a bool index) from the
  executor's read and write paths inside a model forward (the position
  frame's per-row ``first_real`` in ``encoding.py`` is budgeted with the
  frame, not here);
* the write landing gathers its positions once and copies nothing it does
  not have to: no ``clone`` from ``_written_value``, exactly one from the
  hook (the tensor the in-place edit works on);
* our kernel-launching glue per forward stays within a budget (the values
  are in the assertions; view ops and the engine's own grouped-experts
  forward, which replaces the library's, are not counted).

The pinned document is dense — one position per row, a ``swap`` on
``block_output`` — so the ragged landings (and the ``padded_masked``
landing's former boolean-mask select), the scalar operands of ``scale`` /
``clamp``, and a read through the ``expert:`` face (``_expert_selected``
selects its hits with a boolean mask and counts them per row; the width
there is data-dependent) are not under this census.

"Per forward" is per *phase*: ``Census.forward`` advances on the
model's forward pre-hook and never resets, so phase ``k`` runs from the
``k``-th forward's start to the next one's (the reads finalized after a
forward returns fall in its phase), and whatever runs before the first
forward is phase 0, which nothing here asserts on — the counts are
conservative, not leaky.
"""

from __future__ import annotations

import collections
import contextlib
import importlib
import os
import sys
from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
from causalab.neural.shared.gather import _dense_index
from causalab.protocol.schema import PROTOCOL_VERSION

from tests.protocol._docs import saved

from ._drive import base_data_section, executor_for
from .test_train import BASES, COUNTERFACTUALS

pytestmark = pytest.mark.smoke

OURS = "/causalab/neural/"
#: our grouped-experts forward replaces the library's — model work, not glue
EXEMPT = ("experts_path.py",)


def _library_roots() -> tuple[str, ...]:
    """Where the model's own code lives, read off the installed packages —
    not a `site-packages/` spelling, which an editable or vendored install
    would miss, attributing the model's kernels to the first of our frames
    above them (the executor's forward, on the stack for every op)."""
    roots = []
    for name in ("transformers", "fla"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if module.__file__:
            roots.append(os.path.dirname(module.__file__) + os.sep)
    return tuple(roots)


LIBRARY = _library_roots()
#: ops that stall the launch queue on a device: a scalar read back
#: (`.item()`), an explicit `nonzero`, and a boolean-mask select — which the
#: dispatcher shows only as `index` with a bool index (📐 probed: the
#: `nonzero` inside its kernel runs with the mode popped), so the census
#: names it `index[bool]` from the arguments. A host-visible `_to_copy`
#: (`.tolist()`, `.cpu()`) is the fourth family, left out because the same
#: op is also a plain dtype cast
SYNCS = frozenset({"_local_scalar_dense", "nonzero", "index[bool]"})


def _has_bool_index(indices: Any) -> bool:
    items = indices if isinstance(indices, (list, tuple)) else (indices,)
    return any(isinstance(t, torch.Tensor) and t.dtype == torch.bool for t in items)


#: ops that launch nothing: views, metadata, autograd plumbing
VIEWS = frozenset(
    {
        "slice", "select", "view", "_unsafe_view", "reshape", "unsqueeze",
        "squeeze", "permute", "transpose", "t", "expand", "alias", "detach",
        "as_strided", "empty", "empty_like", "empty_strided",
    }
)  # fmt: skip


class Census(TorchDispatchMode):
    """Per model forward: our kernel-launching ops by (site, op) and the
    model's op total."""

    def __init__(self, ours: str = OURS) -> None:
        super().__init__()
        self.ours_prefix = ours
        self.forward = 0
        self.ours: dict[int, collections.Counter[tuple[str, str]]] = (
            collections.defaultdict(collections.Counter)
        )
        self.model: collections.Counter[int] = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func.overloadpacket.__name__)
        if name == "index" and len(args) > 1 and _has_bool_index(args[1]):
            # a boolean-mask select: the `nonzero` its kernel issues runs
            # with this mode popped, so the mask is only visible here
            name = "index[bool]"
        frame = sys._getframe(1)
        while frame is not None:
            filename = frame.f_code.co_filename
            if self.ours_prefix in filename:
                if not filename.endswith(EXEMPT) and name not in VIEWS:
                    site = f"{filename.split('/causalab/')[-1]}:{frame.f_code.co_name}"
                    self.ours[self.forward][(site, name)] += 1
                break
            if any(part in filename for part in LIBRARY):
                self.model[self.forward] += 1
                break
            frame = frame.f_back
        return func(*args, **(kwargs or {}))


@contextlib.contextmanager
def _counting(bundle: ModelBundle):
    census = Census()

    def entered(_m: Any, _a: Any, _k: Any) -> None:
        census.forward += 1

    handle = bundle.model.register_forward_pre_hook(entered, with_kwargs=True)
    try:
        with census:
            yield census
    finally:
        handle.remove()


def _swap_doc(layer: int = 1) -> dict:
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "method": {
            "intervened_models": {
                "original_counterfactual": {
                    "input": "counterfactual",
                    "reads": ["v_cf"],
                },
                "patched": {"input": "base", "reads": ["logits"], "writes": ["patch"]},
            },
            "sites": {
                "tgt": {"component": "block_output", "layers": [layer]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "v_cf": {"site": "tgt", "pos": {"index": -1}},
                "logits": {"site": "lm_head", "pos": {"index": -1}},
            },
            "writes": {
                "patch": {"site": "tgt", "pos": {"index": -1}, "do": {"swap": "v_cf"}}
            },
            "save": [saved("logits", "patched", "l.safetensors")],
        },
    }


def _by_site(census: Census, forward: int, site_suffix: str) -> dict[str, int]:
    return {
        op: n
        for (site, op), n in census.ours[forward].items()
        if site.endswith(site_suffix)
    }


def _run(bundle: ModelBundle) -> Census:
    ex = executor_for(
        _swap_doc(), bundle, base_texts=BASES, counterfactual_texts=COUNTERFACTUALS
    )
    with _counting(bundle) as census:
        ex.dense_value("logits")
    assert census.forward == 2  # the counterfactual read's, then the patched
    # the attribution itself: a model forward that put no op under the
    # library's roots means LIBRARY missed the installed package, not that
    # our glue grew
    assert census.model[2] > 0, LIBRARY
    return census


def test_index_tensors_are_built_once_per_table(qwen35moe_bundle: ModelBundle) -> None:
    _dense_index.cache_clear()
    cold = _run(qwen35moe_bundle)
    # the attribution itself: a cold cache must *show* the construction
    # (`torch.tensor(list)` dispatches as `lift_fresh`), or the absence
    # below says nothing. On this fixture it shows in the first forward (the
    # counterfactual read resolves the same `[-1]` table the patched
    # forward's write and logits read then reuse), but that is the texts'
    # widths, not the cache's doing — either phase will do
    built_cold = {f: _by_site(cold, f, "gather.py:_dense_index") for f in (1, 2)}
    assert built_cold[1] or built_cold[2], built_cold
    assert sum(_by_site(cold, 2, "gather.py:gather_positions").values()) == 2
    misses = _dense_index.cache_info().misses
    warm = _run(qwen35moe_bundle)  # the same tables, the same device
    for forward in (1, 2):
        built = _by_site(warm, forward, "gather.py:_dense_index")
        assert not built, (forward, built)
    # and the dispatch-independent reading of the same fact
    info = _dense_index.cache_info()
    assert info.misses == misses and info.hits > 0, info


def test_the_census_names_the_syncs_it_looks_for() -> None:
    """The detector, not the engine: a boolean-mask select reaches the
    dispatcher as ``index`` with a bool index (its own ``nonzero`` runs with
    the mode popped), so the absence asserted below has to be of names this
    mode can actually produce — the probe, written down."""
    census = Census(ours=os.path.dirname(__file__) + os.sep)
    with census:
        t = torch.randn(4)
        _ = t[torch.tensor([True, False, True, False])]
        _ = t.sum().item()
    named = {op for (_site, op) in census.ours[0]}
    assert SYNCS & named == {"index[bool]", "_local_scalar_dense"}, (
        f"{named} — an extra 'nonzero' means torch now surfaces the re-entrant "
        "call inside index's kernel to modes, not that our code grew a sync"
    )


def test_our_code_reads_no_scalar_back_inside_a_forward(
    qwen35moe_bundle: ModelBundle,
) -> None:
    """A ``.item()`` (``_local_scalar_dense``) inside a model forward is a
    device sync on the launch queue; none comes from the executor's read and
    write paths. The position frame's per-row ``first_real`` (``encoding.py``,
    📐 52 per forward on this shape) is the frame's own cost, excluded here as
    it is from the budget."""
    census = _run(qwen35moe_bundle)
    for forward in (1, 2):
        syncs = {
            key: n
            for key, n in census.ours[forward].items()
            if key[1] in SYNCS and "encoding.py" not in key[0]
        }
        assert not syncs, (forward, syncs)


def test_the_landing_copies_only_what_the_hook_edits_in_place(
    qwen35moe_bundle: ModelBundle,
) -> None:
    census = _run(qwen35moe_bundle)
    patched = census.ours[2]
    assert not _by_site(census, 2, "writes.py:_written_value"), dict(patched)
    assert _by_site(census, 2, "executor.py:out_hook") == {"clone": 1}, dict(patched)
    assert _by_site(census, 2, "writes.py:_apply_writes_to_contract") == {
        "index_put_": 1
    }, dict(patched)


def test_the_glue_budget_of_the_interchange_forward(
    qwen35moe_bundle: ModelBundle,
) -> None:
    census = _run(qwen35moe_bundle)

    def ours(forward: int) -> dict[tuple[str, str], int]:
        # the frame's per-row `first_real` and the lazy encode of a role's
        # batch (encoding.py) are the position frame's own cost, budgeted
        # with it, not here
        return {
            key: n
            for key, n in census.ours[forward].items()
            if "encoding.py" not in key[0]
        }

    # 📐 measured: the read forward gathers once; the patched forward gathers
    # the write's positions, clones the hook's tensor, scatters back and
    # gathers the logits — four launches around 3.4 k of the model's
    assert sum(ours(1).values()) <= 4, ours(1)
    assert sum(ours(2).values()) <= 8, ours(2)
    assert census.model[2] > 100 * sum(ours(2).values())
