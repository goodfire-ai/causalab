"""Write-set transactionality and fire counts (spec §4 "Fires").

An intervened model's write set installs together, runs in one forward and
tears down together. A reading of the installation path found the set
**atomic by construction** — one batched build before the single
``ExitStack``, edits on cloned activations, publish after the forward, tables
after the whole point loop — so a member that fails
to resolve or mismatches its operand's shape refuses the point with no
table, no capture and no receipt recording a subset. That property only
*happened* to hold; the first half of this file pins it, so a refactor that
publishes before the forward returns fails a test rather than a campaign.

The same reading found the one real defect: fire counts were neither counted nor
compared. A write installed on a module the forward never calls — 📐 the
DeltaNet ``conv1d`` module, whose forward goes through a module-global
function (``test_sites_round4_deltanet.py::test_the_conv1d_module_never_fires``)
— fired zero times and the un-intervened forward was scored as an
intervention. The second half is that observable: every member's firings per
forward are counted (``neural/shared/fires.py``), compared with what its kind
declares (once; a ``delta_state`` write once per addressed step), refused
when they differ, and recorded in the run receipt under ``fires``.

The zero- and double-fire cases are constructed the way the measured one
arises: the module is still on the tree and still resolves, but the forward
reaches its math without going through ``Module.__call__`` (no hook fires),
or goes through it twice (a shared or looped block). Every refusal has its
valid-work twin beside it.
"""

from __future__ import annotations

import contextlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterator

import pytest
import torch

from causalab.cli import main, register_model_key
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared.fires import (
    FireTally,
    GroupFires,
    check_fires,
    group_label,
)
from causalab.protocol import RUN_RECORD_NAME, run_protocol
from causalab.protocol.rules.errors import ProtocolError
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.receipt import FIRES_KEY

from tests.neural.engines.pytorch_hooks._drive import base_data_section, executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA, TINY_QWEN35_MOE
from tests.neural.engines.pytorch_hooks.test_prefix_resume import (
    _campaign,
    _engine_bundle,
)
from tests.neural.engines.pytorch_hooks.test_train import (
    ANSWERS,
    BASES,
    COUNTERFACTUALS,
)
from tests.protocol._docs import UNWRITTEN, aggregation, saved
from tests.protocol._env import CORPUS_DIR, FIXTURES

REPO = Path(__file__).resolve().parents[4]

#: The patched model's readout every swap document tabulates as `ce.json`.
CE = aggregation("cross_entropy", target="label")

#: the two DeltaNet-fixture texts of one state interchange — the same length
#: in tokens, because a ``delta_state`` operand must cover exactly the write's
#: addressed steps (``executor.writes._state_operand``)
STATE_BASE = "the quick brown fox jumps"


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #


def _swap_doc(
    layers: list[int],
    *,
    component: str = "attention_output",
    operand_sites: dict[int, tuple[str, int]] | None = None,
    mismatch_last: bool = False,
    fixture_data: bool = False,
) -> dict[str, Any]:
    """One intervened model carrying one swap per layer in ``layers``, scored
    at lm_head — the band shape of ``test_band_patch_run.py``.

    ``operand_sites`` reads a member's operand somewhere else than its own
    address — ``{write layer: (component, layer)}`` — so a test that silences
    one module's hooks can keep the operand read on a module it did not touch
    and exercise the *write* side alone.

    ``mismatch_last`` makes the last member's operand a read of the MLP's
    hidden activation one block up: a ``(rows, 1, d_mlp)`` tensor swapped into
    a ``(rows, 1, d_model)`` slice, which the mechanism refuses as a shape
    mismatch inside the hook — after every earlier member has already fired.
    """
    operand_sites = operand_sites or {}
    sites: dict[str, Any] = {
        **{f"a{i}": {"component": component, "layers": [i]} for i in layers},
        **{
            f"src{i}": {"component": comp, "layers": [layer]}
            for i, (comp, layer) in operand_sites.items()
        },
        "lm_head": {"component": "lm_head"},
    }
    reads: dict[str, Any] = {
        **{
            f"v_a{i}": {
                "site": f"src{i}" if i in operand_sites else f"a{i}",
                "pos": {"index": -1},
            }
            for i in layers
        },
        "logits": {"site": "lm_head", "pos": {"index": -1}},
    }
    writes = {
        f"w{i}": {"site": f"a{i}", "pos": {"index": -1}, "do": {"swap": f"v_a{i}"}}
        for i in layers
    }
    if mismatch_last:
        last = layers[-1]
        # one block above the write's address: rule 21 wants the operand at
        # or above where it lands, and the MLP's hidden width is not d_model
        assert last >= 1
        sites["narrow"] = {"component": "mlp_activation", "layers": [last - 1]}
        reads["v_narrow"] = {"site": "narrow", "pos": {"index": -1}}
        del reads[f"v_a{last}"]
        writes[f"w{last}"]["do"] = {"swap": "v_narrow"}
    # every operand is read on the un-intervened counterfactual; the readout
    # on the patched base
    operands = [name for name in reads if name != "logits"]
    data = (
        {
            "base": {"dataset": "weekdays/data#train", "field": "input"},
            "counterfactual": {
                "dataset": "weekdays/data#train",
                "field": "counterfactual_inputs[0]",
            },
        }
        if fixture_data
        else base_data_section(with_counterfactual=True)
    )
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": TINY_LLAMA, "revision": "main", "dtype": "fp32"},
        "data": data,
        "method": {
            "intervened_models": {
                UNWRITTEN: {"input": "counterfactual", "reads": operands},
                "patched": {
                    "input": "base",
                    "reads": ["logits"],
                    "writes": [f"w{i}" for i in layers],
                },
            },
            "sites": sites,
            "reads": reads,
            "writes": writes,
            "save": [saved("logits", "patched", "ce.json", dict(CE))],
        },
    }


def _state_doc() -> dict[str, Any]:
    """A ``delta_state`` interchange over **every** step of one row: the state
    writer fires once per addressed step, so its declared count is the row's
    token count, not one."""
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": TINY_QWEN35_MOE, "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "method": {
            "intervened_models": {
                UNWRITTEN: {"input": "counterfactual", "reads": ["s_cf"]},
                "patched": {"input": "base", "reads": ["after"], "writes": ["patch"]},
            },
            "sites": {
                "state": {"component": "delta_state", "layers": [0]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "s_cf": {"site": "state", "pos": {"all": True}},
                "after": {"site": "lm_head", "pos": {"index": -1}},
            },
            "writes": {
                "patch": {"site": "state", "pos": {"all": True}, "do": {"swap": "s_cf"}}
            },
            "save": [saved("after", "patched", "after.safetensors")],
        },
    }


# --------------------------------------------------------------------------- #
# the two shapes a wrong fire count takes, constructed on the fixture
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _bypassed(module: torch.nn.Module) -> Iterator[None]:
    """For the duration: the module's math runs, its hooks never fire.

    The forward that owns ``module`` still calls it, but the call reaches
    ``forward`` without ``Module.__call__`` — exactly how the DeltaNet forward
    reaches ``conv1d``'s weights through a module-global function. The site
    still resolves to this very object, so a write installs on it as usual.
    """
    original = type(module)

    class Bypassed(original):  # type: ignore[misc, valid-type]
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

    Bypassed.__name__ = original.__name__
    module.__class__ = Bypassed
    try:
        yield
    finally:
        module.__class__ = original


@contextlib.contextmanager
def _called_twice(module: torch.nn.Module) -> Iterator[None]:
    """For the duration: every call of ``module`` runs it — hooks included —
    twice on the same input, the shape of a block a looped or weight-shared
    forward visits more than once."""
    original = type(module)

    class Twice(original):  # type: ignore[misc, valid-type]
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            torch.nn.Module.__call__(self, *args, **kwargs)
            return torch.nn.Module.__call__(self, *args, **kwargs)

    Twice.__name__ = original.__name__
    module.__class__ = Twice
    try:
        yield
    finally:
        module.__class__ = original


def _executor(raw: dict[str, Any], bundle: ModelBundle, **kwargs: Any) -> PointExecutor:
    return executor_for(
        raw,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
        **kwargs,
    )


def _clean_logits(bundle: ModelBundle) -> torch.Tensor:
    encoded = bundle.tokenizer(BASES, return_tensors="pt", padding=True)
    with torch.no_grad():
        return bundle.model(**encoded).logits.clone()


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    return _engine_bundle()


@pytest.fixture(scope="module")
def moe_bundle() -> ModelBundle:
    return load_model(TINY_QWEN35_MOE)


# --------------------------------------------------------------------------- #
# unit: the tally, torch-free
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestTally:
    def test_a_member_that_never_fired_is_refused_by_name(self) -> None:
        tally = FireTally()
        tally.declare(("w0", "w1"), 1)
        tally.fired(("w0",))
        with pytest.raises(ProtocolError) as err:
            check_fires("patched on base", tally)
        assert err.value.reason == "component_unavailable"
        assert err.value.code == "P4"
        text = str(err.value)
        assert "'w1'" in text and "fired 0 times" in text and "patched on base" in text
        assert "un-intervened forward" in text

    def test_a_member_fired_twice_is_refused_naming_the_count(self) -> None:
        tally = FireTally()
        tally.declare(("w0",), 1)
        tally.fired(("w0",))
        tally.fired(("w0",))
        with pytest.raises(
            ProtocolError, match="fired 2 times in one forward, not the 1"
        ):
            check_fires("patched on base", tally)

    def test_the_zero_member_is_named_first_when_several_are_off(self) -> None:
        tally = FireTally()
        tally.declare(("a", "b"), 1)
        tally.fired(("a",))
        tally.fired(("a",))
        with pytest.raises(
            ProtocolError, match=r"write 'b' .* fired 0 times.*'a' \(2 of 1\)"
        ):
            check_fires("g", tally)

    def test_the_declared_count_passes(self) -> None:
        tally = FireTally()
        tally.declare(("w0",), 1)
        tally.declare(("state",), 3)
        tally.fired(("w0",))
        for step in (2, 4, 5):
            tally.fired(("state",), step=step)
        check_fires("g", tally)  # no raise
        assert tally.counts == {"w0": 1, "state": 3}

    def test_the_group_record_is_layout_invariant(self) -> None:
        """Two windows of one group: a module-kind member records its
        per-forward count (1, not 2), a state write the distinct steps over
        both windows' rows (the union, not the sum)."""
        fires = GroupFires()
        first, second = FireTally(), FireTally()
        for tally, steps in ((first, (3, 4)), (second, (4, 5))):
            tally.declare(("w0",), 1)
            tally.fired(("w0",))
            tally.declare(("state",), len(steps))
            for step in steps:
                tally.fired(("state",), step=step)
            check_fires("g", tally)
            fires.fold(tally)
        assert fires.record() == {"w0": 1, "state": 3}

    def test_the_group_label_is_the_ledgers_edit_group(self) -> None:
        assert group_label("patched", "base") == "patched on base"


# --------------------------------------------------------------------------- #
# unit: the executor counts, refuses and records
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestFireCounts:
    def test_two_members_at_two_layers_record_one_fire_each(self, bundle) -> None:
        """A two-member set at two layers records ``{w0: 1, w1: 1}``
        — one firing per member per forward, whatever address each is at.
        The un-intervened group carries no writes and so no record."""
        executor = _executor(_swap_doc([0, 1]), bundle)
        executor.run_all()
        assert executor.fires == {("patched", "base"): {"w0": 1, "w1": 1}}

    def test_two_members_at_one_address_each_count_the_shared_hooks_firing(
        self, bundle
    ) -> None:
        """Two writes at one address are one hook with two entries
        (``_resolve_write_addresses``); the hook fires once and both members
        record that firing — counting addresses would record one member."""
        raw = _swap_doc([1])
        raw["method"]["writes"]["w1b"] = {
            "site": "a1",
            "pos": {"index": -1},
            "do": {"add_scaled": {"op": "v_a1", "alpha": 0.5}},
        }
        raw["method"]["intervened_models"]["patched"]["writes"].append("w1b")
        executor = _executor(raw, bundle)
        executor.run_all()
        assert executor.fires[("patched", "base")] == {"w1": 1, "w1b": 1}

    @pytest.mark.parametrize("batch_rows", (None, 1))
    def test_the_record_does_not_depend_on_the_row_layout(
        self, bundle, batch_rows: int | None
    ) -> None:
        """Whole or one row per forward, the record is the per-forward count:
        every window is checked at the same declared count, and the receipt
        may differ between layouts at ``execution.batch_rows`` alone (§8)."""
        executor = _executor(_swap_doc([0, 1]), bundle, batch_rows=batch_rows)
        executor.run_all()
        assert executor.fires == {("patched", "base"): {"w0": 1, "w1": 1}}

    def test_a_served_group_records_the_counts_of_the_pass_that_ran(
        self, bundle
    ) -> None:
        """Two points sharing a digest — the second is
        served the first's captures (§3) and records the first's counts, not
        zeroes for a forward it never ran."""
        docs, handles, cache = _campaign([_swap_doc([0, 1]), _swap_doc([0, 1])])
        first = _executor(_swap_doc([0, 1]), bundle, interning=handles[0])
        first.run_all()
        served_before = len(cache.executed)
        second = _executor(_swap_doc([0, 1]), bundle, interning=handles[1])
        second.run_all()
        assert len(cache.executed) == served_before, "the second point ran a pass"
        key = handles[1].keys[("patched", "base")]
        assert cache.fires[key] == {"w0": 1, "w1": 1}
        assert second.fires == first.fires == {("patched", "base"): {"w0": 1, "w1": 1}}

    def test_a_member_whose_module_the_forward_never_calls_is_refused(
        self, bundle
    ) -> None:
        """T18, the zero-fire case (the measured ``conv1d`` shape): a write on
        a module the forward reaches without ``Module.__call__`` fires zero
        times. Refused by name with ``component_unavailable`` — not scored —
        and nothing of the group is recorded or marked as run."""
        raw = _swap_doc(
            [1], component="mlp_output", operand_sites={1: ("mlp_output", 0)}
        )
        executor = _executor(raw, bundle)
        with _bypassed(bundle.blocks[1].mlp):
            with pytest.raises(ProtocolError) as err:
                executor.run_all()
        assert err.value.reason == "component_unavailable"
        assert "write 'w1' in forward group 'patched on base' fired 0 times" in str(
            err.value
        )
        assert executor.fires == {}
        assert ("patched", "base") not in executor._groups_run
        # the twin: the same document with the module called as usual
        twin = _executor(raw, bundle)
        twin.run_all()
        assert twin.fires == {("patched", "base"): {"w1": 1}}

    def test_a_member_the_forward_calls_twice_is_refused(self, bundle) -> None:
        """T18's other side: a module the forward visits twice fires the write
        twice — a tensor the document did not name would be written. Refused
        naming the count; the same document on the ordinary tree runs."""
        raw = _swap_doc(
            [1], component="mlp_output", operand_sites={1: ("mlp_output", 0)}
        )
        executor = _executor(raw, bundle)
        with _called_twice(bundle.blocks[1].mlp):
            with pytest.raises(ProtocolError) as err:
                executor.run_all()
        assert err.value.reason == "component_unavailable"
        assert "fired 2 times in one forward, not the 1" in str(err.value)
        assert executor.fires == {}

    def test_a_state_write_over_several_steps_fires_once_per_step(
        self, moe_bundle
    ) -> None:
        """The declared count per kind: a ``delta_state`` write over
        every step of the row fires once per addressed step and passes with
        that count — a hardcoded "exactly once" would refuse this legitimate
        campaign."""
        steps = len(moe_bundle.tokenizer(STATE_BASE)["input_ids"])
        assert steps >= 3
        executor = executor_for(
            _state_doc(),
            moe_bundle,
            base_texts=[STATE_BASE],
            counterfactual_texts=[STATE_BASE],
        )
        executor.run_all()
        assert executor.fires == {("patched", "base"): {"patch": steps}}

    def test_a_mismatched_second_member_leaves_nothing_behind(self, bundle) -> None:
        """T17: the second member is a ``swap`` whose
        operand's shape mismatches its slice, refused *inside the hook* —
        after the first member has already edited layer 0's activation. Yet
        nothing partial is observable: no capture for the group's digest, no
        pass tallied, no fire record, the group not marked as run, and the
        model's own forward bit-identical to what it produced before.

        *Mutation this must catch:* ``_publish`` (or the tally's record)
        moved before the forward returns — the digest would then have a
        capture, or ``executed`` an entry, for a pass that never completed.
        """
        raw = _swap_doc([0, 1], mismatch_last=True)
        before = _clean_logits(bundle)
        docs, handles, cache = _campaign([raw])
        executor = _executor(raw, bundle, interning=handles[0])
        with pytest.raises(ProtocolError, match="does not broadcast"):
            executor.run_all()
        key = handles[0].keys[("patched", "base")]
        assert not any(
            stored == key or (isinstance(stored, tuple) and stored[0] == key)
            for stored in cache.captured
        ), "a capture was published for a pass that did not complete"
        # `executed` holds the group key of a keyed pass (`_publish`), the
        # model/input label of an unkeyed one — neither may be tallied
        assert key not in cache.executed
        assert "patched/base" not in cache.executed
        assert key not in cache.fires
        assert executor.fires == {}
        assert ("patched", "base") not in executor._groups_run
        # the operand reads on `original` did run — the failure was the
        # write set's, and only the write set's
        assert ("original_counterfactual", "counterfactual") in executor._groups_run
        assert torch.equal(_clean_logits(bundle), before)


# --------------------------------------------------------------------------- #
# smoke: the run receipt
# --------------------------------------------------------------------------- #


def _env(tmp_path: Path) -> ResolutionEnv:
    artifacts = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    register_model_key({"model": {"key": TINY_LLAMA, "revision": "main"}})
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts),
    )


def _receipt(out: Path) -> dict[str, Any]:
    return json.loads((out / RUN_RECORD_NAME).read_text())


@pytest.mark.smoke
class TestReceipt:
    def test_the_receipt_records_every_members_count_beside_execution(
        self, tmp_path: Path
    ) -> None:
        """The ``fires`` block: per point digest, per forward group, each write
        member's count — beside the ``execution`` block, which stays exactly
        the two execution parameters it is declared to hold."""
        out = tmp_path / "run"
        result = run_protocol(
            _swap_doc([0, 1], fixture_data=True),
            _env(tmp_path),
            PytorchHooksEngine(),
            out,
            record=True,
        )
        receipt = _receipt(out)
        (point,) = receipt["points"]
        assert receipt[FIRES_KEY] == {
            point["digest"]: {"patched on base": {"w0": 1, "w1": 1}}
        }
        assert receipt["execution"] == {
            "batch_rows": None,
            "device": "cpu",
            "fit_rows": None,
            "model_source": "loaded",
            "parallel": {
                "data": 1,
                "data_mode": "points",
                "pipeline": 1,
                "context": 1,
                "tensor": 1,
                "expert": 1,
                "world": 1,
                "launcher": "solo",
            },
        }
        (summary,) = result.summaries
        assert summary["fires"] == {"patched on base": {"w0": 1, "w1": 1}}
        assert (out / "ce.json").is_file()

    def test_without_a_receipt_the_counts_stay_in_the_result(
        self, tmp_path: Path
    ) -> None:
        """The default run: the counts are tallied and checked as before, and
        come back in the result's summaries, but nothing beside the table is
        written — no receipt, so no ``fires`` block on disk."""
        out = tmp_path / "run"
        result = run_protocol(
            _swap_doc([0, 1], fixture_data=True),
            _env(tmp_path),
            PytorchHooksEngine(),
            out,
        )
        (summary,) = result.summaries
        assert summary["fires"] == {"patched on base": {"w0": 1, "w1": 1}}
        assert (out / "ce.json").is_file()
        assert not (out / RUN_RECORD_NAME).exists()

    def test_a_zero_fire_member_refuses_the_run_with_no_table_and_no_counts(
        self, tmp_path: Path
    ) -> None:
        """T18 through the receipt: the run is refused, no table reaches disk,
        and the receipt — written before execution — carries no ``fires``
        block rather than a subset."""
        out = tmp_path / "run"
        raw = _swap_doc(
            [1],
            component="mlp_output",
            operand_sites={1: ("mlp_output", 0)},
            fixture_data=True,
        )
        with _bypassed(_engine_bundle().blocks[1].mlp):
            with pytest.raises(ProtocolError, match="fired 0 times"):
                run_protocol(
                    raw, _env(tmp_path), PytorchHooksEngine(), out, record=True
                )
        assert not (out / "ce.json").exists()
        assert FIRES_KEY not in _receipt(out)
        assert "execution" in _receipt(out)  # what the run *was* still stands

    def test_a_mismatched_member_refuses_the_run_with_no_table_and_no_counts(
        self, tmp_path: Path
    ) -> None:
        """T17 through the receipt: the mid-set shape refusal writes no table
        and records no fires — the receipt says what the run was and nothing
        about a partial write."""
        out = tmp_path / "run"
        raw = _swap_doc([0, 1], mismatch_last=True, fixture_data=True)
        with pytest.raises(ProtocolError, match="does not broadcast"):
            run_protocol(raw, _env(tmp_path), PytorchHooksEngine(), out, record=True)
        assert not (out / "ce.json").exists()
        assert FIRES_KEY not in _receipt(out)


# --------------------------------------------------------------------------- #
# smoke: the existing multi-write campaigns (T19)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def roots(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    from tests.protocol._env import write_rot_fixture

    artifacts = tmp_path_factory.mktemp("artifacts")
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    write_rot_fixture(artifacts)
    return FIXTURES / "data", artifacts


#: every corpus document with more than one write that runs on the CPU tier,
#: with the overrides ``test_run_corpus.py`` retargets it with
MULTI_WRITE_CORPUS: dict[str, tuple[str, ...]] = {
    "03_path_patching_im.json": (
        "sites.sender.layers=0",
        "sites.sender.head=1",
        "sites.receiver.layers=1",
        "sites.a10.layers=0",
        "sites.a11.layers=1",
    ),
    "06_hydra_effect_im.json": (
        "sites.abl.layers=0",
        "sites.probe14.layers=0",
        "sites.probe20.layers=1",
        "sites.resid_final.layers=1",
    ),
    "15_circuit_edges_im.json": (
        "sites.nm_9_6.layers=0",
        "sites.nm_9_6.head=1",
        "sites.nm_9_9.layers=0",
        "sites.nm_9_9.head=2",
        "sites.nm_10_0.layers=0",
        "sites.nm_10_0.head=0",
        "sites.ctl_10_7.layers=0",
        "sites.ctl_10_7.head=3",
        "sites.recv.layers=1",
    ),
}


@pytest.mark.smoke
@pytest.mark.parametrize("name", sorted(MULTI_WRITE_CORPUS), ids=lambda n: n[:2])
def test_every_multi_write_corpus_document_runs_with_its_counts_recorded(
    name: str, roots: tuple[Path, Path], tmp_path: Path
) -> None:
    """T19: the repo's multi-write documents still run, and the receipt
    records, per point and intervened model, exactly the members that model
    names — each fired once per forward. The documents' pins are untouched
    (``tests/protocol/test_corpus.py``): a fire count is a receipt fact."""
    data_root, artifacts_root = roots
    argv = [
        "run",
        "--engine",
        "auto",
        str(CORPUS_DIR / name),
        "--data-root",
        str(data_root),
        "--artifacts-root",
        str(artifacts_root),
        "--out",
        str(tmp_path),
        "--set",
        f"model.key={TINY_LLAMA}",
        "--set",
        "model.dtype=fp32",
        "--record",
    ]
    for item in MULTI_WRITE_CORPUS[name]:
        argv += ["--set", item]
    assert main(argv) == 0
    receipt = _receipt(tmp_path)
    raw = json.loads((CORPUS_DIR / name).read_text())
    models = raw["method"]["intervened_models"]
    assert sum(len(m.get("writes", [])) for m in models.values()) > 1
    fires = receipt[FIRES_KEY]
    assert set(fires) == {point["digest"] for point in receipt["points"]}
    # the fires block covers the groups that land a write; an un-intervened
    # model's group has no member to count
    written = {m: spec for m, spec in models.items() if spec.get("writes")}
    for groups in fires.values():
        assert set(groups) == {
            group_label(model, spec["input"]) for model, spec in written.items()
        }
        for model, spec in written.items():
            assert groups[group_label(model, spec["input"])] == {
                write: 1 for write in spec["writes"]
            }
