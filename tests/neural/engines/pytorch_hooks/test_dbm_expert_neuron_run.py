"""MoE-layer neuron DBM end to end: ``dbm_expert_neuron.json`` fits an
expert-keyed gate on the routed interior beside a plain gate on the shared
expert, ``dbm_expert_neuron_apply.json`` replays both (spec §2.5 ``group:
expert_neuron``).

Both presets run on the tiny MoE fixture (128 experts, top-10, ``d_expert``
32, shared inner 32) through the real CLI, as ``test_dbm_head_run.py`` does
for the head-grouped gate. What is checked: the routed bundle holds the whole
``(experts, d_expert)`` table and its header says so, the shared bundle is a
plain per-neuron ``theta`` with separate keys, ``fit_diagnostics.json`` counts
expert-neuron units, ``routing_mismatch.json`` is written per (write, layer,
example) and agrees with a hand computation from saved ``expert_idx`` reads
of both inputs, and the apply reproduces the fit's number exactly — so the
reloaded table is the fit's hard mask over the same experts.

The offline half: both presets validate against the registered Qwen3.6 entry
with no model loaded, the fit's canonical form has ``params.theta`` as
``[256, 512]`` on the routed gate, and the group is refused by name on every
component except ``expert_activation`` and ``expert_neuron_output``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from causalab.cli import main
from causalab.neural.engines.pytorch_hooks.train import _regularizer, fit_diagnostics
from causalab.neural.shared.featurizers import Gate
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.pipeline import compile_protocol
from causalab.io.env import build_artifact_identity, read_safetensors_metadata

from tests.neural.engines.pytorch_hooks.conftest import TINY_QWEN35_MOE
from tests.protocol._env import FIXTURES, build_env
from tests.tables import frame as table_frame
from tests._helpers.paths import PROTOCOLS_DIR


REPO = Path(__file__).resolve().parents[4]
PROTOCOLS = PROTOCOLS_DIR

TINY = {"model.key": TINY_QWEN35_MOE, "model.dtype": "fp32"}
#: 📐 fixture numbers: 128 experts, top-10, d_expert 32, shared inner 32
EXPERTS, TOP_K, D_EXPERT, D_SHARED = 128, 10, 32, 32
LAYER = 0


# --------------------------------------------------------------------------- #
# offline: the presets validate, the table is in the canonical form
# --------------------------------------------------------------------------- #


def _stamped_gate(
    root: Path,
    rel: str,
    theta: torch.Tensor,
    *,
    site: dict,
    group: str | None = None,
    group_map: list | None = None,
) -> None:
    """What ``dbm_expert_neuron.json`` leaves under ``fit/`` on Qwen3.6."""
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    identity = build_artifact_identity(
        model_key="Qwen/Qwen3.6-35B-A3B",
        model_revision="main",
        model_dtype="bf16",
        site=site,
        group=group,
        group_map=group_map,
        dtype="fp32",
        trained_on="weekdays/train",
        engine="pytorch_hooks",
        commit="fixture",
    )
    save_file({"theta": theta}, str(target), metadata=identity)


def _stamped_fit(root: Path) -> None:
    _stamped_gate(
        root,
        "fit/routed_gate.safetensors",
        torch.zeros(256, 512),
        site={"component": "expert_activation", "layers": [19]},
        group="expert_neuron",
        group_map=[256, 512],
    )
    _stamped_gate(
        root,
        "fit/shared_gate.safetensors",
        torch.zeros(512),
        site={"component": "shared_expert_activation", "layers": [19]},
    )


@pytest.mark.unit
class TestPresetsOffline:
    def test_the_fit_preset_validates_with_the_whole_expert_table(
        self, tmp_path: Path
    ) -> None:
        loaded = compile_protocol(
            PROTOCOLS / "dbm_expert_neuron.json", env=build_env(tmp_path)
        )
        featurizers = loaded.canonical["method"]["featurizers"]
        routed = featurizers["routed_gate"]
        assert routed["group"] == "expert_neuron"
        assert routed["width"] == 8 * 512  # Qwen3.6: top-8 slots of d_expert 512
        assert routed["params"] == {"theta": [256, 512]}  # 256 experts x 512
        shared = featurizers["shared_gate"]
        assert "group" not in shared
        assert shared["params"] == {"theta": [512]}
        assert loaded.document.train is not None
        assert loaded.document.train.eval is not None

    def test_the_apply_preset_validates_against_the_fits_bundles(
        self, tmp_path: Path
    ) -> None:
        _stamped_fit(tmp_path)
        loaded = compile_protocol(
            PROTOCOLS / "dbm_expert_neuron_apply.json", env=build_env(tmp_path)
        )
        assert loaded.document.train is None
        featurizers = loaded.canonical["method"]["featurizers"]
        assert "content_digest" in featurizers["routed_gate"]
        assert "content_digest" in featurizers["shared_gate"]

    def test_the_apply_preset_refuses_a_per_coordinate_routed_bundle(
        self, tmp_path: Path
    ) -> None:
        """A routed gate fitted without the group would read 8 x 512 slot
        parameters as an expert table; the stamp says what it was fitted as."""
        _stamped_fit(tmp_path)
        _stamped_gate(
            tmp_path,
            "fit/routed_gate.safetensors",
            torch.zeros(8 * 512),
            site={"component": "expert_activation", "layers": [19]},
        )
        with pytest.raises(ValidationError) as err:
            compile_protocol(
                PROTOCOLS / "dbm_expert_neuron_apply.json", env=build_env(tmp_path)
            )
        assert err.value.rule == 15
        assert "missing 'group'" in str(err.value)

    @pytest.mark.parametrize(
        "component",
        ["shared_expert_activation", "expert_output", "block_output"],
    )
    def test_the_group_is_refused_on_any_other_component_by_name(
        self, tmp_path: Path, component: str
    ) -> None:
        with pytest.raises(ValidationError) as err:
            compile_protocol(
                PROTOCOLS / "dbm_expert_neuron.json",
                env=build_env(tmp_path),
                overrides={"sites.routed.component": component},
            )
        assert err.value.rule == 23
        message = str(err.value)
        assert f"'{component}'" in message and "expert_neuron" in message
        assert err.value.path == "featurizers.routed_gate.group"

    def test_a_group_on_a_read_only_component_is_refused_by_the_write_policy_first(
        self, tmp_path: Path
    ) -> None:
        """`router_logits` is a component no write may change, and the
        capability registry (rule 4, §2.4) says so before the gate's group is
        ever examined — the refusal names the write, not the group. Rule 23
        would refuse the same document one step later; the earlier, more
        specific refusal is the one a user should see."""
        with pytest.raises(ValidationError) as err:
            compile_protocol(
                PROTOCOLS / "dbm_expert_neuron.json",
                env=build_env(tmp_path),
                overrides={"sites.routed.component": "router_logits"},
            )
        assert err.value.rule == 4
        message = str(err.value)
        assert "'router_logits'" in message and "no write may change" in message
        assert err.value.path == "writes.mask_routed.do"


@pytest.mark.unit
class TestExpertNeuronTrainTerms:
    def test_l1_is_the_mean_over_the_whole_table(self) -> None:
        """The penalty counts selected (expert, neuron) units over the table,
        not over the slots a token happens to fill."""
        gate = Gate(TOP_K * D_EXPERT, group="expert_neuron", groups=(EXPERTS, D_EXPERT))
        gate.temperature = 0.5
        with torch.no_grad():
            gate.theta.normal_(generator=torch.Generator().manual_seed(0))
        term = _regularizer("l1", ["routed_gate"], {"routed_gate": gate})
        assert torch.equal(term, torch.sigmoid(gate.theta / 0.5).mean())
        assert term.numel() == 1 and term.requires_grad

    def test_fit_diagnostics_count_expert_neuron_units(self) -> None:
        gate = Gate(TOP_K * D_EXPERT, group="expert_neuron", groups=(EXPERTS, D_EXPERT))
        with torch.no_grad():
            gate.theta.fill_(-3.0)
            gate.theta[5, :7] = 3.0
            gate.theta[9, 0] = 0.5  # undecided, yet selected by theta > 0
        report = fit_diagnostics({"routed_gate": gate})["routed_gate"]
        assert report["width"] == TOP_K * D_EXPERT
        assert report["groups"] == EXPERTS * D_EXPERT
        assert report["hard_mask_size"] == 8.0
        assert report["decisive_fraction"] == pytest.approx(
            1 - 1 / (EXPERTS * D_EXPERT)
        )


# --------------------------------------------------------------------------- #
# end to end: fit on the fixture, apply
# --------------------------------------------------------------------------- #


def _pipeline(component: str = "expert_activation") -> dict:
    """The fit preset retargeted at the fixture, plus two plain ``expert_idx``
    reads at the write's position — the routing of both inputs the hand
    computation of ``routing_mismatch.json`` needs — then the apply preset.
    ``--set`` replaces sections rather than creating keys, so the fit's
    ``sites``, ``reads`` and ``save`` are the preset's own with those added."""
    preset = json.loads((PROTOCOLS / "dbm_expert_neuron.json").read_text())
    sites = {
        **preset["method"]["sites"],
        "routed": {"component": component, "layers": [LAYER]},
        "shared": {"component": "shared_expert_activation", "layers": [LAYER]},
        "idx": {"component": "expert_idx", "layers": [LAYER]},
    }
    # the routing index on each input, read on the un-intervened model of
    # that input (§2.9): the shipped preset's `original_counterfactual`
    # takes `idx_cf`; `idx_base` needs the network on base, declared here
    routing_reads = {
        "idx_base": {"site": "idx", "pos": -1},
        "idx_cf": {"site": "idx", "pos": -1},
    }
    routing_models = {"idx_base": "original_base", "idx_cf": "original_counterfactual"}
    models = preset["method"]["intervened_models"]
    routing_ims = {
        **models,
        "original_counterfactual": {
            **models["original_counterfactual"],
            "reads": [*models["original_counterfactual"]["reads"], "idx_cf"],
        },
        "original_base": {"input": "base", "reads": ["idx_base"]},
    }
    routing_saves = [
        {
            "read": name,
            "model": routing_models[name],
            "file_path": f"{name}.safetensors",
        }
        for name in routing_reads
    ]
    return {
        "version": "1",
        "description": "fit an expert-keyed DBM gate beside a shared-expert gate, then apply both",
        "output_dir": "dbm_expert_neuron",
        "steps": {
            "fit": {
                "type": "intervention_protocol",
                "document": str(PROTOCOLS / "dbm_expert_neuron.json"),
                "set": {
                    **TINY,
                    "sites": sites,
                    "intervened_models": routing_ims,
                    "reads": {**preset["method"]["reads"], **routing_reads},
                    "save": [*preset["method"]["save"], *routing_saves],
                    # Pin the tiny fixture for both fitting and evaluation;
                    # shipped presets use the full task tables.
                    "data.base.dataset": "weekdays/train",
                    "data.counterfactual.dataset": "weekdays/train",
                    "train.eval.split": "weekdays/test",
                    "train.steps": {"epochs": 1},
                    "train.batch": {"pairs": 2},
                },
            },
            "apply": {
                "type": "intervention_protocol",
                "document": str(PROTOCOLS / "dbm_expert_neuron_apply.json"),
                "set": {
                    **TINY,
                    "sites.routed.component": component,
                    "sites.routed.layers": LAYER,
                    "sites.shared.layers": LAYER,
                    # score the split the fit reported on, so the two numbers
                    # are the same question asked twice
                    "data.base.dataset": "weekdays/train",
                    "data.counterfactual.dataset": "weekdays/train",
                },
            },
        },
    }


def _run_workflow(base: Path, document: dict) -> tuple[int, Path]:
    artifacts = base / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    path = base / "wf.json"
    path.write_text(json.dumps(document, indent=2))
    out = base / "run"
    code = main(
        [
            "run",
            "--engine",
            "auto",
            str(path),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts),
            "--out",
            str(out),
        ]
    )
    return code, out / document["output_dir"]


def _counts(records: list[dict]) -> list[dict]:
    return [
        {key: value for key, value in row.items() if key not in ("point", "coords")}
        for row in records
    ]


@pytest.fixture(scope="module", params=["expert_activation", "expert_neuron_output"])
def expert_run(tmp_path_factory: pytest.TempPathFactory, request) -> Path:
    code, run = _run_workflow(
        tmp_path_factory.mktemp("expert_neuron"), _pipeline(request.param)
    )
    assert code == 0
    return run


@pytest.mark.smoke
class TestFitThenApply:
    def test_the_routed_bundle_holds_the_expert_table_and_says_so(
        self, expert_run: Path
    ) -> None:
        bundle = expert_run / "fit/routed_gate.safetensors"
        theta = load_file(str(bundle))["theta"]
        assert theta.shape == (EXPERTS, D_EXPERT)
        assert theta.abs().max() > 0
        header = read_safetensors_metadata(bundle)
        assert header is not None
        assert header["group"] == "expert_neuron"
        assert json.loads(header["group_map"]) == [EXPERTS, D_EXPERT]
        (entry,) = json.loads(header["entries"]).values()
        assert entry["group"] == "expert_neuron"
        assert json.loads(entry["group_map"]) == [EXPERTS, D_EXPERT]

    def test_the_shared_bundle_is_a_separate_per_neuron_gate(
        self, expert_run: Path
    ) -> None:
        bundle = expert_run / "fit/shared_gate.safetensors"
        theta = load_file(str(bundle))["theta"]
        assert theta.shape == (D_SHARED,)
        header = read_safetensors_metadata(bundle)
        assert header is not None
        assert "group" not in header and "group_map" not in header
        assert json.loads(header["site"])["component"] == "shared_expert_activation"

    def test_the_diagnostics_count_units_per_gate(self, expert_run: Path) -> None:
        (record,) = json.loads((expert_run / "fit/fit_diagnostics.json").read_text())
        routed = record["featurizers"]["routed_gate"]
        assert routed["width"] == TOP_K * D_EXPERT
        assert routed["groups"] == EXPERTS * D_EXPERT
        theta = load_file(str(expert_run / "fit/routed_gate.safetensors"))["theta"]
        assert routed["hard_mask_size"] == float((theta > 0).sum())
        shared = record["featurizers"]["shared_gate"]
        assert shared["width"] == D_SHARED and "groups" not in shared

    def test_routing_mismatch_matches_the_saved_routing_of_both_inputs(
        self, expert_run: Path
    ) -> None:
        """One record per (write, layer, example) for the expert-keyed write
        and none for the shared one; ``mismatched`` is the number of base
        slots whose expert the counterfactual token did not activate, computed
        here from the saved ``expert_idx`` reads of both inputs."""
        records = json.loads((expert_run / "fit/routing_mismatch.json").read_text())
        idx_base = load_file(str(expert_run / "fit/idx_base.safetensors"))
        idx_cf = load_file(str(expert_run / "fit/idx_cf.safetensors"))
        (base,) = idx_base.values()
        (cf,) = idx_cf.values()
        assert base.shape == cf.shape == (4, 1, TOP_K)  # the fixture's 4 pairs
        assert {record["write"] for record in records} == {"mask_routed"}
        assert [record["example"] for record in records] == [0, 1, 2, 3]
        for record in records:
            i = record["example"]
            assert record["layer"] == LAYER and record["slots"] == TOP_K
            active = set(cf[i, 0].tolist())
            expected = sum(int(e) not in active for e in base[i, 0].tolist())
            assert record["mismatched"] == expected
        # the fixture routes the two inputs differently enough that the rule
        # actually decided something, and not so differently that it never
        # found a source
        assert 0 < sum(r["mismatched"] for r in records) < 4 * TOP_K

    def test_the_apply_reproduces_the_fit_exactly(self, expert_run: Path) -> None:
        """Same table, same hard mask, same routing rule, same number."""
        fitted = table_frame(expert_run / "fit/iia.json")
        applied = table_frame(expert_run / "apply/iia.json")
        assert len(applied) == len(fitted) == 4
        assert list(applied["value"]) == pytest.approx(list(fitted["value"]), abs=0.0)
        fit_mismatch = json.loads(
            (expert_run / "fit/routing_mismatch.json").read_text()
        )
        apply_mismatch = json.loads(
            (expert_run / "apply/routing_mismatch.json").read_text()
        )
        # the point digest and coords differ by design; the counts must not
        assert _counts(apply_mismatch) == _counts(fit_mismatch)
        manifest = json.loads((expert_run / "workflow.json").read_text())
        assert manifest["steps"]["apply"]["status"] == "completed"
