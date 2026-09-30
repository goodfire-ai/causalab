"""Attention-head DBM end to end: ``dbm_head.json`` fits a head-grouped gate,
``dbm_head_apply.json`` replays it (spec §2.5 ``group: head``).

Both presets run on the tiny MoE fixture at both of its head-major families —
``attention_premix`` on the full-attention layer 3 (query heads) and
``delta_premix`` on the Gated DeltaNet layer 0 (value heads) — through the real
CLI, as ``test_dbm_apply_run.py`` does for the per-coordinate gate. What is
checked: the saved bundle's header carries the group kind and the derived
``(heads, head_dim)`` map, ``fit_diagnostics.json`` counts heads, and the apply
reproduces the fit's number exactly — so the reloaded mask is the fit's hard
mask over the same heads and nothing softer. A replay at the other family's
layer is refused by the stamped site, not by a shape somewhere downstream.

The offline half: both presets validate against the registered Qwen3.6 entry
with no model loaded, the fit's canonical form has one ``theta`` per query
head, and the train loop's L1 over a grouped gate is the mean over heads.
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

#: the fixture's two head-major families: (id, component, layer, heads, head_dim)
FAMILIES = {
    "full_attention": ("attention_premix", 3, 8, 32),
    "deltanet": ("delta_premix", 0, 8, 32),
}


# --------------------------------------------------------------------------- #
# offline: the presets validate, the grouping is in the canonical form
# --------------------------------------------------------------------------- #


def _stamped_head_gate(
    root: Path, *, theta_len: int, site: dict, group_map: list
) -> None:
    """What ``dbm_head.json`` leaves at ``fit/gate.safetensors`` on Qwen3.6."""
    target = root / "fit/gate.safetensors"
    target.parent.mkdir(parents=True, exist_ok=True)
    identity = build_artifact_identity(
        model_key="Qwen/Qwen3.6-35B-A3B",
        model_revision="main",
        model_dtype="bf16",
        site=site,
        group="head",
        group_map=group_map,
        dtype="fp32",
        trained_on="weekdays/train",
        engine="pytorch_hooks",
        commit="fixture",
    )
    save_file({"theta": torch.zeros(theta_len)}, str(target), metadata=identity)


@pytest.mark.unit
class TestPresetsOffline:
    def test_the_fit_preset_validates_with_one_theta_per_query_head(
        self, tmp_path: Path
    ) -> None:
        loaded = compile_protocol(PROTOCOLS / "dbm_head.json", env=build_env(tmp_path))
        gate = loaded.canonical["method"]["featurizers"]["gate"]
        assert gate["group"] == "head"
        assert gate["width"] == 16 * 256  # Qwen3.6: 16 query heads of 256
        assert gate["params"] == {"theta": [16]}
        assert loaded.document.train is not None
        assert loaded.document.train.eval is not None

    def test_the_apply_preset_validates_against_the_fits_bundle(
        self, tmp_path: Path
    ) -> None:
        _stamped_head_gate(
            tmp_path,
            theta_len=16,
            site={"component": "attention_premix", "layers": [19]},
            group_map=[16, 256],
        )
        loaded = compile_protocol(
            PROTOCOLS / "dbm_head_apply.json", env=build_env(tmp_path)
        )
        assert loaded.document.train is None
        assert "content_digest" in loaded.canonical["method"]["featurizers"]["gate"]

    def test_the_apply_preset_refuses_a_deltanet_fit_by_its_site(
        self, tmp_path: Path
    ) -> None:
        """A value-head mask (32 × 128) replayed at the query-head site: the
        stamped site disagrees first, and the refusal names it."""
        _stamped_head_gate(
            tmp_path,
            theta_len=32,
            site={"component": "delta_premix", "layers": [18]},
            group_map=[32, 128],
        )
        with pytest.raises(ValidationError) as err:
            compile_protocol(PROTOCOLS / "dbm_head_apply.json", env=build_env(tmp_path))
        assert err.value.rule == 15
        assert "ArtifactIdentity mismatch on 'site'" in str(err.value)

    def test_the_fit_preset_refuses_a_deltanet_layer_by_stream(
        self, tmp_path: Path
    ) -> None:
        """``attention_premix`` exists only on a full-attention layer; the
        DeltaNet analogue is ``delta_premix``, which the description says."""
        with pytest.raises(ValidationError) as err:
            compile_protocol(
                PROTOCOLS / "dbm_head.json",
                env=build_env(tmp_path),
                overrides={"sites.target.layers": 18},
            )
        assert err.value.rule == 4
        assert (
            "delta"
            in json.load((PROTOCOLS / "dbm_head.json").open())["header"]["description"]
        )


@pytest.mark.unit
class TestGroupedTrainTerms:
    def test_l1_on_a_grouped_gate_is_the_mean_over_heads(self) -> None:
        """The penalty is the selected-head count (scaled by 1/heads), not a
        mean over the coordinates the heads span."""
        gate = Gate(4 * 16, group="head", groups=(4, 16))
        gate.temperature = 0.5
        with torch.no_grad():
            gate.theta.copy_(torch.tensor([3.0, -3.0, 0.5, -0.5]))
        term = _regularizer("l1", ["gate"], {"gate": gate})
        expected = torch.sigmoid(gate.theta / 0.5).mean()
        assert torch.equal(term, expected)
        assert term.numel() == 1 and term.requires_grad

    def test_fit_diagnostics_count_heads(self) -> None:
        gate = Gate(4 * 16, group="head", groups=(4, 16))
        with torch.no_grad():
            gate.theta.copy_(torch.tensor([3.0, -3.0, 0.5, -0.5]))
        report = fit_diagnostics({"gate": gate})["gate"]
        assert report["width"] == 64.0 and report["groups"] == 4.0
        assert report["hard_mask_size"] == 2.0
        assert report["decisive_fraction"] == 0.5  # ±3 are decisive, ±0.5 are not

    def test_an_ungrouped_gate_reports_no_group_count(self) -> None:
        report = fit_diagnostics({"gate": Gate(6)})["gate"]
        assert report["width"] == 6.0 and "groups" not in report


# --------------------------------------------------------------------------- #
# end to end: fit on the fixture, apply, both families
# --------------------------------------------------------------------------- #


def _pipeline(component: str, layer: int, *, apply_site: dict | None = None) -> dict:
    site = {"sites.target.component": component, "sites.target.layers": layer}
    return {
        "version": "1",
        "description": "fit a head-grouped DBM gate, then apply it without re-fitting",
        "output_dir": "dbm_head",
        "steps": {
            "fit": {
                "type": "intervention_protocol",
                "document": str(PROTOCOLS / "dbm_head.json"),
                "set": {
                    **TINY,
                    **site,
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
                "document": str(PROTOCOLS / "dbm_head_apply.json"),
                "set": {
                    **TINY,
                    **(apply_site or site),
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


@pytest.fixture(scope="module", params=sorted(FAMILIES))
def head_run(request, tmp_path_factory: pytest.TempPathFactory):
    component, layer, heads, head_dim = FAMILIES[request.param]
    code, run = _run_workflow(
        tmp_path_factory.mktemp(request.param), _pipeline(component, layer)
    )
    assert code == 0
    return run, heads, head_dim


@pytest.mark.smoke
class TestFitThenApply:
    def test_the_bundle_holds_one_theta_per_head_and_says_so(self, head_run) -> None:
        run, heads, head_dim = head_run
        bundle = run / "fit/gate.safetensors"
        theta = load_file(str(bundle))["theta"]
        assert theta.shape == (heads,)
        header = read_safetensors_metadata(bundle)
        assert header is not None
        assert header["group"] == "head"
        assert json.loads(header["group_map"]) == [heads, head_dim]
        # a single-point fit stamps per entry too — the record a selection reads
        (entry,) = json.loads(header["entries"]).values()
        assert entry["group"] == "head" and json.loads(entry["group_map"]) == [
            heads,
            head_dim,
        ]

    def test_the_diagnostics_count_heads(self, head_run) -> None:
        run, heads, head_dim = head_run
        (record,) = json.loads((run / "fit/fit_diagnostics.json").read_text())
        gate = record["featurizers"]["gate"]
        assert gate["width"] == heads * head_dim and gate["groups"] == heads
        assert 0 <= gate["hard_mask_size"] <= heads
        theta = load_file(str(run / "fit/gate.safetensors"))["theta"]
        assert gate["hard_mask_size"] == float((theta > 0).sum())

    def test_the_apply_reproduces_the_fit_exactly(self, head_run) -> None:
        """Same heads, same hard mask, same number: anything but exact means
        the reloaded gate is not the object the fit was scored through."""
        run, _, _ = head_run
        fitted = table_frame(run / "fit/iia.json")
        applied = table_frame(run / "apply/iia.json")
        assert len(applied) == len(fitted) == 4  # the weekdays/train fixture rows
        assert list(applied["value"]) == pytest.approx(list(fitted["value"]), abs=0.0)
        manifest = json.loads((run / "workflow.json").read_text())
        assert manifest["steps"]["apply"]["status"] == "completed"


@pytest.mark.smoke
def test_a_fit_replayed_at_the_other_familys_layer_is_refused(
    tmp_path: Path, capsys
) -> None:
    """Query heads fitted at layer 3, replayed at the DeltaNet layer 0's value
    heads: the fixture's two maps happen to coincide (8 × 32), which is exactly
    why the stamped site — component and layer — has to be what refuses."""
    component, layer, _, _ = FAMILIES["full_attention"]
    other, other_layer, _, _ = FAMILIES["deltanet"]
    code, _ = _run_workflow(
        tmp_path,
        _pipeline(
            component,
            layer,
            apply_site={
                "sites.target.component": other,
                "sites.target.layers": other_layer,
            },
        ),
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "[V15]" in err and "ArtifactIdentity mismatch on 'site'" in err
