"""DBM's held-out path: fit a gate, save it, apply it (spec §2.5 ``file_path``).

The gap this closes. ``FEATURIZER_SLOTS["gate"] = ("theta",)`` makes a trained
gate **saveable**, and §2.5 says ``file_path`` is legal on every kind — but
``_build_stage`` dispatched ``subspace``/``pca``/``standardize``/``sae`` and
then raised, so a saved gate could never be loaded back. DBM therefore had no
apply document and no held-out number at all, while DAS had both
(``weekdays_das_sweep`` → ``weekdays_das_apply``). Two independent A3B runs hit
it and reported a DBM fit's own ``iia.json`` — a *train* score — as a
localization result.

The load-bearing assertion is the last one: an apply against the fit's own
split must reproduce the fit's number **exactly**. Anything less than exact
means the reloaded mask is not the mask that was scored — a soft σ(θ/T)
instead of the hard ``θ > 0`` split, a re-initialised θ, or a silently
truncated one.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from causalab.cli import main
from causalab.neural.shared.featurizers import Gate, build_stack
from causalab.io.tensor_files import TensorBundle
from causalab.protocol.rules.errors import ProtocolError
from causalab.io.env import read_safetensors_metadata
from causalab.protocol.schema import FeaturizerSpec
from causalab.protocol.registry.shapes import FeatureShape, bs_flat_heads

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._docs import UNWRITTEN, aggregation, saved, term
from tests.protocol._env import FIXTURES, fixture_input_overrides
from tests.tables import frame as table_frame
from tests._helpers.paths import PROTOCOLS_DIR

pytestmark = pytest.mark.smoke

#: The masked model's margin: the fit's objective and its `iia.json` table.
IIA = aggregation("logit_diff", a="cf_answer", b="base_answer")

REPO = Path(__file__).resolve().parents[4]
METHODS = str(PROTOCOLS_DIR)


def _fixture_inputs(name: str) -> dict[str, str]:
    """The shipped document's dataset refs, retargeted onto the fixture tables
    (tiny-random cannot tokenize every weekday; see tests/protocol/_env.py)."""
    return fixture_input_overrides(json.loads((Path(METHODS) / name).read_text()))


# tiny-random on CPU runs fp32 while dbm.json declares bf16 — and the
# realization is part of a fit bundle's identity (§8), so fit and apply have
# to name the same one
TINY = {"model.key": TINY_LLAMA, "model.dtype": "fp32"}

WIDTH = 8


# --------------------------------------------------------------------------- #
# the unit: what comes back off disk
# --------------------------------------------------------------------------- #


def _gate_loader(theta: torch.Tensor):
    def load_tensors(_path: str) -> TensorBundle:
        return TensorBundle(tensors={"theta": theta}, entry_coords={})

    return load_tensors


def _loaded_gate(
    theta: torch.Tensor,
    *,
    width: int = WIDTH,
    group: str | None = None,
    site_shape: FeatureShape | None = None,
) -> Gate:
    """A ``file_path`` gate built at a ``width``-wide site; a grouped one
    derives its map from ``site_shape`` exactly as the executor does."""
    stack = build_stack(
        "gate",
        {
            "gate": FeaturizerSpec(
                kind="gate", group=group, file_path="gate.safetensors"
            )
        },
        width=width,
        load_tensors=_gate_loader(theta),
        stage_cache={},
        site_shape=site_shape,
        site_component=None if site_shape is None else "attention_premix",
    )
    (stage,) = stack.stages
    return stage


def test_a_loaded_gate_selects_exactly_the_fitted_coordinates() -> None:
    """``build_stack`` puts every stage in eval mode, so a loaded gate is the
    hard ``θ > 0`` split — the one a fit's reported score was computed
    through, not the soft mask the train loop optimizes."""
    theta = torch.tensor([1.0, -1.0, 0.5, -0.25, 3.0, -2.0, 0.0, 0.125])
    gate = _loaded_gate(theta)
    assert isinstance(gate, Gate)
    x = torch.ones(WIDTH)
    kept, _ = gate.featurize(x)
    assert torch.equal(kept.nonzero().flatten(), (theta > 0).nonzero().flatten())
    # θ == 0 is *not* selected: the split is strict, as Gate._mask writes it
    assert kept[6] == 0.0


def test_a_loaded_gate_is_the_same_object_a_trained_one_is_after_eval() -> None:
    """The apply path must not re-derive anything. A gate trained in-process
    and put in eval mode, and the same θ round-tripped through a file, have to
    mask identically — otherwise an apply number is not comparable with the
    fit it came from."""
    theta = torch.randn(WIDTH, generator=torch.Generator().manual_seed(0))
    trained = Gate(WIDTH)
    with torch.no_grad():
        trained.theta.copy_(theta)
    trained.eval()
    x = torch.randn(4, WIDTH, generator=torch.Generator().manual_seed(1))
    assert torch.equal(trained.featurize(x)[0], _loaded_gate(theta).featurize(x)[0])
    assert torch.equal(trained.featurize(x)[1], _loaded_gate(theta).featurize(x)[1])


def test_a_loaded_gate_is_not_trainable() -> None:
    """Applying a mask is not resuming a fit — §2.5 forbids a ``file_path``
    featurizer in ``train.params``, and the stage says so itself."""
    assert not _loaded_gate(torch.ones(WIDTH)).theta.requires_grad


def test_a_gate_fitted_at_another_width_refuses() -> None:
    """A mask is a set of coordinates of one activation. Loading a 2048-wide
    gate at a 4096-wide site used to reach the swap and die in the featurize
    matmul; declared width against real tensor is the check that catches it
    (§2.5, widths derive from the site)."""
    with pytest.raises(ProtocolError, match="wide but the site here"):
        _loaded_gate(torch.ones(WIDTH + 1))


# --------------------------------------------------------------------------- #
# the same contract for a HEAD-grouped fit (§2.5 `group`), whose theta is its
# head count rather than its width — so "the width it was fitted at" needs both
# numbers, and a theta matching neither is what has to be refused.
# --------------------------------------------------------------------------- #

#: WIDTH laid out as 4 heads of head_dim 2 — the site shape a head-grouped gate
#: derives its ``(heads, head_dim)`` map from.
HEADS, HEAD_DIM = 4, 2
HEAD_SITE = bs_flat_heads(HEADS, HEAD_DIM)


def test_a_loaded_grouped_gate_masks_whole_groups() -> None:
    """The acceptance test on the apply path: θ is the head count, and the
    hard-eval mask it produces is constant within each head."""
    theta = torch.tensor([1.0, -1.0, 0.5, -2.0])
    gate = _loaded_gate(theta, group="head", site_shape=HEAD_SITE)
    assert gate.groups == (HEADS, HEAD_DIM) and gate.theta.numel() == HEADS
    kept, _ = gate.featurize(torch.ones(WIDTH))
    assert kept.tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def test_a_grouped_gate_reloads_to_the_object_the_fit_scored() -> None:
    """Same claim as the coordinate-wise case, which is the whole point of a
    grouped fit being saveable: the apply number is comparable with the fit's
    only if the reloaded stage masks identically."""
    theta = torch.randn(HEADS, generator=torch.Generator().manual_seed(0))
    trained = Gate(WIDTH, group="head", groups=(HEADS, HEAD_DIM))
    with torch.no_grad():
        trained.theta.copy_(theta)
    trained.eval()
    x = torch.randn(4, WIDTH, generator=torch.Generator().manual_seed(1))
    loaded = _loaded_gate(theta, group="head", site_shape=HEAD_SITE)
    assert torch.equal(trained.featurize(x)[0], loaded.featurize(x)[0])
    assert torch.equal(trained.featurize(x)[1], loaded.featurize(x)[1])


def test_a_grouped_gate_refuses_a_theta_that_matches_neither_number() -> None:
    """A coordinate-wise θ loaded as a grouped gate is the dangerous case: it
    is the right length for *a* gate at this site, just not this one."""
    with pytest.raises(ProtocolError, match="8 heads but the site here is 4"):
        _loaded_gate(torch.ones(WIDTH), group="head", site_shape=HEAD_SITE)


def test_a_coordinate_wise_gate_refuses_a_grouped_theta() -> None:
    """And the converse, which is how a `group` dropped from an apply document
    surfaces — as a refusal naming the width, not as a mask over the wrong
    coordinates."""
    with pytest.raises(ProtocolError, match="4 wide but the site here is 8"):
        _loaded_gate(torch.ones(HEADS))


# --------------------------------------------------------------------------- #
# end to end: dbm.json fits, dbm_apply.json applies
# --------------------------------------------------------------------------- #


def _pipeline(apply_set: dict | None = None) -> dict:
    return {
        "version": "1",
        "description": "fit a DBM gate, then apply it without re-fitting",
        "output_dir": "dbm",
        "steps": {
            "fit": {
                "type": "intervention_protocol",
                "document": f"{METHODS}/dbm.json",
                "set": {
                    **TINY,
                    **_fixture_inputs("dbm.json"),
                    "sites.target.layers": 0,
                    "train.steps": {"epochs": 1},
                    "train.batch": {"pairs": 2},
                },
            },
            "apply": {
                "type": "intervention_protocol",
                "document": f"{METHODS}/dbm_apply.json",
                "set": {
                    **TINY,
                    **_fixture_inputs("dbm_apply.json"),
                    "sites.target.layers": 0,
                    # score the split the fit reported on, so the two numbers
                    # are the same question asked twice
                    "data.base.dataset": "weekdays/data#train",
                    "data.counterfactual.dataset": "weekdays/data#train",
                    **(apply_set or {}),
                },
            },
        },
    }


def _run_workflow(base: Path, document: dict) -> tuple[int, Path]:
    artifacts = base / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    wf_dir = base / "workflows"
    wf_dir.mkdir()
    path = wf_dir / "wf.json"
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


@pytest.fixture(scope="module")
def dbm_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    code, run = _run_workflow(tmp_path_factory.mktemp("dbm"), _pipeline())
    assert code == 0
    return run


def test_the_fit_saves_a_gate_the_apply_can_read(dbm_run: Path) -> None:
    theta = load_file(str(dbm_run / "fit/gate.safetensors"))["theta"]
    assert theta.ndim == 1
    manifest = json.loads((dbm_run / "workflow.json").read_text())
    assert manifest["steps"]["apply"]["status"] == "completed"


def test_the_applied_mask_reproduces_the_fit_exactly(dbm_run: Path) -> None:
    """The whole point of an apply document: the same mask, scored again.

    On the fit's own split the two must agree to the bit. When the apply
    document then names a *held-out* split, any difference is the
    generalization the study is after — and not, as before, the difference
    between a train score and nothing."""
    fitted = table_frame(dbm_run / "fit/iia.json")
    applied = table_frame(dbm_run / "apply/iia.json")
    assert len(applied) == len(fitted) == 2  # the weekdays/data#train fixture rows
    assert list(applied["value"]) == pytest.approx(list(fitted["value"]), abs=0.0)


def test_a_gate_fitted_at_another_site_refuses(tmp_path: Path, capsys) -> None:
    """The ArtifactIdentity check (§2.5), same as the ``subspace`` case: the
    stamped site is part of what the fit is, so applying an L0 mask at L1 is
    refused rather than scored — and refused for that reason, not because
    something downstream tripped over a shape."""
    code, _ = _run_workflow(tmp_path, _pipeline({"sites.target.layers": 1}))
    assert code == 1
    err = capsys.readouterr().err
    assert "[V15]" in err and "ArtifactIdentity mismatch on 'site'" in err


# --------------------------------------------------------------------------- #
# end to end: a HEAD-grouped fit through the real executor (§2.5 `group`)
#
# The unit tests above build a `Gate` from a site shape handed to them. This is
# the wiring test: nothing in the document says how many parameters the gate
# has, so if the executor failed to resolve the group the fit would simply
# train a 4·head_dim-wide coordinate gate and save it — silently, and with a
# perfectly plausible number. The assertion is the parameter count on disk.
# --------------------------------------------------------------------------- #


def _head_grouped_document() -> dict:
    """`dbm.json`'s shape, moved onto the attention interior and grouped.

    `attention_premix` is the o-projection's input: `(batch, position,
    heads·head_dim)`, so it is the component with a head axis that a write can
    land on. `block_output` — dbm.json's site — deliberately has none, which is
    what rule 23 refuses.
    """
    return {
        "header": {
            "protocol_version": "4",
            "description": "a head-grouped DBM gate: one theta per attention head",
        },
        "model": {"key": TINY_LLAMA, "revision": "main", "dtype": "fp32"},
        "data": {
            "base": {"dataset": "weekdays/train", "field": "input"},
            "counterfactual": {
                "dataset": "weekdays/train",
                "field": "counterfactual_inputs[0]",
            },
        },
        "method": {
            "intervened_models": {
                UNWRITTEN: {"input": "counterfactual", "reads": ["v_cf"]},
                "masked": {"input": "base", "reads": ["logits"], "writes": ["mask"]},
            },
            "sites": {
                "target": {"component": "attention_premix", "layers": [0]},
                "lm_head": {"component": "lm_head"},
            },
            "featurizers": {"gate": {"kind": "gate", "group": "head"}},
            "reads": {
                "v_cf": {"site": "target", "pos": -1, "featurizer": "gate"},
                "logits": {"site": "lm_head", "pos": -1},
            },
            "writes": {
                "mask": {
                    "site": "target",
                    "pos": -1,
                    "featurizer": "gate",
                    "do": {"swap": "v_cf"},
                }
            },
            "train": {
                "objective": [
                    [1.0, term("logits", "masked", dict(IIA))],
                    [0.01, {"l1": "gate"}],
                ],
                "params": ["gate"],
                "optimizer": {"name": "adamw", "lr": 1e-3},
                "steps": {"epochs": 1},
                "batch": {"pairs": 2},
                "seed": 0,
            },
            "save": [
                saved("logits", "masked", "iia.json", dict(IIA)),
                {"value": "gate", "site": "target", "file_path": "gate.safetensors"},
            ],
        },
    }


@pytest.fixture(scope="module")
def grouped_fit(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("grouped-dbm")
    # `_run_workflow` owns `base / "workflows"`, so the fit document lives
    # beside it and is referenced absolutely
    documents = base / "documents"
    documents.mkdir()
    inline = documents / "grouped_dbm.json"
    inline.write_text(json.dumps(_head_grouped_document(), indent=2))
    code, run = _run_workflow(
        base,
        {
            "version": "1",
            "description": "fit a head-grouped DBM gate",
            "output_dir": "grouped",
            "steps": {
                "fit": {
                    "type": "intervention_protocol",
                    "document": str(inline),
                    "set": {},
                }
            },
        },
    )
    assert code == 0
    return run


def test_a_head_grouped_fit_saves_one_parameter_per_head(grouped_fit: Path) -> None:
    """The head-grouping contract end to end: H parameters, resolved from the model's own
    config through the document's one `group` field, and the bundle stamps
    the group kind and the derived map."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(TINY_LLAMA)
    heads = config.num_attention_heads
    theta = load_file(str(grouped_fit / "fit/gate.safetensors"))["theta"]
    assert theta.shape == (heads,)
    header = read_safetensors_metadata(grouped_fit / "fit/gate.safetensors")
    assert header is not None and header["group"] == "head"
    assert json.loads(header["group_map"]) == [heads, config.hidden_size // heads]


def test_dropping_the_group_makes_the_same_document_coordinate_wise(
    tmp_path: Path,
) -> None:
    """The difference is attributable to `group` and nothing else.

    Same site, same model, same train block — only the one field removed — and
    the parameter count goes from H to the full feature width. Without this the
    test above would also pass if `attention_premix` merely happened to be H
    wide. It is also the compatibility check on the fit side: an ungrouped
    document stamps no `group` and no `group_map`, as before grouping existed.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(TINY_LLAMA)
    document = _head_grouped_document()
    del document["method"]["featurizers"]["gate"]["group"]
    documents = tmp_path / "documents"
    documents.mkdir()
    inline = documents / "coordinate_dbm.json"
    inline.write_text(json.dumps(document, indent=2))
    code, run = _run_workflow(
        tmp_path,
        {
            "version": "1",
            "description": "the same fit, coordinate-wise",
            "output_dir": "coordinate_wise",
            "steps": {
                "fit": {
                    "type": "intervention_protocol",
                    "document": str(inline),
                    "set": {},
                }
            },
        },
    )
    assert code == 0
    theta = load_file(str(run / "fit/gate.safetensors"))["theta"]
    assert theta.shape == (config.hidden_size,)
    assert config.hidden_size != config.num_attention_heads
    header = read_safetensors_metadata(run / "fit/gate.safetensors")
    assert header is not None
    assert "group" not in header and "group_map" not in header


def test_the_group_is_part_of_the_documents_identity(grouped_fit: Path) -> None:
    """A grouped fit and a coordinate-wise one at the same site are different
    experiments, so they must not share a digest: the authored `group` is in
    the canonical form, and the derived map is stamped into the bundle."""
    grouped = json.loads((grouped_fit / "fit/_step.json").read_text())
    plain = _head_grouped_document()
    del plain["method"]["featurizers"]["gate"]["group"]
    assert grouped["document_digest"] != _digest_of(plain)


def _digest_of(document: dict) -> str:
    """The document digest, resolved the way a run resolves it."""
    from transformers import AutoConfig

    from causalab.protocol.schema.explicit import canonicalize, digest
    from causalab.protocol.registry import model_info_from_hf_config
    from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv

    info = model_info_from_hf_config(TINY_LLAMA, AutoConfig.from_pretrained(TINY_LLAMA))
    env = ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=FIXTURES / "artifacts"),
        model_info=lambda key: info,
    )
    return digest(canonicalize(document, env))


def test_rule_23_refuses_at_load_in_a_real_run(tmp_path: Path, capsys) -> None:
    """The refusal fires on the run path, not only against the static registry.

    This is the asymmetry worth pinning: the *canonicalizer* decides rule 23
    and the *executor* builds the stage, and they read the same registry. If
    canonicalization ever ran without model info while the executor kept
    resolving groups, an illegal group would sail through load and then be
    silently built — two vocabularies of what a head is, which is the thing one
    registry exists to prevent. `tiny-random-LlamaForCausalLM` is not a
    statically registered model, so this exercises the config-resolved path.
    """
    document = _head_grouped_document()
    document["method"]["sites"]["target"]["component"] = "block_output"
    documents = tmp_path / "documents"
    documents.mkdir()
    inline = documents / "illegal_group.json"
    inline.write_text(json.dumps(document, indent=2))
    code, _ = _run_workflow(
        tmp_path,
        {
            "version": "1",
            "description": "a head group on a component with no head axis",
            "output_dir": "illegal",
            "steps": {
                "fit": {
                    "type": "intervention_protocol",
                    "document": str(inline),
                    "set": {},
                }
            },
        },
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "[V23]" in err and "no head axis" in err
