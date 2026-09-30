"""``tests/_helpers/fabricated_outputs.py``: the fabricated files of a protocol
step have the form a real run writes, the values follow the seeding rule of
the module docstring, and the stand-in tokenizer gives each string one id.

The unit tests read two fixture documents in ``tests/protocol/fixtures``:
the scan of ``direct_effect_scan.json`` (72 points over a swept probe site;
three reads on a model that lands no write, one read on a model that lands
one, and one ``top_k`` metric) and the subject read of
``subject_embeddings.json`` (one read over a variable-width subject, on the
six facts of ``facts/known``).
`TestTheFabricatedBundles` reads the fit presets of the method library
(``demos/methods/protocols``), which name registry models and so compile
with no model loaded.

`TestTheFormOfARealRun` runs four paper documents and four fit presets on a
tiny model through
the workflow runner and a real engine, and holds the fabricated files to
the files that run writes. It is the check that keeps the fabricator an
allowed stand-in under the mocking policy of ``docs/TESTS.md``.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open

from causalab.analysis import random_mask
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.io.step_record import EXAMPLE_COLUMN, axes_for
from causalab.io.tables import read_table
from causalab.neural.shared.featurizers.stages import (
    ORTHONORMAL_TOLERANCE,
    orthonormality_deviation,
)
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.registry import get_model_info
from causalab.tasks import TASKS_ROOT
from causalab.workflow.document import LoadedWorkflow, load_workflow
from causalab.workflow.runner import run_workflow
from causalab.workflow.steps import InnerProtocol
from tests._helpers.fabricated_outputs import (
    SPAN_MAX,
    VOCAB,
    WIDTH,
    StandInTokenizer,
    fabricate_protocol_outputs,
)
from tests.neural.engines.pytorch_hooks.conftest import TINY_QWEN35_MOE
from tests.protocol._env import FIXTURES, write_pca_fixture

REPO = Path(__file__).resolve().parents[2]
PAPERS = REPO / "demos" / "papers"
#: The paper tables, then the fixture tables (``facts/``) of the two fixture
#: documents below.
ENV = ResolutionEnv(
    datasets=FileDatasets(
        root=PAPERS / "artifacts" / "data", fallback_roots=(FIXTURES / "data",)
    ),
    artifacts=FileArtifacts(root=REPO),
)
FILES = ("a.safetensors", "z.safetensors", "lens.safetensors", "ml_token.json")
METHODS = REPO / "demos" / "methods" / "protocols"

#: A swept scan with a vocabulary read on a written model and a ``top_k``
#: metric. No shipped paper workflow has one.
SCAN = FIXTURES / "direct_effect_scan.json"


def _scan_step() -> InnerProtocol:
    """The scan as a workflow step would load it."""
    return InnerProtocol.enumerate(compile_protocol(SCAN, env=ENV), ENV)


@pytest.fixture(scope="module")
def clean(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("clean")
    fabricate_protocol_outputs("clean", _scan_step(), FILES, ENV, out)
    return out


def _step(document: str) -> InnerProtocol:
    """One paper protocol as a workflow step would load it."""
    return InnerProtocol.enumerate(
        compile_protocol(PAPERS / "protocols" / document, env=ENV), ENV
    )


#: The one read over a variable-width subject; no paper workflow ships one
#: since ``rome_fig1`` no longer has its noise-scale step.
SUBJECT_READ = FIXTURES / "subject_embeddings.json"


def _subject_step() -> InnerProtocol:
    """The subject read as a workflow step would load it."""
    return InnerProtocol.enumerate(compile_protocol(SUBJECT_READ, env=ENV), ENV)


def _entries(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


class TestTheFabricatedFiles:
    """The helper's own rules, on paper documents, with no model."""

    pytestmark = pytest.mark.unit

    def test_the_files_have_the_engine_shape(self, clean: Path) -> None:
        """One entry per point keyed by its coordinates, each ``(rows, 1,
        width)`` in the model dtype, with the ``entries`` table beside it; one
        metric row per point and example; the axes in ``_step.json``."""
        assert sorted(p.name for p in clean.iterdir()) == sorted([*FILES, "_step.json"])
        a = _entries(clean / "a.safetensors")
        assert len(a) == 72
        assert "a[probe.component=attention_output,probe.layers=0]" in a
        assert all(tensor.shape == (1, 1, WIDTH) for tensor in a.values())
        assert all(tensor.dtype == torch.bfloat16 for tensor in a.values())
        assert all(
            t.shape == (1, 1, VOCAB)
            for t in _entries(clean / "lens.safetensors").values()
        )
        with safe_open(str(clean / "a.safetensors"), framework="pt") as handle:
            metadata = handle.metadata() or {}
        entries = json.loads(metadata["entries"])
        assert entries["a[probe.component=mlp_output,probe.layers=35]"] == {
            "coords": {"probe.component": "mlp_output", "probe.layers": 35},
            "site": '{"component": "mlp_output", "layers": [35]}',
            "slot": "a",
            "trained_on": "facts/prompt",
        }
        assert metadata["model_dtype"] == "bf16"
        rows = read_table(clean / "ml_token.json")
        assert len(rows) == 72
        assert {EXAMPLE_COLUMN, "metric", "value", "unit", "eligible"} <= set(rows[0])
        assert {"sites.probe.component", "sites.probe.layers"} <= set(rows[0])
        assert axes_for(clean / "ml_token.json") == (
            "sites.probe.component",
            "sites.probe.layers",
        )

    def test_a_read_over_a_variable_is_ragged(self, tmp_path: Path) -> None:
        """The subject of each fact is as wide as the tokenizer makes it, so the
        engine saves the flat gather and its widths. The fabricated widths run
        from 1 to `SPAN_MAX` and differ between rows."""
        fabricate_protocol_outputs(
            "noise",
            _subject_step(),
            ["subject_embeddings.safetensors"],
            ENV,
            tmp_path,
        )
        saved = _entries(tmp_path / "subject_embeddings.safetensors")
        assert sorted(saved) == ["e", "e.widths"]
        widths = saved["e.widths"]
        assert widths.shape == (6,) and widths.dtype == torch.int64
        assert 1 <= int(widths.min()) < int(widths.max()) <= SPAN_MAX
        assert saved["e"].shape == (int(widths.sum()), WIDTH)
        assert saved["e"].dtype == torch.float32

    @pytest.mark.parametrize(
        "pos",
        [
            {"span": [0, 2], "scope": {"variable": "subject"}},
            {"indices": [0, 1], "scope": {"variable": "subject"}},
            {"span": [0, 2], "relative_to": {"variable": "subject"}},
            {"indices": [1, 2], "relative_to": {"variable": "subject"}},
        ],
        ids=["scoped-span", "scoped-indices", "relative-span", "relative-indices"],
    )
    def test_a_scoped_or_relative_set_is_refused(
        self, pos: dict[str, Any], tmp_path: Path
    ) -> None:
        """The engine's width for such a read follows from the set, and under
        a scope from the anchor's length too (``resolve_position`` in
        ``causalab/protocol/positions/encoding.py``). A seeded width per row
        would be a ragged shape the engine does not write for it."""
        step = InnerProtocol.enumerate(
            compile_protocol(
                SUBJECT_READ,
                env=ENV,
                overrides={"positions.subject": pos},
            ),
            ENV,
        )
        with pytest.raises(
            AssertionError,
            match=r"^step 'noise': subject_embeddings\.safetensors: .*a span or "
            r"indices set inside a scope or relative_to anchor",
        ):
            fabricate_protocol_outputs(
                "noise", step, ["subject_embeddings.safetensors"], ENV, tmp_path
            )

    def test_a_scoped_index_is_one_token(self, tmp_path: Path) -> None:
        """The valid twin of the refusal above: an ``index`` inside a scope
        addresses one token on every row."""
        step = InnerProtocol.enumerate(
            compile_protocol(
                SUBJECT_READ,
                env=ENV,
                overrides={
                    "positions.subject": {"index": 0, "scope": {"variable": "subject"}}
                },
            ),
            ENV,
        )
        fabricate_protocol_outputs(
            "noise", step, ["subject_embeddings.safetensors"], ENV, tmp_path
        )
        saved = _entries(tmp_path / "subject_embeddings.safetensors")
        assert {key: tuple(t.shape) for key, t in saved.items()} == {"e": (6, 1, WIDTH)}

    def test_a_read_depends_on_what_decides_it(self, clean: Path) -> None:
        """``z`` and ``logits`` sit on a model that lands no write at a fixed
        site, so every point holds the same value; ``a`` moves with its swept
        site; ``lens`` sits on the written model and moves with the point."""
        z = list(_entries(clean / "z.safetensors").values())
        assert all(torch.equal(z[0], tensor) for tensor in z)
        a = list(_entries(clean / "a.safetensors").values())
        assert not torch.equal(a[0], a[1])
        lens = list(_entries(clean / "lens.safetensors").values())
        assert not torch.equal(lens[0], lens[1])
        tokens = {row["value"] for row in read_table(clean / "ml_token.json")}
        assert len(tokens) == 1, "the clean run's top token is one token at every point"

    def test_a_second_fabrication_writes_the_same_bytes(
        self, clean: Path, tmp_path: Path
    ) -> None:
        fabricate_protocol_outputs("clean", _scan_step(), FILES, ENV, tmp_path)
        for name in FILES:
            assert (tmp_path / name).read_bytes() == (clean / name).read_bytes(), name

    def test_a_file_no_entry_saves_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(
            AssertionError, match=r"^step 'clean': no save entry writes \['b\.json'\]"
        ):
            fabricate_protocol_outputs("clean", _scan_step(), ["b.json"], ENV, tmp_path)

    @pytest.mark.parametrize(
        ("document", "file", "refusal"),
        [
            (
                "rome_fig1_knockout.json",
                "location_ledger.json",
                "not the derived 'location_ledger' record",
            ),
            (
                "lookbacks_clean_accuracy.json",
                "correct_cf.json",
                "reads on the 'base' input only, not on 'counterfactual'",
            ),
        ],
    )
    def test_what_it_does_not_fabricate_is_refused(
        self, document: str, file: str, refusal: str, tmp_path: Path
    ) -> None:
        """A derived record, and a read on the counterfactual input, whose rows
        this helper does not model. The valid twins are the tests above and
        `TestTheFabricatedBundles`."""
        with pytest.raises(
            AssertionError,
            match=rf"^step 's': {re.escape(file)}: .*{re.escape(refusal)}",
        ):
            fabricate_protocol_outputs("s", _step(document), [file], ENV, tmp_path)

    def test_the_stand_in_tokenizer_gives_each_string_one_id(self) -> None:
        tokenizer = StandInTokenizer(size=2)
        assert tokenizer.encode(" Monday") == [0]
        assert tokenizer.encode("Monday") == [1]
        assert tokenizer.encode(" Monday", add_special_tokens=False) == [0]
        assert tokenizer.decode([1, 0]) == "Monday Monday"
        assert len(tokenizer) == 2
        with pytest.raises(AssertionError, match="holds 2 strings"):
            tokenizer.encode("Tuesday")


@pytest.fixture(scope="module")
def method_env(tmp_path_factory: pytest.TempPathFactory) -> ResolutionEnv:
    """The environment ``tests/demos/test_methods.py`` loads the method
    library in: the shipped task tables, over the protocol tests' artifact
    fixtures and the basis ``das_pca_init.json`` starts from."""
    root = tmp_path_factory.mktemp("artifacts")
    shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
    write_pca_fixture(root)
    return ResolutionEnv(
        datasets=FileDatasets(root=TASKS_ROOT, fallback_roots=(TASKS_ROOT,)),
        artifacts=FileArtifacts(root=root),
    )


def _fit(
    document: str,
    env: ResolutionEnv,
    files: list[str],
    out: Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fabricate the ``files`` of one fit preset; per file, its tensors, its
    file-level fields and its ``entries`` records."""
    step = InnerProtocol.enumerate(
        compile_protocol(METHODS / document, env=env, overrides=overrides), env
    )
    fabricate_protocol_outputs("fit", step, files, env, out)
    saved: dict[str, dict[str, Any]] = {}
    for file in files:
        with safe_open(str(out / file), framework="pt") as handle:
            metadata = dict(handle.metadata() or {})
        saved[file] = {
            "tensors": _entries(out / file),
            "entries": json.loads(metadata.pop("entries")),
            "metadata": metadata,
        }
    return saved


class TestTheFabricatedBundles:
    """Featurizer bundles: the stage the engine trains, built by the engine's
    own ``build_stack`` at the site width the model registry gives."""

    pytestmark = pytest.mark.unit

    def test_a_rotation_is_an_orthonormal_frame_at_the_site_width(
        self, method_env: ResolutionEnv, tmp_path: Path
    ) -> None:
        """``das_boundless.json`` fits a rank-64 rotation at the residual
        stream of Qwen2.5-7B. Its bundle holds one ``weight`` entry, as wide
        as the model's hidden size, whose columns are orthonormal."""
        saved = _fit("das_boundless.json", method_env, ["rot.safetensors"], tmp_path)
        rot = saved["rot.safetensors"]
        hidden = get_model_info("Qwen/Qwen2.5-7B").hidden_size
        (weight,) = rot["tensors"].values()
        assert sorted(rot["tensors"]) == ["weight"]
        assert weight.shape == (hidden, 64) and weight.dtype == torch.float32
        assert orthonormality_deviation(weight) <= ORTHONORMAL_TOLERANCE
        assert rot["metadata"]["k"] == "64"
        assert rot["metadata"]["parametrization"] == "cayley"
        assert rot["metadata"]["engine"] == "fabricated"
        assert rot["entries"]["weight"]["slot"] == "weight"
        assert rot["entries"]["weight"]["coords"] == {}

    def test_a_boundary_gate_holds_one_theta_in_the_unit_interval(
        self, method_env: ResolutionEnv, tmp_path: Path
    ) -> None:
        """The boundary gate behind that rotation is one scalar, the kept
        fraction of the rotation's columns (``Gate`` docstring)."""
        saved = _fit("das_boundless.json", method_env, ["bnd.safetensors"], tmp_path)
        bnd = saved["bnd.safetensors"]
        theta = bnd["tensors"]["theta"]
        assert theta.shape == (1,) and theta.dtype == torch.float32
        assert 0.0 <= float(theta[0]) <= 1.0
        assert bnd["metadata"]["parametrization"] == "boundary"

    def test_a_head_gate_holds_one_theta_per_head(
        self, method_env: ResolutionEnv, tmp_path: Path
    ) -> None:
        """``dbm_head.json`` groups its gate by attention head, so the gate
        has one ``theta`` per query head and stamps the head map."""
        saved = _fit("dbm_head.json", method_env, ["gate.safetensors"], tmp_path)
        gate = saved["gate.safetensors"]
        info = get_model_info("Qwen/Qwen3.6-35B-A3B")
        assert gate["tensors"]["theta"].shape == (info.num_heads,)
        assert gate["metadata"]["group"] == "head"
        assert json.loads(gate["metadata"]["group_map"]) == [
            info.num_heads,
            info.head_dim,
        ]

    def test_a_swept_fit_saves_one_entry_per_point(
        self, method_env: ResolutionEnv, tmp_path: Path
    ) -> None:
        """A fit swept over ``k`` writes one rotation per point, keyed by the
        point's coordinates, and each point's rotation is its own draw."""
        saved = _fit(
            "das.json",
            method_env,
            ["rot.safetensors"],
            tmp_path,
            {"featurizers.rot.k": {"sweep": [2, 8]}},
        )
        rot = saved["rot.safetensors"]
        hidden = get_model_info("Qwen/Qwen2.5-7B").hidden_size
        assert {key: tuple(t.shape) for key, t in rot["tensors"].items()} == {
            "weight[k=2]": (hidden, 2),
            "weight[k=8]": (hidden, 8),
        }
        assert rot["entries"]["weight[k=8]"]["coords"] == {"k": 8}
        assert "k" not in rot["metadata"], "a swept field is stamped per entry"
        first, second = rot["tensors"]["weight[k=2]"], rot["tensors"]["weight[k=8]"]
        assert not torch.equal(first, second[:, :2])

    def test_a_fabricated_gate_is_a_gate_random_mask_reads(
        self, method_env: ResolutionEnv, tmp_path: Path
    ) -> None:
        """The DBM control reads a fabricated head gate as it reads a fitted
        one, and draws its ``top_k`` heads."""
        _fit("dbm_head.json", method_env, ["gate.safetensors"], tmp_path)
        out = tmp_path / "control.safetensors"
        random_mask.main(
            {"gate": tmp_path / "gate.safetensors", "seed": 0, "top_k": 3},
            {"gate": out},
        )
        theta = _entries(out)["theta"]
        heads = get_model_info("Qwen/Qwen3.6-35B-A3B").num_heads
        assert theta.shape == (heads,) and int((theta > 0).sum()) == 3

    def test_a_featurizer_that_starts_from_a_saved_bundle_is_refused(
        self, method_env: ResolutionEnv, tmp_path: Path
    ) -> None:
        """``das_pca_init.json`` starts its rotation from a PCA basis. The
        engine records that basis in the bundle, and this helper does not
        load one."""
        with pytest.raises(
            AssertionError,
            match=r"^step 'fit': rot\.safetensors: .*starts from a saved file",
        ):
            _fit("das_pca_init.json", method_env, ["rot.safetensors"], tmp_path)


# --------------------------------------------------------------------------- #
# the form of a real run
# --------------------------------------------------------------------------- #

#: Ungated, two layers, 16 wide, a 32000-token vocabulary
#: (``tests/_helpers/tiny.py``).
TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"

#: Documents whose kinds of files a shipped script reads, on the tiny model:
#: a swept single-position scan with a vocabulary read on a written model
#: and a ``top_k`` metric (``clean``, a test fixture), a read over a
#: variable-width subject on six rows (``noise``, a test fixture), a greedy
#: ``decode`` (``baseline``, four tokens on one prompt) and a harvest at a
#: named ``index`` position (``harvest``).
#: Only the model and its revision, the layers and the decode budget are
#: changed: the paper documents pin their own models' snapshots.
PARITY: dict[str, Any] = {
    "version": "1",
    "output_dir": "parity",
    "steps": {
        "clean": {
            "type": "intervention_protocol",
            "document": str(SCAN),
            "set": {
                "model.key": TINY_LLAMA,
                "model.revision": "main",
                "sites.probe.layers": {"sweep": [0, 1]},
                "sites.final.layers": [1],
            },
        },
        "noise": {
            "type": "intervention_protocol",
            "document": str(SUBJECT_READ),
            "set": {"model.key": TINY_LLAMA, "model.revision": "main"},
        },
        "baseline": {
            "type": "intervention_protocol",
            "document": "../protocols/mlp_steering_baseline.json",
            "set": {
                "model.key": TINY_LLAMA,
                "model.revision": "main",
                "data.base.dataset": "facts/prompt",
                "positions.continuation.generated.max_new_tokens": 4,
            },
        },
        "harvest": {
            "type": "intervention_protocol",
            "document": "../protocols/manifold_fig4_harvest.json",
            "set": {
                "model.key": TINY_LLAMA,
                "model.revision": "main",
                "sites.target.layers": [1],
            },
        },
    },
}

#: The one entry field only a loaded model reports (module docstring of the
#: helper).
LOADED_ONLY = "loaded_attn_implementation"


def _tensor_form(path: Path) -> dict[str, Any]:
    """A bundle without its numbers and widths: the file-level fields and
    their values, each tensor's dtype, rank and leading axes (a ragged
    gather's row count is the sum of its widths, checked here), and each
    ``entries`` record."""
    with safe_open(str(path), framework="pt") as handle:
        metadata = dict(handle.metadata() or {})
        shapes = {
            key: (handle.get_slice(key).get_dtype(), handle.get_slice(key).get_shape())
            for key in handle.keys()
        }
        widths = {
            key: handle.get_tensor(key)
            for key in handle.keys()
            if key.endswith(".widths")
        }
    tensors: dict[str, Any] = {}
    for key, (dtype, shape) in shapes.items():
        sidecar = widths.get(f"{key}.widths")
        if sidecar is not None:
            assert shape[0] == int(sidecar.sum()), f"{path.name}: {key} rows"
            tensors[key] = (dtype, len(shape), "ragged")
        elif key in widths:
            tensors[key] = (dtype, tuple(shape))
        else:
            tensors[key] = (dtype, tuple(shape[:-1]))
    entries = json.loads(metadata.pop("entries"))
    engine = metadata.pop("engine")
    return {
        "metadata": metadata,
        "engine": engine,
        "tensors": tensors,
        "entries": entries,
    }


def _table_form(path: Path) -> dict[str, Any]:
    """A table without its numbers: its columns, the JSON types each column
    holds, its row count and the axes a script groups it by."""
    rows = read_table(path)
    types: dict[str, set[str]] = {}
    for row in rows:
        for column, value in row.items():
            types.setdefault(column, set()).add(type(value).__name__)
    return {"rows": len(rows), "types": types, "axes": axes_for(path)}


@pytest.mark.smoke
class TestTheFormOfARealRun:
    """The fabricated files of four paper steps against the files a real
    engine writes for the same steps on the tiny model."""

    @pytest.fixture(scope="class")
    def runs(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> tuple[LoadedWorkflow, Path, Path]:
        from transformers import AutoConfig

        from causalab.neural.shared.engine_router import route
        from causalab.protocol.registry import model_info_from_hf_config

        info = model_info_from_hf_config(
            TINY_LLAMA, AutoConfig.from_pretrained(TINY_LLAMA)
        )
        env = ResolutionEnv(
            datasets=ENV.datasets, artifacts=ENV.artifacts, model_info=lambda _: info
        )
        loaded = load_workflow(PARITY, env, workflow_dir=PAPERS / "workflows")
        real = tmp_path_factory.mktemp("real")
        result = run_workflow(loaded, env, real, route("auto", device="cpu"))
        assert {e["status"] for e in result.manifest["steps"].values()} == {"completed"}
        fabricated = tmp_path_factory.mktemp("fabricated")
        tokenizer = StandInTokenizer()
        for name in PARITY["steps"]:
            files = {
                entry.file_path for entry in loaded.inner[name].point_documents[0].save
            }
            fabricate_protocol_outputs(
                name, loaded.inner[name], files, env, fabricated / name, tokenizer
            )
        return loaded, real / "parity", fabricated

    def test_the_fabricated_files_have_the_form_of_a_real_run(
        self, runs: tuple[LoadedWorkflow, Path, Path]
    ) -> None:
        """Same files, tensor keys, dtypes, ranks and leading axes, file-level
        fields and ``entries`` records, table columns, column types and row
        counts. Only the widths, the numbers, the engine's name and the
        loaded attention backend differ."""
        loaded, real, fabricated = runs
        for name in PARITY["steps"]:
            files = sorted(
                {
                    entry.file_path
                    for entry in loaded.inner[name].point_documents[0].save
                }
            )
            for file in files:
                ours, theirs = fabricated / name / file, real / name / file
                where = f"{name}/{file}"
                if file.endswith(".json"):
                    assert _table_form(ours) == _table_form(theirs), where
                    continue
                mine, engine = _tensor_form(ours), _tensor_form(theirs)
                assert mine["engine"] == "fabricated", where
                for record in engine["entries"].values():
                    assert record.pop(LOADED_ONLY), f"{where}: the engine stamps it"
                for key in ("metadata", "tensors", "entries"):
                    assert mine[key] == engine[key], f"{where}: {key}"


#: The fit presets of the method library, retargeted as
#: ``tests/neural/engines/pytorch_hooks/test_dbm_head_run.py`` retargets
#: ``dbm_head.json``: the tiny Qwen3.5 MoE at its full-attention layer 3, in
#: fp32, on the protocol tests' weekdays fixture, whose answers are one token
#: under that tokenizer and not under the tiny Llama's. The steps are a
#: rank-4 rotation behind a boundary gate (``boundless``), a gate per
#: coordinate (``dbm``), a gate per attention head (``dbm_head``) and a
#: rotation swept over its rank (``das``). One epoch of two pairs, because
#: only the form is compared.
FIT_MODEL = {
    "model.key": TINY_QWEN35_MOE,
    "model.dtype": "fp32",
    "sites.target.layers": [3],
}
FIT_DATA = {
    "data.base.dataset": "weekdays/train",
    "data.counterfactual.dataset": "weekdays/train",
    "train.eval.split": "weekdays/test",
    "train.steps": {"epochs": 1},
    "train.batch": {"pairs": 2},
}
PARITY_FITS: dict[str, Any] = {
    "version": "1",
    "output_dir": "fits",
    "steps": {
        "boundless": {
            "type": "intervention_protocol",
            "document": str(METHODS / "das_boundless.json"),
            "set": {**FIT_MODEL, "featurizers.rot.k": 4, **FIT_DATA},
        },
        "dbm": {
            "type": "intervention_protocol",
            "document": str(METHODS / "dbm.json"),
            "set": {**FIT_MODEL, **FIT_DATA},
        },
        "dbm_head": {
            "type": "intervention_protocol",
            "document": str(METHODS / "dbm_head.json"),
            "set": {**FIT_MODEL, **FIT_DATA},
        },
        "das": {
            "type": "intervention_protocol",
            "document": str(METHODS / "das.json"),
            "set": {
                **FIT_MODEL,
                "featurizers.rot.k": {"sweep": [2, 4]},
                **FIT_DATA,
            },
        },
    },
}


def _bundle_form(path: Path) -> dict[str, Any]:
    """A featurizer bundle without its numbers: each tensor's dtype and whole
    shape, the file-level fields and each ``entries`` record, less the
    engine's name and the backend a loaded model reports."""
    with safe_open(str(path), framework="pt") as handle:
        metadata = dict(handle.metadata() or {})
        shapes = {
            key: (handle.get_slice(key).get_dtype(), handle.get_slice(key).get_shape())
            for key in handle.keys()
        }
    entries = json.loads(metadata.pop("entries"))
    for record in (metadata, *entries.values()):
        record.pop(LOADED_ONLY, None)
    return {
        "engine": metadata.pop("engine"),
        "metadata": metadata,
        "tensors": shapes,
        "entries": entries,
    }


@pytest.mark.smoke
class TestTheFormOfARealFit:
    """The fabricated files of four fit presets against the files a real
    engine writes when it fits them on the tiny model."""

    @pytest.fixture(scope="class")
    def fits(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> tuple[LoadedWorkflow, Path, Path]:
        from transformers import AutoConfig

        from causalab.neural.shared.engine_router import route
        from causalab.protocol.registry import model_info_from_hf_config

        info = model_info_from_hf_config(
            TINY_QWEN35_MOE, AutoConfig.from_pretrained(TINY_QWEN35_MOE)
        )
        env = ResolutionEnv(
            datasets=FileDatasets(root=FIXTURES / "data"),
            artifacts=FileArtifacts(root=REPO),
            model_info=lambda _: info,
        )
        loaded = load_workflow(PARITY_FITS, env, workflow_dir=METHODS)
        real = tmp_path_factory.mktemp("real")
        result = run_workflow(loaded, env, real, route("auto", device="cpu"))
        assert {e["status"] for e in result.manifest["steps"].values()} == {"completed"}
        fabricated = tmp_path_factory.mktemp("fabricated")
        tokenizer = StandInTokenizer()
        for name in PARITY_FITS["steps"]:
            files = {
                entry.file_path for entry in loaded.inner[name].point_documents[0].save
            }
            fabricate_protocol_outputs(
                name, loaded.inner[name], files, env, fabricated / name, tokenizer
            )
        return loaded, real / "fits", fabricated

    def test_the_fabricated_bundles_have_the_form_of_a_real_fit(
        self, fits: tuple[LoadedWorkflow, Path, Path]
    ) -> None:
        """Same bundle files, tensor keys, dtypes and whole shapes, file-level
        fields and ``entries`` records; the metric tables beside them have
        the same columns, column types and row counts. Only the numbers, the
        engine's name and the loaded attention backend differ."""
        loaded, real, fabricated = fits
        bundles = 0
        for name in PARITY_FITS["steps"]:
            for entry in loaded.inner[name].point_documents[0].save:
                file = entry.file_path
                ours, theirs = fabricated / name / file, real / name / file
                where = f"{name}/{file}"
                if file.endswith(".json"):
                    assert _table_form(ours) == _table_form(theirs), where
                    continue
                mine, engine = _bundle_form(ours), _bundle_form(theirs)
                assert mine.pop("engine") == "fabricated", where
                assert engine.pop("engine") != "fabricated", where
                assert mine == engine, where
                bundles += 1
        assert bundles == 5, "rot and bnd, two gates, one swept rotation"
