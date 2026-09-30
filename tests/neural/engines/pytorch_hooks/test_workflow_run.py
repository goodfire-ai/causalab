"""The weekdays pipeline shape end to end on tiny-random, through the real CLI.

locate scan → `causalab.workflow.scripts.select` the best cell → DAS fit at it (via in-run
artifact refs) → apply the fitted rotation (via a run-tree file_path,
ArtifactIdentity checked) → heatmap and lines plots. This is the anchor for the
v2 rewrite: the *pipeline* is the same one v1 ran, so if the numbers and the
provenance still arrive, the vocabulary change did not cost anything.

The workflow references the shipped method presets with per-step ``set``
overrides that retarget model/layers at tiny scale. The **tap axis is not
overridden**: the shipped document's own two positions are what runs here, so
CI exercises the axis a user gets. It used to override to two plain indices,
which is why nothing noticed that the shipped axis named a variable
(``subject``) no task-generated table has, and that a bare variable *window* is
ragged across weekday rows — rule 19's refusal ([V19]) for a write that
declares no ``ragged`` policy (spec §2.8; ``test_ragged_writes.py`` runs that
spelling under ``exact_length_buckets``). Both are fixed in the document: the
second position is the last token of the ``entity`` span, which is the position
the scan asks about.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from safetensors.torch import load_file

from causalab.cli import main
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._env import FIXTURES, fixture_input_overrides, write_rot_fixture
from tests.tables import frame as table_frame
from tests._helpers.paths import PROTOCOLS_DIR

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[4]
METHODS = str(PROTOCOLS_DIR)  # absolute: the workflow file lives in tmp

OUTPUT_DIR = "tiny_weekdays"


def _fixture_inputs(name: str) -> dict[str, str]:
    """The shipped document's dataset refs, retargeted onto the fixture tables."""
    return fixture_input_overrides(json.loads((Path(METHODS) / name).read_text()))


def _tiny_workflow() -> dict:
    tiny = {"model.key": TINY_LLAMA, "model.dtype": "fp32"}
    return {
        "version": "1",
        "description": "the weekdays pipeline shape at tiny scale",
        "output_dir": OUTPUT_DIR,
        "steps": {
            "locate": {
                "type": "intervention_protocol",
                "document": f"{METHODS}/weekdays_locate_scan.json",
                "set": {
                    **tiny,
                    **_fixture_inputs("weekdays_locate_scan.json"),
                    "sites.target.layers": {"sweep": {"range": [0, 2]}},
                },
            },
            "best": {
                "type": "script",
                "script": {"module": "causalab.workflow.scripts.select"},
                "inputs": {
                    "table": {"step": "locate", "file": "iia.json"},
                    "choose": "max",
                    "emit": {
                        "best_layer": "sites.target.layers",
                        "best_pos": "positions.tap",
                    },
                },
                "outputs": {
                    "values": {
                        "file": "values.json",
                        "keys": {"best_layer": 0, "best_pos": {"index": -1}},
                    }
                },
            },
            "fit": {
                "type": "intervention_protocol",
                "document": f"{METHODS}/weekdays_das_sweep.json",
                "set": {
                    **tiny,
                    **_fixture_inputs("weekdays_das_sweep.json"),
                    "positions.best": {"artifact": "best", "key": "best_pos"},
                    "sites.target.layers": {"artifact": "best", "key": "best_layer"},
                    "featurizers.rot.k": 2,
                    "train.seed": 0,
                    "train.steps": {"epochs": 1},
                    "train.batch": {"pairs": 2},
                },
            },
            "apply": {
                "type": "intervention_protocol",
                "document": f"{METHODS}/weekdays_das_apply.json",
                "set": {
                    **tiny,
                    **_fixture_inputs("weekdays_das_apply.json"),
                    "sites.target.layers": {"artifact": "best", "key": "best_layer"},
                    "featurizers.rot.k": 2,
                    "featurizers.rot.file_path": "fit/rot.safetensors",
                },
            },
            "scan_heatmap": {
                "type": "script",
                "script": {"module": "causalab.io.plots.workflow_figures"},
                "inputs": {
                    "table": {"step": "locate", "file": "iia.json"},
                    "plot": "heatmap",
                    "x": "sites.target.layers",
                    "y": "positions.tap",
                },
                "outputs": {
                    "figure": "scan_iia.png",
                    "plotted": {"file": "scan_iia.json"},
                },
            },
            "iia_lines": {
                "type": "script",
                "script": {"module": "causalab.io.plots.workflow_figures"},
                "inputs": {
                    "table": {"step": "locate", "file": "logit_diff.json"},
                    "plot": "lines",
                    "x": "sites.target.layers",
                    "series": "positions.tap",
                },
                "outputs": {
                    "figure": "ld_by_layer.png",
                    "plotted": {"file": "ld_by_layer.json"},
                },
            },
        },
    }


def _run(base: Path, artifacts: Path, wf: Path, *extra: str) -> int:
    return main(
        [
            "run",
            "--engine",
            "auto",
            str(wf),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts),
            "--out",
            str(base / "runs"),
            *extra,
        ]
    )


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """One full CLI run of the tiny pipeline; the assertions below share it."""
    base = tmp_path_factory.mktemp("wf")
    artifacts = base / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    write_rot_fixture(artifacts)
    wf_dir = base / "workflows"
    wf_dir.mkdir()
    wf = wf_dir / "tiny_weekdays.json"
    wf.write_text(json.dumps(_tiny_workflow(), indent=2))
    assert _run(base, artifacts, wf) == 0
    # the document names its own directory; the CLI supplied only the root (§1.1)
    return base / "runs" / OUTPUT_DIR, wf_dir, base


def test_the_document_names_its_own_run_directory(pipeline_run):
    out, _, base = pipeline_run
    assert out == base / "runs" / OUTPUT_DIR
    assert out.is_dir()


def test_run_tree_holds_every_step(pipeline_run):
    out, _, _ = pipeline_run
    assert (out / "locate/iia.json").is_file()
    assert (out / "best/values.json").is_file()
    assert (out / "fit/rot.safetensors").is_file()
    assert (out / "apply/iia.json").is_file()
    # the declared output is the plotted data; the image lands beside it
    assert (out / "scan_heatmap/scan_iia.json").is_file()
    assert (out / "scan_heatmap/scan_iia.png").stat().st_size > 0
    assert (out / "iia_lines/ld_by_layer.png").stat().st_size > 0


def test_every_step_publishes_a_record(pipeline_run):
    """`_step.json` is what lets a downstream script group by the producer's
    sweep axes without the document model deriving it (§6)."""
    out, _, _ = pipeline_run
    for step in ("locate", "best", "fit", "apply", "scan_heatmap"):
        record = json.loads((out / step / "_step.json").read_text())
        assert record["status"] == "completed"
    # a protocol step's record carries the same execution block protocol.json
    # does — the engine loaded its model, so the source is `loaded` (§8)
    for step in ("locate", "fit", "apply"):
        record = json.loads((out / step / "_step.json").read_text())
        assert record["execution"]["model_source"] == "loaded"
    locate = json.loads((out / "locate/_step.json").read_text())
    # the order is the producing document's axis order, which the consumer
    # never needs to know — it groups by all of them
    assert set(locate["axes"]) == {"sites.target.layers", "positions.tap"}
    assert json.loads((out / "best/_step.json").read_text())["axes"] == []


def test_select_chose_the_argmax_cell(pipeline_run):
    out, _, _ = pipeline_run
    chosen = json.loads((out / "best/values.json").read_text())
    assert chosen["best_layer"] in (0, 1)
    # the shipped tap axis, unoverridden: the last prompt token, or the last
    # token of the entity span
    assert chosen["best_pos"] in (
        {"index": -1},
        {"index": -1, "scope": {"variable": "entity"}},
    )
    frame = table_frame(out / "locate/iia.json")
    grouped = frame.groupby(["sites.target.layers", "positions.tap"])["value"].mean()
    best_key = grouped.idxmax()
    assert chosen["best_layer"] == best_key[0]
    assert chosen["best_pos"] == json.loads(best_key[1])


def test_fit_consumed_the_selected_cell_and_stamped_it(pipeline_run):
    """The in-run artifact wiring is provable from the fit bundle's
    ArtifactIdentity: the stamped site carries the layer `best` chose."""
    from causalab.io.env import read_safetensors_metadata

    out, _, _ = pipeline_run
    chosen = json.loads((out / "best/values.json").read_text())
    stamped = read_safetensors_metadata(out / "fit/rot.safetensors")
    assert stamped is not None
    site = json.loads(stamped["site"])
    assert site["layers"] == [chosen["best_layer"]]  # the stamp carries the band
    assert stamped["model_key"] == TINY_LLAMA
    assert stamped["k"] == "2"
    weight = load_file(str(out / "fit/rot.safetensors"))["weight"]
    assert weight.shape == (16, 2)


def test_apply_scored_the_test_split_through_the_fitted_rotation(pipeline_run):
    out, _, _ = pipeline_run
    iia = table_frame(out / "apply/iia.json")
    assert len(iia) == 2  # the weekdays/data#test fixture rows
    assert iia["value"].dtype.kind == "f"


def test_locate_table_carries_coordinate_columns(pipeline_run):
    out, _, _ = pipeline_run
    frame = table_frame(out / "locate/iia.json")
    assert {"sites.target.layers", "positions.tap", "value"} <= set(frame.columns)
    assert len(frame) == 4 * 2  # 4 points x 2 examples


def test_the_plotted_table_is_the_aggregated_data(pipeline_run):
    """A figure is a rendering; what the record keeps is the rows behind it."""
    out, _, _ = pipeline_run
    plotted = table_frame(out / "scan_heatmap/scan_iia.json")
    assert len(plotted) == 4  # one row per (layer, tap) cell
    assert set(plotted.columns) == {"sites.target.layers", "positions.tap", "value"}


def test_run_manifest_records_the_whole_run(pipeline_run):
    out, _, _ = pipeline_run
    manifest = json.loads((out / "workflow.json").read_text())
    assert "workflow_digest" not in manifest  # no run-level identity (§7)
    assert manifest["output_dir"] == OUTPUT_DIR
    assert manifest["steps"]["locate"]["points"] == 4
    assert len(manifest["steps"]["locate"]["point_digests"]) == 4  # provenance units
    assert manifest["steps"]["fit"]["engine"] == "pytorch_hooks"
    assert all(entry["status"] == "completed" for entry in manifest["steps"].values())
    # the run manifest stamps the fully resolved inner digests (§7)
    assert len(manifest["steps"]["apply"]["document_digest"]) == 64
    # a script step records its script's hash — where reproducibility lives
    assert len(manifest["steps"]["best"]["script_sha256"]) == 64
    assert manifest["steps"]["best"]["is_deterministic"] is True
    assert "nondeterministic" not in manifest


def test_no_save_section_and_nothing_is_copied(pipeline_run):
    """The run tree IS the publication (§0): there is no second copy of a
    step's outputs at the run root."""
    out, _, _ = pipeline_run
    top_level = {p.name for p in out.iterdir()}
    assert top_level == {
        "workflow.json",
        "events.jsonl",  # the run's event stream, beside the manifest (§4.3)
        "locate",
        "best",
        "fit",
        "apply",
        "scan_heatmap",
        "iia_lines",
    }


def test_explain_reports_the_derived_schedule(pipeline_run, capsys):
    _, wf_dir, _ = pipeline_run
    code = main(
        [
            "explain",
            "--engine",
            "auto",
            str(wf_dir / "tiny_weekdays.json"),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(FIXTURES / "artifacts"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "schedule" in out
    assert "script causalab.workflow.scripts.select" in out


@pytest.mark.smoke
def test_resume_reuses_a_step_and_a_script_edit_busts_it(
    tmp_path_factory: pytest.TempPathFactory,
):
    """Why the script hash is in the digest: without it, `--resume` would skip a
    step whose code changed."""
    base = tmp_path_factory.mktemp("resume")
    artifacts = base / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    write_rot_fixture(artifacts)
    wf_dir = base / "workflows"
    wf_dir.mkdir()
    scripts = wf_dir / "scripts"
    scripts.mkdir()
    script = scripts / "count.py"
    script.write_text(
        "import json\n"
        "def main(inputs, outputs):\n"
        "    from causalab.io.tables import read_table, write_table\n"
        "    rows = read_table(inputs['table'])\n"
        "    write_table(outputs['out'], [{'n': len(rows), 'tag': 'first'}])\n"
    )
    document = {
        "version": "1",
        "output_dir": "resume_probe",
        "steps": {
            "locate": {
                "type": "intervention_protocol",
                "document": f"{METHODS}/weekdays_locate_scan.json",
                "set": {
                    **_fixture_inputs("weekdays_locate_scan.json"),
                    "model.key": TINY_LLAMA,
                    "sites.target.layers": 0,
                    "positions.tap": {"index": -1},
                },
            },
            "count": {
                "type": "script",
                "script": {"path": "scripts/count.py"},
                "inputs": {"table": {"step": "locate", "file": "iia.json"}},
                "outputs": {
                    "out": {
                        "file": "count.json",
                        "columns": {"n": "int64", "tag": "string"},
                    }
                },
            },
        },
    }
    wf = wf_dir / "resume.json"
    wf.write_text(json.dumps(document, indent=2))
    assert _run(base, artifacts, wf) == 0
    run_root = base / "runs" / "resume_probe"
    assert table_frame(run_root / "count/count.json")["tag"].iloc[0] == "first"

    # unchanged document + unchanged script: --resume reuses both steps
    assert _run(base, artifacts, wf, "--resume") == 0
    manifest = json.loads((run_root / "workflow.json").read_text())
    assert manifest["steps"]["count"]["status"] == "reused"
    assert manifest["steps"]["locate"]["status"] == "reused"

    # edit only the SCRIPT: `script_sha256` is in the step's identity (§7),
    # so the step digest has moved and --resume re-runs that step alone
    script.write_text(script.read_text().replace("'first'", "'second'"))
    assert _run(base, artifacts, wf, "--resume") == 0
    assert table_frame(run_root / "count/count.json")["tag"].iloc[0] == "second"
    manifest = json.loads((run_root / "workflow.json").read_text())
    assert manifest["steps"]["count"]["status"] == "completed"
    assert manifest["steps"]["locate"]["status"] == "reused"
