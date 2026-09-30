"""Single-source CLI with real isolated workers, outputs, and native CPU traces."""

import json
from pathlib import Path
import re

import pytest

from causalab.cli import main
from causalab.measurement.analysis.receipts import load_measurement
from examples.measurements.single import prepare

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("observations", [False, True])
def test_single_source_workflow_collection_capture_and_resume(
    tmp_path, monkeypatch, observations
):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    document, bindings = prepare(tmp_path / "inputs", observations=observations)
    output = tmp_path / "run"
    arguments = [
        "measure",
        str(document),
        "--bindings",
        str(bindings),
        "--out",
        str(output),
    ]
    assert main(arguments) == 0
    assert {p.name for p in (output / "deployment").iterdir()} == {"source"}
    index = json.loads((output / "reports/study.json").read_text())
    assert set(index["cases"]) == {"cold", "resident"}
    for case in ("cold", "resident"):
        record, samples = load_measurement(
            output / f"collections/{case}.source.json",
            require_observations=observations,
        )
        assert len(samples) == 1
        sample = samples[0, 0]
        assert sample["seconds"] > 0
        assert any(
            path.endswith("logits.safetensors") for path in sample["output_files"]
        )
        for path in sample["output_files"]:
            assert (output / "collections" / path).is_file()
        if not observations:
            assert sample["observation_status"] == "not_requested"
            assert "observations" not in sample
            assert not list((output / "collections/blocks").glob("**/*_numerics"))
        else:
            assert sample["tensors"]
        report = index["cases"][case]["measurement"]
        assert report["collection_status"] == "completed"
        captures = report["captures"]
        assert {c["mode"] for c in captures} == (
            {"cold", "warm"} if case == "cold" else {"warm"}
        )
        for capture in captures:
            assert capture["status"] == "completed", capture
            assert capture["observation_check"]["status"] == (
                "compared" if observations else "not_requested"
            )
            from urllib.parse import unquote

            trace_path = output / "reports" / unquote(capture["artifact_links"][0])
            trace = json.loads(trace_path.read_text())
            names = {event.get("name") for event in trace["traceEvents"]}
            assert "step:inference" in names
        assert "comparisons" not in index["cases"][case]
    for report_path in (output / "reports").glob("*.html"):
        for link in re.findall(r'href="([^"]+)"', report_path.read_text()):
            if not link.startswith(("#", "https://", "http://")):
                from urllib.parse import unquote, urlparse

                target = urlparse(link)
                path = Path(unquote(target.path))
                assert (
                    path if path.is_absolute() else report_path.parent / path
                ).exists()
    publications = {
        path: path.read_bytes()
        for name in ("block.json", "publication.json")
        for path in output.rglob(name)
    }
    assert publications
    assert main([*arguments, "--resume"]) == 0
    assert all(path.read_bytes() == data for path, data in publications.items())
    # Saved workflow outputs remain integrity checked even without observations.
    victim = next(
        path for path in sample["output_files"] if path.endswith("logits.safetensors")
    )
    (output / "collections" / victim).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed|mismatch"):
        from causalab.measurement.study.controller import run

        run(document, bindings, output, resume=True)
