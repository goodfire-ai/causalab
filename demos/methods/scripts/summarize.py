"""Reduce every run tree under ``demos/methods/runs/`` to a small results file.

``results/protocols/<name>.json`` (or ``results/workflows/<name>.json``) holds

    {document, document_digest, model, engine, device, dtype, date,
     elapsed_s, metrics}

where ``metrics`` is scalar summaries only: for every metric table a run
saved, the mean over eligible rows — and, when the table has sweep axes, the
best cell (the coordinates with the highest mean) and the mean at it. A
workflow contributes one such block per step plus whatever ``values.json`` a
select step emitted. ``document_digest`` is the *committed* document's digest,
computed the way ``causalab validate`` reports it, so
``tests/demos/test_methods.py`` can tell a results file from a stale one.

    uv run python demos/methods/scripts/summarize.py

Nothing here reads a tensor: bundles stay in the run tree, which is ignored.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

METHODS = Path(__file__).resolve().parents[1]
REPO = METHODS.parents[1]
RUNS = METHODS / "runs"
RESULTS = METHODS / "results"
#: Files in a run tree that are not metric tables.
NOT_TABLES = {
    "protocol.json",
    "_step.json",
    "_run_all.json",
    "workflow.json",
    "values.json",
    "fit_diagnostics.json",
    "routing_mismatch.json",
    "train_eval.json",
}
#: Scalar-bearing sidecars a fit writes, copied through when present.
SIDECARS = ("fit_diagnostics.json", "routing_mismatch.json", "values.json")


#: The committed digest pins of every shipped intervention specification, kept by
#: ``tests/protocol/test_shipped_digests.py`` against the same fixture
#: environment ``causalab validate`` uses offline. A results file quotes the
#: pin rather than recomputing it, so the chain is results -> pins ->
#: documents, and each link is one test.
PINS = REPO / "tests/protocol/shipped_digests.json"


def document_digest(document: Path) -> str:
    """An intervention specification's pinned campaign digest; a workflow's own digest,
    loaded against the repo root the way ``causalab validate`` loads it."""
    raw = json.loads(document.read_text())
    if "steps" in raw:
        from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
        from causalab.tasks import TASKS_ROOT
        from causalab.workflow.document import load_workflow

        env = ResolutionEnv(
            datasets=FileDatasets(root=TASKS_ROOT, fallback_roots=(TASKS_ROOT,)),
            artifacts=FileArtifacts(root=REPO),
        )
        return load_workflow(document, env).digest
    pins = json.loads(PINS.read_text())
    if document.name not in pins:
        # minimal_cpu.json: its model registers only from the HF config, so it
        # has no canonical form offline (test_shipped_digests.EXCLUDED)
        return "unpinned: not in tests/protocol/shipped_digests.json"
    return pins[document.name]["document"]


def run_digest(run_dir: Path) -> str | None:
    """The digest of the document *as run* (``--record``'s receipt): equal to
    the pin unless the document carries an artifact whose content the fixture
    environment stands in for (``das_pca_init.json``'s basis)."""
    receipt = run_dir / "protocol.json"
    if not receipt.is_file():
        return None
    return json.loads(receipt.read_text()).get("document_digest")


def spectrum_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A ``fit_pca`` spectrum: the variance the first 1, 4 and all components hold."""
    ratios = [r.get("explained_variance_ratio") for r in rows]
    if not ratios or any(not isinstance(v, (int, float)) for v in ratios):
        return None
    cumulative = []
    total = 0.0
    for v in ratios:
        total += float(v)
        cumulative.append(round(total, 4))
    return {
        "components": len(ratios),
        "explained_variance_ratio_cumulative": {
            "1": cumulative[0],
            "4": cumulative[min(3, len(cumulative) - 1)],
            str(len(cumulative)): cumulative[-1],
        },
    }


def table_summary(path: Path) -> dict[str, Any] | None:
    payload = json.loads(path.read_text())
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return None
    if "value" not in rows[0]:
        return None
    axes = sorted(
        k
        for k in rows[0]
        if k.split(".")[0] in ("sites", "positions", "featurizers", "train", "params")
    )
    by_cell: dict[tuple, list[float]] = defaultdict(list)
    n_rows = 0
    for row in rows:
        if row.get("eligible") is False or not isinstance(
            row.get("value"), (int, float)
        ):
            continue
        n_rows += 1
        by_cell[tuple(row.get(a) for a in axes)].append(float(row["value"]))
    if not by_cell:
        return {"rows": len(rows), "eligible_rows": 0}
    all_values = [v for vs in by_cell.values() for v in vs]
    out: dict[str, Any] = {
        "metric": rows[0].get("metric"),
        "unit": rows[0].get("unit"),
        "rows": len(rows),
        "eligible_rows": n_rows,
        "mean": round(statistics.fmean(all_values), 4),
    }
    if axes and len(by_cell) > 1:
        means = {cell: statistics.fmean(vs) for cell, vs in by_cell.items()}
        best = max(means, key=means.get)
        out["cells"] = len(means)
        out["best_cell"] = {a: _coord(c) for a, c in zip(axes, best)}
        out["best_mean"] = round(means[best], 4)
        out["worst_mean"] = round(min(means.values()), 4)
    return out


def _coord(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def safetensors_shapes(path: Path) -> dict[str, list[int]]:
    """Slot -> shape from the header alone; the weights are never read."""
    import struct

    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return {
        slot: spec["shape"] for slot, spec in header.items() if slot != "__metadata__"
    }


def text_table_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A ``top_k`` or ``decode`` table: no number to average, so the row count,
    how many matched their anchor, and the commonest first token."""
    from collections import Counter

    tokens: Counter[str] = Counter()
    for row in rows:
        value = row.get("value")
        if isinstance(value, str) and value.startswith("{"):
            try:
                tokens[json.loads(value)["tokens"][0]] += 1
            except (KeyError, IndexError, json.JSONDecodeError, TypeError):
                pass
    out: dict[str, Any] = {
        "metric": rows[0].get("metric"),
        "rows": len(rows),
        "matched_rows": sum(1 for r in rows if r.get("matched")),
        "numeric": False,
    }
    if tokens:
        out["commonest_top1"] = dict(tokens.most_common(3))
    return out


def train_eval_summary(path: Path) -> dict[str, Any]:
    """The held-out score a fit selected on, per point: one point flat, a
    sweep as the best point and the range."""
    points = json.loads(path.read_text())
    if not isinstance(points, list) or not points:
        return {}
    metric = next(iter(points[0].get("metrics", {})), None)
    if metric is None:
        return {"points": len(points)}
    scores = [
        (p["metrics"][metric], p) for p in points if metric in p.get("metrics", {})
    ]
    best_score, best = max(scores, key=lambda sp: sp[0])
    out = {
        "metric": metric,
        "split": best.get("split"),
        "selected": best.get("selected"),
        "points": len(points),
        "best": round(best_score, 4),
    }
    if len(points) > 1:
        out["best_coords"] = best.get("coords")
        out["worst"] = round(min(sc for sc, _ in scores), 4)
    else:
        out["passes"] = best.get("passes")
    return out


#: The fit-diagnostic scalars worth keeping per featurizer.
DIAGNOSTIC_KEYS = (
    "width",
    "groups",
    "decisive_fraction",
    "hard_mask_size",
    "boundary",
    "frozen_units",
)


def fit_diagnostics_summary(path: Path) -> dict[str, Any]:
    """Per featurizer: the scalars ``dbm.json``'s description says to read
    before believing a number (``decisive_fraction``, ``hard_mask_size``);
    across a sweep, their ranges."""
    points = json.loads(path.read_text())
    if not isinstance(points, list) or not points:
        return {}
    out: dict[str, Any] = {"points": len(points)}
    names = {name for p in points for name in p.get("featurizers", {})}
    for name in sorted(names):
        block: dict[str, Any] = {}
        for key in DIAGNOSTIC_KEYS:
            values = [
                p["featurizers"][name][key]
                for p in points
                if key in p.get("featurizers", {}).get(name, {})
            ]
            if not values:
                continue
            block[key] = (
                round(values[0], 4)
                if len(values) == 1
                else {"min": round(min(values), 4), "max": round(max(values), 4)}
            )
        out[name] = block
    return out


def routing_mismatch_summary(path: Path) -> dict[str, Any]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        return {}
    mismatched = sum(r.get("mismatched", 0) for r in rows)
    slots = sum(r.get("slots", 0) for r in rows)
    return {
        "examples": len(rows),
        "mismatched_slots": mismatched,
        "slots": slots,
        "fraction": round(mismatched / slots, 4) if slots else None,
    }


def values_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {"values": payload}


REDUCERS = {
    "train_eval.json": train_eval_summary,
    "fit_diagnostics.json": fit_diagnostics_summary,
    "routing_mismatch.json": routing_mismatch_summary,
    "values.json": values_summary,
}


def summarize_dir(directory: Path) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for table in sorted(directory.glob("*.json")):
        if table.name in REDUCERS:
            reduced = REDUCERS[table.name](table)
            if reduced:
                block[table.stem] = reduced
            continue
        if table.name in NOT_TABLES:
            continue
        payload = json.loads(table.read_text())
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if isinstance(rows, list) and rows and "explained_variance_ratio" in rows[0]:
            summary = spectrum_summary(rows)
            if summary is not None:
                block[table.stem] = summary
            continue
        summary = table_summary(table)
        if (
            summary is not None
            and summary.get("eligible_rows") == 0
            and isinstance(rows, list)
            and rows
        ):
            summary = text_table_summary(rows)
        if summary is not None:
            block[table.stem] = summary
    tensors = {}
    for bundle in sorted(directory.glob("*.safetensors")):
        tensors[bundle.name] = safetensors_shapes(bundle)
    if tensors:
        block["saved_tensors"] = tensors
    return block


def summarize_run(run_dir: Path, kind: str) -> dict[str, Any] | None:
    sidecar_path = run_dir / "_run_all.json"
    if not sidecar_path.is_file():
        return None
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("returncode") != 0:
        return None
    document = METHODS / sidecar["document"]
    metrics: dict[str, Any]
    if kind == "workflow":
        metrics = {}
        for step in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            block = summarize_dir(step)
            if block:
                metrics[step.name] = block
    else:
        metrics = summarize_dir(run_dir)
    model = sidecar["model"]
    if model is None and kind == "workflow":
        # a workflow has no model block; its steps' documents do
        raw = json.loads(document.read_text())
        models = sorted(
            {
                json.loads((document.parent / step["document"]).read_text())["model"][
                    "key"
                ]
                for step in raw["steps"].values()
                if "document" in step
            }
        )
        model = models[0] if len(models) == 1 else models
    return {
        "document": sidecar["document"],
        "document_digest": document_digest(document),
        "run_digest": run_digest(run_dir),
        "model": model,
        "engine": sidecar["engine"],
        "device": sidecar["device"],
        "dtype": sidecar["dtype"],
        "date": sidecar["date"],
        "elapsed_s": sidecar["elapsed_s"],
        "metrics": metrics,
    }


#: Documents that run only as a step of a workflow — their inputs are another
#: step's outputs — and the (workflow run, step) whose tree reports on them.
STEP_RUNS = {
    "weekdays_das_sweep": ("weekdays", "fit"),
    "weekdays_das_apply": ("weekdays", "apply"),
    "pca_harvest": ("pca_basis", "harvest"),
}


def summarize_step_run(name: str, workflow: str, step: str) -> dict[str, Any] | None:
    """A results file for a document from the workflow step that ran it: the
    workflow's sidecar for the run facts, the step's own tree for the metrics."""
    run_dir = RUNS / "workflows" / workflow
    sidecar_path = run_dir / "_run_all.json"
    step_dir = run_dir / step
    if not sidecar_path.is_file() or not step_dir.is_dir():
        return None
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("returncode") != 0:
        return None
    document = METHODS / "protocols" / f"{name}.json"
    record = (
        json.loads((step_dir / "_step.json").read_text())
        if (step_dir / "_step.json").is_file()
        else {}
    )
    raw = json.loads(document.read_text())
    return {
        "document": f"protocols/{name}.json",
        "document_digest": document_digest(document),
        "run_digest": record.get("document_digest"),
        "ran_as": f"workflows/{workflow}.json#{step}",
        "model": raw["model"]["key"],
        "engine": record.get("engine", sidecar["engine"]),
        "device": sidecar["device"],
        "dtype": raw["model"].get("dtype", "engine default"),
        "date": sidecar["date"],
        "elapsed_s": None,  # the workflow's sidecar times the whole chain
        "metrics": summarize_dir(step_dir),
    }


def main() -> int:
    written = 0
    for name, (workflow, step) in STEP_RUNS.items():
        result = summarize_step_run(name, workflow, step)
        if result is None:
            continue
        target = RESULTS / "protocols" / f"{name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(result, indent=2) + "\n"
        assert len(text) < 8000, f"{target} is {len(text)} bytes; reduce further"
        target.write_text(text)
        written += 1
        print(f"wrote {target.relative_to(REPO)} ({len(text)} bytes)")
    for kind, root in (("protocol", RUNS), ("workflow", RUNS / "workflows")):
        if not root.is_dir():
            continue
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if run_dir.name in ("workflows", "artifacts") or run_dir.name.startswith(
                "_"
            ):
                continue
            result = summarize_run(run_dir, kind)
            if result is None:
                continue
            target = (
                RESULTS
                / ("protocols" if kind == "protocol" else "workflows")
                / f"{run_dir.name}.json"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(result, indent=2, sort_keys=False) + "\n"
            assert len(text) < 8000, f"{target} is {len(text)} bytes; reduce further"
            target.write_text(text)
            written += 1
            print(f"wrote {target.relative_to(REPO)} ({len(text)} bytes)")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
