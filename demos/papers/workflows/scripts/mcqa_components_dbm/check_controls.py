"""Score the two fitted masks against their controls on the held-out pairs.

A fitted DBM mask keeps ``k`` of ``n`` units and scores some IIA on the
held-out pairs. Two controls say what that number means:

* ``full``: every unit kept. For the component gates this swaps all 56
  attention and MLP outputs at the answer slot; for the head gate it swaps all
  12 heads of layer 22, which is the layer's whole attention output.
* ``random``: ``k`` units drawn uniformly from the ``n``, one draw per seed.
  The draw spans all 56 gates of the component fit together, because each of
  its bundles holds one unit and a per-bundle draw
  (``causalab.analysis.random_mask``) could only keep or drop that unit.

A third run, ``replay``, writes the fitted mask itself back at the same hard
split. Its IIA must equal the ``apply`` step's to within ``REPLAY_TOLERANCE``,
which checks that the control bundles load and score the way the fitted ones
do; the script stops on a mismatch before it writes the summary. Each of the workflow's six
fit steps (two variants at three l1 weights) gets its own three controls.

Each control is a copy of the fitted bundles with ``theta`` set to +1 on the
kept units and -1 on the others, the hard ``theta > 0`` split of a sigmoid
gate one unit either side of the threshold, as ``causalab.analysis.random_mask``
writes it; every header field is copied, so the apply documents load the
copies at the address they name. Each control runs the package's apply
document with ``--artifacts-root`` pointed at the copies.

Run after the workflow, from ``demos/papers/``, on the machine that ran it::

    python workflows/scripts/mcqa_components_dbm/check_controls.py --device cuda

It writes ``controls/`` into the run tree and the summary
``artifacts/figures/mcqa_components_dbm/controls.json``: per mask, the held-out
IIA of the fit, the replay, the full swap and every random draw.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

__all__ = ["main"]

#: ``demos/papers/``: this script sits in ``workflows/scripts/mcqa_components_dbm/``.
PAPERS = Path(__file__).resolve().parents[3]
RUN = PAPERS / "artifacts" / "output" / "mcqa_components_dbm"
FIGURES = PAPERS / "artifacts" / "figures" / "mcqa_components_dbm"
PROTOCOLS = PAPERS / "protocols"
#: The l1 weights of the workflow's fit steps, by step suffix: onboarding 09's three.
ARMS: dict[str, float] = {"": 0.01, "_mid": 0.3, "_hi": 3.0}
#: The largest difference allowed between the replay's IIA and the ``apply``
#: step's. Both score the same mask on the same pairs, so any difference past
#: float noise means a control bundle loads or scores differently.
REPLAY_TOLERANCE = 1e-9
#: variant -> (fit step prefix, the fit directory its apply document names,
#: apply document). The component fit saves one bundle per gate, the head fit
#: one bundle of 12 heads.
VARIANTS: dict[str, tuple[str, str, str]] = {
    "components": ("", "fit", "mcqa_components_dbm_apply.json"),
    "heads": ("head_", "head_fit", "mcqa_components_dbm_head_apply.json"),
}


def mean_iia(table: Path) -> float:
    """The mean of a per-example ``match`` table."""
    rows = json.loads(table.read_text())
    values = [float(r["value"]) for r in rows if r.get("eligible", True)]
    return sum(values) / len(values)


def fitted_units(fit: Path) -> list[tuple[str, int, bool]]:
    """(bundle file, unit, kept) for every unit the fit trained, the kept
    flag being the fit's own hard split from its ``rank.json``.

    The units come sorted by bundle file name as a string, then by unit, so
    ``g_attn10`` comes before ``g_attn2``. ``check`` draws the random masks
    from a population in this same order, so the order sets which units a
    seed draws."""
    rows = json.loads((fit / "rank.json").read_text())
    units = []
    for row in rows:
        name = row["featurizer"]
        bundle = "gate.safetensors" if name == "gate" else f"{name}.safetensors"
        units.append((bundle, int(row["unit"]), bool(row["hard"])))
    return sorted(units)


def write_control(fit: Path, root: Path, step: str, kept: set[tuple[str, int]]) -> None:
    """Copies of the fit's bundles under ``root/<step>/`` with ``theta`` at +1
    on ``kept`` and -1 elsewhere, every header field unchanged."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    target = root / step
    target.mkdir(parents=True, exist_ok=True)
    for bundle in sorted({b for b, _ in _all_units(fit)}):
        with safe_open(str(fit / bundle), framework="pt") as fh:
            metadata = dict(fh.metadata())
            tensors = {key: fh.get_tensor(key) for key in fh.keys()}
        theta = tensors["theta"]
        flat = torch.full((theta.numel(),), -1.0, dtype=theta.dtype)
        for unit in range(theta.numel()):
            if (bundle, unit) in kept:
                flat[unit] = 1.0
        tensors["theta"] = flat.reshape(theta.shape)
        save_file(tensors, str(target / bundle), metadata=metadata)


def _all_units(fit: Path) -> list[tuple[str, int]]:
    return [(b, u) for b, u, _ in fitted_units(fit)]


def _order(unit: tuple[str, int]) -> tuple[str, int, int]:
    """Bundles by component, then layer, then unit: g_attn2 before g_attn10."""
    stem = unit[0].removesuffix(".safetensors")
    digits = "".join(c for c in stem if c.isdigit())
    return (stem.rstrip("0123456789"), int(digits or 0), unit[1])


def label(bundle: str, unit: int, mask: str) -> str:
    """``g_attn22`` for a component gate, ``head 7`` for a head of the head gate."""
    return f"head {unit}" if mask == "heads" else bundle.removesuffix(".safetensors")


def score(document: str, artifacts_root: Path, out: Path, device: str) -> float:
    """Run one apply document against the bundles under ``artifacts_root``
    and return its held-out IIA."""
    if out.exists():
        shutil.rmtree(out)
    command = [
        sys.executable,
        "-m",
        "causalab.cli",
        "run",
        str(PROTOCOLS / document),
        "--engine",
        "auto",
        "--data-root",
        str(PAPERS / "artifacts" / "data"),
        "--artifacts-root",
        str(artifacts_root),
        "--out",
        str(out),
        "--device",
        device,
    ]
    subprocess.run(command, check=True, cwd=PAPERS)
    return mean_iia(out / "iia.json")


def check(run: Path, device: str, draws: int) -> dict[str, Any]:
    """Every control of every fit step of the workflow, keyed
    ``<variant><suffix>`` (``components``, ``heads_hi``, ...)."""
    summary: dict[str, Any] = {}
    for variant, (prefix, named_dir, document) in VARIANTS.items():
        for suffix, weight in ARMS.items():
            mask = f"{variant}{suffix}"
            fit = run / f"{prefix}fit{suffix}"
            units = fitted_units(fit)
            everything = {(b, u) for b, u, _ in units}
            fitted = {(b, u) for b, u, k in units if k}
            controls = {"replay": fitted, "full": everything}
            population = sorted(everything)
            for seed in range(draws):
                drawn = random.Random(seed).sample(population, len(fitted))
                controls[f"random_{seed}"] = set(drawn)
            fitted_iia = mean_iia(run / f"{prefix}apply{suffix}" / "iia.json")
            scores: dict[str, float] = {}
            for name, kept in controls.items():
                root = run / "controls" / mask / name
                # the apply document names its fit's bundles under
                # `named_dir`, so the copies go there
                write_control(fit, root, named_dir, kept)
                scores[name] = score(document, root, root / "out", device)
                # the replay runs first: stop before the random draws when
                # the control bundles do not reproduce the fit
                if (
                    name == "replay"
                    and abs(scores[name] - fitted_iia) > REPLAY_TOLERANCE
                ):
                    raise SystemExit(
                        f"{mask}: replay {scores[name]} != apply {fitted_iia}"
                    )
            randoms = [scores[f"random_{s}"] for s in range(draws)]
            summary[mask] = {
                "l1_weight": weight,
                "units": len(everything),
                "kept": [label(b, u, variant) for b, u in sorted(fitted, key=_order)],
                "iia_fitted": fitted_iia,
                "iia_replay": scores["replay"],
                "iia_full": scores["full"],
                "iia_random": randoms,
                "iia_random_mean": sum(randoms) / len(randoms),
                "random_seeds": list(range(draws)),
            }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--run", type=Path, default=RUN, help="the workflow's output directory"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--draws", type=int, default=10, help="random masks per fit")
    parser.add_argument(
        "--summary",
        type=Path,
        default=FIGURES / "controls.json",
        help="where the summary goes",
    )
    args = parser.parse_args(argv)
    summary = check(args.run, args.device, args.draws)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
