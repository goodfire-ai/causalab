"""Capture (or check) the parallel golden record on the 2-GPU node.

    HF_HUB_OFFLINE=1 uv run python tests/golden/update_parallel_goldens.py \\
        --i-have-reviewed-the-diff [--only inference,das,dbm,das_dense] [--keep ROOT]

Runs every document of the parallel golden (``tests/golden/_parallel``) on
the real ``Qwen/Qwen3.6-35B-A3B`` — the inference document at world 1 and
at ``dp=2``, ``pp=2``, ``tp=2``, ``ep=2``; the DAS fit at ``attention_query``
at ``pp=2``, ``tp=2``, ``dp=2:rows``; the expert-neuron DBM fit at
``expert_activation`` at ``pp=2``, ``ep=2`` — and the DAS fit on the dense
``Qwen/Qwen3-4B-Instruct-2507`` in fp32 (``das_dense``) at ``tp=2``,
``dp=2:rows`` — as ``causalab run … --device
cuda`` subprocesses, one geometry at a time, measures every output class
against the world-1 run, and writes ``tests/golden/parallel_goldens.json``:
per document and banded geometry the classes' measured maximum, scale, dtype
and band (``max(3 × max, 2 × ulp(dtype, scale), 1e-3)``), the exact
geometries' measured ``0.0``, the loader's bytes and the fits' peak memory
per rank, the capture's context and the rule's justification. It **refuses
to write** when a receipt differs beyond ``execution.parallel``, when any
rank's load report is not ``1 / world`` for its sharded parameters, when a
geometry the design holds exact is not — byte for byte, and every class
(the recorded gradients included) at ``0.0`` — unless ``--explain-inexact
'das pp=2: <why>'`` moves it to the banded set with the explanation written
into the record, or without the review flag after printing the per-class
diff against the committed record — the idiom of
``tests/golden/drift/update_drift_goldens.py`` and
``tests/protocol/update_corpus_digests.py``. ``--only`` captures the named
documents and keeps the committed record's others — the second-family
documents (``tests/golden/_parallel/families.py``: ``inference_gemma2_9b``,
``inference_llama31_8b``, each on its own model) are captured by name
alone, never by a bare capture; ``--keep ROOT`` runs
under ``ROOT`` and resumes any run whose receipt is there; ``--check``
re-measures and exits non-zero if the committed record does not replay,
writing nothing.

The large model's documents (``tests/golden/_parallel/large.py``:
``large``, ``das_large`` on ``meta-llama/Llama-3.1-70B``, eight cards,
``pp=4`` the oracle in place of world 1) are captured by name only —
``--only large,das_large`` — and the device gate is the largest world the
named documents need; their blocks carry the pre-flight's ``estimate``
beside the measured ``memory`` and refuse a peak above the estimate.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from tests.golden import _parallel as par

#: Every document the capture knows: the two-rank record's four by default,
#: the large model's two (eight cards, ``--only large,das_large``).


class BadExplanation(ValueError):
    """An ``--explain-inexact`` argument not of the form ``<doc> <geometry>: <why>``."""


def parse_explanations(items: list[str]) -> dict[str, dict[str, str]]:
    """``['das pp=2: the head ...']`` → ``{"das": {"pp=2": "the head ..."}}``."""
    out: dict[str, dict[str, str]] = {}
    for item in items:
        head, sep, why = item.partition(":")
        parts = head.split()
        if not sep or len(parts) != 2 or not why.strip():
            raise BadExplanation(
                f"--explain-inexact {item!r} is not '<document> <geometry>: <why>'"
            )
        name, geometry = parts
        if name not in par.CAPTURABLE:
            raise BadExplanation(f"--explain-inexact names no document {name!r}")
        out.setdefault(name, {})[geometry] = why.strip()
    return out


def print_diff(
    committed: dict[str, Any], fresh: dict[str, Any], names: list[str]
) -> None:
    for name in names:
        block = fresh["documents"][name]
        before = (
            committed.get("documents", {}).get(name, {})
            if committed.get("format") == par.FORMAT
            else {}
        )
        for geometry, classes in block["geometries"].items():
            for kind, entry in classes.items():
                value = par.entry_value(kind, entry)
                old = before.get("geometries", {}).get(geometry, {}).get(kind)
                where = (
                    f" at scale {entry['scale']!r} {entry['dtype']}"
                    if kind != par.ROUTING
                    else ""
                )
                if old is None:
                    print(
                        f"  + {name} {geometry} {kind}: {value!r}{where}, band {entry['band']!r}"
                    )
                else:
                    print(
                        f"    {name} {geometry} {kind}: {par.entry_value(kind, old)!r} -> "
                        f"{value!r}{where}, band {old['band']!r} -> {entry['band']!r}"
                    )
        for geometry, worst in block["exact"].items():
            print(f"    {name} {geometry}: exact ({worst!r})")
        for geometry, why in block["inexact"].items():
            print(f"  ! {name} {geometry}: inexact, explained: {why}")
        for geometry, ranks in block.get("memory", {}).items():
            peaks = {r: v["peak_bytes_allocated"] for r, v in ranks.items()}
            print(f"    {name} {geometry}: peak bytes allocated {peaks}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="verify, do not write")
    parser.add_argument("--i-have-reviewed-the-diff", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="write here instead")
    parser.add_argument(
        "--keep", type=Path, default=None, help="run under this directory and keep it"
    )
    parser.add_argument(
        "--only",
        default=",".join(par.DOCUMENTS),
        help=(
            "the documents to capture, comma-separated (default: the A3B tier, "
            f"{', '.join(par.DOCUMENTS)}; the family documents "
            f"{', '.join(par.families.FAMILIES)} by name; the large model's "
            f"{', '.join(par.LARGE_DOCUMENTS)} by name, on eight CUDA devices)"
        ),
    )
    parser.add_argument(
        "--explain-inexact",
        action="append",
        default=[],
        metavar="'DOC GEOMETRY: WHY'",
        help="hold an exact geometry banded instead, writing the explanation",
    )
    args = parser.parse_args(argv)
    names = [n for n in args.only.split(",") if n]
    unknown = [n for n in names if n not in par.CAPTURABLE]
    if unknown:
        print(
            f"refused: no document {unknown}; the documents are {list(par.CAPTURABLE)}",
            file=sys.stderr,
        )
        return 2
    try:
        explanations = parse_explanations(args.explain_inexact)
    except BadExplanation as bad:
        print(f"refused: {bad}", file=sys.stderr)
        return 2
    needed = max(par.large.devices_needed(par.CAPTURABLE[name]) for name in names)
    if torch.cuda.device_count() < needed:
        print(
            f"refused: capturing {names} needs {needed} CUDA devices; this node "
            f"shows {torch.cuda.device_count()}",
            file=sys.stderr,
        )
        return 2
    root = args.keep or Path(tempfile.mkdtemp(prefix="parallel-goldens-"))
    root.mkdir(parents=True, exist_ok=True)
    committed = (
        par.load_record() if par.RECORD.exists() else par.pending_record(par.A3B)
    )
    blocks: dict[str, dict[str, Any]] = {}
    refusals: list[str] = []
    for name in names:
        document = par.CAPTURABLE[name]
        block, problems = par.capture(
            root, document, par.realization_of(document), explanations.get(name, {})
        )
        blocks[name] = block
        refusals += [f"{name} {p}" for p in problems]
    if refusals:
        print(
            "the capture is not certifiable; the record is not written:",
            file=sys.stderr,
        )
        for refusal in refusals:
            print(f"  {refusal}", file=sys.stderr)
        return 2
    record = par.make_record(committed, par.A3B, blocks)
    target = args.out or par.RECORD
    if args.check:
        problems: list[str] = []
        for name in names:
            try:
                problems += par.compare_records(committed, record, name)
            except par.StaleRecord as stale:
                problems.append(str(stale))
        if problems:
            print(f"{target} does not replay:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        print(f"{target} replays")
        return 0
    if committed.get("format") != par.FORMAT:
        print(
            f"the committed record is format {committed.get('format')!r}; the new "
            f"record is format {par.FORMAT} and carries only the documents captured now"
        )
    print_diff(committed, record, names)
    if not args.i_have_reviewed_the_diff:
        print("dry run: pass --i-have-reviewed-the-diff to write the record")
        return 1
    target.write_text(par.render(record))
    print(f"wrote {target} (outputs under {root})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
