"""Paper-golden tier: real-model runs asserted against paper-provenance values.

Every asserted number in tests/golden/paper_goldens.json traces to a
published paper figure/table — never to a pinning run of this stack. Documents live in tests/golden/protocols/ (identity pinned in
golden_digests.json); fixture datasets are seeded, committed JSON produced
by tests/golden/fixtures/generators/.

Run on a GPU box:

    uv run pytest tests/golden -m golden

Gated models (meta-llama/Llama-3.1-8B, google/gemma-2-2b-it) need a
licensed ``HF_TOKEN`` in the environment — without one the load 401s with
nothing naming the cause. CUDA is preferred; Apple MPS works for the
smaller documents (the 1,152-row hours document wants ~35GB free).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from causalab.cli import main

from causalab.io.tables import read_table
from tests.golden._env import FIXTURES, GOLDEN_PROTOCOLS, GOLDENS_FILE


def _frame(path: Path) -> pd.DataFrame:
    """One JSON metric table as a DataFrame — tables are JSON on disk."""
    return pd.DataFrame(read_table(path))


pytestmark = pytest.mark.golden

GOLDENS = json.loads(GOLDENS_FILE.read_text())["goldens"]

# golden ids each test claims; test_structural cross-checks this registry
# against paper_goldens.json so no value can go silently unasserted
COVERED: dict[str, tuple[str, ...]] = {
    "test_ioi_clean_logit_diff": ("ioi.clean_logit_diff", "ioi.io_over_s_rate"),
    "test_hours_baseline_accuracy": ("arithmetic.hours_baseline_acc",),
    "test_rome_average_total_effect": ("rome.ate",),
    "test_rome_hidden_state_aie_peak": ("rome.hidden_aie_peak",),
    "test_rome_mlp_window_aie_peak": ("rome.mlp_window_aie_peak",),
    "test_mixing_positional_shares": (
        "mixing.positional_share_edges",
        "mixing.positional_share_middle",
    ),
    "test_arithmetic_steering_diagonal": ("arithmetic.steering_top1_fraction",),
}


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    pytest.skip("golden tier needs an accelerator (cuda or mps)")


def skip_if_pending(golden_id: str) -> None:
    """A pending goldens entry has open provenance/calibration questions
    (recorded in its notes); its test stays wired but skips."""
    if GOLDENS[golden_id].get("pending"):
        pytest.skip(f"{golden_id} is pending: {GOLDENS[golden_id]['notes'][:160]}…")


def assert_golden(golden_id: str, measured: float) -> None:
    """One assertion path for every golden: prints measured next to the
    band so a near-miss is diagnosable straight from the log."""
    entry = GOLDENS[golden_id]
    if entry["sidedness"] == "at_least":
        floor = entry["floor"]
        print(f"{golden_id}: measured {measured:.4f}, floor {floor}")
        assert measured >= floor, f"{golden_id}: {measured:.4f} < floor {floor}"
    else:
        lo, hi = entry["band"]
        print(f"{golden_id}: measured {measured:.4f}, band [{lo}, {hi}]")
        assert lo <= measured <= hi, (
            f"{golden_id}: {measured:.4f} outside [{lo}, {hi}] "
            f"(paper value {entry['value']})"
        )


def run_document(name: str, out: Path, **extra: str) -> None:
    """Run one golden document through the real CLI. Precision is the
    document's own (§2.1) — the pinned digest covers it, so a golden cannot
    be measured at a precision its record does not name."""
    argv = [
        "run",
        "--engine",
        "auto",
        str(GOLDEN_PROTOCOLS / name),
        "--data-root",
        str(FIXTURES / "data"),
        "--artifacts-root",
        str(out / "artifacts"),
        "--out",
        str(out),
        "--device",
        _device(),
    ]
    for key, value in extra.items():
        argv += ["--set", f"{key}={value}"]
    assert main(argv) == 0


def test_ioi_clean_logit_diff(tmp_path):
    run_document("ioi_clean_logit_diff_im.json", tmp_path)
    ld = _frame(tmp_path / "logit_diff.json")["value"]
    assert len(ld) == 512
    assert_golden("ioi.clean_logit_diff", float(ld.mean()))
    assert_golden("ioi.io_over_s_rate", float((ld > 0).mean()))


def test_hours_baseline_accuracy(tmp_path):
    # precision is the document's own now (§2.1) — hours_baseline_im.json
    # declares bf16, and the pinned digest covers it
    run_document("hours_baseline_im.json", tmp_path)
    acc = _frame(tmp_path / "acc.json")["value"]
    assert len(acc) == 1152
    assert_golden("arithmetic.hours_baseline_acc", float(acc.mean()))


def test_rome_average_total_effect(tmp_path):
    """ATE = mean over facts and noise seeds of P_clean - P_corrupted,
    recovered from the cross_entropy tables (p = exp(-ce)); the clean
    read is off the seed axis, so its per-seed rows are identical and the
    per-example mean collapses them.

    One document per subject-token width (the engine refuses ragged
    edits); pooling the per-fact effects across the width shards restores
    the loose-filter width distribution — a single-width sample biases the
    ATE (diagnosed +16pts on the width-4 bucket alone)."""
    import numpy as np

    effects = []
    clean_all, corr_all = [], []
    for width in (2, 3, 4, 5):
        out = tmp_path / f"w{width}"
        run_document(f"rome_ate_w{width}_im.json", out)
        ce_clean = _frame(out / "ce_clean.json")
        ce_corr = _frame(out / "ce_corr.json")
        p_clean = np.exp(-ce_clean.groupby("example_id")["value"].mean())
        p_corr = (
            ce_corr.assign(p=np.exp(-ce_corr["value"]))
            .groupby("example_id")["p"]
            .mean()
        )
        effects.append(p_clean - p_corr)
        clean_all.append(p_clean)
        corr_all.append(p_corr)
    pooled = pd.concat(effects)
    print(
        f"n={len(pooled)}; clean baseline {pd.concat(clean_all).mean():.4f}, "
        f"corrupted {pd.concat(corr_all).mean():.4f} (paper: 0.270 / 0.0847)"
    )
    assert_golden("rome.ate", float(pooled.mean() * 100))


def _axis_column(frame: pd.DataFrame, needle: str) -> str:
    matches = [c for c in frame.columns if needle in c]
    assert len(matches) == 1, f"axis column {needle!r}: {list(frame.columns)}"
    return matches[0]


def test_rome_hidden_state_aie_peak(tmp_path):
    """AIE(l) = mean over facts and seeds of P_restored(l) - P_corrupted,
    pooled over the width shards; peak asserted against Fig 2a."""
    import numpy as np

    per_layer: dict[int, list[pd.Series]] = {}
    for width in (2, 3, 4, 5):
        out = tmp_path / f"w{width}"
        run_document(f"rome_aie_w{width}_im.json", out)
        ce_corr = _frame(out / "ce_corr.json")
        ce_rest = _frame(out / "ce_rest.json")
        p_corr = (
            ce_corr.assign(p=np.exp(-ce_corr["value"]))
            .groupby("example_id")["p"]
            .mean()
        )
        layer_col = _axis_column(ce_rest, "hidden.layers")
        rest = ce_rest.assign(p=np.exp(-ce_rest["value"]))
        for layer, group in rest.groupby(layer_col):
            p_rest = group.groupby("example_id")["p"].mean()
            per_layer.setdefault(int(layer), []).append(p_rest - p_corr)
    aie = {layer: pd.concat(parts).mean() * 100 for layer, parts in per_layer.items()}
    peak_layer = max(aie, key=aie.get)
    print(f"AIE by layer: { {k: round(v, 2) for k, v in sorted(aie.items())} }")
    print(f"peak layer {peak_layer} (paper: 15)")
    assert peak_layer < 32, f"peak at layer {peak_layer} — not early-to-middle"
    assert_golden("rome.hidden_aie_peak", float(aie[peak_layer]))


def test_rome_mlp_window_aie_peak(tmp_path):
    """Fig 2b: restore a 10-layer MLP window [layer-4, layer+5] (clipped) at the
    last subject token in the corrupted run. Ten simultaneous swap sites
    cannot ride one sweep axis, so one document per window center is
    generated here (the parity-goldens build-doc-per-case pattern) —
    generated documents are not digest-pinned, unlike the authored ones."""
    import json as json_lib

    import numpy as np

    centers = list(range(4, 48, 5))
    aie: dict[int, list[pd.Series]] = {}
    for width in (2, 3, 4, 5):
        for center in centers:
            layers = [
                layer for layer in range(center - 4, center + 6) if 0 <= layer < 48
            ]
            doc = {
                "header": {
                    "protocol_version": "4",
                    "description": f"generated: ROME MLP window restore, center {center}, width-{width} shard",
                },
                "model": {"key": "gpt2-xl", "revision": "main", "dtype": "fp32"},
                "data": {
                    "base": {"dataset": f"counterfact/facts_w{width}", "field": "input"}
                },
                "method": {
                    "positions": {
                        "last_subject": {"index": -1, "scope": {"variable": "subject"}}
                    },
                    "sites": {
                        "emb": {"component": "embeddings"},
                        "lm_head": {"component": "lm_head"},
                        **{
                            f"mlp{layer}": {
                                "component": "mlp_output",
                                "layers": [layer],
                            }
                            for layer in layers
                        },
                    },
                    "reads": {
                        "logits_corr": {"site": "lm_head", "pos": -1},
                        "logits_rest": {"site": "lm_head", "pos": -1},
                        **{
                            f"v{layer}": {"site": f"mlp{layer}", "pos": "last_subject"}
                            for layer in layers
                        },
                    },
                    "writes": {
                        "noise": {
                            "site": "emb",
                            "pos": {"variable": "subject"},
                            "do": {
                                "gaussian": {
                                    "seed": 7,
                                    "scale": 0.144681,
                                    "axis": "tp_duplicated",
                                }
                            },
                        },
                        **{
                            f"rest{layer}": {
                                "site": f"mlp{layer}",
                                "pos": "last_subject",
                                "do": {"swap": f"v{layer}"},
                            }
                            for layer in layers
                        },
                    },
                    "intervened_models": {
                        # the clean window values, read on the un-intervened network
                        "original": {
                            "input": "base",
                            "reads": [f"v{layer}" for layer in layers],
                        },
                        "corrupted": {
                            "input": "base",
                            "reads": ["logits_corr"],
                            "writes": ["noise"],
                        },
                        "restored": {
                            "input": "base",
                            "reads": ["logits_rest"],
                            "writes": ["noise"] + [f"rest{layer}" for layer in layers],
                        },
                    },
                    "save": [
                        {
                            "read": "logits_corr",
                            "model": "corrupted",
                            "aggregation": {
                                "kind": "cross_entropy",
                                "target": "answer",
                            },
                            "file_path": "ce_corr.json",
                        },
                        {
                            "read": "logits_rest",
                            "model": "restored",
                            "aggregation": {
                                "kind": "cross_entropy",
                                "target": "answer",
                            },
                            "file_path": "ce_rest.json",
                        },
                    ],
                },
            }
            doc_path = tmp_path / f"mlp_c{center}_w{width}.json"
            doc_path.write_text(json_lib.dumps(doc))
            out = tmp_path / f"out_c{center}_w{width}"
            argv = [
                "run",
                "--engine",
                "auto",
                str(doc_path),
                "--data-root",
                str(FIXTURES / "data"),
                "--artifacts-root",
                str(out / "artifacts"),
                "--out",
                str(out),
                "--device",
                _device(),
            ]
            assert main(argv) == 0
            ce_corr = _frame(out / "ce_corr.json")
            ce_rest = _frame(out / "ce_rest.json")
            diff = np.exp(-ce_rest["value"].to_numpy()) - np.exp(
                -ce_corr["value"].to_numpy()
            )
            aie.setdefault(center, []).append(pd.Series(diff))
    pooled = {c: pd.concat(parts).mean() * 100 for c, parts in aie.items()}
    peak = max(pooled, key=pooled.get)
    print(
        f"MLP-window AIE by center: { {k: round(v, 2) for k, v in sorted(pooled.items())} }"
    )
    print(f"peak center {peak}")
    assert_golden("rome.mlp_window_aie_peak", float(pooled[peak]))


def test_mixing_positional_shares(tmp_path):
    """Fig 2: positional share by query-position bucket at the most-mixed
    layer. Attribution is candidate-relative (the paper's Fig 9 works in
    mean logits over candidate outputs): each row carries logit_diff of
    every mechanism's predicted genre against the original answer, and is
    attributed to the mechanism with the largest positive shift — or to
    none when the original answer still dominates. Shares are normalized
    among attributed rows and read at layer 18 — the paper's own
    intervention layer for gemma-2-2b-it ("the last layer before
    retrieval starts", named as layers 16-18, anchored at ~18). A max-attribution heuristic is wrong here: past retrieval
    (L20+) the patch carries the counterfactual's finished answer and
    reflexive sweeps to ~100% everywhere (layer table of one H100 run of
    this test: L16 edges 100/74% vs middle 28%; L18 90/65% vs 17%; L22+
    ref≈100%).
    One document per bucket keeps the single-batch lm_head forward within
    GPU memory; the scan stays in the document so the retrieval
    transition remains visible in the saved tables."""
    merged_parts = []
    for bucket in ("first", "middle", "last"):
        out = tmp_path / bucket
        run_document(f"mixing_scan_{bucket}_im.json", out)
        frames = {}
        for mech in ("pos", "lex", "ref"):
            frame = _frame(out / f"ld_{mech}.json")
            layer_col = _axis_column(frame, "target.layers")
            frames[mech] = frame.rename(columns={layer_col: "layer"})
        part = frames["pos"][["example_id", "layer"]].copy()
        diffs = pd.DataFrame(
            {mech: frames[mech]["value"].to_numpy() for mech in ("pos", "lex", "ref")}
        )
        best = diffs.idxmax(axis=1)
        winning = diffs.max(axis=1) > 0
        for mech in ("pos", "lex", "ref"):
            part[mech] = ((best == mech) & winning).astype(float).to_numpy()
        part["bucket"] = bucket
        merged_parts.append(part)
    merged = pd.concat(merged_parts, ignore_index=True)
    merged["attributed"] = merged[["pos", "lex", "ref"]].sum(axis=1)

    by_layer = merged.groupby("layer")["attributed"].mean()
    print(
        f"attribution by layer: { {int(k): round(v, 3) for k, v in by_layer.items()} }"
    )
    intervention_layer = 18  # the paper's own layer for gemma-2-2b-it
    print(f"reading shares at the paper's layer {intervention_layer}")

    at_best = merged[merged["layer"] == intervention_layer]
    shares = {}
    for bucket, group in at_best.groupby("bucket"):
        attributed = group["attributed"].sum()
        assert attributed > 0, f"no attributed rows in bucket {bucket!r}"
        shares[bucket] = 100 * group["pos"].sum() / attributed
    print(f"positional shares: { {k: round(v, 1) for k, v in shares.items()} }")
    edge_rows = at_best[at_best["bucket"] != "middle"]
    edges = 100 * edge_rows["pos"].sum() / edge_rows["attributed"].sum()
    assert_golden("mixing.positional_share_edges", float(edges))
    assert_golden("mixing.positional_share_middle", float(shares["middle"]))


def test_arithmetic_steering_diagonal(tmp_path):
    """Fig 7 diagonal, the task author's operationalization: steer every
    baseline-correct hours prompt toward each of the 24 targets (layer-18
    residual, last token, alpha=10, periods {2,5,10,20,50}); a target
    counts when its hour token has the highest prompt-averaged probability
    over the 24 hour tokens. Anti-gaming constraints are structural here: all 24 targets run, the period set is
    fixed in tests/golden/_steering.py, and the only prompt filter is
    baseline correctness.

    Hybrid by necessity: affine Fourier probes are not a v1 train
    objective, so the pinned harvest document collects the residuals and
    the probes are fitted here (the package-pinned recipe: Adam, lr 1e-3,
    500 epochs); the per-prompt Eq. 4 correction is a pytorch_fn edit
    (local-only) in per-target generated documents."""
    import json as json_lib

    import numpy as np
    import torch
    from safetensors.torch import load_file

    from tests.golden import _steering

    # 1. baseline: keep the prompts the model answers correctly
    base_out = tmp_path / "baseline"
    run_document("hours_baseline_im.json", base_out)
    acc = _frame(base_out / "acc.json")
    rows = json_lib.loads((FIXTURES / "data" / "hours" / "all.json").read_text())
    correct = [
        rows[int(e)] for e, v in zip(acc["example_id"], acc["value"]) if v == 1.0
    ]
    print(f"baseline-correct prompts: {len(correct)}/1152")

    # 2. harvest addition residuals (pinned document)
    harvest_out = tmp_path / "harvest"
    run_document("addition_harvest_im.json", harvest_out)
    acts = (
        load_file(str(harvest_out / "acts_l18.safetensors"))["acts"].squeeze(1).float()
    )
    sums = torch.tensor(
        [
            r["sum"]
            for r in json_lib.loads(
                (FIXTURES / "data" / "addition" / "pairs.json").read_text()
            )
        ],
        dtype=torch.float32,
    )
    assert acts.shape[0] == len(sums)

    # 3. fit the affine sine/cosine probes (package-pinned recipe)
    probes: dict[int, dict] = {}
    for period in _steering.PERIODS:
        entry = {}
        for kind, fn in (("sin", torch.sin), ("cos", torch.cos)):
            target = fn(2 * torch.pi * sums / period)
            w = torch.zeros(acts.shape[1], requires_grad=True)
            b = torch.zeros(1, requires_grad=True)
            opt = torch.optim.Adam([w, b], lr=1e-3)
            for _ in range(500):
                opt.zero_grad()
                loss = ((acts @ w + b - target) ** 2).mean()
                loss.backward()
                opt.step()
            entry[f"w_{kind}"] = w.detach()
            entry[f"b_{kind}"] = float(b.detach())
            print(f"T={period} {kind}: final MSE {float(loss):.4f}")
        probes[period] = entry
    _steering.configure(probes)

    # 4. per-target steering documents over the correct prompts
    data_root = tmp_path / "data" / "hours"
    data_root.mkdir(parents=True)
    (data_root / "correct.json").write_text(json_lib.dumps(correct))
    groups = {f"{h:02d}": [f"{h:02d}"] for h in range(24)}
    hits = 0
    for target in range(24):
        doc = {
            "header": {
                "protocol_version": "4",
                "description": f"generated: hours steering toward target {target:02d}",
            },
            "model": {
                "key": "meta-llama/Llama-3.1-8B",
                "revision": "main",
                "dtype": "bf16",
            },
            "data": {"base": {"dataset": "hours/correct", "field": "input"}},
            "method": {
                "sites": {
                    "l18": {"component": "block_output", "layers": [18]},
                    "lm_head": {"component": "lm_head"},
                },
                "reads": {"logits": {"site": "lm_head", "pos": -1}},
                "code": {
                    "steer_fn": {
                        "locator": f"tests.golden._steering.apply_target_{target}",
                        "description": "Eq. 4 / Alg. 1, per-prompt (§2.8.1)",
                    }
                },
                "writes": {
                    "steer": {
                        "site": "l18",
                        "pos": -1,
                        "do": {"pytorch_fn": {"code": "steer_fn"}},
                    }
                },
                "intervened_models": {
                    "steered": {
                        "input": "base",
                        "reads": ["logits"],
                        "writes": ["steer"],
                    }
                },
                "save": [
                    {
                        "read": "logits",
                        "model": "steered",
                        "aggregation": {
                            "kind": "class_probs",
                            "groups": groups,
                        },
                        "file_path": "hour_probs.json",
                    }
                ],
            },
        }
        doc_path = tmp_path / f"steer_{target:02d}.json"
        doc_path.write_text(json_lib.dumps(doc))
        out = tmp_path / f"steer_out_{target:02d}"
        argv = [
            "run",
            "--engine",
            "auto",
            str(doc_path),
            "--data-root",
            str(tmp_path / "data"),
            "--artifacts-root",
            str(out / "artifacts"),
            "--out",
            str(out),
            "--device",
            _device(),
        ]
        assert main(argv) == 0
        table = _frame(out / "hour_probs.json")
        means = {
            name: float(
                np.mean(
                    [
                        json_lib.loads(v)[name] if isinstance(v, str) else v[name]
                        for v in table["value"]
                    ]
                )
            )
            for name in groups
        }
        top = max(means, key=means.get)
        hits += top == f"{target:02d}"
        print(f"target {target:02d}: top {top} (p={means[top]:.3f})")
    assert_golden("arithmetic.steering_top1_fraction", hits / 24)
