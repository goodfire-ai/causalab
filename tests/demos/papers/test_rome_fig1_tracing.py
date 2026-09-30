"""The rome_fig1 tracing documents against ROME's own algorithm, on a tiny GPT-2.

The package claims that ``protocols/rome_fig1_trace_state.json`` and
``protocols/rome_fig1_trace_window.json``, fed the draw of the ``noise_draw``
step, compute what ROME's ``trace_with_patch`` computes (lines 133-230 of
<https://github.com/kmeng01/rome/blob/0874014cd9837e4365f3e6f3c71400ef11509e04/experiments/causal_trace.py>):
row 0 of a batch runs clean; rows 1 to n add ``noise * RandomState(1).randn``
to the subject's token embeddings; the states named are put back from row 0;
and the value is the mean over rows 1 to n of the answer's probability at the
last position. The MLP and attention windows are
``range(max(0, c - w // 2), min(L, c - (-w // 2)))`` (line 427).

This test runs the package workflow's ``noise_draw``, ``state`` and
``window`` steps through ``causalab run`` on
``hf-internal-testing/tiny-random-gpt2``, with only the model, the draw's
shape and the grid changed: five layers, four of the fifteen prompt tokens (the subject's first
and last, the next one and the last), and a window of width 4, so that the
clipping at both ends and the asymmetric split both show. The oracle is plain
``transformers`` forward hooks that copy ROME's function, with its float64
draw. Every value of the three panels must agree.

The tolerance is fp32 rounding on one CPU. The draw is stored in float32 and
scaled in fp32 by the documents, where ROME adds the float64 product, which
moves a sum by at most one fp32 unit. Measured on an Apple M-series CPU
(2026-09-28): the largest relative difference is 2.2e-7, against ``RTOL``
1e-5. The values themselves move by 1.9e-2 (relative) across the grid. Two
mutations show the check can fail: the paper text's window rule
(Appendix B.2, ``[l* - 4, l* + 5]``, here ``[c - 1, c + 2]``) moves a value by
up to 8.8e-3, and a draw from seed 2 by up to 1.2e-2.
"""

from __future__ import annotations

import functools
import json
import shutil
from collections import defaultdict
from typing import Any

import numpy as np
import pytest
import torch

from causalab.cli import main as causalab_main

from tests._helpers.tiny import TINY_RANDOM_GPT2_MODEL_NAME, tiny_random_gpt2_model
from tests.demos.papers._scripts import PAPERS, SCRIPTS, load_script

pytestmark = pytest.mark.smoke

fig1_figure = load_script("rome_fig1", "fig1_figure")

PROMPT = "The Space Needle is in downtown"
SUBJECT = "The Space Needle"
#: One token under the tiny tokenizer (`` Seattle`` is three, and a
#: cross-entropy target must be one token).
ANSWER = " the"
SAMPLES = 10
ALPHA = 0.1
WIDTH = 4
#: fp32 rounding on one CPU, relative to the probability (see the docstring).
RTOL = 1e-5


@functools.lru_cache(maxsize=1)
def _tokens() -> tuple[list[int], int, int]:
    """The prompt's ids, the subject's token count, the answer's id."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_RANDOM_GPT2_MODEL_NAME)
    ids = tokenizer.encode(PROMPT)
    subject = tokenizer.encode(SUBJECT)
    assert ids[: len(subject)] == subject, "the subject is the prompt's prefix"
    (answer,) = tokenizer.encode(ANSWER)
    return ids, len(subject), answer


def _taps() -> list[int]:
    """The swept tokens: the subject's first and last, the first after it,
    and the last, where the two sites of the paper sit."""
    ids, subject_tokens, _ = _tokens()
    return [0, subject_tokens - 1, subject_tokens, len(ids) - 1]


def _retarget(name: str, layers: int) -> dict[str, Any]:
    """A package document with the tiny model and its grid, nothing else."""
    doc = json.loads((PAPERS / "protocols" / name).read_text())
    doc["model"] = {
        "key": TINY_RANDOM_GPT2_MODEL_NAME,
        "revision": "main",
        "dtype": "fp32",
    }
    method = doc["method"]
    method["positions"]["tap"]["index"] = {"sweep": _taps()}
    if "axes" in doc:
        doc["axes"]["center"]["range"] = [0, layers]
        doc["axes"]["window"]["rule"]["clipped_band"]["width"] = WIDTH
    else:
        method["sites"]["restore"]["layers"]["sweep"]["range"] = [0, layers]
    return doc


@pytest.fixture(scope="module")
def traced(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """The figure script's table of the tiny run: one value per panel,
    layer and token.

    The run is the package workflow with its knockout steps dropped: the
    ``noise_draw`` step writes the draw for the tiny model's shape, and the
    ``state`` and ``window`` steps load it from the run tree."""
    root = tmp_path_factory.mktemp("rome_fig1_tracing")
    _, subject_tokens, _ = _tokens()
    config = tiny_random_gpt2_model().config
    data = root / "data" / "rome_fig1"
    data.mkdir(parents=True)
    row = {"input": PROMPT, "subject": SUBJECT, "base_answer": ANSWER, "split": "all"}
    (data / "data.json").write_text(json.dumps([row] * SAMPLES))
    (root / "protocols").mkdir()
    for name in ("rome_fig1_trace_state.json", "rome_fig1_trace_window.json"):
        doc = _retarget(name, config.n_layer)
        (root / "protocols" / name).write_text(json.dumps(doc, indent=2))
    workflow = json.loads((PAPERS / "workflows" / "rome_fig1.json").read_text())
    workflow["steps"] = {
        name: workflow["steps"][name] for name in ("noise_draw", "state", "window")
    }
    workflow["steps"]["noise_draw"]["inputs"].update(
        samples=SAMPLES,
        tokens=subject_tokens,
        width=config.n_embd,
        model_key=TINY_RANDOM_GPT2_MODEL_NAME,
        model_revision="main",
    )
    scripts = root / "workflows" / "scripts" / "rome_fig1"
    scripts.mkdir(parents=True)
    shutil.copy(SCRIPTS / "rome_fig1" / "noise_draw.py", scripts / "noise_draw.py")
    (root / "workflows" / "rome_fig1.json").write_text(json.dumps(workflow, indent=2))
    code = causalab_main(
        [
            "run",
            str(root / "workflows" / "rome_fig1.json"),
            "--engine",
            "auto",
            "--data-root",
            str(root / "data"),
            "--out",
            str(root / "out"),
            "--device",
            "cpu",
            "--batch-rows",
            "64",
        ]
    )
    assert code == 0
    return fig1_figure.load(root / "out" / workflow["output_dir"])


def _module(model: Any, component: str, layer: int) -> torch.nn.Module:
    block = model.transformer.h[layer]
    return {
        "block_output": block,
        "mlp_output": block.mlp,
        "attention_output": block.attn,
    }[component]


def rome_trace(
    restore: list[tuple[str, int, int]],
    *,
    seed: int = 1,
) -> float:
    """ROME's ``trace_with_patch`` on the tiny model: each ``(component,
    layer, token)`` in ``restore`` is put back from the clean row 0."""
    model = tiny_random_gpt2_model()
    ids, subject_tokens, answer = _tokens()
    batch = torch.tensor([ids] * (SAMPLES + 1))
    rs = np.random.RandomState(seed)
    noise = ALPHA * torch.from_numpy(
        rs.randn(SAMPLES, subject_tokens, model.config.n_embd)
    )

    def corrupt(module: Any, args: Any, output: torch.Tensor) -> torch.Tensor:
        output[1:, :subject_tokens] += noise
        return output

    def put_back(tokens: list[int]) -> Any:
        def hook(module: Any, args: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            for token in tokens:
                hidden[1:, token] = hidden[0, token]
            return output

        return hook

    by_module: dict[torch.nn.Module, list[int]] = defaultdict(list)
    for component, layer, token in restore:
        by_module[_module(model, component, layer)].append(token)
    handles = [model.transformer.wte.register_forward_hook(corrupt)]
    handles += [m.register_forward_hook(put_back(t)) for m, t in by_module.items()]
    try:
        with torch.no_grad():
            logits = model(batch).logits
    finally:
        for handle in handles:
            handle.remove()
    return float(torch.softmax(logits[1:, -1], dim=-1).mean(dim=0)[answer])


def rome_window(center: int, layers: int, text_rule: bool = False) -> range:
    """ROME's window (``causal_trace.py`` line 427), or the paper text's."""
    if text_rule:
        return range(
            max(0, center - WIDTH // 2 + 1), min(layers, center + WIDTH // 2 + 1)
        )
    return range(max(0, center - WIDTH // 2), min(layers, center - (-WIDTH // 2)))


PANELS = {"e": "block_output", "f": "mlp_output", "g": "attention_output"}


def _oracle(panel: str, layer: int, token: int, **mutation: Any) -> float:
    layers = tiny_random_gpt2_model().config.n_layer
    component = PANELS[panel]
    text_rule = mutation.pop("text_rule", False)
    band = [layer] if panel == "e" else rome_window(layer, layers, text_rule)
    return rome_trace([(component, lb, token) for lb in band], **mutation)


def test_every_value_is_romes_trace_with_patch(traced: Any) -> None:
    table, _ = traced
    layers = tiny_random_gpt2_model().config.n_layer
    assert len(table) == 3 * layers * len(_taps())
    assert set(table["n"]) == {SAMPLES}
    ours = table["p_restored"].to_numpy()
    expected = np.array(
        [_oracle(r.panel, r.layer, r.position) for r in table.itertuples()]
    )
    np.testing.assert_allclose(ours, expected, rtol=RTOL, atol=0.0)
    # the values move with what is restored, by far more than the tolerance
    assert (expected.max() - expected.min()) / expected.max() > 100 * RTOL


def test_the_starred_tokens_are_the_subject(traced: Any) -> None:
    table, labels = traced
    _, subject_tokens, _ = _tokens()
    assert sorted(set(table["position"])) == _taps()
    starred = {i for i, t in labels.items() if t.endswith("*")}
    assert starred == set(range(subject_tokens))


def test_the_corrupted_and_clean_runs_sit_in_panel_e(traced: Any) -> None:
    """The last layer at an earlier token is the corrupted run; at the last
    token it is the clean run (ROME's row 0)."""
    table, _ = traced
    ids, _, _ = _tokens()
    assert fig1_figure.corrupted_score(table) == pytest.approx(rome_trace([]), rel=RTOL)
    last = table[(table["panel"] == "e") & (table["layer"] == table["layer"].max())]
    clean = last[last["position"] == len(ids) - 1]["p_restored"].item()
    layers = tiny_random_gpt2_model().config.n_layer
    everything = [("block_output", layers - 1, t) for t in range(len(ids))]
    assert clean == pytest.approx(rome_trace(everything), rel=RTOL)


@pytest.mark.parametrize(
    "mutation", [{"text_rule": True}, {"seed": 2}], ids=["text_window", "seed_2"]
)
def test_a_wrong_window_or_draw_would_fail_the_check(
    traced: Any, mutation: dict[str, Any]
) -> None:
    table, _ = traced
    panels = ["f", "g"] if "text_rule" in mutation else ["e", "f", "g"]
    rows = table[table["panel"].isin(panels)]
    ours = rows["p_restored"].to_numpy()
    wrong = np.array(
        [_oracle(r.panel, r.layer, r.position, **mutation) for r in rows.itertuples()]
    )
    assert np.max(np.abs(ours - wrong) / wrong) > 100 * RTOL
