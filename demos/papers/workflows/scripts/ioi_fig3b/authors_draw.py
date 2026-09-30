"""Rebuild the 100 pairs a fresh process of the authors' code draws for Figure 3b.

The authors' path-patching scripts load GPT-2 with
``EasyTransformer.from_pretrained`` and then draw
``IOIDataset(prompt_type="mixed", N=100, prepend_bos=False)`` and its pABC
prompts by three flips (Easy-Transformer ``experiments.py`` lines 73-103 at
``ea15315``; ``real_notebook.py`` lines 122-151 at ``373cd15``). Loading the
model builds an ``EasyTransformerConfig``, whose default ``seed`` of 42 seeds
Python's ``random``, numpy and torch (``EasyTransformerConfig.py`` lines 111
and 131-132, ``utils.py`` lines 201-204 at ``ea15315``). Nothing draws a
random number between the two, so a fresh process draws the same 100 pairs
every time.

This script imports the authors' ``ioi_dataset.py`` from a clone, checks its
bytes, seeds as the model load does, draws the pairs and the flips as the
scripts do, and writes them as a table of this package,
``<out>/ioi_fig3b/data.json``. The workflow runs on it unchanged with
``--data-root <out>``. A flip may repeat a name on the pABC side, and the
causal model of ``build_dataset.py`` takes the three slots as they are.

Usage, from ``demos/papers/``::

    git clone https://github.com/redwoodresearch/Easy-Transformer
    git -C Easy-Transformer checkout ea15315dd24481e9e2ac5c3ef335d82907a1dc34
    python workflows/scripts/ioi_fig3b/authors_draw.py --easy-transformer Easy-Transformer --out DIR
    causalab run workflows/ioi_fig3b.json --engine auto --data-root DIR --out DIR/output --device cuda
    python workflows/scripts/ioi_fig3b/fig3b_figure.py --artifacts DIR/output/ioi_fig3b --figures DIR/figures
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import random
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = ["authors_prompts", "pair", "main"]

HERE = Path(__file__).resolve().parent
GENERATOR = "authors_draw.py"
#: The authors' dataset code inside a clone, and the sha256 of its bytes at
#: ``ea15315``; ``ioi_dataset.py`` at ``373cd15``, the last commit to change
#: it before arXiv v1, has the same bytes.
AUTHORS_FILE = "easy_transformer/ioi_dataset.py"
AUTHORS_SHA256 = "94a72de90e426382322fd690669024a0d2d4bd72a45ed9ce820a99ae9173d231"
#: ``EasyTransformerConfig.seed``'s default, and the pairs the scripts draw.
SEED = 42
N = 100


def _builder() -> ModuleType:
    """``build_dataset.py`` beside this script, for its causal model."""
    spec = importlib.util.spec_from_file_location(
        "ioi_fig3b_build", HERE / "build_dataset.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authors_prompts(
    easy_transformer: Path, n: int = N
) -> tuple[list[dict], list[dict]]:
    """The pIOI prompts and their pABC flips, as the authors' code draws them
    in a fresh process: seeded as ``EasyTransformer.from_pretrained`` seeds,
    then ``IOIDataset`` and the IO, S and S1 flips (``experiments.py`` lines
    79-103)."""
    path = easy_transformer / AUTHORS_FILE
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != AUTHORS_SHA256:
        raise ValueError(
            f"{path} has sha256 {digest}, not {AUTHORS_SHA256} (Easy-Transformer ea15315)"
        )
    # the module imports matplotlib.pyplot, which needs no display here
    os.environ.setdefault("MPLBACKEND", "Agg")
    spec = importlib.util.spec_from_file_location("ioi_dataset", path)
    assert spec is not None and spec.loader is not None
    authors = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(authors)
    import numpy as np
    import torch

    # EasyTransformerConfig.__post_init__ -> utils.set_seed_everywhere(42)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    dataset = authors.IOIDataset(prompt_type="mixed", N=n, prepend_bos=False)
    abc = (
        dataset.gen_flipped_prompts(("IO", "RAND"))
        .gen_flipped_prompts(("S", "RAND"))
        .gen_flipped_prompts(("S1", "RAND"))
    )
    return dataset.ioi_prompts, abc.ioi_prompts


def _slots(template: str, text: str) -> dict[str, str]:
    """The names in the ``io``, ``s`` and ``third`` slots of ``text``, a
    prompt of ``template`` with its place and object filled."""
    pieces = re.split(r"\{(io|s|third)\}", template)
    pattern = "".join(
        f"(?P<{piece}>\\w+)" if i % 2 else re.escape(piece)
        for i, piece in enumerate(pieces)
    )
    match = re.fullmatch(pattern, text)
    if match is None:
        raise ValueError(f"{text!r} is not a prompt of {template!r}")
    return match.groupdict()


def pair(model: Any, templates: list[str], base: dict, abc: dict) -> dict[str, Any]:
    """One example of the package's table from one of the authors' pIOI
    prompts and its pABC flip: the same template, place and object, the
    names in the three slots of each, and each prompt without its last word,
    the answer."""
    place, obj = base["[PLACE]"], base["[OBJECT]"]
    prompt, answer = base["text"].rsplit(" ", 1)
    if answer != base["IO"]:
        raise ValueError(f"{base['text']!r} does not end with its IO {base['IO']!r}")
    names = {"io": base["IO"], "s": base["S"], "third": base["S"]}
    template = next(
        (t for t in templates if t.format(place=place, object=obj, **names) == prompt),
        None,
    )
    if template is None:
        raise ValueError(f"{prompt!r} is none of the package's templates")
    filled = template.replace("{place}", place).replace("{object}", obj)
    abc_names = _slots(filled, abc["text"].rsplit(" ", 1)[0])
    common = {"template": template, "place": place, "object": obj}
    return {
        "input": model.new_trace({**common, **names}),
        "counterfactual_inputs": [model.new_trace({**common, **abc_names})],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--easy-transformer",
        type=Path,
        required=True,
        help="a clone of redwoodresearch/Easy-Transformer at ea15315",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the data root: ioi_fig3b/data.json is written under it",
    )
    args = parser.parse_args(argv)
    builder = _builder()
    model = builder.ioi_model()
    bases, abcs = authors_prompts(args.easy_transformer)
    examples = [pair(model, builder.TEMPLATES, b, a) for b, a in zip(bases, abcs)]
    dataset = builder.serialize(model, examples, GENERATOR, SEED)
    for row, base, abc in zip(dataset.rows, bases, abcs):
        # the table holds the authors' prompts, less the answer word
        assert row["input"] == base["text"].rsplit(" ", 1)[0]
        assert row["counterfactual_inputs"] == [abc["text"].rsplit(" ", 1)[0]]
    path = args.out / builder.TASK / "data.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = builder.write_dataset_table(dataset.rows, path)
    print(f"{path}: {len(dataset.rows)} rows, digest {digest[:12]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
