"""The parallel golden's second-family documents (``docs/model_parallelism.md``
§10.6 "the second-family rows", §11): the inference document on two dense
families the vocabulary decision made served geometries of — **gemma2**
(``google/gemma-2-9b``) and **llama** (``meta-llama/Llama-3.1-8B``) — each
held in bf16 on two cards, as the A3B's inference document is held:
``dp=2`` byte for byte, ``tp=2`` within a format-2 band measured on the
card, the loader at ``1 / world`` of every sharded parameter, the residency
rule clean. Llama's checkpoint is bf16 on disk; gemma-2-9b's is **stored in
fp32** (``tests/golden/parallel_headers_gemma2_9b.json``), so its load
converts — staged on the host, the card holding the bf16 model and nothing
else (§5.3 "the load's peak under a dtype conversion").
The llama document holds ``pp=2`` byte for byte too;
gemma2 **ties its head to its embedding**, which ``pp`` refuses by name at
the load (``sharding.place_stage``, §6.6), so its document names no
pipeline geometry and the golden holds the refusal instead
(`REFUSED`). Gemma2's config ships an ``embed_tokens →
embedding_rowwise`` row the derivation declines (§6.1, §11): the plan's
``unapplied`` provenance, which ``dry-run … --parallel tp=2`` names and the
golden and its CPU guard both hold.

Each document is the boundary document written for the **tiny Llama**
(`inference.author_for`: the module-boundary and attention-interior
classes, no routing table) retargeted to the family's realization — its
own, written into its block of the record (``Document.realization``, as
``das_dense``) — so the record stays one file, the A3B's at the top and
each family's model in its block.
"""

from __future__ import annotations

from tests.golden._parallel import inference
from tests.golden._parallel.runs import Document, Realization
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA

__all__ = [
    "DENSE_FIXTURE",
    "FAMILIES",
    "GEMMA2",
    "GEMMA2_9B",
    "GEMMA2_9B_MODEL",
    "LLAMA",
    "LLAMA31_8B",
    "LLAMA31_8B_MODEL",
    "REFUSED",
    "UNAPPLIED",
    "document",
]

#: The fixture the dense documents are written for before retargeting.
DENSE_FIXTURE = TINY_LLAMA

GEMMA2_9B_MODEL = "google/gemma-2-9b"
LLAMA31_8B_MODEL = "meta-llama/Llama-3.1-8B"

#: The realizations: native bf16 on CUDA, the tapped layer mid-tower (any
#: layer of a dense model is a full-attention layer; the sweep is the layer
#: and the fourth after it, `inference.sweep`).
GEMMA2_9B = Realization(GEMMA2_9B_MODEL, "bf16", "cuda", 20)
LLAMA31_8B = Realization(LLAMA31_8B_MODEL, "bf16", "cuda", 15)


def document(
    name: str,
    realization: Realization,
    *,
    exact: tuple[str, ...],
    banded: tuple[str, ...],
) -> Document:
    """The boundary document on a dense family's realization."""
    return Document(
        name=name,
        exact=exact,
        banded=banded,
        author=inference.author_for(DENSE_FIXTURE),
        describe=inference.describe,
        measure=inference.measure_for(()),
        classes=inference.DENSE_CLASSES,
        realization=realization,
    )


GEMMA2 = document("inference_gemma2_9b", GEMMA2_9B, exact=("dp=2",), banded=("tp=2",))
LLAMA = document(
    "inference_llama31_8b", LLAMA31_8B, exact=("dp=2", "pp=2"), banded=("tp=2",)
)

#: The family documents by name, in capture order.
FAMILIES: dict[str, Document] = {d.name: d for d in (GEMMA2, LLAMA)}

#: Per document, the geometry the design refuses by name on that model and
#: the words the refusal carries: gemma2's tied head under a pipeline.
REFUSED: dict[str, tuple[str, str]] = {GEMMA2.name: ("pp=2", "tie_word_embeddings")}

#: The vocabulary row gemma2's plan declines, as ``dry-run`` names it
#: (``protocol/cli.py``: ``not applied: <pattern> (<style>)``).
UNAPPLIED = "not applied: embed_tokens (embedding_rowwise)"
