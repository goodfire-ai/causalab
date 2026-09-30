"""Shared resolution-environment construction for the protocol tests.

One helper used by both the pytest fixtures (conftest.py) and the
digest-pin regeneration script (update_corpus_digests.py), so the pinned
digests and the asserting tests are guaranteed to resolve against identical
fixture content.

Two fixtures are generated rather than committed: ``rot_k8.safetensors``
(corpus file 09's ``file_path`` featurizer) and ``block_output_L<n>.safetensors``
(the PCA basis ``demos/methods/protocols/das_pca_init.json`` starts from). Their
bytes are deterministic — sorted-key header JSON, zero-filled weight — so the
content digests inside the canonical forms that hash them are stable across
machines and sessions.
"""

from __future__ import annotations

import dataclasses
import json
import struct
from pathlib import Path
from typing import Any, Mapping

from causalab.neural.shared.sweep import Expansion, Point, signed_steps
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import StepRecord
from causalab.protocol.lowering import Axis
from causalab.protocol.registry.models import get_model_info
from causalab.protocol.schema import Document, parse_document
from causalab.tasks import TASKS_ROOT
from causalab.io.env import (
    FileArtifacts,
    FileDatasets,
    ResolutionEnv,
    build_artifact_identity,
)

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS_DIR = Path(__file__).parent.parent / "protocols"

#: The model the fixture corpus (``tests/protocols/*_im.json``) and the other
#: CPU-tier fixture documents name: ungated, so the run door can load its
#: tokenizer on CI without a Hub token (``tests/test_no_gated_models.py``).
#: Registry-only otherwise — no CPU test downloads its weights.
CORPUS_MODEL = "Qwen/Qwen3-8B"
#: The model the shipped method documents (``demos/methods/protocols/
#: weekdays_*.json``, ``das_pca_init.json``) name — the smallest Qwen2.5 base
#: checkpoint that mostly does the weekdays task (demos/methods/README.md).
#: Ungated, like `CORPUS_MODEL`; the CPU tier only ever *loads* those
#: documents (``test_shipped_digests.py``, ``test_init_basis_identity.py``),
#: never runs them, and a load touches the registry, not the Hub — but the
#: artifacts they name must carry its stamp, and their pinned canonical forms
#: hash those artifacts' bytes, so the stamp is not free to move.
SHIPPED_MODEL = "Qwen/Qwen2.5-7B"
#: The cell the shipped weekdays documents sit at: the best cell of
#: ``weekdays_locate_scan.json`` on `SHIPPED_MODEL` (the README says how
#: it was found). Every ``layers: [n]`` in those documents is this number.
SHIPPED_LAYER = 23

#: Corpus 09's fitted-DAS bundle, stamped for `CORPUS_MODEL`.
ROT_FIXTURE_RELPATH = "artifacts/weekdays/qwen3_8b/subspace/rot_k8.safetensors"
#: ``weekdays_das_apply.json``'s, the same bundle stamped for `SHIPPED_MODEL`.
SHIPPED_ROT_FIXTURE_RELPATH = "artifacts/weekdays/qwen25_7b/subspace/rot_k8.safetensors"
#: ``das_pca_init.json``'s PCA basis (a shipped preset, so `SHIPPED_MODEL`).
PCA_FIXTURE_RELPATH = (
    f"artifacts/weekdays/qwen25_7b/pca/block_output_L{SHIPPED_LAYER}.safetensors"
)


def write_rot_fixture(artifacts_root: Path) -> Path:
    """The deterministic fitted-DAS bundles the apply documents load: corpus
    09's (`ROT_FIXTURE_RELPATH`, returned) and the shipped
    ``weekdays_das_apply.json``'s (`SHIPPED_ROT_FIXTURE_RELPATH`) —
    the same bundle, each stamped for the model its document names. Stamped
    identity for (the model @ main in bf16, block_output at the shipped layer, k=8, cayley,
    fp32 params), weight zeros. Load-time checks read only the header.

    ``model_dtype`` is the *model's* precision and ``dtype`` the featurizer
    params', and they differ on purpose: a fit runs a bf16 backbone with fp32
    featurizers (``train.precision``), and both are stamped."""
    _write_rot_bundle(
        artifacts_root / SHIPPED_ROT_FIXTURE_RELPATH, SHIPPED_MODEL, SHIPPED_LAYER
    )
    return _write_rot_bundle(artifacts_root / ROT_FIXTURE_RELPATH, CORPUS_MODEL, 18)


def _write_rot_bundle(target: Path, model_key: str, layer: int) -> Path:
    """One zero-weight rank-8 bundle stamped for ``model_key`` at block
    ``layer`` — corpus 09 sits at L18 (its own pinned cell), the shipped apply
    document at `SHIPPED_LAYER`."""
    identity = build_artifact_identity(
        model_key=model_key,
        model_revision="main",
        # 09 declares bf16, matching the fit it applies (weekdays_das_sweep);
        # model_dtype is part of ArtifactIdentity, so the fixture stamps it too
        model_dtype="bf16",
        tokenizer=model_key,
        site={"component": "block_output", "layers": [layer]},
        k=8,
        parametrization="cayley",
        dtype="fp32",
        trained_on="weekdays/data#train",
        engine="pytorch_hooks",
        commit="fixture",
    )
    # This fixture is 09's "previously fitted" artifact, and artifacts fitted
    # before the backend→engine rename carry the old stamp key. Keeping the old
    # key keeps 09's content_digest (hence its pinned canonical form)
    # byte-stable across the rename, and keeps the loader's tolerance of
    # pre-rename bundles under test. Flip to "engine" only with a corpus
    # re-pin.
    identity["backend"] = identity.pop("engine")
    return write_zero_bundle(
        target, identity, (get_model_info(model_key).hidden_size, 8)
    )


def write_pca_fixture(artifacts_root: Path) -> Path:
    """A deterministic PCA-basis bundle matching ``das_pca_init.json``'s
    ``init``: stamped the way the workflow runner stamps a
    ``causalab.analysis.fit_pca`` output over a harvest of
    (`SHIPPED_MODEL` @ main in bf16, block_output at `SHIPPED_LAYER`) — model
    realization and site
    inherited from the harvest, the basis's own rank (16) and fp32 dtype,
    engine ``script`` — and the single-entry ``entries`` table
    ``step_io.stamp_tensor`` writes. Weight zeros: load-time checks read only
    the header."""
    identity = build_artifact_identity(
        model_key=SHIPPED_MODEL,
        model_revision="main",
        model_dtype="bf16",
        site={"component": "block_output", "layers": [SHIPPED_LAYER]},
        k=16,
        dtype="fp32",
        engine="script",
    )
    identity["entries"] = json.dumps(
        {"weight": {"slot": "weight", "coords": {}}}, sort_keys=True
    )
    return write_zero_bundle(
        artifacts_root / PCA_FIXTURE_RELPATH,
        identity,
        (get_model_info(SHIPPED_MODEL).hidden_size, 16),
    )


def write_zero_bundle(
    target: Path, metadata: dict[str, str], shape: tuple[int, int]
) -> Path:
    """One fp32 ``weight`` of ``shape``, zero-filled, under a sorted-key
    header carrying ``metadata`` (none at all when empty — an unstamped
    bundle) — byte-deterministic, so the content digest a document's
    canonical form takes from it is stable across machines."""
    target.parent.mkdir(parents=True, exist_ok=True)
    n_bytes = shape[0] * shape[1] * 4
    header: dict[str, object] = {
        "weight": {"dtype": "F32", "shape": list(shape), "data_offsets": [0, n_bytes]},
    }
    if metadata:
        header["__metadata__"] = metadata
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    with target.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        fh.write(bytes(n_bytes))
    return target


def build_env(artifacts_root: Path) -> ResolutionEnv:
    """The test resolution environment: committed JSON fixture tables for
    datasets — with the shipped task tables behind them, as the CLI has it, so
    a shipped document and a fixture document load in the same env —
    ``artifacts_root`` (a copy of fixtures/artifacts plus the generated bundle)
    for artifacts."""
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data", fallback_roots=(TASKS_ROOT,)),
        artifacts=FileArtifacts(root=artifacts_root),
        tokenizers=fixture_tokenizer,
    )


#: model keys the fixture environment could not load a tokenizer for
_UNLOADABLE: set[str] = set()


def fixture_tokenizer(key: str, revision: str, *, loader: Any = None) -> Any:
    """The tokenizer service of the fixture environment. A key that
    loads — a tiny fixture model a real engine will run, or the ungated
    `CORPUS_MODEL` whose tokenizer alone (~10 MB) the corpus documents
    resolve positions with — gets **its own** tokenizer: the run door's
    resolution must be the engine's, and the executor refuses a frame its
    tokenizer does not reproduce. A key the Hub has no repo for (``test``,
    registered by the tests) gets the tiny GPT-2 fixture's tokenizer: those
    documents run through stub engines that load nothing, and the run door
    still needs a tokenizer to resolve with. The fallback catches every load
    failure — transformers raises ``OSError`` for a gated repo without a
    token too — so it is **not** what keeps gated keys out of the tier;
    ``tests/test_no_gated_models.py`` is. ``loader`` is the real loader when a test has
    patched the module attribute to this function (else the two would
    recurse)."""
    from tests._helpers.tiny import TINY_RANDOM_GPT2_MODEL_NAME

    if loader is None:
        from causalab.io.tokenizer import load_tokenizer as loader
    if key in _UNLOADABLE:
        return loader(TINY_RANDOM_GPT2_MODEL_NAME)
    try:
        return loader(key, revision)
    except (OSError, ValueError):
        _UNLOADABLE.add(key)
        return loader(TINY_RANDOM_GPT2_MODEL_NAME)


#: The shipped weekdays table spells every weekday, and tiny-random's
#: sentencepiece tokenizer splits " Thursday" / " Wednesday" into several
#: pieces — so a `match` or `logit_diff` over it is refused at tiny scale
#: ([P2], no closed metric kind for multi-token answers). The tiny-scale smoke
#: runs therefore retarget a shipped document's dataset refs onto the 4-row
#: fixture table (whose days happen to be single tokens there), exactly the way
#: they retarget the model. Keyed by the ref a shipped document names.
FIXTURE_INPUTS = {"natural_domains_arithmetic/data/weekdays": "weekdays/data"}


def fixture_input_overrides(document: Mapping[str, Any]) -> dict[str, str]:
    """`set` entries pointing ``document``'s dataset refs at the fixture tables:
    one per data role, plus ``train.eval.split`` when the document trains."""

    def fixture(ref: str) -> str | None:
        base, _, fragment = ref.partition("#")
        if base not in FIXTURE_INPUTS:
            return None
        return (
            f"{FIXTURE_INPUTS[base]}#{fragment}" if fragment else FIXTURE_INPUTS[base]
        )

    out: dict[str, str] = {}
    # the four groups (docs/intervention_protocol.md §1): the inputs are
    # `data`, the fit is `method.train`
    for role, spec in document.get("data", {}).items():
        if (mapped := fixture(spec["dataset"])) is not None:
            out[f"data.{role}.dataset"] = mapped
    split = document.get("method", {}).get("train", {}).get("eval", {}).get("split")
    if split is not None and (mapped := fixture(split)) is not None:
        out["train.eval.split"] = mapped
    return out


@dataclasses.dataclass(frozen=True)
class Steps:
    """A compiled document's steps as the engine enumerates and signs them:
    the expansion in the canonical order, each step's parse, its canonical
    form and its digest — the spellings the
    tests read where they read the compiler's points before the sweep moved
    engine-side (``loaded.expansion``, ``point_documents``,
    ``canonical_points``, ``point_digests``)."""

    expansion: Expansion
    documents: tuple[Document, ...]
    canonical: tuple[Mapping[str, Any], ...]
    digests: tuple[str, ...]

    @property
    def points(self) -> tuple[Point, ...]:
        return self.expansion.points

    @property
    def axes(self) -> tuple[Axis, ...]:
        return self.expansion.axes

    @property
    def coords(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(point.coords for point in self.points)

    @property
    def records(self) -> tuple[StepRecord, ...]:
        """The steps as ``RunResult.steps`` carries them."""
        return tuple(
            StepRecord(index=i, coords=p.coords, digest=d)
            for i, (p, d) in enumerate(zip(self.points, self.digests))
        )


def steps_of(compiled: CompiledProtocol, env: ResolutionEnv) -> Steps:
    """Enumerate and sign the steps of ``compiled`` against ``env`` — the one
    call every test makes where it used to read the points off the compile."""
    signed = signed_steps(compiled, env)
    return Steps(
        expansion=Expansion(
            axes=compiled.axes,
            points=tuple(Point(coords=s.coords, raw=s.raw) for s in signed),
        ),
        documents=tuple(parse_document(s.raw) for s in signed),
        canonical=tuple(s.canonical for s in signed),
        digests=tuple(s.digest for s in signed),
    )
