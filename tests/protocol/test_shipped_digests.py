"""Golden pins for every shipped intervention specification: its campaign
canonical form as BYTES, its campaign digest, and its per-point digests.

Two corpora are covered — the presets a user copies from
(``demos/methods/protocols/*.json``) and the spec's golden corpus
(``tests/protocols/*_im.json``). The corpus already pins its digests in
``corpus_digests.json`` (``test_corpus.py``); this module pins the canonical
bytes for both corpora and cross-checks its digest pins against that file, so
there is one truth about what a shipped document digests to.

The pins guard the *canonical form*, byte for byte. A drift here means the
loader canonicalizes a document differently than it did — which spec §7 treats
as a loader migration (bump ``version``, ship a migration), never a routine
edit. Regenerate with ``uv run python tests/protocol/update_shipped_digests.py``
and review the diff; the canonical files are the bytes ``canonical_bytes``
returns, so the diff of a pretty-printed pair below shows exactly what moved.

Three kinds of document, by how they load offline:

* **standalone** — loads against the fixture environment (``env``), the way
  ``causalab validate`` loads it with no run tree;
* **deferred** (`DEFERRED`) — a preset whose ``file_path`` names a
  workflow step's output, so it is refused standalone under rule 15 (§5.15)
  and loads only as a step after the one that writes the file. Pinned the way
  the workflow's validate path compiles such a step-dependent document — through
  [`DeferredArtifacts`][causalab.workflow.document.DeferredArtifacts], whose file digest is
  a placeholder — so the pinned canonical form carries ``"0" * 64`` where a run
  would carry the artifact's content digest. (The workflow records the
  *authored* digest for such a step and discards this one, so these pins guard
  the canonical form, not a digest any run tree records.)
* **excluded** (`EXCLUDED`) — not pinnable offline at all; each row
  says why, and a test checks the reason still holds.
"""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.schema.explicit import canonical_bytes
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.pipeline import compile_protocol
from causalab.io.env import ResolutionEnv
from causalab.workflow.document import DeferredArtifacts

from tests.protocol._env import CORPUS_DIR, steps_of
from tests.protocol.test_protocol_presets import RUN_TREE_ONLY
from tests._helpers.paths import PROTOCOLS_DIR

pytestmark = pytest.mark.unit

HERE = Path(__file__).parent
REPO = HERE.resolve().parents[1]
PRESETS_DIR = PROTOCOLS_DIR
PINS_PATH = HERE / "shipped_digests.json"
CORPUS_PINS_PATH = HERE / "corpus_digests.json"
CANONICAL_DIR = HERE / "canonical"

REGENERATE = (
    "regenerate with `uv run python tests/protocol/update_shipped_digests.py`, "
    "review the diff, and treat it as a loader migration (spec §7) — never a "
    "silent re-pin"
)

#: The placeholder [`DeferredArtifacts`][causalab.workflow.document.DeferredArtifacts] answers for a deferred file's
#: content digest.
DEFERRED_FILE_DIGEST = "0" * 64

#: Presets whose ``file_path`` names a run-tree artifact (preset → the path it
#: names), refused standalone under rule 15 and pinned through a deferring
#: store instead — see the module docstring. The one table is
#: ``test_protocol_presets.RUN_TREE_ONLY``; this is the same rows under the
#: name this module's docstring uses.
DEFERRED: dict[str, str] = dict(RUN_TREE_ONLY)


@dataclasses.dataclass(frozen=True)
class Exclusion:
    """Why a shipped document has no pin: ``reason`` for the reader,
    ``model_key`` for the check that the reason still holds."""

    reason: str
    model_key: str


#: Documents with no pin. ``minimal_cpu.json``'s model is registered in the
#: protocol model registry only when an engine loads it from its HF config
#: (``pytorch_hooks.loading.load_model``) — there is no static row — and the
#: canonical form derives site widths from that registration, so the document
#: has no canonical form a fresh interpreter can reach offline (its standalone
#: load is refused under rule 4 until another test has loaded the model).
EXCLUDED: dict[str, Exclusion] = {
    "minimal_cpu.json": Exclusion(
        reason="model key has no static registry row; registered only by an "
        "engine loading the HF config, so not canonicalizable offline",
        model_key="hf-internal-testing/tiny-random-LlamaForCausalLM",
    ),
}

CORPUS_FILES = sorted(path.name for path in CORPUS_DIR.glob("*.json"))
PRESET_FILES = sorted(path.name for path in PRESETS_DIR.glob("*.json"))
COVERED = sorted(name for name in CORPUS_FILES + PRESET_FILES if name not in EXCLUDED)

#: Empty before the first regeneration, so the regenerator can import this
#: module's tables; the completeness test then says what is missing.
PINS: dict[str, Any] = json.loads(PINS_PATH.read_text()) if PINS_PATH.exists() else {}
CORPUS_PINS: dict[str, Any] = json.loads(CORPUS_PINS_PATH.read_text())


def document_path(name: str) -> Path:
    """Where a covered document lives — the two corpora share no name."""
    return (CORPUS_DIR if name in CORPUS_FILES else PRESETS_DIR) / name


def deferring(env: ResolutionEnv) -> ResolutionEnv:
    """``env`` with every artifact file check deferred to run time — the store
    the workflow validate path hands a step-dependent document. No step names:
    the presets name their artifacts by ``file_path``, never by step ref."""
    return ResolutionEnv(
        datasets=env.datasets,
        artifacts=DeferredArtifacts(
            outer=env.artifacts, step_names=frozenset(), representatives={}
        ),
        model_info=env.model_info,
    )


def shipped_env(name: str, env: ResolutionEnv) -> ResolutionEnv:
    """The environment a covered document is loaded — and its steps signed —
    with: the deferring store for a preset that names a run-tree path."""
    return deferring(env) if name in DEFERRED else env


def load_shipped(name: str, env: ResolutionEnv) -> CompiledProtocol:
    """Load one covered document the way its pin was made."""
    return compile_protocol(document_path(name), env=shipped_env(name, env))


def _pretty(raw: bytes) -> list[str] | None:
    """``raw`` pretty-printed for a line diff, or ``None`` when it is not JSON
    (a truncated write, a merge marker in the canonical file)."""
    try:
        return json.dumps(json.loads(raw), indent=1, sort_keys=True).splitlines()
    except ValueError:
        return None


def _excerpt(raw: bytes, offset: int, width: int = 40) -> str:
    return repr(raw[max(offset - width, 0) : offset + width])


def describe_drift(name: str, pinned: bytes, actual: bytes) -> str:
    """What a canonical-bytes mismatch looks like to a reader: the first
    differing offset and a unified diff of the two forms, pretty-printed.
    When either side is not JSON the diff gives way to the raw bytes around
    the offset, so a corrupt canonical file still yields this message rather
    than an error while the assertion message is built."""
    offset = next(
        (i for i, (a, b) in enumerate(zip(pinned, actual)) if a != b),
        min(len(pinned), len(actual)),
    )
    pretty_pinned, pretty_actual = _pretty(pinned), _pretty(actual)
    if pretty_pinned is None or pretty_actual is None:
        shown = (
            "one side is not valid JSON — raw bytes around the first difference:\n"
            f"pinned: {_excerpt(pinned, offset)}\n"
            f"actual: {_excerpt(actual, offset)}"
        )
    else:
        diff = list(
            difflib.unified_diff(
                pretty_pinned, pretty_actual, "pinned", "actual", lineterm="", n=2
            )
        )
        shown = "\n".join(diff[:60]) + ("\n..." if len(diff) > 60 else "")
    return (
        f"{name}: canonical form drifted from tests/protocol/canonical/{name} "
        f"(first difference at byte {offset}; {len(pinned)} pinned vs "
        f"{len(actual)} actual bytes). The loader now canonicalizes this "
        f"document differently — if intended, {REGENERATE}.\n{shown}"
    )


# --------------------------------------------------------------------------- #
# the pins
# --------------------------------------------------------------------------- #


class TestShippedPins:
    @pytest.mark.parametrize("name", COVERED)
    def test_canonical_bytes(self, env: ResolutionEnv, name: str) -> None:
        loaded = load_shipped(name, env)
        actual = canonical_bytes(loaded.canonical)
        pinned = (CANONICAL_DIR / name).read_bytes()
        assert actual == pinned, describe_drift(name, pinned, actual)

    @pytest.mark.parametrize("name", COVERED)
    def test_document_digest(self, env: ResolutionEnv, name: str) -> None:
        loaded = load_shipped(name, env)
        assert loaded.digests.document == PINS[name]["document"], (
            f"{name}: campaign digest drifted — the canonical form changed; "
            f"if intended, {REGENERATE}"
        )

    @pytest.mark.parametrize("name", COVERED)
    def test_point_digests(self, env: ResolutionEnv, name: str) -> None:
        loaded = load_shipped(name, env)
        assert (
            list(steps_of(loaded, shipped_env(name, env)).digests)
            == PINS[name]["points"]
        ), (
            f"{name}: per-point digests drifted — a point's canonical form or "
            f"the expansion's point order changed; if intended, {REGENERATE}"
        )

    @pytest.mark.parametrize("name", COVERED)
    def test_pinned_digest_is_the_digest_of_the_pinned_bytes(self, name: str) -> None:
        """The two pin files are one record: the digest pin is sha256 of the
        canonical file, with no loader in between."""
        pinned = (CANONICAL_DIR / name).read_bytes()
        assert hashlib.sha256(pinned).hexdigest() == PINS[name]["document"], (
            f"{name}: shipped_digests.json and canonical/{name} disagree — one "
            f"was edited without the other; {REGENERATE}"
        )


# --------------------------------------------------------------------------- #
# completeness and honesty of the tables
# --------------------------------------------------------------------------- #


class TestCoverage:
    def test_the_corpora_share_no_name(self) -> None:
        """One canonical file per document name, so the names must not collide."""
        assert not set(CORPUS_FILES) & set(PRESET_FILES)

    def test_every_shipped_document_is_covered_or_excluded(self) -> None:
        """A new preset or corpus file must enter the pins (regenerate) or
        `EXCLUDED` with a reason — and nothing may be both."""
        shipped = sorted(CORPUS_FILES + PRESET_FILES)
        assert set(EXCLUDED) <= set(shipped), "EXCLUDED names a file that is gone"
        assert not set(EXCLUDED) & set(COVERED)
        assert sorted(COVERED + list(EXCLUDED)) == shipped
        assert sorted(PINS) == COVERED, (
            f"shipped_digests.json does not pin exactly the covered documents; "
            f"{REGENERATE}"
        )
        canonical_files = sorted(path.name for path in CANONICAL_DIR.iterdir())
        assert canonical_files == COVERED, (
            f"tests/protocol/canonical/ holds a stale or missing file; {REGENERATE}"
        )

    def test_a_corrupt_canonical_file_still_yields_the_drift_message(self) -> None:
        """A pinned file that is not JSON (a truncated write, a merge marker)
        must produce the readable drift message, not an error raised while
        the assertion message is built."""
        name = COVERED[0]
        pinned = (CANONICAL_DIR / name).read_bytes()
        cut = len(pinned) // 2
        for corrupt in (pinned[:cut], pinned[:cut] + b"<<<<<<< HEAD\n" + pinned[cut:]):
            message = describe_drift(name, corrupt, pinned)
            assert message.startswith(f"{name}: canonical form drifted")
            assert f"first difference at byte {cut}" in message
            assert "not valid JSON" in message
            assert "pinned: b'" in message and "actual: b'" in message
            assert REGENERATE in message

    def test_the_drift_message_diffs_two_valid_forms(self) -> None:
        name = COVERED[0]
        pinned = (CANONICAL_DIR / name).read_bytes()
        actual = json.dumps({**json.loads(pinned), "version": -1}).encode()
        message = describe_drift(name, pinned, actual)
        assert "--- pinned" in message and "+++ actual" in message
        assert "not valid JSON" not in message

    def test_deferred_presets_still_name_a_run_tree_path(self) -> None:
        """The deferred rows stay honest: each still names its run-tree path,
        is still refused standalone under rule 15, and its pin still carries
        the deferring store's placeholder where the content digest goes."""
        assert set(DEFERRED) <= set(PRESET_FILES)
        for name, run_tree_path in DEFERRED.items():
            raw = (PRESETS_DIR / name).read_text()
            assert run_tree_path in raw, f"{name} no longer loads {run_tree_path}"
            assert DEFERRED_FILE_DIGEST in (CANONICAL_DIR / name).read_text(), (
                f"{name}: the pin carries no deferred-file placeholder — it was "
                "not made through the deferring store"
            )
            assert PINS[name].get("store") == "deferred", (
                f"{name}: shipped_digests.json must mark a deferred pin with "
                f'"store": "deferred"; {REGENERATE}'
            )
        marked = sorted(name for name, pin in PINS.items() if "store" in pin)
        assert marked == sorted(DEFERRED)

    def test_deferred_presets_are_refused_standalone(self, env: ResolutionEnv) -> None:
        for name in DEFERRED:
            with pytest.raises(ValidationError) as caught:
                compile_protocol(PRESETS_DIR / name, env=env)
            assert caught.value.rule == 15, (
                f"{name} loads standalone now — pin it standalone instead"
            )

    def test_excluded_reasons_still_hold(self) -> None:
        """``minimal_cpu.json`` stays excluded only while its model has no
        static registry row. Checked in a fresh interpreter: this session may
        already have registered the model by loading it."""
        probe = (
            "import sys\n"
            "from causalab.protocol.rules.errors import ValidationError\n"
            "from causalab.protocol.registry import get_model_info\n"
            "try:\n"
            "    get_model_info(sys.argv[1])\n"
            "except ValidationError as err:\n"
            "    sys.exit(0 if err.rule == 4 else 2)\n"
            "sys.exit(1)\n"
        )
        for name, exclusion in EXCLUDED.items():
            raw = json.loads((PRESETS_DIR / name).read_text())
            assert raw["model"]["key"] == exclusion.model_key, (
                f"{name} names another model now — revisit its exclusion"
            )
            result = subprocess.run(
                [sys.executable, "-c", probe, exclusion.model_key],
                capture_output=True,
                text=True,
                cwd=REPO,
            )
            assert result.returncode == 0, (
                f"{name}: {exclusion.model_key!r} is in the offline registry now "
                f"(exit {result.returncode}) — pin the document instead of "
                f"excluding it\n{result.stderr}"
            )


class TestOneTruth:
    @pytest.mark.parametrize("name", CORPUS_FILES)
    def test_corpus_pins_agree_with_corpus_digests_json(self, name: str) -> None:
        """The golden corpus is pinned twice (here and in ``corpus_digests.json``);
        the two records must say the same thing, so a re-pin of one without the
        other is caught."""
        assert PINS[name]["document"] == CORPUS_PINS[name]["document"], (
            f"{name}: shipped_digests.json and corpus_digests.json disagree on "
            "the campaign digest — regenerate both (update_shipped_digests.py "
            "and update_corpus_digests.py) from one tree"
        )
        assert PINS[name]["points"] == CORPUS_PINS[name]["points"], (
            f"{name}: shipped_digests.json and corpus_digests.json disagree on "
            "the point digests — regenerate both from one tree"
        )


def pin_of(
    loaded: CompiledProtocol, env: ResolutionEnv, *, deferred: bool
) -> dict[str, Any]:
    """The pins-file row for one loaded document (the regenerator's): the
    campaign digest and the step digests as the engine signs them against the
    environment the document was loaded with."""
    row: dict[str, Any] = {
        "document": loaded.digests.document,
        "points": list(steps_of(loaded, env).digests),
    }
    if deferred:
        row["store"] = "deferred"
    return row
