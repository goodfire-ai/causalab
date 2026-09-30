"""Checks over the replication packages under ``demos/papers/``.

A package is a page, its documents, tables and scripts, not the
seven-section demo format, so ``test_demos.py`` does not read it. The layout
is the one ``docs/paper_replications.md`` fixes, shared by every package and
keyed by the package name: the page ``<name>.md``, the protocols
``protocols/<name>_*.json``, the workflow ``workflows/<name>.json``, its
scripts under ``workflows/scripts/<name>/``, the tables and other external
inputs under ``artifacts/data/<name>/``, the committed figures under
``artifacts/figures/<name>/``. A JSON file's kind follows from its content.
A workflow has ``steps``, an intervention specification has ``header``, and a
JSON array under ``artifacts/data/`` is a table.

The suite checks the mechanical half of the format:

* every page's H1 names the method or the content it teaches: no
  ``CausaLab`` prefix, no figure or table label, and not the title of the
  paper its blockquote cites;
* every page's figure subheadings follow a page variant: none, or
  ``### Original`` and then ``### Replication`` or
  ``### Verification with <method>``;
* every intervention specification validates against the package's own tables. The
  apply half of a fit and apply pair loads a run-tree artifact, so its
  workflow validates it instead. Every model a package names has a row in
  the static registry, so its documents validate offline, as
  ``scripts/standalone_smoke.py`` requires;
* every workflow loads, so its script locators and cross-step references
  resolve;
* every committed table of a package with a builder is a fresh build of it
  (``workflows/scripts/<name>/build_dataset.py --check``). The bytes a document's digest names
  are then the bytes the recipe produces. A package without a builder treats
  its table as given;
* every committed copy of an onboarding Original is a fresh copy of it
  (``workflows/scripts/<name>/copy_original.py --check``);
* every ``python`` fence of a page is the source of the definition it shows,
  taken from the module its ``title`` names;
* every script step runs on fabricated outputs of the protocol steps it
  reads, through the runner's own input resolution and output check, and
  writes the columns and keys it declares. The runner checks a declaration
  only after the script has run, so a stale declaration or a script that
  reads a column the engine does not write otherwise fails on the cluster,
  after the model steps;
* every metric answer of every inner document that does not depend on an
  earlier step resolves with its model's tokenizer, and none is glued to its
  prompt: the check ``causalab validate --tokenizer`` runs, which ``run``
  runs before step 1 (answers) and before each step's weights (positions).
  Only tokenizers load. A package on a gated model (`GATED`) is checked by
  ``tests/golden/test_paper_answer_columns.py`` instead, because the CPU
  tier has no Hub token (``tests/test_no_gated_models.py``).
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping

import pytest

from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.io.tables import read_table
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.rules.data import check_data_columns
from causalab.protocol.rules.errors import ProtocolWarning, ValidationError
from causalab.workflow.document import (
    LoadedWorkflow,
    Reference,
    ScriptStep,
    load_workflow,
    resolve_script,
)
from causalab.workflow.runner import check_tokenization, script_call, verify_output
from tests._helpers.fabricated_outputs import (
    StandInTokenizer,
    fabricate_protocol_outputs,
)
from tests._helpers.tracked import tracked_child_dirs
from tests.step_scripts import run_step
from tests.test_no_gated_models import GATED

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
PAPERS = REPO / "demos" / "papers"
PROTOCOLS = PAPERS / "protocols"
WORKFLOWS = PAPERS / "workflows"
#: The one data root of every package: ``<name>/<table>`` resolves here.
DATA = PAPERS / "artifacts" / "data"
#: The only directories directly under ``demos/papers/`` (``docs/paper_replications.md``, Layout).
LAYOUT_DIRS = frozenset({"artifacts", "protocols", "workflows"})


def _replications() -> list[str]:
    """Every package name: one workflow ``workflows/<name>.json`` each."""
    return sorted(path.stem for path in WORKFLOWS.glob("*.json"))


def _replication_of(path: Path) -> str:
    """The package a protocol belongs to: the longest name its file name
    starts with, followed by ``_``."""
    names = [n for n in _replications() if path.name.startswith(f"{n}_")]
    assert names, f"{path.name} names no package"
    return max(names, key=len)


def _env() -> ResolutionEnv:
    """Dataset references resolve against ``artifacts/data/``, where each
    package keeps its tables under its own name."""
    return ResolutionEnv(
        datasets=FileDatasets(root=DATA),
        artifacts=FileArtifacts(root=REPO),
    )


def _kind(path: Path) -> str | None:
    """``table``, ``workflow`` or ``protocol`` from the JSON's own shape, and
    ``None`` for anything else, such as a values file."""
    with path.open() as handle:
        if handle.read(64).lstrip().startswith("["):
            return "table"
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        return None
    if "steps" in payload:
        return "workflow"
    if "header" in payload:
        return "protocol"
    return None


def _json_files(kind: str, name: str | None = None) -> list[Path]:
    """The JSON files of one kind, for one package or all of them: tables
    under ``artifacts/data/<name>/``, protocols ``protocols/<name>_*.json``,
    workflows ``workflows/<name>.json``."""
    out = []
    for package in [name] if name else _replications():
        if kind == "table":
            candidates = sorted((DATA / package).glob("*.json"))
        elif kind == "protocol":
            candidates = [
                path
                for path in sorted(PROTOCOLS.glob(f"{package}_*.json"))
                if _replication_of(path) == package
            ]
        else:
            candidates = [WORKFLOWS / f"{package}.json"]
        out.extend(path for path in candidates if _kind(path) == kind)
    return out


def test_layout() -> None:
    """``demos/papers/`` holds the index, one page per package, and the three
    shared directories. Every package has its page and its workflow, and
    every protocol, script folder and data folder belongs to a package. A
    page that is not a package page is a second page on a package's
    workflow, ``<name>_<topic>.md`` (``docs/paper_replications.md``, Name).

    Directories count only when they hold a tracked file, so the ignored
    ``__pycache__`` an old checkout keeps under a renamed package is not a
    stray, and a committed stray still is."""
    names = set(_replications())
    dirs = tracked_child_dirs(PAPERS)
    assert dirs == LAYOUT_DIRS, f"unexpected directories {sorted(dirs - LAYOUT_DIRS)}"
    for name in names:
        assert (PAPERS / f"{name}.md").is_file(), f"{name}: no {name}.md"
    for path in PROTOCOLS.glob("*.json"):
        _replication_of(path)
    for page in PAPERS.glob("*.md"):
        if page.name != "README.md" and page.stem not in names:
            _replication_of(page)
    for parent in (WORKFLOWS / "scripts", DATA, PAPERS / "artifacts" / "figures"):
        strays = tracked_child_dirs(parent) - names
        assert not strays, (
            f"{parent.relative_to(PAPERS)}: {sorted(strays)} name no package"
        )


# --------------------------------------------------------------------------- #
# the page title
# --------------------------------------------------------------------------- #

#: Every page under ``demos/papers/`` except the index.
PAGES = sorted(path for path in PAPERS.glob("*.md") if path.name != "README.md")
H1 = re.compile(r"^# (.+)$", re.MULTILINE)
#: A figure or table label, the retired ``… Figure 2a`` and ``… Table 5`` style.
LABEL = re.compile(r"\b(?:fig(?:ure)?|table)\.?\s*\d", re.IGNORECASE)
#: The paper title: the bold text of the citation blockquote.
BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)


def _words(text: str) -> list[str]:
    """Lower-case words, so that a title matches across case and line breaks."""
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())


def _contains(words: list[str], part: list[str]) -> bool:
    """Whether ``part``, a non-empty word list, appears as a run of
    consecutive words in ``words``."""
    assert part, "an empty title part"
    return any(
        words[i : i + len(part)] == part for i in range(len(words) - len(part) + 1)
    )


def _cited_title(text: str) -> str:
    """The bold text of the first blockquote on the page."""
    quote = re.search(r"^>.*(?:\n>.*)*", text, re.MULTILINE)
    assert quote, "no citation blockquote"
    bold = BOLD.search(quote.group(0).replace("\n>", " "))
    assert bold, "the citation blockquote has no bold title"
    return " ".join(bold.group(1).split())


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_the_title_names_the_method(page: Path) -> None:
    """The H1 names what the page teaches (``docs/paper_replications.md``,
    Parts, in order): it does not open with ``CausaLab``, carries no figure
    or table label, and does not repeat the cited paper's title, whole or up
    to its colon. The citation blockquote names the paper."""
    text = page.read_text()
    h1 = H1.search(text)
    assert h1, f"{page.name}: no H1"
    title = h1.group(1).strip()
    assert not title.lower().startswith("causalab"), f"{page.name}: {title!r}"
    assert not LABEL.search(title), f"{page.name}: {title!r} names a figure or table"
    cited = _cited_title(text).rstrip(".")
    for part in {cited, cited.split(":", 1)[0]}:
        assert not _contains(_words(title), _words(part)), (
            f"{page.name}: {title!r} repeats the paper title {part!r}"
        )


# --------------------------------------------------------------------------- #
# the figure subheadings
# --------------------------------------------------------------------------- #

#: A ``###`` subheading; above the first ``##`` heading these name the figures.
SUBHEADING = re.compile(r"^### (.+)$", re.MULTILINE)
#: The subheading of this run's figure: a replication of the Original, or a
#: verification of it with another method (``### Verification with DBM``).
OWN_FIGURE = re.compile(r"Replication|Verification with \S.*")


def _figure_subheadings(text: str) -> list[str]:
    """The ``###`` subheadings above the page's first ``##`` heading."""
    section = re.search(r"^## ", text, re.MULTILINE)
    return SUBHEADING.findall(text[: section.start() if section else len(text)])


def _is_a_page_variant(subheadings: list[str]) -> bool:
    """Whether the figure subheadings follow a page variant
    (``docs/paper_replications.md``, Page variants): none for a page with no
    source figure, else ``Original`` and then the page's own figure."""
    if not subheadings:
        return True
    return (
        len(subheadings) == 2
        and subheadings[0] == "Original"
        and OWN_FIGURE.fullmatch(subheadings[1]) is not None
    )


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_the_figures_follow_a_page_variant(page: Path) -> None:
    """A page shows no figure subheading, or ``### Original`` and then
    ``### Replication`` or ``### Verification with <method>``."""
    subheadings = _figure_subheadings(page.read_text())
    assert _is_a_page_variant(subheadings), f"{page.name}: {subheadings}"


@pytest.mark.parametrize(
    "subheadings",
    [
        ["Replication"],
        ["Original"],
        ["Replication", "Original"],
        ["Original", "Verification"],
        ["Original", "Verification with DBM", "Replication"],
    ],
    ids=repr,
)
def test_the_figure_rule_refuses_other_subheadings(subheadings: list[str]) -> None:
    """An Original without the page's own figure, the figures in the other
    order, a verification that names no method, or a third figure is no page
    variant."""
    assert not _is_a_page_variant(subheadings)


def test_the_figure_rule_reads_above_the_first_section() -> None:
    """Only subheadings before the first ``##`` heading name figures; the
    chunk subheadings below it do not count."""
    text = "# T\n\n### Original\n\n### Verification with DBM\n\n## Impl\n\n### Load\n"
    assert _figure_subheadings(text) == ["Original", "Verification with DBM"]


def _named_by_a_workflow(document: Path) -> bool:
    """Whether a workflow of the package runs ``document`` as a step. That is
    the one case where a standalone ``[V15]`` refusal (a run-tree
    ``file_path``) is expected."""
    return any(
        f"../protocols/{document.name}" in workflow.read_text()
        for workflow in _json_files("workflow", _replication_of(document))
    )


@pytest.mark.parametrize(
    "document", _json_files("protocol"), ids=lambda p: str(p.relative_to(PAPERS))
)
def test_protocol_validates(document: Path) -> None:
    env = _env()
    try:
        check_data_columns(compile_protocol(document, env=env), env)
    except ValidationError as err:
        if "[V15]" in str(err) and _named_by_a_workflow(document):
            pytest.skip(
                f"{document.name} loads a run-tree artifact; its workflow validates it"
            )
        raise


def _script_steps() -> list[tuple[str, str]]:
    """Every ``(package, script step)``, read from the workflow files as
    written, so collection loads no workflow. ``test_workflow_validates``
    holds the list to the steps each loaded workflow schedules."""
    out = []
    for package in _replications():
        steps = json.loads((WORKFLOWS / f"{package}.json").read_text())["steps"]
        out.extend(
            (package, name)
            for name, step in steps.items()
            if step.get("type") == "script"
        )
    return out


_SCRIPT_STEPS = _script_steps()


@pytest.fixture(scope="module")
def loaded_workflows() -> Iterator[Callable[[str], LoadedWorkflow]]:
    """The workflow of each package with a script step, loaded once for the
    module and dropped at its end. The ``arithmetic_fig2a`` load takes about
    a minute, and two tests read it."""
    loads: dict[str, LoadedWorkflow] = {}

    def load(package: str) -> LoadedWorkflow:
        if package not in loads:
            loads[package] = load_workflow(WORKFLOWS / f"{package}.json", _env())
        return loads[package]

    yield load
    loads.clear()


@pytest.mark.parametrize(
    "workflow", _json_files("workflow"), ids=lambda p: str(p.relative_to(PAPERS))
)
def test_workflow_validates(
    workflow: Path, loaded_workflows: Callable[[str], LoadedWorkflow]
) -> None:
    """The workflow loads, and the script steps it schedules, a mounted
    workflow's included, are the ones ``_script_steps`` lists. No script
    step then goes unchecked by ``test_script_steps_write_their_declarations``."""
    listed = {step for package, step in _SCRIPT_STEPS if package == workflow.stem}
    loaded = (
        loaded_workflows(workflow.stem) if listed else load_workflow(workflow, _env())
    )
    scheduled = {
        name
        for name, step in loaded.document.steps.items()
        if isinstance(step, ScriptStep)
    }
    assert scheduled == listed, (
        f"{workflow.name} schedules the script steps {sorted(scheduled)}, and "
        f"_script_steps lists {sorted(listed)}: extend _script_steps to read "
        "the steps the workflow mounts"
    )


def _strings(value: Any) -> set[str]:
    """Every string in ``value``: a key, or each value of a swept one."""
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        return {text for item in value.values() for text in _strings(item)}
    if isinstance(value, list):
        return {text for item in value for text in _strings(item)}
    return set()


def _model_keys(workflow: Path, sets: Mapping[str, Any] | None = None) -> set[str]:
    """The model keys the workflow's intervention specifications can run:
    each document's ``model.key``, every value of a swept one, a step's
    ``set`` of it, and the same through a nested ``workflow`` step, whose
    ``set`` is per inner step (``sets``). Read from the files, so collection
    loads no workflow."""
    keys: set[str] = set()
    for name, step in json.loads(workflow.read_text())["steps"].items():
        document = step.get("document")
        if not isinstance(document, str):
            continue
        overrides = {**(sets or {}).get(name, {}), **step.get("set", {})}
        if step.get("type") == "workflow":
            keys |= _model_keys(workflow.parent / document, overrides)
            continue
        raw = json.loads((workflow.parent / document).read_text())
        keys |= _strings(overrides.get("model.key", raw["model"]["key"]))
    return keys


def _on_a_gated_model(workflow: Path) -> bool:
    return any(key.startswith(GATED) for key in _model_keys(workflow))


def check_answers_resolve(workflow: Path) -> None:
    """``causalab validate --tokenizer`` on the package's workflow: every
    static inner document's positions and metric answers resolve with its
    model's tokenizer (``runner.check_tokenization``), and no answer warns
    as glued to its prompt. Shared with the golden tier's gated packages."""
    env = _env()
    loaded = load_workflow(workflow, env)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProtocolWarning)
        checked = check_tokenization(loaded, env, positions=True)
    assert checked, f"{workflow.name}: no inner document compiles at load"


@pytest.mark.parametrize(
    "workflow",
    [w for w in _json_files("workflow") if not _on_a_gated_model(w)],
    ids=lambda p: str(p.relative_to(PAPERS)),
)
def test_answer_columns_resolve(workflow: Path) -> None:
    check_answers_resolve(workflow)


@pytest.mark.parametrize("replication", _replications())
def test_tables_reproduce(replication: str) -> None:
    builder = WORKFLOWS / "scripts" / replication / "build_dataset.py"
    if not builder.is_file():
        pytest.skip(f"{replication} ships no build_dataset.py; its tables are given")
    tables = _json_files("table", replication)
    assert tables, f"{replication} commits no table"
    for table in tables:
        result = subprocess.run(
            [sys.executable, str(builder), "--out", str(table), "--check"],
            capture_output=True,
            text=True,
            cwd=PAPERS,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "replication",
    [
        name
        for name in _replications()
        if (WORKFLOWS / "scripts" / name / "copy_original.py").is_file()
    ],
)
def test_original_copy_is_current(replication: str) -> None:
    """A package whose Original is an onboarding figure commits a copy of the
    values that figure draws. ``copy_original.py --check`` fails when the copy
    or the onboarding result has changed since, so a re-run of the onboarding
    page cannot leave the Original stale. It reads committed files only."""
    script = WORKFLOWS / "scripts" / replication / "copy_original.py"
    result = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True,
        text=True,
        cwd=PAPERS,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------- #
# script steps on fabricated protocol outputs
# --------------------------------------------------------------------------- #


class _StandInClassifier:
    """What ``score_toxicity.Classifier`` offers, without its download.

    ``labels`` are the six labels of ``unitary/toxic-bert``
    (https://huggingface.co/unitary/toxic-bert/blob/main/config.json), and
    each probability is a fixed function of the text."""

    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def __init__(self, key: str, revision: str | None = None) -> None:
        del key, revision  # every checkpoint stands in the same way

    def probabilities(self, texts: list[str]) -> list[list[float]]:
        return [
            [byte / 255 for byte in hashlib.sha256(text.encode()).digest()[:6]]
            for text in texts
        ]


def _offline_classifier(module: ModuleType) -> None:
    """Replace ``score_toxicity.Classifier`` in the loaded script."""
    assert hasattr(module, "Classifier"), (
        "score_toxicity.py defines no Classifier; update its stand-in"
    )
    setattr(module, "Classifier", _StandInClassifier)


#: The script steps that load a model the CPU tier does not have, and the
#: stand-in that replaces that one object of the loaded module. The rest of
#: the script runs as shipped.
SCRIPT_STAND_INS: dict[tuple[str, str], Callable[[ModuleType], None]] = {
    ("mlp_steering", "grade"): _offline_classifier,
}


@contextlib.contextmanager
def _script_imports(package: str) -> Iterator[None]:
    """Undo what running a package's script changes in this process: the
    ``sys.path`` entries it adds and the sibling modules it imports. Both
    would otherwise stay for the rest of the session."""
    path = list(sys.path)
    before = set(sys.modules)
    scripts = (WORKFLOWS / "scripts" / package).resolve()
    try:
        yield
    finally:
        sys.path[:] = path
        for key in set(sys.modules) - before:
            file = getattr(sys.modules.get(key), "__file__", None)
            if file and Path(file).resolve().is_relative_to(scripts):
                del sys.modules[key]


def _run_script(
    package: str, name: str, loaded: LoadedWorkflow, root: Path
) -> dict[str, str]:
    """Run one script step as the runner runs one in process, and check every
    output it declares; returns ``{file: check}``.

    The runner's own `script_call` resolves the inputs, `ScriptCall.stamp`
    finishes the outputs, and `verify_output` checks each one with the
    runner's refusal text. The test imports the script itself, so that a
    stand-in can replace one object of it."""
    step = loaded.document.steps[name]
    assert isinstance(step, ScriptStep)
    assert not (step.runtime and step.runtime.get("isolate")), (
        f"{package}/{name} runs isolated, and this check runs a script in process"
    )
    step_dir = root / name
    step_dir.mkdir(parents=True)
    call = script_call(name, loaded, root, step_dir)
    script = resolve_script(step, loaded.workflow_dir, f"steps.{name}.script")
    with _script_imports(package):
        spec = importlib.util.spec_from_file_location(
            f"_papers_{package}_{name}", script
        )
        assert spec is not None and spec.loader is not None, script
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        stand_in = SCRIPT_STAND_INS.get((package, name))
        if stand_in is not None:
            stand_in(module)
        run_step(module, call.inputs, call.outputs)
    call.stamp()
    checks = {}
    for slot, decl in step.outputs.items():
        what = f"step {name!r}: output {decl.file!r}"  # as `_verify_outputs`
        checks[decl.file] = verify_output(call.outputs[slot], decl, what=what)
        if decl.columns is not None:
            # an empty table satisfies any column declaration
            assert read_table(call.outputs[slot]), f"{what} wrote no rows"
    return checks


def _run_script_steps(
    package: str, root: Path, loaded: LoadedWorkflow
) -> dict[str, dict[str, str] | Exception]:
    """Every script step of ``package`` in schedule order, over fabricated
    outputs of the protocol steps they read; per step, its checks or the
    error it raised. A step after a failed step is refused, naming it. One
    stand-in tokenizer serves every protocol step, as one model's tokenizer
    serves a real run."""
    steps = loaded.document.steps
    wanted: dict[str, set[str]] = {}
    for step in steps.values():
        if not isinstance(step, ScriptStep):
            continue
        for value in step.inputs.values():
            if isinstance(value, Reference) and value.step is not None:
                if not isinstance(steps[value.step], ScriptStep):
                    wanted.setdefault(value.step, set()).add(str(value.file))
    tokenizer = StandInTokenizer()
    outcomes: dict[str, dict[str, str] | Exception] = {}
    for name in loaded.order:
        if name in wanted:
            assert name in loaded.inner, (
                f"{package}/{name}: a script reads it, and only a protocol step's "
                "outputs are fabricated"
            )
            fabricate_protocol_outputs(
                name, loaded.inner[name], wanted[name], _env(), root / name, tokenizer
            )
            continue
        if not isinstance(steps[name], ScriptStep):
            continue
        failed = [
            dep
            for dep in loaded.dependencies[name]
            if isinstance(outcomes.get(dep), Exception)
        ]
        if failed:
            outcomes[name] = AssertionError(f"upstream step(s) {failed} failed")
            continue
        try:
            outcomes[name] = _run_script(package, name, loaded, root)
        except Exception as err:  # re-raised by the step's own test case
            outcomes[name] = err
    return outcomes


@pytest.fixture(scope="module")
def script_runs(
    tmp_path_factory: pytest.TempPathFactory,
    loaded_workflows: Callable[[str], LoadedWorkflow],
) -> Callable[[str], dict[str, dict[str, str] | Exception]]:
    """Each package's script steps, run once for the module."""
    runs: dict[str, dict[str, dict[str, str] | Exception]] = {}

    def run(package: str) -> dict[str, dict[str, str] | Exception]:
        if package not in runs:
            runs[package] = _run_script_steps(
                package, tmp_path_factory.mktemp(package), loaded_workflows(package)
            )
        return runs[package]

    return run


@pytest.mark.parametrize(
    ("package", "step"), _SCRIPT_STEPS, ids=[f"{p}/{s}" for p, s in _SCRIPT_STEPS]
)
def test_script_steps_write_their_declarations(
    script_runs: Callable[[str], dict[str, dict[str, str] | Exception]],
    package: str,
    step: str,
) -> None:
    """A script step, run on fabricated outputs of the protocol steps it
    reads (``tests/_helpers/fabricated_outputs.py``), writes every output it
    declares, with the declared columns and keys (`verify_output`).

    The fabricated tables and bundles come from the engine's own writers, and
    their columns, tensor keys, tensor ranks and dtypes and entry fields
    match a real run on a tiny model. The numbers are made up. The check
    catches a declaration the script no longer matches and a script that
    reads a column the engine does not write. It does not check column
    dtypes, the numbers a script computes, or answers the real tokenizer
    splits into several tokens."""
    outcome = script_runs(package)[step]
    if isinstance(outcome, Exception):
        raise outcome
    assert outcome, f"{package}/{step} declares no output"


# --------------------------------------------------------------------------- #
# the README's chunked copy of a specification
# --------------------------------------------------------------------------- #

#: The sections of ``method`` (``docs/intervention_protocol.md`` §1). A README
#: chunk that opens with one of these keys is a piece of ``method``; any other
#: top-level key is a piece of the document root.
METHOD_SECTIONS = frozenset(
    {
        "intervened_models",
        "positions",
        "sites",
        "featurizers",
        "params",
        "reads",
        "writes",
        "train",
        "save",
        "code",
    }
)

FENCE = re.compile(r"^```json[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
COMMENT = re.compile(r"(?:^|(?<=\s))//[^\n]*", re.MULTILINE)
#: A ``##`` heading, which ends the chunks that follow a Full JSON block.
SECTION = re.compile(r"^## ", re.MULTILINE)


def _is_whole(fence: str) -> bool:
    """A fence that opens with ``{`` is a whole copy, such as a Full JSON
    block; every other ``json`` fence is a chunk."""
    return COMMENT.sub("", fence).lstrip().startswith("{")


def _file_equal_to(whole: str) -> Path:
    """The protocol file a whole copy equals once its comments are gone, or a
    path that names the failure, which the tests then report as missing."""
    payload = json.loads(COMMENT.sub("", whole))
    for path in sorted(PROTOCOLS.glob("*.json")):
        if json.loads(path.read_text()) == payload:
            return path
    return PROTOCOLS / "<no protocol file equals the Full JSON block>"


def _copies(readme: Path) -> list[tuple[Path, list[str], list[str]]]:
    """Every ``(file, chunks, wholes)`` a page documents, in order. Each copy
    is anchored on a Full JSON block: the file is the protocol the block
    equals, and the chunks are the fences after the block up to the next
    ``##`` heading."""
    text = readme.read_text()
    out: list[tuple[Path, list[str], list[str]]] = []
    fences = list(FENCE.finditer(text))
    for index, match in enumerate(fences):
        if not _is_whole(match.group(1)):
            continue
        heading = SECTION.search(text, match.end())
        end = heading.start() if heading else len(text)
        chunks = [
            m.group(1)
            for m in fences[index + 1 :]
            if m.start() < end and not _is_whole(m.group(1))
        ]
        out.append((_file_equal_to(match.group(1)), chunks, [match.group(1)]))
    return out


def _chunked_copies(readme: Path) -> list[tuple[Path, list[str]]]:
    """Every ``(file, chunks)`` a page documents (`_copies`)."""
    return [(document, chunks) for document, chunks, _ in _copies(readme)]


def _merge_chunks(chunks: list[str]) -> dict:
    """The document the chunks spell: the fragments merged, ``method`` sections
    into ``method`` and every other key at the root."""
    merged: dict = {}
    for chunk in chunks:
        fragment = json.loads("{" + COMMENT.sub("", chunk) + "}")
        for key, value in fragment.items():
            if key in METHOD_SECTIONS:
                merged.setdefault("method", {})[key] = value
            else:
                merged[key] = value
    return merged


#: Every page follows the tutorial layout of ``docs/paper_replications.md``
#: (The chunks): one page per package, plus any second page
#: ``<name>_<topic>.md`` on a package's workflow (``test_layout``).
TUTORIALS = frozenset(page.stem for page in PAGES)

_CHUNKED = [
    (readme, document, chunks)
    for readme in sorted(PAPERS / f"{name}.md" for name in TUTORIALS)
    for document, chunks in _chunked_copies(readme)
]


@pytest.mark.parametrize(
    ("readme", "document", "chunks"),
    _CHUNKED,
    ids=[f"{r.stem}/{d.name}" for r, d, _ in _CHUNKED],
)
def test_the_chunked_copy_is_the_file(
    readme: Path, document: Path, chunks: list[str]
) -> None:
    """A tutorial README shows its specification key by key, and shows all
    of it: the chunks, merged (``method`` sections into ``method``, the rest
    at the root), equal the file minus its ``header``. A chunk that drifts
    from the file, a key left out, or a whole copy in one fence fails here."""
    assert document.is_file(), f"{readme.stem}: {document.name} is not a file"
    assert chunks, f"{readme.stem}: no json fences before {document.name}"
    expected = json.loads(document.read_text())
    assert "header" in expected
    expected.pop("header")
    assert _merge_chunks(chunks) == expected


def test_every_tutorial_shows_a_chunked_copy() -> None:
    """Each page in `TUTORIALS` exists and shows at least one chunked copy,
    anchored on a Full JSON block."""
    for package in sorted(TUTORIALS):
        readme = PAPERS / f"{package}.md"
        assert readme.is_file(), package
        assert _chunked_copies(readme), f"{package}: no chunked copy in its README"


_WHOLE = [
    (readme, document, whole)
    for readme in sorted(PAPERS / f"{name}.md" for name in TUTORIALS)
    for document, _, wholes in _copies(readme)
    for whole in wholes
]


@pytest.mark.parametrize(
    ("readme", "document", "whole"),
    _WHOLE,
    ids=[f"{r.stem}/{d.name}" for r, d, _ in _WHOLE],
)
def test_the_whole_copy_is_the_file(readme: Path, document: Path, whole: str) -> None:
    """A page's Full JSON block equals its file, header included, once its
    ``//`` comments are gone."""
    assert json.loads(COMMENT.sub("", whole)) == json.loads(document.read_text())


# --------------------------------------------------------------------------- #
# the page's python chunk: a causal model's equations, copied from the library
# --------------------------------------------------------------------------- #

#: A ``python`` fence whose ``title`` names the repository file it copies from,
#: ```` ```python title="causalab/tasks/MCQA/causal_models.py" ````. The chunk
#: merge above reads ``json`` fences only, so these never enter it.
PYTHON_FENCE = re.compile(
    r"^```python(?P<info>[^\n]*)\n(?P<body>.*?)^```", re.MULTILINE | re.DOTALL
)
TITLE = re.compile(r'title="(?P<path>[^"]+\.py)"')


def _python_chunks(readme: Path) -> list[tuple[str | None, str]]:
    """Every ``(source path, body)`` of a page's ``python`` fences; the path is
    ``None`` when the fence names no file."""
    out = []
    for match in PYTHON_FENCE.finditer(readme.read_text()):
        title = TITLE.search(match.group("info"))
        out.append((title.group("path") if title else None, match.group("body")))
    return out


def _definition_source(path: str, name: str) -> str:
    """The source of the top-level definition ``name`` in the module at
    ``path``, decorators included.

    ``inspect.getsource`` of the imported object is the check, but a
    ``@mechanism`` definition is a ``ModelDefinition``, not a function, and
    ``inspect.getsource`` refuses it. So the module's own
    ``inspect.getsource`` is parsed and the definition cut out of it at the
    lines ``ast`` reports, which is the text ``inspect.getsource`` gives for
    an undecorated function."""
    module = importlib.import_module(path.removesuffix(".py").replace("/", "."))
    source = inspect.getsource(module)
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
                lines = source.splitlines(keepends=True)
                return "".join(lines[first - 1 : node.end_lineno])
    raise AssertionError(f"{path} defines no top-level {name!r}")


def _python_chunk_mismatch(path: str | None, body: str) -> str | None:
    """Why a ``python`` chunk is not a verbatim copy of the definition it
    shows, or ``None`` when it is. The chunk holds one top-level definition,
    and its name selects the source to compare."""
    if path is None:
        return "the python fence names no source file in its title"
    definitions = [
        node.name
        for node in ast.parse(body).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if len(definitions) != 1:
        return f"the python chunk holds {len(definitions)} definitions, not one"
    source = _definition_source(path, definitions[0])
    if body != source:
        return f"the python chunk differs from {definitions[0]!r} in {path}"
    return None


_PYTHON = [
    (readme, path, body)
    for readme in sorted(PAPERS / f"{name}.md" for name in TUTORIALS)
    if readme.is_file()
    for path, body in _python_chunks(readme)
]


@pytest.mark.parametrize(
    ("readme", "path", "body"),
    _PYTHON,
    ids=[f"{r.stem}/{p}" for r, p, _ in _PYTHON],
)
def test_the_python_chunk_is_the_source(
    readme: Path, path: str | None, body: str
) -> None:
    """A page that shows a causal model's equations shows the library's text:
    the ``python`` chunk equals the source of the definition it names."""
    assert _python_chunk_mismatch(path, body) is None, readme.stem


def test_the_python_check_refuses_a_one_line_edit() -> None:
    """A copy of the MCQA equations that differs from the library by one line
    fails the check, and the verbatim copy passes it."""
    path = "causalab/tasks/MCQA/causal_models.py"
    source = _definition_source(path, "equations")
    assert _python_chunk_mismatch(path, source) is None
    lines = source.splitlines(keepends=True)
    edited = next(i for i, line in enumerate(lines) if "answer_position = V(" in line)
    lines[edited] = lines[edited].replace("choices.index(color)", "0")
    assert _python_chunk_mismatch(path, "".join(lines)) is not None


def test_python_fences_do_not_enter_the_chunk_merge(tmp_path: Path) -> None:
    """A ``python`` fence between the Full JSON block and the next ``##``
    heading is not a chunk: the merge reads the ``json`` fences alone."""
    document = sorted(PROTOCOLS.glob("rome_fig1_knockout.json"))[0]
    payload = json.loads(document.read_text())
    root = {k: v for k, v in payload.items() if k not in ("header", "method")}
    page = tmp_path / "page.md"
    page.write_text(
        "```json\n"
        + json.dumps(payload)
        + "\n```\n\n```python\ndef f():\n    return {}\n```\n\n"
        + "".join(
            "```json\n" + json.dumps({key: value})[1:-1] + "\n```\n"
            for key, value in [*root.items(), *payload["method"].items()]
        )
        + "\n## Next\n"
    )
    [(found, chunks)] = _chunked_copies(page)
    assert found == document
    assert _merge_chunks(chunks) == {k: v for k, v in payload.items() if k != "header"}


def test_a_gated_key_behind_a_set_or_a_nested_workflow_is_found(
    tmp_path: Path,
) -> None:
    """A step's ``set`` of ``model.key`` and a nested workflow's step both
    reach the gated-model filter, so neither package lands in the CPU tier."""
    doc = {"model": {"key": "gpt2"}}
    (tmp_path / "doc.json").write_text(json.dumps(doc))
    inner = {
        "steps": {"fit": {"type": "intervention_protocol", "document": "doc.json"}}
    }
    (tmp_path / "inner.json").write_text(json.dumps(inner))
    outer = {
        "steps": {
            "plain": {"type": "intervention_protocol", "document": "doc.json"},
            "tail": {
                "type": "workflow",
                "document": "inner.json",
                "set": {"fit": {"model.key": f"{GATED[0]}any-model"}},
            },
        }
    }
    (tmp_path / "outer.json").write_text(json.dumps(outer))
    assert _model_keys(tmp_path / "outer.json") == {"gpt2", f"{GATED[0]}any-model"}
    assert _on_a_gated_model(tmp_path / "outer.json")
    assert not _on_a_gated_model(tmp_path / "inner.json")
