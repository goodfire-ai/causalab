"""A fit's declared splits are endpoint-disjoint across tables (§2.2, §5 rule 22).

Rule 22's three resolver-side refusals (``test_dataset_splits.py``) are each a
property of one table. A fit consumes rows in two roles — the training rows its
``data`` refs select, the held-out rows ``train.eval.split`` names — and when
those are two *different* tables nothing checked that they share no prompt: two
files called ``train`` and ``test`` asserted their relationship in their names.
This file is the fourth refusal, and its fail-closed twins.

The fixture tables are built, not hand-written: the two-fold ``weekdays/data``
comes from [`causalab.tasks.splits.generate_split_dataset`][], whose folds are
group-disjoint by construction, and the one-split tables beside it are its
folds re-labelled — plus one deliberately broken copy with a single row carried
across tables, which is the whole of what the refusal has to catch.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from causalab.cli import main
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.rules.data import HELD_OUT_ROLE, check_fit_splits, fit_roles
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.rules.data import check_data_columns
from causalab.io.env import (
    FileArtifacts,
    FileDatasets,
    ResolutionEnv,
    endpoints,
)
from causalab.protocol.schema import parse_document
from causalab.tasks import TASKS_ROOT
from causalab.protocol.rules.document import validate_document
from causalab.tables import table_bytes
from causalab.tasks.loader import load_task
from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig
from causalab.tasks.serialize import serialize_examples, write_dataset_table
from causalab.tasks.splits import generate_split_dataset

from tests.protocol._docs import LOGIT_DIFF, base_doc, in_order, term
from tests.protocol._env import CORPUS_DIR, FIXTURES, steps_of
from tests.protocol.test_protocol_presets import RUN_TREE_ONLY
from tests._helpers.paths import PROTOCOLS_DIR

from tests._helpers.demos import data_root, demo_protocols


pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
PROTOCOLS = PROTOCOLS_DIR
SPEC = REPO / "docs/intervention_protocol.md"
DEMO_FITS = [path for path in demo_protocols() if path.name.endswith("_fit.json")]

#: the two-fold table the group-disjoint builder writes — `#train` / `#test`
TABLE = "weekdays/data"
#: its train fold as a one-split table of its own, and its test fold likewise:
#: two *files* whose disjointness only the bytes can vouch for
POOL = "weekdays/pool"
HELD = "weekdays/held"
#: HELD plus the first row of POOL — the hand-broken twin, shrunk to the one
#: leak that matters
LEAKY = "weekdays/leaky"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("fit-splits-data")
    task = load_task(
        "natural_domains_arithmetic",
        task_cfg=NaturalDomainConfig(domain_type="weekdays"),
    )
    folds = generate_split_dataset(task, seed=0, fractions={"train": 0.5, "test": 0.5})
    table = serialize_examples(
        task.causal_model,
        folds.examples,
        split=folds.splits,
        target_variables=["result"],
    )
    rows = table.rows
    write_dataset_table(rows, root / f"{TABLE}.json")
    pool = [{**row, "split": "all"} for row in rows if row["split"] == "train"]
    held = [{**row, "split": "all"} for row in rows if row["split"] == "test"]
    assert pool and held
    write_dataset_table(pool, root / f"{POOL}.json")
    write_dataset_table(held, root / f"{HELD}.json")
    write_dataset_table(held + [pool[0]], root / f"{LEAKY}.json")
    return root


@pytest.fixture(scope="module")
def datasets(root: Path) -> FileDatasets:
    return FileDatasets(root)


@pytest.fixture(scope="module")
def env(root: Path) -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(root), artifacts=FileArtifacts(root=root)
    )


def fit_doc(training: str, held_out: str | None) -> dict[str, Any]:
    """A minimal fit: the interchange base document with a trained subspace on
    the patch, training on ``training`` and — when given — evaluating on
    ``held_out``. Nothing else differs between two calls."""
    doc = base_doc()
    doc["data"]["base"]["dataset"] = training
    doc["data"]["counterfactual"]["dataset"] = training
    doc["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    doc["method"]["writes"]["patch"]["featurizer"] = "rot"
    doc["method"]["train"] = {
        "objective": [[1.0, term("logits", "patched", dict(LOGIT_DIFF))]],
        "params": ["rot"],
        "optimizer": {"name": "adamw", "lr": 0.01},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    if held_out is not None:
        doc["method"]["train"]["eval"] = {
            "every": {"epochs": 1},
            "split": held_out,
            "aggregations": {"ld": term("logits", "patched", dict(LOGIT_DIFF))},
        }
    doc["method"]["save"].append(
        {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"}
    )
    return in_order(doc)


def parsed(raw: dict[str, Any]):
    doc = parse_document(raw)
    validate_document(doc)
    return doc


def all_endpoints(rows: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in rows:
        out |= endpoints(row)
    return out


class _NeverRead:
    """A resolver that refuses to be asked: the no-`train` twin must not reach it."""

    def rows(self, ref: str) -> list[dict[str, Any]]:
        raise AssertionError(
            f"rows({ref!r}) was read for a document with nothing to compare"
        )

    def digest(self, ref: str) -> str:
        raise AssertionError("digest was read")

    def columns(self, ref: str) -> tuple[str, ...]:
        raise AssertionError("columns was read")


# --------------------------------------------------------------------------- #
# the refusal
# --------------------------------------------------------------------------- #


def test_a_fit_across_two_tables_sharing_a_prompt_is_refused(datasets: FileDatasets):
    """The document is valid — the leak is in the bytes two refs resolve to,
    which is why no load-time rule could see it. The message names both refs,
    both roles and the first shared prompt, and carries rule 22's slug."""
    doc = parsed(fit_doc(POOL, LEAKY))
    with pytest.raises(ValidationError) as err:
        check_fit_splits(doc, datasets)
    assert (err.value.rule, err.value.rule_id, err.value.path) == (
        22,
        "split_declaration",
        HELD_OUT_ROLE,
    )
    message = str(err.value)
    assert "leak across tables" in message
    assert f"{POOL!r} (the training rows, data.base)" in message
    assert f"{LEAKY!r} (the held-out rows, train.eval.split)" in message
    carried = datasets.rows(POOL)[0]  # the row the broken copy carries
    first_shared = sorted(endpoints(carried))[0]
    assert repr(first_shared) in message


def test_the_leak_is_caught_at_either_endpoint(root: Path, datasets: FileDatasets):
    """A training *counterfactual* reappearing as a held-out *base* is the
    same leak as the other way round: the check reads both sides of the pair
    on both roles, not `input` against `input`."""
    pool = datasets.rows(POOL)
    held = datasets.rows(HELD)
    prompt = pool[0]["counterfactual_inputs"][0]
    assert prompt not in all_endpoints(held)  # the builder kept them apart
    crossed = [{**held[0], "input": prompt}] + held[1:]
    write_dataset_table(crossed, root / "weekdays/crossed.json")
    with pytest.raises(ValidationError, match=re.escape(repr(prompt))) as err:
        check_fit_splits(parsed(fit_doc(POOL, "weekdays/crossed")), datasets)
    assert err.value.rule == 22


def test_the_refusal_comes_before_any_executor_is_built(
    env: ResolutionEnv, tmp_path: Path
):
    """The run seam. ``execute_request`` is what every engine's ``execute`` and
    the workflow runner's step both call, and the check sits in it *before*
    the executor factory: no executor, so no encoded batch, no forward group
    and no minibatch. The factory and the train loop here fail if reached."""
    from causalab.neural.shared.execution import execute_request
    from causalab.protocol.engine import RunContext

    loaded = compile_protocol(
        fit_doc(POOL, LEAKY), env=env
    )  # the compile itself is fine
    # a workflow door's context: the engine records nothing beside the outputs
    run = RunContext(output_dir=tmp_path, env=env, record=False)
    built: list[Any] = []

    def factory(*args: Any) -> Any:
        built.append(args)
        raise AssertionError("an executor was built for a leaking fit")

    def trainer(*args: Any) -> Any:
        raise AssertionError("the train loop ran on a leaking fit")

    with pytest.raises(ValidationError) as err:
        execute_request(
            loaded,
            run,
            engine_name="none",
            executor_factory=factory,
            train_runner=trainer,
        )
    assert err.value.rule == 22 and err.value.rule_id == "split_declaration"
    assert built == []
    assert not any(tmp_path.iterdir())  # nothing was written either


def test_validate_data_refuses_the_leaky_fit_and_passes_its_twin(
    root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """The pure-verb seam: ``causalab validate --data`` is where the other
    row-level checks (rules 4, 20, 25) are reached, and this one joins them —
    a refusal with no model and no weights, and a clean twin beside it."""
    leaky = tmp_path / "leaky.json"
    leaky.write_text(json.dumps(fit_doc(POOL, LEAKY)))
    args = ["--data", "--data-root", str(root), "--artifacts-root", str(root)]
    assert main(["validate", "--engine", "auto", str(leaky), *args]) == 1
    err = capsys.readouterr().err
    assert "[V22]" in err and "leak across tables" in err and LEAKY in err

    clean = tmp_path / "clean.json"
    clean.write_text(json.dumps(fit_doc(f"{TABLE}#train", f"{TABLE}#test")))
    assert main(["validate", "--engine", "auto", str(clean), *args]) == 0
    # `validate` runs the data rules by default (`--data` names
    # the default), so the bare verb refuses the leaky fit too
    assert (
        main(["validate", "--engine", "auto", str(leaky), "--data-root", str(root)])
        == 1
    )


# --------------------------------------------------------------------------- #
# fail-closed twins: every valid case still passes
# --------------------------------------------------------------------------- #


def test_two_splits_of_one_table_pass(env: ResolutionEnv):
    """(i) The shape every shipped fit has (`dbm.json`, `das.json`, corpus
    04/05): rule 22's third refusal already made these disjoint, and the
    fourth agrees — through the check and through the whole --data pass."""
    doc = parsed(fit_doc(f"{TABLE}#train", f"{TABLE}#test"))
    check_fit_splits(doc, env.datasets)
    check_data_columns(
        compile_protocol(fit_doc(f"{TABLE}#train", f"{TABLE}#test"), env=env), env
    )


def test_the_same_ref_twice_is_the_visible_ablation_and_passes(datasets: FileDatasets):
    """(ii) Naming one ref for both roles is how a deliberate train-equals-test
    ablation is spelled (`env._check_splits_are_disjoint`). Every prompt
    is shared, and nothing is refused — even on the broken table."""
    for ref in (POOL, LEAKY, f"{TABLE}#train"):
        check_fit_splits(parsed(fit_doc(ref, ref)), datasets)


def test_two_tables_with_no_shared_endpoint_pass(datasets: FileDatasets):
    """(iii) Two files may well be disjoint; the point is that it is now the
    bytes that say so, not the names."""
    assert not all_endpoints(datasets.rows(POOL)) & all_endpoints(datasets.rows(HELD))
    check_fit_splits(parsed(fit_doc(POOL, HELD)), datasets)


def test_a_document_without_a_train_block_never_touches_the_resolver():
    """(iv) No `train`, one role, nothing to compare — and a fit with no `eval`
    likewise. Neither reads a table."""
    check_fit_splits(parse_document(base_doc()), _NeverRead())  # type: ignore[arg-type]
    check_fit_splits(parsed(fit_doc(POOL, None)), _NeverRead())  # type: ignore[arg-type]


def test_fit_roles_names_every_ref_a_fit_consumes():
    doc = parsed(fit_doc(POOL, HELD))
    assert fit_roles(doc) == (
        [("data.base", POOL), ("data.counterfactual", POOL)],
        HELD,
    )
    assert fit_roles(parse_document(base_doc()))[1] is None


def _fits(path: Path) -> bool:
    """Whether the document declares a ``train`` block (in its ``method``, §1)."""
    return "train" in json.loads(path.read_text()).get("method", {})


def _shipped_fits() -> list[tuple[str, Path, Path]]:
    """(id, document, data root) for every committed document with a `train`
    block: the standalone presets and the corpus against the fixture tables,
    the demo fits against their own demo's data."""
    out: list[tuple[str, Path, Path]] = []
    for path in sorted(PROTOCOLS.glob("*.json")):
        if path.name not in RUN_TREE_ONLY and _fits(path):
            out.append((f"protocols/{path.name}", path, FIXTURES / "data"))
    for path in sorted(CORPUS_DIR.glob("*.json")):
        if _fits(path):
            out.append((f"corpus/{path.name}", path, FIXTURES / "data"))
    for path in DEMO_FITS:
        out.append(
            (f"{path.parents[1].name}/{path.name}", path, data_root(path.parents[1]))
        )
    return out


SHIPPED_FITS = _shipped_fits()


def test_the_shipped_fits_were_found():
    """The twin below is not vacuous: the presets that fit, the corpus fits
    and the demo fits are all in the list."""
    ids = {name for name, _, _ in SHIPPED_FITS}
    assert {"protocols/dbm_head.json", "protocols/dbm_expert_neuron.json"} <= ids
    assert {"protocols/das_pca_init.json", "corpus/05_dbm_im.json"} <= ids
    assert any(name.startswith("onboarding_tutorial/") for name in ids)


@pytest.mark.parametrize(
    ("document", "data_root"),
    [(doc, data_root) for _, doc, data_root in SHIPPED_FITS],
    ids=[name for name, _, _ in SHIPPED_FITS],
)
def test_every_shipped_fit_passes(
    document: Path, data_root: Path, artifacts_root: Path
):
    """(v) Every committed fit — the shipped presets on the two splits of the
    shipped weekdays table, the corpus and demo fits on the fixtures — loads
    and passes the fourth refusal at every point. The shipped tables sit behind
    the fixture root here as they do behind any `--data-root`."""
    env = ResolutionEnv(
        datasets=FileDatasets(root=data_root, fallback_roots=(TASKS_ROOT,)),
        artifacts=FileArtifacts(root=artifacts_root),
    )
    loaded: CompiledProtocol = compile_protocol(document, env=env)
    assert loaded.document.train is not None
    for point in steps_of(loaded, env).documents:
        check_fit_splits(point, env.datasets)


def test_the_fixture_train_and_test_tables_share_no_endpoint():
    """The tiny-scale run tests retarget `dbm_head.json` and
    `dbm_expert_neuron.json` onto `weekdays/train` and `weekdays/test` — two
    files. Those two used to share three of their four prompts, so the presets
    ran a leaking fit in every engine test; the refusal is what made it
    visible. The fixture is honest now, and a regression here is a leak, not a
    typo."""
    datasets = FileDatasets(FIXTURES / "data")
    train = all_endpoints(datasets.rows("weekdays/train"))
    test = all_endpoints(datasets.rows("weekdays/test"))
    assert train and test and not train & test
    assert (
        len(datasets.rows("weekdays/test")) == 2
    )  # the row count the run tests assume


# --------------------------------------------------------------------------- #
# T13: content identity covers the fold half — no new field
# --------------------------------------------------------------------------- #


def test_t13_two_fits_differing_only_in_their_training_split_have_different_identities(
    env: ResolutionEnv,
):
    """T13: method identity is the fit document's digest plus its fold
    tables' digests, and the data's content identity already supplies the
    fold half. Two documents identical except for which fold their training
    ref selects canonicalize to different digests, and the *only* difference
    in their canonical forms is the `data` entries' rows digest. No field
    names a fold, a group key or a control family."""
    a = compile_protocol(fit_doc(f"{TABLE}#train", None), env=env)
    b = compile_protocol(fit_doc(f"{TABLE}#test", None), env=env)
    assert steps_of(a, env).digests[0] != steps_of(b, env).digests[0]
    ca, cb = steps_of(a, env).canonical[0], steps_of(b, env).canonical[0]
    assert {k: v for k, v in ca.items() if k != "data"} == {
        k: v for k, v in cb.items() if k != "data"
    }
    for role in ("base", "counterfactual"):
        assert ca["data"][role]["digest"] != cb["data"][role]["digest"]
        assert (
            set(ca["data"][role])
            == set(cb["data"][role])
            == {"dataset", "field", "digest"}
        )
    assert set(ca["method"]["train"]) == set(cb["method"]["train"])
    for word in ("fold", "group_key", "control", "replay"):
        assert word not in json.dumps(ca), word


def test_t13_mutation_the_digest_is_the_rows_not_the_ref_string(
    env: ResolutionEnv, datasets: FileDatasets
):
    """The mutation: the identity must come from the *rows*, or two refs to
    the same rows would be two methods and two folds under one ref one method.
    A whole-table ref and its `#all` fragment select the same rows and carry
    the same data digest, which is the sha256 of exactly those rows' bytes;
    the two folds of one table carry two."""
    whole = steps_of(compile_protocol(fit_doc(POOL, None), env=env), env).canonical[0][
        "data"
    ]["base"]["digest"]
    fragment = steps_of(
        compile_protocol(fit_doc(f"{POOL}#all", None), env=env), env
    ).canonical[0]["data"]["base"]["digest"]
    assert (
        whole
        == fragment
        == hashlib.sha256(table_bytes(datasets.rows(POOL))).hexdigest()
    )
    assert datasets.digest(f"{TABLE}#train") != datasets.digest(f"{TABLE}#test")


# --------------------------------------------------------------------------- #
# T14, shrunk: the builder's folds pass, a broken copy fails
# --------------------------------------------------------------------------- #


def test_t14_the_builders_folds_are_group_disjoint_and_pass(datasets: FileDatasets):
    """ "Require group-disjoint folds": `causalab.tasks.splits` partitions the
    unique inputs into groups before pairing, so the two folds share no
    endpoint — a fact rule 22's third refusal re-checks on every read, and the
    fourth agrees whichever way the folds are named (as two splits of the one
    table, or as the test fold serialized on its own)."""
    train = datasets.rows(f"{TABLE}#train")  # reading it ran the per-table check
    test = datasets.rows(f"{TABLE}#test")
    assert train and test and not all_endpoints(train) & all_endpoints(test)
    check_fit_splits(parsed(fit_doc(f"{TABLE}#train", f"{TABLE}#test")), datasets)
    check_fit_splits(parsed(fit_doc(f"{TABLE}#train", HELD)), datasets)


def test_t14_a_hand_broken_copy_of_the_test_fold_is_refused(datasets: FileDatasets):
    """The same folds, with one training row copied into the held-out table:
    the refusal names the fold, the broken copy, both roles and the prompt."""
    with pytest.raises(ValidationError) as err:
        check_fit_splits(parsed(fit_doc(f"{TABLE}#train", LEAKY)), datasets)
    message = str(err.value)
    assert err.value.rule == 22
    assert f"'{TABLE}#train' (the training rows, data.base)" in message
    assert f"{LEAKY!r} (the held-out rows, {HELD_OUT_ROLE})" in message
    carried = datasets.rows(POOL)[0]
    assert any(repr(prompt) in message for prompt in endpoints(carried))


# --------------------------------------------------------------------------- #
# the spec says what the code does
# --------------------------------------------------------------------------- #


def test_the_spec_documents_split_requirements():
    """Rule 22 documents table splits and the training/evaluation boundary."""
    text = SPEC.read_text()
    section = text.split("## 5. Validation")[1].split("\n## ")[0]
    rule = section.split("\n22. **split_declaration**")[1].split("\n23. **")[0]
    rule = " ".join(rule.split())
    for field in ("`split`", "`#<split>`", "`train.eval.split`"):
        assert field in rule, field
    assert "both endpoints" in rule
    assert "different references" in rule
    assert "same reference" in rule
    assert "disjoint endpoints" in text.split("### 2.2 `data`")[1].split("### 2.2.1")[0]
    assert (
        "disjoint endpoints" in text.split("### 2.11 `train`")[1].split("### 2.12")[0]
    )
