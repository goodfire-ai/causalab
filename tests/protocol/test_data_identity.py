"""Forward-group data identity is the *content* of the rows a role reads,
never the ref's name (spec §3, §8).

``_data_identity`` used to be ``f"{dataset}#{field}"`` — a name. Two
different tables under one name shared a forward group, and one table under
two names did not. The identity is now the canonical form's content digest of
the selected rows (§2.2) plus the field, so the interning identity and the
provenance identity name a table the same way. These tests are the witness
that the old identity is gone: every one of T1–T3 passes a name-keyed
identity through the same document and fails on it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from causalab.neural.shared.execution import _data_identity, campaign_plans
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.pipeline import compile_protocol
from causalab.neural.shared.plan import ForwardGroup, interned_groups
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.tables import table_bytes

from tests.protocol._docs import UNWRITTEN, base_doc, in_order, saved
from tests.protocol._env import CORPUS_DIR, FIXTURES, steps_of


pytestmark = pytest.mark.unit

SCAN = CORPUS_DIR / "07_weekdays_locate_scan_im.json"
DAS_SWEEP = CORPUS_DIR / "08_weekdays_das_sweep_im.json"
TABLE = FIXTURES / "data" / "weekdays" / "data.json"
#: The one ref every weekdays corpus document reads, on both roles.
REF = "weekdays/data#train"


def _rows() -> list[dict[str, Any]]:
    return json.loads(TABLE.read_text())


def _edited_rows() -> list[dict[str, Any]]:
    """The fixture table with one train row's prompt reworded — same name,
    same shape, same columns, different content. The entity survives the
    edit so the ``{"variable": "entity"}`` anchor still finds it."""
    rows = _rows()
    row = next(r for r in rows if r["split"] == "train")
    row["input"] = row["input"].replace("tomorrow", "then tomorrow")
    return rows


def _env(root: Path, tables: dict[str, list[dict[str, Any]]]) -> ResolutionEnv:
    """A data root holding ``tables`` (ref → rows) and an empty artifact
    root — none of the documents here loads a ``file_path``."""
    for ref, rows in tables.items():
        target = root / "data" / f"{ref}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(rows, indent=1))
    (root / "artifacts").mkdir(exist_ok=True)
    shutil.copytree(FIXTURES / "artifacts", root / "artifacts", dirs_exist_ok=True)
    return ResolutionEnv(
        datasets=FileDatasets(root=root / "data"),
        artifacts=FileArtifacts(root=root / "artifacts"),
    )


def _plans(loaded: CompiledProtocol, env: ResolutionEnv):
    steps = steps_of(loaded, env)
    return campaign_plans(steps.documents, steps.canonical)


def _keys(plans) -> set[str]:
    return {group.key for plan in plans for group in plan.groups}


def _by_model(plans, model: str) -> set[str]:
    return {g.key for plan in plans for g in plan.groups if g.model == model}


def _name_keyed(doc) -> dict[str, str]:
    """The identity this PR removed, for the negative controls below."""
    out: dict[str, str] = {}
    for role, value in doc.data.items():
        entries = value if isinstance(value, tuple) else (value,)
        for j, spec in enumerate(entries):
            name = role if not isinstance(value, tuple) else f"{role}[{j}]"
            out[name] = f"{spec.dataset}#{spec.field}"
    return out


# --------------------------------------------------------------------------- #
# T1 — same name, different content → different identity
# --------------------------------------------------------------------------- #


def test_same_name_different_content_never_interns(tmp_path: Path) -> None:
    """Two data roots, each with ``weekdays/data.json``, one row's prompt
    reworded in the second. Every forward group of corpus 07 — the shared
    counterfactual harvest and all 64 patched groups — differs between the
    two loads, because the rows differ. Under the name-keyed identity the
    two sets were identical: the very same 65 digests for different data."""
    env_one = _env(tmp_path / "a", {"weekdays/data": _rows()})
    env_two = _env(tmp_path / "b", {"weekdays/data": _edited_rows()})
    one, two = compile_protocol(SCAN, env=env_one), compile_protocol(SCAN, env=env_two)
    steps_one, steps_two = steps_of(one, env_one), steps_of(two, env_two)
    plans_one, plans_two = _plans(one, env_one), _plans(two, env_two)

    assert _keys(plans_one).isdisjoint(_keys(plans_two))
    assert _by_model(plans_one, "original_counterfactual").isdisjoint(
        _by_model(plans_two, "original_counterfactual")
    )
    assert _by_model(plans_one, "patched").isdisjoint(_by_model(plans_two, "patched"))
    # the per-role identity itself moved on both roles — both read the table
    ident_one = _data_identity(steps_one.documents[0], steps_one.canonical[0])
    ident_two = _data_identity(steps_two.documents[0], steps_two.canonical[0])
    assert ident_one["base"] != ident_two["base"]
    assert ident_one["counterfactual"] != ident_two["counterfactual"]
    # negative control: the name-keyed identity cannot tell them apart
    assert _name_keyed(steps_one.documents[0]) == _name_keyed(steps_two.documents[0])


# --------------------------------------------------------------------------- #
# T2 — two names, same content → same identity
# --------------------------------------------------------------------------- #


def test_a_drawn_role_is_identified_by_its_eval_member(tmp_path: Path) -> None:
    """§2.2 ``draw``: the identity carries the field the forward tokenizes
    (`DataRole.resolved_field`), so a drawn role at ``eval: 0`` is the *same*
    identity as the fixed ``counterfactual_inputs[0]`` over the same rows —
    identical texts, and a fit's own forwards bypass the store — while
    ``eval: 1`` is another. Before the change the identity carried the bare
    column, and its docstring was not literally true."""
    env = _env(tmp_path, {"weekdays/data": _rows()})
    fixed = compile_protocol(SCAN, env=env)
    raw = json.loads(json.dumps(dict(fixed.tree)))

    def drawn(eval_member: int) -> CompiledProtocol:
        doc = json.loads(json.dumps(raw))
        doc["data"]["counterfactual"] = {
            **doc["data"]["counterfactual"],
            "field": "counterfactual_inputs",
            "draw": {"kind": "uniform", "eval": eval_member},
        }
        return compile_protocol(doc, env=env)

    def identity(loaded: CompiledProtocol) -> str:
        return _data_identity(
            steps_of(loaded, env).documents[0], steps_of(loaded, env).canonical[0]
        )["counterfactual"]

    assert identity(drawn(0)) == identity(fixed)
    assert identity(drawn(0)).endswith("#counterfactual_inputs[0]")
    assert identity(drawn(1)) != identity(fixed)


def test_same_content_under_two_names_interns(tmp_path: Path) -> None:
    """The table copied byte-for-byte to ``other/renamed.json`` and the
    document pointed at it with ``--set`` on both roles: every group key
    is equal, and interning the two campaigns together owes exactly what
    one of them owes (65, not 130). Under the name-keyed identity the two
    loads shared nothing."""
    env = _env(tmp_path, {"weekdays/data": _rows(), "other/renamed": _rows()})
    one = compile_protocol(SCAN, env=env)
    two = compile_protocol(
        SCAN,
        env=env,
        overrides={
            "data.base.dataset": "other/renamed#train",
            "data.counterfactual.dataset": "other/renamed#train",
        },
    )
    assert steps_of(two, env).documents[0].data["base"].dataset == "other/renamed#train"
    # the documents are different campaigns (the ref is authored content) …
    assert one.digests.document != two.digests.document
    plans_one, plans_two = _plans(one, env), _plans(two, env)
    # … whose forwards are the same forwards
    assert _keys(plans_one) == _keys(plans_two)
    assert len(interned_groups(plans_one)) == 65
    assert len(interned_groups(plans_one + plans_two)) == len(
        interned_groups(plans_one)
    )
    assert _data_identity(
        steps_of(one, env).documents[0], steps_of(one, env).canonical[0]
    ) == _data_identity(
        steps_of(two, env).documents[0], steps_of(two, env).canonical[0]
    )
    # negative control
    assert _name_keyed(steps_of(one, env).documents[0]) != _name_keyed(
        steps_of(two, env).documents[0]
    )


# --------------------------------------------------------------------------- #
# T3 — a name-hashing reimplementation cannot pass
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class _TableDouble:
    """A ``DatasetResolver`` that answers every ref with one fixed table —
    so the ref string carries no information at all, and only content can."""

    table: tuple[dict[str, Any], ...]

    def digest(self, ref: str) -> str:
        return hashlib.sha256(table_bytes(self.rows(ref))).hexdigest()

    def columns(self, ref: str) -> tuple[str, ...]:
        cols: set[str] = set()
        for row in self.rows(ref):
            cols.update(row)
        return tuple(sorted(cols))

    def rows(self, ref: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self.table]


def _double_env(rows: list[dict[str, Any]], root: Path) -> ResolutionEnv:
    return ResolutionEnv(
        datasets=_TableDouble(table=tuple(rows)),
        artifacts=FileArtifacts(root=root),
    )


def test_identity_is_the_content_digest_and_not_the_ref(tmp_path: Path) -> None:
    """Against the resolver protocol: the same ref ``weekdays/data#train``
    resolves to table A in one environment and table B in another. The
    identities differ, each one *is* the canonical content digest plus the
    field, and neither contains a character of the ref. A reimplementation
    hashing ``f"{dataset}#{field}"`` fails the first assertion; one hashing
    the name alongside the content fails the last."""
    doc = in_order(base_doc())
    env_one = _double_env(_rows(), tmp_path)
    env_two = _double_env(_edited_rows(), tmp_path)
    one = compile_protocol(doc, env=env_one, base_dir=tmp_path)
    two = compile_protocol(doc, env=env_two, base_dir=tmp_path)
    steps_one, steps_two = steps_of(one, env_one), steps_of(two, env_two)
    pdoc_one, canon_one = steps_one.documents[0], steps_one.canonical[0]
    pdoc_two, canon_two = steps_two.documents[0], steps_two.canonical[0]
    ident_one = _data_identity(pdoc_one, canon_one)
    ident_two = _data_identity(pdoc_two, canon_two)

    assert ident_one != ident_two
    assert set(ident_one) == {"base", "counterfactual"}
    for role in ("base", "counterfactual"):
        digest = canon_one["data"][role]["digest"]
        field = str(pdoc_one.data[role].field)
        assert ident_one[role] == f"{digest}#{field}"
        assert ident_one[role].startswith(digest)
        assert len(digest) == 64 and int(digest, 16) >= 0
        assert "weekdays" not in ident_one[role]
        assert "data#train" not in ident_one[role]
    # the two roles read different fields of the same rows — one table, two
    # identities, and the digest half is shared
    assert (
        ident_one["base"].split("#", 1)[0]
        == ident_one["counterfactual"].split("#", 1)[0]
    )
    assert ident_one["base"] != ident_one["counterfactual"]
    # the digest is exactly the sha256 of the selected rows' canonical bytes
    assert (
        ident_one["base"].split("#", 1)[0]
        == hashlib.sha256(table_bytes(_rows())).hexdigest()
    )
    # and the group keys move with it
    assert _keys(_plans(one, env_one)).isdisjoint(_keys(_plans(two, env_two)))


def test_a_canonical_form_without_a_digest_is_refused(tmp_path: Path) -> None:
    """Fail closed: handed a canonical form that stamps no digest for a role,
    the identity refuses rather than naming the ref — there is no fallback to
    the name."""
    from causalab.protocol.rules.errors import ProtocolError

    env = _double_env(_rows(), tmp_path)
    one = compile_protocol(in_order(base_doc()), env=env, base_dir=tmp_path)
    steps = steps_of(one, env)
    pdoc, canon = steps.documents[0], dict(steps.canonical[0])
    stripped = {k: dict(v) for k, v in canon["data"].items()}
    del stripped["base"]["digest"]
    canon["data"] = stripped
    with pytest.raises(ProtocolError, match="no content digest"):
        _data_identity(pdoc, canon)
    with pytest.raises(ProtocolError, match="lockstep"):
        campaign_plans(steps.documents, ())
    # a shape mismatch is refused, never broadcast: a single-valued role handed
    # a list form (or a tuple-valued role handed one mapping) is another
    # point's canonical form
    canon = dict(steps.canonical[0])
    canon["data"] = {
        **canon["data"],
        "counterfactual": [canon["data"]["counterfactual"]],
    }
    with pytest.raises(ProtocolError, match="single-valued"):
        _data_identity(pdoc, canon)


def test_tuple_valued_counterfactual_indexes_the_canonical_list(tmp_path: Path) -> None:
    """A tuple-valued ``counterfactual`` is a list in the canonical form;
    role ``counterfactual[j]`` reads entry ``j`` of it, as ``resolve_roles``
    names the roles. Two entries reading two fields of one table share the
    digest half and differ in the field half."""
    doc = base_doc()
    doc["data"]["counterfactual"] = [
        {"dataset": REF, "field": "counterfactual_inputs[0]"},
        {"dataset": REF, "field": "cf_answer"},
    ]
    doc["method"]["intervened_models"][UNWRITTEN]["input"] = "counterfactual[0]"
    doc["method"]["intervened_models"]["original_counterfactual_1"] = {
        "input": "counterfactual[1]",
        "reads": ["v_cf2"],
    }
    doc["method"]["reads"]["v_cf2"] = {"site": "tgt", "pos": -1}
    doc["method"]["save"].append(
        saved("v_cf2", "original_counterfactual_1", "cf2.safetensors")
    )
    env = _double_env(_rows(), tmp_path)
    loaded = compile_protocol(in_order(doc), env=env, base_dir=tmp_path)
    steps = steps_of(loaded, env)
    pdoc, canon = steps.documents[0], steps.canonical[0]
    assert isinstance(canon["data"]["counterfactual"], list)
    ident = _data_identity(pdoc, canon)
    assert set(ident) == {"base", "counterfactual[0]", "counterfactual[1]"}
    digest = canon["data"]["base"]["digest"]
    assert ident["counterfactual[0]"] == f"{digest}#counterfactual_inputs[0]"
    assert ident["counterfactual[1]"] == f"{digest}#cf_answer"
    for j in (0, 1):
        assert ident[f"counterfactual[{j}]"].startswith(
            canon["data"]["counterfactual"][j]["digest"]
        )


# --------------------------------------------------------------------------- #
# T4 — the fold half lines up
# --------------------------------------------------------------------------- #


def test_the_fold_half_is_the_canonical_data_digest(env) -> None:
    """Corpus 08 fits on ``weekdays/data#train`` and evaluates on
    ``weekdays/data#test``. The ``base`` identity is exactly the canonical
    ``data.base.digest`` — the fold digest method identity is built from —
    and it differs from ``train.eval.digest``: two splits of one table are
    two identities, though they share a file and a name."""
    loaded = compile_protocol(DAS_SWEEP, env=env)
    for pdoc, canon in zip(
        steps_of(loaded, env).documents, steps_of(loaded, env).canonical
    ):
        ident = _data_identity(pdoc, canon)
        base_digest = canon["data"]["base"]["digest"]
        assert ident["base"] == f"{base_digest}#input"
        assert ident["counterfactual"] == f"{base_digest}#counterfactual_inputs[0]"
        assert canon["method"]["train"]["eval"]["digest"] != base_digest
    # the digests are the resolver's, not a second hashing of the same rows
    assert canon["data"]["base"]["digest"] == env.datasets.digest(REF)
    assert canon["method"]["train"]["eval"]["digest"] == env.datasets.digest(
        "weekdays/data#test"
    )


# --------------------------------------------------------------------------- #
# T5a — valid work still interns, through the real identity
# --------------------------------------------------------------------------- #


def test_locate_scan_still_owes_65_with_the_real_identity(env) -> None:
    """The counts ``test_corpus.py`` pins with a hand-written identity, run
    through ``campaign_plans`` with the content identity: 07's 128 group
    instances still intern to 65 (one shared harvest with 32 taps + 64
    patched), because every point reads one table."""
    loaded = compile_protocol(SCAN, env=env)
    plans = _plans(loaded, env)
    assert sum(plan.num_forwards for plan in plans) == 128
    groups = interned_groups(plans)
    assert len(groups) == 65
    (harvest,) = [g for g in groups if g.model == "original_counterfactual"]
    assert len(harvest.taps) == 32
    assert len([g for g in groups if g.model != "original_counterfactual"]) == 64


def test_das_sweep_still_interns_one_harvest_with_the_real_identity(env) -> None:
    loaded = compile_protocol(DAS_SWEEP, env=env)
    plans = _plans(loaded, env)
    assert len(_by_model(plans, "original_counterfactual")) == 1
    assert len(_by_model(plans, "patched")) == 9
    assert len(interned_groups(plans)) == 10


def test_the_tiny_scan_shape_holds_with_the_real_identity(env) -> None:
    """The shape ``test_forward_interning`` runs on tiny-llama, planned here
    without a model: 4 points x 2 groups intern to 5 (4 patched + 1 shared
    harvest carrying one tap per layer)."""
    loaded = compile_protocol(
        SCAN,
        env=env,
        overrides={
            "sites.target.layers": {"sweep": [0, 1]},
            "positions.tap": {"sweep": [{"index": -1}, {"index": -2}]},
        },
    )
    plans = _plans(loaded, env)
    groups: list[ForwardGroup] = list(interned_groups(plans))
    assert sum(plan.num_forwards for plan in plans) == 8
    assert len(groups) == 5
    (harvest,) = [g for g in groups if g.model == "original_counterfactual"]
    assert len(harvest.taps) == 2
