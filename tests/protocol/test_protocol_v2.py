"""Intervention protocol v2 — one file, four groups (spec §1, §7, §9).

What this file pins:

* the **shape**: four required groups, a header of three fields, the twelve
  method sections in their table; unknown groups and sections refuse with
  suggestions; a v1 document (top-level ``version``) refuses by name and points
  at ``causalab migrate``; a workflow handed to the parser refuses as one;
* **paths are section-rooted** (§1): ``sites.target.layers`` is the spelling in
  ``--set``, in sweep axis ids and in the tree-path helpers, and never carries
  the group;
* the **canonical form** keeps ``protocol_version`` and drops ``title`` and
  ``description`` (§7), so renaming a document moves no digest — the property
  ``test_method_presets`` used to ``xfail`` on;
* there is **no method digest**: the two digests a compile reports are the
  document's and its points' — the identities ``--resume`` compares — and a
  loaded protocol and a run receipt carry no third;
* **migrate** is the pure v1 → v2 rewrite: total on flat v1 documents,
  idempotent on v2 and on workflows, refusing on a split document; the
  markdown rewrite touches only whole v1 examples; and the census — the spec's
  §1 tables and the code's constants are one list.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from causalab.cli import main
from causalab.protocol.lowering import AXES_KEY
from causalab.protocol.schema.explicit import canonicalize, digest
from causalab.protocol.rules.errors import ParseError, ProtocolWarning
from causalab.protocol.pipeline import compile_protocol
from causalab.io.sources import apply_overrides
from causalab.protocol.migrate import (
    format_document,
    migrate_document,
    migrate_markdown,
)
from causalab.protocol.schema import (
    GROUP_ORDER,
    HEADER_FIELDS,
    METHOD_SECTIONS,
    PROTOCOL_VERSION,
    REQUIRED_METHOD_SECTIONS,
    SECTION_ORDER,
    dotted_path,
    parse_document,
    tree_path,
)
from causalab.protocol.lowering import find_axes

from tests.protocol._docs import base_doc, in_order
from tests.protocol._env import FIXTURES, fixture_input_overrides, steps_of
from tests.protocol.test_site_layers import v3_base_doc
from tests._helpers.paths import PROTOCOLS_DIR

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "docs" / "intervention_protocol.md"


# --------------------------------------------------------------------------- #
# shape
# --------------------------------------------------------------------------- #


def test_the_four_groups_parse_and_flatten_to_attributes():
    doc = parse_document(base_doc())
    assert doc.protocol_version == PROTOCOL_VERSION
    assert doc.title is None and doc.description is None
    assert set(doc.sites) == {"tgt", "lm_head"}
    assert doc.raw["method"]["sites"]["tgt"]["layers"] == [3]  # grouped as authored


def test_header_carries_title_and_description():
    raw = base_doc()
    raw["header"].update(title="Interchange on gpt2", description="the intent")
    doc = parse_document(raw)
    assert doc.title == "Interchange on gpt2"
    assert doc.description == "the intent"


@pytest.mark.parametrize("group", GROUP_ORDER)
def test_every_group_is_required(group):
    raw = base_doc()
    del raw[group]
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert err.value.code == "P2" and group in str(err.value)


def test_unknown_group_refuses_with_a_suggestion():
    raw = base_doc()
    raw["methods"] = raw.pop("method")
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert err.value.code == "P3" and "method" in str(err.value)


def test_a_method_section_at_the_top_level_is_an_unknown_group():
    """The twelve sections live under `method`; the refusal names the groups."""
    raw = base_doc()
    raw["sites"] = raw["method"].pop("sites")
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert err.value.code == "P3" and "'sites'" in str(err.value)


def test_unknown_method_section_refuses_with_a_suggestion():
    raw = base_doc()
    raw["method"]["read"] = raw["method"].pop("reads")
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert err.value.code == "P3" and "reads" in str(err.value)


@pytest.mark.parametrize("section", sorted(REQUIRED_METHOD_SECTIONS))
def test_required_method_sections(section):
    raw = base_doc()
    del raw["method"][section]
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert err.value.code == "P2" and section in str(err.value)


def test_header_fields_are_closed_and_free_text():
    raw = base_doc()
    raw["header"]["titel"] = "x"
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert err.value.code == "P3" and "title" in str(err.value)
    raw = base_doc()
    raw["header"]["title"] = {"sweep": ["a", "b"]}
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert "free text" in str(err.value)


def test_protocol_version_is_required_and_pinned():
    raw = base_doc()
    del raw["header"]["protocol_version"]
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert "protocol_version" in str(err.value)
    raw = base_doc()
    raw["header"]["protocol_version"] = "1"
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert "unsupported protocol_version '1'" in str(err.value)
    raw = base_doc()
    raw["header"]["protocol_version"] = 2  # an integer is not the string "2"
    with pytest.raises(ParseError):
        parse_document(raw)


def test_rule_18_a_v1_document_is_refused_by_name_and_told_how_to_migrate():
    v1 = {"version": "1", **base_doc()["method"]}
    v1["model"] = base_doc()["model"]
    v1["data"] = base_doc()["data"]
    with pytest.raises(ParseError) as err:
        parse_document(v1)
    assert "v1 document" in str(err.value)
    assert "causalab migrate" in str(err.value)


def test_a_workflow_handed_to_the_parser_is_refused_as_one():
    with pytest.raises(ParseError) as err:
        parse_document({"version": "1", "output_dir": "r", "steps": {}})
    assert "workflow document" in str(err.value)


def test_unconventional_order_warns_for_groups_and_for_method_sections():
    raw = base_doc()
    reordered = {"model": raw["model"], "header": raw["header"]}
    reordered.update({k: v for k, v in raw.items() if k not in reordered})
    with pytest.warns(ProtocolWarning, match="groups are not"):
        parse_document(reordered)
    raw = base_doc()
    save = raw["method"].pop("save")
    raw["method"] = {"save": save, **raw["method"]}
    with pytest.warns(ProtocolWarning, match="method sections are not"):
        parse_document(raw)
    assert list(in_order(raw)["method"]) == [
        s for s in METHOD_SECTIONS if s in raw["method"]
    ]


# --------------------------------------------------------------------------- #
# paths are section-rooted
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "dotted, path",
    [
        ("sites.target.layers", ("method", "sites", "target", "layers")),
        ("model.dtype", ("model", "dtype")),
        ("data.base.field", ("data", "base", "field")),
        ("train.seed", ("method", "train", "seed")),
        ("header.title", ("header", "title")),
    ],
)
def test_tree_path_and_dotted_path_are_inverses(dotted, path):
    assert tree_path(dotted) == path
    assert dotted_path(path) == dotted


def test_set_addresses_sections_never_groups():
    raw = base_doc()
    out = apply_overrides(raw, {"sites.tgt.layers": 5, "model.dtype": "bf16"})
    assert out["method"]["sites"]["tgt"]["layers"] == 5
    assert out["model"]["dtype"] == "bf16"
    with pytest.raises(ParseError) as err:
        apply_overrides(raw, {"method.sites.tgt.layers": 5})
    assert "paths start at the section" in str(err.value)


def test_set_on_the_bare_group_is_refused_not_a_traceback():
    """``--set method=…`` has no section to point at; the refusal says so."""
    with pytest.raises(ParseError) as err:
        apply_overrides(base_doc(), {"method": {}})
    assert "name a section" in str(err.value) and err.value.path == "method"


def test_set_addresses_an_indexed_first_segment():
    """``save`` is the method's one list, so the documented ``[i]`` syntax has
    to survive the group prefix: the index is part of the segment, not of the
    section's name."""
    assert tree_path("save[0].file_path") == ("method", "save[0]", "file_path")
    assert dotted_path(("method", "save[0]", "file_path")) == "save[0].file_path"
    out = apply_overrides(base_doc(), {"save[0].file_path": "renamed.json"})
    assert out["method"]["save"][0]["file_path"] == "renamed.json"


def test_at_once_families_expand_inside_the_method_group(env):
    """§3.1's wrappers are method sugar: the shipped band-patch preset, whose
    sites, reads and writes are families, loads to the members it denotes, and
    a wrapper where no name identity exists is refused by its section-rooted
    path."""
    from causalab.protocol.rules.errors import ValidationError

    preset = PROTOCOLS_DIR / "attention_band_patch.json"
    loaded = compile_protocol(
        preset,
        env=env,
        overrides=fixture_input_overrides(json.loads(preset.read_text())),
    )
    assert {f"a{layer}" for layer in range(9, 17)} <= set(loaded.document.sites)
    assert "at_once" not in json.dumps(loaded.tree["method"])  # the sugar is gone
    stray = base_doc()
    stray["method"]["save"][0]["aggregation"]["a"] = {
        "at_once": ["cf_answer", "base_answer"]
    }
    with pytest.raises(ValidationError) as err:
        compile_protocol(stray, env=env)
    assert err.value.rule == 28 and err.value.path == "save[0].aggregation.a"


def test_sweep_axis_ids_are_section_rooted():
    raw = base_doc()
    raw["method"]["sites"]["tgt"]["layers"] = {"sweep": [1, 2]}
    (axis,) = find_axes(raw)
    assert axis.path == ("method", "sites", "tgt", "layers")
    assert axis.id == "sites.tgt.layers"


# --------------------------------------------------------------------------- #
# canonical form and digests
# --------------------------------------------------------------------------- #


def test_title_and_description_are_not_in_the_digest(env):
    plain = compile_protocol(base_doc(), env=env)
    named = base_doc()
    named["header"].update(title="a name", description="what it is for")
    renamed = compile_protocol(named, env=env)
    assert renamed.digests.document == plain.digests.document
    assert steps_of(renamed, env).digests == steps_of(plain, env).digests
    assert renamed.canonical["header"] == {"protocol_version": PROTOCOL_VERSION}
    assert list(renamed.canonical) == list(GROUP_ORDER)


def test_protocol_version_is_in_the_digest(env):
    canonical = canonicalize(base_doc(), env)
    assert canonical["header"]["protocol_version"] == PROTOCOL_VERSION
    other = json.loads(json.dumps(canonical))
    other["header"]["protocol_version"] = "5"
    assert digest(other) != digest(canonical)


def test_there_is_no_method_digest(env):
    """A compile reports the document digest and nothing else (§7; the point
    digests are the engine's to sign): the method digest was a
    third identity that ``--resume`` never compared and no record needed, so
    a loaded protocol has no such field and a sugar-only respelling is one
    experiment under every digest there is."""
    import dataclasses

    sugar = compile_protocol(base_doc(), env=env)
    assert not hasattr(sugar, "method_digest")
    assert [f.name for f in dataclasses.fields(sugar.digests)] == ["document"]
    explicit_raw = base_doc()
    explicit_raw["method"]["reads"]["v_cf"]["pos"] = {"index": -1}
    explicit = compile_protocol(explicit_raw, env=env)
    assert explicit.digests.document == sugar.digests.document
    assert steps_of(explicit, env).digests == steps_of(sugar, env).digests


# --------------------------------------------------------------------------- #
# migrate
# --------------------------------------------------------------------------- #


def _v1(doc: dict[str, Any]) -> dict[str, Any]:
    """The v1 spelling of a grouped document, for the round trip — fed the
    protocol-3 ancestor, since a v1 body is spelled like one (§9)."""
    out: dict[str, Any] = {"version": "1"}
    if "description" in doc["header"]:
        out["description"] = doc["header"]["description"]
    out["model"] = doc["model"]
    out["data"] = doc["data"]
    out.update(doc["method"])
    return out


def test_migrate_regroups_a_flat_v1_document_and_is_idempotent():
    v2 = v3_base_doc()
    v2["header"]["description"] = "the intent"
    current = base_doc()
    current["header"]["description"] = "the intent"
    assert migrate_document(_v1(v2)) == current
    assert migrate_document(current) == current
    workflow = {"version": "1", "output_dir": "r", "steps": {}}
    assert migrate_document(workflow) == workflow


def test_migrate_drops_type_and_the_neural_model_alias():
    v1 = _v1(v3_base_doc())
    v1["type"] = "protocol"
    v1["neural_model"] = v1.pop("model")
    assert migrate_document(v1) == base_doc()


def test_migrate_refuses_a_split_document_and_a_method_file():
    v1 = _v1(v3_base_doc())
    split = {
        "version": "1",
        "application": {"model": v1["model"], "data": v1["data"]},
        "method": {
            k: v for k, v in v1.items() if k not in ("version", "model", "data")
        },
    }
    with pytest.raises(ParseError, match="split"):
        migrate_document(split)
    with pytest.raises(ParseError, match="method file"):
        migrate_document({"version": "1", "type": "method", "reads": {}, "save": []})
    with pytest.raises(ParseError, match="unsupported version"):
        migrate_document({"version": "0", "model": {}, "data": {}})


def test_format_document_writes_shallow_objects_on_one_line():
    text = format_document(base_doc())
    assert text.endswith("\n")
    assert json.loads(text) == base_doc()
    assert '"tgt": {"component": "block_output", "layers": [3]}' in text
    assert '"patch": {"site": "tgt", "pos": -1, "do": {"swap": "v_cf"}}' in text
    assert text.startswith(
        '{\n  "header": {"protocol_version": "' + PROTOCOL_VERSION + '"},\n  "model":'
    )


def test_migrate_markdown_rewrites_whole_v1_examples_only():
    v1 = _v1(v3_base_doc())
    prose = (
        "Some prose.\n\n```json\n"
        + json.dumps(v1, indent=2)
        + '\n```\n\nA fragment:\n\n```json\n{"sites": {"target": {...}}}\n```\n\n'
        'A workflow:\n\n```json\n{"version": "1", "output_dir": "r", "steps": {}}\n```\n\n'
        "  Indented:\n\n  ```json\n"
        + "\n".join("  " + line for line in json.dumps(v1, indent=2).splitlines())
        + "\n  ```\n"
    )
    out = migrate_markdown(prose)
    assert out.count('"protocol_version": "4"') == 2
    assert '{"sites": {"target": {...}}}' in out  # the fragment is untouched
    assert (
        '{"version": "1", "output_dir": "r", "steps": {}}' in out
    )  # so is the workflow
    assert "\n  ```json\n  {\n" in out  # indentation kept
    assert migrate_markdown(out) == out


def test_the_migrate_verb_rewrites_in_place_and_check_reports(tmp_path, capsys):
    v1 = _v1(v3_base_doc())
    document = tmp_path / "old.json"
    document.write_text(json.dumps(v1))
    workflow = tmp_path / "wf.json"
    workflow.write_text(json.dumps({"version": "1", "output_dir": "r", "steps": {}}))
    assert main(["migrate", "--check", str(document), str(workflow)]) == 1
    assert "would migrate" in capsys.readouterr().out
    assert json.loads(document.read_text()) == v1  # nothing written
    assert main(["migrate", str(document), str(workflow)]) == 0
    assert json.loads(document.read_text()) == base_doc()
    assert json.loads(workflow.read_text())["version"] == "1"
    assert main(["migrate", "--check", str(document), str(workflow)]) == 0
    # a missing path, a directory and a YAML document are refusals on stderr,
    # not tracebacks — and the files before them stay migrated
    again = tmp_path / "again.json"
    again.write_text(json.dumps(v1))
    yaml_doc = tmp_path / "old.yaml"
    yaml_doc.write_text("version: '1'\n")
    assert (
        main(
            [
                "migrate",
                str(again),
                str(tmp_path / "missing.json"),
                str(tmp_path),
                str(yaml_doc),
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert err.count("refused:") == 3
    assert "missing.json" in err and "YAML" in err and "comments" in err
    assert json.loads(again.read_text()) == base_doc()
    assert yaml_doc.read_text() == "version: '1'\n"
    assert (
        main(
            [
                "validate",
                "--engine",
                "auto",
                str(document),
                "--data-root",
                str(FIXTURES / "data"),
                "--artifacts-root",
                str(tmp_path),
            ]
        )
        == 0
    )


# --------------------------------------------------------------------------- #
# census — the spec's §1 tables are the code's constants
# --------------------------------------------------------------------------- #


def _section(heading: str) -> str:
    depth = len(heading) - len(heading.lstrip("#"))
    body = SPEC.read_text().split(heading, 1)[1]
    stop = re.compile(rf"^#{{1,{depth}}} ", re.MULTILINE)
    match = stop.search(body)
    return body if match is None else body[: match.start()]


def _tables(text: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if all(set(cell) <= set("-: ") for cell in cells):
                continue
            current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def _keys(table: list[list[str]], column: str) -> tuple[str, ...]:
    index = table[0].index(column)
    return tuple(row[index].strip("`") for row in table[1:])


def test_the_spec_section_one_tables_are_the_constants():
    tables = _tables(_section("## 1. Document layout"))
    groups = next(t for t in tables if _keys(t, "key")[0] == "header")
    # The ✓ rows are the required groups; the – rows are the optional pre-gate
    # groups the compiler lowers before the shape gate (axes, §3.2), so a
    # second optional group must be declared beside AXES_KEY as well.
    group_flags = tuple(zip(_keys(groups, "key"), _keys(groups, "required")))
    assert tuple(key for key, flag in group_flags if flag == "✓") == GROUP_ORDER
    assert tuple(key for key, flag in group_flags if flag == "–") == (AXES_KEY,)
    assert _keys(groups, "key") == (*GROUP_ORDER, AXES_KEY)
    header = next(t for t in tables if _keys(t, "key")[0] == "protocol_version")
    assert _keys(header, "key") == HEADER_FIELDS
    method = next(t for t in tables if _keys(t, "key")[0] == "intervened_models")
    assert _keys(method, "key") == METHOD_SECTIONS
    required = {
        key
        for key, flag in zip(_keys(method, "key"), _keys(method, "required"))
        if flag == "✓"
    }
    assert required == REQUIRED_METHOD_SECTIONS
    assert SECTION_ORDER == ("model", "data", *METHOD_SECTIONS)
