"""Build the bundled, model-agnostic relation JSON for ``subject_object_relations``.

Ingests the LRE-relation DAS bundle at ``SOURCE_DIR`` (Llama-3.1-8B provenance)
and writes, per relation, a compact JSON carrying only the *relation content* a
causalab task needs — distinct subjects, the deterministic subject→object map,
distinct objects, and the deduped prompt templates. **All Llama-specific token /
position fields are dropped** (``prompt``, ``prompt_token_ids``, ``gold_first_id``,
``subj_last_idx``/``subj_last``, ``prompt_last_idx``/``prompt_last``,
``pred_first_id``, ``subj_char_*``, ``seq_len``): causalab recomputes tokens and
positions per tokenizer.

The bundle's ``dataset/filtered.jsonl`` is the **authoritative** ingest source —
every row has ``subject`` / ``object`` / ``template`` / ``template_idx``. The 11
"bias" relations ship a *stripped* ``filtered.meta.json`` (no ``templates`` list,
empty ``object_first_ids``), so templates and objects are read from the JSONL, not
the meta. Group / effective-dimension provenance comes from the bundle-level
``effective_dimension.parquet`` (site ``last_token``); ``category`` / ``range_name``
come from the per-relation meta when present.

Source templates use a positional ``{}`` subject slot; they are rewritten to the
named ``{subject}`` placeholder causalab's token-position parser requires.

Run (from the repo root)::

    uv run python causalab/tasks/subject_object_relations/sources/build_relations.py

Outputs (committed, ~<3 MB total; runtime never reads ``external artifact storage``):
    sources/relations/<relation>.json   (35 files)
    sources/manifest.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# Provenance only — the generated JSON is committed, so runtime never touches this.
SOURCE_DIR = Path("<lre-relations-source>")

_HERE = Path(__file__).resolve().parent
RELATIONS_DIR = _HERE / "relations"
MANIFEST_PATH = _HERE / "manifest.json"

# Fields that must NEVER appear in the committed JSON (Llama-8B-specific).
_FORBIDDEN_KEYS = {
    "prompt",
    "prompt_token_ids",
    "gold_first_id",
    "pred_first_id",
    "subj_last_idx",
    "subj_last",
    "prompt_last_idx",
    "prompt_last",
    "subj_char_start",
    "subj_char_end",
    "seq_len",
    "candidate_first_ids",
    "object_first_ids",
}

_SUBJECT_PLACEHOLDER = "{subject}"

# Matches a *literal* backslash-u / backslash-x escape that survived into the
# decoded string — i.e. the upstream JSONL double-escaped the value, so
# ``json.loads`` yields a real backslash rather than the intended character.
# Handles lowercase ``\uXXXX`` / ``\xXX`` only; uppercase ``\U########`` and
# astral surrogate-pair escapes are NOT decoded (none occur in this data). So
# this is a targeted decoder for the observed double-escape, not an exhaustive
# "no stray backslash" scrubber.
_LITERAL_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})|\\x([0-9a-fA-F]{2})")


def _decode_literal_escapes(s: str) -> str:
    """Decode surviving literal ``\\uXXXX`` / ``\\xXX`` escapes to real characters.

    Root-cause normalization for values that the upstream ``filtered.jsonl``
    double-escaped: e.g. the file byte sequence ``"Bras\\\\u00edlia"`` decodes via
    ``json.loads`` to the 13-char literal ``Bras\\u00edlia`` (a real backslash),
    which the model can never emit. Applied uniformly to every extracted subject /
    object / template so the class of bug is fixed at ingest, not per-relation.
    Strings with no such escape are returned unchanged.
    """

    def _repl(m: re.Match[str]) -> str:
        hex_digits = m.group(1) or m.group(2)
        return chr(int(hex_digits, 16))

    return _LITERAL_ESCAPE_RE.sub(_repl, s)


def _to_named_template(raw: str) -> str:
    """Rewrite a source template's single positional ``{}`` slot to ``{subject}``.

    causalab's token-position parser (``causalab/tasks/token_positions.py``)
    matches ``\\{([^}]+)\\}`` named placeholders and fills them from trace
    variables, so a positional ``{}`` slot would not resolve.
    """
    n = raw.count("{}")
    if n != 1:
        raise ValueError(
            f"expected exactly one positional '{{}}' subject slot, got {n}: {raw!r}"
        )
    return raw.replace("{}", _SUBJECT_PLACEHOLDER)


def _discover_relations() -> list[str]:
    """Every source subdir carrying a ``dataset/filtered.jsonl`` — the in-scope set."""
    out = []
    for child in sorted(SOURCE_DIR.iterdir()):
        if child.is_dir() and (child / "dataset" / "filtered.jsonl").is_file():
            out.append(child.name)
    return out


def _load_provenance() -> dict[str, dict]:
    """Per-relation group / C / eff_k / compact from ``effective_dimension.parquet``.

    One row per (relation, site); we take the ``last_token`` (answer-token) site,
    the site the effective-dimension law is stated over.
    """
    df = pd.read_parquet(SOURCE_DIR / "effective_dimension.parquet")
    df = df[df["site"] == "last_token"]
    prov: dict[str, dict] = {}
    for _, row in df.iterrows():
        prov[str(row["relation"])] = {
            "group": str(row["group"]),
            "bundle": str(row["bundle"]),
            "C": int(row["C"]),
            "n_distinct_obj": int(row["n_distinct_obj"]),
            "eff_k": None if pd.isna(row["eff_k"]) else int(row["eff_k"]),
            "compact": bool(row["compact"]),
        }
    return prov


def _build_relation(name: str, prov: dict) -> dict:
    """Extract the model-agnostic content for one relation from its JSONL + meta."""
    rel_dir = SOURCE_DIR / name
    jsonl = rel_dir / "dataset" / "filtered.jsonl"

    subjects: list[str] = []
    objects: list[str] = []
    templates: list[str] = []
    subject_to_object: dict[str, str] = {}
    n_rows = 0
    n_conflicts = 0

    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        n_rows += 1
        # Normalize upstream double-escapes (literal \uXXXX / \xXX) at ingest,
        # uniformly across every relation, so no committed value carries a real
        # backslash the model can never emit.
        subj = _decode_literal_escapes(row["subject"])
        obj = _decode_literal_escapes(row["object"])
        tmpl = _to_named_template(_decode_literal_escapes(row["template"]))

        if subj not in subject_to_object:
            subject_to_object[subj] = obj
            subjects.append(subj)
        elif subject_to_object[subj] != obj:
            # Non-deterministic subject→object in the source. Keep the first
            # (deterministic) mapping; count it so the build surfaces the drift.
            n_conflicts += 1
        if obj not in objects:
            objects.append(obj)
        if tmpl not in templates:
            templates.append(tmpl)

    # category / range_name from the per-relation meta when present (the bias
    # relations' stripped meta carries neither category nor a templates list).
    meta_path = rel_dir / "dataset" / "filtered.meta.json"
    category = None
    range_name = None
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        category = meta.get("category")
        range_name = meta.get("range_name")

    record = {
        "relation": name,
        "group": prov.get("group"),
        "category": category,
        "range_name": range_name,
        "templates": templates,
        "subjects": subjects,
        "subject_to_object": subject_to_object,
        "objects": objects,
        "provenance": {
            "bundle": prov.get("bundle"),
            "C": prov.get("C"),
            "eff_k": prov.get("eff_k"),
            "compact": prov.get("compact"),
            "n_source_rows": n_rows,
            "n_subject_object_conflicts": n_conflicts,
        },
    }
    return record


def _assert_no_forbidden(record: dict) -> None:
    """Belt-and-suspenders: no Llama token/position key leaked into the output."""
    blob = json.dumps(record)
    # Only guard against forbidden fields appearing as JSON *keys*.
    for key in _FORBIDDEN_KEYS:
        if f'"{key}"' in blob:
            raise AssertionError(
                f"relation {record['relation']!r} JSON contains forbidden key {key!r}"
            )


def _assert_no_literal_escapes(record: dict) -> None:
    """No extracted string value retains a literal ``\\u`` / ``\\x`` escape.

    Post-normalization guard: catches an upstream double-escape that
    `_decode_literal_escapes` failed to cover, before it round-trips into a
    committed, unscoreable value.
    """

    def _walk(obj) -> None:
        if isinstance(obj, str):
            if _LITERAL_ESCAPE_RE.search(obj):
                raise AssertionError(
                    f"relation {record['relation']!r} has an unescaped literal "
                    f"backslash-escape in a string value: {obj!r}"
                )
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _walk(k)
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(record)


def main() -> None:
    RELATIONS_DIR.mkdir(parents=True, exist_ok=True)
    relations = _discover_relations()
    prov_all = _load_provenance()

    manifest_relations: dict[str, dict] = {}
    total_bytes = 0
    print(
        f"{'relation':35s} {'group':11s} {'subj':>5s} {'obj':>4s} {'tmpl':>4s} {'conf':>4s}"
    )
    for name in relations:
        record = _build_relation(name, prov_all.get(name, {}))
        _assert_no_forbidden(record)
        _assert_no_literal_escapes(record)

        out_path = RELATIONS_DIR / f"{name}.json"
        payload = json.dumps(record, ensure_ascii=False, indent=1, sort_keys=True)
        out_path.write_text(payload, encoding="utf-8")
        total_bytes += len(payload.encode("utf-8"))

        manifest_relations[name] = {
            "group": record["group"],
            "category": record["category"],
            "n_subjects": len(record["subjects"]),
            "n_objects": len(record["objects"]),
            "n_templates": len(record["templates"]),
            "C": record["provenance"]["C"],
            "eff_k": record["provenance"]["eff_k"],
            "compact": record["provenance"]["compact"],
        }
        print(
            f"{name:35s} {str(record['group']):11s} "
            f"{len(record['subjects']):5d} {len(record['objects']):4d} "
            f"{len(record['templates']):4d} "
            f"{record['provenance']['n_subject_object_conflicts']:4d}"
        )

    manifest = {
        "source": str(SOURCE_DIR),
        "n_relations": len(relations),
        "note": (
            "Model-agnostic relation content extracted from the LRE-relation DAS "
            "bundle; all Llama-3.1-8B token/position fields dropped. filtered.jsonl "
            "is the authoritative ingest source."
        ),
        "relations": manifest_relations,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"\nWrote {len(relations)} relation files + manifest.json "
        f"({total_bytes / 1e6:.2f} MB of relation JSON)."
    )


if __name__ == "__main__":
    main()
