"""Validate resolved dataset tables and fitted artifacts.

Checks cover column references, row roles, fit splits, and the identity fields
required when a document loads an artifact. They run against the compiled
specification and its resolution environment."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from causalab.causal.pair_validation import (
    EDIT_GROUPS_COLUMN,
    EditGroupError,
    parse_edit_groups,
)
from causalab.causal.scoring import ScoringError, ScoringMismatch, check_scoring
from causalab.causal.scoring import declared_modes
from causalab.protocol.bundles import entry_selection, select_entry, selector_slot
from causalab.protocol.schema.explicit import canonical_model
from causalab.protocol.results import EXAMPLE_ID_COLUMN, example_id_defect
from causalab.protocol.identity import (
    build_artifact_identity,
    check_artifact_identity,
    single_site_featurizers,
)
from causalab.protocol.registry import site_group_map
from causalab.io.env import (
    DatasetResolver,
    ResolutionEnv,
    endpoints,
    split_dataset_ref,
)
from causalab.protocol.rules.document import im_writes
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.schema import (
    metric_column_fields,
    FEATURIZER_SLOTS,
    HARD_CONCRETE_STRETCH,
    MINIMUM_COUNT_FIELD,
    DataRole,
    Document,
    FeaturizerSpec,
    AggregationSpec,
    PositionSpec,
)
from causalab.protocol.positions.spans import walk

if TYPE_CHECKING:
    from causalab.protocol.compiled import CompiledProtocol

__all__ = [
    "HELD_OUT_ROLE",
    "check_data_columns",
    "check_fit_splits",
    "check_loaded_featurizers",
    "check_minimum_counts",
    "check_row_roles",
    "check_start_site",
    "fit_roles",
    "maximum_eligible_count",
]


def _is_default_stretch(stamped: Any) -> bool:
    """Whether a bundle's stamped ``stretch`` (a JSON list, or the list itself)
    is the unauthored default, which a document authoring none implies."""
    try:
        values = json.loads(stamped) if isinstance(stamped, str) else list(stamped)
        lo, hi = float(values[0]), float(values[1])
    except (TypeError, ValueError, IndexError, json.JSONDecodeError):
        return False
    return (lo, hi) == HARD_CONCRETE_STRETCH


def check_loaded_featurizers(
    doc: Document,
    env: ResolutionEnv,
    *,
    coords: Mapping[str, Any] | None = None,
) -> None:
    """§2.5/§8: every ``file_path`` load is checked at load time. A
    featurizer bundle's stamped ArtifactIdentity must match what the
    document implies (model, site record, k, parametrization, dtype, and —
    for a grouped gate — the group and the map the registry derives for it);
    a ``params`` entry's file must exist, and its identity — when stamped —
    must name the same model (free constant tensors may come from outside
    causalab, so an unstamped params file is existence-checked only; a
    stamped one must not contradict the document).

    A ``subspace`` with an ``init`` basis is checked the same way against the
    fields a *starting point* has to share with the fit — the model
    realization and the site — and not against the fields the basis owns
    (its rank, its dtype, the data it was fitted on).

    A bundle written by a *swept* producer stamps per entry, not per file
    (§8): the fields that differ between points live in the header's
    ``entries`` table. The check therefore looks at the record of the entry
    this document selects, whenever that entry is knowable here — an
    authored ``entry``, or a bundle holding exactly one record for the slot.
    When the selection is the executing point's (implicit matching off its
    own coordinates, §2.5), an ``init`` basis is checked on its file-level
    stamp — the model realization and site a start must share with the fit
    are the producer's, so a swept producer stamps them file-wide unless it
    swept them.

    A ``subspace`` start whose producer swept the site stamps the site per
    entry, and each point starts from the entry at its own site. ``coords``
    are the executing point's sweep coordinates. With them, the check
    selects that point's entry from the header as the build will, and holds
    its recorded site to the point's ([`check_start_site`][]). The engine
    passes them for every point before any weights load (``check_steps``).
    Without them (the compiler's representatives), the site is left to that
    per-point check. Any other field that differs between entries, and a
    gate start's site, is reachable only through an authored ``entry``, and
    the load refuses asking for one."""
    defers = getattr(env.artifacts, "defers", None)

    for pname, pspec in doc.params.items():
        if not isinstance(pspec.file_path, str):
            continue
        if defers is not None and defers(pspec.file_path):
            continue  # a run-tree path inside a workflow — checked at run time
        stamped = env.artifacts.read_identity(pspec.file_path)  # V15 if missing
        if stamped is not None:
            stamped = _entry_identity(
                stamped,
                slot=selector_slot(pspec.entry, "value"),
                authored=pspec.entry,
                what=f"params entry {pname!r} ({pspec.file_path})",
            )
        if stamped is not None:
            check_artifact_identity(
                stamped,
                {"model_key": doc.model.key, "model_revision": doc.model.revision},
                what=f"params entry {pname!r} ({pspec.file_path})",
            )

    for fname, spec in doc.featurizers.items():
        if isinstance(spec.file_path, str):
            if defers is not None and defers(spec.file_path):
                continue  # a run-tree path inside a workflow — checked at run time
            expected: dict[str, Any] = {
                **_featurizer_realization(doc, fname),
                "k": spec.k,
                "parametrization": spec.parametrization,
                "dtype": spec.dtype if spec.dtype is not None else "fp32",
                # the two `init_*` keys are asked of a bundle ONLY when this
                # document authors `init` (§8): a document without one implies
                # nothing about a start, so a bundle stamped before the keys
                # existed still loads under it, and the keys stay write-only
                # provenance everywhere else
                **_init_expectation(spec, env, defers),
            }
            if spec.stretch is not None:
                # a hard-concrete gate's hard split is θ > logit((½−γ)/(ζ−γ)),
                # so a bundle fitted at one stretch is a different mask under
                # another: the document's authored stretch is asked of the
                # bundle (the `group` precedent — only an AUTHORED value is
                # expected; the unauthored default is the reverse check below).
                # β is not compared: it does not enter the eval-mode split, and
                # a loaded gate is only ever read in eval mode — the loop puts
                # `train.params` stages in training mode, and a `file_path`
                # featurizer may not be one (§2.5)
                expected["stretch"] = list(spec.stretch)
            if isinstance(spec.group, str):
                # the map a grouped gate was fitted over is derivable offline
                # from the registry, so "16 heads of 256" against "32 heads of
                # 128" is refused here, by name, and not by a width mismatch
                # deep in the build. Only a document that AUTHORS a group
                # expects one: an ungrouped document derives neither key, so a
                # per-coordinate gate fitted before ``group`` existed still
                # loads (§2.5, "absent, nothing changes")
                expected["group"] = spec.group
                site_record = expected.get("site")
                if isinstance(site_record, Mapping) and isinstance(
                    site_record.get("component"), str
                ):
                    info = env.model_info(str(doc.model.key))
                    head = site_record.get("head")
                    expert = site_record.get("expert")
                    expected["group_map"] = list(
                        site_group_map(
                            info,
                            spec.group,
                            site_record["component"],
                            head=head if isinstance(head, int) else None,
                            expert=expert if isinstance(expert, int) else None,
                        )
                    )
            if isinstance(spec.axis, str):
                # §2.5 `axis`: a mask over positions is not a mask over
                # coordinates, and the bundle says which it is — asked here,
                # by name, as `group` is, not by a width mismatch at the build
                expected["axis"] = spec.axis
            what = f"featurizer {fname!r} ({spec.file_path})"
            stamped = env.artifacts.read_identity(spec.file_path)
            slot = FEATURIZER_SLOTS.get(
                spec.kind if isinstance(spec.kind, str) else "identity", ()
            )
            resolved = (
                _entry_identity(stamped, slot=slot[0], authored=spec.entry, what=what)
                if stamped is not None and slot
                else stamped
            )
            if resolved is None:
                continue  # only the executing point can select — checked at build
            if resolved.get("group") is not None and not isinstance(spec.group, str):
                # the reverse mismatch: the bundle was fitted per head, and this
                # document's per-coordinate gate would read its H parameters as H
                # coordinates. Not a key the expectation carries, so said here
                raise ValidationError(
                    15,
                    f"{what}: the bundle was fitted with group {resolved['group']!r} "
                    "but this document declares no group on the gate — a mask over "
                    "units is not a mask over coordinates (§2.5)",
                )
            if resolved.get("axis") is not None and not isinstance(spec.axis, str):
                # the reverse: a bundle fitted over positions read by a
                # per-coordinate gate would take its W parameters as W
                # coordinates. Not a key the expectation carries, so said here
                raise ValidationError(
                    15,
                    f"{what}: the bundle was fitted over {resolved['axis']!r} but "
                    "this document declares no axis on the gate — a mask over "
                    "positions is not a mask over coordinates (§2.5)",
                )
            stamped_pool = resolved.get("pool")
            if stamped_pool is not None and not isinstance(spec.pool, str):
                # a pooled theta is a ranking relative to its co-members (§2.5
                # `pool`): read alone it is a different experiment. The other
                # direction is open — an UNSTAMPED bundle may join a pooled
                # readout, which is how a joint ranking of separately fitted
                # sigmoid masks (DBM's MIB curve) is cut — so `pool` is not an
                # expectation the bundle must carry, only one it may not
                # contradict
                raise ValidationError(
                    15,
                    f"{what}: the bundle was fitted in pool "
                    f"{stamped_pool!r} but this document declares no pool on "
                    "the gate — a pooled theta is a ranking relative to its "
                    "co-members, so a cut through it alone is a different "
                    "experiment (§2.5)",
                )
            if (
                stamped_pool is not None
                and isinstance(spec.pool, str)
                and str(stamped_pool) != spec.pool
            ):
                raise ValidationError(
                    15,
                    f"{what}: ArtifactIdentity mismatch on 'pool' — the bundle was "
                    f"fitted in pool {stamped_pool!r}, the document reads it "
                    f"in {spec.pool!r} (§2.5)",
                )
            stamped_param = resolved.get("parametrization")
            if (
                spec.kind == "gate"
                and stamped_param not in (None, "sigmoid")
                and not isinstance(spec.parametrization, str)
            ):
                # the reverse mismatch again: a clamp-fitted bundle's hard mask
                # is θ > ½, and a document declaring no parametrization would
                # read it as a sigmoid gate's θ > 0. Not a key the expectation
                # carries (absent means sigmoid, §2.5), so said here
                raise ValidationError(
                    15,
                    f"{what}: the bundle was fitted with parametrization "
                    f"{stamped_param!r} but this document declares none on the "
                    "gate (sigmoid) — the two hard masks differ (θ > ½ against "
                    "θ > 0), so declare the same parametrization (§2.5)",
                )
            stamped_stretch = resolved.get("stretch")
            if (
                spec.kind == "gate"
                and stamped_stretch is not None
                and spec.stretch is None
                and not _is_default_stretch(stamped_stretch)
            ):
                # the reverse mismatch once more: the bundle was fitted at a
                # stretch whose hard split is not θ > 0, and a document that
                # authors none would read it at the default's. Not a key the
                # expectation carries (absent means the default, §2.5)
                raise ValidationError(
                    15,
                    f"{what}: the bundle was fitted at stretch {stamped_stretch!r} "
                    "but this document authors none on the gate (the default "
                    f"{list(HARD_CONCRETE_STRETCH)}) — the two hard masks split θ "
                    "at different thresholds, so declare the same stretch (§2.5)",
                )
            check_artifact_identity(
                resolved,
                {key: value for key, value in expected.items() if value is not None},
                what=what,
            )
        elif spec.init is not None and "file_path" in spec.init:
            # a gate's `init.fill` is a value, not an artifact: nothing to check
            init_path = str(spec.init["file_path"])
            if defers is not None and defers(init_path):
                continue  # a run-tree path inside a workflow — checked at run time
            what = f"featurizer {fname!r} init ({init_path})"
            (start_slot,) = FEATURIZER_SLOTS[
                spec.kind if isinstance(spec.kind, str) else "subspace"
            ][:1]
            stamped = env.artifacts.read_identity(init_path)
            if stamped is None:
                raise ValidationError(
                    15,
                    f"{what}: the basis carries no ArtifactIdentity metadata — "
                    "a fit's starting point enters its record, so an "
                    "unverifiable one is refused (§2.5)",
                )
            resolved = _entry_identity(
                stamped,
                slot=start_slot,
                authored=spec.init.get("entry"),
                what=what,
            )
            # the basis must have been fitted where the subspace is trained:
            # this model realization, this site. Its rank, dtype and dataset
            # are its own — a wider basis seeds the fit by its first k columns
            # (the width and column count are checked at build, where the
            # tensor is), and a PCA over one corpus may start a fit on another
            expected = {
                key: value
                for key, value in _featurizer_realization(doc, fname).items()
                if value is not None
            }
            if resolved is None:
                # several entries, none authored: the executing point selects
                # (§2.5). The fields a start has to share with the fit are
                # the producer's, not the point's, so a swept producer stamps
                # them file-level whenever its points agree on them, and the
                # file-level stamp is checked here. The site is the one field
                # a site sweep varies. A subspace producer swept over the
                # sites the fit sweeps stamps it per entry; each point selects
                # the entry its own site coordinates name, and that entry's
                # site is checked per point. A gate start has no per-point
                # check, so its site stays a field only an authored `entry`
                # reaches, like any other per-entry field
                resolved = {k: v for k, v in stamped.items() if k != "entries"}
                site = expected.get("site")
                if spec.kind == "subspace" and "site" not in resolved and site:
                    expected = {k: v for k, v in expected.items() if k != "site"}
                    selected = (
                        None
                        if coords is None
                        else _selected_start(
                            stamped, slot=start_slot, coords=coords, name=fname
                        )
                    )
                    if selected is not None:
                        key, identity = selected
                        check_start_site(identity, site, what=f"{what} entry {key!r}")
                per_entry = sorted(key for key in expected if key not in resolved)
                if per_entry:
                    raise ValidationError(
                        15,
                        f"{what}: the bundle holds several {start_slot!r} entries and "
                        f"its file-level ArtifactIdentity carries no {per_entry} "
                        "— those fields differ between its entries, so the "
                        "document must author 'init.entry' to name the one it "
                        "starts from (§2.5)",
                    )
            check_artifact_identity(resolved, expected, what=what)


def _init_expectation(
    spec: FeaturizerSpec, env: ResolutionEnv, defers: Any
) -> dict[str, Any]:
    """What a document that authors ``init`` implies about the *start* of a
    bundle it loads (§8): the basis's data ref, and the columns a
    rank-``k`` fit takes from it. Empty for a document without
    ``init`` — which is what keeps a bundle stamped before the ``init_*`` keys
    existed loadable under every document that never asked for a start — and
    empty when the basis cannot be read here (a deferred run-tree path, a
    swept basis whose entry only the executing point can pick), where the
    build re-reads it. The parser refuses ``init`` beside ``file_path`` on the
    same featurizer (§2.5), so today no parsed document reaches a load with
    this non-empty; the expectation is built here regardless so the identity
    contract is stated once, at the loader, and a bundle is never held to
    fewer fields than its document authors."""
    if spec.init is None or "file_path" not in spec.init:
        return {}  # no start, or a gate's `fill`: a value, not an artifact
    init_path = str(spec.init["file_path"])
    if defers is not None and defers(init_path):
        return {}
    stamped = env.artifacts.read_identity(init_path)
    if stamped is None:
        return {}
    kind = spec.kind if isinstance(spec.kind, str) else "subspace"
    basis = _entry_identity(
        stamped,
        slot=FEATURIZER_SLOTS[kind][0],
        authored=spec.init.get("entry"),
        what=f"init ({init_path})",
    )
    if basis is None:
        return {}
    expected: dict[str, Any] = {"init_trained_on": basis.get("trained_on")}
    if kind == "subspace":
        # the columns a rank-k fit takes; a gate's start is the whole theta
        expected["init_components"] = (
            list(range(spec.k)) if isinstance(spec.k, int) else None
        )
    return expected


def check_start_site(
    entry: Mapping[str, Any],
    site: Mapping[str, Any],
    *,
    what: str,
) -> None:
    """Refuse the ``subspace`` start entry a point selected when the entry's
    recorded site is not the point's site (§2.5).

    ``entry`` is the selected entry's identity: the file-level stamp,
    overlaid with the entry's record. ``site`` is the point's site as a
    stamp records it, and ``what`` names the featurizer, the file and the
    entry key. The load calls this for every point before any weights load
    ([`check_loaded_featurizers`][] with ``coords``), and the build calls it
    again for a caller that builds a stack directly."""
    want = build_artifact_identity(site=dict(site))["site"]
    got = entry.get("site")
    if got is not None and str(got) == want:
        return
    recorded = "records no site" if got is None else f"was stamped {str(got)!r}"
    raise ValidationError(
        15,
        f"{what}: ArtifactIdentity mismatch on 'site': the point that selects "
        f"this entry runs at {want!r}, but the entry {recorded} (§2.5). A "
        "point selects its start entry by its own sweep coordinates, so the "
        "producer must key each entry by the site it records. To start from "
        "another entry, author 'init.entry'.",
    )


def _selected_start(
    stamped: Mapping[str, Any],
    *,
    slot: str,
    coords: Mapping[str, Any],
    name: str,
) -> tuple[str, dict[str, Any]] | None:
    """The key and identity of the start entry a point with ``coords``
    selects from a bundle's header, as the build selects it
    ([`entry_selection`][causalab.protocol.bundles.entry_selection],
    [`select_entry`][causalab.protocol.bundles.select_entry]).

    ``None`` when the header has no entries table or the selection does not
    resolve. The build then selects over the tensors themselves and reports
    its own refusal."""
    table = _entries_table(stamped)
    if not table:
        return None
    want, implicit = entry_selection(None, coords, name)
    try:
        key = select_entry(
            table.keys(),
            slot,
            want,
            what=name,
            coords_by_key=table,
            implicit=implicit,
        )
    except ValidationError:
        return None
    return key, _overlaid(stamped, table[key])


def _entries_table(stamped: Mapping[str, Any]) -> dict[str, Any] | None:
    """The header's ``entries`` table, or ``None`` when it has none or it
    does not parse."""
    raw = stamped.get("entries")
    if not isinstance(raw, str):
        return None
    try:
        table = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return table if isinstance(table, dict) and table else None


def _overlaid(stamped: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    """The file-level stamp overlaid with one entry's record (§8)."""
    merged = {k: v for k, v in stamped.items() if k != "entries"}
    merged.update({k: v for k, v in record.items() if k not in ("slot", "coords")})
    return merged


def _featurizer_realization(doc: Document, fname: str) -> dict[str, Any]:
    """What a bundle fitted *for* one featurizer must have been fitted
    against: the model realization (§8 — a rotation fitted in bf16 does not
    apply to fp32 activations just because the shapes agree) and, when the
    featurizer is used at exactly one site, that site's record
    ([`single_site_featurizers`][])."""
    realization = canonical_model(doc.raw["model"])
    expected: dict[str, Any] = {
        "model_key": doc.model.key,
        "model_revision": doc.model.revision,
        "model_dtype": realization["dtype"],
        "model_quantization": realization.get("quantization"),
        "model_attn_implementation": realization.get("attn_implementation"),
    }
    site = single_site_featurizers(doc).get(fname)
    if site is not None:
        expected["site"] = site
    return expected


def _entry_identity(
    stamped: Mapping[str, Any],
    *,
    slot: str,
    authored: Any,
    what: str,
) -> Mapping[str, Any] | None:
    """The identity of the one bundle entry a spec selects: the file-level
    stamp, overlaid with that entry's record from the header's ``entries``
    table (§8).

    Returns ``None`` when the table holds several candidates and the
    document authored no ``entry`` — the selection is then the executing
    point's, and so is the check. A bundle with no table at all is
    file-level only, which is exactly what an un-swept or hand-made bundle
    stamps.
    """
    table = _entries_table(stamped)
    if table is None:
        return stamped
    try:
        key = select_entry(
            table.keys(),
            slot,
            authored,
            what=what,
            coords_by_key=table,
        )
    except ValidationError as err:
        if authored:
            raise  # an authored selection that misses is a load error
        if err.reason == "empty_selector":
            # nothing authored and several candidates: the selection is the
            # executing point's, so the check is deferred, not failed. The
            # reason code is the contract; the message text is not.
            return None
        raise
    return _overlaid(stamped, table[key])


#: A ``<column>[j]`` field selector (§2.2). Same shape as the engine's
#: ``encoding._LIST_FIELD``; kept here so ``protocol/`` needs no import from
#: ``neural/`` to answer a question about a table.
_LIST_FIELD = re.compile(r"^([A-Za-z0-9_]+)\[(\d+)\]$")


def check_data_columns(compiled: CompiledProtocol, env: ResolutionEnv) -> list[str]:
    """The ``validate --data`` pass (§2.2) over a compiled result: every dataset
    field selector, every metric column reference and every
    ``column``/``variable`` position must exist in the resolved tables. Returns
    the checked names (for reporting); raises on a miss.

    A pass *after* the build rather than a stage of it, and deliberately: it
    reads every row of every table (prompt variables live in the rows), which
    ``digest``, ``explain`` and a workflow's load of ten inner documents
    should not pay for — the pipeline's ``validate(…, data=True)`` is where
    it runs, and the ``causalab validate`` verb pays for it by default since
    the protocol refactor (``--data`` names that default).

    References are checked against the **base** role's table, not the union of
    all of them (§2.2, rule 20). Base is the schema of a paired row, and the
    executor already believes it: ``rows_for_metrics`` returns
    ``role_rows["base"]`` and nothing else
    (``neural/shared/executor/base.py``). Checking the union accepted a
    superset of what the run can serve — a counterfactual-only column passed
    load, the model was brought up, and only then did the metric die looking
    for a column in a base row.

    A column *position* is resolved at run time against the role of the read
    it positions, not against base, so a bare subset rule would leave the
    mirror-image hole: a position on a counterfactual read naming a base-only
    column. Rule 20 closes it by requiring the two roles' column sets to be
    **equal** when they name different datasets — automatic, and free, for
    every document that names one table for both sides, which is every
    document this repo ships and the shape §3 recommends.

    **Every axis value, not just the first.** A swept axis is a set of
    documents (§3), and the coordinate that names a bad column need not be
    coordinate 0 — a tap swept over
    ``[{"index": -1}, {"variable": "subject"}]`` names its variable only at
    coordinate 1, and a pass reading the first point alone would see only the
    index. The pass runs over
    the compile's representatives — one concrete step per axis value
    ([`representatives`][causalab.protocol.compiled.CompiledProtocol.representatives]) —
    so every value every axis takes is checked once (a column a *combination*
    of axis values names, and no single value does, is the engine's to
    refuse, per step). Table reads are cached per dataset ref, so the cost is
    the number of distinct refs.

    A ``variable`` position is checked for **existence only**. What a prompt
    variable resolves to — a char span, hence a token count — needs a
    tokenizer, and the pure verbs load none (``ResolutionEnv`` carries one as a
    service that the run door alone calls — ``pipeline.resolve_positions``
    — and stays torch- and network-free here). So a variable that no
    role can name is refused here, while a variable whose window turns out to
    be ragged across rows is still the run's refusal ([V19]). That split is
    the whole of what is answerable without a model.

    **The string mode** (§2.2, §2.10). A table built from a task carries
    its ``string_mode`` in every row; a ``match``
    metric's ``mode`` is held to the table's mode under §2.10's translation
    table ([`causalab.causal.scoring.check_scoring`][]) — a ``prefix`` table
    under ``mode: exact`` is refused under rule 4, naming both modes and the
    derivation, because the document names a reference (an answer that is one
    token) the table does not resolve. An unrecorded table compares nothing.
    The same check runs again before the first forward, where the run receipt
    records its result.

    **The edit groups** (§2.2, §5 item 27). A row may declare which spans of
    its pair move together (``edit_groups``, [`causalab.causal.pair_validation`][]);
    the shape of that declaration — spans inside the row's texts, the same
    number of constituents on both sides, an ``atomic`` group with two or
    more — is document-decidable and refused here under rule 27. Whether a
    run addresses an atomic group whole is the tokenizer's question and is
    refused before the first forward (``ExecutorBase.check_edit_groups``).
    A row without the column declares nothing and is untouched.
    """
    columns_by_ref: dict[str, set[str]] = {}
    variables_by_role: dict[tuple[str, str], set[str]] = {}
    #: The role a paired row's schema comes from (§2.2). Named rather than
    #: spelled inline so the reason is visible at every use.
    BASE = "data.base"

    def columns_of(ref: str) -> set[str]:
        if ref not in columns_by_ref:
            columns_by_ref[ref] = set(env.datasets.columns(ref))
        return columns_by_ref[ref]

    rows_by_ref: dict[str, list[dict[str, Any]]] = {}

    def rows_of(ref: str) -> list[dict[str, Any]]:
        if ref not in rows_by_ref:
            rows_by_ref[ref] = env.datasets.rows(ref)
        return rows_by_ref[ref]

    def variables_of(ref: str, field: str) -> set[str]:
        key = (ref, field)
        if key not in variables_by_role:
            variables_by_role[key] = _role_variables(rows_of(ref), field)
        return variables_by_role[key]

    refs: list[str] = []
    seen: set[tuple[str, str]] = set()

    def record(where: str, name: str) -> bool:
        """Report the reference, and say whether it still needs checking."""
        refs.append(name)
        if (where, name) in seen:
            return False
        seen.add((where, name))
        return True

    for doc in compiled.representatives:
        columns_by_role: dict[str, set[str]] = {}
        datasets: dict[str, str] = {}
        variables: set[str] = set()
        for where, role in _data_roles(doc):
            if not isinstance(role.dataset, str):
                continue
            columns_by_role[where] = columns_of(role.dataset)
            datasets[where] = role.dataset
            field = str(role.field)
            base_field = field.split("[", 1)[0]
            if base_field not in columns_of(role.dataset):
                raise ValidationError(
                    4,
                    f"data field {field!r} is not a column of {role.dataset!r}",
                    path=where,
                )
            # the field the forward reads — `<column>[eval]` for a drawn role
            # (§2.2) — so a per-member `_variables` sibling is read as the
            # engine reads it, not skipped on the bare column
            variables.update(variables_of(role.dataset, role.resolved_field))
        columns = columns_by_role.get(BASE, set())
        for agg in doc.aggregations():
            qname, metric = agg.label, agg.spec
            # which fields are columns is one predicate, shared with the
            # run-time eligibility check (§2.10, `metric_column_fields`)
            for field, value in metric_column_fields(metric).items():
                if not record(f"{agg.owner}.aggregation.{field}", value):
                    continue
                if value not in columns:
                    raise ValidationError(
                        4,
                        f"aggregation {qname!r} references column {value!r}"
                        + _why_not_in_base(value, columns_by_role, datasets, base=BASE),
                        path=f"{agg.owner}.aggregation.{field}",
                    )
        # a declared decision threshold against the most rows the base table
        # can make eligible (§2.10 "Eligibility"): needs the rows, so it is
        # this pass's and not the bare load's, like the column half above
        base_ref = datasets.get(BASE)
        if base_ref is not None:
            check_minimum_counts(doc, rows_of(base_ref), base_ref)
        # an `example_id` column is the rows' label (§2.2): present on every
        # row, non-empty, unique — or absent, and the row index labels them
        for where, ref in datasets.items():
            if not record(f"{where}.{EXAMPLE_ID_COLUMN}", ref):
                continue
            defect = example_id_defect(rows_of(ref))
            if defect is not None:
                raise ValidationError(
                    4,
                    f"dataset {ref!r}: {defect} — an {EXAMPLE_ID_COLUMN} column must "
                    "label every row, uniquely",
                    path=where,
                )
        for where, name in _column_position_refs(doc):
            if not record(where, name):
                continue
            if name not in columns:
                raise ValidationError(
                    4,
                    f"position {where} references column {name!r}"
                    + _why_not_in_base(name, columns_by_role, datasets, base=BASE),
                    path=where,
                )
        # a prompt variable resolves per role: the role's <col>_variables
        # sibling first, then a same-named column (§2.3). Either spelling
        # counts, and unlike a metric or a column position this one stays a
        # union over roles: a variable lives in a `<field>_variables` sibling,
        # so two roles reading different fields of the *same* table legitimately
        # name different variables — which is the shape every shipped
        # multi-role document has.
        resolvable = variables | set().union(*columns_by_role.values(), set())
        for where, name in _variable_position_refs(doc):
            if not record(where, name):
                continue
            if name not in resolvable:
                raise ValidationError(
                    4,
                    f"position {where} references prompt variable {name!r}, which "
                    f"none of the resolved datasets provide — no role's "
                    f"'<field>_variables' names it and there is no {name!r} "
                    f"column (have {sorted(resolvable)})",
                    path=where,
                )
        # the table's recorded string_mode against every `match` mode the
        # document declares (§2.10's translation table); metric rows are base
        # rows, so the base table is the one held to
        base_ref = datasets.get(BASE)
        if base_ref is not None:
            _check_scoring_identity(doc, rows_of(base_ref), base_ref)
            _check_edit_groups(rows_of(base_ref), base_ref)
        # last, so that a document which *references* a counterfactual-only
        # column is told about the reference — the actionable half — rather
        # than about the schema mismatch underneath it
        _check_roles_agree_with_base(columns_by_role, datasets, base=BASE)
        check_row_roles(doc, env)
        check_fit_splits(doc, env.datasets)  # §5 rule 22, across the tables a fit names
    return refs


def maximum_eligible_count(
    metric: AggregationSpec, rows: Sequence[Mapping[str, Any]]
) -> tuple[int, dict[str, int]]:
    """The most rows of ``rows`` a metric's decision rule could be evaluated
    over (§2.10 "Eligibility"), and per column the rows that cannot be.

    A row is eligible only if every **column** the kind names carries a value
    for it — not ``null``, not absent, not an empty list of forms. That is
    the one structural fact about a row's eligibility the resolved table can
    settle without a model: a row whose answer column is empty is an excluded
    measurement at run time (``protocol/answers.py::excluded_rows``,
    the same predicate) whatever the model does, so no run can make more rows
    eligible than this. Fields that are not columns (``k``, ``by``,
    ``tokens``, ``groups``, a ``kl``/``js`` target read, ``match.mode``)
    exclude nothing here — [`metric_column_fields`][causalab.protocol.schema.types.metric_column_fields]
    is the one predicate both sides apply.
    """
    columns = list(metric_column_fields(metric).values())
    empty: dict[str, int] = {}
    eligible = 0
    for row in rows:
        missing = [
            column
            for column in columns
            if row.get(column) is None
            or (isinstance(row.get(column), list) and not row.get(column))
        ]
        if missing:
            for column in missing:
                empty[column] = empty.get(column, 0) + 1
        else:
            eligible += 1
    return eligible, empty


def check_minimum_counts(
    doc: Document, rows: Sequence[Mapping[str, Any]], ref: str
) -> None:
    """Rule 4's threshold half (§2.10 "Eligibility", §5): a metric's
    ``minimum_count`` above the resolved base table's
    [`maximum_eligible_count`][] is a decision rule the data cannot meet —
    a reference to more eligible rows than resolve — refused naming the
    maximum, the table's size and the empty columns. A threshold at exactly
    the maximum passes; a metric with none makes no claim.

    Held to the *maximum* and not to the row count on purpose: the table's
    ``n_considered`` is what every row would contribute if it could be
    scored, and a table three of whose ten answers are empty cannot make a
    threshold of ten, whatever the model does. Run-time exclusions the table
    cannot foresee (a row whose address aligns on nothing, §4.1) lower the
    cell's ``n_eligible`` further; that number is the cell's to report, not
    this pass's to predict.
    """
    for agg in doc.aggregations():
        qname, metric = agg.label, agg.spec
        threshold = metric.minimum_count
        if threshold is None:
            continue
        maximum, empty = maximum_eligible_count(metric, rows)
        if threshold <= maximum:
            continue
        why = (
            " — "
            + ", ".join(f"{n} carry no value in {col!r}" for col, n in empty.items())
            if empty
            else ""
        )
        raise ValidationError(
            4,
            f"aggregation {qname!r} declares minimum_count={threshold}, but the resolved "
            f"base table {ref!r} can make at most {maximum} of its {len(rows)} "
            f"rows eligible{why} (§2.10 'Eligibility'). A decision rule the data "
            "cannot meet is refused before any forward; declare a threshold of "
            f"at most {maximum}, or a table that carries the answers",
            path=f"{agg.owner}.aggregation.{MINIMUM_COUNT_FIELD}",
        )


def _check_scoring_identity(
    doc: Document, rows: list[dict[str, Any]], ref: str
) -> None:
    """Rule 4's scoring half: a ``match`` ``mode`` the base table's recorded
    ``string_mode`` contradicts is a reference that does not resolve
    ([`causalab.causal.scoring.check_scoring`][]; §2.2, §2.10). A malformed
    column — rows disagreeing on the mode, or a mode outside ``STRING_MODES``
    — is refused the same way: the table cannot be held to a claim it does
    not make cleanly."""
    by_label = {}
    owners: dict[str, str] = {}
    for agg in doc.aggregations():
        by_label.setdefault(agg.label, agg.spec)
        owners.setdefault(agg.label, agg.owner)
    modes = declared_modes(by_label)
    try:
        check_scoring(rows, modes, where=ref)
    except ScoringMismatch as err:
        raise ValidationError(
            4, str(err), path=f"{owners.get(err.metric, 'save')}.aggregation.mode"
        ) from err
    except ScoringError as err:
        raise ValidationError(4, f"dataset {ref!r}: {err}", path="data.base") from err


def _check_edit_groups(rows: list[dict[str, Any]], ref: str) -> None:
    """Rule 27's data half: a row's ``edit_groups`` declaration is well-formed
    ([`causalab.causal.pair_validation.parse_edit_groups`][]) — a span that is not
    inside the pair's text, sides that disagree on their constituent count or
    an ``atomic`` group of one is a span that is not well-formed, refused at
    ``validate --data`` with the row named. Rows without the column are not
    read."""
    for index, row in enumerate(rows):
        if row.get(EDIT_GROUPS_COLUMN) is None:
            continue
        try:
            parse_edit_groups(row)
        except EditGroupError as err:
            raise ValidationError(
                27,
                f"dataset {ref!r} row {index}: {err} (sec. 2.2 `edit_groups`)",
                path="data.base",
            ) from err


def check_row_roles(doc: Document, env: ResolutionEnv) -> None:
    """Rule 25 — declared row roles match the resolved data (§2.8.1).

    A ``code`` declaration that says what the rows of its batch *are* has
    made a checkable claim, and this is where it is checked: the rows a
    referenced function receives are the resolved rows of the input role of
    every intervened_model the write is in force on, so the declared roles
    have to add up to that table's length.

    The ROME replication motivated this check. Its corruption function assumed one
    clean row followed by ten corrupted ones and inferred the roles from
    physical batch positions; nothing anywhere said eleven, so a twelve-row
    batch would have run and produced numbers. Written down, the same
    assumption is arithmetic the loader can do before a model is brought up.

    Needs the resolved tables, so — like rule 4's column half and rule 20 —
    it belongs to the ``validate --data`` pass rather than the bare load. It
    is *also* called from [`run_protocol`][], before
    the engine is chosen: the column half of that pass is a wider claim whose
    blast radius on existing documents is unknown, while this one can only
    fire on a document that carries a ``code`` declaration with row roles, and
    a run that goes ahead on a batch the function does not describe is exactly
    the failure this rule exists to prevent.
    """
    if not doc.code:
        return
    users: dict[str, set[str]] = {}
    for mname, im in doc.intervened_models.items():
        for ename in im_writes(im.writes):
            write = doc.writes.get(ename)
            if write is None or str(write.do.mechanism) != "pytorch_fn":
                continue
            named = write.do.payload["code"]
            if isinstance(named, str):
                users.setdefault(named, set()).add(str(im.input))

    for name, spec in doc.code.items():
        declared = spec.declared_rows
        if declared is None:
            continue
        for role_name in sorted(users.get(name, ())):
            role = doc.data.get(_role_key(role_name))
            role = _role_member(role, role_name)
            if role is None or not isinstance(role.dataset, str):
                continue
            actual = len(env.datasets.rows(role.dataset))
            if actual != declared:
                spelled = ", ".join(f"{r.role}:{r.rows}" for r in spec.row_roles)
                raise ValidationError(
                    25,
                    f"code {name!r} declares {declared} rows ({spelled}) but the "
                    f"resolved {role_name!r} table {role.dataset!r} has {actual} — "
                    "the row convention a local function assumes is part of the "
                    "protocol, so a batch it does not describe is a load error "
                    "(§2.8.1)",
                    path=f"code.{name}.row_roles",
                )


_ROLE_INDEX = re.compile(r"^([A-Za-z_]+)\[(\d+)\]$")


def _role_key(role_name: str) -> str:
    """``counterfactual[2]`` selects a member of the ``counterfactual`` list."""
    match = _ROLE_INDEX.match(role_name)
    return match.group(1) if match else role_name


def _role_member(
    role: DataRole | tuple[DataRole, ...] | None, role_name: str
) -> DataRole | None:
    if isinstance(role, tuple):
        match = _ROLE_INDEX.match(role_name)
        index = int(match.group(2)) if match else 0
        return role[index] if index < len(role) else None
    return role


def _data_roles(doc: Document) -> list[tuple[str, DataRole]]:
    """``(path, role)`` for every data role, base first (§2.2).

    The path is what an error message and a ``ValidationError.path`` should
    say — ``data.base``, ``data.counterfactual``, ``data.counterfactual[2]`` —
    so a rejection names the role and not just the column.
    """
    out: list[tuple[str, DataRole]] = []
    for name, value in doc.data.items():
        if isinstance(value, tuple):
            out.extend(
                (f"data.{name}[{index}]", role) for index, role in enumerate(value)
            )
        else:
            out.append((f"data.{name}", value))
    return out


def _check_roles_agree_with_base(
    columns_by_role: Mapping[str, set[str]],
    datasets: Mapping[str, str],
    *,
    base: str,
) -> None:
    """Rule 20 — base is the schema of a paired row (§2.2).

    Two claims, and the second is narrower than the first for a reason:

    * every non-base role's columns are a **subset** of base's. A column only
      a counterfactual carries is a column no metric can read, because
      ``rows_for_metrics`` serves base rows; accepting it here only moves the
      failure past the model load.
    * when a role names a **different dataset** from base, the two column sets
      are **equal**. A column position is resolved against the role of the read
      it positions, not against base, so under a bare subset rule a position on
      a counterfactual read could name a base-only column and still fail at
      run time. Naming one table for both sides — every document this repo
      ships, and the shape §3 recommends — satisfies this for free.
    """
    if base not in columns_by_role:
        return
    for where, columns in columns_by_role.items():
        if where == base:
            continue
        extra = sorted(columns - columns_by_role[base])
        if extra:
            raise ValidationError(
                20,
                f"{where} carries column(s) {extra} that {base} does not — base "
                f"is the schema of a paired row (§2.2), so a column only a "
                f"counterfactual role has is one no metric can read. Move it to "
                f"{base}'s dataset ({datasets[base]!r}), or drop it.",
                path=where,
            )
        if datasets[where] == datasets[base]:
            continue  # one table for both sides: nothing left to check
        missing = sorted(columns_by_role[base] - columns)
        if missing:
            raise ValidationError(
                20,
                f"{where} names a different dataset from {base} "
                f"({datasets[where]!r} vs {datasets[base]!r}) and is missing "
                f"column(s) {missing}. Roles on different tables must carry "
                f"identical column sets: a column position resolves against the "
                f"role of the read it positions, so a base-only column would "
                f"fail at run time rather than here (§2.2).",
                path=where,
            )


def _why_not_in_base(
    name: str,
    columns_by_role: Mapping[str, set[str]],
    datasets: Mapping[str, str],
    *,
    base: str,
) -> str:
    """The tail of a rule-4 rejection: why this column is not readable.

    Naming the role that *does* carry it is the difference between a message
    that reads as a typo and one that reads as the rule it is — the reference
    resolves, just not where the run will look.
    """
    holders = sorted(
        where
        for where, columns in columns_by_role.items()
        if where != base and name in columns
    )
    if not holders:
        return (
            f", which no resolved dataset provides (base is {base} on "
            f"{datasets.get(base, '?')!r})"
        )
    named = ", ".join(f"{where} on {datasets[where]!r}" for where in holders)
    return (
        f", which {base} on {datasets.get(base, '?')!r} does not provide — it is "
        f"a column of {named}. Column references resolve against base, the "
        f"schema of a paired row (§2.2), so this would load and then fail at run "
        f"time looking for it in a base row."
    )


def _role_variables(rows: list[dict[str, Any]], field: str) -> set[str]:
    """The prompt variables one data role can name, from its rows.

    Mirrors ``causalab.protocol.positions.encoding.variable_value``, which is
    the authority at run time — torch-free and protocol-side, so
    this duplicate (predating that move) could become one import. Only the *sibling* half is here —
    the plain-column fallback is the caller's ``columns`` set.

    Union across rows, matching ``FileDatasets.columns``: a table whose rows
    disagree about their variables is a table defect, and refusing the
    *document* for it would point at the wrong thing."""
    match = _LIST_FIELD.match(field)
    column = match.group(1) if match else field
    index = int(match.group(2)) if match else None
    found: set[str] = set()
    for row in rows:
        sibling = row.get(f"{column}_variables")
        if index is not None and isinstance(sibling, list):
            sibling = sibling[index] if index < len(sibling) else None
        if isinstance(sibling, Mapping):
            found.update(str(key) for key in sibling)
    return found


def _variable_position_refs(doc: Document) -> list[tuple[str, str]]:
    """``(where, variable)`` for every prompt-variable position in a document —
    the named entries plus the inline specs on reads and writes, and the
    ``scope``/``relative_to`` anchors spelled as a variable (§2.3)."""
    found: list[tuple[str, str]] = []

    def visit(where: str, spec: Any) -> None:
        if not isinstance(spec, PositionSpec):
            return
        for inner in walk(spec):  # a span's members and anchors too (§2.3)
            if isinstance(inner.variable, str):
                found.append((where, inner.variable))
            anchor = inner.scope if inner.scope is not None else inner.relative_to
            if inner.anchor_source == "variable" and isinstance(anchor, str):
                found.append((where, anchor))

    for name, entry in doc.positions.items():
        visit(f"positions.{name}", entry)
    for section, table in (("reads", doc.reads), ("writes", doc.writes)):
        for name, spec in table.items():
            visit(f"{section}.{name}.pos", spec.pos)
    return found


def _column_position_refs(doc: Document) -> list[tuple[str, str]]:
    """``(where, column)`` for every ``column`` position in a document — the
    named entries plus the inline specs on reads and writes (§2.3)."""
    found: list[tuple[str, str]] = []
    if doc.segments is not None:
        # a segment located from a column needs the column (§2.2.1)
        if doc.segments.system is not None:
            found.append(("segments.system", doc.segments.system.column))
        for name, source in doc.segments.declare.items():
            found.append((f"segments.declare.{name}", source.column))

    def visit(where: str, spec: Any) -> None:
        if not isinstance(spec, PositionSpec):
            return
        for inner in walk(spec):  # a span's members and anchors too (§2.3)
            if isinstance(inner.column, str):
                found.append((where, inner.column))
            anchor = inner.scope if inner.scope is not None else inner.relative_to
            if inner.anchor_source == "column" and isinstance(anchor, str):
                found.append((where, anchor))

    for name, entry in doc.positions.items():
        visit(f"positions.{name}", entry)
    for section, table in (("reads", doc.reads), ("writes", doc.writes)):
        for name, spec in table.items():
            visit(f"{section}.{name}.pos", spec.pos)
    return found


# --------------------------------------------------------------------------- #
# rule 22, the fourth refusal — a fit's splits are endpoint-disjoint across tables
# --------------------------------------------------------------------------- #

#: The path of the held-out role — the one ``train`` field that names a
#: dataset ref. There is no third role on a ``train`` block today
#: (``checkpoint.file_path`` is an artifact path, not a table); one that
#: arrives joins [`fit_roles`][] and is checked by the same loop.
HELD_OUT_ROLE = "train.eval.split"


def fit_roles(doc: Document) -> tuple[list[tuple[str, str]], str | None]:
    """The refs a fit consumes, by role: ``(path, ref)`` for every training
    role (``data.base``, ``data.counterfactual``, ``data.counterfactual[j]``,
    the spelling [`check_data_columns`][] reports under) and the held-out
    ref, or ``None`` when the fit declares no ``eval``.

    A value that is still a sweep wrapper is skipped: the check runs on
    expanded points, where every ref is one string.
    """
    training: list[tuple[str, str]] = []
    for name, value in doc.data.items():
        roles = value if isinstance(value, tuple) else (value,)
        for index, role in enumerate(roles):
            path = (
                f"data.{name}[{index}]" if isinstance(value, tuple) else f"data.{name}"
            )
            if isinstance(role.dataset, str):
                training.append((path, role.dataset))
    held_out: str | None = None
    if doc.train is not None and doc.train.eval is not None:
        split = doc.train.eval.get("split")
        if isinstance(split, str):
            held_out = split
    return training, held_out


def check_fit_splits(doc: Document, datasets: DatasetResolver) -> None:
    """Refuse a fit whose training rows and held-out rows share a prompt at
    either endpoint, unless both roles name the very same ref [V22].

    An endpoint is what [`endpoints`][] says: the
    row's ``input`` and every ``counterfactual_inputs`` string — the prompts a
    row puts in front of the model, on both sides of the pair. The leak that
    matters is a training base reappearing as a held-out counterfactual, which
    reports a training score under a held-out name.

    A document with no ``train`` block, or a fit with no ``eval``, has one role
    and nothing to compare; it returns without touching the resolver.
    """
    if doc.train is None:
        return
    training, held_out = fit_roles(doc)
    if held_out is None:
        return
    rows_by_ref: dict[str, list[dict[str, Any]]] = {}

    def rows(ref: str) -> list[dict[str, Any]]:
        if ref not in rows_by_ref:
            rows_by_ref[ref] = datasets.rows(ref)
        return rows_by_ref[ref]

    held_out_endpoints: set[str] | None = None
    for path, ref in training:
        if ref == held_out:
            continue  # one ref, twice: the visible train-equals-test ablation
        if held_out_endpoints is None:
            held_out_endpoints = set()
            for row in rows(held_out):
                held_out_endpoints |= endpoints(row)
        for row in rows(ref):
            shared = sorted(endpoints(row) & held_out_endpoints)
            if shared:
                base, _ = split_dataset_ref(ref)
                raise ValidationError(
                    22,
                    f"fit splits leak across tables: the prompt {shared[0]!r} "
                    f"appears in both {ref!r} (the training rows, {path}) and "
                    f"{held_out!r} (the held-out rows, {HELD_OUT_ROLE}). A fit's "
                    f"splits must be endpoint-disjoint across the tables it names "
                    f"(§2.2) — select two splits of one table ({base}#train and "
                    f"{base}#test, built by causalab.tasks.splits.generate_split_dataset), "
                    f"or "
                    f"name the same ref for both roles to spell a deliberate "
                    f"train-equals-test ablation where it is visible",
                    path=HELD_OUT_ROLE,
                )
