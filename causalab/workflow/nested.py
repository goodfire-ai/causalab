"""Load reusable workflows and mount their steps in the parent schedule.

Inner names are prefixed with the containing step. References resolve within
the corresponding run subtree. Validation checks recursive inclusion, supported
control declarations, and conditional dependencies across nesting boundaries."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from causalab.protocol.rules.errors import ProtocolError, suggest
from causalab.io.sources import load_text
from causalab.io.env import ResolutionEnv
from causalab.workflow.document import (
    ConditionalStep,
    DecisionStep,
    LoadedWorkflow,
    ProtocolStep,
    Reference,
    ScriptStep,
    Step,
    WorkflowError,
    WorkflowStep,
    is_workflow,
    load_workflow,
)

__all__ = [
    "NESTED_RULE",
    "SEPARATOR",
    "check_nested",
    "check_step_set",
    "containers",
    "describe",
    "expand_skips",
    "flatten_edges",
    "lay_over_set",
    "load_nested",
    "local_name",
    "local_skips",
    "locate",
    "manifest_block",
    "members",
    "mount",
    "parse_step_set",
    "parse_workflow_step",
    "prefix_skips",
    "qualified",
    "rebase",
    "sub_roots",
]

#: The checklist rule this layer refuses under (§5).
NESTED_RULE = 20

#: Between an outer step's name and an inner step's: outside rule 3's
#: alphabet, so a flattened name never collides with an authored one, and the
#: run-tree path separator, so ``<step>/<inner>/<file>`` is one path.
SEPARATOR = "/"

#: The inner kinds a ``workflow`` step's ``set`` may name (§2.10): the two
#: that carry a ``set`` of their own. A nested ``workflow`` step is not among
#: them — ``set`` reaches one level in this version.
_DOCUMENT_KINDS = ("intervention_protocol", "behavioral")


def _refuse(message: str, path: str) -> WorkflowError:
    return WorkflowError(NESTED_RULE, message, path=path)


# --------------------------------------------------------------------------- #
# names
# --------------------------------------------------------------------------- #


def qualified(outer: str, inner: str) -> str:
    """``<outer>/<inner>`` — or ``inner`` alone under the run root."""
    return f"{outer}{SEPARATOR}{inner}" if outer else inner


def local_name(rel: str, name: str) -> str | None:
    """The name ``name`` has inside the sub-root ``rel`` (``tail/fit@0`` under
    ``tail`` is ``fit@0``), or ``None`` when it is not under that root."""
    if not rel:
        return name
    prefix = rel + SEPARATOR
    return name[len(prefix) :] if name.startswith(prefix) else None


def prefix_skips(rel: str, names: Iterable[Any]) -> list[str]:
    """An inner conditional's ``skipped`` names, spelled under its sub-root."""
    return [qualified(rel, str(name)) for name in names]


def expand_skips(steps: Mapping[str, Step], names: Iterable[str]) -> list[str]:
    """A skipped set with every ``workflow`` step replaced by its members —
    ``on_false: ["tail"]`` skips every ``tail/*`` (children included, §2.9),
    each its own ``skipped`` entry (§8)."""
    out: set[str] = set()
    for name in names:
        if isinstance(steps.get(name), WorkflowStep):
            out |= members(steps, name)
        else:
            out.add(name)
    return sorted(out)


def members(steps: Mapping[str, Step], container: str) -> set[str]:
    """The steps a ``workflow`` step contributes to the run — every flattened
    name under it that is work, never a deeper container."""
    prefix = container + SEPARATOR
    return {
        name
        for name, step in steps.items()
        if name.startswith(prefix) and not isinstance(step, WorkflowStep)
    }


def local_skips(
    rel: str, skipped_by: Mapping[str, Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    """The run's ``skipped_by`` map as an inner workflow sees it: keyed by
    inner-local names (a ``selected`` join inside the inner reads it, §2.9)."""
    out: dict[str, Mapping[str, Any]] = {}
    for name, by in skipped_by.items():
        local = local_name(rel, name)
        if local is not None:
            out[local] = by
    return out


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def parse_workflow_step(
    raw: Mapping[str, Any],
    path: str,
    *,
    document: str,
    requires_receipt: Mapping[str, str] | None,
    after: tuple[str, ...],
    description: str | None,
) -> WorkflowStep:
    """A ``workflow`` step (§2.10): ``document`` a workflow document relative
    to the workflow file, ``set`` the nested form — inner step name → that
    step's own ``set`` map — each value a map of dotted paths to literals."""
    return WorkflowStep(
        type="workflow",
        document=document,
        set=parse_step_set(raw.get("set", {}), f"{path}.set"),
        requires_receipt=requires_receipt,
        after=after,
        description=description,
    )


def parse_step_set(raw: Any, path: str) -> dict[str, dict[str, Any]]:
    """The nested ``set`` form, shape-checked (its names are held to the inner
    document by [`check_step_set`][] once that document is read)."""
    shape = (
        "'set' on a workflow step maps inner step names to that step's own "
        '\'set\' — {"<inner step>": {"<dotted path>": value}} (§2.10)'
    )
    if not isinstance(raw, Mapping):
        raise _refuse(shape, path)
    out: dict[str, dict[str, Any]] = {}
    for inner, overrides in raw.items():
        if not isinstance(inner, str) or not inner:
            raise _refuse(f"{shape}; got the key {inner!r}", path)
        where = f"{path}.{inner}"
        if not isinstance(overrides, Mapping):
            raise _refuse(
                f"'set.{inner}' is {inner!r}'s own 'set' — a map of dotted paths "
                f"to values — got {type(overrides).__name__}",
                where,
            )
        for key in overrides:
            if not isinstance(key, str) or not key:
                raise _refuse(
                    f"'set.{inner}' maps dotted paths to values; got the key {key!r}",
                    where,
                )
        out[inner] = dict(overrides)
    return out


def check_step_set(
    inner_raw: Mapping[str, Any],
    step_set: Mapping[str, Mapping[str, Any]],
    *,
    document: str | None,
    path: str,
) -> None:
    """Rule 20: every inner name a ``workflow`` step's ``set`` names is a step
    of the inner document, and one with a ``set`` of its own to lay over — a
    document step (`_DOCUMENT_KINDS`); a script, a decision, a
    conditional or a nested ``workflow`` step is refused naming the field."""
    steps_raw = inner_raw.get("steps")
    if not isinstance(steps_raw, Mapping):
        return  # the inner parse refuses the document's shape (rule 1)
    which = f" of {document!r}" if document is not None else ""
    for inner in step_set:
        where = f"{path}.{inner}"
        target = steps_raw.get(inner)
        if target is None:
            raise _refuse(
                f"'set' names {inner!r}, which is not a step{which}"
                f"{suggest(inner, sorted(str(n) for n in steps_raw))}",
                where,
            )
        kind = target.get("type") if isinstance(target, Mapping) else None
        if kind not in _DOCUMENT_KINDS:
            reach = (
                " — 'set' reaches one level: a nested workflow step is "
                "parametrised from its own document in this version"
                if kind == "workflow"
                else ""
            )
            raise _refuse(
                f"'set' names {inner!r}, a {kind!r} step{which} — only a document "
                f"step ({' · '.join(_DOCUMENT_KINDS)}) has a 'set' to lay over"
                f"{reach}",
                where,
            )


def lay_over_set(
    raw: Mapping[str, Any], step_set: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """The inner document's raw tree with the outer's ``set`` laid over the
    named steps' own ``set`` (outer wins, key by key) — before the inner
    parse, so rule 8 checks every dotted key against the intervention
    specification exactly as it checks an authored one."""
    check_step_set(raw, step_set, document=None, path="set")
    if not isinstance(raw.get("steps"), Mapping):
        # the inner parse refuses the document's shape (rule 1): this runs
        # before it, so the raw tree passes through untouched and the outer
        # wraps that refusal as its own — never a KeyError out of `dict([])`
        return dict(raw)
    steps_raw = dict(raw["steps"])
    for inner, overrides in step_set.items():
        target = dict(steps_raw[inner])
        authored = target.get("set")
        merged: dict[str, Any] = dict(authored) if isinstance(authored, Mapping) else {}
        merged.update(overrides)
        target["set"] = merged
        steps_raw[inner] = target
    return {**raw, "steps": steps_raw}


# --------------------------------------------------------------------------- #
# load time: the inner load, the mount, the flattened edges, rule 20
# --------------------------------------------------------------------------- #


def _labels(paths: Sequence[Path]) -> list[str]:
    names = [path.name for path in paths]
    if len(set(names)) < len(set(paths)):
        return [str(path) for path in paths]  # two files share a name: spell them
    return names


def load_nested(
    name: str,
    step: WorkflowStep,
    workflow_dir: Path,
    env: ResolutionEnv,
    chain: Sequence[Path],
) -> LoadedWorkflow:
    """Load the document a ``workflow`` step names, under this checklist in
    full (rule 20, §2.10). ``chain`` is every document being loaded above
    this one, outermost first: a document already in it is refused naming
    the chain — the one cap on depth (finite files, finite depth)."""
    where = f"steps.{name}"
    doc_path = (workflow_dir / step.document).resolve()
    if not doc_path.is_file():
        raise WorkflowError(4, f"document {step.document!r} not found", path=where)
    if doc_path in chain:
        start = list(chain).index(doc_path)
        trail = " -> ".join(_labels((*chain[start:], doc_path)))
        raise _refuse(
            f"workflow {step.document!r} includes itself ({trail}) — a nested "
            "workflow is a finite tree of documents, never a loop",
            f"{where}.document",
        )
    try:
        raw: dict[str, Any] = dict(load_text(doc_path))
    except ProtocolError as err:
        raise _refuse(
            f"workflow {step.document!r} does not load: {err}", f"{where}.document"
        ) from err
    if not is_workflow(raw):
        raise _refuse(
            f"document {step.document!r} is not a workflow document — it has no "
            "'steps' section (§1); a 'workflow' step nests a workflow document, "
            "and an intervention specification runs under an "
            "'intervention_protocol' step",
            f"{where}.document",
        )
    check_step_set(raw, step.set, document=step.document, path=f"{where}.set")
    try:
        return load_workflow(
            raw,
            env,
            workflow_dir=doc_path.parent,
            _including=(*chain, doc_path),
            _step_set=step.set,
        )
    except WorkflowError as err:
        raise _refuse(
            f"workflow {step.document!r} does not load: {err}", where
        ) from err


def rebase(step: Step, prefix: str) -> Step:
    """The step as the outer table sees it: every step name it holds —
    ``after``, ``requires_receipt.step``, an input's ``step``, a decision's
    ``values.step``, a conditional's producer and sides — spelled under
    ``prefix``, so the flattened table is one consistent graph. The inner's
    own table keeps the unprefixed step; that is the one the runner executes
    under the sub-root."""

    def q(inner: Any) -> str:
        return qualified(prefix, str(inner))

    changes: dict[str, Any] = {"after": tuple(q(other) for other in step.after)}
    if step.requires_receipt is not None:
        changes["requires_receipt"] = {
            **step.requires_receipt,
            "step": q(step.requires_receipt["step"]),
        }
    if isinstance(step, ScriptStep):
        changes["inputs"] = {
            slot: (
                dataclasses.replace(value, step=q(value.step))
                if isinstance(value, Reference) and value.step is not None
                else value
            )
            for slot, value in step.inputs.items()
        }
    elif isinstance(step, DecisionStep):
        changes["values"] = dataclasses.replace(step.values, step=q(step.values.step))
    elif isinstance(step, ConditionalStep):
        decision = dict(step.predicate.get("decision", {}))
        decision["step"] = q(decision.get("step"))
        changes["predicate"] = {**step.predicate, "decision": decision}
        changes["on_true"] = tuple(q(gated) for gated in step.on_true)
        changes["on_false"] = tuple(q(gated) for gated in step.on_false)
    return dataclasses.replace(step, **changes)


def mount(
    name: str,
    nested: LoadedWorkflow,
    *,
    steps: dict[str, Step],
    deps: dict[str, set[str]],
    inner: dict[str, Any],
    inner_digests: dict[str, str],
    inner_digest_kind: dict[str, str],
    step_digests: dict[str, str],
    children: dict[str, tuple[str, ...]],
    unchecked_paths: list[str],
    mounted: dict[str, str],
) -> None:
    """Add a loaded inner workflow's steps to the outer tables under
    ``<name>/<inner>`` (§2.10): each step (rebased), its inner edges, its
    compiled document and digests where it has them, its identity, a
    fan-out's children (§2.9), and the absolute paths load could not check.
    ``mounted`` records ``{flattened name: the workflow step it came in
    through}``. The container's own edges — ``after``, a receipt, a
    conditional's gate — are derived by the outer's loop and folded onto the
    inner roots by [`flatten_edges`][]."""
    for local, step in nested.document.steps.items():
        flat = qualified(name, local)
        steps[flat] = rebase(step, name)
        mounted[flat] = name
        deps[flat] = {
            qualified(name, upstream) for upstream in nested.dependencies.get(local, ())
        }
        if local in nested.inner:
            inner[flat] = nested.inner[local]
        if local in nested.inner_digests:
            inner_digests[flat] = nested.inner_digests[local]
            inner_digest_kind[flat] = nested.inner_digest_kind[local]
        if local in nested.step_digests:
            step_digests[flat] = nested.step_digests[local]
    for parent, kids in nested.children.items():
        children[qualified(name, parent)] = tuple(qualified(name, k) for k in kids)
    unchecked_paths.extend(
        f"{name}{SEPARATOR}{item}" for item in nested.unchecked_paths
    )


def flatten_edges(
    steps: Mapping[str, Step], deps: dict[str, set[str]]
) -> dict[str, set[str]]:
    """Take every ``workflow`` step out of the edge map (§2.10, §6): an edge
    **to** a container becomes edges to each of its members (``after:
    ["tail"]`` waits for every ``tail/*``; a skip among them propagates as
    over any ``after`` edge); the container's own edges — its ``after``, its
    receipt's producer, the conditional that gates it — are inherited by the
    members with no upstream inside it, so the whole nested workflow runs
    after them. Containers leave ``deps``; the schedule (rule 5) is over
    work alone. Returns each container's outer edges, for the record."""
    containers = [n for n, s in steps.items() if isinstance(s, WorkflowStep)]
    of = {c: members(steps, c) for c in containers}

    def expand(edges: Iterable[str]) -> set[str]:
        out: set[str] = set()
        for upstream in edges:
            out |= of[upstream] if upstream in of else {upstream}
        return out

    outer = {c: expand(deps.get(c, set())) for c in containers}
    roots = {c: {m for m in of[c] if not (deps[m] & of[c])} for c in containers}
    for c in containers:
        for root in roots[c]:
            deps[root] |= outer[c]
    for name in list(deps):
        if name in of:
            continue
        deps[name] = expand(deps[name])
    for c in containers:
        deps.pop(c, None)
    return outer


def check_nested(
    steps: Mapping[str, Step],
    nested: Mapping[str, LoadedWorkflow],
    mounted: Mapping[str, str],
) -> None:
    """Rule 20 over the flattened table (§2.10, §5), before rules 14 and 18
    would call the same shapes by another name: a ``workflow`` step publishes
    no file and writes no receipt, so a ``requires_receipt`` or a
    ``predicate`` naming it is refused (name ``<step>/<inner>``; a ``values``
    or a ``{"step": …}`` reference naming it was refused by rule 4's
    ``outputs_of`` already, at the same path and under this rule); a
    per-child conditional (§2.9) names no nested step as its producer or on
    a side; no step declares a ``workflow`` step, or one of its inner steps,
    as a control; and a nested workflow declares no ``control`` and no
    ``waive`` — in this version, because the controls ledger is keyed by
    step name at one root. The fail-closed consequence: an outer that
    engages rule 14 (some own step authoring ``control`` or ``waive``, the
    predicate ``_check_controls`` gates its coverage clause on) commits to a
    control or a waiver for every fit it runs — and a fit inside a nested
    document can carry neither, while the outer's declarations cannot reach
    it, so such a nesting is refused here naming the inner fit rather than
    loading a fit the outer's controls do not hold (§2.10). A fit is what
    rule 14 calls one: a protocol step whose loaded document has a
    ``train`` section."""

    def container(target: Any) -> bool:
        return isinstance(steps.get(str(target)), WorkflowStep)

    def publishes_nothing(target: Any, field: str, path: str) -> None:
        if container(target):
            raise _refuse(
                f"{field!r} names {str(target)!r}, a workflow step — a nested "
                "workflow publishes no file and writes no receipt of its own; "
                f"name the inner step that does, '{target}/<step>' (§2.10)",
                path,
            )

    for name, step in steps.items():
        if name in mounted:
            continue  # its own load checked it, against its own table
        where = f"steps.{name}"
        if step.requires_receipt is not None:
            publishes_nothing(
                step.requires_receipt["step"],
                "requires_receipt",
                f"{where}.requires_receipt.step",
            )
        # a decision's `values` and a script input's `{"step": …}` naming a
        # `workflow` step are not checked here: the rule-4 loops run first and
        # their `outputs_of` refuses a `WorkflowStep` target at the same path,
        # under this rule
        if isinstance(step, ConditionalStep):
            producer = step.predicate.get("decision", {}).get("step")
            publishes_nothing(producer, "predicate", f"{where}.predicate.decision.step")
            if step.scope != "global":
                # a per-child scope (§2.9) follows one fan-out, and the outer's
                # expansion is over its own steps: across the boundary the
                # conditional would load and never expand, so it is refused
                # rather than stranded — the fan-out it follows and the
                # conditional sit in one document
                named = (
                    ("predicate.decision.step", producer),
                    *(("on_true", gated) for gated in step.on_true),
                    *(("on_false", gated) for gated in step.on_false),
                )
                for field, target in named:
                    if str(target) not in mounted:
                        continue
                    # `mounted` names the container the step came in through
                    # at this level; the advice names the INNERMOST one — the
                    # deepest `workflow` step whose sub-root the target is
                    # under, whose document is the one that declares the
                    # fan-out (`a/b`, `per_child.json`; never `a`, `mid.json`)
                    # never empty: `str(target) in mounted` puts the target
                    # under its top-level container `mounted[target]`, the
                    # outermost candidate — the default states the invariant
                    through = max(
                        (
                            holder_name
                            for holder_name, candidate in steps.items()
                            if isinstance(candidate, WorkflowStep)
                            and str(target).startswith(holder_name + SEPARATOR)
                        ),
                        key=len,
                        default=mounted[str(target)],
                    )
                    holder = steps[through]
                    assert isinstance(holder, WorkflowStep)
                    raise _refuse(
                        f"scope {step.scope!r} on {name!r} names {str(target)!r} in "
                        f"{field!r}, a step of the nested workflow {through!r} — a "
                        "per-child conditional and the fan-out it follows sit in "
                        "one workflow document in this version: declare the "
                        f"per-child conditional inside {through!r}'s own document, "
                        f"{holder.document!r} (§2.10)",
                        f"{where}.scope",
                    )
        elif isinstance(step, ProtocolStep) and step.control is not None:
            of = str(step.control.get("of"))
            if container(of) or of in mounted:
                what = (
                    "a workflow step" if container(of) else "a nested workflow's step"
                )
                raise _refuse(
                    f"'control.of' names {of!r}, {what} — the controls ledger is "
                    "keyed by step name at one root, so a control and its target "
                    "sit in one workflow document in this version (§2.10)",
                    f"{where}.control.of",
                )
    # the outer's own protocol steps — the table `_check_controls` runs over
    # (§2.10) — and its engagement, by the same predicate
    engaged = any(
        isinstance(step, ProtocolStep)
        and (step.control is not None or step.waive is not None)
        for name, step in steps.items()
        if name not in mounted
    )
    for name, workflow in nested.items():
        step = steps[name]
        assert isinstance(step, WorkflowStep)
        # `workflow.document.steps` is the inner's own flattened table, so a
        # fit two levels down is visited here under `name` as well
        for local, inner_step in workflow.document.steps.items():
            if not isinstance(inner_step, ProtocolStep):
                continue
            declared = (
                "control"
                if inner_step.control is not None
                else "waive"
                if inner_step.waive is not None
                else None
            )
            if declared is not None:
                raise _refuse(
                    f"workflow {step.document!r} declares {declared!r} on its step "
                    f"{local!r} — a nested workflow declares no control and no "
                    "waive in this version: the controls ledger is keyed by step "
                    "name at one root (§2.10)",
                    f"steps.{name}.document",
                )
            if engaged and workflow.inner[local].compiled.document.train is not None:
                raise _refuse(
                    f"workflow {step.document!r} contains the fit "
                    f"{qualified(name, local)!r}, and this document engages the "
                    "controls layer (a step declares 'control' or 'waive') — a "
                    "nested fit is not held by the outer's controls in this "
                    "version: the inner document declares none (rule 20) and "
                    "the ledger is keyed by step name at one root, so the fit "
                    "would load with no control and no waiver where rule 14 "
                    "requires one; declare the control in the outer's own "
                    "steps, or waive it there once the ledger keys by flattened "
                    "name (§2.2, §2.10)",
                    f"steps.{name}.document",
                )


# --------------------------------------------------------------------------- #
# run time: where a flattened step lives
# --------------------------------------------------------------------------- #


def _walk(
    loaded: LoadedWorkflow, name: str
) -> Iterator[tuple[LoadedWorkflow, str, str]]:
    """Each ``workflow`` step a flattened name descends through, outermost
    first, as ``(its owner, its local name there, its owner's sub-root)`` —
    the one walk [`locate`][] and [`containers`][] read, so the two can
    never disagree about where a step lives."""
    owner, rel, rest = loaded, "", name
    while SEPARATOR in rest:
        head, _, tail = rest.partition(SEPARATOR)
        if head not in owner.nested:
            return
        yield owner, head, rel
        owner, rel, rest = owner.nested[head], qualified(rel, head), tail


def locate(loaded: LoadedWorkflow, name: str) -> tuple[LoadedWorkflow, str, str]:
    """``(owner, local, rel)`` for a flattened step name: the loaded workflow
    whose table holds the step, its name there, and the sub-root it executes
    under (``tail/fit@0`` → ``(loaded.nested["tail"], "fit@0", "tail")``; an
    outer step is ``(loaded, name, "")``)."""
    owner, rel = loaded, ""
    for holder, head, at in _walk(loaded, name):
        owner, rel = holder.nested[head], qualified(at, head)
    return owner, local_name(rel, name) or name, rel


def containers(
    loaded: LoadedWorkflow, name: str
) -> tuple[tuple[LoadedWorkflow, str, str], ...]:
    """Every ``workflow`` step a flattened name sits inside, outermost first,
    each as ``(its owner, its local name, its owner's sub-root)`` — the
    receipts the runner checks before an inner step is allocated (§2.10)."""
    return tuple(_walk(loaded, name))


def sub_roots(loaded: LoadedWorkflow, rel: str = "") -> list[str]:
    """Every nested sub-root of a run, deepest first — the directories a clean
    run prunes when nothing landed in them."""
    out: list[str] = []
    for name, workflow in loaded.nested.items():
        here = qualified(rel, name)
        out.extend(sub_roots(workflow, here))
        out.append(here)
    return out


def describe(name: str, step: WorkflowStep, nested: LoadedWorkflow) -> str:
    """One ``explain`` line: ``tail: workflow tail.json — 3 step(s)``; the
    inner lines follow, indented. The nesting identity the outer entry carries
    is not printed: it is the inner steps' identities folded, and those are
    what ``--resume`` compares (§7)."""
    return f"{name}: workflow {step.document} — {len(nested.order)} step(s)"


def manifest_block(loaded: LoadedWorkflow, rel: str = "") -> dict[str, dict[str, Any]]:
    """The manifest's ``nested`` map (§8, record-only, never canonical and
    never compared by ``--resume``): per ``workflow`` step, its document, the
    inner digest the outer entry carries, and the flattened steps it
    contributed."""
    out: dict[str, dict[str, Any]] = {}
    for name, workflow in loaded.nested.items():
        here = qualified(rel, name)
        step = loaded.document.steps[name]
        assert isinstance(step, WorkflowStep)
        out[here] = {
            "document": step.document,
            "workflow_digest": workflow.digest,
            "steps": [qualified(here, inner) for inner in workflow.order],
        }
        out.update(manifest_block(workflow, here))
    return out
