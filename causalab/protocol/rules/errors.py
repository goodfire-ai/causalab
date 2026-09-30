"""Define structured errors and rule codes.

``ParseError`` identifies malformed documents with a P code.
``ValidationError`` identifies a failed validation rule with a V code.
Reason codes describe the specific availability or alignment failure."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Literal, Mapping, Sequence, get_args

__all__ = [
    "REASON_CODES",
    "RULES",
    "RULES_BY_NUMBER",
    "ParseError",
    "ProtocolError",
    "ProtocolWarning",
    "ReasonCode",
    "Rule",
    "ValidationError",
    "ValidationErrors",
    "lookup_rule",
    "raise_distinct",
    "rule_registry",
    "suggest",
]


@dataclass(frozen=True)
class Rule:
    """One item of the §5 load-error checklist.

    ``slug`` is the rule's identity — the name code and spec share, never
    reused for a different rule. ``number`` is the label the spec prints and
    the code renders as ``V<number>``: it is frozen at first landing, so a
    later rule takes the next unused number and no rule is ever renumbered.
    Gaps are legal; a retired rule leaves its number retired. ``title`` is
    the item's opening clause, as the spec words it.
    """

    slug: str
    number: int
    title: str
    #: The featurizer kinds the rule is *about* — ``gate`` for group legality
    #: and a gate's score-table start — so a method page can list what a
    #: document of its family is refused for. Empty for a rule that concerns
    #: every document alike (the census holds every name here to the schema's
    #: ``FEATURIZER_KINDS``; ``errors`` sits below ``schema`` and cannot import
    #: it).
    kinds: frozenset[str] = frozenset()

    @property
    def code(self) -> str:
        """The user-visible code, ``V<number>``."""
        return f"V{self.number}"


def rule_registry(rules: Iterable[Rule]) -> dict[str, Rule]:
    """The slug-keyed registry of ``rules``, refusing any collision.

    Two rules sharing a slug or a number is the merge that this registry
    exists to catch — two PRs each appending "the next" rule — so it is
    refused at import, not at the first ``ValidationError`` that happens to
    hit one of them.
    """
    out: dict[str, Rule] = {}
    by_number: dict[int, str] = {}
    for rule in rules:
        if rule.slug in out:
            raise ValueError(f"checklist rule slug {rule.slug!r} is declared twice")
        if rule.number in by_number:
            raise ValueError(
                f"checklist rule number {rule.number} is claimed by both "
                f"{by_number[rule.number]!r} and {rule.slug!r}; a new rule takes "
                f"the next unused number"
            )
        out[rule.slug] = rule
        by_number[rule.number] = rule.slug
    return out


#: The §5 checklist, in spec order. Adding a rule means appending one entry
#: here with the next unused number and one item to §5 that opens with the
#: same slug and number; the census in ``tests/protocol/test_validation_rules
#: .py`` holds the two identical. Two PRs that both take the next number
#: conflict on this one literal and one spec line, and the one that lands
#: second takes the number after — no raise site and no test moves, because a
#: rule is named by its slug, not by its place in the list.
#:
#: Six of the rules are not plain load rejections, and each says so in §5:
#: ``section_order`` (2) warns (order carries no meaning, see
#: [`ProtocolWarning`][]) and ``write_widths_uniform`` (19) is checked
#: when the run encodes its inputs. ``base_is_paired_schema`` (20) needs the
#: resolved tables, so it is part of the ``validate --data`` pass.
#: ``split_declaration`` (22) is raised by the *resolver*, not by a load
#: pass: it is a property of one table, so it belongs where a ref becomes
#: rows and every verb — ``run`` included — has to go through it.
#: ``group_legality`` (23) is decided as the document canonicalizes, from
#: static model config, before any weight loads.
#: ``row_roles`` (25) needs the resolved tables too, so it joins
#: ``base_is_paired_schema`` in the ``validate --data`` pass.
#: ``scores_init`` (32) is decided in two halves: ``keep`` against the unit
#: count as the document canonicalizes (the width is derived there, as for
#: rule 23), the table's coverage of the units at build, where the table is
#: read.
#:
#: Read-only: the only way a rule enters this table is the literal below, so
#: a runtime insertion cannot mint a code the spec does not list.
RULES: Mapping[str, Rule] = MappingProxyType(
    rule_registry(
        [
            Rule("strict_keys", 1, "Strict keys"),
            Rule("section_order", 2, "Sections in the sec. 1 order"),
            Rule("global_namespace", 3, "Global namespace"),
            Rule("references_resolve", 4, "Every reference resolves"),
            Rule("read_bindings", 5, "Read references resolve to one model"),
            Rule("write_operands", 6, "Writes carry no `model`/`input`/conditions"),
            Rule(
                "write_membership",
                7,
                "Every write and every read is in ≥ 1 intervened_model; the model graph is acyclic",
            ),
            Rule(
                "one_absolute_write",
                8,
                "Per (site, overlapping pos, model): ≤ 1 absolute write",
            ),
            Rule(
                "dims_disjoint",
                9,
                "`dims` selections co-occurring at one address in one model are disjoint",
            ),
            Rule(
                "save_manifest",
                10,
                "`save` non-empty; entry shapes exact; one entry per value",
            ),
            Rule(
                "sink_rule", 11, "Sink rule: every read on every model has a consumer"
            ),
            Rule("featurizer_legality", 12, "Featurizer legality"),
            Rule(
                "pytorch_fn_local",
                13,
                "`pytorch_fn` present ⇒ refused unless the selected engine is local",
            ),
            Rule("sweep_wrappers", 14, "Sweep wrappers well-formed"),
            Rule("artifact_fields_resolve", 15, "Artifact-valued fields resolve"),
            Rule("generation_read_only", 16, "Generation is read-only"),
            Rule("realization_coherent", 17, "The model's realization is coherent"),
            Rule("shape", 18, "Shape (§1)"),
            Rule("write_widths_uniform", 19, "Write widths are uniform"),
            Rule("base_is_paired_schema", 20, "`base` is the schema of a paired row"),
            Rule("operand_reachability", 21, "Operand reachability"),
            Rule("split_declaration", 22, "Split declaration (sec. 2.2)"),
            Rule("group_legality", 23, "Group legality", kinds=frozenset({"gate"})),
            Rule(
                "code_declaration",
                24,
                "A `code` declaration agrees with the source it names (sec. 2.8.1)",
            ),
            Rule(
                "row_roles",
                25,
                "Declared row roles match the resolved data (sec. 2.8.1)",
            ),
            Rule(
                "alignment_declared",
                26,
                "A declared `alignment` fits its address (sec. 2.3)",
            ),
            Rule(
                "segment_declared",
                27,
                "A `segment` anchor names a declared segment and a span is "
                "well-formed (sec. 2.2.1, sec. 2.3)",
            ),
            Rule("family_wrappers", 28, "`at_once` families well-formed"),
            Rule(
                "kl_operands_compatible",
                29,
                "`kl` operands are comparable (sec. 2.10)",
            ),
            Rule(
                "train_engine_supported",
                30,
                "The routed engine can execute the fit as authored (sec. 2.11)",
            ),
            Rule(
                "metric_position_scalar",
                31,
                "A metric reduces one position per example (sec. 2.10)",
            ),
            Rule(
                "scores_init",
                32,
                "A gate's `init.from_scores` fits the gate",
                kinds=frozenset({"gate"}),
            ),
        ]
    )
)

#: The same rules by their frozen number — the form the raise sites use.
RULES_BY_NUMBER: Mapping[int, Rule] = MappingProxyType(
    {rule.number: rule for rule in RULES.values()}
)


def lookup_rule(rule: int | str) -> Rule:
    """The [`Rule`][] a number or slug names; unknown ones are refused.

    ``bool`` is an ``int`` to Python but names no rule, so ``True`` is not
    rule 1.
    """
    if isinstance(rule, bool):
        found = None
    elif isinstance(rule, int):
        found = RULES_BY_NUMBER.get(rule)
    else:
        found = RULES.get(rule)
    if found is None:
        raise AssertionError(f"unknown checklist rule: {rule!r}")
    return found


#: Why a refusal about a component, a mechanism or a selector fired — a
#: closed vocabulary, carried by
#: [`ProtocolError.reason`][] next to the rule code. A rule number says
#: *which check* refused (V4: a reference did not resolve); the reason says
#: *what kind of fact* was missing, in a form a caller can branch on without
#: parsing prose. Spec §2.4 tabulates the eight and which check emits each;
#: ``tests/protocol/test_vocabulary_census.py`` holds the table to this
#: literal. Six are emitted today (``unsupported_mechanism``: a write policy
#: refused the mechanism; ``component_unavailable``: the model, the layer or
#: the engine has no such tensor; ``empty_selector``: a selector matched no
#: unit; ``alignment_missing`` / ``alignment_ambiguous``: a position's value
#: occurs zero or several times in a row, or an authored answer form has no
#: counterpart in the form the data carries — ``protocol/positions/alignment.py``;
#: ``chat_template_missing``: a document declares the chat frame and the
#: tokenizer carries no chat template to render it — ``protocol/positions/
#: framing.py``); the other two are declared here so no later change invents
#: its own spelling.
ReasonCode = Literal[
    "unsupported_mechanism",
    "component_unavailable",
    "alignment_missing",
    "alignment_ambiguous",
    "empty_selector",
    "chat_template_missing",
    "ragged_write_unsupported",
    "overlapping_write_unproven",
]
REASON_CODES: tuple[ReasonCode, ...] = get_args(ReasonCode)


class ProtocolError(ValueError):
    """Base class for every intervention-protocol load rejection.

    Subclasses set [`code`][]; the message always leads with it so a bare
    ``str(err)`` in a log or CI transcript identifies the rule without the
    exception type. ``reason`` is the optional [`ReasonCode`][] — checked
    against the vocabulary here, so no code path can invent one.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        reason: ReasonCode | None = None,
    ) -> None:
        if reason is not None and reason not in REASON_CODES:
            raise AssertionError(
                f"unknown reason code {reason!r}; expected one of {REASON_CODES}"
            )
        self.code = code
        self.path = path
        self.reason = reason
        #: the message without the code/path prefix — so an aggregate can
        #: re-format a violation rather than re-parse its rendered form
        self.message = message
        where = f" at {path}" if path else ""
        super().__init__(f"[{code}]{where} {message}")


class ParseError(ProtocolError):
    """The document is not a well-formed protocol object (strict parse).

    Codes:

    * ``P1`` — not valid JSON/YAML, or the top level is not an object
    * ``P2`` — a section or field has the wrong type / shape
    * ``P3`` — unknown key (strict keys; suggestions offered)
    * ``P4`` — a closed enum received an unknown value (suggestions offered)
    * ``P5`` — a derived field was authored (spec §6)
    """

    def __init__(self, code: str, message: str, *, path: str | None = None) -> None:
        if code not in {"P1", "P2", "P3", "P4", "P5"}:
            raise AssertionError(f"unknown ParseError code {code!r}")
        super().__init__(code, message, path=path)


class ValidationError(ProtocolError):
    """A well-formed document violates checklist rule ``rule`` (spec §5).

    ``rule`` names an entry of [`RULES`][] either by its frozen number or
    by its slug; the two spellings build the same error. [`rule`][] is the
    number and [`rule_id`][] the slug — the tests' contract is one failing
    document per rule, asserted by either.
    """

    def __init__(
        self,
        rule: int | str,
        message: str,
        *,
        path: str | None = None,
        reason: ReasonCode | None = None,
    ) -> None:
        found = lookup_rule(rule)
        #: the rule's frozen number (the ``<n>`` of ``V<n>``)
        self.rule: int = found.number
        #: the rule's stable id
        self.rule_id: str = found.slug
        super().__init__(found.code, message, path=path, reason=reason)


class ValidationErrors(ValidationError):
    """Several *independent* checklist violations, reported together.

    Reporting one violation at a time was never wrong, only expensive: each
    round trip costs an edit and a re-run, and §5's rules are independent of
    one another, so the second failure was already knowable when the first was
    reported. A document with three unrelated problems should say so once.

    A **subclass** of [`ValidationError`][] rather than a sibling, so every
    caller that catches one keeps working and every test that asserts a rule
    number still can: `rule`, `code` and `path` are the
    first violation's, in checklist order. [`errors`][] is the whole list,
    and ``str`` names each violation with its own path.

    Raised only for two or more. One violation is raised as itself, so the
    single-error message a reader (and a test) sees is unchanged.
    """

    def __init__(self, errors: Sequence[ValidationError]) -> None:
        if len(errors) < 2:
            raise AssertionError(
                "ValidationErrors reports two or more violations; raise a lone "
                "ValidationError as itself"
            )
        #: every violation, in checklist order
        self.errors: tuple[ValidationError, ...] = tuple(errors)
        first = self.errors[0]
        super().__init__(
            first.rule, first.message, path=first.path, reason=first.reason
        )

    def __str__(self) -> str:
        listing = "\n".join(f"  {err}" for err in self.errors)
        return (
            f"{len(self.errors)} independent checklist violations "
            f"(docs/intervention_protocol.md §5):\n{listing}"
        )


class ProtocolWarning(UserWarning):
    """A document is legal, but not written the conventional way (§5 rule 2).

    The one thing the checklist recommends rather than requires is the order
    of the top-level sections. Order carries no meaning: canonicalization
    emits the sections in the §1 order however they were authored, so two
    documents differing only in order have one digest, one plan and one run.
    Refusing the unconventional one rejected a document that was already, byte
    for byte, the same experiment.

    So the rule is a warning, and it names the order it recommends. Nothing
    downstream depends on it being heeded; a reader does.
    """


def suggest(unknown: str, known: Iterable[str]) -> str:
    """A ``did you mean …?`` suffix for unknown-key/enum rejections.

    Returns an empty string when nothing is close enough — the caller can
    always append the result unconditionally.
    """
    candidates: Sequence[str] = difflib.get_close_matches(
        str(unknown), [str(k) for k in known], n=3
    )
    if not candidates:
        return ""
    quoted = ", ".join(repr(c) for c in candidates)
    return f" — did you mean {quoted}?"


def raise_distinct(refused: list[ValidationError]) -> None:
    """Report what several steps refused on, once.

    One failing step raises exactly what it raised. Several raise their
    *distinct* violations together — a sweep whose every step breaks one rule
    the same way is one violation, and is raised as the first step's, so a
    single-rule refusal keeps the text the loader always gave it. The
    aggregation every per-step pass shares: the compiler's checklist over its
    representatives (``pipeline``) and the engine's per-step rules
    (``neural/shared/step_rules.py``)."""
    if not refused:
        return
    if len(refused) == 1:
        raise refused[0]
    distinct: dict[tuple[str, str | None, str], ValidationError] = {}
    for err in refused:
        each = err.errors if isinstance(err, ValidationErrors) else (err,)
        for violation in each:
            distinct.setdefault(
                (violation.code, violation.path, violation.message), violation
            )
    if len(distinct) == 1:
        raise next(iter(distinct.values()))
    raise ValidationErrors(tuple(distinct.values()))
