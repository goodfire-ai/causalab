"""Model-free tokenizer pre-flight checks for tasks.

Catches a class of tokenization gotchas *before* any model is loaded or run —
the kind that otherwise stays invisible until a baseline run returns ~0%
accuracy. The motivating case: a task whose
``prompt_suffix`` ends in a space (e.g. ``"\\nAnswer: "``). BPE/sentencepiece
tokenizers encode that trailing space as its own *orphan* whitespace token, so
the model's next-token target becomes the *bare* answer form (``"Kate"``)
rather than the leading-space merged form (``" Kate"``) that the task's
declared answer tokens expect. Every checker comparison then fails, and the
prompt also confuses the model's continuation entirely.

The accepted answer forms come from the task's ``CausalModel.output_tokens``
declaration — the single source of each value's surface forms (see
[`task_forms_resolver`][]).

The checks here are deliberately model-free: they need only a *tokenizer*
(no weights, no GPU, sub-second on CPU), so both task setup and the
experiment run phase can run them as a blocking pre-flight gate.

Detection is tokenizer-family robust. A naive "decode the prompt's last token
and look for a space" breaks on sentencepiece tokenizers (a trailing-space
token decodes to ``''``). Instead we use the fast tokenizer's offset mapping
to flag any token whose character span lies entirely in the prompt's
trailing-whitespace region — an orphan whitespace token — and fall back to a
string-level check only when offsets are unavailable (slow tokenizers).

CLI::

    python -m causalab.tasks.preflight --task entity_binding --model meta-llama/Llama-3.2-1B-Instruct

Exits non-zero if any ``error``-severity finding is raised (the blocking gate).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Finding:
    """One pre-flight observation. ``error`` blocks; ``warning`` is advisory."""

    severity: Severity
    message: str
    sample_index: int


def _orphan_whitespace_tokens(
    tokenizer: Any, text: str
) -> list[tuple[int, tuple[int, int]]] | None:
    """Tokens lying entirely in the trailing-whitespace region of ``text``.

    Returns a list of ``(token_id, (start, end))`` for each orphan whitespace
    token, or ``None`` when the tokenizer cannot produce an offset mapping
    (slow tokenizers) so the caller can fall back to a string-level check.
    """
    last_content = len(text.rstrip())
    if last_content == len(text):
        return []  # no trailing whitespace at all
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except (TypeError, ValueError, NotImplementedError):
        return None  # slow tokenizer — no offsets available
    offsets = enc.get("offset_mapping")
    ids = enc.get("input_ids")
    if offsets is None or ids is None:
        return None
    return [
        (tid, (start, end))
        for tid, (start, end) in zip(ids, offsets)
        if end > start and start >= last_content
    ]


def _content_token_count(tokenizer: Any, text: str) -> int:
    """Number of tokens in ``text`` that decode to non-whitespace content.

    Whitespace/structural tokens (e.g. the sentencepiece metaspace, which
    decodes to ``''``) are not counted, so " Monday" reads as one content
    token regardless of tokenizer family.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    return sum(1 for tid in ids if tokenizer.decode([tid]).strip())


def check_prompt_tokenization(
    samples: Iterable[tuple[str, str]],
    forms_for: Callable[[str], Sequence[str]],
    tokenizer: Any,
) -> list[Finding]:
    """Run the model-free tokenizer checks over ``(raw_input, raw_output)`` pairs.

    ``forms_for(raw_output)`` returns the accepted surface forms of an answer
    value (e.g. ``[" Kate", "Kate"]``) — the task's declared answer-token
    contract, sourced from ``CausalModel.output_tokens``; see
    [`run_task_preflight`][]. Keeping the check over a plain forms-resolver
    keeps it decoupled from how the task declares those forms.

    Flags, per sample:

    * **error** — the prompt ends with an orphan whitespace token while the
      declared answer forms include a leading-space variant. The model's
      next-token target then diverges from the expected token.
    * **warning** — the expected answer is multi-token (token-level
      interventions and probability scoring degrade; a prefix-match checker
      still works at the string level).
    """
    findings: list[Finding] = []
    for i, (raw_input, raw_output) in enumerate(samples):
        forms = list(forms_for(raw_output))
        leading_space_forms = [f for f in forms if f[:1] == " "]

        # --- error: orphan trailing-whitespace token ---
        if leading_space_forms:
            orphans = _orphan_whitespace_tokens(tokenizer, raw_input)
            triggered = False
            detail = ""
            if orphans is None:
                # Slow tokenizer: no offsets. Fall back to the string surface —
                # a prompt ending in a space (incl. "\n ") is the footgun.
                triggered = raw_input.endswith((" ", "\t"))
                detail = "prompt ends with trailing whitespace"
            elif orphans:
                triggered = True
                tid = orphans[0][0]
                detail = (
                    f"prompt ends with an orphan whitespace token "
                    f"(id {tid}, {tokenizer.decode([tid])!r})"
                )
            if triggered:
                findings.append(
                    Finding(
                        "error",
                        f"Tokenization mismatch: {detail}. The model's next-token "
                        f"target becomes the bare form {raw_output!r}, but the task's "
                        f"declared answer forms expect a leading-space form "
                        f"(e.g. {leading_space_forms[0]!r}). Fix: drop the trailing "
                        f"whitespace from prompt_suffix / the template so the prompt "
                        f'ends on a non-whitespace character (use "\\nAnswer:" not '
                        f'"\\nAnswer: ").',
                        i,
                    )
                )

        # --- warning: multi-token expected answer ---
        primary = (
            leading_space_forms[0]
            if leading_space_forms
            else (forms[0] if forms else raw_output)
        )
        n_content = _content_token_count(tokenizer, primary)
        if n_content > 1:
            findings.append(
                Finding(
                    "warning",
                    f"Expected answer {primary!r} tokenizes to {n_content} content "
                    f"tokens; token-level interventions and probability scoring "
                    f"degrade for multi-token answers (a prefix-match checker still "
                    f"works at the string level).",
                    i,
                )
            )
    return findings


def task_forms_resolver(task: Any) -> Callable[[str], Sequence[str]]:
    """Return ``forms_for(raw_output) -> [surface form, ...]`` for ``task``.

    Answer forms come from ``CausalModel.output_tokens`` — the single declaration
    of each value's accepted surface forms. For a value it does not
    declare (e.g. a sentinel like ``"UNKNOWN"``), assume the conventional BPE
    leading-space + bare forms, so the orphan-whitespace check still covers every
    sample (it is a prompt-level property, not a per-value one).
    """
    output_tokens = getattr(task.causal_model, "output_tokens", None) or {}
    by_value: dict[str, list[str]] = {}
    for var_map in output_tokens.values():
        for value, forms in var_map.items():
            if forms:
                by_value.setdefault(str(value).strip(), list(forms))

    def forms_for(raw_output: str) -> Sequence[str]:
        return by_value.get(str(raw_output).strip()) or [f" {raw_output}", raw_output]

    return forms_for


def run_task_preflight(task: Any, tokenizer: Any, n_samples: int = 8) -> list[Finding]:
    """Sample ``n_samples`` inputs from ``task`` and run the tokenizer checks.

    Individual sampling failures are tolerated (a flaky sampler shouldn't sink
    the whole gate). Raises ``RuntimeError`` only if *no* sample could be drawn,
    so the caller can distinguish "couldn't run" from "found problems".
    """
    cm = task.causal_model
    samples = []
    last_error: Exception | None = None
    for _ in range(n_samples):
        try:
            trace = cm.sample_input()
        except Exception as exc:  # noqa: BLE001 — surface a clean message instead
            last_error = exc
            continue
        samples.append((trace["raw_input"], trace["raw_output"]))
    if not samples:
        raise RuntimeError(
            f"could not draw any sample from task {task.name!r}: {last_error}"
        )
    return check_prompt_tokenization(samples, task_forms_resolver(task), tokenizer)


def format_findings(findings: Sequence[Finding]) -> str:
    """Human-readable, deduplicated summary of findings.

    Identical messages across samples are collapsed (with a sample count) so a
    prompt-wide gotcha doesn't print ``n_samples`` times.
    """
    if not findings:
        return "✓ No tokenization issues detected."
    by_message: dict[tuple[str, str], list[int]] = {}
    for f in findings:
        by_message.setdefault((f.severity, f.message), []).append(f.sample_index)
    lines = []
    for (severity, message), indices in by_message.items():
        marker = "✗ ERROR" if severity == "error" else "⚠ WARNING"
        count = f" (×{len(indices)})" if len(indices) > 1 else f" (sample {indices[0]})"
        lines.append(f"{marker}{count}: {message}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m causalab.tasks.preflight",
        description=(
            "Model-free tokenizer pre-flight check for a task. Loads only the "
            "tokenizer (no model weights) and flags tokenization gotchas — e.g. a "
            "trailing-space prompt suffix that orphans into its own whitespace "
            "token — before any model run. Exits non-zero if an error is found."
        ),
    )
    parser.add_argument(
        "--task", required=True, help="Task name (shipped or session-local)"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name/path; only its tokenizer is loaded (no weights).",
    )
    parser.add_argument("--n-samples", type=int, default=8)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    from causalab.tasks.loader import load_task

    print(f"Tokenizer pre-flight — task={args.task!r} model={args.model!r}")
    try:
        task = load_task(args.task)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        findings = run_task_preflight(task, tokenizer, n_samples=args.n_samples)
    except Exception as exc:  # noqa: BLE001 — clean exit instead of a traceback
        print(f"⚠ PRE-FLIGHT COULD NOT RUN: {exc}")
        print(
            "\nThis is an infrastructure issue, not a tokenization finding (a "
            "factory task may need its run config to sample). Fall back to the "
            "model-dependent token-alignment test before running."
        )
        return 2

    print(format_findings(findings))

    if any(f.severity == "error" for f in findings):
        print(
            "\nPRE-FLIGHT FAILED: fix the tokenization issue(s) above before "
            "running the model. These would otherwise surface as ~0% baseline "
            "accuracy."
        )
        return 1
    print("\nPRE-FLIGHT PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
