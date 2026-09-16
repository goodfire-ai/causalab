"""Task-setup figure renderer (analysis-neutral, stdlib-only).

Renders a self-contained ``task_setup.html`` page that summarizes the task(s) an
experiment runs: one card per task with its name + target variable, an optional
one-line description, and a few worked ``prompt -> expected answer`` examples.

It is a **pure renderer** — plain data in, an HTML string out. It imports only
the standard library (no numpy / plotly, no model / analysis / runner imports),
so it stays within the ``causalab.io`` layering rule (invariant 3,
``tests/test_architecture_layering.py``). No shipped producer currently loads
tasks, samples examples, and calls :func:`write_task_setup_html`. Callers must
prepare the mappings below and invoke the renderer directly; the artifact
viewer can embed the resulting HTML as an iframe figure.

A task is a mapping::

    {
        "name": "comparative_degree",          # required
        "target_variable": "degree",           # optional
        "description": "Rate A vs B on 1-5.",  # optional one-liner
        "examples": [                          # optional; rendered as-available
            {"prompt": "...", "answer": "..."},
            ...
        ],
    }

A task with no examples renders a note rather than crashing, mirroring the
artifact viewer's "drop, don't fail" handling of missing figures.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["render_task_setup_html", "write_task_setup_html"]


def _esc(value: Any) -> str:
    """HTML-escape a value (``None`` -> empty string)."""
    return html.escape("" if value is None else str(value))


def _example_block(example: Mapping[str, Any], index: int) -> str:
    """Render one ``prompt -> expected answer`` example."""
    prompt = _esc(example.get("prompt"))
    answer = _esc(example.get("answer"))
    return (
        '<div class="example">'
        f'<div class="ex-label">Example {index}</div>'
        f'<pre class="prompt">{prompt}</pre>'
        f'<div class="answer"><span class="arrow">&rarr; expected</span> '
        f"<code>{answer}</code></div>"
        "</div>"
    )


def _task_card(task: Mapping[str, Any]) -> str:
    """Render one task as a card: header + optional description + examples."""
    name = _esc(task.get("name") or "task")
    target = task.get("target_variable")
    target_html = (
        f'<span class="muted">target: <code>{_esc(target)}</code></span>'
        if target
        else ""
    )
    description = task.get("description")
    desc_html = f'<p class="desc">{_esc(description)}</p>' if description else ""

    examples = list(task.get("examples") or [])
    if examples:
        body = "".join(_example_block(ex, i) for i, ex in enumerate(examples, start=1))
    else:
        body = '<p class="empty">No examples available.</p>'

    return (
        f'<section class="card"><h2>{name}{target_html}</h2>{desc_html}{body}</section>'
    )


def render_task_setup_html(
    tasks: Sequence[Mapping[str, Any]],
    *,
    title: str = "Task setup",
) -> str:
    """Render the task-setup page as a self-contained HTML string.

    Args:
        tasks: one mapping per task (see module docstring for the shape).
        title: page ``<h1>`` and ``<title>``.

    Returns:
        A complete ``<!doctype html>`` document. With no tasks, renders an empty
        state rather than failing.
    """
    if tasks:
        cards = "".join(_task_card(task) for task in tasks)
    else:
        cards = '<section class="card"><p class="empty">No tasks to show.</p></section>'
    title_esc = _esc(title)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title_esc}</title><style>{_CSS}</style></head><body>"
        f"<header><h1>{title_esc}</h1></header><main>{cards}</main></body></html>"
    )


def write_task_setup_html(
    tasks: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    title: str = "Task setup",
) -> Path:
    """Render the page and write it to ``output_path`` (creating parents)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_task_setup_html(tasks, title=title), encoding="utf-8")
    return out


_CSS = """
:root{--fg:#1a1a1a;--muted:#777;--line:#e2e2e2;--accent:#2b5fa8}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:var(--fg);margin:0;line-height:1.5;background:#fff}
header{background:#0f1c2e;color:#fff;padding:20px 28px}
header h1{margin:0;font-size:20px}
main{max-width:1080px;margin:0 auto;padding:20px 28px 60px}
.card{border:1px solid var(--line);border-radius:8px;padding:14px 18px;margin:16px 0}
.card h2{font-size:18px;border-bottom:2px solid var(--accent);padding-bottom:6px;margin:0 0 10px;display:flex;align-items:baseline;gap:12px}
.muted{color:var(--muted);font-weight:400;font-size:.7em}
.desc{font-size:14px;color:#444;margin:0 0 12px}
.example{border-left:3px solid var(--line);padding:2px 0 2px 12px;margin:12px 0}
.ex-label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px}
pre.prompt{white-space:pre-wrap;word-break:break-word;background:#f6f8fb;border:1px solid var(--line);border-radius:6px;padding:8px 10px;margin:0;font-size:13px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.answer{margin-top:6px;font-size:14px}
.answer .arrow{color:var(--accent);font-weight:600;margin-right:6px}
code{background:#eef0f3;padding:1px 5px;border-radius:4px;font-size:.9em;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.empty{color:var(--muted);font-style:italic;margin:6px 0}
"""
