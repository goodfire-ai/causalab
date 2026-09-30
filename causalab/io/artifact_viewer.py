#!/usr/bin/env python3
"""Generic, spec-driven HTML artifact viewer.

Renders a single self-contained ``index.html`` that lets a reader browse a tree
of experiment artifacts (figures + headings + captions, *no* interpretive prose
— that lives in ``REPORT.md``). Which artifacts to show is **not** hardcoded: it
comes entirely from a declarative spec (``viewer_spec.yaml``), so this module
carries zero analysis-specific knowledge and stays within the ``causalab.io``
layering rule (no imports from ``analyses``/``methods``/``runner``).

Spec schema (YAML)
------------------
Top level::

    title:    "..."            # page <h1>
    subtitle: "..."            # page subheader (optional)
    intro:                     # optional metadata block; values render as Markdown
      Featurizer: "..."        # (or give `intro` a single string -> Markdown block)
      Target LM:  "..."
    sections: [ <section>, ... ]

A ``section`` is either ``{heading, items}`` or a ``{repeat: ...}`` block.

An ``item`` is one of:

  * a *figure*  -> ``{caption, candidates: [glob, ...], height?: <css|int>}``
  * a *row*     -> ``{row: [<figure>, ...]}``  (side-by-side, responsive 2-col)

A ``repeat`` block re-renders a nested structure once per matched directory::

    repeat:
      over: "*/"                      # glob (relative to root) of instance dirs
      label_from: {kind: dirname}     # or {kind: json, path: <rel>, key: <k>}
      heading: "{label}"              # {label} interpolated from label_from
      sections: [ <section>, ... ]    # candidates re-rooted at each instance dir

Resolution
----------
For each figure, ``candidates`` globs are tried **in list order** against the
artifact root; the first glob that matches any file wins (the author's semantic
ordering is authoritative). Within that glob's matches, filetype preference
breaks ties: interactive/static HTML (iframe) > image (``<img>``) > PDF (iframe,
last resort) — see [`FILETYPE_PREFERENCE`][]. A glob ending in ``.*`` is how an
author says "prefer the PNG twin over the PDF". Nothing matched -> the figure
slot is **dropped** (and a build warning emitted), not rendered as a permanent
placeholder; a row, section, or repeat-instance left with no surviving figure is
dropped too. There is no PDF->PNG conversion; emit PNG figures upstream to get
inline images.

The artifact root is typically the **session directory** (not just
``artifacts/``), so candidates under ``plan/figures/`` and ``artifacts/`` both
resolve; the ``**/``-prefixed globs survive the wider root. The viewer's own
``out/figures/`` copies are excluded from resolution, so re-running against a
session-wide root never re-matches them — and the ``figures/`` tree is cleared at
the start of each build, so rebuilds are idempotent (stable file set and size).

A figure may carry an optional ``height`` (iframe embeds only): an integer is
read as pixels (``480`` -> ``480px``), a string is used verbatim (``"60vh"``);
the default is `_DEFAULT_IFRAME_HEIGHT`. Images always fit their width.

The ``intro`` block renders Markdown: a mapping renders each *value* as inline
Markdown (tables, code, links), and a plain string renders as a Markdown block.

Example (default selection shape)
---------------------------------
The default selection when the DAG ends in
``path_steering`` (globs are ``**/``-prefixed to survive the variable
``{task}/{model}/[{sweep}/]`` middle path; interactive ``.html`` is preferred,
then the ``.*`` PNG/PDF twin)::

    title: "..."
    sections:
      - heading: "Counterfactual geometry"
        items:
          - {caption: "Activation manifold (spline)", candidates: ["**/activation_manifold/**/visualization/features_3d.html"]}
          - {caption: "Hellinger output manifold",     candidates: ["**/output_manifold/**/hellinger_pca_3d.html"]}
      - heading: "Path steering — causal faithfulness"
        items:
          - {caption: "Dual-manifold embedding (subspace vs behaviour)", candidates: ["**/path_steering/**/vis/dual_manifold.html"]}
          - caption: "Receptive field — argmax-class decision map over the top subspace PCs (interactive slice / class / path controls)"
            candidates: ["**/path_steering/**/vis/receptive_field.html"]
          - row:
              - {caption: "Isometry MDS — geometric", candidates: ["**/path_steering/**/vis/isometry/geometric/isometry_mds.html"]}
              - {caption: "Isometry MDS — linear",    candidates: ["**/path_steering/**/vis/isometry/linear/isometry_mds.html"]}

The receptive-field figure is a standard part of the ``path_steering`` selection.
Its viz is opt-in upstream, so the run must enable it
(``path_steering.visualizations += [receptive_field]``) for the figure to exist;
otherwise that figure slot is simply dropped.

CLI::

    python -m causalab.io.artifact_viewer --spec PATH --root PATH --out PATH
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover — pyyaml is a direct dependency
    raise ImportError("PyYAML is required for causalab.io.artifact_viewer.") from exc

try:
    import markdown as _markdown
except ImportError:  # pragma: no cover - markdown ships transitively via tensorboard
    _markdown = None  # intro Markdown degrades to escaped plain text

# Representative filetype order, high -> low preference. Images (.jpg/.svg/...)
# are ranked at the .png tier; see _EXT_TIERS for the full grouping.
FILETYPE_PREFERENCE: tuple[str, ...] = (".html", ".png", ".pdf")

_IMG_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"}
)
# Ordered tiers used for tie-breaking within one candidate's matches.
_EXT_TIERS: tuple[frozenset[str], ...] = (
    frozenset({".html"}),  # interactive / static HTML -> iframe
    _IMG_EXTS,  # raster / vector images -> <img>
    frozenset({".pdf"}),  # last resort -> iframe
)

# Default iframe height (HTML / PDF embeds). A figure may override per-item via
# its ``height`` key; images always fit their width and ignore it.
_DEFAULT_IFRAME_HEIGHT = "70vh"

# Markdown extensions enabled for the ``intro`` block (GFM-ish: tables + fences).
_MARKDOWN_EXTENSIONS: tuple[str, ...] = ("tables", "fenced_code", "sane_lists")


# --------------------------------------------------------------------------- #
# Spec loading & validation
# --------------------------------------------------------------------------- #
def load_spec(spec_path: str | Path) -> dict[str, Any]:
    """Parse and lightly validate a ``viewer_spec.yaml``.

    Raises:
        ValueError: the spec is not a mapping, lacks a ``sections`` list, or
            contains an item that is none of figure / row / repeat.
    """
    data = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"viewer spec must be a mapping, got {type(data).__name__}")
    _validate_spec(data)
    return data


def _validate_spec(spec: dict[str, Any]) -> None:
    sections = spec.get("sections")
    if not isinstance(sections, list):
        raise ValueError("viewer spec must contain a `sections` list")
    for section in sections:
        _validate_section(section)


def _validate_section(section: Any) -> None:
    if not isinstance(section, dict):
        raise ValueError(f"section must be a mapping, got {type(section).__name__}")
    if "repeat" in section:
        repeat = section["repeat"]
        if not isinstance(repeat, dict) or not isinstance(repeat.get("sections"), list):
            raise ValueError("a `repeat` block needs a nested `sections` list")
        for sub in repeat["sections"]:
            _validate_section(sub)
        return
    if "heading" not in section or not isinstance(section.get("items"), list):
        raise ValueError(
            "section needs a `heading` and an `items` list (or a `repeat`)"
        )
    for item in section["items"]:
        _validate_item(item)


def _validate_item(item: Any) -> None:
    if not isinstance(item, dict):
        raise ValueError(f"item must be a mapping, got {type(item).__name__}")
    if "row" in item:
        if not isinstance(item["row"], list):
            raise ValueError("`row` must be a list of figures")
        for fig in item["row"]:
            _validate_figure(fig)
        return
    _validate_figure(item)


def _validate_figure(fig: Any) -> None:
    if not isinstance(fig, dict) or "caption" not in fig or "candidates" not in fig:
        raise ValueError(
            "figure item needs `caption` and `candidates` (or use `row`/`repeat`)"
        )
    if not isinstance(fig["candidates"], list):
        raise ValueError("`candidates` must be a list of glob strings")


# --------------------------------------------------------------------------- #
# Build state
# --------------------------------------------------------------------------- #
@dataclass
class _Build:
    out_dir: Path
    copy_assets: bool
    asset_index: int = 0
    used_anchors: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def _resolve_candidates(
    candidates: list[str], base: Path, build: _Build
) -> Path | None:
    """Return the best existing file for an ordered candidate-glob list, or None.

    Globs resolve relative to ``base``; an absolute glob is rejected with a
    warning. The viewer's own ``out/figures/`` copies are excluded, so resolving
    against a session-wide root never re-matches assets from a previous build.
    """
    out_root = build.out_dir.resolve()
    for glob in candidates:
        if Path(glob).is_absolute():
            build.warnings.append(
                f"candidate glob {glob!r} is absolute; skipping (specs must use "
                f"globs relative to --root)."
            )
            continue
        hits = sorted(
            p for p in base.glob(glob) if p.is_file() and not _within(p, out_root)
        )
        if not hits:
            continue
        best = _pick_by_filetype(hits)
        if best is None:
            continue  # matched only unrenderable types; try the next candidate
        _maybe_warn_multi(hits, base, build, glob)
        return best
    return None


def _pick_by_filetype(hits: list[Path]) -> Path | None:
    """Pick the highest-preference renderable file among a glob's matches."""
    for tier in _EXT_TIERS:
        tier_hits = sorted(p for p in hits if p.suffix.lower() in tier)
        if tier_hits:
            return tier_hits[0]
    return None


def _maybe_warn_multi(hits: list[Path], base: Path, build: _Build, glob: str) -> None:
    firsts: set[str] = set()
    for p in hits:
        try:
            rel = p.relative_to(base)
        except ValueError:
            continue
        if rel.parts:
            firsts.add(rel.parts[0])
    if len(firsts) > 1:
        build.warnings.append(
            f"glob {glob!r} matched files across {len(firsts)} top-level dirs "
            f"{sorted(firsts)}; only the first is shown. Use a `repeat` block to "
            f"render each one separately."
        )


# --------------------------------------------------------------------------- #
# Asset copying & HTML embedding
# --------------------------------------------------------------------------- #
def _copy_asset(src: Path, dest_prefix: str, caption: str, build: _Build) -> str:
    """Copy ``src`` into ``out_dir/figures/...`` (or reference in place) -> relative URL."""
    if not build.copy_assets:
        rel = Path(_relpath(src, build.out_dir))
        return rel.as_posix()
    build.asset_index += 1
    ext = src.suffix.lower()
    name = f"{build.asset_index:03d}-{_slug(caption)}{ext}"
    rel = f"figures/{dest_prefix}/{name}" if dest_prefix else f"figures/{name}"
    dst = build.out_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not (dst.exists() and dst.stat().st_size == src.stat().st_size):
        shutil.copy(src, dst)
    return rel


def _embed(rel_url: str, caption: str, height: str | None = None) -> str:
    ext = Path(rel_url).suffix.lower()
    cap = _esc(caption)
    if ext in _IMG_EXTS:
        return (
            f'<figure><img src="{_esc(rel_url)}" loading="lazy" '
            f'style="width:100%;border:1px solid #d0d0d0;border-radius:6px;background:#fff">'
            f"<figcaption>{cap}</figcaption></figure>"
        )
    h = _esc(height or _DEFAULT_IFRAME_HEIGHT)
    return (
        f'<figure><iframe src="{_esc(rel_url)}" loading="lazy" '
        f'style="width:100%;height:{h};border:1px solid #d0d0d0;border-radius:6px;background:#fff">'
        f"</iframe><figcaption>{cap}</figcaption></figure>"
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _render_item(
    item: dict[str, Any], base: Path, dest_prefix: str, build: _Build
) -> str | None:
    """Render an item, or None if every figure it holds is unresolved."""
    if "row" in item:
        figs = [
            rendered
            for f in item["row"]
            if (rendered := _render_figure(f, base, dest_prefix, build)) is not None
        ]
        if not figs:
            return None
        if len(figs) == 1:
            return figs[0]  # lone survivor -> standalone, not a half-empty row
        return f'<div class="row">{"".join(figs)}</div>'
    return _render_figure(item, base, dest_prefix, build)


def _render_figure(
    fig: dict[str, Any], base: Path, dest_prefix: str, build: _Build
) -> str | None:
    """Resolve + embed one figure, or None (with a warning) if nothing matched."""
    src = _resolve_candidates(fig["candidates"], base, build)
    if src is None:
        build.warnings.append(
            f"figure {fig['caption']!r} dropped: no candidate resolved "
            f"(globs {fig['candidates']})."
        )
        return None
    rel = _copy_asset(src, dest_prefix, fig["caption"], build)
    return _embed(rel, fig["caption"], _coerce_height(fig.get("height")))


def _render_section(
    section: dict[str, Any], base: Path, dest_prefix: str, level: int, build: _Build
) -> tuple[str | None, list[tuple[str, str]]]:
    if "repeat" in section:
        return _render_repeat(section["repeat"], base, dest_prefix, level, build)
    heading = section.get("heading", "")
    sec_prefix = f"{dest_prefix}/{_slug(heading)}" if dest_prefix else _slug(heading)
    items = [
        rendered
        for item in section["items"]
        if (rendered := _render_item(item, base, sec_prefix, build)) is not None
    ]
    if not items:
        return None, []  # every slot dropped -> drop the section (and its TOC entry)
    anchor = _unique_anchor(_slug(heading) or "section", build)
    hlvl = max(2, min(level, 6))
    parts = [f'<section id="{anchor}" class="card"><h{hlvl}>{_esc(heading)}</h{hlvl}>']
    parts.extend(items)
    parts.append("</section>")
    toc = [(anchor, heading)] if level == 2 else []
    return "".join(parts), toc


def _render_repeat(
    block: dict[str, Any], base: Path, dest_prefix: str, level: int, build: _Build
) -> tuple[str | None, list[tuple[str, str]]]:
    over = block.get("over", "*/")
    label_from = block.get("label_from", {"kind": "dirname"})
    heading_tmpl = block.get("heading", "{label}")
    nested = block["sections"]
    instances = sorted(p for p in base.glob(over) if p.is_dir())
    if not instances:
        # No directories matched at all -> a glob-misconfig diagnostic, distinct
        # from an artifact that simply hasn't been produced yet.
        note = (
            f'<section class="card"><p class="ph">No instances matched '
            f"<code>{_esc(over)}</code>.</p></section>"
        )
        return note, []
    body: list[str] = []
    toc: list[tuple[str, str]] = []
    for inst in instances:
        label = _instance_label(inst, label_from)
        heading = heading_tmpl.replace("{label}", label)
        inst_prefix = (
            f"{dest_prefix}/{_slug(label) or inst.name}"
            if dest_prefix
            else (_slug(label) or inst.name)
        )
        sub_parts: list[str] = []
        for sub in nested:
            sub_html, _ = _render_section(sub, inst, inst_prefix, level + 1, build)
            if sub_html is not None:
                sub_parts.append(sub_html)
        if not sub_parts:
            continue  # this instance resolved nothing -> drop it
        anchor = _unique_anchor(_slug(heading) or _slug(inst.name) or "instance", build)
        hlvl = max(2, min(level, 6))
        body.append(
            f'<section id="{anchor}" class="card instance"><h{hlvl}>{_esc(heading)}</h{hlvl}>'
        )
        body.extend(sub_parts)
        body.append("</section>")
        toc.append((anchor, heading))
    if not body:
        return None, []
    return "".join(body), toc


def _instance_label(inst: Path, label_from: dict[str, Any]) -> str:
    kind = label_from.get("kind", "dirname")
    if kind == "json":
        data = _load_json(inst / label_from.get("path", ""))
        key = label_from.get("key", "")
        if isinstance(data, dict) and key in data:
            return str(data[key])
    return inst.name


# --------------------------------------------------------------------------- #
# Page assembly
# --------------------------------------------------------------------------- #
def _intro(spec: dict[str, Any]) -> tuple[str, bool]:
    intro = spec.get("intro")
    if isinstance(intro, str) and intro.strip():
        body = f'<div class="intro-md">{_render_markdown(intro)}</div>'
    elif isinstance(intro, dict) and intro:
        rows = "".join(
            f"<dt>{_esc(k)}</dt><dd>{_render_markdown_inline(v)}</dd>"
            for k, v in intro.items()
        )
        body = f'<dl class="intro">{rows}</dl>'
    else:
        return "", False
    html_str = (
        '<section id="overview" class="card"><h2>Overview &amp; resources</h2>'
        f"{body}</section>"
    )
    return html_str, True


def _toc(entries: list[tuple[str, str]]) -> str:
    if not entries:
        return ""
    items = "".join(
        f'<li><a href="#{a}">{_esc(label)}</a></li>' for a, label in entries
    )
    return f"<nav><b>Index</b><ol>{items}</ol></nav>"


def _assemble(
    spec: dict[str, Any], intro_html: str, toc_html: str, body_html: str
) -> str:
    title = _esc(spec.get("title", "Artifact viewer"))
    subtitle = spec.get("subtitle")
    sub_html = f"<p>{_esc(subtitle)}</p>" if subtitle else ""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>{_CSS}</style></head><body>"
        f"<header><h1>{title}</h1>{sub_html}</header><main>"
        f"{intro_html}{toc_html}{body_html}</main></body></html>"
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def build_viewer(
    spec: dict[str, Any] | str | Path,
    artifact_root: str | Path,
    out_dir: str | Path,
    *,
    copy_assets: bool = True,
) -> Path:
    """Render a self-contained artifact-viewer page.

    Resolves every figure item's candidate globs against ``artifact_root``,
    applies filetype preference, copies the chosen files into ``out_dir/figures/``
    (when ``copy_assets``), and writes ``out_dir/index.html`` referencing them by
    relative path. Returns the path to the written ``index.html``.
    """
    if isinstance(spec, (str, Path)):
        spec = load_spec(spec)
    else:
        _validate_spec(spec)

    base = Path(artifact_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Idempotent rebuild: clear figures/ up front so re-runs don't accumulate
    # stale copies (recreated lazily as assets are copied).
    fig_dir = out / "figures"
    if copy_assets and fig_dir.exists():
        shutil.rmtree(fig_dir)
    build = _Build(out_dir=out, copy_assets=copy_assets)

    intro_html, has_intro = _intro(spec)
    toc: list[tuple[str, str]] = (
        [("overview", "Overview & resources")] if has_intro else []
    )
    body_parts: list[str] = []
    for section in spec.get("sections", []):
        section_html, section_toc = _render_section(section, base, "", 2, build)
        if section_html is None:
            continue  # section had no resolvable figures -> dropped
        body_parts.append(section_html)
        toc.extend(section_toc)

    index = out / "index.html"
    index.write_text(
        _assemble(spec, intro_html, _toc(toc), "".join(body_parts)), encoding="utf-8"
    )

    files = [p for p in fig_dir.rglob("*") if p.is_file()] if fig_dir.exists() else []
    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"[artifact_viewer] wrote {index} ({len(files)} figures, {total_mb:.1f} MB)")
    for warning in build.warnings:
        print(f"[artifact_viewer] warning: {warning}", file=sys.stderr)
    return index


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _esc(value: Any) -> str:
    return html.escape(str(value)) if value is not None else ""


def _slug(text: Any) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return s[:48]


def _unique_anchor(base_slug: str, build: _Build) -> str:
    anchor = base_slug
    n = 2
    while anchor in build.used_anchors:
        anchor = f"{base_slug}-{n}"
        n += 1
    build.used_anchors.add(anchor)
    return anchor


def _relpath(target: Path, start: Path) -> str:
    import os

    return os.path.relpath(target, start)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _within(path: Path, root: Path) -> bool:
    """True if ``path`` is ``root`` itself or nested under it (both resolved)."""
    try:
        resolved = path.resolve()
    except OSError:  # pragma: no cover - resolve() rarely raises on existing files
        return False
    return resolved == root or root in resolved.parents


def _coerce_height(value: Any) -> str | None:
    """Normalise a figure ``height``: int/float -> ``"<n>px"``, str -> verbatim."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return f"{int(value)}px"
    text = str(value).strip()
    return text or None


def _render_markdown(text: Any) -> str:
    """Render ``text`` as a Markdown block (escaped plain text if unavailable)."""
    source = str(text)
    if _markdown is None:  # pragma: no cover - markdown ships transitively
        return _esc(source)
    return _markdown.markdown(source, extensions=list(_MARKDOWN_EXTENSIONS)).strip()


def _render_markdown_inline(value: Any) -> str:
    """Render Markdown, unwrapping a lone enclosing ``<p>`` so short values stay inline."""
    rendered = _render_markdown(value)
    match = re.fullmatch(r"<p>(.*)</p>", rendered, re.DOTALL)
    match = re.fullmatch(r"<p>(.*?)</p>", rendered, re.DOTALL)
    return match.group(1) if match else rendered


_CSS = """
:root{--fg:#1a1a1a;--muted:#777;--line:#e2e2e2;--accent:#2b5fa8}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:var(--fg);margin:0;line-height:1.5}
header{background:#0f1c2e;color:#fff;padding:28px 40px}
header h1{margin:0 0 6px;font-size:24px}
header p{margin:0;color:#b9c6d6;font-size:14px}
main{max-width:1180px;margin:0 auto;padding:24px 40px 80px}
h2{font-size:21px;border-bottom:2px solid var(--accent);padding-bottom:6px;margin-top:40px}
h3{font-size:16px;color:var(--accent);margin-top:26px;margin-bottom:8px}
h4{font-size:14px;color:var(--accent);margin-top:18px;margin-bottom:6px}
.muted{color:var(--muted);font-weight:400;font-size:.85em}
nav{background:#f6f8fb;border:1px solid var(--line);border-radius:8px;padding:14px 20px;margin:20px 0}
nav ol{margin:6px 0 0;padding-left:20px;columns:2}
nav a{color:var(--accent);text-decoration:none}
nav a:hover{text-decoration:underline}
dl.intro{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;font-size:14px;margin:8px 0 0}
dl.intro dt{font-weight:600;color:#244}
dl.intro dd{margin:0}
dl.intro dd>p:first-child{margin-top:0}
dl.intro dd>p:last-child{margin-bottom:0}
.intro-md{font-size:14px}
.intro-md>:first-child{margin-top:0}
.intro-md>:last-child{margin-bottom:0}
.card table{border-collapse:collapse;font-size:13px;margin:6px 0}
.card th,.card td{border:1px solid var(--line);padding:4px 8px;text-align:left}
.card th{background:#f6f8fb}
figure{margin:14px 0}
figcaption{font-size:13px;color:#444;margin-top:6px;padding-left:2px}
.ph{display:flex;align-items:center;justify-content:center;height:120px;background:#fff5f5;border:1px dashed #e0a0a0;border-radius:6px;color:#a33;font-size:14px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.card{scroll-margin-top:12px}
.instance{border-top:1px solid var(--line);padding-top:4px}
code{background:#eef0f3;padding:1px 5px;border-radius:4px;font-size:.9em}
@media(max-width:820px){.row{grid-template-columns:1fr}nav ol{columns:1}}
"""


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m causalab.io.artifact_viewer",
        description="Render a self-contained HTML artifact viewer from a viewer_spec.yaml.",
    )
    parser.add_argument("--spec", required=True, help="path to viewer_spec.yaml")
    parser.add_argument(
        "--root",
        required=True,
        help=(
            "root that candidate globs resolve against — typically the session "
            "dir, so plan/figures/ and artifacts/ both resolve"
        ),
    )
    parser.add_argument(
        "--out", required=True, help="output dir (index.html + figures/ written here)"
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="reference matched files in place (relative) instead of copying into out/figures/",
    )
    args = parser.parse_args(argv)
    path = build_viewer(args.spec, args.root, args.out, copy_assets=not args.no_copy)
    print(path)


if __name__ == "__main__":
    _main()
