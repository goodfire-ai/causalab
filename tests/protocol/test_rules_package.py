"""The ``causalab/protocol/rules/`` package keeps every namespace it absorbed,
and rule 4's address half is the checklist's, not the canonicalizer's.

The §5 checklist lives in one package —
``rules/errors.py`` (the rule table and the rejection classes),
``rules/document.py`` (``validate.py``, plus rule 27 from ``segments.py`` and
the site-address refusals from ``canonical.py``), ``rules/code.py`` (the check
half of ``code.py``), ``rules/data.py`` (the check half of ``loader.py`` plus
``fit_splits.py``) and ``rules/capability.py`` (the §8 vocabulary and
derivation from ``engine.py``, rules 13 and 30 from ``validate.py``, the
routing shortfall from ``compile.py``). The one-beat star-import shims at the
old paths are deleted. Three things hold the move:

* **standalone import** — every ``rules`` submodule imports on its own in a
  fresh interpreter without pulling ``torch`` in;
* **an acyclic runtime graph** — ``capability`` never imports ``document``
  (``document`` imports it for rules 13 and 30), and the package ``__init__``
  is a docstring and nothing else, so nothing can re-enter a half-executed
  ``__init__``;
* **the one behaviour change** — the five refusals ``explicit._canon_site``
  used to raise (a layer outside the tower, a declared stream the layer does
  not carry, a component whose mixer the layer is not, a head outside the
  component's head space or on a component with none, a component the entry
  lacks) are raised by [`validate_document`][causalab.protocol.rules.document.validate_document] under rule 4 with the same
  text, path and reason, and ``canonicalize()`` of the same document now
  returns. Every door still refuses the same document with the same words —
  the compiler's ``validate`` stage precedes its ``canonicalize`` stage — and
  a document with an illegal site *and* an unrelated violation is now
  reported as one [`ValidationErrors`][causalab.protocol.rules.errors.ValidationErrors] naming both, where before the
  site refusal waited for the other to be fixed.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.schema.explicit import canonicalize
from causalab.protocol.pipeline import STAGES
from causalab.protocol.rules.errors import ValidationError, ValidationErrors
from causalab.protocol.rules.document import validate_document
from causalab.protocol.schema import parse_document

from tests._helpers.refusal_snapshot import A3B, ENV
from tests.protocol._docs import base_doc, in_order


pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "causalab" / "protocol" / "rules"
SUBMODULES = ("errors", "document", "code", "data", "capability")


# The namespace-preservation census and the pins on the one-beat star-import
# shims (``protocol/errors.py``, ``validate.py``, ``fit_splits.py``) were
# deleted with the shims.


# --------------------------------------------------------------------------- #
# standalone import, torch-free
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("submodule", SUBMODULES)
def test_a_submodule_imports_standalone_without_torch(submodule: str) -> None:
    code = (
        "import sys\n"
        f"import causalab.protocol.rules.{submodule}\n"
        "print('torch' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", "torch was imported"


# --------------------------------------------------------------------------- #
# the runtime graph
# --------------------------------------------------------------------------- #


def _runtime_package_imports(path: Path) -> set[str]:
    """The ``rules`` siblings a module imports at module level outside
    ``if TYPE_CHECKING:``."""
    out: set[str] = set()
    prefix = "causalab.protocol.rules."
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING":
            continue
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(prefix):
                out.add(node.module[len(prefix) :])
    return out


def test_the_runtime_import_graph_is_acyclic() -> None:
    edges = {
        name: _runtime_package_imports(PACKAGE / f"{name}.py") for name in SUBMODULES
    }
    assert edges["errors"] == set()
    assert edges["code"] == set()
    assert edges["capability"] == {"errors"}
    assert edges["document"] == {"capability", "code", "errors"}
    assert edges["data"] == {"document", "errors"}


def test_the_package_init_is_a_docstring_and_nothing_else() -> None:
    body = ast.parse((PACKAGE / "__init__.py").read_text()).body
    assert len(body) == 1
    assert isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
    docstring = body[0].value.value
    assert isinstance(docstring, str)
    assert "errors" in docstring and "capability" in docstring


# --------------------------------------------------------------------------- #
# the behaviour change: rule 4's address half is the checklist's
# --------------------------------------------------------------------------- #


def _doc(model: str, site: dict[str, Any]) -> dict[str, Any]:
    raw = base_doc()
    raw["model"]["key"] = model
    raw["method"]["sites"]["tgt"] = site
    return in_order(raw)


#: (document, path, reason, message) — the message pinned to the byte the
#: canonicalizer raised before the checklist took the address check over.
SITE_REFUSALS: dict[str, tuple[dict[str, Any], str, str | None, str]] = {
    "layer_out_of_range": (
        _doc("gpt2", {"component": "block_output", "layers": [99]}),
        "sites.tgt.layers",
        None,
        "site 'tgt': layer 99 out of range for the 12-layer model 'gpt2'",
    ),
    "declared_stream_is_not_the_layers": (
        _doc(
            A3B,
            {"component": "block_output", "layers": [0], "stream": "full_attention"},
        ),
        "sites.tgt.stream",
        "component_unavailable",
        "site 'tgt': stream 'full_attention' is declared at layer 0, but that "
        "layer of 'Qwen/Qwen3.6-35B-A3B' carries 'linear_attention' — on a "
        "hybrid tower the stream is a per-layer fact, not a model-wide one",
    ),
    "component_mixer_is_not_the_layers": (
        _doc(A3B, {"component": "attention_premix", "layers": [0]}),
        "sites.tgt.component",
        "component_unavailable",
        "site 'tgt': component 'attention_premix' exists only on a "
        "'full_attention' mixer, but layer 0 of 'Qwen/Qwen3.6-35B-A3B' carries "
        "'linear_attention' — there is no such tensor at this layer. Layers "
        "carrying 'full_attention': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]",
    ),
    "head_out_of_range": (
        _doc("gpt2", {"component": "attention_premix", "layers": [3], "head": 99}),
        "sites.tgt.head",
        None,
        "site 'tgt': head 99 out of range (12 heads in 'attention_premix''s head "
        "space, (batch, position, head·feature))",
    ),
    "no_head_space": (
        _doc("gpt2", {"component": "block_output", "layers": [3], "head": 0}),
        "sites.tgt.head",
        None,
        "site 'tgt': component 'block_output' has no head axis: its shape is "
        "(batch, position, feature): so head 0 would be validated and then "
        "silently dropped. Name a component that has heads ('attention_premix'), "
        "or drop the 'head' field.",
    ),
    "component_unavailable_on_this_entry": (
        _doc("gpt2", {"component": "routed_output", "layers": [3]}),
        "sites.tgt.component",
        "component_unavailable",
        "site 'tgt': component 'routed_output' needs a sparse-MoE block (the "
        "entry declares no experts), which model 'gpt2' does not have: there "
        "is no such tensor on this model",
    ),
}


def _validate(raw: dict[str, Any]) -> None:
    validate_document(parse_document(raw), model_info=ENV.model_info)


@pytest.mark.parametrize("case", sorted(SITE_REFUSALS))
def test_an_illegal_site_is_refused_by_the_checklist_under_rule_4(case: str) -> None:
    raw, path, reason, message = SITE_REFUSALS[case]
    with pytest.raises(ValidationError) as excinfo:
        _validate(raw)
    err = excinfo.value
    assert not isinstance(err, ValidationErrors), "one violation, raised as itself"
    assert err.code == "V4"
    assert err.path == path
    assert err.reason == reason
    assert err.message == message


@pytest.mark.parametrize("case", sorted(SITE_REFUSALS))
def test_canonicalize_no_longer_refuses_an_illegal_site(case: str) -> None:
    """A direct ``canonicalize()`` of the same document returns its canonical
    form: the address check is the checklist's, not canonicalization's — which
    is also what lets ``pipeline.build`` canonicalize such a document (the
    checklist is ``pipeline.validate``'s, after the build, and no longer a
    ``STAGES`` entry) and leave the refusal to ``validate``."""
    raw, _path, _reason, _message = SITE_REFUSALS[case]
    canonical = canonicalize(raw, ENV)
    assert isinstance(canonical, dict)
    assert (
        canonical["method"]["sites"]["tgt"]["component"]
        == (raw["method"]["sites"]["tgt"]["component"])
    )
    assert "validate" not in STAGES and "canonicalize" in STAGES


def test_an_illegal_site_and_another_violation_are_reported_together() -> None:
    """Before the move the checklist raised the rule-11 dead declaration alone and
    the site refusal surfaced only once that was fixed, at canonicalization;
    both are independent facts about the document and are now one report."""
    raw = _doc("gpt2", {"component": "block_output", "layers": [99]})
    raw["method"]["sites"]["spare"] = {"component": "block_output", "layers": [5]}
    raw = in_order(raw)
    with pytest.raises(ValidationErrors) as excinfo:
        _validate(raw)
    reported = {(e.code, e.path) for e in excinfo.value.errors}
    assert reported == {("V4", "sites.tgt.layers"), ("V11", "sites.spare")}
    assert "layer 99 out of range for the 12-layer model 'gpt2'" in str(excinfo.value)


def test_an_unregistered_model_has_no_address_half_to_check() -> None:
    """Before the lift, ``canonicalize`` refused an unknown ``model.key`` before
    ``_canon_site`` looked at any site, so the address checks never ran on a
    model the registry does not know — and ``validate_document`` never asked
    the registry at all. Both facts hold: a document validated on its own
    against an unregistered key (an executor test's ``"test"``, run on a
    loaded bundle) still validates, and the canonicalizer still refuses the
    key under rule 4 at ``model.key``."""
    raw = _doc(
        "rules-package/unregistered", {"component": "block_output", "layers": [99]}
    )
    _validate(raw)  # no refusal: nothing to check a layer against
    with pytest.raises(ValidationError) as excinfo:
        canonicalize(raw, ENV)
    assert excinfo.value.code == "V4" and excinfo.value.path == "model.key"


def test_only_the_registrys_unknown_key_refusal_skips_the_address_half() -> None:
    """The address check swallows exactly the registry's own refusal (rule 4
    at ``model.key``): the document's other rules still run and are the
    report. Any other ``ValidationError`` the ``model_info`` service raises is
    that service's finding and propagates as itself."""
    raw = _doc("gpt2", {"component": "block_output", "layers": [99]})
    raw["method"]["sites"]["spare"] = {"component": "block_output", "layers": [5]}
    doc = parse_document(in_order(raw))

    def unknown_key(key: str) -> Any:
        raise ValidationError(4, f"model {key!r} is not registered", path="model.key")

    with pytest.raises(ValidationError) as excinfo:
        validate_document(doc, model_info=unknown_key)
    assert not isinstance(excinfo.value, ValidationErrors)
    assert (excinfo.value.code, excinfo.value.path) == ("V11", "sites.spare")

    def some_other_refusal(key: str) -> Any:
        raise ValidationError(4, "not the registry's refusal", path="sites.tgt.layers")

    legal = parse_document(in_order(base_doc()))
    with pytest.raises(ValidationError) as excinfo:
        validate_document(legal, model_info=some_other_refusal)
    err = excinfo.value
    assert not isinstance(err, ValidationErrors)
    assert (err.code, err.path, err.message) == (
        "V4",
        "sites.tgt.layers",
        "not the registry's refusal",
    )


def test_a_legal_document_still_validates_and_canonicalizes() -> None:
    raw = in_order(base_doc())
    _validate(raw)
    assert isinstance(canonicalize(raw, ENV), dict)


def test_validate_document_keeps_its_signature() -> None:
    import inspect

    params = inspect.signature(validate_document).parameters
    assert list(params) == [
        "doc",
        "engine_is_local",
        "engine_capabilities",
        "model_info",
    ]
    assert all(
        params[p].kind is inspect.Parameter.KEYWORD_ONLY
        for p in ("engine_is_local", "engine_capabilities", "model_info")
    )
