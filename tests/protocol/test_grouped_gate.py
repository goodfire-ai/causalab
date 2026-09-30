"""Group legality (spec §5.23) and the grouped gate's pure layer (§2.5 ``group``).

`Gate` was coordinate-wise by construction: `theta = zeros(width)`, one
parameter per channel, and an attention-head mask — the primitive every
head-level DBM study needs — had to be assembled campaign-side out of a
coordinate gate plus a hand-built index map. `featurizers.<name>.group` makes
it a declaration, with two values: `head` (one θ per head of a head-major
component) and `expert_neuron` (one θ per (expert, neuron) of the routed
expert table). The map is derived, never authored.

What this file pins is the part that makes the feature *cheap*, and it is a
property of the pure layer rather than of the engine:

* the map is **derived from the registry** — `(heads, head_dim)` off the
  component's shape, `(num_experts, d_expert)` off the model's expert table —
  so "how many parameters" needs no model, no tokenizer and no torch, which is
  why `group` resolves in `canonical`, next to the width it already derives;
* a `group` the site cannot honour is a **load** error (§5.23, rule 23), not a
  silent collapse to one parameter: a component with no such axis, a site that
  already selects one member of it, a basis-changing stage before the gate,
  two sites laying the units out differently. A loaded (`file_path`) grouped
  gate is checked the same way;
* **`group` absent changes nothing.** `_canon_featurizer` starts from
  `dict(entry)`, so a field no document authored stays out of the canonical
  form — the digest of every document that predates this feature is
  byte-identical, and a per-coordinate bundle stamped before the field existed
  still loads under an ungrouped document. `coordinate` is not a value: the default has no spelling.
"""

from __future__ import annotations

import dataclasses
import json
import struct
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.schema.explicit import canonicalize, digest
from causalab.protocol.rules.errors import ParseError, ValidationError
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.registry import component_width, get_model_info, site_group_map
from causalab.protocol.schema import GATE_GROUPS, GATE_MAPS, parse_document

from tests.protocol._docs import base_doc, in_order
from tests.protocol._env import build_env


pytestmark = pytest.mark.unit

#: gpt2's attention interior: 12 heads of 64, so a head-grouped gate over it
#: has 12 parameters over a 768-wide site.
GPT2_HEADS = 12
GPT2_HEAD_DIM = 64
#: The registered Qwen3.6 entry: 16 query heads of 256 at `attention_premix`,
#: 32 value heads of 128 at `delta_premix`, 256 experts of d_expert 512 routed
#: top-8. Registry-only — no weights anywhere in this file.
A3B = "Qwen/Qwen3.6-35B-A3B"
A3B_EXPERTS, A3B_D_EXPERT, A3B_TOP_K = 256, 512, 8


def gate_doc(
    *,
    group: Any = "head",
    component: str = "attention_premix",
    chain: Any = "g",
    featurizers: dict[str, Any] | None = None,
    site_extra: dict[str, Any] | None = None,
    model: str | None = None,
    layer: int = 3,
) -> dict[str, Any]:
    """`base_doc` with a gate on the target site, trained.

    A gate is only interesting as a trained featurizer, and `train` is what
    obliges the `save` entry, so the builder carries both — a document that
    declares a gate and trains nothing is refused for a reason this file is
    not about.
    """
    doc = base_doc()
    if model is not None:
        doc["model"] = {"key": model, "revision": "main", "dtype": "bf16"}
    doc["method"]["sites"]["tgt"] = {
        "component": component,
        "layers": [layer],
        **(site_extra or {}),
    }
    gate: dict[str, Any] = {"kind": "gate"}
    if group is not None:
        gate["group"] = group
    doc["method"]["featurizers"] = {"g": gate, **(featurizers or {})}
    doc["method"]["reads"]["v_cf"]["featurizer"] = chain
    doc["method"]["writes"]["patch"]["featurizer"] = chain
    doc["method"]["train"] = {
        "objective": [[1.0, fit_term(doc)]],
        "params": ["g"],
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
        "seed": 0,
    }
    doc["method"]["save"].append(
        {"value": "g", "site": "tgt", "file_path": "g.safetensors"}
    )
    return in_order(doc)


def apply_doc(
    *, group: Any = "head", component: str = "attention_premix", layer: int = 3
) -> dict[str, Any]:
    """`gate_doc` as the replay document: the gate loads `fit/g.safetensors`,
    nothing is trained, nothing but the metric is saved."""
    doc = gate_doc(group=group, component=component, layer=layer)
    doc["method"]["featurizers"]["g"]["file_path"] = "fit/g.safetensors"
    del doc["method"]["train"]
    doc["method"]["save"] = [doc["method"]["save"][0]]  # the aggregation alone
    return in_order(doc)


def fit_term(doc: dict[str, Any]) -> dict[str, Any]:
    """The `ld` aggregation as an objective term's fields (§2.11): the read,
    the model and the aggregation the save entry `ld.json` carries."""
    (entry,) = [e for e in doc["method"]["save"] if e["file_path"] == "ld.json"]
    return {k: entry[k] for k in ("read", "model", "aggregation")}


def canon_gate(env: Any, **kwargs: Any) -> dict[str, Any]:
    return canonicalize(gate_doc(**kwargs), env)["method"]["featurizers"]["g"]


def expect_rule(rule: int, env: Any, raw: dict[str, Any]) -> ValidationError:
    """Rule 23 needs the model's declared axes, so — like rule 4's width half —
    it is decided while the document canonicalizes, and `load` is the surface
    that runs it. No model is loaded: `ModelInfo` is static config."""
    with pytest.raises(ValidationError) as err:
        compile_protocol(raw, env=env)
    assert err.value.rule == rule, f"expected V{rule}, got {err.value}"
    return err.value


# -- the map is derived from the registry ----------------------------------- #


def test_the_map_resolves_with_no_model_loaded() -> None:
    """Model-free by construction: `ModelInfo` is static config, and this is
    the whole resolution — for both group kinds."""
    assert site_group_map(get_model_info("gpt2"), "head", "attention_premix") == (
        GPT2_HEADS,
        GPT2_HEAD_DIM,
    )
    info = get_model_info(A3B)
    assert site_group_map(info, "head", "attention_premix") == (16, 256)
    assert site_group_map(info, "head", "delta_premix") == (32, 128)
    assert site_group_map(info, "expert_neuron", "expert_activation") == (
        A3B_EXPERTS,
        A3B_D_EXPERT,
    )


def test_a_map_covers_every_coordinate_of_a_head_with_that_head() -> None:
    """`(heads, head_dim)` tiles the site exactly, and one head's width — the
    slice a `head` field would select — is the group width, so group *h* is
    head *h*: consecutive runs of `head_dim` coordinates share a parameter."""
    info = get_model_info("gpt2")
    heads, head_dim = site_group_map(info, "head", "attention_premix")
    assert heads * head_dim == component_width(info, "attention_premix")
    assert head_dim == component_width(info, "attention_premix", head=0)
    assert [i // head_dim for i in range(heads * head_dim)] == [
        h for h in range(heads) for _ in range(head_dim)
    ]


def test_a_head_grouped_gate_has_exactly_one_parameter_per_head(env: Any) -> None:
    """The head contract, in the canonical form: H parameters over a component
    with H heads."""
    entry = canon_gate(env)
    assert entry["width"] == GPT2_HEADS * GPT2_HEAD_DIM
    assert entry["params"]["theta"] == [GPT2_HEADS]
    assert entry["group"] == "head"


@pytest.mark.parametrize("component", ["expert_activation", "expert_neuron_output"])
def test_an_expert_keyed_gate_has_the_whole_expert_table(env: Any, component) -> None:
    """The expert contract: `num_experts × d_expert` parameters whatever `top_k`
    experts a token activates — the site is `top_k` slots wide, the table is
    not."""
    entry = canon_gate(env, group="expert_neuron", component=component, model=A3B)
    assert entry["width"] == A3B_TOP_K * A3B_D_EXPERT
    assert entry["params"]["theta"] == [A3B_EXPERTS, A3B_D_EXPERT]


# -- what is refused, and where ------------------------------------------- #


def test_group_is_a_closed_vocabulary() -> None:
    with pytest.raises(ParseError) as err:
        parse_document(gate_doc(group="heads"))
    assert err.value.code == "P4"
    assert "head" in str(err.value)  # the suggestion


def test_coordinate_is_not_a_value() -> None:
    """The default has no spelling: an absent `group` is the per-coordinate
    gate, and `coordinate` is refused like any unknown value. Writing it would
    give the default a canonical form of its own — and, stamped, would refuse
    every per-coordinate bundle fitted before the field existed."""
    assert "coordinate" not in GATE_GROUPS
    with pytest.raises(ParseError) as err:
        parse_document(gate_doc(group="coordinate"))
    assert err.value.code == "P4"


def test_group_is_a_gate_field_only() -> None:
    """A subspace's parameters are a basis over the whole axis; there is no
    per-head basis to tie them to, so the field is not in its vocabulary."""
    with pytest.raises(ParseError) as err:
        parse_document(
            gate_doc(
                group=None,
                featurizers={"rot": {"kind": "subspace", "k": 4, "group": "head"}},
            )
        )
    assert err.value.code == "P3"


def test_rule_23_a_group_the_component_has_no_axis_for_is_a_load_error(
    env: Any,
) -> None:
    err = expect_rule(23, env, gate_doc(component="block_output"))
    assert "no head axis" in str(err)
    assert "block_output" in str(err)
    assert err.path == "featurizers.g.group"


def test_rule_23_expert_neuron_off_expert_activation_is_a_load_error(
    env: Any,
) -> None:
    """The expert table is laid out one expert's neurons per routed slot by
    `expert_activation` and `expert_neuron_output`; on the shared expert the per-coordinate gate is
    already one parameter per neuron."""
    err = expect_rule(
        23,
        env,
        gate_doc(
            group="expert_neuron", component="shared_expert_activation", model=A3B
        ),
    )
    assert "'shared_expert_activation'" in str(err) and "expert_neuron" in str(err)


def test_rule_23_expert_neuron_on_a_model_with_no_expert_table_is_refused(
    env: Any,
) -> None:
    """gpt2 declares no experts: nothing to key the parameters by. Refused by
    the component first — gpt2 has no `expert_activation` — which is the same
    rule number either way."""
    err = expect_rule(
        23, env, gate_doc(group="expert_neuron", component="attention_premix")
    )
    assert "expert_neuron" in str(err)


def test_rule_23_a_site_that_already_selects_one_head_is_refused(env: Any) -> None:
    """H groups over one head is one group — a coordinate-wise gate under a
    name that claims otherwise."""
    err = expect_rule(23, env, gate_doc(site_extra={"head": 3}))
    assert "already selects head 3" in str(err)


def test_rule_23_a_grouped_gate_after_a_rotation_is_refused(env: Any) -> None:
    """The chain rule with teeth: after a k=8 rotation, coordinate 5 is a
    coordinate of a *fitted basis*, not a channel of head 0."""
    err = expect_rule(
        23,
        env,
        gate_doc(
            chain=["rot", "g"],
            featurizers={"rot": {"kind": "subspace", "k": 8}},
        ),
    )
    assert "must be the first stage of its chain" in str(err)


def test_rule_23_a_grouped_gate_after_a_standardize_is_refused(env: Any) -> None:
    """The rule is positional, not about which kinds preserve a basis: a grouped
    gate acts on the component's own coordinates, so it is the first stage of
    its chain or it is refused — a per-coordinate ``standardize`` ahead of it
    included."""
    err = expect_rule(
        23,
        env,
        gate_doc(
            chain=["z", "g"],
            featurizers={"z": {"kind": "standardize"}},
        ),
    )
    assert "must be the first stage of its chain" in str(err)


def test_rule_23_two_sites_laid_out_differently_are_refused(env: Any) -> None:
    """One featurizer, one parameter set, so one map: the A3B's two head-major
    families are both 4096 wide — the width check passes — and lay their heads
    out as 16 × 256 and 32 × 128."""
    # the operand is read at the shallower site so rule 21 has nothing to say;
    # the second write lands deeper, at the other family
    doc = gate_doc(model=A3B, component="delta_premix", layer=0)
    doc["method"]["sites"]["other"] = {"component": "attention_premix", "layers": [3]}
    doc["method"]["writes"]["patch_other"] = {
        "site": "other",
        "pos": -1,
        "featurizer": "g",
        "do": {"swap": "v_cf"},
    }
    doc["method"]["intervened_models"]["patched"]["writes"].append("patch_other")
    err = expect_rule(23, env, in_order(doc))
    assert "laid out differently" in str(err)
    assert "(16, 256)" in str(err) and "(32, 128)" in str(err)


# -- the cheapness claim --------------------------------------------------- #


def test_an_ungrouped_gate_is_unchanged_byte_for_byte(env: Any) -> None:
    """The property that makes the feature cheap, asserted rather than
    assumed: a document that authors no `group` canonicalizes to exactly what
    it did before the field existed, so **no existing digest moves**."""
    doc = gate_doc(group=None, component="block_output")
    entry = canonicalize(doc, env)["method"]["featurizers"]["g"]
    assert "group" not in entry
    assert "group_map" not in entry and "group_width" not in entry
    assert entry["params"]["theta"] == [entry["width"]]


def test_adding_a_group_changes_the_digest(env: Any) -> None:
    """The converse: a grouped gate is a different intervention from a
    coordinate-wise one, so it must not share its identity."""
    plain = digest(canonicalize(gate_doc(group=None), env))
    grouped = digest(canonicalize(gate_doc(group="head"), env))
    assert plain != grouped


# -- the two surfaces a width-gated check would exempt --------------------- #
#
# Rule 23 is decided *above* `_canon_featurizer`'s early returns, because none
# of its clauses needs the width. Gated on the width, two whole surfaces slip
# through, and the second is the one the rule is most for.
#
# These go through `canonicalize` with a permissive artifact store rather than
# through `load`, because rule 15 legitimately runs first: an artifact that does
# not exist is refused before anything can be said about its grouping. Stubbing
# the store is what isolates the clause under test from that ordering.


class _AnyArtifact:
    """An artifact store where every path exists, with a fixed digest."""

    def file_digest(self, path: str) -> str:
        return "0" * 64

    def identity(self, path: str) -> dict[str, Any] | None:
        return None


def permissive_env(env: Any) -> Any:
    return dataclasses.replace(env, artifacts=_AnyArtifact())


def expect_canon_rule(rule: int, env: Any, raw: dict[str, Any]) -> ValidationError:
    with pytest.raises(ValidationError) as err:
        canonicalize(raw, permissive_env(env))
    assert err.value.rule == rule, f"expected V{rule}, got {err.value}"
    return err.value


def test_rule_23_refuses_a_loaded_grouped_gate(env: Any) -> None:
    """A `file_path` gate is the entire apply/replay surface, and its branch
    returns from `_canon_featurizer` before the width is derived at all. Left
    below that return, `{"kind": "gate", "group": "head", "file_path": …}` on
    a headless component would not be a load error but a run-time one."""
    err = expect_canon_rule(23, env, apply_doc(component="block_output"))
    assert "no head axis" in str(err)


def test_rule_23_refuses_a_grouped_gate_after_an_sae(env: Any) -> None:
    """The clause that a width-gated check structurally could not reach.

    `_raw_stage_output_width` returns `None` for an `sae` — its dictionary size
    lives in the bundle, not the spec — so `_derived_width` gives up there. An
    `sae` before a grouped gate is *exactly* what the basis clause exists to
    refuse: after it, coordinate 5 is a dictionary feature, not a channel of
    head 0.
    """
    err = expect_canon_rule(
        23,
        env,
        gate_doc(
            chain=["dict", "g"],
            featurizers={
                "dict": {"kind": "sae", "file_path": "artifacts/sae.safetensors"}
            },
        ),
    )
    assert "must be the first stage of its chain" in str(err)


def test_a_loaded_ungrouped_gate_is_untouched_by_the_hoist(env: Any) -> None:
    """Non-vacuity for the hoist: a `file_path` gate with no group must still
    canonicalize, and must not grow a group. The check moves *when* the group
    is resolved, not *whether* an ungrouped document is touched."""
    entry = canonicalize(
        apply_doc(group=None, component="block_output"), permissive_env(env)
    )["method"]["featurizers"]["g"]
    assert "group" not in entry and "group_map" not in entry
    assert entry["content_digest"] == "0" * 64


def test_a_loaded_grouped_gate_keeps_its_group_in_the_canonical_form(env: Any) -> None:
    """And the legal case the hoist newly reaches: an apply document whose gate
    *is* grouped legally canonicalizes with the authored group, so a loaded
    per-head mask is not confusable with a coordinate-wise one; the derived map
    is the bundle's to stamp (rule 15 compares it), not the document's."""
    entry = canonicalize(apply_doc(), permissive_env(env))["method"]["featurizers"]["g"]
    assert entry["group"] == "head"
    assert "group_map" not in entry


# -- an ungrouped bundle from before `group` still loads --------------------- #
#
# Every gate bundle fitted before `group` existed (every demo-fitted gate among
# them) was fitted per coordinate, with an ArtifactIdentity that had no `group`
# and no `group_map`. `group` enters the loader's expectation ONLY when the document
# authors a group; an ungrouped document must accept such a header unchanged,
# and there is no re-fit migration. The header below is written by hand with
# exactly the key set that earlier code stamps on a single-point gate fit —
# not through `build_artifact_identity`, whose key set is the extended one.

#: The file-level identity keys of a gate bundle written before `group` (a
#: single-point fit; `k`/`parametrization`/`model_quantization`/`tokenizer`/
#: `trained_on_digest` are absent because the stamp drops `None`).
MAIN_ERA_GATE_HEADER_KEYS = (
    "commit",
    "dtype",
    "engine",
    "model_dtype",
    "model_key",
    "model_revision",
    "produced_by",
    "site",
    "trained_on",
)


def _write_gate_header(
    root: Path,
    rel: str,
    *,
    theta_len: int,
    site: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, str]:
    """A hand-written fitted-gate bundle: a stamped header and a zero ``theta``
    of ``theta_len`` fp32 entries, no tensor library involved. Without
    ``extra`` the header is byte-for-byte what earlier code writes for an
    ungrouped gate; ``extra`` adds the grouped header's ``group``/``group_map``."""
    metadata: dict[str, str] = {
        "commit": "fixture",
        "dtype": "fp32",
        "engine": "pytorch_hooks",
        "model_dtype": "fp32",
        "model_key": "gpt2",
        "model_revision": "main",
        "produced_by": "0" * 64,
        "site": json.dumps(site, sort_keys=True),
        "trained_on": "weekdays/data#train",
    }
    for key, value in (extra or {}).items():
        metadata[key] = value if isinstance(value, str) else json.dumps(value)
    header = {
        "__metadata__": metadata,
        "theta": {
            "dtype": "F32",
            "shape": [theta_len],
            "data_offsets": [0, 4 * theta_len],
        },
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        fh.write(bytes(4 * theta_len))
    return metadata


def test_a_main_era_ungrouped_gate_loads_under_an_ungrouped_document(
    tmp_path: Path,
) -> None:
    """The backward-compatibility clause: no `group`, no `group_map`, and the load is
    accepted — `group` is not expected of a document that authors none."""
    metadata = _write_gate_header(
        tmp_path,
        "fit/g.safetensors",
        theta_len=GPT2_HEADS * GPT2_HEAD_DIM,
        site={"component": "block_output", "layers": [3]},
    )
    assert set(metadata) == set(MAIN_ERA_GATE_HEADER_KEYS)
    loaded = compile_protocol(
        apply_doc(group=None, component="block_output"), env=build_env(tmp_path)
    )
    entry = loaded.canonical["method"]["featurizers"]["g"]
    assert "group" not in entry and "content_digest" in entry


def test_a_main_era_ungrouped_gate_is_refused_by_a_grouped_document(
    tmp_path: Path,
) -> None:
    """The document wants a per-head mask; the bundle is a per-coordinate one
    and cannot prove otherwise. Refused by the missing key, by name."""
    _write_gate_header(
        tmp_path,
        "fit/g.safetensors",
        theta_len=GPT2_HEADS * GPT2_HEAD_DIM,
        site={"component": "attention_premix", "layers": [3]},
    )
    with pytest.raises(ValidationError) as err:
        compile_protocol(apply_doc(group="head"), env=build_env(tmp_path))
    assert err.value.rule == 15
    assert "missing 'group'" in str(err.value)


def test_a_grouped_bundle_is_refused_by_an_ungrouped_document(tmp_path: Path) -> None:
    """The reverse: a bundle fitted per head would have its H parameters read
    as H coordinates. Not a key the expectation carries, so said explicitly."""
    _write_gate_header(
        tmp_path,
        "fit/g.safetensors",
        theta_len=GPT2_HEADS,
        site={"component": "attention_premix", "layers": [3]},
        extra={"group": "head", "group_map": [GPT2_HEADS, GPT2_HEAD_DIM]},
    )
    with pytest.raises(ValidationError) as err:
        compile_protocol(apply_doc(group=None), env=build_env(tmp_path))
    assert err.value.rule == 15
    assert "fitted with group 'head'" in str(err.value)


def test_a_grouped_bundle_reloads_under_the_document_that_fitted_it(
    tmp_path: Path,
) -> None:
    """Valid work still passes: the same header with
    the group and the derived map stamped loads under the grouped document."""
    _write_gate_header(
        tmp_path,
        "fit/g.safetensors",
        theta_len=GPT2_HEADS,
        site={"component": "attention_premix", "layers": [3]},
        extra={"group": "head", "group_map": [GPT2_HEADS, GPT2_HEAD_DIM]},
    )
    compile_protocol(apply_doc(group="head"), env=build_env(tmp_path))


def test_a_grouped_bundle_with_another_map_is_refused(tmp_path: Path) -> None:
    """Same site, same group kind, another head layout stamped: refused by the
    map, not by a width mismatch somewhere in the build."""
    _write_gate_header(
        tmp_path,
        "fit/g.safetensors",
        theta_len=GPT2_HEADS,
        site={"component": "attention_premix", "layers": [3]},
        extra={"group": "head", "group_map": [6, 128]},
    )
    with pytest.raises(ValidationError) as err:
        compile_protocol(apply_doc(group="head"), env=build_env(tmp_path))
    assert err.value.rule == 15
    assert "ArtifactIdentity mismatch on 'group_map'" in str(err.value)


# -- §2.5 `group: site` — one parameter over the whole site ----------------- #

#: gpt2's dense widths: `mlp_output` and `embeddings` are hidden-wide, and a
#: site-grouped gate over either has exactly one parameter.
GPT2_HIDDEN = 768


def test_the_site_map_is_one_group_over_the_whole_width() -> None:
    """Model-free like the other two maps, and defined on every feature-space
    component — the MIB node set is heads, MLP blocks and the embedding, so
    the map has to exist where `head` does not."""
    info = get_model_info("gpt2")
    assert site_group_map(info, "site", "mlp_output") == (1, GPT2_HIDDEN)
    assert site_group_map(info, "site", "embeddings") == (1, GPT2_HIDDEN)
    assert site_group_map(info, "site", "attention_premix") == (
        1,
        GPT2_HEADS * GPT2_HEAD_DIM,
    )


def test_a_site_gate_over_one_head_is_legal(env: Any) -> None:
    """Unlike `head`, no site field collapses `site` into a lie: one θ over one
    head's slice is one unit over one unit. The map is that slice."""
    info = get_model_info("gpt2")
    assert site_group_map(info, "site", "attention_premix", head=3) == (
        1,
        GPT2_HEAD_DIM,
    )
    entry = canon_gate(env, group="site", site_extra={"head": 3})
    assert entry["params"]["theta"] == [1]


def test_a_site_grouped_gate_has_exactly_one_parameter(env: Any) -> None:
    """In the canonical form: one θ over a hidden-wide site, on a layered
    component and on the layer-less embedding alike."""
    entry = canon_gate(env, group="site", component="mlp_output")
    assert entry["width"] == GPT2_HIDDEN
    assert entry["params"]["theta"] == [1]
    assert entry["group"] == "site"
    raw = gate_doc(group="site", component="embeddings")
    del raw["method"]["sites"]["tgt"]["layers"]  # `embeddings` is layer-less
    entry = canonicalize(raw, env)["method"]["featurizers"]["g"]
    assert entry["width"] == GPT2_HIDDEN
    assert entry["params"]["theta"] == [1]


def test_rule_23_a_site_gate_needs_a_feature_space() -> None:
    """An attention pattern has no coordinate axis for one parameter to cover;
    the registry refuses the map by rule 23 before any width is asked for."""
    from causalab.protocol.registry import component_shape, gate_group_map

    info = get_model_info("gpt2")
    with pytest.raises(ValidationError) as err:
        gate_group_map(
            "site",
            component_shape(info, "attention_probs"),
            0,
            component="attention_probs",
        )
    assert err.value.rule == 23
    assert "no feature axis" in str(err.value)


def test_rule_23_a_site_gate_after_a_rotation_is_refused(env: Any) -> None:
    """`site` means the component's own coordinates, like the other two groups:
    one θ over a fitted basis is a different object and is refused the same
    way."""
    err = expect_rule(
        23,
        env,
        gate_doc(
            group="site",
            component="mlp_output",
            chain=["rot", "g"],
            featurizers={"rot": {"kind": "subspace", "k": 8}},
        ),
    )
    assert "must be the first stage of its chain" in str(err)


def test_a_site_bundle_reloads_only_under_a_site_document(tmp_path: Path) -> None:
    """The three-way identity check `head` has: a one-parameter bundle stamped
    `site` loads under `site`, is refused by `head` (another group) and by an
    ungrouped document (which would read its one entry as one coordinate)."""
    site = {"component": "attention_premix", "layers": [3]}
    _write_gate_header(
        tmp_path,
        "fit/g.safetensors",
        theta_len=1,
        site=site,
        extra={"group": "site", "group_map": [1, GPT2_HEADS * GPT2_HEAD_DIM]},
    )
    compile_protocol(apply_doc(group="site"), env=build_env(tmp_path))
    with pytest.raises(ValidationError) as err:
        compile_protocol(apply_doc(group="head"), env=build_env(tmp_path))
    assert err.value.rule == 15
    with pytest.raises(ValidationError) as err:
        compile_protocol(apply_doc(group=None), env=build_env(tmp_path))
    assert err.value.rule == 15
    assert "fitted with group 'site'" in str(err.value)


def test_a_site_group_changes_the_digest_and_head_does_not_move(env: Any) -> None:
    """Authoring `site` is a new experiment (a new digest); the vocabulary
    growing moves no existing `head` or ungrouped document."""
    plain = digest(canonicalize(gate_doc(group=None, component="mlp_output"), env))
    site = digest(canonicalize(gate_doc(group="site", component="mlp_output"), env))
    assert plain != site
    head = canonicalize(gate_doc(group="head"), env)
    assert head["method"]["featurizers"]["g"]["params"]["theta"] == [GPT2_HEADS]


# -- §2.5 `parametrization` on a gate --------------------------------------- #


def _clamp(doc: dict[str, Any]) -> dict[str, Any]:
    doc["method"]["featurizers"]["g"]["parametrization"] = "clamp"
    return doc


def test_gate_parametrization_is_its_own_vocabulary() -> None:
    """One field, an enum per kind: a gate's map is `sigmoid` | `clamp`, a
    subspace's the rotation triple — and neither accepts the other's words."""
    parse_document(_clamp(gate_doc()))
    doc = gate_doc()
    doc["method"]["featurizers"]["g"]["parametrization"] = "cayley"
    with pytest.raises(ParseError) as err:
        parse_document(doc)
    assert err.value.code == "P4" and "clamp" in str(err.value)
    with pytest.raises(ParseError) as err:
        parse_document(
            gate_doc(
                featurizers={
                    "rot": {"kind": "subspace", "k": 4, "parametrization": "clamp"}
                }
            )
        )
    assert err.value.code == "P4" and "cayley" in str(err.value)


def test_an_absent_parametrization_has_no_spelling_in_the_canonical_form(
    env: Any,
) -> None:
    """`sigmoid` is the gate as it always was: unauthored, the canonical form
    carries no key and no existing digest moves; authored, `clamp` is a
    different experiment and the digest says so."""
    assert "parametrization" not in canon_gate(env)
    before = digest(canonicalize(gate_doc(), env))
    after = digest(canonicalize(_clamp(gate_doc()), env))
    assert (
        canonicalize(_clamp(gate_doc()), env)["method"]["featurizers"]["g"][
            "parametrization"
        ]
        == "clamp"
    )
    assert before != after


def test_rule_4_refuses_a_temperature_anneal_on_a_clamp_gate(env: Any) -> None:
    doc = _clamp(gate_doc())
    doc["method"]["train"]["anneal"] = {"g.theta.temperature": [1.0, 0.01, 0.5]}
    err = expect_rule(4, env, in_order(doc))
    assert GATE_MAPS["clamp"].no_temperature_because in str(err)
    sigmoid = gate_doc()
    sigmoid["method"]["train"]["anneal"] = {"g.theta.temperature": [1.0, 0.01, 0.5]}
    compile_protocol(in_order(sigmoid), env=env)  # the sigmoid gate anneals as before


def test_a_clamp_bundle_is_refused_by_a_document_declaring_no_parametrization(
    tmp_path: Path,
) -> None:
    """The reverse mismatch, said explicitly like `group`'s: a clamp-fitted
    bundle's hard mask is θ > ½, and a sigmoid document would read it at θ > 0."""
    _write_gate_header(
        tmp_path,
        "fit/g.safetensors",
        theta_len=GPT2_HEADS * GPT2_HEAD_DIM,
        site={"component": "block_output", "layers": [3]},
        extra={"parametrization": "clamp"},
    )
    with pytest.raises(ValidationError) as err:
        compile_protocol(
            apply_doc(group=None, component="block_output"), env=build_env(tmp_path)
        )
    assert err.value.rule == 15
    assert "fitted with parametrization 'clamp'" in str(err.value)


def test_an_unstamped_bundle_is_refused_by_a_clamp_document(tmp_path: Path) -> None:
    """A bundle from before the field existed is a sigmoid gate and cannot
    prove otherwise — refused by the missing key, by name."""
    _write_gate_header(
        tmp_path,
        "fit/g.safetensors",
        theta_len=GPT2_HEADS * GPT2_HEAD_DIM,
        site={"component": "block_output", "layers": [3]},
    )
    with pytest.raises(ValidationError) as err:
        compile_protocol(
            _clamp(apply_doc(group=None, component="block_output")),
            env=build_env(tmp_path),
        )
    assert err.value.rule == 15
    assert "missing 'parametrization'" in str(err.value)


def test_a_bundle_reloads_under_the_map_it_was_fitted_with(tmp_path: Path) -> None:
    """Valid work still passes: clamp under clamp, and the `sigmoid` stamp a
    new fit writes under a document that declares nothing."""
    _write_gate_header(
        tmp_path,
        "fit/g.safetensors",
        theta_len=GPT2_HEADS * GPT2_HEAD_DIM,
        site={"component": "block_output", "layers": [3]},
        extra={"parametrization": "clamp"},
    )
    compile_protocol(
        _clamp(apply_doc(group=None, component="block_output")), env=build_env(tmp_path)
    )
    _write_gate_header(
        tmp_path,
        "fit/g.safetensors",
        theta_len=GPT2_HEADS * GPT2_HEAD_DIM,
        site={"component": "block_output", "layers": [3]},
        extra={"parametrization": "sigmoid"},
    )
    compile_protocol(
        apply_doc(group=None, component="block_output"), env=build_env(tmp_path)
    )


# -- §2.5 `init` on a gate ---------------------------------------------------- #


def test_a_gate_fill_enters_the_canonical_form_as_authored(env: Any) -> None:
    """A fill is a value, not an artifact: no digest to hash, nothing to check
    at load, and a different start is a different experiment."""
    doc = gate_doc()
    doc["method"]["featurizers"]["g"]["init"] = {"fill": 0.99}
    canon = canonicalize(doc, env)["method"]["featurizers"]["g"]
    assert canon["init"] == {"fill": 0.99}
    assert "content_digest" not in canon["init"]
    assert digest(canonicalize(doc, env)) != digest(canonicalize(gate_doc(), env))
    assert "init" not in canon_gate(env)  # absent, the midpoint start: no key
    compile_protocol(in_order(doc), env=env)  # and nothing to refuse at load


# -- §2.11 `train.control` in the canonical form ----------------------------- #


def test_a_controllers_defaults_are_materialized_like_an_optimizers(env: Any) -> None:
    """Two spellings of one controller digest alike; a controlled document is
    a different experiment from the same document without the controller."""
    target = "train.objective.sparsity.weight"
    pid = {
        "kind": "pid",
        "signal": {"hard_mask_size": "g"},
        "setpoint": {"ramp": [12, 0, 0.5]},
        "gains": {"kp": 0.1, "ki": 0.001},
    }
    terse = gate_doc()
    terse["method"]["train"]["objective"] = {
        "fit": {"weight": 1.0, **fit_term(terse)},
        "sparsity": {"weight": 0.025, "l1": "g"},
    }
    terse["method"]["train"]["control"] = {target: pid}
    spelled = json.loads(json.dumps(terse))
    spelled["method"]["train"]["control"][target] = {
        **pid,
        "gains": {**pid["gains"], "kd": 0.0},
        "space": "log",
        "bounds": [1e-8, 1e8],
        "d_clip": 5.0,
    }
    canon = canonicalize(terse, env)["method"]["train"]["control"][target]
    assert canon["space"] == "log" and canon["gains"]["kd"] == 0.0
    assert canon["bounds"] == [1e-8, 1e8] and canon["d_clip"] == 5.0
    assert digest(canonicalize(terse, env)) == digest(canonicalize(spelled, env))
    plain = json.loads(json.dumps(terse))
    del plain["method"]["train"]["control"]
    assert digest(canonicalize(plain, env)) != digest(canonicalize(terse, env))
    compile_protocol(in_order(terse), env=env)


def test_a_control_signal_over_one_gate_and_over_a_list_of_one_are_one_document(
    env: Any,
) -> None:
    """The signal's canonical spelling is the list (the `layers` fold, §2.4):
    `"g"` and `["g"]` digest alike, and a list over several trained gates is
    the many-layer DBM's one signal — their kept counts summed (§2.11)."""
    target = "train.objective.sparsity.weight"

    def controlled(signal: Any) -> dict[str, Any]:
        doc = gate_doc()
        doc["method"]["train"]["objective"] = {
            "fit": {"weight": 1.0, **fit_term(doc)},
            "sparsity": {"weight": 0.025, "l1": "g"},
        }
        doc["method"]["train"]["control"] = {
            target: {
                "kind": "pid",
                "signal": {"hard_mask_size": signal},
                "setpoint": {"ramp": [12, 0, 0.5]},
                "gains": {"kp": 0.1, "ki": 0.001},
            }
        }
        return doc

    bare, listed = controlled("g"), controlled(["g"])
    canon = canonicalize(bare, env)["method"]["train"]["control"][target]
    assert canon["signal"] == {"hard_mask_size": ["g"]}
    assert digest(canonicalize(bare, env)) == digest(canonicalize(listed, env))
    compile_protocol(in_order(listed), env=env)


# -- §2.12 `trajectory` under rule 10 ----------------------------------------- #


def _with_trajectory(doc: dict[str, Any], **entry: Any) -> dict[str, Any]:
    doc["method"]["save"].append(
        {
            "kind": "trajectory",
            "every": {"count": 4},
            "file_path": "trajectory.safetensors",
            **entry,
        }
    )
    return in_order(doc)


def test_rule_10_holds_a_trajectory_to_its_shape(env: Any) -> None:
    compile_protocol(_with_trajectory(gate_doc()), env=env)  # a fit, one bundle: fine
    err = expect_rule(
        10, env, _with_trajectory(gate_doc(), file_path="trajectory.json")
    )
    assert "safetensors" in str(err)
    err = expect_rule(10, env, _with_trajectory(apply_doc()))
    assert "needs a train section" in str(err)
    twice = _with_trajectory(gate_doc())
    twice["method"]["save"].append(
        {"kind": "trajectory", "every": {"count": 2}, "file_path": "other.safetensors"}
    )
    err = expect_rule(10, env, in_order(twice))
    assert "saved twice" in str(err)  # a non-value entry's value is its kind


# -- a hard-concrete gate's stretch is part of the identity ------------------ #


def _hard_concrete_apply(stretch: list[float] | None) -> dict[str, Any]:
    doc = apply_doc(group=None, component="block_output")
    gate = doc["method"]["featurizers"]["g"]
    gate["parametrization"] = "hard_concrete"
    if stretch is not None:
        gate["stretch"] = stretch
    return doc


def _hard_concrete_bundle(tmp_path: Path, stretch: list[float] | None) -> None:
    extra: dict[str, Any] = {"parametrization": "hard_concrete"}
    if stretch is not None:
        extra["stretch"] = stretch
    _write_gate_header(
        tmp_path,
        "fit/g.safetensors",
        theta_len=GPT2_HEADS * GPT2_HEAD_DIM,
        site={"component": "block_output", "layers": [3]},
        extra=extra,
    )


def test_an_authored_stretch_is_asked_of_the_bundle(tmp_path: Path) -> None:
    """The hard split is `θ > logit((½−γ)/(ζ−γ))`, so a bundle fitted at one
    stretch is another mask under another: an authored stretch is compared
    (the `group` precedent), a matching one loads, a different one refuses by
    name."""
    _hard_concrete_bundle(tmp_path, [-0.1, 1.5])
    compile_protocol(_hard_concrete_apply([-0.1, 1.5]), env=build_env(tmp_path))
    with pytest.raises(ValidationError) as err:
        compile_protocol(_hard_concrete_apply([-0.2, 1.2]), env=build_env(tmp_path))
    assert err.value.rule == 15 and "'stretch'" in str(err.value)


def test_a_non_default_stretch_bundle_is_refused_by_a_document_authoring_none(
    tmp_path: Path,
) -> None:
    """The reverse mismatch, said like `parametrization`'s: a bundle fitted at
    `[-0.1, 1.5]` splits θ at ≈ −0.51, and a document authoring no stretch
    would read it at the default's 0 — a different mask than the fit reported,
    silently, were the key write-only provenance."""
    _hard_concrete_bundle(tmp_path, [-0.1, 1.5])
    with pytest.raises(ValidationError) as err:
        compile_protocol(_hard_concrete_apply(None), env=build_env(tmp_path))
    assert err.value.rule == 15 and "fitted at stretch" in str(err.value)


def test_a_default_stretch_bundle_loads_under_a_document_authoring_none(
    tmp_path: Path,
) -> None:
    """A fit that authored nothing stamps the default; the apply that authors
    nothing implies the same default — the two thresholds agree, so it loads
    (and so does one that spells the default out)."""
    _hard_concrete_bundle(tmp_path, [-0.1, 1.1])
    compile_protocol(_hard_concrete_apply(None), env=build_env(tmp_path))
    compile_protocol(_hard_concrete_apply([-0.1, 1.1]), env=build_env(tmp_path))


def _position_apply_doc(*, axis: bool) -> dict[str, Any]:
    """`apply_doc` over a fixed three-position window, the loaded gate a
    position gate when ``axis`` is set and a per-coordinate one otherwise."""
    doc = apply_doc(group=None, component="block_output")
    for entry in (doc["method"]["reads"]["v_cf"], doc["method"]["writes"]["patch"]):
        entry["pos"] = {"span": [0, 3]}
    if axis:
        doc["method"]["featurizers"]["g"]["axis"] = "position"
    return in_order(doc)


def test_a_position_gate_bundle_is_refused_by_a_document_without_axis(
    tmp_path: Path,
) -> None:
    """§2.5 ``axis`` at load time, the `group` shape: a bundle fitted over
    positions read by a per-coordinate gate would take its W parameters as W
    coordinates — refused at compile by name (rule 15), not by a width
    mismatch at the build. Before the clause only the build refused it."""
    _write_gate_header(
        tmp_path,
        "fit/g.safetensors",
        theta_len=3,
        site={"component": "block_output", "layers": [3]},
        extra={"axis": "position"},
    )
    with pytest.raises(ValidationError) as err:
        compile_protocol(_position_apply_doc(axis=False), env=build_env(tmp_path))
    assert err.value.rule == 15 and "fitted over 'position'" in str(err.value)


def test_a_coordinate_gate_bundle_is_refused_by_a_position_gate_document(
    tmp_path: Path,
) -> None:
    """The other direction: the document wants a mask over positions and the
    bundle cannot prove it is one — refused by the missing key, by name."""
    _write_gate_header(
        tmp_path,
        "fit/g.safetensors",
        theta_len=GPT2_HEADS * GPT2_HEAD_DIM,
        site={"component": "block_output", "layers": [3]},
    )
    with pytest.raises(ValidationError) as err:
        compile_protocol(_position_apply_doc(axis=True), env=build_env(tmp_path))
    assert err.value.rule == 15 and "missing 'axis'" in str(err.value)


def test_a_position_gate_document_authors_top_k_beside_axis(tmp_path: Path) -> None:
    """§2.5: `top_k` beside `axis: position` is a legal pairing at the
    document level — the parser refuses `group` and `pool` beside `axis`, not
    `top_k` — and both reach the canonical entry (the cut itself, counting
    positions, is `test_featurizers.py`'s)."""
    _write_gate_header(
        tmp_path,
        "fit/g.safetensors",
        theta_len=3,
        site={"component": "block_output", "layers": [3]},
        extra={"axis": "position"},
    )
    doc = apply_doc(group=None, component="block_output")
    for entry in (doc["method"]["reads"]["v_cf"], doc["method"]["writes"]["patch"]):
        entry["pos"] = {"span": [0, 3]}
    doc["method"]["featurizers"]["g"]["axis"] = "position"
    doc["method"]["featurizers"]["g"]["top_k"] = 2
    loaded = compile_protocol(in_order(doc), env=build_env(tmp_path))
    entry = loaded.canonical["method"]["featurizers"]["g"]
    assert entry["axis"] == "position" and entry["top_k"] == 2


def test_a_position_gate_bundle_loads_under_a_document_that_spells_axis(
    tmp_path: Path,
) -> None:
    _write_gate_header(
        tmp_path,
        "fit/g.safetensors",
        theta_len=3,
        site={"component": "block_output", "layers": [3]},
        extra={"axis": "position"},
    )
    loaded = compile_protocol(_position_apply_doc(axis=True), env=build_env(tmp_path))
    entry = loaded.canonical["method"]["featurizers"]["g"]
    assert entry["axis"] == "position" and "content_digest" in entry
