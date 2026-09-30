"""Canonicalization: the canonical-stamp principle (spec §7)."""

from __future__ import annotations

import copy
import json

import pytest

from causalab.protocol.schema.explicit import canonical_bytes, canonicalize, digest
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.rules.document import validate_document
from causalab.protocol.schema import inline_train_saves, parse_document

from tests.protocol._env import CORPUS_DIR, steps_of
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.tables import inline_rows, table_bytes
from tests.protocol._docs import (
    UNWRITTEN,
    aggregation,
    base_doc,
    base_only_doc,
    in_order,
    inline_doc,
    saved,
    term,
)

pytestmark = pytest.mark.unit


def test_pos_sugar_and_alias_canonicalize(env):
    raw = base_doc()
    canonical = canonicalize(raw, env)
    assert canonical["method"]["reads"]["v_cf"]["pos"] == {"index": -1}
    # dtype materializes like every other default: an authored document may
    # be silent about precision, a canonical one never is (§2.1)
    assert canonical["model"] == {"key": "gpt2", "revision": "main", "dtype": "fp32"}


def test_all_pos_sugar_canonicalizes(env):
    """Both spellings land on one canonical form, so a document authored
    either way digests identically."""
    sugar, explicit = base_doc(), base_doc()
    sugar["method"]["reads"]["v_cf"]["pos"] = "all"
    explicit["method"]["reads"]["v_cf"]["pos"] = {"all": True}
    assert canonicalize(sugar, env)["method"]["reads"]["v_cf"]["pos"] == {"all": True}
    assert digest(canonicalize(sugar, env)) == digest(canonicalize(explicit, env))


def test_all_positions_changes_the_digest(env):
    """The position is part of the address, so it is part of the record."""
    raw = base_doc()
    raw["method"]["reads"]["v_cf"]["pos"] = "all"
    assert digest(canonicalize(raw, env)) != digest(canonicalize(base_doc(), env))


def test_dataset_digest_stamped(env):
    canonical = canonicalize(base_doc(), env)
    stamped = canonical["data"]["base"]["digest"]
    assert stamped == env.datasets.digest("weekdays/data#train")
    assert len(stamped) == 64


def test_a_role_less_data_block_digests_as_base(env):
    """The short and the explicit spelling of a single-input document are one
    document (§2.2, §7): same canonical ``data``, same digest, and the short
    one passes the document rules as the explicit one does."""
    short, explicit = base_only_doc(), base_only_doc()
    short["data"] = explicit["data"]["base"]
    validate_document(parse_document(short), model_info=env.model_info)
    assert canonicalize(short, env)["data"] == canonicalize(explicit, env)["data"]
    assert "base" in canonicalize(short, env)["data"]
    assert digest(canonicalize(short, env)) == digest(canonicalize(explicit, env))


def test_an_inline_role_canonicalizes_with_its_derived_fields(env):
    """§2.2, §7: the authored prompts stay, and the column and the ref every
    reader of a canonical role expects materialize next to the digest."""
    prompts = ["The Space Needle is located in"]
    role = canonicalize(inline_doc(*prompts), env)["data"]["base"]
    assert role["inputs"] == prompts
    assert role["field"] == "input"
    assert role["dataset"].startswith("inline:")
    assert role["digest"] == role["dataset"][len("inline:") :]
    assert len(role["digest"]) == 64


def test_an_inline_table_and_a_file_table_with_equal_rows_are_one_dataset(
    env, tmp_path
):
    """The digest is over the rows, never the ref's name (§2.2, §7,
    ``_data_identity``): a document that inlines its prompts and one that
    reads them from a file of the same rows stamp the same digest, so they
    intern into one forward group."""
    prompts = ["The Space Needle is located in", "The Eiffel Tower is located in"]
    root = tmp_path / "data"
    (root / "spots").mkdir(parents=True)
    (root / "spots" / "data.json").write_bytes(table_bytes(inline_rows(prompts)))
    file_env = ResolutionEnv(
        datasets=FileDatasets(root=root), artifacts=FileArtifacts(root=tmp_path)
    )
    inline = canonicalize(inline_doc(*prompts), env)["data"]["base"]
    on_disk = inline_doc(*prompts)
    on_disk["data"] = {"dataset": "spots/data", "field": "input"}
    from_file = canonicalize(on_disk, file_env)["data"]["base"]
    assert inline["digest"] == from_file["digest"]


def test_an_inline_document_compiles_with_its_data_checked(env):
    """The demo shape end to end: ``validate --data`` resolves the inline
    rows, checks the field against their columns, and the compiled protocol
    records the table under its ref."""
    compiled = compile_protocol(inline_doc(), env=env, data=True)
    (ref,) = compiled.data
    assert ref.startswith("inline:")
    assert compiled.data[ref].columns == ("input", "split")
    assert compiled.data[ref].digest == ref[len("inline:") :]


def test_im_write_lists_sorted(env):
    raw = base_doc()
    raw["method"]["writes"]["another"] = {
        "site": "tgt",
        "pos": -1,
        "do": {"add_scaled": {"op": "v_cf", "alpha": 1.0}},
    }
    raw["method"]["intervened_models"]["patched"]["writes"] = [
        "patch",
        "another",
    ]
    canonical = canonicalize(raw, env)
    assert canonical["method"]["intervened_models"]["patched"]["writes"] == [
        "another",
        "patch",
    ]


def test_writes_during_generation_is_canonical_only_when_true(env):
    """§2.9: an authored ``false`` and the absent field are one intervention
    (prefill-only) and digest identically; ``true`` enters the canonical form
    and changes the digest, as a different intervention must."""
    raw = base_doc()
    raw["method"]["positions"] = {
        "tail": {"generated": {"max_new_tokens": 8}, "index": -1}
    }
    raw["method"]["reads"]["logits"]["pos"] = "tail"
    raw["method"]["writes"]["patch"]["do"] = {"swap": 0.0}
    del raw["method"]["reads"]["v_cf"]
    absent = canonicalize(in_order(raw), env)
    assert (
        "writes_during_generation"
        not in absent["method"]["intervened_models"]["patched"]
    )

    off = copy.deepcopy(raw)
    off["method"]["intervened_models"]["patched"]["writes_during_generation"] = False
    assert digest(canonicalize(in_order(off), env)) == digest(absent)

    on = copy.deepcopy(raw)
    on["method"]["intervened_models"]["patched"]["writes_during_generation"] = True
    canonical = canonicalize(in_order(on), env)
    assert canonical["method"]["intervened_models"]["patched"] == {
        "input": "base",
        "reads": ["logits"],
        "writes": ["patch"],
        "writes_during_generation": True,
    }
    assert digest(canonical) != digest(absent)


def test_train_defaults_materialized(env):
    loaded = compile_protocol(CORPUS_DIR / "04_das_im.json", env=env)
    train = loaded.canonical["method"]["train"]
    assert train["optimizer"]["betas"] == [0.9, 0.999]
    assert train["optimizer"]["eps"] == 1e-8
    assert train["optimizer"]["schedule"] == "constant"
    assert train["precision"] == {"feature": "fp32", "loss": "fp32"}
    # the model's own precision has one home, and it is the model section
    assert loaded.canonical["model"]["dtype"] == "bf16"
    assert "digest" in train["eval"]  # eval.split is a dataset ref too


def test_a_linear_anneal_has_one_canonical_spelling(env):
    """§2.11 / sec. 7: `{"from", "to", "frac"}` with an unauthored or `linear`
    shape canonicalizes to `[from, to, frac]`, so the mapping form moves no
    digest a list-spelled document already has; a geometric schedule keeps
    the mapping with its shape spelled, since the list has no place for it."""
    raw = json.loads((CORPUS_DIR / "05_dbm_im.json").read_text())
    train = raw["method"]["train"]
    assert "anneal" in train, "the DBM corpus document anneals its temperature"
    (target,) = train["anneal"]
    listed = digest(canonicalize(raw, env))
    mapped = copy.deepcopy(raw)
    start, end, frac = train["anneal"][target]
    mapped["method"]["train"]["anneal"][target] = {
        "from": start,
        "to": end,
        "frac": frac,
    }
    assert digest(canonicalize(mapped, env)) == listed
    mapped["method"]["train"]["anneal"][target]["shape"] = "linear"
    assert digest(canonicalize(mapped, env)) == listed
    mapped["method"]["train"]["anneal"][target]["shape"] = "geometric"
    canonical = canonicalize(mapped, env)
    assert canonical["method"]["train"]["anneal"][target] == {
        "from": start,
        "to": end,
        "frac": frac,
        "shape": "geometric",
    }
    assert digest(canonical) != listed


def test_featurizer_widths_derived(env):
    loaded = compile_protocol(CORPUS_DIR / "04_das_im.json", env=env)
    rot = loaded.canonical["method"]["featurizers"]["rot"]
    assert rot["width"] == 4096
    assert rot["params"] == {"weight": [4096, 8]}
    assert rot["dtype"] == "fp32"


def test_gate_width_derived(env):
    loaded = compile_protocol(CORPUS_DIR / "05_dbm_im.json", env=env)
    gate = loaded.canonical["method"]["featurizers"]["gate"]
    assert gate["params"] == {"theta": [4096]}


def test_loaded_featurizer_hashed_not_shaped(env):
    loaded = compile_protocol(CORPUS_DIR / "09_das_apply_im.json", env=env)
    rot = loaded.canonical["method"]["featurizers"]["rot"]
    assert "content_digest" in rot and len(rot["content_digest"]) == 64
    assert "params" not in rot  # loaded bundles are identified by their bytes


def _loaded_rotation_at(components: tuple[str, ...]) -> dict:
    """The apply document's *loaded* rotation named from one layer-18 site per
    component — one name at several sites (§2.5)."""
    raw = json.loads((CORPUS_DIR / "09_das_apply_im.json").read_text())
    method = raw["method"]
    method["sites"]["target"]["component"] = components[0]
    for i, component in enumerate(components[1:], start=2):
        site = f"target{i}"
        method["sites"][site] = {"component": component, "layers": [18]}
        method["reads"][f"v_cf{i}"] = {"site": site, "pos": -1, "featurizer": "rot"}
        method["intervened_models"][UNWRITTEN]["reads"].append(f"v_cf{i}")
        method["writes"][f"patch{i}"] = {
            "site": site,
            "pos": -1,
            "featurizer": "rot",
            "do": {"swap": f"v_cf{i}"},
        }
        method["intervened_models"]["patched"]["writes"].append(f"patch{i}")
    return in_order(raw)


def test_a_loaded_featurizer_at_two_sites_is_held_to_one_width_at_load(env):
    """§2.5, one name at several sites, on a *loaded* featurizer: rule 4's
    one-width check ran on the fitted path only, so a loaded rotation named
    from a 4096-wide and a 12288-wide site reached the build and was refused
    there, after its weights were read. It is refused at load now — while its
    canonical form still records its bytes and no width (one description of a
    loaded bundle), and two sites of one width are the tie they always were."""
    same = canonicalize(_loaded_rotation_at(("block_output", "attention_output")), env)
    rot = same["method"]["featurizers"]["rot"]
    assert "content_digest" in rot and "width" not in rot and "params" not in rot
    with pytest.raises(ValidationError) as err:
        canonicalize(_loaded_rotation_at(("block_output", "mlp_activation")), env)
    assert err.value.rule == 4
    assert "one featurizer, one width" in str(err.value)
    assert "4096" in str(err.value) and "12288" in str(err.value)
    assert err.value.path == "featurizers.rot"


def test_swept_document_keeps_wrappers(env):
    loaded = compile_protocol(CORPUS_DIR / "08_weekdays_das_sweep_im.json", env=env)
    doc_form = loaded.canonical
    assert doc_form["method"]["featurizers"]["rot"]["k"] == {"sweep": [8, 16, 32]}
    point_form = steps_of(loaded, env).canonical[0]
    assert point_form["method"]["featurizers"]["rot"]["k"] == 8
    assert point_form["method"]["featurizers"]["rot"]["params"] == {"weight": [4096, 8]}


def test_canonical_bytes_are_sorted_and_minimal(env):
    canonical = canonicalize(base_doc(), env)
    blob = canonical_bytes(canonical).decode()
    assert ": " not in blob and ", " not in blob
    assert blob.index('"data"') < blob.index('"model"')  # sorted keys


def test_out_of_range_layer_refused(env):
    """Rule 4's address half is the checklist's since the rules package: the
    canonicalizer keeps the folds and returns; ``validate_document`` refuses,
    with the environment's static model metadata."""
    raw = base_doc()
    raw["method"]["sites"]["tgt"]["layers"] = 40  # gpt2 has 12 layers
    with pytest.raises(Exception) as err:
        validate_document(parse_document(raw), model_info=env.model_info)
    assert "[V4]" in str(err.value)
    assert canonicalize(raw, env)["method"]["sites"]["tgt"]["layers"] == [40]


MATCH = aggregation("match", expected="label")


def test_match_mode_default_materialized(env):
    """Optional metric fields are materialized like ``train.optimizer``
    defaults (§2.10): the two spellings of "exact" are one canonical form, so
    adding the field cannot split the digest of documents that omit it."""
    omitted = base_doc()
    omitted["method"]["save"].append(saved("logits", "patched", "m.json", MATCH))
    spelled = copy.deepcopy(omitted)
    spelled["method"]["save"][1]["aggregation"]["mode"] = "exact"
    assert (
        canonicalize(omitted, env)["method"]["save"][1]["aggregation"]["mode"]
        == "exact"
    )
    assert digest(canonicalize(omitted, env)) == digest(canonicalize(spelled, env))


def test_first_token_mode_is_a_different_document(env):
    """...and a real semantic choice still moves the digest."""
    exact = base_doc()
    exact["method"]["save"].append(saved("logits", "patched", "m.json", MATCH))
    first = copy.deepcopy(exact)
    first["method"]["save"][1]["aggregation"]["mode"] = "first_token"
    assert digest(canonicalize(exact, env)) != digest(canonicalize(first, env))


def test_column_position_canonicalizes_verbatim(env):
    """A column position is data the canonical form carries as authored — no
    derivation, so the digest names the column the document reads."""
    raw = base_doc()
    raw["method"]["positions"] = {"subj": {"column": "entity"}}
    raw["method"]["reads"]["v_cf"]["pos"] = "subj"
    canonical = canonicalize(in_order(raw), env)
    assert canonical["method"]["positions"]["subj"] == {"column": "entity"}


def test_generated_frame_canonicalizes_verbatim(env):
    """The frame selector is authored data with no defaults to materialize,
    so it carries through untouched — and a different budget is a different
    document, because the position enters the read's closure."""
    raw = base_doc()
    raw["method"]["positions"] = {
        "tail": {"generated": {"max_new_tokens": 8}, "index": -1}
    }
    raw["method"]["reads"]["v_cf"]["pos"] = "tail"
    canonical = canonicalize(in_order(raw), env)
    assert canonical["method"]["positions"]["tail"] == {
        "generated": {"max_new_tokens": 8},
        "index": -1,
    }
    # deep-copied: canonicalize passes this section through by reference, so
    # mutating a shared nested dict would rewrite the form just measured
    longer = copy.deepcopy(in_order(raw))
    longer["method"]["positions"]["tail"]["generated"]["max_new_tokens"] = 16
    assert digest(canonical) != digest(canonicalize(longer, env))


def test_prompt_frame_documents_digest_unchanged(env):
    """The new field is absent, not defaulted, on every position that does
    not ask for a continuation — the reason no existing digest moves."""
    canonical = canonicalize(base_doc(), env)
    assert "generated" not in canonical["method"]["reads"]["v_cf"]["pos"]


def test_subspace_init_hashes_the_basis(env):
    """The basis a fit starts from is part of the experiment: its bytes enter
    the canonical form as ``init.content_digest``, the way a loaded
    featurizer's do (§7) — while the featurizer stays a *fit*, so its param
    shape is still derived. A missing basis is a load error, never a default."""
    from causalab.protocol.rules.errors import ValidationError

    from tests.protocol._env import PCA_FIXTURE_RELPATH

    raw = base_doc()
    raw["method"]["featurizers"] = {
        "rot": {
            "kind": "subspace",
            "k": 2,
            "parametrization": "cayley",
            "init": {"file_path": PCA_FIXTURE_RELPATH},
        }
    }
    raw["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    raw["method"]["writes"]["patch"]["featurizer"] = "rot"
    raw = in_order(raw)
    canonical = canonicalize(raw, env)
    rot = canonical["method"]["featurizers"]["rot"]
    assert rot["init"]["file_path"] == PCA_FIXTURE_RELPATH
    assert len(rot["init"]["content_digest"]) == 64
    assert rot["params"] == {"weight": [768, 2]}  # gpt2's width: still a fit
    assert canonicalize(copy.deepcopy(raw), env) == canonical  # stable
    # the authored tree is left alone — canonicalization copies
    assert raw["method"]["featurizers"]["rot"]["init"] == {
        "file_path": PCA_FIXTURE_RELPATH
    }
    raw["method"]["featurizers"]["rot"]["init"]["file_path"] = (
        "nowhere/basis.safetensors"
    )
    with pytest.raises(ValidationError) as err:
        canonicalize(raw, env)
    assert err.value.rule == 15


# §2.5 gate `group: head` — one parameter per head, derived offline ---------- #

A3B = "Qwen/Qwen3.6-35B-A3B"


def _head_gate_doc(component: str, layer: int, chain=("gate",)) -> dict:
    """A DBM-shaped document on the registered Qwen3.6 entry, whose two
    head-major components have *different* head layouts (16 × 256 query heads,
    32 × 128 value heads) — so the derived count is checkable per family."""
    raw = base_doc()
    raw["model"] = {"key": A3B, "revision": "main", "dtype": "bf16"}
    raw["method"]["sites"]["tgt"] = {"component": component, "layers": [layer]}
    raw["method"]["featurizers"] = {
        "gate": {"kind": "gate", "group": "head"},
        "rot": {"kind": "subspace", "k": 64, "parametrization": "cayley"},
    }
    ref = list(chain) if len(chain) > 1 else chain[0]
    raw["method"]["reads"]["v_cf"]["featurizer"] = ref
    raw["method"]["writes"]["patch"]["featurizer"] = ref
    return in_order(raw)


def test_a_head_grouped_gate_has_one_theta_per_query_head(env):
    gate = canonicalize(_head_gate_doc("attention_premix", 19), env)["method"][
        "featurizers"
    ]["gate"]
    assert gate["width"] == 16 * 256
    assert gate["params"] == {"theta": [16]}


def test_a_head_grouped_gate_on_deltanet_counts_value_heads(env):
    """``delta_premix`` is the out-projection's input in value-head space, so
    the same declaration masks 32 value heads there — one code path, two
    families, two counts."""
    gate = canonicalize(_head_gate_doc("delta_premix", 18), env)["method"][
        "featurizers"
    ]["gate"]
    assert gate["width"] == 32 * 128
    assert gate["params"] == {"theta": [32]}


def test_a_head_grouped_gate_on_a_headless_component_is_refused_by_name(env):
    with pytest.raises(ValidationError) as err:
        canonicalize(_head_gate_doc("block_output", 19), env)
    assert err.value.rule == 23
    assert "'block_output'" in str(err.value) and "no head axis" in str(err.value)
    assert err.value.path == "featurizers.gate.group"


def test_a_head_grouped_gate_after_a_rotation_is_refused(env):
    """After ``rot`` the coordinates are a rotated basis; "head" names nothing
    there, so the grouping is refused rather than tiling 64 rotated columns
    into 256-wide "heads"."""
    with pytest.raises(ValidationError) as err:
        canonicalize(_head_gate_doc("attention_premix", 19, ("rot", "gate")), env)
    assert err.value.rule == 23
    assert "must be the first stage of its chain" in str(err.value)


# §2.5 gate `group: expert_neuron` — the whole expert table, derived offline -- #


def _expert_gate_doc(component: str, layer: int = 19) -> dict:
    """The head-gate document with the gate re-keyed by expert neuron on a
    MoE component; Qwen3.6 routes over 256 experts of d_expert 512, top-8."""
    raw = _head_gate_doc(component, layer)
    raw["method"]["featurizers"]["gate"]["group"] = "expert_neuron"
    return raw


def test_an_expert_keyed_gate_has_the_whole_expert_table(env):
    gate = canonicalize(_expert_gate_doc("expert_activation"), env)["method"][
        "featurizers"
    ]["gate"]
    assert gate["group"] == "expert_neuron"
    assert gate["width"] == 8 * 512  # the site: top-8 slots of d_expert 512
    assert gate["params"] == {"theta": [256, 512]}  # the table: every expert


@pytest.mark.parametrize(
    "component", ["shared_expert_activation", "expert_output", "block_output"]
)
def test_an_expert_keyed_gate_off_expert_activation_is_refused_by_name(env, component):
    with pytest.raises(ValidationError) as err:
        canonicalize(_expert_gate_doc(component), env)
    assert err.value.rule == 23
    assert f"'{component}'" in str(err.value) and "expert_neuron" in str(err.value)
    assert err.value.path == "featurizers.gate.group"


def test_the_ungrouped_gate_canonical_form_is_untouched(env):
    """No ``group`` key appears in an ungrouped gate's canonical form, so no
    existing document's digest moves (the corpus pins say the same)."""
    raw = _head_gate_doc("attention_premix", 19)
    del raw["method"]["featurizers"]["gate"]["group"]
    gate = canonicalize(raw, env)["method"]["featurizers"]["gate"]
    assert "group" not in gate
    assert gate["params"] == {"theta": [16 * 256]}


# --------------------------------------------------------------------------- #
# train.objective regularizer lists (§2.11): a set, canonicalized as one
# --------------------------------------------------------------------------- #


#: The corpus DBM document's fit term (§2.11): the cross-entropy of the
#: masked model's logits against the label, the aggregation its `ce.json`
#: save entry tabulates.
CE = term(
    "logits",
    "masked",
    aggregation("cross_entropy", target="label"),
)


def _two_gate_dbm(env, objective):
    """The corpus DBM fit with a second gate at the layer below."""
    raw = copy.deepcopy(
        dict(compile_protocol(CORPUS_DIR / "05_dbm_im.json", env=env).tree)
    )
    raw["method"]["sites"]["below"] = dict(raw["method"]["sites"]["target"])
    raw["method"]["sites"]["below"]["layers"] = [
        raw["method"]["sites"]["target"]["layers"][0] - 1
    ]
    raw["method"]["featurizers"]["gate_below"] = {"kind": "gate"}
    raw["method"]["reads"]["v_below"] = dict(
        raw["method"]["reads"]["v_cf"], site="below", featurizer="gate_below"
    )
    # the new read is taken where `v_cf` is (§2.9)
    for model in raw["method"]["intervened_models"].values():
        if "v_cf" in model["reads"]:
            model["reads"] = [*model["reads"], "v_below"]
    raw["method"]["writes"]["mask_below"] = dict(
        raw["method"]["writes"]["mask"],
        site="below",
        featurizer="gate_below",
        do={"swap": "v_below"},
    )
    raw["method"]["intervened_models"]["masked"]["writes"] = ["mask", "mask_below"]
    raw["method"]["train"]["params"] = ["gate", "gate_below"]
    # the corpus saves name its objective terms (§2.12); spelled inline, the
    # saves hold whatever `objective` this test puts in their place
    raw["method"]["save"] = inline_train_saves(raw["method"])
    raw["method"]["train"]["objective"] = objective
    raw["method"]["save"].append(
        {"value": "gate_below", "site": "below", "file_path": "gate_below.safetensors"}
    )
    return raw


def test_a_regularizer_reduce_passes_through_unmaterialized(env):
    """§2.11 ``reduce``: authored, it is in the canonical form as written, the
    names beside it still sorted; absent, nothing is added — so no document
    without it moves its digest."""
    plain = _two_gate_dbm(env, [[1.0, CE], [0.01, {"l1": ["gate_below", "gate"]}]])
    canonical = canonicalize(plain, env)["method"]["train"]["objective"]
    assert "reduce" not in canonical[1][1]
    summed = _two_gate_dbm(
        env, [[1.0, CE], [0.01, {"l1": ["gate_below", "gate"], "reduce": "sum"}]]
    )
    canonical = canonicalize(summed, env)["method"]["train"]["objective"]
    assert canonical[1][1] == {"l1": ["gate", "gate_below"], "reduce": "sum"}
    assert digest(canonicalize(summed, env)) != digest(canonicalize(plain, env))


def test_a_regularizer_list_is_sorted(env):
    raw = _two_gate_dbm(env, [[1.0, CE], [0.01, {"l1": ["gate_below", "gate"]}]])
    canonical = canonicalize(raw, env)
    assert canonical["method"]["train"]["objective"][1] == [
        0.01,
        {"l1": ["gate", "gate_below"]},
    ]
    other = _two_gate_dbm(env, [[1.0, CE], [0.01, {"l1": ["gate", "gate_below"]}]])
    assert digest(canonicalize(other, env)) == digest(canonical)


def test_a_one_name_list_collapses_to_the_name(env):
    """``{"l1": ["gate"]}`` is ``{"l1": "gate"}``: the corpus DBM document's
    digest is the same whichever way its one gate is written."""
    listed = copy.deepcopy(
        dict(compile_protocol(CORPUS_DIR / "05_dbm_im.json", env=env).tree)
    )
    objective = listed["method"]["train"]["objective"]
    assert objective["l1"] == {"weight": 0.01, "l1": "gate"}
    objective["l1"]["l1"] = ["gate"]
    plain = compile_protocol(CORPUS_DIR / "05_dbm_im.json", env=env)
    assert canonicalize(listed, env)["method"]["train"]["objective"]["l1"] == {
        "weight": 0.01,
        "l1": "gate",
    }
    assert digest(canonicalize(listed, env)) == plain.digests.document


def test_the_named_form_keeps_its_names_and_sorts_its_lists(env):
    raw = _two_gate_dbm(
        env,
        {
            "fit": {"weight": 1.0, **CE},
            "sparsity": {
                "weight": {"sweep": [0.01, 0.1]},
                "l1": ["gate_below", "gate"],
            },
        },
    )
    loaded = compile_protocol(raw, env=env)
    assert loaded.canonical["method"]["train"]["objective"] == {
        "fit": {"weight": 1.0, **CE},
        "sparsity": {"weight": {"sweep": [0.01, 0.1]}, "l1": ["gate", "gate_below"]},
    }
    assert [
        p["method"]["train"]["objective"]["sparsity"]["weight"]
        for p in steps_of(loaded, env).canonical
    ] == [
        0.01,
        0.1,
    ]
    assert len(set(steps_of(loaded, env).digests)) == 2


# --------------------------------------------------------------------------- #
# §2.2 `shuffle` is in the canonical form exactly when authored (§7)
# --------------------------------------------------------------------------- #

CORPUS_PINS = json.loads(
    (CORPUS_DIR.parent / "protocol" / "corpus_digests.json").read_text()
)


def _corpus_02():
    return json.loads((CORPUS_DIR / "02_interchange_im.json").read_text())


def _with_shuffle(raw, seed):
    raw = copy.deepcopy(raw)
    raw["data"]["counterfactual"]["shuffle"] = {"seed": seed}
    return raw


def test_an_unshuffled_document_keeps_its_canonical_bytes_and_digest(env):
    """Corpus 02 without ``shuffle`` canonicalizes to its committed pin — the
    verb adds nothing when unauthored, so no shipped digest moved with it. The
    corpus pins prove this for every document; this names the mechanism."""
    loaded = compile_protocol(_corpus_02(), env=env)
    pin = CORPUS_PINS["02_interchange_im.json"]
    assert loaded.digests.document == pin["document"]
    assert list(steps_of(loaded, env).digests) == pin["points"]
    for role in loaded.canonical["data"].values():
        assert "shuffle" not in role
    assert digest(loaded.canonical) == pin["document"]
    assert canonical_bytes(loaded.canonical)  # the bytes the digest is of


def test_an_authored_shuffle_passes_through_and_moves_the_digest(env):
    """The same document with ``shuffle: {seed: 0}`` is a different value:
    ``shuffle`` sits on the counterfactual role's canonical entry beside the
    stamped digest, the document and point digests both move, and the base
    role's entry is byte-identical. The mutation that drops ``shuffle`` from
    the canonical form fails here and in the two-seeds test below."""
    plain = compile_protocol(_corpus_02(), env=env)
    shuffled = compile_protocol(_with_shuffle(_corpus_02(), 0), env=env)
    cf = shuffled.canonical["data"]["counterfactual"]
    assert cf["shuffle"] == {"seed": 0}
    assert {k: v for k, v in cf.items() if k != "shuffle"} == (
        plain.canonical["data"]["counterfactual"]
    )
    assert shuffled.canonical["data"]["base"] == (plain.canonical["data"]["base"])
    assert shuffled.digests.document != plain.digests.document
    assert steps_of(shuffled, env).digests != steps_of(plain, env).digests


def test_two_seeds_are_two_digests(env):
    zero = compile_protocol(_with_shuffle(_corpus_02(), 0), env=env)
    one = compile_protocol(_with_shuffle(_corpus_02(), 1), env=env)
    assert zero.digests.document != one.digests.document
    assert steps_of(zero, env).digests != steps_of(one, env).digests
    assert zero.canonical["data"]["counterfactual"]["shuffle"] == {"seed": 0}
    assert one.canonical["data"]["counterfactual"]["shuffle"] == {"seed": 1}


def _with_draw(raw, eval_member=None):
    raw = copy.deepcopy(raw)
    role = raw["data"]["counterfactual"]
    role["field"] = "counterfactual_inputs"  # the bare list column
    role["draw"] = (
        {"kind": "uniform"}
        if eval_member is None
        else {"kind": "uniform", "eval": eval_member}
    )
    return raw


def test_an_undrawn_document_keeps_its_canonical_bytes_and_digest(env):
    """§7: ``draw`` adds nothing when unauthored — corpus 02 keeps its
    committed pin and no role's canonical entry holds the block (``shuffle``'s
    mechanism, one verb over)."""
    loaded = compile_protocol(_corpus_02(), env=env)
    assert loaded.digests.document == CORPUS_PINS["02_interchange_im.json"]["document"]
    for role in loaded.canonical["data"].values():
        assert "draw" not in role


def test_an_authored_draw_passes_through_and_moves_the_digest(env):
    """The same document with a bare ``field`` and ``draw: {kind: uniform}``
    is a different value: ``draw`` sits on the counterfactual role's canonical
    entry beside the stamped digest with the field as the bare column, the
    document and point digests both move, and the base role's entry is
    byte-identical. The mutation that drops ``draw`` from the canonical form
    fails here; the one that materializes ``eval`` fails on the last lines —
    ``draw`` with an explicit ``eval: 0`` is a different value from ``draw``
    without it (§7: only when authored, down to the nested key)."""
    plain = compile_protocol(_corpus_02(), env=env)
    drawn = compile_protocol(_with_draw(_corpus_02()), env=env)
    plain_cf = plain.canonical["data"]["counterfactual"]
    cf = drawn.canonical["data"]["counterfactual"]
    assert cf["draw"] == {"kind": "uniform"}
    assert cf["field"] == "counterfactual_inputs"
    assert plain_cf["field"] == "counterfactual_inputs[0]"
    assert {k: v for k, v in cf.items() if k not in ("draw", "field")} == {
        k: v for k, v in plain_cf.items() if k != "field"
    }
    assert drawn.canonical["data"]["base"] == (plain.canonical["data"]["base"])
    assert drawn.digests.document != plain.digests.document
    assert steps_of(drawn, env).digests != steps_of(plain, env).digests
    explicit = compile_protocol(_with_draw(_corpus_02(), 0), env=env)
    assert explicit.canonical["data"]["counterfactual"]["draw"] == {
        "kind": "uniform",
        "eval": 0,
    }
    assert explicit.digests.document != drawn.digests.document


def _position_gate_doc(
    second_component: str | None = None, *, loaded: bool = False
) -> dict:
    """A position gate over `{"span": [0, 3]}` on Qwen3.6's `attention_premix`
    (4096 wide), and — when given — named from a second site on another
    component too (§2.5 "one name, several sites"); ``loaded`` gives it a
    `file_path` (canonicalize hashes the bytes, it does not read them)."""
    raw = base_doc()
    raw["model"] = {"key": A3B, "revision": "main", "dtype": "bf16"}
    method = raw["method"]
    method["sites"]["tgt"] = {"component": "attention_premix", "layers": [19]}
    method["featurizers"] = {
        "pg": {
            "kind": "gate",
            "axis": "position",
            **({"file_path": "fit/pg.safetensors"} if loaded else {}),
        }
    }
    method["reads"]["v_cf"]["pos"] = {"span": [0, 3]}
    method["reads"]["v_cf"]["featurizer"] = "pg"
    method["writes"]["patch"]["pos"] = {"span": [0, 3]}
    method["writes"]["patch"]["featurizer"] = "pg"
    if second_component is not None:
        method["sites"]["tgt2"] = {"component": second_component, "layers": [23]}
        name = "pg"
        method["reads"]["v2"] = {
            **method["reads"]["v_cf"],
            "site": "tgt2",
            "featurizer": name,
        }
        method["intervened_models"][UNWRITTEN]["reads"].append("v2")
        method["writes"]["patch2"] = {
            **method["writes"]["patch"],
            "site": "tgt2",
            "featurizer": name,
            "do": {"swap": "v2"},
        }
        method["intervened_models"]["patched"]["writes"] = ["patch", "patch2"]
    return in_order(raw)


def test_a_position_gates_canonical_width_and_params_are_the_window(env):
    """§2.5 ``axis``: θ is one entry per addressed position, so the canonical
    form records `width` and `params.theta` as the window length — `[3]`, not
    the site's 4096 — the invariant `group` established (the canonical form
    states the real tensor shape). Before the change it recorded `[4096]`."""
    gate = canonicalize(_position_gate_doc(), env)["method"]["featurizers"]["pg"]
    assert gate["width"] == 3 and gate["params"] == {"theta": [3]}


def test_a_grouped_feature_gate_composes_with_a_position_gate_when_first(env):
    """Positions ⊗ heads: `["gate", "pg"]` with `gate` grouped by
    head canonicalizes — the grouped gate first (rule 23), sized by the site
    (16 query heads of 256), the position gate by the window (3). The other
    order is refused by rule 23, which holds a grouped gate to the head of its
    chain; the `axis` bullet says which order to write."""
    raw = _position_gate_doc()
    raw["method"]["featurizers"]["gate"] = {"kind": "gate", "group": "head"}
    raw["method"]["reads"]["v_cf"]["featurizer"] = ["gate", "pg"]
    raw["method"]["writes"]["patch"]["featurizer"] = ["gate", "pg"]
    featurizers = canonicalize(raw, env)["method"]["featurizers"]
    assert featurizers["gate"]["width"] == 16 * 256
    assert featurizers["gate"]["params"] == {"theta": [16]}
    assert featurizers["pg"]["width"] == 3 and featurizers["pg"]["params"] == {
        "theta": [3]
    }
    raw["method"]["reads"]["v_cf"]["featurizer"] = ["pg", "gate"]
    raw["method"]["writes"]["patch"]["featurizer"] = ["pg", "gate"]
    with pytest.raises(ValidationError) as err:
        canonicalize(raw, env)
    assert err.value.rule == 23 and "must be the first stage of its chain" in str(
        err.value
    )


def test_a_position_gate_may_span_sites_of_different_feature_widths(env):
    """θ is the window either way, so one position gate at a 4096-wide and a
    2048-wide site is one gate — accepted offline as `build_stack` accepts it,
    where the feature-width rule would have refused a legal document."""
    featurizers = canonicalize(_position_gate_doc("block_output"), env)["method"][
        "featurizers"
    ]
    assert featurizers["pg"]["width"] == 3 and featurizers["pg"]["params"] == {
        "theta": [3]
    }


def test_a_loaded_position_gate_is_held_to_one_window_at_load(tmp_path):
    """The loaded path of §2.5 "one name, several sites" for a position gate:
    its one width is one *window*, so a loaded position gate named from a
    4096-wide and a 2048-wide site over one `[0, 3)` canonicalizes (bytes, no
    width — a loaded bundle's one description), and over two windows it is
    refused at load under rule 4, not at the build after its θ is read."""
    from tests.protocol._env import build_env

    bundle = tmp_path / "fit" / "pg.safetensors"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"canonicalize hashes bytes, it does not read them")
    raw = _position_gate_doc("block_output", loaded=True)
    gate = canonicalize(raw, build_env(tmp_path))["method"]["featurizers"]["pg"]
    assert "content_digest" in gate and "width" not in gate and "params" not in gate
    raw["method"]["writes"]["patch2"]["pos"] = {"span": [0, 4]}
    raw["method"]["reads"]["v2"]["pos"] = {"span": [0, 4]}
    with pytest.raises(ValidationError) as err:
        canonicalize(raw, build_env(tmp_path))
    assert err.value.rule == 4 and "one gate, one window" in str(err.value)
    assert err.value.path == "featurizers.pg.axis"
