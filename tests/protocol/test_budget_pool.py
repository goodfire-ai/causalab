"""§2.5 ``pool`` and ``k_schedule.of`` at the protocol layer: how the two
fields parse, what rule 4 holds a pool to, that an unpooled document's digest
does not move, and the rule-15 identity check on a pooled bundle.

A pool is one budget over gates at several sites. What this file pins is
everything that can be decided with no model loaded: the vocabulary, the
agreement a pool's members owe each other, and the stamp a pooled bundle
carries so a cut through one member alone is refused rather than scored.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.schema.explicit import canonicalize, digest
from causalab.protocol.rules.errors import ParseError, ValidationError
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.schema import K_SCHEDULE_OF, parse_document

from tests.protocol._docs import LOGIT_DIFF, UNWRITTEN, base_doc, in_order, term
from tests.protocol._env import build_env


pytestmark = pytest.mark.unit

FIXED = {"kind": "fixed", "k": 2}


def pool_doc(
    gate_a: dict[str, Any],
    gate_b: dict[str, Any] | None = None,
    *,
    train: bool = True,
    params: list[str] | None = None,
) -> dict[str, Any]:
    """`base_doc` with gate `a` at the target site and, when given, gate `b` at
    a second site (`mlp_output`, layer 3), both written into one intervened
    model — the shape a pool exists for."""
    doc = base_doc()
    method = doc["method"]
    method["featurizers"] = {"a": {"kind": "gate", **gate_a}}
    method["reads"]["v_cf"]["featurizer"] = "a"
    method["writes"]["patch"]["featurizer"] = "a"
    names = ["a"]
    if gate_b is not None:
        method["featurizers"]["b"] = {"kind": "gate", **gate_b}
        method["sites"]["mlp"] = {"component": "mlp_output", "layers": [3]}
        method["reads"]["v_cf_b"] = {"site": "mlp", "pos": -1, "featurizer": "b"}
        method["intervened_models"][UNWRITTEN]["reads"].append("v_cf_b")
        method["writes"]["patch_b"] = {
            "site": "mlp",
            "pos": -1,
            "featurizer": "b",
            "do": {"swap": "v_cf_b"},
        }
        method["intervened_models"]["patched"]["writes"].append("patch_b")
        names.append("b")
    if train:
        method["train"] = {
            "objective": [[1.0, term("logits", "patched", dict(LOGIT_DIFF))]],
            "params": params if params is not None else names,
            "optimizer": {"name": "adam", "lr": 0.1},
            "steps": {"epochs": 1},
            "batch": {"pairs": 2},
        }
        for name in params if params is not None else names:
            method["save"].append(
                {
                    "value": name,
                    "site": "tgt" if name == "a" else "mlp",
                    "file_path": f"{name}.safetensors",
                }
            )
    return in_order(doc)


def budget(**extra: Any) -> dict[str, Any]:
    return {"parametrization": "budget", "k_schedule": FIXED, **extra}


# -- parse ------------------------------------------------------------------- #


def test_pool_and_of_parse_and_of_is_a_closed_vocabulary() -> None:
    doc = parse_document(
        pool_doc(
            budget(pool="mib", k_schedule={**FIXED, "of": "kept"}),
            budget(pool="mib", k_schedule={**FIXED, "of": "kept"}),
        )
    )
    assert doc.featurizers["a"].pool == "mib" and doc.featurizers["b"].pool == "mib"
    assert doc.featurizers["a"].k_schedule == {"kind": "fixed", "k": 2, "of": "kept"}
    assert K_SCHEDULE_OF == ("patched", "kept")
    with pytest.raises(ParseError) as err:
        parse_document(pool_doc(budget(k_schedule={**FIXED, "of": "clean"})))
    assert err.value.code == "P4"


@pytest.mark.parametrize(
    "gate, needle",
    [
        ({"pool": "mib"}, "requires budget during fitting"),  # sigmoid fit
        ({"parametrization": "clamp", "pool": "mib"}, "requires budget during fitting"),
        (budget(pool=""), "needs a name"),
        (budget(pool={"sweep": ["p", "q"]}), "string"),
    ],
)
def test_pool_is_a_budget_gates_name_and_nothing_else(gate, needle) -> None:
    with pytest.raises(ParseError) as err:
        parse_document(pool_doc(gate))
    assert needle in str(err.value)


def test_a_loaded_gate_may_author_a_pool_under_any_map() -> None:
    """`k_schedule` is a training-time object and refused on a loaded gate;
    `pool` is not — the apply cuts the pooled ranking at `top_k`, and a
    pooled READOUT takes a sigmoid bundle too (DBM's joint MIB ranking)."""
    loaded = {"file_path": "fit/a.safetensors", "top_k": 3, "pool": "mib"}
    doc = parse_document(pool_doc({"parametrization": "budget", **loaded}, train=False))
    assert doc.featurizers["a"].pool == "mib" and doc.featurizers["a"].top_k == 3
    doc = parse_document(pool_doc(loaded, train=False))
    assert doc.featurizers["a"].pool == "mib"
    with pytest.raises(ParseError, match="needs a top_k"):
        parse_document(
            pool_doc({"file_path": "fit/a.safetensors", "pool": "mib"}, train=False)
        )


# -- rule 4: what makes members one pool ------------------------------------- #


def _expect_rule(rule: int, raw: dict[str, Any], tmp_path: Path) -> ValidationError:
    with pytest.raises(ValidationError) as err:
        compile_protocol(raw, env=build_env(tmp_path))
    assert err.value.rule == rule, err.value
    return err.value


def test_rule_4_members_agree_on_the_schedule(tmp_path: Path) -> None:
    err = _expect_rule(
        4,
        pool_doc(
            budget(pool="mib"), budget(pool="mib", k_schedule={"kind": "fixed", "k": 3})
        ),
        tmp_path,
    )
    assert "disagree on 'k_schedule'" in str(err)
    err = _expect_rule(
        4,
        pool_doc(budget(pool="mib"), budget(pool="mib", stop_grad_shift=True)),
        tmp_path,
    )
    assert "disagree on 'stop_grad_shift'" in str(err)


def test_rule_4_a_pool_is_fitted_together(tmp_path: Path) -> None:
    err = _expect_rule(
        4,
        pool_doc(budget(pool="mib"), budget(pool="mib"), params=["a"]),
        tmp_path,
    )
    assert "one pool, one fit" in str(err)


def test_rule_4_a_swept_map_with_a_non_budget_arm_is_not_a_pool(tmp_path: Path) -> None:
    doc = pool_doc(budget(pool="mib"), budget(pool="mib"))
    doc["method"]["featurizers"]["b"]["parametrization"] = {
        "sweep": ["budget", "sigmoid"]
    }
    with pytest.raises((ParseError, ValidationError)):
        compile_protocol(doc, env=build_env(tmp_path))


def test_a_valid_pool_loads_and_a_pool_of_one_is_the_lone_gate(tmp_path: Path) -> None:
    env = build_env(tmp_path)
    compile_protocol(pool_doc(budget(pool="mib"), budget(pool="mib")), env=env)
    compile_protocol(pool_doc(budget(pool="mib")), env=env)


# -- canonical form ---------------------------------------------------------- #


def test_pool_and_of_enter_the_canonical_form_only_when_authored(
    tmp_path: Path,
) -> None:
    """The `group` precedent: an existing budget document is byte-identical,
    and authoring either field is a new experiment."""
    env = build_env(tmp_path)
    plain = canonicalize(pool_doc(budget(), budget()), env)
    entry = plain["method"]["featurizers"]["a"]
    assert "pool" not in entry and "of" not in entry["k_schedule"]
    pooled = canonicalize(pool_doc(budget(pool="mib"), budget(pool="mib")), env)
    assert pooled["method"]["featurizers"]["a"]["pool"] == "mib"
    assert digest(plain) != digest(pooled)
    kept = canonicalize(
        pool_doc(
            budget(k_schedule={**FIXED, "of": "kept"}),
            budget(k_schedule={**FIXED, "of": "kept"}),
        ),
        env,
    )
    assert kept["method"]["featurizers"]["a"]["k_schedule"]["of"] == "kept"
    assert digest(plain) != digest(kept)


# -- rule 15: a pooled bundle is only a pooled bundle ------------------------ #


def _write_gate_header(
    root: Path, rel: str, *, theta_len: int, extra: dict[str, Any]
) -> None:
    metadata: dict[str, str] = {
        "commit": "fixture",
        "dtype": "fp32",
        "engine": "pytorch_hooks",
        "model_dtype": "fp32",
        "model_key": "gpt2",
        "model_revision": "main",
        "site": json.dumps(
            {"component": "block_output", "layers": [3]}, sort_keys=True
        ),
        "trained_on": "weekdays/data#train",
        "parametrization": "budget",
    }
    for key, value in extra.items():
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


def _apply(gate: dict[str, Any]) -> dict[str, Any]:
    return pool_doc(
        {
            "parametrization": "budget",
            "file_path": "fit/a.safetensors",
            "top_k": 3,
            **gate,
        },
        train=False,
    )


def test_a_pooled_bundle_reloads_only_under_a_pooled_document(tmp_path: Path) -> None:
    _write_gate_header(
        tmp_path,
        "fit/a.safetensors",
        theta_len=768,
        extra={"pool": "mib", "pool_units": "1536"},
    )
    compile_protocol(_apply({"pool": "mib"}), env=build_env(tmp_path))
    with pytest.raises(ValidationError) as err:
        compile_protocol(_apply({}), env=build_env(tmp_path))
    assert err.value.rule == 15 and "fitted in pool 'mib'" in str(err.value)
    with pytest.raises(ValidationError) as err:
        compile_protocol(_apply({"pool": "other"}), env=build_env(tmp_path))
    assert err.value.rule == 15 and "'pool'" in str(err.value)


def test_an_unpooled_bundle_may_join_a_pooled_readout(tmp_path: Path) -> None:
    """The DBM case: separately fitted masks read out through one joint cut.
    The stamp is an expectation the bundle may not contradict, not one it
    must carry."""
    _write_gate_header(tmp_path, "fit/a.safetensors", theta_len=768, extra={})
    compile_protocol(_apply({"pool": "mib"}), env=build_env(tmp_path))


def test_rule_4_readout_members_agree_on_the_map(tmp_path: Path) -> None:
    """The joint ranking is over raw θ, so a clamp member (θ in [0, 1]) and a
    sigmoid one (logits) would not share a scale: `parametrization` is rule 4's
    fourth agreement field."""
    loaded = {"file_path": "fit/a.safetensors", "top_k": 3, "pool": "mib"}
    raw = pool_doc(loaded, {**loaded, "parametrization": "clamp"}, train=False)
    raw["method"]["featurizers"]["b"]["file_path"] = "fit/b.safetensors"
    err = _expect_rule(4, raw, tmp_path)
    assert "disagree on 'parametrization'" in str(err)


def test_a_swept_file_path_is_a_loaded_member(tmp_path: Path) -> None:
    """Rule 4's one definition of "loaded" is `file_path is not None`, so a
    readout pool over swept bundle paths is not told it "is not a budget
    gate" or that it "mixes fitted and loaded gates"."""
    loaded = {
        "file_path": {"sweep": ["fit/a.safetensors", "fit/a2.safetensors"]},
        "top_k": 3,
        "pool": "mib",
    }
    other = {"file_path": "fit/b.safetensors", "top_k": 3, "pool": "mib"}
    raw = pool_doc(loaded, other, train=False)
    with pytest.raises(ValidationError) as err:
        compile_protocol(raw, env=build_env(tmp_path))
    assert err.value.rule != 4 or "budget gate" not in str(err.value)
    assert "mixes fitted and loaded" not in str(err.value)


def test_a_stamped_pool_of_another_size_is_refused_at_the_link() -> None:
    """`pool_units` is the pool's shape: a θ fitted against 1536 co-member
    units read as one of 8 is not the same ranking, whatever the name says."""
    import torch

    from causalab.neural.shared.featurizers import Gate, link_budget_pools
    from causalab.protocol.rules.errors import ProtocolError
    from causalab.protocol.schema import FeaturizerSpec

    theta = torch.randn(4)
    gates = {
        n: Gate.from_theta(theta, parametrization="budget", top_k=3, pool="mib")
        for n in "ab"
    }
    gates["a"].stamped_pool_units = 1536
    specs = {
        n: FeaturizerSpec(
            kind="gate",
            parametrization="budget",
            top_k=3,
            pool="mib",
            file_path=f"{n}.st",
        )
        for n in "ab"
    }
    with pytest.raises(ProtocolError, match="fitted in a pool of 1536 units"):
        link_budget_pools(specs, gates, lambda n: gates[n])
    for g in gates.values():
        g.stamped_pool_units = 8
    link_budget_pools(specs, gates, lambda n: gates[n])
    assert gates["a"].pool.units == 8
