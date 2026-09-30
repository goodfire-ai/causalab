"""The load-time identity check on a ``subspace``'s ``init`` basis (spec §2.5).

A fit that starts from a saved basis is only meaningful if the basis was
fitted where the subspace is trained: on the same model realization, at the
same site. Those fields are compared exactly as they are for a ``file_path``
load, and a mismatch refuses **naming the field**. What is deliberately *not*
compared is what the basis owns — its rank (a wider basis seeds by its first
``k`` columns; the column count is a build-time check, where the tensor is),
its params dtype, and the data it was fitted on (a PCA over one corpus may
start a fit on another).

The document under test is the shipped ``das_pca_init.json`` itself, overridden
the way ``--set`` would, against the generated PCA fixture that matches it
(``tests/protocol/_env.py``).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from causalab.neural.shared.step_rules import check_steps
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.registry.models import get_model_info
from causalab.io.env import build_artifact_identity

from tests.protocol._docs import base_doc, in_order
from tests.protocol._env import (
    PCA_FIXTURE_RELPATH,
    SHIPPED_LAYER,
    SHIPPED_MODEL,
    build_env,
    write_zero_bundle,
    steps_of,
)
from tests._helpers.paths import PROTOCOLS_DIR


pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
PRESET = PROTOCOLS_DIR / "das_pca_init.json"

#: The fixture basis's stamp, as ``write_pca_fixture`` writes it: what a
#: ``fit_pca`` output over a harvest of the shipped preset's model @ main
#: (bf16) at ``block_output`` `SHIPPED_LAYER` carries after the runner
#: stamps it.
BASIS_STAMP: dict[str, Any] = {
    "model_key": SHIPPED_MODEL,
    "model_revision": "main",
    "model_dtype": "bf16",
    "site": {"component": "block_output", "layers": [SHIPPED_LAYER]},
    "k": 16,
    "dtype": "fp32",
    "engine": "script",
}


def _refusal(env, **overrides: Any) -> str:
    with pytest.raises(ValidationError) as err:
        compile_protocol(PRESET, env=env, overrides=overrides)
    assert err.value.rule == 15
    return str(err.value)


def test_the_matching_basis_loads(env):
    loaded = compile_protocol(PRESET, env=env)
    assert len(steps_of(loaded, env).points) == 15
    for point in steps_of(loaded, env).documents:
        assert point.featurizers["rot"].init == {"file_path": PCA_FIXTURE_RELPATH}


def test_another_layer_refuses_naming_the_site(env):
    message = _refusal(env, **{"sites.target.layers": 17})
    assert "'site'" in message and "init" in message


def test_another_site_record_refuses_naming_the_site(env):
    message = _refusal(env, **{"sites.target.component": "attention_output"})
    assert "'site'" in message


def test_another_model_dtype_refuses_naming_the_dtype(env):
    """A basis fitted over bf16 activations does not seed a fit over fp32
    ones — the same rule a loaded rotation follows, and the same field."""
    message = _refusal(env, **{"model.dtype": "fp32"})
    assert "'model_dtype'" in message


def test_another_revision_refuses_naming_the_revision(env):
    message = _refusal(env, **{"model.revision": "v2"})
    assert "'model_revision'" in message


def test_another_model_key_refuses_naming_the_key(env):
    message = _refusal(env, **{"model.key": "gpt2", "sites.target.layers": 3})
    assert "'model_key'" in message


def test_a_larger_k_than_the_basis_holds_is_not_a_load_question(env):
    """The header does not carry the tensor's shape, and the rank stamped on
    the basis is the basis's own — the column count is checked at build,
    where the tensor is (``featurizers._init_basis``)."""
    loaded = compile_protocol(PRESET, env=env, overrides={"featurizers.rot.k": 64})
    assert len(steps_of(loaded, env).points) == 3


def _env_with_basis(
    root: Path,
    stamp: dict[str, Any] | None,
    entries: dict[str, dict[str, Any]] | None = None,
) -> Any:
    """A resolution environment whose ``init`` basis carries ``stamp`` at file
    level (or no metadata at all for ``None``) and ``entries`` in its header
    table — the single record a script step writes unless given; the fixture
    datasets otherwise."""
    metadata = build_artifact_identity(**stamp) if stamp is not None else {}
    if stamp is not None:
        table = entries or {"weight": {"slot": "weight", "coords": {}}}
        metadata["entries"] = json.dumps(table, sort_keys=True)
    write_zero_bundle(
        root / PCA_FIXTURE_RELPATH,
        metadata,
        (get_model_info(SHIPPED_MODEL).hidden_size, 16),
    )
    return build_env(root)


def _swept_entries(**per_entry: Any) -> dict[str, dict[str, Any]]:
    """The ``entries`` table a k-swept DAS fit writes: one ``weight`` record
    per rank, named by its coordinates, plus ``per_entry`` fields stamped on
    every record."""
    return {
        f"weight[k={k}]": {
            "slot": "weight",
            "coords": {"k": k},
            "k": str(k),
            **per_entry,
        }
        for k in (4, 8)
    }


def test_a_swept_basis_whose_points_agree_loads_off_the_file_level_stamp(tmp_path):
    """A fitted rotation reused as a warm start, no ``entry`` authored: the
    executing point selects among the entries, but the model realization and
    site are the producer's, stamped file-wide, and are checked here."""
    stamp = {key: value for key, value in BASIS_STAMP.items() if key != "k"}
    env = _env_with_basis(tmp_path, stamp, _swept_entries())
    assert len(steps_of(compile_protocol(PRESET, env=env), env).points) == 15


def test_a_swept_basis_from_another_model_refuses_naming_the_key(tmp_path):
    """The hole a deferred check would leave: nothing at build compares the
    model, so a swept bundle from the wrong model must refuse at load."""
    stamp = {**BASIS_STAMP, "model_key": "gpt2"}
    del stamp["k"]
    env = _env_with_basis(tmp_path, stamp, _swept_entries())
    with pytest.raises(ValidationError, match="mismatch on 'model_key'"):
        compile_protocol(PRESET, env=env)


def test_a_swept_basis_from_another_layer_refuses_naming_the_site(tmp_path):
    stamp = {**BASIS_STAMP, "site": {"component": "block_output", "layers": [3]}}
    del stamp["k"]
    env = _env_with_basis(tmp_path, stamp, _swept_entries())
    with pytest.raises(ValidationError, match="mismatch on 'site'"):
        compile_protocol(PRESET, env=env)


def test_a_site_stamped_per_entry_is_left_to_the_executing_point(tmp_path):
    """A producer that swept the site stamps it per entry. A ``subspace``
    start from such a bundle compiles without an authored ``entry``: the
    compiler's representatives carry no coordinates, so the load checks the
    file-level stamp and leaves the site to the per-point check below."""
    stamp = {
        key: value for key, value in BASIS_STAMP.items() if key not in ("k", "site")
    }
    site = json.dumps(
        {"component": "block_output", "layers": [SHIPPED_LAYER]}, sort_keys=True
    )
    env = _env_with_basis(tmp_path, stamp, _swept_entries(site=site))
    assert len(steps_of(compile_protocol(PRESET, env=env), env).points) == 15
    # naming the entry reaches its record, and the check runs against it
    loaded = compile_protocol(
        PRESET,
        env=env,
        overrides={
            "featurizers.rot.init": {
                "file_path": PCA_FIXTURE_RELPATH,
                "entry": {"k": 8},
            }
        },
    )
    assert len(steps_of(loaded, env).points) == 15


def test_the_per_point_check_holds_each_selected_entry_to_its_site(tmp_path):
    """Given a point's coordinates, the load selects the start entry that
    point selects and compares the entry's recorded site with the point's.
    The ``k=8`` entry records layer 3, so the engine's per-step pass refuses
    the ``k=8`` points naming that entry, and passes the ``k=4`` points.
    The points whose ``k`` no entry holds are left to the build."""
    stamp = {
        key: value for key, value in BASIS_STAMP.items() if key not in ("k", "site")
    }
    entries = _swept_entries()
    entries["weight[k=4]"]["site"] = json.dumps(
        {"component": "block_output", "layers": [SHIPPED_LAYER]}, sort_keys=True
    )
    entries["weight[k=8]"]["site"] = json.dumps(
        {"component": "block_output", "layers": [3]}, sort_keys=True
    )
    env = _env_with_basis(tmp_path, stamp, entries)
    steps = steps_of(compile_protocol(PRESET, env=env), env)
    coords = [point.coords for point in steps.expansion.points]
    at_k = {
        k: [i for i, point in enumerate(coords) if point["featurizers.rot.k"] == k]
        for k in (4, 8)
    }
    check_steps(
        [steps.documents[i] for i in at_k[4]],
        env,
        coords=[coords[i] for i in at_k[4]],
    )
    with pytest.raises(ValidationError) as err:
        check_steps(steps.documents, env, coords=coords)
    message = str(err.value)
    assert err.value.rule == 15
    assert "entry 'weight[k=8]'" in message and "mismatch on 'site'" in message
    assert "'weight[k=4]'" not in message


def test_an_authored_entry_at_another_site_still_refuses_at_load(tmp_path):
    """Leaving the site to the build is only for a selection the load cannot
    see. An authored ``entry`` is visible, so its record is checked here."""
    stamp = {
        key: value for key, value in BASIS_STAMP.items() if key not in ("k", "site")
    }
    site = json.dumps({"component": "block_output", "layers": [3]}, sort_keys=True)
    env = _env_with_basis(tmp_path, stamp, _swept_entries(site=site))
    with pytest.raises(ValidationError, match="mismatch on 'site'"):
        compile_protocol(
            PRESET,
            env=env,
            overrides={
                "featurizers.rot.init": {
                    "file_path": PCA_FIXTURE_RELPATH,
                    "entry": {"k": 8},
                }
            },
        )


def test_another_per_entry_field_still_needs_an_authored_entry(tmp_path):
    """Only the site is a field a site sweep varies. A model revision stamped
    per entry is not the point's to choose, so the load asks for an
    ``entry``."""
    stamp = {
        key: value
        for key, value in BASIS_STAMP.items()
        if key not in ("k", "model_revision")
    }
    env = _env_with_basis(tmp_path, stamp, _swept_entries(model_revision="main"))
    with pytest.raises(ValidationError, match="must author 'init.entry'"):
        compile_protocol(PRESET, env=env)


def test_a_basis_over_another_dataset_is_accepted(tmp_path):
    """The dataset identity is the one field a starting point need not share:
    the preset trains on ``natural_domains_arithmetic/data/weekdays#train``
    and this basis says it saw
    something else, and that is a legal start."""
    env = _env_with_basis(tmp_path, {**BASIS_STAMP, "trained_on": "weekdays/other"})
    assert len(steps_of(compile_protocol(PRESET, env=env), env).points) == 15


def test_a_basis_with_its_own_rank_and_dtype_is_accepted(tmp_path):
    env = _env_with_basis(tmp_path, {**BASIS_STAMP, "k": 64, "dtype": "bf16"})
    assert len(steps_of(compile_protocol(PRESET, env=env), env).points) == 15


def test_an_unstamped_basis_refuses(tmp_path):
    env = _env_with_basis(tmp_path, None)
    with pytest.raises(ValidationError, match="carries no ArtifactIdentity"):
        compile_protocol(PRESET, env=env)


def test_a_missing_basis_is_a_load_error(env):
    message = _refusal(
        env, **{"featurizers.rot.init.file_path": "artifacts/nowhere.safetensors"}
    )
    assert "not found" in message


# --------------------------------------------------------------------------- #
# the two `init_*` keys (§8): recorded on every fit from a basis, *expected*
# of a loaded bundle only when the loading document authors `init`
# --------------------------------------------------------------------------- #
#
# `ARTIFACT_IDENTITY_KEYS` grew two keys. What has to stay true is that the
# check iterates the keys the *document* implies, never the key list: every
# subspace bundle fitted before the keys existed (no `init_*` in its header)
# still loads under a document that never asked for a start, and only a
# document that authors `init` holds a bundle to those fields.

INIT_KEYS = ("init_trained_on", "init_components")
#: the data ref the basis a fit starts from was fitted over — the value
#: `init_trained_on` is held to under a document that authors `init`
BASIS_TRAINED_ON = "weekdays/data#train"
ROT_RELPATH = "artifacts/weekdays/llama31_8b/subspace/rot_k2.safetensors"


def _rot_doc(**featurizer: Any) -> dict[str, Any]:
    """``base_doc`` (gpt2, block_output L3) applying a *loaded* rank-2
    rotation on its patch — the shape of every apply document."""
    raw = base_doc()
    raw["method"]["featurizers"] = {
        "rot": {
            "kind": "subspace",
            "k": 2,
            "parametrization": "cayley",
            "file_path": ROT_RELPATH,
            **featurizer,
        }
    }
    raw["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    raw["method"]["writes"]["patch"]["featurizer"] = "rot"
    return in_order(raw)


def _main_era_stamp(**extra: Any) -> dict[str, Any]:
    """The header a subspace fit stamped *before* the ``init_*`` keys existed:
    exactly what ``_rot_doc`` implies, in the base's key set, no start."""
    return {
        "model_key": "gpt2",
        "model_revision": "main",
        "model_dtype": "fp32",
        "site": {"component": "block_output", "layers": [3]},
        "k": 2,
        "parametrization": "cayley",
        "dtype": "fp32",
        "trained_on": "weekdays/data#train",
        "engine": "pytorch_hooks",
        **extra,
    }


def _env_with_rot(root: Path, stamp: dict[str, Any], *, basis: bool = False) -> Any:
    metadata = build_artifact_identity(**stamp)
    metadata["entries"] = json.dumps(
        {"weight": {"slot": "weight", "coords": {}}}, sort_keys=True
    )
    write_zero_bundle(root / ROT_RELPATH, metadata, (768, 2))
    if basis:
        _env_with_basis(
            root,
            {
                **BASIS_STAMP,
                "model_key": "gpt2",
                "site": stamp["site"],
                "trained_on": BASIS_TRAINED_ON,
            },
        )
    return build_env(root)


def _init_document(raw: dict[str, Any]) -> Any:
    """A parsed ``Document`` whose loaded featurizer *also* authors ``init``.
    The parser refuses the pair on one featurizer (§2.5), so the document is
    assembled past it — this is the loader's contract under test, not the
    parser's."""
    from causalab.protocol.schema import parse_document

    doc = parse_document(raw)
    rot = dataclasses.replace(
        doc.featurizers["rot"], init={"file_path": PCA_FIXTURE_RELPATH}
    )
    return dataclasses.replace(doc, featurizers={**doc.featurizers, "rot": rot})


def test_a_main_era_bundle_loads_under_a_document_without_init(tmp_path):
    """(a) The backward-compatibility constraint: no `init_*` in the header, none asked for —
    the extended key list refuses nothing that loaded before it."""
    env = _env_with_rot(tmp_path, _main_era_stamp())
    loaded = compile_protocol(_rot_doc(), env=env)
    assert len(steps_of(loaded, env).points) == 1
    stamped = env.artifacts.read_identity(ROT_RELPATH)
    assert stamped is not None and not (set(stamped) & set(INIT_KEYS))


def test_the_same_bundle_is_refused_under_a_document_with_init(tmp_path):
    """(b) A document that authors `init` holds the bundle it loads to the
    start it names, and a bundle that recorded none is refused naming the
    field — the fail-closed twin of (a)."""
    from causalab.protocol.rules.data import check_loaded_featurizers

    env = _env_with_rot(tmp_path, _main_era_stamp(), basis=True)
    doc = _init_document(_rot_doc())
    with pytest.raises(ValidationError) as err:
        check_loaded_featurizers(doc, env)
    assert err.value.rule == 15
    assert "missing 'init_trained_on'" in str(err.value)


def test_a_bundle_stamped_with_init_loads_under_its_init_document(tmp_path):
    """(c) The valid twin of (b): a fit that recorded the basis it started
    from — the basis's data ref, the columns taken — loads under the document
    naming that basis, and a fit from a basis over *other* data does not."""
    from causalab.protocol.rules.data import check_loaded_featurizers

    started = {"init_trained_on": BASIS_TRAINED_ON, "init_components": [0, 1]}
    env = _env_with_rot(tmp_path, _main_era_stamp(**started), basis=True)
    check_loaded_featurizers(_init_document(_rot_doc()), env)
    # (a) again, on the stamped bundle: a document without `init` asks nothing
    # of the start, so the extra fields are provenance it simply carries
    assert len(steps_of(compile_protocol(_rot_doc(), env=env), env).points) == 1
    other = _env_with_rot(
        tmp_path / "other",
        _main_era_stamp(**{**started, "init_trained_on": "weekdays/data#other"}),
        basis=True,
    )
    with pytest.raises(ValidationError, match="mismatch on 'init_trained_on'"):
        check_loaded_featurizers(_init_document(_rot_doc()), other)


def test_the_header_helpers_live_in_resolve_and_are_re_exported_by_step_io():
    """`entry_table` / `entry_identity` moved to the identity schema's owner;
    `causalab.io.step_io` keeps both importable so a script that imports them
    from there keeps working."""
    from causalab.io import step_io
    from causalab.io import env as resolve

    assert step_io.entry_table is resolve.entry_table
    assert step_io.entry_identity is resolve.entry_identity
    assert {"entry_table", "entry_identity"} <= set(step_io.__all__)
    header = {
        "trained_on": "weekdays/data#train",
        "entries": json.dumps(
            {"weight[k=2]": {"slot": "weight", "coords": {"k": 2}, "k": "2"}}
        ),
    }
    assert resolve.entry_table(header) == {
        "weight[k=2]": {"slot": "weight", "coords": {"k": 2}, "k": "2"}
    }
    assert resolve.entry_identity(header, "weight[k=2]") == {
        "trained_on": "weekdays/data#train",
        "k": "2",
    }
