"""User code identified by content (§2.8.1, §5 rules 24 and 25).

A ``pytorch_fn`` used to name a function and nothing else, so the function's
body, its arguments, the files it opened, the environment it read and the row
convention it assumed were all outside the protocol digest. The ``ROME``
replication is the case: its corruption function took its noise scale from an
externally selected file and assumed an eleven-row batch — one clean row
followed by ten corrupted — and neither fact could change the document's
identity.

One class of test per clause of that, and each one is mutation-checked: the
drift is injected and the refusal asserted, *and* the un-drifted document is
asserted to still load. A refusal that also fires on valid work is not a
check, it is an outage.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.engine import Engine
from causalab.protocol.rules.errors import ParseError, ValidationError
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.pipeline import check_engine, compile_protocol
from causalab.protocol.rules.data import check_data_columns

from tests.protocol._docs import UNWRITTEN, base_doc, in_order


pytestmark = pytest.mark.unit


#: The clean body of the corruption function every test starts from: it takes
#: the feature slice, one declared scalar argument, and the row-role bounds
#: the declaration hands it instead of assuming them.
CLEAN_SOURCE = """
def corrupt(f, scale=1.0, *, row_roles=None):
    lo, hi = (row_roles or {}).get("corrupted", (0, 0))
    out = f.clone()
    out[lo:hi] = out[lo:hi] * scale
    return out


def plain(f):
    return f
"""


def write_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, source: str
) -> Path:
    """Put ``source`` on ``sys.path`` as a top-level module and return its
    file. Nothing imports it — resolution reads the file — so the same name
    may be rewritten in place and re-resolved."""
    target = tmp_path / f"{name}.py"
    target.write_text(textwrap.dedent(source).lstrip())
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    return target


def doc_with_code(locator: str, **declaration: Any) -> dict[str, Any]:
    """A minimal one-write document whose write is a declared ``pytorch_fn``.

    The counterfactual half of ``base_doc`` goes: with no ``swap`` operand the
    counterfactual read has no consumer, so the read and the un-intervened
    model that listed it go with it (§2.9), and the point here is the code
    reference, not the read graph.
    """
    doc = base_doc()
    del doc["method"]["reads"]["v_cf"]
    del doc["method"]["intervened_models"][UNWRITTEN]
    del doc["data"]["counterfactual"]
    doc["method"]["code"] = {"corrupt": {"locator": locator, **declaration}}
    doc["method"]["writes"]["patch"]["do"] = {"pytorch_fn": {"code": "corrupt"}}
    return in_order(doc)


def load_doc(raw: dict[str, Any], env: Any) -> CompiledProtocol:
    """Compile a ``pytorch_fn`` document and hold it to a local engine — rule
    13's question, put to ``check_engine`` as the document's own requirement
    plus ``pytorch_fn_local``."""
    compiled = compile_protocol(raw, env=env)
    check_engine(compiled, compiled.capabilities | {"pytorch_fn_local"})
    return compiled


# --------------------------------------------------------------------------- #
# acceptance 1 — editing the body of a referenced function moves the digest
# --------------------------------------------------------------------------- #


class TestSourceIsIdentity:
    def test_editing_the_body_changes_the_document_digest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: Any
    ) -> None:
        module = write_module(tmp_path, monkeypatch, "codeid_body", CLEAN_SOURCE)
        raw = doc_with_code("codeid_body.corrupt", args={"scale": 0.5})

        before = load_doc(raw, env).digests.document
        # the guard against the guard: re-loading the *same* tree must agree,
        # or "the digest changed" says nothing about the edit
        assert load_doc(raw, env).digests.document == before

        module.write_text(module.read_text().replace("* scale", "+ scale"))
        importlib.invalidate_caches()
        after = load_doc(raw, env).digests.document

        assert after != before, "an edited function body left the digest alone"

    def test_the_hash_is_the_source_and_is_in_the_canonical_form(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: Any
    ) -> None:
        import hashlib

        module = write_module(tmp_path, monkeypatch, "codeid_stamp", CLEAN_SOURCE)
        loaded = load_doc(
            doc_with_code("codeid_stamp.corrupt", args={"scale": 0.5}), env
        )
        entry = loaded.canonical["method"]["code"]["corrupt"]
        assert entry["source_module"] == "codeid_stamp"
        assert entry["source_sha256"] == hashlib.sha256(module.read_bytes()).hexdigest()

    def test_an_unrelated_file_does_not_move_the_digest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: Any
    ) -> None:
        """The other half of the same claim: the digest names *this* module,
        not the directory it happens to sit in."""
        write_module(tmp_path, monkeypatch, "codeid_neighbour", CLEAN_SOURCE)
        raw = doc_with_code("codeid_neighbour.corrupt", args={"scale": 0.5})
        before = load_doc(raw, env).digests.document
        (tmp_path / "somebody_else.py").write_text("x = 1\n")
        importlib.invalidate_caches()
        assert load_doc(raw, env).digests.document == before

    def test_editing_an_imported_sibling_changes_the_document_digest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: Any
    ) -> None:
        """The module boundary is not where behaviour stops depending on
        bytes: a constant the function reads from a sibling module is in the
        digest through the declared import closure (§2.8.1), while
        ``source_sha256`` keeps meaning the defining module alone. A file-only
        hash leaves the digest still here — and so does the package's
        ``tree_digest``, which never sees a module beside a user package."""
        import hashlib

        helper = write_module(tmp_path, monkeypatch, "codeid_sibling", "SCALE = 2.0\n")
        write_module(
            tmp_path,
            monkeypatch,
            "codeid_with_sibling",
            "from codeid_sibling import SCALE\n" + CLEAN_SOURCE,
        )
        raw = doc_with_code("codeid_with_sibling.corrupt", args={"scale": 0.5})
        first = load_doc(raw, env)
        assert load_doc(raw, env).digests.document == first.digests.document

        helper.write_text("SCALE = 3.0\n")
        importlib.invalidate_caches()
        second = load_doc(raw, env)
        assert second.digests.document != first.digests.document, (
            "an edited imported module left the digest alone"
        )
        before = first.canonical["method"]["code"]["corrupt"]
        after = second.canonical["method"]["code"]["corrupt"]
        assert before["source_sha256"] == after["source_sha256"]
        assert before["closure"] == {
            "codeid_sibling.py": hashlib.sha256(b"SCALE = 2.0\n").hexdigest()
        }
        assert after["closure"] == {
            "codeid_sibling.py": hashlib.sha256(b"SCALE = 3.0\n").hexdigest()
        }
        assert before["closure_sha256"] != after["closure_sha256"]

    def test_a_type_checking_only_import_does_not_move_the_digest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: Any
    ) -> None:
        """An ``if TYPE_CHECKING:`` block never executes, so nothing in it is
        a dependence: the module it names is not in the closure and editing
        it moves nothing. The upper bound on what the closure covers."""
        hint = write_module(tmp_path, monkeypatch, "codeid_hint", "class Hint: ...\n")
        write_module(
            tmp_path,
            monkeypatch,
            "codeid_typing_only",
            "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n"
            "    from codeid_hint import Hint\n" + CLEAN_SOURCE,
        )
        raw = doc_with_code("codeid_typing_only.corrupt", args={"scale": 0.5})
        first = load_doc(raw, env)
        assert "closure" not in first.canonical["method"]["code"]["corrupt"]
        hint.write_text("class Hint:\n    pass\n")
        importlib.invalidate_caches()
        assert load_doc(raw, env).digests.document == first.digests.document

    def test_a_declared_data_input_is_content_digested(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        artifacts_root: Path,
        env: Any,
    ) -> None:
        """The ROME case's first half: the externally selected noise-scale file is in
        the digest, so changing it is a different protocol."""
        write_module(
            tmp_path,
            monkeypatch,
            "codeid_scale",
            """
            import json

            def corrupt(f):
                return f * json.load(open("noise_scale.json"))["scale"]
            """,
        )
        scale_file = artifacts_root / "noise_scale.json"
        scale_file.write_text('{"scale": 0.1}')
        raw = doc_with_code(
            "codeid_scale.corrupt", data_inputs={"scale": "noise_scale.json"}
        )

        loaded = load_doc(raw, env)
        first = loaded.canonical["method"]["code"]["corrupt"]["data_input_digests"][
            "scale"
        ]
        before = loaded.digests.document

        scale_file.write_text('{"scale": 0.2}')
        after = load_doc(raw, env)
        assert (
            after.canonical["method"]["code"]["corrupt"]["data_input_digests"]["scale"]
            != first
        )
        assert after.digests.document != before, (
            "the noise scale changed and the protocol identity did not"
        )

    def test_a_missing_data_input_is_a_load_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: Any
    ) -> None:
        write_module(
            tmp_path,
            monkeypatch,
            "codeid_missing",
            """
            def corrupt(f):
                return f
            """,
        )
        with pytest.raises(ValidationError) as err:
            load_doc(
                doc_with_code(
                    "codeid_missing.corrupt", data_inputs={"scale": "nope.json"}
                ),
                env,
            )
        assert err.value.rule == 15


# --------------------------------------------------------------------------- #
# acceptance 2 — declared row roles must match the resolved data
# --------------------------------------------------------------------------- #


class TestRowRoles:
    def _doc(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return doc_with_code("codeid_rows.corrupt", args={"scale": 0.5}, row_roles=rows)

    @pytest.fixture(autouse=True)
    def _module(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        write_module(tmp_path, monkeypatch, "codeid_rows", CLEAN_SOURCE)

    @staticmethod
    def _base_ref() -> str:
        """The dataset ref the document under test reads for its ``base``
        role — the table rule 25 measures the declaration against."""
        return doc_with_code("codeid_rows.plain")["data"]["base"]["dataset"]

    def _resolved_rows(self, env: Any) -> int:
        """That table's length, resolved exactly as ``check_row_roles`` does
        (``env.datasets.rows`` on the ref, split fragment included) rather
        than a row count written down here a second time."""
        return len(env.datasets.rows(self._base_ref()))

    def test_matching_row_roles_load(self, env: Any) -> None:
        """Valid work is not refused: the refusal below must not fire here."""
        total = self._resolved_rows(env)
        raw = self._doc(
            [{"role": "clean", "rows": 1}, {"role": "corrupted", "rows": total - 1}]
        )
        loaded = load_doc(raw, env)
        check_data_columns(loaded, env)  # the --data pass, where rule 25 lives

    def test_mismatched_row_roles_are_a_load_error(self, env: Any) -> None:
        raw = self._doc(
            [{"role": "clean", "rows": 1}, {"role": "corrupted", "rows": 10}]
        )
        loaded = load_doc(raw, env)  # the bare load cannot see the tables
        with pytest.raises(ValidationError) as err:
            check_data_columns(loaded, env)
        assert err.value.rule == 25
        message = str(err.value)
        assert "11" in message and str(self._resolved_rows(env)) in message

    def test_the_fixture_really_has_that_many_rows(self, env: Any) -> None:
        """Guard against the guard, on the table the document *actually*
        reads: the ref the loader resolves is the one the count above comes
        from, that table has at least the two rows the matching test needs
        (one clean, one corrupted), and it is not eleven rows long — else the
        mismatch test would pass for the wrong reason."""
        loaded = load_doc(doc_with_code("codeid_rows.plain"), env)
        assert loaded.document.data["base"].dataset == self._base_ref()
        total = self._resolved_rows(env)
        assert total >= 2
        assert total != 11

    def test_no_row_roles_declares_nothing_and_checks_nothing(self, env: Any) -> None:
        """A declaration that says nothing about rows makes no claim, so rule
        25 has nothing to check and must not invent one."""
        loaded = load_doc(doc_with_code("codeid_rows.plain"), env)
        check_data_columns(loaded, env)

    def test_an_empty_row_role_list_is_refused(self, env: Any) -> None:
        with pytest.raises(ParseError):
            load_doc(self._doc([]), env)

    def test_a_run_refuses_before_an_engine_is_chosen(
        self, tmp_path: Path, env: Any
    ) -> None:
        """The refusal has to land before weights do, so ``run_protocol``
        checks rule 25 itself rather than trusting a prior
        ``validate --data``. The engine list is a sentinel: touching it at all
        would mean the check ran too late."""
        from causalab.protocol.pipeline import run_protocol

        class Explodes(Engine):
            name = "explodes"

            def execute(self, compiled: Any, run: Any) -> Any:  # pragma: no cover
                raise AssertionError("an engine was reached")

        raw = self._doc(
            [{"role": "clean", "rows": 1}, {"role": "corrupted", "rows": 10}]
        )
        with pytest.raises(ValidationError) as err:
            run_protocol(load_doc(raw, env), env, Explodes(), tmp_path / "out")
        assert err.value.rule == 25

    def test_row_roles_reach_the_runtime_as_bounds(self) -> None:
        """The declaration replaces inferring roles from physical positions:
        the mechanism hands the function the bounds it declared."""
        from causalab.neural.shared.mechanisms import row_role_bounds
        from causalab.protocol.schema import CodeSpec, RowRole

        spec = CodeSpec(
            locator="x.y",
            row_roles=(RowRole("clean", 1), RowRole("corrupted", 10)),
        )
        assert row_role_bounds(spec) == {"clean": (0, 1), "corrupted": (1, 11)}


# --------------------------------------------------------------------------- #
# rule 24 — undeclared reads, refused where detectable
# --------------------------------------------------------------------------- #


ENV_READER = """
import os

def corrupt(f):
    return f * float(os.environ["ROME_NOISE_SCALE"])
"""

FILE_READER = """
import json
from pathlib import Path

def corrupt(f):
    return f * json.loads(Path("noise_scale.json").read_text())["scale"]
"""


class TestUndeclaredReads:
    def test_an_undeclared_environment_read_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: Any
    ) -> None:
        write_module(tmp_path, monkeypatch, "codeid_env", ENV_READER)
        with pytest.raises(ValidationError) as err:
            load_doc(doc_with_code("codeid_env.corrupt"), env)
        assert err.value.rule == 24
        assert "ROME_NOISE_SCALE" in str(err.value)

    def test_declaring_the_environment_variable_loads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: Any
    ) -> None:
        write_module(tmp_path, monkeypatch, "codeid_env_ok", ENV_READER)
        raw = doc_with_code("codeid_env_ok.corrupt", env_inputs=["ROME_NOISE_SCALE"])
        loaded = load_doc(raw, env)
        assert loaded.canonical["method"]["code"]["corrupt"]["env_inputs"] == [
            "ROME_NOISE_SCALE"
        ]

    def test_an_undeclared_file_read_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: Any
    ) -> None:
        write_module(tmp_path, monkeypatch, "codeid_file", FILE_READER)
        with pytest.raises(ValidationError) as err:
            load_doc(doc_with_code("codeid_file.corrupt"), env)
        assert err.value.rule == 24
        assert "noise_scale.json" in str(err.value)

    def test_declaring_the_file_loads(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        artifacts_root: Path,
        env: Any,
    ) -> None:
        write_module(tmp_path, monkeypatch, "codeid_file_ok", FILE_READER)
        (artifacts_root / "noise_scale.json").write_text('{"scale": 0.1}')
        load_doc(
            doc_with_code(
                "codeid_file_ok.corrupt", data_inputs={"scale": "noise_scale.json"}
            ),
            env,
        )

    def test_a_function_that_reads_nothing_is_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: Any
    ) -> None:
        """The scan must be quiet on the ordinary case."""
        write_module(
            tmp_path,
            monkeypatch,
            "codeid_quiet",
            """
            def corrupt(f):
                return f * 2.0
            """,
        )
        load_doc(doc_with_code("codeid_quiet.corrupt"), env)


# --------------------------------------------------------------------------- #
# rule 24 — arguments, and locators that name no source
# --------------------------------------------------------------------------- #


class TestDeclaredArguments:
    def test_an_undeclared_required_argument_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: Any
    ) -> None:
        write_module(
            tmp_path,
            monkeypatch,
            "codeid_args",
            """
            def corrupt(f, scale):
                return f * scale
            """,
        )
        with pytest.raises(ValidationError) as err:
            load_doc(doc_with_code("codeid_args.corrupt"), env)
        assert err.value.rule == 24
        assert "scale" in str(err.value)

    def test_declaring_it_loads_and_puts_the_value_in_the_digest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: Any
    ) -> None:
        write_module(
            tmp_path,
            monkeypatch,
            "codeid_args_ok",
            """
            def corrupt(f, scale):
                return f * scale
            """,
        )
        first = load_doc(
            doc_with_code("codeid_args_ok.corrupt", args={"scale": 0.1}), env
        )
        second = load_doc(
            doc_with_code("codeid_args_ok.corrupt", args={"scale": 0.2}), env
        )
        assert first.digests.document != second.digests.document

    def test_an_argument_the_function_does_not_take_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: Any
    ) -> None:
        write_module(
            tmp_path,
            monkeypatch,
            "codeid_args_extra",
            """
            def corrupt(f):
                return f
            """,
        )
        with pytest.raises(ValidationError) as err:
            load_doc(
                doc_with_code("codeid_args_extra.corrupt", args={"scale": 0.1}), env
            )
        assert err.value.rule == 24

    def test_a_function_with_no_readable_def_is_not_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: Any
    ) -> None:
        """A closure factory (the arithmetic golden's ``apply_target_<n>``)
        has no ``def`` to read. The source hash still covers it; the checks
        that need a signature do not run, and must not refuse."""
        write_module(
            tmp_path,
            monkeypatch,
            "codeid_factory",
            """
            def _make(n):
                def apply(f):
                    return f * n
                return apply

            corrupt = _make(2)
            """,
        )
        loaded = load_doc(doc_with_code("codeid_factory.corrupt"), env)
        assert loaded.canonical["method"]["code"]["corrupt"]["source_sha256"]


class TestLocators:
    def test_a_locator_naming_no_python_source_is_refused(self, env: Any) -> None:
        with pytest.raises(ValidationError) as err:
            load_doc(doc_with_code("no_such_package_anywhere.corrupt"), env)
        assert err.value.rule == 24

    def test_resolution_does_not_import_the_module(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: Any
    ) -> None:
        """The whole reason the hash can reach the digest: ``validate`` and
        ``digest`` stay torch-free because nothing here is imported."""
        write_module(
            tmp_path,
            monkeypatch,
            "codeid_explodes",
            """
            raise RuntimeError("importing this module is a test failure")

            def corrupt(f):
                return f
            """,
        )
        load_doc(doc_with_code("codeid_explodes.corrupt"), env)
        assert "codeid_explodes" not in sys.modules

    def test_resolution_does_not_import_the_closure_either(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: Any
    ) -> None:
        """The closure walk is as static as the hash: a member that raises at
        import time is read, hashed and parsed — and lands in the manifest —
        without ever being imported."""
        write_module(
            tmp_path,
            monkeypatch,
            "codeid_exploding_helper",
            'raise RuntimeError("importing this module is a test failure")\n',
        )
        write_module(
            tmp_path,
            monkeypatch,
            "codeid_imports_exploder",
            "import codeid_exploding_helper\n" + CLEAN_SOURCE,
        )
        loaded = load_doc(doc_with_code("codeid_imports_exploder.corrupt"), env)
        closure = loaded.canonical["method"]["code"]["corrupt"]["closure"]
        assert set(closure) == {"codeid_exploding_helper.py"}
        assert "codeid_exploding_helper" not in sys.modules
        assert "codeid_imports_exploder" not in sys.modules

    def test_a_module_outside_the_repository_is_not_in_the_closure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: Any
    ) -> None:
        """Third-party code is runtime identity, not document identity: a
        module importable from another ``sys.path`` entry — a stand-in for
        site-packages — is excluded from the closure, so bumping it moves no
        digest. The exclusion is by *root*, not by absence: the module really
        is importable from where it sits."""
        site = tmp_path / "site-packages"
        site.mkdir()
        fake = site / "codeid_fakelib.py"
        fake.write_text("VERSION = 1\n")
        monkeypatch.syspath_prepend(str(site))
        own = tmp_path / "own"
        own.mkdir()
        write_module(
            own,
            monkeypatch,
            "codeid_uses_fakelib",
            "import codeid_fakelib\n" + CLEAN_SOURCE,
        )
        raw = doc_with_code("codeid_uses_fakelib.corrupt", args={"scale": 0.5})
        first = load_doc(raw, env)
        assert importlib.util.find_spec("codeid_fakelib") is not None
        assert "closure" not in first.canonical["method"]["code"]["corrupt"]

        fake.write_text("VERSION = 2\n")
        importlib.invalidate_caches()
        assert load_doc(raw, env).digests.document == first.digests.document

    def test_a_locator_into_an_installed_module_declares_no_closure(
        self, env: Any
    ) -> None:
        """The other side of the boundary: the hashed module *itself* is
        installed (the stdlib here; site-packages is the same case). The
        locator puts that file in the digest as ``source_sha256`` — the
        document named it — but its imports are runtime identity, so the
        closure is empty: not the two hundred stdlib modules ``posixpath``
        reaches, which would move the digest on every Python patch release
        and cost seconds per load. The bound is generous (the walk it forbids
        takes over a second; the load takes milliseconds)."""
        import hashlib
        import time

        from causalab.protocol.identity import resolve_locator

        raw = doc_with_code("posixpath.join")
        start = time.perf_counter()
        loaded = load_doc(raw, env)
        elapsed = time.perf_counter() - start
        entry = loaded.canonical["method"]["code"]["corrupt"]
        stdlib_file = resolve_locator("posixpath.join").path
        assert entry["source_module"] == "posixpath"
        assert (
            entry["source_sha256"]
            == hashlib.sha256(stdlib_file.read_bytes()).hexdigest()
        )
        assert "closure" not in entry and "closure_sha256" not in entry
        assert elapsed < 0.5, f"loading a stdlib locator took {elapsed:.2f}s"


# --------------------------------------------------------------------------- #
# the shape of the declaration itself
# --------------------------------------------------------------------------- #


class TestTheDeclarationIsTheOnlyForm:
    def test_a_bare_qualname_is_refused_and_says_what_to_write(self, env: Any) -> None:
        doc = doc_with_code("os.path.join")
        doc["method"]["writes"]["patch"]["do"] = {
            "pytorch_fn": {"qualname": "torch.relu"}
        }
        with pytest.raises(ParseError) as err:
            load_doc(doc, env)
        assert "code" in str(err.value)

    def test_an_undeclared_code_name_is_an_unresolved_reference(self, env: Any) -> None:
        doc = doc_with_code("os.path.join")
        doc["method"]["writes"]["patch"]["do"] = {"pytorch_fn": {"code": "corupt"}}
        with pytest.raises(ValidationError) as err:
            load_doc(doc, env)
        assert err.value.rule == 4
        assert "corrupt" in str(err.value)  # the suggestion

    @pytest.mark.parametrize(
        "field",
        [
            "source_sha256",
            "source_module",
            "closure",
            "closure_sha256",
            "data_input_digests",
        ],
    )
    def test_derived_fields_are_not_authorable(self, field: str, env: Any) -> None:
        doc = doc_with_code("os.path.join")
        doc["method"]["code"]["corrupt"][field] = "whatever"
        with pytest.raises(ParseError) as err:
            load_doc(doc, env)
        assert err.value.code == "P5"

    def test_a_code_name_shares_the_global_namespace(self, env: Any) -> None:
        doc = doc_with_code("os.path.join")
        doc["method"]["code"]["lm_head"] = doc["method"]["code"].pop("corrupt")
        doc["method"]["writes"]["patch"]["do"] = {"pytorch_fn": {"code": "lm_head"}}
        with pytest.raises(ValidationError) as err:
            load_doc(doc, env)
        assert err.value.rule == 3
