"""The shared compile cache (``neural/shared/compile_cache.py``): the toolchain
signature separates what must not mix, the layout under the root, that the
root is opt-in through the variable alone, what ``configure`` does to the
compilers' environment and to the root — personal or shared — with the
toolchain injected, since the CPU tier has no CUDA to detect one from, and
that the loaders call it."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import stat
import sys
import threading
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from causalab.neural.shared import compile_cache
from causalab.neural.shared.compile_cache import (
    ENV,
    GROUP_SHARED_MODE,
    VARIABLES,
    Toolchain,
    configure,
    is_shared,
    layout,
    python_abi,
)

pytestmark = pytest.mark.unit

H100 = Toolchain(
    torch="2.9.0+cu128",
    cuda="12.8",
    triton="3.5.0",
    tilelang="0.1.14",
    fla="0.5.2",
    transformers="5.16.1",
    python="cpython-310-x86_64-linux-gnu",
    gpu="NVIDIA H100 80GB HBM3",
    capability="9.0",
)

FIELDS = [field.name for field in dataclasses.fields(Toolchain)]
OPTIONAL = ["cuda", "triton", "tilelang", "fla", "transformers"]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for variable in (ENV, *VARIABLES.values()):
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture(autouse=True)
def kept_umask():
    """A shared root widens the process umask; every test gets its own back."""
    previous = os.umask(0o022)
    os.umask(previous)
    yield previous
    os.umask(previous)


def _another_group_of_ours(path: Path) -> int:
    """A group the test user belongs to other than the one ``path`` has, so a
    root can be given a group its subdirectories would not inherit by
    default; skipped where the user is in one group only."""
    others = set(os.getgroups()) - {path.stat().st_gid}
    if not others:
        pytest.skip("the test user belongs to a single group")
    return min(others)


def _skip_where_the_group_is_inherited(root: Path, group: int) -> None:
    """Skip where a directory made under ``root`` inherits ``group`` on its
    own (BSD semantics): there, adoption cannot be told from inheritance, so
    the tests that assert it only run on Linux."""
    probe = root / "probe"
    probe.mkdir()
    inherited = probe.stat().st_gid == group
    probe.rmdir()
    if inherited:
        pytest.skip("this filesystem hands the parent's group down itself")


def _umask() -> int:
    current = os.umask(0o077)
    os.umask(current)
    return current


class TestSignature:
    def test_is_a_short_hex_content_hash(self) -> None:
        signature = H100.signature()
        assert len(signature) == 16
        int(signature, 16)
        assert Toolchain(**dataclasses.asdict(H100)).signature() == signature

    @given(field=st.sampled_from(FIELDS), value=st.text(min_size=1, max_size=12))
    @settings(max_examples=60, deadline=None)
    def test_any_field_moving_moves_the_signature(self, field: str, value: str) -> None:
        current = getattr(H100, field)
        if value == current:
            return
        moved = dataclasses.replace(H100, **{field: value})
        assert moved.signature() != H100.signature()

    @given(field=st.sampled_from(OPTIONAL))
    @settings(max_examples=10, deadline=None)
    def test_an_absent_package_is_its_own_case(self, field: str) -> None:
        """``None`` is a value of its own — not the string that spells it, and
        not the empty one: a run without the FLA extra must not land where a
        run with an oddly named version of it would."""
        absent = dataclasses.replace(H100, **{field: None})
        for stand_in in ("None", "none", "", "null"):
            other = dataclasses.replace(H100, **{field: stand_in})
            assert absent.signature() != other.signature()

    def test_the_python_abi_is_part_of_it(self) -> None:
        """Triton's launcher module is keyed on its source and the platform,
        not the interpreter, so two CPython minors must not share a Triton
        directory."""
        other = dataclasses.replace(H100, python="cpython-312-x86_64-linux-gnu")
        assert other.signature() != H100.signature()
        assert "python" in FIELDS

    def test_python_abi_names_this_interpreter(self) -> None:
        tag = python_abi()
        assert tag
        assert "%d" % sys.version_info.major in tag

    def test_detect_is_none_off_cuda(self) -> None:
        assert Toolchain.detect("cpu") is None

    def test_detect_reads_a_device_list_at_its_first_device(self, monkeypatch) -> None:
        # the loaders pass ``--device`` as given, and a comma list places the
        # layers across devices (DeviceMap.parse); torch.device alone refuses
        # the list, which failed the GPU golden tests
        import torch

        asked: list[int] = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_capability",
            lambda index: asked.append(index) or (9, 0),
        )
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "GPU")
        toolchain = Toolchain.detect("cuda:1,cuda:0")
        assert toolchain is not None
        assert toolchain.capability == "9.0"
        assert asked == [1]

    def test_detect_is_none_for_a_device_list_without_cuda(self, monkeypatch) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert Toolchain.detect("cuda:0,cuda:1") is None


class TestLayout:
    def test_every_compiler_has_its_directory_under_the_signature(
        self, tmp_path
    ) -> None:
        chosen = layout(tmp_path, H100)
        home = tmp_path / H100.signature()
        assert chosen.home == home
        assert chosen.triton == home / "triton"
        assert chosen.tilelang == home / "tilelang"
        assert chosen.inductor == home / "inductor"
        assert chosen.manifest == home / "toolchain.json"
        assert chosen.environment() == {
            "TRITON_CACHE_DIR": str(home / "triton"),
            "TILELANG_CACHE_DIR": str(home / "tilelang"),
            "TORCHINDUCTOR_CACHE_DIR": str(home / "inductor"),
        }
        # every directory is under the home by construction
        assert all(path.parent == home for path in chosen.directories())

    def test_a_compile_policy_names_its_own_inductor_directory(self, tmp_path) -> None:
        plain = layout(tmp_path, H100)
        policy = layout(tmp_path, H100, inductor_policy="aten-cumsum-exp-sum")
        assert policy.triton == plain.triton
        assert policy.inductor != plain.inductor
        assert policy.inductor.name.startswith("inductor-aten-cumsum-exp-sum-")

    def test_policies_that_read_alike_never_share_a_directory(self, tmp_path) -> None:
        """The readable slug is many-to-one; the hash of the exact text is not."""
        names = {
            layout(tmp_path, H100, inductor_policy=text).inductor.name
            for text in ("aten-cumsum+exp", "aten cumsum exp", "aten/cumsum/exp")
        }
        assert len(names) == 3
        assert all(name.startswith("inductor-aten-cumsum-exp-") for name in names)

    def test_two_toolchains_never_share_a_directory(self, tmp_path) -> None:
        other = dataclasses.replace(H100, triton="3.7.1")
        a, b = layout(tmp_path, H100), layout(tmp_path, other)
        assert a.home != b.home
        assert not set(a.directories()) & set(b.directories())

    def test_the_root_may_name_the_home_directory(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        assert layout("~/cache", H100).root == tmp_path / "cache"


class TestRoot:
    """The root is opt-in: the variable alone names it — a path — and unset
    or empty leaves every compiler on its own default."""

    def test_an_unset_variable_runs_without_a_root(self) -> None:
        before = dict(os.environ)
        assert configure("cuda", toolchain=H100) is None
        assert dict(os.environ) == before

    def test_the_variable_names_the_root(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv(ENV, str(tmp_path / "mine"))
        chosen = configure("cuda", toolchain=H100)
        assert chosen is not None and chosen.root == tmp_path / "mine"
        assert chosen.triton.is_dir()

    def test_the_empty_variable_runs_without_a_root(self, monkeypatch) -> None:
        monkeypatch.setenv(ENV, "")
        assert configure("cuda", toolchain=H100) is None
        assert "TRITON_CACHE_DIR" not in os.environ

    def test_shared_is_read_off_the_roots_mode(self) -> None:
        # the group-write bit an operator set (`mkdir -m 2770`) is what makes
        # a root shared; the umask's own bits make a personal one
        assert not is_shared(0o755)
        assert is_shared(0o2770)
        assert is_shared(0o770)


class TestSharedRoot:
    """A root several users write into — one whose mode is group-writable —
    gets group-writable, setgid directories and a process umask that lets the
    compilers' own kernel directories be too."""

    def test_directories_are_group_writable_and_the_umask_allows_group_writes(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        root = tmp_path / "shared"
        root.mkdir()
        os.chmod(root, 0o2770)  # what the docs tell an operator to create
        os.umask(0o022)
        monkeypatch.setenv(ENV, str(root))
        with caplog.at_level(logging.INFO, logger=compile_cache.__name__):
            chosen = configure("cuda", toolchain=H100)
        assert chosen is not None
        for directory in (chosen.home, *chosen.directories()):
            assert stat.S_IMODE(directory.stat().st_mode) == GROUP_SHARED_MODE == 0o2770
        assert _umask() == 0o002
        assert stat.S_IMODE(chosen.manifest.stat().st_mode) == 0o660
        assert any("group-shared" in r.getMessage() for r in caplog.records)

    def test_a_tighter_umask_only_loses_the_group_write_bit(
        self, monkeypatch, tmp_path
    ) -> None:
        root = tmp_path / "shared"
        root.mkdir()
        os.chmod(root, 0o2770)
        os.umask(0o077)
        monkeypatch.setenv(ENV, str(root))
        assert configure("cuda", toolchain=H100) is not None
        assert _umask() == 0o057

    def test_a_world_writable_shared_root_is_warned_about(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        """The case that most needs the signal: whoever creates
        ``<root>/<signature>`` first supplies the artifacts every job loads."""
        root = tmp_path / "shared"
        root.mkdir()
        os.chmod(root, 0o2777)
        monkeypatch.setenv(ENV, str(root))
        with caplog.at_level(logging.WARNING, logger=compile_cache.__name__):
            chosen = configure("cuda", toolchain=H100)
        assert chosen is not None
        assert any("world-writable" in r.getMessage() for r in caplog.records)
        assert stat.S_IMODE(chosen.triton.stat().st_mode) == 0o2770

    def test_directories_created_under_a_shared_root_take_its_group(
        self, monkeypatch, tmp_path
    ) -> None:
        """A ``770`` root without setgid: ``mkdir`` would give each new
        directory its creator's primary group, and the ``2770`` applied next
        would lock the root's other users out of it."""
        root = tmp_path / "shared"
        root.mkdir()
        other = _another_group_of_ours(root)
        os.chown(root, -1, other)
        os.chmod(root, 0o770)  # group-writable, no setgid
        _skip_where_the_group_is_inherited(root, other)
        monkeypatch.setenv(ENV, str(root))
        chosen = configure("cuda", toolchain=H100)
        assert chosen is not None
        for directory in (chosen.home, *chosen.directories()):
            assert directory.stat().st_gid == other
            assert stat.S_IMODE(directory.stat().st_mode) == GROUP_SHARED_MODE
        assert chosen.manifest.stat().st_gid == other  # setgid home hands it on

    def test_a_group_that_cannot_be_adopted_only_warns(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        root = tmp_path / "shared"
        root.mkdir()
        other = _another_group_of_ours(root)
        os.chown(root, -1, other)
        os.chmod(root, 0o770)
        _skip_where_the_group_is_inherited(root, other)
        monkeypatch.setenv(ENV, str(root))

        def refusing(path, uid, gid):
            raise PermissionError(f"{path}: Operation not permitted")

        monkeypatch.setattr(os, "chown", refusing)
        with caplog.at_level(logging.WARNING, logger=compile_cache.__name__):
            chosen = configure("cuda", toolchain=H100)
        assert chosen is not None
        assert any("root's group" in r.getMessage() for r in caplog.records)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory modes")
    def test_a_shared_root_that_refuses_us_leaves_the_umask_alone(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        """Group-writable, so shared, but denied to its owner (``070``: the
        owner class is checked first): no cache, so no process-wide price."""
        root = tmp_path / "shared"
        root.mkdir()
        os.chmod(root, 0o070)
        os.umask(0o022)
        monkeypatch.setenv(ENV, str(root))
        try:
            with caplog.at_level(logging.WARNING, logger=compile_cache.__name__):
                assert configure("cuda", toolchain=H100) is None
        finally:
            os.chmod(root, 0o700)  # so tmp_path can be cleaned up
        assert _umask() == 0o022
        assert "TRITON_CACHE_DIR" not in os.environ
        assert any("not usable" in r.getMessage() for r in caplog.records)

    def test_a_personal_root_leaves_the_umask_alone(
        self, monkeypatch, tmp_path
    ) -> None:
        os.umask(0o022)
        monkeypatch.setenv(ENV, str(tmp_path / "mine"))
        assert configure("cuda", toolchain=H100) is not None
        assert _umask() == 0o022

    def test_a_manifest_that_cannot_be_written_does_not_switch_the_cache_off(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        """A home another user created whose bits refuse us: the manifest is
        documentation, the caches work without it."""
        root = tmp_path / "shared"
        root.mkdir()
        os.chmod(root, 0o2770)
        monkeypatch.setenv(ENV, str(root))

        def refusing(path, content):
            raise PermissionError(f"{path}: Operation not permitted")

        monkeypatch.setattr(compile_cache, "_write_manifest", refusing)
        with caplog.at_level(logging.WARNING, logger=compile_cache.__name__):
            chosen = configure("cuda", toolchain=H100)
        assert chosen is not None
        assert os.environ["TRITON_CACHE_DIR"] == str(chosen.triton)
        assert any("manifest" in r.getMessage() for r in caplog.records)


class TestConfigure:
    def test_nothing_happens_without_a_root(self) -> None:
        before = dict(os.environ)
        assert configure("cuda", toolchain=H100) is None
        assert dict(os.environ) == before

    def test_nothing_happens_off_cuda(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv(ENV, str(tmp_path))
        assert configure("cpu") is None
        assert not any(variable in os.environ for variable in VARIABLES.values())
        assert list(tmp_path.iterdir()) == []

    def test_the_root_from_the_environment_points_every_compiler(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setenv(ENV, str(tmp_path))
        chosen = configure("cuda", toolchain=H100)
        assert chosen is not None
        for variable, value in chosen.environment().items():
            assert os.environ[variable] == value
            assert Path(value).is_dir()
        manifest = json.loads(chosen.manifest.read_text())
        assert manifest == dataclasses.asdict(H100)

    def test_an_explicit_root_wins_over_the_environment(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setenv(ENV, str(tmp_path / "env"))
        chosen = configure("cuda", root=tmp_path / "explicit", toolchain=H100)
        assert chosen is not None and chosen.root == tmp_path / "explicit"

    def test_a_compiler_variable_set_by_hand_is_overridden_with_a_warning(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        monkeypatch.setenv(ENV, str(tmp_path))
        monkeypatch.setenv("TRITON_CACHE_DIR", "/elsewhere/triton")
        with caplog.at_level(logging.INFO, logger=compile_cache.__name__):
            chosen = configure("cuda", toolchain=H100)
        assert chosen is not None
        assert os.environ["TRITON_CACHE_DIR"] == str(chosen.triton)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "/elsewhere/triton" in message and str(tmp_path) in message
        # the message names the root, not an environment variable that may
        # not be set (an explicit `root` argument is the compile runner's path)
        assert ENV not in message

    def test_inductors_own_default_is_not_an_operators_setting(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        """Inductor writes its default directory back into the environment on
        first use; finding it there is not a hand-set value and rates INFO."""
        monkeypatch.setenv(ENV, str(tmp_path))
        monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_someone")
        monkeypatch.setattr(
            compile_cache,
            "_is_compilers_own_default",
            lambda variable, value: variable == "TORCHINDUCTOR_CACHE_DIR",
        )
        with caplog.at_level(logging.INFO, logger=compile_cache.__name__):
            assert configure("cuda", toolchain=H100) is not None
        overrides = [r for r in caplog.records if "overrides" in r.getMessage()]
        assert [r.levelno for r in overrides] == [logging.INFO]

    def test_the_real_default_check_reads_torch(self) -> None:
        """A canary on a private torch path: ``_is_compilers_own_default``
        guards its import so production degrades, but this test imports it
        bare on purpose — when a torch bump moves ``default_cache_dir``, the
        repair is to update the path there, not to guard this test, or every
        Inductor default would log at WARNING from then on."""
        from torch._inductor.runtime.cache_dir_utils import default_cache_dir

        assert compile_cache._is_compilers_own_default(
            "TORCHINDUCTOR_CACHE_DIR", default_cache_dir()
        )
        assert not compile_cache._is_compilers_own_default(
            "TORCHINDUCTOR_CACHE_DIR", "/somewhere/else"
        )
        assert not compile_cache._is_compilers_own_default(
            "TRITON_CACHE_DIR", default_cache_dir()
        )

    def test_idempotent_and_the_manifest_is_written_once(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setenv(ENV, str(tmp_path))
        first = configure("cuda", toolchain=H100)
        assert first is not None
        first.manifest.write_text(
            '{"kept": true}\n'
        )  # a later call must not rewrite it
        second = configure("cuda", toolchain=H100)
        assert second == first
        assert first.manifest.read_text() == '{"kept": true}\n'
        assert not list(first.home.glob(".toolchain-*"))  # no temporary left behind

    def test_an_empty_manifest_is_repaired(self, monkeypatch, tmp_path) -> None:
        """A zero-length file is what an unclean shutdown between the write
        and the writeback leaves; the next run replaces it."""
        monkeypatch.setenv(ENV, str(tmp_path))
        home = tmp_path / H100.signature()
        home.mkdir(parents=True)
        (home / "toolchain.json").write_bytes(b"")
        chosen = configure("cuda", toolchain=H100)
        assert chosen is not None
        assert json.loads(chosen.manifest.read_text()) == dataclasses.asdict(H100)

    def test_two_toolchains_under_one_root_keep_apart(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setenv(ENV, str(tmp_path))
        a = configure("cuda", toolchain=H100)
        b = configure(
            "cuda", toolchain=dataclasses.replace(H100, gpu="NVIDIA A100 80GB")
        )
        assert a is not None and b is not None
        assert a.home != b.home
        assert os.environ["TRITON_CACHE_DIR"] == str(b.triton)
        assert {p.name for p in tmp_path.iterdir()} == {a.signature, b.signature}

    def test_the_policy_reaches_only_inductor(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv(ENV, str(tmp_path))
        chosen = configure("cuda", toolchain=H100, inductor_policy="aten-cumsum")
        assert chosen is not None
        assert "inductor-aten-cumsum-" in os.environ["TORCHINDUCTOR_CACHE_DIR"]
        assert os.environ["TRITON_CACHE_DIR"] == str(chosen.triton)

    def test_created_directories_copy_a_personal_roots_permission_bits(
        self, monkeypatch, tmp_path
    ) -> None:
        """A pre-existing personal root's bits (here: closed to the group, so
        not shared) carry onto the directories created under it."""
        root = tmp_path / "mine"
        root.mkdir()
        os.chmod(root, 0o2750)
        monkeypatch.setenv(ENV, str(root))
        chosen = configure("cuda", toolchain=H100)
        assert chosen is not None
        for directory in (chosen.home, *chosen.directories()):
            assert stat.S_IMODE(directory.stat().st_mode) == 0o2750
        assert stat.S_IMODE(chosen.manifest.stat().st_mode) == 0o640

    def test_a_fresh_root_follows_the_umask_and_its_manifest_is_readable(
        self, monkeypatch, tmp_path
    ) -> None:
        """A root created here has no bits to copy: the tree follows the
        umask, and so does the manifest — not ``mkstemp``'s owner-only mode."""
        monkeypatch.setenv(ENV, str(tmp_path / "fresh"))
        os.umask(0o022)
        chosen = configure("cuda", toolchain=H100)
        assert chosen is not None
        assert stat.S_IMODE(chosen.manifest.stat().st_mode) == 0o644
        assert stat.S_IMODE(chosen.triton.stat().st_mode) == 0o755

    def test_a_root_created_world_writable_is_warned_about_too(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        """Tooling running under ``umask 000`` creates the root itself, so
        the warning must not depend on the root having existed already."""
        monkeypatch.setenv(ENV, str(tmp_path / "fresh"))
        os.umask(0)
        with caplog.at_level(logging.WARNING, logger=compile_cache.__name__):
            chosen = configure("cuda", toolchain=H100)
        assert chosen is not None
        assert stat.S_IMODE(chosen.root.stat().st_mode) & stat.S_IWOTH
        assert any("world-writable" in r.getMessage() for r in caplog.records)
        # a world-writable root is group-writable, so it is shared, and the
        # tree under it — where the artifacts live — is closed to others
        assert stat.S_IMODE(chosen.triton.stat().st_mode) == GROUP_SHARED_MODE

    def test_a_directory_another_job_created_is_left_alone(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        """The loser of a create race must not chmod the winner's directory,
        and a chmod refused (another owner) must not disable the cache."""
        root = tmp_path / "shared"
        root.mkdir()
        os.chmod(root, 0o2770)
        home = root / H100.signature()
        (home / "triton").mkdir(parents=True)  # "the other job got there first"
        os.chmod(home / "triton", 0o755)
        monkeypatch.setenv(ENV, str(root))
        real_chmod = os.chmod

        def refusing_chmod(path, mode, *args, **kwargs):
            if Path(path) == home / "tilelang":
                raise PermissionError("Operation not permitted")
            return real_chmod(path, mode, *args, **kwargs)

        monkeypatch.setattr(os, "chmod", refusing_chmod)
        with caplog.at_level(logging.WARNING, logger=compile_cache.__name__):
            chosen = configure("cuda", toolchain=H100)
        assert chosen is not None
        assert os.environ["TRITON_CACHE_DIR"] == str(chosen.triton)
        # the pre-existing directory keeps the bits its creator gave it
        assert stat.S_IMODE((home / "triton").stat().st_mode) == 0o755
        assert not any("not usable" in r.getMessage() for r in caplog.records)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory modes")
    def test_an_unusable_root_warns_and_leaves_the_compilers_alone(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        """A performance setting never fails a model load: a root that cannot
        be written is logged and the compilers keep their own defaults."""
        blocked = tmp_path / "blocked"
        blocked.mkdir()
        os.chmod(blocked, 0o500)
        monkeypatch.setenv(ENV, str(blocked))
        monkeypatch.setenv("TRITON_CACHE_DIR", "/kept/triton")
        try:
            with caplog.at_level(logging.WARNING, logger=compile_cache.__name__):
                assert configure("cuda", toolchain=H100) is None
        finally:
            os.chmod(blocked, 0o700)
        assert os.environ["TRITON_CACHE_DIR"] == "/kept/triton"
        assert "TILELANG_CACHE_DIR" not in os.environ
        assert any("not usable" in r.getMessage() for r in caplog.records)

    def test_racing_writers_leave_one_whole_manifest(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setenv(ENV, str(tmp_path))
        errors: list[BaseException] = []

        def run() -> None:
            try:
                configure("cuda", toolchain=H100)
            except BaseException as err:  # pragma: no cover - reported below
                errors.append(err)

        threads = [threading.Thread(target=run) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        home = tmp_path / H100.signature()
        assert json.loads((home / "toolchain.json").read_text()) == dataclasses.asdict(
            H100
        )
        assert not list(home.glob(".toolchain-*"))


class TestLoadersCallIt:
    """The loaders are the delivery mechanism: a model landing on its device
    configures the cache before its first forward, on both engines' paths.

    Each test clears the loader's process-wide cache so the spy sees a real
    load; a later test in the same session reloads the tiny model once
    (about a second), the price of pinning the call."""

    def test_the_hooks_loader_and_the_caller_owned_path(self, monkeypatch) -> None:
        from causalab.neural.engines.pytorch_hooks import loading
        from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA

        calls: list[str] = []
        monkeypatch.setattr(
            loading, "configure_compile_cache", lambda device: calls.append(device)
        )
        loading.load_model.cache_clear()
        bundle = loading.load_model(TINY_LLAMA, device="cpu")
        assert calls == ["cpu"]
        loading.ModelBundle.from_model(
            bundle.model,
            bundle.tokenizer,
            key=TINY_LLAMA,
            revision="main",
            dtype="fp32",
        )
        assert calls == ["cpu", "cpu"]

    def test_the_nnsight_loader(self, monkeypatch) -> None:
        pytest.importorskip("nnsight")
        from causalab.neural.engines.nnsight_tracing import loading
        from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA

        calls: list[str] = []
        monkeypatch.setattr(
            loading, "configure_compile_cache", lambda device: calls.append(device)
        )
        loading.load_model.cache_clear()
        loading.load_model(TINY_LLAMA, device="cpu")
        assert calls == ["cpu"]
