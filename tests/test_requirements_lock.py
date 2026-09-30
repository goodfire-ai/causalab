"""``requirements.lock.txt`` is a hash-locked, resolver-free install.

`uv.lock` already pins every version, but only `uv` reads it. A consumer
installing causalab into an existing environment — a cluster job image, a
reviewer reproducing a number — has neither `uv` nor a reason to let a resolver
choose anything, and reproducing a number needs exactly what the missing file
provides: every dependency pinned, every artifact named by its SHA-256, no
resolution step.

CI regenerating and diffing is the real guard against staleness. These tests
guard the other failure, which a diff cannot see because a hand-edit changes
both sides of it: whether the file *is* what it claims to be. They are
near-instant — parsing 2.4k lines of text, no install, no network — so they run
in the ordinary unit tier while the actual clean-venv install is the separate
check that `docs/standalone_install.md` describes.

The first of them is not hypothetical: the initial export carried ~450 ANSI
escape sequences, because `uv` colours its comment lines even when stdout is a
pipe. Invisible in a terminal, invisible in review, and a `pip` parse error for
the one consumer the file exists for.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "requirements.lock.txt"
SCRIPT = REPO / "scripts/export_requirements_lock.py"

#: A requirement line: `name==version` optionally followed by a marker and the
#: line-continuation backslash the hashes hang off.
REQUIREMENT = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;]+)")

#: Lines `uv export` legitimately emits that are not requirements. An index URL
#: is not a resolution step — `pip` still installs only what the file pins — but
#: it *is* a fetch target, so it is listed here rather than pattern-matched
#: away. None appear today; the first `[[tool.uv.index]]` block (a CUDA index,
#: say) would add one, and this list is where that becomes a decision rather
#: than a test that reads as "the file is broken".
OPTIONS = ("--hash=", "--index-url", "--extra-index-url", "--find-links", "--no-binary")


def _requirements() -> dict[str, list[str]]:
    """Every pinned requirement, mapped to the hash lines that follow it.

    Keyed on the pin *as written*, not on the name: a universal export emits
    the same package more than once under mutually exclusive markers (four do
    today — `contourpy`, `networkx`, `numpy`, `scipy` — split on the Python
    version), and merging those blocks under one name would let a hashless
    second block hide behind its twin's hashes while `pip --require-hashes`
    refuses the whole file.
    """
    out: dict[str, list[str]] = {}
    current: str | None = None
    for raw in LOCK.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = REQUIREMENT.match(line)
        if match:
            current = line.rstrip(" \\")
            out.setdefault(current, [])
            # a hash may sit on the pin's own line
            if "--hash=" in line:
                out[current].append(line)
            continue
        if line.startswith("--hash=") and current is not None:
            out[current].append(line)
    return out


def test_the_lock_exists_and_was_parsed() -> None:
    """A census over a file the parser failed to read passes vacuously."""
    assert LOCK.exists(), (
        f"{LOCK.name} is missing — see scripts/export_requirements_lock.py"
    )
    assert len(_requirements()) > 50, "parsed implausibly few requirements"


def test_marker_split_pins_are_kept_apart() -> None:
    """The parser's own precondition: a package exported twice under exclusive
    markers is two pins, each answering for its own hashes. Collapsing them by
    name is how `test_every_requirement_names_at_least_one_hash` would go
    vacuous in the one shape this file actually has."""
    pins = _requirements()
    numpy_pins = sorted(
        pin for pin in pins if REQUIREMENT.match(pin).group("name") == "numpy"
    )
    assert len(numpy_pins) >= 2, (
        f"expected numpy pinned once per Python-version marker, found {numpy_pins}"
    )
    for pin in numpy_pins:
        assert pins[pin], f"no --hash= for: {pin!r}"


def test_the_lock_is_plain_ascii_text_pip_can_parse() -> None:
    """The bug this file was born from: `uv` colours comment lines even into a
    pipe, so the export carried ANSI escapes until `--color never` was passed.

    Checked as bytes, because the escapes render invisibly.
    """
    data = LOCK.read_bytes()
    assert b"\x1b" not in data, (
        f"{LOCK.name} carries {data.count(bytes([0x1B]))} ANSI escape "
        "sequences — "
        "regenerate it with scripts/export_requirements_lock.py"
    )
    assert b"\r\n" not in data, "CRLF line endings would break the continuations"


def test_every_requirement_is_exactly_pinned() -> None:
    """No range, no compatible-release, no bare name. A resolver-free install
    means the file leaves nothing to resolve."""
    for raw in LOCK.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(OPTIONS):
            continue
        assert REQUIREMENT.match(line), (
            f"{LOCK.name} has a line that is not an exact pin: {line!r}"
        )


def test_every_requirement_names_at_least_one_hash() -> None:
    """`pip install --require-hashes` refuses the whole file if any single
    requirement lacks a hash, so one gap disables the feature entirely."""
    missing = sorted(pin for pin, hashes in _requirements().items() if not hashes)
    assert not missing, f"no --hash= for: {missing}"


def test_every_hash_is_a_sha256() -> None:
    for pin, hashes in _requirements().items():
        for entry in hashes:
            for found in re.findall(r"--hash=(\S+)", entry):
                assert re.fullmatch(r"sha256:[0-9a-f]{64}", found), (
                    f"{pin!r} carries a hash that is not a sha256: {found!r}"
                )


def test_nothing_unhashable_reached_the_lock() -> None:
    """The reason the extras are excluded, asserted rather than trusted.

    `nnsight` is a **git** dependency: a git revision has no artifact hash, so a
    single such entry makes `--require-hashes` impossible for every consumer.
    Editable and local-path entries are the same problem in a different
    spelling, and the project itself is excluded for exactly that reason
    (`--no-emit-project`).
    """
    text = LOCK.read_text(encoding="utf-8")
    for forbidden in ("git+", " @ ", "-e ", "file://"):
        assert forbidden not in text, (
            f"{LOCK.name} contains {forbidden!r} — an unhashable requirement "
            "disables --require-hashes for the whole file"
        )
    assert "nnsight" not in text
    assert "\ncausalab==" not in text and not text.startswith("causalab==")


def test_optional_attention_packages_stay_out_of_the_base_lock() -> None:
    """A normal install must not build optional GPU extensions."""
    names = {REQUIREMENT.match(pin).group("name") for pin in _requirements()}
    assert names.isdisjoint(
        {"flash-attn", "flash-linear-attention", "fla-core", "causal-conv1d"}
    )


def test_the_lock_is_universal_not_per_platform() -> None:
    """It has to be byte-identical whatever machine regenerates it, or the CI
    diff is a guarantee of failure rather than a check.

    `uv export` emits the lockfile's whole marker set, so the linux-only CUDA
    wheels and the win32-only packages are present *with their markers* even
    when the export ran on macOS. Their absence would mean the export narrowed
    to one platform.
    """
    text = LOCK.read_text(encoding="utf-8")
    assert re.search(r"^nvidia-\S+==\S+ ; .*sys_platform == 'linux'", text, re.M), (
        "no marker-guarded linux CUDA wheel — the export looks platform-specific"
    )
    assert "sys_platform == 'win32'" in text


def test_the_lock_says_how_to_regenerate_itself() -> None:
    """A generated file a reader cannot regenerate is a file that will be
    hand-edited."""
    head = LOCK.read_text(encoding="utf-8")[:500]
    assert "scripts/export_requirements_lock.py" in head
    assert "do not hand-edit" in head


def test_the_check_reports_rather_than_crashes_under_a_legacy_locale(
    tmp_path: Path,
) -> None:
    """The header carries an em dash, and `Path.read_text()` with no encoding
    takes the locale's. Under a C locale with UTF-8 mode off that is ASCII, and
    the script raised `UnicodeDecodeError` on the committed file instead of
    saying whether it was stale — a crash, in a bare container or a cron shell,
    dressed as a red gate.

    Run as a subprocess because the encoding is fixed at interpreter start.
    Both `LC_ALL=C` and `PYTHONUTF8=0` are needed to reproduce: since 3.7 the
    bare C locale switches UTF-8 mode *on* (PEP 540), so `LC_ALL=C` alone would
    pass with or without the fix. `export()` is stubbed to return the committed
    text, so this is about the encoding of the read, the write and the report
    — not about `uv` being on PATH or the lock being current.
    """
    program = textwrap.dedent(
        """
        import importlib.util, pathlib, sys
        script, lock, committed = (pathlib.Path(a) for a in sys.argv[1:])
        spec = importlib.util.spec_from_file_location("export_lock", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.LOCK = lock
        module.export = lambda: committed.read_bytes().decode("utf-8")
        wrote = module.main([])             # write path: the em dash goes out
        current = module.main(["--check"])  # read path: and comes back
        lock.write_bytes(lock.read_bytes() + b"stale==0\\n")
        stale = module.main(["--check"])    # report path: a diff and a message
        print("RC", wrote, current, stale)
        """
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
    env.update(LC_ALL="C", LANG="C", LC_CTYPE="C", PYTHONUTF8="0")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(SCRIPT),
            str(tmp_path / "lock.txt"),
            str(LOCK),
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "RC 0 0 1" in result.stdout, result.stdout
    assert "is current" in result.stdout
    assert "is stale" in result.stdout
