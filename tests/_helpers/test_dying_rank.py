"""The dying rank's modes (``tests/_helpers/dying_rank.py``): torch-free at
import — a spawn parent that hands it in as ``entry`` must stay so —, the
five modes spelled by one variable and by a leading flag, an unknown word
refused by name, and a process that is not the victim armed with nothing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests._helpers import dying_rank

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]


def test_the_module_imports_no_torch() -> None:
    probe = (
        "import sys; import tests._helpers.dying_rank as d; "
        "print(sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]", out.stdout


def test_the_modes_and_their_flags() -> None:
    assert dying_rank.MODES == (
        "exit",
        "stop",
        "wedge",
        "exit-before-join",
        "exit-after-join",
        "mark",
    )
    assert dying_rank.FLAGS == {f"--{m}": m for m in dying_rank.MODES}
    assert dying_rank.mode_from({}) == "exit"
    assert dying_rank.mode_from({dying_rank.MODE_VARIABLE: "wedge"}) == "wedge"
    with pytest.raises(
        ValueError, match="CAUSALAB_TEST_DYING_MODE='hang' is not one of"
    ):
        dying_rank.mode_from({dying_rank.MODE_VARIABLE: "hang"})


def test_a_leading_flag_sets_the_mode_for_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    seen: list[list[str]] = []
    # ``setenv`` first, so the fixture's teardown restores the variable to
    # *absent*: ``main`` writes ``os.environ`` itself, and a ``delenv`` after
    # that write would be undone at teardown — the value it deleted put
    # back — leaving every later spawned child of this process a stopped
    # victim (the two watchdog smokes then fail their bounds)
    monkeypatch.setenv(dying_rank.MODE_VARIABLE, "placeholder")
    del os.environ[dying_rank.MODE_VARIABLE]
    monkeypatch.setattr(dying_rank, "entry", lambda argv: seen.append(list(argv)) or 7)
    assert dying_rank.main(["--stop", "run", "doc.json"]) == 7
    assert seen == [["run", "doc.json"]]
    assert os.environ[dying_rank.MODE_VARIABLE] == "stop"
    del os.environ[dying_rank.MODE_VARIABLE]
    assert dying_rank.main(["run", "--parallel", "tp=2"]) == 7
    assert dying_rank.MODE_VARIABLE not in os.environ, "no flag: the environment's word"


def test_a_process_that_is_not_the_victim_is_armed_with_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(dying_rank.RANK_VARIABLE, raising=False)
    assert dying_rank.arm() is None
    monkeypatch.setenv(dying_rank.RANK_VARIABLE, "1")
    monkeypatch.setenv("RANK", "0")
    assert dying_rank.arm() is None


def test_the_mark_records_where_and_when_and_the_mark_mode_runs_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """``_act`` in the ``mark`` mode writes the events' clock and the place
    into the mark file and returns; the file is what the placement test
    (``test_dying_rank_placement_run.py``) reads."""
    from datetime import datetime

    mark = tmp_path / "mark"
    monkeypatch.setenv(dying_rank.MARK_VARIABLE, str(mark))
    dying_rank._act(
        "1", "mark", "after its first collective (rowwise reduce of Linear(16->16))"
    )  # pyright: ignore[reportPrivateUsage]
    stamp, _, where = mark.read_text().rstrip("\n").partition(" ")
    assert datetime.fromisoformat(stamp).tzinfo is not None
    assert where == "after its first collective (rowwise reduce of Linear(16->16))"
    assert "dying rank 1: marking the moment and running on" in capsys.readouterr().err
