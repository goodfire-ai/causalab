"""Missing standalone controller entrypoints are refused before deployment."""

import subprocess

import pytest

from causalab.measurement.deployment import remote

pytestmark = pytest.mark.unit


def test_launch_requires_supervisor_in_controller_archive(tmp_path, monkeypatch):
    repository = tmp_path / "controller"
    transfer = repository / "causalab/measurement/deployment/transfer.py"
    transfer.parent.mkdir(parents=True)
    transfer.write_text("# Only the transfer entrypoint exists in this revision.\n")
    for args in (
        ["init", "--quiet"],
        ["add", "."],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "test controller",
        ],
    ):
        subprocess.run(["git", "-C", str(repository), *args], check=True)
    monkeypatch.setattr(remote, "__file__", str(transfer.with_name("remote.py")))

    class NoNetwork:
        def __init__(self, host):
            pass

        def call(self, *args, **kwargs):
            pytest.fail("controller sources must be validated before SSH")

    monkeypatch.setattr(remote, "SSH", NoNetwork)
    receipt = tmp_path / "job.json"
    args = remote.parser().parse_args(
        [
            "launch",
            str(tmp_path / "study.json"),
            "--bindings",
            str(tmp_path / "bindings.json"),
            "--host",
            "unused",
            "--receipt",
            str(receipt),
        ]
    )
    with pytest.raises(subprocess.CalledProcessError) as error:
        remote.launch(args)
    assert error.value.cmd[-1].endswith(":causalab/remote/supervisor.py")
    assert not receipt.exists()
