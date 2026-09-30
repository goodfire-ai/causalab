"""Public measurement imports must stay usable without numerical dependencies."""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_public_collection_and_script_packages_are_lightweight():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import sys

from causalab.measurement import Operation, collect
from causalab.measurement import collection

assert Operation is collection.Operation
assert collect is collection.collect
for name in ('study', 'runtime', 'capture', 'analysis', 'deployment'):
    importlib.import_module('causalab.measurement.' + name)
assert not {'torch', 'numpy', 'safetensors'} & sys.modules.keys()
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
