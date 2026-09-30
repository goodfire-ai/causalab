"""The package's one shared precondition: no default ``torch.distributed``
group before or after any test here. ``test_mesh.py`` initialises torch's
``fake`` backend as a default group and its fixture destroys it;
``test_serving.py`` asserts on the group's absence. A group that leaked
would otherwise fail whichever test next read the state, naming the wrong
subject — this fixture names the test that left it, and clears it so the
rest of the package is judged on its own."""

from __future__ import annotations

from typing import Iterator

import pytest
import torch.distributed as dist


@pytest.fixture(autouse=True)
def _no_default_process_group(request: pytest.FixtureRequest) -> Iterator[None]:
    assert not dist.is_initialized(), (
        f"a default process group was left initialised before {request.node.nodeid}"
    )
    yield
    if dist.is_initialized():
        dist.destroy_process_group()
        pytest.fail(f"{request.node.nodeid} left the default process group initialised")
