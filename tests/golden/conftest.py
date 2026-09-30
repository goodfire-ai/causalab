"""One model resident per golden module.

The golden tier is one ``pytest -n 0 -m golden`` process on one accelerator,
module after module in collection order. The engine loaders cache up to four
models each (``functools.lru_cache``), so a module that loads through them —
every paper golden does, through its protocol run — leaves those models alive
for the next module; the per-test reclamation in ``tests/conftest.py``
frees only dead ones. Before this fixture, the two ``Qwen/Qwen3.6-35B-A3B``
modules protected the module *after* them by clearing the cache in their own
teardown, and were protected themselves only by sorting before
``test_paper_goldens.py`` — ``test_readout_a3b.py`` does not, and hit a CUDA
OOM with the paper goldens' GPT-2 XL and Llama-3.1-8B still cached.

So the invariant is enforced here, at the boundary, for every golden module:
the loader caches are emptied and the accelerator drained before a module's
first fixture runs and again after its last one is finalized. A module fixture
that loads a model needs no teardown of its own. Gated on the ``golden`` marker
(module-level ``pytestmark``) so the CPU guards beside these suites — and the
CPU tiers' shared tiny-model fixtures — keep the cache they rely on.
"""

from __future__ import annotations

import pytest

from tests._helpers.resident_models import evict_resident_models


@pytest.fixture(autouse=True, scope="module")
def _one_model_resident(request: pytest.FixtureRequest):
    if request.node.get_closest_marker("golden") is None:
        yield
        return
    evict_resident_models()
    yield
    evict_resident_models()
