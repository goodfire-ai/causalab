"""A module standing in for one this rank does not hold (``docs/model_parallelism.md`` §6.5).

Under pipeline placement a rank keeps its stage's layers and replaces the
rest by an identity. The identity alone would leave the site resolver, the
stream table and the placement lookup blind on a block another stage owns —
``blocks[layer].self_attn`` has to exist for a site there to resolve to *a*
module with a stable identity, the mixer child has to be named for the
stream check, a projection's ``out_features`` has to be readable for the
interior's width rule. [`Shadowed`][] is the identity that also carries
the replaced module as its **shadow**: every attribute the stand-in lacks
falls through to it (the ``_StandIn`` pattern the resume swap uses), while
the shadow is registered neither as a child nor in the state dict — the
loader never materialises it, the device map never sees it, and a forward
through the stand-in is the identity.

Kept in the shared layer because the shared resolver and the placement
lookup (``placements.module_path``) read it; the engine's pipeline placement
composes it with transformers' own ``PipelineIdentityLayer``.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["Shadowed"]


class Shadowed(torch.nn.Identity):
    """An identity module whose missing attributes fall through to the module
    it replaced (module docstring). The shadow is held in a plain list so it
    is not registered as a child: ``parameters()``, ``state_dict()`` and
    ``named_modules()`` see an empty identity."""

    def __init__(self, shadow: torch.nn.Module) -> None:
        super().__init__()
        self._shadow = [shadow]

    @property
    def shadow(self) -> torch.nn.Module:
        return self._shadow[0]

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            # read off ``__dict__``: were the list ever missing, ``self._shadow``
            # would recurse into this method instead of raising
            shadow = self.__dict__.get("_shadow")
            if shadow is None:
                raise
            return getattr(shadow[0], name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """The first argument, as transformers' ``PipelineIdentityLayer`` returns it."""
        return args[0] if args else next(iter(kwargs.values()))
