"""Neural execution for compiled interventions.

``engines`` supplies PyTorch hooks and nnsight tracing. Both use ``shared``
for protocol execution; ``shared.engine_router`` resolves the engine choice.
Authoring helpers live in ``causalab.analysis.sequences``; task position
helpers live in ``causalab.tasks.token_positions``.
"""
