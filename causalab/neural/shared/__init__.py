"""Services shared by the execution engines.

These modules handle frames, tensor layouts, write math, metrics, results,
and the loaded model's component addresses. ``engine_router`` resolves the
``--engine`` choice and constructs the engine lazily, so it imports without
torch. Engines supply model loading and the operations that capture tensors
and apply writes.
"""
