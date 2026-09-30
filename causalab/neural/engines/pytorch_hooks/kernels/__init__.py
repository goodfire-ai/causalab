"""Fused Triton kernels for the hooks engine.

Each kernel has a Torch reference that specifies operation order and a
plan that checks device and shape support before use.
"""
