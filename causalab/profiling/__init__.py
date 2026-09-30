"""Native profiler adapters; importing the registry never imports torch."""

from __future__ import annotations

from importlib import import_module

from .base import CaptureRequest, ProfilerBackend, ProfilerUnavailable

__all__ = ["CaptureRequest", "ProfilerBackend", "ProfilerUnavailable", "get_backend"]


def get_backend(name: str) -> ProfilerBackend:
    if name not in {"torch", "nsys", "ncu"}:
        raise ValueError(f"unknown profiler backend: {name!r}")
    return import_module(f"{__name__}.{name}").BACKEND
