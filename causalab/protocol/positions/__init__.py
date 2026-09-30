"""Resolve token positions for intervention specifications.

The caller supplies a tokenizer. Position frames hold Python values so the
protocol layer can describe addresses independently of device tensors. The
service handles plain text and chat frames, semantic spans, pair alignment,
and location ledgers."""

from __future__ import annotations
