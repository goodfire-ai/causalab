"""Readers for *foreign* SAE checkpoints.

``causalab.io`` is the lowest application layer (docs/CODEBASE.md invariant 3):
it reads raw tensors off disk and returns plain tensors + metadata dicts. This
module is the sanctioned home for reading **foreign** SAE checkpoints — third-
party ``.pt`` blobs that are NOT causalab's own (safetensors + ``.meta.json``)
artifacts, so the ``torch.load`` ban in [`causalab.io.artifacts`][]
(docs/CODEBASE.md serialization policy) does not apply here. Reading a foreign
checkpoint with ``weights_only=False`` is the documented exception.

Two readers live here:

- [`read_sae_decoder`][] — a vanilla SAE's ``decoder.weight`` (one decoder
  direction per ``d_sae`` index). The cluster path stacks several directions
  into a basis via ``methods.sae.decoder_subspace``; that basis math stays in
  ``methods/`` — this module only reads the raw tensor.
- [`load_block_sae_frame`][] — one block of a block/Grassmannian SAE
  (``GrassmannianCoderSparse``). Each block is already a K-dim orthonormal
  Stiefel subspace ``B_raw[block_id]`` of shape ``(d_model, K)``, so this returns
  it directly as a ``(d_model, k_alive)`` frame — no basis math, no QR, no GPU,
  and no dependency on the external block-SAE training repo.

Both use ``weights_only=False`` and therefore assume the checkpoint comes from a
**trusted source**: unpickling can execute arbitrary code, so never point these
at user-uploaded or otherwise untrusted files.
"""

from __future__ import annotations

import logging
import pathlib
import re
import sys

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _load_foreign_checkpoint(checkpoint_path: str) -> dict:
    """``torch.load`` a foreign SAE checkpoint dict (trusted source; module docstring).

    Returns the top-level dict. Raises ``ValueError`` if the blob is not a dict.
    """
    # Checkpoints pickled on Python >=3.12 reference ``pathlib._local``, absent
    # on 3.10. Register the alias before unpickling for cross-version pickle
    # compatibility. This is an intentional process-global side effect — leave
    # it in place: on 3.12+ the stdlib ships its own ``pathlib._local`` so the
    # ``setdefault`` is a no-op; only on <3.12 does the alias take effect.
    sys.modules.setdefault("pathlib._local", pathlib)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"Unexpected SAE checkpoint format in {checkpoint_path}: "
            f"top-level object is {type(ckpt).__name__}, expected a dict."
        )
    return ckpt


def read_sae_decoder(checkpoint_path: str) -> tuple[Tensor, int | None]:
    """Read the decoder weight (and ``d_model`` hint) from a vanilla SAE checkpoint.

    Pulls ``decoder.weight`` directly from the checkpoint's ``state_dict`` — no
    SAE class needed. Returns the raw decoder tensor in
    its on-disk orientation plus ``d_model`` from ``model_config`` when present
    (``None`` otherwise); ``methods.sae.decoder_subspace`` resolves orientation.

    Trusted-source ``weights_only=False`` (see module docstring).
    """
    ckpt = _load_foreign_checkpoint(checkpoint_path)
    state_dict = ckpt.get("state_dict", ckpt)
    if "decoder.weight" not in state_dict:
        raise KeyError(
            f"No 'decoder.weight' in {checkpoint_path}. "
            f"state_dict keys: {sorted(state_dict)[:20]}"
        )
    decoder = state_dict["decoder.weight"]
    model_config = ckpt.get("model_config", ckpt.get("config", {}))
    d_model = model_config.get("d_model") if isinstance(model_config, dict) else None
    return decoder, (int(d_model) if d_model is not None else None)


def _parse_layer_from_config(config: dict) -> int | None:
    """Best-effort residual-stream layer index from a block-SAE config.

    Block-SAE checkpoints do not store the layer explicitly, but the training
    ``config`` records the activation source — e.g. ``dataset='model.layers.19'``
    or ``source='.../fineweb-L19'``. Parse the first match; ``None`` when nothing
    resolves (the caller then supplies the layer).
    """
    if not isinstance(config, dict):
        return None
    dataset = config.get("dataset")
    if isinstance(dataset, str):
        m = re.search(r"layers?[._/](\d+)", dataset)
        if m:
            return int(m.group(1))
    for key in ("source", "tag"):
        val = config.get(key)
        if isinstance(val, str):
            m = re.search(r"\bL(\d+)\b", val)
            if m:
                return int(m.group(1))
    return None


def load_block_sae_frame(checkpoint_path: str, block_id: int) -> tuple[Tensor, dict]:
    """Return one block of a block/Grassmannian SAE as a ``(d_model, k_alive)`` frame.

    A block-SAE (``GrassmannianCoderSparse``) stores ``state_dict["B_raw"]`` of
    shape ``(n_groups, d_model, K)``; each block ``B_raw[block_id]`` is already a
    K-dim orthonormal Stiefel frame (the SAE trains with ``enforce_ortho=True``).
    This returns that frame restricted to its **alive** columns
    (``dim_mask[block_id] > 0``; faithfulness step F3) and unit-normalized per
    column (F1 — exact for K=1, a no-op when columns are already orthonormal).

    ``block_id`` indexes the ``n_groups`` axis — one block = one K-dim subspace.
    This is deliberately NOT ``feature_id`` (which, in [`read_sae_decoder`][]'s
    world, indexes the ``d_sae`` axis = a single decoder direction); conflating
    the two would be a category error.

    The returned frame is the **subspace span only**. It does NOT reproduce the
    SAE's gating or biases (``b_gate`` + top-l0 gate, ``b_dec``, ``log_gamma``):
    those are not part of the subspace geometry. A fixed-subspace IIA on this
    frame tests *"the subspace's coordinates,"* not *"the gated SAE-feature
    activation"* (reproducing the gated activation needs the external SAE class +
    GPU; out of scope here). The activation rescale (``scale``) commutes with a
    linear projection and a same-space interchange, so it does not change the
    subspace; it is returned in ``meta`` for magnitude-sensitive consumers.

    Args:
        checkpoint_path: Path to a block/Grassmannian SAE ``.pt``.
        block_id: Index into the ``n_groups`` axis (one K-dim Stiefel subspace).

    Returns:
        ``(frame, meta)`` — ``frame`` is ``(d_model, k_alive)`` with orthonormal
        columns; ``meta`` carries ``{d_model, group_size, n_groups, scale, layer,
        hook_site, model_id, b_gate_block, dim_mask_block, has_b_dec,
        has_log_gamma}`` (``hook_site``/``model_id`` are ``None`` — not stored in
        the checkpoint; the caller supplies them).

    Pure tensor read: no GPU, no model load, no dependency on the external
    block-SAE repo (``config``/``model_kwargs`` are plain dicts, ``state_dict``
    is plain tensors). Trusted-source ``weights_only=False`` (module docstring).
    """
    ckpt = _load_foreign_checkpoint(checkpoint_path)
    state_dict = ckpt.get("state_dict", ckpt)
    if "B_raw" not in state_dict:
        raise KeyError(
            f"No 'B_raw' in {checkpoint_path}; this does not look like a block/"
            f"Grassmannian SAE checkpoint. state_dict keys: {sorted(state_dict)[:20]}"
        )
    b_raw = state_dict["B_raw"]
    if b_raw.ndim != 3:
        raise ValueError(
            "Expected B_raw of shape (n_groups, d_model, K); got "
            f"{tuple(b_raw.shape)} in {checkpoint_path}."
        )
    n_groups, d_model, group_size = (
        int(b_raw.shape[0]),
        int(b_raw.shape[1]),
        int(b_raw.shape[2]),
    )
    bid = int(block_id)
    if not 0 <= bid < n_groups:
        raise IndexError(
            f"block_id={bid} out of range for {n_groups} blocks in {checkpoint_path}."
        )
    frame = b_raw[bid].to(torch.float32)  # (d_model, K)

    # F3: restrict to alive dims. `dim_mask` zeroes dead columns within a block,
    # so the effective subspace spans only the alive columns.
    dim_mask = state_dict.get("dim_mask")
    dim_mask_block = None
    if dim_mask is not None:
        block_mask = dim_mask[bid]
        dim_mask_block = block_mask.to(torch.float32).tolist()
        alive = block_mask > 0  # (K,)
        if int(alive.sum()) == 0:
            raise ValueError(
                f"block_id={bid} in {checkpoint_path} has no alive dims "
                "(dim_mask all zero); it does not define a subspace."
            )
        frame = frame[:, alive]

    # F1: unit-normalize columns. Block frames are already orthonormal
    # (enforce_ortho=True), so this is a no-op for K>1 and exactly the K=1
    # normalization. We do NOT QR / cross-orthogonalize here — basis math stays
    # in methods/, and these frames are orthonormal by construction.
    col_norms = frame.norm(dim=0, keepdim=True)
    if torch.any(col_norms <= 1e-12):
        raise ValueError(
            f"block_id={bid} in {checkpoint_path} has a zero-norm column; "
            "cannot unit-normalize."
        )
    frame = (frame / col_norms).contiguous()

    config = ckpt.get("config")
    config = config if isinstance(config, dict) else {}
    model_kwargs = ckpt.get("model_kwargs")
    model_kwargs = model_kwargs if isinstance(model_kwargs, dict) else {}

    def _first(*vals: object) -> object:
        for v in vals:
            if v is not None:
                return v
        return None

    def _meta_scalar(v: object) -> object:
        """Coerce a metadata value to a JSON-friendly form without assuming it is scalar.

        Foreign checkpoints don't guarantee a scalar-per-block gate or a global
        scalar ``scale``: a *per-feature* gate has shape ``(n_groups, group_size)``,
        so ``b_gate[bid]`` is then a length-K vector, not a scalar. Convert a
        0-d / 1-element tensor to ``float``, a multi-element tensor to a list, and
        pass Python numbers through. This records a non-scalar field faithfully
        instead of letting ``float(tensor)`` crash the entire frame load.
        """
        if v is None:
            return None
        if torch.is_tensor(v):
            return float(v) if v.numel() == 1 else v.detach().to(torch.float32).tolist()
        return float(v)

    b_gate = state_dict.get("b_gate")
    meta = {
        "d_model": int(_first(model_kwargs.get("d"), config.get("d"), d_model)),
        "group_size": int(
            _first(model_kwargs.get("group_size"), config.get("group_size"), group_size)
        ),
        "n_groups": int(
            _first(model_kwargs.get("n_groups"), config.get("n_groups"), n_groups)
        ),
        "scale": _meta_scalar(ckpt.get("scale")),
        "layer": _parse_layer_from_config(config),
        # hook_site / model_id are not stored in these checkpoints; the caller
        # supplies them (block SAEs train on resid_post of a base model).
        "hook_site": None,
        "model_id": None,
        "b_gate_block": _meta_scalar(b_gate[bid]) if b_gate is not None else None,
        "dim_mask_block": dim_mask_block,
        "has_b_dec": "b_dec" in state_dict,
        "has_log_gamma": "log_gamma" in state_dict,
    }
    logger.info(
        "Loaded block-SAE frame: block_id=%d -> (d_model=%d, k_alive=%d) from %s "
        "(n_groups=%d, K=%d).",
        bid,
        int(frame.shape[0]),
        int(frame.shape[1]),
        checkpoint_path,
        n_groups,
        group_size,
    )
    return frame, meta
