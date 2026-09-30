"""Operation microbenchmarks over a real causal LM and Causalab featurizers.

This is a compute-path harness, not the protocol compiler / dataset / disk
pipeline. The injected model makes the same paths testable on tiny CPU models.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from causalab.neural.shared.featurizers import Gate, Subspace
from causalab.sol.benchmark import Case
from causalab.sol.model import Workload, validate_integer
from causalab.sol.qwen36_v2 import CONTRACT


class _CaptureComplete(Exception):
    """Private control flow: capture has finished before the unused suffix."""


class _CachedBlock(torch.nn.Module):
    def forward(self, hidden_states, **_kwargs):
        return hidden_states


class TorchOperations:
    def __init__(
        self,
        model: Any,
        *,
        batch: int,
        sequence: int,
        updates: int,
        source_batches: int,
        eval_batches: int,
        suffix_layers: int,
        rank: int,
        device: str,
        seed: int = 0,
        data_rank: int = 0,
        sync_gradients: Callable[[Any], None] = lambda _: None,
    ) -> None:
        for name, value in {
            "batch": batch,
            "sequence": sequence,
            "updates": updates,
            "source_batches": source_batches,
            "eval_batches": eval_batches,
            "suffix_layers": suffix_layers,
            "rank": rank,
        }.items():
            validate_integer(name, value)
        self.model = model.eval().requires_grad_(False)
        self.batch, self.sequence = batch, sequence
        self.layers = model.model.layers
        self.target_layer = len(self.layers) - suffix_layers - 1
        if self.target_layer < 0:
            raise ValueError(
                "a block-output intervention needs 1 <= suffix_layers < model depth"
            )
        if source_batches > updates:
            raise ValueError(
                "source_batches must not exceed updates so every cold batch is used"
            )
        if eval_batches % 2:
            raise ValueError("eval_batches must be even: source + intervened forwards")
        if rank > model.config.hidden_size:
            raise ValueError("rank exceeds model width")
        self.device = device
        # Preserve original layer indices and mask selection when skipping a prefix.
        self.cached_layers = torch.nn.ModuleList(
            [_CachedBlock() for _ in range(self.target_layer + 1)]
            + list(self.layers[self.target_layer + 1 :])
        )
        self.updates, self.source_batches, self.eval_batches = (
            updates,
            source_batches,
            eval_batches,
        )
        self.sync_gradients = sync_gradients
        rng = torch.Generator().manual_seed(seed + data_rank)
        vocab = model.config.vocab_size

        def tokens():
            # Exclude the special-token tail of Qwen's vocabulary. Synthetic
            # tokens are deterministic, not a representative language corpus.
            return torch.randint(
                0, min(vocab, 200000), (batch, sequence), generator=rng
            ).to(device)

        self.base = [tokens() for _ in range(source_batches)]
        self.source = [tokens() for _ in range(source_batches)]
        self.eval_base = [tokens() for _ in range(eval_batches // 2)]
        self.eval_source = [tokens() for _ in range(eval_batches // 2)]
        self.labels = torch.randint(0, vocab, (batch,), generator=rng).to(device)
        d = model.config.hidden_size
        self.stages = {
            "subspace": Subspace(d, rank, "cayley", seed=seed).to(device),
            "dbm": Gate(d).to(device),
        }
        self.initial = {
            kind: {
                key: value.detach().clone() for key, value in stage.state_dict().items()
            }
            for kind, stage in self.stages.items()
        }
        self.last_output: Any = None
        self.cache: dict[int, Any] = {}
        self.base_cache: dict[int, Any] = {}
        self.counts: dict[str, int] = {}
        self.stage: Any = None
        self.optimizer: Any = None

    def _forward(
        self,
        tokens: Any,
        category: str,
        *,
        source: Any = None,
        capture: bool = False,
        full_capture: bool = False,
        prefix: Any = None,
    ) -> tuple[Any, Any]:
        captured = None
        fired = [0]

        def intervene(hidden):
            if self.stage is None:
                replacement = source
            else:
                cf, _ = self.stage.featurize(source.float())
                _, residual = self.stage.featurize(hidden[:, -1, :].float())
                replacement = self.stage.inverse(cf, residual)
            changed = hidden.clone()
            changed[:, -1, :] = replacement.to(hidden.dtype)
            return changed

        def hook(_module, _args, output):
            nonlocal captured
            fired[0] += 1
            hidden = output[0] if isinstance(output, tuple) else output
            if capture:
                captured = (
                    (hidden if full_capture else hidden[:, -1, :]).detach().clone()
                )
                raise _CaptureComplete
            if source is None:
                return None
            changed = intervene(hidden)
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        handle = (
            self.layers[self.target_layer].register_forward_hook(hook)
            if prefix is None and (capture or source is not None)
            else None
        )
        try:
            kwargs = {"logits_to_keep": 1}
            if prefix is not None:
                self.model.model.layers = self.cached_layers
                logits = self.model(
                    inputs_embeds=intervene(prefix), use_cache=False, **kwargs
                ).logits
            else:
                logits = self.model(input_ids=tokens, use_cache=False, **kwargs).logits
        except _CaptureComplete:
            if not capture or captured is None:
                raise
            logits = None
        finally:
            self.model.model.layers = self.layers
            if handle is not None:
                handle.remove()
        if handle is not None and fired[0] != 1:
            raise RuntimeError(
                f"intervention hook fired {fired[0]} times, expected once"
            )
        self.counts[category] += 1
        self.counts["block_executions"] += (
            self.target_layer + 1
            if capture
            else len(self.layers) - self.target_layer - 1
            if prefix is not None
            else len(self.layers)
        )
        self.counts["head_token_rows"] += 0 if capture else self.batch
        return logits, captured

    def _capture(self, tokens: Any, category: str, *, full: bool = False):
        logits, captured = self._forward(
            tokens, category, capture=True, full_capture=full
        )
        del logits
        return captured

    def case(self, workload: Workload) -> Case:
        if workload.contract and workload.contract.get("id") != CONTRACT:
            raise ValueError("workload reference and execution contract differ")
        name = workload.name
        training = name.endswith("_train")
        kind = (
            "subspace"
            if name.startswith("subspace_")
            else "dbm"
            if name.startswith("dbm_")
            else None
        )
        expected = {
            "source_forwards": 0,
            "base_forwards": 0,
            "eval_forwards": 0,
            "backwards": 0,
            "updates": 0,
        }
        if training:
            expected.update(
                source_forwards=self.source_batches,
                base_forwards=self.updates,
                eval_forwards=self.eval_batches,
                backwards=self.updates,
                updates=self.updates,
            )
        elif name in {"interchange", "subspace_apply", "dbm_apply"}:
            expected.update(source_forwards=1, base_forwards=1)
        elif name in {"inference", "activation_harvest"}:
            expected.update(base_forwards=1)
        else:
            raise ValueError(f"unsupported benchmark operation {name}")

        prefix_depth = self.target_layer + 1
        if training:
            blocks = (
                2 * self.source_batches * prefix_depth
                + self.updates * (len(self.layers) - prefix_depth)
                + (self.eval_batches // 2) * (prefix_depth + len(self.layers))
            )
            heads = self.updates + self.eval_batches // 2
        else:
            blocks = (
                prefix_depth
                if name == "activation_harvest"
                else len(self.layers) + expected["source_forwards"] * prefix_depth
            )
            heads = 0 if name == "activation_harvest" else 1
        expected.update(
            base_prefix_forwards=self.source_batches if training else 0,
            block_executions=blocks,
            head_token_rows=heads * self.batch,
        )

        def reset():
            self.cache.clear()
            self.base_cache.clear()
            self.last_output = None
            self.counts = dict.fromkeys(expected, 0)
            self.stage = None if kind is None else self.stages[kind]
            self.optimizer = None
            if kind is not None:
                self.stage.load_state_dict(self.initial[kind])
                self.stage.zero_grad(set_to_none=True)
                self.stage.train(training)
                if kind == "dbm":
                    self.stage.temperature = 1.0
                if training:
                    self.optimizer = torch.optim.AdamW(
                        self.stage.parameters(), lr=0.001, weight_decay=0
                    )
                elif kind == "dbm":
                    # A nontrivial, deterministic hard mask, not an all-zero swap.
                    with torch.no_grad():
                        self.stage.theta[::2] = 1
                        self.stage.theta[1::2] = -1

        def run():
            if training:
                for step in range(self.updates):
                    batch_index = step % self.source_batches
                    if batch_index not in self.cache:
                        with torch.no_grad():
                            self.cache[batch_index] = self._capture(
                                self.source[batch_index], "source_forwards"
                            )
                    if kind == "dbm":
                        progress = min(1.0, step / max(1, int(0.5 * self.updates)))
                        self.stage.temperature = 1.0 + (0.01 - 1.0) * progress
                    if batch_index not in self.base_cache:
                        with torch.no_grad():
                            self.base_cache[batch_index] = self._capture(
                                self.base[batch_index],
                                "base_prefix_forwards",
                                full=True,
                            )
                    self.optimizer.zero_grad(set_to_none=True)
                    logits, _ = self._forward(
                        self.base[batch_index],
                        "base_forwards",
                        source=self.cache[batch_index],
                        prefix=self.base_cache.get(batch_index),
                    )
                    loss = torch.nn.functional.cross_entropy(
                        logits[:, -1, :].float(), self.labels
                    )
                    if kind == "dbm":
                        loss = (
                            loss
                            + 0.01
                            * torch.sigmoid(
                                self.stage.theta / self.stage.temperature
                            ).mean()
                        )
                    loss.backward()
                    self.counts["backwards"] += 1
                    self.sync_gradients(self.stage)
                    self.optimizer.step()
                    self.counts["updates"] += 1
                    self.last_output = loss.detach()
                    del logits, loss
                self.stage.eval()
                with torch.no_grad():
                    for base, src in zip(self.eval_base, self.eval_source):
                        self.last_output = None
                        capture = self._capture(src, "eval_forwards")
                        logits, _ = self._forward(base, "eval_forwards", source=capture)
                        self.last_output = logits[:, -1, :].detach()
                        del logits
            else:
                with torch.no_grad():
                    source = None
                    if expected["source_forwards"]:
                        source = self._capture(self.source[0], "source_forwards")
                    logits, capture = self._forward(
                        self.base[0],
                        "base_forwards",
                        source=source,
                        capture=name == "activation_harvest",
                    )
                    self.last_output = (
                        capture
                        if name == "activation_harvest"
                        else logits[:, -1, :].detach()
                    )
            return dict(self.counts)

        def validate():
            if (
                self.last_output is None
                or not torch.isfinite(self.last_output).all().item()
            ):
                raise RuntimeError(f"{name} produced nonfinite or missing output")
            if training:
                if any(
                    not torch.isfinite(p).all().item() for p in self.stage.parameters()
                ):
                    raise RuntimeError(
                        "training produced nonfinite featurizer parameters"
                    )
                if any(p.grad is None for p in self.stage.parameters()):
                    raise RuntimeError(
                        "training failed to produce featurizer gradients"
                    )
                if any(p.grad is not None for p in self.model.parameters()):
                    raise RuntimeError("backbone must remain frozen")

        return Case(workload, reset, run, validate, expected)
