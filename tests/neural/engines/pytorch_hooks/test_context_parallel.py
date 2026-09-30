"""Context parallelism through the engine's own seams (``docs/model_parallelism.md``
§6.4, §8.4, §10.4), under the ``SimulatedWorld`` at drawn schedules (tapes
and seeds, ``parallel_strategies.schedules``) over drawn data seeds:

- **attention with all-gather KV** through the real ``attention_interface``
  dispatcher, a faithful miniature of the library's eager function and a
  left-padded frame: each rank's query chunk against the gathered keys and
  values, under the whole frame's mask rows, equals the whole-frame
  attention at that chunk. The mask, the gathered keys and values and a
  ``attention_key`` read are the same tensors bit for bit (a gather moves
  no arithmetic); the scores and the output are the whole frame's rows up
  to the GEMM's blocking over a shorter query axis — measured ``0.0`` on
  CPU for these shapes over the first twenty data seeds at ``cp ∈ {2, 3}``,
  pinned at an
  fp32 band of ``1e-6`` rather than asserted exact, since a BLAS may block a
  shorter GEMM differently. A pattern swap
  at the global pattern lands each rank's rows and feeds the library's own
  value multiply;
- **the DeltaNet state handed chunk to chunk** with the library's own
  kernels: the chunked gated-delta kernel over the chunks, each with the
  state the rank below produced, equals the unsplit kernel within a band —
  the kernel's 64-position blocks fall differently, so its sums
  re-associate (measured ``1.8e-7`` on these tensors over the first twenty
  data seeds at ``cp ∈ {2, 3}``, pinned ``2e-5``: a hundred times, room for a longer
  frame's blocks); the per-step recurrent path (``_stepwise``, what a state read
  or write runs) equals the unsplit loop **bit for bit**, since every
  step's state is the same kernel call in the same order; a rank that skips
  its send is a typed refusal naming the waiting rank;
- **the real engine on the tiny Llama** at a simulated ``cp=2``: every rank
  runs ``run_protocol`` over its own replicated copy of the model (the
  loader keyed by ``(geometry, rank)``) with the corpus interchange
  document extended to read and write at positions in **both** chunks, and
  its outputs equal the world-1 run within the band below, receipts equal
  but for ``execution.parallel``, under every drawn schedule.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings
from safetensors.torch import load_file
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from causalab.cli import register_model_key
from causalab.neural.engines.pytorch_hooks import delta_interface
from causalab.neural.engines.pytorch_hooks.attention_interface import (
    InterfaceTap,
    attention_interface_taps,
)
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.executor import (
    _interface_capture,
    _interface_edit,
)
from causalab.neural.engines.pytorch_hooks.sharding import Sharding
from causalab.neural.shared.kernels import torch_implementation
from causalab.neural.shared.layout import to_contract
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.context import (
    SequenceFrame,
    activate,
    handoff_state,
    local_causal_mask,
)
from causalab.neural.shared.parallel.fragments import Fragments
from causalab.neural.shared.parallel.placement import SequenceSharded
from causalab.neural.shared.parallel.taps import TapFragments
from causalab.neural.shared.sites import ResolvedSite
from causalab.protocol.parallel import ParallelGeometry, sequence_chunks
from causalab.protocol.pipeline import run_protocol
from causalab.protocol.registry.shapes import attention_pattern, bhsd, bshd
from causalab.io.tables import read_table
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import (
    Abandoned,
    Hang,
    Schedule,
    SimulatedWorld,
    groups_for,
)
from tests.protocol._env import CORPUS_DIR, FIXTURES, build_env

from .conftest import TINY_LLAMA

# pyright: reportPrivateUsage=false

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _seeded(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _left_padded(rows: int, padded_len: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, padded_len + 1, (rows,), generator=generator)
    mask = torch.zeros(rows, padded_len, dtype=torch.long)
    for row, length in enumerate(lengths.tolist()):
        mask[row, padded_len - length :] = 1
    return mask


# --------------------------------------------------------------------------- #
# attention: all-gather KV inside the eager function
# --------------------------------------------------------------------------- #

BATCH, HEADS, SEQ, DIM = 2, 4, 7, 8
#: The fp32 band for the rows of a GEMM computed over a shorter query axis
#: (module docstring); the mask, the gathered K/V and a key read are exact.
ATTENTION_BAND = 1e-6


def eager_attention_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The library's eager function in miniature, the mask add included —
    resolved by ``module_eager_attention`` from this module, where
    `_Mixer` lives."""
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    out = torch.matmul(probs, value).transpose(1, 2).contiguous()
    return out, probs


class _Mixer(torch.nn.Module):
    num_key_value_groups = 1


def _site(component: str, slot: str, shape: Any) -> ResolvedSite:
    return ResolvedSite(
        module=None,
        kind="interface" if slot != "probs" else "out",
        shape=shape,
        component=component,
        interface_slot=slot,
    )


#: The manager installs one process-global entry, so a world's ranks share
#: one manager entered once around the run over a live tap table.
_KEEP = -1


def _attention_program(
    seed: int, *, tapped: bool
) -> Callable[[int, Collective], dict[str, torch.Tensor]]:
    q = _seeded((BATCH, HEADS, SEQ, DIM), seed)
    k = _seeded((BATCH, HEADS, SEQ, DIM), seed + 1)
    v = _seeded((BATCH, HEADS, SEQ, DIM), seed + 2)
    mask = _left_padded(BATCH, SEQ, seed + 3)
    replacement = torch.softmax(_seeded((BATCH, HEADS, SEQ, SEQ), seed + 4), dim=-1)
    scaling = DIM**-0.5
    taps: dict[int, tuple[InterfaceTap, ...]] = {_KEEP: ()}

    def program(rank: int, c: Collective) -> dict[str, torch.Tensor]:
        frame = SequenceFrame(c, mask)
        chunk = frame.chunk
        local = {
            name: t[:, :, chunk.start : chunk.stop]
            for name, t in (("q", q), ("k", k), ("v", v))
        }
        # what the model would hand the function for the local chunk alone:
        # the local causal mask, which the wrapper replaces under cp > 1
        local_mask = local_causal_mask(
            mask[:, chunk.start : chunk.stop], range(len(chunk)), q.dtype
        )
        sink: dict[Any, torch.Tensor] = {}
        mixer = _Mixer()
        if tapped:
            fragments = Fragments(c)
            rows = TapFragments(fragments, SequenceSharded(2), frame=frame)
            positions = TapFragments(fragments, SequenceSharded(1), frame=frame)

            def swap(contract: torch.Tensor) -> None:
                contract.copy_(replacement)

            taps[id(mixer)] = (
                InterfaceTap(
                    slot="probs",
                    edit=_interface_edit(
                        _site("attention_probs", "probs", attention_pattern(HEADS)),
                        swap,
                        BATCH,
                        rows,
                    ),
                ),
                InterfaceTap(
                    slot="key",
                    read=_interface_capture(
                        sink,
                        "key",
                        _site("attention_key", "key", bhsd(HEADS, DIM)),
                        BATCH,
                        rows,
                    ),
                ),
                InterfaceTap(
                    slot="scores",
                    read=_interface_capture(
                        sink,
                        "scores",
                        _site("attention_scores", "scores", attention_pattern(HEADS)),
                        BATCH,
                        rows,
                    ),
                ),
                InterfaceTap(
                    slot="z",
                    read=_interface_capture(
                        sink,
                        "z",
                        _site("attention_z", "z", bshd(HEADS, DIM)),
                        BATCH,
                        positions,
                    ),
                ),
            )
        with activate(frame):
            out, weights = ALL_ATTENTION_FUNCTIONS["eager"](
                mixer, local["q"], local["k"], local["v"], local_mask, scaling=scaling
            )
        return {
            **sink,
            "out": out,
            "weights": weights,
            "chunk": torch.tensor([chunk.start, chunk.stop]),
        }

    program.q, program.k, program.v, program.mask = q, k, v, mask  # type: ignore[attr-defined]
    program.replacement, program.scaling, program.taps = replacement, scaling, taps  # type: ignore[attr-defined]
    return program


def _run_attention(
    program: Any, world: SimulatedWorld
) -> list[dict[str, torch.Tensor]]:
    with attention_interface_taps(program.taps):
        return world.run(program)


def _whole_attention(program: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The whole-frame reference: the library's own function over every
    position, the whole frame's eager mask."""
    mask = local_causal_mask(program.mask, range(0, SEQ), program.q.dtype)
    scores = torch.matmul(program.q, program.k.transpose(2, 3)) * program.scaling + mask
    out, weights = eager_attention_forward(
        _Mixer(), program.q, program.k, program.v, mask, program.scaling
    )
    return out, weights, scores


@pytest.mark.property
class TestAttentionAllGatherKV:
    @pytest.mark.parametrize("context", [2, 3])
    @given(schedule=ps.schedules(), seed=ps.seeds())
    @example(schedule=[], seed=0)
    @_SETTINGS
    def test_each_ranks_query_chunk_sees_the_whole_frame(
        self, context: int, schedule: Schedule, seed: int
    ) -> None:
        """Untapped: the wrapper alone gathers K and V and swaps the mask."""
        program = _attention_program(seed, tapped=False)
        world = SimulatedWorld(
            groups_for(context, context=context), world=context, schedule=schedule
        )
        out, weights, _ = _whole_attention(program)
        for rank, got in enumerate(_run_attention(program, world)):
            chunk = sequence_chunks(SEQ, context)[rank]
            assert got["weights"].shape == (BATCH, HEADS, len(chunk), SEQ)
            assert torch.allclose(
                got["weights"],
                weights[:, :, chunk.start : chunk.stop],
                atol=ATTENTION_BAND,
                rtol=0,
            )
            assert torch.allclose(
                got["out"],
                out[:, chunk.start : chunk.stop],
                atol=ATTENTION_BAND,
                rtol=0,
            )

    @given(schedule=ps.schedules(), seed=ps.seeds())
    @_SETTINGS
    def test_reads_are_the_whole_frames_and_a_pattern_swap_lands_the_chunk(
        self, schedule: Schedule, seed: int
    ) -> None:
        program = _attention_program(seed, tapped=True)
        world = SimulatedWorld(groups_for(2, context=2), world=2, schedule=schedule)
        _, _, scores = _whole_attention(program)
        swapped_out = (
            torch.matmul(program.replacement, program.v).transpose(1, 2).contiguous()
        )
        for rank, got in enumerate(_run_attention(program, world)):
            chunk = sequence_chunks(SEQ, 2)[rank]
            # a gather moves no arithmetic: the key read is the global key exactly
            assert torch.equal(
                got["key"], to_contract(program.k, bhsd(HEADS, DIM), batch_size=BATCH)
            )
            # the scores are the whole frame's rows (the mask is the same
            # tensor; the GEMM ran over a shorter query axis)
            assert torch.allclose(got["scores"], scores, atol=ATTENTION_BAND, rtol=0)
            # the swap landed this rank's rows of the global pattern, and the
            # library's own value multiply consumed them
            assert torch.equal(
                got["weights"], program.replacement[:, :, chunk.start : chunk.stop]
            )
            assert torch.allclose(
                got["out"],
                swapped_out[:, chunk.start : chunk.stop],
                atol=ATTENTION_BAND,
                rtol=0,
            )
            assert torch.allclose(
                got["z"],
                to_contract(swapped_out, bshd(HEADS, DIM), batch_size=BATCH),
                atol=ATTENTION_BAND,
                rtol=0,
            )

    def test_the_gathered_keys_values_and_mask_are_exact(self) -> None:
        """The three tensors the wrapper builds, checked in the wrapper's
        own terms: bit for bit the whole frame's."""
        program = _attention_program(11, tapped=False)
        whole_mask = local_causal_mask(program.mask, range(0, SEQ), torch.float32)

        def check(rank: int, c: Collective) -> None:
            frame = SequenceFrame(c, program.mask)
            chunk = frame.chunk
            assert torch.equal(
                frame.gather(program.k[:, :, chunk.start : chunk.stop], 2), program.k
            )
            assert torch.equal(
                frame.gather(program.v[:, :, chunk.start : chunk.stop], 2), program.v
            )
            assert torch.equal(
                frame.attention_mask(torch.float32),
                whole_mask[:, :, chunk.start : chunk.stop],
            )

        SimulatedWorld(groups_for(2, context=2), world=2, schedule=1).run(check)

    def test_without_a_frame_the_dispatcher_is_the_library_function(self) -> None:
        """No frame bound and no tap: the wrapper is not even installed."""
        program = _attention_program(5, tapped=False)
        mixer = _Mixer()
        with attention_interface_taps({}):
            assert "eager" not in ALL_ATTENTION_FUNCTIONS
        with attention_interface_taps({id(mixer): ()}):
            out, weights = ALL_ATTENTION_FUNCTIONS["eager"](
                mixer, program.q, program.k, program.v, None, scaling=program.scaling
            )
        expected_out, expected_weights = eager_attention_forward(
            mixer, program.q, program.k, program.v, None, program.scaling
        )
        assert torch.equal(out, expected_out) and torch.equal(weights, expected_weights)

    def test_two_managers_on_one_mixer_are_refused(self) -> None:
        mixer = _Mixer()
        with attention_interface_taps({id(mixer): (InterfaceTap(slot="query"),)}):
            with pytest.raises(ValueError, match="already tapped"):
                with attention_interface_taps({id(mixer): (InterfaceTap(slot="z"),)}):
                    pass
        assert "eager" not in ALL_ATTENTION_FUNCTIONS


# --------------------------------------------------------------------------- #
# the DeltaNet state handed chunk to chunk
# --------------------------------------------------------------------------- #

DB, DL, DH, DK, DV = 2, 9, 4, 8, 8
#: The chunked kernel's blocks fall differently on a split frame (module
#: docstring): measured ``1.8e-7``, pinned a hundred times above.
DELTA_BAND = 2e-5


def _kernels() -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as modeling

    return (
        torch_implementation(modeling.torch_chunk_gated_delta_rule),
        torch_implementation(modeling.torch_recurrent_gated_delta_rule),
        modeling.l2norm,
    )


def _delta_inputs(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "q": torch.randn(DB, DL, DH, DK, generator=generator),
        "k": torch.randn(DB, DL, DH, DK, generator=generator),
        "v": torch.randn(DB, DL, DH, DV, generator=generator),
        "g": -torch.rand(DB, DL, DH, generator=generator),
        "beta": torch.rand(DB, DL, DH, generator=generator),
    }


def _delta_program(
    seed: int, *, stepwise: bool, skip_send: bool = False
) -> Callable[[int, Collective], dict[str, torch.Tensor]]:
    chunk_kernel, recurrent, l2norm = _kernels()
    x = _delta_inputs(seed)
    mask = torch.ones(DB, DL, dtype=torch.long)

    def program(rank: int, c: Collective) -> dict[str, torch.Tensor]:
        frame = SequenceFrame(c, mask)
        s = slice(frame.chunk.start, frame.chunk.stop)
        q, k, v, g, beta = (x[name][:, s] for name in ("q", "k", "v", "g", "beta"))

        def run(initial: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
            if stepwise:
                out, final, *_ = delta_interface._stepwise(
                    recurrent, l2norm, q, k, v, g, beta, initial, True
                )
                return out, final
            return chunk_kernel(
                q,
                k,
                v,
                g,
                beta,
                initial_state=initial,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        if skip_send and rank == 0:
            out, _ = run(None)
            return {"out": out}
        out, final = handoff_state(
            frame, run, shape=(DB, DH, DK, DV), dtype=torch.float32, device=v.device
        )
        return {"out": frame.gather(out, 1), "final": final}

    program.inputs = x  # type: ignore[attr-defined]
    return program


def _unsplit(seed: int, *, stepwise: bool) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_kernel, recurrent, l2norm = _kernels()
    x = _delta_inputs(seed)
    if stepwise:
        out, final, *_ = delta_interface._stepwise(
            recurrent, l2norm, x["q"], x["k"], x["v"], x["g"], x["beta"], None, True
        )
        return out, final
    return chunk_kernel(
        x["q"],
        x["k"],
        x["v"],
        x["g"],
        x["beta"],
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )


@pytest.mark.property
class TestDeltaHandoff:
    @pytest.mark.parametrize("context", [2, 3])
    @given(schedule=ps.schedules(), seed=ps.seeds())
    @_SETTINGS
    def test_the_stepwise_path_over_chunks_is_the_unsplit_loop_bit_for_bit(
        self, context: int, schedule: Schedule, seed: int
    ) -> None:
        """``_stepwise`` threads the received state into the same recurrent
        kernel call the unsplit loop makes at that step: nothing re-associates."""
        out, final = _unsplit(seed, stepwise=True)
        world = SimulatedWorld(
            groups_for(context, context=context), world=context, schedule=schedule
        )
        results = world.run(_delta_program(seed, stepwise=True))
        for got in results:
            assert torch.equal(got["out"], out)
        assert torch.equal(results[-1]["final"], final)

    @pytest.mark.parametrize("context", [2, 3])
    @given(schedule=ps.schedules(), seed=ps.seeds())
    @_SETTINGS
    def test_the_chunked_kernel_over_chunks_equals_the_unsplit_kernel_within_the_band(
        self, context: int, schedule: Schedule, seed: int
    ) -> None:
        out, final = _unsplit(seed, stepwise=False)
        world = SimulatedWorld(
            groups_for(context, context=context), world=context, schedule=schedule
        )
        results = world.run(_delta_program(seed, stepwise=False))
        for got in results:
            worst = (got["out"] - out).abs().max().item()
            assert worst <= DELTA_BAND, worst
        assert (results[-1]["final"] - final).abs().max().item() <= DELTA_BAND

    def test_a_rank_that_skips_the_state_send_is_refused_by_name(self) -> None:
        world = SimulatedWorld(groups_for(2, context=2), world=2, schedule=0)
        with pytest.raises((Hang, Abandoned)) as err:
            world.run(_delta_program(0, stepwise=False, skip_send=True))
        assert "rank 1" in str(err.value) and "recv" in str(err.value)


# --------------------------------------------------------------------------- #
# the real engine on the tiny Llama at a simulated cp=2
# --------------------------------------------------------------------------- #

INTERCHANGE = CORPUS_DIR / "02_interchange_im.json"
TABLES = ("iia.json", "logit_diff.json")
LAYER = 1
#: The fp32 band for the engine at ``cp=2`` against world 1: every read and
#: write is a gather or a chunk of the same tensors, and the attention
#: GEMMs run over a shorter query axis (``ATTENTION_BAND`` per layer, twice
#: composed through the residual and the head). Pinned at ``1e-5``, above
#: the maximum the runs below reach.
ENGINE_BAND = 1e-5


def _document(tmp: Path) -> Path:
    """Corpus 02 on the tiny Llama: the swap at the answer slot (the last
    chunk), a read and a swap at the first position (the first chunk), the
    residual and the logits saved for both."""
    doc = json.loads(INTERCHANGE.read_text())
    doc["model"] = {"key": TINY_LLAMA, "revision": "main", "dtype": "fp32"}
    method = doc["method"]
    method["sites"]["target"]["layers"] = [LAYER]
    reads, writes, models, save = (
        method["reads"],
        method["writes"],
        method["intervened_models"],
        method["save"],
    )
    reads["first"] = {"site": "target", "pos": 0}
    reads["first_cf"] = {"site": "target", "pos": 0}
    writes["patch_first"] = {"site": "target", "pos": 0, "do": {"swap": "first_cf"}}
    reads["logits_first"] = {"site": "lm_head", "pos": -1}
    reads["patched_all"] = {"site": "target", "pos": "all"}
    # the un-intervened model on base is a declared model with no writes (§2.9)
    models["original_base"] = {"input": "base", "reads": ["first"]}
    models["original_counterfactual"]["reads"].append("first_cf")
    models["patched_first"] = {
        "input": "base",
        "reads": ["logits_first"],
        "writes": ["patch_first"],
    }
    models["patched"]["reads"].append("patched_all")
    for read, model in (
        ("v_cf", "original_counterfactual"),
        ("first", "original_base"),
        ("logits", "patched"),
        ("logits_first", "patched_first"),
        ("patched_all", "patched"),
    ):
        save.append({"read": read, "model": model, "file_path": f"{read}.safetensors"})
    target = tmp / "context.json"
    target.write_text(json.dumps(doc, indent=2))
    return target


def _artifacts(tmp: Path) -> Path:
    root = tmp / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
    return root


def _solo(document: Path, artifacts: Path, out: Path) -> Path:
    register_model_key(json.loads(document.read_text()))
    run_protocol(
        document,
        build_env(artifacts),
        PytorchHooksEngine(device="cpu"),
        out,
        record=True,
    )
    return out


def _engine_program(
    document: Path, artifacts: Path, out_root: Path
) -> Callable[[int, Collective], str]:
    geometry = ParallelGeometry(context=2)

    def program(rank: int, c: Collective) -> str:
        engine = PytorchHooksEngine(
            device="cpu",
            parallel=geometry,
            collective=c,
            sharding=Sharding(geometry=geometry, rank=rank, meshes={}),
        )
        out = out_root / f"rank{rank}"
        run_protocol(document, build_env(artifacts), engine, out, record=True)
        return str(out)

    return program


def _max_diffs(solo: Path, parallel: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for path in sorted(solo.glob("*.safetensors")):
        a, b = load_file(str(path)), load_file(str(parallel / path.name))
        assert set(a) == set(b), path.name
        worst = 0.0
        for key in a:
            if a[key].shape != b[key].shape:
                worst = math.inf
            elif a[key].dtype.is_floating_point:
                worst = max(
                    worst, (a[key].double() - b[key].double()).abs().max().item()
                )
            elif not torch.equal(a[key], b[key]):
                worst = math.inf
        out[path.stem] = worst
    for name in TABLES:
        for x, y in zip(read_table(solo / name), read_table(parallel / name)):
            for key in x:
                if isinstance(x[key], float):
                    out[name] = max(
                        out.get(name, 0.0), abs(float(x[key]) - float(y[key]))
                    )
                else:
                    assert x[key] == y[key], (name, key)
    return out


@pytest.mark.property
class TestEngineAtContextTwo:
    @pytest.fixture(scope="class")
    def solo(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
        tmp = tmp_path_factory.mktemp("cp-llama")
        artifacts = _artifacts(tmp)
        document = _document(tmp)
        return document, artifacts, _solo(document, artifacts, tmp / "solo")

    @given(schedule=ps.schedules())
    @_SETTINGS
    def test_reads_and_writes_in_both_chunks_equal_world_one(
        self, solo: tuple[Path, Path, Path], tmp_path: Path, schedule: Schedule
    ) -> None:
        document, artifacts, solo_out = solo
        world = SimulatedWorld(groups_for(2, context=2), world=2, schedule=schedule)
        outs = world.run(_engine_program(document, artifacts, tmp_path))
        for rank, out in enumerate(outs):
            diffs = _max_diffs(solo_out, Path(out))
            worst = max(diffs.values())
            assert worst <= ENGINE_BAND, (rank, diffs)
            receipt = json.loads((Path(out) / "protocol.json").read_text())
            assert receipt["execution"]["parallel"]["context"] == 2
            reference = json.loads((solo_out / "protocol.json").read_text())
            del receipt["execution"]["parallel"], reference["execution"]["parallel"]
            assert json.dumps(receipt, sort_keys=True) == json.dumps(
                reference, sort_keys=True
            )
