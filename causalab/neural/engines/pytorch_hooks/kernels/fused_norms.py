"""Bind supported norm and rotary forwards to fused Triton kernels.

Plans check device, dtype, shape, strides, activation, and kernel options
before tensor operations. Norms require frozen weights and supported
reduction widths. Unsupported calls use the module's forward.
``norm_reference`` specifies operation order and rounding; golden tests
check outputs and gradients at the tested H100 workflow shapes. Fusion
reduces kernel launches: each norm uses one launch in each direction, and
each rotary embedding uses one launch per tensor.

Each norm uses one autograd node. Views are created inside ``forward`` and
outputs retain the caller's shape, avoiding extra view-backward work.
These functions support first-order gradients.

``fused_norm_path`` temporarily binds norm class forwards and modeling
modules' rotary functions. Eligibility requires source text matching
``PINNED`` after comments and docstrings are removed. Nested bindings stay
in place; exit restores prior functions. Source drift disables a binding
and fails its canary test. The executor enters this context after the short
delta kernel context and before taps. Module hooks continue to run.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import functools
import inspect
import textwrap
import weakref
from typing import Any, Callable, Iterator, cast

import torch

from causalab.neural.engines.pytorch_hooks.kernels import norm_reference as reference
from causalab.neural.engines.pytorch_hooks.kernels import norm_triton as kernels
from causalab.neural.engines.pytorch_hooks.kernels.norm_triton import RowLaunch
from causalab.neural.shared.kernel_options import FusedNormOptions
from causalab.neural.shared.kernels import modeling_modules, torch_implementation

__all__ = [
    "MAX_WIDTH",
    "MIN_WIDTH",
    "PINNED",
    "NormPlan",
    "Targets",
    "canonical_source",
    "fused_gated_rms_norm",
    "fused_norm_path",
    "fused_rms_norm",
    "fused_rotary",
    "plan_norm",
    "plan_rotary",
    "rotary_dispatcher",
    "row_launch",
    "targets_of",
]

#: The narrowest row the reduce order is modelled for (32 lanes of one
#: element) and the widest: above 4096 the reduce splits across warps for 16
#: or more rows (``row_sum_config`` refuses that too), and for fewer rows —
#: one decode row at 6144, say — ``row_sum_config`` would admit a launch
#: whose elementwise tail needs a block above 4096, which no kernel here
#: tiles.
MIN_WIDTH = 32
MAX_WIDTH = 4096

_FLOATS = (torch.float32, torch.float16, torch.bfloat16)
#: The tensor types the kernels may read through a raw pointer. A wrapper
#: subclass — a replicated ``DTensor`` on a ``tp > 1`` norm weight
#: (``docs/model_parallelism.md`` §5.2: ``replicated_with_grad_allreduce`` on
#: ``q_norm`` / ``k_norm``), a ``FakeTensor`` under a trace — passes every
#: shape and device check while having no storage of its own to hand a
#: Triton launch, so it is refused by type before anything is read off it.
_PLAIN_TENSORS = (torch.Tensor, torch.nn.Parameter)

#: The gated norm's activation the kernel spells (``ActivationSiluKernel.cu``).
_GATED_ACTIVATION = "silu"

#: The functions each kernel mirrors, as [`canonical_source`][] renders
#: them (decorators, comments and docstrings dropped): transformers 5.16
#: ``modeling_qwen3_5_moe.py``. A class binds as a norm when its ``forward``
#: and ``_norm`` are the first two; as a gated norm when its ``forward`` is
#: the third; a modeling module binds its rotary embedding when its
#: ``rotate_half`` and ``apply_rotary_pos_emb`` are the last two.
PINNED: dict[str, str] = {
    "norm_forward": (
        "def forward(self, x):\n"
        "    output = self._norm(x.float())\n"
        "    output = output * (1.0 + self.weight.float())\n"
        "    return output.type_as(x)"
    ),
    "norm_norm": (
        "def _norm(self, x):\n"
        "    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)"
    ),
    "gated_forward": (
        "def forward(self, hidden_states, gate=None):\n"
        "    input_dtype = hidden_states.dtype\n"
        "    hidden_states = hidden_states.to(torch.float32)\n"
        "    variance = hidden_states.pow(2).mean(-1, keepdim=True)\n"
        "    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)\n"
        "    hidden_states = self.weight * hidden_states.to(input_dtype)\n"
        "    hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))\n"
        "    return hidden_states.to(input_dtype)"
    ),
    "rotate_half": (
        "def rotate_half(x):\n"
        "    x1 = x[..., :x.shape[-1] // 2]\n"
        "    x2 = x[..., x.shape[-1] // 2:]\n"
        "    return torch.cat((-x2, x1), dim=-1)"
    ),
    "apply_rotary": (
        "def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):\n"
        "    cos = cos.unsqueeze(unsqueeze_dim)\n"
        "    sin = sin.unsqueeze(unsqueeze_dim)\n"
        "    rotary_dim = cos.shape[-1]\n"
        "    (q_rot, q_pass) = (q[..., :rotary_dim], q[..., rotary_dim:])\n"
        "    (k_rot, k_pass) = (k[..., :rotary_dim], k[..., rotary_dim:])\n"
        "    q_embed = q_rot * cos + rotate_half(q_rot) * sin\n"
        "    k_embed = k_rot * cos + rotate_half(k_rot) * sin\n"
        "    q_embed = torch.cat([q_embed, q_pass], dim=-1)\n"
        "    k_embed = torch.cat([k_embed, k_pass], dim=-1)\n"
        "    return (q_embed, k_embed)"
    ),
}


def canonical_source(fn: Callable[..., Any]) -> str | None:
    """``fn``'s source with decorators, comments and a docstring dropped,
    re-rendered by ``ast.unparse`` — what [`PINNED`][] is compared to.
    ``None`` for a function whose source cannot be read."""
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    tree = ast.parse(textwrap.dedent(source))
    node = tree.body[0]
    if not isinstance(node, ast.FunctionDef):
        return None
    node.decorator_list = []
    body = node.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        node.body = body[1:]
    return ast.unparse(node)


# ---------------------------------------------------------------- plans


@functools.lru_cache(maxsize=256)
def row_launch(rows: int, width: int) -> RowLaunch | None:
    """The launch for ``(rows, width)`` — a pure function of the two ints,
    cached: the plan runs on every norm call of every forward."""
    if width < MIN_WIDTH or width > MAX_WIDTH or rows < 1:
        return None
    try:
        config = reference.row_sum_config(rows, width)
    except reference.UnsupportedReduction:
        return None
    chunk = config.lanes * config.vec
    return RowLaunch(
        rows=rows,
        width=width,
        lanes=config.lanes,
        vec=config.vec,
        steps=-(-width // chunk),
        mean_factor=reference.mean_factor(rows, width),
        reciprocal=reference.reciprocal(width),
    )


@dataclasses.dataclass(frozen=True)
class NormPlan:
    """One admitted norm call: the row geometry the kernels index ``x`` (and
    the gate) by — ints, so that nothing here is a recorded view — and the
    launch."""

    rows: int
    row_stride: int
    gate_stride: int | None
    launch: RowLaunch

    @property
    def width(self) -> int:
        return self.launch.width


def _wrapped(tensor: object) -> bool:
    """A tensor subclass other than a parameter (`_PLAIN_TENSORS`)."""
    return isinstance(tensor, torch.Tensor) and type(tensor) not in _PLAIN_TENSORS


def plan_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor | None = None,
    *,
    options: FusedNormOptions,
    activation: str | None = None,
) -> NormPlan | None:
    """Whether the norm call ``(x, weight)`` — the gated norm's when ``gate``
    is given — runs the fused kernel (module docstring). ``activation`` is
    the gated module's and part of the call: a gated call names ``silu`` or
    is refused, the kernel spelling nothing else. No tensor op is run and no
    view is taken."""
    if not options.enabled("gated_norm" if gate is not None else "norm"):
        return None
    if _wrapped(weight) or _wrapped(x) or _wrapped(gate):
        return None
    if x.device.type != "cuda" or not kernels.available():
        return None
    if x.dtype not in _FLOATS or weight.dtype not in _FLOATS:
        return None
    if weight.requires_grad or weight.ndim != 1 or not weight.is_contiguous():
        return None
    if weight.device != x.device:
        return None
    width = x.shape[-1]
    if weight.shape[0] != width:
        return None
    if gate is not None:
        if activation != _GATED_ACTIVATION:
            return None
        # ``weight * h.to(dtype)`` promotes otherwise; ``gate`` must pair rows
        if weight.dtype != x.dtype or gate.dtype not in _FLOATS:
            return None
        if gate.shape != x.shape or gate.device != x.device:
            return None
    geometry = reference.row_geometry(x)
    if geometry is None:
        return None
    gate_stride = None
    if gate is not None:
        gate_geometry = reference.row_geometry(gate)
        if gate_geometry is None:
            return None
        gate_stride = gate_geometry.row_stride
    launch = row_launch(geometry.rows, width)
    if launch is None:
        return None
    return NormPlan(
        rows=geometry.rows,
        row_stride=geometry.row_stride,
        gate_stride=gate_stride,
        launch=launch,
    )


def plan_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, options: FusedNormOptions
) -> bool:
    """Whether ``apply_rotary_pos_emb``'s per-tensor half runs the fused
    kernel for ``x (B, H, T, D)`` with ``cos`` / ``sin`` already unsqueezed."""
    if not options.enabled("rotary"):
        return False
    if x.device.type != "cuda" or not kernels.available():
        return False
    if x.ndim != 4 or cos.ndim != 4 or sin.ndim != 4:
        return False
    if x.dtype not in _FLOATS or cos.dtype != x.dtype or sin.dtype != x.dtype:
        return False
    if cos.requires_grad or sin.requires_grad:
        return False
    if cos.shape != sin.shape or x.stride(-1) != 1:
        return False
    if cos.stride(-1) != 1 or sin.stride(-1) != 1:
        return False
    rot = cos.shape[-1]
    half = rot // 2
    if rot % 2 or half < 2 or half & (half - 1) or rot > x.shape[-1]:
        return False
    if x.shape[-1] - rot > kernels.MAX_PASS_BLOCK:
        return False
    return all(c in (1, s) for c, s in zip(cos.shape[:-1], x.shape[:-1]))


# ------------------------------------------------------- autograd functions


def _rows(x: torch.Tensor, row_stride: int, plan: NormPlan) -> torch.Tensor:
    """``x`` as the ``(rows, width)`` view the plan describes — taken where
    grad mode is off (inside a ``Function``), so nothing is recorded."""
    return x.as_strided((plan.rows, plan.width), (row_stride, 1))


def _grad_rows(grad: torch.Tensor) -> torch.Tensor:
    """An output gradient as ``(rows, width)``: its own storage when its
    leading dimensions merge, a contiguous copy otherwise."""
    rows = reference.rows_view(grad)
    if rows is None:
        rows = reference.rows_view(grad.contiguous())
    assert rows is not None
    return rows


class _FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, x: torch.Tensor, weight: torch.Tensor, eps: float, plan: NormPlan
    ) -> torch.Tensor:
        y, rstd = kernels.rms_norm_forward(
            _rows(x, plan.row_stride, plan), weight, eps, plan.launch, shape=x.shape
        )
        ctx.save_for_backward(x, weight, rstd)
        ctx.plan = plan
        return y

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[Any, ...]:  # type: ignore[override]
        x, weight, rstd = ctx.saved_tensors
        plan: NormPlan = ctx.plan
        dx = kernels.rms_norm_backward(
            _grad_rows(grad),
            _rows(x, plan.row_stride, plan),
            weight,
            rstd,
            plan.launch,
            shape=x.shape,
        )
        return dx, None, None, None


class _FusedGatedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        plan: NormPlan,
    ) -> torch.Tensor:
        assert plan.gate_stride is not None
        o, rstd = kernels.gated_rms_norm_forward(
            _rows(x, plan.row_stride, plan),
            _rows(gate, plan.gate_stride, plan),
            weight,
            eps,
            plan.launch,
            shape=x.shape,
        )
        ctx.save_for_backward(x, gate, weight, rstd)
        ctx.plan = plan
        return o

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[Any, ...]:  # type: ignore[override]
        x, gate, weight, rstd = ctx.saved_tensors
        plan: NormPlan = ctx.plan
        assert plan.gate_stride is not None
        dx, dgate = kernels.gated_rms_norm_backward(
            _grad_rows(grad),
            _rows(x, plan.row_stride, plan),
            _rows(gate, plan.gate_stride, plan),
            weight,
            rstd,
            plan.launch,
            shape=x.shape,
        )
        return dx, dgate, None, None, None


class _FusedRotary(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(cos, sin)
        return kernels.rotary_forward(x, cos, sin)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[Any, ...]:  # type: ignore[override]
        cos, sin = ctx.saved_tensors
        if grad.stride(-1) != 1:
            grad = grad.contiguous()
        return kernels.rotary_backward(grad, cos, sin), None, None


def fused_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, plan: NormPlan
) -> torch.Tensor:
    """``Qwen3_5MoeRMSNorm.forward(x)`` through the kernel, in ``x``'s shape:
    one autograd node between ``x`` and the result."""
    return cast(torch.Tensor, _FusedRMSNorm.apply(x, weight, eps, plan))


def fused_gated_rms_norm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    plan: NormPlan,
) -> torch.Tensor:
    """``Qwen3_5MoeRMSNormGated.forward(x, gate)`` through the kernel."""
    return cast(torch.Tensor, _FusedGatedRMSNorm.apply(x, gate, weight, eps, plan))


def fused_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """One tensor of ``apply_rotary_pos_emb`` through the kernel; ``cos`` /
    ``sin`` unsqueezed, broadcast to ``x``'s leading shape here."""
    lead = tuple(x.shape[:-1]) + (cos.shape[-1],)
    return cast(torch.Tensor, _FusedRotary.apply(x, cos.expand(lead), sin.expand(lead)))


# ------------------------------------------------------------------ binding


@dataclasses.dataclass(frozen=True)
class Targets:
    """What [`fused_norm_path`][] rebinds for one model: the norm classes,
    the gated-norm classes and the modeling modules whose rotary embedding
    is the pinned one."""

    norms: tuple[type, ...]
    gated: tuple[type, ...]
    rotary: tuple[Any, ...]

    @property
    def any(self) -> bool:
        return bool(self.norms or self.gated or self.rotary)


_TARGETS: "weakref.WeakKeyDictionary[torch.nn.Module, Targets]" = (
    weakref.WeakKeyDictionary()
)

#: Set on every dispatcher this module installs — what a nested
#: [`fused_norm_path`][] finds already bound and leaves alone.
_DISPATCHER_MARK = "__fused_norm_dispatcher__"


def _is_pinned(fn: Any, key: str) -> bool:
    """Whether ``fn`` is the pinned function — read through the dispatchers
    of a live entry (each carries ``__wrapped__``), so the scan answers the
    same under an enclosing binding and a model first seen inside one does
    not cache an empty [`Targets`][] for good."""
    if fn is None:
        return False
    return canonical_source(torch_implementation(fn)) == PINNED[key]


def _is_dispatcher(fn: Any) -> bool:
    return getattr(fn, _DISPATCHER_MARK, False) is True


def targets_of(model: torch.nn.Module) -> Targets:
    """The classes and modules of ``model`` whose source is the pinned one
    (module docstring), scanned once per model object."""
    cached = _TARGETS.get(model)
    if cached is not None:
        return cached
    norms: list[type] = []
    gated: list[type] = []
    seen_classes: set[type] = set()
    for module in model.modules():
        cls = type(module)
        if cls in seen_classes:
            continue
        seen_classes.add(cls)
        forward = cls.__dict__.get("forward")
        if forward is None:
            continue
        if _is_pinned(forward, "norm_forward") and _is_pinned(
            getattr(cls, "_norm", None), "norm_norm"
        ):
            norms.append(cls)
        elif _is_pinned(forward, "gated_forward"):
            gated.append(cls)
    rotary = [
        modeling
        for modeling in modeling_modules(model)
        if _is_pinned(getattr(modeling, "apply_rotary_pos_emb", None), "apply_rotary")
        and _is_pinned(getattr(modeling, "rotate_half", None), "rotate_half")
    ]
    found = Targets(norms=tuple(norms), gated=tuple(gated), rotary=tuple(rotary))
    _TARGETS[model] = found
    return found


def _mark(dispatcher: Callable[..., Any], original: Callable[..., Any]) -> None:
    dispatcher.__wrapped__ = original  # type: ignore[attr-defined]
    setattr(dispatcher, _DISPATCHER_MARK, True)


def _norm_forward(
    options: FusedNormOptions, original: Callable[..., Any]
) -> Callable[..., Any]:
    def forward(self: Any, x: torch.Tensor) -> torch.Tensor:
        plan = plan_norm(x, self.weight, options=options)
        if plan is None:
            return original(self, x)
        return fused_rms_norm(x, self.weight, float(self.eps), plan)

    _mark(forward, original)
    return forward


def _gated_forward(
    options: FusedNormOptions, original: Callable[..., Any]
) -> Callable[..., Any]:
    def forward(
        self: Any, hidden_states: torch.Tensor, gate: torch.Tensor | None = None
    ) -> torch.Tensor:
        if gate is None:
            return original(self, hidden_states, gate)
        plan = plan_norm(
            hidden_states,
            self.weight,
            gate,
            options=options,
            activation=self.activation,
        )
        if plan is None:
            return original(self, hidden_states, gate)
        return fused_gated_rms_norm(
            hidden_states, gate, self.weight, float(self.variance_epsilon), plan
        )

    _mark(forward, original)
    return forward


def rotary_dispatcher(
    options: FusedNormOptions, original: Callable[..., Any]
) -> Callable[..., Any]:
    def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_u = cos.unsqueeze(unsqueeze_dim)
        sin_u = sin.unsqueeze(unsqueeze_dim)
        if plan_rotary(q, cos_u, sin_u, options) and plan_rotary(
            k, cos_u, sin_u, options
        ):
            return fused_rotary(q, cos_u, sin_u), fused_rotary(k, cos_u, sin_u)
        return original(q, k, cos, sin, unsqueeze_dim)

    _mark(apply_rotary_pos_emb, original)
    return apply_rotary_pos_emb


@contextlib.contextmanager
def fused_norm_path(
    model: torch.nn.Module, options: FusedNormOptions | None = None
) -> Iterator[None]:
    """While active, the pinned norm classes ``model`` uses and its modeling
    module's rotary embedding dispatch to the fused kernels where
    [`plan_norm`][] / [`plan_rotary`][] admit the call (module
    docstring). ``None`` reads the options from the environment; an empty
    option set installs nothing. A target already bound by an enclosing
    entry is left to it, options included: the outer entry's kernel set
    decides for as long as it stands, and an inner, narrower or empty set
    is not consulted — a bisection scopes the outermost entry. Restored on
    exit either way."""
    if options is None:
        options = FusedNormOptions.from_env()
    if not options.kernels:
        yield
        return
    targets = targets_of(model)
    if not targets.any:
        yield
        return
    rebound: list[tuple[Any, str, Any]] = []

    def bind(target: Any, name: str, wrap: Callable[[Any], Any]) -> None:
        original = target.__dict__[name]
        if _is_dispatcher(original):
            return
        rebound.append((target, name, original))
        setattr(target, name, wrap(original))

    try:
        for cls in targets.norms:
            bind(cls, "forward", functools.partial(_norm_forward, options))
        for cls in targets.gated:
            bind(cls, "forward", functools.partial(_gated_forward, options))
        for modeling in targets.rotary:
            bind(
                modeling,
                "apply_rotary_pos_emb",
                functools.partial(rotary_dispatcher, options),
            )
        yield
    finally:
        for target, name, original in reversed(rebound):
            setattr(target, name, original)
