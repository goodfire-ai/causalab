"""Drive an intervention specification through the reference engine in-process.

Test documents bypass dataset resolution: rows are handed to the executor
directly (the executor's own seam), while the document still parses and
validates through the real loader path — a test document is exactly as
valid as a real one."""

from __future__ import annotations

from typing import Any

from causalab.neural.engines.pytorch_hooks.executor import Interning, PointExecutor
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle


def bundle_loader(files: dict[str, dict[str, Any]]) -> Any:
    """See ``tests._helpers.engines.bundle_loader``; re-exported so the
    reference tests keep their import."""
    from tests._helpers.engines import bundle_loader as _bundle_loader

    return _bundle_loader(files)


def executor_for(
    doc_raw: dict[str, Any],
    bundle: ModelBundle,
    *,
    base_texts: list[str],
    counterfactual_texts: list[str] | None = None,
    extra_columns: dict[str, list[Any]] | None = None,
    load_tensors: Any = None,
    load_table: Any = None,
    grad_enabled: bool = False,
    interning: Interning | None = None,
    batch_rows: int | None = None,
) -> PointExecutor:
    """The reference engine's executor over ``doc_raw``: the engine-generic
    ``tests._helpers.engines.executor_for`` with ``PointExecutor`` filled
    in, so the parity suite and these tests build rows one way."""
    from tests._helpers.engines import executor_for as _executor_for

    return _executor_for(
        PointExecutor,
        doc_raw,
        bundle,
        base_texts=base_texts,
        counterfactual_texts=counterfactual_texts,
        extra_columns=extra_columns,
        load_tensors=load_tensors,
        load_table=load_table,
        grad_enabled=grad_enabled,
        interning=interning,
        batch_rows=batch_rows,
    )


def base_data_section(with_counterfactual: bool) -> dict[str, Any]:
    data: dict[str, Any] = {"base": {"dataset": "inline", "field": "input"}}
    if with_counterfactual:
        data["counterfactual"] = {
            "dataset": "inline",
            "field": "counterfactual_inputs[0]",
        }
    return data
