from __future__ import annotations

from typing import Any, Iterable, List, Sequence

import numpy as np

from ..aggregators._chunking import select_adaptive_chunk_size
from ..aggregators.coordinate_wise._tiling import as_flat_batch, flatten_gradients
from ..configs.backend import get_backend
from ..engine.graph.batch import FlatTensorBatch
from ..engine.graph.batch import _is_disabled as _flat_batch_disabled
from ..engine.graph.subtask import SubTask
from ..engine.storage.shared_store import (
    SharedTensorHandle,
    cleanup_tensor,
    open_tensor,
    register_tensor,
)
from .base import PreAggregator

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


def _to_like(arr: np.ndarray, like: Any) -> Any:
    if _HAS_TORCH and isinstance(like, torch.Tensor):  # type: ignore[arg-type]
        return torch.from_numpy(arr).to(dtype=like.dtype, device=like.device)
    be = get_backend()
    return be.asarray(arr, like=like)


class Clipping(PreAggregator):
    """Static norm clipping pre-aggregator."""

    name = "pre-agg/clipping"
    supports_subtasks = True
    max_subtasks_inflight = 0

    def __init__(self, threshold: float = 2.0, *, chunk_size: int = 32) -> None:
        if threshold < 0:
            raise ValueError("threshold must be >= 0")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.threshold = float(threshold)
        self.chunk_size = int(chunk_size)
        self._in_handle: SharedTensorHandle | None = None
        self._in_owned: bool = False
        self._out_handle: SharedTensorHandle | None = None
        self._flat_shape: tuple[int, ...] | None = None
        self._like_template: Any | None = None

    def pre_aggregate(self, xs: Sequence[Any]) -> List[Any]:
        if not xs:
            raise ValueError("xs must be non-empty")
        flat_shape, flat = flatten_gradients(xs)
        clipped = _clip_rows(flat, self.threshold)
        outputs: List[Any] = []
        for row in clipped:
            reshaped = row.reshape(flat_shape)
            outputs.append(
                _to_like(reshaped, xs[0] if not isinstance(xs, FlatTensorBatch) else xs.like)
            )
        return outputs

    def create_subtasks(self, inputs, *, context):  # type: ignore[override]
        xs = inputs.get(self.input_key)
        if xs is None:
            return []
        if isinstance(xs, FlatTensorBatch):
            if len(xs) == 0:
                return []
            input_batch = xs
            in_owned = False
        else:
            if not isinstance(xs, Sequence) or not xs:
                return []
            input_batch = as_flat_batch(xs)
            in_owned = True

        n, feature_dim = input_batch.handle.shape
        if n == 0:
            if in_owned:
                cleanup_tensor(input_batch.handle)
            return []

        self._in_handle = input_batch.handle
        self._in_owned = in_owned
        self._flat_shape = input_batch.flat_shape
        self._like_template = input_batch.like

        out_array = np.zeros((n, feature_dim), dtype=np.dtype(input_batch.handle.dtype))
        out_handle = register_tensor(out_array)
        self._out_handle = out_handle

        metadata = getattr(context, "metadata", None) or {}
        pool_size = int(metadata.get("pool_size") or 0)
        chunk = select_adaptive_chunk_size(n, self.chunk_size, pool_size=pool_size)

        in_handle = input_batch.handle
        threshold = self.threshold

        def _iter() -> Iterable[SubTask]:
            chunk_id = 0
            for start in range(0, n, chunk):
                end = min(n, start + chunk)
                yield SubTask(
                    fn=_clipping_chunk,
                    args=(in_handle, out_handle, start, end, threshold),
                    kwargs={},
                    name=f"clipping_chunk_{chunk_id}",
                )
                chunk_id += 1

        return _iter()

    def reduce_subtasks(self, partials, inputs, *, context):  # type: ignore[override]
        if not partials:
            self._cleanup_state()
            return super().compute(inputs, context=context)
        if self._out_handle is None or self._flat_shape is None or self._like_template is None:
            raise RuntimeError("Clipping missing state for reduction.")

        out_handle = self._out_handle
        flat_shape = self._flat_shape
        like = self._like_template

        if self._in_owned and self._in_handle is not None:
            cleanup_tensor(self._in_handle)
        self._in_handle = None
        self._in_owned = False
        self._out_handle = None
        self._flat_shape = None
        self._like_template = None

        if _flat_batch_disabled():
            try:
                with open_tensor(out_handle) as flat:
                    data = np.array(flat, copy=True)
                outputs: List[Any] = []
                for i in range(data.shape[0]):
                    reshaped = data[i].reshape(flat_shape)
                    outputs.append(_to_like(reshaped, like))
                return outputs
            finally:
                cleanup_tensor(out_handle)

        return FlatTensorBatch(
            handle=out_handle,
            flat_shape=flat_shape,
            like=like,
            owns_handle=True,
        )

    def _cleanup_state(self) -> None:
        if self._in_owned and self._in_handle is not None:
            cleanup_tensor(self._in_handle)
        if self._out_handle is not None:
            cleanup_tensor(self._out_handle)
        self._in_handle = None
        self._in_owned = False
        self._out_handle = None
        self._flat_shape = None
        self._like_template = None


def _clip_rows(flat: np.ndarray, threshold: float) -> np.ndarray:
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    denom = np.maximum(norms, 1e-12)
    factors = np.minimum(1.0, threshold / denom)
    return flat * factors


def _clipping_chunk(
    in_handle: SharedTensorHandle,
    out_handle: SharedTensorHandle,
    start: int,
    end: int,
    threshold: float,
):
    with open_tensor(in_handle) as src, open_tensor(out_handle) as dst:
        chunk = np.asarray(src[start:end])
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        denom = np.maximum(norms, 1e-12)
        factors = np.minimum(1.0, threshold / denom)
        dst[start:end, :] = chunk * factors
    return start, None


__all__ = ["Clipping"]
