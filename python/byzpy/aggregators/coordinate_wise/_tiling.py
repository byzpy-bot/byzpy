from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ...engine.graph.batch import FlatTensorBatch
from ...engine.storage.shared_store import SharedTensorHandle, open_tensor, register_tensor

try:  # optional torch dependency
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


def flatten_gradients(gradients: Sequence[Any]) -> tuple[tuple[int, ...], np.ndarray]:
    if isinstance(gradients, FlatTensorBatch):
        with open_tensor(gradients.handle) as arr:
            flat = np.array(arr, copy=True)
        return gradients.flat_shape, flat
    arrays = [_to_numpy(g) for g in gradients]
    stacked = np.stack(arrays, axis=0)
    shape = stacked.shape[1:]
    flat = stacked.reshape(stacked.shape[0], -1)
    return shape, flat


def as_flat_batch(gradients: Any) -> FlatTensorBatch:
    """
    Normalize gradients into a :class:`FlatTensorBatch`.

    If ``gradients`` is already a FlatTensorBatch, it is returned directly
    (no copy, no re-register). Otherwise the sequence is flattened, stacked
    into a contiguous ``(n, feature_dim)`` array, and registered into shared
    memory.
    """
    if isinstance(gradients, FlatTensorBatch):
        return gradients
    flat_shape, flat = flatten_gradients(gradients)
    handle = register_tensor(flat)
    like = gradients[0] if hasattr(gradients, "__getitem__") else next(iter(gradients))
    return FlatTensorBatch(handle=handle, flat_shape=flat_shape, like=like, owns_handle=True)


def _to_numpy(grad: Any) -> np.ndarray:
    if _HAS_TORCH and isinstance(grad, torch.Tensor):  # type: ignore[arg-type]
        return grad.detach().cpu().numpy()
    if isinstance(grad, np.ndarray):
        return grad
    if isinstance(grad, SharedTensorHandle):
        with open_tensor(grad) as arr:
            return np.array(arr, copy=True)
    if isinstance(grad, dict) and {"name", "shape", "dtype"} <= grad.keys():
        handle = SharedTensorHandle(**grad)
        with open_tensor(handle) as arr:
            return np.array(arr, copy=True)
    return np.asarray(grad)


__all__ = ["flatten_gradients", "as_flat_batch"]
