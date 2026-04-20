"""
Flat tensor batch intermediate representation.

This module provides :class:`FlatTensorBatch`, a lightweight wrapper around a
shared-memory ``(n, feature_dim)`` buffer that behaves like ``Sequence[Any]``
while avoiding repeated flatten/stack/register round-trips between operators.

The goal is to let two subtask-capable operators chain together without the
downstream operator having to reflatten and re-register the intermediate
result. When an operator's ``reduce_subtasks`` returns a ``FlatTensorBatch``,
the next operator's ``create_subtasks`` can detect it and reuse the backing
handle directly.

Because ``FlatTensorBatch`` implements the ``Sequence`` protocol, code paths
that treat intermediates as ``List[Tensor]`` continue to work unchanged. The
handle lifecycle is managed by the scheduler based on graph-level
"last-consumer" information; see :meth:`byzpy.engine.graph.graph.ComputationGraph.last_use_map`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Iterator, List

import numpy as np

from ..storage.shared_store import SharedTensorHandle, cleanup_tensor, open_tensor

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


def _is_disabled() -> bool:
    """Return True when the feature flag disables the FlatTensorBatch path."""
    return os.environ.get("BYZPY_DISABLE_FLAT_BATCH", "").lower() in ("1", "true", "yes")


@dataclass
class FlatTensorBatch(Sequence):
    """
    Shared-memory backed batch of row-aligned tensors.

    This represents ``n`` tensors of identical shape ``flat_shape`` stored
    contiguously in a single ``(n, prod(flat_shape))`` shared-memory buffer.

    Parameters
    ----------
    handle : SharedTensorHandle
        Shared-memory handle holding the flat data.
    flat_shape : tuple[int, ...]
        Shape of a single item (the ``...`` in ``(n, ...)``).
    like : Any
        Reference tensor used to determine target dtype / device / backend
        when items are materialized as individual tensors.
    owns_handle : bool, optional
        Whether this batch is responsible for cleaning up the shared memory
        on :meth:`release`. Default True.

    Notes
    -----
    The ``Sequence`` protocol (``__len__``, ``__iter__``, ``__getitem__``)
    returns per-item tensors shaped like the original inputs. This can be used
    in places that previously received a ``List[Tensor]``, but each such
    access incurs a shared-memory read and conversion; prefer
    :meth:`materialize_list` when you need all items or pass the batch through
    to the next operator for zero-copy reuse.
    """

    handle: SharedTensorHandle
    flat_shape: tuple[int, ...]
    like: Any
    owns_handle: bool = True
    _released: bool = field(default=False, init=False, repr=False)

    @property
    def n(self) -> int:
        return int(self.handle.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.handle.shape[1])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> Any:  # type: ignore[override]
        if self._released:
            raise RuntimeError("FlatTensorBatch has already been released")
        if index < 0:
            index += self.n
        if index < 0 or index >= self.n:
            raise IndexError(index)
        with open_tensor(self.handle) as flat:
            row = np.array(flat[index], copy=True)
        reshaped = row.reshape(self.flat_shape)
        return _to_like(reshaped, self.like)

    def __iter__(self) -> Iterator[Any]:
        if self._released:
            raise RuntimeError("FlatTensorBatch has already been released")
        with open_tensor(self.handle) as flat:
            data = np.array(flat, copy=True)
        for i in range(self.n):
            reshaped = data[i].reshape(self.flat_shape)
            yield _to_like(reshaped, self.like)

    def materialize_list(self) -> List[Any]:
        """Return a ``List[Tensor]`` equivalent to ``list(self)`` in one SHM read."""
        if self._released:
            raise RuntimeError("FlatTensorBatch has already been released")
        with open_tensor(self.handle) as flat:
            data = np.array(flat, copy=True)
        return [_to_like(data[i].reshape(self.flat_shape), self.like) for i in range(self.n)]

    def as_numpy(self) -> np.ndarray:
        """Return a contiguous ``(n, feature_dim)`` numpy copy of the batch."""
        if self._released:
            raise RuntimeError("FlatTensorBatch has already been released")
        with open_tensor(self.handle) as flat:
            return np.array(flat, copy=True)

    def release(self) -> None:
        """Unlink the backing shared memory if we own it. Idempotent."""
        if self._released:
            return
        self._released = True
        if self.owns_handle:
            cleanup_tensor(self.handle)

    def disown(self) -> None:
        """
        Relinquish ownership without releasing. Callers that transfer the
        backing ``handle`` to another owner (e.g., an operator's internal
        state) should invoke this to prevent double-free by the scheduler.
        """
        self.owns_handle = False


def _to_like(arr: np.ndarray, like: Any) -> Any:
    if _HAS_TORCH and isinstance(like, torch.Tensor):  # type: ignore[arg-type]
        return torch.from_numpy(arr).to(dtype=like.dtype, device=like.device)
    if isinstance(like, np.ndarray):
        return np.asarray(arr, dtype=like.dtype)
    return arr


def release_if_batch(value: Any) -> None:
    """Release a value if it is a FlatTensorBatch we own; otherwise no-op."""
    if isinstance(value, FlatTensorBatch):
        value.release()


__all__ = ["FlatTensorBatch", "release_if_batch"]
