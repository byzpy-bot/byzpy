from __future__ import annotations

import random
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..aggregators._chunking import select_adaptive_chunk_size
from ..aggregators.coordinate_wise._tiling import as_flat_batch
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


class Bucketing(PreAggregator):
    """
    Bucketing pre-aggregator: group vectors into buckets and average.

    This pre-aggregator randomly permutes input vectors, splits them into
    consecutive buckets of a specified size, and returns the mean of each
    bucket. This reduces the number of vectors while preserving some
    statistical properties.

    Algorithm:
    1. Randomly permute the input vectors (or use provided permutation).
    2. Split into consecutive buckets of size ``bucket_size``.
    3. Return the mean of each bucket.

    Parameters
    ----------
    bucket_size : int
        Number of vectors per bucket. Must be >= 1. The last bucket may be
        smaller if the number of vectors is not divisible by bucket_size.
    feature_chunk_size : int, optional
        Size of feature chunks for parallel processing. Default is 8192.
    perm : Optional[Iterable[int]], optional
        Explicit permutation of indices. If None, a random permutation is
        generated. Must be a permutation of range(n) where n is the number
        of input vectors.
    rng : Optional[random.Random], optional
        Random number generator for shuffling. If None, a new generator is
        created.

    Examples
    --------
    >>> preagg = Bucketing(bucket_size=4)
    >>> vectors = [torch.randn(100) for _ in range(10)]
    >>> result = preagg.pre_aggregate(vectors)
    >>> len(result)  # 10 vectors -> ceil(10/4) = 3 buckets
    3
    >>> assert all(v.shape == (100,) for v in result)

    Notes
    -----
    - Output length is ceil(n / bucket_size) where n is input length.
    - Supports parallel execution via subtasks for large feature dimensions.
    - Time complexity: O(n * d) where n is number of vectors and d is
      dimension. With subtasks: O(n * d / workers).
    - Memory complexity: O(n * d) for stacking vectors.
    """

    name = "pre-agg/bucketing"
    supports_subtasks = True
    max_subtasks_inflight = 0

    def __init__(
        self,
        bucket_size: int,
        *,
        feature_chunk_size: int = 8192,
        perm: Optional[Iterable[int]] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        if bucket_size < 1:
            raise ValueError("bucket_size must be >= 1")
        self.bucket_size = int(bucket_size)
        if feature_chunk_size <= 0:
            raise ValueError("feature_chunk_size must be > 0")
        self.feature_chunk_size = int(feature_chunk_size)
        self.perm = None if perm is None else [int(i) for i in perm]
        self.rng = rng or random.Random()
        self._active_handle: SharedTensorHandle | None = None
        self._active_owned: bool = False
        self._flat_shape: tuple[int, ...] | None = None
        self._bucket_count: int | None = None
        self._like_template: Any | None = None

    def pre_aggregate(self, xs: Sequence[Any]) -> List[Any]:
        if not xs:
            raise ValueError("xs must be a non-empty sequence")

        n = len(xs)
        be = get_backend()
        if isinstance(xs, FlatTensorBatch):
            like = xs.like
        else:
            like = xs[0]

        order = self._resolve_order(n)

        arrs = [be.asarray(xs[i], like=like) for i in order]
        X = be.stack(arrs, axis=0)

        out: List[Any] = []
        for start in range(0, n, self.bucket_size):
            stop = min(start + self.bucket_size, n)
            chunk = X[start:stop]
            mean = be.mean(chunk, axis=0)
            out.append(mean)
        return out

    def create_subtasks(self, inputs, *, context):  # type: ignore[override]
        xs = inputs.get(self.input_key)
        if xs is None:
            return []
        if isinstance(xs, FlatTensorBatch):
            if len(xs) == 0:
                return []
            input_batch = xs
            active_owned = False
        else:
            if not isinstance(xs, Sequence) or not xs:
                return []
            input_batch = as_flat_batch(xs)
            active_owned = True

        n = len(input_batch)
        order = self._resolve_order(n)

        self._active_handle = input_batch.handle
        self._active_owned = active_owned
        self._flat_shape = input_batch.flat_shape
        self._like_template = input_batch.like

        bucket_index_arrays: List[np.ndarray] = []
        for start in range(0, n, self.bucket_size):
            stop = min(start + self.bucket_size, n)
            bucket_index_arrays.append(np.asarray(order[start:stop], dtype=np.int64))
        bucket_count = len(bucket_index_arrays)
        self._bucket_count = bucket_count

        metadata = getattr(context, "metadata", None) or {}
        pool_size = int(metadata.get("pool_size") or 0)
        bucket_chunk = select_adaptive_chunk_size(
            bucket_count,
            max(1, self.feature_chunk_size),
            pool_size=pool_size,
            allow_small_chunks=True,
        )

        in_handle = input_batch.handle

        def _iter() -> Iterable[SubTask]:
            chunk_id = 0
            for start in range(0, bucket_count, bucket_chunk):
                end = min(bucket_count, start + bucket_chunk)
                yield SubTask(
                    fn=_bucketing_bucket_chunk,
                    args=(in_handle, tuple(bucket_index_arrays[start:end]), start),
                    kwargs={},
                    name=f"bucketing_chunk_{chunk_id}",
                )
                chunk_id += 1

        return _iter()

    def reduce_subtasks(self, partials, inputs, *, context):  # type: ignore[override]
        if not partials:
            self._cleanup_state()
            return super().compute(inputs, context=context)

        if (
            self._active_handle is None
            or self._flat_shape is None
            or self._bucket_count is None
            or self._like_template is None
        ):
            raise RuntimeError("Bucketing missing reduction state.")

        feature_dim = int(np.prod(self._flat_shape))
        buckets = self._bucket_count

        out_dtype = np.dtype(self._active_handle.dtype)
        assembled = np.zeros((buckets, feature_dim), dtype=out_dtype)

        for idx, part in enumerate(partials):
            try:
                offset, chunk = part
            except Exception as exc:  # pragma: no cover
                raise ValueError(
                    f"Bucketing received malformed partial at index {idx}: {part!r}"
                ) from exc
            chunk_np = np.asarray(chunk)
            rows = chunk_np.shape[0]
            if chunk_np.dtype != out_dtype:
                assembled[offset : offset + rows, :] = chunk_np.astype(out_dtype, copy=False)
            else:
                assembled[offset : offset + rows, :] = chunk_np

        flat_shape = self._flat_shape
        like = self._like_template

        if self._active_owned:
            cleanup_tensor(self._active_handle)
        self._active_handle = None
        self._active_owned = False
        self._flat_shape = None
        self._bucket_count = None
        self._like_template = None

        if _flat_batch_disabled():
            results: List[Any] = []
            for bucket_idx in range(buckets):
                flat_bucket = assembled[bucket_idx]
                reshaped = flat_bucket.reshape(flat_shape)
                results.append(_to_like(reshaped, like))
            return results

        out_handle = register_tensor(assembled)
        return FlatTensorBatch(
            handle=out_handle,
            flat_shape=flat_shape,
            like=like,
            owns_handle=True,
        )

    def _cleanup_state(self) -> None:
        if self._active_owned and self._active_handle is not None:
            cleanup_tensor(self._active_handle)
        self._active_handle = None
        self._active_owned = False
        self._flat_shape = None
        self._bucket_count = None
        self._like_template = None

    def _resolve_order(self, n: int) -> List[int]:
        if self.perm is None:
            order = list(range(n))
            self.rng.shuffle(order)
            return order
        if len(self.perm) != n or sorted(self.perm) != list(range(n)):
            raise ValueError("perm must be a permutation of range(n)")
        return list(self.perm)


def _bucketing_bucket_chunk(
    handle: SharedTensorHandle,
    bucket_indices: Tuple[np.ndarray, ...],
    offset: int,
) -> tuple[int, np.ndarray]:
    with open_tensor(handle) as flat:
        feature_dim = flat.shape[1]
        out = np.zeros((len(bucket_indices), feature_dim), dtype=flat.dtype)
        for idx, indices in enumerate(bucket_indices):
            if len(indices) == 0:
                continue
            rows = flat[indices]
            out[idx, :] = np.mean(rows, axis=0)
    return offset, out


def _to_like(arr: np.ndarray, like: Any) -> Any:
    if _HAS_TORCH and isinstance(like, torch.Tensor):  # type: ignore[arg-type]
        return torch.from_numpy(arr).to(dtype=like.dtype, device=like.device)
    be = get_backend()
    return be.asarray(arr, like=like)
