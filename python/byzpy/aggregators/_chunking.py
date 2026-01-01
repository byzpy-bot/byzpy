from __future__ import annotations

import math
import os
from typing import Optional


def _read_int_env(name: str) -> int | None:
    val = os.getenv(name)
    if val is None:
        return None
    try:
        parsed = int(val, 10)
    except ValueError:
        return None
    return max(1, parsed)


def _read_float_env(name: str) -> float | None:
    val = os.getenv(name)
    if val is None:
        return None
    try:
        parsed = float(val)
    except ValueError:
        return None
    return max(parsed, 0.0)


def select_adaptive_chunk_size(
    total_items: int,
    configured_chunk: int,
    *,
    pool_size: Optional[int] = None,
    min_chunks_per_worker: int = 4,
    max_shrink_factor: int = 8,
    allow_small_chunks: bool = False,
) -> int:
    """
    Heuristic for chunk sizing: keep enough subtasks in flight
    to saturate the worker pool while avoiding tiny fragments.
    Environment overrides:
        BYZPY_CHUNK_MIN_PER_WORKER: int >=1 overriding ``min_chunks_per_worker``.
        BYZPY_CHUNK_MAX_SHRINK: int >=1 overriding ``max_shrink_factor``.
        BYZPY_CHUNK_TARGET_FACTOR: float >=0 scaling the chunks/worker target.
    """
    if total_items <= 0:
        return 0
    chunk = max(1, int(configured_chunk))
    if pool_size is None or pool_size <= 1:
        return min(chunk, total_items)

    env_min = _read_int_env("BYZPY_CHUNK_MIN_PER_WORKER")
    env_max_shrink = _read_int_env("BYZPY_CHUNK_MAX_SHRINK")
    env_target = _read_float_env("BYZPY_CHUNK_TARGET_FACTOR")

    min_chunks_per_worker = max(1, int(env_min or min_chunks_per_worker))
    max_shrink_factor = max(1, int(env_max_shrink or max_shrink_factor))
    target_factor = env_target if env_target is not None and env_target > 0 else 1.0

    if (not allow_small_chunks) and total_items <= chunk * pool_size:
        return min(chunk, total_items)

    effective_chunk = min(chunk, total_items)
    current_chunks = max(1, math.ceil(total_items / effective_chunk))
    desired_chunks = math.ceil(min_chunks_per_worker * pool_size * target_factor)
    target_chunks = max(desired_chunks, current_chunks)
    auto_chunk = max(1, math.ceil(total_items / target_chunks))
    min_chunk = max(1, effective_chunk // max_shrink_factor)

    tuned_chunk = max(min_chunk, auto_chunk)
    return min(effective_chunk, tuned_chunk, total_items)


__all__ = ["select_adaptive_chunk_size"]
