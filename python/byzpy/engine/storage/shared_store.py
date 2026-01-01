from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Iterator, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class SharedTensorHandle:
    name: str
    shape: Tuple[int, ...]
    dtype: str


SharedHandleLike = Union["SharedTensorHandle", dict]


def register_tensor(array: np.ndarray) -> SharedTensorHandle:
    arr = np.asarray(array)
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    try:
        buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        buf[:] = arr
    finally:
        shm.close()
    return SharedTensorHandle(name=shm.name, shape=arr.shape, dtype=str(arr.dtype))


@contextmanager
def open_tensor(handle: SharedHandleLike) -> Iterator[np.ndarray]:
    if isinstance(handle, dict):
        handle = SharedTensorHandle(**handle)
    shm = shared_memory.SharedMemory(name=handle.name)
    try:
        arr = np.ndarray(handle.shape, dtype=np.dtype(handle.dtype), buffer=shm.buf)
        yield arr
    finally:
        shm.close()


def cleanup_tensor(handle: SharedHandleLike) -> None:
    if isinstance(handle, dict):
        handle = SharedTensorHandle(**handle)
    try:
        shm = shared_memory.SharedMemory(name=handle.name)
    except FileNotFoundError:
        return
    try:
        shm.unlink()
    finally:
        shm.close()
