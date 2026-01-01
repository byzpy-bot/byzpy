from __future__ import annotations

import os
import shutil
import subprocess
from typing import List

# Core dependencies required for any install of byzpy
_CPU_DEPENDENCIES: List[str] = [
    "torch>=2.2,<2.6",
    "torchvision>=0.17,<0.21",
    "numpy>=1.24,<2.1",
    "matplotlib>=3.8,<3.11",
    "cloudpickle>=2.2,<4.0",
]

# GPU-specific dependencies that should only be present when CUDA hardware is available.
_GPU_DEPENDENCIES: List[str] = [
    "cupy-cuda12x>=13.2.0",
    "ucxx-cu12>=0.38",
]

# Developer tooling for running the project's internal test-suite.
_DEV_DEPENDENCIES: List[str] = [
    "pytest==8.3.3",
    "pytest-asyncio==0.23.7",
    "pytest-cov==5.0.0",
]

_FORCE_GPU_ENV = "BYZPY_FORCE_GPU"
_FORCE_CPU_ENV = "BYZPY_FORCE_CPU"


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _has_nvidia_smi() -> bool:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return False
    try:
        completed = subprocess.run(
            [binary],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _has_cuda_env() -> bool:
    # CUDA_VISIBLE_DEVICES is populated with comma separated GPU ids when CUDA is usable.
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible and cuda_visible.strip() not in {"", "-1"}:
        return True
    if os.environ.get("CUDA_HOME"):
        return True
    return False


def _should_install_gpu() -> bool:
    if _env_flag(_FORCE_CPU_ENV):
        return False
    if _env_flag(_FORCE_GPU_ENV):
        return True
    return _has_nvidia_smi() or _has_cuda_env()


def get_dependencies() -> List[str]:
    deps = list(_CPU_DEPENDENCIES)
    if _should_install_gpu():
        deps.extend(_GPU_DEPENDENCIES)
    return deps


def get_gpu_optional_dependencies() -> List[str]:
    return list(_GPU_DEPENDENCIES)


def get_dev_optional_dependencies() -> List[str]:
    return list(_DEV_DEPENDENCIES)
