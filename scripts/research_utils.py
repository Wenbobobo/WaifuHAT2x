"""Small, dependency-light helpers shared by retained research tools."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch


_STAT_ALIASES = {
    "inference_seconds": (
        "inference_seconds",
        "gpu_inference_seconds",
        "compute_seconds",
        "seconds",
    ),
    "model_load_seconds": ("model_load_seconds", "load_seconds"),
    "assembly_seconds": (
        "assembly_seconds",
        "output_assembly_seconds",
        "host_assembly_seconds",
        "device_assembly_seconds",
    ),
    "transfer_seconds": (
        "transfer_seconds",
        "device_to_host_seconds",
        "d2h_seconds",
    ),
    "native_scale": ("native_scale", "scale"),
    "peak_vram_bytes": (
        "peak_vram_bytes",
        "peak_memory_allocated_bytes",
        "peak_memory_bytes",
    ),
    "peak_reserved_vram_bytes": (
        "peak_reserved_vram_bytes",
        "peak_memory_reserved_bytes",
    ),
    "tile_count": ("tile_count", "tiles"),
    "precision": ("precision", "dtype"),
    "tile": ("tile", "effective_tile"),
    "overlap": ("overlap", "effective_overlap"),
    "batch_tiles": ("batch_tiles", "effective_batch_tiles"),
    "device_assembly": ("device_assembly", "assembled_on_device"),
    "h2d_seconds": ("h2d_seconds",),
    "forward_seconds": ("forward_seconds",),
    "gpu_postprocess_seconds": ("gpu_postprocess_seconds",),
    "d2h_seconds": ("d2h_seconds",),
    "cpu_prepare_seconds": ("cpu_prepare_seconds",),
    "gpu_timing_backend": ("gpu_timing_backend",),
    "gpu_timing_error": ("gpu_timing_error",),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def safe_name(value: str, limit: int = 80) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return (normalized or "benchmark")[:limit]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def stats_mapping(stats: object) -> dict[str, Any]:
    if is_dataclass(stats) and not isinstance(stats, type):
        payload = asdict(stats)
    elif hasattr(stats, "_asdict"):
        payload = stats._asdict()  # type: ignore[attr-defined]
    elif hasattr(stats, "__dict__"):
        payload = vars(stats)
    else:
        payload = {}
        for aliases in _STAT_ALIASES.values():
            for name in aliases:
                if hasattr(stats, name):
                    payload[name] = getattr(stats, name)
    return json_safe(payload)


def save_lossless_png(array: np.ndarray, path: Path, compress_level: int) -> float:
    started = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "L" if array.ndim == 2 else "RGB"
    Image.fromarray(array, mode=mode).save(
        path,
        format="PNG",
        compress_level=compress_level,
        optimize=False,
    )
    return time.perf_counter() - started


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def environment_report() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu: dict[str, Any] | None = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(properties.total_memory),
            "multiprocessor_count": int(properties.multi_processor_count),
            "device_index": 0,
        }
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cuda_api_available": cuda_available,
        "gpu": gpu,
    }
