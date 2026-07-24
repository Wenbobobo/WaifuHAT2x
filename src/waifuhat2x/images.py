from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


IMAGE_EXTENSIONS: Final = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class ResolutionPlan:
    upscale: bool
    native_scale: int
    output_width: int
    output_height: int
    reason: str


def plan_resolution(
    width: int,
    height: int,
    target_short_edge: int,
    max_long_edge_for_sr: int,
    available_scales: tuple[int, ...],
    max_upscale_factor: int,
    max_output_long_edge: int,
    max_output_megapixels: float,
) -> ResolutionPlan:
    """Choose the smallest native SR scale that can reach the requested short edge."""
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive")
    short_edge = min(width, height)
    long_edge = max(width, height)
    if short_edge >= target_short_edge:
        return ResolutionPlan(False, 1, width, height, "short edge already meets target")
    if long_edge > max_long_edge_for_sr:
        return ResolutionPlan(False, 1, width, height, "long edge exceeds SR safety limit")

    scales = tuple(sorted({scale for scale in available_scales if 1 < scale <= max_upscale_factor}))
    if not scales:
        raise ValueError("No supported upscale factor is available")
    reaching = [scale for scale in scales if short_edge * scale >= target_short_edge]
    native_scale = reaching[0] if reaching else scales[-1]

    if short_edge * native_scale < target_short_edge:
        plan = ResolutionPlan(
            True,
            native_scale,
            width * native_scale,
            height * native_scale,
            f"maximum {native_scale}x remains below target",
        )
    else:
        if width <= height:
            output_width = target_short_edge
            output_height = max(1, round(height * target_short_edge / width))
        else:
            output_height = target_short_edge
            output_width = max(1, round(width * target_short_edge / height))
        plan = ResolutionPlan(
            True,
            native_scale,
            output_width,
            output_height,
            f"{native_scale}x then resize to target short edge",
        )
    output_pixels = plan.output_width * plan.output_height
    if (
        max(plan.output_width, plan.output_height) > max_output_long_edge
        or output_pixels > max_output_megapixels * 1_000_000
    ):
        return ResolutionPlan(False, 1, width, height, "planned output exceeds safety limit")
    return plan


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).div_(255.0)


def is_grayscale(image: Image.Image, tolerance: int = 3) -> bool:
    if image.mode in {"1", "L", "I;16", "I", "F"}:
        return True
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    # Chroma-subsampled WebP/JPEG can create stronger colored fringes around a
    # small percentage of hard black/white edges. The first two checks tolerate
    # that sparse noise and reject broad tints; the strongest-1% mean prevents a
    # compact saturated stamp or page number from disappearing into the global
    # average. Strong chroma is deliberately biased toward preserving RGB:
    # a false color classification costs a little space, while a false grayscale
    # classification permanently discards information in replace mode.
    histogram = np.bincount(spread.ravel(), minlength=256)
    pixel_count = spread.size
    cumulative = np.cumsum(histogram)
    percentile_rank = (pixel_count - 1) * 0.95
    lower_rank = int(percentile_rank)
    upper_rank = min(lower_rank + 1, pixel_count - 1)
    lower_chroma = int(np.searchsorted(cumulative, lower_rank + 1))
    upper_chroma = int(np.searchsorted(cumulative, upper_rank + 1))
    percentile_95 = lower_chroma + (upper_chroma - lower_chroma) * (
        percentile_rank - lower_rank
    )
    chroma_sum = int(histogram @ np.arange(256, dtype=np.int64))
    if percentile_95 > tolerance or chroma_sum / pixel_count > tolerance:
        return False
    strong = spread > max(32, 4 * tolerance)
    if strong.shape[0] >= 3 and strong.shape[1] >= 3:
        dense_chroma = strong[:-2, :-2].copy()
        for y_offset in range(3):
            for x_offset in range(3):
                if y_offset or x_offset:
                    dense_chroma &= strong[
                        y_offset : y_offset + strong.shape[0] - 2,
                        x_offset : x_offset + strong.shape[1] - 2,
                    ]
        if bool(dense_chroma.any()):
            return False
    very_strong_threshold = max(128, 16 * tolerance)
    if very_strong_threshold < len(histogram) and int(histogram[very_strong_threshold:].sum()) >= 4:
        return False
    tail_count = max(1, (pixel_count + 99) // 100)
    remaining = tail_count
    strongest_chroma_total = 0
    for chroma in range(255, -1, -1):
        selected = min(remaining, int(histogram[chroma]))
        strongest_chroma_total += selected * chroma
        remaining -= selected
        if remaining == 0:
            break
    return strongest_chroma_total / tail_count <= max(32, 4 * tolerance)


def srgb_to_linear(values: torch.Tensor) -> torch.Tensor:
    return torch.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(values: torch.Tensor) -> torch.Tensor:
    return torch.where(values <= 0.0031308, values * 12.92, 1.055 * values.clamp_min(0) ** (1 / 2.4) - 0.055)


def resize_linear_light(array: np.ndarray, width: int, height: int) -> np.ndarray:
    """Lanczos-like antialiased resize in linear light using PyTorch area-aware bicubic."""
    if array.shape[1] == width and array.shape[0] == height:
        return array
    channels = 1 if array.ndim == 2 else array.shape[2]
    source = torch.from_numpy(array.copy()).float().div_(255.0)
    if channels == 1:
        source = source.unsqueeze(0).unsqueeze(0)
    else:
        source = source.permute(2, 0, 1).unsqueeze(0)
    linear = srgb_to_linear(source)
    resized = F.interpolate(linear, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
    result = linear_to_srgb(resized).clamp_(0, 1).mul_(255).round().to(torch.uint8)
    if channels == 1:
        return result[0, 0].numpy()
    return result[0].permute(1, 2, 0).contiguous().numpy()


def output_path_for(source: Path, input_root: Path, output_root: Path, extension: str) -> Path:
    relative = source.relative_to(input_root)
    return (output_root / relative).with_suffix(f".{extension.lower()}")
