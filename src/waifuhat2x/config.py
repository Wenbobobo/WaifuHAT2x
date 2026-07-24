from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import tomllib


@dataclass(frozen=True)
class PathsConfig:
    input: Path
    output: Path
    models: Path


@dataclass(frozen=True)
class ProcessingConfig:
    profile: str = "hat-auto"
    target_short_edge: int = 1600
    real_hat_sharper_min_short_edge: int = 1000
    max_long_edge_for_sr: int = 3200
    max_upscale_factor: int = 4
    max_output_long_edge: int = 6400
    max_output_megapixels: float = 24.0
    precision: str = "auto"
    tile: int = 256
    overlap: int = 64
    hat_tile: int = 256
    hat_overlap: int = 32
    batch_tiles: int = 1
    device_assembly: bool = True
    model_cache_size: int = 2
    grayscale_tolerance: int = 3
    linear_light_downscale: bool = True
    hat_tile_candidates: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        candidates = self.hat_tile_candidates or (self.hat_tile,)
        object.__setattr__(self, "hat_tile_candidates", tuple(candidates))


@dataclass(frozen=True)
class OutputConfig:
    mode: str = "mirror"
    format: str = "jxl"
    webp_lossless: bool = True
    webp_method: int = 4
    copy_non_images: bool = True
    overwrite: bool = False
    existing_jxl_policy: str = "error"
    allow_lossy_replace: bool = False
    allow_metadata_loss: bool = False
    allow_alpha_flatten: bool = False
    allow_bit_depth_loss: bool = False


@dataclass(frozen=True)
class JxlConfig:
    distance: float = 0.3
    effort: int = 7
    threads: int = 4
    workers: int = 1
    queue_depth: int = 2
    verify_decode: bool = True


@dataclass(frozen=True)
class AppConfig:
    source_file: Path
    paths: PathsConfig
    processing: ProcessingConfig
    output: OutputConfig
    jxl: JxlConfig


def _resolve(base: Path, value: str) -> Path:
    if os.name == "posix":
        windows = re.match(r"^([A-Za-z]):[\\/](.*)$", value)
        if windows:
            tail = windows.group(2).replace("\\", "/")
            return (Path("/mnt") / windows.group(1).lower() / tail).resolve()
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> AppConfig:
    source = Path(path).resolve()
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    base = source.parent
    path_values = raw.get("paths", {})
    processing_values = dict(raw.get("processing", {}))
    if "hat_tile_candidates" in processing_values:
        candidates = processing_values["hat_tile_candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(
                "processing.hat_tile_candidates must be a non-empty TOML array"
            )
        processing_values["hat_tile_candidates"] = tuple(candidates)
    processing = ProcessingConfig(**processing_values)
    output = OutputConfig(**raw.get("output", {}))

    if processing.target_short_edge < 1:
        raise ValueError("processing.target_short_edge must be positive")
    if processing.profile.lower() == "real-hat-auto":
        threshold = processing.real_hat_sharper_min_short_edge
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, int)
            or not 1 <= threshold < processing.target_short_edge
        ):
            raise ValueError(
                "processing.real_hat_sharper_min_short_edge must satisfy "
                "1 <= threshold < processing.target_short_edge for real-hat-auto"
            )
        if processing.max_upscale_factor != 4:
            raise ValueError("processing.max_upscale_factor must be 4 for real-hat-auto")
        if processing.model_cache_size < 2:
            raise ValueError(
                "processing.model_cache_size must be at least 2 for real-hat-auto"
            )
    if processing.max_long_edge_for_sr < processing.target_short_edge:
        raise ValueError("processing.max_long_edge_for_sr must be >= target_short_edge")
    if processing.max_upscale_factor not in (2, 4):
        raise ValueError("processing.max_upscale_factor must be 2 or 4")
    if processing.max_output_long_edge < processing.target_short_edge:
        raise ValueError("processing.max_output_long_edge must be >= target_short_edge")
    if processing.max_output_megapixels <= 0:
        raise ValueError("processing.max_output_megapixels must be positive")
    for tile_name, overlap_name in (("tile", "overlap"), ("hat_tile", "hat_overlap")):
        tile = getattr(processing, tile_name)
        overlap = getattr(processing, overlap_name)
        if tile <= 0 or tile % 16:
            raise ValueError(f"processing.{tile_name} must be a positive multiple of 16")
        if overlap < 0 or overlap % 8:
            raise ValueError(f"processing.{overlap_name} must be a non-negative multiple of 8")
        if (tile + 2 * overlap) % 16:
            raise ValueError(f"{tile_name} + 2 * {overlap_name} must be divisible by 16")
    candidates = processing.hat_tile_candidates
    for candidate in candidates:
        if isinstance(candidate, bool) or not isinstance(candidate, int):
            raise ValueError(
                "processing.hat_tile_candidates must contain only integers"
            )
        if candidate <= 0 or candidate % 16:
            raise ValueError(
                "processing.hat_tile_candidates must contain positive multiples of 16"
            )
        if (candidate + 2 * processing.hat_overlap) % 16:
            raise ValueError(
                "each HAT tile candidate + 2 * processing.hat_overlap "
                "must be divisible by 16"
            )
    if candidates[0] != processing.hat_tile:
        raise ValueError(
            "processing.hat_tile_candidates must start with processing.hat_tile"
        )
    if len(set(candidates)) != len(candidates):
        raise ValueError("processing.hat_tile_candidates must contain unique values")
    if processing.batch_tiles not in {1, 2, 4, 8}:
        raise ValueError("processing.batch_tiles must be one of 1, 2, 4, or 8")
    if not isinstance(processing.device_assembly, bool):
        raise ValueError("processing.device_assembly must be true or false")
    if not 1 <= processing.model_cache_size <= 4:
        raise ValueError("processing.model_cache_size must be between 1 and 4")
    if processing.precision not in {"auto", "fp16", "bf16", "fp32"}:
        raise ValueError("processing.precision must be auto, fp16, bf16, or fp32")
    if output.format.lower() not in {"jxl", "webp", "png"}:
        raise ValueError("output.format must be jxl, webp, or png")
    if output.mode not in {"mirror", "replace"}:
        raise ValueError("output.mode must be mirror or replace")
    if output.mode == "replace" and output.format.lower() != "jxl":
        raise ValueError("output.mode = replace currently requires output.format = jxl")
    if output.mode == "replace" and output.overwrite:
        raise ValueError("output.overwrite must remain false in replace mode")
    if output.existing_jxl_policy not in {"error", "replace"}:
        raise ValueError("output.existing_jxl_policy must be error or replace")
    if output.mode != "replace" and output.existing_jxl_policy != "error":
        raise ValueError("output.existing_jxl_policy is only available in replace mode")
    jxl = JxlConfig(**raw.get("jxl", {}))
    if not 0 <= jxl.distance <= 15:
        raise ValueError("jxl.distance must be between 0 and 15")
    if not 1 <= jxl.effort <= 10:
        raise ValueError("jxl.effort must be between 1 and 10")
    if jxl.threads < 1 or jxl.workers != 1 or jxl.queue_depth < 1:
        raise ValueError("JXL requires threads >= 1, workers = 1, and queue_depth >= 1")
    if output.mode == "replace" and jxl.distance != 0 and not output.allow_lossy_replace:
        raise ValueError(
            "Lossy JXL replacement requires output.allow_lossy_replace = true or jxl.distance = 0"
        )
    if output.mode == "replace" and not jxl.verify_decode:
        raise ValueError("output.mode = replace requires jxl.verify_decode = true")

    return AppConfig(
        source_file=source,
        paths=PathsConfig(
            input=_resolve(base, path_values.get("input", "Test")),
            output=_resolve(base, path_values.get("output", "Output")),
            models=_resolve(base, path_values.get("models", "models")),
        ),
        processing=processing,
        output=output,
        jxl=jxl,
    )
