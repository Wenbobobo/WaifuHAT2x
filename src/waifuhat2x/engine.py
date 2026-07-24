from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import gc
import time
from typing import Iterator
import weakref

import numpy as np
from spandrel import ImageModelDescriptor, ModelLoader
import torch
from torch.nn import functional as F


HAT_TILE_STRATEGY = "min-padded-work-v1"
HAT_TILE_ESTIMATOR = "ceil(width/tile)*ceil(height/tile)*(tile+2*overlap)^2"


@dataclass(frozen=True)
class TileWorkEstimate:
    tile: int
    tiles_x: int
    tiles_y: int
    tile_count: int
    expanded_edge: int
    expanded_tile_area: int
    estimated_work: int


def estimate_hat_tile_work(
    width: int, height: int, tile: int, overlap: int
) -> TileWorkEstimate:
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive")
    if tile < 1:
        raise ValueError("Tile size must be positive")
    if overlap < 0:
        raise ValueError("Tile overlap must be non-negative")
    tiles_x = (width + tile - 1) // tile
    tiles_y = (height + tile - 1) // tile
    expanded_edge = tile + 2 * overlap
    tile_count = tiles_x * tiles_y
    expanded_tile_area = expanded_edge**2
    return TileWorkEstimate(
        tile=tile,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
        tile_count=tile_count,
        expanded_edge=expanded_edge,
        expanded_tile_area=expanded_tile_area,
        estimated_work=tile_count * expanded_tile_area,
    )


def choose_hat_tile(
    width: int, height: int, candidates: tuple[int, ...], overlap: int
) -> tuple[int, tuple[TileWorkEstimate, ...]]:
    if not candidates:
        raise ValueError("At least one HAT tile candidate is required")
    estimates = tuple(
        estimate_hat_tile_work(width, height, tile, overlap) for tile in candidates
    )
    selected = min(
        estimates, key=lambda estimate: (estimate.estimated_work, estimate.tile)
    )
    return selected.tile, estimates


@dataclass(frozen=True)
class InferenceStats:
    seconds: float
    native_scale: int
    peak_vram_bytes: int
    tile_count: int
    precision: str
    tile: int
    overlap: int
    batch_tiles: int
    assembly: str
    model_load_seconds: float
    model_cache_hit: bool
    h2d_seconds: float | None = None
    forward_seconds: float | None = None
    gpu_postprocess_seconds: float | None = None
    d2h_seconds: float | None = None
    cpu_prepare_seconds: float | None = None
    gpu_timing_backend: str | None = None
    gpu_timing_error: str | None = None
    gpu_timing_warning: str | None = None
    gpu_event_total_seconds: float | None = None
    gpu_event_scale_to_wall: float | None = None
    gpu_event_raw_seconds: dict[str, float] | None = None
    model_load_interval_ns: tuple[int, int] | None = None
    peak_reserved_vram_bytes: int | None = None
    inference_interval_ns: tuple[int, int] | None = None
    tile_candidates: tuple[int, ...] = ()
    tile_strategy: str = "fixed"
    tile_estimator: str | None = None
    tile_estimates: tuple[TileWorkEstimate, ...] = ()


class _GpuPhaseRecorder:
    """Optional HIP-event timing behind PyTorch's CUDA-compatible ROCm API."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.error: str | None = None
        self._events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def _disable(self, error: BaseException) -> None:
        self.enabled = False
        self.error = f"{type(error).__name__}: {error}"
        self._events.clear()

    def begin(self) -> torch.cuda.Event | None:
        if not self.enabled:
            return None
        try:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        except Exception as exc:
            self._disable(exc)
            return None

    def end(self, name: str, started: torch.cuda.Event | None) -> None:
        if not self.enabled or started is None:
            return
        try:
            ended = torch.cuda.Event(enable_timing=True)
            ended.record()
            self._events.setdefault(name, []).append((started, ended))
        except Exception as exc:
            self._disable(exc)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = self.begin()
        try:
            yield
        finally:
            self.end(name, started)

    def seconds(self) -> dict[str, float] | None:
        if not self.enabled or self.error is not None:
            return None
        try:
            return {
                name: sum(start.elapsed_time(end) for start, end in pairs) / 1000.0
                for name, pairs in self._events.items()
            }
        except Exception as exc:
            self._disable(exc)
            return None


class _HatMaskCache:
    """Small device-side LRU for Spandrel HAT's deterministic attention mask."""

    max_entries = 2

    def __init__(self, model: torch.nn.Module) -> None:
        original = getattr(model, "calculate_mask")
        if not callable(original):
            raise TypeError("HAT calculate_mask is not callable")

        model_attributes = vars(model)
        self._had_instance_attribute = "calculate_mask" in model_attributes
        self._original_instance_value = model_attributes.get("calculate_mask")
        original_function = getattr(original, "__func__", None)
        original_owner = getattr(original, "__self__", None)
        if original_owner is model and callable(original_function):
            self._original_function: Callable[..., object] | None = original_function
            self._original_callable: Callable[..., object] | None = None
        else:
            self._original_function = None
            self._original_callable = original

        self._model_ref: weakref.ReferenceType[torch.nn.Module] = weakref.ref(model)
        self._entries: OrderedDict[
            tuple[int, int, torch.dtype, torch.device], torch.Tensor
        ] = OrderedDict()

    @staticmethod
    def cache_key(
        x_size: tuple[object, object], dtype: torch.dtype, device: torch.device
    ) -> tuple[int, int, torch.dtype, torch.device]:
        height, width = x_size
        return int(height), int(width), dtype, torch.device(device)

    @staticmethod
    def _model_device(model: torch.nn.Module) -> torch.device:
        for parameter in model.parameters():
            return parameter.device
        for buffer in model.buffers():
            return buffer.device
        raise RuntimeError("HAT model has no tensor from which to infer its device")

    def _call_original(
        self, model: torch.nn.Module, x_size: object, dtype: torch.dtype
    ) -> object:
        if self._original_function is not None:
            return self._original_function(model, x_size, dtype)
        if self._original_callable is not None:
            return self._original_callable(x_size, dtype)
        raise RuntimeError("HAT attention-mask cache has been detached")

    def __call__(self, x_size: object, dtype: torch.dtype) -> object:
        model = self._model_ref()
        if model is None:
            raise RuntimeError("HAT model no longer exists")

        try:
            device = self._model_device(model)
            key = self.cache_key(x_size, dtype, device)  # type: ignore[arg-type]
        except Exception:
            return self._call_original(model, x_size, dtype)

        try:
            cached = self._entries.pop(key)
        except KeyError:
            pass
        else:
            self._entries[key] = cached
            return cached

        mask = self._call_original(model, x_size, dtype)
        if not isinstance(mask, torch.Tensor):
            return mask
        try:
            device_mask = mask.to(device=device)
        except Exception:
            return mask

        self._entries[key] = device_mask
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return device_mask

    def detach(self) -> None:
        model = self._model_ref()
        if model is not None and vars(model).get("calculate_mask") is self:
            if self._had_instance_attribute:
                setattr(model, "calculate_mask", self._original_instance_value)
            else:
                delattr(model, "calculate_mask")
        self._entries.clear()
        self._original_function = None
        self._original_callable = None
        self._original_instance_value = None


class UpscaleEngine:
    def __init__(
        self,
        precision: str,
        tile: int,
        overlap: int,
        batch_tiles: int = 1,
        hat_tile: int | None = None,
        hat_overlap: int | None = None,
        device_assembly: bool = True,
        model_cache_size: int = 2,
        collect_gpu_timing: bool = False,
        hat_tile_candidates: tuple[int, ...] | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm GPU is unavailable to PyTorch (torch.cuda.is_available() is false)")
        self.device = torch.device("cuda:0")
        self.precision = precision
        self.dtype = torch.float32
        self.tile = tile
        self.overlap = overlap
        self.hat_tile = hat_tile if hat_tile is not None else tile
        self.hat_overlap = hat_overlap if hat_overlap is not None else overlap
        self.hat_tile_candidates = tuple(
            hat_tile_candidates
            if hat_tile_candidates is not None
            else (self.hat_tile,)
        )
        if not self.hat_tile_candidates:
            raise ValueError("hat_tile_candidates must not be empty")
        if self.hat_tile_candidates[0] != self.hat_tile:
            raise ValueError("hat_tile_candidates must start with hat_tile")
        self.batch_tiles = batch_tiles
        self.device_assembly = device_assembly
        self.model_cache_size = model_cache_size
        self.collect_gpu_timing = collect_gpu_timing
        self._last_gpu_phase_seconds: dict[str, float] | None = None
        self._last_gpu_timing_error: str | None = None
        self._last_cpu_prepare_seconds: float | None = None
        self._last_inference_interval_ns: tuple[int, int] | None = None
        if batch_tiles < 1:
            raise ValueError("batch_tiles must be positive")
        if model_cache_size < 1:
            raise ValueError("model_cache_size must be positive")
        self._models: OrderedDict[Path, tuple[ImageModelDescriptor, torch.dtype]] = OrderedDict()
        self._hat_mask_caches: dict[Path, _HatMaskCache] = {}
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    @property
    def device_name(self) -> str:
        return torch.cuda.get_device_name(self.device)

    def _load(
        self, path: Path
    ) -> tuple[ImageModelDescriptor, torch.dtype, float, bool]:
        path = path.resolve()
        if path in self._models:
            descriptor, dtype = self._models.pop(path)
            self._models[path] = (descriptor, dtype)
            self.dtype = dtype
            return descriptor, dtype, 0.0, True

        while len(self._models) >= self.model_cache_size:
            evicted_path, (evicted, _) = self._models.popitem(last=False)
            self._detach_hat_mask_cache(evicted_path)
            del evicted
            gc.collect()
            torch.cuda.empty_cache()

        started = time.perf_counter()
        descriptor = ModelLoader().load_from_file(path)
        if not isinstance(descriptor, ImageModelDescriptor):
            raise TypeError(f"Model is not a single-image model: {path}")
        descriptor.eval().to(self.device)
        requested = self.precision
        if requested == "auto":
            if descriptor.supports_half:
                requested = "fp16"
            elif getattr(descriptor, "supports_bfloat16", False):
                requested = "bf16"
            else:
                requested = "fp32"
        if requested == "fp16":
            if not descriptor.supports_half:
                raise RuntimeError(f"Model does not support FP16: {path.name}")
            descriptor.half()
            self.dtype = torch.float16
        elif requested == "bf16":
            if not getattr(descriptor, "supports_bfloat16", False):
                raise RuntimeError(f"Model does not support BF16: {path.name}")
            descriptor.bfloat16()
            self.dtype = torch.bfloat16
        else:
            descriptor.float()
            self.dtype = torch.float32
        self._attach_hat_mask_cache(path, descriptor)
        self._models[path] = (descriptor, self.dtype)
        return descriptor, self.dtype, time.perf_counter() - started, False

    def close(self) -> None:
        for cache in self._hat_mask_caches.values():
            cache.detach()
        self._hat_mask_caches.clear()
        self._models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _spandrel_hat_model(descriptor: ImageModelDescriptor) -> torch.nn.Module | None:
        architecture = getattr(descriptor, "architecture", None)
        model = getattr(descriptor, "model", None)
        model_type = type(model)
        if (
            getattr(architecture, "id", None) != "HAT"
            or model_type.__name__ != "HAT"
            or not model_type.__module__.startswith("spandrel.architectures.HAT.")
            or not isinstance(model, torch.nn.Module)
            or not callable(getattr(model, "calculate_mask", None))
        ):
            return None
        return model

    def _attach_hat_mask_cache(
        self, path: Path, descriptor: ImageModelDescriptor
    ) -> _HatMaskCache | None:
        model = self._spandrel_hat_model(descriptor)
        if model is None:
            return None
        self._detach_hat_mask_cache(path)
        try:
            cache = _HatMaskCache(model)
            setattr(model, "calculate_mask", cache)
        except Exception:
            return None
        self._hat_mask_caches[path] = cache
        return cache

    def _detach_hat_mask_cache(self, path: Path) -> None:
        cache = self._hat_mask_caches.pop(path, None)
        if cache is not None:
            cache.detach()

    @staticmethod
    def _positions(height: int, width: int, tile: int) -> list[tuple[int, int]]:
        return [(y, x) for y in range(0, height, tile) for x in range(0, width, tile)]

    @staticmethod
    def _crop_core_fp32(
        result: torch.Tensor, scale: int, tile: int, overlap: int
    ) -> torch.Tensor:
        crop_start = overlap * scale
        crop_end = (overlap + tile) * scale
        return result[:, :, crop_start:crop_end, crop_start:crop_end].float().clamp_(0, 1)

    def _upscale_host_assembly(
        self,
        image: torch.Tensor,
        descriptor: ImageModelDescriptor,
        dtype: torch.dtype,
        scale: int,
        tile: int,
        overlap: int,
        grayscale_output: bool,
    ) -> tuple[np.ndarray, float, int, int]:
        collect_timing = getattr(self, "collect_gpu_timing", False)
        prepare_started_ns = time.perf_counter_ns() if collect_timing else None
        recorder = _GpuPhaseRecorder(collect_timing)
        self._last_gpu_phase_seconds = None
        self._last_gpu_timing_error = None
        self._last_cpu_prepare_seconds = None
        self._last_inference_interval_ns = None
        scale = int(descriptor.scale)
        _, _, height, width = image.shape
        padded_height = ((height + tile - 1) // tile) * tile
        padded_width = ((width + tile - 1) // tile) * tile
        pad_right = padded_width - width + overlap
        pad_bottom = padded_height - height + overlap
        padding_mode = "reflect"
        if overlap >= height or overlap >= width or pad_right >= width or pad_bottom >= height:
            padding_mode = "replicate"
        padded = F.pad(image, (overlap, pad_right, overlap, pad_bottom), mode=padding_mode)

        channels = 1 if grayscale_output else int(descriptor.output_channels)
        output = np.empty((height * scale, width * scale, channels), dtype=np.uint8)
        positions = self._positions(padded_height, padded_width, tile)
        if prepare_started_ns is not None:
            self._last_cpu_prepare_seconds = (
                time.perf_counter_ns() - prepare_started_ns
            ) / 1_000_000_000
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        started_ns = time.perf_counter_ns()
        started = time.perf_counter()
        total_event = recorder.begin()

        with torch.inference_mode():
            for offset in range(0, len(positions), self.batch_tiles):
                group = positions[offset : offset + self.batch_tiles]
                host_tiles = torch.cat(
                    [
                        padded[
                            :, :, y : y + tile + 2 * overlap, x : x + tile + 2 * overlap
                        ]
                        for y, x in group
                    ],
                    dim=0,
                )
                with recorder.phase("h2d"):
                    tiles = host_tiles.to(
                        device=self.device, dtype=dtype, non_blocking=False
                    )
                with recorder.phase("forward"):
                    result = descriptor(tiles)
                with recorder.phase("gpu_postprocess"):
                    result = self._crop_core_fp32(result, scale, tile, overlap)
                    if grayscale_output:
                        result = (
                            result[:, 0:1] * 0.2126
                            + result[:, 1:2] * 0.7152
                            + result[:, 2:3] * 0.0722
                        )
                    result = result.mul_(255).round_().to(torch.uint8)
                with recorder.phase("d2h"):
                    result = result.cpu()
                for index, (y, x) in enumerate(group):
                    core_height = min(tile, height - y)
                    core_width = min(tile, width - x)
                    if core_height <= 0 or core_width <= 0:
                        continue
                    tile_array = result[index, :, : core_height * scale, : core_width * scale]
                    tile_array = tile_array.permute(1, 2, 0).contiguous().numpy()
                    output[y * scale : (y + core_height) * scale, x * scale : (x + core_width) * scale] = tile_array

        recorder.end("gpu_total", total_event)
        torch.cuda.synchronize(self.device)
        ended = time.perf_counter()
        ended_ns = time.perf_counter_ns()
        self._last_inference_interval_ns = (started_ns, ended_ns)
        self._last_gpu_phase_seconds = recorder.seconds()
        self._last_gpu_timing_error = recorder.error
        elapsed = ended - started
        peak = torch.cuda.max_memory_allocated(self.device)
        if grayscale_output:
            output = output[:, :, 0]
        return output, elapsed, peak, len(positions)

    def _upscale_device_assembly(
        self,
        image: torch.Tensor,
        descriptor: ImageModelDescriptor,
        dtype: torch.dtype,
        scale: int,
        tile: int,
        overlap: int,
        grayscale_output: bool,
    ) -> tuple[np.ndarray, float, int, int]:
        collect_timing = getattr(self, "collect_gpu_timing", False)
        prepare_started_ns = time.perf_counter_ns() if collect_timing else None
        recorder = _GpuPhaseRecorder(collect_timing)
        self._last_gpu_phase_seconds = None
        self._last_gpu_timing_error = None
        self._last_cpu_prepare_seconds = None
        self._last_inference_interval_ns = None
        _, _, height, width = image.shape
        padded_height = ((height + tile - 1) // tile) * tile
        padded_width = ((width + tile - 1) // tile) * tile
        pad_right = padded_width - width + overlap
        pad_bottom = padded_height - height + overlap
        padding_mode = "reflect"
        if overlap >= height or overlap >= width or pad_right >= width or pad_bottom >= height:
            padding_mode = "replicate"
        padded = F.pad(image, (overlap, pad_right, overlap, pad_bottom), mode=padding_mode)
        positions = self._positions(padded_height, padded_width, tile)
        channels = 1 if grayscale_output else int(descriptor.output_channels)
        if prepare_started_ns is not None:
            self._last_cpu_prepare_seconds = (
                time.perf_counter_ns() - prepare_started_ns
            ) / 1_000_000_000

        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        started_ns = time.perf_counter_ns()
        started = time.perf_counter()
        total_event = recorder.begin()
        with recorder.phase("h2d"):
            padded_device = padded.to(
                device=self.device, dtype=dtype, non_blocking=False
            )
        output_device = torch.empty(
            (channels, height * scale, width * scale),
            dtype=torch.uint8,
            device=self.device,
        )

        with torch.inference_mode():
            for offset in range(0, len(positions), self.batch_tiles):
                group = positions[offset : offset + self.batch_tiles]
                parts = [
                    padded_device[:, :, y : y + tile + 2 * overlap, x : x + tile + 2 * overlap]
                    for y, x in group
                ]
                tiles = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
                with recorder.phase("forward"):
                    result = descriptor(tiles)
                with recorder.phase("gpu_postprocess"):
                    result = self._crop_core_fp32(result, scale, tile, overlap)
                    if grayscale_output:
                        result = (
                            result[:, 0:1] * 0.2126
                            + result[:, 1:2] * 0.7152
                            + result[:, 2:3] * 0.0722
                        )
                    result = result.mul_(255).round_().to(torch.uint8)
                    for index, (y, x) in enumerate(group):
                        core_height = min(tile, height - y)
                        core_width = min(tile, width - x)
                        if core_height <= 0 or core_width <= 0:
                            continue
                        output_device[
                            :,
                            y * scale : (y + core_height) * scale,
                            x * scale : (x + core_width) * scale,
                        ] = result[index, :, : core_height * scale, : core_width * scale]

        with recorder.phase("gpu_postprocess"):
            contiguous_output = output_device.permute(1, 2, 0).contiguous()
        with recorder.phase("d2h"):
            output = contiguous_output.cpu().numpy()
        recorder.end("gpu_total", total_event)
        torch.cuda.synchronize(self.device)
        ended = time.perf_counter()
        ended_ns = time.perf_counter_ns()
        self._last_inference_interval_ns = (started_ns, ended_ns)
        self._last_gpu_phase_seconds = recorder.seconds()
        self._last_gpu_timing_error = recorder.error
        elapsed = ended - started
        peak = torch.cuda.max_memory_allocated(self.device)
        if grayscale_output:
            output = output[:, :, 0]
        return output, elapsed, peak, len(positions)

    def upscale(
        self, image: torch.Tensor, model_path: Path, grayscale_output: bool
    ) -> tuple[np.ndarray, InferenceStats]:
        load_started_ns = time.perf_counter_ns()
        descriptor, dtype, load_seconds, cache_hit = self._load(model_path)
        load_ended_ns = time.perf_counter_ns()
        scale = int(descriptor.scale)
        if descriptor.architecture.id == "HAT":
            overlap = self.hat_overlap
            candidates = getattr(self, "hat_tile_candidates", (self.hat_tile,))
            if len(candidates) > 1:
                height, width = (int(value) for value in image.shape[-2:])
                tile, tile_estimates = choose_hat_tile(
                    width, height, candidates, overlap
                )
                tile_strategy = HAT_TILE_STRATEGY
                tile_estimator = HAT_TILE_ESTIMATOR
            else:
                tile = candidates[0]
                tile_estimates = ()
                tile_strategy = "fixed"
                tile_estimator = None
        else:
            tile = self.tile
            overlap = self.overlap
            candidates = (tile,)
            tile_estimates = ()
            tile_strategy = "fixed"
            tile_estimator = None
        if self.device_assembly:
            output, elapsed, peak, tile_count = self._upscale_device_assembly(
                image, descriptor, dtype, scale, tile, overlap, grayscale_output
            )
            assembly = "device"
        else:
            output, elapsed, peak, tile_count = self._upscale_host_assembly(
                image, descriptor, dtype, scale, tile, overlap, grayscale_output
            )
            assembly = "host"
        peak_reserved = torch.cuda.max_memory_reserved(self.device)
        precision = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}[
            dtype
        ]
        raw_phase_seconds = self._last_gpu_phase_seconds or {}
        raw_gpu_total = raw_phase_seconds.get("gpu_total")
        event_scale = (
            elapsed / raw_gpu_total
            if raw_gpu_total is not None and raw_gpu_total > 0
            else None
        )
        phase_seconds = {
            name: seconds * event_scale
            for name, seconds in raw_phase_seconds.items()
            if name != "gpu_total" and event_scale is not None
        }
        timing_warning = None
        if event_scale is not None and abs(event_scale - 1.0) > 0.02:
            timing_warning = (
                "HIP Event clock differed from synchronized perf_counter wall time; "
                f"phase durations were scaled by {event_scale:.6f}."
            )
        return output, InferenceStats(
            seconds=elapsed,
            native_scale=scale,
            peak_vram_bytes=peak,
            tile_count=tile_count,
            precision=precision,
            tile=tile,
            overlap=overlap,
            batch_tiles=self.batch_tiles,
            assembly=assembly,
            model_load_seconds=load_seconds,
            model_cache_hit=cache_hit,
            h2d_seconds=phase_seconds.get("h2d"),
            forward_seconds=phase_seconds.get("forward"),
            gpu_postprocess_seconds=phase_seconds.get("gpu_postprocess"),
            d2h_seconds=phase_seconds.get("d2h"),
            cpu_prepare_seconds=self._last_cpu_prepare_seconds,
            gpu_timing_backend=(
                "torch.cuda.Event/ROCm-HIP+perf_counter_calibration"
                if phase_seconds
                else None
            ),
            gpu_timing_error=self._last_gpu_timing_error,
            gpu_timing_warning=timing_warning,
            gpu_event_total_seconds=raw_gpu_total,
            gpu_event_scale_to_wall=event_scale,
            gpu_event_raw_seconds=(raw_phase_seconds or None),
            model_load_interval_ns=(
                (load_started_ns, load_ended_ns) if not cache_hit else None
            ),
            peak_reserved_vram_bytes=peak_reserved,
            inference_interval_ns=self._last_inference_interval_ns,
            tile_candidates=candidates,
            tile_strategy=tile_strategy,
            tile_estimator=tile_estimator,
            tile_estimates=tile_estimates,
        )
