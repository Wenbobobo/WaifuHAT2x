from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import gc
import weakref

import numpy as np
import pytest
import torch
from torch.nn import functional as F

import waifuhat2x.engine as engine_module
from waifuhat2x.engine import (
    HAT_TILE_ESTIMATOR,
    HAT_TILE_STRATEGY,
    UpscaleEngine,
    _GpuPhaseRecorder,
    _HatMaskCache,
    choose_hat_tile,
    estimate_hat_tile_work,
)


class NearestDescriptor:
    scale = 2
    output_channels = 3

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(image.shape[0])
        return F.interpolate(image, scale_factor=self.scale, mode="nearest")


class HAT(torch.nn.Module):
    def __init__(self, *, with_parameter: bool = True) -> None:
        super().__init__()
        if with_parameter:
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.mask_calls: list[tuple[tuple[int, int], torch.dtype]] = []

    def calculate_mask(
        self, x_size: tuple[int, int], dtype: torch.dtype
    ) -> torch.Tensor:
        height, width = x_size
        self.mask_calls.append(((height, width), dtype))
        values = torch.arange(height * width, dtype=torch.float32).reshape(
            1, 1, height, width
        )
        return values.to(dtype=dtype)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        mask = self.calculate_mask((image.shape[2], image.shape[3]), image.dtype)
        return image + mask.to(image.device)


HAT.__module__ = "spandrel.architectures.HAT.__arch.HAT"


class FakeImageModelDescriptor:
    supports_half = False
    supports_bfloat16 = True
    scale = 1
    output_channels = 3

    def __init__(self, model: HAT, architecture_id: str = "HAT") -> None:
        self.model = model
        self.architecture = SimpleNamespace(id=architecture_id)

    def eval(self) -> FakeImageModelDescriptor:
        self.model.eval()
        return self

    def to(self, device: torch.device) -> FakeImageModelDescriptor:
        self.model.to(device)
        return self

    def half(self) -> FakeImageModelDescriptor:
        self.model.half()
        return self

    def bfloat16(self) -> FakeImageModelDescriptor:
        self.model.bfloat16()
        return self

    def float(self) -> FakeImageModelDescriptor:
        self.model.float()
        return self


def test_gpu_phase_recorder_accumulates_event_time_and_fails_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter((0.0, 4.0, 4.0, 11.5))

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing
            self.timestamp = next(timestamps)

        def record(self) -> None:
            pass

        def elapsed_time(self, other: FakeEvent) -> float:
            return other.timestamp - self.timestamp

    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    recorder = _GpuPhaseRecorder(True)
    with recorder.phase("forward"):
        pass
    with recorder.phase("forward"):
        pass

    assert recorder.seconds() == {"forward": pytest.approx(0.0115)}
    assert recorder.error is None

    class BrokenEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing

        def record(self) -> None:
            raise RuntimeError("timing unavailable")

    monkeypatch.setattr(torch.cuda, "Event", BrokenEvent)
    broken = _GpuPhaseRecorder(True)
    with broken.phase("h2d"):
        pass
    assert broken.seconds() is None
    assert broken.error == "RuntimeError: timing unavailable"


def test_upscale_reports_peak_reserved_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = SimpleNamespace(
        scale=4,
        architecture=SimpleNamespace(id="HAT"),
    )
    engine = object.__new__(UpscaleEngine)
    engine.device = torch.device("cpu")
    engine.device_assembly = True
    engine.hat_tile = 256
    engine.hat_overlap = 32
    engine.tile = 256
    engine.overlap = 64
    engine.batch_tiles = 1
    engine._last_gpu_phase_seconds = None
    engine._last_gpu_timing_error = None
    engine._last_cpu_prepare_seconds = 0.0
    engine._last_inference_interval_ns = (100, 200)
    monkeypatch.setattr(
        engine,
        "_load",
        lambda _path: (descriptor, torch.float32, 0.0, True),
    )
    monkeypatch.setattr(
        engine,
        "_upscale_device_assembly",
        lambda *_args: (np.zeros((8, 8, 3), dtype=np.uint8), 0.1, 1234, 1),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 2345)

    _output, stats = engine.upscale(
        torch.zeros((3, 2, 2)), Path("model.pth"), grayscale_output=False
    )

    assert stats.peak_vram_bytes == 1234
    assert stats.peak_reserved_vram_bytes == 2345
    assert stats.inference_interval_ns == (100, 200)
    assert stats.tile == 256
    assert stats.tile_candidates == (256,)
    assert stats.tile_strategy == "fixed"
    assert stats.tile_estimator is None
    assert stats.tile_estimates == ()


def test_hat_tile_work_estimator_uses_padded_tile_area() -> None:
    estimate = estimate_hat_tile_work(
        width=1000, height=1500, tile=256, overlap=32
    )

    assert estimate.tile == 256
    assert estimate.tiles_x == 4
    assert estimate.tiles_y == 6
    assert estimate.tile_count == 24
    assert estimate.expanded_edge == 320
    assert estimate.expanded_tile_area == 102400
    assert estimate.estimated_work == 2457600


def test_hat_tile_selection_minimizes_work_and_breaks_ties_toward_smaller() -> None:
    selected, estimates = choose_hat_tile(
        width=513, height=513, candidates=(256, 320), overlap=32
    )

    assert selected == 320
    assert [estimate.tile for estimate in estimates] == [256, 320]
    assert estimates[1].estimated_work < estimates[0].estimated_work

    tied, _ = choose_hat_tile(
        width=3, height=3, candidates=(4, 2), overlap=0
    )
    assert tied == 2


def test_upscale_selects_hat_tile_from_input_tensor_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = SimpleNamespace(
        scale=4,
        architecture=SimpleNamespace(id="HAT"),
    )
    engine = object.__new__(UpscaleEngine)
    engine.device = torch.device("cpu")
    engine.device_assembly = True
    engine.hat_tile = 256
    engine.hat_tile_candidates = (256, 320)
    engine.hat_overlap = 32
    engine.tile = 192
    engine.overlap = 64
    engine.batch_tiles = 1
    engine._last_gpu_phase_seconds = None
    engine._last_gpu_timing_error = None
    engine._last_cpu_prepare_seconds = 0.0
    engine._last_inference_interval_ns = (100, 200)
    monkeypatch.setattr(
        engine,
        "_load",
        lambda _path: (descriptor, torch.float32, 0.0, True),
    )
    selected: dict[str, int] = {}

    def fake_assembly(*args: object) -> tuple[np.ndarray, float, int, int]:
        selected["tile"] = int(args[4])
        return np.zeros((8, 8, 3), dtype=np.uint8), 0.1, 1234, 4

    monkeypatch.setattr(engine, "_upscale_device_assembly", fake_assembly)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 2345)

    _output, stats = engine.upscale(
        torch.empty((1, 3, 513, 513)),
        Path("model.pth"),
        grayscale_output=False,
    )

    assert selected["tile"] == 320
    assert stats.tile == 320
    assert stats.tile_candidates == (256, 320)
    assert stats.tile_strategy == HAT_TILE_STRATEGY
    assert stats.tile_estimator == HAT_TILE_ESTIMATOR
    assert [estimate.tile for estimate in stats.tile_estimates] == [256, 320]


def test_non_hat_upscale_keeps_fixed_tile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = SimpleNamespace(
        scale=4,
        architecture=SimpleNamespace(id="ESRGAN"),
    )
    engine = object.__new__(UpscaleEngine)
    engine.device = torch.device("cpu")
    engine.device_assembly = True
    engine.hat_tile = 256
    engine.hat_tile_candidates = (256, 320)
    engine.hat_overlap = 32
    engine.tile = 192
    engine.overlap = 64
    engine.batch_tiles = 1
    engine._last_gpu_phase_seconds = None
    engine._last_gpu_timing_error = None
    engine._last_cpu_prepare_seconds = 0.0
    engine._last_inference_interval_ns = (100, 200)
    monkeypatch.setattr(
        engine,
        "_load",
        lambda _path: (descriptor, torch.float32, 0.0, True),
    )
    selected: dict[str, int] = {}

    def fake_assembly(*args: object) -> tuple[np.ndarray, float, int, int]:
        selected["tile"] = int(args[4])
        return np.zeros((8, 8, 3), dtype=np.uint8), 0.1, 1234, 1

    monkeypatch.setattr(engine, "_upscale_device_assembly", fake_assembly)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 2345)

    _output, stats = engine.upscale(
        torch.empty((1, 3, 513, 513)),
        Path("model.pth"),
        grayscale_output=False,
    )

    assert selected["tile"] == 192
    assert stats.tile == 192
    assert stats.tile_candidates == (192,)
    assert stats.tile_strategy == "fixed"
    assert stats.tile_estimator is None
    assert stats.tile_estimates == ()


@pytest.fixture
def cpu_engine(monkeypatch: pytest.MonkeyPatch) -> UpscaleEngine:
    engine = object.__new__(UpscaleEngine)
    engine.device = torch.device("cpu")
    engine.batch_tiles = 3
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: 0)
    return engine


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_crop_before_fp32_clamp_is_exactly_equivalent_to_legacy_order(
    dtype: torch.dtype,
) -> None:
    result = torch.linspace(-0.5, 1.5, steps=3 * 3 * 14 * 14, dtype=torch.float32)
    result = result.reshape(3, 3, 14, 14).to(dtype)
    result[0, 0, 0, 0] = float("nan")
    result[0, 0, 0, 1] = float("inf")
    result[0, 0, 0, 2] = -float("inf")

    legacy = result.clone().float().clamp_(0, 1)[:, :, 4:10, 4:10]
    optimized = UpscaleEngine._crop_core_fp32(
        result.clone(), scale=2, tile=3, overlap=2
    )

    torch.testing.assert_close(optimized, legacy, rtol=0, atol=0, equal_nan=True)
    if dtype == torch.bfloat16:
        assert optimized.untyped_storage().nbytes() == optimized.numel() * optimized.element_size()


def test_hat_mask_cache_keys_lru_and_safe_fallback() -> None:
    model = HAT()
    cache = _HatMaskCache(model)
    setattr(model, "calculate_mask", cache)

    first = model.calculate_mask((4, 4), torch.float32)
    second = model.calculate_mask((4, 4), torch.float32)
    assert second is first
    assert model.mask_calls == [((4, 4), torch.float32)]

    model.calculate_mask((8, 4), torch.float32)
    model.calculate_mask((8, 4), torch.bfloat16)
    assert len(cache._entries) == 2
    model.calculate_mask((4, 4), torch.float32)
    assert model.mask_calls == [
        ((4, 4), torch.float32),
        ((8, 4), torch.float32),
        ((8, 4), torch.bfloat16),
        ((4, 4), torch.float32),
    ]

    cpu_key = cache.cache_key((4, 4), torch.float32, torch.device("cpu"))
    cuda_key = cache.cache_key((4, 4), torch.float32, torch.device("cuda:0"))
    assert cpu_key != cuda_key

    cache.detach()
    assert "calculate_mask" not in vars(model)
    assert model.calculate_mask.__func__ is HAT.calculate_mask
    assert not cache._entries

    no_device_model = HAT(with_parameter=False)
    fallback = _HatMaskCache(no_device_model)
    setattr(no_device_model, "calculate_mask", fallback)
    no_device_model.calculate_mask((2, 2), torch.float32)
    no_device_model.calculate_mask((2, 2), torch.float32)
    assert no_device_model.mask_calls == [
        ((2, 2), torch.float32),
        ((2, 2), torch.float32),
    ]
    assert not fallback._entries
    fallback.detach()


def test_hat_mask_cache_preserves_output_exactly_and_rejects_unsafe_models() -> None:
    engine = object.__new__(UpscaleEngine)
    engine._hat_mask_caches = {}
    model = HAT()
    descriptor = FakeImageModelDescriptor(model)
    image = torch.linspace(0, 1, steps=3 * 4 * 5).reshape(1, 3, 4, 5)
    expected = model(image)
    state_before = {name: value.clone() for name, value in model.state_dict().items()}

    cache = engine._attach_hat_mask_cache(
        Path("safe.pth"), descriptor  # type: ignore[arg-type]
    )
    assert cache is not None
    assert model.state_dict().keys() == state_before.keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[name], rtol=0, atol=0)
    first = model(image)
    second = model(image)
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, expected, rtol=0, atol=0)
    assert model.mask_calls == [
        ((4, 5), torch.float32),
        ((4, 5), torch.float32),
    ]

    unsafe = FakeImageModelDescriptor(HAT(), architecture_id="Other")
    original = unsafe.model.calculate_mask
    assert (
        engine._attach_hat_mask_cache(  # type: ignore[arg-type]
            Path("unsafe.pth"), unsafe
        )
        is None
    )
    assert unsafe.model.calculate_mask == original

    engine._detach_hat_mask_cache(Path("safe.pth"))
    assert "calculate_mask" not in vars(model)


def test_hat_mask_cache_does_not_keep_the_model_alive() -> None:
    engine = object.__new__(UpscaleEngine)
    engine._hat_mask_caches = {}
    model = HAT()
    descriptor = FakeImageModelDescriptor(model)
    model_ref = weakref.ref(model)
    cache = engine._attach_hat_mask_cache(
        Path("weak.pth"), descriptor  # type: ignore[arg-type]
    )
    assert cache is not None
    model.calculate_mask((4, 4), torch.float32)

    del descriptor
    del model
    gc.collect()

    assert model_ref() is None
    assert cache._model_ref() is None
    engine._detach_hat_mask_cache(Path("weak.pth"))


def test_hat_mask_cache_is_detached_on_model_eviction_and_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptors = {
        "first.pth": FakeImageModelDescriptor(HAT()),
        "second.pth": FakeImageModelDescriptor(HAT()),
    }

    class FakeLoader:
        def load_from_file(self, path: Path) -> FakeImageModelDescriptor:
            return descriptors[path.name]

    monkeypatch.setattr(engine_module, "ModelLoader", FakeLoader)
    monkeypatch.setattr(engine_module, "ImageModelDescriptor", FakeImageModelDescriptor)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    engine = object.__new__(UpscaleEngine)
    engine.device = torch.device("cpu")
    engine.precision = "fp32"
    engine.dtype = torch.float32
    engine.model_cache_size = 1
    engine._models = OrderedDict()
    engine._hat_mask_caches = {}

    first_path = Path("first.pth").resolve()
    second_path = Path("second.pth").resolve()
    first, _, _, _ = engine._load(first_path)
    first_cache = engine._hat_mask_caches[first_path]
    first.model.calculate_mask((4, 4), torch.float32)
    assert len(first_cache._entries) == 1

    second, _, _, _ = engine._load(second_path)
    second_cache = engine._hat_mask_caches[second_path]
    assert first_path not in engine._hat_mask_caches
    assert not first_cache._entries
    assert "calculate_mask" not in vars(first.model)
    assert "calculate_mask" in vars(second.model)

    second.model.calculate_mask((4, 4), torch.float32)
    assert len(second_cache._entries) == 1
    engine.close()
    assert not engine._models
    assert not engine._hat_mask_caches
    assert not second_cache._entries
    assert "calculate_mask" not in vars(second.model)


@pytest.mark.parametrize("grayscale_output", [False, True])
def test_host_and_device_assembly_match_with_batches_and_boundary_tiles(
    cpu_engine: UpscaleEngine, grayscale_output: bool
) -> None:
    image = torch.linspace(-0.2, 1.2, steps=1 * 3 * 5 * 7, dtype=torch.float32)
    image = image.reshape(1, 3, 5, 7)
    expected = F.interpolate(image.to(torch.bfloat16), scale_factor=2, mode="nearest")
    expected = expected.float().clamp_(0, 1)
    if grayscale_output:
        expected = (
            expected[:, 0:1] * 0.2126
            + expected[:, 1:2] * 0.7152
            + expected[:, 2:3] * 0.0722
        )
    expected = expected.mul_(255).round_().to(torch.uint8)[0]
    expected_array = expected.permute(1, 2, 0).contiguous().numpy()
    if grayscale_output:
        expected_array = expected_array[:, :, 0]

    outputs: list[np.ndarray] = []
    for assembly in (
        cpu_engine._upscale_host_assembly,
        cpu_engine._upscale_device_assembly,
    ):
        descriptor = NearestDescriptor()
        output, _, peak, tile_count = assembly(
            image=image,
            descriptor=descriptor,  # type: ignore[arg-type]
            dtype=torch.bfloat16,
            scale=2,
            tile=4,
            overlap=1,
            grayscale_output=grayscale_output,
        )
        assert descriptor.batch_sizes == [3, 1]
        assert tile_count == 4
        assert peak == 0
        np.testing.assert_array_equal(output, expected_array)
        outputs.append(output)

    np.testing.assert_array_equal(outputs[0], outputs[1])
