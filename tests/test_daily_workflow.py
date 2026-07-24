from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil

from PIL import Image
import pytest

from waifuhat2x.config import (
    AppConfig,
    JxlConfig,
    OutputConfig,
    PathsConfig,
    ProcessingConfig,
)
from waifuhat2x.jxl import JxlEncoder as RealJxlEncoder
from waifuhat2x.jxl import JxlStats
from waifuhat2x.engine import InferenceStats
from waifuhat2x.pipeline import PIPELINE_SCHEMA_VERSION, _pipeline_signature, run_pipeline
from waifuhat2x.state import StateStore, file_sha256


class FakeWorkflowEncoder:
    def __init__(self, config: JxlConfig) -> None:
        self.config = config

    def version(self) -> str:
        return "fake-jxl"

    def verify(
        self,
        path: Path,
        expected_width: int | None = None,
        expected_height: int | None = None,
    ) -> tuple[int, int]:
        content = path.read_bytes()
        if content == b"high-jxl":
            dimensions = (160, 240)
        elif content == b"low-jxl":
            dimensions = (80, 120)
        elif content == b"corrupt-jxl":
            raise RuntimeError("djxl verification failed")
        else:
            dimensions = (expected_width or 160, expected_height or 240)
        if expected_width is not None and dimensions != (expected_width, expected_height):
            raise RuntimeError("JXL dimension mismatch")
        return dimensions

    def encode_resized(
        self,
        _array: object,
        _destination: Path,
        output_width: int,
        output_height: int,
        *,
        temporary: Path,
        **_kwargs: object,
    ) -> JxlStats:
        content = f"candidate-{output_width}x{output_height}".encode()
        temporary.write_bytes(content)
        return JxlStats(
            seconds=0.01,
            bytes=len(content),
            temporary=temporary,
            sha256=hashlib.sha256(content).hexdigest(),
        )

    def encode_lossless_jpeg(self, *_args: object, **_kwargs: object) -> JxlStats:
        raise AssertionError("JPEG reconstruction is not expected in this fixture")

    def commit(self, temporary: Path, destination: Path, expected_sha256: str) -> None:
        assert hashlib.sha256(temporary.read_bytes()).hexdigest() == expected_sha256
        os.replace(temporary, destination)


class GpuMustNotStart:
    def __init__(self, **_kwargs: object) -> None:
        raise AssertionError("GPU must not start for this workflow fixture")


def _config(root: Path) -> AppConfig:
    models = root / "models" / "hat"
    models.mkdir(parents=True)
    (models / "HAT-S_SRx2.pth").touch()
    (models / "HAT-S_SRx4.pth").touch()
    return AppConfig(
        source_file=root / "config.toml",
        paths=PathsConfig(root / "library", root / "unused-output", root / "models"),
        processing=ProcessingConfig(
            target_short_edge=160,
            max_long_edge_for_sr=320,
            max_output_long_edge=640,
        ),
        output=OutputConfig(
            mode="replace",
            format="jxl",
            existing_jxl_policy="replace",
            allow_lossy_replace=True,
            allow_metadata_loss=True,
        ),
        jxl=JxlConfig(distance=0.5),
    )


def _real_hat_config(root: Path) -> AppConfig:
    config = _config(root)
    hat_root = config.paths.models / "hat"
    (hat_root / "Real_HAT_GAN_SRx4.pth").write_bytes(b"normal-checkpoint")
    (hat_root / "Real_HAT_GAN_SRx4_sharper.pth").write_bytes(b"sharper-checkpoint")
    return replace(
        config,
        processing=replace(
            config.processing,
            profile="real-hat-auto",
            target_short_edge=1600,
            real_hat_sharper_min_short_edge=1000,
            max_long_edge_for_sr=3200,
            max_upscale_factor=4,
            max_output_long_edge=6400,
        ),
    )


def _schema_6_signature(config: AppConfig) -> str:
    processing = asdict(config.processing)
    processing.pop("real_hat_sharper_min_short_edge")
    payload = {
        "schema": 6,
        "processing": processing,
        "output": asdict(config.output),
        "jxl": asdict(config.jxl),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class FakeWorkflowEngine:
    starts = 0

    def __init__(self, **_kwargs: object) -> None:
        type(self).starts += 1

    @property
    def device_name(self) -> str:
        return "fake-gpu"

    def upscale(self, _tensor: object, _model: Path, grayscale_output: bool) -> tuple[object, InferenceStats]:
        shape = (300, 200) if grayscale_output else (300, 200, 3)
        array = __import__("numpy").zeros(shape, dtype="uint8")
        return array, InferenceStats(
            seconds=0.01,
            native_scale=2,
            peak_vram_bytes=1,
            tile_count=1,
            precision="bf16",
            tile=256,
            overlap=32,
            batch_tiles=1,
            assembly="device",
            model_load_seconds=0.0,
            model_cache_hit=False,
        )

    def close(self) -> None:
        pass


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    engine: type[object] = GpuMustNotStart,
    encoder: type[object] = FakeWorkflowEncoder,
) -> None:
    monkeypatch.setattr("waifuhat2x.pipeline.JxlEncoder", encoder)
    monkeypatch.setattr("waifuhat2x.pipeline.UpscaleEngine", engine)


def test_jxl_only_page_is_skipped_as_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    source_root = config.paths.input
    source_root.mkdir()
    destination = source_root / "001.JXL"
    destination.write_bytes(b"high-jxl")
    original_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
    _patch_runtime(monkeypatch)

    summary = run_pipeline(config)

    assert hashlib.sha256(destination.read_bytes()).hexdigest() == original_hash
    assert summary.processed == 0
    assert summary.jxl_skipped == 1
    assert summary.existing_jxl_replaced == 0
    assert summary.sr_pages == 0
    assert not (source_root / ".waifuhat2x-state.json").exists()
    assert not (source_root / ".waifuhat2x-worklist.jsonl").exists()


def test_low_resolution_source_is_super_resolved_and_replaces_existing_jxl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    fixture_root = tmp_path / "read-only-fixture"
    fixture_root.mkdir()
    fixture_source = fixture_root / "001.png"
    Image.new("RGB", (100, 150), (20, 30, 40)).save(fixture_source)
    fixture_destination = fixture_root / "001.jxl"
    fixture_destination.write_bytes(b"high-jxl")
    source_root = config.paths.input
    shutil.copytree(fixture_root, source_root)
    source = source_root / "001.png"
    destination = source_root / "001.jxl"

    class DecodeGuardEncoder(FakeWorkflowEncoder):
        events: list[str] = []

        def encode_resized(
            self,
            array: object,
            encoded_destination: Path,
            output_width: int,
            output_height: int,
            **kwargs: object,
        ) -> JxlStats:
            stats = super().encode_resized(
                array,
                encoded_destination,
                output_width,
                output_height,
                **kwargs,
            )
            assert stats.temporary is not None
            assert source.exists()
            self.verify(stats.temporary, output_width, output_height)
            type(self).events.append("decoded")
            return stats

        def commit(
            self, temporary: Path, committed_destination: Path, expected_sha256: str
        ) -> None:
            assert type(self).events == ["decoded"]
            assert source.exists()
            super().commit(temporary, committed_destination, expected_sha256)
            type(self).events.append("committed")

    FakeWorkflowEngine.starts = 0
    _patch_runtime(monkeypatch, FakeWorkflowEngine, DecodeGuardEncoder)

    summary = run_pipeline(config)

    assert not source.exists()
    assert destination.read_bytes() == b"candidate-160x240"
    assert fixture_source.exists()
    assert fixture_destination.read_bytes() == b"high-jxl"
    assert DecodeGuardEncoder.events == ["decoded", "committed"]
    assert summary.processed == 1
    assert summary.existing_jxl_replaced == 1
    assert summary.existing_jxl_adopted == 0
    assert summary.sr_pages == 1
    assert FakeWorkflowEngine.starts == 1
    assert not (source_root / ".waifuhat2x-state.json").exists()
    assert not (source_root / ".waifuhat2x-worklist.jsonl").exists()


def test_final_jxl_decode_failure_retains_source_and_output_ready_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    source_root = config.paths.input
    source_root.mkdir()
    source = source_root / "001.png"
    Image.new("RGB", (160, 240), (20, 30, 40)).save(source)
    destination = source.with_suffix(".jxl")

    original_mark_output_ready = StateStore.mark_output_ready

    def corrupt_after_output_ready(
        state: StateStore, key: str, output: Path
    ) -> None:
        original_mark_output_ready(state, key, output)
        output.write_bytes(b"corrupt-jxl")

    monkeypatch.setattr(StateStore, "mark_output_ready", corrupt_after_output_ready)
    _patch_runtime(monkeypatch)

    summary = run_pipeline(config)

    assert source.exists()
    assert destination.read_bytes() == b"corrupt-jxl"
    assert summary.failed == 1
    assert summary.processed == 0
    state = StateStore(source_root / ".waifuhat2x-state.json")
    record = state.record("001.png")
    assert record is not None
    assert record["phase"] == "output_ready"
    assert record["verified"] is True
    assert (source_root / ".waifuhat2x-worklist.jsonl").exists()


@pytest.mark.parametrize("short_edge", [999, 1000, 1001])
@pytest.mark.parametrize("layout", ["portrait", "landscape"])
def test_real_hat_routes_exif_transposed_short_edge_in_full_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    short_edge: int,
    layout: str,
) -> None:
    config = _real_hat_config(tmp_path)
    source_root = config.paths.input
    source_root.mkdir()
    long_edge = short_edge + 101
    displayed_size = (
        (short_edge, long_edge) if layout == "portrait" else (long_edge, short_edge)
    )
    stored_size = (displayed_size[1], displayed_size[0])
    source = source_root / f"{layout}-{short_edge}.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", stored_size, (20, 30, 40)).save(source, exif=exif)

    selected_models: list[Path] = []
    tensor_shapes: list[tuple[int, int]] = []

    class CapturingRealHatEngine(FakeWorkflowEngine):
        def upscale(
            self, tensor: object, model: Path, grayscale_output: bool
        ) -> tuple[object, InferenceStats]:
            selected_models.append(model)
            shape = getattr(tensor, "shape")
            tensor_shapes.append((int(shape[-2]), int(shape[-1])))
            array, stats = super().upscale(tensor, model, grayscale_output)
            return array, replace(stats, native_scale=4)

    _patch_runtime(monkeypatch, CapturingRealHatEngine)

    summary = run_pipeline(config)

    expected_variant = "normal" if short_edge < 1000 else "sharper"
    expected_name = (
        "Real_HAT_GAN_SRx4.pth"
        if expected_variant == "normal"
        else "Real_HAT_GAN_SRx4_sharper.pth"
    )
    output_width = 1600 if layout == "portrait" else round(long_edge * 1600 / short_edge)
    output_height = round(long_edge * 1600 / short_edge) if layout == "portrait" else 1600
    destination = source.with_suffix(".jxl")

    assert selected_models == [config.paths.models / "hat" / expected_name]
    assert tensor_shapes == [(displayed_size[1], displayed_size[0])]
    assert destination.read_bytes() == f"candidate-{output_width}x{output_height}".encode()
    assert min(output_width, output_height) == 1600
    assert not source.exists()
    assert summary.sr_pages == 1
    assert summary.target_unmet == 0


@pytest.mark.parametrize(
    "missing_filename",
    ["Real_HAT_GAN_SRx4.pth", "Real_HAT_GAN_SRx4_sharper.pth"],
)
def test_real_hat_pipeline_fails_before_processing_when_either_weight_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_filename: str,
) -> None:
    config = _real_hat_config(tmp_path)
    source_root = config.paths.input
    source_root.mkdir()
    source = source_root / "001.png"
    Image.new("RGB", (999, 1100), (20, 30, 40)).save(source)
    (config.paths.models / "hat" / missing_filename).unlink()
    _patch_runtime(monkeypatch)

    with pytest.raises(FileNotFoundError, match=missing_filename):
        run_pipeline(config)

    assert source.exists()
    assert not source.with_suffix(".jxl").exists()
    assert not (source_root / ".waifuhat2x-state.json").exists()
    assert not (source_root / ".waifuhat2x-worklist.jsonl").exists()


@pytest.mark.parametrize("phase", ["prepared", "encoded", "output_ready"])
def test_schema_6_replace_transactions_replan_without_early_source_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    config = _config(tmp_path)
    source_root = config.paths.input
    source_root.mkdir()
    source = source_root / "001.png"
    Image.new("RGB", (100, 150), (20, 30, 40)).save(source)
    destination = source_root / "001.jxl"
    destination.write_bytes(b"high-jxl")
    temporary = source_root / ".001.jxl.schema6.part"
    temporary.write_bytes(b"schema-6-candidate")
    state = StateStore(source_root / ".waifuhat2x-state.json")
    old_signature = _schema_6_signature(config)
    current_signature = _pipeline_signature(config)
    assert PIPELINE_SCHEMA_VERSION == 7
    assert old_signature != current_signature
    model = config.paths.models / "hat" / "HAT-S_SRx2.pth"
    fingerprint = state.fingerprint(
        source,
        source_root,
        file_sha256(model),
        old_signature,
    )
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        temporary.name,
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 160,
            "output_height": 240,
            "native_scale": 2,
            "previous_output_sha256": file_sha256(destination),
        },
    )
    if phase in {"encoded", "output_ready"}:
        state.mark_encoded(
            "001.png",
            temporary,
            temporary.stat().st_size,
            file_sha256(temporary),
        )
    if phase == "output_ready":
        os.replace(temporary, destination)
        state.mark_output_ready("001.png", destination)
        # Model a crash that left a sibling transaction artifact even though
        # the durable output had already reached output_ready.
        temporary.write_bytes(b"stale-schema-6-part")
    state.save()

    source_present_during_replan: list[bool] = []
    transaction_events: list[str] = []

    class SourceGuardEngine(FakeWorkflowEngine):
        def upscale(
            self, tensor: object, model_path: Path, grayscale_output: bool
        ) -> tuple[object, InferenceStats]:
            source_present_during_replan.append(source.exists())
            return super().upscale(tensor, model_path, grayscale_output)

    class SourceGuardEncoder(FakeWorkflowEncoder):
        def verify(
            self,
            path: Path,
            expected_width: int | None = None,
            expected_height: int | None = None,
        ) -> tuple[int, int]:
            assert source.exists()
            final = path == destination
            assert (expected_width, expected_height) == (160, 240)
            transaction_events.append("final-djxl" if final else "candidate-djxl")
            return super().verify(path, expected_width, expected_height)

        def encode_resized(
            self,
            array: object,
            encoded_destination: Path,
            output_width: int,
            output_height: int,
            **kwargs: object,
        ) -> JxlStats:
            stats = super().encode_resized(
                array,
                encoded_destination,
                output_width,
                output_height,
                **kwargs,
            )
            assert stats.temporary is not None
            self.verify(stats.temporary, output_width, output_height)
            return stats

        def commit(
            self, candidate: Path, committed_destination: Path, expected_sha256: str
        ) -> None:
            assert source.exists()
            assert transaction_events == ["candidate-djxl"]
            super().commit(candidate, committed_destination, expected_sha256)
            transaction_events.append("committed")

    _patch_runtime(monkeypatch, SourceGuardEngine, SourceGuardEncoder)

    summary = run_pipeline(config)

    assert source_present_during_replan == [True]
    assert transaction_events == ["candidate-djxl", "committed", "final-djxl"]
    assert not source.exists()
    assert destination.read_bytes() == b"candidate-160x240"
    assert not temporary.exists()
    assert not list(source_root.glob("*.part"))
    assert summary.processed == 1
    assert summary.sr_pages == 1
    assert not state.path.exists()


def test_schema_6_prepared_replans_through_real_jxl_without_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    try:
        encoder_probe = RealJxlEncoder(config.jxl)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    assert encoder_probe.version()

    source_root = config.paths.input
    source_root.mkdir()
    source = source_root / "001.png"
    Image.new("RGB", (100, 150), (20, 30, 40)).save(source)
    destination = source.with_suffix(".jxl")
    temporary = source_root / ".001.jxl.schema6-real.part"
    temporary.write_bytes(b"stale-schema-6-candidate")
    state = StateStore(source_root / ".waifuhat2x-state.json")
    model = config.paths.models / "hat" / "HAT-S_SRx2.pth"
    fingerprint = state.fingerprint(
        source,
        source_root,
        file_sha256(model),
        _schema_6_signature(config),
    )
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        temporary.name,
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 160,
            "output_height": 240,
            "native_scale": 2,
            "previous_output_sha256": None,
        },
    )
    state.save()

    transaction_events: list[str] = []

    class RecordingRealEncoder(RealJxlEncoder):
        def verify(
            self,
            path: Path,
            expected_width: int | None = None,
            expected_height: int | None = None,
        ) -> tuple[int, int]:
            assert source.exists()
            final = path == destination
            assert (expected_width, expected_height) == (160, 240)
            transaction_events.append("final-djxl" if final else "candidate-djxl")
            return super().verify(path, expected_width, expected_height)

        def commit(
            self, candidate: Path, committed_destination: Path, expected_sha256: str
        ) -> None:
            assert source.exists()
            assert transaction_events == ["candidate-djxl"]
            super().commit(candidate, committed_destination, expected_sha256)
            transaction_events.append("committed")

    FakeWorkflowEngine.starts = 0
    _patch_runtime(monkeypatch, FakeWorkflowEngine, RecordingRealEncoder)

    summary = run_pipeline(config)

    assert transaction_events == ["candidate-djxl", "committed", "final-djxl"]
    assert not source.exists()
    assert not temporary.exists()
    assert RealJxlEncoder(config.jxl).verify(destination, 160, 240) == (160, 240)
    assert summary.processed == 1
    assert summary.sr_pages == 1
    assert FakeWorkflowEngine.starts == 1
    assert not state.path.exists()


@pytest.mark.parametrize("stale_context", ["threshold_signature", "selected_model_hash"])
def test_real_hat_recovery_discards_old_normal_candidate_and_replans_sharper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale_context: str,
) -> None:
    config = _real_hat_config(tmp_path)
    old_config = replace(
        config,
        processing=replace(
            config.processing,
            real_hat_sharper_min_short_edge=1001,
        ),
    )
    assert _pipeline_signature(old_config) != _pipeline_signature(config)
    source_root = config.paths.input
    source_root.mkdir()
    source = source_root / "001.png"
    Image.new("RGB", (1000, 1101), (20, 30, 40)).save(source)
    destination = source_root / "001.jxl"
    destination.write_bytes(b"high-jxl")
    temporary = source_root / ".001.jxl.old-threshold.part"
    temporary.write_bytes(b"old-normal-candidate")
    normal = config.paths.models / "hat" / "Real_HAT_GAN_SRx4.pth"
    sharper = config.paths.models / "hat" / "Real_HAT_GAN_SRx4_sharper.pth"
    state = StateStore(source_root / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(
        source,
        source_root,
        file_sha256(normal),
        (
            _pipeline_signature(old_config)
            if stale_context == "threshold_signature"
            else _pipeline_signature(config)
        ),
    )
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        temporary.name,
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 1762,
            "native_scale": 4,
            "model_label": "Real-HAT-GAN-x4-normal",
            "previous_output_sha256": file_sha256(destination),
        },
    )
    state.mark_encoded(
        "001.png",
        temporary,
        temporary.stat().st_size,
        file_sha256(temporary),
    )
    state.save()

    selected_models: list[Path] = []
    source_present_during_replan: list[bool] = []

    class CapturingThresholdEngine(FakeWorkflowEngine):
        def upscale(
            self, tensor: object, model: Path, grayscale_output: bool
        ) -> tuple[object, InferenceStats]:
            selected_models.append(model)
            source_present_during_replan.append(source.exists())
            array, stats = super().upscale(tensor, model, grayscale_output)
            return array, replace(stats, native_scale=4)

    _patch_runtime(monkeypatch, CapturingThresholdEngine)

    summary = run_pipeline(config)

    assert selected_models == [sharper]
    assert source_present_during_replan == [True]
    assert not source.exists()
    assert not temporary.exists()
    assert destination.read_bytes() == b"candidate-1600x1762"
    assert summary.processed == 1
    assert summary.sr_pages == 1


def test_recovery_replans_encoded_candidate_after_selected_model_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    source_root = config.paths.input
    source_root.mkdir()
    source = source_root / "001.png"
    Image.new("RGB", (100, 150), (20, 30, 40)).save(source)
    destination = source_root / "001.jxl"
    destination.write_bytes(b"high-jxl")
    temporary = source_root / ".001.jxl.old-model.part"
    temporary.write_bytes(b"candidate-from-old-model")
    state = StateStore(source_root / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(
        source,
        source_root,
        "model-hash-before-weight-replacement",
        _pipeline_signature(config),
    )
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        temporary.name,
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 160,
            "output_height": 240,
            "native_scale": 2,
            "previous_output_sha256": file_sha256(destination),
        },
    )
    state.mark_encoded(
        "001.png", temporary, temporary.stat().st_size, file_sha256(temporary)
    )
    state.save()
    FakeWorkflowEngine.starts = 0
    _patch_runtime(monkeypatch, FakeWorkflowEngine)

    summary = run_pipeline(config)

    assert not source.exists()
    assert not temporary.exists()
    assert destination.read_bytes() == b"candidate-160x240"
    assert summary.processed == 1
    assert summary.sr_pages == 1
    assert FakeWorkflowEngine.starts == 1
    assert not state.path.exists()


def test_recovery_replans_output_ready_transaction_after_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    source_root = config.paths.input
    source_root.mkdir()
    source = source_root / "001.png"
    Image.new("RGB", (100, 150), (20, 30, 40)).save(source)
    destination = source_root / "001.jxl"
    destination.write_bytes(b"candidate-for-old-source")
    state = StateStore(source_root / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(
        source,
        source_root,
        "model-hash-for-old-source",
        _pipeline_signature(config),
    )
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        ".001.jxl.missing-candidate.part",
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 160,
            "output_height": 240,
            "native_scale": 2,
            "previous_output_sha256": None,
        },
    )
    record = state.record("001.png")
    assert record is not None
    record["verified"] = True
    state.mark_output_ready("001.png", destination)
    state.save()
    Image.new("RGB", (100, 150), (50, 60, 70)).save(source)
    FakeWorkflowEngine.starts = 0
    _patch_runtime(monkeypatch, FakeWorkflowEngine)

    summary = run_pipeline(config)

    assert not source.exists()
    assert destination.read_bytes() == b"candidate-160x240"
    assert summary.processed == 1
    assert summary.deferred == 0
    assert summary.sr_pages == 1
    assert FakeWorkflowEngine.starts == 1
    assert not state.path.exists()


def test_meaningful_alpha_blocks_existing_jxl_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    source_root = config.paths.input
    source_root.mkdir()
    source = source_root / "001.png"
    image = Image.new("RGBA", (100, 150), (20, 30, 40, 255))
    image.putpixel((0, 0), (20, 30, 40, 254))
    image.save(source)
    destination = source_root / "001.jxl"
    destination.write_bytes(b"high-jxl")
    _patch_runtime(monkeypatch)

    summary = run_pipeline(config)

    assert source.exists()
    assert destination.read_bytes() == b"high-jxl"
    assert summary.failed == 1
    assert summary.existing_jxl_replaced == 0
    assert (source_root / ".waifuhat2x-worklist.jsonl").exists()


def test_existing_jxl_is_overwritten_without_decoding_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    source_root = config.paths.input
    source_root.mkdir()
    source = source_root / "001.png"
    Image.new("RGB", (160, 240), (20, 30, 40)).save(source)
    destination = source_root / "001.jxl"
    destination.write_bytes(b"corrupt-jxl")
    _patch_runtime(monkeypatch)

    summary = run_pipeline(config)

    assert not source.exists()
    assert destination.read_bytes() == b"candidate-160x240"
    assert summary.failed == 0
    assert summary.processed == 1
    assert summary.transcoded_pages == 1
    assert summary.existing_jxl_replaced == 1
    assert not (source_root / ".waifuhat2x-worklist.jsonl").exists()
