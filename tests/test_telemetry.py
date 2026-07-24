from __future__ import annotations

from dataclasses import asdict, replace
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pytest

from waifuhat2x.cli import build_parser
from waifuhat2x.config import (
    AppConfig,
    JxlConfig,
    OutputConfig,
    PathsConfig,
    ProcessingConfig,
)
from waifuhat2x.jxl import JxlEncoder
from waifuhat2x.engine import InferenceStats, TileWorkEstimate
from waifuhat2x.pipeline import RunSummary, run_pipeline
from waifuhat2x.telemetry import RunTelemetry


def _mirror_config(root: Path) -> AppConfig:
    model_root = root / "models" / "hat"
    model_root.mkdir(parents=True)
    (model_root / "HAT-S_SRx2.pth").touch()
    (model_root / "HAT-S_SRx4.pth").touch()
    return AppConfig(
        source_file=root / "config.toml",
        paths=PathsConfig(root / "input", root / "output", root / "models"),
        processing=ProcessingConfig(
            target_short_edge=160,
            max_long_edge_for_sr=320,
            max_output_long_edge=640,
        ),
        output=OutputConfig(mode="mirror", format="png"),
        jxl=JxlConfig(),
    )


def test_run_pipeline_writes_versioned_relative_metrics(tmp_path: Path) -> None:
    config = _mirror_config(tmp_path)
    config.paths.input.mkdir()
    Image.new("RGB", (160, 240), (20, 30, 40)).save(config.paths.input / "page.png")
    metrics_root = tmp_path / "metrics"

    summary = run_pipeline(config, metrics_dir=metrics_root)

    run_directories = list(metrics_root.iterdir())
    assert len(run_directories) == 1
    run_directory = run_directories[0]
    assert summary.metrics_directory == str(run_directory.resolve())
    assert summary.metrics_write_errors == 0

    pages = [
        json.loads(line)
        for line in (run_directory / "pages.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(pages) == 1
    page = pages[0]
    assert page["schema_version"] == 1
    assert page["source"] == "page.png"
    assert page["destination"] == "page.png"
    assert page["status"] == "complete"
    assert page["details"]["model_label"] == "none"
    assert set(page["timing"]["spans"]) >= {
        "read",
        "decode_exif",
        "analyze_plan",
        "hash_and_fingerprint",
        "preprocess",
        "commit",
    }

    job = json.loads((run_directory / "job.json").read_text(encoding="utf-8"))
    assert job["schema_version"] == 1
    assert job["status"] == "complete"
    assert job["pages_written"] == 1
    assert set(job["timing"]["stage_spans"]) >= {
        "recovery",
        "discovery",
        "preflight",
        "worklist",
    }
    assert "cumulative_service_seconds" in job["timing"]
    assert "interval_summary" in job["timing"]


def test_metrics_are_disabled_by_default(tmp_path: Path) -> None:
    config = _mirror_config(tmp_path)
    config.paths.input.mkdir()
    Image.new("L", (160, 240), 128).save(config.paths.input / "page.png")

    summary = run_pipeline(config)

    assert summary.metrics_directory is None
    assert summary.metrics_write_errors == 0
    assert not (tmp_path / "metrics").exists()


def test_metrics_directory_cannot_overlap_processing_roots(tmp_path: Path) -> None:
    config = _mirror_config(tmp_path)
    config.paths.input.mkdir()
    Image.new("L", (160, 240), 128).save(config.paths.input / "page.png")

    with pytest.raises(ValueError, match="must not overlap"):
        run_pipeline(config, metrics_dir=config.paths.input / "metrics")

    assert not config.paths.output.exists()
    assert not (config.paths.input / "metrics").exists()


def test_interval_report_distinguishes_union_service_and_overlap(tmp_path: Path) -> None:
    started_ns = time.perf_counter_ns()
    telemetry = RunTelemetry.create(tmp_path / "metrics", started_ns=started_ns)
    page = telemetry.page("chapter/001.png", 1, 1, destination="chapter/001.jxl")
    page.add_interval("engine_path", (started_ns + 1_000, started_ns + 3_000))
    page.add_interval("gpu_inference", (started_ns + 1_200, started_ns + 2_800))
    page.add_interval("jxl_service", (started_ns + 2_000, started_ns + 4_000))
    page.set_service_seconds("gpu_synchronized_inference", 0.000002)
    page.set_service_seconds("cjxl", 0.000001)
    page.finish("complete")

    telemetry.finalize(
        status="complete",
        wall_seconds=0.000010,
        summary=asdict(RunSummary(processed=1)),
        context={},
    )

    assert telemetry.run_dir is not None
    job = json.loads((telemetry.run_dir / "job.json").read_text(encoding="utf-8"))
    intervals = job["timing"]["interval_summary"]
    assert intervals["engine_path"]["cumulative_seconds"] == pytest.approx(0.000002)
    assert intervals["engine_path"]["union_seconds"] == pytest.approx(0.000002)
    assert intervals["engine_jxl_relationship"]["overlap_seconds"] == pytest.approx(
        0.000001
    )
    assert intervals["engine_jxl_relationship"]["busy_union_seconds"] == pytest.approx(
        0.000003
    )
    assert intervals["gpu_jxl_relationship"]["overlap_seconds"] == pytest.approx(
        0.0000008
    )
    assert intervals["gpu_jxl_relationship"]["busy_union_seconds"] == pytest.approx(
        0.0000028
    )


def test_page_report_failure_is_nonfatal_and_recorded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    telemetry = RunTelemetry.create(
        tmp_path / "metrics", started_ns=time.perf_counter_ns()
    )
    assert telemetry._pages_handle is not None
    telemetry._pages_handle.close()

    page = telemetry.page("001.png", 1, 1)
    page.finish("complete")
    errors = telemetry.finalize(
        status="complete",
        wall_seconds=0.01,
        summary={},
        context={},
    )

    assert errors == 1
    assert "processing results remain authoritative" in capsys.readouterr().err
    assert telemetry.run_dir is not None
    report = json.loads((telemetry.run_dir / "job.json").read_text(encoding="utf-8"))
    assert report["write_errors"]


def test_finalize_records_close_error_in_job_summary(tmp_path: Path) -> None:
    telemetry = RunTelemetry.create(
        tmp_path / "metrics", started_ns=time.perf_counter_ns()
    )
    assert telemetry._pages_handle is not None

    class CloseErrorHandle:
        def __init__(self, wrapped: object) -> None:
            self.wrapped = wrapped

        def flush(self) -> None:
            self.wrapped.flush()  # type: ignore[attr-defined]

        def close(self) -> None:
            self.wrapped.close()  # type: ignore[attr-defined]
            raise OSError("simulated close failure")

    telemetry._pages_handle = CloseErrorHandle(telemetry._pages_handle)
    errors = telemetry.finalize(
        status="complete",
        wall_seconds=0.01,
        summary={"metrics_write_errors": 0},
        context={},
    )

    assert errors == 1
    assert telemetry.run_dir is not None
    report = json.loads((telemetry.run_dir / "job.json").read_text(encoding="utf-8"))
    assert report["summary"]["metrics_write_errors"] == errors
    assert len(report["write_errors"]) == errors


def test_finalize_cleanup_failure_is_nonfatal(tmp_path: Path) -> None:
    telemetry = RunTelemetry.create(
        tmp_path / "metrics", started_ns=time.perf_counter_ns()
    )
    assert telemetry.run_dir is not None
    (telemetry.run_dir / "job.json.tmp").mkdir()

    errors = telemetry.finalize(
        status="complete",
        wall_seconds=0.01,
        summary={},
        context={},
    )

    assert errors == 2
    assert not (telemetry.run_dir / "job.json").exists()
    assert (telemetry.run_dir / "job.json.tmp").is_dir()


def test_finalize_contains_unexpected_internal_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    telemetry = RunTelemetry.create(
        tmp_path / "metrics", started_ns=time.perf_counter_ns()
    )

    def fail_interval_summary(_wall_seconds: float) -> dict[str, object]:
        raise OSError("simulated report construction failure")

    monkeypatch.setattr(telemetry, "_interval_summary", fail_interval_summary)

    assert (
        telemetry.finalize(
            status="fatal_error",
            wall_seconds=0.01,
            summary=None,
            context={},
            error=RuntimeError("original pipeline failure"),
        )
        == 1
    )


def test_jxl_stats_split_trustworthy_worker_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder = object.__new__(JxlEncoder)
    encoder.cjxl = Path("cjxl")
    encoder.config = JxlConfig(verify_decode=True)
    monkeypatch.setattr(encoder, "_environment", lambda: {})
    monkeypatch.setattr(encoder, "verify", lambda *_args, **_kwargs: (3, 2))

    class FakeProcess:
        def __init__(self, command: list[object], **_kwargs: object) -> None:
            Path(command[2]).write_bytes(b"jxl-candidate")
            self.stdin = BytesIO()
            self.stderr = BytesIO()

        def wait(self) -> int:
            return 0

        def kill(self) -> None:
            pass

    monkeypatch.setattr("waifuhat2x.jxl.subprocess.Popen", FakeProcess)
    destination = tmp_path / "page.jxl"

    stats = encoder.encode(
        np.zeros((2, 3), dtype=np.uint8), destination, finalize=False
    )

    assert stats.temporary is not None and stats.temporary.is_file()
    assert stats.sha256 is not None
    assert stats.service_interval_ns is not None
    assert stats.cjxl_interval_ns is not None
    assert stats.djxl_interval_ns is not None
    assert stats.candidate_hash_interval_ns is not None
    assert stats.commit_interval_ns is None
    assert stats.cjxl_seconds >= 0
    assert stats.djxl_seconds >= 0
    assert stats.candidate_hash_seconds >= 0
    assert stats.commit_seconds == 0


def test_cli_accepts_metrics_directory() -> None:
    args = build_parser().parse_args(["--metrics-dir", "metrics"])

    assert args.metrics_dir == Path("metrics")


def test_sr_metrics_enable_and_report_gpu_phase_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _mirror_config(tmp_path)
    config = replace(
        config,
        processing=replace(
            config.processing,
            hat_tile_candidates=(256, 320),
        ),
    )
    config.paths.input.mkdir()
    Image.new("L", (80, 120), 96).save(config.paths.input / "page.png")
    constructor_kwargs: dict[str, object] = {}

    class TimedEngine:
        def __init__(self, **kwargs: object) -> None:
            constructor_kwargs.update(kwargs)

        @property
        def device_name(self) -> str:
            return "fake-rocm"

        def upscale(
            self, _tensor: object, _model: Path, grayscale_output: bool
        ) -> tuple[np.ndarray, InferenceStats]:
            assert grayscale_output
            load_start = time.perf_counter_ns()
            load_end = time.perf_counter_ns()
            return np.zeros((240, 160), dtype=np.uint8), InferenceStats(
                seconds=0.25,
                native_scale=2,
                peak_vram_bytes=1234,
                peak_reserved_vram_bytes=2345,
                tile_count=1,
                precision="bf16",
                tile=256,
                overlap=32,
                batch_tiles=1,
                assembly="device",
                model_load_seconds=0.05,
                model_cache_hit=False,
                h2d_seconds=0.01,
                forward_seconds=0.20,
                gpu_postprocess_seconds=0.03,
                d2h_seconds=0.01,
                cpu_prepare_seconds=0.002,
                gpu_timing_backend=(
                    "torch.cuda.Event/ROCm-HIP+perf_counter_calibration"
                ),
                gpu_event_total_seconds=0.25,
                gpu_event_scale_to_wall=1.0,
                gpu_event_raw_seconds={
                    "gpu_total": 0.25,
                    "h2d": 0.01,
                    "forward": 0.20,
                    "gpu_postprocess": 0.03,
                    "d2h": 0.01,
                },
                model_load_interval_ns=(load_start, load_end),
                inference_interval_ns=(load_end, time.perf_counter_ns()),
                tile_candidates=(256, 320),
                tile_strategy="min-padded-work-v1",
                tile_estimator=(
                    "ceil(width/tile)*ceil(height/tile)*(tile+2*overlap)^2"
                ),
                tile_estimates=(
                    TileWorkEstimate(
                        tile=256,
                        tiles_x=1,
                        tiles_y=1,
                        tile_count=1,
                        expanded_edge=320,
                        expanded_tile_area=102400,
                        estimated_work=102400,
                    ),
                    TileWorkEstimate(
                        tile=320,
                        tiles_x=1,
                        tiles_y=1,
                        tile_count=1,
                        expanded_edge=384,
                        expanded_tile_area=147456,
                        estimated_work=147456,
                    ),
                ),
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr("waifuhat2x.pipeline.UpscaleEngine", TimedEngine)
    summary = run_pipeline(config, metrics_dir=tmp_path / "metrics")

    assert constructor_kwargs["collect_gpu_timing"] is True
    assert constructor_kwargs["hat_tile_candidates"] == (256, 320)
    assert summary.metrics_directory is not None
    pages_path = Path(summary.metrics_directory) / "pages.jsonl"
    page = json.loads(pages_path.read_text(encoding="utf-8").splitlines()[0])
    services = page["timing"]["cumulative_service_seconds"]
    assert services["h2d"] == pytest.approx(0.01)
    assert services["forward"] == pytest.approx(0.20)
    assert services["gpu_postprocess"] == pytest.approx(0.03)
    assert services["d2h"] == pytest.approx(0.01)
    assert services["engine_cpu_prepare"] == pytest.approx(0.002)
    assert page["details"]["engine_component_timing"]["unavailable"] == []
    assert page["details"]["gpu_timing_backend"] == (
        "torch.cuda.Event/ROCm-HIP+perf_counter_calibration"
    )
    assert page["details"]["peak_vram_bytes"] == 1234
    assert page["details"]["peak_reserved_vram_bytes"] == 2345
    assert page["details"]["tile"] == 256
    assert page["details"]["tile_candidates"] == [256, 320]
    assert page["details"]["tile_strategy"] == "min-padded-work-v1"
    assert page["details"]["tile_estimator"] == (
        "ceil(width/tile)*ceil(height/tile)*(tile+2*overlap)^2"
    )
    assert page["details"]["tile_estimates"] == [
        {
            "tile": 256,
            "tiles_x": 1,
            "tiles_y": 1,
            "tile_count": 1,
            "expanded_edge": 320,
            "expanded_tile_area": 102400,
            "estimated_work": 102400,
        },
        {
            "tile": 320,
            "tiles_x": 1,
            "tiles_y": 1,
            "tile_count": 1,
            "expanded_edge": 384,
            "expanded_tile_area": 147456,
            "estimated_work": 147456,
        },
    ]
    assert "model_load" in page["timing"]["spans"]
    assert "gpu_inference" in page["timing"]["spans"]
