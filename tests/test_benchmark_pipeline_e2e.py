from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from PIL import Image
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_pipeline_e2e as benchmark  # noqa: E402


def _write_base_config(root: Path) -> Path:
    model_root = root / "models" / "hat"
    model_root.mkdir(parents=True)
    for name in benchmark.REAL_HAT_MODELS:
        (model_root / name).write_bytes(name.encode())
    config = root / "config.toml"
    config.write_text(
        """
[paths]
input = "unused-input"
output = "unused-output"
models = "models"

[processing]
profile = "real-hat-auto"
target_short_edge = 1600
real_hat_sharper_min_short_edge = 1000
max_long_edge_for_sr = 3200
max_upscale_factor = 4
max_output_long_edge = 6400
max_output_megapixels = 24.0
precision = "bf16"
tile = 256
overlap = 64
hat_tile = 256
hat_tile_candidates = [256]
hat_overlap = 32
batch_tiles = 1
device_assembly = true
model_cache_size = 2
grayscale_tolerance = 3
linear_light_downscale = true

[output]
mode = "replace"
format = "jxl"
copy_non_images = true
overwrite = false
existing_jxl_policy = "replace"
allow_lossy_replace = true
allow_metadata_loss = true
allow_alpha_flatten = false
allow_bit_depth_loss = false

[jxl]
distance = 0.5
effort = 7
threads = 4
workers = 1
queue_depth = 2
verify_decode = true
""".lstrip(),
        encoding="utf-8",
    )
    return config


def _write_manifest(root: Path) -> tuple[Path, Path, dict[str, dict[str, Any]]]:
    inputs = root / "representative" / "inputs"
    inputs.mkdir(parents=True)
    files = {
        "01_normal.png": ("L", (999, 1401), 128),
        "02_sharper.png": ("RGB", (1000, 1400), (10, 30, 80)),
    }
    pages = []
    pixel_counts = []
    for index, (name, (mode, size, fill)) in enumerate(files.items(), start=1):
        path = inputs / name
        Image.new(mode, size, fill).save(path)
        width, height = size
        short_edge = min(size)
        route = "normal" if index == 1 else "sharper"
        gray = mode == "L"
        odd = bool(width % 2 or height % 2)
        pixel_counts.append(width * height)
        pages.append(
            {
                "index": index,
                "copied_path": f"inputs/{name}",
                "copied_sha256": benchmark.sha256_file(path),
                "width": width,
                "height": height,
                "short_edge": short_edge,
                "long_edge": max(size),
                "pixels": width * height,
                "route": route,
                "grayscale": gray,
                "odd_dimension": odd,
            }
        )
    manifest = inputs.parent / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "real_hat_representative_manifest",
                "selected_counts": {"normal": 1, "sharper": 1},
                "coverage": {
                    "exact_threshold": 1,
                    "grayscale": 1,
                    "rgb_or_color": 1,
                    "odd_dimension": 1,
                    "minimum_selected_pixels": min(pixel_counts),
                    "maximum_selected_pixels": max(pixel_counts),
                },
                "pages": pages,
            }
        ),
        encoding="utf-8",
    )
    return manifest, inputs, benchmark.input_snapshot(inputs)


def _telemetry(
    *,
    reserved: float | None = 12 * 1024**3,
    route_counts: dict[str, int] | None = None,
    tile_candidates: tuple[int, ...] = (256,),
    overlap: int = 32,
) -> dict[str, Any]:
    page_count = sum((route_counts or {"normal": 1}).values())
    strategy = "fixed" if len(tile_candidates) == 1 else "min-padded-work-v1"
    selected_tile_counts = {}
    for index, tile in enumerate(tile_candidates):
        count = page_count // len(tile_candidates) + (
            index < page_count % len(tile_candidates)
        )
        if count:
            selected_tile_counts[str(tile)] = count
    return {
        "route_counts": route_counts or {"normal": 1},
        "page_status_counts": {"complete": page_count},
        "peak_allocated_vram_bytes": 2 * 1024**3,
        "peak_reserved_vram_bytes": reserved,
        "reserved_vram_source": (
            "pages.details.peak_reserved_vram_bytes"
            if reserved is not None
            else "unavailable"
        ),
        "tile_execution": {
            "selected_tile_counts": selected_tile_counts,
            "candidate_sets": [
                {"candidates": list(tile_candidates), "pages": page_count}
            ],
            "strategy_counts": {strategy: page_count},
            "estimator_counts": {
                (
                    "ceil(width/tile)*ceil(height/tile)*(tile+2*overlap)^2"
                    if len(tile_candidates) > 1
                    else "none"
                ): page_count
            },
            "overlap_counts": {str(overlap): page_count},
        },
        "phases": {
            "engine_path": {"cumulative_seconds": 8.0, "union_seconds": 8.0},
            "jxl_service": {"cumulative_seconds": 1.0, "union_seconds": 1.0},
        },
        "services": {"forward": 7.0, "cjxl": 0.8},
        "engine_jxl_relationship": {"overlap_seconds": 0.5},
        "gpu_jxl_relationship": {"overlap_seconds": 0.4},
    }


def _run(
    wall: float,
    *,
    digest: str = "a" * 64,
    reserved: float | None = 12 * 1024**3,
    run_id: str = "run",
    route_counts: dict[str, int] | None = None,
    page_count: int = 1,
    tile_candidates: tuple[int, ...] = (256,),
    overlap: int = 32,
    role: str = "repeat",
    index: int = 1,
    parent_session_id: str = "protocol-session",
) -> dict[str, Any]:
    attestation = {
        name: {"benchmark_path": f"{run_id}/{name}.json", "sha256": "e" * 64}
        for name in (
            "child_spec",
            "child_config",
            "attempt_status",
            "completion_marker",
        )
    }
    return {
        "pipeline_summary": {
            "wall_seconds": wall,
            "failed": 0,
            "deferred": 0,
            "target_unmet": 0,
            "metrics_write_errors": 0,
        },
        "jxl_outputs": [
            {
                "path": f"page-{index:02d}.jxl",
                "bytes": 10,
                "sha256": digest,
            }
            for index in range(1, page_count + 1)
        ],
        "telemetry": _telemetry(
            reserved=reserved,
            route_counts=route_counts,
            tile_candidates=tile_candidates,
            overlap=overlap,
        ),
        "attempt": f"runs/{run_id}/attempt-001",
        "role": role,
        "index": index,
        "parent_session_id": parent_session_id,
        "pair_id": f"{role}-{index}",
        "attestation": attestation,
        "input_snapshot_unchanged": True,
        "isolation": {
            "fresh_roots_required_at_child_start": True,
            "input_output_disjoint": True,
            "input_metrics_disjoint": True,
            "input_cache_disjoint": True,
            "output_metrics_disjoint": True,
            "output_cache_disjoint": True,
            "metrics_cache_disjoint": True,
        },
        "owned_roots": {
            "output": f"/synthetic-benchmark/{run_id}/output",
            "metrics": f"/synthetic-benchmark/{run_id}/metrics",
            "cache": f"/synthetic-benchmark/{run_id}/cache",
        },
    }


def test_parser_defaults_define_fixed_256_vs_adaptive_256_320_protocol() -> None:
    parser = benchmark.build_parser()
    args = parser.parse_args(
        ["--manifest", "manifest.json", "--output-root", "benchmark"]
    )

    assert args.tile == [256]
    assert args.adaptive_tiles == [256, 320]
    assert args.overlap == [32]
    assert args.warmups == 1
    assert args.repeats == 3
    assert args.resume is True
    assert args.fixed_only is False
    assert args.adaptive_only is False
    assert args.baseline_overlap is None
    assert benchmark.validate_arguments(parser, args) == [
        benchmark.BenchmarkConfiguration((256,), 32),
        benchmark.BenchmarkConfiguration((256, 320), 32),
    ]

    fixed_only = parser.parse_args(
        [
            "--manifest",
            "manifest.json",
            "--output-root",
            "benchmark",
            "--fixed-only",
            "--tile",
            "256",
            "320",
        ]
    )
    assert benchmark.validate_arguments(parser, fixed_only) == [
        benchmark.BenchmarkConfiguration((256,), 32),
        benchmark.BenchmarkConfiguration((320,), 32),
    ]

    adaptive_only = parser.parse_args(
        [
            "--manifest",
            "manifest.json",
            "--output-root",
            "benchmark",
            "--adaptive-only",
            "--baseline-overlap",
            "32",
            "--adaptive-tiles",
            "256",
            "320",
            "--overlap",
            "32",
            "16",
        ]
    )
    assert benchmark.validate_arguments(parser, adaptive_only) == [
        benchmark.BenchmarkConfiguration((256, 320), 32),
        benchmark.BenchmarkConfiguration((256, 320), 16),
    ]


def test_parser_rejects_fixed_only_with_adaptive_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = benchmark.build_parser()

    with pytest.raises(SystemExit) as raised:
        parser.parse_args(
            [
                "--manifest",
                "manifest.json",
                "--output-root",
                "benchmark",
                "--fixed-only",
                "--adaptive-only",
            ]
        )

    assert raised.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--tile", "257"), "divisible by 16"),
        (("--overlap", "7"), "divisible by 8"),
        (("--tile", "32", "--overlap", "32"), "smaller"),
        (("--tile", "256", "256"), "duplicate fixed"),
        (("--adaptive-tiles", "256", "256"), "duplicate --adaptive"),
        (("--adaptive-tiles", "256"), "at least two"),
        (("--baseline-overlap", "32"), "requires --adaptive-only"),
        (
            (
                "--adaptive-only",
                "--baseline-overlap",
                "24",
                "--overlap",
                "32",
                "16",
            ),
            "must be present",
        ),
        (
            ("--adaptive-only", "--overlap", "32", "32"),
            "duplicate --overlap",
        ),
    ],
)
def test_argument_validation_rejects_non_comparable_matrix(
    extra: tuple[str, ...], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = benchmark.build_parser()
    args = parser.parse_args(
        ["--manifest", "manifest.json", "--output-root", "benchmark", *extra]
    )

    with pytest.raises(SystemExit) as raised:
        benchmark.validate_arguments(parser, args)

    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_manifest_requires_exact_hashed_input_set_and_threshold_routes(
    tmp_path: Path,
) -> None:
    manifest, inputs, snapshot = _write_manifest(tmp_path)

    loaded, actual, routes, coverage = benchmark.load_representative_manifest(
        manifest, inputs, 1000
    )

    assert loaded["selected_counts"] == {"normal": 1, "sharper": 1}
    assert actual == snapshot
    assert routes == {"normal": 1, "sharper": 1}
    assert coverage["exact_threshold"] == 1
    assert coverage["grayscale"] == 1
    assert coverage["rgb_or_color"] == 1

    (inputs / "unexpected.txt").write_text("not in manifest", encoding="utf-8")
    with pytest.raises(ValueError, match="file set differs"):
        benchmark.load_representative_manifest(manifest, inputs, 1000)


def test_manifest_rejects_route_drift(tmp_path: Path) -> None:
    manifest, inputs, _snapshot = _write_manifest(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["pages"][0]["route"] = "sharper"
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="fact drift"):
        benchmark.load_representative_manifest(manifest, inputs, 1000)


def test_manifest_rejects_claimed_dimensions_without_redecoding_trust(
    tmp_path: Path,
) -> None:
    manifest, inputs, _snapshot = _write_manifest(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["pages"][0]["width"] = 998
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="fact drift.*width"):
        benchmark.load_representative_manifest(manifest, inputs, 1000)


def test_isolation_rejects_input_overlap_and_preexisting_child_output(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "page.png").write_bytes(b"page")

    with pytest.raises(ValueError, match="must not overlap"):
        benchmark.validate_isolated_roots(
            inputs,
            inputs / "output",
            tmp_path / "metrics",
            tmp_path / "cache",
            require_fresh=True,
        )

    output = tmp_path / "output"
    output.mkdir()
    (output / "unmanaged.bin").write_bytes(b"occupied")
    with pytest.raises(ValueError, match="already exists and is non-empty"):
        benchmark.validate_isolated_roots(
            inputs,
            output,
            tmp_path / "metrics",
            tmp_path / "cache",
            require_fresh=True,
        )


@pytest.mark.parametrize(
    "configuration",
    [
        benchmark.BenchmarkConfiguration((256,), 32),
        benchmark.BenchmarkConfiguration((256, 320), 24),
    ],
)
def test_rendered_config_forces_mirror_jxl_and_exact_tile_candidates(
    tmp_path: Path, configuration: benchmark.BenchmarkConfiguration
) -> None:
    base = benchmark.load_config(_write_base_config(tmp_path))
    inputs = tmp_path / "representative"
    output = tmp_path / "isolated-output"
    rendered = benchmark.render_child_config(
        base,
        input_root=inputs,
        output_root=output,
        configuration=configuration,
    )
    child_config = tmp_path / "child.toml"
    child_config.write_text(rendered, encoding="utf-8")

    loaded = benchmark.load_config(child_config)

    assert loaded.paths.input == inputs.resolve()
    assert loaded.paths.output == output.resolve()
    assert loaded.output.mode == "mirror"
    assert loaded.output.format == "jxl"
    assert loaded.output.copy_non_images is False
    assert loaded.output.overwrite is False
    assert loaded.output.existing_jxl_policy == "error"
    assert loaded.processing.hat_tile == configuration.primary_tile
    assert loaded.processing.hat_tile_candidates == configuration.tile_candidates
    assert loaded.processing.hat_overlap == configuration.overlap
    assert loaded.processing.model_cache_size == 2


def test_configuration_record_and_fingerprint_distinguish_fixed_from_adaptive() -> None:
    fixed = benchmark.BenchmarkConfiguration((256,), 32)
    adaptive = benchmark.BenchmarkConfiguration((256, 320), 32)

    assert fixed.record() == {
        "strategy": "fixed",
        "hat_tile": 256,
        "hat_tile_candidates": [256],
        "hat_overlap": 32,
        "selection_formula": None,
    }
    assert adaptive.record()["strategy"] == "min-padded-work-v1"
    assert "ceil(width/tile)" in adaptive.record()["selection_formula"]
    assert benchmark._run_fingerprint("parent", fixed, "repeat", 1) != (
        benchmark._run_fingerprint("parent", adaptive, "repeat", 1)
    )

    schedule = benchmark.build_execution_schedule(
        [fixed, adaptive], warmups=1, repeats=3
    )
    assert [(role, index, item.strategy) for role, index, item in schedule] == [
        ("warmup", 1, "fixed"),
        ("warmup", 1, "min-padded-work-v1"),
        ("repeat", 1, "fixed"),
        ("repeat", 1, "min-padded-work-v1"),
        ("repeat", 2, "min-padded-work-v1"),
        ("repeat", 2, "fixed"),
        ("repeat", 3, "fixed"),
        ("repeat", 3, "min-padded-work-v1"),
    ]

    candidates = {
        (role, index, configuration): (
            Path(f"attempt-{position}"),
            {"parent_session_id": "session-a"},
        )
        for position, (role, index, configuration) in enumerate(schedule, start=1)
    }
    complete, sessions = benchmark.coherent_reusable_plan(candidates, schedule)
    assert complete == candidates
    assert sessions == {"session-a"}

    partial = dict(list(candidates.items())[:-1])
    assert benchmark.coherent_reusable_plan(partial, schedule)[0] == {}
    candidates[schedule[-1]] = (
        Path("attempt-mixed"),
        {"parent_session_id": "session-b"},
    )
    assert benchmark.coherent_reusable_plan(candidates, schedule)[0] == {}


def test_child_config_validation_rejects_candidate_strategy_drift(
    tmp_path: Path,
) -> None:
    base = benchmark.load_config(_write_base_config(tmp_path))
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    inputs = tmp_path / "inputs"
    output = attempt / "output"
    configuration = benchmark.BenchmarkConfiguration((256, 320), 32)
    config_path = attempt / "config.toml"
    config_path.write_text(
        benchmark.render_child_config(
            base,
            input_root=inputs,
            output_root=output,
            configuration=configuration,
        ),
        encoding="utf-8",
    )
    spec = {
        "attempt_root": str(attempt),
        "config_path": str(config_path),
        "config_sha256": benchmark.sha256_file(config_path),
        "input_root": str(inputs),
        "output_root": str(output),
        "configuration": configuration.record(),
    }

    loaded = benchmark._validate_child_config(spec)
    assert loaded.processing.hat_tile_candidates == (256, 320)

    spec["configuration"] = {
        **configuration.record(),
        "strategy": "fixed",
    }
    with pytest.raises(ValueError, match="not canonical"):
        benchmark._validate_child_config(spec)


def test_production_semantics_reject_route_target_or_jxl_drift(tmp_path: Path) -> None:
    config_path = _write_base_config(tmp_path)
    benchmark.validate_production_semantics(benchmark.load_config(config_path))

    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        text.replace("target_short_edge = 1600", "target_short_edge = 1500").replace(
            "distance = 0.5", "distance = 0.6"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target_short_edge.*jxl.distance"):
        benchmark.validate_production_semantics(benchmark.load_config(config_path))


def test_model_inventory_rejects_nonofficial_checkpoint_hash(tmp_path: Path) -> None:
    config = benchmark.load_config(_write_base_config(tmp_path))

    with pytest.raises(ValueError, match="Official Real-HAT checkpoint hash mismatch"):
        benchmark.resolve_real_hat_models(config)

    assert benchmark.OFFICIAL_REAL_HAT_SHA256 == {
        "Real_HAT_GAN_SRx4.pth": (
            "f5b1e3bbbb05147ca2beefcc715279cb647d7976cbda67d62ea7e6e20d5ffcc7"
        ),
        "Real_HAT_GAN_SRx4_sharper.pth": (
            "5800b67136006eb8cab3b4ed7c8d73b6a195bb18e6cc709b674f9aa069c00271"
        ),
    }


def test_telemetry_summary_distinguishes_allocated_and_reserved_vram(
    tmp_path: Path,
) -> None:
    job = {
        "run_id": "run",
        "pages_written": 1,
        "timing": {
            "stage_spans": {"discovery": [{"duration_seconds": 0.2}]},
            "cumulative_service_seconds": {"forward": 1.5},
            "interval_summary": {
                "engine_path": {
                    "cumulative_seconds": 2.0,
                    "union_seconds": 2.0,
                },
                "engine_jxl_relationship": {"overlap_seconds": 0.1},
                "gpu_jxl_relationship": {"overlap_seconds": 0.08},
            },
        },
    }
    page = {
        "type": "waifuhat2x-page-metrics",
        "run_id": "run",
        "status": "complete",
        "details": {
            "model_label": "Real-HAT-GAN-x4-sharper",
            "peak_vram_bytes": 100,
            "peak_reserved_vram_bytes": 200,
            "tile": 320,
            "tile_candidates": [256, 320],
            "tile_strategy": "min-padded-work-v1",
            "tile_estimator": ("ceil(width/tile)*ceil(height/tile)*(tile+2*overlap)^2"),
            "overlap": 32,
        },
    }

    report = benchmark.summarize_telemetry(job, [page])

    assert report["route_counts"] == {"sharper": 1}
    assert report["peak_allocated_vram_bytes"] == 100
    assert report["peak_reserved_vram_bytes"] == 200
    assert report["reserved_vram_source"] == "pages.details.peak_reserved_vram_bytes"
    assert report["tile_execution"] == {
        "selected_tile_counts": {"320": 1},
        "candidate_sets": [{"candidates": [256, 320], "pages": 1}],
        "strategy_counts": {"min-padded-work-v1": 1},
        "estimator_counts": {
            "ceil(width/tile)*ceil(height/tile)*(tile+2*overlap)^2": 1
        },
        "overlap_counts": {"32": 1},
    }
    assert report["phases"]["stage:discovery"]["union_seconds"] == pytest.approx(0.2)
    assert report["gpu_jxl_relationship"]["overlap_seconds"] == pytest.approx(0.08)

    del page["details"]["peak_reserved_vram_bytes"]
    unavailable = benchmark.summarize_telemetry(job, [page])
    assert unavailable["peak_reserved_vram_bytes"] is None
    assert unavailable["reserved_vram_source"] == "unavailable"


def test_configuration_aggregation_checks_cv_hash_routes_vram_and_phases() -> None:
    repeats = [_run(10.0), _run(10.1), _run(9.9)]

    report = benchmark.aggregate_configuration(
        configuration=benchmark.BenchmarkConfiguration((256,), 32),
        warmups=[_run(11.0)],
        repeats=repeats,
        expected_routes={"normal": 1},
        expected_pages=1,
        max_cv_percent=3.0,
        max_reserved_vram_bytes=14 * 1024**3,
    )

    assert report["batch_wall_seconds"]["mean"] == pytest.approx(10.0)
    assert report["batch_wall_seconds"]["cv_percent"] < 3.0
    assert report["jxl_byte_deterministic"] is True
    assert report["configuration"]["hat_tile_candidates"] == [256]
    assert report["peak_reserved_vram_bytes"] == 12 * 1024**3
    assert report["phase_breakdown"]["engine_path"][
        "union_share_of_batch_wall_percent"
    ] == pytest.approx(80.0)
    assert report["qualification"]["valid_for_performance_decision"] is True


def test_configuration_aggregation_never_substitutes_allocated_for_reserved() -> None:
    repeats = [_run(10.0, reserved=None), _run(10.0, reserved=None)]

    report = benchmark.aggregate_configuration(
        configuration=benchmark.BenchmarkConfiguration((256, 320), 32),
        warmups=[],
        repeats=repeats,
        expected_routes={"normal": 1},
        expected_pages=1,
        max_cv_percent=3.0,
        max_reserved_vram_bytes=14 * 1024**3,
    )

    assert report["peak_allocated_vram_bytes"] == 2 * 1024**3
    assert report["peak_reserved_vram_bytes"] is None
    assert report["reserved_vram_status"] == "unavailable"
    checks = report["qualification"]["checks"]
    assert checks["reserved_vram_available"] is False
    assert report["qualification"]["valid_for_performance_decision"] is False


def test_configuration_aggregation_rejects_failed_or_over_vram_warmup() -> None:
    warmup = _run(
        11.0,
        reserved=15 * 1024**3,
        role="warmup",
        run_id="warmup",
    )
    warmup["pipeline_summary"]["failed"] = 1

    report = benchmark.aggregate_configuration(
        configuration=benchmark.BenchmarkConfiguration((256,), 32),
        warmups=[warmup],
        repeats=[
            _run(10.0, run_id="repeat-1"),
            _run(10.0, run_id="repeat-2", index=2),
            _run(10.0, run_id="repeat-3", index=3),
        ],
        expected_routes={"normal": 1},
        expected_pages=1,
        max_cv_percent=3.0,
        max_reserved_vram_bytes=14 * 1024**3,
    )

    checks = report["qualification"]["checks"]
    assert checks["no_failed_or_deferred_pages"] is False
    assert checks["reserved_vram_within_limit"] is False
    assert report["peak_reserved_vram_bytes"] == 15 * 1024**3


def test_configuration_aggregation_detects_jxl_byte_drift() -> None:
    report = benchmark.aggregate_configuration(
        configuration=benchmark.BenchmarkConfiguration((256, 320), 32),
        warmups=[],
        repeats=[_run(10.0), _run(10.0, digest="b" * 64)],
        expected_routes={"normal": 1},
        expected_pages=1,
        max_cv_percent=3.0,
        max_reserved_vram_bytes=14 * 1024**3,
    )

    assert report["jxl_byte_deterministic"] is False
    assert report["qualification"]["checks"]["jxl_byte_deterministic"] is False


def _protocol_report(
    configuration: benchmark.BenchmarkConfiguration,
    walls: list[float],
    prefix: str,
    *,
    warmup_count: int = 1,
) -> dict[str, Any]:
    routes = {"normal": 9, "sharper": 21}
    return benchmark.aggregate_configuration(
        configuration=configuration,
        warmups=[
            _run(
                walls[0] + 10,
                run_id=f"{prefix}-warmup-{index}",
                route_counts=routes,
                page_count=30,
                tile_candidates=configuration.tile_candidates,
                overlap=configuration.overlap,
                role="warmup",
                index=index,
            )
            for index in range(1, warmup_count + 1)
        ],
        repeats=[
            _run(
                wall,
                run_id=f"{prefix}-repeat-{index}",
                route_counts=routes,
                page_count=30,
                tile_candidates=configuration.tile_candidates,
                overlap=configuration.overlap,
                role="repeat",
                index=index,
            )
            for index, wall in enumerate(walls, start=1)
        ],
        expected_routes=routes,
        expected_pages=30,
        max_cv_percent=3.0,
        max_reserved_vram_bytes=14 * 1024**3,
    )


def _production_coverage() -> dict[str, int]:
    return {
        "page_count": 30,
        "exact_threshold": 2,
        "grayscale": 17,
        "rgb_or_color": 13,
        "odd_dimension": 13,
        "minimum_selected_pixels": 195_072,
        "maximum_selected_pixels": 3_840_000,
    }


@pytest.mark.parametrize(
    ("configurations", "baseline_overlap", "message"),
    [
        (
            [
                benchmark.BenchmarkConfiguration((256, 320), 16),
                benchmark.BenchmarkConfiguration((256, 320), 24),
            ],
            32,
            "exactly one adaptive baseline",
        ),
        (
            [
                benchmark.BenchmarkConfiguration((256, 320), 32),
                benchmark.BenchmarkConfiguration((256, 320), 32),
                benchmark.BenchmarkConfiguration((256, 320), 16),
            ],
            32,
            "Duplicate adaptive overlap",
        ),
        (
            [
                benchmark.BenchmarkConfiguration((256, 320), 32),
                benchmark.BenchmarkConfiguration((256,), 16),
            ],
            32,
            "identical adaptive tile set",
        ),
        (
            [benchmark.BenchmarkConfiguration((256, 320), 32)],
            32,
            "at least one candidate overlap",
        ),
    ],
)
def test_adaptive_overlap_matrix_rejects_missing_duplicate_or_noncandidate(
    configurations: list[benchmark.BenchmarkConfiguration],
    baseline_overlap: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.validate_adaptive_overlap_matrix(
            configurations, baseline_overlap
        )


def test_adaptive_overlap_gate_uses_explicit_baseline_independent_of_order() -> None:
    baseline_configuration = benchmark.BenchmarkConfiguration((256, 320), 32)
    candidate_configuration = benchmark.BenchmarkConfiguration((256, 320), 16)
    baseline = _protocol_report(
        baseline_configuration, [100.0, 100.0, 100.0], "adaptive-o32"
    )
    candidate = _protocol_report(
        candidate_configuration, [90.0, 90.0, 90.0], "adaptive-o16"
    )
    reports = [candidate, baseline]
    comparison = benchmark.build_adaptive_overlap_comparison(
        reports,
        baseline_overlap=32,
        min_wall_reduction_percent=3.0,
    )
    by_overlap = {
        item["configuration"]["hat_overlap"]: item for item in comparison
    }

    assert by_overlap[32]["is_adaptive_overlap_baseline"] is True
    assert by_overlap[32]["meets_minimum_wall_reduction"] is None
    assert by_overlap[16]["is_adaptive_overlap_baseline"] is False
    assert by_overlap[16][
        "wall_reduction_vs_adaptive_baseline_percent"
    ] == pytest.approx(10.0)
    assert by_overlap[16]["meets_minimum_wall_reduction"] is True

    gate = benchmark.adaptive_overlap_qualification(
        reports,
        comparison,
        expected_configurations=[baseline_configuration, candidate_configuration],
        baseline_overlap=32,
        expected_routes={"normal": 9, "sharper": 21},
        expected_pages=30,
        representative_coverage=_production_coverage(),
        warmups=1,
        repeats=3,
        max_cv_percent=3.0,
        max_reserved_vram_bytes=14 * 1024**3,
        min_wall_reduction_percent=3.0,
    )

    assert gate["valid_for_performance_decision"] is True
    assert all(gate["checks"].values())
    assert gate["required_protocol"]["baseline_overlap"] == 32

    slow_candidate = _protocol_report(
        candidate_configuration, [98.0, 98.0, 98.0], "adaptive-o16-slow"
    )
    slow_reports = [slow_candidate, baseline]
    slow_comparison = benchmark.build_adaptive_overlap_comparison(
        slow_reports,
        baseline_overlap=32,
        min_wall_reduction_percent=3.0,
    )
    rejected = benchmark.adaptive_overlap_qualification(
        slow_reports,
        slow_comparison,
        expected_configurations=[baseline_configuration, candidate_configuration],
        baseline_overlap=32,
        expected_routes={"normal": 9, "sharper": 21},
        expected_pages=30,
        representative_coverage=_production_coverage(),
        warmups=1,
        repeats=3,
        max_cv_percent=3.0,
        max_reserved_vram_bytes=14 * 1024**3,
        min_wall_reduction_percent=3.0,
    )
    assert (
        rejected["checks"]["all_candidate_wall_reductions_at_least_minimum"]
        is False
    )
    assert rejected["valid_for_performance_decision"] is False


def test_production_gate_requires_complete_protocol_and_three_percent_speedup() -> None:
    baseline = _protocol_report(
        benchmark.BenchmarkConfiguration((256,), 32),
        [100.0, 100.0, 100.0],
        "fixed",
    )
    adaptive = _protocol_report(
        benchmark.BenchmarkConfiguration((256, 320), 32),
        [95.0, 95.0, 95.0],
        "adaptive",
    )
    reports = [baseline, adaptive]
    comparison = benchmark.build_comparison(reports, min_wall_reduction_percent=3.0)

    assert comparison[0]["is_fixed_256_baseline"] is True
    assert comparison[1]["wall_reduction_vs_fixed_256_percent"] == pytest.approx(5.0)
    assert comparison[1]["meets_minimum_wall_reduction"] is True

    gate = benchmark.production_qualification(
        reports,
        comparison,
        expected_routes={"normal": 9, "sharper": 21},
        expected_pages=30,
        representative_coverage=_production_coverage(),
        warmups=1,
        repeats=3,
        max_cv_percent=3.0,
        max_reserved_vram_bytes=14 * 1024**3,
        min_wall_reduction_percent=3.0,
    )

    assert gate["valid_for_production_decision"] is True
    assert all(gate["checks"].values())
    assert gate["required_protocol"]["min_wall_reduction_percent"] == 3.0

    slow_adaptive = _protocol_report(
        benchmark.BenchmarkConfiguration((256, 320), 32),
        [98.0, 98.0, 98.0],
        "adaptive-slow",
    )
    slow_reports = [baseline, slow_adaptive]
    slow_comparison = benchmark.build_comparison(
        slow_reports, min_wall_reduction_percent=3.0
    )
    rejected = benchmark.production_qualification(
        slow_reports,
        slow_comparison,
        expected_routes={"normal": 9, "sharper": 21},
        expected_pages=30,
        representative_coverage=_production_coverage(),
        warmups=1,
        repeats=3,
        max_cv_percent=3.0,
        max_reserved_vram_bytes=14 * 1024**3,
        min_wall_reduction_percent=3.0,
    )

    assert rejected["checks"]["adaptive_wall_reduction_at_least_minimum"] is False
    assert rejected["valid_for_production_decision"] is False


@pytest.mark.parametrize(
    ("warmup_count", "expected_valid"),
    [(0, False), (1, True), (2, True)],
)
def test_production_gate_requires_at_least_one_warmup_and_counts_actual_pairs(
    warmup_count: int, expected_valid: bool
) -> None:
    reports = [
        _protocol_report(
            benchmark.BenchmarkConfiguration((256,), 32),
            [100.0, 100.0, 100.0],
            f"fixed-w{warmup_count}",
            warmup_count=warmup_count,
        ),
        _protocol_report(
            benchmark.BenchmarkConfiguration((256, 320), 32),
            [95.0, 95.0, 95.0],
            f"adaptive-w{warmup_count}",
            warmup_count=warmup_count,
        ),
    ]
    comparison = benchmark.build_comparison(
        reports, min_wall_reduction_percent=3.0
    )
    gate = benchmark.production_qualification(
        reports,
        comparison,
        expected_routes={"normal": 9, "sharper": 21},
        expected_pages=30,
        representative_coverage=_production_coverage(),
        warmups=warmup_count,
        repeats=3,
        max_cv_percent=3.0,
        max_reserved_vram_bytes=14 * 1024**3,
        min_wall_reduction_percent=3.0,
    )

    assert gate["checks"]["at_least_one_warmup_per_configuration"] is (
        warmup_count >= 1
    )
    assert gate["checks"]["all_input_snapshots_unchanged"] is True
    assert gate["checks"]["fresh_isolated_owned_roots_per_run"] is True
    assert gate["checks"]["attempt_and_completion_integrity"] is True
    assert gate["checks"]["single_parent_session_and_complete_pairs"] is True
    assert gate["valid_for_production_decision"] is expected_valid
    assert gate["required_protocol"]["minimum_warmups_per_configuration"] == 1


def test_production_gate_rejects_wrong_representative_contract() -> None:
    reports = [
        _protocol_report(
            benchmark.BenchmarkConfiguration((256,), 32),
            [100.0, 100.0, 100.0],
            "fixed",
        ),
        _protocol_report(
            benchmark.BenchmarkConfiguration((256, 320), 32),
            [95.0, 95.0, 95.0],
            "adaptive",
        ),
    ]
    comparison = benchmark.build_comparison(reports, min_wall_reduction_percent=3.0)

    gate = benchmark.production_qualification(
        reports,
        comparison,
        expected_routes={"normal": 10, "sharper": 20},
        expected_pages=30,
        representative_coverage={**_production_coverage(), "exact_threshold": 0},
        warmups=0,
        repeats=3,
        max_cv_percent=4.0,
        max_reserved_vram_bytes=15 * 1024**3,
        min_wall_reduction_percent=2.0,
    )

    assert gate["checks"]["representative_routes_normal_9_sharper_21"] is False
    assert gate["checks"]["representative_boundary_and_image_modes_redecoded"] is False
    assert gate["checks"]["at_least_one_warmup_per_configuration"] is False
    assert gate["checks"]["cv_limit_is_no_looser_than_3_percent"] is False
    assert gate["checks"]["reserved_vram_limit_is_no_looser_than_14_gib"] is False
    assert gate["checks"]["speed_gate_is_no_looser_than_3_percent"] is False
    assert gate["valid_for_production_decision"] is False


def _write_complete_attempt(
    attempt: Path,
    fingerprint: str,
    input_root: Path,
    expected_input: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], Path]:
    attempt.mkdir(parents=True)
    configuration_spec = benchmark.BenchmarkConfiguration((256,), 32)
    configuration = configuration_spec.record()
    base = benchmark.load_config(_write_base_config(attempt))
    output = attempt / "output"
    config_path = attempt / "config.toml"
    config_path.write_text(
        benchmark.render_child_config(
            base,
            input_root=input_root,
            output_root=output,
            configuration=configuration_spec,
        ),
        encoding="utf-8",
    )
    output.mkdir()
    jxl = output / "page.jxl"
    jxl.write_bytes(b"jxl")
    metrics = attempt / "metrics" / "run"
    metrics.mkdir(parents=True)
    page = {
        "type": "waifuhat2x-page-metrics",
        "schema_version": 1,
        "run_id": "run-id",
        "status": "complete",
        "details": {
            "model_label": "Real-HAT-GAN-x4-normal",
            "peak_vram_bytes": 1,
            "peak_reserved_vram_bytes": 2,
            "tile": 256,
            "tile_candidates": [256],
            "tile_strategy": "fixed",
            "tile_estimator": None,
            "overlap": 32,
        },
    }
    pages = metrics / "pages.jsonl"
    pages.write_text(json.dumps(page) + "\n", encoding="utf-8")
    job = metrics / "job.json"
    job_data = {
        "type": "waifuhat2x-job-metrics",
        "schema_version": 1,
        "status": "complete",
        "run_id": "run-id",
        "pages_written": 1,
        "context": {
            "output_mode": "mirror",
            "output_format": "jxl",
            "input_root": str(input_root.resolve()),
            "output_root": str(output.resolve()),
        },
        "timing": {
            "stage_spans": {},
            "cumulative_service_seconds": {},
            "interval_summary": {},
        },
    }
    job.write_text(json.dumps(job_data), encoding="utf-8")
    result_path = attempt / "result.json"
    spec_path = attempt / "spec.json"
    model_inventory = {}
    for name in benchmark.REAL_HAT_MODELS:
        model_path = (attempt / "models" / "hat" / name).resolve()
        digest = benchmark.sha256_file(model_path)
        model_inventory[name] = {
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": digest,
            "official_sha256": digest,
        }
    cache_slot_root = attempt.parent.parent / "caches"
    cache_root = cache_slot_root / attempt.name
    benchmark.write_json(
        spec_path,
        {
            "schema_version": benchmark.SCHEMA_VERSION,
            "kind": benchmark.CHILD_SPEC_KIND,
            "fingerprint": fingerprint,
            "attempt_root": str(attempt.resolve()),
            "role": "repeat",
            "index": 1,
            "parent_session_id": "test-session",
            "pair_id": "repeat-1",
            "configuration": configuration,
            "input_root": str(input_root.resolve()),
            "input_snapshot": expected_input,
            "models": model_inventory,
            "config_path": str(config_path.resolve()),
            "config_sha256": benchmark.sha256_file(config_path),
            "output_root": str(output.resolve()),
            "metrics_root": str((attempt / "metrics").resolve()),
            "cache_root": str(cache_root.resolve()),
            "result_path": str(result_path.resolve()),
        },
    )
    result = {
        "schema_version": benchmark.SCHEMA_VERSION,
        "kind": benchmark.CHILD_RESULT_KIND,
        "status": "complete",
        "fingerprint": fingerprint,
        "role": "repeat",
        "index": 1,
        "parent_session_id": "test-session",
        "pair_id": "repeat-1",
        "configuration": configuration,
        "models_before": model_inventory,
        "models_after": model_inventory,
        "input_snapshot_before": expected_input,
        "input_snapshot_after": expected_input,
        "spec_path": "spec.json",
        "spec_sha256": benchmark.sha256_file(spec_path),
        "config_path": "config.toml",
        "config_sha256": benchmark.sha256_file(config_path),
        "job": {
            "path": "metrics/run/job.json",
            "sha256": benchmark.sha256_file(job),
        },
        "pages": {
            "path": "metrics/run/pages.jsonl",
            "sha256": benchmark.sha256_file(pages),
            "count": 1,
        },
        "jxl_outputs": [
            {
                "path": "page.jxl",
                "bytes": 3,
                "sha256": benchmark.sha256_file(jxl),
            }
        ],
        "pipeline_summary": {
            "wall_seconds": 1.0,
            "failed": 0,
            "deferred": 0,
            "target_unmet": 0,
            "metrics_write_errors": 0,
        },
        "telemetry": benchmark.summarize_telemetry(job_data, [page]),
        "isolation": {
            "fresh_roots_required_at_child_start": True,
            "input_output_disjoint": True,
            "input_metrics_disjoint": True,
            "input_cache_disjoint": True,
            "output_metrics_disjoint": True,
            "output_cache_disjoint": True,
            "metrics_cache_disjoint": True,
        },
        "owned_roots": {
            "output": str(output.resolve()),
            "metrics": str((attempt / "metrics").resolve()),
            "cache": str(cache_root.resolve()),
        },
    }
    benchmark.write_json(result_path, result)
    benchmark.write_json(
        attempt / "attempt.json",
        {
            "schema_version": benchmark.SCHEMA_VERSION,
            "kind": "real_hat_pipeline_e2e_attempt",
            "status": "complete",
            "fingerprint": fingerprint,
            "returncode": 0,
            "timed_out": False,
            "termination": {"process_group_checked": True},
        },
    )
    benchmark.write_json(
        attempt / "completion.json",
        {
            "schema_version": benchmark.SCHEMA_VERSION,
            "kind": benchmark.COMPLETION_KIND,
            "fingerprint": fingerprint,
            "result_path": "result.json",
            "result_sha256": benchmark.sha256_file(result_path),
            "spec_sha256": benchmark.sha256_file(spec_path),
            "config_sha256": benchmark.sha256_file(config_path),
            "job_sha256": benchmark.sha256_file(job),
            "pages_sha256": benchmark.sha256_file(pages),
        },
    )
    return model_inventory, cache_slot_root


def test_resume_reuses_only_complete_fingerprint_and_hash_matching_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    (input_root / "page.png").write_bytes(b"page")
    expected_input = benchmark.input_snapshot(input_root)
    slot = tmp_path / "slot"
    complete = slot / "attempt-001"
    models, cache_slot_root = _write_complete_attempt(
        complete, "fingerprint", input_root, expected_input
    )
    for name, metadata in models.items():
        monkeypatch.setitem(
            benchmark.OFFICIAL_REAL_HAT_SHA256, name, metadata["sha256"]
        )
    incomplete = slot / "attempt-002"
    incomplete.mkdir(parents=True)
    (incomplete / "result.json").write_text("{}", encoding="utf-8")

    def find(fingerprint: str = "fingerprint") -> tuple[Path, dict[str, Any]] | None:
        return benchmark.find_reusable_result(
            slot,
            fingerprint=fingerprint,
            expected_input_root=input_root,
            expected_input=expected_input,
            expected_models=models,
            expected_configuration=benchmark.BenchmarkConfiguration((256,), 32),
            expected_role="repeat",
            expected_index=1,
            cache_slot_root=cache_slot_root,
        )

    reusable = find()

    assert reusable is not None
    assert reusable[0] == complete

    completion_marker = json.loads(
        (complete / "completion.json").read_text(encoding="utf-8")
    )
    completion_marker["result_sha256"] = "0" * 64
    benchmark.write_json(complete / "completion.json", completion_marker)
    assert find() is None
    completion_marker["result_sha256"] = benchmark.sha256_file(complete / "result.json")
    benchmark.write_json(complete / "completion.json", completion_marker)

    attempt_status = json.loads((complete / "attempt.json").read_text(encoding="utf-8"))
    attempt_status["status"] = "running"
    benchmark.write_json(complete / "attempt.json", attempt_status)
    assert find() is None
    attempt_status["status"] = "complete"
    benchmark.write_json(complete / "attempt.json", attempt_status)

    attempt_status["timed_out"] = True
    benchmark.write_json(complete / "attempt.json", attempt_status)
    assert find() is None
    attempt_status["timed_out"] = False
    attempt_status["returncode"] = 1
    benchmark.write_json(complete / "attempt.json", attempt_status)
    assert find() is None
    attempt_status["returncode"] = 0
    benchmark.write_json(complete / "attempt.json", attempt_status)

    assert find("different") is None
    (complete / "output" / "page.jxl").write_bytes(b"changed")
    assert find() is None


def test_nonempty_benchmark_root_requires_matching_resume_summary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "benchmark"
    root.mkdir()
    (root / "unmanaged.txt").write_text("occupied", encoding="utf-8")

    with pytest.raises(ValueError, match="no resumable"):
        benchmark._prepare_benchmark_root(
            root,
            resume=True,
            fingerprint="fingerprint",
            cache_root=root / "caches",
        )

    (root / "benchmark_summary.json").write_text(
        json.dumps(
            {
                "schema_version": benchmark.SCHEMA_VERSION,
                "kind": benchmark.SUMMARY_KIND,
                "fingerprint": "fingerprint",
                "output_root": str(root.resolve()),
                "cache_root": str((root / "caches").resolve()),
            }
        ),
        encoding="utf-8",
    )
    previous = benchmark._prepare_benchmark_root(
        root,
        resume=True,
        fingerprint="fingerprint",
        cache_root=root / "caches",
    )
    assert previous is not None


def test_next_attempt_never_reuses_incomplete_directory(tmp_path: Path) -> None:
    slot = tmp_path / "slot"
    (slot / "attempt-001").mkdir(parents=True)
    (slot / "attempt-003").mkdir()

    assert benchmark.next_attempt_root(slot).name == "attempt-004"


def test_output_session_lease_rejects_concurrent_owner(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    first = benchmark.BenchmarkSessionLease(root)
    second = benchmark.BenchmarkSessionLease(root)
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="Another benchmark process"):
            second.acquire()
    finally:
        first.release()

    lease = json.loads((root / benchmark.SESSION_LOCK_NAME).read_text(encoding="utf-8"))
    assert lease["pid"] > 0
    assert lease["output_root"] == str(root.resolve())
    assert (
        benchmark._prepare_benchmark_root(
            root,
            resume=True,
            fingerprint="fingerprint",
            cache_root=root / "caches",
        )
        is None
    )
    second.acquire()
    second.release()


def test_child_runner_uses_current_python_and_isolated_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    cache = tmp_path / "cache"
    spec = attempt / "spec.json"
    benchmark.write_json(
        spec,
        {
            "fingerprint": "fingerprint",
            "cache_root": str(cache),
        },
    )
    captured: dict[str, Any] = {}

    class FakeProcess:
        pid = 42

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            captured["command"] = command
            captured["env"] = kwargs["env"]

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

        def poll(self) -> int:
            return 0

    monkeypatch.setattr(benchmark.subprocess, "Popen", FakeProcess)

    report = benchmark.run_child_process(
        spec,
        attempt / "child.log",
        60.0,
        backend_environment={"TORCH_BLAS_PREFER_HIPBLASLT": "1"},
    )

    assert report["returncode"] == 0
    assert captured["command"][:3] == [
        sys.executable,
        str(Path(benchmark.__file__).resolve()),
        benchmark.CHILD_FLAG,
    ]
    assert captured["env"]["TORCHINDUCTOR_CACHE_DIR"] == str(cache.resolve())
    assert captured["env"]["TORCH_BLAS_PREFER_HIPBLASLT"] == "1"
    assert report["backend_environment"] == {"TORCH_BLAS_PREFER_HIPBLASLT": "1"}
    assert (
        json.loads((attempt / "attempt.json").read_text(encoding="utf-8"))["status"]
        == "complete"
    )


def test_child_runner_reaps_child_on_parent_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    spec = attempt / "spec.json"
    benchmark.write_json(
        spec,
        {
            "fingerprint": "fingerprint",
            "cache_root": str(tmp_path / "cache"),
        },
    )

    class InterruptedProcess:
        pid = 43
        stopped = False

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            if not self.stopped:
                raise KeyboardInterrupt
            return -1

        def poll(self) -> int | None:
            return -1 if self.stopped else None

    process = InterruptedProcess()

    def terminate(target: InterruptedProcess) -> dict[str, Any]:
        assert target is process
        target.stopped = True
        return {"method": "test"}

    monkeypatch.setattr(
        benchmark.subprocess, "Popen", lambda *_args, **_kwargs: process
    )
    monkeypatch.setattr(benchmark, "_terminate_process_tree", terminate)

    with pytest.raises(KeyboardInterrupt):
        benchmark.run_child_process(spec, attempt / "child.log", 60.0)

    assert process.stopped is True
    assert (
        json.loads((attempt / "attempt.json").read_text(encoding="utf-8"))["status"]
        == "running"
    )


def test_bounded_wait_never_falls_back_to_unbounded_wait() -> None:
    class StuckProcess:
        pid = 44

        def wait(self, timeout: float | None = None) -> int:
            assert timeout == 0.25
            raise subprocess.TimeoutExpired("child", timeout)

    with pytest.raises(RuntimeError, match="did not exit within 0.25s"):
        benchmark._bounded_wait(StuckProcess(), 0.25)  # type: ignore[arg-type]
