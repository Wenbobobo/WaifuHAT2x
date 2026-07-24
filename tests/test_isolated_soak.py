from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
import pytest

import scripts.isolated_soak as soak
from waifuhat2x.config import load_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _interval() -> dict[str, Any]:
    return {
        "clock": "cpu_monotonic",
        "start_offset_ns": 0,
        "end_offset_ns": 10_000_000,
        "duration_seconds": 0.01,
    }


def _json_string(value: Path) -> str:
    return json.dumps(str(value.resolve()))


def _write_base_config(root: Path) -> tuple[Path, Path, Path]:
    source_root = root / "production-library"
    source_root.mkdir()
    model_root = root / "models"
    hat_root = model_root / "hat"
    hat_root.mkdir(parents=True)
    normal = hat_root / "Real_HAT_GAN_SRx4.pth"
    sharper = hat_root / "Real_HAT_GAN_SRx4_sharper.pth"
    normal.write_bytes(b"normal-checkpoint")
    sharper.write_bytes(b"sharper-checkpoint")
    model_manifest = root / "model_sources.toml"
    model_manifest.write_text(
        f"""
[models.normal]
filename = "hat/Real_HAT_GAN_SRx4.pth"
sha256 = "{_sha256(normal)}"

[models.sharper]
filename = "hat/Real_HAT_GAN_SRx4_sharper.pth"
sha256 = "{_sha256(sharper)}"
""".lstrip(),
        encoding="utf-8",
    )
    config = root / "base.toml"
    config.write_text(
        f"""
[paths]
input = {_json_string(source_root)}
output = {_json_string(root / "unused-output")}
models = {_json_string(model_root)}

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
hat_tile_candidates = [256, 320]
hat_overlap = 16
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
    return config, model_manifest, source_root


def _page_record(
    *, path: Path, manifest_root: Path, source_name: str, index: int, route: str
) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
        grayscale = image.mode == "L"
        source_mode = image.mode
        source_format = image.format
    digest = _sha256(path)
    return {
        "source": source_name,
        "width": width,
        "height": height,
        "short_edge": min(width, height),
        "long_edge": max(width, height),
        "pixels": width * height,
        "route": route,
        "grayscale": grayscale,
        "odd_dimension": bool(width % 2 or height % 2),
        "source_mode": source_mode,
        "source_format": source_format,
        "file_bytes": path.stat().st_size,
        "index": index,
        "source_sha256": digest,
        "copied_path": path.relative_to(manifest_root).as_posix(),
        "copied_sha256": digest,
    }


def _write_manifest(root: Path, source_root: Path) -> Path:
    representative = root / "representative"
    inputs = representative / "inputs"
    inputs.mkdir(parents=True)
    normal = inputs / "01_normal.png"
    sharper = inputs / "02_sharper.png"
    Image.new("RGB", (9, 11), (12, 34, 56)).save(normal)
    Image.new("L", (1000, 1001), 127).save(sharper)
    records = [
        _page_record(
            path=normal,
            manifest_root=representative,
            source_name="formal/normal.png",
            index=1,
            route="normal",
        ),
        _page_record(
            path=sharper,
            manifest_root=representative,
            source_name="formal/sharper.png",
            index=2,
            route="sharper",
        ),
    ]
    manifest = {
        "schema_version": 1,
        "kind": soak.MANIFEST_KIND,
        "source_root": str(source_root.resolve()),
        "source_is_read_only": True,
        "selected_counts": {"normal": 1, "sharper": 1},
        "coverage": {
            "exact_threshold": 1,
            "grayscale": 1,
            "rgb_or_color": 1,
            "odd_dimension": 2,
            "minimum_selected_pixels": 99,
            "maximum_selected_pixels": 1001000,
        },
        "discovery": {"errors": 0},
        "pages": records,
    }
    path = representative / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def _render_environment(tmp_path: Path) -> dict[str, Any]:
    config, model_manifest, source_root = _write_base_config(tmp_path)
    manifest = _write_manifest(tmp_path, source_root)
    representative = manifest.parent
    output_root = representative / "mirror-output"
    metrics_root = representative / "metrics-first"
    isolated_config = representative / "mirror.toml"
    result = soak.render_isolated_config(
        base_config_path=config,
        manifest_path=manifest,
        output_root=output_root,
        metrics_root=metrics_root,
        output_path=isolated_config,
        model_manifest_path=model_manifest,
    )
    return {
        "base_config": config,
        "model_manifest": model_manifest,
        "manifest": manifest,
        "input_root": representative / "inputs",
        "output_root": output_root,
        "metrics_root": metrics_root,
        "isolated_config": isolated_config,
        "render_result": result,
    }


def _model_inventory(environment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    config = load_config(environment["isolated_config"])
    return soak._model_inventory(config, environment["model_manifest"])


def _expected_pages(environment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    base = load_config(environment["base_config"])
    _, expected, _ = soak._manifest_pages(
        environment["manifest"], environment["input_root"], base
    )
    return expected


def _common_page_details(
    key: str, fact: dict[str, Any], models: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    route = fact["route"]
    return {
        "source_extension": Path(key).suffix.lower(),
        "source_bytes": fact["bytes"],
        "source_dimensions": [fact["width"], fact["height"]],
        "source_short_edge": fact["short_edge"],
        "grayscale": fact["grayscale"],
        "upscale": True,
        "native_scale": 4,
        "planned_output_dimensions": [fact["output_width"], fact["output_height"]],
        "plan_reason": fact["plan_reason"],
        "model_label": soak.REAL_HAT_LABELS[route],
        "model_checkpoint": models[route]["filename"],
    }


def _write_run(
    environment: dict[str, Any], *, run_kind: str, metrics_root: Path | None = None
) -> Path:
    config = load_config(environment["isolated_config"])
    from waifuhat2x.pipeline import _pipeline_signature

    pipeline_signature = _pipeline_signature(config)
    expected = _expected_pages(environment)
    models = _model_inventory(environment)
    output_root = environment["output_root"]
    if not output_root.exists():
        output_root.mkdir()
        for key, fact in expected.items():
            destination = output_root / Path(fact["destination"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(f"jxl:{key}".encode())
        (output_root / soak.LOCK_NAME).write_text("pid=1\n", encoding="utf-8")
        state: dict[str, dict[str, Any]] = {}
        for key, fact in expected.items():
            source = environment["input_root"] / Path(key)
            destination = output_root / Path(fact["destination"])
            state[key] = {
                "size": source.stat().st_size,
                "mtime_ns": source.stat().st_mtime_ns,
                "source_sha256": _sha256(source),
                "source_root": str(environment["input_root"].resolve()),
                "model_sha256": models[fact["route"]]["sha256"],
                "pipeline_signature": pipeline_signature,
                "phase": "committed",
                "destination": fact["destination"],
                "output_size": destination.stat().st_size,
                "output_sha256": _sha256(destination),
            }
        (output_root / soak.STATE_NAME).write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
    else:
        state = json.loads((output_root / soak.STATE_NAME).read_text(encoding="utf-8"))
        assert next(iter(state.values()))["pipeline_signature"] == pipeline_signature

    metrics_root = metrics_root or environment["metrics_root"]
    run_root = metrics_root / "run-001"
    run_root.mkdir(parents=True)
    run_id = f"{run_kind}-run"
    pages: list[dict[str, Any]] = []
    first = run_kind == "first"
    output_bytes = 0
    for key, fact in sorted(expected.items(), key=lambda item: item[1]["index"]):
        details = _common_page_details(key, fact, models)
        if first:
            destination = output_root / Path(fact["destination"])
            details.update(
                precision="bf16",
                batch_tiles=1,
                tile_candidates=[256, 320],
                overlap=16,
                tile=256 if fact["route"] == "normal" else 320,
                peak_reserved_vram_bytes=4 * 1024**3,
                output_bytes=destination.stat().st_size,
            )
            output_bytes += destination.stat().st_size
            span_names = {
                "read",
                "decode_exif",
                "analyze_plan",
                "hash_and_fingerprint",
                "engine_path",
                "gpu_inference",
                "jxl_service",
                "cjxl",
                "djxl",
                "candidate_hash",
                "commit",
            }
        else:
            span_names = {"read", "decode_exif", "analyze_plan", "hash_and_fingerprint"}
        pages.append(
            {
                "type": soak.PAGE_METRICS_TYPE,
                "schema_version": 1,
                "run_id": run_id,
                "index": fact["index"],
                "total": len(expected),
                "source": key,
                "status": "complete" if first else "skipped",
                "details": details,
                "timing": {
                    "clock": "perf_counter_ns",
                    "spans": {name: [_interval()] for name in sorted(span_names)},
                    "cumulative_service_seconds": {},
                },
                "destination": fact["destination"],
            }
        )
    target_unmet = sum(fact["target_unmet"] for fact in expected.values())
    summary = {
        "processed": len(expected) if first else 0,
        "skipped": 0 if first else len(expected),
        "copied": 0,
        "ignored": 0,
        "jxl_skipped": 0,
        "failed": 0,
        "sr_pages": len(expected) if first else 0,
        "transcoded_pages": 0,
        "replaced_sources": 0,
        "existing_jxl_adopted": 0,
        "existing_jxl_replaced": 0,
        "external_jxl_recoveries": 0,
        "deferred": 0,
        "target_unmet": target_unmet if first else 0,
        "output_bytes": output_bytes if first else 0,
        "metrics_directory": str(run_root.resolve()),
        "metrics_write_errors": 0,
    }
    common_intervals = {"read", "decode_exif", "analyze_plan", "hash_and_fingerprint"}
    first_intervals = {
        "engine_path",
        "gpu_inference",
        "jxl_service",
        "cjxl",
        "djxl",
        "candidate_hash",
        "commit",
    }
    interval_summary = {
        name: {
            "count": len(expected),
            "cumulative_seconds": 0.02,
            "union_seconds": 0.02,
            "critical_path_percent": 2.0,
        }
        for name in common_intervals | (first_intervals if first else set())
    }
    services = (
        {
            "gpu_synchronized_inference": 0.02,
            "forward": 0.02,
            "cjxl": 0.02,
            "djxl": 0.02,
            "candidate_hash": 0.02,
        }
        if first
        else {}
    )
    job = {
        "type": soak.JOB_METRICS_TYPE,
        "schema_version": 1,
        "run_id": run_id,
        "status": "complete",
        "wall_seconds": 1.0,
        "pages_written": len(expected),
        "write_errors": [],
        "context": {
            "input_root": str(config.paths.input),
            "output_root": str(config.paths.output),
            "output_mode": "mirror",
            "output_format": "jxl",
            "processing_profile": "real-hat-auto",
            "pipeline_signature": pipeline_signature,
        },
        "summary": {**summary, "wall_seconds": 1.0},
        "timing": {
            "clock": "perf_counter_ns",
            "stage_spans": {
                name: [_interval()]
                for name in ("recovery", "discovery", "preflight", "worklist")
            },
            "cumulative_service_seconds": services,
            "interval_summary": interval_summary,
        },
    }
    (run_root / "job.json").write_text(
        json.dumps(job, indent=2) + "\n", encoding="utf-8"
    )
    (run_root / "pages.jsonl").write_text(
        "".join(json.dumps(page, separators=(",", ":")) + "\n" for page in pages),
        encoding="utf-8",
    )
    return metrics_root


def _decoder(calls: list[tuple[Path, int, int]] | None = None) -> soak.Decoder:
    def decode(path: Path, width: int, height: int) -> tuple[int, int]:
        if calls is not None:
            calls.append((path, width, height))
        return width, height

    return decode


def _verify(
    environment: dict[str, Any],
    *,
    output: Path,
    metrics_root: Path | None = None,
    run_kind: str = "first",
    baseline: Path | None = None,
    decoder: soak.Decoder | None = None,
) -> dict[str, Any]:
    return soak.verify_soak(
        base_config_path=environment["base_config"],
        isolated_config_path=environment["isolated_config"],
        manifest_path=environment["manifest"],
        metrics_root=metrics_root or environment["metrics_root"],
        output_path=output,
        expected_pages=2,
        expected_normal=1,
        expected_sharper=1,
        max_reserved_vram_gib=14,
        run_kind=run_kind,
        baseline_path=baseline,
        model_manifest_path=environment["model_manifest"],
        min_exact_threshold=1,
        decoder=decoder or _decoder(),
    )


def test_render_config_inherits_production_semantics_and_keeps_roots_fresh(
    tmp_path: Path,
) -> None:
    environment = _render_environment(tmp_path)
    base = load_config(environment["base_config"])
    isolated = load_config(environment["isolated_config"])

    assert isolated.processing == base.processing
    assert isolated.processing.hat_overlap == 16
    assert isolated.jxl == base.jxl
    assert isolated.paths.input == environment["input_root"].resolve()
    assert isolated.paths.output == environment["output_root"].resolve()
    assert isolated.output.mode == "mirror"
    assert isolated.output.copy_non_images is False
    assert isolated.output.existing_jxl_policy == "error"
    assert not environment["output_root"].exists()
    assert not environment["metrics_root"].exists()
    assert environment["render_result"]["route_counts"] == {
        "normal": 1,
        "sharper": 1,
    }


def test_render_config_rejects_overlapping_output_root(tmp_path: Path) -> None:
    config, model_manifest, source_root = _write_base_config(tmp_path)
    manifest = _write_manifest(tmp_path, source_root)

    with pytest.raises(ValueError, match="input and output roots overlap"):
        soak.render_isolated_config(
            base_config_path=config,
            manifest_path=manifest,
            output_root=manifest.parent / "inputs" / "output",
            metrics_root=manifest.parent / "metrics",
            output_path=manifest.parent / "isolated.toml",
            model_manifest_path=model_manifest,
        )


def test_render_config_rejects_input_hash_drift(tmp_path: Path) -> None:
    config, model_manifest, source_root = _write_base_config(tmp_path)
    manifest = _write_manifest(tmp_path, source_root)
    (manifest.parent / "inputs" / "01_normal.png").write_bytes(b"changed")

    with pytest.raises(ValueError, match="hash mismatch"):
        soak.render_isolated_config(
            base_config_path=config,
            manifest_path=manifest,
            output_root=manifest.parent / "output",
            metrics_root=manifest.parent / "metrics",
            output_path=manifest.parent / "isolated.toml",
            model_manifest_path=model_manifest,
        )


def test_render_config_refuses_to_write_inside_production_input(tmp_path: Path) -> None:
    config, model_manifest, source_root = _write_base_config(tmp_path)
    manifest = _write_manifest(tmp_path, source_root)
    protected_parent = source_root / "generated"

    with pytest.raises(ValueError, match="overlaps production input root"):
        soak.render_isolated_config(
            base_config_path=config,
            manifest_path=manifest,
            output_root=manifest.parent / "output",
            metrics_root=manifest.parent / "metrics",
            output_path=protected_parent / "isolated.toml",
            model_manifest_path=model_manifest,
        )

    assert not protected_parent.exists()


def test_render_config_requires_production_overlap_16(tmp_path: Path) -> None:
    config, model_manifest, source_root = _write_base_config(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "hat_overlap = 16", "hat_overlap = 32"
        ),
        encoding="utf-8",
    )
    manifest = _write_manifest(tmp_path, source_root)

    with pytest.raises(ValueError, match="processing.hat_overlap"):
        soak.render_isolated_config(
            base_config_path=config,
            manifest_path=manifest,
            output_root=manifest.parent / "output",
            metrics_root=manifest.parent / "metrics",
            output_path=manifest.parent / "isolated.toml",
            model_manifest_path=model_manifest,
        )


def test_first_run_verification_binds_all_artifacts(tmp_path: Path) -> None:
    environment = _render_environment(tmp_path)
    _write_run(environment, run_kind="first")
    calls: list[tuple[Path, int, int]] = []
    report_path = environment["manifest"].parent / "first-attestation.json"

    report = _verify(environment, output=report_path, decoder=_decoder(calls))

    assert report["valid"] is True
    assert report["run_kind"] == "first"
    assert report["page_evidence"]["route_counts"] == {
        "normal": 1,
        "sharper": 1,
    }
    assert report["state"]["records"] == 2
    assert len(calls) == 2
    assert report_path.is_file()


def test_verify_refuses_report_inside_model_root_before_decoding(
    tmp_path: Path,
) -> None:
    environment = _render_environment(tmp_path)
    _write_run(environment, run_kind="first")
    protected_parent = load_config(environment["base_config"]).paths.models / "reports"
    calls: list[tuple[Path, int, int]] = []

    with pytest.raises(ValueError, match="overlaps model root"):
        _verify(
            environment,
            output=protected_parent / "attestation.json",
            decoder=_decoder(calls),
        )

    assert not protected_parent.exists()
    assert calls == []


def test_incremental_verification_requires_exact_baseline_inventories(
    tmp_path: Path,
) -> None:
    environment = _render_environment(tmp_path)
    _write_run(environment, run_kind="first")
    baseline = environment["manifest"].parent / "first-attestation.json"
    _verify(environment, output=baseline)
    incremental_metrics = environment["manifest"].parent / "metrics-incremental"
    _write_run(environment, run_kind="incremental", metrics_root=incremental_metrics)
    incremental = environment["manifest"].parent / "incremental-attestation.json"

    report = _verify(
        environment,
        output=incremental,
        metrics_root=incremental_metrics,
        run_kind="incremental",
        baseline=baseline,
    )

    assert report["valid"] is True
    assert report["job_summary"]["processed"] == 0
    assert report["job_summary"]["skipped"] == 2
    assert report["checks"]["incremental_inventories_unchanged"] is True


@pytest.mark.parametrize(
    ("mutation", "error_pattern"),
    [
        ("loosen_vram_gate", "acceptance thresholds"),
        ("disable_first_run_check", "first-run checks"),
        ("inflate_first_run_peak", "first-run page evidence"),
    ],
)
def test_incremental_verification_rejects_weakened_baseline(
    tmp_path: Path, mutation: str, error_pattern: str
) -> None:
    environment = _render_environment(tmp_path)
    _write_run(environment, run_kind="first")
    baseline = environment["manifest"].parent / "first-attestation.json"
    _verify(environment, output=baseline)

    baseline_data = json.loads(baseline.read_text(encoding="utf-8"))
    if mutation == "loosen_vram_gate":
        baseline_data["expected"]["max_reserved_vram_bytes"] = 15 * 1024**3
    elif mutation == "disable_first_run_check":
        baseline_data["checks"]["final_jxl_redecoded"] = False
    else:
        baseline_data["page_evidence"]["peak_reserved_vram_bytes"] = 15 * 1024**3
    baseline.write_text(json.dumps(baseline_data, indent=2) + "\n", encoding="utf-8")

    incremental_metrics = environment["manifest"].parent / "metrics-incremental"
    _write_run(environment, run_kind="incremental", metrics_root=incremental_metrics)

    with pytest.raises(ValueError, match=error_pattern):
        _verify(
            environment,
            output=environment["manifest"].parent / "incremental-attestation.json",
            metrics_root=incremental_metrics,
            run_kind="incremental",
            baseline=baseline,
        )


@pytest.mark.parametrize("residual", [soak.WORKLIST_NAME, ".candidate.part"])
def test_verification_rejects_transaction_residue(
    tmp_path: Path, residual: str
) -> None:
    environment = _render_environment(tmp_path)
    _write_run(environment, run_kind="first")
    (environment["output_root"] / residual).write_bytes(b"residual")

    with pytest.raises(ValueError, match="worklist|part file"):
        _verify(environment, output=tmp_path / "attestation.json")


def test_verification_requires_committed_hash_bound_mirror_state(
    tmp_path: Path,
) -> None:
    environment = _render_environment(tmp_path)
    _write_run(environment, run_kind="first")
    state_path = environment["output_root"] / soak.STATE_NAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    next(iter(state.values()))["phase"] = "prepared"
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="state phase mismatch"):
        _verify(environment, output=tmp_path / "attestation.json")


def test_verification_recomputes_pipeline_signature_from_isolated_config(
    tmp_path: Path,
) -> None:
    environment = _render_environment(tmp_path)
    metrics_root = _write_run(environment, run_kind="first")
    job_path = metrics_root / "run-001" / "job.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["context"]["pipeline_signature"] = "b" * 64
    job_path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pipeline signature differs"):
        _verify(environment, output=tmp_path / "attestation.json")


def test_verification_accepts_complete_runtime_index_permutation(
    tmp_path: Path,
) -> None:
    environment = _render_environment(tmp_path)
    metrics_root = _write_run(environment, run_kind="first")
    pages_path = metrics_root / "run-001" / "pages.jsonl"
    pages = [
        json.loads(line) for line in pages_path.read_text(encoding="utf-8").splitlines()
    ]
    pages[0]["index"], pages[1]["index"] = pages[1]["index"], pages[0]["index"]
    pages_path.write_text(
        "".join(json.dumps(page) + "\n" for page in pages), encoding="utf-8"
    )

    report = _verify(environment, output=tmp_path / "attestation.json")

    assert report["checks"]["metrics_complete_and_bound"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_checkpoint", "Real_HAT_GAN_SRx4_sharper.pth", "model_checkpoint"),
        ("native_scale", 2, "native_scale"),
        ("planned_output_dimensions", [1, 1], "planned_output_dimensions"),
        ("peak_reserved_vram_bytes", 15 * 1024**3, "reserved VRAM"),
    ],
)
def test_verification_rejects_page_evidence_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    environment = _render_environment(tmp_path)
    metrics_root = _write_run(environment, run_kind="first")
    pages_path = metrics_root / "run-001" / "pages.jsonl"
    pages = [
        json.loads(line) for line in pages_path.read_text(encoding="utf-8").splitlines()
    ]
    pages[0]["details"][field] = value
    pages_path.write_text(
        "".join(json.dumps(page) + "\n" for page in pages), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        _verify(environment, output=tmp_path / "attestation.json")


@pytest.mark.parametrize("damage", ["empty", "negative-duration"])
def test_verification_rejects_empty_or_invalid_required_page_spans(
    tmp_path: Path, damage: str
) -> None:
    environment = _render_environment(tmp_path)
    metrics_root = _write_run(environment, run_kind="first")
    pages_path = metrics_root / "run-001" / "pages.jsonl"
    pages = [
        json.loads(line) for line in pages_path.read_text(encoding="utf-8").splitlines()
    ]
    if damage == "empty":
        pages[0]["timing"]["spans"]["engine_path"] = []
    else:
        pages[0]["timing"]["spans"]["engine_path"][0]["duration_seconds"] = -0.1
    pages_path.write_text(
        "".join(json.dumps(page) + "\n" for page in pages), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="non-empty list|invalid interval"):
        _verify(environment, output=tmp_path / "attestation.json")


@pytest.mark.parametrize("damage", ["empty-stage", "wrong-count"])
def test_verification_rejects_invalid_job_timing(tmp_path: Path, damage: str) -> None:
    environment = _render_environment(tmp_path)
    metrics_root = _write_run(environment, run_kind="first")
    job_path = metrics_root / "run-001" / "job.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    if damage == "empty-stage":
        job["timing"]["stage_spans"]["recovery"] = []
    else:
        job["timing"]["interval_summary"]["gpu_inference"]["count"] = 1
    job_path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty list|interval count"):
        _verify(environment, output=tmp_path / "attestation.json")


@pytest.mark.parametrize("write_errors", [0, ["disk full"]])
def test_verification_rejects_invalid_job_write_errors(
    tmp_path: Path, write_errors: object
) -> None:
    environment = _render_environment(tmp_path)
    metrics_root = _write_run(environment, run_kind="first")
    job_path = metrics_root / "run-001" / "job.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["write_errors"] = write_errors
    job_path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="page/write counts"):
        _verify(environment, output=tmp_path / "attestation.json")


def test_verification_rejects_final_decode_failure(tmp_path: Path) -> None:
    environment = _render_environment(tmp_path)
    _write_run(environment, run_kind="first")

    def reject(_path: Path, _width: int, _height: int) -> tuple[int, int]:
        raise RuntimeError("djxl failed")

    with pytest.raises(RuntimeError, match="djxl failed"):
        _verify(
            environment,
            output=tmp_path / "attestation.json",
            decoder=reject,
        )


def test_verification_refuses_to_overwrite_report_before_decoding(
    tmp_path: Path,
) -> None:
    environment = _render_environment(tmp_path)
    _write_run(environment, run_kind="first")
    report = tmp_path / "attestation.json"
    report.write_text("keep", encoding="utf-8")
    calls: list[tuple[Path, int, int]] = []

    with pytest.raises(FileExistsError, match="already exists"):
        _verify(environment, output=report, decoder=_decoder(calls))

    assert report.read_text(encoding="utf-8") == "keep"
    assert calls == []


def test_verification_rejects_multiple_metrics_runs(tmp_path: Path) -> None:
    environment = _render_environment(tmp_path)
    metrics_root = _write_run(environment, run_kind="first")
    duplicate = metrics_root / "run-002"
    duplicate.mkdir()
    (duplicate / "job.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one job.json"):
        _verify(environment, output=tmp_path / "attestation.json")


def test_verification_rejects_isolated_config_drift(tmp_path: Path) -> None:
    environment = _render_environment(tmp_path)
    config_path = environment["isolated_config"]
    content = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        content.replace("hat_overlap = 16", "hat_overlap = 24"), encoding="utf-8"
    )
    _write_run(environment, run_kind="first")

    with pytest.raises(ValueError, match="drifted from base"):
        _verify(environment, output=tmp_path / "attestation.json")


def test_incremental_verification_rejects_changed_jxl_bytes(tmp_path: Path) -> None:
    environment = _render_environment(tmp_path)
    _write_run(environment, run_kind="first")
    baseline = environment["manifest"].parent / "first-attestation.json"
    _verify(environment, output=baseline)
    incremental_metrics = environment["manifest"].parent / "metrics-incremental"
    _write_run(environment, run_kind="incremental", metrics_root=incremental_metrics)
    expected = _expected_pages(environment)
    first_fact = next(iter(expected.values()))
    destination = environment["output_root"] / Path(first_fact["destination"])
    destination.write_bytes(b"changed-jxl")
    state_path = environment["output_root"] / soak.STATE_NAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    record = state[next(iter(expected))]
    record["output_size"] = destination.stat().st_size
    record["output_sha256"] = _sha256(destination)
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="JXL inventory changed"):
        _verify(
            environment,
            output=tmp_path / "incremental-attestation.json",
            metrics_root=incremental_metrics,
            run_kind="incremental",
            baseline=baseline,
        )
