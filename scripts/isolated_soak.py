from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tomllib
from typing import Any, Callable

from PIL import Image, ImageOps

from waifuhat2x.config import AppConfig, load_config
from waifuhat2x.images import IMAGE_EXTENSIONS, is_grayscale, plan_resolution


REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "waifuhat2x-isolated-soak-attestation"
MANIFEST_KIND = "real_hat_representative_manifest"
PAGE_METRICS_TYPE = "waifuhat2x-page-metrics"
JOB_METRICS_TYPE = "waifuhat2x-job-metrics"
STATE_NAME = ".waifuhat2x-state.json"
WORKLIST_NAME = ".waifuhat2x-worklist.jsonl"
LOCK_NAME = ".waifuhat2x.lock"
REAL_HAT_FILENAMES = {
    "normal": "hat/Real_HAT_GAN_SRx4.pth",
    "sharper": "hat/Real_HAT_GAN_SRx4_sharper.pth",
}
REAL_HAT_LABELS = {
    "normal": "Real-HAT-GAN-x4-normal",
    "sharper": "Real-HAT-GAN-x4-sharper",
}
HEX_DIGITS = frozenset("0123456789abcdef")


Decoder = Callable[[Path, int, int], tuple[int, int]]


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_exclusive(path: Path, content: str) -> None:
    if path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    _write_exclusive(path, encoded)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in HEX_DIGITS for character in value)
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    first = left.expanduser().resolve()
    second = right.expanduser().resolve()
    return first == second or first in second.parents or second in first.parents


def _safe_relative(path: Path, root: Path, description: str) -> str:
    resolved = path.resolve()
    base = root.resolve()
    try:
        relative = resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"{description} escapes its required root: {path}") from exc
    if not relative.parts:
        raise ValueError(f"{description} must be below its required root: {path}")
    return relative.as_posix()


def _assert_regular_file(path: Path, description: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular non-symlink file: {path}")


def _assert_directory(path: Path, description: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{description} must be a non-symlink directory: {path}")


def _assert_fresh_path(path: Path, description: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{description} already exists: {path}")


def _validate_isolated_roots(
    *,
    source_root: Path,
    model_root: Path,
    input_root: Path,
    output_root: Path,
    metrics_root: Path,
) -> None:
    isolated = {
        "input": input_root.resolve(),
        "output": output_root.resolve(),
        "metrics": metrics_root.resolve(),
    }
    names = list(isolated)
    for index, name in enumerate(names):
        for other_name in names[index + 1 :]:
            _expect(
                not _paths_overlap(isolated[name], isolated[other_name]),
                f"Isolated {name} and {other_name} roots overlap",
            )
    protected = {
        "production input": source_root.resolve(),
        "model": model_root.resolve(),
    }
    for isolated_name, isolated_path in isolated.items():
        for protected_name, protected_path in protected.items():
            _expect(
                not _paths_overlap(isolated_path, protected_path),
                f"Isolated {isolated_name} root overlaps {protected_name} root",
            )


def _protect_artifact_path(path: Path, config: AppConfig, description: str) -> None:
    for root_name, root in {
        "production input": config.paths.input,
        "model": config.paths.models,
    }.items():
        _expect(
            not _paths_overlap(path, root),
            f"{description} overlaps {root_name} root",
        )


def _validate_base_semantics(config: AppConfig) -> None:
    processing_expected = {
        "profile": "real-hat-auto",
        "target_short_edge": 1600,
        "real_hat_sharper_min_short_edge": 1000,
        "max_long_edge_for_sr": 3200,
        "max_upscale_factor": 4,
        "max_output_long_edge": 6400,
        "max_output_megapixels": 24.0,
        "precision": "bf16",
        "batch_tiles": 1,
        "device_assembly": True,
        "model_cache_size": 2,
        "grayscale_tolerance": 3,
        "linear_light_downscale": True,
        "hat_tile": 256,
        "hat_tile_candidates": (256, 320),
        "hat_overlap": 16,
    }
    jxl_expected = {
        "distance": 0.5,
        "effort": 7,
        "threads": 4,
        "workers": 1,
        "queue_depth": 2,
        "verify_decode": True,
    }
    mismatches: list[str] = []
    for name, expected in processing_expected.items():
        actual = getattr(config.processing, name)
        if actual != expected:
            mismatches.append(f"processing.{name}={actual!r}, expected {expected!r}")
    for name, expected in jxl_expected.items():
        actual = getattr(config.jxl, name)
        if actual != expected:
            mismatches.append(f"jxl.{name}={actual!r}, expected {expected!r}")
    if mismatches:
        raise ValueError("Base production semantics drifted: " + "; ".join(mismatches))


def _model_manifest_path(base_config_path: Path, requested: Path | None) -> Path:
    return (
        requested.expanduser().resolve()
        if requested is not None
        else (base_config_path.resolve().parent / "model_sources.toml").resolve()
    )


def _model_inventory(
    config: AppConfig, model_manifest_path: Path
) -> dict[str, dict[str, Any]]:
    _assert_regular_file(model_manifest_path, "Model manifest")
    with model_manifest_path.open("rb") as handle:
        manifest = tomllib.load(handle)
    entries = manifest.get("models")
    if not isinstance(entries, dict):
        raise ValueError("Model manifest has no [models] table")

    inventory: dict[str, dict[str, Any]] = {}
    for route, relative_name in REAL_HAT_FILENAMES.items():
        matches = [
            value
            for value in entries.values()
            if isinstance(value, dict) and value.get("filename") == relative_name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Model manifest must contain exactly one {relative_name} entry"
            )
        expected_sha256 = matches[0].get("sha256")
        if not _is_sha256(expected_sha256):
            raise ValueError(
                f"Model manifest has an invalid SHA-256 for {relative_name}"
            )
        path = (config.paths.models / Path(relative_name)).resolve()
        _assert_regular_file(path, f"{route} checkpoint")
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Checkpoint hash mismatch for {path}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        inventory[route] = {
            "filename": Path(relative_name).name,
            "relative_path": relative_name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual_sha256,
        }
    return inventory


def _input_file_set(input_root: Path) -> set[str]:
    files: set[str] = set()
    for path in input_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Isolated input contains a symlink: {path}")
        if path.is_file():
            files.add(path.relative_to(input_root).as_posix())
    return files


def _manifest_pages(
    manifest_path: Path,
    input_root: Path,
    config: AppConfig,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, int]]:
    _assert_regular_file(manifest_path, "Representative manifest")
    _assert_directory(input_root, "Representative input")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != MANIFEST_KIND:
        raise ValueError("Unsupported representative manifest")
    if manifest.get("source_is_read_only") is not True:
        raise ValueError("Representative manifest does not declare a read-only source")
    source_root = manifest.get("source_root")
    if (
        not isinstance(source_root, str)
        or Path(source_root).resolve() != config.paths.input
    ):
        raise ValueError(
            "Representative manifest source root differs from the base config"
        )
    discovery = manifest.get("discovery")
    if not isinstance(discovery, dict) or discovery.get("errors") != 0:
        raise ValueError("Representative manifest contains discovery errors")
    raw_pages = manifest.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise ValueError("Representative manifest has no pages")

    expected: dict[str, dict[str, Any]] = {}
    indexes: set[int] = set()
    routes: Counter[str] = Counter()
    exact_threshold = 0
    grayscale_count = 0
    color_count = 0
    odd_count = 0
    pixels: list[int] = []
    for raw in raw_pages:
        if not isinstance(raw, dict):
            raise ValueError("Representative page must be a JSON object")
        index = raw.get("index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 1
            or index in indexes
        ):
            raise ValueError(f"Invalid representative page index: {index!r}")
        indexes.add(index)
        copied_path = raw.get("copied_path")
        copied_sha256 = raw.get("copied_sha256")
        if not isinstance(copied_path, str) or not _is_sha256(copied_sha256):
            raise ValueError(f"Representative page {index} has no copied path/hash")
        relative_declared = Path(copied_path)
        if relative_declared.is_absolute() or ".." in relative_declared.parts:
            raise ValueError(f"Unsafe copied path in representative page {index}")
        copied = (manifest_path.parent / relative_declared).resolve()
        key = _safe_relative(copied, input_root, "Representative copied input")
        if key in expected:
            raise ValueError(f"Duplicate representative copied input: {key}")
        if copied.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported representative input extension: {copied}")
        _assert_regular_file(copied, "Representative copied input")
        actual_sha256 = _sha256(copied)
        if actual_sha256 != copied_sha256 or raw.get("source_sha256") != copied_sha256:
            raise ValueError(f"Representative copied input hash mismatch: {copied}")
        if raw.get("file_bytes") != copied.stat().st_size:
            raise ValueError(f"Representative copied input size mismatch: {copied}")

        with Image.open(copied) as opened:
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError(
                    f"Animated representative input is unsupported: {copied}"
                )
            image = ImageOps.exif_transpose(opened)
            image.load()
        width, height = image.size
        short_edge = min(width, height)
        long_edge = max(width, height)
        gray = is_grayscale(image, config.processing.grayscale_tolerance)
        odd = bool(width % 2 or height % 2)
        route = (
            "normal"
            if short_edge < config.processing.real_hat_sharper_min_short_edge
            else "sharper"
        )
        decoded = {
            "width": width,
            "height": height,
            "short_edge": short_edge,
            "long_edge": long_edge,
            "pixels": width * height,
            "route": route,
            "grayscale": gray,
            "odd_dimension": odd,
        }
        for name, actual in decoded.items():
            if raw.get(name) != actual:
                raise ValueError(
                    f"Representative fact drift for page {index} {name}: "
                    f"{raw.get(name)!r} != {actual!r}"
                )
        plan = plan_resolution(
            width,
            height,
            config.processing.target_short_edge,
            config.processing.max_long_edge_for_sr,
            (4,),
            config.processing.max_upscale_factor,
            config.processing.max_output_long_edge,
            config.processing.max_output_megapixels,
        )
        if not plan.upscale or plan.native_scale != 4:
            raise ValueError(
                f"Representative page is not a Real-HAT x4 SR page: {copied}"
            )
        destination = Path(key).with_suffix(".jxl").as_posix()
        expected[key] = {
            **decoded,
            "index": index,
            "path": str(copied),
            "bytes": copied.stat().st_size,
            "sha256": actual_sha256,
            "destination": destination,
            "output_width": plan.output_width,
            "output_height": plan.output_height,
            "plan_reason": plan.reason,
            "target_unmet": "remains below target" in plan.reason,
        }
        routes[route] += 1
        exact_threshold += int(
            short_edge == config.processing.real_hat_sharper_min_short_edge
        )
        grayscale_count += int(gray)
        color_count += int(not gray)
        odd_count += int(odd)
        pixels.append(width * height)

    actual_files = _input_file_set(input_root)
    if actual_files != set(expected):
        missing = sorted(set(expected) - actual_files)
        extra = sorted(actual_files - set(expected))
        raise ValueError(
            f"Representative input file set differs from manifest; missing={missing}, extra={extra}"
        )
    route_counts = dict(sorted(routes.items()))
    if manifest.get("selected_counts") != route_counts:
        raise ValueError("Representative selected_counts differs from decoded routes")
    coverage = manifest.get("coverage")
    expected_coverage = {
        "exact_threshold": exact_threshold,
        "grayscale": grayscale_count,
        "rgb_or_color": color_count,
        "odd_dimension": odd_count,
        "minimum_selected_pixels": min(pixels),
        "maximum_selected_pixels": max(pixels),
    }
    if not isinstance(coverage, dict) or any(
        coverage.get(name) != value for name, value in expected_coverage.items()
    ):
        raise ValueError("Representative coverage differs from decoded input facts")
    return manifest, expected, route_counts


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return f"[{', '.join(_toml_literal(item) for item in value)}]"
    raise TypeError(f"Unsupported TOML value: {value!r}")


def _render_config_text(
    base: AppConfig, input_root: Path, output_root: Path
) -> tuple[str, Any]:
    mirror_output = replace(
        base.output,
        mode="mirror",
        format="jxl",
        copy_non_images=False,
        overwrite=False,
        existing_jxl_policy="error",
        allow_lossy_replace=False,
        allow_metadata_loss=False,
        allow_alpha_flatten=False,
        allow_bit_depth_loss=False,
    )
    sections = [
        (
            "paths",
            {
                "input": str(input_root.resolve()),
                "output": str(output_root.resolve()),
                "models": str(base.paths.models.resolve()),
            },
        ),
        ("processing", asdict(base.processing)),
        ("output", asdict(mirror_output)),
        ("jxl", asdict(base.jxl)),
    ]
    lines: list[str] = []
    for section, values in sections:
        lines.append(f"[{section}]")
        lines.extend(
            f"{name} = {_toml_literal(value)}" for name, value in values.items()
        )
        lines.append("")
    return "\n".join(lines), mirror_output


def render_isolated_config(
    *,
    base_config_path: Path,
    manifest_path: Path,
    output_root: Path,
    metrics_root: Path,
    output_path: Path,
    model_manifest_path: Path | None = None,
) -> dict[str, Any]:
    base_config_path = base_config_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    metrics_root = metrics_root.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    input_root = (manifest_path.parent / "inputs").resolve()
    base = load_config(base_config_path)
    _protect_artifact_path(output_path, base, "Rendered config path")
    _validate_base_semantics(base)
    _validate_isolated_roots(
        source_root=base.paths.input,
        model_root=base.paths.models,
        input_root=input_root,
        output_root=output_root,
        metrics_root=metrics_root,
    )
    for root, name in ((output_root, "Output root"), (metrics_root, "Metrics root")):
        _assert_fresh_path(root, name)
    _assert_fresh_path(output_path, "Rendered config")
    for root_name, root in {
        "input": input_root,
        "output": output_root,
        "metrics": metrics_root,
    }.items():
        _expect(
            not _paths_overlap(output_path, root),
            f"Rendered config path overlaps isolated {root_name} root",
        )
    manifest, pages, routes = _manifest_pages(manifest_path, input_root, base)
    resolved_model_manifest = _model_manifest_path(
        base_config_path, model_manifest_path
    )
    models = _model_inventory(base, resolved_model_manifest)
    rendered, expected_output = _render_config_text(base, input_root, output_root)
    _write_exclusive(output_path, rendered)
    isolated = load_config(output_path)
    if isolated.processing != base.processing or isolated.jxl != base.jxl:
        raise RuntimeError("Rendered config changed processing or JXL semantics")
    if isolated.output != expected_output:
        raise RuntimeError("Rendered config has incorrect mirror output semantics")
    if (
        isolated.paths.input != input_root
        or isolated.paths.output != output_root
        or isolated.paths.models != base.paths.models
    ):
        raise RuntimeError("Rendered config has incorrect processing roots")
    return {
        "config": str(output_path),
        "config_sha256": _sha256(output_path),
        "manifest_sha256": _sha256(manifest_path),
        "page_count": len(pages),
        "route_counts": routes,
        "hat_overlap": isolated.processing.hat_overlap,
        "models": models,
        "source_manifest": {
            "kind": manifest["kind"],
            "source_root": manifest["source_root"],
        },
    }


def _load_isolated_config(
    *,
    base_config_path: Path,
    isolated_config_path: Path,
    manifest_path: Path,
    metrics_root: Path,
) -> tuple[AppConfig, AppConfig]:
    base = load_config(base_config_path)
    isolated = load_config(isolated_config_path)
    _validate_base_semantics(base)
    input_root = (manifest_path.parent / "inputs").resolve()
    if isolated.processing != base.processing or isolated.jxl != base.jxl:
        raise ValueError("Isolated config drifted from base processing/JXL semantics")
    expected_output = replace(
        base.output,
        mode="mirror",
        format="jxl",
        copy_non_images=False,
        overwrite=False,
        existing_jxl_policy="error",
        allow_lossy_replace=False,
        allow_metadata_loss=False,
        allow_alpha_flatten=False,
        allow_bit_depth_loss=False,
    )
    if isolated.output != expected_output:
        raise ValueError("Isolated config does not have required mirror semantics")
    if isolated.paths.input != input_root or isolated.paths.models != base.paths.models:
        raise ValueError("Isolated config input/model roots differ from expected roots")
    _validate_isolated_roots(
        source_root=base.paths.input,
        model_root=base.paths.models,
        input_root=isolated.paths.input,
        output_root=isolated.paths.output,
        metrics_root=metrics_root,
    )
    return base, isolated


def _metrics_artifacts(
    metrics_root: Path,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    _assert_directory(metrics_root, "Metrics root")
    if any(path.is_symlink() for path in metrics_root.rglob("*")):
        raise ValueError("Metrics root contains a symlink")
    job_paths = sorted(metrics_root.rglob("job.json"))
    if len(job_paths) != 1:
        raise ValueError(
            f"Metrics root must contain exactly one job.json, found {len(job_paths)}"
        )
    run_root = job_paths[0].parent.resolve()
    _expect(
        run_root.parent == metrics_root.resolve(), "Metrics run must be a direct child"
    )
    pages_path = run_root / "pages.jsonl"
    _assert_regular_file(job_paths[0], "Job metrics")
    _assert_regular_file(pages_path, "Page metrics")
    job = _read_json(job_paths[0])
    pages: list[dict[str, Any]] = []
    with pages_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank page metrics line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Page metrics line {line_number} is not an object")
            pages.append(value)
    return run_root, job, pages


def _validate_job(
    *,
    job: dict[str, Any],
    run_root: Path,
    config: AppConfig,
    expected_pages: int,
    expected_target_unmet: int,
    run_kind: str,
    expected_pipeline_signature: str,
) -> str:
    if (
        job.get("type") != JOB_METRICS_TYPE
        or job.get("schema_version") != 1
        or job.get("status") != "complete"
    ):
        raise ValueError("Job metrics is not a complete schema-1 report")
    run_id = job.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Job metrics has no run_id")
    write_errors = job.get("write_errors")
    if (
        job.get("pages_written") != expected_pages
        or not isinstance(write_errors, list)
        or write_errors
    ):
        raise ValueError("Job metrics page/write counts are invalid")
    context = job.get("context")
    if not isinstance(context, dict):
        raise ValueError("Job metrics has no context")
    expected_context = {
        "output_mode": "mirror",
        "output_format": "jxl",
        "processing_profile": "real-hat-auto",
    }
    for name, expected in expected_context.items():
        actual = context.get(name)
        if str(actual).lower() != expected:
            raise ValueError(f"Job context {name} differs from isolated config")
    if (
        Path(str(context.get("input_root", ""))).resolve() != config.paths.input
        or Path(str(context.get("output_root", ""))).resolve() != config.paths.output
    ):
        raise ValueError("Job context processing roots differ from isolated config")
    pipeline_signature = context.get("pipeline_signature")
    if pipeline_signature != expected_pipeline_signature:
        raise ValueError("Job context pipeline signature differs from isolated config")
    summary = job.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("Job metrics has no summary")
    first_run = run_kind == "first"
    exact_values = {
        "processed": expected_pages if first_run else 0,
        "skipped": 0 if first_run else expected_pages,
        "copied": 0,
        "ignored": 0,
        "jxl_skipped": 0,
        "failed": 0,
        "sr_pages": expected_pages if first_run else 0,
        "transcoded_pages": 0,
        "replaced_sources": 0,
        "existing_jxl_adopted": 0,
        "existing_jxl_replaced": 0,
        "external_jxl_recoveries": 0,
        "deferred": 0,
        "target_unmet": expected_target_unmet if first_run else 0,
        "metrics_write_errors": 0,
    }
    for name, expected in exact_values.items():
        if summary.get(name) != expected:
            raise ValueError(
                f"Job summary {name}={summary.get(name)!r}, expected {expected!r}"
            )
    if Path(str(summary.get("metrics_directory", ""))).resolve() != run_root:
        raise ValueError("Job summary metrics_directory differs from metrics run")
    _validate_job_timing(job, expected_pages=expected_pages, run_kind=run_kind)
    return expected_pipeline_signature


def _spans(page: dict[str, Any]) -> dict[str, Any]:
    timing = page.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("Page metrics has no timing object")
    spans = timing.get("spans")
    if not isinstance(spans, dict):
        raise ValueError("Page metrics has no timing spans")
    return spans


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _validate_span_map(
    spans: dict[str, Any], required: set[str], description: str
) -> None:
    missing = sorted(required - set(spans))
    if missing:
        raise ValueError(f"{description} is missing required spans: {missing}")
    for name, raw_intervals in spans.items():
        if not isinstance(raw_intervals, list) or not raw_intervals:
            raise ValueError(f"{description} span {name} must be a non-empty list")
        for interval in raw_intervals:
            if not isinstance(interval, dict):
                raise ValueError(f"{description} span {name} has a non-object interval")
            start = interval.get("start_offset_ns")
            end = interval.get("end_offset_ns")
            duration = interval.get("duration_seconds")
            clock = interval.get("clock")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or start < 0
                or not isinstance(end, int)
                or isinstance(end, bool)
                or end < start
                or not _finite_nonnegative(duration)
                or not isinstance(clock, str)
                or not clock
            ):
                raise ValueError(f"{description} span {name} has an invalid interval")
            measured = (end - start) / 1_000_000_000
            tolerance = max(1e-9, measured * 1e-6)
            if abs(float(duration) - measured) > tolerance:
                raise ValueError(
                    f"{description} span {name} duration does not match offsets"
                )


def _validate_job_timing(
    job: dict[str, Any], *, expected_pages: int, run_kind: str
) -> None:
    wall_seconds = job.get("wall_seconds")
    if not _finite_nonnegative(wall_seconds) or float(wall_seconds) == 0:
        raise ValueError("Job wall_seconds must be finite and positive")
    summary = job.get("summary")
    if not isinstance(summary, dict) or summary.get("wall_seconds") != wall_seconds:
        raise ValueError("Job and summary wall_seconds differ")
    timing = job.get("timing")
    if not isinstance(timing, dict) or timing.get("clock") != "perf_counter_ns":
        raise ValueError("Job timing has an invalid clock")
    stage_spans = timing.get("stage_spans")
    if not isinstance(stage_spans, dict):
        raise ValueError("Job timing has no stage spans")
    _validate_span_map(
        stage_spans,
        {"recovery", "discovery", "preflight", "worklist"},
        "Job timing",
    )

    services = timing.get("cumulative_service_seconds")
    interval_summary = timing.get("interval_summary")
    if not isinstance(services, dict) or not isinstance(interval_summary, dict):
        raise ValueError("Job timing service or interval summary is missing")
    for name, seconds in services.items():
        if not _finite_nonnegative(seconds):
            raise ValueError(f"Job timing service {name} is invalid")

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
    required_intervals = common_intervals | (
        first_intervals if run_kind == "first" else set()
    )
    missing = sorted(required_intervals - set(interval_summary))
    if missing:
        raise ValueError(f"Job timing interval summary is incomplete: {missing}")
    for name in required_intervals:
        record = interval_summary[name]
        if not isinstance(record, dict) or record.get("count") != expected_pages:
            raise ValueError(f"Job timing interval count is invalid for {name}")
        cumulative = record.get("cumulative_seconds")
        union = record.get("union_seconds")
        percent = record.get("critical_path_percent")
        if not all(
            _finite_nonnegative(value) for value in (cumulative, union, percent)
        ):
            raise ValueError(f"Job timing interval values are invalid for {name}")
        if (
            float(union) > float(cumulative) + 1e-9
            or float(union) > float(wall_seconds) + 1e-9
            or float(percent) > 100 + 1e-6
        ):
            raise ValueError(
                f"Job timing interval relationships are invalid for {name}"
            )

    first_services = {
        "gpu_synchronized_inference",
        "forward",
        "cjxl",
        "djxl",
        "candidate_hash",
    }
    if run_kind == "first":
        missing_services = sorted(first_services - set(services))
        if missing_services:
            raise ValueError(f"Job timing services are incomplete: {missing_services}")
    else:
        unexpected = sorted(
            (first_intervals & set(interval_summary)) | (first_services & set(services))
        )
        if unexpected:
            raise ValueError(
                f"Incremental job unexpectedly contains GPU/JXL timing: {unexpected}"
            )


def _validate_pages(
    *,
    pages: list[dict[str, Any]],
    expected: dict[str, dict[str, Any]],
    job_run_id: str,
    config: AppConfig,
    model_inventory: dict[str, dict[str, Any]],
    run_kind: str,
    max_reserved_vram_bytes: int,
) -> dict[str, Any]:
    if len(pages) != len(expected):
        raise ValueError(f"Page metrics count {len(pages)} != {len(expected)}")
    observed: set[str] = set()
    indexes: set[int] = set()
    routes: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    selected_tiles: Counter[int] = Counter()
    peak_reserved = 0
    common_spans = {"read", "decode_exif", "analyze_plan", "hash_and_fingerprint"}
    gpu_jxl_spans = {
        "engine_path",
        "gpu_inference",
        "jxl_service",
        "cjxl",
        "djxl",
        "candidate_hash",
        "commit",
    }
    required_first_spans = common_spans | gpu_jxl_spans
    prohibited_incremental_spans = gpu_jxl_spans | {"engine_setup", "model_load"}
    for page in pages:
        if (
            page.get("type") != PAGE_METRICS_TYPE
            or page.get("schema_version") != 1
            or page.get("run_id") != job_run_id
        ):
            raise ValueError("Page metrics type/schema/run_id mismatch")
        source = page.get("source")
        if not isinstance(source, str) or source not in expected or source in observed:
            raise ValueError(f"Unexpected or duplicate page source: {source!r}")
        observed.add(source)
        index = page.get("index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index in indexes
            or not 1 <= index <= len(expected)
        ):
            raise ValueError(f"Invalid page metrics index for {source}")
        indexes.add(index)
        if page.get("total") != len(expected):
            raise ValueError(f"Page metrics total mismatch for {source}")
        status = page.get("status")
        expected_status = "complete" if run_kind == "first" else "skipped"
        if status != expected_status:
            raise ValueError(
                f"Page status {status!r} != {expected_status!r} for {source}"
            )
        if page.get("destination") != expected[source]["destination"]:
            raise ValueError(f"Page destination mismatch for {source}")
        details = page.get("details")
        if not isinstance(details, dict):
            raise ValueError(f"Page details are missing for {source}")
        route = expected[source]["route"]
        detail_values = {
            "source_extension": Path(source).suffix.lower(),
            "source_bytes": expected[source]["bytes"],
            "source_dimensions": [
                expected[source]["width"],
                expected[source]["height"],
            ],
            "source_short_edge": expected[source]["short_edge"],
            "grayscale": expected[source]["grayscale"],
            "upscale": True,
            "native_scale": 4,
            "planned_output_dimensions": [
                expected[source]["output_width"],
                expected[source]["output_height"],
            ],
            "plan_reason": expected[source]["plan_reason"],
            "model_label": REAL_HAT_LABELS[route],
            "model_checkpoint": model_inventory[route]["filename"],
        }
        for name, value in detail_values.items():
            if details.get(name) != value:
                raise ValueError(
                    f"Page detail {name}={details.get(name)!r}, expected {value!r} for {source}"
                )
        spans = _spans(page)
        if run_kind == "first":
            _validate_span_map(spans, required_first_spans, f"Page timing for {source}")
            first_values = {
                "precision": config.processing.precision,
                "batch_tiles": config.processing.batch_tiles,
                "tile_candidates": list(config.processing.hat_tile_candidates),
                "overlap": config.processing.hat_overlap,
            }
            for name, value in first_values.items():
                if details.get(name) != value:
                    raise ValueError(
                        f"First-run page detail {name} mismatch for {source}"
                    )
            tile = details.get("tile")
            if tile not in config.processing.hat_tile_candidates:
                raise ValueError(f"Selected tile is not configured for {source}")
            selected_tiles[int(tile)] += 1
            reserved = details.get("peak_reserved_vram_bytes")
            if (
                not isinstance(reserved, int)
                or isinstance(reserved, bool)
                or reserved < 1
            ):
                raise ValueError(f"Page has no peak reserved VRAM evidence: {source}")
            if reserved > max_reserved_vram_bytes:
                raise ValueError(f"Page exceeds reserved VRAM limit: {source}")
            peak_reserved = max(peak_reserved, reserved)
        else:
            _validate_span_map(spans, common_spans, f"Page timing for {source}")
            present = sorted(prohibited_incremental_spans & set(spans))
            if present:
                raise ValueError(
                    f"Incremental page unexpectedly contains GPU/JXL spans for {source}: {present}"
                )
            forbidden_details = {
                "precision",
                "tile",
                "tile_candidates",
                "overlap",
                "peak_reserved_vram_bytes",
            }
            present_details = sorted(forbidden_details & set(details))
            if present_details:
                raise ValueError(
                    f"Incremental page unexpectedly contains engine details for {source}: "
                    f"{present_details}"
                )
        routes[route] += 1
        statuses[str(status)] += 1
    if observed != set(expected):
        raise ValueError("Page metrics source set differs from representative inputs")
    if indexes != set(range(1, len(expected) + 1)):
        raise ValueError("Page metrics indexes are not a complete runtime sequence")
    return {
        "route_counts": dict(sorted(routes.items())),
        "status_counts": dict(sorted(statuses.items())),
        "selected_tile_counts": {
            str(tile): count for tile, count in sorted(selected_tiles.items())
        },
        "peak_reserved_vram_bytes": peak_reserved if run_kind == "first" else None,
    }


def _output_inventory(
    output_root: Path,
    expected: dict[str, dict[str, Any]],
    decoder: Decoder,
) -> tuple[list[dict[str, Any]], int]:
    _assert_directory(output_root, "Mirror output")
    worklist = output_root / WORKLIST_NAME
    if worklist.exists() or worklist.is_symlink():
        raise ValueError("Completed mirror run retained its worklist")
    expected_jxl = {fact["destination"] for fact in expected.values()}
    allowed_non_jxl = {STATE_NAME, LOCK_NAME}
    actual_jxl: set[str] = set()
    output_bytes = 0
    for path in output_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Mirror output contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(output_root).as_posix()
        if path.name.endswith(".part") or ".part" in path.name:
            raise ValueError(f"Completed mirror run retained a part file: {path}")
        if path.suffix.lower() == ".jxl":
            actual_jxl.add(relative)
        elif relative not in allowed_non_jxl:
            raise ValueError(f"Unexpected non-JXL mirror output: {relative}")
    if actual_jxl != expected_jxl:
        missing = sorted(expected_jxl - actual_jxl)
        extra = sorted(actual_jxl - expected_jxl)
        raise ValueError(f"JXL output set mismatch; missing={missing}, extra={extra}")

    by_destination = {fact["destination"]: fact for fact in expected.values()}
    inventory: list[dict[str, Any]] = []
    for relative in sorted(actual_jxl):
        path = output_root / Path(relative)
        _assert_regular_file(path, "JXL output")
        fact = by_destination[relative]
        dimensions = decoder(path, fact["output_width"], fact["output_height"])
        if dimensions != (fact["output_width"], fact["output_height"]):
            raise ValueError(f"Final JXL decoder returned wrong dimensions for {path}")
        size = path.stat().st_size
        output_bytes += size
        inventory.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": _sha256(path),
                "decoded_dimensions": list(dimensions),
            }
        )
    return inventory, output_bytes


def _validate_state(
    *,
    output_root: Path,
    input_root: Path,
    expected: dict[str, dict[str, Any]],
    output_inventory: list[dict[str, Any]],
    model_inventory: dict[str, dict[str, Any]],
    pipeline_signature: str,
) -> dict[str, Any]:
    state_path = output_root / STATE_NAME
    _assert_regular_file(state_path, "Mirror state")
    state = _read_json(state_path)
    if set(state) != set(expected):
        raise ValueError("Mirror state source set differs from representative inputs")
    outputs = {item["path"]: item for item in output_inventory}
    for key, fact in expected.items():
        record = state.get(key)
        if not isinstance(record, dict):
            raise ValueError(f"Mirror state record is invalid for {key}")
        source_path = input_root / Path(key)
        source_stat = source_path.stat()
        route = fact["route"]
        values = {
            "size": fact["bytes"],
            "mtime_ns": source_stat.st_mtime_ns,
            "source_sha256": fact["sha256"],
            "source_root": str(input_root.resolve()),
            "model_sha256": model_inventory[route]["sha256"],
            "pipeline_signature": pipeline_signature,
            "phase": "committed",
            "destination": fact["destination"],
            "output_size": outputs[fact["destination"]]["bytes"],
            "output_sha256": outputs[fact["destination"]]["sha256"],
        }
        for name, expected_value in values.items():
            if record.get(name) != expected_value:
                raise ValueError(
                    f"Mirror state {name} mismatch for {key}: "
                    f"{record.get(name)!r} != {expected_value!r}"
                )
    return {
        "path": str(state_path),
        "bytes": state_path.stat().st_size,
        "sha256": _sha256(state_path),
        "records": len(state),
    }


def _input_inventory(expected: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"path": key, "bytes": fact["bytes"], "sha256": fact["sha256"]}
        for key, fact in sorted(expected.items())
    ]


def _validate_baseline(
    *,
    baseline_path: Path,
    isolated_config_path: Path,
    manifest_path: Path,
    input_inventory: list[dict[str, Any]],
    output_inventory: list[dict[str, Any]],
    state_report: dict[str, Any],
    model_inventory: dict[str, dict[str, Any]],
    expected_gate: dict[str, Any],
) -> dict[str, Any]:
    _assert_regular_file(baseline_path, "Baseline attestation")
    baseline = _read_json(baseline_path)
    if (
        baseline.get("schema_version") != REPORT_SCHEMA_VERSION
        or baseline.get("kind") != REPORT_KIND
        or baseline.get("valid") is not True
        or baseline.get("run_kind") != "first"
    ):
        raise ValueError("Incremental baseline is not a valid first-run attestation")
    files = baseline.get("files")
    if not isinstance(files, dict):
        raise ValueError("Incremental baseline has no file bindings")
    if files.get("config_sha256") != _sha256(isolated_config_path) or files.get(
        "manifest_sha256"
    ) != _sha256(manifest_path):
        raise ValueError("Incremental config/manifest differs from first-run baseline")
    if baseline.get("expected") != expected_gate:
        raise ValueError("Incremental baseline used different acceptance thresholds")
    required_checks = {
        "roots_are_isolated",
        "base_semantics_inherited",
        "manifest_inputs_redecoded_and_unchanged",
        "metrics_complete_and_bound",
        "routes_models_plans_and_native_scale_bound",
        "reserved_vram_within_limit",
        "final_jxl_redecoded",
        "mirror_state_committed_and_hash_bound",
        "worklist_and_part_files_absent",
    }
    checks = baseline.get("checks")
    if not isinstance(checks, dict) or any(
        checks.get(name) is not True for name in required_checks
    ):
        raise ValueError("Incremental baseline does not prove all first-run checks")
    page_evidence = baseline.get("page_evidence")
    if not isinstance(page_evidence, dict):
        raise ValueError("Incremental baseline has no first-run page evidence")
    expected_pages = expected_gate["pages"]
    peak_reserved = page_evidence.get("peak_reserved_vram_bytes")
    selected_tiles = page_evidence.get("selected_tile_counts")
    if (
        page_evidence.get("route_counts") != expected_gate["routes"]
        or page_evidence.get("status_counts") != {"complete": expected_pages}
        or not isinstance(selected_tiles, dict)
        or sum(selected_tiles.values()) != expected_pages
        or not isinstance(peak_reserved, int)
        or isinstance(peak_reserved, bool)
        or not 0 < peak_reserved <= expected_gate["max_reserved_vram_bytes"]
    ):
        raise ValueError("Incremental baseline first-run page evidence is invalid")
    first_summary = baseline.get("job_summary")
    if not isinstance(first_summary, dict) or any(
        first_summary.get(name) != value
        for name, value in {
            "processed": expected_pages,
            "skipped": 0,
            "failed": 0,
            "deferred": 0,
            "metrics_write_errors": 0,
        }.items()
    ):
        raise ValueError("Incremental baseline first-run job summary is invalid")
    comparisons = {
        "input inventory": (baseline.get("input_inventory"), input_inventory),
        "JXL inventory": (baseline.get("output_inventory"), output_inventory),
        "model inventory": (baseline.get("models"), model_inventory),
    }
    for description, (before, after) in comparisons.items():
        if before != after:
            raise ValueError(f"Incremental {description} changed")
    baseline_state = baseline.get("state")
    if (
        not isinstance(baseline_state, dict)
        or baseline_state.get("sha256") != state_report["sha256"]
    ):
        raise ValueError("Incremental mirror state bytes changed")
    return {"path": str(baseline_path), "sha256": _sha256(baseline_path)}


def verify_soak(
    *,
    base_config_path: Path,
    isolated_config_path: Path,
    manifest_path: Path,
    metrics_root: Path,
    output_path: Path,
    expected_pages: int,
    expected_normal: int,
    expected_sharper: int,
    max_reserved_vram_gib: float,
    run_kind: str = "first",
    baseline_path: Path | None = None,
    model_manifest_path: Path | None = None,
    min_exact_threshold: int = 2,
    decoder: Decoder | None = None,
) -> dict[str, Any]:
    if run_kind not in {"first", "incremental"}:
        raise ValueError("run_kind must be first or incremental")
    if expected_pages < 1 or expected_normal < 1 or expected_sharper < 1:
        raise ValueError("Expected page and route counts must be positive")
    if expected_normal + expected_sharper != expected_pages:
        raise ValueError("Expected route counts must sum to expected pages")
    if (
        not math.isfinite(max_reserved_vram_gib)
        or max_reserved_vram_gib <= 0
        or min_exact_threshold < 0
    ):
        raise ValueError(
            "VRAM limit must be positive and threshold coverage non-negative"
        )
    if run_kind == "incremental" and baseline_path is None:
        raise ValueError("Incremental verification requires a baseline attestation")
    if run_kind == "first" and baseline_path is not None:
        raise ValueError(
            "First-run verification must not receive a baseline attestation"
        )

    base_config_path = base_config_path.expanduser().resolve()
    isolated_config_path = isolated_config_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    metrics_root = metrics_root.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    base, isolated = _load_isolated_config(
        base_config_path=base_config_path,
        isolated_config_path=isolated_config_path,
        manifest_path=manifest_path,
        metrics_root=metrics_root,
    )
    _protect_artifact_path(output_path, base, "Soak attestation path")
    _assert_fresh_path(output_path, "Soak attestation")
    for root_name, root in {
        "input": isolated.paths.input,
        "output": isolated.paths.output,
        "metrics": metrics_root,
    }.items():
        _expect(
            not _paths_overlap(output_path, root),
            f"Attestation path overlaps isolated {root_name} root",
        )
    manifest, expected, routes = _manifest_pages(
        manifest_path, isolated.paths.input, base
    )
    required_routes = {"normal": expected_normal, "sharper": expected_sharper}
    max_reserved_vram_bytes = int(max_reserved_vram_gib * 1024**3)
    expected_gate = {
        "pages": expected_pages,
        "routes": required_routes,
        "minimum_exact_threshold": min_exact_threshold,
        "max_reserved_vram_bytes": max_reserved_vram_bytes,
    }
    if len(expected) != expected_pages or routes != required_routes:
        raise ValueError(
            f"Representative count/routes mismatch: pages={len(expected)}, routes={routes}"
        )
    coverage = manifest.get("coverage", {})
    if int(coverage.get("exact_threshold", -1)) < min_exact_threshold:
        raise ValueError("Representative set has insufficient exact-threshold coverage")
    resolved_model_manifest = _model_manifest_path(
        base_config_path, model_manifest_path
    )
    models = _model_inventory(isolated, resolved_model_manifest)
    run_root, job, pages = _metrics_artifacts(metrics_root)
    expected_target_unmet = sum(fact["target_unmet"] for fact in expected.values())
    from waifuhat2x.pipeline import _pipeline_signature

    current_pipeline_signature = _pipeline_signature(isolated)
    pipeline_signature = _validate_job(
        job=job,
        run_root=run_root,
        config=isolated,
        expected_pages=expected_pages,
        expected_target_unmet=expected_target_unmet,
        run_kind=run_kind,
        expected_pipeline_signature=current_pipeline_signature,
    )
    page_report = _validate_pages(
        pages=pages,
        expected=expected,
        job_run_id=str(job["run_id"]),
        config=isolated,
        model_inventory=models,
        run_kind=run_kind,
        max_reserved_vram_bytes=max_reserved_vram_bytes,
    )
    if page_report["route_counts"] != required_routes:
        raise ValueError("Page telemetry route counts differ from expected routes")

    if decoder is None:
        from waifuhat2x.jxl import JxlEncoder

        encoder = JxlEncoder(isolated.jxl)

        def decoder(path: Path, width: int, height: int) -> tuple[int, int]:
            return encoder.verify(path, width, height)

    outputs, output_bytes = _output_inventory(isolated.paths.output, expected, decoder)
    summary = job["summary"]
    expected_output_bytes = output_bytes if run_kind == "first" else 0
    if summary.get("output_bytes") != expected_output_bytes:
        raise ValueError(
            f"Job output_bytes={summary.get('output_bytes')!r}, "
            f"expected {expected_output_bytes}"
        )
    inputs = _input_inventory(expected)
    state = _validate_state(
        output_root=isolated.paths.output,
        input_root=isolated.paths.input,
        expected=expected,
        output_inventory=outputs,
        model_inventory=models,
        pipeline_signature=pipeline_signature,
    )
    baseline = None
    if baseline_path is not None:
        baseline = _validate_baseline(
            baseline_path=baseline_path.expanduser().resolve(),
            isolated_config_path=isolated_config_path,
            manifest_path=manifest_path,
            input_inventory=inputs,
            output_inventory=outputs,
            state_report=state,
            model_inventory=models,
            expected_gate=expected_gate,
        )
    model_inventory_after = _model_inventory(isolated, resolved_model_manifest)
    if model_inventory_after != models:
        raise ValueError("Real-HAT checkpoints changed during verification")

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "valid": True,
        "run_kind": run_kind,
        "created_utc": _utc_now(),
        "expected": expected_gate,
        "roots": {
            "input": str(isolated.paths.input),
            "output": str(isolated.paths.output),
            "metrics": str(metrics_root),
            "production_input": str(base.paths.input),
        },
        "files": {
            "base_config": str(base_config_path),
            "base_config_sha256": _sha256(base_config_path),
            "config": str(isolated_config_path),
            "config_sha256": _sha256(isolated_config_path),
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "model_manifest": str(resolved_model_manifest),
            "model_manifest_sha256": _sha256(resolved_model_manifest),
            "job": str(run_root / "job.json"),
            "job_sha256": _sha256(run_root / "job.json"),
            "pages": str(run_root / "pages.jsonl"),
            "pages_sha256": _sha256(run_root / "pages.jsonl"),
        },
        "config_semantics": {
            "processing": asdict(isolated.processing),
            "output": asdict(isolated.output),
            "jxl": asdict(isolated.jxl),
        },
        "manifest": {
            "selected_counts": routes,
            "coverage": manifest["coverage"],
        },
        "models": models,
        "job_summary": summary,
        "page_evidence": page_report,
        "input_inventory": inputs,
        "output_inventory": outputs,
        "state": state,
        "baseline": baseline,
        "checks": {
            "roots_are_isolated": True,
            "base_semantics_inherited": True,
            "manifest_inputs_redecoded_and_unchanged": True,
            "metrics_complete_and_bound": True,
            "routes_models_plans_and_native_scale_bound": True,
            "reserved_vram_within_limit": True,
            "final_jxl_redecoded": True,
            "mirror_state_committed_and_hash_bound": True,
            "worklist_and_part_files_absent": True,
            "incremental_inventories_unchanged": run_kind == "incremental",
        },
    }
    _write_exclusive_json(output_path, report)
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render and attest an isolated production-pipeline soak run."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render-config")
    render.add_argument("--base-config", type=Path, default=Path("config.toml"))
    render.add_argument("--manifest", type=Path, required=True)
    render.add_argument("--output-root", type=Path, required=True)
    render.add_argument("--metrics-root", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--model-manifest", type=Path)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--base-config", type=Path, default=Path("config.toml"))
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--metrics-root", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--model-manifest", type=Path)
    verify.add_argument("--run-kind", choices=("first", "incremental"), default="first")
    verify.add_argument("--baseline", type=Path)
    verify.add_argument("--expected-pages", type=_positive_int, default=100)
    verify.add_argument("--expected-normal", type=_positive_int, default=30)
    verify.add_argument("--expected-sharper", type=_positive_int, default=70)
    verify.add_argument("--min-exact-threshold", type=_nonnegative_int, default=2)
    verify.add_argument("--max-reserved-vram-gib", type=float, default=14.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "render-config":
        result = render_isolated_config(
            base_config_path=args.base_config,
            manifest_path=args.manifest,
            output_root=args.output_root,
            metrics_root=args.metrics_root,
            output_path=args.output,
            model_manifest_path=args.model_manifest,
        )
    else:
        result = verify_soak(
            base_config_path=args.base_config,
            isolated_config_path=args.config,
            manifest_path=args.manifest,
            metrics_root=args.metrics_root,
            output_path=args.output,
            expected_pages=args.expected_pages,
            expected_normal=args.expected_normal,
            expected_sharper=args.expected_sharper,
            max_reserved_vram_gib=args.max_reserved_vram_gib,
            run_kind=args.run_kind,
            baseline_path=args.baseline,
            model_manifest_path=args.model_manifest,
            min_exact_threshold=args.min_exact_threshold,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
