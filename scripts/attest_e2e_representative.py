from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

from PIL import Image, ImageOps

from waifuhat2x.images import is_grayscale


SCHEMA_VERSION = 1
E2E_SCHEMA_VERSIONS = {3, 4}
E2E_SUMMARY_KIND = "real_hat_pipeline_e2e_benchmark"
E2E_RESULT_KIND = "real_hat_pipeline_e2e_child_result"
E2E_COMPLETION_KIND = "real_hat_pipeline_e2e_completion"
REQUIRED_ROUTES = {"normal": 9, "sharper": 21}
REQUIRED_PAGE_COUNT = 30


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def inspect_manifest(
    manifest_path: Path, threshold: int
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "real_hat_representative_manifest"
    ):
        raise ValueError("Unsupported representative manifest")
    inputs_root = (manifest_path.parent / "inputs").resolve()
    raw_pages = manifest.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise ValueError("Representative manifest has no pages")

    facts: dict[str, dict[str, Any]] = {}
    indexes: set[int] = set()
    route_counts: Counter[str] = Counter()
    exact_threshold = grayscale = color = odd_dimension = 0
    pixel_counts: list[int] = []
    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            raise ValueError("Representative manifest page must be an object")
        index = int(raw_page.get("index", 0))
        if index < 1 or index in indexes:
            raise ValueError(f"Invalid or duplicate page index: {index}")
        indexes.add(index)
        copied_path = raw_page.get("copied_path")
        copied_sha256 = raw_page.get("copied_sha256")
        if not isinstance(copied_path, str) or not isinstance(copied_sha256, str):
            raise ValueError(f"Page {index} has no copied path/hash")
        source = (manifest_path.parent / copied_path).resolve()
        if not is_relative_to(source, inputs_root) or not source.is_file():
            raise ValueError(f"Page {index} is not an isolated copied input: {source}")
        digest = sha256_file(source)
        if digest != copied_sha256:
            raise ValueError(f"Copied input hash changed for page {index}: {source}")
        with Image.open(source) as opened:
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError(
                    f"Representative page is animated/multi-frame: {source}"
                )
            image = ImageOps.exif_transpose(opened)
            image.load()
        width, height = image.size
        short_edge = min(width, height)
        pixels = width * height
        gray = is_grayscale(image)
        odd = bool(width % 2 or height % 2)
        route = "normal" if short_edge < threshold else "sharper"
        claimed = {
            "width": width,
            "height": height,
            "short_edge": short_edge,
            "long_edge": max(width, height),
            "pixels": pixels,
            "route": route,
            "grayscale": gray,
            "odd_dimension": odd,
        }
        for name, actual in claimed.items():
            if raw_page.get(name) != actual:
                raise ValueError(
                    f"Manifest fact drift for page {index} {name}: "
                    f"{raw_page.get(name)!r} != {actual!r}"
                )
        relative = source.relative_to(inputs_root).as_posix()
        if relative in facts:
            raise ValueError(f"Duplicate copied input path: {relative}")
        facts[relative] = {
            "index": index,
            "source_path": str(source),
            "source_sha256": digest,
            "source_bytes": source.stat().st_size,
            "width": width,
            "height": height,
            "short_edge": short_edge,
            "route": route,
            "grayscale": gray,
            "odd_dimension": odd,
        }
        route_counts[route] += 1
        exact_threshold += int(short_edge == threshold)
        grayscale += int(gray)
        color += int(not gray)
        odd_dimension += int(odd)
        pixel_counts.append(pixels)

    coverage = {
        "page_count": len(facts),
        "route_counts": dict(sorted(route_counts.items())),
        "exact_threshold": exact_threshold,
        "grayscale": grayscale,
        "rgb_or_color": color,
        "odd_dimension": odd_dimension,
        "minimum_selected_pixels": min(pixel_counts),
        "maximum_selected_pixels": max(pixel_counts),
    }
    if len(facts) != REQUIRED_PAGE_COUNT:
        raise ValueError(f"Expected {REQUIRED_PAGE_COUNT} pages, found {len(facts)}")
    if coverage["route_counts"] != REQUIRED_ROUTES:
        raise ValueError(
            f"Representative route coverage drifted: {coverage['route_counts']}"
        )
    if exact_threshold < 2 or not grayscale or not color or not odd_dimension:
        raise ValueError(f"Representative coverage is incomplete: {coverage}")
    manifest_coverage = manifest.get("coverage")
    if not isinstance(manifest_coverage, dict):
        raise ValueError("Manifest has no coverage record")
    for name in (
        "exact_threshold",
        "grayscale",
        "rgb_or_color",
        "odd_dimension",
        "minimum_selected_pixels",
        "maximum_selected_pixels",
    ):
        if manifest_coverage.get(name) != coverage[name]:
            raise ValueError(f"Manifest coverage drift for {name}")
    if manifest.get("selected_counts") != coverage["route_counts"]:
        raise ValueError("Manifest selected_counts drifted")
    return facts, coverage


def page_route(model_label: object) -> str | None:
    if not isinstance(model_label, str):
        return None
    if model_label.endswith("-normal"):
        return "normal"
    if model_label.endswith("-sharper"):
        return "sharper"
    return None


def verify_page_records(
    pages: list[dict[str, Any]], expected: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    observed: dict[str, dict[str, Any]] = {}
    for page in pages:
        if (
            page.get("type") != "waifuhat2x-page-metrics"
            or page.get("schema_version") != 1
            or page.get("status") != "complete"
        ):
            raise ValueError(
                "Page telemetry is incomplete or has an unsupported schema"
            )
        source = page.get("source")
        details = page.get("details")
        if not isinstance(source, str) or not isinstance(details, dict):
            raise ValueError("Page telemetry has no source/details")
        if source in observed:
            raise ValueError(f"Duplicate page telemetry source: {source}")
        fact = expected.get(source)
        if fact is None:
            raise ValueError(f"Unexpected page telemetry source: {source}")
        dimensions = details.get("source_dimensions")
        actual = {
            "index": page.get("index"),
            "width": dimensions[0]
            if isinstance(dimensions, list) and len(dimensions) == 2
            else None,
            "height": dimensions[1]
            if isinstance(dimensions, list) and len(dimensions) == 2
            else None,
            "short_edge": details.get("source_short_edge"),
            "route": page_route(details.get("model_label")),
            "grayscale": details.get("grayscale"),
        }
        for name, value in actual.items():
            if value != fact[name]:
                raise ValueError(
                    f"Telemetry fact drift for {source} {name}: {value!r} != {fact[name]!r}"
                )
        checkpoint = details.get("model_checkpoint")
        expected_checkpoint = (
            "Real_HAT_GAN_SRx4.pth"
            if fact["route"] == "normal"
            else "Real_HAT_GAN_SRx4_sharper.pth"
        )
        if checkpoint != expected_checkpoint:
            raise ValueError(f"Telemetry checkpoint drift for {source}: {checkpoint!r}")
        observed[source] = actual
    if set(observed) != set(expected):
        raise ValueError(
            "Page telemetry source set does not match the isolated manifest"
        )
    return {
        "page_count": len(observed),
        "route_counts": dict(
            sorted(Counter(item["route"] for item in observed.values()).items())
        ),
        "per_source_facts_match": True,
    }


def verify_attempt(
    completion_path: Path,
    expected: dict[str, dict[str, Any]],
    *,
    expected_schema_version: int | None = None,
) -> dict[str, Any]:
    attempt_root = completion_path.parent.resolve()
    completion = read_json(completion_path)
    if (
        completion.get("schema_version") not in E2E_SCHEMA_VERSIONS
        or completion.get("kind") != E2E_COMPLETION_KIND
    ):
        raise ValueError(f"Invalid completion marker: {completion_path}")
    completion_schema = int(completion["schema_version"])
    if (
        expected_schema_version is not None
        and completion_schema != expected_schema_version
    ):
        raise ValueError(f"Completion/summary schema mismatch: {completion_path}")
    result_name = completion.get("result_path")
    if not isinstance(result_name, str):
        raise ValueError(f"Completion marker has no result path: {completion_path}")
    result_path = (attempt_root / result_name).resolve()
    if not is_relative_to(result_path, attempt_root) or not result_path.is_file():
        raise ValueError(f"Unsafe or missing result path: {result_path}")
    if sha256_file(result_path) != completion.get("result_sha256"):
        raise ValueError(f"Result hash does not match completion marker: {result_path}")
    result = read_json(result_path)
    if (
        result.get("schema_version") != completion_schema
        or result.get("kind") != E2E_RESULT_KIND
        or result.get("status") != "complete"
    ):
        raise ValueError(f"Invalid complete child result: {result_path}")
    for field, completion_field in (
        ("spec_sha256", "spec_sha256"),
        ("config_sha256", "config_sha256"),
    ):
        if result.get(field) != completion.get(completion_field):
            raise ValueError(f"Completion/result {field} mismatch: {result_path}")
    pages_record = result.get("pages")
    job_record = result.get("job")
    if not isinstance(pages_record, dict) or not isinstance(job_record, dict):
        raise ValueError(f"Result has no telemetry evidence: {result_path}")
    pages_path = (attempt_root / str(pages_record.get("path", ""))).resolve()
    job_path = (attempt_root / str(job_record.get("path", ""))).resolve()
    for path, record, completion_key in (
        (pages_path, pages_record, "pages_sha256"),
        (job_path, job_record, "job_sha256"),
    ):
        if not is_relative_to(path, attempt_root) or not path.is_file():
            raise ValueError(f"Unsafe or missing child artifact: {path}")
        digest = sha256_file(path)
        if digest != record.get("sha256") or digest != completion.get(completion_key):
            raise ValueError(f"Child artifact hash mismatch: {path}")
    pages = [
        json.loads(line) for line in pages_path.read_text(encoding="utf-8").splitlines()
    ]
    if not all(isinstance(page, dict) for page in pages):
        raise ValueError(f"Invalid pages JSONL: {pages_path}")
    page_report = verify_page_records(pages, expected)  # type: ignore[arg-type]
    if pages_record.get("count") != len(pages):
        raise ValueError(f"Result page count mismatch: {result_path}")
    return {
        "attempt": str(attempt_root),
        "completion_sha256": sha256_file(completion_path),
        "result_sha256": sha256_file(result_path),
        "pages_sha256": sha256_file(pages_path),
        "job_sha256": sha256_file(job_path),
        "role": result.get("role"),
        "index": result.get("index"),
        "configuration": result.get("configuration"),
        "page_evidence": page_report,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Attest an existing Real-HAT E2E run by re-reading isolated images "
            "and binding every per-page route to completion hashes."
        )
    )
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.threshold < 1:
        raise SystemExit("threshold must be positive")
    benchmark_root = args.benchmark_root.resolve()
    manifest_path = args.manifest.resolve()
    summary_path = benchmark_root / "benchmark_summary.json"
    summary = read_json(summary_path)
    if (
        summary.get("schema_version") not in E2E_SCHEMA_VERSIONS
        or summary.get("kind") != E2E_SUMMARY_KIND
        or summary.get("status") != "complete"
    ):
        raise ValueError("The E2E benchmark has not completed")
    production = summary.get("production_qualification")
    if not isinstance(production, dict) or not isinstance(
        production.get("valid_for_production_decision"), bool
    ):
        raise ValueError("The E2E summary has no final production qualification")
    expected, coverage = inspect_manifest(manifest_path, args.threshold)
    completion_paths = sorted((benchmark_root / "runs").rglob("completion.json"))
    expected_attempts = len(summary.get("runs", []))
    if expected_attempts < 1 or len(completion_paths) != expected_attempts:
        raise ValueError(
            f"Completion marker count drift: {len(completion_paths)} != {expected_attempts}"
        )
    summary_schema = int(summary["schema_version"])
    attempts = [
        verify_attempt(
            path,
            expected,
            expected_schema_version=summary_schema,
        )
        for path in completion_paths
    ]
    if any(
        sha256_file(Path(fact["source_path"])) != fact["source_sha256"]
        for fact in expected.values()
    ):
        raise RuntimeError("Isolated representative input changed during attestation")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "real_hat_e2e_representative_attestation",
        "status": "complete",
        "created_at": utc_now(),
        "benchmark_root": str(benchmark_root),
        "benchmark_summary": {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
            "production_qualification": production["valid_for_production_decision"],
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "threshold": args.threshold,
        "coverage": coverage,
        "per_source_expectations": expected,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "checks": {
            "manifest_facts_redecoded_after_exif": True,
            "manifest_coverage_matches_redecoded_inputs": True,
            "all_attempt_completion_hashes_valid": True,
            "all_page_sources_dimensions_routes_and_models_match": True,
            "isolated_input_hashes_unchanged": True,
        },
    }
    output = (
        args.output.resolve()
        if args.output
        else benchmark_root / "representative_attestation.json"
    )
    if output.exists():
        raise FileExistsError(output)
    write_json(output, payload)
    print(output)


if __name__ == "__main__":
    main()
