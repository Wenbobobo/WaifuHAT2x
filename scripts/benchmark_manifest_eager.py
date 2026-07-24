from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Iterator

import numpy as np
from PIL import Image, ImageOps
import torch

try:
    from scripts.research_utils import (
        environment_report,
        json_safe,
        pixel_sha256,
        safe_name,
        save_lossless_png,
        sha256_file,
        stats_mapping,
        utc_now,
        write_json,
    )
except ModuleNotFoundError:
    from research_utils import (  # type: ignore[no-redef]
        environment_report,
        json_safe,
        pixel_sha256,
        safe_name,
        save_lossless_png,
        sha256_file,
        stats_mapping,
        utc_now,
        write_json,
    )
from waifuhat2x.engine import UpscaleEngine
from waifuhat2x.images import pil_to_tensor


STEADY_ROCTX_RANGE = "real_hat_steady_rounds"
BACKEND_ENVIRONMENT_KEYS = ("TORCH_BLAS_PREFER_HIPBLASLT",)


class RocTxProfilerControl:
    def __init__(self) -> None:
        try:
            module = importlib.import_module("roctx")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ROCTx Python bindings are required for --rocprof-selected-regions"
            ) from exc
        self._resume = getattr(module, "profilerResume", None)
        self._pause = getattr(module, "profilerPause", None)
        self._range_push = getattr(module, "rangePush", None)
        self._range_pop = getattr(module, "rangePop", None)
        if not all(
            callable(operation)
            for operation in (
                self._resume,
                self._pause,
                self._range_push,
                self._range_pop,
            )
        ):
            raise RuntimeError("ROCTx Python bindings have incomplete control APIs")

    @staticmethod
    def _check_result(operation: str, result: Any) -> None:
        if result not in (None, 0):
            raise RuntimeError(f"ROCTx {operation} failed with code {result}")

    def resume(self) -> None:
        self._check_result("profilerResume", self._resume(0))
        try:
            result = self._range_push(STEADY_ROCTX_RANGE)
            if result is not None and (not isinstance(result, int) or result < 0):
                raise RuntimeError(f"ROCTx rangePush failed with code {result}")
        except BaseException:
            self._check_result("profilerPause", self._pause(0))
            raise

    def pause(self) -> None:
        try:
            result = self._range_pop()
            if result is not None and (not isinstance(result, int) or result < 0):
                raise RuntimeError(f"ROCTx rangePop failed with code {result}")
        finally:
            self._check_result("profilerPause", self._pause(0))


def stop_selected_region(control: RocTxProfilerControl) -> None:
    try:
        torch.cuda.synchronize()
    finally:
        control.pause()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def load_manifest(
    path: Path, indexes: list[int] | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "real_hat_representative_manifest":
        raise ValueError(f"Unsupported manifest kind: {manifest.get('kind')!r}")
    by_index = {int(page["index"]): page for page in manifest["pages"]}
    order = indexes or sorted(by_index)
    if len(order) != len(set(order)):
        raise ValueError("--page-indexes must not contain duplicates")
    missing = [index for index in order if index not in by_index]
    if missing:
        raise ValueError(f"Manifest has no page indexes: {missing}")
    return manifest, [by_index[index] for index in order]


def load_tensor(path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.load()
    return pil_to_tensor(image), image.size


def metric(values: list[float]) -> dict[str, float | int | None]:
    mean = statistics.fmean(values)
    cv = statistics.pstdev(values) / mean if len(values) > 1 and mean else None
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": mean,
        "minimum": min(values),
        "maximum": max(values),
        "cv_percent": cv * 100 if cv is not None else None,
    }


def estimate_tile_work(
    width: int, height: int, tile: int, overlap: int
) -> dict[str, int]:
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    if tile < 1:
        raise ValueError("tile must be positive")
    if overlap < 0 or overlap >= tile:
        raise ValueError("overlap must be non-negative and smaller than tile")
    tiles_x = math.ceil(width / tile)
    tiles_y = math.ceil(height / tile)
    tile_count = tiles_x * tiles_y
    expanded_edge = tile + 2 * overlap
    expanded_tile_area = expanded_edge**2
    return {
        "tile": tile,
        "tiles_x": tiles_x,
        "tiles_y": tiles_y,
        "tile_count": tile_count,
        "expanded_edge": expanded_edge,
        "expanded_tile_area": expanded_tile_area,
        "estimated_work": tile_count * expanded_tile_area,
    }


def choose_adaptive_tile(
    width: int, height: int, candidates: list[int], overlap: int
) -> tuple[int, list[dict[str, int]]]:
    ordered = sorted(set(candidates))
    if not ordered:
        raise ValueError("adaptive tile candidates must not be empty")
    estimates = [estimate_tile_work(width, height, tile, overlap) for tile in ordered]
    selected = min(estimates, key=lambda item: (item["estimated_work"], item["tile"]))
    return selected["tile"], estimates


def runtime_code_fingerprints() -> list[dict[str, str | int]]:
    paths = {
        Path(__file__).resolve(),
        Path(inspect.getfile(UpscaleEngine)).resolve(),
        Path(inspect.getfile(pil_to_tensor)).resolve(),
    }
    return [
        {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(paths, key=str)
    ]


def build_engine(
    tile_candidates: list[int], overlap: int, collect_gpu_timing: bool
) -> UpscaleEngine:
    candidates = tuple(tile_candidates)
    return UpscaleEngine(
        precision="bf16",
        tile=candidates[0],
        overlap=overlap,
        hat_tile=candidates[0],
        hat_overlap=overlap,
        batch_tiles=1,
        device_assembly=True,
        model_cache_size=2,
        collect_gpu_timing=collect_gpu_timing,
        hat_tile_candidates=candidates,
    )


@contextmanager
def fixed_engine_tile(engine: UpscaleEngine, selected_tile: int) -> Iterator[None]:
    previous = (engine.tile, engine.hat_tile, engine.hat_tile_candidates)
    engine.tile = selected_tile
    engine.hat_tile = selected_tile
    engine.hat_tile_candidates = (selected_tile,)
    try:
        yield
    finally:
        engine.tile, engine.hat_tile, engine.hat_tile_candidates = previous


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a deterministic Real-HAT manifest with one dual-model cache."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--page-indexes", type=positive_int, nargs="+")
    parser.add_argument("--normal-model", type=Path, required=True)
    parser.add_argument("--sharper-model", type=Path, required=True)
    parser.add_argument("--threshold", type=positive_int, default=1000)
    tile_group = parser.add_mutually_exclusive_group(required=True)
    tile_group.add_argument("--tile", type=positive_int)
    tile_group.add_argument("--adaptive-tiles", type=positive_int, nargs="+")
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--rounds", type=positive_int, default=3)
    parser.add_argument("--warmups-per-model", type=positive_int, default=1)
    parser.add_argument("--warmup-crop", type=positive_int, default=320)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--save-first-round", action="store_true")
    parser.add_argument("--png-compress-level", type=int, choices=range(10), default=1)
    parser.add_argument("--gpu-phase-timing", action="store_true")
    parser.add_argument("--rocprof-selected-regions", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tile_candidates = (
        sorted(set(args.adaptive_tiles)) if args.adaptive_tiles else [args.tile]
    )
    if args.overlap < 0 or any(args.overlap >= tile for tile in tile_candidates):
        raise SystemExit("overlap must be non-negative and smaller than every tile")
    tile_mode = "adaptive-estimated-work" if args.adaptive_tiles else "fixed"
    manifest_path = args.manifest.resolve()
    manifest, pages = load_manifest(manifest_path, args.page_indexes)
    if {str(page["route"]) for page in pages} != {"normal", "sharper"}:
        raise ValueError(
            "A dual-model batch must contain both normal and sharper pages"
        )
    manifest_root = manifest_path.parent
    normal_model = args.normal_model.resolve()
    sharper_model = args.sharper_model.resolve()
    for model in (normal_model, sharper_model):
        if not model.is_file():
            raise FileNotFoundError(model)

    run_root = (args.output_root.resolve() / safe_name(args.run_name)).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    summary_path = run_root / "batch_summary.json"
    decoded: list[dict[str, Any]] = []
    for page in pages:
        copied = page.get("copied_path")
        if not copied:
            raise ValueError(f"Page {page['index']} has no isolated copied_path")
        source = (manifest_root / copied).resolve()
        if manifest_root not in source.parents or not source.is_file():
            raise ValueError(f"Unsafe or missing isolated source: {source}")
        if sha256_file(source) != page["copied_sha256"]:
            raise ValueError(f"Copied source hash changed: {source}")
        tensor, size = load_tensor(source)
        route = "normal" if min(size) < args.threshold else "sharper"
        if route != page["route"]:
            raise ValueError(
                f"Route drift for page {page['index']}: {route} != {page['route']}"
            )
        selected_tile, tile_estimates = choose_adaptive_tile(
            size[0], size[1], tile_candidates, args.overlap
        )
        decoded.append(
            {
                "page": page,
                "path": source,
                "tensor": tensor,
                "size": size,
                "route": route,
                "selected_tile": selected_tile,
                "tile_estimates": tile_estimates,
            }
        )

    model_payload = {
        "normal": {
            "path": str(normal_model),
            "sha256": sha256_file(normal_model),
            "bytes": normal_model.stat().st_size,
        },
        "sharper": {
            "path": str(sharper_model),
            "sha256": sha256_file(sharper_model),
            "bytes": sharper_model.stat().st_size,
        },
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "real_hat_manifest_eager_benchmark",
        "status": "running",
        "started_at": utc_now(),
        "environment": environment_report(),
        # This is read in the child process after its backend environment has been
        # established, so a research wrapper can attest which BLAS preference ran.
        "backend_environment": {
            name: os.environ.get(name) for name in BACKEND_ENVIRONMENT_KEYS
        },
        "manifest": str(manifest_path),
        "page_order": [int(item["page"]["index"]) for item in decoded],
        "runtime_code": runtime_code_fingerprints(),
        "models": model_payload,
        "configuration": {
            "precision": "bf16",
            "tile_mode": tile_mode,
            "tile": args.tile,
            "adaptive_tiles": tile_candidates if args.adaptive_tiles else None,
            "adaptive_selection_formula": (
                "ceil(width/tile) * ceil(height/tile) * (tile + 2*overlap)^2; "
                "minimum wins, ties use the smaller tile"
                if args.adaptive_tiles
                else None
            ),
            "overlap": args.overlap,
            "batch_tiles": 1,
            "device_assembly": True,
            "model_cache_size": 2,
            "threshold": args.threshold,
            "rounds": args.rounds,
            "warmups_per_model": args.warmups_per_model,
            "warmup_crop": args.warmup_crop,
            "gpu_phase_timing": args.gpu_phase_timing,
        },
        "profiler_control": {
            "rocprof_selected_regions": args.rocprof_selected_regions,
            "scope": (
                "all steady rounds only"
                if args.rocprof_selected_regions
                else "disabled"
            ),
            "range_name": (
                STEADY_ROCTX_RANGE if args.rocprof_selected_regions else None
            ),
        },
        "methodology": {
            "decoded_once": True,
            "round_wall": "perf_counter around ordered engine.upscale calls plus pixel hashing; PNG saving occurs after the measured round",
            "model_wall": "sum of per-page engine.upscale wall durations",
            "pixel_hash": "SHA-256 of contiguous uint8 output pixels",
            "page_order_fixed_across_rounds": True,
            "tile_plan": [
                {
                    "index": int(item["page"]["index"]),
                    "input_size": list(item["size"]),
                    "selected_tile": item["selected_tile"],
                    "estimates": item["tile_estimates"],
                }
                for item in decoded
            ],
        },
        "warmups": [],
        "rounds": [],
    }
    write_json(summary_path, summary)
    engine = build_engine(tile_candidates, args.overlap, args.gpu_phase_timing)
    profiler_control: RocTxProfilerControl | None = None
    profiling_resumed = False
    try:
        load_reports = []
        for route, model in (("normal", normal_model), ("sharper", sharper_model)):
            _descriptor, _dtype, seconds, cache_hit = engine._load(model)
            load_reports.append(
                {"route": route, "seconds": seconds, "cache_hit": cache_hit}
            )
        summary["model_preloads"] = load_reports

        warmup_pairs = sorted(
            {(str(item["route"]), int(item["selected_tile"])) for item in decoded}
        )
        for route, selected_tile in warmup_pairs:
            model = normal_model if route == "normal" else sharper_model
            example = next(
                item
                for item in decoded
                if item["route"] == route and item["selected_tile"] == selected_tile
            )
            tensor = example["tensor"][:, :, : args.warmup_crop, : args.warmup_crop]
            for index in range(1, args.warmups_per_model + 1):
                with fixed_engine_tile(engine, selected_tile):
                    started = time.perf_counter()
                    output, stats = engine.upscale(
                        tensor, model, grayscale_output=False
                    )
                summary["warmups"].append(
                    {
                        "route": route,
                        "tile": selected_tile,
                        "index": index,
                        "wall_seconds": time.perf_counter() - started,
                        "pixel_sha256": pixel_sha256(output),
                        "stats": stats_mapping(stats),
                    }
                )
                del output

        if args.rocprof_selected_regions:
            profiler_control = RocTxProfilerControl()
            torch.cuda.synchronize()
            profiler_control.resume()
            profiling_resumed = True

        for round_index in range(1, args.rounds + 1):
            retained: list[tuple[dict[str, Any], np.ndarray]] = []
            page_reports: list[dict[str, Any]] = []
            round_started = time.perf_counter()
            for item in decoded:
                page = item["page"]
                model = normal_model if item["route"] == "normal" else sharper_model
                selected_tile = int(item["selected_tile"])
                page_started = time.perf_counter()
                output, stats = engine.upscale(
                    item["tensor"], model, grayscale_output=bool(page["grayscale"])
                )
                upscale_wall = time.perf_counter() - page_started
                stats_payload = stats_mapping(stats)
                peak_reserved_vram_bytes = int(
                    stats_payload["peak_reserved_vram_bytes"]
                )
                if int(stats_payload["tile"]) != selected_tile:
                    raise RuntimeError(
                        f"Engine tile drift for page {page['index']}: "
                        f"{stats_payload['tile']} != {selected_tile}"
                    )
                hash_started = time.perf_counter()
                digest = pixel_sha256(output)
                hash_seconds = time.perf_counter() - hash_started
                page_reports.append(
                    {
                        "index": int(page["index"]),
                        "route": item["route"],
                        "source": str(item["path"]),
                        "input_size": list(item["size"]),
                        "selected_tile": selected_tile,
                        "tile_estimates": item["tile_estimates"],
                        "upscale_wall_seconds": upscale_wall,
                        "pixel_hash_seconds": hash_seconds,
                        "pixel_sha256": digest,
                        "output_shape": list(output.shape),
                        "stats": stats_payload,
                        "peak_reserved_vram_bytes": peak_reserved_vram_bytes,
                    }
                )
                if args.save_first_round and round_index == 1:
                    retained.append((page_reports[-1], output))
                else:
                    del output
                print(
                    f"round={round_index}/{args.rounds} page={page['index']} route={item['route']} "
                    f"tile={selected_tile} wall={upscale_wall:.4f}s",
                    flush=True,
                )
            round_loop_wall = time.perf_counter() - round_started
            round_report = {
                "round": round_index,
                "loop_wall_seconds": round_loop_wall,
                "model_wall_seconds": sum(
                    item["upscale_wall_seconds"] for item in page_reports
                ),
                "page_count": len(page_reports),
                "pages": page_reports,
            }
            summary["rounds"].append(round_report)
            write_json(summary_path, summary)

            if retained:
                output_root = run_root / "outputs-round-01"
                for page_report, output in retained:
                    output_path = (
                        output_root
                        / f"{page_report['index']:02d}_{page_report['route']}.png"
                    )
                    page_report["png_encode_seconds"] = save_lossless_png(
                        output, output_path, args.png_compress_level
                    )
                    page_report["png_path"] = str(output_path.resolve())
                    page_report["png_sha256"] = sha256_file(output_path)
                    del output
                write_json(summary_path, summary)

        if profiler_control is not None:
            try:
                stop_selected_region(profiler_control)
            finally:
                profiling_resumed = False

        round_walls = [float(item["loop_wall_seconds"]) for item in summary["rounds"]]
        model_walls = [float(item["model_wall_seconds"]) for item in summary["rounds"]]
        by_page: dict[int, set[str]] = {}
        for round_report in summary["rounds"]:
            for page_report in round_report["pages"]:
                by_page.setdefault(int(page_report["index"]), set()).add(
                    str(page_report["pixel_sha256"])
                )
        peak_allocated = max(
            int(page["stats"]["peak_vram_bytes"])
            for round_report in summary["rounds"]
            for page in round_report["pages"]
        )
        peak_reserved = max(
            int(page["peak_reserved_vram_bytes"])
            for round_report in summary["rounds"]
            for page in round_report["pages"]
        )
        summary["steady_state"] = {
            "round_loop_wall_seconds": metric(round_walls),
            "round_model_wall_seconds": metric(model_walls),
            "pixel_deterministic": all(len(hashes) == 1 for hashes in by_page.values()),
            "unique_hashes_per_page": {
                str(index): sorted(hashes) for index, hashes in sorted(by_page.items())
            },
            "peak_vram_bytes": peak_allocated,
            "peak_reserved_vram_bytes": peak_reserved,
        }
        summary["status"] = "complete"
        summary["finished_at"] = utc_now()
        write_json(summary_path, summary)
        print(json.dumps(json_safe(summary["steady_state"]), indent=2), flush=True)
    except BaseException as exc:
        control_error: BaseException | None = None
        if profiling_resumed and profiler_control is not None:
            try:
                stop_selected_region(profiler_control)
            except BaseException as observed_error:
                control_error = observed_error
            finally:
                profiling_resumed = False
        summary["status"] = "error"
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if control_error is not None:
            summary["profiler_control_error"] = {
                "type": type(control_error).__name__,
                "message": str(control_error),
            }
        summary["finished_at"] = utc_now()
        write_json(summary_path, summary)
        if control_error is not None:
            raise RuntimeError(f"ROCTx cleanup failed: {control_error}") from exc
        raise
    finally:
        engine.close()


if __name__ == "__main__":
    main()
