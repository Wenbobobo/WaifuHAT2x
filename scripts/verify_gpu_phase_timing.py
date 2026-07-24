from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from waifuhat2x.engine import UpscaleEngine
from waifuhat2x.images import is_grayscale, pil_to_tensor


_GPU_PHASE_FIELDS = (
    "h2d_seconds",
    "forward_seconds",
    "gpu_postprocess_seconds",
    "d2h_seconds",
)


def _pixel_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _crop(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be left,top,right,bottom")
    left, top, right, bottom = parts
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise argparse.ArgumentTypeError("crop coordinates are invalid")
    return parts


def _timing_validation_errors(stats: object) -> list[str]:
    errors: list[str] = []
    missing = [name for name in _GPU_PHASE_FIELDS if getattr(stats, name, None) is None]
    if missing:
        errors.append(f"missing calibrated phases: {', '.join(missing)}")

    elapsed = getattr(stats, "seconds", None)
    raw_total = getattr(stats, "gpu_event_total_seconds", None)
    scale = getattr(stats, "gpu_event_scale_to_wall", None)
    raw = getattr(stats, "gpu_event_raw_seconds", None)
    inference_interval = getattr(stats, "inference_interval_ns", None)
    for label, value in (
        ("engine wall time", elapsed),
        ("raw GPU Event total", raw_total),
        ("GPU Event scale", scale),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            errors.append(f"{label} is missing or non-positive")

    if not isinstance(raw, dict) or "gpu_total" not in raw:
        errors.append("raw GPU Event phase map is missing gpu_total")

    if (
        not isinstance(inference_interval, tuple)
        or len(inference_interval) != 2
        or not all(isinstance(value, int) for value in inference_interval)
        or inference_interval[1] < inference_interval[0]
    ):
        errors.append("synchronized inference interval is missing or invalid")
    elif isinstance(elapsed, (int, float)) and math.isfinite(elapsed):
        interval_seconds = (inference_interval[1] - inference_interval[0]) / 1_000_000_000
        tolerance = max(0.005, elapsed * 0.01)
        if abs(interval_seconds - elapsed) > tolerance:
            errors.append(
                "synchronized inference interval differs from engine wall time "
                f"({interval_seconds:.6f}s vs {elapsed:.6f}s)"
            )

    calibrated = [getattr(stats, name, None) for name in _GPU_PHASE_FIELDS]
    if (
        isinstance(elapsed, (int, float))
        and math.isfinite(elapsed)
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in calibrated)
    ):
        calibrated_sum = sum(calibrated)
        tolerance = max(0.005, elapsed * 0.02)
        if calibrated_sum > elapsed + tolerance:
            errors.append(
                "calibrated GPU phase sum exceeds synchronized engine wall time "
                f"({calibrated_sum:.6f}s > {elapsed:.6f}s + {tolerance:.6f}s)"
            )
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify that optional ROCm HIP-event timing preserves exact SR pixels."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop", type=_crop)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if args.tile < 1 or args.overlap < 0 or args.overlap >= args.tile:
        raise SystemExit("tile/overlap values are invalid")

    with Image.open(input_path) as opened:
        image = ImageOps.exif_transpose(opened)
        image.load()
    if args.crop is not None:
        image = image.crop(args.crop)
    grayscale = is_grayscale(image)
    tensor = pil_to_tensor(image)

    engine = UpscaleEngine(
        precision=args.precision,
        tile=args.tile,
        overlap=args.overlap,
        hat_tile=args.tile,
        hat_overlap=args.overlap,
        batch_tiles=1,
        device_assembly=True,
        model_cache_size=1,
        collect_gpu_timing=False,
    )
    try:
        baseline, baseline_stats = engine.upscale(tensor, model_path, grayscale)
        engine.collect_gpu_timing = True
        timed, timed_stats = engine.upscale(tensor, model_path, grayscale)
    finally:
        engine.close()

    baseline_hash = _pixel_sha256(baseline)
    timed_hash = _pixel_sha256(timed)
    timing_errors = _timing_validation_errors(timed_stats)
    result = {
        "schema_version": 1,
        "kind": "waifuhat2x-gpu-phase-timing-verification",
        "input": str(input_path),
        "model": str(model_path),
        "crop": list(args.crop) if args.crop is not None else None,
        "configuration": {
            "precision": args.precision,
            "tile": args.tile,
            "overlap": args.overlap,
            "grayscale": grayscale,
        },
        "baseline_pixel_sha256": baseline_hash,
        "timed_pixel_sha256": timed_hash,
        "pixel_exact": baseline_hash == timed_hash,
        "timing_validation_errors": timing_errors,
        "baseline_stats": asdict(baseline_stats),
        "timed_stats": asdict(timed_stats),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["pixel_exact"] or timing_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
