from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

from PIL import Image

from waifuhat2x.config import load_config
from waifuhat2x.images import ResolutionPlan, plan_resolution
from waifuhat2x.models import available_scales, real_hat_variant
from waifuhat2x.pipeline import _discover_files, _normalized_path_key


def _display_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as opened:
        if getattr(opened, "n_frames", 1) != 1:
            raise ValueError("animated or multi-page image")
        width, height = opened.size
        orientation = int(opened.getexif().get(274, 1))
        if orientation in {5, 6, 7, 8}:
            width, height = height, width
        return width, height


def _category(plan: ResolutionPlan) -> str:
    if plan.upscale:
        return f"hat_x{plan.native_scale}"
    return {
        "short edge already meets target": "transcode_target_met",
        "long edge exceeds SR safety limit": "transcode_long_edge_safety",
        "planned output exceeds safety limit": "transcode_output_safety",
    }.get(plan.reason, "transcode_other")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only workload count; does not hash files, load HAT, or write the library."
    )
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    config = load_config(args.config)
    root = config.paths.input
    discovery = _discover_files(root, include_metadata=False)
    scales = available_scales(
        config.paths.models,
        config.processing.profile,
        True,
    )
    counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    conflict_keys: set[str] = set()
    real_hat_profile = config.processing.profile.lower() == "real-hat-auto"

    def inspect(
        source: Path,
    ) -> tuple[Path, int | None, ResolutionPlan | None, str | None]:
        try:
            width, height = _display_dimensions(source)
            plan = plan_resolution(
                width,
                height,
                config.processing.target_short_edge,
                config.processing.max_long_edge_for_sr,
                scales,
                config.processing.max_upscale_factor,
                config.processing.max_output_long_edge,
                config.processing.max_output_megapixels,
            )
            return source, min(width, height), plan, None
        except Exception as exc:
            return source, None, None, f"{type(exc).__name__}: {exc}"

    last_real_hat_variant: str | None = None
    real_hat_model_switches = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (source, source_short_edge, plan, error) in enumerate(
            executor.map(inspect, discovery.images), start=1
        ):
            relative = source.relative_to(root)
            companion_key = _normalized_path_key(relative.with_suffix(".jxl"))
            if companion_key in discovery.jxl_by_key:
                conflict_keys.add(companion_key)
            if error is not None:
                errors.append({"source": relative.as_posix(), "error": error})
            else:
                assert plan is not None
                counts[_category(plan)] += 1
                if "remains below target" in plan.reason:
                    counts["target_unmet"] += 1
                if real_hat_profile and plan.upscale:
                    assert source_short_edge is not None
                    variant = real_hat_variant(
                        source_short_edge,
                        config.processing.real_hat_sharper_min_short_edge,
                    )
                    counts[f"real_hat_{variant}"] += 1
                    if source_short_edge == config.processing.real_hat_sharper_min_short_edge:
                        counts["real_hat_threshold_exact"] += 1
                    if last_real_hat_variant is not None and variant != last_real_hat_variant:
                        real_hat_model_switches += 1
                    last_real_hat_variant = variant
            if index % 500 == 0:
                print(f"inspected {index}/{len(discovery.images)}", file=sys.stderr)

    sr_total = sum(value for key, value in counts.items() if key.startswith("hat_x"))
    transcode_total = sum(
        value for key, value in counts.items() if key.startswith("transcode_")
    )
    result = {
        "root": str(root),
        "source_images": len(discovery.images),
        "existing_jxl_files": len(discovery.jxl_by_key),
        "jxl_only_skipped": len(discovery.jxl_by_key) - len(conflict_keys),
        "same_stem_jxl_replace_targets": len(conflict_keys),
        "sr_total": sr_total,
        "hat_x2": counts["hat_x2"],
        "hat_x4": counts["hat_x4"],
        "real_hat_sharper_min_short_edge": (
            config.processing.real_hat_sharper_min_short_edge if real_hat_profile else None
        ),
        "real_hat_normal": counts["real_hat_normal"],
        "real_hat_sharper": counts["real_hat_sharper"],
        "real_hat_threshold_exact": counts["real_hat_threshold_exact"],
        "real_hat_model_switches": real_hat_model_switches,
        "target_unmet": counts["target_unmet"],
        "transcode_total": transcode_total,
        "transcode_target_met": counts["transcode_target_met"],
        "transcode_long_edge_safety": counts["transcode_long_edge_safety"],
        "transcode_output_safety": counts["transcode_output_safety"],
        "transcode_other": counts["transcode_other"],
        "errors": len(errors),
        "error_items": errors[:100],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
