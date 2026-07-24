from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image, ImageOps

from waifuhat2x.config import AppConfig, load_config
from waifuhat2x.images import IMAGE_EXTENSIONS, is_grayscale, plan_resolution
from waifuhat2x.pipeline import _discover_files


@dataclass(frozen=True)
class Candidate:
    source: str
    width: int
    height: int
    short_edge: int
    long_edge: int
    pixels: int
    route: str
    grayscale: bool
    odd_dimension: bool
    source_mode: str
    source_format: str | None
    file_bytes: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect(path: Path, root: Path, threshold: int, config: AppConfig) -> Candidate | None:
    with Image.open(path) as opened:
        if getattr(opened, "n_frames", 1) != 1:
            return None
        source_mode = opened.mode
        source_format = opened.format
        image = ImageOps.exif_transpose(opened)
        image.load()
    width, height = image.size
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
    if not plan.upscale:
        return None
    short_edge = min(width, height)
    return Candidate(
        source=path.relative_to(root).as_posix(),
        width=width,
        height=height,
        short_edge=short_edge,
        long_edge=max(width, height),
        pixels=width * height,
        route="normal" if short_edge < threshold else "sharper",
        grayscale=is_grayscale(image),
        odd_dimension=bool(width % 2 or height % 2),
        source_mode=source_mode,
        source_format=source_format,
        file_bytes=path.stat().st_size,
    )


def _even_fill(pool: list[Candidate], count: int, selected: list[Candidate]) -> None:
    remaining = [candidate for candidate in pool if candidate not in selected]
    while len(selected) < count and remaining:
        if len(selected) == count - 1:
            index = len(remaining) // 2
        else:
            fraction = (len(selected) + 1) / (count + 1)
            index = min(len(remaining) - 1, round(fraction * (len(remaining) - 1)))
        selected.append(remaining.pop(index))


def _pick(pool: list[Candidate], count: int, threshold: int) -> list[Candidate]:
    if len(pool) < count:
        raise ValueError(f"route has only {len(pool)} candidates, but {count} were requested")
    ordered = sorted(pool, key=lambda item: (item.short_edge, item.long_edge, item.source))
    selected: list[Candidate] = []

    def add(candidate: Candidate | None) -> None:
        if candidate is not None and candidate not in selected and len(selected) < count:
            selected.append(candidate)

    exact = [item for item in ordered if item.short_edge == threshold]
    add(next((item for item in exact if item.grayscale), None))
    add(next((item for item in exact if not item.grayscale), None))
    for item in exact:
        add(item)
        if len([chosen for chosen in selected if chosen.short_edge == threshold]) >= 2:
            break
    add(min(ordered, key=lambda item: (item.pixels, item.source)))
    add(max(ordered, key=lambda item: (item.pixels, item.source)))
    add(next((item for item in ordered if item.grayscale), None))
    add(next((item for item in ordered if not item.grayscale), None))
    add(next((item for item in ordered if item.odd_dimension), None))
    add(ordered[0])
    add(ordered[-1])
    _even_fill(ordered, count, selected)
    return sorted(selected, key=lambda item: (item.route, item.short_edge, item.long_edge, item.source))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, read-only Real-HAT benchmark sample manifest."
    )
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--input-root", type=Path)
    parser.add_argument(
        "--worklist",
        type=Path,
        help="Optional waifuhat2x JSONL worklist; avoids recursively scanning existing JXL files.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=1000)
    parser.add_argument("--normal-count", type=int, default=9)
    parser.add_argument("--sharper-count", type=int, default=21)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--copy-inputs",
        action="store_true",
        help="Copy selected files to output-root/inputs without modifying the source library.",
    )
    return parser


def _worklist_sources(path: Path, source_root: Path) -> tuple[list[Path], dict[str, int]]:
    sources: list[Path] = []
    seen: set[str] = set()
    item_lines = 0
    with path.open("r", encoding="utf-8") as worklist:
        header = json.loads(next(worklist))
        if header.get("type") != "waifuhat2x-worklist":
            raise ValueError(f"Not a waifuhat2x worklist: {path}")
        for line in worklist:
            item_lines += 1
            item = json.loads(line)
            relative = Path(item["source"])
            source = (source_root / relative).resolve()
            if source_root not in source.parents or source.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsafe or unsupported worklist source: {relative}")
            key = relative.as_posix()
            if key in seen:
                continue
            seen.add(key)
            sources.append(source)
    declared_count = int(header.get("count", -1))
    if declared_count != len(sources):
        raise ValueError(
            f"Worklist declares {declared_count} items but contains {len(sources)} unique sources"
        )
    return sources, {
        "declared_count": declared_count,
        "item_lines": item_lines,
        "unique_sources": len(sources),
        "duplicate_lines": item_lines - len(sources),
    }


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.threshold < 1
        or args.normal_count < 1
        or args.sharper_count < 1
        or args.workers < 1
    ):
        raise SystemExit("threshold, route counts, and workers must be positive")
    config_path = args.config.resolve()
    config = load_config(config_path)
    source_root = (args.input_root or config.paths.input).resolve()
    output_root = args.output_root.resolve()
    if args.worklist:
        worklist_path = args.worklist.resolve()
        worklist_sources, worklist_stats = _worklist_sources(worklist_path, source_root)
        missing_sources = [path for path in worklist_sources if not path.is_file()]
        source_paths = [path for path in worklist_sources if path.is_file()]
    else:
        worklist_path = None
        worklist_stats = None
        missing_sources = []
        source_paths = list(_discover_files(source_root, include_metadata=False).images)

    candidates: list[Candidate] = []
    errors: list[dict[str, str]] = []

    def inspect(path: Path) -> tuple[Candidate | None, str | None]:
        try:
            return _inspect(path, source_root, args.threshold, config), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (path, result) in enumerate(
            zip(source_paths, executor.map(inspect, source_paths), strict=True), start=1
        ):
            candidate, error = result
            if candidate is not None:
                candidates.append(candidate)
            if error is not None:
                errors.append({"source": path.relative_to(source_root).as_posix(), "error": error})
            if index % 500 == 0:
                print(f"inspected {index}/{len(source_paths)}", flush=True)

    normal_pool = [item for item in candidates if item.route == "normal"]
    sharper_pool = [item for item in candidates if item.route == "sharper"]
    selected = _pick(normal_pool, args.normal_count, args.threshold) + _pick(
        sharper_pool, args.sharper_count, args.threshold
    )
    selected = sorted(selected, key=lambda item: (item.route != "normal", item.short_edge, item.source))

    output_root.mkdir(parents=True, exist_ok=False)
    inputs_root = output_root / "inputs"
    records: list[dict[str, object]] = []
    for index, candidate in enumerate(selected, start=1):
        source = source_root / Path(candidate.source)
        copied_path: Path | None = None
        if args.copy_inputs:
            inputs_root.mkdir(exist_ok=True)
            copied_path = inputs_root / f"{index:02d}_{candidate.route}_{source.name}"
            shutil.copy2(source, copied_path)
        record = asdict(candidate)
        record.update(
            {
                "index": index,
                "source_sha256": _sha256(source),
                "copied_path": copied_path.relative_to(output_root).as_posix() if copied_path else None,
                "copied_sha256": _sha256(copied_path) if copied_path else None,
            }
        )
        records.append(record)

    route_counts = {
        "normal": sum(item.route == "normal" for item in selected),
        "sharper": sum(item.route == "sharper" for item in selected),
    }
    manifest = {
        "schema_version": 1,
        "kind": "real_hat_representative_manifest",
        "source_root": str(source_root),
        "source_is_read_only": True,
        "worklist": str(worklist_path) if worklist_path else None,
        "worklist_stats": worklist_stats,
        "threshold_semantics": f"short_edge < {args.threshold}: normal; otherwise: sharper",
        "candidate_counts": {"normal": len(normal_pool), "sharper": len(sharper_pool)},
        "selected_counts": route_counts,
        "coverage": {
            "exact_threshold": sum(item.short_edge == args.threshold for item in selected),
            "grayscale": sum(item.grayscale for item in selected),
            "rgb_or_color": sum(not item.grayscale for item in selected),
            "odd_dimension": sum(item.odd_dimension for item in selected),
            "minimum_selected_pixels": min(item.pixels for item in selected),
            "maximum_selected_pixels": max(item.pixels for item in selected),
        },
        "discovery": {
            "source_images": len(source_paths),
            "stale_missing_worklist_sources": len(missing_sources),
            "eligible_sr_candidates": len(candidates),
            "errors": len(errors),
            "error_items": errors[:100],
        },
        "pages": records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **manifest["coverage"], **route_counts}, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
