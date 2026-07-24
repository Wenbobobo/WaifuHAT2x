from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


CATEGORIES = ("text", "screentone", "diagonal")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def boundary_hits(box: tuple[int, int, int, int], tiles: tuple[int, int]) -> list[str]:
    x, y, width, height = box
    right = x + width
    bottom = y + height
    hits: list[str] = []
    for tile in sorted(set(tiles)):
        for position in range(tile, right + 1, tile):
            if x < position < right:
                hits.append(f"t{tile}:x{position}")
        for position in range(tile, bottom + 1, tile):
            if y < position < bottom:
                hits.append(f"t{tile}:y{position}")
    return hits


def blind_left_is_a(seed: str, roi_id: str) -> bool:
    digest = hashlib.sha256(f"{seed}\0{roi_id}".encode()).digest()
    return not bool(digest[0] & 1)


def overlap_fraction(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    intersection_width = max(
        0,
        min(first_x + first_width, second_x + second_width) - max(first_x, second_x),
    )
    intersection_height = max(
        0,
        min(first_y + first_height, second_y + second_height) - max(first_y, second_y),
    )
    intersection = intersection_width * intersection_height
    smaller_area = min(first_width * first_height, second_width * second_height)
    return intersection / smaller_area if smaller_area else 0.0


def validate_annotation_inventory(
    annotations: dict[str, Any],
    minimum_per_category: int,
    allowed_pages: set[int] | None,
) -> dict[str, Any]:
    if annotations.get("schema_version") != 1 or not isinstance(annotations.get("rois"), list):
        raise ValueError("Expected annotation schema_version 1 with a rois list")
    seen_ids: set[str] = set()
    boxes_by_page: dict[int, list[tuple[str, tuple[int, int, int, int]]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    page_category: dict[int, Counter[str]] = defaultdict(Counter)
    for annotation in annotations["rois"]:
        roi_id = str(annotation["id"])
        if roi_id in seen_ids:
            raise ValueError(f"Duplicate ROI id: {roi_id}")
        seen_ids.add(roi_id)
        category = str(annotation["category"])
        if category not in CATEGORIES:
            raise ValueError(f"Unsupported ROI category {category!r}: {roi_id}")
        page_index = int(annotation["page_index"])
        if allowed_pages is not None and page_index not in allowed_pages:
            raise ValueError(f"ROI {roi_id} uses disallowed page {page_index}")
        box = tuple(int(value) for value in annotation["box"])
        if len(box) != 4:
            raise ValueError(f"ROI box must be x,y,width,height: {roi_id}")
        x, y, width, height = box
        if x < 0 or y < 0 or width < 1 or height < 1:
            raise ValueError(f"ROI box must be positive and in bounds: {roi_id}")
        for other_id, other_box in boxes_by_page[page_index]:
            if overlap_fraction(box, other_box) > 0.5:
                raise ValueError(
                    f"ROI {roi_id} substantially overlaps {other_id} on page {page_index}"
                )
        boxes_by_page[page_index].append((roi_id, box))
        counts[category] += 1
        page_category[page_index][category] += 1

    deficient = {
        category: counts[category]
        for category in CATEGORIES
        if counts[category] < minimum_per_category
    }
    if deficient:
        raise ValueError(
            f"Need at least {minimum_per_category} ROIs per category; found {deficient}"
        )
    used_pages = set(boxes_by_page)
    if allowed_pages is not None and used_pages != allowed_pages:
        raise ValueError(
            f"Annotation pages must exactly match {sorted(allowed_pages)}; found {sorted(used_pages)}"
        )
    return {
        "total": len(seen_ids),
        "counts": {category: counts[category] for category in CATEGORIES},
        "per_page_per_category": {
            str(page): {
                category: page_category[page][category] for category in CATEGORIES
            }
            for page in sorted(page_category)
        },
        "pages": sorted(used_pages),
        "duplicate_or_substantially_overlapping_boxes": 0,
    }


def load_benchmark_tile_plan(path: Path) -> tuple[dict[int, int], dict[int, str], str]:
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or not payload.get("rounds"):
        raise ValueError(f"Benchmark summary is not complete: {resolved}")
    plan: dict[int, int] = {}
    first_round_hashes: dict[int, str] = {}
    for round_index, round_report in enumerate(payload["rounds"], start=1):
        for page in round_report["pages"]:
            page_index = int(page["index"])
            tile = int(page["selected_tile"])
            previous = plan.setdefault(page_index, tile)
            if previous != tile:
                raise ValueError(
                    f"Tile selection drift for page {page_index} in {resolved}: {previous} != {tile}"
                )
            if round_index == 1:
                first_round_hashes[page_index] = str(page["png_sha256"])
    return plan, first_round_hashes, sha256_file(resolved)


def configuration_tiles_for_page(
    page_index: int,
    plans: tuple[dict[int, int] | None, dict[int, int] | None],
    declared_tiles: tuple[int | None, int | None],
) -> tuple[int, int]:
    actual: list[int] = []
    for config_index, (plan, declared) in enumerate(zip(plans, declared_tiles)):
        if plan is None:
            if declared is None:
                raise ValueError(
                    f"Configuration {config_index + 1} needs a benchmark summary "
                    "or a declared fixed tile"
                )
            tile = declared
        else:
            selected = plan.get(page_index)
            if selected is None:
                raise ValueError(
                    f"Page {page_index} is absent from config {config_index + 1} summary"
                )
            tile = selected
            if declared is not None and tile != declared:
                raise ValueError(
                    f"Page {page_index} uses tile {tile}, not declared tile {declared}, "
                    f"in config {config_index + 1}"
                )
        actual.append(tile)
    return actual[0], actual[1]


def fitted(image: Image.Image, size: int) -> Image.Image:
    contained = ImageOps.contain(image.convert("RGB"), (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - contained.width) // 2
    y = (size - contained.height) // 2
    canvas.paste(contained, (x, y))
    return canvas


def contact_sheet(
    rows: list[dict[str, Any]],
    output: Path,
    display_size: int,
    native_pixel_display: bool,
) -> None:
    label_width = 180
    gutter = 12
    header_height = 34
    row_height = display_size + 46
    width = label_width + display_size * 2 + gutter * 3
    height = header_height + row_height * len(rows)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        small = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        small = font
    draw.text((gutter, 8), "ROI", fill="black", font=font)
    draw.text((label_width + gutter * 2, 8), "A", fill="black", font=font)
    draw.text((label_width + display_size + gutter * 3, 8), "B", fill="black", font=font)
    for row_index, row in enumerate(rows):
        top = header_height + row_index * row_height
        draw.line((0, top, width, top), fill=(190, 190, 190), width=1)
        draw.text((gutter, top + 10), row["id"], fill="black", font=font)
        draw.text(
            (gutter, top + 34),
            f"page {row['page_index']}\n{row['category']}\n{', '.join(row['boundary_hits'])}",
            fill=(50, 50, 50),
            font=small,
            spacing=3,
        )
        with Image.open(row["a_path"]) as a_opened:
            if native_pixel_display:
                if a_opened.size != (display_size, display_size):
                    raise ValueError(
                        f"Native display requires {display_size}x{display_size} crops: "
                        f"{row['id']} A is {a_opened.size}"
                    )
                a = a_opened.convert("RGB").copy()
            else:
                a = fitted(a_opened, display_size)
        with Image.open(row["b_path"]) as b_opened:
            if native_pixel_display:
                if b_opened.size != (display_size, display_size):
                    raise ValueError(
                        f"Native display requires {display_size}x{display_size} crops: "
                        f"{row['id']} B is {b_opened.size}"
                    )
                b = b_opened.convert("RGB").copy()
            else:
                b = fitted(b_opened, display_size)
        sheet.paste(a, (label_width + gutter * 2, top + 34))
        sheet.paste(b, (label_width + display_size + gutter * 3, top + 34))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", compress_level=1, optimize=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic blind A/B contact sheets for annotated tile-boundary ROIs."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--config-a-root", type=Path, required=True)
    parser.add_argument("--config-b-root", type=Path, required=True)
    parser.add_argument("--config-a-label", required=True)
    parser.add_argument("--config-b-label", required=True)
    parser.add_argument("--config-a-summary", type=Path)
    parser.add_argument("--config-b-summary", type=Path)
    parser.add_argument("--tile-a", type=int)
    parser.add_argument("--tile-b", type=int)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minimum-per-category", type=int, default=20)
    parser.add_argument("--rows-per-sheet", type=int, default=10)
    parser.add_argument("--display-size", type=int, default=288)
    parser.add_argument("--native-pixel-display", action="store_true")
    parser.add_argument("--allowed-pages", type=int, nargs="+")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    positive = (
        args.scale,
        args.minimum_per_category,
        args.rows_per_sheet,
        args.display_size,
        *(tile for tile in (args.tile_a, args.tile_b) if tile is not None),
    )
    if any(value < 1 for value in positive):
        raise SystemExit("tile, scale, counts, and display size must be positive")
    annotations_path = args.annotations.resolve()
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    allowed_pages = set(args.allowed_pages) if args.allowed_pages else None
    inventory = validate_annotation_inventory(
        annotations, args.minimum_per_category, allowed_pages
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    roots = (args.config_a_root.resolve(), args.config_b_root.resolve())
    labels = (args.config_a_label, args.config_b_label)
    declared_tiles = (args.tile_a, args.tile_b)
    summaries = (args.config_a_summary, args.config_b_summary)
    summary_plans: list[dict[int, int] | None] = []
    summary_hashes: list[dict[int, str] | None] = []
    summary_file_hashes: list[str | None] = []
    for summary in summaries:
        if summary is None:
            summary_plans.append(None)
            summary_hashes.append(None)
            summary_file_hashes.append(None)
        else:
            plan, hashes, file_hash = load_benchmark_tile_plan(summary)
            summary_plans.append(plan)
            summary_hashes.append(hashes)
            summary_file_hashes.append(file_hash)
    records: list[dict[str, Any]] = []
    reveal: list[dict[str, Any]] = []
    counts = {category: 0 for category in CATEGORIES}

    for annotation in annotations["rois"]:
        roi_id = str(annotation["id"])
        category = str(annotation["category"])
        counts[category] += 1
        page_index = int(annotation["page_index"])
        box_values = tuple(int(value) for value in annotation["box"])
        x, y, width, height = box_values
        actual_tiles = configuration_tiles_for_page(
            page_index,
            (summary_plans[0], summary_plans[1]),
            declared_tiles,
        )
        hits = boundary_hits(box_values, actual_tiles)
        if not hits:
            raise ValueError(
                f"ROI does not cross an actual tile boundary {actual_tiles}: {roi_id}"
            )
        source_paths = tuple(root / f"{page_index:02d}_{annotation['route']}.png" for root in roots)
        images: list[Image.Image] = []
        try:
            for source in source_paths:
                if not source.is_file():
                    raise FileNotFoundError(source)
                opened = Image.open(source)
                opened.load()
                images.append(opened)
            for config_index, source in enumerate(source_paths):
                expected_hashes = summary_hashes[config_index]
                if expected_hashes is not None:
                    expected_hash = expected_hashes[page_index]
                    actual_hash = sha256_file(source)
                    if actual_hash != expected_hash:
                        raise ValueError(
                            f"First-round PNG hash drift for {roi_id}, config {config_index + 1}"
                        )
            if images[0].size != images[1].size:
                raise ValueError(f"Configuration output dimensions differ for {roi_id}")
            output_box = (x * args.scale, y * args.scale, (x + width) * args.scale, (y + height) * args.scale)
            if output_box[2] > images[0].width or output_box[3] > images[0].height:
                raise ValueError(f"ROI exceeds page output bounds: {roi_id}")
            crops = (images[0].crop(output_box), images[1].crop(output_box))
            left_a = blind_left_is_a(args.seed, roi_id)
            order = (0, 1) if left_a else (1, 0)
            category_root = output_root / "rois" / category
            a_path = category_root / f"{roi_id}-A.png"
            b_path = category_root / f"{roi_id}-B.png"
            category_root.mkdir(parents=True, exist_ok=True)
            crops[order[0]].save(a_path, format="PNG", compress_level=1, optimize=False)
            crops[order[1]].save(b_path, format="PNG", compress_level=1, optimize=False)
        finally:
            for image in images:
                image.close()
        record = {
            "id": roi_id,
            "page_index": page_index,
            "route": annotation["route"],
            "category": category,
            "input_box": list(box_values),
            "output_box": list(output_box),
            "actual_tile_set": sorted(set(actual_tiles)),
            "boundary_hits": hits,
            "a_path": str(a_path.resolve()),
            "b_path": str(b_path.resolve()),
            "a_sha256": sha256_file(a_path),
            "b_sha256": sha256_file(b_path),
        }
        records.append(record)
        reveal.append(
            {
                "id": roi_id,
                "A": labels[order[0]],
                "B": labels[order[1]],
                "A_tile": actual_tiles[order[0]],
                "B_tile": actual_tiles[order[1]],
                "source_a": str(source_paths[order[0]]),
                "source_b": str(source_paths[order[1]]),
            }
        )

    sheets: list[dict[str, Any]] = []
    for category in CATEGORIES:
        category_rows = [record for record in records if record["category"] == category]
        for offset in range(0, len(category_rows), args.rows_per_sheet):
            part = offset // args.rows_per_sheet + 1
            path = output_root / "contact-sheets" / f"{category}-{part:02d}.png"
            contact_sheet(
                category_rows[offset : offset + args.rows_per_sheet],
                path,
                args.display_size,
                args.native_pixel_display,
            )
            sheets.append(
                {
                    "category": category,
                    "part": part,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
            )

    public = {
        "schema_version": 1,
        "kind": "blind_tile_boundary_roi_manifest",
        "annotations": str(annotations_path),
        "blind_labels": ["A", "B"],
        "mapping_file": "mapping-reveal.json (do not open before scoring)",
        "counts": counts,
        "inventory": inventory,
        "tile_boundaries_input_pixels": sorted(
            {tile for record in records for tile in record["actual_tile_set"]}
        ),
        "tile_boundary_source": "per-page selected_tile from each benchmark summary",
        "scale": args.scale,
        "contact_sheet_display": {
            "pixels": args.display_size,
            "native_output_pixels_1_to_1": args.native_pixel_display,
        },
        "rois": records,
        "contact_sheets": sheets,
    }
    hidden = {
        "schema_version": 1,
        "kind": "blind_tile_boundary_roi_reveal",
        "seed_sha256": hashlib.sha256(args.seed.encode()).hexdigest(),
        "configurations": {
            labels[0]: str(roots[0]),
            labels[1]: str(roots[1]),
        },
        "benchmark_summary_sha256": {
            labels[0]: summary_file_hashes[0],
            labels[1]: summary_file_hashes[1],
        },
        "mapping": reveal,
    }
    write_json(output_root / "blind-manifest.json", public)
    write_json(output_root / "mapping-reveal.json", hidden)
    validation_report = {
        "schema_version": 1,
        "kind": "blind_tile_boundary_roi_validation",
        "annotations_sha256": sha256_file(annotations_path),
        "inventory": inventory,
        "source_summary_sha256": {
            "configuration_1": summary_file_hashes[0],
            "configuration_2": summary_file_hashes[1],
        },
        "roi_hashes": {
            record["id"]: {"A": record["a_sha256"], "B": record["b_sha256"]}
            for record in records
        },
        "contact_sheet_hashes": {
            f"{sheet['category']}-{sheet['part']:02d}": sheet["sha256"]
            for sheet in sheets
        },
    }
    write_json(output_root / "validation-report.json", validation_report)
    print(
        json.dumps(
            {"output": str(output_root), "counts": counts, "contact_sheets": len(sheets)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
