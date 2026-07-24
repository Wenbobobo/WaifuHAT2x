from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.build_blind_boundary_rois import (
        CATEGORIES,
        boundary_hits,
        configuration_tiles_for_page,
        load_benchmark_tile_plan,
        validate_annotation_inventory,
        write_json,
    )
except ModuleNotFoundError:
    from build_blind_boundary_rois import (  # type: ignore[no-redef]
        CATEGORIES,
        boundary_hits,
        configuration_tiles_for_page,
        load_benchmark_tile_plan,
        validate_annotation_inventory,
        write_json,
    )


NEW_ROIS: dict[str, list[tuple[str, int, tuple[int, int, int, int]]]] = {
    "text": [
        ("text-new-01", 10, (216, 16, 80, 80)),
        ("text-new-02", 10, (472, 80, 80, 80)),
        ("text-new-03", 10, (728, 144, 80, 80)),
        ("text-new-04", 10, (216, 232, 80, 80)),
        ("text-new-05", 10, (472, 280, 80, 80)),
        ("text-new-06", 10, (728, 360, 80, 80)),
        ("text-new-07", 10, (216, 472, 80, 80)),
        ("text-new-08", 10, (472, 552, 80, 80)),
        ("text-new-09", 10, (728, 632, 80, 80)),
        ("text-new-10", 10, (216, 712, 80, 80)),
        ("text-new-11", 10, (472, 792, 80, 80)),
        ("text-new-12", 10, (728, 872, 80, 80)),
        ("text-new-13", 10, (472, 984, 80, 80)),
    ],
    "screentone": [
        ("screen-candidate-01", 18, (216, 16, 80, 80)),
        ("screen-candidate-02", 18, (984, 16, 80, 80)),
        ("screen-candidate-04", 18, (968, 216, 80, 80)),
        ("screen-candidate-05", 18, (280, 360, 80, 80)),
        ("screen-candidate-06", 18, (920, 360, 80, 80)),
        ("screen-candidate-10", 18, (984, 472, 80, 80)),
        ("screen-candidate-13", 18, (728, 728, 80, 80)),
        ("screen-candidate-14", 18, (984, 728, 80, 80)),
        ("screen-candidate-16", 18, (328, 920, 80, 80)),
    ],
    "diagonal": [
        ("diagonal-candidate-01", 18, (216, 984, 80, 80)),
        ("diagonal-candidate-02", 18, (472, 984, 80, 80)),
        ("diagonal-candidate-04", 18, (984, 984, 80, 80)),
        ("diagonal-candidate-05", 18, (280, 1200, 80, 80)),
        ("diagonal-candidate-07", 18, (920, 1200, 80, 80)),
        ("diagonal-candidate-08", 18, (216, 1272, 80, 80)),
        ("diagonal-candidate-10", 18, (728, 1272, 80, 80)),
        ("diagonal-candidate-11", 18, (984, 1272, 80, 80)),
        ("diagonal-candidate-13", 18, (472, 1496, 80, 80)),
    ],
}

EXCLUDED_REUSED_IDS = {"text-20", "diagonal-06", "diagonal-08"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build reviewed overlap A/B annotations at actual selected-tile boundaries."
    )
    parser.add_argument("--base-annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--o32-summary", type=Path, required=True)
    parser.add_argument("--o24-summary", type=Path, required=True)
    parser.add_argument("--o16-summary", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_path = args.base_annotations.resolve()
    manifest_path = args.manifest.resolve()
    review_root = args.review_root.resolve()
    for review_name in (
        "candidate-review-text.png",
        "candidate-review-screentone.png",
        "candidate-review-diagonal.png",
    ):
        if not (review_root / review_name).is_file():
            raise FileNotFoundError(review_root / review_name)

    base = json.loads(base_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = {int(page["index"]): page for page in manifest["pages"]}
    o32_plan, _o32_hashes, o32_summary_hash = load_benchmark_tile_plan(args.o32_summary)
    o24_plan, _o24_hashes, o24_summary_hash = load_benchmark_tile_plan(args.o24_summary)
    o16_plan, _o16_hashes, o16_summary_hash = load_benchmark_tile_plan(args.o16_summary)

    by_category: dict[str, list[dict[str, Any]]] = {
        category: [] for category in CATEGORIES
    }
    for annotation in base["rois"]:
        if str(annotation["id"]) in EXCLUDED_REUSED_IDS:
            continue
        box = tuple(int(value) for value in annotation["box"])
        if not boundary_hits(box, (320, 320)):
            continue
        category = str(annotation["category"])
        by_category[category].append(
            {
                "page_index": int(annotation["page_index"]),
                "route": annotation["route"],
                "category": category,
                "box": list(box),
                "provenance": {
                    "kind": "reused_reviewed_t320_roi",
                    "source_annotation_id": annotation["id"],
                },
            }
        )

    for category, candidates in NEW_ROIS.items():
        for candidate_id, page_index, box in candidates:
            by_category[category].append(
                {
                    "page_index": page_index,
                    "route": pages[page_index]["route"],
                    "category": category,
                    "box": list(box),
                    "provenance": {
                        "kind": "reviewed_overlap_candidate",
                        "review_candidate_id": candidate_id,
                    },
                }
            )

    rois: list[dict[str, Any]] = []
    for category in CATEGORIES:
        rows = by_category[category]
        if len(rows) != 20:
            raise RuntimeError(f"Expected 20 {category} ROIs, found {len(rows)}")
        for index, row in enumerate(rows, start=1):
            row["id"] = f"{category}-{index:02d}"
            rois.append(row)

    for row in rois:
        page_index = int(row["page_index"])
        box = tuple(int(value) for value in row["box"])
        for comparison, candidate_plan in (("o24", o24_plan), ("o16", o16_plan)):
            actual_tiles = configuration_tiles_for_page(
                page_index, (o32_plan, candidate_plan), (None, None)
            )
            hits = boundary_hits(box, actual_tiles)
            if not hits:
                raise RuntimeError(
                    f"{row['id']} does not cross an actual o32/{comparison} tile "
                    f"boundary {actual_tiles}"
                )

    annotations = {
        "schema_version": 1,
        "kind": "tile_boundary_roi_annotations",
        "coordinate_space": "input pixels before x4 upscale",
        "comparison": "adaptive t256/t320 overlap32 versus overlap24 and overlap16",
        "selection_scope": "actual selected-tile boundaries from both sides of each A/B comparison",
        "review_basis": {
            "grid_root": str(review_root),
            "candidate_sheets": [
                str(review_root / f"candidate-review-{category}.png")
                for category in CATEGORIES
            ],
        },
        "box_policy": "80x80 input ROI; every ROI crosses an actual tile edge in both pairwise protocols",
        "source_summary_sha256": {
            "o32": o32_summary_hash,
            "o24": o24_summary_hash,
            "o16": o16_summary_hash,
        },
        "rois": rois,
    }
    inventory = validate_annotation_inventory(
        annotations, minimum_per_category=20, allowed_pages={4, 5, 10, 12, 13, 18}
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(output_root / "annotations.json", annotations)
    write_json(
        output_root / "validation-report.json",
        {
            "schema_version": 1,
            "kind": "overlap_boundary_roi_annotation_validation",
            "inventory": inventory,
            "source_summary_sha256": annotations["source_summary_sha256"],
            "all_pairwise_actual_boundary_checks": True,
        },
    )
    print(json.dumps({"output": str(output_root), "inventory": inventory}, indent=2))


if __name__ == "__main__":
    main()
