from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw


MODEL_FILES = {
    "current-hat-s-auto": "30_current_hat_s_auto_target1600.png",
    "real-hat-normal": "32_real_hat_gan_x4_target1600.png",
    "real-hat-sharper": "33_real_hat_gan_sharper_x4_target1600.png",
}


def edge_map(image: Image.Image) -> np.ndarray:
    values = np.asarray(image.convert("L"), dtype=np.float32)
    edges = np.zeros_like(values)
    edges[:, 1:] += np.abs(values[:, 1:] - values[:, :-1])
    edges[1:, :] += np.abs(values[1:, :] - values[:-1, :])
    return edges


def select_rois(image: Image.Image, size: int, count: int) -> list[tuple[int, int, int, int]]:
    width, height = image.size
    size = min(size, width, height)
    stride = max(1, size // 3)
    edges = edge_map(image)
    candidates: list[tuple[float, int, int]] = []
    for top in range(0, max(1, height - size + 1), stride):
        for left in range(0, max(1, width - size + 1), stride):
            candidates.append((float(edges[top : top + size, left : left + size].mean()), left, top))
    candidates.sort(reverse=True)
    selected: list[tuple[int, int, int, int]] = []
    for _score, left, top in candidates:
        box = (left, top, left + size, top + size)
        if any(
            abs(left - prior[0]) < size // 2 and abs(top - prior[1]) < size // 2
            for prior in selected
        ):
            continue
        selected.append(box)
        if len(selected) == count:
            break
    return selected


def label_image(image: Image.Image, label: str, width: int | None = None) -> Image.Image:
    if width is not None and image.width != width:
        height = round(image.height * width / image.width)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    bar = 40
    canvas = Image.new("RGB", (image.width, image.height + bar), "white")
    canvas.paste(image.convert("RGB"), (0, bar))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 11), label, fill="black")
    return canvas


def horizontal_sheet(images: list[Image.Image]) -> Image.Image:
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    sheet = Image.new("RGB", (width, height), "white")
    left = 0
    for image in images:
        sheet.paste(image, (left, 0))
        left += image.width
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Build anonymized whole-page and detail SR comparisons.")
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--thumbnail-width", type=int, default=420)
    parser.add_argument("--roi-size", type=int, default=384)
    parser.add_argument("--roi-count", type=int, default=3)
    args = parser.parse_args()

    model_names = list(MODEL_FILES)
    random.Random(args.seed).shuffle(model_names)
    labels = {chr(ord("A") + index): name for index, name in enumerate(model_names)}
    args.output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"seed": args.seed, "labels": labels, "samples": {}}

    for sample_root in sorted(path for path in args.comparison_root.iterdir() if path.is_dir()):
        model_images: dict[str, Image.Image] = {}
        for name, filename in MODEL_FILES.items():
            path = sample_root / filename
            if not path.is_file():
                break
            with Image.open(path) as opened:
                model_images[name] = opened.convert("RGB").copy()
        if len(model_images) != len(MODEL_FILES):
            continue

        sample_output = args.output_root / sample_root.name
        sample_output.mkdir(parents=True, exist_ok=True)
        ordered = [
            label_image(model_images[labels[label]], label, args.thumbnail_width)
            for label in sorted(labels)
        ]
        full_path = sample_output / "whole-page-blind.png"
        horizontal_sheet(ordered).save(full_path, compress_level=3)

        reference = model_images["current-hat-s-auto"]
        rois = select_rois(reference, args.roi_size, args.roi_count)
        roi_paths: list[str] = []
        for index, box in enumerate(rois, start=1):
            crops = [
                label_image(model_images[labels[label]].crop(box), label)
                for label in sorted(labels)
            ]
            path = sample_output / f"detail-{index:02d}-blind.png"
            horizontal_sheet(crops).save(path, compress_level=3)
            roi_paths.append(str(path))
        report["samples"][sample_root.name] = {
            "whole_page": str(full_path),
            "rois": [list(box) for box in rois],
            "detail_sheets": roi_paths,
        }

    (args.output_root / "blind_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Blind comparison written to {args.output_root.resolve()}")
    print("Do not inspect blind_manifest.json until visual scoring is complete.")


if __name__ == "__main__":
    main()
