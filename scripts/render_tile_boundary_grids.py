from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser(description="Render source previews with tile-boundary grids.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--page-indexes", type=int, nargs="+", required=True)
    parser.add_argument("--tiles", type=int, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if any(value < 1 for value in (*args.page_indexes, *args.tiles)):
        raise SystemExit("page indexes and tiles must be positive")
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = {int(page["index"]): page for page in manifest["pages"]}
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    colors = ((230, 30, 40), (20, 105, 230), (0, 150, 90))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for page_index in args.page_indexes:
        page = pages[page_index]
        source = (manifest_path.parent / page["copied_path"]).resolve()
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
        draw = ImageDraw.Draw(image)
        for tile_index, tile in enumerate(args.tiles):
            color = colors[tile_index % len(colors)]
            for x in range(tile, image.width, tile):
                draw.line((x, 0, x, image.height), fill=color, width=2)
                draw.rectangle((x + 2, 2 + tile_index * 22, x + 76, 21 + tile_index * 22), fill="white")
                draw.text((x + 4, 3 + tile_index * 22), f"t{tile} x{x}", fill=color, font=font)
            for y in range(tile, image.height, tile):
                draw.line((0, y, image.width, y), fill=color, width=2)
                draw.rectangle((2, y + 2 + tile_index * 22, 92, y + 21 + tile_index * 22), fill="white")
                draw.text((4, y + 3 + tile_index * 22), f"t{tile} y{y}", fill=color, font=font)
        image.save(output_root / f"{page_index:02d}_grid.png", format="PNG", compress_level=1)


if __name__ == "__main__":
    main()
