from __future__ import annotations

import argparse
import json
from pathlib import Path

from waifuhat2x.config import load_config
from waifuhat2x.pipeline import (
    _discover_files,
    _normalized_path_key,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect source images whose destination JXL already exists."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    config = load_config(args.config)
    discovery = _discover_files(root, include_metadata=False)
    report = []
    conflict_keys: set[str] = set()
    for source in discovery.images:
        relative = source.relative_to(root)
        companion_key = _normalized_path_key(relative.with_suffix(".jxl"))
        destination = discovery.jxl_by_key.get(companion_key)
        if destination is None:
            continue
        conflict_keys.add(companion_key)
        row: dict[str, object] = {
            "source": relative.as_posix(),
            "jxl": destination.relative_to(root).as_posix(),
        }
        try:
            if destination.is_symlink() or not destination.is_file():
                raise RuntimeError("existing JXL is not a regular file")
            row.update(
                jxl_bytes=destination.stat().st_size,
                action="replace_existing_jxl_after_source_processing",
            )
        except Exception as exc:
            row.update(action="error", error=f"{type(exc).__name__}: {exc}")
        report.append(row)
    summary = {
        "root": str(root),
        "policy": config.output.existing_jxl_policy,
        "source_images": len(discovery.images),
        "jxl_input_skipped": len(discovery.jxl_by_key),
        "conflicts": len(report),
        "jxl_only_skipped": len(discovery.jxl_by_key) - len(conflict_keys),
        "replace": sum(
            row.get("action") == "replace_existing_jxl_after_source_processing"
            for row in report
        ),
        "errors": sum(row.get("action") == "error" for row in report),
        "items": report,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
