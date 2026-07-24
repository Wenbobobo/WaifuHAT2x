from __future__ import annotations

import json
from pathlib import Path
import tomllib

from download_models import _gdrive_ticket


def main() -> None:
    with Path("model_sources.toml").open("rb") as handle:
        manifest = tomllib.load(handle)
    tickets = []
    for name, item in manifest.get("models", {}).items():
        if item.get("kind") != "gdrive":
            continue
        if (Path("models") / item["filename"]).is_file():
            continue
        url, cookies, size = _gdrive_ticket(item["id"])
        tickets.append(
            {
                "name": name,
                "filename": item["filename"],
                "url": url,
                "cookies": cookies,
                "size": size,
            }
        )
    print(json.dumps(tickets, ensure_ascii=False))


if __name__ == "__main__":
    main()
