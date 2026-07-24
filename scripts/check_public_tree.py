"""Reject local data and identity markers from the tracked public source tree."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_FILENAMES = {"config.toml", ".env"}
FORBIDDEN_SUFFIXES = {
    ".avif",
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".jxl",
    ".onnx",
    ".png",
    ".pth",
    ".tif",
    ".tiff",
    ".webp",
}
PATTERNS = {
    "local account or home path": re.compile(
        r"(?i)(?:/mnt/[a-z]/users/|[a-z]:[\\/](?:users|documents|desktop|appdata)[\\/])"
    ),
    "local project path": re.compile(
        r"(?i)(?:/mnt/[a-z]/projects/|[a-z]:[\\/]projects[\\/])"
    ),
    "production sync path": re.compile(r"(?i)synchronizing[\\/]manga"),
    "known local identity": re.compile(r"(?i)wen" + "bo"),
    "common GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "private key block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def tracked_paths() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [
        ROOT / Path(item.decode("utf-8"))
        for item in completed.stdout.split(b"\0")
        if item
    ]


def main() -> int:
    findings: list[str] = []
    for path in tracked_paths():
        relative = path.relative_to(ROOT)
        if not path.is_file():
            continue
        if path.name.lower() in FORBIDDEN_FILENAMES:
            findings.append(f"forbidden tracked filename: {relative}")
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(f"forbidden tracked artifact: {relative}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label}: {relative}")
    if findings:
        print("Public-tree check failed:", file=sys.stderr)
        print("\n".join(f"- {finding}" for finding in findings), file=sys.stderr)
        return 1
    print("Public-tree check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
