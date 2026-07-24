from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from urllib.request import Request, urlopen


VERSION = "0.12.0"
ARCHIVE_NAME = "jxl-debs-amd64-ubuntu-24.04.tar"
URL = f"https://github.com/libjxl/libjxl/releases/download/v{VERSION}/{ARCHIVE_NAME}"
SHA256 = "01a20d64069b6e760e1ecacfe39bbb7dba5057c71ef5c17c98fee35b37436df9"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(target: Path) -> None:
    partial = target.with_suffix(target.suffix + ".part")
    request = Request(URL, headers={"User-Agent": "WaifuHAT2x/0.1"})
    with urlopen(request) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    partial.replace(target)


def child_environment(root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    library = root / "usr/lib/x86_64-linux-gnu"
    previous = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = f"{library}:{previous}" if previous else str(library)
    return environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime_root.expanduser().resolve()
    downloads = runtime / "downloads"
    install_root = runtime / f"jxl-{VERSION}"
    marker = install_root / ".installed"
    cjxl = install_root / "usr/bin/cjxl"
    djxl = install_root / "usr/bin/djxl"
    downloads.mkdir(parents=True, exist_ok=True)

    expected_marker = f"v{VERSION}\n{SHA256}\n"
    if marker.is_file() and marker.read_text(encoding="utf-8") == expected_marker:
        if cjxl.is_file() and djxl.is_file():
            environment = child_environment(install_root)
            subprocess.run([cjxl, "--version"], check=True, env=environment)
            subprocess.run([djxl, "--version"], check=True, env=environment)
            return

    archive = downloads / ARCHIVE_NAME
    if not archive.is_file() or file_hash(archive) != SHA256:
        archive.unlink(missing_ok=True)
        print(f"Downloading libjxl {VERSION} official Ubuntu 24.04 bundle...")
        download(archive)
    actual = file_hash(archive)
    if actual != SHA256:
        archive.unlink(missing_ok=True)
        raise RuntimeError(f"libjxl SHA-256 mismatch: expected {SHA256}, got {actual}")

    staging = runtime / f".jxl-{VERSION}-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    debs = staging / "debs"
    debs.mkdir()
    with tarfile.open(archive, "r:") as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or not member.name.endswith(".deb"):
                continue
            name = Path(member.name).name
            target = debs / name
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"Unable to read {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)

    extracted = staging / "root"
    extracted.mkdir()
    for package in sorted(debs.glob("*.deb")):
        subprocess.run(["dpkg-deb", "-x", package, extracted], check=True)
    for executable in ("cjxl", "djxl"):
        if not (extracted / "usr/bin" / executable).is_file():
            raise RuntimeError(f"Official libjxl bundle did not contain usr/bin/{executable}")

    if install_root.exists():
        shutil.rmtree(install_root)
    extracted.replace(install_root)
    marker.write_text(expected_marker, encoding="utf-8")
    shutil.rmtree(staging)
    environment = child_environment(install_root)
    subprocess.run([cjxl, "--version"], check=True, env=environment)
    subprocess.run([djxl, "--version"], check=True, env=environment)


if __name__ == "__main__":
    main()
