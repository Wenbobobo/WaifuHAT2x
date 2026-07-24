from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib
from urllib.request import Request, urlopen
import zipfile


CHUNK = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _download_range(url: str, segment: Path, start: int, end: int) -> None:
    expected = end - start + 1
    existing = segment.stat().st_size if segment.exists() else 0
    if existing > expected:
        segment.unlink()
        existing = 0
    if existing == expected:
        return
    for attempt in range(4):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "WaifuHAT2x/1.0",
                    "Range": f"bytes={start + existing}-{end}",
                },
            )
            with urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"Server ignored range request: HTTP {response.status}")
                with segment.open("ab") as output:
                    shutil.copyfileobj(response, output, length=CHUNK)
            if segment.stat().st_size != expected:
                raise RuntimeError(
                    f"Incomplete range {start}-{end}: {segment.stat().st_size}/{expected} bytes"
                )
            return
        except Exception:
            if attempt == 3:
                raise
            existing = segment.stat().st_size if segment.exists() else 0
            time.sleep(1 + attempt)


def download_url(url: str, target: Path, connections: int = 8) -> None:
    partial = target.with_suffix(target.suffix + ".part")
    request = Request(url, headers={"User-Agent": "WaifuHAT2x/1.0"}, method="HEAD")
    with urlopen(request, timeout=60) as response:
        resolved_url = response.geturl()
        total = int(response.headers.get("Content-Length", "0"))
        ranged = response.headers.get("Accept-Ranges", "").lower() == "bytes"
    if not ranged or total <= 0 or connections == 1:
        request = Request(url, headers={"User-Agent": "WaifuHAT2x/1.0"})
        with urlopen(request, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=CHUNK)
        partial.replace(target)
        return

    workers = min(connections, max(1, (total + 32 * 1024**2 - 1) // (32 * 1024**2)))
    span = (total + workers - 1) // workers
    ranges = [(index, index * span, min(total - 1, (index + 1) * span - 1)) for index in range(workers)]
    segments = [partial.with_name(partial.name + f".{index:02d}") for index, _, _ in ranges]
    # Reuse an interrupted old single-stream download as the beginning of range 0.
    if partial.exists() and not segments[0].exists() and partial.stat().st_size <= ranges[0][2] + 1:
        partial.replace(segments[0])
    print(f"  parallel download: {total / 1024**2:.1f} MiB in {workers} resumable ranges")
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="model-download") as executor:
        futures = [
            executor.submit(_download_range, resolved_url, segment, start, end)
            for segment, (_, start, end) in zip(segments, ranges, strict=True)
        ]
        for index, future in enumerate(futures, start=1):
            future.result()
            print(f"  range {index}/{workers} complete", flush=True)
    with partial.open("wb") as output:
        for segment in segments:
            with segment.open("rb") as source:
                shutil.copyfileobj(source, output, length=CHUNK)
    if partial.stat().st_size != total:
        raise RuntimeError(f"Assembled download has {partial.stat().st_size}/{total} bytes")
    for segment in segments:
        segment.unlink()
    partial.replace(target)


def _find_aria2() -> Path | None:
    # The Windows helper owns optional Windows aria2 integration. In WSL, use a
    # native binary when available and otherwise use the portable gdown path.
    if found := shutil.which("aria2c"):
        return Path(found)
    return None


def _gdrive_ticket_once(file_id: str) -> tuple[str, str, int]:
    from gdown.download import _get_session, get_url_from_gdrive_confirmation

    user_agent = "WaifuHAT2x/1.0"
    session = _get_session(proxy=None, use_cookies=True, user_agent=user_agent)
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        while True:
            response = session.get(url, stream=True, timeout=120)
            if "Content-Disposition" in response.headers:
                final_url = response.url
                size = int(response.headers.get("Content-Length", "0"))
                response.close()
                cookies = "; ".join(f"{cookie.name}={cookie.value}" for cookie in session.cookies)
                return final_url, cookies, size
            contents = response.text
            response.close()
            url = get_url_from_gdrive_confirmation(contents)
    finally:
        session.close()


def _gdrive_ticket(file_id: str) -> tuple[str, str, int]:
    for attempt in range(5):
        try:
            return _gdrive_ticket_once(file_id)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 + attempt * 2)
    raise AssertionError("unreachable")


def download_gdrive(file_id: str, target: Path) -> None:
    partial = target.with_suffix(target.suffix + ".part")

    def download_with_gdown() -> None:
        import gdown

        result = gdown.download(id=file_id, output=str(partial), quiet=False, resume=True)
        if not result:
            raise RuntimeError(f"Google Drive download failed: {file_id}")
        partial.replace(target)

    aria2 = _find_aria2()
    if aria2 is None:
        download_with_gdown()
        return

    # gdown uses randomized suffixes for interrupted temporary files. Preserve
    # the largest one and let aria2 continue from those already downloaded bytes.
    legacy = sorted(
        partial.parent.glob(partial.name + "*.part"),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    if not partial.exists() and legacy:
        legacy[0].replace(partial)
    url, cookies, expected = _gdrive_ticket(file_id)
    if aria2.suffix.lower() == ".exe":
        output_directory = subprocess.run(
            ["wslpath", "-w", str(partial.parent)],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    else:
        output_directory = str(partial.parent)
    print(f"  aria2: {expected / 1024**2:.1f} MiB in 8 resumable ranges")
    command = [
        str(aria2),
        "--continue=true",
        "--max-connection-per-server=8",
        "--split=8",
        "--min-split-size=1M",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--summary-interval=5",
        f"--dir={output_directory}",
        f"--out={partial.name}",
        f"--header=Cookie: {cookies}",
        "--user-agent=WaifuHAT2x/1.0",
        url,
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"  aria2 unavailable ({type(exc).__name__}); resuming with gdown", flush=True)
        partial.with_name(partial.name + ".aria2").unlink(missing_ok=True)
        download_with_gdown()
        return
    if expected and partial.stat().st_size != expected:
        raise RuntimeError(
            f"Google Drive size mismatch for {target.name}: {partial.stat().st_size}/{expected}"
        )
    for extra in legacy:
        if extra.exists() and extra != partial:
            extra.unlink()
    control = partial.with_name(partial.name + ".aria2")
    control.unlink(missing_ok=True)
    result = str(partial)
    if not result:
        raise RuntimeError(f"Google Drive download failed: {file_id}")
    partial.replace(target)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (root / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe zip member: {member.filename}")
        bundle.extractall(root)


def archive_manifest(archive: Path) -> dict[str, str]:
    """Derive per-file hashes from an archive whose outer SHA-256 was verified."""
    result: dict[str, str] = {}
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.is_dir():
                continue
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe zip member: {member.filename}")
            digest = hashlib.sha256()
            with bundle.open(member) as source:
                while block := source.read(CHUNK):
                    digest.update(block)
            result[relative.as_posix()] = digest.hexdigest()
    return result


def extraction_matches(destination: Path, expected: dict[str, str], marker: Path) -> bool:
    if not destination.is_dir() or not marker.is_file():
        return False
    actual_files = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path != marker
    }
    if actual_files != set(expected):
        return False
    return all(sha256(destination / relative) == digest for relative, digest in expected.items())


def contained(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise RuntimeError(f"Manifest path escapes model directory: {candidate}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="model_sources.toml")
    parser.add_argument("--models", default="models")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    root = Path(args.models).resolve()
    downloads = root / ".downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("rb") as handle:
        manifest = tomllib.load(handle)

    calculated: list[tuple[str, str]] = []
    for section, entries in manifest.items():
        for name, item in entries.items():
            is_archive = section == "archives"
            target = contained(
                root,
                downloads / item["filename"] if is_archive else root / item["filename"],
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = item.get("sha256", "").lower()
            if target.exists() and (not expected or sha256(target) == expected):
                print(f"Present: {target.relative_to(root)}")
            else:
                if target.exists():
                    target.unlink()
                print(f"Downloading {name} -> {target.relative_to(root)}")
                if item["kind"] == "gdrive":
                    download_gdrive(item["id"], target)
                else:
                    download_url(item["url"], target)
            actual = sha256(target)
            if expected and actual != expected:
                target.unlink(missing_ok=True)
                raise RuntimeError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
            calculated.append((actual, target.relative_to(root).as_posix()))
            if is_archive:
                extract_to = contained(root, root / item["extract_to"])
                marker = extract_to / f".{actual}.extracted"
                members = archive_manifest(target)
                for relative, digest in members.items():
                    calculated.append((digest, (extract_to.relative_to(root) / relative).as_posix()))
                staging = contained(root, extract_to.with_name(f".{extract_to.name}.{actual}.staging"))
                backup = contained(root, extract_to.with_name(f".{extract_to.name}.backup"))
                if not extract_to.exists() and backup.exists():
                    backup.replace(extract_to)
                if not extraction_matches(extract_to, members, marker):
                    if staging.exists():
                        shutil.rmtree(staging)
                    print(f"Extracting {target.name} -> {extract_to.relative_to(root)}")
                    safe_extract(target, staging)
                    staging_marker = staging / marker.name
                    staging_marker.touch()
                    if not extraction_matches(staging, members, staging_marker):
                        shutil.rmtree(staging)
                        raise RuntimeError(f"Extracted model verification failed: {target.name}")
                    if backup.exists():
                        shutil.rmtree(backup)
                    if extract_to.exists():
                        extract_to.replace(backup)
                    try:
                        staging.replace(extract_to)
                    except Exception:
                        if backup.exists() and not extract_to.exists():
                            backup.replace(extract_to)
                        raise
                    if backup.exists():
                        shutil.rmtree(backup)
                else:
                    print(f"Verified: {extract_to.relative_to(root)} ({len(members)} files)")

    checksum_file = root / "checksums.sha256"
    checksum_file.write_text("".join(f"{digest}  {path}\n" for digest, path in calculated), encoding="utf-8")
    print(f"Wrote {checksum_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Model setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
