from __future__ import annotations

from dataclasses import dataclass, asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SourceFingerprint:
    size: int
    mtime_ns: int
    source_sha256: str
    source_root: str
    model_sha256: str
    pipeline_signature: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.data: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.data = {}

    @staticmethod
    def fingerprint(
        source: Path, source_root: Path, model_sha256: str, pipeline_signature: str
    ) -> SourceFingerprint:
        stat = source.stat()
        return SourceFingerprint(
            stat.st_size,
            stat.st_mtime_ns,
            file_sha256(source),
            str(source_root.resolve()),
            model_sha256,
            pipeline_signature,
        )

    @staticmethod
    def fingerprint_bytes(
        content: bytes,
        mtime_ns: int,
        source_root: Path,
        model_sha256: str,
        pipeline_signature: str,
    ) -> SourceFingerprint:
        return SourceFingerprint(
            len(content),
            mtime_ns,
            hashlib.sha256(content).hexdigest(),
            str(source_root.resolve()),
            model_sha256,
            pipeline_signature,
        )

    def matches(self, key: str, value: SourceFingerprint, output: Path) -> bool:
        if not output.is_file():
            return False
        record = self.data.get(key)
        if not isinstance(record, dict):
            return False
        expected = asdict(value)
        if any(record.get(field) != expected_value for field, expected_value in expected.items()):
            return False
        stat = output.stat()
        return (
            record.get("output_size") == stat.st_size
            and record.get("output_sha256") == file_sha256(output)
        )

    def owns_output(self, key: str, output: Path) -> bool:
        record = self.data.get(key)
        if not isinstance(record, dict) or output.is_symlink() or not output.is_file():
            return False
        stat = output.stat()
        return (
            record.get("output_size") == stat.st_size
            and record.get("output_sha256") == file_sha256(output)
        )

    def manages_output_path(self, key: str, output: Path) -> bool:
        """Return whether state explicitly associates ``key`` with this path.

        Older state records did not persist the destination.  They are accepted
        only while their output hash is still intact; the next successful run
        upgrades them to the explicit-path schema.
        """
        record = self.data.get(key)
        if not isinstance(record, dict):
            return False
        recorded = record.get("destination")
        if isinstance(recorded, str):
            relative = Path(recorded)
            if relative.is_absolute() or ".." in relative.parts:
                return False
            return (self.path.parent / relative).resolve() == output.resolve()
        return self.owns_output(key, output)

    def prepare_replace(
        self,
        key: str,
        value: SourceFingerprint,
        destination_relative: str,
        temporary_relative: str,
        details: Mapping[str, Any],
    ) -> None:
        self.data[key] = {
            **asdict(value),
            **details,
            "phase": "prepared",
            "destination": destination_relative,
            "temporary": temporary_relative,
        }

    def prepare_adopt(
        self,
        key: str,
        value: SourceFingerprint,
        destination_relative: str,
        output: Path,
        expected_sha256: str,
        details: Mapping[str, Any],
    ) -> None:
        if output.is_symlink() or not output.is_file():
            raise RuntimeError(f"Existing JXL is not a regular file: {output}")
        stat = output.stat()
        if file_sha256(output) != expected_sha256:
            raise RuntimeError(f"Existing JXL changed before adoption: {output}")
        self.data[key] = {
            **asdict(value),
            **details,
            "action": "adopt_existing_jxl",
            "phase": "output_ready",
            "destination": destination_relative,
            "output_size": stat.st_size,
            "output_sha256": expected_sha256,
            "verified": True,
        }

    def mark_output_ready(self, key: str, output: Path) -> None:
        record = self.data[key]
        if record.get("verified") is not True:
            raise RuntimeError("Cannot mark an unverified JXL candidate as output-ready")
        stat = output.stat()
        output_sha256 = file_sha256(output)
        if record.get("phase") == "encoded" and (
            record.get("candidate_size") != stat.st_size
            or record.get("candidate_sha256") != output_sha256
        ):
            raise RuntimeError("Committed JXL no longer matches the verified candidate")
        record.update(
            phase="output_ready",
            output_size=stat.st_size,
            output_sha256=output_sha256,
        )

    def mark_encoded(
        self, key: str, temporary: Path, expected_size: int, expected_sha256: str
    ) -> None:
        record = self.data[key]
        stat = temporary.stat()
        if stat.st_size != expected_size or file_sha256(temporary) != expected_sha256:
            raise RuntimeError("Verified JXL candidate changed before journal persistence")
        record.update(
            phase="encoded",
            verified=True,
            candidate_size=expected_size,
            candidate_sha256=expected_sha256,
        )

    def mark_committed(self, key: str) -> None:
        self.data[key].update(phase="committed", source_removed=True)

    def discard(self, key: str) -> None:
        self.data.pop(key, None)

    def record(self, key: str) -> dict[str, Any] | None:
        value = self.data.get(key)
        return value if isinstance(value, dict) else None

    def update(self, key: str, value: SourceFingerprint, output: Path) -> None:
        stat = output.stat()
        try:
            destination = output.resolve().relative_to(self.path.parent.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError(f"Managed output escapes state directory: {output}") from exc
        self.data[key] = {
            **asdict(value),
            "phase": "committed",
            "destination": destination,
            "output_size": stat.st_size,
            "output_sha256": file_sha256(output),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        if self.data:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(self.data, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        else:
            temporary.unlink(missing_ok=True)
            self.path.unlink(missing_ok=True)
        try:
            directory = os.open(self.path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        except OSError:
            return
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


class RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError("Another WaifuHAT2x process is already using this output directory") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid={os.getpid()}\n")
        self.handle.flush()
        return self

    def __exit__(self, *_: object) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
