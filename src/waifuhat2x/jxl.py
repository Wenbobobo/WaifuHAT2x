from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

import numpy as np
from PIL import Image

from .config import JxlConfig
from .images import resize_linear_light


JXL_VERSION = "0.12.0"


@dataclass(frozen=True)
class JxlStats:
    seconds: float
    bytes: int
    temporary: Path | None = None
    sha256: str | None = None
    postprocess_seconds: float = 0.0
    cjxl_seconds: float = 0.0
    djxl_seconds: float = 0.0
    candidate_hash_seconds: float = 0.0
    commit_seconds: float = 0.0
    service_interval_ns: tuple[int, int] | None = None
    resize_interval_ns: tuple[int, int] | None = None
    cjxl_interval_ns: tuple[int, int] | None = None
    djxl_interval_ns: tuple[int, int] | None = None
    candidate_hash_interval_ns: tuple[int, int] | None = None
    commit_interval_ns: tuple[int, int] | None = None


class JxlEncoder:
    def __init__(self, config: JxlConfig) -> None:
        runtime = Path(
            os.environ.get("WAIFUHAT_RUNTIME_ROOT", Path.home() / ".local/share/waifuhat2x")
        ).expanduser()
        self.root = runtime / f"jxl-{JXL_VERSION}"
        self.cjxl = self.root / "usr/bin/cjxl"
        self.djxl = self.root / "usr/bin/djxl"
        self.config = config
        if not self.cjxl.is_file() or not self.djxl.is_file():
            raise FileNotFoundError(f"JPEG XL {JXL_VERSION} tools are missing; run install.bat: {self.root}")

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        library = self.root / "usr/lib/x86_64-linux-gnu"
        previous = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = f"{library}:{previous}" if previous else str(library)
        return environment

    def version(self) -> str:
        result = subprocess.run(
            [self.cjxl, "--version"], check=True, text=True, capture_output=True, env=self._environment()
        )
        return (result.stdout or result.stderr).strip()

    def _temporary(self, destination: Path, temporary: Path | None) -> Path:
        candidate = temporary or destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.part"
        )
        if candidate.parent.resolve() != destination.parent.resolve() or candidate == destination:
            raise ValueError("JXL temporary file must be a distinct sibling of the destination")
        candidate.unlink(missing_ok=True)
        return candidate

    def verify(
        self, path: Path, expected_width: int | None = None, expected_height: int | None = None
    ) -> tuple[int, int]:
        verified = subprocess.run(
            [self.djxl, path, "--disable_output", "-v"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=self._environment(),
        )
        message = verified.stderr.decode("utf-8", errors="replace")
        if verified.returncode:
            raise RuntimeError(f"djxl verification failed: {message.strip()}")
        dimensions = re.search(r"(?m)^(\d+) x (\d+),", message)
        if dimensions is None:
            raise RuntimeError("djxl verification did not report decoded dimensions")
        width, height = map(int, dimensions.groups())
        if expected_width is not None and (width, height) != (expected_width, expected_height):
            raise RuntimeError(
                f"JXL dimension mismatch: expected {expected_width}x{expected_height}, got {width}x{height}"
            )
        return width, height

    @staticmethod
    def _durable_replace(temporary: Path, destination: Path) -> None:
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        except OSError:
            return
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    def commit(self, temporary: Path, destination: Path, expected_sha256: str) -> None:
        digest = hashlib.sha256()
        with temporary.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("Verified JXL candidate changed before commit")
        self._durable_replace(temporary, destination)

    def encode(
        self,
        array: np.ndarray,
        destination: Path,
        *,
        temporary: Path | None = None,
        distance: float | None = None,
        finalize: bool = True,
    ) -> JxlStats:
        if array.dtype != np.uint8 or array.ndim not in (2, 3):
            raise ValueError("JXL input must be an 8-bit grayscale or RGB array")
        if array.ndim == 3 and array.shape[2] != 3:
            raise ValueError("JXL color input must have exactly three channels")
        pixels = np.ascontiguousarray(array)
        height, width = pixels.shape[:2]
        magic = "P5" if pixels.ndim == 2 else "P6"
        header = f"{magic}\n{width} {height}\n255\n".encode("ascii")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._temporary(destination, temporary)
        selected_distance = self.config.distance if distance is None else distance
        command = [
            str(self.cjxl),
            "-",
            str(temporary),
            "--streaming_input",
            "-d",
            f"{selected_distance:g}",
            "-e",
            str(self.config.effort),
            "--num_threads",
            str(self.config.threads),
        ]
        if selected_distance == 0:
            command.extend(["-m", "1"])
        service_started_ns = time.perf_counter_ns()
        cjxl_started_ns = service_started_ns
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=self._environment(),
        )
        assert process.stdin is not None
        try:
            process.stdin.write(header)
            process.stdin.write(memoryview(pixels).cast("B"))
            process.stdin.close()
            stderr = process.stderr.read() if process.stderr is not None else b""
            return_code = process.wait()
        except Exception:
            process.kill()
            process.wait()
            temporary.unlink(missing_ok=True)
            raise
        cjxl_ended_ns = time.perf_counter_ns()
        if return_code:
            temporary.unlink(missing_ok=True)
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"cjxl failed with code {return_code}: {message}")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("cjxl returned success but produced no output")

        djxl_interval_ns: tuple[int, int] | None = None
        if self.config.verify_decode:
            try:
                djxl_started_ns = time.perf_counter_ns()
                self.verify(temporary, width, height)
                djxl_interval_ns = (djxl_started_ns, time.perf_counter_ns())
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        hash_started_ns = time.perf_counter_ns()
        digest = self._file_sha256(temporary)
        hash_interval_ns = (hash_started_ns, time.perf_counter_ns())
        size = temporary.stat().st_size
        commit_interval_ns: tuple[int, int] | None = None
        if finalize:
            commit_started_ns = time.perf_counter_ns()
            self.commit(temporary, destination, digest)
            commit_interval_ns = (commit_started_ns, time.perf_counter_ns())
        service_ended_ns = time.perf_counter_ns()
        common = {
            "seconds": (service_ended_ns - service_started_ns) / 1_000_000_000,
            "bytes": size,
            "cjxl_seconds": (cjxl_ended_ns - cjxl_started_ns) / 1_000_000_000,
            "djxl_seconds": (
                (djxl_interval_ns[1] - djxl_interval_ns[0]) / 1_000_000_000
                if djxl_interval_ns is not None
                else 0.0
            ),
            "candidate_hash_seconds": (
                hash_interval_ns[1] - hash_interval_ns[0]
            )
            / 1_000_000_000,
            "commit_seconds": (
                (commit_interval_ns[1] - commit_interval_ns[0]) / 1_000_000_000
                if commit_interval_ns is not None
                else 0.0
            ),
            "service_interval_ns": (service_started_ns, service_ended_ns),
            "cjxl_interval_ns": (cjxl_started_ns, cjxl_ended_ns),
            "djxl_interval_ns": djxl_interval_ns,
            "candidate_hash_interval_ns": hash_interval_ns,
            "commit_interval_ns": commit_interval_ns,
        }
        if finalize:
            return JxlStats(**common)
        return JxlStats(temporary=temporary, sha256=digest, **common)

    def encode_resized(
        self,
        array: np.ndarray,
        destination: Path,
        output_width: int,
        output_height: int,
        *,
        linear_light: bool,
        temporary: Path | None = None,
        distance: float | None = None,
        finalize: bool = True,
    ) -> JxlStats:
        """Resize on the JXL worker, then encode the resulting pixels.

        Keeping both operations in the same bounded worker queue lets the main
        thread submit a native-scale SR result and immediately start the next
        GPU inference.  The future owns ``array`` until this method completes,
        so queue depth remains the upper bound on pending full-resolution
        arrays.
        """
        if array.dtype != np.uint8 or array.ndim not in (2, 3):
            raise ValueError("JXL input must be an 8-bit grayscale or RGB array")
        if array.ndim == 3 and array.shape[2] != 3:
            raise ValueError("JXL color input must have exactly three channels")
        if output_width < 1 or output_height < 1:
            raise ValueError("JXL output dimensions must be positive")

        service_started_ns = time.perf_counter_ns()
        postprocess_seconds = 0.0
        resize_interval_ns: tuple[int, int] | None = None
        pixels = array
        if (array.shape[1], array.shape[0]) != (output_width, output_height):
            resize_started_ns = time.perf_counter_ns()
            if linear_light:
                pixels = resize_linear_light(array, output_width, output_height)
            else:
                pixels = np.asarray(
                    Image.fromarray(array).resize(
                        (output_width, output_height), Image.Resampling.LANCZOS
                    )
                )
            resize_interval_ns = (resize_started_ns, time.perf_counter_ns())
            postprocess_seconds = (
                resize_interval_ns[1] - resize_interval_ns[0]
            ) / 1_000_000_000

        stats = self.encode(
            pixels,
            destination,
            temporary=temporary,
            distance=distance,
            finalize=finalize,
        )
        service_ended_ns = (
            stats.service_interval_ns[1]
            if stats.service_interval_ns is not None
            else time.perf_counter_ns()
        )
        return JxlStats(
            seconds=stats.seconds,
            bytes=stats.bytes,
            temporary=stats.temporary,
            sha256=stats.sha256,
            postprocess_seconds=postprocess_seconds,
            cjxl_seconds=stats.cjxl_seconds,
            djxl_seconds=stats.djxl_seconds,
            candidate_hash_seconds=stats.candidate_hash_seconds,
            commit_seconds=stats.commit_seconds,
            service_interval_ns=(service_started_ns, service_ended_ns),
            resize_interval_ns=resize_interval_ns,
            cjxl_interval_ns=stats.cjxl_interval_ns,
            djxl_interval_ns=stats.djxl_interval_ns,
            candidate_hash_interval_ns=stats.candidate_hash_interval_ns,
            commit_interval_ns=stats.commit_interval_ns,
        )

    def encode_lossless_jpeg(
        self,
        source_bytes: bytes,
        destination: Path,
        *,
        temporary: Path | None = None,
        finalize: bool = True,
    ) -> JxlStats:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._temporary(destination, temporary)
        service_started_ns = time.perf_counter_ns()
        cjxl_started_ns = service_started_ns
        result = subprocess.run(
            [
                self.cjxl,
                "-",
                temporary,
                "--lossless_jpeg=1",
                "--effort",
                str(self.config.effort),
                "--num_threads",
                str(self.config.threads),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            input=source_bytes,
            env=self._environment(),
        )
        cjxl_ended_ns = time.perf_counter_ns()
        if result.returncode:
            temporary.unlink(missing_ok=True)
            message = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"lossless JPEG reconstruction failed: {message}")
        try:
            djxl_started_ns = time.perf_counter_ns()
            self.verify(temporary)
            djxl_interval_ns = (djxl_started_ns, time.perf_counter_ns())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        hash_started_ns = time.perf_counter_ns()
        digest = self._file_sha256(temporary)
        hash_interval_ns = (hash_started_ns, time.perf_counter_ns())
        size = temporary.stat().st_size
        commit_interval_ns: tuple[int, int] | None = None
        if finalize:
            commit_started_ns = time.perf_counter_ns()
            self.commit(temporary, destination, digest)
            commit_interval_ns = (commit_started_ns, time.perf_counter_ns())
        service_ended_ns = time.perf_counter_ns()
        common = {
            "seconds": (service_ended_ns - service_started_ns) / 1_000_000_000,
            "bytes": size,
            "cjxl_seconds": (cjxl_ended_ns - cjxl_started_ns) / 1_000_000_000,
            "djxl_seconds": (
                djxl_interval_ns[1] - djxl_interval_ns[0]
            )
            / 1_000_000_000,
            "candidate_hash_seconds": (
                hash_interval_ns[1] - hash_interval_ns[0]
            )
            / 1_000_000_000,
            "commit_seconds": (
                (commit_interval_ns[1] - commit_interval_ns[0]) / 1_000_000_000
                if commit_interval_ns is not None
                else 0.0
            ),
            "service_interval_ns": (service_started_ns, service_ended_ns),
            "cjxl_interval_ns": (cjxl_started_ns, cjxl_ended_ns),
            "djxl_interval_ns": djxl_interval_ns,
            "candidate_hash_interval_ns": hash_interval_ns,
            "commit_interval_ns": commit_interval_ns,
        }
        if finalize:
            return JxlStats(**common)
        return JxlStats(temporary=temporary, sha256=digest, **common)
