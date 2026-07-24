from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from typing import Any
import uuid


SCHEMA_VERSION = 2
BENCHMARK_SCHEMA_VERSION = 1
REAL_HAT_THRESHOLD = 1000
STEADY_ROCTX_RANGE = "real_hat_steady_rounds"
OFFICIAL_REAL_HAT_CHECKPOINTS = {
    "Real_HAT_GAN_SRx4.pth": (
        "f5b1e3bbbb05147ca2beefcc715279cb647d7976cbda67d62ea7e6e20d5ffcc7"
    ),
    "Real_HAT_GAN_SRx4_sharper.pth": (
        "5800b67136006eb8cab3b4ed7c8d73b6a195bb18e6cc709b674f9aa069c00271"
    ),
}
MIN_END_TO_END_SHARE_PERCENT = 15.0
MIN_GPU_SHARE_PERCENT = 20.0
MIN_PROTOTYPE_SPEEDUP = 1.3
MIN_ESTIMATED_END_TO_END_GAIN_PERCENT = 3.0
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ROCM_72_PATTERN = re.compile(
    r"^\s*rocm_version\s*:\s*(7\.2(?:\.\d+)?)\s*$", re.IGNORECASE | re.MULTILINE
)
PROFILER_DATA_LOSS_PATTERN = re.compile(
    r"\b(?:dropped|lost)(?:\s+(?P<count>\d+))?\s+"
    r"(?:record|records|event|events)\b|"
    r"\bbuffer\s+(?:overflow|overrun)\b|\btrace\s+truncat(?:ed|ion)\b",
    re.IGNORECASE,
)
PROFILER_WARNING_PATTERN = re.compile(r"\bwarn(?:ing)?\b", re.IGNORECASE)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite value greater than 0")
    return parsed


def safe_name(value: str, limit: int = 80) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in value
    ).strip("-._")
    return (cleaned or "native-rocprof")[:limit]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return is_relative_to(left_resolved, right_resolved) or is_relative_to(
        right_resolved, left_resolved
    )


def validate_sha256(value: Any, *, label: str) -> str:
    digest = str(value)
    if not SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"Invalid SHA-256 for {label}: {value!r}")
    return digest


def snapshot_inputs(
    manifest: Path, selected_pages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    records = [("manifest", manifest)] + [
        (f"page:{page['index']}", Path(str(page["path"]))) for page in selected_pages
    ]
    snapshot: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for role, raw_path in records:
        path = raw_path.resolve()
        if path in seen:
            raise ValueError(f"Duplicate isolated input path: {path}")
        seen.add(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Unsafe or missing isolated input: {path}")
        snapshot.append(
            {
                "role": role,
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return snapshot


def verify_input_snapshot(
    expected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actual: list[dict[str, Any]] = []
    for record in expected:
        path = Path(str(record["path"])).resolve()
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Isolated profiler input disappeared: {path}")
        observed = {
            "role": str(record["role"]),
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        if observed != record:
            raise RuntimeError(
                f"Isolated profiler input changed during profiling: {path}"
            )
        actual.append(observed)
    return actual


def validate_isolation(
    *,
    production_root: Path,
    manifest: Path,
    run_root: Path,
    selected_pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    production = production_root.resolve()
    manifest_root = manifest.resolve().parent
    output = run_root.resolve()
    if not production.is_dir():
        raise ValueError(
            f"--production-root must be an existing directory: {production}"
        )
    checks = [
        ("production_root", production, "manifest_root", manifest_root),
        ("production_root", production, "profile_output", output),
        ("manifest_root", manifest_root, "profile_output", output),
    ]
    for page in selected_pages:
        checks.append(
            (
                "production_root",
                production,
                f"isolated_page:{page['index']}",
                Path(str(page["path"])).resolve(),
            )
        )
    proof: list[dict[str, Any]] = []
    for left_name, left, right_name, right in checks:
        overlap = paths_overlap(left, right)
        proof.append(
            {
                "left": left_name,
                "left_path": str(left),
                "right": right_name,
                "right_path": str(right),
                "overlap": overlap,
            }
        )
        if overlap:
            raise ValueError(
                f"Isolation violation: {left_name} overlaps {right_name}: "
                f"{left} <-> {right}"
            )
    return proof


def validate_rocprof_version(
    completed: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"rocprofv3 --version failed with code {completed.returncode}: "
            f"{stderr or stdout}"
        )
    combined = "\n".join(part for part in (stdout, stderr) if part)
    match = ROCM_72_PATTERN.search(combined)
    if match is None:
        raise RuntimeError(
            "rocprofv3 --version does not report the required ROCm 7.2.x runtime"
        )
    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": completed.returncode,
        "rocm_version": match.group(1),
    }


def build_roctx_environment(
    rocprof_invocation: Path,
    *,
    base_environment: dict[str, str] | None = None,
    python_version: tuple[int, int] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    environment = dict(base_environment if base_environment is not None else os.environ)
    invocation = rocprof_invocation.absolute()
    configured_root = environment.get("ROCM_PATH")
    rocm_root = (
        Path(configured_root).resolve()
        if configured_root
        else invocation.parent.parent.resolve()
    )
    version = python_version or (sys.version_info.major, sys.version_info.minor)
    site_packages = (
        rocm_root / "lib" / f"python{version[0]}.{version[1]}" / "site-packages"
    ).resolve()
    if not (site_packages / "roctx").is_dir():
        raise RuntimeError(
            "ROCm 7.2 ROCTx Python bindings are unavailable for the active Python: "
            f"{site_packages / 'roctx'}"
        )
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(site_packages)
        if not existing
        else os.pathsep.join((str(site_packages), existing))
    )
    return environment, {
        "rocm_root": str(rocm_root),
        "roctx_site_packages": str(site_packages),
    }


def validate_roctx_preflight(
    completed: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    if completed.returncode != 0:
        raise RuntimeError(
            "ROCTx Python preflight failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    if completed.stdout.strip() != "roctx-control-ok":
        raise RuntimeError("ROCTx Python preflight returned an unexpected result")
    return {
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "exit_code": completed.returncode,
    }


def assert_native_linux(
    *,
    system: str | None = None,
    release: str | None = None,
    version: str | None = None,
    environment: dict[str, str] | None = None,
) -> None:
    actual_system = system if system is not None else platform.system()
    actual_release = release if release is not None else platform.release()
    actual_version = version if version is not None else platform.version()
    actual_environment = environment if environment is not None else dict(os.environ)
    marker = f"{actual_release} {actual_version}".lower()
    wsl_environment = any(
        actual_environment.get(name) for name in ("WSL_DISTRO_NAME", "WSL_INTEROP")
    )
    if actual_system != "Linux":
        raise RuntimeError("rocprofv3 profiling is restricted to native Linux")
    if "microsoft" in marker or "wsl" in marker or wsl_environment:
        raise RuntimeError(
            "rocprofv3 profiling is disabled under WSL; use native Linux or a "
            "remote native ROCm host"
        )


def validate_checkpoint(path: Path, expected_name: str) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.name != expected_name or not resolved.is_file():
        raise ValueError(f"Expected checkpoint {expected_name}: {resolved}")
    digest = sha256_file(resolved)
    expected = OFFICIAL_REAL_HAT_CHECKPOINTS[expected_name]
    if digest != expected:
        raise ValueError(
            f"Official checkpoint SHA-256 mismatch for {expected_name}: {digest}"
        )
    return {"path": str(resolved), "sha256": digest, "bytes": resolved.stat().st_size}


def validate_isolated_manifest(
    path: Path, page_indexes: list[int]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path.is_symlink():
        raise ValueError(f"Manifest must not be a symlink: {path}")
    manifest_path = path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_root = manifest_path.parent
    payload = read_json(manifest_path)
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "real_hat_representative_manifest"
    ):
        raise ValueError(
            "Only an isolated Real-HAT representative manifest is accepted"
        )
    if len(page_indexes) != len(set(page_indexes)):
        raise ValueError("--page-indexes must not contain duplicates")
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        raise ValueError("Manifest pages must be a list")
    pages: dict[int, dict[str, Any]] = {}
    for page in raw_pages:
        if not isinstance(page, dict):
            raise ValueError("Each manifest page must be an object")
        index = int(page["index"])
        if index in pages:
            raise ValueError(f"Manifest contains duplicate page index: {index}")
        pages[index] = page
    missing = [index for index in page_indexes if index not in pages]
    if missing:
        raise ValueError(f"Manifest has no page indexes: {missing}")
    selected: list[dict[str, Any]] = []
    routes: set[str] = set()
    for index in page_indexes:
        page = pages[index]
        copied_path = page.get("copied_path")
        if not isinstance(copied_path, str) or not copied_path:
            raise ValueError(f"Page {index} has no isolated copied_path")
        copied_candidate = manifest_root / copied_path
        if copied_candidate.is_symlink():
            raise ValueError(
                f"Isolated input must not be a symlink: {copied_candidate}"
            )
        copied = copied_candidate.resolve()
        if not is_relative_to(copied, manifest_root) or not copied.is_file():
            raise ValueError(
                f"Unsafe or missing isolated input for page {index}: {copied}"
            )
        digest = sha256_file(copied)
        if digest != page.get("copied_sha256"):
            raise ValueError(f"Isolated input hash changed for page {index}: {copied}")
        route = str(page.get("route"))
        if route not in {"normal", "sharper"}:
            raise ValueError(f"Invalid route for page {index}: {route!r}")
        routes.add(route)
        selected.append(
            {
                "index": index,
                "route": route,
                "path": str(copied),
                "sha256": digest,
                "bytes": copied.stat().st_size,
            }
        )
    if routes != {"normal", "sharper"}:
        raise ValueError(
            "The profiling subset must exercise both official Real-HAT models"
        )
    return payload, selected


def build_profile_command(
    *,
    rocprofv3: Path,
    trace_root: Path,
    benchmark_script: Path,
    manifest: Path,
    page_indexes: list[int],
    normal_model: Path,
    sharper_model: Path,
    benchmark_output_root: Path,
    tile: int | None,
    adaptive_tiles: list[int] | None,
    overlap: int,
    rounds: int,
    warmups_per_model: int,
    warmup_crop: int,
) -> list[str]:
    command = [
        str(rocprofv3),
        "--kernel-trace",
        "--marker-trace",
        "--selected-regions",
        "--stats",
        "--output-config",
        "--output-format",
        "csv",
        "json",
        "--output-directory",
        str(trace_root),
        "--output-file",
        "real-hat",
        "--",
        sys.executable,
        str(benchmark_script),
        "--manifest",
        str(manifest),
        "--page-indexes",
        *(str(index) for index in page_indexes),
        "--normal-model",
        str(normal_model),
        "--sharper-model",
        str(sharper_model),
        "--threshold",
        str(REAL_HAT_THRESHOLD),
    ]
    if adaptive_tiles:
        command.extend(["--adaptive-tiles", *(str(tile) for tile in adaptive_tiles)])
    elif tile is not None:
        command.extend(["--tile", str(tile)])
    else:
        raise ValueError("A fixed tile or adaptive tile candidates are required")
    command.extend(
        [
            "--overlap",
            str(overlap),
            "--rounds",
            str(rounds),
            "--warmups-per-model",
            str(warmups_per_model),
            "--warmup-crop",
            str(warmup_crop),
            "--output-root",
            str(benchmark_output_root),
            "--run-name",
            "profiled",
            "--gpu-phase-timing",
            "--rocprof-selected-regions",
        ]
    )
    return command


def process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_process_group_exit(process_group_id: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while process_group_exists(process_group_id):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))
    return True


def terminate_process_group(
    process: subprocess.Popen[Any], *, grace_seconds: float = 10.0
) -> None:
    process_group_id = process.pid
    for selected_signal in (signal.SIGTERM, signal.SIGKILL):
        if not process_group_exists(process_group_id):
            break
        try:
            os.killpg(process_group_id, selected_signal)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            pass
        if wait_for_process_group_exit(process_group_id, grace_seconds):
            break
    if process_group_exists(process_group_id):
        raise RuntimeError(
            f"Unable to terminate profiler process group {process_group_id}"
        )
    try:
        process.wait(timeout=0)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Profiler leader {process.pid} survived process-group cleanup"
        ) from None


def run_process(
    command: list[str],
    timeout_seconds: float,
    *,
    environment: dict[str, str] | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> tuple[int, float]:
    if (stdout_path is None) != (stderr_path is None):
        raise ValueError("Profiler stdout and stderr paths must be supplied together")
    if stdout_path is not None and stdout_path.resolve() == stderr_path.resolve():
        raise ValueError("Profiler stdout and stderr paths must be distinct")
    started = time.perf_counter()
    stdout_handle = None
    stderr_handle = None
    try:
        if stdout_path is not None and stderr_path is not None:
            stdout_handle = stdout_path.open("xb")
            stderr_handle = stderr_path.open("xb")
        process = subprocess.Popen(
            command,
            start_new_session=True,
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except BaseException as exc:
            try:
                terminate_process_group(process)
            except BaseException as cleanup_error:
                raise RuntimeError(
                    f"Profiler process-group cleanup failed: {cleanup_error}"
                ) from exc
            if isinstance(exc, subprocess.TimeoutExpired):
                raise TimeoutError(
                    f"rocprofv3 exceeded {timeout_seconds:.0f} seconds"
                ) from None
            raise
        if process_group_exists(process.pid):
            terminate_process_group(process)
            raise RuntimeError(
                "rocprofv3 leader exited while descendants remained in its process group"
            )
        return return_code, time.perf_counter() - started
    finally:
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()


def inspect_profiler_logs(paths: list[Path]) -> dict[str, Any]:
    warnings: list[str] = []
    data_loss: list[str] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or unsafe profiler log: {path}")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped:
                continue
            diagnostic = f"{path.name}:{line_number}: {stripped[:500]}"
            loss_match = PROFILER_DATA_LOSS_PATTERN.search(stripped)
            if loss_match is not None and (
                loss_match.groupdict().get("count") is None
                or int(loss_match.group("count")) > 0
            ):
                data_loss.append(diagnostic)
            elif PROFILER_WARNING_PATTERN.search(stripped):
                warnings.append(diagnostic)
    if data_loss:
        raise RuntimeError(
            "rocprofv3 reported dropped or truncated trace data: "
            + " | ".join(data_loss[:20])
        )
    return {
        "data_loss_detected": False,
        "manual_review_required": bool(warnings),
        "warnings": warnings[:50],
    }


def artifact_inventory(root: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Profiler artifacts must not contain symlinks: {path}")
        if path.is_file():
            inventory.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    return inventory


def relative_artifact_record(root: Path, path: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if path.is_symlink() or not is_relative_to(resolved, resolved_root):
        raise ValueError(f"Unsafe profiler artifact: {path}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": resolved.relative_to(resolved_root).as_posix(),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def verify_profiler_control_traces(paths: list[Path]) -> dict[str, int]:
    calls = {"roctxProfilerResume": 0, "roctxProfilerPause": 0}
    ranges: list[tuple[int, int]] = []
    process_ids: set[int] = set()
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "Function",
                "Process_Id",
                "Start_Timestamp",
                "End_Timestamp",
            }
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"Unsupported marker trace columns in {path}")
            for row in reader:
                process_ids.add(int(row["Process_Id"]))
                function = row.get("Function")
                if function in calls:
                    calls[function] += 1
                if function == STEADY_ROCTX_RANGE:
                    start = int(row["Start_Timestamp"])
                    end = int(row["End_Timestamp"])
                    if end < start:
                        raise ValueError(f"Negative ROCTx range duration in {path}")
                    ranges.append((start, end))
    if calls != {"roctxProfilerResume": 1, "roctxProfilerPause": 1}:
        raise RuntimeError(
            "Expected exactly one global ROCTx resume/pause pair around steady rounds; "
            f"observed {calls}"
        )
    if len(ranges) != 1:
        raise RuntimeError(
            f"Expected exactly one {STEADY_ROCTX_RANGE!r} marker range; "
            f"observed {len(ranges)}"
        )
    if len(process_ids) != 1:
        raise RuntimeError(
            f"Expected exactly one profiled process; observed {sorted(process_ids)}"
        )
    return {
        **calls,
        "process_id": next(iter(process_ids)),
        "range_start_ns": ranges[0][0],
        "range_end_ns": ranges[0][1],
    }


def verify_kernels_within_selected_range(
    rows: list[dict[str, Any]], control: dict[str, int]
) -> None:
    start = int(control["range_start_ns"])
    end = int(control["range_end_ns"])
    outside = [
        row for row in rows if int(row["start_ns"]) < start or int(row["end_ns"]) > end
    ]
    if outside:
        raise RuntimeError(
            f"{len(outside)} kernel dispatches fall outside the steady ROCTx range"
        )


def run_profile(args: argparse.Namespace) -> int:
    assert_native_linux()
    rocprof_executable = shutil.which("rocprofv3")
    if not rocprof_executable:
        raise RuntimeError("rocprofv3 is unavailable; install ROCprofiler-SDK")
    rocprof_invocation = Path(rocprof_executable).absolute()
    rocprofv3 = rocprof_invocation.resolve()
    version_completed = subprocess.run(
        [str(rocprofv3), "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rocprof_version = validate_rocprof_version(version_completed)
    child_environment, roctx_paths = build_roctx_environment(rocprof_invocation)
    roctx_completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import roctx; "
                "assert callable(roctx.profilerResume); "
                "assert callable(roctx.profilerPause); "
                "assert callable(roctx.rangePush); "
                "assert callable(roctx.rangePop); "
                "print('roctx-control-ok')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=child_environment,
    )
    roctx_preflight = validate_roctx_preflight(roctx_completed)
    manifest_argument = args.manifest
    _manifest_payload, selected_pages = validate_isolated_manifest(
        manifest_argument, args.page_indexes
    )
    manifest = manifest_argument.resolve()
    models = {
        "normal": validate_checkpoint(args.normal_model, "Real_HAT_GAN_SRx4.pth"),
        "sharper": validate_checkpoint(
            args.sharper_model, "Real_HAT_GAN_SRx4_sharper.pth"
        ),
    }
    output_root = args.output_root.resolve()
    run_root = (output_root / safe_name(args.run_name)).resolve()
    production_root = args.production_root.resolve()
    isolation_proof = validate_isolation(
        production_root=production_root,
        manifest=manifest,
        run_root=run_root,
        selected_pages=selected_pages,
    )
    if run_root.exists():
        raise FileExistsError(run_root)
    input_snapshot_before = snapshot_inputs(manifest, selected_pages)
    trace_root = run_root / "trace"
    benchmark_output_root = run_root / "benchmark"
    trace_root.mkdir(parents=True)
    benchmark_script = (Path(__file__).parent / "benchmark_manifest_eager.py").resolve()
    command = build_profile_command(
        rocprofv3=rocprofv3,
        trace_root=trace_root,
        benchmark_script=benchmark_script,
        manifest=manifest,
        page_indexes=args.page_indexes,
        normal_model=args.normal_model.resolve(),
        sharper_model=args.sharper_model.resolve(),
        benchmark_output_root=benchmark_output_root,
        tile=args.tile,
        adaptive_tiles=args.adaptive_tiles,
        overlap=args.overlap,
        rounds=args.rounds,
        warmups_per_model=args.warmups_per_model,
        warmup_crop=args.warmup_crop,
    )
    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": "real_hat_native_rocprof_plan",
        "status": "running",
        "started_at": utc_now(),
        "environment": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": sys.version,
            "executable": sys.executable,
            "rocprofv3": str(rocprofv3),
            "rocprofv3_version": rocprof_version,
            "roctx_paths": roctx_paths,
            "roctx_preflight": roctx_preflight,
        },
        "production_root": str(production_root),
        "profile_output": str(run_root),
        "benchmark_script": {
            "path": str(benchmark_script),
            "sha256": sha256_file(benchmark_script),
            "bytes": benchmark_script.stat().st_size,
        },
        "manifest": {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
            "selected_pages": selected_pages,
        },
        "input_snapshot_before": input_snapshot_before,
        "input_snapshot_before_sha256": canonical_sha256(input_snapshot_before),
        "models": models,
        "configuration": {
            "precision": "bf16",
            "tile_mode": (
                "adaptive-estimated-work" if args.adaptive_tiles else "fixed"
            ),
            "tile": args.tile,
            "adaptive_tiles": (
                sorted(set(args.adaptive_tiles)) if args.adaptive_tiles else None
            ),
            "adaptive_selection_formula": (
                "ceil(width/tile) * ceil(height/tile) * (tile + 2*overlap)^2; "
                "minimum wins, ties use the smaller tile"
                if args.adaptive_tiles
                else None
            ),
            "overlap": args.overlap,
            "batch_tiles": 1,
            "device_assembly": True,
            "model_cache_size": 2,
            "threshold": REAL_HAT_THRESHOLD,
            "rounds": args.rounds,
            "warmups_per_model": args.warmups_per_model,
            "warmup_crop": args.warmup_crop,
            "gpu_phase_timing": True,
            "rocprof_selected_regions": True,
            "selected_region_scope": "all steady rounds only",
            "selected_region_name": STEADY_ROCTX_RANGE,
            "timeout_seconds": args.timeout_seconds,
        },
        "command": command,
        "safety": {
            "native_linux_required": True,
            "wsl_rejected": True,
            "isolated_manifest_required": True,
            "official_checkpoint_hashes_required": True,
            "explicit_production_root_required": True,
            "path_overlap_checks": isolation_proof,
            "input_hashes_checked_before_and_after": True,
            "rocm_7_2_required": True,
            "selected_regions_required": True,
        },
    }
    profile_plan_path = run_root / "profile_plan.json"
    write_json(profile_plan_path, plan)
    profiler_stdout = run_root / "rocprof.stdout.log"
    profiler_stderr = run_root / "rocprof.stderr.log"
    result = dict(plan)
    result.update(
        {
            "kind": "real_hat_native_rocprof_result",
            "profile_plan": relative_artifact_record(run_root, profile_plan_path),
        }
    )
    try:
        return_code, elapsed = run_process(
            command,
            args.timeout_seconds,
            environment=child_environment,
            stdout_path=profiler_stdout,
            stderr_path=profiler_stderr,
        )
        profiler_diagnostics = inspect_profiler_logs([profiler_stdout, profiler_stderr])
        input_snapshot_after = verify_input_snapshot(input_snapshot_before)
        result.update(
            {
                "finished_at": utc_now(),
                "return_code": return_code,
                "profiled_wall_seconds": elapsed,
                "profiler_diagnostics": profiler_diagnostics,
                "input_snapshot_after": input_snapshot_after,
                "input_snapshot_after_sha256": canonical_sha256(input_snapshot_after),
            }
        )
        kernel_traces = sorted(trace_root.rglob("*kernel_trace.csv"))
        marker_traces = sorted(trace_root.rglob("*marker_api_trace.csv"))
        benchmark_summary = benchmark_output_root / "profiled" / "batch_summary.json"
        if return_code != 0:
            raise RuntimeError(f"rocprofv3 child exited with code {return_code}")
        if not kernel_traces:
            raise RuntimeError("rocprofv3 produced no kernel_trace.csv")
        if not marker_traces:
            raise RuntimeError("rocprofv3 produced no marker_api_trace.csv")
        control_calls = verify_profiler_control_traces(marker_traces)
        verify_kernels_within_selected_range(
            read_kernel_rows(kernel_traces, process_id=control_calls["process_id"]),
            control_calls,
        )
        if not benchmark_summary.is_file():
            raise RuntimeError("The profiled benchmark produced no batch_summary.json")
        benchmark_payload = read_json(benchmark_summary)
        validate_profiled_benchmark_summary(benchmark_payload, result)
        result["status"] = "complete"
        result["roctx_control_calls"] = control_calls
        result["kernel_traces"] = [
            relative_artifact_record(run_root, path) for path in kernel_traces
        ]
        result["marker_traces"] = [
            relative_artifact_record(run_root, path) for path in marker_traces
        ]
        result["profiler_logs"] = [
            relative_artifact_record(run_root, profiler_stdout),
            relative_artifact_record(run_root, profiler_stderr),
        ]
        result["benchmark_summary"] = relative_artifact_record(
            run_root, benchmark_summary
        )
        result["artifacts"] = artifact_inventory(run_root)
        result["artifacts_sha256"] = canonical_sha256(result["artifacts"])
        profile_result_path = run_root / "profile_result.json"
        write_json(profile_result_path, result)
        completion = {
            "schema_version": SCHEMA_VERSION,
            "kind": "real_hat_native_rocprof_completion",
            "status": "complete",
            "finished_at": utc_now(),
            "profile_result": relative_artifact_record(run_root, profile_result_path),
            "profile_plan": result["profile_plan"],
            "artifacts_sha256": result["artifacts_sha256"],
            "kernel_traces": result["kernel_traces"],
            "marker_traces": result["marker_traces"],
            "profiler_logs": result["profiler_logs"],
            "profiler_diagnostics": result["profiler_diagnostics"],
            "benchmark_summary": result["benchmark_summary"],
            "input_snapshot_before_sha256": result["input_snapshot_before_sha256"],
            "input_snapshot_after_sha256": result["input_snapshot_after_sha256"],
        }
        write_json(run_root / "completion.json", completion)
        print(run_root)
        return 0
    except BaseException as exc:
        snapshot_error: BaseException | None = None
        try:
            input_snapshot_after = verify_input_snapshot(input_snapshot_before)
            result["input_snapshot_after"] = input_snapshot_after
            result["input_snapshot_after_sha256"] = canonical_sha256(
                input_snapshot_after
            )
        except BaseException as observed_error:
            snapshot_error = observed_error
        result.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "artifacts": artifact_inventory(run_root),
            }
        )
        result["artifacts_sha256"] = canonical_sha256(result["artifacts"])
        if snapshot_error is not None:
            result["input_snapshot_error"] = (
                f"{type(snapshot_error).__name__}: {snapshot_error}"
            )
        write_json(run_root / "profile_result.json", result)
        if snapshot_error is not None:
            raise RuntimeError(str(snapshot_error)) from exc
        raise


def read_kernel_rows(paths: list[Path], *, process_id: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows_before = len(rows)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "Kernel_Name",
                "Start_Timestamp",
                "End_Timestamp",
                "Stream_Id",
                "Agent_Id",
                "Queue_Id",
            }
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"Unsupported kernel trace columns in {path}")
            for raw in reader:
                start = int(raw["Start_Timestamp"])
                end = int(raw["End_Timestamp"])
                if end < start:
                    raise ValueError(f"Negative kernel duration in {path}")
                rows.append(
                    {
                        "kernel_name": raw["Kernel_Name"],
                        "start_ns": start,
                        "end_ns": end,
                        "duration_ns": end - start,
                        "stream_id": str(raw["Stream_Id"]).strip(),
                        "agent_id": str(raw["Agent_Id"]).strip(),
                        "queue_id": str(raw["Queue_Id"]).strip(),
                        "process_id": str(process_id),
                        "source": str(path),
                    }
                )
        if len(rows) == rows_before:
            raise ValueError(f"Kernel trace contains no dispatches: {path}")
    if not rows:
        raise ValueError("Kernel traces contain no dispatches")
    return rows


def benchmark_output_hashes(payload: dict[str, Any], *, label: str) -> dict[str, str]:
    steady = payload.get("steady_state")
    if not isinstance(steady, dict) or steady.get("pixel_deterministic") is not True:
        raise ValueError(f"{label} did not produce deterministic output pixels")
    raw_hashes = steady.get("unique_hashes_per_page")
    if not isinstance(raw_hashes, dict) or not raw_hashes:
        raise ValueError(f"{label} has no per-page output hashes")
    hashes: dict[str, str] = {}
    for page, values in raw_hashes.items():
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(f"{label} has non-unique hashes for page {page}")
        hashes[str(page)] = validate_sha256(values[0], label=f"{label} page {page}")
    return hashes


def normalize_kernel_patterns(patterns: list[str]) -> list[str]:
    normalized: list[str] = []
    for raw_pattern in patterns:
        pattern = str(raw_pattern).strip()
        if not pattern:
            raise ValueError("Kernel patterns must not be blank")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid kernel pattern {pattern!r}: {exc}") from exc
        if pattern in normalized:
            raise ValueError(f"Duplicate kernel pattern: {pattern!r}")
        normalized.append(pattern)
    if not normalized:
        raise ValueError("At least one kernel pattern is required")
    return normalized


def _shape(value: Any, *, label: str, dimensions: int | tuple[int, ...]) -> list[int]:
    allowed = (dimensions,) if isinstance(dimensions, int) else dimensions
    if not isinstance(value, list) or len(value) not in allowed:
        raise ValueError(f"{label} must have {allowed} dimensions")
    shape: list[int] = []
    for raw_dimension in value:
        if isinstance(raw_dimension, bool) or not isinstance(raw_dimension, int):
            raise ValueError(f"{label} dimensions must be integers")
        if raw_dimension < 1:
            raise ValueError(f"{label} dimensions must be positive")
        shape.append(raw_dimension)
    return shape


def profile_workload_binding(
    profile_result: dict[str, Any], profiled_summary: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    manifest = profile_result.get("manifest")
    selected_pages = (
        manifest.get("selected_pages") if isinstance(manifest, dict) else None
    )
    if not isinstance(manifest, dict) or not isinstance(selected_pages, list):
        raise ValueError("Profile result has no selected-page manifest provenance")
    manifest_sha = validate_sha256(manifest.get("sha256"), label="profile manifest")
    page_order = profiled_summary.get("page_order")
    expected_order = [int(page["index"]) for page in selected_pages]
    if page_order != expected_order:
        raise ValueError(
            "Profile workload page order differs from the selected manifest"
        )

    input_snapshot = profile_result.get("input_snapshot_before")
    if not isinstance(input_snapshot, list):
        raise ValueError("Profile result has no input snapshot")
    snapshot_by_role: dict[str, dict[str, Any]] = {}
    for record in input_snapshot:
        if not isinstance(record, dict):
            raise ValueError("Profile input snapshot contains an invalid record")
        role = str(record.get("role", ""))
        if not role or role in snapshot_by_role:
            raise ValueError(
                "Profile input snapshot contains a blank or duplicate role"
            )
        snapshot_by_role[role] = record
    manifest_snapshot = snapshot_by_role.get("manifest")
    if (
        not isinstance(manifest_snapshot, dict)
        or manifest_snapshot.get("sha256") != manifest_sha
    ):
        raise ValueError("Profile manifest hash is not bound to the input snapshot")
    snapshot_sha = canonical_sha256(input_snapshot)
    if snapshot_sha != profile_result.get("input_snapshot_before_sha256"):
        raise ValueError("Profile input snapshot fingerprint is invalid")

    rounds = profiled_summary.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("Profiled benchmark has no steady rounds")
    canonical_reports: dict[int, dict[str, Any]] = {}
    for round_index, round_report in enumerate(rounds, start=1):
        pages = round_report.get("pages") if isinstance(round_report, dict) else None
        if not isinstance(pages, list):
            raise ValueError(f"Profiled benchmark round {round_index} has no pages")
        observed_order = [int(page.get("index", 0)) for page in pages]
        if observed_order != expected_order:
            raise ValueError(
                f"Profiled benchmark round {round_index} has a different page order"
            )
        for page in pages:
            index = int(page["index"])
            report = {
                "route": str(page.get("route", "")),
                "source": str(Path(str(page.get("source", ""))).resolve()),
                "input_size": _shape(
                    page.get("input_size"),
                    label=f"profiled page {index} input size",
                    dimensions=2,
                ),
                "output_shape": _shape(
                    page.get("output_shape"),
                    label=f"profiled page {index} output shape",
                    dimensions=(2, 3),
                ),
            }
            previous = canonical_reports.setdefault(index, report)
            if report != previous:
                raise ValueError(
                    f"Profiled page {index} workload dimensions changed across rounds"
                )

    pages_binding: list[dict[str, Any]] = []
    output_shapes: dict[str, list[int]] = {}
    for page in selected_pages:
        index = int(page["index"])
        route = str(page.get("route", ""))
        if route not in {"normal", "sharper"}:
            raise ValueError(f"Profiled page {index} has an invalid route")
        source = str(Path(str(page.get("path", ""))).resolve())
        input_sha = validate_sha256(
            page.get("sha256"), label=f"profiled page {index} input"
        )
        input_bytes = int(page.get("bytes", 0))
        snapshot = snapshot_by_role.get(f"page:{index}")
        if (
            input_bytes < 1
            or not isinstance(snapshot, dict)
            or snapshot.get("path") != source
            or snapshot.get("sha256") != input_sha
            or snapshot.get("bytes") != input_bytes
        ):
            raise ValueError(
                f"Profiled page {index} is not bound to its input snapshot"
            )
        report = canonical_reports[index]
        if report["route"] != route or report["source"] != source:
            raise ValueError(
                f"Profiled page {index} route or source differs from the manifest"
            )
        pages_binding.append(
            {
                "index": index,
                "route": route,
                "input_sha256": input_sha,
                "input_bytes": input_bytes,
                "input_size": report["input_size"],
                "output_shape": report["output_shape"],
            }
        )
        output_shapes[str(index)] = report["output_shape"]
    return (
        {
            "manifest_sha256": manifest_sha,
            "input_snapshot_sha256": snapshot_sha,
            "page_order": expected_order,
            "pages": pages_binding,
        },
        output_shapes,
    )


def model_signature(models: Any, *, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(models, dict) or set(models) != {"normal", "sharper"}:
        raise ValueError(f"{label} must bind both Real-HAT models")
    signature: dict[str, dict[str, Any]] = {}
    for route in ("normal", "sharper"):
        record = models[route]
        if not isinstance(record, dict):
            raise ValueError(f"{label} model record is invalid for {route}")
        size = int(record.get("bytes", 0))
        if size < 1:
            raise ValueError(f"{label} model size is invalid for {route}")
        signature[route] = {
            "sha256": validate_sha256(
                record.get("sha256"), label=f"{label} {route} model"
            ),
            "bytes": size,
        }
    return signature


def expected_benchmark_configuration(
    profile_result: dict[str, Any],
) -> dict[str, Any]:
    configuration = profile_result.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("Profile result has no valid configuration")
    keys = (
        "precision",
        "tile_mode",
        "tile",
        "adaptive_tiles",
        "adaptive_selection_formula",
        "overlap",
        "batch_tiles",
        "device_assembly",
        "model_cache_size",
        "threshold",
        "rounds",
        "warmups_per_model",
        "warmup_crop",
        "gpu_phase_timing",
    )
    return {key: configuration.get(key) for key in keys}


def validate_benchmark_summary(
    payload: dict[str, Any], *, label: str, selected_regions: bool
) -> dict[str, str]:
    if (
        payload.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or payload.get("kind") != "real_hat_manifest_eager_benchmark"
        or payload.get("status") != "complete"
    ):
        raise ValueError(f"{label} is not a complete eager benchmark summary")
    profiler_control = payload.get("profiler_control")
    expected_control = {
        "rocprof_selected_regions": selected_regions,
        "scope": "all steady rounds only" if selected_regions else "disabled",
        "range_name": STEADY_ROCTX_RANGE if selected_regions else None,
    }
    if profiler_control != expected_control:
        raise ValueError(f"{label} profiler-control provenance is invalid")
    configuration = payload.get("configuration")
    if (
        not isinstance(configuration, dict)
        or configuration.get("gpu_phase_timing") is not True
    ):
        raise ValueError(f"{label} must include HIP Event phase timing")
    rounds = payload.get("rounds")
    expected_rounds = int(configuration.get("rounds", 0))
    if not isinstance(rounds, list) or len(rounds) != expected_rounds:
        raise ValueError(f"{label} steady round count is inconsistent")
    environment = payload.get("environment")
    gpu = environment.get("gpu") if isinstance(environment, dict) else None
    gpu_memory = gpu.get("total_memory_bytes") if isinstance(gpu, dict) else None
    compute_units = gpu.get("multiprocessor_count") if isinstance(gpu, dict) else None
    if (
        not isinstance(environment, dict)
        or environment.get("cuda_api_available") is not True
        or not environment.get("torch")
        or not environment.get("torch_hip")
        or not isinstance(gpu, dict)
        or not gpu.get("name")
        or not isinstance(gpu_memory, int)
        or gpu_memory <= 0
        or not isinstance(compute_units, int)
        or compute_units <= 0
    ):
        raise ValueError(f"{label} has no valid ROCm device environment")
    page_order = payload.get("page_order")
    if not isinstance(page_order, list) or len(page_order) != len(set(page_order)):
        raise ValueError(f"{label} page order is invalid")
    output_hashes = benchmark_output_hashes(payload, label=label)
    if set(output_hashes) != {str(index) for index in page_order}:
        raise ValueError(f"{label} page order and output hashes disagree")
    return output_hashes


def validate_profiled_benchmark_summary(
    payload: dict[str, Any], profile_result: dict[str, Any]
) -> dict[str, str]:
    output_hashes = validate_benchmark_summary(
        payload, label="profiled benchmark", selected_regions=True
    )
    if payload.get("configuration") != expected_benchmark_configuration(profile_result):
        raise ValueError("Profiled benchmark configuration does not match profile plan")
    manifest = profile_result.get("manifest")
    if not isinstance(manifest, dict) or payload.get("manifest") != manifest.get(
        "path"
    ):
        raise ValueError("Profiled benchmark manifest does not match profile plan")
    selected_pages = manifest.get("selected_pages")
    expected_order = [int(page["index"]) for page in selected_pages]
    if payload.get("page_order") != expected_order:
        raise ValueError("Profiled benchmark page order does not match profile plan")
    if model_signature(
        payload.get("models"), label="profiled benchmark"
    ) != model_signature(profile_result.get("models"), label="profile result"):
        raise ValueError("Profiled benchmark model hashes do not match profile plan")
    benchmark_script = profile_result.get("benchmark_script")
    runtime_code = payload.get("runtime_code")
    if (
        not isinstance(benchmark_script, dict)
        or not isinstance(runtime_code, list)
        or benchmark_script not in runtime_code
    ):
        raise ValueError("Profiled benchmark runtime code does not bind its launcher")
    return output_hashes


def baseline_metrics(payload: dict[str, Any]) -> dict[str, float]:
    round_walls: list[float] = []
    forward_services: list[float] = []
    page_order = [int(index) for index in payload["page_order"]]
    expected_hashes = benchmark_output_hashes(payload, label="unprofiled baseline")
    for expected_round, round_report in enumerate(payload["rounds"], start=1):
        if not isinstance(round_report, dict):
            raise ValueError("Baseline contains an invalid round report")
        if int(round_report.get("round", 0)) != expected_round:
            raise ValueError("Baseline round indexes are not contiguous")
        wall = float(round_report.get("loop_wall_seconds", 0.0))
        if not math.isfinite(wall) or wall <= 0:
            raise ValueError("Baseline round wall must be finite and positive")
        pages = round_report.get("pages")
        if not isinstance(pages, list) or len(pages) != len(page_order):
            raise ValueError("Baseline round page count is inconsistent")
        observed_order = [
            int(page.get("index", 0)) if isinstance(page, dict) else 0 for page in pages
        ]
        if observed_order != page_order:
            raise ValueError("Baseline round page order differs from page_order")
        forward = 0.0
        for page in pages:
            page_index = int(page["index"])
            digest = validate_sha256(
                page.get("pixel_sha256"),
                label=f"baseline round {expected_round} page {page_index}",
            )
            if digest != expected_hashes[str(page_index)]:
                raise ValueError("Baseline round output hash differs from steady state")
            stats = page.get("stats") if isinstance(page, dict) else None
            if not isinstance(stats, dict):
                raise ValueError("Baseline page has no timing stats")
            seconds = stats.get("forward_seconds")
            if not isinstance(seconds, (int, float)):
                raise ValueError("Baseline page has no forward HIP Event timing")
            seconds = float(seconds)
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError("Baseline forward timing is invalid")
            forward += seconds
        if forward <= 0 or forward > wall:
            raise ValueError(
                "Baseline forward service must be positive and fit within its "
                "single-stream round wall"
            )
        round_walls.append(wall)
        forward_services.append(forward)
    end_to_end = statistics.median(round_walls)
    forward_service = statistics.median(forward_services)
    recorded = payload["steady_state"].get("round_loop_wall_seconds")
    if not isinstance(recorded, dict) or not math.isclose(
        float(recorded.get("median", -1.0)), end_to_end, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError("Baseline recorded median wall does not match its rounds")
    return {
        "end_to_end_seconds": end_to_end,
        "forward_service_seconds": forward_service,
    }


def validate_baseline_summary(
    baseline: dict[str, Any],
    profiled: dict[str, Any],
    profile_result: dict[str, Any],
) -> tuple[dict[str, float], dict[str, str]]:
    baseline_hashes = validate_benchmark_summary(
        baseline, label="unprofiled baseline", selected_regions=False
    )
    profiled_hashes = validate_profiled_benchmark_summary(profiled, profile_result)
    if baseline.get("configuration") != profiled.get("configuration"):
        raise ValueError("Baseline and profiled benchmark configurations differ")
    if baseline.get("manifest") != profiled.get("manifest"):
        raise ValueError("Baseline and profiled benchmark manifests differ")
    if baseline.get("page_order") != profiled.get("page_order"):
        raise ValueError("Baseline and profiled page orders differ")
    if baseline.get("runtime_code") != profiled.get("runtime_code"):
        raise ValueError("Baseline and profiled runtime code fingerprints differ")
    if baseline.get("environment") != profiled.get("environment"):
        raise ValueError("Baseline and profiled ROCm device environments differ")
    if model_signature(baseline.get("models"), label="baseline") != model_signature(
        profiled.get("models"), label="profiled benchmark"
    ):
        raise ValueError("Baseline and profiled model hashes differ")
    if baseline_hashes != profiled_hashes:
        raise ValueError("Profiling changed benchmark output pixel hashes")
    return baseline_metrics(baseline), baseline_hashes


def analyze_kernel_rows(
    rows: list[dict[str, Any]],
    patterns: list[str],
    *,
    baseline_end_to_end_seconds: float,
    baseline_forward_service_seconds: float,
    prototype_speedup: float | None,
    prototype_outputs_match: bool,
    evidence_blockers: list[str] | None = None,
) -> dict[str, Any]:
    if (
        not math.isfinite(baseline_end_to_end_seconds)
        or baseline_end_to_end_seconds <= 0
    ):
        raise ValueError("Baseline end-to-end wall must be finite and positive")
    if (
        not math.isfinite(baseline_forward_service_seconds)
        or baseline_forward_service_seconds <= 0
        or baseline_forward_service_seconds > baseline_end_to_end_seconds
    ):
        raise ValueError(
            "Baseline forward service must be finite, positive, and no larger "
            "than end-to-end wall"
        )
    if prototype_speedup is not None and (
        not math.isfinite(prototype_speedup) or prototype_speedup <= 0
    ):
        raise ValueError("Prototype speedup must be finite and positive")
    execution_lanes = {
        (
            str(row.get("source", "")).strip(),
            str(row.get("process_id", "")).strip(),
            str(row.get("agent_id", "")).strip(),
            str(row.get("queue_id", "")).strip(),
            str(row.get("stream_id", "")).strip(),
        )
        for row in rows
    }
    if any(not all(lane) for lane in execution_lanes) or len(execution_lanes) != 1:
        raise ValueError(
            "Custom-operator attribution requires exactly one trace/process-agent-queue-stream lane"
        )
    source, process_id, agent_id, queue_id, stream_id = next(iter(execution_lanes))
    compiled = [re.compile(pattern) for pattern in patterns]
    total_ns = sum(int(row["duration_ns"]) for row in rows)
    if total_ns <= 0:
        raise ValueError("Kernel traces contain no positive service time")
    matched_rows = [
        row
        for row in rows
        if any(pattern.search(str(row["kernel_name"])) for pattern in compiled)
    ]
    matched_ns = sum(int(row["duration_ns"]) for row in matched_rows)
    gpu_share = 100.0 * matched_ns / total_ns
    estimated_baseline_segment_seconds = (
        baseline_forward_service_seconds * gpu_share / 100.0
    )
    end_to_end_share = (
        100.0 * estimated_baseline_segment_seconds / baseline_end_to_end_seconds
    )
    segment_gate = (
        end_to_end_share >= MIN_END_TO_END_SHARE_PERCENT
        or gpu_share >= MIN_GPU_SHARE_PERCENT
    )
    estimated_gain = None
    speedup_gate = False
    gain_gate = False
    if prototype_speedup is not None:
        estimated_gain = end_to_end_share * (1.0 - 1.0 / prototype_speedup)
        speedup_gate = prototype_speedup >= MIN_PROTOTYPE_SPEEDUP
        gain_gate = estimated_gain >= MIN_ESTIMATED_END_TO_END_GAIN_PERCENT
    blockers: list[str] = list(evidence_blockers or [])
    if not segment_gate:
        blockers.append("target segment is below both attribution thresholds")
    if prototype_speedup is None:
        blockers.append("prototype speedup is not supplied")
    elif not speedup_gate:
        blockers.append("prototype speedup is below 1.3x")
    if prototype_speedup is not None and not gain_gate:
        blockers.append("estimated end-to-end gain is below 3%")
    if not prototype_outputs_match:
        blockers.append("prototype output hash equivalence is not confirmed")

    by_kernel: dict[str, dict[str, int]] = {}
    for row in rows:
        name = str(row["kernel_name"])
        aggregate = by_kernel.setdefault(name, {"calls": 0, "duration_ns": 0})
        aggregate["calls"] += 1
        aggregate["duration_ns"] += int(row["duration_ns"])
    top_kernels = [
        {
            "kernel_name": name,
            "calls": values["calls"],
            "service_seconds": values["duration_ns"] / 1e9,
            "service_share_percent": 100.0 * values["duration_ns"] / total_ns,
        }
        for name, values in sorted(
            by_kernel.items(), key=lambda item: item[1]["duration_ns"], reverse=True
        )[:30]
    ]
    return {
        "kernel_dispatches": len(rows),
        "execution_lane": {
            "trace": source,
            "process_id": process_id,
            "agent_id": agent_id,
            "queue_id": queue_id,
            "stream_id": stream_id,
        },
        "total_kernel_service_seconds": total_ns / 1e9,
        "matched_kernel_dispatches": len(matched_rows),
        "matched_kernel_service_seconds": matched_ns / 1e9,
        "matched_gpu_service_share_percent": gpu_share,
        "baseline_end_to_end_seconds": baseline_end_to_end_seconds,
        "baseline_forward_service_seconds": baseline_forward_service_seconds,
        "estimated_baseline_segment_seconds": estimated_baseline_segment_seconds,
        "estimated_end_to_end_share_percent": end_to_end_share,
        "prototype_speedup": prototype_speedup,
        "prototype_outputs_match": prototype_outputs_match,
        "estimated_end_to_end_gain_percent": estimated_gain,
        "gates": {
            "segment_share": segment_gate,
            "prototype_speedup": speedup_gate,
            "estimated_end_to_end_gain": gain_gate,
            "hash_equivalence": prototype_outputs_match,
        },
        "custom_operator_eligible": (
            segment_gate
            and speedup_gate
            and gain_gate
            and prototype_outputs_match
            and not evidence_blockers
        ),
        "blockers": blockers,
        "top_kernels": top_kernels,
    }


def resolve_artifact_record(
    root: Path, record: Any, *, label: str
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} artifact record is invalid")
    raw_relative = record.get("path")
    if not isinstance(raw_relative, str) or not raw_relative:
        raise ValueError(f"{label} artifact path is invalid")
    relative = Path(raw_relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} artifact path is unsafe: {raw_relative}")
    resolved_root = root.resolve()
    candidate = resolved_root / relative
    cursor = candidate
    while cursor != resolved_root:
        if cursor.is_symlink():
            raise ValueError(f"{label} artifact path contains a symlink: {candidate}")
        cursor = cursor.parent
    path = candidate.resolve()
    if not is_relative_to(path, resolved_root) or not path.is_file():
        raise ValueError(f"{label} artifact is missing or unsafe: {path}")
    expected_sha = validate_sha256(record.get("sha256"), label=label)
    expected_bytes = int(record.get("bytes", -1))
    normalized = {
        "path": path.relative_to(resolved_root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if normalized["sha256"] != expected_sha or normalized["bytes"] != expected_bytes:
        raise ValueError(f"{label} artifact hash or size changed: {path}")
    if normalized["path"] != Path(raw_relative).as_posix():
        raise ValueError(f"{label} artifact path is not canonical: {raw_relative}")
    return path, normalized


def validate_profile_command(profile_result: dict[str, Any]) -> None:
    command = profile_result.get("command")
    if not isinstance(command, list) or not all(
        isinstance(item, str) for item in command
    ):
        raise ValueError("Profile result has no valid command")
    if command.count("--") != 1:
        raise ValueError("Profile command has an invalid application delimiter")
    delimiter = command.index("--")
    profiler = command[:delimiter]
    child = command[delimiter + 1 :]
    for flag in ("--kernel-trace", "--marker-trace", "--selected-regions"):
        if profiler.count(flag) != 1:
            raise ValueError(f"Profile command must contain exactly one {flag}")
    if "--runtime-trace" in profiler:
        raise ValueError(
            "Profile command must not trace preload or warmup runtime activity"
        )
    if child.count("--rocprof-selected-regions") != 1:
        raise ValueError("Profile child did not enable ROCTx steady-round control")


def validate_profile_evidence(profile_root: Path) -> dict[str, Any]:
    root = profile_root.resolve()
    if profile_root.is_symlink() or not root.is_dir():
        raise ValueError(f"Profile root is missing or unsafe: {root}")
    completion_path = root / "completion.json"
    if completion_path.is_symlink() or not completion_path.is_file():
        raise ValueError("Profile root has no safe completion marker")
    completion = read_json(completion_path)
    if (
        completion.get("schema_version") != SCHEMA_VERSION
        or completion.get("kind") != "real_hat_native_rocprof_completion"
        or completion.get("status") != "complete"
    ):
        raise ValueError("Profile root has no valid completion marker")

    result_path, result_record = resolve_artifact_record(
        root, completion.get("profile_result"), label="profile result"
    )
    if result_path != root / "profile_result.json":
        raise ValueError("Completion marker points to an unexpected profile result")
    result = read_json(result_path)
    if (
        result.get("schema_version") != SCHEMA_VERSION
        or result.get("kind") != "real_hat_native_rocprof_result"
        or result.get("status") != "complete"
        or Path(str(result.get("profile_output"))).resolve() != root
    ):
        raise ValueError("Profile result is incomplete or bound to another output root")
    validate_profile_command(result)

    plan_path, plan_record = resolve_artifact_record(
        root, result.get("profile_plan"), label="profile plan"
    )
    if completion.get("profile_plan") != plan_record:
        raise ValueError("Completion and result disagree on the profile plan")
    plan = read_json(plan_path)
    if (
        plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("kind") != "real_hat_native_rocprof_plan"
        or plan.get("status") != "running"
    ):
        raise ValueError("Profile plan is invalid")
    bound_plan_fields = (
        "environment",
        "production_root",
        "profile_output",
        "benchmark_script",
        "manifest",
        "input_snapshot_before",
        "input_snapshot_before_sha256",
        "models",
        "configuration",
        "command",
        "safety",
    )
    if any(plan.get(key) != result.get(key) for key in bound_plan_fields):
        raise ValueError("Profile result does not match its immutable plan")

    inventory = result.get("artifacts")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Profile result has no artifact inventory")
    inventory_sha = canonical_sha256(inventory)
    if (
        result.get("artifacts_sha256") != inventory_sha
        or completion.get("artifacts_sha256") != inventory_sha
    ):
        raise ValueError("Profile artifact inventory fingerprint is invalid")
    inventory_by_path: dict[str, dict[str, Any]] = {}
    for index, raw_record in enumerate(inventory):
        _path, record = resolve_artifact_record(
            root, raw_record, label=f"inventory artifact {index}"
        )
        if record["path"] in inventory_by_path:
            raise ValueError(f"Duplicate artifact inventory path: {record['path']}")
        inventory_by_path[record["path"]] = record
    if inventory_by_path.get(plan_record["path"]) != plan_record:
        raise ValueError("Artifact inventory does not bind the profile plan")
    completion_record = relative_artifact_record(root, completion_path)
    expected_current_inventory = sorted(
        [*inventory_by_path.values(), result_record, completion_record],
        key=lambda record: str(record["path"]),
    )
    if artifact_inventory(root) != expected_current_inventory:
        raise ValueError("Profile root contains unrecorded or missing artifacts")

    def validate_trace_set(
        field: str, pattern: str
    ) -> tuple[list[Path], list[dict[str, Any]]]:
        result_records = result.get(field)
        if not isinstance(result_records, list) or not result_records:
            raise ValueError(f"Profile result has no {field}")
        if completion.get(field) != result_records:
            raise ValueError(f"Completion and result disagree on {field}")
        resolved_records: list[dict[str, Any]] = []
        paths: list[Path] = []
        for index, raw_record in enumerate(result_records):
            path, record = resolve_artifact_record(
                root, raw_record, label=f"{field} {index}"
            )
            if inventory_by_path.get(record["path"]) != record:
                raise ValueError(f"Artifact inventory does not bind {field} {index}")
            paths.append(path)
            resolved_records.append(record)
        observed_paths = sorted((root / "trace").rglob(pattern))
        if (
            any(path.is_symlink() for path in observed_paths)
            or sorted(paths) != observed_paths
        ):
            raise ValueError(f"Recorded and observed {field} differ")
        return paths, resolved_records

    kernel_paths, kernel_records = validate_trace_set(
        "kernel_traces", "*kernel_trace.csv"
    )
    marker_paths, marker_records = validate_trace_set(
        "marker_traces", "*marker_api_trace.csv"
    )
    raw_log_records = result.get("profiler_logs")
    if (
        not isinstance(raw_log_records, list)
        or len(raw_log_records) != 2
        or completion.get("profiler_logs") != raw_log_records
    ):
        raise ValueError("Completion has no valid profiler log binding")
    log_records: list[dict[str, Any]] = []
    log_paths: list[Path] = []
    for index, raw_record in enumerate(raw_log_records):
        path, record = resolve_artifact_record(
            root, raw_record, label=f"profiler log {index}"
        )
        if inventory_by_path.get(record["path"]) != record:
            raise ValueError(f"Artifact inventory does not bind profiler log {index}")
        log_paths.append(path)
        log_records.append(record)
    if {path.name for path in log_paths} != {
        "rocprof.stdout.log",
        "rocprof.stderr.log",
    }:
        raise ValueError("Profiler log filenames are invalid")
    profiler_diagnostics = inspect_profiler_logs(log_paths)
    if (
        result.get("profiler_diagnostics") != profiler_diagnostics
        or completion.get("profiler_diagnostics") != profiler_diagnostics
    ):
        raise ValueError("Profiler diagnostics are not bound to logs and completion")
    control_calls = verify_profiler_control_traces(marker_paths)
    verify_kernels_within_selected_range(
        read_kernel_rows(kernel_paths, process_id=control_calls["process_id"]),
        control_calls,
    )
    if result.get("roctx_control_calls") != control_calls:
        raise ValueError("Profile result ROCTx control counts are invalid")

    benchmark_path, benchmark_record = resolve_artifact_record(
        root, result.get("benchmark_summary"), label="profiled benchmark summary"
    )
    if completion.get("benchmark_summary") != benchmark_record:
        raise ValueError("Completion and result disagree on the benchmark summary")
    if inventory_by_path.get(benchmark_record["path"]) != benchmark_record:
        raise ValueError("Artifact inventory does not bind the benchmark summary")
    profiled_summary = read_json(benchmark_path)
    validate_profiled_benchmark_summary(profiled_summary, result)

    before = result.get("input_snapshot_before")
    after = result.get("input_snapshot_after")
    if not isinstance(before, list) or before != after:
        raise ValueError("Profiler input snapshots are missing or changed")
    before_sha = canonical_sha256(before)
    after_sha = canonical_sha256(after)
    if (
        result.get("input_snapshot_before_sha256") != before_sha
        or result.get("input_snapshot_after_sha256") != after_sha
        or completion.get("input_snapshot_before_sha256") != before_sha
        or completion.get("input_snapshot_after_sha256") != after_sha
    ):
        raise ValueError("Profiler input snapshot fingerprints are invalid")
    verify_input_snapshot(before)

    manifest = result.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Profile result has no manifest binding")
    selected_pages = manifest.get("selected_pages")
    if not isinstance(selected_pages, list):
        raise ValueError("Profile result has no selected-page binding")
    isolation_proof = validate_isolation(
        production_root=Path(str(result.get("production_root"))),
        manifest=Path(str(manifest.get("path"))),
        run_root=root,
        selected_pages=selected_pages,
    )
    safety = result.get("safety")
    if (
        not isinstance(safety, dict)
        or safety.get("path_overlap_checks") != isolation_proof
    ):
        raise ValueError("Profile isolation proof is invalid")

    environment = result.get("environment")
    version = (
        environment.get("rocprofv3_version") if isinstance(environment, dict) else None
    )
    version_output = (
        "\n".join(
            part
            for part in (
                str(version.get("stdout", "")) if isinstance(version, dict) else "",
                str(version.get("stderr", "")) if isinstance(version, dict) else "",
            )
            if part
        )
        if isinstance(version, dict)
        else ""
    )
    version_match = ROCM_72_PATTERN.search(version_output)
    if (
        not isinstance(version, dict)
        or version.get("exit_code") != 0
        or version_match is None
        or version_match.group(1) != str(version.get("rocm_version"))
    ):
        raise ValueError("Profile result is not bound to rocprofv3 on ROCm 7.2.x")
    preflight = environment.get("roctx_preflight")
    roctx_paths = environment.get("roctx_paths")
    if (
        not isinstance(preflight, dict)
        or preflight.get("exit_code") != 0
        or preflight.get("stdout") != "roctx-control-ok"
        or not isinstance(roctx_paths, dict)
        or not roctx_paths.get("roctx_site_packages")
    ):
        raise ValueError("Profile result has no valid ROCTx Python preflight binding")

    return {
        "root": root,
        "completion_path": completion_path,
        "completion_record": completion_record,
        "completion": completion,
        "profile_result_path": result_path,
        "profile_result_record": result_record,
        "result": result,
        "profile_plan_path": plan_path,
        "profile_plan_record": plan_record,
        "kernel_paths": kernel_paths,
        "kernel_records": kernel_records,
        "marker_records": marker_records,
        "roctx_control": control_calls,
        "log_paths": log_paths,
        "log_records": log_records,
        "profiler_diagnostics": profiler_diagnostics,
        "benchmark_path": benchmark_path,
        "benchmark_record": benchmark_record,
        "profiled_summary": profiled_summary,
    }


def bind_bundled_file(summary_path: Path, record: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} is not an artifact record")
    raw_path = record.get("path")
    relative = Path(str(raw_path))
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise ValueError(f"{label} path is unsafe: {raw_path!r}")
    root = summary_path.resolve().parent
    candidate = root / relative
    cursor = candidate
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError(f"{label} path contains a symlink: {candidate}")
        cursor = cursor.parent
    path = candidate.resolve()
    if not is_relative_to(path, root) or not path.is_file() or path == summary_path:
        raise ValueError(f"{label} file is missing or unsafe: {path}")
    size = path.stat().st_size
    expected_bytes = record.get("bytes")
    if (
        not isinstance(expected_bytes, int)
        or expected_bytes < 1
        or size != expected_bytes
    ):
        raise ValueError(f"{label} size changed: {path}")
    expected_sha = validate_sha256(record.get("sha256"), label=label)
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha:
        raise ValueError(f"{label} hash changed: {path}")
    return {"path": str(path), "sha256": observed_sha, "bytes": size}


def load_prototype_run_summary(
    path: Path,
    *,
    label: str,
    expected_role: str,
    profile_result: dict[str, Any],
    profiled_summary: dict[str, Any],
    kernel_patterns: list[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    if path.is_symlink():
        raise ValueError(f"{label} summary must not be a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = read_json(resolved)
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "real_hat_prototype_run"
        or payload.get("status") != "complete"
        or payload.get("role") != expected_role
    ):
        raise ValueError(f"{label} is not a complete {expected_role} run summary")
    if model_signature(payload.get("models"), label=label) != model_signature(
        profile_result.get("models"), label="profile result"
    ):
        raise ValueError(f"{label} model hashes differ from the profile")
    if payload.get("configuration") != profiled_summary.get("configuration"):
        raise ValueError(f"{label} configuration differs from the profile")
    if payload.get("environment") != profiled_summary.get("environment"):
        raise ValueError(f"{label} ROCm device environment differs from the profile")

    expected_workload, expected_output_shapes = profile_workload_binding(
        profile_result, profiled_summary
    )
    if payload.get("workload") != expected_workload:
        raise ValueError(f"{label} workload differs from the profiled input set")
    expected_segment = {
        "kind": "rocprof-kernel-regex-set",
        "kernel_patterns": normalize_kernel_patterns(kernel_patterns),
    }
    if payload.get("target_segment") != expected_segment:
        raise ValueError(f"{label} target segment differs from the analysis target")

    implementation = payload.get("implementation")
    implementation_name = (
        str(implementation.get("name", "")).strip()
        if isinstance(implementation, dict)
        else ""
    )
    raw_artifacts = (
        implementation.get("artifacts") if isinstance(implementation, dict) else None
    )
    if (
        not implementation_name
        or not isinstance(raw_artifacts, list)
        or not raw_artifacts
    ):
        raise ValueError(f"{label} has no implementation artifact provenance")
    implementation_artifacts = [
        bind_bundled_file(resolved, record, label=f"{label} implementation {index}")
        for index, record in enumerate(raw_artifacts)
    ]
    implementation_paths = {record["path"] for record in implementation_artifacts}
    if len(implementation_paths) != len(implementation_artifacts):
        raise ValueError(f"{label} repeats an implementation artifact")

    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError(f"{label} has no actual output files")
    output_hashes: dict[str, str] = {}
    output_files: list[dict[str, Any]] = []
    output_paths: set[str] = set()
    for index, output in enumerate(outputs):
        if not isinstance(output, dict):
            raise ValueError(f"{label} output {index} is invalid")
        identifier = str(output.get("id", "")).strip()
        if not identifier or identifier in output_hashes:
            raise ValueError(f"{label} has a blank or duplicate output id")
        artifact = bind_bundled_file(
            resolved, output, label=f"{label} output {identifier}"
        )
        output_shape = _shape(
            output.get("shape"),
            label=f"{label} output {identifier} shape",
            dimensions=(2, 3),
        )
        if output.get("format") != "raw-uint8-contiguous":
            raise ValueError(
                f"{label} output {identifier} must be contiguous raw uint8 pixels"
            )
        if expected_output_shapes.get(identifier) != output_shape:
            raise ValueError(
                f"{label} output {identifier} shape differs from the eager baseline"
            )
        if artifact["bytes"] != math.prod(output_shape):
            raise ValueError(
                f"{label} output {identifier} byte count does not match its uint8 shape"
            )
        if artifact["path"] in output_paths or artifact["path"] in implementation_paths:
            raise ValueError(f"{label} repeats an output or implementation artifact")
        output_paths.add(artifact["path"])
        output_hashes[identifier] = artifact["sha256"]
        output_files.append(
            {
                "id": identifier,
                "format": "raw-uint8-contiguous",
                "shape": output_shape,
                **artifact,
            }
        )
    eager_output_hashes = benchmark_output_hashes(
        profiled_summary, label="profiled benchmark"
    )
    expected_output_ids = set(eager_output_hashes)
    if set(output_hashes) != expected_output_ids:
        raise ValueError(f"{label} outputs do not cover the profiled page set")
    if output_hashes != eager_output_hashes:
        raise ValueError(
            f"{label} actual raw output pixels differ from the eager baseline"
        )

    performance = payload.get("performance")
    raw_samples = (
        performance.get("segment_wall_seconds")
        if isinstance(performance, dict)
        else None
    )
    if (
        not isinstance(performance, dict)
        or performance.get("warmup_excluded") is not True
        or performance.get("single_stream") is not True
        or not isinstance(raw_samples, list)
        or len(raw_samples) < 5
    ):
        raise ValueError(
            f"{label} has no valid steady single-stream performance samples"
        )
    samples = [float(value) for value in raw_samples]
    if any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError(f"{label} performance samples must be finite and positive")
    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    cv_percent = statistics.pstdev(samples) / mean * 100.0
    if int(performance.get("iterations", 0)) != len(samples) or not math.isclose(
        float(performance.get("median_segment_wall_seconds", 0.0)),
        median,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{label} performance summary is internally inconsistent")
    if cv_percent >= 3.0:
        raise ValueError(f"{label} performance CV must be below 3%")

    evidence = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "role": expected_role,
        "implementation": {
            "name": implementation_name,
            "artifacts": implementation_artifacts,
            "artifacts_sha256": canonical_sha256(implementation_artifacts),
        },
        "models_sha256": canonical_sha256(
            model_signature(payload.get("models"), label=label)
        ),
        "configuration_sha256": canonical_sha256(payload["configuration"]),
        "environment_sha256": canonical_sha256(payload["environment"]),
        "workload": expected_workload,
        "workload_sha256": canonical_sha256(expected_workload),
        "target_segment": expected_segment,
        "target_segment_sha256": canonical_sha256(expected_segment),
        "performance": {
            "iterations": len(samples),
            "segment_wall_seconds": samples,
            "median_segment_wall_seconds": median,
            "cv_percent": cv_percent,
            "samples_sha256": canonical_sha256(samples),
        },
        "output_count": len(output_hashes),
        "mapping_sha256": canonical_sha256(output_hashes),
        "output_files": output_files,
    }
    return output_hashes, evidence


def analyze_profile(args: argparse.Namespace) -> int:
    kernel_patterns = normalize_kernel_patterns(args.kernel_pattern)
    prototype_values = (
        args.prototype_baseline_summary,
        args.prototype_candidate_summary,
    )
    if any(value is not None for value in prototype_values) and not all(
        value is not None for value in prototype_values
    ):
        raise ValueError(
            "Prototype analysis requires both baseline and candidate run summaries"
        )
    evidence = validate_profile_evidence(args.profile_root)
    profile_root = evidence["root"]
    manifest_root = Path(str(evidence["result"]["manifest"]["path"])).resolve().parent
    baseline_path = args.baseline_summary.resolve()
    if (
        args.baseline_summary.is_symlink()
        or not baseline_path.is_file()
        or is_relative_to(baseline_path, profile_root)
        or is_relative_to(baseline_path, manifest_root)
    ):
        raise ValueError(
            "Baseline summary must be a separate, safe unprofiled artifact"
        )
    production_root = Path(str(evidence["result"]["production_root"])).resolve()
    if is_relative_to(baseline_path, production_root):
        raise ValueError("Baseline summary must not come from the production library")
    baseline = read_json(baseline_path)
    metrics, baseline_output_hashes = validate_baseline_summary(
        baseline, evidence["profiled_summary"], evidence["result"]
    )

    prototype_evidence: dict[str, Any] | None = None
    prototype_outputs_match = False
    prototype_speedup: float | None = None
    if args.prototype_baseline_summary is not None:
        baseline_run_path = args.prototype_baseline_summary.resolve()
        candidate_run_path = args.prototype_candidate_summary.resolve()
        if baseline_run_path == candidate_run_path:
            raise ValueError("Prototype run summaries must be two distinct files")
        prototype_baseline, prototype_baseline_evidence = load_prototype_run_summary(
            args.prototype_baseline_summary,
            label="prototype baseline",
            expected_role="baseline",
            profile_result=evidence["result"],
            profiled_summary=evidence["profiled_summary"],
            kernel_patterns=kernel_patterns,
        )
        prototype_candidate, prototype_candidate_evidence = load_prototype_run_summary(
            args.prototype_candidate_summary,
            label="prototype candidate",
            expected_role="candidate",
            profile_result=evidence["result"],
            profiled_summary=evidence["profiled_summary"],
            kernel_patterns=kernel_patterns,
        )
        baseline_output_paths = {
            record["path"] for record in prototype_baseline_evidence["output_files"]
        }
        candidate_output_paths = {
            record["path"] for record in prototype_candidate_evidence["output_files"]
        }
        if baseline_output_paths & candidate_output_paths:
            raise ValueError("Prototype runs must use distinct actual output files")
        baseline_implementation = prototype_baseline_evidence["implementation"]
        candidate_implementation = prototype_candidate_evidence["implementation"]
        baseline_implementation_signature = sorted(
            (record["sha256"], record["bytes"])
            for record in baseline_implementation["artifacts"]
        )
        candidate_implementation_signature = sorted(
            (record["sha256"], record["bytes"])
            for record in candidate_implementation["artifacts"]
        )
        if baseline_implementation_signature == candidate_implementation_signature:
            raise ValueError(
                "Prototype baseline and candidate implementations are identical"
            )
        prototype_speedup = (
            prototype_baseline_evidence["performance"]["median_segment_wall_seconds"]
            / prototype_candidate_evidence["performance"]["median_segment_wall_seconds"]
        )
        prototype_outputs_match = prototype_baseline == prototype_candidate
        prototype_evidence = {
            "baseline": prototype_baseline_evidence,
            "candidate": prototype_candidate_evidence,
            "output_ids": sorted(prototype_baseline),
            "outputs_match": prototype_outputs_match,
            "derived_speedup": prototype_speedup,
        }

    rows = read_kernel_rows(
        evidence["kernel_paths"],
        process_id=evidence["roctx_control"]["process_id"],
    )
    evidence_blockers = (
        ["rocprofv3 warnings require manual review before custom-operator work"]
        if evidence["profiler_diagnostics"]["manual_review_required"]
        else []
    )
    analysis = analyze_kernel_rows(
        rows,
        kernel_patterns,
        baseline_end_to_end_seconds=metrics["end_to_end_seconds"],
        baseline_forward_service_seconds=metrics["forward_service_seconds"],
        prototype_speedup=prototype_speedup,
        prototype_outputs_match=prototype_outputs_match,
        evidence_blockers=evidence_blockers,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "real_hat_native_rocprof_analysis",
        "status": "complete",
        "created_at": utc_now(),
        "profile_root": str(profile_root),
        "profile_evidence": {
            "completion": evidence["completion_record"],
            "profile_result": evidence["profile_result_record"],
            "profile_plan": evidence["profile_plan_record"],
            "kernel_traces": evidence["kernel_records"],
            "marker_traces": evidence["marker_records"],
            "profiler_logs": evidence["log_records"],
            "profiler_diagnostics": evidence["profiler_diagnostics"],
            "benchmark_summary": evidence["benchmark_record"],
            "configuration_sha256": canonical_sha256(
                evidence["profiled_summary"]["configuration"]
            ),
            "environment_sha256": canonical_sha256(
                evidence["profiled_summary"]["environment"]
            ),
        },
        "baseline": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
            "bytes": baseline_path.stat().st_size,
            "configuration_sha256": canonical_sha256(baseline["configuration"]),
            "environment_sha256": canonical_sha256(baseline["environment"]),
            "output_hashes_sha256": canonical_sha256(baseline_output_hashes),
            "metrics": metrics,
            "unprofiled": True,
        },
        "prototype": prototype_evidence,
        "kernel_patterns": kernel_patterns,
        "methodology": {
            "gpu_share": (
                "matched kernel service / all steady-round single-stream kernel service"
            ),
            "baseline_segment_estimate": (
                "unprofiled forward HIP Event service * traced kernel service share"
            ),
            "estimated_gain": "Amdahl estimate using the unprofiled end-to-end wall",
            "prototype_speedup": (
                "ratio of medians from two bound single-stream target-segment runs"
            ),
            "trace_overhead_excluded_from_baseline": True,
            "multi_stream_attribution_rejected": True,
            "required_order": [
                "existing ATen or Inductor formulation",
                "MIOpen or rocBLAS",
                "semantically equivalent SDPA",
                "Triton prototype",
                "HIP or CK only after Triton",
            ],
        },
        "thresholds": {
            "minimum_end_to_end_share_percent": MIN_END_TO_END_SHARE_PERCENT,
            "minimum_gpu_share_percent": MIN_GPU_SHARE_PERCENT,
            "minimum_prototype_speedup": MIN_PROTOTYPE_SPEEDUP,
            "minimum_estimated_end_to_end_gain_percent": (
                MIN_ESTIMATED_END_TO_END_GAIN_PERCENT
            ),
        },
        "analysis": analysis,
    }
    output = args.output.resolve() if args.output else profile_root / "analysis.json"
    protected = {
        baseline_path,
        evidence["completion_path"],
        evidence["profile_result_path"],
        evidence["profile_plan_path"],
        evidence["benchmark_path"],
        *evidence["kernel_paths"],
        *evidence["log_paths"],
    }
    if args.prototype_baseline_summary is not None:
        protected.update(
            {
                args.prototype_baseline_summary.resolve(),
                args.prototype_candidate_summary.resolve(),
            }
        )
        for run in (prototype_evidence["baseline"], prototype_evidence["candidate"]):
            protected.update(
                Path(record["path"])
                for record in [
                    *run["implementation"]["artifacts"],
                    *run["output_files"],
                ]
            )
    if (
        output in protected
        or output.exists()
        or is_relative_to(output, profile_root / "trace")
        or is_relative_to(output, profile_root / "benchmark")
        or is_relative_to(output, production_root)
        or is_relative_to(output, manifest_root)
    ):
        raise ValueError(f"Unsafe or existing analysis output path: {output}")
    write_json(output, payload)
    print(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Real-HAT rocprofv3 only on native Linux, or apply the custom-op "
            "attribution gates to an existing trace."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Collect a native-Linux rocprofv3 trace")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument(
        "--production-root",
        type=Path,
        required=True,
        help="Read-only production library root; must not overlap inputs or outputs",
    )
    run.add_argument("--page-indexes", type=positive_int, nargs="+", required=True)
    run.add_argument("--normal-model", type=Path, required=True)
    run.add_argument("--sharper-model", type=Path, required=True)
    tile_group = run.add_mutually_exclusive_group(required=True)
    tile_group.add_argument("--tile", type=positive_int)
    tile_group.add_argument("--adaptive-tiles", type=positive_int, nargs="+")
    run.add_argument("--overlap", type=positive_int, default=32)
    run.add_argument("--rounds", type=positive_int, default=1)
    run.add_argument("--warmups-per-model", type=positive_int, default=1)
    run.add_argument("--warmup-crop", type=positive_int, default=320)
    run.add_argument("--timeout-seconds", type=positive_float, default=3600.0)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--run-name", required=True)
    run.set_defaults(handler=run_profile)

    analyze = subparsers.add_parser(
        "analyze", help="Apply attribution and custom-operator gates"
    )
    analyze.add_argument("--profile-root", type=Path, required=True)
    analyze.add_argument("--kernel-pattern", action="append", required=True)
    analyze.add_argument("--baseline-summary", type=Path, required=True)
    analyze.add_argument("--prototype-baseline-summary", type=Path)
    analyze.add_argument("--prototype-candidate-summary", type=Path)
    analyze.add_argument("--output", type=Path)
    analyze.set_defaults(handler=analyze_profile)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        candidates = args.adaptive_tiles or ([args.tile] if args.tile else [])
        if args.overlap >= min(candidates):
            raise SystemExit("overlap must be smaller than every tile")
        if any((tile + 2 * args.overlap) % 16 for tile in candidates):
            raise SystemExit("tile + 2*overlap must be divisible by 16")
    if args.command == "analyze":
        prototype_values = (
            args.prototype_baseline_summary,
            args.prototype_candidate_summary,
        )
        if any(value is not None for value in prototype_values) and not all(
            value is not None for value in prototype_values
        ):
            raise SystemExit(
                "prototype analysis requires baseline and candidate run summaries"
            )
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
