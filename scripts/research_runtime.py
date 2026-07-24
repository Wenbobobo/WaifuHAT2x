"""Isolated, process-level Real-HAT runtime research.

This runner deliberately does not import torch. Every GPU candidate runs in a
fresh child process after its backend environment has been established. It
accepts only copied representative pages and refuses to run while the
production watchdog or worker is active.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np
from PIL import Image, ImageOps


RESEARCH_SCHEMA = "research-v1"
RESEARCH_KIND = "real_hat_runtime_research"
DEFAULT_MANIFEST = Path("benchmark/representative/manifest.json")
DEFAULT_NORMAL_MODEL = Path("models/hat/Real_HAT_GAN_SRx4.pth")
DEFAULT_SHARPER_MODEL = Path("models/hat/Real_HAT_GAN_SRx4_sharper.pth")
DEFAULT_MICRO_INDEXES = (1, 2, 10, 12)
DEFAULT_CANARY_INDEXES = (1, 2, 5, 9, 10, 11, 12, 13, 18, 24, 26, 30)
DEFAULT_COLD_INDEXES = (1, 2, 10, 12)
DEFAULT_TILES = (256, 320)
DEFAULT_OVERLAP = 16
BACKEND_ENVIRONMENT_KEY = "TORCH_BLAS_PREFER_HIPBLASLT"
BACKEND_CANDIDATES = ("default", "hipblaslt")
PRODUCTION_PROCESS_PATTERN = re.compile(
    r"(?:run_with_watchdog\.py|\bpython(?:\d+(?:\.\d+)*)?\b.*\s-m\s+waifuhat2x\b)",
    re.IGNORECASE,
)


class ResearchError(RuntimeError):
    """A safety, evidence, or candidate execution precondition failed."""


@dataclass(frozen=True)
class Candidate:
    identifier: str
    backend_environment: Mapping[str, str | None]


@dataclass(frozen=True)
class IsolatedPage:
    index: int
    route: str
    copied_path: str
    copied_sha256: str
    width: int
    height: int
    short_edge: int
    long_edge: int
    grayscale: bool
    odd_dimension: bool
    source_mode: str
    source_format: str
    file_bytes: int

    def record(self) -> dict[str, Any]:
        # Do not repeat the original formal-library path from the manifest.
        return {
            "index": self.index,
            "route": self.route,
            "copied_path": self.copied_path,
            "copied_sha256": self.copied_sha256,
            "width": self.width,
            "height": self.height,
            "short_edge": self.short_edge,
            "long_edge": self.long_edge,
            "grayscale": self.grayscale,
            "odd_dimension": self.odd_dimension,
            "source_mode": self.source_mode,
            "source_format": self.source_format,
            "file_bytes": self.file_bytes,
        }


@dataclass(frozen=True)
class ManifestInventory:
    path: Path
    input_root: Path
    sha256: str
    pages: tuple[IsolatedPage, ...]
    threshold: int


@dataclass
class EagerOutcome:
    record: dict[str, Any]
    payload: dict[str, Any]


@dataclass
class PipelineOutcome:
    record: dict[str, Any]
    payload: dict[str, Any]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ResearchError(f"Expected a JSON object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def paths_overlap(left: Path, right: Path) -> bool:
    first = left.expanduser().resolve(strict=False)
    second = right.expanduser().resolve(strict=False)
    return relative_to(first, second) or relative_to(second, first)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite value")
    return parsed


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return normalized or "run"


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise ResearchError(f"Required regular file is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _assert_no_symlink_components(path: Path, root: Path) -> None:
    root = root.resolve()
    resolved = path.resolve()
    if not relative_to(resolved, root) or resolved == root:
        raise ResearchError(f"Path escapes isolated root: {path}")
    current = root
    for part in resolved.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ResearchError(f"Isolated path traverses a symbolic link: {current}")


def _manifest_page(
    raw: Any,
    *,
    manifest_root: Path,
    input_root: Path,
    threshold: int,
) -> IsolatedPage:
    if not isinstance(raw, dict):
        raise ResearchError("Manifest pages must be objects")
    try:
        index = int(raw["index"])
        copied_path = str(raw["copied_path"])
        expected_sha = str(raw["copied_sha256"])
        route = str(raw["route"])
        width = int(raw["width"])
        height = int(raw["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchError("Manifest page is missing a required field") from exc
    copied_relative = Path(copied_path)
    if (
        index < 1
        or copied_relative.is_absolute()
        or ".." in copied_relative.parts
        or route not in {"normal", "sharper"}
        or width < 1
        or height < 1
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
    ):
        raise ResearchError(f"Manifest page {index} is not canonical")
    source = (manifest_root / copied_relative).resolve()
    _assert_no_symlink_components(source, input_root)
    if source.is_symlink() or not source.is_file():
        raise ResearchError(f"Isolated representative page is unsafe or missing: {source}")
    if sha256_file(source) != expected_sha:
        raise ResearchError(f"Isolated representative page hash drifted: {source}")
    with Image.open(source) as opened:
        if getattr(opened, "n_frames", 1) != 1:
            raise ResearchError(f"Representative page is animated: {source}")
        image = ImageOps.exif_transpose(opened)
        image.load()
    actual_width, actual_height = image.size
    if (actual_width, actual_height) != (width, height):
        raise ResearchError(
            f"Representative dimensions drifted for page {index}: "
            f"{(actual_width, actual_height)} != {(width, height)}"
        )
    short_edge = min(width, height)
    expected_route = "normal" if short_edge < threshold else "sharper"
    if route != expected_route:
        raise ResearchError(
            f"Representative route drifted for page {index}: {route} != {expected_route}"
        )
    return IsolatedPage(
        index=index,
        route=route,
        copied_path=copied_path,
        copied_sha256=expected_sha,
        width=width,
        height=height,
        short_edge=short_edge,
        long_edge=max(width, height),
        grayscale=bool(raw.get("grayscale")),
        odd_dimension=bool(raw.get("odd_dimension")),
        source_mode=str(raw.get("source_mode", "")),
        source_format=str(raw.get("source_format", "")),
        file_bytes=int(raw.get("file_bytes", source.stat().st_size)),
    )


def load_manifest(
    manifest_path: Path, indexes: Sequence[int], threshold: int
) -> ManifestInventory:
    manifest = manifest_path.expanduser().resolve()
    if manifest.is_symlink() or not manifest.is_file():
        raise ResearchError(f"Representative manifest is unsafe or missing: {manifest}")
    payload = read_json(manifest)
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "real_hat_representative_manifest"
    ):
        raise ResearchError(f"Unsupported representative manifest: {manifest}")
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise ResearchError("Representative manifest has no pages")
    manifest_root = manifest.parent
    input_root = (manifest_root / "inputs").resolve()
    if input_root.is_symlink() or not input_root.is_dir():
        raise ResearchError(f"Isolated representative inputs are missing: {input_root}")
    pages_by_index: dict[int, IsolatedPage] = {}
    for raw in raw_pages:
        page = _manifest_page(
            raw,
            manifest_root=manifest_root,
            input_root=input_root,
            threshold=threshold,
        )
        if page.index in pages_by_index:
            raise ResearchError(f"Duplicate representative page index: {page.index}")
        pages_by_index[page.index] = page
    if len(indexes) != len(set(indexes)):
        raise ResearchError("Requested page indexes must not contain duplicates")
    missing = [index for index in indexes if index not in pages_by_index]
    if missing:
        raise ResearchError(f"Representative manifest lacks page indexes: {missing}")
    pages = tuple(pages_by_index[index] for index in indexes)
    if not pages:
        raise ResearchError("At least one isolated representative page is required")
    return ManifestInventory(
        path=manifest,
        input_root=input_root,
        sha256=sha256_file(manifest),
        pages=pages,
        threshold=threshold,
    )


def require_dual_route(pages: Sequence[IsolatedPage]) -> None:
    routes = {page.route for page in pages}
    if routes != {"normal", "sharper"}:
        raise ResearchError(
            "The manifest eager runner requires both normal and sharper pages"
        )


def production_processes() -> list[dict[str, Any]]:
    """Return watchdog/worker processes, or fail closed when process inspection fails."""

    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ResearchError("Cannot establish the production process safety gate") from exc
    if completed.returncode != 0:
        raise ResearchError(
            "Production process safety gate failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    active: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        pid, command = int(fields[0]), fields[1]
        if PRODUCTION_PROCESS_PATTERN.search(command):
            active.append({"pid": pid, "command": command})
    return active


def require_idle_production() -> None:
    active = production_processes()
    if active:
        details = "; ".join(f"pid={item['pid']} {item['command']}" for item in active)
        raise ResearchError(
            "Refusing GPU research while a production watchdog or worker is active: "
            + details
        )


def candidate_definitions(names: Sequence[str]) -> list[Candidate]:
    if len(names) != len(set(names)):
        raise ResearchError("Candidate names must not contain duplicates")
    candidates: list[Candidate] = []
    for name in names:
        if name == "default":
            candidates.append(Candidate(name, {BACKEND_ENVIRONMENT_KEY: None}))
        elif name == "hipblaslt":
            candidates.append(Candidate(name, {BACKEND_ENVIRONMENT_KEY: "1"}))
        else:
            raise ResearchError(f"Unsupported backend candidate: {name}")
    if not candidates:
        raise ResearchError("At least one backend candidate is required")
    return candidates


def child_environment(candidate: Candidate) -> dict[str, str]:
    environment = os.environ.copy()
    for name, value in candidate.backend_environment.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    return environment


def actual_backend_environment(environment: Mapping[str, str]) -> dict[str, str | None]:
    return {BACKEND_ENVIRONMENT_KEY: environment.get(BACKEND_ENVIRONMENT_KEY)}


def _toml_paths(config_path: Path) -> dict[str, Path]:
    try:
        import tomllib
    except ModuleNotFoundError as exc:  # pragma: no cover - Python 3.12 requires it.
        raise ResearchError("Python tomllib is required for research isolation") from exc
    config = config_path.expanduser().resolve()
    if config.is_symlink() or not config.is_file():
        raise ResearchError(f"Configuration is unsafe or missing: {config}")
    payload = tomllib.loads(config.read_text(encoding="utf-8"))
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        raise ResearchError("Configuration has no [paths] table")
    result: dict[str, Path] = {}
    for name in ("input", "output", "models"):
        value = paths.get(name)
        if not isinstance(value, str) or not value:
            raise ResearchError(f"Configuration paths.{name} must be a non-empty string")
        raw = Path(value).expanduser()
        result[name] = (config.parent / raw).resolve() if not raw.is_absolute() else raw.resolve()
    return result


def prepare_output_root(
    output_root: Path,
    *,
    inventory: ManifestInventory | None,
    config_path: Path | None,
) -> Path:
    root = output_root.expanduser().resolve(strict=False)
    if root.exists() or root.is_symlink():
        raise ResearchError(f"Research output root must be new: {root}")
    forbidden: list[tuple[str, Path]] = []
    if inventory is not None:
        forbidden.extend(
            (
                ("isolated input", inventory.input_root),
                ("representative manifest root", inventory.path.parent),
            )
        )
    if config_path is not None:
        for name, path in _toml_paths(config_path).items():
            if name in {"input", "output"}:
                forbidden.append((f"configured {name}", path))
    for label, path in forbidden:
        if paths_overlap(root, path):
            raise ResearchError(f"Research output overlaps {label}: {root} <-> {path}")
    root.mkdir(parents=True, exist_ok=False)
    return root


def metric(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "cv_percent": None,
        }
    mean = statistics.fmean(values)
    cv = statistics.pstdev(values) / mean * 100 if len(values) > 1 and mean else None
    return {
        "count": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "cv_percent": cv,
    }


def percent_reduction(baseline: float, candidate: float) -> float | None:
    if baseline <= 0 or candidate <= 0:
        return None
    return (baseline - candidate) / baseline * 100


def selected_environment() -> dict[str, str | None]:
    keys = (
        "HSA_OVERRIDE_GFX_VERSION",
        "PYTORCH_ROCM_ARCH",
        "ROCM_PATH",
        "WAIFUHAT_RUNTIME_ROOT",
        BACKEND_ENVIRONMENT_KEY,
    )
    return {name: os.environ.get(name) for name in keys}


def session_environment(
    *,
    command: str,
    config_path: Path | None,
    inventory: ManifestInventory | None,
    normal_model: Path | None,
    sharper_model: Path | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": RESEARCH_SCHEMA,
        "kind": "real_hat_runtime_environment",
        "command": command,
        "created_at": utc_now(),
        "project_root": str(project_root()),
        "runner": file_record(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "parent_environment": selected_environment(),
    }
    if config_path is not None:
        payload["config"] = file_record(config_path)
    if inventory is not None:
        payload["manifest"] = {
            "path": str(inventory.path),
            "sha256": inventory.sha256,
            "input_root": str(inventory.input_root),
            "threshold": inventory.threshold,
        }
    if normal_model is not None and sharper_model is not None:
        payload["models"] = {
            "normal": file_record(normal_model),
            "sharper": file_record(sharper_model),
        }
    return payload


def initialize_session(
    *,
    command: str,
    output_root: Path,
    config_path: Path | None,
    inventory: ManifestInventory | None,
    normal_model: Path | None,
    sharper_model: Path | None,
    candidates: Sequence[Candidate],
    schedule: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = prepare_output_root(
        output_root, inventory=inventory, config_path=config_path
    )
    environment = session_environment(
        command=command,
        config_path=config_path,
        inventory=inventory,
        normal_model=normal_model,
        sharper_model=sharper_model,
    )
    write_json(root / "environment.json", environment)
    candidate_payload = {
        "schema_version": RESEARCH_SCHEMA,
        "kind": "real_hat_runtime_candidates",
        "created_at": utc_now(),
        "candidates": [
            {
                "id": candidate.identifier,
                "backend_environment": dict(candidate.backend_environment),
                "effective_child_environment": actual_backend_environment(
                    child_environment(candidate)
                ),
            }
            for candidate in candidates
        ],
        "schedule": [dict(item) for item in schedule],
    }
    write_json(root / "candidate.json", candidate_payload)
    rows = [] if inventory is None else [page.record() for page in inventory.pages]
    write_jsonl(root / "pages.jsonl", rows)
    summary = {
        "schema_version": RESEARCH_SCHEMA,
        "kind": RESEARCH_KIND,
        "status": "running",
        "command": command,
        "created_at": utc_now(),
        "output_root": str(root),
        "environment": {
            "path": "environment.json",
            "sha256": sha256_file(root / "environment.json"),
        },
        "candidates": {
            "path": "candidate.json",
            "sha256": sha256_file(root / "candidate.json"),
        },
        "pages": {
            "path": "pages.jsonl",
            "sha256": sha256_file(root / "pages.jsonl"),
            "count": len(rows),
        },
        "runs": [],
    }
    write_json(root / "summary.json", summary)
    return {"root": root, "summary": summary}


def finish_session(
    session: dict[str, Any], *, status: str, **fields: Any
) -> dict[str, Any]:
    summary = session["summary"]
    summary.update(fields)
    summary["status"] = status
    summary["finished_at"] = utc_now()
    write_json(Path(session["root"]) / "summary.json", summary)
    return summary


def update_session_runs(session: dict[str, Any], runs: Sequence[Mapping[str, Any]]) -> None:
    summary = session["summary"]
    summary["runs"] = [dict(run) for run in runs]
    summary["updated_at"] = utc_now()
    write_json(Path(session["root"]) / "summary.json", summary)


def execute_child(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    log_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(environment),
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            returncode = process.wait(timeout=10)
    report = {
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": time.perf_counter() - started,
        "command": list(command),
        "log": {
            "path": log_path.name,
            "sha256": sha256_file(log_path),
            "bytes": log_path.stat().st_size,
        },
    }
    if returncode != 0:
        raise ResearchError(
            f"Research child failed with exit code {returncode}; inspect {log_path}"
        )
    return report


def eager_script() -> Path:
    path = project_root() / "scripts" / "benchmark_manifest_eager.py"
    if not path.is_file():
        raise ResearchError(f"Manifest eager benchmark is missing: {path}")
    return path


def _require_eager_payload(
    payload: Mapping[str, Any],
    *,
    expected_indexes: Sequence[int],
    candidate: Candidate,
    expected_rounds: int,
) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "real_hat_manifest_eager_benchmark"
        or payload.get("status") != "complete"
    ):
        raise ResearchError("Manifest eager child did not produce a complete summary")
    if payload.get("page_order") != list(expected_indexes):
        raise ResearchError("Manifest eager page order differs from the requested plan")
    rounds = payload.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != expected_rounds:
        raise ResearchError("Manifest eager round count differs from the requested plan")
    actual = payload.get("backend_environment")
    expected = actual_backend_environment(child_environment(candidate))
    if actual != expected:
        raise ResearchError(
            "Manifest eager child did not attest the expected backend environment: "
            f"{actual!r} != {expected!r}"
        )


def run_eager_child(
    *,
    root: Path,
    sequence: int,
    phase: str,
    candidate: Candidate,
    inventory: ManifestInventory,
    normal_model: Path,
    sharper_model: Path,
    tile: int,
    overlap: int,
    warmups: int,
    rounds: int,
    warmup_crop: int,
    timeout_seconds: float,
    save_first_round: bool,
    pair: int | None = None,
) -> EagerOutcome:
    label = safe_name(f"{sequence:03d}-{phase}-{candidate.identifier}-t{tile}")
    run_root = root / "runs" / label
    run_root.mkdir(parents=True, exist_ok=False)
    output_root = run_root / "eager-output"
    command = [
        sys.executable,
        str(eager_script()),
        "--manifest",
        str(inventory.path),
        "--page-indexes",
        *(str(page.index) for page in inventory.pages),
        "--normal-model",
        str(normal_model.resolve()),
        "--sharper-model",
        str(sharper_model.resolve()),
        "--threshold",
        str(inventory.threshold),
        "--tile",
        str(tile),
        "--overlap",
        str(overlap),
        "--rounds",
        str(rounds),
        "--warmups-per-model",
        str(warmups),
        "--warmup-crop",
        str(warmup_crop),
        "--output-root",
        str(output_root),
        "--run-name",
        "eager",
    ]
    if save_first_round:
        command.append("--save-first-round")
    child_env = child_environment(candidate)
    process = execute_child(
        command,
        environment=child_env,
        log_path=run_root / "child.log",
        timeout_seconds=timeout_seconds,
    )
    summary_path = output_root / "eager" / "batch_summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        raise ResearchError(f"Manifest eager child produced no summary: {summary_path}")
    payload = read_json(summary_path)
    _require_eager_payload(
        payload,
        expected_indexes=[page.index for page in inventory.pages],
        candidate=candidate,
        expected_rounds=rounds,
    )
    record = {
        "sequence": sequence,
        "phase": phase,
        "pair": pair,
        "candidate": candidate.identifier,
        "tile": tile,
        "overlap": overlap,
        "backend_environment": actual_backend_environment(child_env),
        "process": process,
        "summary": {
            "path": str(summary_path.relative_to(root)),
            "sha256": sha256_file(summary_path),
            "bytes": summary_path.stat().st_size,
        },
    }
    write_json(run_root / "run.json", record)
    return EagerOutcome(record=record, payload=payload)


def eager_pages(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rounds = payload.get("rounds")
    if not isinstance(rounds, list):
        raise ResearchError("Manifest eager summary has no rounds")
    pages: list[dict[str, Any]] = []
    for round_payload in rounds:
        if not isinstance(round_payload, dict) or not isinstance(
            round_payload.get("pages"), list
        ):
            raise ResearchError("Manifest eager summary has malformed page records")
        for page in round_payload["pages"]:
            if not isinstance(page, dict):
                raise ResearchError("Manifest eager page record is not an object")
            pages.append(page)
    return pages


def eager_page_hashes(payload: Mapping[str, Any]) -> dict[int, set[str]]:
    result: dict[int, set[str]] = defaultdict(set)
    for page in eager_pages(payload):
        index = int(page["index"])
        digest = page.get("pixel_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ResearchError(f"Manifest eager page {index} has no canonical pixel hash")
        result[index].add(digest)
    return dict(result)


def all_eager_hashes_deterministic(outcomes: Sequence[EagerOutcome]) -> bool:
    by_page: dict[int, set[str]] = defaultdict(set)
    for outcome in outcomes:
        for index, hashes in eager_page_hashes(outcome.payload).items():
            by_page[index].update(hashes)
    return bool(by_page) and all(len(hashes) == 1 for hashes in by_page.values())


def eager_hash_determinism_by_tile(
    outcomes: Sequence[EagerOutcome],
) -> dict[int, bool]:
    """Check repeat determinism within each fixed-tile microbenchmark cell."""
    by_tile: dict[int, list[EagerOutcome]] = defaultdict(list)
    for outcome in outcomes:
        tile = outcome.record.get("tile")
        if not isinstance(tile, int) or isinstance(tile, bool) or tile < 1:
            raise ResearchError("Fixed-tile outcome has no valid tile")
        by_tile[tile].append(outcome)
    return {
        tile: all_eager_hashes_deterministic(tile_outcomes)
        for tile, tile_outcomes in sorted(by_tile.items())
    }


def eager_round_wall(payload: Mapping[str, Any]) -> float:
    rounds = payload.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 1:
        raise ResearchError("Paired eager comparison requires exactly one measured round")
    value = rounds[0].get("loop_wall_seconds") if isinstance(rounds[0], dict) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ResearchError("Manifest eager round has no positive wall time")
    return float(value)


def eager_round_page_map(payload: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    rounds = payload.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 1:
        raise ResearchError("Paired eager comparison requires exactly one measured round")
    raw_pages = rounds[0].get("pages") if isinstance(rounds[0], dict) else None
    if not isinstance(raw_pages, list):
        raise ResearchError("Manifest eager round has no pages")
    result: dict[int, dict[str, Any]] = {}
    for raw in raw_pages:
        if not isinstance(raw, dict):
            raise ResearchError("Manifest eager round has a malformed page")
        index = int(raw.get("index", 0))
        if index < 1 or index in result:
            raise ResearchError("Manifest eager round has duplicate page indexes")
        value = raw.get("upscale_wall_seconds")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ResearchError(f"Manifest eager page {index} has no positive wall time")
        result[index] = raw
    return result


def output_png_path(outcome: EagerOutcome, page: Mapping[str, Any], root: Path) -> Path:
    raw = page.get("png_path")
    if not isinstance(raw, str) or not raw:
        raise ResearchError("Canary eager run did not retain its first-round PNG")
    path = Path(raw).resolve()
    if not relative_to(path, root.resolve()) or path.is_symlink() or not path.is_file():
        raise ResearchError(f"Canary PNG escapes its owned research root: {path}")
    expected = page.get("png_sha256")
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise ResearchError(f"Canary PNG hash drifted: {path}")
    return path


def compare_png_sets(
    baseline: EagerOutcome, candidate: EagerOutcome, root: Path
) -> dict[str, Any]:
    baseline_pages = eager_round_page_map(baseline.payload)
    candidate_pages = eager_round_page_map(candidate.payload)
    if set(baseline_pages) != set(candidate_pages):
        raise ResearchError("Canary candidate page set differs from its baseline")
    histogram = np.zeros(256, dtype=np.int64)
    squared_error = 0
    channel_count = 0
    page_hash_equal = True
    for index in sorted(baseline_pages):
        base_page = baseline_pages[index]
        candidate_page = candidate_pages[index]
        if base_page.get("pixel_sha256") != candidate_page.get("pixel_sha256"):
            page_hash_equal = False
        baseline_path = output_png_path(baseline, base_page, root)
        candidate_path = output_png_path(candidate, candidate_page, root)
        with Image.open(baseline_path) as opened:
            baseline_array = np.asarray(opened).astype(np.int16)
        with Image.open(candidate_path) as opened:
            candidate_array = np.asarray(opened).astype(np.int16)
        if baseline_array.shape != candidate_array.shape:
            raise ResearchError(
                f"Canary output shape differs for page {index}: "
                f"{baseline_array.shape} != {candidate_array.shape}"
            )
        difference = np.abs(baseline_array - candidate_array)
        histogram += np.bincount(difference.reshape(-1), minlength=256)[:256]
        squared_error += int(np.square(difference, dtype=np.int64).sum())
        channel_count += int(difference.size)
    if channel_count < 1:
        raise ResearchError("Canary image comparison observed no pixels")
    cumulative = np.cumsum(histogram)
    percentile_rank = math.ceil(channel_count * 0.95)
    p95 = int(np.searchsorted(cumulative, percentile_rank, side="left"))
    max_difference = int(np.flatnonzero(histogram)[-1])
    mse = squared_error / channel_count
    infinite_psnr = squared_error == 0
    psnr = None if infinite_psnr else 10.0 * math.log10((255.0**2) / mse)
    return {
        "page_count": len(baseline_pages),
        "channel_count": channel_count,
        "pixel_hash_equal": page_hash_equal,
        "max_abs_difference": max_difference,
        "p95_abs_difference": p95,
        "mse": mse,
        "psnr_db": psnr,
        "psnr_infinite": infinite_psnr,
    }


def combine_png_comparisons(comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not comparisons:
        raise ResearchError("No canary image comparisons were recorded")
    total_channels = sum(int(item["channel_count"]) for item in comparisons)
    weighted_mse = (
        sum(float(item["mse"]) * int(item["channel_count"]) for item in comparisons)
        / total_channels
    )
    infinite_psnr = weighted_mse == 0
    psnr = (
        None
        if infinite_psnr
        else 10.0 * math.log10((255.0**2) / weighted_mse)
    )
    return {
        "pair_count": len(comparisons),
        "page_comparisons": sum(int(item["page_count"]) for item in comparisons),
        "channel_count": total_channels,
        "pixel_hash_equal": all(bool(item["pixel_hash_equal"]) for item in comparisons),
        "max_abs_difference": max(int(item["max_abs_difference"]) for item in comparisons),
        "p95_abs_difference": max(int(item["p95_abs_difference"]) for item in comparisons),
        "mse": weighted_mse,
        "psnr_db": psnr,
        "psnr_infinite": infinite_psnr,
    }


def paired_eager_performance(
    pairs: Sequence[tuple[EagerOutcome, EagerOutcome]]
) -> dict[str, Any]:
    if not pairs:
        raise ResearchError("No paired eager measurements were recorded")
    baseline_walls: list[float] = []
    candidate_walls: list[float] = []
    ratios: list[float] = []
    by_cell: dict[tuple[str, int], list[float]] = defaultdict(list)
    for baseline, candidate in pairs:
        baseline_wall = eager_round_wall(baseline.payload)
        candidate_wall = eager_round_wall(candidate.payload)
        baseline_walls.append(baseline_wall)
        candidate_walls.append(candidate_wall)
        ratios.append(candidate_wall / baseline_wall)
        baseline_pages = eager_round_page_map(baseline.payload)
        candidate_pages = eager_round_page_map(candidate.payload)
        if set(baseline_pages) != set(candidate_pages):
            raise ResearchError("Paired eager page sets differ")
        for index, base_page in baseline_pages.items():
            candidate_page = candidate_pages[index]
            route = str(base_page.get("route"))
            tile = int(base_page.get("selected_tile", 0))
            if (
                route not in {"normal", "sharper"}
                or tile < 1
                or candidate_page.get("route") != route
                or int(candidate_page.get("selected_tile", 0)) != tile
            ):
                raise ResearchError(f"Paired eager route/tile drifted for page {index}")
            by_cell[(route, tile)].append(
                float(candidate_page["upscale_wall_seconds"])
                / float(base_page["upscale_wall_seconds"])
            )
    baseline_mean = statistics.fmean(baseline_walls)
    candidate_mean = statistics.fmean(candidate_walls)
    cells = [
        {
            "route": route,
            "tile": tile,
            "candidate_to_baseline_ratio": metric(values),
            "regression_over_2_percent": any(value > 1.02 for value in values),
        }
        for (route, tile), values in sorted(by_cell.items())
    ]
    return {
        "pair_count": len(pairs),
        "baseline_wall_seconds": metric(baseline_walls),
        "candidate_wall_seconds": metric(candidate_walls),
        "candidate_to_baseline_ratio": metric(ratios),
        "wall_reduction_percent": percent_reduction(baseline_mean, candidate_mean),
        "route_tile": cells,
        "no_route_tile_regression_over_2_percent": not any(
            item["regression_over_2_percent"] for item in cells
        ),
    }


def interleaved_pair_schedule(candidates: Sequence[Candidate], pairs: int) -> list[dict[str, Any]]:
    if pairs < 1:
        raise ResearchError("At least one paired round is required")
    if {candidate.identifier for candidate in candidates} != set(BACKEND_CANDIDATES):
        return [
            {"pair": pair, "candidate": candidate.identifier}
            for pair in range(1, pairs + 1)
            for candidate in candidates
        ]
    schedule: list[dict[str, Any]] = []
    for pair in range(1, pairs + 1):
        identifiers = ("default", "hipblaslt") if pair % 2 else ("hipblaslt", "default")
        schedule.extend(
            {"pair": pair, "candidate": identifier} for identifier in identifiers
        )
    return schedule


def resolve_research_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = project_root()
    config = args.config.expanduser().resolve()
    normal = args.normal_model.expanduser()
    sharper = args.sharper_model.expanduser()
    normal = (root / normal).resolve() if not normal.is_absolute() else normal.resolve()
    sharper = (root / sharper).resolve() if not sharper.is_absolute() else sharper.resolve()
    file_record(config)
    file_record(normal)
    file_record(sharper)
    return config, normal, sharper


def dry_run_record(
    *,
    session: dict[str, Any],
    plan: Mapping[str, Any],
) -> int:
    finish_session(session, status="planned", plan=dict(plan))
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


def run_micro(args: argparse.Namespace) -> int:
    config, normal_model, sharper_model = resolve_research_paths(args)
    inventory = load_manifest(args.manifest, args.page_indexes, args.threshold)
    require_dual_route(inventory.pages)
    candidates = candidate_definitions(args.candidates)
    tiles = tuple(args.tiles)
    if len(tiles) != len(set(tiles)):
        raise ResearchError("Micro tile values must not contain duplicates")
    if any(args.overlap >= tile for tile in tiles):
        raise ResearchError("Micro overlap must be smaller than every tile")
    schedule: list[dict[str, Any]] = []
    sequence = 1
    for tile_index, tile in enumerate(tiles):
        ordered = list(candidates if tile_index % 2 == 0 else reversed(candidates))
        for candidate in ordered:
            schedule.append(
                {
                    "sequence": sequence,
                    "phase": "micro",
                    "candidate": candidate.identifier,
                    "tile": tile,
                }
            )
            sequence += 1
    session = initialize_session(
        command="micro",
        output_root=args.output_root,
        config_path=config,
        inventory=inventory,
        normal_model=normal_model,
        sharper_model=sharper_model,
        candidates=candidates,
        schedule=schedule,
    )
    if args.dry_run:
        return dry_run_record(session=session, plan={"schedule": schedule})
    try:
        require_idle_production()
        outcomes: list[EagerOutcome] = []
        for item in schedule:
            candidate = next(
                value for value in candidates if value.identifier == item["candidate"]
            )
            outcome = run_eager_child(
                root=session["root"],
                sequence=int(item["sequence"]),
                phase="micro",
                candidate=candidate,
                inventory=inventory,
                normal_model=normal_model,
                sharper_model=sharper_model,
                tile=int(item["tile"]),
                overlap=args.overlap,
                warmups=args.warmups,
                rounds=args.rounds,
                warmup_crop=min(int(item["tile"]), args.warmup_crop),
                timeout_seconds=args.timeout_seconds,
                save_first_round=False,
            )
            outcomes.append(outcome)
            update_session_runs(session, [outcome.record for outcome in outcomes])

        by_candidate_tile = {
            (outcome.record["candidate"], int(outcome.record["tile"])): outcome
            for outcome in outcomes
        }
        cells: list[dict[str, Any]] = []
        comparisons: list[dict[str, Any]] = []
        for outcome in outcomes:
            by_route: dict[str, list[float]] = defaultdict(list)
            for page in eager_pages(outcome.payload):
                by_route[str(page["route"])].append(float(page["upscale_wall_seconds"]))
            round_walls = [
                float(round_payload["loop_wall_seconds"])
                for round_payload in outcome.payload["rounds"]
            ]
            cells.append(
                {
                    "candidate": outcome.record["candidate"],
                    "tile": outcome.record["tile"],
                    "round_wall_seconds": metric(round_walls),
                    "route_upscale_wall_seconds": {
                        route: metric(values) for route, values in sorted(by_route.items())
                    },
                    "pixel_deterministic": all_eager_hashes_deterministic([outcome]),
                }
            )
        for tile in tiles:
            baseline = by_candidate_tile.get(("default", tile))
            candidate = by_candidate_tile.get(("hipblaslt", tile))
            if baseline is None or candidate is None:
                continue
            base_pages: dict[str, list[float]] = defaultdict(list)
            candidate_pages: dict[str, list[float]] = defaultdict(list)
            for page in eager_pages(baseline.payload):
                base_pages[str(page["route"])].append(float(page["upscale_wall_seconds"]))
            for page in eager_pages(candidate.payload):
                candidate_pages[str(page["route"])].append(
                    float(page["upscale_wall_seconds"])
                )
            route_comparisons = []
            for route in sorted(set(base_pages) | set(candidate_pages)):
                base_mean = statistics.fmean(base_pages[route])
                candidate_mean = statistics.fmean(candidate_pages[route])
                route_comparisons.append(
                    {
                        "route": route,
                        "wall_reduction_percent": percent_reduction(
                            base_mean, candidate_mean
                        ),
                        "candidate_to_baseline_ratio": candidate_mean / base_mean,
                        "regression_over_2_percent": candidate_mean / base_mean > 1.02,
                    }
                )
            baseline_round = [
                float(item["loop_wall_seconds"])
                for item in baseline.payload["rounds"]
            ]
            candidate_round = [
                float(item["loop_wall_seconds"])
                for item in candidate.payload["rounds"]
            ]
            comparisons.append(
                {
                    "tile": tile,
                    "baseline_round_wall_seconds": metric(baseline_round),
                    "candidate_round_wall_seconds": metric(candidate_round),
                    "wall_reduction_percent": percent_reduction(
                        statistics.fmean(baseline_round), statistics.fmean(candidate_round)
                    ),
                    "route_cells": route_comparisons,
                    "pixel_hash_equal": eager_page_hashes(baseline.payload)
                    == eager_page_hashes(candidate.payload),
                }
            )
        reductions = [
            float(item["wall_reduction_percent"])
            for item in comparisons
            if item["wall_reduction_percent"] is not None
        ]
        critical_regression = any(
            cell["regression_over_2_percent"]
            for comparison in comparisons
            for cell in comparison["route_cells"]
        )
        candidate_outcomes = [
            outcome for outcome in outcomes if outcome.record["candidate"] == "hipblaslt"
        ]
        candidate_determinism_by_tile = (
            eager_hash_determinism_by_tile(candidate_outcomes)
            if candidate_outcomes
            else None
        )
        decision = {
            "candidate_hashes_deterministic_by_tile": candidate_determinism_by_tile,
            "all_candidate_hashes_deterministic": (
                all(candidate_determinism_by_tile.values())
                if candidate_determinism_by_tile is not None
                else None
            ),
            "no_route_tile_regression_over_2_percent": not critical_regression,
            "mean_wall_reduction_percent": statistics.fmean(reductions) if reductions else None,
            "meets_micro_speed_gate": bool(reductions)
            and statistics.fmean(reductions) >= 3.0
            and not critical_regression,
        }
        finish_session(
            session,
            status="complete",
            micro={"cells": cells, "comparisons": comparisons, "decision": decision},
        )
        return 0
    except BaseException as exc:
        finish_session(
            session,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise


def run_cold(args: argparse.Namespace) -> int:
    config, normal_model, sharper_model = resolve_research_paths(args)
    inventory = load_manifest(args.manifest, args.page_indexes, args.threshold)
    require_dual_route(inventory.pages)
    candidates = candidate_definitions(args.candidates)
    tiles = tuple(args.tiles)
    if len(tiles) != len(set(tiles)) or any(args.overlap >= tile for tile in tiles):
        raise ResearchError("Cold attribution tile/overlap configuration is invalid")
    schedule = [
        {
            "sequence": sequence,
            "phase": "fresh-process",
            "candidate": candidate.identifier,
            "tile": tile,
        }
        for sequence, (candidate, tile) in enumerate(
            (
                (candidate, tile)
                for tile in tiles
                for candidate in candidates
            ),
            start=1,
        )
    ]
    session = initialize_session(
        command="cold",
        output_root=args.output_root,
        config_path=config,
        inventory=inventory,
        normal_model=normal_model,
        sharper_model=sharper_model,
        candidates=candidates,
        schedule=schedule,
    )
    attribution_scope = {
        "fresh_python_process_per_candidate_tile": True,
        "persistent_cache_claim": False,
        "cache_policy": (
            "The runner neither clears nor treats driver, MIOpen, rocBLAS, or "
            "hipBLAS persistent state as cold. Results attribute only model preload "
            "and first-versus-steady execution inside a fresh Python process."
        ),
    }
    if args.dry_run:
        return dry_run_record(
            session=session, plan={"schedule": schedule, "attribution_scope": attribution_scope}
        )
    try:
        require_idle_production()
        outcomes: list[EagerOutcome] = []
        for item in schedule:
            candidate = next(
                value for value in candidates if value.identifier == item["candidate"]
            )
            tile = int(item["tile"])
            outcome = run_eager_child(
                root=session["root"],
                sequence=int(item["sequence"]),
                phase="fresh-process",
                candidate=candidate,
                inventory=inventory,
                normal_model=normal_model,
                sharper_model=sharper_model,
                tile=tile,
                overlap=args.overlap,
                warmups=args.warmups,
                # One full repeat gives every route a same-page steady observation
                # after its first complete-page shape has occurred.
                rounds=2,
                warmup_crop=tile,
                timeout_seconds=args.timeout_seconds,
                save_first_round=False,
            )
            outcomes.append(outcome)
            update_session_runs(session, [outcome.record for outcome in outcomes])
        cells = []
        for outcome in outcomes:
            preloads = outcome.payload.get("model_preloads")
            warmups = outcome.payload.get("warmups")
            if not isinstance(preloads, list) or not isinstance(warmups, list):
                raise ResearchError("Fresh-process eager summary lacks preload/warmup records")
            preload_by_route = {
                str(item["route"]): float(item["seconds"])
                for item in preloads
                if isinstance(item, dict) and isinstance(item.get("seconds"), (int, float))
            }
            warmups_by_route: dict[str, list[float]] = defaultdict(list)
            for item in warmups:
                if not isinstance(item, dict):
                    raise ResearchError("Fresh-process warmup record is malformed")
                warmups_by_route[str(item["route"])].append(float(item["wall_seconds"]))
            complete_pages_by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
            raw_rounds = outcome.payload.get("rounds")
            if not isinstance(raw_rounds, list) or len(raw_rounds) != 2:
                raise ResearchError("Fresh-process run did not retain two full rounds")
            for round_payload in raw_rounds:
                if not isinstance(round_payload, dict) or not isinstance(
                    round_payload.get("pages"), list
                ):
                    raise ResearchError("Fresh-process full-page record is malformed")
                for page in round_payload["pages"]:
                    if not isinstance(page, dict):
                        raise ResearchError("Fresh-process full-page item is malformed")
                    complete_pages_by_route[str(page.get("route"))].append(page)
            for route in ("normal", "sharper"):
                values = warmups_by_route.get(route, [])
                if len(values) != args.warmups or route not in preload_by_route:
                    raise ResearchError("Fresh-process warmup count differs from the plan")
                complete = complete_pages_by_route.get(route, [])
                if len(complete) < 2:
                    raise ResearchError(
                        "Fresh-process attribution needs a first and subsequent complete page "
                        f"for route {route}"
                    )
                first_complete = complete[0]
                first_index = int(first_complete.get("index", 0))
                first_complete_wall = float(first_complete.get("upscale_wall_seconds", 0.0))
                if first_index < 1 or first_complete_wall <= 0:
                    raise ResearchError("Fresh-process first complete-page timing is invalid")
                same_page_steady = [
                    float(page.get("upscale_wall_seconds", 0.0))
                    for page in complete[1:]
                    if int(page.get("index", 0)) == first_index
                    and float(page.get("upscale_wall_seconds", 0.0)) > 0
                ]
                subsequent_complete = [
                    float(page.get("upscale_wall_seconds", 0.0))
                    for page in complete[1:]
                    if float(page.get("upscale_wall_seconds", 0.0)) > 0
                ]
                cells.append(
                    {
                        "candidate": outcome.record["candidate"],
                        "tile": outcome.record["tile"],
                        "route": route,
                        "model_preload_seconds": preload_by_route[route],
                        "first_warmup_seconds": values[0],
                        "steady_warmup_seconds": metric(values[1:]),
                        "first_to_steady_ratio": (
                            values[0] / statistics.fmean(values[1:])
                            if len(values) > 1 and statistics.fmean(values[1:]) > 0
                            else None
                        ),
                        "first_complete_page": {
                            "index": first_index,
                            "selected_tile": int(first_complete.get("selected_tile", 0)),
                            "upscale_wall_seconds": first_complete_wall,
                        },
                        "subsequent_complete_page_seconds": metric(subsequent_complete),
                        "same_page_second_round_seconds": metric(same_page_steady),
                        "first_complete_to_same_page_steady_ratio": (
                            first_complete_wall / statistics.fmean(same_page_steady)
                            if same_page_steady
                            and statistics.fmean(same_page_steady) > 0
                            else None
                        ),
                    }
                )
        finish_session(
            session,
            status="complete",
            cold_attribution={"scope": attribution_scope, "cells": cells},
        )
        return 0
    except BaseException as exc:
        finish_session(
            session,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise


def run_canary12(args: argparse.Namespace) -> int:
    config, normal_model, sharper_model = resolve_research_paths(args)
    inventory = load_manifest(args.manifest, args.page_indexes, args.threshold)
    require_dual_route(inventory.pages)
    candidates = candidate_definitions(args.candidates)
    if tuple(sorted(set(args.adaptive_tiles))) != tuple(args.adaptive_tiles):
        raise ResearchError("Canary adaptive tiles must be unique and sorted")
    if args.overlap >= min(args.adaptive_tiles):
        raise ResearchError("Canary overlap must be smaller than every adaptive tile")
    pair_schedule = interleaved_pair_schedule(candidates, args.pairs)
    schedule = [
        {
            "sequence": sequence,
            "phase": "canary12",
            "pair": item["pair"],
            "candidate": item["candidate"],
            "adaptive_tiles": list(args.adaptive_tiles),
        }
        for sequence, item in enumerate(pair_schedule, start=1)
    ]
    session = initialize_session(
        command="canary12",
        output_root=args.output_root,
        config_path=config,
        inventory=inventory,
        normal_model=normal_model,
        sharper_model=sharper_model,
        candidates=candidates,
        schedule=schedule,
    )
    if args.dry_run:
        return dry_run_record(session=session, plan={"schedule": schedule})
    try:
        require_idle_production()
        outcomes: list[EagerOutcome] = []
        # The existing eager runner has one adaptive-tile mode.  Keeping this
        # conversion here makes the same per-page selection formula observable.
        for item in schedule:
            candidate = next(
                value for value in candidates if value.identifier == item["candidate"]
            )
            outcome = run_adaptive_eager_child(
                root=session["root"],
                sequence=int(item["sequence"]),
                phase="canary12",
                pair=int(item["pair"]),
                candidate=candidate,
                inventory=inventory,
                normal_model=normal_model,
                sharper_model=sharper_model,
                tiles=tuple(args.adaptive_tiles),
                overlap=args.overlap,
                warmups=args.warmups,
                rounds=1,
                warmup_crop=args.warmup_crop,
                timeout_seconds=args.timeout_seconds,
                save_first_round=True,
            )
            outcomes.append(outcome)
            update_session_runs(session, [outcome.record for outcome in outcomes])

        by_pair: dict[int, dict[str, EagerOutcome]] = defaultdict(dict)
        for outcome in outcomes:
            pair = outcome.record.get("pair")
            if not isinstance(pair, int):
                raise ResearchError("Canary outcome has no pairing key")
            identifier = str(outcome.record["candidate"])
            if identifier in by_pair[pair]:
                raise ResearchError(f"Duplicate canary candidate for pair {pair}")
            by_pair[pair][identifier] = outcome
        comparable_pairs: list[tuple[EagerOutcome, EagerOutcome]] = []
        quality_pairs: list[dict[str, Any]] = []
        if set(candidate.identifier for candidate in candidates) == set(BACKEND_CANDIDATES):
            for pair in range(1, args.pairs + 1):
                matching = by_pair.get(pair, {})
                if set(matching) != set(BACKEND_CANDIDATES):
                    raise ResearchError(f"Canary pair {pair} is incomplete")
                baseline = matching["default"]
                candidate = matching["hipblaslt"]
                comparable_pairs.append((baseline, candidate))
                quality = compare_png_sets(baseline, candidate, Path(session["root"]))
                quality_pairs.append({"pair": pair, **quality})
        performance = (
            paired_eager_performance(comparable_pairs) if comparable_pairs else None
        )
        image_quality = combine_png_comparisons(quality_pairs) if quality_pairs else None
        baseline_outcomes = [
            outcome for outcome in outcomes if outcome.record["candidate"] == "default"
        ]
        candidate_outcomes = [
            outcome
            for outcome in outcomes
            if outcome.record["candidate"] == "hipblaslt"
        ]
        quality_gate = (
            image_quality is not None
            and image_quality["p95_abs_difference"] == 0
            and image_quality["max_abs_difference"] <= 1
            and (
                image_quality["psnr_infinite"]
                or float(image_quality["psnr_db"]) >= 90.0
            )
            and all_eager_hashes_deterministic(baseline_outcomes)
            and all_eager_hashes_deterministic(candidate_outcomes)
        )
        performance_gate = (
            performance is not None
            and performance["wall_reduction_percent"] is not None
            and float(performance["wall_reduction_percent"]) >= 5.0
            and bool(performance["no_route_tile_regression_over_2_percent"])
        )
        finish_session(
            session,
            status="complete",
            canary12={
                "paired_performance": performance,
                "image_quality_by_pair": quality_pairs,
                "image_quality": image_quality,
                "determinism": {
                    "baseline": all_eager_hashes_deterministic(baseline_outcomes)
                    if baseline_outcomes
                    else None,
                    "candidate": all_eager_hashes_deterministic(candidate_outcomes)
                    if candidate_outcomes
                    else None,
                },
                "decision": {
                    "quality_gate": quality_gate,
                    "performance_gate": performance_gate,
                    "eligible_for_final30": quality_gate and performance_gate,
                },
            },
        )
        return 0
    except BaseException as exc:
        finish_session(
            session,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise


def run_adaptive_eager_child(
    *,
    root: Path,
    sequence: int,
    phase: str,
    pair: int | None,
    candidate: Candidate,
    inventory: ManifestInventory,
    normal_model: Path,
    sharper_model: Path,
    tiles: tuple[int, ...],
    overlap: int,
    warmups: int,
    rounds: int,
    warmup_crop: int,
    timeout_seconds: float,
    save_first_round: bool,
) -> EagerOutcome:
    label = safe_name(f"{sequence:03d}-{phase}-{candidate.identifier}-adaptive")
    run_root = root / "runs" / label
    run_root.mkdir(parents=True, exist_ok=False)
    output_root = run_root / "eager-output"
    command = [
        sys.executable,
        str(eager_script()),
        "--manifest",
        str(inventory.path),
        "--page-indexes",
        *(str(page.index) for page in inventory.pages),
        "--normal-model",
        str(normal_model.resolve()),
        "--sharper-model",
        str(sharper_model.resolve()),
        "--threshold",
        str(inventory.threshold),
        "--adaptive-tiles",
        *(str(tile) for tile in tiles),
        "--overlap",
        str(overlap),
        "--rounds",
        str(rounds),
        "--warmups-per-model",
        str(warmups),
        "--warmup-crop",
        str(warmup_crop),
        "--output-root",
        str(output_root),
        "--run-name",
        "eager",
    ]
    if save_first_round:
        command.append("--save-first-round")
    child_env = child_environment(candidate)
    process = execute_child(
        command,
        environment=child_env,
        log_path=run_root / "child.log",
        timeout_seconds=timeout_seconds,
    )
    summary_path = output_root / "eager" / "batch_summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        raise ResearchError(f"Adaptive eager child produced no summary: {summary_path}")
    payload = read_json(summary_path)
    _require_eager_payload(
        payload,
        expected_indexes=[page.index for page in inventory.pages],
        candidate=candidate,
        expected_rounds=rounds,
    )
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict) or configuration.get("adaptive_tiles") != list(tiles):
        raise ResearchError("Adaptive eager child did not retain the requested tile set")
    record = {
        "sequence": sequence,
        "phase": phase,
        "pair": pair,
        "candidate": candidate.identifier,
        "adaptive_tiles": list(tiles),
        "overlap": overlap,
        "backend_environment": actual_backend_environment(child_env),
        "process": process,
        "summary": {
            "path": str(summary_path.relative_to(root)),
            "sha256": sha256_file(summary_path),
            "bytes": summary_path.stat().st_size,
        },
    }
    write_json(run_root / "run.json", record)
    return EagerOutcome(record=record, payload=payload)


def e2e_module() -> Any:
    root = str(project_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from scripts import benchmark_pipeline_e2e as e2e
    except ModuleNotFoundError as exc:
        raise ResearchError("The isolated E2E benchmark module is unavailable") from exc
    return e2e


def run_pipeline_child(
    *,
    root: Path,
    session_id: str,
    sequence: int,
    phase: str,
    pair: int | None,
    candidate: Candidate,
    config_path: Path,
    inventory: ManifestInventory,
    adaptive_tiles: tuple[int, ...],
    overlap: int,
    timeout_seconds: float,
) -> PipelineOutcome:
    e2e = e2e_module()
    base = e2e.load_config(config_path)
    e2e.validate_production_semantics(base)
    manifest, input_snapshot, routes, coverage = e2e.load_representative_manifest(
        inventory.path, inventory.input_root, inventory.threshold
    )
    if len(input_snapshot) != 30 or routes != {"normal": 9, "sharper": 21}:
        raise ResearchError("Final30 requires the complete 9-normal/21-sharper manifest")
    if manifest.get("pages") is None or coverage.get("exact_threshold", 0) < 2:
        raise ResearchError("Final30 manifest lacks the required threshold coverage")
    models = e2e.resolve_real_hat_models(base)
    configuration = e2e.BenchmarkConfiguration(tuple(adaptive_tiles), overlap)
    label = safe_name(f"{sequence:03d}-{phase}-{candidate.identifier}-adaptive")
    attempt = root / "runs" / label
    attempt.mkdir(parents=True, exist_ok=False)
    child_output = attempt / "output"
    child_metrics = attempt / "metrics"
    child_cache = attempt / "cache"
    e2e.validate_isolated_roots(
        inventory.input_root,
        child_output,
        child_metrics,
        child_cache,
        require_fresh=True,
    )
    child_config = e2e.render_child_config(
        base,
        input_root=inventory.input_root,
        output_root=child_output,
        configuration=configuration,
    )
    config_file = attempt / "config.toml"
    config_file.write_text(child_config, encoding="utf-8", newline="\n")
    fingerprint = canonical_hash(
        {
            "research_schema": RESEARCH_SCHEMA,
            "candidate": candidate.identifier,
            "backend_environment": actual_backend_environment(child_environment(candidate)),
            "config": sha256_file(config_file),
            "manifest": inventory.sha256,
            "models": models,
            "configuration": configuration.record(),
            "input_snapshot": input_snapshot,
            "sequence": sequence,
            "phase": phase,
            "pair": pair,
        }
    )
    spec = {
        "schema_version": e2e.SCHEMA_VERSION,
        "kind": e2e.CHILD_SPEC_KIND,
        "fingerprint": fingerprint,
        "attempt_root": str(attempt.resolve()),
        "role": phase,
        "index": sequence,
        "parent_session_id": session_id,
        "pair_id": f"{phase}-{sequence}",
        "configuration": configuration.record(),
        "input_root": str(inventory.input_root),
        "input_snapshot": input_snapshot,
        "models": models,
        "output_root": str(child_output.resolve()),
        "metrics_root": str(child_metrics.resolve()),
        "cache_root": str(child_cache.resolve()),
        "config_path": str(config_file.resolve()),
        "config_sha256": sha256_file(config_file),
        "result_path": str((attempt / "result.json").resolve()),
    }
    spec_path = attempt / "spec.json"
    e2e.write_json(spec_path, spec)
    process = e2e.run_child_process(
        spec_path,
        attempt / "child.log",
        timeout_seconds,
        backend_environment=candidate.backend_environment,
    )
    result = e2e._valid_complete_result(
        attempt,
        fingerprint=fingerprint,
        expected_input_root=inventory.input_root,
        expected_input=input_snapshot,
        expected_models=models,
        expected_configuration=configuration,
        expected_role=phase,
        expected_index=sequence,
        expected_cache_root=child_cache,
    )
    if process.get("returncode") != 0 or result is None:
        raise ResearchError(f"Final30 child left incomplete E2E evidence: {attempt}")
    expected_backend = actual_backend_environment(child_environment(candidate))
    if process.get("backend_environment") != expected_backend:
        raise ResearchError("Final30 child launcher did not retain the requested backend")
    runtime = result.get("runtime")
    actual_backend = (
        runtime.get("relevant_environment", {}).get(BACKEND_ENVIRONMENT_KEY)
        if isinstance(runtime, dict)
        else None
    )
    if actual_backend != expected_backend[BACKEND_ENVIRONMENT_KEY]:
        raise ResearchError(
            "Final30 child did not attest the expected backend environment: "
            f"{actual_backend!r} != {expected_backend[BACKEND_ENVIRONMENT_KEY]!r}"
        )
    result_path = attempt / "result.json"
    record = {
        "sequence": sequence,
        "phase": phase,
        "pair": pair,
        "candidate": candidate.identifier,
        "adaptive_tiles": list(adaptive_tiles),
        "overlap": overlap,
        "backend_environment": expected_backend,
        "attempt": str(attempt.relative_to(root)),
        "result": {
            "path": str(result_path.relative_to(root)),
            "sha256": sha256_file(result_path),
            "bytes": result_path.stat().st_size,
        },
        "process": process,
    }
    write_json(attempt / "research-run.json", record)
    return PipelineOutcome(record=record, payload=result)


def pipeline_wall(payload: Mapping[str, Any]) -> float:
    summary = payload.get("pipeline_summary")
    value = summary.get("wall_seconds") if isinstance(summary, dict) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ResearchError("Final30 child has no positive complete-pipeline wall time")
    return float(value)


def pipeline_jxl_hashes(payload: Mapping[str, Any]) -> dict[str, str]:
    outputs = payload.get("jxl_outputs")
    if not isinstance(outputs, list):
        raise ResearchError("Final30 child has no JXL output inventory")
    result: dict[str, str] = {}
    for item in outputs:
        if not isinstance(item, dict):
            raise ResearchError("Final30 JXL output record is malformed")
        path = item.get("path")
        digest = item.get("sha256")
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or path in result
        ):
            raise ResearchError("Final30 JXL output inventory is not canonical")
        result[path] = digest
    if len(result) != 30:
        raise ResearchError("Final30 JXL output inventory does not contain 30 pages")
    return result


def all_pipeline_jxl_deterministic(outcomes: Sequence[PipelineOutcome]) -> bool:
    inventories = [pipeline_jxl_hashes(outcome.payload) for outcome in outcomes]
    return bool(inventories) and all(item == inventories[0] for item in inventories[1:])


def paired_pipeline_performance(
    pairs: Sequence[tuple[PipelineOutcome, PipelineOutcome]]
) -> dict[str, Any]:
    if not pairs:
        raise ResearchError("No paired final30 measurements were recorded")
    baseline = [pipeline_wall(item[0].payload) for item in pairs]
    candidate = [pipeline_wall(item[1].payload) for item in pairs]
    ratios = [other / reference for reference, other in zip(baseline, candidate)]
    jxl_equal = [
        pipeline_jxl_hashes(reference.payload) == pipeline_jxl_hashes(other.payload)
        for reference, other in pairs
    ]
    baseline_mean = statistics.fmean(baseline)
    candidate_mean = statistics.fmean(candidate)
    return {
        "pair_count": len(pairs),
        "baseline_wall_seconds": metric(baseline),
        "candidate_wall_seconds": metric(candidate),
        "candidate_to_baseline_ratio": metric(ratios),
        "wall_reduction_percent": percent_reduction(baseline_mean, candidate_mean),
        "jxl_byte_equal_by_pair": jxl_equal,
        "jxl_byte_equal_all_pairs": all(jxl_equal),
    }


def run_final30(args: argparse.Namespace) -> int:
    config, normal_model, sharper_model = resolve_research_paths(args)
    indexes = tuple(range(1, 31))
    if tuple(args.page_indexes) != indexes:
        raise ResearchError("Final30 always requires the complete fixed 30-page manifest")
    inventory = load_manifest(args.manifest, indexes, args.threshold)
    candidates = candidate_definitions(args.candidates)
    tiles = tuple(args.adaptive_tiles)
    if tuple(sorted(set(tiles))) != tiles or args.overlap >= min(tiles):
        raise ResearchError("Final30 adaptive tile/overlap configuration is invalid")
    schedule: list[dict[str, Any]] = []
    sequence = 1
    for candidate in candidates:
        schedule.append(
            {
                "sequence": sequence,
                "phase": "warmup",
                "pair": None,
                "candidate": candidate.identifier,
            }
        )
        sequence += 1
    for item in interleaved_pair_schedule(candidates, args.pairs):
        schedule.append(
            {
                "sequence": sequence,
                "phase": "repeat",
                "pair": item["pair"],
                "candidate": item["candidate"],
            }
        )
        sequence += 1
    session = initialize_session(
        command="final30",
        output_root=args.output_root,
        config_path=config,
        inventory=inventory,
        normal_model=normal_model,
        sharper_model=sharper_model,
        candidates=candidates,
        schedule=schedule,
    )
    if args.dry_run:
        return dry_run_record(session=session, plan={"schedule": schedule})
    try:
        require_idle_production()
        session_id = uuid.uuid4().hex
        outcomes: list[PipelineOutcome] = []
        for item in schedule:
            candidate = next(
                value for value in candidates if value.identifier == item["candidate"]
            )
            outcome = run_pipeline_child(
                root=session["root"],
                session_id=session_id,
                sequence=int(item["sequence"]),
                phase=str(item["phase"]),
                pair=item["pair"],
                candidate=candidate,
                config_path=config,
                inventory=inventory,
                adaptive_tiles=tiles,
                overlap=args.overlap,
                timeout_seconds=args.timeout_seconds,
            )
            outcomes.append(outcome)
            update_session_runs(session, [outcome.record for outcome in outcomes])
        by_pair: dict[int, dict[str, PipelineOutcome]] = defaultdict(dict)
        for outcome in outcomes:
            if outcome.record["phase"] != "repeat":
                continue
            pair = outcome.record.get("pair")
            if not isinstance(pair, int):
                raise ResearchError("Final30 repeat has no pairing key")
            by_pair[pair][str(outcome.record["candidate"])] = outcome
        comparable_pairs: list[tuple[PipelineOutcome, PipelineOutcome]] = []
        if set(candidate.identifier for candidate in candidates) == set(BACKEND_CANDIDATES):
            for pair in range(1, args.pairs + 1):
                matching = by_pair.get(pair, {})
                if set(matching) != set(BACKEND_CANDIDATES):
                    raise ResearchError(f"Final30 pair {pair} is incomplete")
                comparable_pairs.append((matching["default"], matching["hipblaslt"]))
        paired = paired_pipeline_performance(comparable_pairs) if comparable_pairs else None
        baseline_outcomes = [
            outcome for outcome in outcomes if outcome.record["candidate"] == "default"
        ]
        candidate_outcomes = [
            outcome
            for outcome in outcomes
            if outcome.record["candidate"] == "hipblaslt"
        ]
        candidate_eligible = (
            paired is not None
            and paired["wall_reduction_percent"] is not None
            and float(paired["wall_reduction_percent"]) >= 3.0
            and bool(paired["jxl_byte_equal_all_pairs"])
            and all_pipeline_jxl_deterministic(baseline_outcomes)
            and all_pipeline_jxl_deterministic(candidate_outcomes)
        )
        finish_session(
            session,
            status="complete",
            final30={
                "paired_performance": paired,
                "determinism": {
                    "baseline_jxl_bytes": all_pipeline_jxl_deterministic(baseline_outcomes)
                    if baseline_outcomes
                    else None,
                    "candidate_jxl_bytes": all_pipeline_jxl_deterministic(candidate_outcomes)
                    if candidate_outcomes
                    else None,
                },
                "decision": {
                    "minimum_3_percent_wall_reduction": candidate_eligible,
                    "note": (
                        "Final30 validates complete pipeline/JXL byte stability. "
                        "The canary12 image-quality gate remains mandatory for any "
                        "numeric backend change."
                    ),
                },
            },
        )
        return 0
    except BaseException as exc:
        finish_session(
            session,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise


def tool_probe(command: Sequence[str], timeout_seconds: float = 15.0) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"command": list(command), "available": False}
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "command": [executable, *command[1:]],
            "available": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "command": [executable, *command[1:]],
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[:8000],
        "stderr": completed.stderr.strip()[:8000],
    }


def run_capability(args: argparse.Namespace) -> int:
    config, normal_model, sharper_model = resolve_research_paths(args)
    inventory = load_manifest(args.manifest, args.page_indexes, args.threshold)
    candidates = candidate_definitions(args.candidates)
    schedule = [
        {"probe": "rocprofv3 --version"},
        {"probe": "rocprofv3-avail"},
    ]
    session = initialize_session(
        command="capability",
        output_root=args.output_root,
        config_path=config,
        inventory=inventory,
        normal_model=normal_model,
        sharper_model=sharper_model,
        candidates=candidates,
        schedule=schedule,
    )
    try:
        release = platform.release().lower()
        wsl = bool(os.environ.get("WSL_DISTRO_NAME") or "microsoft" in release or "wsl" in release)
        probes = {
            "rocprofv3_version": tool_probe(["rocprofv3", "--version"]),
            "rocprofv3_avail": tool_probe(["rocprofv3-avail"]),
        }
        finish_session(
            session,
            status="complete",
            capability={
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "is_wsl": wsl,
                },
                "probes": probes,
                "gate": {
                    "gpu_trace_or_pmc_validated": False,
                    "note": (
                        "This command does not attach a profiler or launch a GPU kernel. "
                        "A profiler-led optimization remains blocked until an isolated "
                        "trace/PMC fidelity run is explicitly performed and validated."
                    ),
                },
            },
        )
        return 0
    except BaseException as exc:
        finish_session(
            session,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise


def run_summarize(args: argparse.Namespace) -> int:
    input_roots = [path.expanduser().resolve() for path in args.input]
    if len(input_roots) != len(set(input_roots)):
        raise ResearchError("Summary inputs must not contain duplicates")
    records = []
    for root in input_roots:
        summary_path = root / "summary.json"
        if root.is_symlink() or not root.is_dir() or summary_path.is_symlink():
            raise ResearchError(f"Unsafe research summary input: {root}")
        payload = read_json(summary_path)
        if payload.get("schema_version") != RESEARCH_SCHEMA or payload.get("kind") != RESEARCH_KIND:
            raise ResearchError(f"Unsupported research summary: {summary_path}")
        records.append(
            {
                "root": str(root),
                "summary": {
                    "path": "summary.json",
                    "sha256": sha256_file(summary_path),
                    "bytes": summary_path.stat().st_size,
                },
                "command": payload.get("command"),
                "status": payload.get("status"),
                "finished_at": payload.get("finished_at"),
                "decision": payload.get("canary12", {}).get("decision")
                if isinstance(payload.get("canary12"), dict)
                else payload.get("final30", {}).get("decision")
                if isinstance(payload.get("final30"), dict)
                else payload.get("micro", {}).get("decision")
                if isinstance(payload.get("micro"), dict)
                else None,
            }
        )
    root = prepare_output_root(args.output_root, inventory=None, config_path=None)
    environment = session_environment(
        command="summarize",
        config_path=None,
        inventory=None,
        normal_model=None,
        sharper_model=None,
    )
    write_json(root / "environment.json", environment)
    candidate_payload = {
        "schema_version": RESEARCH_SCHEMA,
        "kind": "real_hat_runtime_summary_inputs",
        "created_at": utc_now(),
        "inputs": records,
    }
    write_json(root / "candidate.json", candidate_payload)
    write_jsonl(root / "pages.jsonl", [])
    summary = {
        "schema_version": RESEARCH_SCHEMA,
        "kind": RESEARCH_KIND,
        "status": "complete",
        "command": "summarize",
        "created_at": utc_now(),
        "finished_at": utc_now(),
        "output_root": str(root),
        "environment": {"path": "environment.json", "sha256": sha256_file(root / "environment.json")},
        "candidates": {"path": "candidate.json", "sha256": sha256_file(root / "candidate.json")},
        "pages": {"path": "pages.jsonl", "sha256": sha256_file(root / "pages.jsonl"), "count": 0},
        "research_runs": records,
    }
    write_json(root / "summary.json", summary)
    print(json.dumps({"output_root": str(root), "runs": len(records)}, ensure_ascii=False))
    return 0


def add_common_execution_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_indexes: Sequence[int],
    default_candidates: Sequence[str] = BACKEND_CANDIDATES,
) -> None:
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--normal-model", type=Path, default=DEFAULT_NORMAL_MODEL)
    parser.add_argument("--sharper-model", type=Path, default=DEFAULT_SHARPER_MODEL)
    parser.add_argument("--threshold", type=positive_int, default=1000)
    parser.add_argument(
        "--page-indexes", type=positive_int, nargs="+", default=list(default_indexes)
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        choices=BACKEND_CANDIDATES,
        nargs="+",
        default=list(default_candidates),
    )
    parser.add_argument("--timeout-seconds", type=positive_float, default=7200.0)
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated Real-HAT backend research over copied representative pages. "
            "GPU commands refuse to run while the production worker/watchdog is active."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    capability = commands.add_parser(
        "capability", description="Record non-invasive ROCm profiler capability metadata."
    )
    add_common_execution_arguments(capability, default_indexes=DEFAULT_MICRO_INDEXES)
    capability.set_defaults(handler=run_capability)

    micro = commands.add_parser(
        "micro", description="Run four fixed route/tile hipBLASLt microbenchmark cells."
    )
    add_common_execution_arguments(micro, default_indexes=DEFAULT_MICRO_INDEXES)
    micro.add_argument("--tiles", type=positive_int, nargs="+", default=list(DEFAULT_TILES))
    micro.add_argument("--overlap", type=nonnegative_int, default=DEFAULT_OVERLAP)
    micro.add_argument("--warmups", type=positive_int, default=2)
    micro.add_argument("--rounds", type=positive_int, default=5)
    micro.add_argument("--warmup-crop", type=positive_int, default=320)
    micro.set_defaults(handler=run_micro)

    cold = commands.add_parser(
        "cold",
        description="Attribute model load and first-versus-steady work in fresh Python children.",
    )
    add_common_execution_arguments(
        cold,
        default_indexes=DEFAULT_COLD_INDEXES,
        default_candidates=("default",),
    )
    cold.add_argument("--tiles", type=positive_int, nargs="+", default=list(DEFAULT_TILES))
    cold.add_argument("--overlap", type=nonnegative_int, default=DEFAULT_OVERLAP)
    cold.add_argument("--warmups", type=positive_int, default=1)
    cold.set_defaults(handler=run_cold)

    canary = commands.add_parser(
        "canary12", description="Run AB/BA 12-page backend screening with PNG quality checks."
    )
    add_common_execution_arguments(canary, default_indexes=DEFAULT_CANARY_INDEXES)
    canary.add_argument(
        "--adaptive-tiles", type=positive_int, nargs="+", default=list(DEFAULT_TILES)
    )
    canary.add_argument("--overlap", type=nonnegative_int, default=DEFAULT_OVERLAP)
    canary.add_argument("--warmups", type=positive_int, default=2)
    canary.add_argument("--pairs", type=positive_int, default=3)
    canary.add_argument("--warmup-crop", type=positive_int, default=320)
    canary.set_defaults(handler=run_canary12)

    final30 = commands.add_parser(
        "final30", description="Run the complete mirror/JXL 30-page backend gate."
    )
    add_common_execution_arguments(final30, default_indexes=tuple(range(1, 31)))
    final30.add_argument(
        "--adaptive-tiles", type=positive_int, nargs="+", default=list(DEFAULT_TILES)
    )
    final30.add_argument("--overlap", type=nonnegative_int, default=DEFAULT_OVERLAP)
    final30.add_argument("--pairs", type=positive_int, default=3)
    final30.set_defaults(handler=run_final30)

    summarize = commands.add_parser(
        "summarize", description="Create a compact index of completed research-v1 runs."
    )
    summarize.add_argument("--input", type=Path, nargs="+", required=True)
    summarize.add_argument("--output-root", type=Path, required=True)
    summarize.set_defaults(handler=run_summarize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("Interrupted; owned research artifacts were retained.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
