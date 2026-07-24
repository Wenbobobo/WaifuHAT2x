from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import time
import unicodedata
import uuid

import numpy as np
from PIL import Image, ImageOps

from .config import AppConfig
from .engine import UpscaleEngine
from .images import (
    IMAGE_EXTENSIONS,
    ResolutionPlan,
    is_grayscale,
    output_path_for,
    pil_to_tensor,
    plan_resolution,
    resize_linear_light,
)
from .jxl import JxlEncoder, JxlStats
from .models import ModelChoice, available_scales, choose_model
from .state import RunLock, SourceFingerprint, StateStore, file_sha256
from .telemetry import PageTelemetry, RunTelemetry


COPYABLE_METADATA_EXTENSIONS = {".xml", ".json", ".txt", ".opf", ".nfo", ".yaml", ".yml"}
PIPELINE_SCHEMA_VERSION = 7
COPYABLE_METADATA_NAMES = {".nomedia"}
INTERNAL_NAMES = {
    ".waifuhat2x-state.json",
    ".waifuhat2x-worklist.jsonl",
    ".waifuhat2x.lock",
}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
TRANSCODE_MODEL_SHA256 = "no-model:transcode-only:v1"


@dataclass
class RunSummary:
    processed: int = 0
    skipped: int = 0
    copied: int = 0
    ignored: int = 0
    jxl_skipped: int = 0
    failed: int = 0
    sr_pages: int = 0
    transcoded_pages: int = 0
    replaced_sources: int = 0
    existing_jxl_adopted: int = 0
    existing_jxl_replaced: int = 0
    external_jxl_recoveries: int = 0
    deferred: int = 0
    target_unmet: int = 0
    inference_seconds: float = 0.0
    postprocess_seconds: float = 0.0
    encoding_seconds: float = 0.0
    output_bytes: int = 0
    wall_seconds: float = 0.0
    metrics_directory: str | None = None
    metrics_write_errors: int = 0


@dataclass(frozen=True)
class PendingEncode:
    future: Future[JxlStats]
    source: Path
    destination: Path
    key: str
    fingerprint: SourceFingerprint
    plan: ResolutionPlan
    index: int
    total: int
    replace_source: bool
    telemetry_page: PageTelemetry
    submitted_ns: int


@dataclass(frozen=True)
class ExistingOutputSnapshot:
    size: int
    sha256: str


@dataclass(frozen=True)
class RecoveryOutcome:
    blocked_keys: frozenset[str]
    external_jxl_keys: frozenset[str]


@dataclass(frozen=True)
class WorkItemSnapshot:
    size: int
    mtime_ns: int
    ctime_ns: int
    device: int
    inode: int


@dataclass(frozen=True)
class DiscoveryResult:
    images: tuple[Path, ...]
    image_snapshots: dict[Path, WorkItemSnapshot]
    jxl_by_key: dict[str, Path]
    metadata: tuple[Path, ...]
    ignored: int


def _pipeline_signature(config: AppConfig) -> str:
    payload = {
        "schema": PIPELINE_SCHEMA_VERSION,
        "processing": asdict(config.processing),
        "output": asdict(config.output),
        "jxl": asdict(config.jxl),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path, cache: dict[Path, str]) -> str:
    resolved = path.resolve()
    if resolved not in cache:
        cache[resolved] = file_sha256(resolved)
    return cache[resolved]


def _flatten_alpha(image: Image.Image) -> Image.Image:
    if "A" not in image.getbands() and "transparency" not in image.info:
        return image
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def _has_meaningful_transparency(image: Image.Image) -> bool:
    if "A" in image.getbands():
        minimum, _maximum = image.getchannel("A").getextrema()
        return int(minimum) < 255
    if "transparency" not in image.info:
        return False
    minimum, _maximum = image.convert("RGBA").getchannel("A").getextrema()
    return int(minimum) < 255


def _declared_bit_depth(opened: Image.Image, source_bytes: bytes) -> int:
    if (
        source_bytes.startswith(PNG_SIGNATURE)
        and len(source_bytes) >= 25
        and source_bytes[12:16] == b"IHDR"
    ):
        return source_bytes[24]
    if opened.format == "TIFF":
        bits = getattr(opened, "tag_v2", {}).get(258, 8)
        if isinstance(bits, (tuple, list)):
            return max(map(int, bits), default=8)
        return int(bits)
    raw_modes = [
        str(tile[3])
        for tile in getattr(opened, "tile", [])
        if len(tile) > 3 and tile[3] is not None
    ]
    if any(";16" in mode for mode in raw_modes):
        return 16
    if opened.mode in {"I", "F"} or ";16" in opened.mode:
        return 32 if opened.mode in {"I", "F"} else 16
    return 8


def _save_atomic(image: Image.Image, destination: Path, config: AppConfig) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    try:
        if config.output.format.lower() == "webp":
            image.save(
                temporary,
                format="WEBP",
                lossless=config.output.webp_lossless,
                quality=100,
                method=config.output.webp_method,
                exact=True,
            )
        else:
            image.save(temporary, format="PNG", compress_level=3, optimize=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _is_copyable_metadata(path: Path) -> bool:
    return path.name.lower() in COPYABLE_METADATA_NAMES or path.suffix.lower() in COPYABLE_METADATA_EXTENSIONS


def _comparison_parts(path: Path) -> tuple[str, ...]:
    parts = path.resolve().parts
    if len(parts) >= 3 and parts[0] == "/" and parts[1].casefold() == "mnt":
        return tuple(part.casefold() for part in parts)
    return parts


def _paths_overlap(first: Path, second: Path) -> bool:
    left = _comparison_parts(first)
    right = _comparison_parts(second)
    return left == right or left[: len(right)] == right or right[: len(left)] == left


def _copy_metadata(source: Path, input_root: Path, output_root: Path) -> None:
    destination = output_root / source.relative_to(input_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    same = (
        destination.is_file()
        and source.stat().st_size == destination.stat().st_size
        and _sha256(source, {}) == _sha256(destination, {})
    )
    if same:
        return
    temporary = destination.with_name(destination.name + ".part")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _preflight_destinations(
    images: list[Path], metadata: list[Path], input_root: Path, output_root: Path, extension: str
) -> None:
    destinations: dict[str, Path] = {}
    pairs = [(source, output_path_for(source, input_root, output_root, extension)) for source in images]
    pairs.extend((source, output_root / source.relative_to(input_root)) for source in metadata)
    for source, destination in pairs:
        key = _normalized_path_key(destination)
        if previous := destinations.get(key):
            raise ValueError(
                f"Two source files map to the same output: {previous} and {source} -> {destination}"
            )
        destinations[key] = source


def _safe_relative(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe transaction path in state: {value}")
    candidate = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"Transaction path traverses a symbolic link: {value}")
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve()
    if (
        not _paths_overlap(root_resolved, candidate_resolved)
        or _comparison_parts(root_resolved) == _comparison_parts(candidate_resolved)
    ):
        raise RuntimeError(f"Transaction path escapes root: {value}")
    return candidate


def _safe_transaction_temporary(
    root: Path,
    value: object,
    source: Path,
    destination: Path,
    key: str,
) -> Path:
    if not isinstance(value, str):
        raise RuntimeError(f"Replace transaction has no temporary path: {key}")
    temporary = _safe_relative(root, value)
    if _normalized_path_key(temporary) == _normalized_path_key(source):
        raise RuntimeError(f"Replace transaction reuses its source as a temporary: {key}")
    if _normalized_path_key(temporary) == _normalized_path_key(destination):
        raise RuntimeError(f"Replace transaction reuses its destination as a temporary: {key}")
    if _normalized_path_key(temporary.parent) != _normalized_path_key(destination.parent):
        raise RuntimeError(
            f"Replace transaction temporary is not beside its destination: {key}"
        )

    prefix = f".{destination.name}."
    suffix = ".part"
    name = temporary.name
    token = name[len(prefix) : -len(suffix)] if name.startswith(prefix) and name.endswith(suffix) else ""
    if not token:
        raise RuntimeError(f"Replace transaction has an unsafe temporary filename: {key}")
    return temporary


def _safe_transaction_destination(
    root: Path,
    value: object,
    source: Path,
    key: str,
) -> Path:
    if not isinstance(value, str):
        raise RuntimeError(f"Replace transaction has no destination: {key}")
    destination = _safe_relative(root, value)
    if _normalized_path_key(destination) == _normalized_path_key(source):
        raise RuntimeError(f"Replace transaction reuses its source as a destination: {key}")
    expected = source.with_suffix(".jxl")
    if _normalized_path_key(destination) != _normalized_path_key(expected):
        raise RuntimeError(
            f"Replace transaction destination is not the source's sibling JXL: {key}"
        )
    return destination


def _record_source_matches(source: Path, record: dict[str, object]) -> bool:
    if not source.is_file() or source.is_symlink():
        return False
    return source.stat().st_size == record.get("size") and file_sha256(source) == record.get("source_sha256")


def _is_lowercase_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_previous_output_hash(
    record: dict[str, object], key: str, phase_name: str
) -> str | None:
    previous_hash = record.get("previous_output_sha256")
    if previous_hash is not None and not _is_lowercase_sha256(previous_hash):
        raise RuntimeError(f"{phase_name} transaction has an invalid previous output hash: {key}")
    return previous_hash


def _validated_recovery_output_details(
    record: dict[str, object],
    key: str,
    phase_name: str,
    *,
    allow_missing_encode_mode: bool = False,
) -> tuple[str, int, int]:
    encode_mode = record.get("encode_mode")
    if encode_mode is None and allow_missing_encode_mode:
        encode_mode = "pixels"
    if encode_mode not in {"pixels", "jpeg_reconstruction"}:
        raise RuntimeError(
            f"{phase_name} transaction has an unknown encode mode {encode_mode!r}: {key}"
        )
    width = record.get("output_width")
    height = record.get("output_height")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width < 1
        or not isinstance(height, int)
        or isinstance(height, bool)
        or height < 1
    ):
        raise RuntimeError(f"{phase_name} transaction has invalid dimensions: {key}")
    return encode_mode, width, height


def _validated_encoded_candidate(record: dict[str, object], key: str) -> tuple[str, int]:
    candidate_hash = record.get("candidate_sha256")
    if record.get("verified") is not True or candidate_hash is None:
        raise RuntimeError(f"Encoded transaction lacks a verified candidate hash: {key}")
    if not _is_lowercase_sha256(candidate_hash):
        raise RuntimeError(f"Encoded transaction has an invalid candidate hash: {key}")
    candidate_size = record.get("candidate_size")
    if (
        not isinstance(candidate_size, int)
        or isinstance(candidate_size, bool)
        or candidate_size < 1
    ):
        raise RuntimeError(f"Encoded transaction has an invalid candidate size: {key}")
    return candidate_hash, candidate_size


def _fsync_parent(path: Path) -> None:
    try:
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _snapshot_for(source: Path) -> WorkItemSnapshot:
    return _snapshot_from_stat(source.stat())


def _snapshot_from_stat(stat: os.stat_result) -> WorkItemSnapshot:
    return WorkItemSnapshot(
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        ctime_ns=stat.st_ctime_ns,
        device=stat.st_dev,
        inode=stat.st_ino,
    )


def _normalized_path_key(path: Path) -> str:
    return unicodedata.normalize("NFC", path.as_posix()).casefold()


def _discover_files(source_root: Path, include_metadata: bool) -> DiscoveryResult:
    images: list[Path] = []
    image_snapshots: dict[Path, WorkItemSnapshot] = {}
    jxl_by_key: dict[str, Path] = {}
    metadata: list[Path] = []
    ignored = 0
    pending_directories = [source_root]

    while pending_directories:
        directory = pending_directories.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: unicodedata.normalize("NFC", entry.name).casefold())
        except FileNotFoundError:
            if directory == source_root:
                raise RuntimeError(f"Input directory disappeared during scan: {source_root}")
            continue
        child_directories: list[Path] = []
        for entry in entries:
            path = Path(entry.path)
            lowered = entry.name.lower()
            suffix = Path(entry.name).suffix.lower()
            if lowered in INTERNAL_NAMES:
                continue
            if entry.is_symlink():
                if suffix in IMAGE_EXTENSIONS or suffix == ".jxl":
                    raise ValueError(f"Symbolic-link image inputs are not supported: {path}")
                ignored += 1
                continue
            if entry.is_dir(follow_symlinks=False):
                child_directories.append(path)
                continue
            if suffix in IMAGE_EXTENSIONS and entry.is_file(follow_symlinks=False):
                images.append(path)
                image_snapshots[path] = _snapshot_from_stat(entry.stat(follow_symlinks=False))
                continue
            if suffix == ".jxl" and entry.is_file(follow_symlinks=False):
                relative = path.relative_to(source_root)
                key = _normalized_path_key(relative)
                if previous := jxl_by_key.get(key):
                    raise ValueError(f"Two JXL files have the same normalized path: {previous} and {path}")
                jxl_by_key[key] = path
                ignored += 1
                continue
            if include_metadata and _is_copyable_metadata(path) and entry.is_file(follow_symlinks=False):
                metadata.append(path)
                continue
            ignored += 1
        pending_directories.extend(reversed(child_directories))

    images.sort(key=lambda path: _normalized_path_key(path.relative_to(source_root)))
    metadata.sort(key=lambda path: _normalized_path_key(path.relative_to(source_root)))
    return DiscoveryResult(tuple(images), image_snapshots, jxl_by_key, tuple(metadata), ignored)


def _snapshot_matches(stat: os.stat_result, snapshot: WorkItemSnapshot) -> bool:
    return (
        stat.st_size == snapshot.size
        and stat.st_mtime_ns == snapshot.mtime_ns
        and stat.st_ctime_ns == snapshot.ctime_ns
        and stat.st_dev == snapshot.device
        and stat.st_ino == snapshot.inode
    )


def _root_identity_matches(root: Path, snapshot: WorkItemSnapshot) -> bool:
    try:
        stat = root.stat()
    except FileNotFoundError:
        return False
    return stat.st_dev == snapshot.device and stat.st_ino == snapshot.inode


def _write_worklist(
    path: Path,
    source_root: Path,
    snapshots: dict[str, WorkItemSnapshot],
    details: dict[str, dict[str, object]],
    pipeline_signature: str,
    root_snapshot: WorkItemSnapshot,
    target_short_edge: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    header = {
        "type": "waifuhat2x-worklist",
        "schema": 2,
        "run_id": uuid.uuid4().hex,
        "created_unix_ns": time.time_ns(),
        "source_root": str(source_root),
        "pipeline_signature": pipeline_signature,
        "root_device": root_snapshot.device,
        "root_inode": root_snapshot.inode,
        "target_short_edge": target_short_edge,
        "count": len(snapshots),
    }
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(header, ensure_ascii=False, separators=(",", ":")) + "\n")
            for key, snapshot in snapshots.items():
                row = {
                    "source": key,
                    "source_snapshot": asdict(snapshot),
                    **details[key],
                }
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_worklist_plan(path: Path, key: str, plan: ResolutionPlan) -> None:
    action = "transcode_only"
    if plan.upscale:
        action = f"hat_x{plan.native_scale}"
    row = {
        "type": "plan",
        "source": key,
        "action": action,
        "planned_output_dimensions": [plan.output_width, plan.output_height],
        "native_scale": plan.native_scale,
        "reason": plan.reason,
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def _commit_source_removal(
    state: StateStore, key: str, source: Path, destination: Path, summary: RunSummary
) -> None:
    record = state.record(key)
    if record is None:
        raise RuntimeError(f"Missing replace transaction for {key}")
    if record.get("phase") != "output_ready" or record.get("verified") is not True:
        raise RuntimeError(
            f"Refusing to remove source without a verified output-ready transaction: {key}"
        )
    if not state.manages_output_path(key, destination) or not state.owns_output(key, destination):
        raise RuntimeError(f"Refusing to remove source because final JXL changed: {destination}")
    if source.exists():
        if not _record_source_matches(source, record):
            raise RuntimeError(f"Source changed during replacement; kept original: {source}")
        source.unlink()
        _fsync_parent(source)
        summary.replaced_sources += 1
        if record.get("action") == "adopt_existing_jxl":
            summary.existing_jxl_adopted += 1
        elif record.get("replaces_existing_jxl") is True:
            summary.existing_jxl_replaced += 1
    # A completed in-place replacement no longer needs an incremental record:
    # the source is gone and .jxl is outside the input allowlist.  Keeping only
    # in-flight transactions prevents the JSON journal from growing and being
    # rewritten for every page in a large library.
    state.discard(key)
    state.save()


def _recorded_source_root_matches(record: dict[str, object], root: Path) -> bool:
    recorded_root = record.get("source_root")
    try:
        return (
            isinstance(recorded_root, str)
            and Path(recorded_root).is_absolute()
            and _comparison_parts(Path(recorded_root)) == _comparison_parts(root)
        )
    except (OSError, ValueError):
        return False


def _recovery_context_issue(
    record: dict[str, object],
    source: Path,
    root: Path,
    pipeline_signature: str,
    model_hash_resolver: Callable[[Path], str],
    action: object,
) -> str | None:
    if not _recorded_source_root_matches(record, root):
        return "source root changed"
    if record.get("pipeline_signature") != pipeline_signature:
        return "pipeline signature changed"
    if action not in {"legacy_replace", "sr", "transcode_only"}:
        return None
    if not _record_source_matches(source, record):
        # The phase-specific recovery path handles a changed source without
        # selecting a model from content that no longer matches the journal.
        return None
    try:
        current_model_hash = model_hash_resolver(source)
    except Exception as exc:
        return f"current model selection failed ({type(exc).__name__}: {exc})"
    if record.get("model_sha256") != current_model_hash:
        return "selected model changed"
    return None


def _restart_stale_replace_transaction(
    state: StateStore,
    key: str,
    record: dict[str, object],
    source: Path,
    destination: Path,
    temporary: Path | None,
    issue: str,
) -> None:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(
            f"Stale replace transaction cannot be redone because its source is unavailable: {key}"
        )

    phase = record.get("phase")
    action = record.get("action", "legacy_replace")
    if phase == "output_ready":
        # The durable destination is authoritative at this phase.  A leftover
        # sibling part can only be an interrupted transaction artifact, and it
        # must not survive when stale context forces the page to be replanned.
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if action != "adopt_existing_jxl" and state.owns_output(key, destination):
            # The candidate is already the durable output. Keep its ownership
            # record so the current run can replace it atomically, but do not
            # honor the stale authorization to remove the source.
            print(f"Recovery: {issue}; retaining source and redoing managed output: {key}")
            return
        state.discard(key)
        state.save()
        print(f"Recovery: {issue}; discarded stale authorization and retained source: {key}")
        return

    if phase == "encoded" and (temporary is None or not temporary.is_file()):
        candidate_hash = record.get("candidate_sha256")
        if destination.is_file() and file_sha256(destination) == candidate_hash:
            # A crash may have renamed the candidate before mark_output_ready.
            # Preserve the known output as managed so replanning can overwrite
            # it without treating it as an unrelated user file.
            state.mark_output_ready(key, destination)
            state.save()
            print(f"Recovery: {issue}; retaining source and redoing managed output: {key}")
            return

    if temporary is not None:
        temporary.unlink(missing_ok=True)
    state.discard(key)
    state.save()
    print(f"Recovery: {issue}; discarded stale candidate and replanning: {key}")


def _verify_output_ready_for_source_removal(
    encoder: JxlEncoder,
    destination: Path,
    record: dict[str, object],
    key: str,
) -> None:
    if record.get("verified") is not True:
        raise RuntimeError(f"Output-ready transaction lacks verified decode evidence: {key}")
    encode_mode, width, height = _validated_recovery_output_details(
        record,
        key,
        "Output-ready",
        allow_missing_encode_mode=record.get("action") == "adopt_existing_jxl",
    )
    _validated_previous_output_hash(record, key, "Output-ready")
    if encode_mode == "jpeg_reconstruction":
        encoder.verify(destination)
        return
    encoder.verify(destination, width, height)


def _verify_prepared_external_completion(
    encoder: JxlEncoder,
    destination: Path,
    record: dict[str, object],
    key: str,
) -> None:
    _encode_mode, width, height = _validated_recovery_output_details(record, key, "Prepared")

    snapshot_before = _stable_snapshot_existing_output(destination)
    previous_hash = _validated_previous_output_hash(record, key, "Prepared")
    if previous_hash is not None and snapshot_before.sha256 == previous_hash:
        raise RuntimeError(
            f"Prepared replace lost its source before a new JXL was committed: {key}"
        )
    encoder.verify(destination, width, height)
    snapshot_after = _stable_snapshot_existing_output(destination)
    if snapshot_before != snapshot_after:
        raise RuntimeError(
            f"Unrecorded final JXL changed during prepared recovery: {destination}"
        )


def _complete_prepared_external_recovery(
    state: StateStore,
    encoder: JxlEncoder,
    source: Path,
    destination: Path,
    temporary: Path | None,
    record: dict[str, object],
    key: str,
    summary: RunSummary,
) -> None:
    if source.is_symlink():
        raise RuntimeError(f"Prepared replace source became a symbolic link: {key}")
    if not destination.exists():
        raise RuntimeError(f"Prepared replace lost its source before commit: {key}")
    _verify_prepared_external_completion(encoder, destination, record, key)
    state.discard(key)
    state.save()
    summary.external_jxl_recoveries += 1
    candidate_note = (
        f"; left the recorded candidate path untouched: {temporary}"
        if temporary is not None
        else ""
    )
    print(
        "Recovery: verified an external JXL completion after the prepared "
        "source disappeared; retained the unowned final JXL and detached "
        f"the stale transaction: {key}{candidate_note}"
    )


def _recover_replace_transactions(
    state: StateStore,
    root: Path,
    encoder: JxlEncoder,
    summary: RunSummary,
    *,
    pipeline_signature: str,
    model_hash_resolver: Callable[[Path], str],
) -> RecoveryOutcome:
    blocked: set[str] = set()
    external_jxl_keys: set[str] = set()
    for key, record in list(state.data.items()):
        if not isinstance(record, dict) or record.get("phase") not in {
            "prepared",
            "encoded",
            "output_ready",
        }:
            continue
        action = record.get("action", "legacy_replace")
        if action not in {
            "legacy_replace",
            "sr",
            "transcode_only",
            "adopt_existing_jxl",
        }:
            raise RuntimeError(f"Replace transaction has an unknown action {action!r}: {key}")
        phase = record.get("phase")
        if action == "adopt_existing_jxl" and phase != "output_ready":
            raise RuntimeError(f"Adopt transaction has an invalid phase {phase!r}: {key}")
        if action != "adopt_existing_jxl" and phase not in {
            "prepared",
            "encoded",
            "output_ready",
        }:
            raise RuntimeError(f"Replace transaction has an invalid phase {phase!r}: {key}")
        source = _safe_relative(root, key)
        if Path(key).suffix.lower() not in IMAGE_EXTENSIONS or source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise RuntimeError(f"Replace transaction source is not a supported image: {key}")
        destination = _safe_transaction_destination(
            root,
            record.get("destination"),
            source,
            key,
        )
        temporary_value = record.get("temporary")
        if action == "adopt_existing_jxl":
            if temporary_value is not None:
                raise RuntimeError(f"Adopt transaction unexpectedly has a temporary: {key}")
            temporary = None
        else:
            temporary = _safe_transaction_temporary(
                root,
                temporary_value,
                source,
                destination,
                key,
            )

        phase_name = str(phase).replace("_", "-").capitalize()
        previous_hash = _validated_previous_output_hash(record, key, phase_name)
        output_details = _validated_recovery_output_details(
            record,
            key,
            phase_name,
            allow_missing_encode_mode=action == "adopt_existing_jxl",
        )
        encoded_candidate: tuple[str, int] | None = None
        if phase == "encoded":
            encoded_candidate = _validated_encoded_candidate(record, key)

        if (
            phase == "prepared"
            and not source.exists()
            and _recorded_source_root_matches(record, root)
        ):
            _complete_prepared_external_recovery(
                state,
                encoder,
                source,
                destination,
                temporary,
                record,
                key,
                summary,
            )
            blocked.add(key)
            external_jxl_keys.add(key)
            continue

        context_issue = _recovery_context_issue(
            record,
            source,
            root,
            pipeline_signature,
            model_hash_resolver,
            action,
        )
        if context_issue is not None:
            if (
                phase == "prepared"
                and not source.exists()
                and _recorded_source_root_matches(record, root)
            ):
                _complete_prepared_external_recovery(
                    state,
                    encoder,
                    source,
                    destination,
                    temporary,
                    record,
                    key,
                    summary,
                )
                blocked.add(key)
                external_jxl_keys.add(key)
                continue
            _restart_stale_replace_transaction(
                state,
                key,
                record,
                source,
                destination,
                temporary,
                context_issue,
            )
            continue

        if phase == "prepared":
            if not source.exists():
                _complete_prepared_external_recovery(
                    state,
                    encoder,
                    source,
                    destination,
                    temporary,
                    record,
                    key,
                    summary,
                )
                blocked.add(key)
                external_jxl_keys.add(key)
                continue
            if destination.is_file():
                current_hash = file_sha256(destination)
                if previous_hash is None or current_hash != previous_hash:
                    raise RuntimeError(
                        "Prepared transaction found an unrecorded final JXL; retaining both files "
                        f"for manual inspection: {destination}"
                    )
            if not _record_source_matches(source, record):
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
                state.discard(key)
                state.save()
                print(f"Recovery: source changed; discarded prepared candidate and replanning: {key}")
                continue
            if temporary is not None and temporary.exists():
                temporary.unlink()
            continue

        if record.get("phase") == "encoded":
            assert encoded_candidate is not None
            assert output_details is not None
            candidate_hash, candidate_size = encoded_candidate
            encode_mode, output_width, output_height = output_details
            if temporary is not None and temporary.exists() and not temporary.is_file():
                raise RuntimeError(f"Encoded JXL candidate is not a regular file: {temporary}")
            if temporary is not None and temporary.is_file():
                if not source.exists() or source.is_symlink():
                    raise RuntimeError(
                        "Encoded transaction lost its source before commit; retaining the "
                        f"verified candidate for manual inspection: {key}"
                    )
                if not _record_source_matches(source, record):
                    temporary.unlink()
                    state.discard(key)
                    state.save()
                    blocked.add(key)
                    summary.deferred += 1
                    print(f"Recovery: source changed; discarded uncommitted candidate: {key}")
                    continue
                if (
                    temporary.stat().st_size != candidate_size
                    or file_sha256(temporary) != candidate_hash
                ):
                    raise RuntimeError(f"Encoded JXL candidate changed during recovery: {temporary}")
                _assert_replace_destination_unchanged(
                    destination, previous_hash
                )
                encoder.commit(temporary, destination, candidate_hash)
            elif not destination.is_file() or file_sha256(destination) != candidate_hash:
                if source.exists() and not _record_source_matches(source, record):
                    _assert_replace_destination_unchanged(
                        destination, previous_hash
                    )
                    state.discard(key)
                    state.save()
                    print(
                        f"Recovery: source changed and verified candidate is missing; "
                        f"retained source and replanning: {key}"
                    )
                    continue
                raise RuntimeError(f"Verified JXL candidate is missing during recovery: {key}")
            if encode_mode == "jpeg_reconstruction":
                encoder.verify(destination)
            else:
                encoder.verify(destination, output_width, output_height)
            state.mark_output_ready(key, destination)
            state.save()

        record = state.record(key)
        if record is not None and record.get("phase") == "output_ready":
            if not state.owns_output(key, destination):
                raise RuntimeError(f"Managed JXL is damaged during recovery: {destination}")
            if source.exists() and not _record_source_matches(source, record):
                print(f"Recovery: source changed; retaining it for a new transaction: {key}")
                if action == "adopt_existing_jxl":
                    # Adoption never modified the existing JXL.  The stale
                    # authorization can be discarded safely; this run defers
                    # the page and the next run plans it from current facts.
                    state.discard(key)
                    state.save()
                    blocked.add(key)
                    summary.deferred += 1
                continue
            _verify_output_ready_for_source_removal(encoder, destination, record, key)
            _commit_source_removal(state, key, source, destination, summary)
            print(f"Recovery: committed replace transaction: {key}")
    return RecoveryOutcome(frozenset(blocked), frozenset(external_jxl_keys))


def _skip_blocked_recovery_source(
    key: str,
    outcome: RecoveryOutcome,
    summary: RunSummary,
) -> bool:
    if key not in outcome.blocked_keys:
        return False
    if key in outcome.external_jxl_keys:
        summary.deferred += 1
        print(
            "Recovery: source reappeared after an external JXL completion was "
            f"verified; deferring it to the next run: {key}"
        )
    return True


def _replace_destination_is_admissible(
    state: StateStore, key: str, destination: Path, output_root: Path
) -> bool:
    if not destination.exists():
        return True
    if state.owns_output(key, destination):
        return True
    record = state.record(key)
    # These phases are not trusted here; matching the journaled path merely
    # lets them reach _recover_replace_transactions, which verifies hashes and
    # dimensions before any source removal.
    if record is None or record.get("phase") not in {"prepared", "encoded", "output_ready"}:
        return False
    try:
        recorded = _safe_relative(output_root, str(record.get("destination", "")))
    except RuntimeError:
        return False
    return _comparison_parts(recorded) == _comparison_parts(destination)


def _assert_replace_destination_unchanged(
    destination: Path, expected_previous_sha256: object
) -> None:
    if expected_previous_sha256 is None:
        if destination.exists():
            raise RuntimeError(
                f"A destination appeared after replace preparation: {destination}"
            )
        return
    if not _is_lowercase_sha256(expected_previous_sha256):
        raise RuntimeError("Replace journal has an invalid previous-output hash")
    if not destination.is_file():
        raise RuntimeError(f"Destination disappeared before candidate commit: {destination}")
    if file_sha256(destination) != expected_previous_sha256:
        raise RuntimeError(f"Destination changed before candidate commit: {destination}")


def _stable_snapshot_existing_output(destination: Path) -> ExistingOutputSnapshot:
    if not destination.is_file() or destination.is_symlink():
        raise RuntimeError(f"Existing JXL is not a regular file: {destination}")
    stat_before = destination.stat()
    output_hash = file_sha256(destination)
    stat_after = destination.stat()
    if (
        stat_before.st_size != stat_after.st_size
        or stat_before.st_mtime_ns != stat_after.st_mtime_ns
        or stat_before.st_ctime_ns != stat_after.st_ctime_ns
        or stat_before.st_dev != stat_after.st_dev
        or stat_before.st_ino != stat_after.st_ino
    ):
        raise RuntimeError(f"Existing JXL changed while it was being hashed: {destination}")
    return ExistingOutputSnapshot(stat_after.st_size, output_hash)


def _mirror_destination_is_admissible(
    state: StateStore, key: str, destination: Path, overwrite: bool
) -> bool:
    if not destination.exists() or overwrite:
        return True
    return state.manages_output_path(key, destination)


def _jxl_interval(stats: JxlStats, name: str) -> tuple[int, int] | None:
    value = getattr(stats, name, None)
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(part, int) for part in value)
    ):
        return value
    return None


def _record_jxl_telemetry(
    page: PageTelemetry, stats: JxlStats, submitted_ns: int
) -> None:
    service_interval = _jxl_interval(stats, "service_interval_ns")
    page.add_interval("jxl_service", service_interval, clock="cpu_monotonic_worker")
    if service_interval is not None and service_interval[0] >= submitted_ns:
        page.add_interval(
            "jxl_queue_wait",
            (submitted_ns, service_interval[0]),
            clock="cpu_monotonic",
        )

    intervals = {
        "jxl_resize": "resize_interval_ns",
        "cjxl": "cjxl_interval_ns",
        "djxl": "djxl_interval_ns",
        "candidate_hash": "candidate_hash_interval_ns",
        "jxl_worker_commit": "commit_interval_ns",
    }
    for metric_name, attribute in intervals.items():
        page.add_interval(
            metric_name,
            _jxl_interval(stats, attribute),
            clock="cpu_monotonic_worker",
        )

    services = {
        "jxl_encode_total": getattr(stats, "seconds", 0.0),
        "jxl_resize": getattr(stats, "postprocess_seconds", 0.0),
        "cjxl": getattr(stats, "cjxl_seconds", 0.0),
        "djxl": getattr(stats, "djxl_seconds", 0.0),
        "candidate_hash": getattr(stats, "candidate_hash_seconds", 0.0),
        "jxl_worker_commit": getattr(stats, "commit_seconds", 0.0),
    }
    for metric_name, seconds in services.items():
        page.set_service_seconds(metric_name, seconds)
    page.set_detail("output_bytes", stats.bytes)


def _choose_page_model(
    config: AppConfig,
    image: Image.Image,
    grayscale: bool,
    plan: ResolutionPlan,
) -> ModelChoice:
    if not plan.upscale:
        raise ValueError("A model can only be selected for an SR page")
    return choose_model(
        config.paths.models,
        config.processing.profile,
        image.height,
        grayscale,
        plan.native_scale,
        source_short_edge=min(image.width, image.height),
        real_hat_sharper_min_short_edge=(
            config.processing.real_hat_sharper_min_short_edge
        ),
    )


def _run_pipeline(config: AppConfig, telemetry: RunTelemetry) -> RunSummary:
    source_root = config.paths.input
    replace_mode = config.output.mode == "replace"
    output_root = source_root if replace_mode else config.paths.output
    if not source_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {source_root}")
    if not replace_mode and _paths_overlap(source_root, output_root):
        raise ValueError("Mirror input and output directories must not contain one another")

    output_root.mkdir(parents=True, exist_ok=True)

    summary = RunSummary()
    started = time.perf_counter()
    signature = _pipeline_signature(config)
    model_hashes: dict[Path, str] = {}
    available_scales_by_grayscale = {
        grayscale: available_scales(
            config.paths.models,
            config.processing.profile,
            grayscale,
        )
        for grayscale in (False, True)
    }
    state = StateStore(output_root / ".waifuhat2x-state.json")
    replace_existing_jxl: set[str] = set()
    destinations: dict[str, Path] = {}
    worklist_path = output_root / ".waifuhat2x-worklist.jsonl"
    worklist_snapshots: dict[str, WorkItemSnapshot] = {}
    worklist_details: dict[str, dict[str, object]] = {}
    root_snapshot: WorkItemSnapshot | None = None
    images: list[Path] = []
    metadata: list[Path] = []
    pending: deque[PendingEncode] = deque()
    executor: ThreadPoolExecutor | None = None
    encoder: JxlEncoder | None = None
    engine: UpscaleEngine | None = None

    def get_engine() -> UpscaleEngine:
        nonlocal engine
        if engine is None:
            engine = UpscaleEngine(
                precision=config.processing.precision,
                tile=config.processing.tile,
                overlap=config.processing.overlap,
                batch_tiles=config.processing.batch_tiles,
                hat_tile=config.processing.hat_tile,
                hat_overlap=config.processing.hat_overlap,
                hat_tile_candidates=config.processing.hat_tile_candidates,
                device_assembly=config.processing.device_assembly,
                model_cache_size=config.processing.model_cache_size,
                collect_gpu_timing=telemetry.enabled,
            )
            print(f"GPU: {engine.device_name}")
        return engine

    def recovery_model_hash(source: Path) -> str:
        with Image.open(source) as opened:
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError("Animated or multi-page images are not supported")
            image = ImageOps.exif_transpose(opened)
            image.load()
        gray = is_grayscale(image, config.processing.grayscale_tolerance)
        plan = plan_resolution(
            image.width,
            image.height,
            config.processing.target_short_edge,
            config.processing.max_long_edge_for_sr,
            available_scales_by_grayscale[gray],
            config.processing.max_upscale_factor,
            config.processing.max_output_long_edge,
            config.processing.max_output_megapixels,
        )
        if not plan.upscale:
            return TRANSCODE_MODEL_SHA256
        choice = _choose_page_model(config, image, gray, plan)
        return _sha256(choice.path, model_hashes)

    def finish(item: PendingEncode) -> None:
        try:
            stats = item.future.result()
            _record_jxl_telemetry(item.telemetry_page, stats, item.submitted_ns)
            with item.telemetry_page.span("commit"):
                if item.replace_source:
                    assert encoder is not None
                    if stats.temporary is None or stats.sha256 is None:
                        raise RuntimeError("Replace encoder did not return a verified candidate")
                    state.mark_encoded(item.key, stats.temporary, stats.bytes, stats.sha256)
                    state.save()
                    record = state.record(item.key)
                    if record is None:
                        raise RuntimeError(f"Missing replace transaction for {item.key}")
                    if not _record_source_matches(item.source, record):
                        stats.temporary.unlink(missing_ok=True)
                        state.discard(item.key)
                        state.save()
                        summary.deferred += 1
                        print(
                            f"[{item.index}/{item.total}] defer source changed before JXL commit "
                            f"{item.source.relative_to(source_root)}"
                        )
                        item.telemetry_page.finish("deferred")
                        return
                    if root_snapshot is None or not _root_identity_matches(
                        source_root, root_snapshot
                    ):
                        raise RuntimeError("Input root changed before JXL commit")
                    _assert_replace_destination_unchanged(
                        item.destination, record.get("previous_output_sha256")
                    )
                    encoder.commit(stats.temporary, item.destination, stats.sha256)
                    shutil.copystat(item.source, item.destination)
                    state.mark_output_ready(item.key, item.destination)
                    state.save()
                    record = state.record(item.key)
                    if record is None:
                        raise RuntimeError(f"Missing output-ready transaction for {item.key}")
                    final_verify_started_ns = time.perf_counter_ns()
                    try:
                        _verify_output_ready_for_source_removal(
                            encoder,
                            item.destination,
                            record,
                            item.key,
                        )
                    finally:
                        final_verify_ended_ns = time.perf_counter_ns()
                        final_verify_interval = (
                            final_verify_started_ns,
                            final_verify_ended_ns,
                        )
                        final_verify_seconds = (
                            final_verify_ended_ns - final_verify_started_ns
                        ) / 1_000_000_000
                        item.telemetry_page.add_interval(
                            "jxl_final_verify",
                            final_verify_interval,
                            clock="cpu_monotonic",
                        )
                        item.telemetry_page.add_interval(
                            "jxl_service",
                            final_verify_interval,
                            clock="cpu_monotonic",
                        )
                        item.telemetry_page.add_interval(
                            "djxl",
                            final_verify_interval,
                            clock="cpu_monotonic",
                        )
                        item.telemetry_page.set_service_seconds(
                            "jxl_final_verify", final_verify_seconds
                        )
                        item.telemetry_page.set_service_seconds(
                            "djxl", stats.djxl_seconds + final_verify_seconds
                        )
                    _commit_source_removal(
                        state, item.key, item.source, item.destination, summary
                    )
                else:
                    shutil.copystat(item.source, item.destination)
                    state.update(item.key, item.fingerprint, item.destination)
                    state.save()
            summary.processed += 1
            summary.sr_pages += int(item.plan.upscale)
            summary.transcoded_pages += int(not item.plan.upscale)
            summary.target_unmet += int("remains below target" in item.plan.reason)
            summary.postprocess_seconds += stats.postprocess_seconds
            summary.encoding_seconds += stats.seconds
            summary.output_bytes += stats.bytes
            verb = "replace" if item.replace_source else "complete"
            print(
                f"[{item.index}/{item.total}] JXL {verb} {item.source.relative_to(source_root)} | "
                f"{item.plan.output_width}x{item.plan.output_height} | "
                f"resize {stats.postprocess_seconds:.2f}s | encode {stats.seconds:.2f}s | "
                f"{stats.bytes / 1024**2:.2f} MiB"
            )
            item.telemetry_page.finish("complete")
        except Exception as exc:
            summary.failed += 1
            print(
                f"[{item.index}/{item.total}] JXL ERROR {item.source.relative_to(source_root)}: "
                f"{type(exc).__name__}: {exc}"
            )
            item.telemetry_page.finish("error", error=exc)

    with RunLock(output_root / ".waifuhat2x.lock"):
        if config.output.format.lower() == "jxl":
            encoder = JxlEncoder(config.jxl)
            executor = ThreadPoolExecutor(max_workers=config.jxl.workers, thread_name_prefix="cjxl")
            print(f"JPEG XL: {encoder.version()}")
        recovery_outcome = RecoveryOutcome(frozenset(), frozenset())
        with telemetry.stage("recovery"):
            if replace_mode:
                assert encoder is not None
                recovery_outcome = _recover_replace_transactions(
                    state,
                    output_root,
                    encoder,
                    summary,
                    pipeline_signature=signature,
                    model_hash_resolver=recovery_model_hash,
                )

        with telemetry.stage("discovery"):
            discovery = _discover_files(source_root, include_metadata=not replace_mode)
        images = list(discovery.images)
        metadata = list(discovery.metadata)
        summary.ignored = discovery.ignored
        summary.jxl_skipped = len(discovery.jxl_by_key)
        with telemetry.stage("preflight"):
            _preflight_destinations(
                images, metadata, source_root, output_root, config.output.format
            )
            processable: list[Path] = []
            for source in images:
                relative = source.relative_to(source_root)
                key = relative.as_posix()
                if _skip_blocked_recovery_source(key, recovery_outcome, summary):
                    continue
                default_destination = output_path_for(
                    source, source_root, output_root, config.output.format
                )
                if replace_mode:
                    companion_key = _normalized_path_key(relative.with_suffix(".jxl"))
                    destination = discovery.jxl_by_key.get(companion_key, default_destination)
                else:
                    destination = default_destination
                destinations[key] = destination
                try:
                    action = "encode_mirror" if not replace_mode else "encode_new"
                    details: dict[str, object] = {
                        "destination": destination.relative_to(output_root).as_posix(),
                        "action": action,
                    }
                    if replace_mode and destination.exists():
                        if config.output.existing_jxl_policy == "error":
                            if not _replace_destination_is_admissible(
                                state, key, destination, output_root
                            ):
                                raise FileExistsError(
                                    "Refusing to overwrite an unmanaged JXL in replace mode: "
                                    f"{destination}"
                                )
                        else:
                            replace_existing_jxl.add(key)
                            details["action"] = "replace_existing_jxl"
                    worklist_snapshots[key] = discovery.image_snapshots[source]
                    worklist_details[key] = details
                    processable.append(source)
                except Exception as exc:
                    summary.failed += 1
                    print(f"preflight ERROR {relative}: {type(exc).__name__}: {exc}")
        images = processable
        root_snapshot = _snapshot_for(source_root)
        with telemetry.stage("worklist"):
            _write_worklist(
                worklist_path,
                source_root,
                worklist_snapshots,
                worklist_details,
                signature,
                root_snapshot,
                config.processing.target_short_edge,
            )

        print(f"Input: {source_root}")
        print(f"Output mode: {'replace in place' if replace_mode else f'mirror -> {output_root}'}")
        print(
            f"Pages: {len(images)}; existing JXL skipped: {summary.jxl_skipped}; "
            f"metadata: {len(metadata)}; ignored: {summary.ignored}"
        )

        try:
            for index, source in enumerate(images, start=1):
                relative = source.relative_to(source_root)
                key = relative.as_posix()
                page = telemetry.page(
                    key,
                    index,
                    len(images),
                    destination=destinations[key].relative_to(output_root).as_posix(),
                )
                page.set_detail("source_extension", source.suffix.lower())
                if len(pending) >= config.jxl.queue_depth:
                    with page.span("queue_backpressure"):
                        while len(pending) >= config.jxl.queue_depth:
                            finish(pending.popleft())
                try:
                    snapshot = worklist_snapshots[key]
                    try:
                        stat_before = source.stat()
                    except FileNotFoundError:
                        summary.deferred += 1
                        print(f"[{index}/{len(images)}] defer missing source {relative}")
                        page.finish("deferred")
                        continue
                    if not _snapshot_matches(stat_before, snapshot):
                        summary.deferred += 1
                        print(f"[{index}/{len(images)}] defer changed source {relative}")
                        page.finish("deferred")
                        continue
                    with page.span("read"):
                        source_bytes = source.read_bytes()
                    page.set_detail("source_bytes", len(source_bytes))
                    try:
                        stat_after = source.stat()
                    except FileNotFoundError:
                        summary.deferred += 1
                        print(f"[{index}/{len(images)}] defer source removed while reading {relative}")
                        page.finish("deferred")
                        continue
                    if (
                        not _snapshot_matches(stat_after, snapshot)
                        or stat_after.st_size != len(source_bytes)
                    ):
                        summary.deferred += 1
                        print(f"[{index}/{len(images)}] defer source changed while reading {relative}")
                        page.finish("deferred")
                        continue
                    with page.span("decode_exif"):
                        with Image.open(BytesIO(source_bytes)) as opened:
                            if getattr(opened, "n_frames", 1) != 1:
                                raise ValueError("Animated or multi-page images are not supported")
                            original_mode = opened.mode
                            declared_bit_depth = _declared_bit_depth(opened, source_bytes)
                            meaningful_metadata = sorted(str(name) for name in opened.info)
                            if getattr(opened, "text", None):
                                meaningful_metadata.append("png_text")
                            if opened.format == "TIFF" and len(
                                getattr(opened, "tag_v2", {})
                            ):
                                meaningful_metadata.append("tiff_tags")
                            image = ImageOps.exif_transpose(opened)
                            image.load()

                    with page.span("analyze_plan"):
                        has_alpha = _has_meaningful_transparency(image)
                        gray = is_grayscale(image, config.processing.grayscale_tolerance)
                        destination = destinations[key]
                        plan = plan_resolution(
                            image.width,
                            image.height,
                            config.processing.target_short_edge,
                            config.processing.max_long_edge_for_sr,
                            available_scales_by_grayscale[gray],
                            config.processing.max_upscale_factor,
                            config.processing.max_output_long_edge,
                            config.processing.max_output_megapixels,
                        )
                        _append_worklist_plan(worklist_path, key, plan)
                    page.set_detail("source_dimensions", [image.width, image.height])
                    page.set_detail("source_short_edge", min(image.width, image.height))
                    page.set_detail("grayscale", gray)
                    page.set_detail("upscale", plan.upscale)
                    page.set_detail("native_scale", plan.native_scale)
                    page.set_detail(
                        "planned_output_dimensions",
                        [plan.output_width, plan.output_height],
                    )
                    page.set_detail("plan_reason", plan.reason)

                    jpeg_reconstruction = (
                        replace_mode
                        and not plan.upscale
                        and source.suffix.lower() in JPEG_EXTENSIONS
                    )
                    if replace_mode and not jpeg_reconstruction:
                        if meaningful_metadata and not config.output.allow_metadata_loss:
                            raise ValueError(
                                "Replacing this page would discard embedded metadata "
                                f"{meaningful_metadata}; set output.allow_metadata_loss = true"
                            )
                        if has_alpha and not config.output.allow_alpha_flatten:
                            raise ValueError(
                                "Replacing this page would flatten transparency; "
                                "set output.allow_alpha_flatten = true"
                            )
                        if declared_bit_depth > 8 and not config.output.allow_bit_depth_loss:
                            raise ValueError(
                                f"Replacing {declared_bit_depth}-bit mode {original_mode} would "
                                "quantize it to 8-bit; "
                                "set output.allow_bit_depth_loss = true"
                            )

                    with page.span("hash_and_fingerprint"):
                        previous_output = (
                            _stable_snapshot_existing_output(destination)
                            if key in replace_existing_jxl
                            else None
                        )

                        choice: ModelChoice | None = None
                        if plan.upscale:
                            choice = _choose_page_model(config, image, gray, plan)
                            model_hash = _sha256(choice.path, model_hashes)
                        else:
                            model_hash = TRANSCODE_MODEL_SHA256

                        fingerprint = state.fingerprint_bytes(
                            source_bytes,
                            stat_after.st_mtime_ns,
                            source_root,
                            model_hash,
                            signature,
                        )
                    page.set_detail("model_label", choice.label if choice else "none")
                    page.set_detail(
                        "model_checkpoint", choice.path.name if choice is not None else None
                    )
                    if (
                        previous_output is None
                        and not config.output.overwrite
                        and state.matches(key, fingerprint, destination)
                    ):
                        if replace_mode:
                            with page.span("commit"):
                                _commit_source_removal(
                                    state, key, source, destination, summary
                                )
                        summary.skipped += 1
                        print(f"[{index}/{len(images)}] skip {relative}")
                        page.finish("skipped")
                        continue
                    if not replace_mode and not _mirror_destination_is_admissible(
                        state, key, destination, config.output.overwrite
                    ):
                        raise FileExistsError(
                            f"Refusing to overwrite an unmanaged mirror output: {destination}"
                        )
                    if (
                        replace_mode
                        and previous_output is None
                        and not _replace_destination_is_admissible(
                            state, key, destination, output_root
                        )
                    ):
                        raise FileExistsError(f"Unmanaged destination exists: {destination}")

                    with page.span("preprocess"):
                        image = _flatten_alpha(image)

                    if plan.upscale:
                        assert choice is not None
                        with page.span("engine_setup"):
                            selected_engine = get_engine()
                        with page.span("preprocess"):
                            tensor = pil_to_tensor(image)
                        with page.span("engine_path"):
                            array, stats = selected_engine.upscale(
                                tensor, choice.path, grayscale_output=gray
                            )
                        page.set_service_seconds(
                            "gpu_synchronized_inference", stats.seconds
                        )
                        page.set_service_seconds("model_load", stats.model_load_seconds)
                        page.add_interval(
                            "model_load",
                            stats.model_load_interval_ns,
                            clock="cpu_monotonic",
                        )
                        page.add_interval(
                            "gpu_inference",
                            stats.inference_interval_ns,
                            clock="cpu_monotonic",
                        )
                        page.set_service_seconds(
                            "engine_cpu_prepare", stats.cpu_prepare_seconds
                        )
                        available_components: list[str] = []
                        for metric_name, attribute in (
                            ("h2d", "h2d_seconds"),
                            ("forward", "forward_seconds"),
                            ("gpu_postprocess", "gpu_postprocess_seconds"),
                            ("d2h", "d2h_seconds"),
                        ):
                            value = getattr(stats, attribute, None)
                            if value is not None:
                                page.set_service_seconds(metric_name, value)
                                available_components.append(metric_name)
                        page.set_detail("gpu_timing_backend", stats.gpu_timing_backend)
                        page.set_detail("gpu_timing_error", stats.gpu_timing_error)
                        page.set_detail("gpu_timing_warning", stats.gpu_timing_warning)
                        page.set_detail(
                            "gpu_event_total_seconds", stats.gpu_event_total_seconds
                        )
                        page.set_detail(
                            "gpu_event_scale_to_wall", stats.gpu_event_scale_to_wall
                        )
                        page.set_detail(
                            "gpu_event_raw_seconds", stats.gpu_event_raw_seconds
                        )
                        page.set_detail(
                            "engine_component_timing",
                            {
                                "available": available_components,
                                "unavailable": sorted(
                                    {
                                        "h2d",
                                        "forward",
                                        "gpu_postprocess",
                                        "d2h",
                                    }
                                    - set(available_components)
                                ),
                                "note": (
                                    "Unavailable components are not inferred from the "
                                    "synchronized engine-path interval."
                                ),
                            },
                        )
                        page.set_detail("precision", stats.precision)
                        page.set_detail("peak_vram_bytes", stats.peak_vram_bytes)
                        page.set_detail(
                            "peak_reserved_vram_bytes",
                            stats.peak_reserved_vram_bytes,
                        )
                        page.set_detail("tile_count", stats.tile_count)
                        page.set_detail("tile", stats.tile)
                        page.set_detail(
                            "tile_candidates", list(stats.tile_candidates)
                        )
                        page.set_detail("tile_strategy", stats.tile_strategy)
                        page.set_detail("tile_estimator", stats.tile_estimator)
                        page.set_detail(
                            "tile_estimates",
                            [asdict(estimate) for estimate in stats.tile_estimates],
                        )
                        page.set_detail("overlap", stats.overlap)
                        page.set_detail("batch_tiles", stats.batch_tiles)
                        page.set_detail("assembly", stats.assembly)
                        page.set_detail("model_cache_hit", stats.model_cache_hit)
                        if encoder is None and (array.shape[1], array.shape[0]) != (
                            plan.output_width,
                            plan.output_height,
                        ):
                            resize_started = time.perf_counter()
                            with page.span("postprocess"):
                                if config.processing.linear_light_downscale:
                                    array = resize_linear_light(
                                        array, plan.output_width, plan.output_height
                                    )
                                else:
                                    array = np.asarray(
                                        Image.fromarray(array).resize(
                                            (plan.output_width, plan.output_height),
                                            Image.Resampling.LANCZOS,
                                        )
                                    )
                            summary.postprocess_seconds += time.perf_counter() - resize_started
                        summary.inference_seconds += stats.seconds
                        peak_gib = stats.peak_vram_bytes / 1024**3
                        tile_candidates = ",".join(
                            str(candidate) for candidate in stats.tile_candidates
                        )
                        tile_estimates = ",".join(
                            f"{estimate.tile}:{estimate.estimated_work}"
                            for estimate in stats.tile_estimates
                        )
                        print(
                            f"[{index}/{len(images)}] SR {relative} | {choice.label} "
                            f"{stats.precision} | {stats.seconds:.2f}s | {peak_gib:.2f} GiB | "
                            f"{stats.tile_count} tiles ({stats.tile}+2x{stats.overlap}, "
                            f"batch {stats.batch_tiles}, {stats.assembly}; "
                            f"{stats.tile_strategy}, candidates [{tile_candidates}], "
                            f"estimator {stats.tile_estimator or 'none'}, "
                            f"work [{tile_estimates}]) | "
                            f"load {stats.model_load_seconds:.2f}s"
                            f"{' cache' if stats.model_cache_hit else ''} | {plan.reason}"
                        )
                    else:
                        with page.span("preprocess"):
                            array = np.asarray(
                                image.convert("L" if gray else "RGB"), dtype=np.uint8
                            )
                        print(
                            f"[{index}/{len(images)}] no SR {relative} | {plan.reason} | "
                            f"keep {plan.output_width}x{plan.output_height}"
                        )

                    temporary: Path | None = None
                    if replace_mode:
                        with page.span("transaction_prepare"):
                            temporary = destination.with_name(
                                f".{destination.name}.{uuid.uuid4().hex}.part"
                            )
                            previous = state.record(key)
                            if previous_output is not None:
                                _assert_replace_destination_unchanged(
                                    destination, previous_output.sha256
                                )
                                previous_hash = previous_output.sha256
                            elif previous is not None and state.owns_output(key, destination):
                                previous_hash = previous.get("output_sha256")
                            elif (
                                previous is not None
                                and previous.get("phase") == "prepared"
                                and isinstance(previous.get("previous_output_sha256"), str)
                            ):
                                previous_hash = previous.get("previous_output_sha256")
                                _assert_replace_destination_unchanged(
                                    destination, previous_hash
                                )
                            else:
                                previous_hash = None
                            state.prepare_replace(
                                key,
                                fingerprint,
                                destination.relative_to(output_root).as_posix(),
                                temporary.relative_to(output_root).as_posix(),
                                {
                                    "action": "sr" if plan.upscale else "transcode_only",
                                    "encode_mode": (
                                        "jpeg_reconstruction"
                                        if jpeg_reconstruction
                                        else "pixels"
                                    ),
                                    "original_width": image.width,
                                    "original_height": image.height,
                                    "output_width": plan.output_width,
                                    "output_height": plan.output_height,
                                    "native_scale": plan.native_scale,
                                    "plan_reason": plan.reason,
                                    "model_label": choice.label if choice else "none",
                                    "previous_output_sha256": previous_hash,
                                    "replaces_existing_jxl": previous_output is not None,
                                },
                            )
                            state.save()

                    if encoder is not None and executor is not None:
                        submitted_ns = time.perf_counter_ns()
                        if jpeg_reconstruction:
                            future = executor.submit(
                                encoder.encode_lossless_jpeg,
                                source_bytes,
                                destination,
                                temporary=temporary,
                                finalize=not replace_mode,
                            )
                        else:
                            distance = 0.0 if replace_mode and not plan.upscale else None
                            future = executor.submit(
                                encoder.encode_resized,
                                array,
                                destination,
                                plan.output_width,
                                plan.output_height,
                                linear_light=config.processing.linear_light_downscale,
                                temporary=temporary,
                                distance=distance,
                                finalize=not replace_mode,
                            )
                        pending.append(
                            PendingEncode(
                                future,
                                source,
                                destination,
                                key,
                                fingerprint,
                                plan,
                                index,
                                len(images),
                                replace_mode,
                                page,
                                submitted_ns,
                            )
                        )
                    else:
                        with page.span("commit"):
                            result_image = Image.fromarray(
                                array, mode="L" if array.ndim == 2 else "RGB"
                            )
                            _save_atomic(result_image, destination, config)
                            shutil.copystat(source, destination)
                            state.update(key, fingerprint, destination)
                            state.save()
                        summary.processed += 1
                        summary.sr_pages += int(plan.upscale)
                        summary.transcoded_pages += int(not plan.upscale)
                        summary.target_unmet += int("remains below target" in plan.reason)
                        summary.output_bytes += destination.stat().st_size
                        page.set_detail("output_bytes", destination.stat().st_size)
                        page.finish("complete")
                except Exception as exc:
                    summary.failed += 1
                    print(f"[{index}/{len(images)}] ERROR {relative}: {type(exc).__name__}: {exc}")
                    page.finish("error", error=exc)
            while pending:
                finish(pending.popleft())
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)
            if engine is not None:
                engine.close()

        if config.output.copy_non_images and not replace_mode:
            for source in metadata:
                _copy_metadata(source, source_root, output_root)
                summary.copied += 1
        state.save()
        if summary.failed == 0 and summary.deferred == 0:
            worklist_path.unlink(missing_ok=True)
            _fsync_parent(worklist_path)
        else:
            print(f"Worklist retained after errors or deferred files: {worklist_path}")

    summary.wall_seconds = time.perf_counter() - started
    return summary


def run_pipeline(config: AppConfig, *, metrics_dir: Path | None = None) -> RunSummary:
    """Run the pipeline, optionally writing best-effort versioned metrics.

    Metrics live outside the processing roots and are deliberately excluded
    from the pipeline signature. Failure to initialize the requested metrics
    directory aborts before any output transaction starts. Later report-write
    failures are warnings: completed image transactions remain authoritative.
    """

    started_ns = time.perf_counter_ns()
    if metrics_dir is not None:
        metrics_root = metrics_dir.expanduser().resolve()
        source_root = config.paths.input.expanduser().resolve()
        output_root = (
            source_root
            if config.output.mode == "replace"
            else config.paths.output.expanduser().resolve()
        )
        if _paths_overlap(metrics_root, source_root) or _paths_overlap(
            metrics_root, output_root
        ):
            raise ValueError("Metrics directory must not overlap input or output directories")

    telemetry = RunTelemetry.create(metrics_dir, started_ns=started_ns)
    context = {
        "input_root": str(config.paths.input),
        "output_root": str(
            config.paths.input
            if config.output.mode == "replace"
            else config.paths.output
        ),
        "output_mode": config.output.mode,
        "output_format": config.output.format,
        "processing_profile": config.processing.profile,
        "pipeline_signature": _pipeline_signature(config),
    }
    try:
        summary = _run_pipeline(config, telemetry)
    except BaseException as exc:
        wall_seconds = (time.perf_counter_ns() - started_ns) / 1_000_000_000
        telemetry.finalize(
            status="fatal_error",
            wall_seconds=wall_seconds,
            summary=None,
            context=context,
            error=exc,
        )
        raise

    summary.wall_seconds = (time.perf_counter_ns() - started_ns) / 1_000_000_000
    summary.metrics_directory = telemetry.output_path
    summary.metrics_write_errors = telemetry.write_error_count
    status = (
        "completed_with_errors"
        if summary.failed or summary.deferred
        else "complete"
    )
    summary.metrics_write_errors = telemetry.finalize(
        status=status,
        wall_seconds=summary.wall_seconds,
        summary=asdict(summary),
        context=context,
    )
    return summary
