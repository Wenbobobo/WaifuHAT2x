from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping
import uuid

from PIL import Image, ImageOps

from waifuhat2x.config import AppConfig, load_config
from waifuhat2x.images import is_grayscale


SCHEMA_VERSION = 4
SUMMARY_KIND = "real_hat_pipeline_e2e_benchmark"
CHILD_SPEC_KIND = "real_hat_pipeline_e2e_child_spec"
CHILD_RESULT_KIND = "real_hat_pipeline_e2e_child_result"
COMPLETION_KIND = "real_hat_pipeline_e2e_completion"
CHILD_FLAG = "--_child-spec"
SESSION_LOCK_NAME = ".real-hat-pipeline-e2e.lock"
REAL_HAT_MODELS = (
    "Real_HAT_GAN_SRx4.pth",
    "Real_HAT_GAN_SRx4_sharper.pth",
)
OFFICIAL_REAL_HAT_SHA256 = {
    "Real_HAT_GAN_SRx4.pth": (
        "f5b1e3bbbb05147ca2beefcc715279cb647d7976cbda67d62ea7e6e20d5ffcc7"
    ),
    "Real_HAT_GAN_SRx4_sharper.pth": (
        "5800b67136006eb8cab3b4ed7c8d73b6a195bb18e6cc709b674f9aa069c00271"
    ),
}
REPRESENTATIVE_IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
ADAPTIVE_TILE_STRATEGY = "min-padded-work-v1"
FIXED_TILE_STRATEGY = "fixed"
ADAPTIVE_SELECTION_FORMULA = (
    "ceil(width/tile)*ceil(height/tile)*(tile+2*overlap)^2; "
    "minimum wins; ties use the smaller tile"
)
PRODUCTION_BASELINE_TILES = (256,)
PRODUCTION_CANDIDATE_TILES = (256, 320)
PRODUCTION_OVERLAP = 32
PRODUCTION_ROUTE_COUNTS = {"normal": 9, "sharper": 21}
PRODUCTION_PAGE_COUNT = 30
PRODUCTION_MIN_WARMUPS = 1
PRODUCTION_REPEATS = 3
PRODUCTION_MAX_CV_PERCENT = 3.0
PRODUCTION_MAX_RESERVED_VRAM_BYTES = 14 * 1024**3
PRODUCTION_MIN_WALL_REDUCTION_PERCENT = 3.0
PRODUCTION_PROCESSING_SEMANTICS = {
    "profile": "real-hat-auto",
    "target_short_edge": 1600,
    "real_hat_sharper_min_short_edge": 1000,
    "max_long_edge_for_sr": 3200,
    "max_upscale_factor": 4,
    "max_output_long_edge": 6400,
    "max_output_megapixels": 24.0,
    "precision": "bf16",
    "batch_tiles": 1,
    "device_assembly": True,
    "model_cache_size": 2,
    "grayscale_tolerance": 3,
    "linear_light_downscale": True,
}
PRODUCTION_JXL_SEMANTICS = {
    "distance": 0.5,
    "effort": 7,
    "threads": 4,
    "workers": 1,
    "queue_depth": 2,
    "verify_decode": True,
}


@dataclass(frozen=True)
class BenchmarkConfiguration:
    tile_candidates: tuple[int, ...]
    overlap: int

    def __post_init__(self) -> None:
        if not self.tile_candidates:
            raise ValueError("A benchmark configuration needs at least one tile")
        if tuple(sorted(set(self.tile_candidates))) != self.tile_candidates:
            raise ValueError("Tile candidates must be unique and sorted")
        if any(tile < 1 or tile % 16 for tile in self.tile_candidates):
            raise ValueError("Tile candidates must be positive multiples of 16")
        if self.overlap < 0 or self.overlap % 8:
            raise ValueError("Overlap must be a non-negative multiple of 8")
        if any(self.overlap >= tile for tile in self.tile_candidates):
            raise ValueError("Overlap must be smaller than every tile candidate")

    @property
    def strategy(self) -> str:
        return (
            FIXED_TILE_STRATEGY
            if len(self.tile_candidates) == 1
            else ADAPTIVE_TILE_STRATEGY
        )

    @property
    def primary_tile(self) -> int:
        return self.tile_candidates[0]

    @property
    def label(self) -> str:
        tiles = "-".join(str(tile) for tile in self.tile_candidates)
        prefix = "fixed" if self.strategy == FIXED_TILE_STRATEGY else "adaptive"
        return f"{prefix}-t{tiles}-o{self.overlap}"

    def record(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "hat_tile": self.primary_tile,
            "hat_tile_candidates": list(self.tile_candidates),
            "hat_overlap": self.overlap,
            "selection_formula": (
                ADAPTIVE_SELECTION_FORMULA
                if self.strategy == ADAPTIVE_TILE_STRATEGY
                else None
            ),
        }

    @classmethod
    def from_record(cls, value: Any) -> BenchmarkConfiguration:
        if not isinstance(value, dict):
            raise ValueError("Benchmark configuration record must be a JSON object")
        raw_candidates = value.get("hat_tile_candidates")
        if not isinstance(raw_candidates, list) or not all(
            type(candidate) is int for candidate in raw_candidates
        ):
            raise ValueError("Benchmark configuration candidates must be a JSON array")
        raw_overlap = value.get("hat_overlap")
        if type(raw_overlap) is not int:
            raise ValueError("Benchmark configuration overlap must be an integer")
        configuration = cls(
            tuple(raw_candidates),
            raw_overlap,
        )
        if value != configuration.record():
            raise ValueError("Benchmark configuration record is not canonical")
        return configuration


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def paths_overlap(left: Path, right: Path) -> bool:
    first = left.expanduser().resolve()
    second = right.expanduser().resolve()
    return first == second or first in second.parents or second in first.parents


class BenchmarkSessionLease:
    """Non-blocking cross-process lease for one benchmark artifact root."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.path = self.root / SESSION_LOCK_NAME
        self._handle: Any = None

    def acquire(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink() or (self.path.exists() and not self.path.is_file()):
            raise ValueError(f"Unsafe benchmark session lock path: {self.path}")
        if self.path.exists() and self.path.stat().st_nlink != 1:
            raise ValueError(
                f"Benchmark session lock must not be hard-linked: {self.path}"
            )
        handle = self.path.open("a+b")
        try:
            if self.path.stat().st_size == 0:
                handle.write(b" ")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            handle.close()
            raise RuntimeError(
                f"Another benchmark process owns this output session: {self.root}"
            ) from exc
        self._handle = handle
        try:
            payload = json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "real_hat_pipeline_e2e_session_lease",
                    "pid": os.getpid(),
                    "acquired_at": utc_now(),
                    "output_root": str(self.root),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            handle.seek(0)
            handle.truncate()
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> BenchmarkSessionLease:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


def safe_relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    base = root.resolve()
    if resolved == base or base not in resolved.parents:
        raise ValueError(f"Path escapes its owned root: {resolved} is not below {base}")
    return resolved.relative_to(base).as_posix()


def resolve_owned_relative(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("Owned artifact path must be a non-empty relative string")
    candidate = (root / Path(relative)).resolve()
    safe_relative(candidate, root)
    return candidate


def assert_absent(path: Path, label: str) -> None:
    if path.exists():
        if path.is_dir() and not any(path.iterdir()):
            raise ValueError(
                f"{label} must be newly created, not an existing empty directory: {path}"
            )
        raise ValueError(f"{label} already exists and is non-empty: {path}")


def validate_isolated_roots(
    input_root: Path,
    output_root: Path,
    metrics_root: Path,
    cache_root: Path,
    *,
    require_fresh: bool,
) -> None:
    input_root = input_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(
            f"Representative input directory is missing: {input_root}"
        )
    owned = {
        "output": output_root.resolve(),
        "metrics": metrics_root.resolve(),
        "cache": cache_root.resolve(),
    }
    for label, path in owned.items():
        if paths_overlap(input_root, path):
            raise ValueError(
                f"{label} directory must not overlap the read-only input: {path}"
            )
        if require_fresh:
            assert_absent(path, label)
    pairs = (("output", "metrics"), ("output", "cache"), ("metrics", "cache"))
    for left, right in pairs:
        if paths_overlap(owned[left], owned[right]):
            raise ValueError(f"Per-run {left} and {right} directories must not overlap")


def input_snapshot(input_root: Path) -> dict[str, dict[str, Any]]:
    root = input_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    snapshot: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"Representative input must not contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        snapshot[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if not snapshot:
        raise ValueError(f"Representative input is empty: {root}")
    return snapshot


def load_representative_manifest(
    manifest_path: Path, input_root: Path, threshold: int
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, int],
    dict[str, int],
]:
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "real_hat_representative_manifest"
    ):
        raise ValueError(f"Unsupported representative manifest: {manifest_path}")
    pages = manifest.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("Representative manifest must contain a non-empty pages list")

    manifest_root = manifest_path.resolve().parent
    root = input_root.resolve()
    indexes: set[int] = set()
    expected: dict[str, dict[str, Any]] = {}
    route_counts: Counter[str] = Counter()
    exact_threshold = 0
    grayscale = 0
    color = 0
    odd_dimension = 0
    pixel_counts: list[int] = []
    for raw_page in pages:
        if not isinstance(raw_page, dict):
            raise ValueError("Representative manifest pages must be JSON objects")
        index = int(raw_page.get("index", 0))
        if index < 1 or index in indexes:
            raise ValueError(f"Invalid or duplicate representative page index: {index}")
        indexes.add(index)
        copied_path = raw_page.get("copied_path")
        expected_sha = raw_page.get("copied_sha256")
        if not isinstance(copied_path, str) or not isinstance(expected_sha, str):
            raise ValueError(f"Page {index} has no copied_path/copied_sha256")
        source = (manifest_root / Path(copied_path)).resolve()
        relative = safe_relative(source, root)
        if source.suffix.lower() not in REPRESENTATIVE_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported representative input extension: {source}")
        if relative in expected:
            raise ValueError(f"Duplicate representative input path: {relative}")
        if not source.is_file() or sha256_file(source) != expected_sha:
            raise ValueError(f"Representative input hash mismatch: {source}")

        with Image.open(source) as opened:
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError(
                    f"Representative page is animated/multi-frame: {source}"
                )
            image = ImageOps.exif_transpose(opened)
            image.load()
        width, height = image.size
        short_edge = min(width, height)
        pixels = width * height
        gray = is_grayscale(image)
        odd = bool(width % 2 or height % 2)
        route = "normal" if short_edge < threshold else "sharper"
        decoded_facts = {
            "width": width,
            "height": height,
            "short_edge": short_edge,
            "long_edge": max(width, height),
            "pixels": pixels,
            "route": route,
            "grayscale": gray,
            "odd_dimension": odd,
        }
        for name, actual_value in decoded_facts.items():
            if raw_page.get(name) != actual_value:
                raise ValueError(
                    f"Representative manifest fact drift at page {index} {name}: "
                    f"{raw_page.get(name)!r} != {actual_value!r}"
                )
        route_counts[route] += 1
        exact_threshold += int(short_edge == threshold)
        grayscale += int(gray)
        color += int(not gray)
        odd_dimension += int(odd)
        pixel_counts.append(pixels)
        expected[relative] = {
            "bytes": source.stat().st_size,
            "sha256": expected_sha,
            "index": index,
            "route": route,
        }

    actual = input_snapshot(root)
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            f"Representative input/manifest file set differs; missing={missing}, extra={extra}"
        )
    for relative, expected_file in expected.items():
        if actual[relative]["sha256"] != expected_file["sha256"]:
            raise ValueError(f"Representative input changed: {relative}")
    calculated_routes = dict(sorted(route_counts.items()))
    if manifest.get("selected_counts") != calculated_routes:
        raise ValueError(
            "Representative manifest selected_counts does not match its page routes"
        )
    coverage = {
        "page_count": len(expected),
        "exact_threshold": exact_threshold,
        "grayscale": grayscale,
        "rgb_or_color": color,
        "odd_dimension": odd_dimension,
        "minimum_selected_pixels": min(pixel_counts),
        "maximum_selected_pixels": max(pixel_counts),
    }
    manifest_coverage = manifest.get("coverage")
    if not isinstance(manifest_coverage, dict):
        raise ValueError("Representative manifest has no coverage record")
    for name, actual_value in coverage.items():
        if name == "page_count":
            continue
        if manifest_coverage.get(name) != actual_value:
            raise ValueError(
                f"Representative manifest coverage drift for {name}: "
                f"{manifest_coverage.get(name)!r} != {actual_value!r}"
            )
    return manifest, actual, calculated_routes, coverage


def resolve_real_hat_models(config: AppConfig) -> dict[str, dict[str, Any]]:
    model_root = config.paths.models.resolve() / "hat"
    result: dict[str, dict[str, Any]] = {}
    for filename in REAL_HAT_MODELS:
        exact = model_root / filename
        matches = [exact] if exact.is_file() else sorted(model_root.rglob(filename))
        matches = [path for path in matches if path.is_file()]
        if not matches:
            raise FileNotFoundError(f"Required Real-HAT checkpoint is missing: {exact}")
        selected = matches[0].resolve()
        actual_sha256 = sha256_file(selected)
        expected_sha256 = OFFICIAL_REAL_HAT_SHA256[filename]
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Official Real-HAT checkpoint hash mismatch for {selected}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        result[filename] = {
            "path": str(selected),
            "bytes": selected.stat().st_size,
            "sha256": actual_sha256,
            "official_sha256": expected_sha256,
        }
    return result


def source_inventory(project_root: Path) -> dict[str, str]:
    candidates = [
        Path(__file__).resolve(),
        *sorted((project_root / "src").rglob("*.py")),
    ]
    candidates.extend(
        path
        for path in (project_root / "pyproject.toml", project_root / "uv.lock")
        if path.is_file()
    )
    return {
        path.resolve().relative_to(project_root.resolve()).as_posix(): sha256_file(path)
        for path in candidates
    }


def runtime_inventory() -> dict[str, Any]:
    runtime_root = Path(
        os.environ.get("WAIFUHAT_RUNTIME_ROOT", Path.home() / ".local/share/waifuhat2x")
    ).expanduser()
    jxl_root = runtime_root / "jxl-0.12.0" / "usr/bin"
    tools: dict[str, Any] = {}
    for name in ("cjxl", "djxl"):
        path = jxl_root / name
        tools[name] = (
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if path.is_file()
            else {"path": str(path.resolve()), "missing": True}
        )
    distributions = {}
    for name in ("numpy", "pillow", "spandrel", "torch", "torchvision", "triton"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    relevant_environment = {
        name: os.environ.get(name)
        for name in (
            "HSA_OVERRIDE_GFX_VERSION",
            "PYTORCH_ROCM_ARCH",
            "ROCM_PATH",
            "WAIFUHAT_RUNTIME_ROOT",
            "TORCH_BLAS_PREFER_HIPBLASLT",
        )
    }
    return {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "distributions": distributions,
        "relevant_environment": relevant_environment,
        "jxl_tools": tools,
    }


def toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return f"[{', '.join(toml_literal(item) for item in value)}]"
    raise TypeError(f"Unsupported TOML value: {value!r}")


def render_child_config(
    base: AppConfig,
    *,
    input_root: Path,
    output_root: Path,
    configuration: BenchmarkConfiguration,
) -> str:
    processing = asdict(base.processing)
    processing.update(
        {
            "tile": configuration.primary_tile,
            "overlap": configuration.overlap,
            "hat_tile": configuration.primary_tile,
            "hat_tile_candidates": list(configuration.tile_candidates),
            "hat_overlap": configuration.overlap,
        }
    )
    output = asdict(base.output)
    output.update(
        {
            "mode": "mirror",
            "format": "jxl",
            "copy_non_images": False,
            "overwrite": False,
            "existing_jxl_policy": "error",
            "allow_lossy_replace": False,
            "allow_metadata_loss": False,
            "allow_alpha_flatten": False,
            "allow_bit_depth_loss": False,
        }
    )
    sections: list[tuple[str, dict[str, Any]]] = [
        (
            "paths",
            {
                "input": str(input_root.resolve()),
                "output": str(output_root.resolve()),
                "models": str(base.paths.models.resolve()),
            },
        ),
        ("processing", processing),
        ("output", output),
        ("jxl", asdict(base.jxl)),
    ]
    lines: list[str] = []
    for name, values in sections:
        lines.append(f"[{name}]")
        lines.extend(f"{key} = {toml_literal(value)}" for key, value in values.items())
        lines.append("")
    return "\n".join(lines)


def validate_production_semantics(config: AppConfig) -> None:
    mismatches = []
    for name, expected in PRODUCTION_PROCESSING_SEMANTICS.items():
        actual = getattr(config.processing, name)
        if actual != expected:
            mismatches.append(f"processing.{name}={actual!r} (expected {expected!r})")
    for name, expected in PRODUCTION_JXL_SEMANTICS.items():
        actual = getattr(config.jxl, name)
        if actual != expected:
            mismatches.append(f"jxl.{name}={actual!r} (expected {expected!r})")
    if mismatches:
        raise ValueError("E2E production semantics drifted: " + "; ".join(mismatches))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated, process-level Real-HAT mirror/JXL benchmarks over a "
            "representative manifest."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Defaults to the manifest's sibling inputs directory.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="Fresh per-child caches; defaults to OUTPUT_ROOT/caches.",
    )
    parser.add_argument(
        "--tile",
        type=positive_int,
        nargs="+",
        default=[256],
        help="Fixed tile sizes; each value becomes one fixed benchmark configuration.",
    )
    parser.add_argument(
        "--adaptive-tiles",
        type=positive_int,
        nargs="+",
        default=[256, 320],
        help="One adaptive candidate set selected by estimated padded work.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fixed-only",
        action="store_true",
        help="Run only --tile fixed configurations and omit the adaptive candidate set.",
    )
    mode.add_argument(
        "--adaptive-only",
        action="store_true",
        help="Run only the --adaptive-tiles candidate set and omit fixed configurations.",
    )
    parser.add_argument("--overlap", type=nonnegative_int, nargs="+", default=[32])
    parser.add_argument(
        "--baseline-overlap",
        type=nonnegative_int,
        help=(
            "Adaptive-only baseline overlap; defaults to processing.hat_overlap from "
            "the base config."
        ),
    )
    parser.add_argument("--warmups", type=nonnegative_int, default=1)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--timeout-seconds", type=positive_float, default=7200.0)
    parser.add_argument("--max-cv-percent", type=positive_float, default=3.0)
    parser.add_argument("--max-reserved-vram-gib", type=positive_float, default=14.0)
    parser.add_argument(
        "--min-wall-reduction-percent", type=positive_float, default=3.0
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse only complete, fingerprint-matching attempts.",
    )
    return parser


def validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> list[BenchmarkConfiguration]:
    if not args.adaptive_only and len(args.tile) != len(set(args.tile)):
        parser.error("duplicate fixed --tile values are not allowed")
    adaptive_tiles = tuple(sorted(args.adaptive_tiles))
    if not args.fixed_only:
        if len(adaptive_tiles) != len(set(adaptive_tiles)):
            parser.error("duplicate --adaptive-tiles values are not allowed")
        if len(adaptive_tiles) < 2:
            parser.error("--adaptive-tiles requires at least two distinct candidates")
    if args.baseline_overlap is not None:
        if not args.adaptive_only:
            parser.error("--baseline-overlap requires --adaptive-only")
        if args.baseline_overlap not in args.overlap:
            parser.error("--baseline-overlap must be present in --overlap")
    if args.adaptive_only and len(args.overlap) != len(set(args.overlap)):
        parser.error("duplicate --overlap values are not allowed with --adaptive-only")

    candidate_groups = [] if args.adaptive_only else [(tile,) for tile in args.tile]
    if not args.fixed_only:
        candidate_groups.append(adaptive_tiles)
    configurations: list[BenchmarkConfiguration] = []
    for candidates in candidate_groups:
        for tile in candidates:
            if tile % 16:
                parser.error("tile candidates must be divisible by 16")
        for overlap in args.overlap:
            if overlap % 8:
                parser.error("--overlap values must be divisible by 8")
            if any(overlap >= tile for tile in candidates):
                parser.error("--overlap must be smaller than every tile candidate")
            configuration = BenchmarkConfiguration(candidates, overlap)
            if configuration in configurations:
                parser.error("duplicate benchmark configurations are not allowed")
            configurations.append(configuration)
    return configurations


def metric_summary(values: list[float]) -> dict[str, Any]:
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


def _phase_report(job: dict[str, Any]) -> dict[str, dict[str, float]]:
    timing = job.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("job.json has no timing object")
    phases: dict[str, dict[str, float]] = {}
    intervals = timing.get("interval_summary")
    if not isinstance(intervals, dict):
        raise ValueError("job.json has no interval_summary")
    for name, raw in intervals.items():
        if name.endswith("_relationship") or not isinstance(raw, dict):
            continue
        phases[name] = {
            "cumulative_seconds": float(raw.get("cumulative_seconds", 0.0)),
            "union_seconds": float(raw.get("union_seconds", 0.0)),
        }
    stage_spans = timing.get("stage_spans")
    if not isinstance(stage_spans, dict):
        raise ValueError("job.json has no stage_spans")
    for name, raw_spans in stage_spans.items():
        if not isinstance(raw_spans, list):
            continue
        seconds = sum(
            float(span.get("duration_seconds", 0.0))
            for span in raw_spans
            if isinstance(span, dict)
        )
        phases[f"stage:{name}"] = {
            "cumulative_seconds": seconds,
            "union_seconds": seconds,
        }
    return phases


def _service_report(job: dict[str, Any]) -> dict[str, float]:
    raw = job.get("timing", {}).get("cumulative_service_seconds", {})
    if not isinstance(raw, dict):
        raise ValueError("job.json cumulative service report is invalid")
    return {str(name): float(seconds) for name, seconds in raw.items()}


def _read_pages(path: Path, job: dict[str, Any]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            page = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid pages.jsonl line {line_number}: {exc}") from exc
        if (
            not isinstance(page, dict)
            or page.get("type") != "waifuhat2x-page-metrics"
            or page.get("schema_version") != 1
        ):
            raise ValueError(f"Invalid page metrics record at line {line_number}")
        if page.get("run_id") != job.get("run_id"):
            raise ValueError("pages.jsonl run_id does not match job.json")
        pages.append(page)
    if len(pages) != int(job.get("pages_written", -1)):
        raise ValueError("pages.jsonl count does not match job.json pages_written")
    return pages


def summarize_telemetry(
    job: dict[str, Any], pages: list[dict[str, Any]]
) -> dict[str, Any]:
    routes: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    selected_tiles: Counter[int] = Counter()
    candidate_sets: Counter[tuple[int, ...]] = Counter()
    tile_strategies: Counter[str] = Counter()
    tile_estimators: Counter[str] = Counter()
    overlaps: Counter[int] = Counter()
    allocated: list[float] = []
    reserved: list[float] = []
    for page in pages:
        statuses[str(page.get("status", "unknown"))] += 1
        details = page.get("details", {})
        if not isinstance(details, dict):
            continue
        label = details.get("model_label")
        if isinstance(label, str):
            if label.endswith("-normal"):
                routes["normal"] += 1
            elif label.endswith("-sharper"):
                routes["sharper"] += 1
        selected_tile = details.get("tile")
        if isinstance(selected_tile, int) and not isinstance(selected_tile, bool):
            selected_tiles[selected_tile] += 1
        raw_candidates = details.get("tile_candidates")
        if (
            isinstance(raw_candidates, list)
            and raw_candidates
            and all(
                isinstance(candidate, int) and not isinstance(candidate, bool)
                for candidate in raw_candidates
            )
        ):
            candidate_sets[tuple(raw_candidates)] += 1
        strategy = details.get("tile_strategy")
        if isinstance(strategy, str):
            tile_strategies[strategy] += 1
        estimator = details.get("tile_estimator")
        tile_estimators[str(estimator) if estimator is not None else "none"] += 1
        overlap = details.get("overlap")
        if isinstance(overlap, int) and not isinstance(overlap, bool):
            overlaps[overlap] += 1
        value = details.get("peak_vram_bytes")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            allocated.append(float(value))
        value = details.get("peak_reserved_vram_bytes")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            reserved.append(float(value))
    interval_summary = job["timing"]["interval_summary"]
    return {
        "route_counts": dict(sorted(routes.items())),
        "page_status_counts": dict(sorted(statuses.items())),
        "peak_allocated_vram_bytes": max(allocated) if allocated else None,
        "peak_reserved_vram_bytes": max(reserved) if reserved else None,
        "reserved_vram_source": (
            "pages.details.peak_reserved_vram_bytes" if reserved else "unavailable"
        ),
        "tile_execution": {
            "selected_tile_counts": {
                str(tile): count for tile, count in sorted(selected_tiles.items())
            },
            "candidate_sets": [
                {"candidates": list(candidates), "pages": count}
                for candidates, count in sorted(candidate_sets.items())
            ],
            "strategy_counts": dict(sorted(tile_strategies.items())),
            "estimator_counts": dict(sorted(tile_estimators.items())),
            "overlap_counts": {
                str(overlap): count for overlap, count in sorted(overlaps.items())
            },
        },
        "phases": _phase_report(job),
        "services": _service_report(job),
        "engine_jxl_relationship": interval_summary.get("engine_jxl_relationship", {}),
        "gpu_jxl_relationship": interval_summary.get("gpu_jxl_relationship", {}),
    }


def _jxl_inventory(output_root: Path) -> list[dict[str, Any]]:
    outputs = sorted(output_root.rglob("*.jxl"), key=lambda item: item.as_posix())
    inventory = []
    for path in outputs:
        if path.is_symlink():
            raise ValueError(f"JXL output must not be a symlink: {path}")
        if not path.is_file():
            continue
        safe_relative(path, output_root)
        inventory.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def _validate_child_config(spec: dict[str, Any]) -> AppConfig:
    attempt_root = Path(spec["attempt_root"]).resolve()
    config_path = Path(spec["config_path"]).resolve()
    if safe_relative(config_path, attempt_root) != "config.toml":
        raise ValueError("Child config must be attempt_root/config.toml")
    if sha256_file(config_path) != spec.get("config_sha256"):
        raise ValueError("Child config hash mismatch")
    config = load_config(config_path)
    input_root = Path(spec["input_root"]).resolve()
    output_root = Path(spec["output_root"]).resolve()
    if config.paths.input != input_root or config.paths.output != output_root:
        raise ValueError("Child config processing roots do not match its spec")
    if config.output.mode != "mirror" or config.output.format.lower() != "jxl":
        raise ValueError("E2E child is restricted to mirror-mode JXL")
    if config.output.overwrite or config.output.existing_jxl_policy != "error":
        raise ValueError("E2E child must not overwrite any output")
    if config.output.copy_non_images:
        raise ValueError("E2E child must not copy metadata into benchmark output")
    validate_production_semantics(config)
    expected = spec["configuration"]
    expected_configuration = BenchmarkConfiguration.from_record(expected)
    expected_candidates = expected_configuration.tile_candidates
    actual_candidates = tuple(config.processing.hat_tile_candidates)
    if (
        config.processing.hat_tile != int(expected["hat_tile"])
        or actual_candidates != expected_candidates
        or config.processing.hat_overlap != int(expected["hat_overlap"])
        or config.processing.hat_tile != actual_candidates[0]
    ):
        raise ValueError("Child tile candidates/strategy/overlap differ from its spec")
    return config


def child_main(spec_path: Path) -> int:
    spec = read_json(spec_path.resolve())
    if (
        spec.get("schema_version") != SCHEMA_VERSION
        or spec.get("kind") != CHILD_SPEC_KIND
    ):
        raise ValueError(f"Unsupported child spec: {spec_path}")
    attempt_root = Path(spec["attempt_root"]).resolve()
    if spec_path.resolve().parent != attempt_root:
        raise ValueError("Child spec must live directly under its attempt root")
    result_path = Path(spec["result_path"]).resolve()
    if safe_relative(result_path, attempt_root) != "result.json":
        raise ValueError("Child result must be attempt_root/result.json")
    if result_path.exists():
        raise FileExistsError(
            f"Refusing to replace an existing child result: {result_path}"
        )
    completion_path = attempt_root / "completion.json"
    if completion_path.exists():
        raise FileExistsError(
            f"Refusing to replace an existing child completion marker: {completion_path}"
        )

    input_root = Path(spec["input_root"]).resolve()
    output_root = Path(spec["output_root"]).resolve()
    metrics_root = Path(spec["metrics_root"]).resolve()
    cache_root = Path(spec["cache_root"]).resolve()
    validate_isolated_roots(
        input_root, output_root, metrics_root, cache_root, require_fresh=True
    )
    expected_input = spec.get("input_snapshot")
    if input_snapshot(input_root) != expected_input:
        raise ValueError("Representative input differs from the parent snapshot")
    config = _validate_child_config(spec)
    models_before = resolve_real_hat_models(config)
    if models_before != spec.get("models"):
        raise ValueError(
            "Child Real-HAT checkpoint inventory differs from its parent spec"
        )
    expected_cache = str(cache_root)
    if os.environ.get("TORCHINDUCTOR_CACHE_DIR") != expected_cache:
        raise ValueError(
            "TORCHINDUCTOR_CACHE_DIR does not match the isolated child cache"
        )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": CHILD_RESULT_KIND,
        "status": "running",
        "fingerprint": spec["fingerprint"],
        "role": spec["role"],
        "index": spec["index"],
        "parent_session_id": spec["parent_session_id"],
        "pair_id": spec["pair_id"],
        "configuration": spec["configuration"],
        "started_at": utc_now(),
        "spec_path": "spec.json",
        "spec_sha256": sha256_file(spec_path),
        "config_path": "config.toml",
        "config_sha256": spec["config_sha256"],
        "models_before": models_before,
        "runtime": runtime_inventory(),
        "input_snapshot_before": expected_input,
        "isolation": {
            "fresh_roots_required_at_child_start": True,
            "input_output_disjoint": not paths_overlap(input_root, output_root),
            "input_metrics_disjoint": not paths_overlap(input_root, metrics_root),
            "input_cache_disjoint": not paths_overlap(input_root, cache_root),
            "output_metrics_disjoint": not paths_overlap(output_root, metrics_root),
            "output_cache_disjoint": not paths_overlap(output_root, cache_root),
            "metrics_cache_disjoint": not paths_overlap(metrics_root, cache_root),
        },
        "owned_roots": {
            "output": str(output_root),
            "metrics": str(metrics_root),
            "cache": str(cache_root),
        },
    }
    write_json(result_path, result)
    started = time.perf_counter()
    try:
        from waifuhat2x.pipeline import run_pipeline

        summary = run_pipeline(config, metrics_dir=metrics_root)
        process_wall = time.perf_counter() - started
        after = input_snapshot(input_root)
        if after != expected_input:
            raise RuntimeError(
                "Read-only representative input changed during the child run"
            )
        models_after = resolve_real_hat_models(config)
        if models_after != models_before:
            raise RuntimeError("Real-HAT checkpoints changed during the child run")
        if summary.metrics_directory is None:
            raise RuntimeError("Pipeline returned no metrics directory")
        metrics_run = Path(summary.metrics_directory).resolve()
        safe_relative(metrics_run, metrics_root)
        job_path = metrics_run / "job.json"
        pages_path = metrics_run / "pages.jsonl"
        if not job_path.is_file() or not pages_path.is_file():
            raise RuntimeError("Pipeline metrics are incomplete")
        job = read_json(job_path)
        if (
            job.get("type") != "waifuhat2x-job-metrics"
            or job.get("schema_version") != 1
            or job.get("status") not in {"complete", "completed_with_errors"}
        ):
            raise RuntimeError("job.json is not a complete pipeline report")
        context = job.get("context", {})
        if (
            context.get("output_mode") != "mirror"
            or str(context.get("output_format", "")).lower() != "jxl"
            or Path(context.get("input_root", "")).resolve() != input_root
            or Path(context.get("output_root", "")).resolve() != output_root
        ):
            raise RuntimeError("job.json context violates E2E isolation")
        pages = _read_pages(pages_path, job)
        jxl_outputs = _jxl_inventory(output_root)
        if len(jxl_outputs) != int(summary.processed):
            raise RuntimeError("JXL output count does not match processed page count")

        result.update(
            {
                "status": "complete",
                "finished_at": utc_now(),
                "process_wall_seconds": process_wall,
                "pipeline_summary": asdict(summary),
                "input_snapshot_after": after,
                "models_after": models_after,
                "job": {
                    "path": safe_relative(job_path, attempt_root),
                    "sha256": sha256_file(job_path),
                    "status": job["status"],
                },
                "pages": {
                    "path": safe_relative(pages_path, attempt_root),
                    "sha256": sha256_file(pages_path),
                    "count": len(pages),
                },
                "jxl_outputs": jxl_outputs,
                "telemetry": summarize_telemetry(job, pages),
            }
        )
        write_json(result_path, result)
        write_json(
            completion_path,
            {
                "schema_version": SCHEMA_VERSION,
                "kind": COMPLETION_KIND,
                "fingerprint": spec["fingerprint"],
                "completed_at": utc_now(),
                "result_path": "result.json",
                "result_sha256": sha256_file(result_path),
                "spec_sha256": result["spec_sha256"],
                "config_sha256": result["config_sha256"],
                "job_sha256": result["job"]["sha256"],
                "pages_sha256": result["pages"]["sha256"],
            },
        )
        return 0
    except BaseException as exc:
        result.update(
            {
                "status": "error",
                "finished_at": utc_now(),
                "process_wall_seconds": time.perf_counter() - started,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        try:
            result["input_snapshot_after"] = input_snapshot(input_root)
        except Exception as snapshot_error:
            result["input_snapshot_error"] = {
                "type": type(snapshot_error).__name__,
                "message": str(snapshot_error),
            }
        write_json(result_path, result)
        raise


def _bounded_wait(process: subprocess.Popen[Any], timeout_seconds: float) -> int:
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Child PID {process.pid} did not exit within {timeout_seconds:g}s"
        ) from exc


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(process_group: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _process_group_exists(process_group):
            return True
        time.sleep(0.05)
    return not _process_group_exists(process_group)


def _terminate_process_tree(process: subprocess.Popen[Any]) -> dict[str, Any]:
    if process.pid <= 1:
        raise RuntimeError(f"Refusing to terminate unsafe child PID: {process.pid}")
    leader_already_exited = process.poll() is not None
    if os.name == "nt":
        taskkill_returncode: int | None = None
        taskkill_timed_out = False
        if not leader_already_exited:
            try:
                taskkill = subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=10,
                )
                taskkill_returncode = taskkill.returncode
            except (subprocess.TimeoutExpired, OSError) as taskkill_error:
                taskkill_timed_out = True
                taskkill_failure = f"{type(taskkill_error).__name__}: {taskkill_error}"
            else:
                taskkill_failure = None
            if (
                taskkill_timed_out or taskkill_returncode != 0
            ) and process.poll() is None:
                process.kill()
            try:
                _bounded_wait(process, 10)
            except RuntimeError:
                if process.poll() is None:
                    process.kill()
                _bounded_wait(process, 5)
        else:
            _bounded_wait(process, 0.1)
        if process.poll() is None:
            raise RuntimeError(f"Child PID {process.pid} survived taskkill/kill")
        return {
            "method": "already-exited" if leader_already_exited else "taskkill",
            "taskkill_returncode": taskkill_returncode,
            "taskkill_timed_out": taskkill_timed_out,
            "taskkill_failure": taskkill_failure if not leader_already_exited else None,
            "process_group_checked": False,
        }

    # start_new_session=True makes the child's PID its owned process-group ID.
    process_group = process.pid
    if process_group == os.getpgrp():
        raise RuntimeError("Refusing to signal the benchmark parent's process group")
    group_existed = _process_group_exists(process_group)
    if group_existed:
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass

    if process.poll() is None:
        try:
            _bounded_wait(process, 5)
        except RuntimeError:
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                process.kill()
            _bounded_wait(process, 5)
    else:
        _bounded_wait(process, 0.1)

    group_killed = False
    if not _wait_for_process_group_exit(process_group, 1):
        try:
            os.killpg(process_group, signal.SIGKILL)
            group_killed = True
        except ProcessLookupError:
            pass
        if not _wait_for_process_group_exit(process_group, 5):
            raise RuntimeError(f"Owned process group {process_group} survived SIGKILL")
    if process.poll() is None:
        raise RuntimeError(f"Child PID {process.pid} was not reaped")
    return {
        "method": "process-group",
        "leader_already_exited": leader_already_exited,
        "process_group_checked": True,
        "process_group_existed": group_existed,
        "process_group_killed": group_killed,
    }


def run_child_process(
    spec_path: Path,
    log_path: Path,
    timeout_seconds: float,
    *,
    backend_environment: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    spec = read_json(spec_path)
    env = os.environ.copy()
    env["TORCHINDUCTOR_CACHE_DIR"] = str(Path(spec["cache_root"]).resolve())
    env["TRITON_CACHE_DIR"] = str(Path(spec["cache_root"]).resolve() / "triton")
    allowed_backend_keys = {"TORCH_BLAS_PREFER_HIPBLASLT"}
    if backend_environment is not None:
        unexpected = set(backend_environment) - allowed_backend_keys
        if unexpected:
            raise ValueError(
                "Unsupported child backend environment keys: "
                + ", ".join(sorted(unexpected))
            )
        for name, value in backend_environment.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    started = time.perf_counter()
    process: subprocess.Popen[Any] | None = None
    termination: dict[str, Any] | None = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    CHILD_FLAG,
                    str(spec_path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                start_new_session=os.name != "nt",
                creationflags=creationflags,
            )
            write_json(
                spec_path.parent / "attempt.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "real_hat_pipeline_e2e_attempt",
                    "status": "running",
                    "pid": process.pid,
                    "started_at": utc_now(),
                    "fingerprint": spec["fingerprint"],
                },
            )
            try:
                returncode = process.wait(timeout=timeout_seconds)
                timed_out = False
            except subprocess.TimeoutExpired:
                termination = _terminate_process_tree(process)
                returncode = 124
                timed_out = True
            except BaseException:
                termination = _terminate_process_tree(process)
                raise
        finally:
            if process is not None:
                if process.poll() is None:
                    termination = _terminate_process_tree(process)
                elif termination is None and os.name != "nt":
                    # The leader can exit while a cjxl/djxl descendant remains in its session.
                    termination = _terminate_process_tree(process)
                _bounded_wait(process, 5)
    report = {
        "returncode": returncode,
        "timed_out": timed_out,
        "termination": termination,
        "wall_seconds": time.perf_counter() - started,
        "log_path": log_path.name,
        "backend_environment": {
            name: env.get(name) for name in sorted(allowed_backend_keys)
        },
    }
    write_json(
        spec_path.parent / "attempt.json",
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "real_hat_pipeline_e2e_attempt",
            "status": "complete" if returncode == 0 else "child_error",
            "finished_at": utc_now(),
            "fingerprint": spec["fingerprint"],
            **report,
        },
    )
    return report


def _valid_complete_result(
    attempt_root: Path,
    *,
    fingerprint: str,
    expected_input_root: Path,
    expected_input: dict[str, dict[str, Any]],
    expected_models: dict[str, dict[str, Any]],
    expected_configuration: BenchmarkConfiguration,
    expected_role: str,
    expected_index: int,
    expected_cache_root: Path,
) -> dict[str, Any] | None:
    result_path = attempt_root / "result.json"
    spec_path = attempt_root / "spec.json"
    config_path = attempt_root / "config.toml"
    attempt_path = attempt_root / "attempt.json"
    completion_path = attempt_root / "completion.json"
    if not all(
        path.is_file()
        for path in (result_path, spec_path, config_path, attempt_path, completion_path)
    ) or any(
        path.is_symlink()
        for path in (result_path, spec_path, config_path, attempt_path, completion_path)
    ):
        return None
    try:
        result = read_json(result_path)
        spec = read_json(spec_path)
        attempt = read_json(attempt_path)
        completion = read_json(completion_path)
        validated_config = _validate_child_config(spec)
        resolved_models = resolve_real_hat_models(validated_config)
        pipeline_summary = result.get("pipeline_summary")
        telemetry = result.get("telemetry")
        jxl_outputs = result.get("jxl_outputs")
        isolation = result.get("isolation")
        BenchmarkConfiguration.from_record(result.get("configuration"))
        valid = all(
            (
                result.get("schema_version") == SCHEMA_VERSION,
                result.get("kind") == CHILD_RESULT_KIND,
                result.get("status") == "complete",
                result.get("fingerprint") == fingerprint,
                result.get("role") == expected_role,
                result.get("index") == expected_index,
                isinstance(result.get("parent_session_id"), str),
                bool(result.get("parent_session_id")),
                result.get("parent_session_id") == spec.get("parent_session_id"),
                result.get("pair_id") == f"{expected_role}-{expected_index}",
                result.get("pair_id") == spec.get("pair_id"),
                result.get("configuration") == expected_configuration.record(),
                result.get("input_snapshot_before") == expected_input,
                result.get("input_snapshot_after") == expected_input,
                result.get("spec_path") == "spec.json",
                result.get("spec_sha256") == sha256_file(spec_path),
                result.get("config_path") == "config.toml",
                result.get("config_sha256") == sha256_file(config_path),
                isinstance(pipeline_summary, dict),
                isinstance(telemetry, dict),
                isinstance(jxl_outputs, list),
                isinstance(isolation, dict),
                all(isolation.values()) if isinstance(isolation, dict) else False,
                isinstance(result.get("owned_roots"), dict),
                isinstance(result.get("models_before"), dict),
                resolved_models == expected_models,
                spec.get("models") == expected_models,
                result.get("models_before") == expected_models,
                result.get("models_before") == result.get("models_after"),
                spec.get("schema_version") == SCHEMA_VERSION,
                spec.get("kind") == CHILD_SPEC_KIND,
                spec.get("fingerprint") == fingerprint,
                Path(spec.get("attempt_root", "")).resolve() == attempt_root,
                spec.get("role") == expected_role,
                spec.get("index") == expected_index,
                spec.get("pair_id") == f"{expected_role}-{expected_index}",
                spec.get("configuration") == result.get("configuration"),
                Path(spec.get("input_root", "")).resolve()
                == expected_input_root.resolve(),
                spec.get("input_snapshot") == expected_input,
                spec.get("models") == expected_models,
                Path(spec.get("output_root", "")).resolve()
                == (attempt_root / "output").resolve(),
                Path(spec.get("metrics_root", "")).resolve()
                == (attempt_root / "metrics").resolve(),
                Path(spec.get("cache_root", "")).resolve()
                == expected_cache_root.resolve(),
                Path(spec.get("config_path", "")).resolve() == config_path,
                Path(spec.get("result_path", "")).resolve() == result_path,
                spec.get("config_sha256") == result.get("config_sha256"),
                attempt.get("schema_version") == SCHEMA_VERSION,
                attempt.get("kind") == "real_hat_pipeline_e2e_attempt",
                attempt.get("status") == "complete",
                attempt.get("fingerprint") == fingerprint,
                attempt.get("returncode") == 0,
                attempt.get("timed_out") is False,
                completion.get("schema_version") == SCHEMA_VERSION,
                completion.get("kind") == COMPLETION_KIND,
                completion.get("fingerprint") == fingerprint,
                completion.get("result_path") == "result.json",
                completion.get("result_sha256") == sha256_file(result_path),
                completion.get("spec_sha256") == result.get("spec_sha256"),
                completion.get("config_sha256") == result.get("config_sha256"),
            )
        )
        if not valid:
            return None
        owned_roots = result["owned_roots"]
        if any(
            Path(owned_roots[name]).resolve() != Path(spec[f"{name}_root"]).resolve()
            for name in ("output", "metrics", "cache")
        ):
            return None
        if os.name != "nt":
            termination = attempt.get("termination")
            if not isinstance(termination, dict) or not termination.get(
                "process_group_checked"
            ):
                return None
        assert isinstance(pipeline_summary, dict)
        assert isinstance(telemetry, dict)
        if not all(
            name in pipeline_summary
            for name in (
                "wall_seconds",
                "failed",
                "deferred",
                "target_unmet",
                "metrics_write_errors",
            )
        ) or not all(
            name in telemetry
            for name in (
                "route_counts",
                "page_status_counts",
                "tile_execution",
                "phases",
                "services",
                "peak_allocated_vram_bytes",
                "peak_reserved_vram_bytes",
            )
        ):
            return None
        job_info = result.get("job")
        pages_info = result.get("pages")
        if not isinstance(job_info, dict) or not isinstance(pages_info, dict):
            return None
        job_path = resolve_owned_relative(attempt_root, job_info.get("path"))
        pages_path = resolve_owned_relative(attempt_root, pages_info.get("path"))
        if (
            not job_path.is_file()
            or not pages_path.is_file()
            or job_path.is_symlink()
            or pages_path.is_symlink()
            or sha256_file(job_path) != job_info.get("sha256")
            or sha256_file(pages_path) != pages_info.get("sha256")
            or completion.get("job_sha256") != job_info.get("sha256")
            or completion.get("pages_sha256") != pages_info.get("sha256")
        ):
            return None
        job = read_json(job_path)
        if (
            job.get("type") != "waifuhat2x-job-metrics"
            or job.get("schema_version") != 1
            or job.get("status") not in {"complete", "completed_with_errors"}
        ):
            return None
        context = job.get("context")
        if (
            not isinstance(context, dict)
            or context.get("output_mode") != "mirror"
            or str(context.get("output_format", "")).lower() != "jxl"
            or Path(context.get("input_root", "")).resolve()
            != expected_input_root.resolve()
            or Path(context.get("output_root", "")).resolve()
            != (attempt_root / "output").resolve()
        ):
            return None
        output_root = attempt_root / "output"
        actual_outputs = _jxl_inventory(output_root)
        if actual_outputs != result.get("jxl_outputs"):
            return None
        parsed_pages = _read_pages(pages_path, job)
        if len(parsed_pages) != int(pages_info.get("count", -1)):
            return None
        if summarize_telemetry(job, parsed_pages) != telemetry:
            return None
        return result
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _attempt_directories(slot_root: Path) -> list[Path]:
    if not slot_root.is_dir():
        return []
    return sorted(
        (
            path
            for path in slot_root.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and path.name.startswith("attempt-")
        ),
        reverse=True,
    )


def find_reusable_result(
    slot_root: Path,
    *,
    fingerprint: str,
    expected_input_root: Path,
    expected_input: dict[str, dict[str, Any]],
    expected_models: dict[str, dict[str, Any]],
    expected_configuration: BenchmarkConfiguration,
    expected_role: str,
    expected_index: int,
    cache_slot_root: Path,
) -> tuple[Path, dict[str, Any]] | None:
    for attempt in _attempt_directories(slot_root):
        result = _valid_complete_result(
            attempt,
            fingerprint=fingerprint,
            expected_input_root=expected_input_root,
            expected_input=expected_input,
            expected_models=expected_models,
            expected_configuration=expected_configuration,
            expected_role=expected_role,
            expected_index=expected_index,
            expected_cache_root=cache_slot_root / attempt.name,
        )
        if result is not None:
            return attempt, result
    return None


def next_attempt_root(slot_root: Path) -> Path:
    attempts = _attempt_directories(slot_root)
    numbers: list[int] = []
    for path in attempts:
        try:
            numbers.append(int(path.name.removeprefix("attempt-")))
        except ValueError:
            continue
    attempt = slot_root / f"attempt-{max(numbers, default=0) + 1:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt


def _run_fingerprint(
    benchmark_fingerprint: str,
    configuration: BenchmarkConfiguration,
    role: str,
    index: int,
) -> str:
    return json_fingerprint(
        {
            "benchmark_fingerprint": benchmark_fingerprint,
            "configuration": configuration.record(),
            "role": role,
            "index": index,
        }
    )


def aggregate_configuration(
    *,
    configuration: BenchmarkConfiguration,
    warmups: list[dict[str, Any]],
    repeats: list[dict[str, Any]],
    expected_routes: dict[str, int],
    expected_pages: int,
    max_cv_percent: float,
    max_reserved_vram_bytes: int,
) -> dict[str, Any]:
    walls = [float(run["pipeline_summary"]["wall_seconds"]) for run in repeats]
    wall_report = metric_summary(walls)
    output_hashes: dict[str, set[str]] = defaultdict(set)
    output_sets: list[dict[str, str]] = []
    for run in repeats:
        mapping = {item["path"]: item["sha256"] for item in run["jxl_outputs"]}
        output_sets.append(mapping)
        for path, digest in mapping.items():
            output_hashes[path].add(digest)
    deterministic = bool(output_sets) and all(
        mapping == output_sets[0] for mapping in output_sets[1:]
    )

    phase_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"cumulative_seconds": 0.0, "union_seconds": 0.0}
    )
    service_totals: dict[str, float] = defaultdict(float)
    observed_runs = warmups + repeats
    reserved_values: list[float] = []
    allocated_values: list[float] = []
    route_runs: list[dict[str, int]] = []
    page_status_runs: list[dict[str, int]] = []
    tile_execution_runs: list[dict[str, Any]] = []
    for run in observed_runs:
        telemetry = run["telemetry"]
        route_runs.append(telemetry["route_counts"])
        page_status_runs.append(telemetry.get("page_status_counts", {}))
        tile_execution_runs.append(telemetry.get("tile_execution", {}))
        reserved = telemetry.get("peak_reserved_vram_bytes")
        allocated = telemetry.get("peak_allocated_vram_bytes")
        if reserved is not None:
            reserved_values.append(float(reserved))
        if allocated is not None:
            allocated_values.append(float(allocated))
    for run in repeats:
        telemetry = run["telemetry"]
        for name, values in telemetry["phases"].items():
            phase_totals[name]["cumulative_seconds"] += float(
                values["cumulative_seconds"]
            )
            phase_totals[name]["union_seconds"] += float(values["union_seconds"])
        for name, seconds in telemetry["services"].items():
            service_totals[name] += float(seconds)
    wall_total = sum(walls)
    phases = {
        name: {
            **values,
            "union_share_of_batch_wall_percent": (
                values["union_seconds"] / wall_total * 100 if wall_total else None
            ),
        }
        for name, values in sorted(phase_totals.items())
    }
    services = {
        name: {
            "cumulative_seconds": seconds,
            "cumulative_share_of_batch_wall_percent": (
                seconds / wall_total * 100 if wall_total else None
            ),
        }
        for name, seconds in sorted(service_totals.items())
    }
    pipeline_summaries = [run["pipeline_summary"] for run in observed_runs]
    failed = sum(int(summary["failed"]) for summary in pipeline_summaries)
    deferred = sum(int(summary["deferred"]) for summary in pipeline_summaries)
    target_unmet = sum(int(summary["target_unmet"]) for summary in pipeline_summaries)
    metrics_errors = sum(
        int(summary["metrics_write_errors"]) for summary in pipeline_summaries
    )
    page_counts = [len(run["jxl_outputs"]) for run in observed_runs]
    route_match = all(routes == expected_routes for routes in route_runs)
    page_status_match = all(
        statuses == {"complete": expected_pages} for statuses in page_status_runs
    )
    page_count_match = all(count == expected_pages for count in page_counts)
    cv = wall_report["cv_percent"]
    reserved_available = len(reserved_values) == len(observed_runs)
    expected_candidates = list(configuration.tile_candidates)
    expected_candidate_sets = [
        {"candidates": expected_candidates, "pages": expected_pages}
    ]
    expected_strategy_counts = {configuration.strategy: expected_pages}
    expected_estimator_counts = {
        (
            ADAPTIVE_SELECTION_FORMULA.split(";", maxsplit=1)[0]
            if configuration.strategy == ADAPTIVE_TILE_STRATEGY
            else "none"
        ): expected_pages
    }
    expected_overlap_counts = {str(configuration.overlap): expected_pages}
    tile_execution_matches = all(
        execution.get("candidate_sets") == expected_candidate_sets
        and execution.get("strategy_counts") == expected_strategy_counts
        and execution.get("estimator_counts") == expected_estimator_counts
        and execution.get("overlap_counts") == expected_overlap_counts
        and sum(execution.get("selected_tile_counts", {}).values()) == expected_pages
        and {int(tile) for tile in execution.get("selected_tile_counts", {})}.issubset(
            configuration.tile_candidates
        )
        for execution in tile_execution_runs
    )
    all_candidates_exercised = all(
        {int(tile) for tile in execution.get("selected_tile_counts", {})}
        == set(configuration.tile_candidates)
        for execution in tile_execution_runs
    )
    checks = {
        "repeat_count": len(repeats) > 0,
        "cv_below_limit": cv is not None and cv < max_cv_percent,
        "jxl_byte_deterministic": deterministic,
        "no_failed_or_deferred_pages": failed == 0 and deferred == 0,
        "metrics_complete": metrics_errors == 0,
        "route_counts_match_manifest": route_match,
        "all_page_metrics_complete": page_status_match,
        "tile_execution_matches_configuration": tile_execution_matches,
        "all_configured_tiles_exercised": all_candidates_exercised,
        "output_count_matches_manifest": page_count_match,
        "reserved_vram_available": reserved_available,
        "reserved_vram_within_limit": (
            reserved_available and max(reserved_values) <= max_reserved_vram_bytes
        ),
        "input_snapshot_unchanged": all(
            run.get("input_snapshot_unchanged") is True for run in warmups + repeats
        ),
        "isolated_fresh_roots": all(
            isinstance(run.get("isolation"), dict) and all(run["isolation"].values())
            for run in warmups + repeats
        ),
        "attempt_completion_integrity": all(
            isinstance(run.get("attestation"), dict)
            and set(run["attestation"])
            == {
                "child_spec",
                "child_config",
                "attempt_status",
                "completion_marker",
            }
            and all(
                isinstance(item, dict)
                and isinstance(item.get("sha256"), str)
                and len(item["sha256"]) == 64
                for item in run["attestation"].values()
            )
            for run in warmups + repeats
        ),
    }
    return {
        "configuration": configuration.record(),
        "warmup_runs": warmups,
        "measured_runs": repeats,
        "batch_wall_seconds": wall_report,
        "jxl_byte_deterministic": deterministic,
        "jxl_hashes_per_path": {
            path: sorted(hashes) for path, hashes in sorted(output_hashes.items())
        },
        "route_counts_per_run": route_runs,
        "page_status_counts_per_run": page_status_runs,
        "tile_execution_per_run": tile_execution_runs,
        "expected_route_counts": expected_routes,
        "page_counts_per_run": page_counts,
        "failures_total": failed,
        "deferred_total": deferred,
        "target_unmet_total": target_unmet,
        "metrics_write_errors_total": metrics_errors,
        "peak_allocated_vram_bytes": max(allocated_values)
        if allocated_values
        else None,
        "peak_reserved_vram_bytes": max(reserved_values) if reserved_values else None,
        "reserved_vram_status": "available" if reserved_available else "unavailable",
        "phase_breakdown": phases,
        "service_breakdown": services,
        "qualification": {
            "valid_for_performance_decision": all(checks.values()),
            "checks": checks,
            "limits": {
                "max_cv_percent": max_cv_percent,
                "max_reserved_vram_bytes": max_reserved_vram_bytes,
            },
        },
    }


def build_comparison(
    reports: list[dict[str, Any]], *, min_wall_reduction_percent: float
) -> list[dict[str, Any]]:
    fixed_256_by_overlap = {
        int(report["configuration"]["hat_overlap"]): report
        for report in reports
        if report["configuration"]["strategy"] == FIXED_TILE_STRATEGY
        and report["configuration"]["hat_tile_candidates"]
        == list(PRODUCTION_BASELINE_TILES)
    }
    comparison: list[dict[str, Any]] = []
    for report in reports:
        configuration = report["configuration"]
        baseline = fixed_256_by_overlap.get(int(configuration["hat_overlap"]))
        baseline_mean = (
            baseline["batch_wall_seconds"]["mean"] if baseline is not None else None
        )
        mean = report["batch_wall_seconds"]["mean"]
        reduction = (
            (baseline_mean - mean) / baseline_mean * 100
            if baseline_mean and mean
            else None
        )
        is_baseline = configuration[
            "strategy"
        ] == FIXED_TILE_STRATEGY and configuration["hat_tile_candidates"] == list(
            PRODUCTION_BASELINE_TILES
        )
        comparison.append(
            {
                "configuration": configuration,
                "baseline_configuration": (
                    baseline["configuration"] if baseline is not None else None
                ),
                "mean_batch_wall_seconds": mean,
                "speedup_vs_fixed_256": (
                    baseline_mean / mean if baseline_mean and mean else None
                ),
                "wall_reduction_vs_fixed_256_percent": reduction,
                "is_fixed_256_baseline": is_baseline,
                "meets_minimum_wall_reduction": (
                    None
                    if is_baseline
                    else reduction is not None
                    and reduction >= min_wall_reduction_percent
                ),
                "valid_for_performance_decision": report["qualification"][
                    "valid_for_performance_decision"
                ],
            }
        )
    return comparison


def validate_adaptive_overlap_matrix(
    configurations: list[BenchmarkConfiguration], baseline_overlap: int
) -> tuple[BenchmarkConfiguration, list[BenchmarkConfiguration]]:
    duplicate_counts = Counter(configurations)
    duplicates = [
        configuration.label
        for configuration, count in duplicate_counts.items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate adaptive overlap configurations: {duplicates}")
    baselines = [
        configuration
        for configuration in configurations
        if configuration.strategy == ADAPTIVE_TILE_STRATEGY
        and configuration.overlap == baseline_overlap
    ]
    if len(baselines) != 1:
        raise ValueError(
            "Adaptive-only matrix requires exactly one adaptive baseline with "
            f"overlap {baseline_overlap}; found {len(baselines)}"
        )
    baseline = baselines[0]
    non_candidates = [
        configuration.label
        for configuration in configurations
        if configuration.strategy != ADAPTIVE_TILE_STRATEGY
        or configuration.tile_candidates != baseline.tile_candidates
    ]
    if non_candidates:
        raise ValueError(
            "Adaptive-only baseline and candidates must use one identical adaptive "
            f"tile set; invalid configurations: {non_candidates}"
        )
    candidates = [
        configuration
        for configuration in configurations
        if configuration.overlap != baseline_overlap
    ]
    if not candidates:
        raise ValueError("Adaptive-only matrix requires at least one candidate overlap")
    return baseline, candidates


def build_adaptive_overlap_comparison(
    reports: list[dict[str, Any]],
    *,
    baseline_overlap: int,
    min_wall_reduction_percent: float,
) -> list[dict[str, Any]]:
    configurations = [
        BenchmarkConfiguration.from_record(report["configuration"])
        for report in reports
    ]
    baseline_configuration, _ = validate_adaptive_overlap_matrix(
        configurations, baseline_overlap
    )
    baseline_index = configurations.index(baseline_configuration)
    baseline_report = reports[baseline_index]
    baseline_mean = baseline_report["batch_wall_seconds"]["mean"]
    if not baseline_mean or not math.isfinite(float(baseline_mean)):
        raise ValueError("Adaptive overlap baseline needs a positive finite mean wall time")

    comparison: list[dict[str, Any]] = []
    for configuration, report in zip(configurations, reports):
        mean = report["batch_wall_seconds"]["mean"]
        reduction = (
            (baseline_mean - mean) / baseline_mean * 100 if mean else None
        )
        is_baseline = configuration == baseline_configuration
        comparison.append(
            {
                "configuration": report["configuration"],
                "baseline_configuration": baseline_report["configuration"],
                "mean_batch_wall_seconds": mean,
                "speedup_vs_adaptive_baseline": (
                    baseline_mean / mean if mean else None
                ),
                "wall_reduction_vs_adaptive_baseline_percent": reduction,
                "is_adaptive_overlap_baseline": is_baseline,
                "meets_minimum_wall_reduction": (
                    None
                    if is_baseline
                    else reduction is not None
                    and reduction >= min_wall_reduction_percent
                ),
                "valid_for_performance_decision": report["qualification"][
                    "valid_for_performance_decision"
                ],
            }
        )
    return comparison


def _shared_protocol_checks(
    reports: list[dict[str, Any]],
    *,
    expected_configuration_count: int,
    expected_routes: dict[str, int],
    expected_pages: int,
    representative_coverage: dict[str, int],
    warmups: int,
    repeats: int,
    max_cv_percent: float,
    max_reserved_vram_bytes: int,
    min_wall_reduction_percent: float,
) -> dict[str, bool]:
    all_runs = [
        run
        for report in reports
        for run in report["warmup_runs"] + report["measured_runs"]
    ]
    expected_run_count = expected_configuration_count * (warmups + repeats)
    artifact_paths = [run.get("attempt") for run in all_runs]
    evidence_sessions = {run.get("parent_session_id") for run in all_runs}
    pair_members: Counter[tuple[str, int, str]] = Counter(
        (
            str(run.get("parent_session_id")),
            int(run["index"]),
            str(run["role"]),
        )
        for run in all_runs
    )
    owned_roots = [
        str(path) for run in all_runs for path in run.get("owned_roots", {}).values()
    ]
    return {
        "representative_page_count_30": expected_pages == PRODUCTION_PAGE_COUNT,
        "representative_routes_normal_9_sharper_21": expected_routes
        == PRODUCTION_ROUTE_COUNTS,
        "representative_boundary_and_image_modes_redecoded": (
            representative_coverage.get("page_count") == PRODUCTION_PAGE_COUNT
            and representative_coverage.get("exact_threshold", 0) >= 2
            and representative_coverage.get("grayscale", 0) > 0
            and representative_coverage.get("rgb_or_color", 0) > 0
            and representative_coverage.get("odd_dimension", 0) > 0
            and representative_coverage.get("minimum_selected_pixels", 0) > 0
            and representative_coverage.get("maximum_selected_pixels", 0)
            >= representative_coverage.get("minimum_selected_pixels", 0)
        ),
        "at_least_one_warmup_per_configuration": warmups
        >= PRODUCTION_MIN_WARMUPS
        and all(len(report["warmup_runs"]) == warmups for report in reports),
        "three_measured_runs_per_configuration": repeats == PRODUCTION_REPEATS
        and all(
            len(report["measured_runs"]) == PRODUCTION_REPEATS for report in reports
        ),
        "cv_limit_is_no_looser_than_3_percent": max_cv_percent
        <= PRODUCTION_MAX_CV_PERCENT,
        "reserved_vram_limit_is_no_looser_than_14_gib": max_reserved_vram_bytes
        <= PRODUCTION_MAX_RESERVED_VRAM_BYTES,
        "speed_gate_is_no_looser_than_3_percent": min_wall_reduction_percent
        >= PRODUCTION_MIN_WALL_REDUCTION_PERCENT,
        "all_configuration_checks_pass": len(reports) == expected_configuration_count
        and all(
            report["qualification"]["valid_for_performance_decision"]
            for report in reports
        ),
        "all_jxl_outputs_byte_deterministic_within_configuration": all(
            report["jxl_byte_deterministic"] for report in reports
        ),
        "all_input_snapshots_unchanged": len(all_runs) == expected_run_count
        and all(run.get("input_snapshot_unchanged") is True for run in all_runs),
        "fresh_isolated_owned_roots_per_run": len(owned_roots) == expected_run_count * 3
        and len(owned_roots) == len(set(owned_roots)),
        "attempt_and_completion_integrity": len(all_runs) == expected_run_count
        and all(
            report["qualification"]["checks"]["attempt_completion_integrity"]
            for report in reports
        )
        and len(artifact_paths) == len(set(artifact_paths)),
        "single_parent_session_and_complete_pairs": len(evidence_sessions) == 1
        and None not in evidence_sessions
        and set(pair_members.values()) == {expected_configuration_count}
        and len(pair_members) == warmups + repeats,
    }


def production_qualification(
    reports: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    *,
    expected_routes: dict[str, int],
    expected_pages: int,
    representative_coverage: dict[str, int],
    warmups: int,
    repeats: int,
    max_cv_percent: float,
    max_reserved_vram_bytes: int,
    min_wall_reduction_percent: float,
) -> dict[str, Any]:
    expected_matrix = [
        BenchmarkConfiguration(PRODUCTION_BASELINE_TILES, PRODUCTION_OVERLAP).record(),
        BenchmarkConfiguration(PRODUCTION_CANDIDATE_TILES, PRODUCTION_OVERLAP).record(),
    ]
    actual_matrix = [report["configuration"] for report in reports]
    candidate_comparisons = [
        item
        for item in comparison
        if item["configuration"]["strategy"] == ADAPTIVE_TILE_STRATEGY
        and item["configuration"]["hat_tile_candidates"]
        == list(PRODUCTION_CANDIDATE_TILES)
        and item["configuration"]["hat_overlap"] == PRODUCTION_OVERLAP
    ]
    checks = {
        "exact_fixed_256_vs_adaptive_256_320_matrix": actual_matrix == expected_matrix,
        **_shared_protocol_checks(
            reports,
            expected_configuration_count=len(expected_matrix),
            expected_routes=expected_routes,
            expected_pages=expected_pages,
            representative_coverage=representative_coverage,
            warmups=warmups,
            repeats=repeats,
            max_cv_percent=max_cv_percent,
            max_reserved_vram_bytes=max_reserved_vram_bytes,
            min_wall_reduction_percent=min_wall_reduction_percent,
        ),
        "adaptive_wall_reduction_at_least_minimum": len(candidate_comparisons) == 1
        and candidate_comparisons[0]["meets_minimum_wall_reduction"] is True,
    }
    return {
        "valid_for_production_decision": all(checks.values()),
        "checks": checks,
        "required_protocol": {
            "configurations": expected_matrix,
            "route_counts": PRODUCTION_ROUTE_COUNTS,
            "page_count": PRODUCTION_PAGE_COUNT,
            "coverage": {
                "exact_threshold_at_least": 2,
                "requires_grayscale": True,
                "requires_color": True,
                "requires_odd_dimension": True,
            },
            "minimum_warmups_per_configuration": PRODUCTION_MIN_WARMUPS,
            "measured_runs_per_configuration": PRODUCTION_REPEATS,
            "max_cv_percent": PRODUCTION_MAX_CV_PERCENT,
            "max_reserved_vram_bytes": PRODUCTION_MAX_RESERVED_VRAM_BYTES,
            "min_wall_reduction_percent": PRODUCTION_MIN_WALL_REDUCTION_PERCENT,
        },
        "effective_limits": {
            "max_cv_percent": max_cv_percent,
            "max_reserved_vram_bytes": max_reserved_vram_bytes,
            "min_wall_reduction_percent": min_wall_reduction_percent,
        },
    }


def adaptive_overlap_qualification(
    reports: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    *,
    expected_configurations: list[BenchmarkConfiguration],
    baseline_overlap: int,
    expected_routes: dict[str, int],
    expected_pages: int,
    representative_coverage: dict[str, int],
    warmups: int,
    repeats: int,
    max_cv_percent: float,
    max_reserved_vram_bytes: int,
    min_wall_reduction_percent: float,
) -> dict[str, Any]:
    expected_baseline, expected_candidates = validate_adaptive_overlap_matrix(
        expected_configurations, baseline_overlap
    )
    actual_configurations = [
        BenchmarkConfiguration.from_record(report["configuration"])
        for report in reports
    ]
    actual_baseline, actual_candidates = validate_adaptive_overlap_matrix(
        actual_configurations, baseline_overlap
    )
    expected_fingerprints = Counter(
        json_fingerprint(configuration.record())
        for configuration in expected_configurations
    )
    actual_fingerprints = Counter(
        json_fingerprint(configuration.record())
        for configuration in actual_configurations
    )
    candidate_comparisons = [
        item for item in comparison if not item["is_adaptive_overlap_baseline"]
    ]
    checks = {
        "exact_requested_adaptive_overlap_matrix": actual_fingerprints
        == expected_fingerprints,
        "exactly_one_adaptive_overlap_baseline": actual_baseline == expected_baseline,
        "all_nonbaseline_overlaps_are_candidates": Counter(actual_candidates)
        == Counter(expected_candidates),
        **_shared_protocol_checks(
            reports,
            expected_configuration_count=len(expected_configurations),
            expected_routes=expected_routes,
            expected_pages=expected_pages,
            representative_coverage=representative_coverage,
            warmups=warmups,
            repeats=repeats,
            max_cv_percent=max_cv_percent,
            max_reserved_vram_bytes=max_reserved_vram_bytes,
            min_wall_reduction_percent=min_wall_reduction_percent,
        ),
        "all_candidate_wall_reductions_at_least_minimum": (
            len(candidate_comparisons) == len(expected_candidates)
            and all(
                item["meets_minimum_wall_reduction"] is True
                for item in candidate_comparisons
            )
        ),
    }
    normalized_matrix = [
        expected_baseline.record(),
        *[
            configuration.record()
            for configuration in sorted(
                expected_candidates, key=lambda configuration: configuration.overlap
            )
        ],
    ]
    return {
        "mode": "adaptive_overlap_candidates_vs_baseline",
        "valid_for_performance_decision": all(checks.values()),
        "checks": checks,
        "required_protocol": {
            "configurations": normalized_matrix,
            "baseline_overlap": baseline_overlap,
            "adaptive_tile_candidates": list(expected_baseline.tile_candidates),
            "route_counts": PRODUCTION_ROUTE_COUNTS,
            "page_count": PRODUCTION_PAGE_COUNT,
            "coverage": {
                "exact_threshold_at_least": 2,
                "requires_grayscale": True,
                "requires_color": True,
                "requires_odd_dimension": True,
            },
            "minimum_warmups_per_configuration": PRODUCTION_MIN_WARMUPS,
            "measured_runs_per_configuration": PRODUCTION_REPEATS,
            "max_cv_percent": PRODUCTION_MAX_CV_PERCENT,
            "max_reserved_vram_bytes": PRODUCTION_MAX_RESERVED_VRAM_BYTES,
            "min_wall_reduction_percent": PRODUCTION_MIN_WALL_REDUCTION_PERCENT,
        },
        "effective_limits": {
            "max_cv_percent": max_cv_percent,
            "max_reserved_vram_bytes": max_reserved_vram_bytes,
            "min_wall_reduction_percent": min_wall_reduction_percent,
        },
    }
def _summary_run_reference(
    attempt: Path, benchmark_root: Path, result: dict[str, Any], *, reused: bool
) -> dict[str, Any]:
    job = dict(result["job"])
    pages = dict(result["pages"])
    job["benchmark_path"] = safe_relative(
        resolve_owned_relative(attempt, job["path"]), benchmark_root
    )
    pages["benchmark_path"] = safe_relative(
        resolve_owned_relative(attempt, pages["path"]), benchmark_root
    )
    return {
        "role": result["role"],
        "index": result["index"],
        "parent_session_id": result["parent_session_id"],
        "pair_id": result["pair_id"],
        "configuration": result["configuration"],
        "attempt": safe_relative(attempt, benchmark_root),
        "result": safe_relative(attempt / "result.json", benchmark_root),
        "attestation": {
            "child_spec": {
                "benchmark_path": safe_relative(attempt / "spec.json", benchmark_root),
                "sha256": sha256_file(attempt / "spec.json"),
            },
            "child_config": {
                "benchmark_path": safe_relative(
                    attempt / "config.toml", benchmark_root
                ),
                "sha256": sha256_file(attempt / "config.toml"),
            },
            "attempt_status": {
                "benchmark_path": safe_relative(
                    attempt / "attempt.json", benchmark_root
                ),
                "sha256": sha256_file(attempt / "attempt.json"),
            },
            "completion_marker": {
                "benchmark_path": safe_relative(
                    attempt / "completion.json", benchmark_root
                ),
                "sha256": sha256_file(attempt / "completion.json"),
            },
        },
        "fingerprint": result["fingerprint"],
        "reused": reused,
        "input_snapshot_unchanged": (
            result["input_snapshot_before"] == result["input_snapshot_after"]
        ),
        "isolation": result["isolation"],
        "owned_roots": result["owned_roots"],
        "pipeline_summary": result["pipeline_summary"],
        "job": job,
        "pages": pages,
        "jxl_outputs": result["jxl_outputs"],
        "telemetry": result["telemetry"],
    }


def _prepare_benchmark_root(
    root: Path,
    *,
    resume: bool,
    fingerprint: str,
    cache_root: Path,
) -> dict[str, Any] | None:
    summary_path = root / "benchmark_summary.json"
    if not root.exists():
        root.mkdir(parents=True)
        return None
    if not root.is_dir():
        raise ValueError(f"Output root is not a directory: {root}")
    entries = [path for path in root.iterdir() if path.name != SESSION_LOCK_NAME]
    if not entries:
        return None
    if not resume:
        raise ValueError(f"Output root already exists and is non-empty: {root}")
    if summary_path.is_symlink() or not summary_path.is_file():
        raise ValueError(
            f"Non-empty output root has no resumable benchmark_summary.json: {root}"
        )
    previous = read_json(summary_path)
    if (
        previous.get("schema_version") != SCHEMA_VERSION
        or previous.get("kind") != SUMMARY_KIND
        or previous.get("fingerprint") != fingerprint
        or Path(previous.get("output_root", "")).resolve() != root
        or Path(previous.get("cache_root", "")).resolve() != cache_root
    ):
        raise ValueError(
            "Existing benchmark root does not match this invocation fingerprint"
        )
    return previous


def build_execution_schedule(
    configurations: list[BenchmarkConfiguration], *, warmups: int, repeats: int
) -> list[tuple[str, int, BenchmarkConfiguration]]:
    schedule: list[tuple[str, int, BenchmarkConfiguration]] = []
    for role, count in (("warmup", warmups), ("repeat", repeats)):
        for index in range(1, count + 1):
            ordered = (
                configurations
                if role == "warmup" or index % 2
                else list(reversed(configurations))
            )
            schedule.extend((role, index, configuration) for configuration in ordered)
    return schedule


def coherent_reusable_plan(
    candidates: dict[
        tuple[str, int, BenchmarkConfiguration], tuple[Path, dict[str, Any]]
    ],
    schedule: list[tuple[str, int, BenchmarkConfiguration]],
) -> tuple[
    dict[tuple[str, int, BenchmarkConfiguration], tuple[Path, dict[str, Any]]],
    set[str],
]:
    sessions = {
        str(result.get("parent_session_id"))
        for _attempt, result in candidates.values()
        if result.get("parent_session_id")
    }
    if len(candidates) == len(schedule) and len(sessions) == 1:
        return candidates, sessions
    return {}, sessions


def run_benchmark(
    args: argparse.Namespace, configurations: list[BenchmarkConfiguration]
) -> int:
    manifest_path = args.manifest.expanduser().resolve()
    input_root = (
        args.input_root.expanduser().resolve()
        if args.input_root is not None
        else (manifest_path.parent / "inputs").resolve()
    )
    output_root = args.output_root.expanduser().resolve()
    cache_root = (
        args.cache_root.expanduser().resolve()
        if args.cache_root is not None
        else (output_root / "caches").resolve()
    )
    if paths_overlap(input_root, output_root) or paths_overlap(input_root, cache_root):
        raise ValueError(
            "Output/cache roots must not overlap the read-only representative input"
        )
    with BenchmarkSessionLease(output_root):
        return _run_benchmark_locked(args, configurations)


def _run_benchmark_locked(
    args: argparse.Namespace, configurations: list[BenchmarkConfiguration]
) -> int:
    config_path = args.config.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    input_root = (
        args.input_root.expanduser().resolve()
        if args.input_root is not None
        else (manifest_path.parent / "inputs").resolve()
    )
    output_root = args.output_root.expanduser().resolve()
    cache_root = (
        args.cache_root.expanduser().resolve()
        if args.cache_root is not None
        else (output_root / "caches").resolve()
    )
    if paths_overlap(input_root, output_root) or paths_overlap(input_root, cache_root):
        raise ValueError(
            "Output/cache roots must not overlap the read-only representative input"
        )
    base = load_config(config_path)
    validate_production_semantics(base)
    adaptive_baseline_overlap: int | None = None
    if args.adaptive_only:
        adaptive_baseline_overlap = (
            int(args.baseline_overlap)
            if args.baseline_overlap is not None
            else int(base.processing.hat_overlap)
        )
        validate_adaptive_overlap_matrix(configurations, adaptive_baseline_overlap)
    manifest, source_snapshot, expected_routes, representative_coverage = (
        load_representative_manifest(
            manifest_path,
            input_root,
            base.processing.real_hat_sharper_min_short_edge,
        )
    )
    models = resolve_real_hat_models(base)
    project_root = Path(__file__).resolve().parents[1]
    execution_schedule = build_execution_schedule(
        configurations, warmups=args.warmups, repeats=args.repeats
    )
    identity = {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "input_root": str(input_root),
        "input_snapshot": source_snapshot,
        "manifest_selected_counts": manifest.get("selected_counts"),
        "expected_route_counts": expected_routes,
        "representative_coverage": representative_coverage,
        "models": models,
        "source_inventory": source_inventory(project_root),
        "runtime": runtime_inventory(),
        "matrix": [configuration.record() for configuration in configurations],
        "execution_schedule": [
            {
                "role": role,
                "index": index,
                "configuration": configuration.record(),
            }
            for role, index, configuration in execution_schedule
        ],
        "warmups": args.warmups,
        "repeats": args.repeats,
        "timeout_seconds": args.timeout_seconds,
        "max_cv_percent": args.max_cv_percent,
        "max_reserved_vram_gib": args.max_reserved_vram_gib,
        "min_wall_reduction_percent": args.min_wall_reduction_percent,
    }
    if adaptive_baseline_overlap is not None:
        identity["qualification_mode"] = "adaptive_overlap_candidates_vs_baseline"
        identity["baseline_overlap"] = adaptive_baseline_overlap
    fingerprint = json_fingerprint(identity)
    previous = _prepare_benchmark_root(
        output_root,
        resume=args.resume,
        fingerprint=fingerprint,
        cache_root=cache_root,
    )
    if cache_root.exists() and any(cache_root.iterdir()) and previous is None:
        raise ValueError(f"Cache root already exists and is non-empty: {cache_root}")
    cache_root.mkdir(parents=True, exist_ok=True)
    parent_session_id = uuid.uuid4().hex
    reusable_candidates: dict[
        tuple[str, int, BenchmarkConfiguration], tuple[Path, dict[str, Any]]
    ] = {}
    if previous is not None:
        for role, index, configuration in execution_schedule:
            label = configuration.label
            slot_root = output_root / "runs" / label / f"{role}-{index:02d}"
            cache_slot_root = cache_root / label / f"{role}-{index:02d}"
            run_fingerprint = _run_fingerprint(fingerprint, configuration, role, index)
            reusable = find_reusable_result(
                slot_root,
                fingerprint=run_fingerprint,
                expected_input_root=input_root,
                expected_input=source_snapshot,
                expected_models=models,
                expected_configuration=configuration,
                expected_role=role,
                expected_index=index,
                cache_slot_root=cache_slot_root,
            )
            if reusable is not None:
                reusable_candidates[(role, index, configuration)] = reusable
    reusable_plan, reusable_sessions = coherent_reusable_plan(
        reusable_candidates, execution_schedule
    )
    reuse_complete_protocol = bool(reusable_plan)
    summary_path = output_root / "benchmark_summary.json"
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": SUMMARY_KIND,
        "status": "running",
        "started_at": previous.get("started_at", utc_now()) if previous else utc_now(),
        "updated_at": utc_now(),
        "fingerprint": fingerprint,
        "output_root": str(output_root),
        "cache_root": str(cache_root),
        "parent_session_id": parent_session_id,
        "resume_evidence": {
            "valid_slots_found": len(reusable_candidates),
            "required_slots": len(execution_schedule),
            "source_session_ids": sorted(reusable_sessions),
            "reused_complete_single_session_protocol": reuse_complete_protocol,
            "partial_protocol_policy": (
                "retain old attempts but rerun the entire protocol in one new session"
            ),
        },
        "identity": identity,
        "methodology": {
            "process_isolation": "one fresh Python child per warmup and measured batch",
            "input_policy": "content-hashed read-only representative input; mirror mode only",
            "output_policy": "fresh output, metrics, and cache directories per child",
            "wall_metric": "RunSummary.wall_seconds for the complete pipeline batch",
            "determinism": "exact relative JXL file set and SHA-256 equality across measured repeats",
            "tile_selection": ADAPTIVE_SELECTION_FORMULA,
            "production_decision_protocol": (
                (
                    f"adaptive {list(configurations[0].tile_candidates)}, overlap "
                    f"baseline {adaptive_baseline_overlap} versus all other requested "
                    "overlaps; 30 pages routed normal=9/sharper=21; at least one "
                    "warmup plus "
                    "three fresh measured children per configuration"
                )
                if adaptive_baseline_overlap is not None
                else (
                    "fixed [256] versus adaptive [256, 320], overlap 32; 30 pages "
                    "routed normal=9/sharper=21; at least one warmup plus three fresh "
                    "measured children"
                )
            ),
            "execution_order": (
                "Both configurations warm before measurement; measured rounds alternate "
                "configuration order to reduce thermal and clock-order bias."
            ),
            "resume": "only complete child results with revalidated config/input/job/pages/JXL hashes",
            "phase_percentages": (
                "Each interval union is divided by total measured batch wall; overlapping phase "
                "percentages are intentionally not additive. Service percentages are cumulative work."
            ),
        },
        "runs": [],
        "configurations": [],
    }
    write_json(summary_path, summary)

    try:
        run_references: list[dict[str, Any]] = []
        role_results_by_configuration: dict[
            BenchmarkConfiguration, dict[str, list[dict[str, Any]]]
        ] = {
            configuration: {"warmup": [], "repeat": []}
            for configuration in configurations
        }
        for role, index, configuration in execution_schedule:
            label = configuration.label
            slot_root = output_root / "runs" / label / f"{role}-{index:02d}"
            cache_slot_root = cache_root / label / f"{role}-{index:02d}"
            run_fingerprint = _run_fingerprint(fingerprint, configuration, role, index)
            reusable = reusable_plan.get((role, index, configuration))
            if reusable is not None:
                attempt, result = reusable
                print(f"reuse {label} {role}-{index:02d}: {attempt.name}", flush=True)
                reused = True
            else:
                attempt = next_attempt_root(slot_root)
                child_output = attempt / "output"
                child_metrics = attempt / "metrics"
                child_cache = cache_slot_root / attempt.name
                validate_isolated_roots(
                    input_root,
                    child_output,
                    child_metrics,
                    child_cache,
                    require_fresh=True,
                )
                child_config = render_child_config(
                    base,
                    input_root=input_root,
                    output_root=child_output,
                    configuration=configuration,
                )
                child_config_path = attempt / "config.toml"
                child_config_path.write_text(
                    child_config, encoding="utf-8", newline="\n"
                )
                spec = {
                    "schema_version": SCHEMA_VERSION,
                    "kind": CHILD_SPEC_KIND,
                    "fingerprint": run_fingerprint,
                    "attempt_root": str(attempt.resolve()),
                    "role": role,
                    "index": index,
                    "parent_session_id": parent_session_id,
                    "pair_id": f"{role}-{index}",
                    "configuration": configuration.record(),
                    "input_root": str(input_root),
                    "input_snapshot": source_snapshot,
                    "models": models,
                    "output_root": str(child_output.resolve()),
                    "metrics_root": str(child_metrics.resolve()),
                    "cache_root": str(child_cache.resolve()),
                    "config_path": str(child_config_path.resolve()),
                    "config_sha256": sha256_file(child_config_path),
                    "result_path": str((attempt / "result.json").resolve()),
                }
                spec_path = attempt / "spec.json"
                write_json(spec_path, spec)
                print(f"run {label} {role}-{index:02d}: {attempt.name}", flush=True)
                process_report = run_child_process(
                    spec_path, attempt / "child.log", args.timeout_seconds
                )
                result = _valid_complete_result(
                    attempt,
                    fingerprint=run_fingerprint,
                    expected_input_root=input_root,
                    expected_input=source_snapshot,
                    expected_models=models,
                    expected_configuration=configuration,
                    expected_role=role,
                    expected_index=index,
                    expected_cache_root=child_cache,
                )
                if process_report["returncode"] != 0 or result is None:
                    raise RuntimeError(
                        f"Child failed or left incomplete evidence: {attempt}; "
                        f"returncode={process_report['returncode']}"
                    )
                reused = False
            reference = _summary_run_reference(
                attempt, output_root, result, reused=reused
            )
            role_results_by_configuration[configuration][role].append(reference)
            run_references.append(reference)
            summary["runs"] = run_references
            summary["updated_at"] = utc_now()
            write_json(summary_path, summary)
            if input_snapshot(input_root) != source_snapshot:
                raise RuntimeError("Representative input changed between child runs")
            if sha256_file(config_path) != identity["config_sha256"]:
                raise RuntimeError("Base configuration changed between child runs")
            if sha256_file(manifest_path) != identity["manifest_sha256"]:
                raise RuntimeError("Representative manifest changed between child runs")
            if source_inventory(project_root) != identity["source_inventory"]:
                raise RuntimeError("Benchmark source code changed between child runs")

        configuration_reports = []
        for configuration in configurations:
            role_results = role_results_by_configuration[configuration]
            configuration_reports.append(
                aggregate_configuration(
                    configuration=configuration,
                    warmups=role_results["warmup"],
                    repeats=role_results["repeat"],
                    expected_routes=expected_routes,
                    expected_pages=len(source_snapshot),
                    max_cv_percent=args.max_cv_percent,
                    max_reserved_vram_bytes=round(args.max_reserved_vram_gib * 1024**3),
                )
            )
            summary["configurations"] = configuration_reports
            summary["updated_at"] = utc_now()
            write_json(summary_path, summary)

        if adaptive_baseline_overlap is not None:
            comparison = build_adaptive_overlap_comparison(
                configuration_reports,
                baseline_overlap=adaptive_baseline_overlap,
                min_wall_reduction_percent=args.min_wall_reduction_percent,
            )
            production_gate = adaptive_overlap_qualification(
                configuration_reports,
                comparison,
                expected_configurations=configurations,
                baseline_overlap=adaptive_baseline_overlap,
                expected_routes=expected_routes,
                expected_pages=len(source_snapshot),
                representative_coverage=representative_coverage,
                warmups=args.warmups,
                repeats=args.repeats,
                max_cv_percent=args.max_cv_percent,
                max_reserved_vram_bytes=round(args.max_reserved_vram_gib * 1024**3),
                min_wall_reduction_percent=args.min_wall_reduction_percent,
            )
            qualified = production_gate["valid_for_performance_decision"]
        else:
            comparison = build_comparison(
                configuration_reports,
                min_wall_reduction_percent=args.min_wall_reduction_percent,
            )
            production_gate = production_qualification(
                configuration_reports,
                comparison,
                expected_routes=expected_routes,
                expected_pages=len(source_snapshot),
                representative_coverage=representative_coverage,
                warmups=args.warmups,
                repeats=args.repeats,
                max_cv_percent=args.max_cv_percent,
                max_reserved_vram_bytes=round(args.max_reserved_vram_gib * 1024**3),
                min_wall_reduction_percent=args.min_wall_reduction_percent,
            )
            qualified = production_gate["valid_for_production_decision"]
        summary.update(
            {
                "status": "complete",
                "finished_at": utc_now(),
                "updated_at": utc_now(),
                "configurations": configuration_reports,
                "comparison": comparison,
                "production_qualification": production_gate,
                "valid_for_performance_decision": qualified,
            }
        )
        write_json(summary_path, summary)
        print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
        return 0 if qualified else 2
    except BaseException as exc:
        summary.update(
            {
                "status": "interrupted"
                if isinstance(exc, KeyboardInterrupt)
                else "error",
                "finished_at": utc_now(),
                "updated_at": utc_now(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_json(summary_path, summary)
        raise


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == CHILD_FLAG:
        raise SystemExit(child_main(Path(sys.argv[2])))
    parser = build_parser()
    args = parser.parse_args()
    configurations = validate_arguments(parser, args)
    try:
        code = run_benchmark(args, configurations)
    except KeyboardInterrupt:
        print(
            "\nInterrupted; completed evidence is retained for --resume.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as exc:
        print(f"Fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
