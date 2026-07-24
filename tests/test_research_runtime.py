from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from PIL import Image
import pytest

from scripts import research_runtime as research


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_page(path: Path, *, size: tuple[int, int], value: int = 0) -> dict[str, Any]:
    image = Image.new("RGB", size, (value, value, value))
    image.save(path)
    return {
        "copied_path": f"inputs/{path.name}",
        "copied_sha256": _sha256(path),
        "width": size[0],
        "height": size[1],
        "file_bytes": path.stat().st_size,
    }


def _manifest(tmp_path: Path) -> Path:
    root = tmp_path / "representative"
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    normal = _write_page(inputs / "normal.png", size=(8, 10))
    sharper = _write_page(inputs / "sharper.png", size=(10, 12))
    pages = [
        {
            "index": 1,
            "route": "normal",
            "grayscale": False,
            "odd_dimension": False,
            "source_mode": "RGB",
            "source_format": "PNG",
            **normal,
        },
        {
            "index": 2,
            "route": "sharper",
            "grayscale": False,
            "odd_dimension": False,
            "source_mode": "RGB",
            "source_format": "PNG",
            **sharper,
        },
    ]
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "real_hat_representative_manifest",
                "pages": pages,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _outcome(
    root: Path, name: str, array: np.ndarray, *, tile: int = 256
) -> research.EagerOutcome:
    output = root / f"{name}.png"
    Image.fromarray(array).save(output)
    page = {
        "index": 1,
        "route": "normal",
        "selected_tile": tile,
        "upscale_wall_seconds": 1.0,
        "pixel_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "png_path": str(output),
        "png_sha256": _sha256(output),
    }
    return research.EagerOutcome(
        record={"candidate": name, "tile": tile},
        payload={"rounds": [{"loop_wall_seconds": 1.0, "pages": [page]}]},
    )


def test_load_manifest_uses_selected_copied_pages_only(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    inventory = research.load_manifest(manifest, [1, 2], threshold=10)

    assert inventory.input_root == manifest.parent / "inputs"
    assert [page.index for page in inventory.pages] == [1, 2]
    assert [page.route for page in inventory.pages] == ["normal", "sharper"]
    assert all("source" not in page.record() for page in inventory.pages)


def test_load_manifest_rejects_route_drift(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["pages"][0]["route"] = "sharper"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(research.ResearchError, match="route drifted"):
        research.load_manifest(manifest, [1], threshold=10)


def test_production_process_gate_detects_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = subprocess.CompletedProcess(
        ["ps"], 0, stdout="  321 python -m waifuhat2x --config config.toml\n", stderr=""
    )
    monkeypatch.setattr(research.subprocess, "run", lambda *_args, **_kwargs: completed)

    assert research.production_processes() == [
        {"pid": 321, "command": "python -m waifuhat2x --config config.toml"}
    ]
    with pytest.raises(research.ResearchError, match="Refusing GPU research"):
        research.require_idle_production()


def test_candidate_environment_clears_or_sets_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(research.BACKEND_ENVIRONMENT_KEY, "inherited")
    baseline, hipblaslt = research.candidate_definitions(["default", "hipblaslt"])

    assert research.actual_backend_environment(research.child_environment(baseline)) == {
        research.BACKEND_ENVIRONMENT_KEY: None
    }
    assert research.actual_backend_environment(research.child_environment(hipblaslt)) == {
        research.BACKEND_ENVIRONMENT_KEY: "1"
    }


def test_interleaved_pair_schedule_is_ab_ba_ab() -> None:
    candidates = research.candidate_definitions(["default", "hipblaslt"])

    schedule = research.interleaved_pair_schedule(candidates, 3)

    assert [(item["pair"], item["candidate"]) for item in schedule] == [
        (1, "default"),
        (1, "hipblaslt"),
        (2, "hipblaslt"),
        (2, "default"),
        (3, "default"),
        (3, "hipblaslt"),
    ]


def test_png_comparison_accepts_single_lsb_difference(tmp_path: Path) -> None:
    baseline = np.zeros((128, 128, 3), dtype=np.uint8)
    candidate = baseline.copy()
    candidate[0, 0, 0] = 1
    base_outcome = _outcome(tmp_path, "baseline", baseline)
    candidate_outcome = _outcome(tmp_path, "candidate", candidate)

    report = research.compare_png_sets(base_outcome, candidate_outcome, tmp_path)

    assert report["pixel_hash_equal"] is False
    assert report["max_abs_difference"] == 1
    assert report["p95_abs_difference"] == 0
    assert report["psnr_infinite"] is False
    assert float(report["psnr_db"]) >= 90.0


def test_micro_determinism_does_not_compare_different_tiles(tmp_path: Path) -> None:
    tile_256 = _outcome(tmp_path, "hipblaslt", np.zeros((8, 8, 3), dtype=np.uint8), tile=256)
    tile_320 = _outcome(tmp_path, "hipblaslt", np.ones((8, 8, 3), dtype=np.uint8), tile=320)

    assert research.eager_hash_determinism_by_tile([tile_256, tile_320]) == {
        256: True,
        320: True,
    }


def test_cold_defaults_to_default_backend_and_full_page_pairs(tmp_path: Path) -> None:
    parser = research.build_parser()

    args = parser.parse_args(["cold", "--output-root", str(tmp_path / "out")])

    assert args.candidates == ["default"]
    assert args.page_indexes == list(research.DEFAULT_COLD_INDEXES)
    assert args.warmups == 1
