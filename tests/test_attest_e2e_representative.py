from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from scripts.attest_e2e_representative import (
    inspect_manifest,
    page_route,
    verify_page_records,
)


def test_page_route_uses_explicit_real_hat_suffixes() -> None:
    assert page_route("Real-HAT-GAN-x4-normal") == "normal"
    assert page_route("Real-HAT-GAN-x4-sharper") == "sharper"
    assert page_route("other") is None


def test_page_records_are_bound_to_source_facts() -> None:
    expected = {
        "001.png": {
            "index": 1,
            "width": 999,
            "height": 1201,
            "short_edge": 999,
            "route": "normal",
            "grayscale": True,
        }
    }
    page = {
        "type": "waifuhat2x-page-metrics",
        "schema_version": 1,
        "index": 1,
        "source": "001.png",
        "status": "complete",
        "details": {
            "source_dimensions": [999, 1201],
            "source_short_edge": 999,
            "grayscale": True,
            "model_label": "Real-HAT-GAN-x4-normal",
            "model_checkpoint": "Real_HAT_GAN_SRx4.pth",
        },
    }

    report = verify_page_records([page], expected)

    assert report["per_source_facts_match"] is True
    assert report["route_counts"] == {"normal": 1}


def test_page_records_reject_swapped_route_with_unchanged_total() -> None:
    expected = {
        "001.png": {
            "index": 1,
            "width": 1000,
            "height": 1400,
            "short_edge": 1000,
            "route": "sharper",
            "grayscale": False,
        }
    }
    page = {
        "type": "waifuhat2x-page-metrics",
        "schema_version": 1,
        "index": 1,
        "source": "001.png",
        "status": "complete",
        "details": {
            "source_dimensions": [1000, 1400],
            "source_short_edge": 1000,
            "grayscale": False,
            "model_label": "Real-HAT-GAN-x4-normal",
            "model_checkpoint": "Real_HAT_GAN_SRx4.pth",
        },
    }

    with pytest.raises(ValueError, match="route"):
        verify_page_records([page], expected)


def test_manifest_inspection_rejects_a_non_isolated_copied_path(tmp_path: Path) -> None:
    outside = tmp_path / "production.png"
    Image.new("RGB", (8, 8), "white").save(outside)
    manifest_root = tmp_path / "representative"
    manifest_root.mkdir()
    manifest = manifest_root / "manifest.json"
    manifest.write_text(
        """
        {
          "schema_version": 1,
          "kind": "real_hat_representative_manifest",
          "coverage": {},
          "pages": [{
            "index": 1,
            "copied_path": "../production.png",
            "copied_sha256": "unused"
          }]
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not an isolated copied input"):
        inspect_manifest(manifest, 1000)
