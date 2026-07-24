from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.build_representative_manifest import _worklist_sources
from scripts.build_blind_boundary_rois import (
    blind_left_is_a,
    boundary_hits,
    configuration_tiles_for_page,
    overlap_fraction,
    validate_annotation_inventory,
)
import scripts.benchmark_manifest_eager as benchmark_manifest_eager
from scripts.benchmark_manifest_eager import (
    build_engine,
    choose_adaptive_tile,
    estimate_tile_work,
    fixed_engine_tile,
    load_manifest,
    metric,
)


def _write_worklist(path: Path, count: int, sources: list[str]) -> None:
    lines = [
        json.dumps({"type": "waifuhat2x-worklist", "count": count}),
        *(json.dumps({"source": source}) for source in sources),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_worklist_sources_deduplicates_repeated_lines(tmp_path: Path) -> None:
    worklist = tmp_path / "worklist.jsonl"
    _write_worklist(worklist, 2, ["a/001.jpg", "a/001.jpg", "b/002.webp"])

    sources, stats = _worklist_sources(worklist, tmp_path.resolve())

    assert [path.relative_to(tmp_path).as_posix() for path in sources] == [
        "a/001.jpg",
        "b/002.webp",
    ]
    assert stats == {
        "declared_count": 2,
        "item_lines": 3,
        "unique_sources": 2,
        "duplicate_lines": 1,
    }


def test_worklist_sources_rejects_header_count_mismatch(tmp_path: Path) -> None:
    worklist = tmp_path / "worklist.jsonl"
    _write_worklist(worklist, 3, ["a/001.jpg", "b/002.webp"])

    with pytest.raises(ValueError, match="declares 3 items.*2 unique"):
        _worklist_sources(worklist, tmp_path.resolve())


def test_manifest_loader_preserves_explicit_page_order(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "real_hat_representative_manifest",
                "pages": [{"index": 1}, {"index": 2}, {"index": 3}],
            }
        ),
        encoding="utf-8",
    )

    _payload, pages = load_manifest(manifest, [3, 1])

    assert [page["index"] for page in pages] == [3, 1]


def test_manifest_loader_rejects_duplicate_page_order(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"kind": "real_hat_representative_manifest", "pages": [{"index": 1}]}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not contain duplicates"):
        load_manifest(manifest, [1, 1])


def test_metric_uses_population_cv() -> None:
    result = metric([1.0, 2.0, 3.0])

    assert result["median"] == 2.0
    assert result["cv_percent"] == pytest.approx(40.8248290463863)


def test_estimate_tile_work_uses_requested_formula() -> None:
    estimate = estimate_tile_work(width=1000, height=1500, tile=256, overlap=32)

    assert estimate == {
        "tile": 256,
        "tiles_x": 4,
        "tiles_y": 6,
        "tile_count": 24,
        "expanded_edge": 320,
        "expanded_tile_area": 102400,
        "estimated_work": 2457600,
    }


def test_choose_adaptive_tile_selects_minimum_estimated_work() -> None:
    selected, estimates = choose_adaptive_tile(
        width=1000, height=1500, candidates=[320, 256, 320], overlap=32
    )

    assert [item["tile"] for item in estimates] == [256, 320]
    assert selected == 256
    assert estimates[0]["estimated_work"] < estimates[1]["estimated_work"]


def test_choose_adaptive_tile_breaks_work_tie_toward_smaller_tile() -> None:
    selected, _estimates = choose_adaptive_tile(
        width=3, height=3, candidates=[4, 2], overlap=0
    )

    # t2 uses four 2x2 tiles and t4 uses one 4x4 tile: both estimate 16 units.
    assert selected == 2


def test_manifest_benchmark_passes_all_hat_tile_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_engine(**kwargs: object) -> object:
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(benchmark_manifest_eager, "UpscaleEngine", fake_engine)

    result = build_engine([256, 320], overlap=24, collect_gpu_timing=True)

    assert type(result) is object
    assert observed["tile"] == 256
    assert observed["hat_tile"] == 256
    assert observed["hat_tile_candidates"] == (256, 320)
    assert observed["overlap"] == 24
    assert observed["hat_overlap"] == 24
    assert observed["collect_gpu_timing"] is True


def test_fixed_engine_tile_restores_adaptive_candidates_after_failure() -> None:
    engine = SimpleNamespace(
        tile=256,
        hat_tile=256,
        hat_tile_candidates=(256, 320),
    )

    with pytest.raises(RuntimeError, match="warmup failed"):
        with fixed_engine_tile(engine, 320):
            assert engine.tile == 320
            assert engine.hat_tile == 320
            assert engine.hat_tile_candidates == (320,)
            raise RuntimeError("warmup failed")

    assert engine.tile == 256
    assert engine.hat_tile == 256
    assert engine.hat_tile_candidates == (256, 320)


@pytest.mark.parametrize(
    ("width", "height", "tile", "overlap"),
    [(0, 10, 8, 1), (10, 0, 8, 1), (10, 10, 0, 0), (10, 10, 8, -1), (10, 10, 8, 8)],
)
def test_estimate_tile_work_rejects_invalid_inputs(
    width: int, height: int, tile: int, overlap: int
) -> None:
    with pytest.raises(ValueError):
        estimate_tile_work(width, height, tile, overlap)


def test_boundary_hits_reports_only_strictly_crossed_boundaries() -> None:
    assert boundary_hits((240, 300, 96, 64), (256, 320)) == [
        "t256:x256",
        "t320:x320",
        "t320:y320",
    ]


def test_boundary_hits_deduplicates_equal_actual_tiles() -> None:
    assert boundary_hits((240, 0, 32, 32), (256, 256)) == ["t256:x256"]


def test_configuration_tiles_for_page_uses_each_summary_plan() -> None:
    plans = ({4: 320, 18: 320}, {4: 320, 18: 256})

    assert configuration_tiles_for_page(4, plans, (None, None)) == (320, 320)
    assert configuration_tiles_for_page(18, plans, (None, None)) == (320, 256)


def test_configuration_tiles_for_page_rejects_declared_tile_drift() -> None:
    with pytest.raises(ValueError, match="uses tile 256, not declared tile 320"):
        configuration_tiles_for_page(18, ({18: 320}, {18: 256}), (320, 320))


def test_blind_assignment_is_deterministic() -> None:
    assert blind_left_is_a("seed", "text-01") == blind_left_is_a("seed", "text-01")


def test_overlap_fraction_uses_smaller_roi_area() -> None:
    assert overlap_fraction((0, 0, 10, 10), (5, 5, 5, 5)) == 1.0
    assert overlap_fraction((0, 0, 10, 10), (10, 10, 5, 5)) == 0.0


def test_annotation_inventory_reports_distribution() -> None:
    annotations = {
        "schema_version": 1,
        "rois": [
            {"id": "text-01", "page_index": 4, "category": "text", "box": [0, 0, 8, 8]},
            {
                "id": "screentone-01",
                "page_index": 4,
                "category": "screentone",
                "box": [20, 0, 8, 8],
            },
            {
                "id": "diagonal-01",
                "page_index": 5,
                "category": "diagonal",
                "box": [0, 0, 8, 8],
            },
        ],
    }

    inventory = validate_annotation_inventory(annotations, 1, {4, 5})

    assert inventory["counts"] == {"text": 1, "screentone": 1, "diagonal": 1}
    assert inventory["per_page_per_category"]["4"] == {
        "text": 1,
        "screentone": 1,
        "diagonal": 0,
    }


def test_annotation_inventory_rejects_substantial_overlap() -> None:
    annotations = {
        "schema_version": 1,
        "rois": [
            {"id": "text-01", "page_index": 4, "category": "text", "box": [0, 0, 8, 8]},
            {"id": "text-02", "page_index": 4, "category": "text", "box": [1, 1, 8, 8]},
            {
                "id": "screentone-01",
                "page_index": 4,
                "category": "screentone",
                "box": [20, 0, 8, 8],
            },
            {
                "id": "diagonal-01",
                "page_index": 4,
                "category": "diagonal",
                "box": [40, 0, 8, 8],
            },
        ],
    }

    with pytest.raises(ValueError, match="substantially overlaps"):
        validate_annotation_inventory(annotations, 1, {4})
