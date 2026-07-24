from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from waifuhat2x.pipeline import (
    RunSummary,
    WorkItemSnapshot,
    _discover_files,
    _has_meaningful_transparency,
    _recover_replace_transactions,
    _write_worklist,
)
from waifuhat2x.state import StateStore, file_sha256


def _snapshot(path: Path) -> WorkItemSnapshot:
    stat = path.stat()
    return WorkItemSnapshot(
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        ctime_ns=stat.st_ctime_ns,
        device=stat.st_dev,
        inode=stat.st_ino,
    )


def test_fully_opaque_rgba_has_no_meaningful_transparency() -> None:
    image = Image.new("RGBA", (3, 2), (10, 20, 30, 255))

    assert not _has_meaningful_transparency(image)


def test_partially_transparent_rgba_has_meaningful_transparency() -> None:
    image = Image.new("RGBA", (3, 2), (10, 20, 30, 255))
    image.putpixel((1, 1), (10, 20, 30, 254))

    assert _has_meaningful_transparency(image)


class _RecordingEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, tuple[int, ...]]] = []

    def verify(self, path: Path, *dimensions: int) -> tuple[int, int]:
        self.calls.append((path, dimensions))
        return (1600, 2400)


def _prepare_adoption(
    root: Path,
) -> tuple[StateStore, Path, Path, str]:
    chapter = root / "chapter"
    chapter.mkdir(parents=True)
    source = chapter / "001.png"
    source.write_bytes(b"unchanged-source")
    destination = chapter / "001.jxl"
    destination.write_bytes(b"already-verified-jxl")
    key = "chapter/001.png"
    state = StateStore(root / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, root, "adopt-model", "pipeline-signature")
    state.prepare_adopt(
        key,
        fingerprint,
        "chapter/001.jxl",
        destination,
        file_sha256(destination),
        {
            "output_width": 1600,
            "output_height": 2400,
            "verified_decode": True,
            "aspect_checked": True,
        },
    )
    state.save()
    return state, source, destination, key


def test_prepare_adopt_recovery_safely_removes_source_and_clears_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "library"
    root.mkdir()
    state, source, destination, key = _prepare_adoption(root)
    # Exercise the persisted journal, not only the in-memory record.
    state = StateStore(state.path)
    summary = RunSummary()
    encoder = _RecordingEncoder()

    outcome = _recover_replace_transactions(
        state,
        root,
        encoder,  # type: ignore[arg-type]
        summary,
        pipeline_signature="pipeline-signature",
        model_hash_resolver=lambda _source: "unused",
    )

    assert outcome.blocked_keys == frozenset()
    assert outcome.external_jxl_keys == frozenset()
    assert not source.exists()
    assert destination.read_bytes() == b"already-verified-jxl"
    assert state.record(key) is None
    assert state.data == {}
    assert not state.path.exists()
    assert summary.replaced_sources == 1
    assert summary.existing_jxl_adopted == 1
    assert encoder.calls == [(destination, (1600, 2400))]


def test_prepare_adopt_recovery_retains_changed_source_and_discards_stale_authorization(
    tmp_path: Path,
) -> None:
    root = tmp_path / "library"
    root.mkdir()
    state, source, destination, key = _prepare_adoption(root)
    source.write_bytes(b"source-changed-after-adoption-was-prepared")
    summary = RunSummary()
    encoder = _RecordingEncoder()

    outcome = _recover_replace_transactions(
        state,
        root,
        encoder,  # type: ignore[arg-type]
        summary,
        pipeline_signature="pipeline-signature",
        model_hash_resolver=lambda _source: "unused",
    )

    assert outcome.blocked_keys == frozenset({key})
    assert outcome.external_jxl_keys == frozenset()
    assert source.exists()
    assert destination.read_bytes() == b"already-verified-jxl"
    assert state.record(key) is None
    assert not state.path.exists()
    assert summary.replaced_sources == 0
    assert summary.existing_jxl_adopted == 0
    assert summary.deferred == 1
    assert encoder.calls == []


def test_schema_2_worklist_writes_action_and_source_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "library"
    root.mkdir()
    source = root / "001.png"
    source.write_bytes(b"image-bytes")
    source_snapshot = _snapshot(source)
    worklist = root / ".waifuhat2x-worklist.jsonl"

    _write_worklist(
        worklist,
        root,
        {"001.png": source_snapshot},
        {
            "001.png": {
                "destination": "001.jxl",
                "action": "replace_existing_jxl",
            }
        },
        "pipeline-signature",
        _snapshot(root),
        1600,
    )

    header, row = [json.loads(line) for line in worklist.read_text(encoding="utf-8").splitlines()]
    assert header["type"] == "waifuhat2x-worklist"
    assert header["schema"] == 2
    assert header["count"] == 1
    assert row["source"] == "001.png"
    assert row["action"] == "replace_existing_jxl"
    assert row["source_snapshot"] == {
        "size": source_snapshot.size,
        "mtime_ns": source_snapshot.mtime_ns,
        "ctime_ns": source_snapshot.ctime_ns,
        "device": source_snapshot.device,
        "inode": source_snapshot.inode,
    }


def test_scandir_discovery_excludes_jxl_and_indexes_uppercase_companion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "library"
    chapter = root / "chapter"
    chapter.mkdir(parents=True)
    source = chapter / "Page.png"
    source.write_bytes(b"png")
    companion = chapter / "Page.JXL"
    companion.write_bytes(b"jxl")
    orphan = chapter / "Orphan.jxl"
    orphan.write_bytes(b"jxl")

    discovery = _discover_files(root, include_metadata=False)

    assert discovery.images == (source,)
    assert companion not in discovery.images
    assert orphan not in discovery.images
    assert discovery.jxl_by_key["chapter/page.jxl"] == companion
    assert discovery.jxl_by_key["chapter/orphan.jxl"] == orphan
    assert discovery.ignored == 2


def test_scandir_discovery_rejects_jxl_extension_case_collision(
    tmp_path: Path,
) -> None:
    root = tmp_path / "library"
    root.mkdir()
    (root / "page.jxl").write_bytes(b"first")
    (root / "page.JXL").write_bytes(b"second")
    names = [path.name for path in root.iterdir()]
    if len(names) != 2:
        pytest.skip("requires a case-sensitive test filesystem")

    with pytest.raises(ValueError, match="same normalized path"):
        _discover_files(root, include_metadata=False)
