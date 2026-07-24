from dataclasses import replace
import os
from pathlib import Path

import pytest

from waifuhat2x.state import StateStore, file_sha256


def test_changed_pipeline_invalidates_state(tmp_path: Path) -> None:
    source = tmp_path / "input.png"
    source.write_bytes(b"pixels")
    output = tmp_path / "output.jxl"
    output.write_bytes(b"jxl")
    state = StateStore(tmp_path / "state.json")
    first = state.fingerprint(source, tmp_path, "model-hash", "pipeline-a")
    state.update("input.png", first, output)
    assert state.matches("input.png", first, output)
    assert not state.matches("input.png", replace(first, pipeline_signature="pipeline-b"), output)


def test_changed_content_with_same_metadata_invalidates_state(tmp_path: Path) -> None:
    source = tmp_path / "input.png"
    source.write_bytes(b"pixels-a")
    output = tmp_path / "output.jxl"
    output.write_bytes(b"jxl-good")
    state = StateStore(tmp_path / "state.json")
    first = state.fingerprint(source, tmp_path, "model-hash", "pipeline")
    state.update("input.png", first, output)

    source_stat = source.stat()
    source.write_bytes(b"pixels-b")
    os.utime(source, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
    changed = state.fingerprint(source, tmp_path, "model-hash", "pipeline")
    assert not state.matches("input.png", changed, output)


def test_changed_output_invalidates_state(tmp_path: Path) -> None:
    source = tmp_path / "input.png"
    source.write_bytes(b"pixels")
    output = tmp_path / "output.jxl"
    output.write_bytes(b"jxl-good")
    state = StateStore(tmp_path / "state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model-hash", "pipeline")
    state.update("input.png", fingerprint, output)

    output.write_bytes(b"jxl-evil")
    assert not state.matches("input.png", fingerprint, output)


def test_output_ready_remains_bound_to_the_verified_candidate(tmp_path: Path) -> None:
    source = tmp_path / "input.png"
    source.write_bytes(b"pixels")
    candidate = tmp_path / ".output.jxl.transaction.part"
    candidate.write_bytes(b"verified-candidate")
    output = tmp_path / "output.jxl"
    output.write_bytes(b"different-valid-jxl")
    state = StateStore(tmp_path / "state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model-hash", "pipeline")
    state.prepare_replace(
        "input.png",
        fingerprint,
        output.name,
        candidate.name,
        {"action": "sr", "output_width": 1600, "output_height": 2400},
    )
    state.mark_encoded(
        "input.png",
        candidate,
        candidate.stat().st_size,
        file_sha256(candidate),
    )

    with pytest.raises(RuntimeError, match="verified candidate"):
        state.mark_output_ready("input.png", output)

    record = state.record("input.png")
    assert record is not None
    assert record["phase"] == "encoded"
