from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from waifuhat2x.pipeline import (
    RunSummary,
    _commit_source_removal,
    _declared_bit_depth,
    _mirror_destination_is_admissible,
    _paths_overlap,
    _preflight_destinations,
    _recover_replace_transactions,
    _replace_destination_is_admissible,
    _skip_blocked_recovery_source,
)
from waifuhat2x.images import IMAGE_EXTENSIONS
from waifuhat2x.jxl import JxlEncoder, JxlStats
from waifuhat2x.state import StateStore, file_sha256


def test_duplicate_stem_is_rejected(tmp_path: Path) -> None:
    input_root = tmp_path / "in"
    output_root = tmp_path / "out"
    first = input_root / "001.png"
    second = input_root / "001.jpg"
    with pytest.raises(ValueError, match="same output"):
        _preflight_destinations([first, second], [], input_root, output_root, "jxl")


def test_generated_jxl_cannot_collide_with_metadata(tmp_path: Path) -> None:
    input_root = tmp_path / "in"
    output_root = tmp_path / "out"
    image = input_root / "001.png"
    metadata = input_root / "001.jxl"
    with pytest.raises(ValueError, match="same output"):
        _preflight_destinations([image], [metadata], input_root, output_root, "jxl")


def test_windows_mount_containment_is_case_insensitive() -> None:
    assert _paths_overlap(Path("/mnt/z/sample-input"), Path("/mnt/z/SAMPLE-input/output"))
    assert not _paths_overlap(Path("/mnt/z/sample-input"), Path("/mnt/z/sample-output"))


def test_image_input_allowlist_excludes_output_and_animated_formats() -> None:
    assert {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"} == IMAGE_EXTENSIONS
    assert ".jxl" not in IMAGE_EXTENSIONS
    assert ".gif" not in IMAGE_EXTENSIONS
    assert ".avif" not in IMAGE_EXTENSIONS


class FakeJxlEncoder:
    def verify(self, _path: Path, *_dimensions: int) -> tuple[int, int]:
        return (1600, 2400)

    def commit(self, temporary: Path, destination: Path, expected_sha256: str) -> None:
        assert file_sha256(temporary) == expected_sha256
        temporary.replace(destination)


class RecordingJxlEncoder(FakeJxlEncoder):
    def __init__(self) -> None:
        self.verify_calls: list[tuple[Path, tuple[int, ...]]] = []

    def verify(self, path: Path, *dimensions: int) -> tuple[int, int]:
        self.verify_calls.append((path, dimensions))
        return super().verify(path, *dimensions)


class FailingJxlEncoder(FakeJxlEncoder):
    def __init__(self, message: str) -> None:
        self.message = message

    def verify(self, _path: Path, *_dimensions: int) -> tuple[int, int]:
        raise RuntimeError(self.message)


class MutatingJxlEncoder(FakeJxlEncoder):
    def verify(self, path: Path, *_dimensions: int) -> tuple[int, int]:
        path.write_bytes(b"tampered-completion")
        return (1600, 2400)


class ReappearingSourceJxlEncoder(FakeJxlEncoder):
    def __init__(self, source: Path) -> None:
        self.source = source

    def verify(self, _path: Path, *_dimensions: int) -> tuple[int, int]:
        self.source.write_bytes(b"source-reappeared")
        return (1600, 2400)


def _recover(
    state: StateStore,
    root: Path,
    encoder: object,
    summary: RunSummary,
    *,
    pipeline_signature: str = "pipeline",
    model_hash: str = "model",
) -> set[str]:
    outcome = _recover_replace_transactions(
        state,
        root,
        encoder,  # type: ignore[arg-type]
        summary,
        pipeline_signature=pipeline_signature,
        model_hash_resolver=lambda _source: model_hash,
    )
    return set(outcome.blocked_keys)


def test_prepared_recovery_discards_uncommitted_part_and_keeps_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "001.png"
    source.write_bytes(b"source")
    temporary = tmp_path / ".001.jxl.transaction.part"
    temporary.write_bytes(b"uncommitted-part")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        "001.jxl",
        temporary.name,
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": None,
        },
    )
    state.save()

    summary = RunSummary()
    blocked = _recover(state, tmp_path, FakeJxlEncoder(), summary)

    assert blocked == set()
    assert source.read_bytes() == b"source"
    assert not temporary.exists()
    assert state.record("001.png") is not None
    assert summary.replaced_sources == 0


@pytest.mark.parametrize(
    "temporary_kind",
    ["source", "destination", "wrong_name", "wrong_directory", "empty_token", "missing"],
)
def test_recovery_rejects_unsafe_temporary_without_changing_library_files(
    tmp_path: Path,
    temporary_kind: str,
) -> None:
    source = tmp_path / "001.png"
    source.write_bytes(b"source")
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"destination")
    nested = tmp_path / "nested"
    nested.mkdir()
    candidates = {
        "source": source,
        "destination": destination,
        "wrong_name": tmp_path / "unrelated.bin",
        "wrong_directory": nested / ".001.jxl.transaction.part",
        "empty_token": tmp_path / ".001.jxl..part",
    }
    candidate = candidates.get(temporary_kind)
    if candidate is not None and candidate not in {source, destination}:
        candidate.write_bytes(b"unrelated")

    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "old-pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        candidate.relative_to(tmp_path).as_posix() if candidate is not None else "unused",
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": file_sha256(destination),
        },
    )
    if candidate is None:
        record = state.record("001.png")
        assert record is not None
        record.pop("temporary")
    state.save()

    with pytest.raises(RuntimeError, match="temporary"):
        _recover(
            state,
            tmp_path,
            FakeJxlEncoder(),
            RunSummary(),
            pipeline_signature="current-pipeline",
        )

    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"destination"
    if candidate is not None and candidate not in {source, destination}:
        assert candidate.read_bytes() == b"unrelated"
    assert state.record("001.png") is not None


def test_output_ready_recovery_never_accepts_source_as_destination(tmp_path: Path) -> None:
    source = tmp_path / "001.png"
    source.write_bytes(b"source-and-forged-output")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        source.name,
        ".001.png.transaction.part",
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": None,
        },
    )
    record = state.record("001.png")
    assert record is not None
    record["verified"] = True
    state.mark_output_ready("001.png", source)
    state.save()

    with pytest.raises(RuntimeError, match="source as a destination"):
        _recover(state, tmp_path, FakeJxlEncoder(), RunSummary())

    assert source.read_bytes() == b"source-and-forged-output"
    assert state.record("001.png") is not None


@pytest.mark.parametrize("source_name", ["metadata.txt", "already.jxl"])
def test_recovery_rejects_non_image_journal_source(
    tmp_path: Path,
    source_name: str,
) -> None:
    source = tmp_path / source_name
    source.write_bytes(b"non-image-source")
    destination = source.with_suffix(".jxl")
    if destination != source:
        destination.write_bytes(b"destination")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "pipeline")
    state.prepare_replace(
        source.name,
        fingerprint,
        destination.name,
        f".{destination.name}.transaction.part",
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": None,
        },
    )
    state.save()

    with pytest.raises(RuntimeError, match="not a supported image"):
        _recover(state, tmp_path, FakeJxlEncoder(), RunSummary())

    assert source.read_bytes() == b"non-image-source"
    if destination != source:
        assert destination.read_bytes() == b"destination"
    assert state.record(source.name) is not None


def test_prepared_recovery_replans_changed_source_in_same_run(tmp_path: Path) -> None:
    source = tmp_path / "001.png"
    source.write_bytes(b"source-v1")
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"old-jxl")
    temporary = tmp_path / ".001.jxl.transaction.part"
    temporary.write_bytes(b"unverified-candidate")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        temporary.name,
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": file_sha256(destination),
        },
    )
    state.save()
    source.write_bytes(b"source-v2")

    summary = RunSummary()
    blocked = _recover(state, tmp_path, FakeJxlEncoder(), summary)

    assert blocked == set()
    assert source.read_bytes() == b"source-v2"
    assert destination.read_bytes() == b"old-jxl"
    assert not temporary.exists()
    assert state.record("001.png") is None
    assert summary.deferred == 0


@pytest.mark.parametrize("source_missing", [False, True])
def test_encoded_replace_transaction_recovers_post_rename_crash(
    tmp_path: Path,
    source_missing: bool,
) -> None:
    source = tmp_path / "001.png"
    source.write_bytes(b"source")
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"verified-jxl")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        "001.jxl",
        ".001.jxl.transaction.part",
        {
            "phase": "prepared",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": None,
        },
    )
    state.mark_encoded("001.png", destination, destination.stat().st_size, file_sha256(destination))
    record = state.record("001.png")
    assert record is not None
    # Simulate the crash window after the verified candidate was atomically renamed.
    record["temporary"] = ".001.jxl.missing-candidate.part"
    state.save()
    if source_missing:
        source.unlink()

    summary = RunSummary()
    _recover(state, tmp_path, FakeJxlEncoder(), summary)
    assert not source.exists()
    assert destination.exists()
    assert state.record("001.png") is None
    assert not state.path.exists()
    assert summary.replaced_sources == int(not source_missing)


def test_png_ihdr_exposes_16_bit_rgb_even_if_pillow_mode_is_rgb() -> None:
    class FakePng:
        format = "PNG"
        mode = "RGB"
        tile = [("zip", (0, 0, 1, 1), 0, "RGB;16B")]

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = (13).to_bytes(4, "big") + b"IHDR" + (1).to_bytes(4, "big") * 2 + bytes([16])
    assert _declared_bit_depth(FakePng(), header + ihdr) == 16  # type: ignore[arg-type]


def test_prepared_transaction_never_adopts_unrecorded_destination(tmp_path: Path) -> None:
    source = tmp_path / "001.png"
    source.write_bytes(b"source")
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"unrelated-valid-looking-jxl")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        "001.jxl",
        ".001.jxl.transaction.part",
        {
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": None,
        },
    )
    with pytest.raises(RuntimeError, match="unrecorded final JXL"):
        _recover(state, tmp_path, FakeJxlEncoder(), RunSummary())
    assert source.exists()
    assert destination.exists()


def _prepared_transaction_with_missing_source(
    tmp_path: Path,
) -> tuple[StateStore, Path, Path, Path]:
    source = tmp_path / "001.png"
    source.write_bytes(b"source")
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"external-completion")
    temporary = tmp_path / ".001.jxl.transaction.part"
    temporary.write_bytes(b"unverified-candidate")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        temporary.name,
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": None,
        },
    )
    state.save()
    source.unlink()
    return state, source, destination, temporary


@pytest.mark.parametrize("previous_hash", [None, "0" * 64])
def test_prepared_missing_source_accepts_stable_exact_external_completion_without_data_loss(
    tmp_path: Path,
    previous_hash: str | None,
) -> None:
    state, source, destination, temporary = _prepared_transaction_with_missing_source(
        tmp_path
    )
    if previous_hash is not None:
        record = state.record("001.png")
        assert record is not None
        record["previous_output_sha256"] = previous_hash
        state.save()
    destination_hash = file_sha256(destination)
    summary = RunSummary()
    encoder = RecordingJxlEncoder()

    blocked = _recover(state, tmp_path, encoder, summary)

    assert blocked == {"001.png"}
    assert encoder.verify_calls == [(destination, (1600, 2400))]
    assert not source.exists()
    assert destination.exists()
    assert file_sha256(destination) == destination_hash
    assert temporary.read_bytes() == b"unverified-candidate"
    assert state.record("001.png") is None
    assert not state.path.exists()
    assert summary.external_jxl_recoveries == 1
    assert summary.deferred == 0
    assert summary.replaced_sources == 0


def test_prepared_missing_source_recovers_before_stale_signature_replanning(
    tmp_path: Path,
) -> None:
    state, source, destination, temporary = _prepared_transaction_with_missing_source(
        tmp_path
    )
    summary = RunSummary()

    blocked = _recover(
        state,
        tmp_path,
        RecordingJxlEncoder(),
        summary,
        pipeline_signature="changed-pipeline",
    )

    assert blocked == {"001.png"}
    assert not source.exists()
    assert destination.read_bytes() == b"external-completion"
    assert temporary.read_bytes() == b"unverified-candidate"
    assert state.record("001.png") is None
    assert summary.external_jxl_recoveries == 1
    assert summary.deferred == 0


def test_prepared_missing_source_with_changed_root_stays_blocked(tmp_path: Path) -> None:
    state, source, destination, temporary = _prepared_transaction_with_missing_source(
        tmp_path
    )
    record = state.record("001.png")
    assert record is not None
    record["source_root"] = str(tmp_path / "old-root")
    state.save()
    summary = RunSummary()

    with pytest.raises(RuntimeError, match="source is unavailable"):
        _recover(state, tmp_path, RecordingJxlEncoder(), summary)

    assert not source.exists()
    assert destination.read_bytes() == b"external-completion"
    assert temporary.read_bytes() == b"unverified-candidate"
    assert state.record("001.png") is not None
    assert summary.external_jxl_recoveries == 0
    assert summary.deferred == 0


def test_reappeared_source_after_external_recovery_is_deferred(tmp_path: Path) -> None:
    state, source, destination, temporary = _prepared_transaction_with_missing_source(
        tmp_path
    )
    summary = RunSummary()
    outcome = _recover_replace_transactions(
        state,
        tmp_path,
        ReappearingSourceJxlEncoder(source),  # type: ignore[arg-type]
        summary,
        pipeline_signature="pipeline",
        model_hash_resolver=lambda _source: "model",
    )

    assert source.read_bytes() == b"source-reappeared"
    assert destination.read_bytes() == b"external-completion"
    assert temporary.read_bytes() == b"unverified-candidate"
    assert state.record("001.png") is None
    assert outcome.blocked_keys == frozenset({"001.png"})
    assert outcome.external_jxl_keys == frozenset({"001.png"})
    assert summary.external_jxl_recoveries == 1
    assert summary.deferred == 0

    assert _skip_blocked_recovery_source("001.png", outcome, summary)
    assert summary.deferred == 1


@pytest.mark.parametrize(
    "message",
    [
        "djxl verification failed",
        "JXL dimension mismatch: expected 1600x2400, got 800x1200",
    ],
)
def test_prepared_missing_source_keeps_everything_when_external_jxl_fails_validation(
    tmp_path: Path,
    message: str,
) -> None:
    state, source, destination, temporary = _prepared_transaction_with_missing_source(
        tmp_path
    )
    destination_hash = file_sha256(destination)

    with pytest.raises(RuntimeError, match=message.split(":")[0]):
        _recover(state, tmp_path, FailingJxlEncoder(message), RunSummary())

    assert not source.exists()
    assert file_sha256(destination) == destination_hash
    assert temporary.read_bytes() == b"unverified-candidate"
    assert state.record("001.png") is not None


def test_prepared_missing_source_rejects_unchanged_previous_jxl(tmp_path: Path) -> None:
    state, source, destination, temporary = _prepared_transaction_with_missing_source(
        tmp_path
    )
    record = state.record("001.png")
    assert record is not None
    record["previous_output_sha256"] = file_sha256(destination)
    state.save()

    with pytest.raises(RuntimeError, match="before a new JXL was committed"):
        _recover(state, tmp_path, FakeJxlEncoder(), RunSummary())

    assert not source.exists()
    assert destination.read_bytes() == b"external-completion"
    assert temporary.read_bytes() == b"unverified-candidate"
    assert state.record("001.png") is not None


@pytest.mark.parametrize(
    "invalid_previous_hash",
    [True, 17, "", "0" * 63, "g" * 64],
)
def test_prepared_missing_source_rejects_invalid_previous_hash(
    tmp_path: Path,
    invalid_previous_hash: object,
) -> None:
    state, source, destination, temporary = _prepared_transaction_with_missing_source(
        tmp_path
    )
    record = state.record("001.png")
    assert record is not None
    record["previous_output_sha256"] = invalid_previous_hash
    state.save()
    destination_hash = file_sha256(destination)
    encoder = RecordingJxlEncoder()
    summary = RunSummary()

    with pytest.raises(RuntimeError, match="invalid previous output hash"):
        _recover(state, tmp_path, encoder, summary)

    assert encoder.verify_calls == []
    assert not source.exists()
    assert file_sha256(destination) == destination_hash
    assert temporary.read_bytes() == b"unverified-candidate"
    assert state.record("001.png") is not None
    assert summary.external_jxl_recoveries == 0
    assert summary.deferred == 0


def test_prepared_missing_source_rejects_symlinked_external_jxl(tmp_path: Path) -> None:
    state, _source, destination, temporary = _prepared_transaction_with_missing_source(
        tmp_path
    )
    external = tmp_path / "external.jxl"
    external.write_bytes(b"external-completion")
    destination.unlink()
    destination.symlink_to(external.name)

    with pytest.raises(
        RuntimeError,
        match="symbolic link|destination is not the source's sibling JXL|not a regular file",
    ):
        _recover(state, tmp_path, FakeJxlEncoder(), RunSummary())

    assert destination.is_symlink()
    assert temporary.read_bytes() == b"unverified-candidate"
    assert state.record("001.png") is not None


def test_prepared_missing_source_detects_external_jxl_verification_race(
    tmp_path: Path,
) -> None:
    state, _source, destination, temporary = _prepared_transaction_with_missing_source(
        tmp_path
    )

    with pytest.raises(RuntimeError, match="changed during prepared recovery"):
        _recover(state, tmp_path, MutatingJxlEncoder(), RunSummary())

    assert destination.read_bytes() == b"tampered-completion"
    assert temporary.read_bytes() == b"unverified-candidate"
    assert state.record("001.png") is not None


def test_recovery_discards_uncommitted_candidate_if_source_changed(tmp_path: Path) -> None:
    source = tmp_path / "001.png"
    source.write_bytes(b"source-v1")
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"old-jxl")
    temporary = tmp_path / ".001.jxl.transaction.part"
    temporary.write_bytes(b"candidate")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        temporary.name,
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": file_sha256(destination),
        },
    )
    state.mark_encoded(
        "001.png", temporary, temporary.stat().st_size, file_sha256(temporary)
    )
    state.save()
    source.write_bytes(b"source-v2")

    summary = RunSummary()
    blocked = _recover(state, tmp_path, FakeJxlEncoder(), summary)

    assert blocked == {"001.png"}
    assert source.read_bytes() == b"source-v2"
    assert destination.read_bytes() == b"old-jxl"
    assert not temporary.exists()
    assert state.record("001.png") is None
    assert summary.deferred == 1


@pytest.mark.parametrize("source_state", ["missing", "symlink"])
def test_encoded_recovery_preserves_verified_candidate_when_source_is_unavailable(
    tmp_path: Path,
    source_state: str,
) -> None:
    source = tmp_path / "001.png"
    source.write_bytes(b"source-v1")
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"old-jxl")
    temporary = tmp_path / ".001.jxl.transaction.part"
    temporary.write_bytes(b"verified-candidate")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        temporary.name,
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": file_sha256(destination),
        },
    )
    state.mark_encoded(
        "001.png", temporary, temporary.stat().st_size, file_sha256(temporary)
    )
    state.save()
    source.unlink()
    if source_state == "symlink":
        replacement = tmp_path / "replacement.png"
        replacement.write_bytes(b"external-source")
        source.symlink_to(replacement.name)

    summary = RunSummary()
    with pytest.raises(RuntimeError, match="lost its source|symbolic link"):
        _recover(state, tmp_path, FakeJxlEncoder(), summary)

    assert destination.read_bytes() == b"old-jxl"
    assert temporary.read_bytes() == b"verified-candidate"
    assert state.record("001.png") is not None
    assert summary.replaced_sources == 0
    assert summary.deferred == 0


def _encoded_recovery_fixture(
    tmp_path: Path,
) -> tuple[StateStore, Path, Path, Path]:
    source = tmp_path / "001.png"
    source.write_bytes(b"source")
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"old-jxl")
    temporary = tmp_path / ".001.jxl.transaction.part"
    temporary.write_bytes(b"verified-candidate")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        temporary.name,
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": file_sha256(destination),
        },
    )
    state.mark_encoded(
        "001.png", temporary, temporary.stat().st_size, file_sha256(temporary)
    )
    state.save()
    return state, source, destination, temporary


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_sha256", "A" * 64, "invalid candidate hash"),
        ("candidate_sha256", "0" * 63, "invalid candidate hash"),
        ("candidate_sha256", True, "invalid candidate hash"),
        ("candidate_size", True, "invalid candidate size"),
        ("candidate_size", 0, "invalid candidate size"),
        ("candidate_size", "123", "invalid candidate size"),
        ("previous_output_sha256", "A" * 64, "invalid previous output hash"),
        ("previous_output_sha256", "g" * 64, "invalid previous output hash"),
        ("previous_output_sha256", True, "invalid previous output hash"),
        ("encode_mode", "unknown", "unknown encode mode"),
        ("output_width", True, "invalid dimensions"),
        ("output_width", 0, "invalid dimensions"),
        ("output_width", "1600", "invalid dimensions"),
        ("output_height", True, "invalid dimensions"),
        ("output_height", 0, "invalid dimensions"),
        ("output_height", "2400", "invalid dimensions"),
    ],
)
def test_encoded_recovery_rejects_invalid_journal_fields_without_mutation(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    state, source, destination, temporary = _encoded_recovery_fixture(tmp_path)
    record = state.record("001.png")
    assert record is not None
    record[field] = value
    state.save()
    source_hash = file_sha256(source)
    destination_hash = file_sha256(destination)
    temporary_hash = file_sha256(temporary)
    encoder = RecordingJxlEncoder()

    with pytest.raises(RuntimeError, match=message):
        _recover(state, tmp_path, encoder, RunSummary())

    assert encoder.verify_calls == []
    assert file_sha256(source) == source_hash
    assert file_sha256(destination) == destination_hash
    assert file_sha256(temporary) == temporary_hash
    assert state.record("001.png") is not None


def test_output_ready_recovery_rejects_invalid_dimensions_before_source_removal(
    tmp_path: Path,
) -> None:
    state, source, destination, temporary = _encoded_recovery_fixture(tmp_path)
    FakeJxlEncoder().commit(temporary, destination, file_sha256(temporary))
    state.mark_output_ready("001.png", destination)
    record = state.record("001.png")
    assert record is not None
    record["output_width"] = 0
    state.save()
    destination_hash = file_sha256(destination)

    with pytest.raises(RuntimeError, match="invalid dimensions"):
        _recover(state, tmp_path, RecordingJxlEncoder(), RunSummary())

    assert source.exists()
    assert file_sha256(destination) == destination_hash
    assert state.record("001.png") is not None


def test_encoded_recovery_rejects_parent_directory_symbolic_link_without_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "library"
    root.mkdir()
    actual = root / "actual"
    actual.mkdir()
    source = actual / "001.png"
    source.write_bytes(b"source")
    destination = actual / "001.jxl"
    destination.write_bytes(b"old-jxl")
    temporary = actual / ".001.jxl.transaction.part"
    temporary.write_bytes(b"verified-candidate")
    alias = root / "alias"
    alias.symlink_to(actual.name, target_is_directory=True)
    state = StateStore(root / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, root, "model", "pipeline")
    state.prepare_replace(
        "alias/001.png",
        fingerprint,
        "alias/001.jxl",
        "alias/.001.jxl.transaction.part",
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": file_sha256(destination),
        },
    )
    state.mark_encoded(
        "alias/001.png", temporary, temporary.stat().st_size, file_sha256(temporary)
    )
    state.save()
    source_hash = file_sha256(source)
    destination_hash = file_sha256(destination)
    temporary_hash = file_sha256(temporary)

    with pytest.raises(RuntimeError, match="traverses a symbolic link"):
        _recover(state, root, RecordingJxlEncoder(), RunSummary())

    assert file_sha256(source) == source_hash
    assert file_sha256(destination) == destination_hash
    assert file_sha256(temporary) == temporary_hash
    assert state.record("alias/001.png") is not None


def test_encoded_recovery_replans_changed_source_when_candidate_was_cleaned(
    tmp_path: Path,
) -> None:
    source = tmp_path / "001.png"
    source.write_bytes(b"source-v1")
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"old-jxl")
    temporary = tmp_path / ".001.jxl.transaction.part"
    temporary.write_bytes(b"verified-candidate")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        temporary.name,
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": file_sha256(destination),
        },
    )
    state.mark_encoded(
        "001.png", temporary, temporary.stat().st_size, file_sha256(temporary)
    )
    state.save()
    temporary.unlink()
    source.write_bytes(b"source-v2")

    summary = RunSummary()
    blocked = _recover(state, tmp_path, FakeJxlEncoder(), summary)

    assert blocked == set()
    assert source.read_bytes() == b"source-v2"
    assert destination.read_bytes() == b"old-jxl"
    assert state.record("001.png") is None
    assert summary.deferred == 0


@pytest.mark.parametrize("stale_context", ["pipeline_signature", "source_root", "model_sha256"])
def test_recovery_discards_encoded_candidate_when_current_context_changed(
    tmp_path: Path, stale_context: str
) -> None:
    source = tmp_path / "001.png"
    source.write_bytes(b"source")
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"old-jxl")
    temporary = tmp_path / ".001.jxl.transaction.part"
    temporary.write_bytes(b"candidate-from-old-context")
    recorded_root = tmp_path / "old-library" if stale_context == "source_root" else tmp_path
    recorded_signature = "old-pipeline" if stale_context == "pipeline_signature" else "pipeline"
    recorded_model = "old-model" if stale_context == "model_sha256" else "model"
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(
        source, recorded_root, recorded_model, recorded_signature
    )
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        temporary.name,
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": file_sha256(destination),
        },
    )
    state.mark_encoded(
        "001.png", temporary, temporary.stat().st_size, file_sha256(temporary)
    )
    state.save()

    summary = RunSummary()
    blocked = _recover(
        state,
        tmp_path,
        FakeJxlEncoder(),
        summary,
        model_hash="new-model" if stale_context == "model_sha256" else "model",
    )

    assert blocked == set()
    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"old-jxl"
    assert not temporary.exists()
    assert state.record("001.png") is None
    assert summary.replaced_sources == 0
    assert summary.deferred == 0


def test_output_ready_from_old_signature_never_removes_source(tmp_path: Path) -> None:
    source = tmp_path / "001.png"
    source.write_bytes(b"source")
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"candidate-from-old-pipeline")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "old-pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        ".001.jxl.missing-candidate.part",
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": None,
        },
    )
    record = state.record("001.png")
    assert record is not None
    record["verified"] = True
    state.mark_output_ready("001.png", destination)
    state.save()

    summary = RunSummary()
    blocked = _recover(
        state,
        tmp_path,
        FakeJxlEncoder(),
        summary,
        pipeline_signature="current-pipeline",
    )

    assert blocked == set()
    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"candidate-from-old-pipeline"
    assert state.record("001.png") is not None
    assert summary.replaced_sources == 0


def test_current_output_ready_without_verified_evidence_never_removes_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "001.png"
    source.write_bytes(b"source")
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"unverified-output")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        ".001.jxl.missing-candidate.part",
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": None,
        },
    )
    record = state.record("001.png")
    assert record is not None
    record.update(
        phase="output_ready",
        output_size=destination.stat().st_size,
        output_sha256=file_sha256(destination),
    )
    state.save()

    with pytest.raises(RuntimeError, match="lacks verified decode evidence"):
        _recover(state, tmp_path, FakeJxlEncoder(), RunSummary())

    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"unverified-output"
    assert state.record("001.png") is not None


def test_source_removal_primitive_rejects_unverified_matching_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "001.png"
    source.write_bytes(b"source")
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"matching-output")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        ".001.jxl.part",
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
        },
    )
    record = state.record("001.png")
    assert record is not None
    record.update(
        phase="output_ready",
        output_size=destination.stat().st_size,
        output_sha256=file_sha256(destination),
    )

    with pytest.raises(RuntimeError, match="verified output-ready"):
        _commit_source_removal(state, "001.png", source, destination, RunSummary())

    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"matching-output"


def test_current_output_ready_is_redecoded_before_source_removal(tmp_path: Path) -> None:
    class RecordingEncoder(FakeJxlEncoder):
        def __init__(self) -> None:
            self.calls: list[tuple[Path, tuple[int, ...]]] = []

        def verify(self, path: Path, *dimensions: int) -> tuple[int, int]:
            self.calls.append((path, dimensions))
            return super().verify(path, *dimensions)

    source = tmp_path / "001.png"
    source.write_bytes(b"source")
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"verified-output")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        ".001.jxl.missing-candidate.part",
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": None,
        },
    )
    record = state.record("001.png")
    assert record is not None
    record["verified"] = True
    state.mark_output_ready("001.png", destination)
    state.save()
    encoder = RecordingEncoder()

    _recover(state, tmp_path, encoder, RunSummary())

    assert encoder.calls == [(destination, (1600, 2400))]
    assert not source.exists()
    assert destination.exists()
    assert state.record("001.png") is None


def test_current_output_ready_decode_failure_retains_source_and_state(
    tmp_path: Path,
) -> None:
    class RejectingEncoder(FakeJxlEncoder):
        def verify(self, _path: Path, *_dimensions: int) -> tuple[int, int]:
            raise RuntimeError("djxl verification failed")

    source = tmp_path / "001.png"
    source.write_bytes(b"source")
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"corrupt-output")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        ".001.jxl.missing-candidate.part",
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": None,
        },
    )
    record = state.record("001.png")
    assert record is not None
    record["verified"] = True
    state.mark_output_ready("001.png", destination)
    state.save()

    with pytest.raises(RuntimeError, match="djxl verification failed"):
        _recover(state, tmp_path, RejectingEncoder(), RunSummary())

    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"corrupt-output"
    assert state.record("001.png") is not None


def test_stale_encoded_transaction_never_claims_recorded_previous_jxl(
    tmp_path: Path,
) -> None:
    source = tmp_path / "001.png"
    source.write_bytes(b"source")
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"user-jxl-before-transaction")
    temporary = tmp_path / ".001.jxl.transaction.part"
    temporary.write_bytes(b"candidate-from-old-pipeline")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "old-pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        temporary.name,
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": file_sha256(destination),
        },
    )
    state.mark_encoded(
        "001.png", temporary, temporary.stat().st_size, file_sha256(temporary)
    )
    temporary.unlink()
    state.save()

    summary = RunSummary()
    blocked = _recover(
        state,
        tmp_path,
        FakeJxlEncoder(),
        summary,
        pipeline_signature="current-pipeline",
    )

    assert blocked == set()
    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"user-jxl-before-transaction"
    assert state.record("001.png") is None
    assert not state.owns_output("001.png", destination)
    assert summary.replaced_sources == 0


def test_recovery_commits_verified_candidate_over_recorded_previous_jxl(
    tmp_path: Path,
) -> None:
    source = tmp_path / "001.png"
    source.write_bytes(b"source")
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"old-jxl")
    temporary = tmp_path / ".001.jxl.transaction.part"
    temporary.write_bytes(b"verified-candidate")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        temporary.name,
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": file_sha256(destination),
            "replaces_existing_jxl": True,
        },
    )
    state.mark_encoded(
        "001.png", temporary, temporary.stat().st_size, file_sha256(temporary)
    )
    state.save()

    summary = RunSummary()
    _recover(state, tmp_path, FakeJxlEncoder(), summary)

    assert not source.exists()
    assert not temporary.exists()
    assert destination.read_bytes() == b"verified-candidate"
    assert state.record("001.png") is None
    assert summary.existing_jxl_replaced == 1


def test_recovery_refuses_candidate_if_previous_jxl_changed(tmp_path: Path) -> None:
    source = tmp_path / "001.png"
    source.write_bytes(b"source")
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"old-jxl")
    previous_hash = file_sha256(destination)
    temporary = tmp_path / ".001.jxl.transaction.part"
    temporary.write_bytes(b"verified-candidate")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        destination.name,
        temporary.name,
        {
            "action": "sr",
            "encode_mode": "pixels",
            "output_width": 1600,
            "output_height": 2400,
            "previous_output_sha256": previous_hash,
        },
    )
    state.mark_encoded(
        "001.png", temporary, temporary.stat().st_size, file_sha256(temporary)
    )
    state.save()
    destination.write_bytes(b"changed-jxl")

    with pytest.raises(RuntimeError, match="Destination changed"):
        _recover(state, tmp_path, FakeJxlEncoder(), RunSummary())

    assert source.exists()
    assert destination.read_bytes() == b"changed-jxl"
    assert temporary.exists()


def test_encoded_replace_phase_is_admissible_only_for_recorded_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "001.png"
    source.write_bytes(b"source")
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"verified-candidate")
    temporary = tmp_path / ".001.jxl.transaction.part"
    temporary.write_bytes(b"verified-candidate")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "pipeline")
    state.prepare_replace(
        "001.png",
        fingerprint,
        "001.jxl",
        temporary.name,
        {"encode_mode": "pixels", "output_width": 1, "output_height": 1},
    )
    state.mark_encoded(
        "001.png", temporary, temporary.stat().st_size, file_sha256(temporary)
    )

    assert _replace_destination_is_admissible(
        state, "001.png", destination, tmp_path
    )
    other = tmp_path / "other.jxl"
    other.write_bytes(b"verified-candidate")
    assert not _replace_destination_is_admissible(state, "001.png", other, tmp_path)


def test_mirror_refuses_unmanaged_output_unless_overwrite_is_explicit(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "001.jxl"
    destination.write_bytes(b"manual-output")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")

    assert not _mirror_destination_is_admissible(
        state, "001.png", destination, overwrite=False
    )
    assert _mirror_destination_is_admissible(
        state, "001.png", destination, overwrite=True
    )


def test_mirror_can_repair_a_damaged_output_at_its_recorded_path(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    destination = tmp_path / "nested" / "001.jxl"
    destination.parent.mkdir()
    destination.write_bytes(b"managed-output")
    state = StateStore(tmp_path / ".waifuhat2x-state.json")
    fingerprint = state.fingerprint(source, tmp_path, "model", "pipeline")
    state.update("source.png", fingerprint, destination)
    destination.write_bytes(b"damaged")

    assert _mirror_destination_is_admissible(
        state, "source.png", destination, overwrite=False
    )
    unrelated = tmp_path / "other" / "001.jxl"
    unrelated.parent.mkdir()
    unrelated.write_bytes(b"damaged")
    assert not _mirror_destination_is_admissible(
        state, "source.png", unrelated, overwrite=False
    )


def _encoder_with_capture(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, np.ndarray]
) -> JxlEncoder:
    encoder = object.__new__(JxlEncoder)

    def fake_encode(array: np.ndarray, _destination: Path, **_kwargs: object) -> JxlStats:
        captured["array"] = array.copy()
        return JxlStats(seconds=0.25, bytes=123)

    monkeypatch.setattr(encoder, "encode", fake_encode)
    return encoder


def test_jxl_worker_uses_linear_light_resize_and_reports_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, np.ndarray] = {}
    calls: list[tuple[int, int]] = []
    encoder = _encoder_with_capture(monkeypatch, captured)

    def fake_linear_resize(array: np.ndarray, width: int, height: int) -> np.ndarray:
        calls.append((width, height))
        return np.full((height, width), int(array[0, 0]), dtype=np.uint8)

    monkeypatch.setattr("waifuhat2x.jxl.resize_linear_light", fake_linear_resize)
    source = np.full((5, 7), 91, dtype=np.uint8)

    stats = encoder.encode_resized(
        source,
        tmp_path / "page.jxl",
        3,
        2,
        linear_light=True,
    )

    assert calls == [(3, 2)]
    assert captured["array"].shape == (2, 3)
    assert stats.seconds == 0.25
    assert stats.postprocess_seconds >= 0.0


def test_jxl_worker_uses_lanczos_resize_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, np.ndarray] = {}
    encoder = _encoder_with_capture(monkeypatch, captured)
    monkeypatch.setattr(
        "waifuhat2x.jxl.resize_linear_light",
        lambda *_args: pytest.fail("linear-light resize must not be used"),
    )
    source = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    expected = np.asarray(
        Image.fromarray(source).resize((4, 3), Image.Resampling.LANCZOS)
    )

    encoder.encode_resized(
        source,
        tmp_path / "page.jxl",
        4,
        3,
        linear_light=False,
    )

    np.testing.assert_array_equal(captured["array"], expected)


def test_jxl_worker_skips_resize_when_dimensions_already_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, np.ndarray] = {}
    encoder = _encoder_with_capture(monkeypatch, captured)
    monkeypatch.setattr(
        "waifuhat2x.jxl.resize_linear_light",
        lambda *_args: pytest.fail("matching dimensions must not be resized"),
    )
    source = np.arange(4 * 6, dtype=np.uint8).reshape(4, 6)

    stats = encoder.encode_resized(
        source,
        tmp_path / "page.jxl",
        6,
        4,
        linear_light=True,
    )

    np.testing.assert_array_equal(captured["array"], source)
    assert stats.postprocess_seconds == 0.0


def test_jxl_worker_resize_failure_propagates_through_future(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder = _encoder_with_capture(monkeypatch, {})

    def fail_resize(*_args: object) -> np.ndarray:
        raise RuntimeError("resize failed")

    monkeypatch.setattr("waifuhat2x.jxl.resize_linear_light", fail_resize)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            encoder.encode_resized,
            np.zeros((8, 8), dtype=np.uint8),
            tmp_path / "page.jxl",
            4,
            4,
            linear_light=True,
        )
        with pytest.raises(RuntimeError, match="resize failed"):
            future.result()
