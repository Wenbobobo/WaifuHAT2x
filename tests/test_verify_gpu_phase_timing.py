from __future__ import annotations

from types import SimpleNamespace

from scripts.verify_gpu_phase_timing import _timing_validation_errors


def _stats(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "seconds": 1.0,
        "h2d_seconds": 0.01,
        "forward_seconds": 0.90,
        "gpu_postprocess_seconds": 0.04,
        "d2h_seconds": 0.01,
        "gpu_event_total_seconds": 1.08,
        "gpu_event_scale_to_wall": 1.0 / 1.08,
        "gpu_event_raw_seconds": {
            "gpu_total": 1.08,
            "h2d": 0.0108,
            "forward": 0.972,
            "gpu_postprocess": 0.0432,
            "d2h": 0.0108,
        },
        "inference_interval_ns": (1_000_000_000, 2_000_000_000),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_gpu_phase_timing_validation_accepts_calibrated_subset() -> None:
    assert _timing_validation_errors(_stats()) == []


def test_gpu_phase_timing_validation_rejects_missing_raw_calibration() -> None:
    errors = _timing_validation_errors(
        _stats(gpu_event_total_seconds=None, gpu_event_raw_seconds=None)
    )

    assert "raw GPU Event total is missing or non-positive" in errors
    assert "raw GPU Event phase map is missing gpu_total" in errors


def test_gpu_phase_timing_validation_rejects_component_sum_over_wall() -> None:
    errors = _timing_validation_errors(_stats(forward_seconds=1.10))

    assert any("phase sum exceeds" in error for error in errors)


def test_gpu_phase_timing_validation_rejects_bad_wall_interval() -> None:
    missing = _timing_validation_errors(_stats(inference_interval_ns=None))
    mismatched = _timing_validation_errors(
        _stats(inference_interval_ns=(1_000_000_000, 2_500_000_000))
    )

    assert "synchronized inference interval is missing or invalid" in missing
    assert any("interval differs" in error for error in mismatched)
