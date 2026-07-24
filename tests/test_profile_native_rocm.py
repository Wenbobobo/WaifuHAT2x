from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

import scripts.profile_native_rocm as profiler
from scripts.profile_native_rocm import (
    SCHEMA_VERSION,
    analyze_kernel_rows,
    analyze_profile,
    artifact_inventory,
    assert_native_linux,
    benchmark_output_hashes,
    build_parser,
    build_profile_command,
    build_roctx_environment,
    canonical_sha256,
    inspect_profiler_logs,
    load_prototype_run_summary,
    read_kernel_rows,
    relative_artifact_record,
    sha256_file,
    snapshot_inputs,
    validate_baseline_summary,
    validate_isolation,
    validate_profile_evidence,
    validate_rocprof_version,
    validate_roctx_preflight,
    verify_input_snapshot,
    write_json,
)


NORMAL_HASH = "a" * 64
SHARPER_HASH = "b" * 64
PAGE_ONE_PIXELS = bytes(range(12))
PAGE_TWO_PIXELS = bytes(range(12, 24))
PAGE_ONE_HASH = hashlib.sha256(PAGE_ONE_PIXELS).hexdigest()
PAGE_TWO_HASH = hashlib.sha256(PAGE_TWO_PIXELS).hexdigest()


def test_native_guard_rejects_windows_and_wsl() -> None:
    with pytest.raises(RuntimeError, match="native Linux"):
        assert_native_linux(
            system="Windows", release="11", version="11", environment={}
        )
    with pytest.raises(RuntimeError, match="disabled under WSL"):
        assert_native_linux(
            system="Linux",
            release="6.6.87.2-microsoft-standard-WSL2",
            version="#1 SMP",
            environment={},
        )
    with pytest.raises(RuntimeError, match="disabled under WSL"):
        assert_native_linux(
            system="Linux",
            release="6.8.0-generic",
            version="#1 SMP",
            environment={"WSL_INTEROP": "/run/WSL/1_interop"},
        )


def test_native_guard_accepts_native_linux() -> None:
    assert_native_linux(
        system="Linux",
        release="6.8.0-generic",
        version="#1 SMP PREEMPT_DYNAMIC",
        environment={},
    )


def test_run_parser_requires_production_root_and_a_tile_mode() -> None:
    parser = build_parser()
    common = [
        "run",
        "--manifest",
        "manifest.json",
        "--page-indexes",
        "1",
        "2",
        "--normal-model",
        "normal.pth",
        "--sharper-model",
        "sharper.pth",
        "--output-root",
        "out",
        "--run-name",
        "test",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args([*common, "--tile", "256"])
    with pytest.raises(SystemExit):
        parser.parse_args([*common, "--production-root", "production"])
    parsed = parser.parse_args(
        [*common, "--production-root", "production", "--tile", "256"]
    )
    assert parsed.production_root == Path("production")


def test_analyze_parser_accepts_only_evidence_files() -> None:
    parser = build_parser()
    parsed = parser.parse_args(
        [
            "analyze",
            "--profile-root",
            "profile",
            "--kernel-pattern",
            "attention",
            "--baseline-summary",
            "baseline.json",
            "--prototype-baseline-summary",
            "baseline-prototype.json",
            "--prototype-candidate-summary",
            "candidate-prototype.json",
        ]
    )
    assert parsed.baseline_summary == Path("baseline.json")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "analyze",
                "--profile-root",
                "profile",
                "--kernel-pattern",
                "attention",
                "--baseline-end-to-end-seconds",
                "10",
            ]
        )


def test_profile_command_selects_only_roctx_controlled_steady_rounds(
    tmp_path: Path,
) -> None:
    command = build_profile_command(
        rocprofv3=tmp_path / "rocprofv3",
        trace_root=tmp_path / "trace",
        benchmark_script=tmp_path / "benchmark.py",
        manifest=tmp_path / "manifest.json",
        page_indexes=[1, 2],
        normal_model=tmp_path / "normal.pth",
        sharper_model=tmp_path / "sharper.pth",
        benchmark_output_root=tmp_path / "benchmark",
        tile=256,
        adaptive_tiles=None,
        overlap=32,
        rounds=3,
        warmups_per_model=1,
        warmup_crop=320,
    )
    delimiter = command.index("--")
    profiler_args = command[:delimiter]
    child_args = command[delimiter + 1 :]
    assert "--selected-regions" in profiler_args
    assert "--kernel-trace" in profiler_args
    assert "--marker-trace" in profiler_args
    assert "--runtime-trace" not in profiler_args
    assert "--rocprof-selected-regions" in child_args


def test_rocprof_version_must_succeed_and_report_rocm_72() -> None:
    report = validate_rocprof_version(
        subprocess.CompletedProcess(
            ["rocprofv3", "--version"],
            0,
            stdout="version: 1.1.0\nrocm_version: 7.2.1\n",
            stderr="",
        )
    )
    assert report["rocm_version"] == "7.2.1"
    with pytest.raises(RuntimeError, match="failed"):
        validate_rocprof_version(
            subprocess.CompletedProcess(
                ["rocprofv3", "--version"], 1, stdout="", stderr="broken"
            )
        )
    with pytest.raises(RuntimeError, match="7.2.x"):
        validate_rocprof_version(
            subprocess.CompletedProcess(
                ["rocprofv3", "--version"],
                0,
                stdout="version: 1.0.0\nrocm_version: 7.1.0\n",
                stderr="",
            )
        )


def test_roctx_environment_is_explicit_and_preflight_is_fail_closed(
    tmp_path: Path,
) -> None:
    rocm = tmp_path / "rocm"
    invocation = rocm / "bin" / "rocprofv3"
    invocation.parent.mkdir(parents=True)
    bindings = rocm / "lib" / "python3.12" / "site-packages" / "roctx"
    bindings.mkdir(parents=True)
    environment, report = build_roctx_environment(
        invocation,
        base_environment={"PYTHONPATH": "existing"},
        python_version=(3, 12),
    )
    expected_site = str(bindings.parent.resolve())
    assert environment["PYTHONPATH"].split(os.pathsep) == [expected_site, "existing"]
    assert report["roctx_site_packages"] == expected_site
    assert (
        validate_roctx_preflight(
            subprocess.CompletedProcess(
                ["python", "-c", "..."], 0, stdout="roctx-control-ok\n", stderr=""
            )
        )["exit_code"]
        == 0
    )
    with pytest.raises(RuntimeError, match="preflight failed"):
        validate_roctx_preflight(
            subprocess.CompletedProcess(
                ["python", "-c", "..."], 1, stdout="", stderr="no roctx"
            )
        )


def test_isolation_rejects_any_production_overlap_and_snapshots_inputs(
    tmp_path: Path,
) -> None:
    production = tmp_path / "production"
    isolated = tmp_path / "isolated"
    output = tmp_path / "profile"
    production.mkdir()
    isolated.mkdir()
    manifest = isolated / "manifest.json"
    page = isolated / "page.png"
    manifest.write_text("{}", encoding="utf-8")
    page.write_bytes(b"page")
    selected = [
        {
            "index": 1,
            "route": "normal",
            "path": str(page.resolve()),
            "sha256": sha256_file(page),
            "bytes": page.stat().st_size,
        }
    ]
    proof = validate_isolation(
        production_root=production,
        manifest=manifest,
        run_root=output,
        selected_pages=selected,
    )
    assert proof and all(check["overlap"] is False for check in proof)
    before = snapshot_inputs(manifest, selected)
    assert verify_input_snapshot(before) == before
    page.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed during profiling"):
        verify_input_snapshot(before)
    with pytest.raises(ValueError, match="Isolation violation"):
        validate_isolation(
            production_root=production,
            manifest=manifest,
            run_root=production / "metrics",
            selected_pages=selected,
        )


class FakeProcess:
    def __init__(self, outcome: Any) -> None:
        self.pid = 4242
        self.outcome = outcome

    def wait(self, timeout: float) -> int:
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return int(self.outcome)


@pytest.mark.parametrize(
    ("outcome", "error"),
    [
        (KeyboardInterrupt(), KeyboardInterrupt),
        (subprocess.TimeoutExpired(["rocprofv3"], 1), TimeoutError),
    ],
)
def test_run_process_cleans_the_pgid_on_every_abnormal_exit(
    monkeypatch: pytest.MonkeyPatch, outcome: BaseException, error: type[BaseException]
) -> None:
    process = FakeProcess(outcome)
    cleaned: list[int] = []
    monkeypatch.setattr(profiler.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        profiler,
        "terminate_process_group",
        lambda observed: cleaned.append(observed.pid),
    )
    with pytest.raises(error):
        profiler.run_process(["rocprofv3"], 1.0, environment={})
    assert cleaned == [process.pid]


def test_run_process_kills_surviving_descendants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess(0)
    cleaned: list[int] = []
    monkeypatch.setattr(profiler.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(profiler, "process_group_exists", lambda _pid: True)
    monkeypatch.setattr(
        profiler,
        "terminate_process_group",
        lambda observed: cleaned.append(observed.pid),
    )
    with pytest.raises(RuntimeError, match="descendants remained"):
        profiler.run_process(["rocprofv3"], 1.0)
    assert cleaned == [process.pid]


class SequencedProcess:
    pid = 5151

    def __init__(self) -> None:
        self.outcomes: list[int | BaseException] = [
            subprocess.TimeoutExpired(["rocprofv3"], 1),
            0,
            0,
        ]

    def wait(self, timeout: float) -> int:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_terminate_process_group_escalates_term_to_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = SequencedProcess()
    group_states = iter((True, True, False))
    waits = iter((False, True))
    signals: list[int] = []
    monkeypatch.setattr(profiler.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(
        profiler, "process_group_exists", lambda _pid: next(group_states)
    )
    monkeypatch.setattr(
        profiler,
        "wait_for_process_group_exit",
        lambda _pid, _timeout: next(waits),
    )
    monkeypatch.setattr(
        profiler.os,
        "killpg",
        lambda _pid, selected_signal: signals.append(selected_signal),
        raising=False,
    )
    profiler.terminate_process_group(process, grace_seconds=0.01)
    assert signals == [profiler.signal.SIGTERM, 9]


def test_run_process_surfaces_cleanup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    process = FakeProcess(KeyboardInterrupt())
    monkeypatch.setattr(profiler.subprocess, "Popen", lambda *args, **kwargs: process)

    def fail_cleanup(_process: FakeProcess) -> None:
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(profiler, "terminate_process_group", fail_cleanup)
    with pytest.raises(RuntimeError, match="process-group cleanup failed"):
        profiler.run_process(["rocprofv3"], 1.0)


def test_profiler_logs_block_data_loss_and_flag_warnings(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_text("profile complete\n", encoding="utf-8")
    stderr.write_text("warning: inspect clock stability\n", encoding="utf-8")
    report = inspect_profiler_logs([stdout, stderr])
    assert report["manual_review_required"] is True
    stderr.write_text("dropped 12 records due to buffer overflow\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dropped or truncated"):
        inspect_profiler_logs([stdout, stderr])


def write_trace(path: Path, *, streams: tuple[str, ...] = ("1",)) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Kernel_Name",
                "Start_Timestamp",
                "End_Timestamp",
                "Stream_Id",
                "Agent_Id",
                "Queue_Id",
            ],
        )
        writer.writeheader()
        start = 0
        for index, stream in enumerate(streams):
            duration = 30_000_000 if index == 0 else 70_000_000
            writer.writerow(
                {
                    "Kernel_Name": "attention_target" if index == 0 else "other_kernel",
                    "Start_Timestamp": start,
                    "End_Timestamp": start + duration,
                    "Stream_Id": stream,
                    "Agent_Id": "gpu-0",
                    "Queue_Id": "queue-0",
                }
            )
            start += duration
        if len(streams) == 1:
            writer.writerow(
                {
                    "Kernel_Name": "other_kernel",
                    "Start_Timestamp": start,
                    "End_Timestamp": start + 70_000_000,
                    "Stream_Id": streams[0],
                    "Agent_Id": "gpu-0",
                    "Queue_Id": "queue-0",
                }
            )


def test_analysis_uses_single_stream_baseline_forward_service_for_amdahl(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "1_kernel_trace.csv"
    write_trace(trace)
    rows = read_kernel_rows([trace], process_id=123)
    report = analyze_kernel_rows(
        rows,
        ["attention"],
        baseline_end_to_end_seconds=10.0,
        baseline_forward_service_seconds=8.0,
        prototype_speedup=1.3,
        prototype_outputs_match=True,
    )
    assert report["matched_gpu_service_share_percent"] == pytest.approx(30.0)
    assert report["estimated_end_to_end_share_percent"] == pytest.approx(24.0)
    assert report["estimated_end_to_end_gain_percent"] == pytest.approx(
        24.0 * (1.0 - 1.0 / 1.3)
    )
    assert report["baseline_forward_service_seconds"] == 8.0
    assert report["custom_operator_eligible"] is True
    blocked = analyze_kernel_rows(
        rows,
        ["attention"],
        baseline_end_to_end_seconds=10.0,
        baseline_forward_service_seconds=8.0,
        prototype_speedup=1.3,
        prototype_outputs_match=True,
        evidence_blockers=["profiler warning requires review"],
    )
    assert blocked["custom_operator_eligible"] is False
    assert "profiler warning requires review" in blocked["blockers"]


def test_analysis_rejects_multi_stream_attribution(tmp_path: Path) -> None:
    trace = tmp_path / "1_kernel_trace.csv"
    write_trace(trace, streams=("1", "2"))
    with pytest.raises(ValueError, match="exactly one"):
        analyze_kernel_rows(
            read_kernel_rows([trace], process_id=123),
            ["attention"],
            baseline_end_to_end_seconds=10.0,
            baseline_forward_service_seconds=8.0,
            prototype_speedup=2.0,
            prototype_outputs_match=True,
        )


def test_analysis_rejects_same_stream_id_from_multiple_trace_files(
    tmp_path: Path,
) -> None:
    first = tmp_path / "1_kernel_trace.csv"
    second = tmp_path / "2_kernel_trace.csv"
    write_trace(first)
    write_trace(second)
    with pytest.raises(ValueError, match="exactly one"):
        analyze_kernel_rows(
            read_kernel_rows([first, second], process_id=123),
            ["attention"],
            baseline_end_to_end_seconds=10.0,
            baseline_forward_service_seconds=8.0,
            prototype_speedup=2.0,
            prototype_outputs_match=True,
        )


def test_analysis_blocks_unverified_or_low_impact_prototype(tmp_path: Path) -> None:
    trace = tmp_path / "1_kernel_trace.csv"
    write_trace(trace)
    report = analyze_kernel_rows(
        read_kernel_rows([trace], process_id=123),
        ["missing"],
        baseline_end_to_end_seconds=10.0,
        baseline_forward_service_seconds=8.0,
        prototype_speedup=2.0,
        prototype_outputs_match=False,
    )
    assert report["custom_operator_eligible"] is False
    assert "target segment is below both attribution thresholds" in report["blockers"]
    assert "estimated end-to-end gain is below 3%" in report["blockers"]
    assert "prototype output hash equivalence is not confirmed" in report["blockers"]


def test_trace_reader_requires_stream_ids(tmp_path: Path) -> None:
    trace = tmp_path / "bad.csv"
    trace.write_text(
        "Kernel_Name,Start_Timestamp,End_Timestamp\nkernel,0,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported kernel trace columns"):
        read_kernel_rows([trace], process_id=123)


def benchmark_configuration() -> dict[str, Any]:
    return {
        "precision": "bf16",
        "tile_mode": "fixed",
        "tile": 256,
        "adaptive_tiles": None,
        "adaptive_selection_formula": None,
        "overlap": 32,
        "batch_tiles": 1,
        "device_assembly": True,
        "model_cache_size": 2,
        "threshold": 1000,
        "rounds": 1,
        "warmups_per_model": 1,
        "warmup_crop": 320,
        "gpu_phase_timing": True,
    }


def benchmark_models(tmp_path: Path) -> dict[str, dict[str, Any]]:
    return {
        "normal": {
            "path": str(tmp_path / "Real_HAT_GAN_SRx4.pth"),
            "sha256": NORMAL_HASH,
            "bytes": 10,
        },
        "sharper": {
            "path": str(tmp_path / "Real_HAT_GAN_SRx4_sharper.pth"),
            "sha256": SHARPER_HASH,
            "bytes": 11,
        },
    }


def benchmark_summary(
    *,
    manifest: Path,
    models: dict[str, dict[str, Any]],
    selected_regions: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "real_hat_manifest_eager_benchmark",
        "status": "complete",
        "environment": {
            "platform": "Linux-6.8.0-x86_64",
            "python": "3.12.11",
            "torch": "2.9.1+rocm7.2.1",
            "torch_hip": "7.2.1",
            "torch_cuda": None,
            "cuda_api_available": True,
            "gpu": {
                "name": "unsupported-fixture-gpu",
                "total_memory_bytes": 25_751_666_688,
                "multiprocessor_count": 96,
                "device_index": 0,
            },
        },
        "manifest": str(manifest.resolve()),
        "page_order": [1, 2],
        "runtime_code": [{"path": "benchmark.py", "sha256": "9" * 64, "bytes": 1}],
        "models": models,
        "configuration": benchmark_configuration(),
        "profiler_control": {
            "rocprof_selected_regions": selected_regions,
            "scope": "all steady rounds only" if selected_regions else "disabled",
            "range_name": "real_hat_steady_rounds" if selected_regions else None,
        },
        "rounds": [
            {
                "round": 1,
                "loop_wall_seconds": 10.0,
                "pages": [
                    {
                        "index": 1,
                        "route": "normal",
                        "source": str((manifest.parent / "one.png").resolve()),
                        "input_size": [800, 1200],
                        "output_shape": [2, 2, 3],
                        "pixel_sha256": PAGE_ONE_HASH,
                        "stats": {"forward_seconds": 4.0},
                    },
                    {
                        "index": 2,
                        "route": "sharper",
                        "source": str((manifest.parent / "two.png").resolve()),
                        "input_size": [1000, 1400],
                        "output_shape": [2, 2, 3],
                        "pixel_sha256": PAGE_TWO_HASH,
                        "stats": {"forward_seconds": 4.0},
                    },
                ],
            }
        ],
        "steady_state": {
            "round_loop_wall_seconds": {"median": 10.0},
            "pixel_deterministic": True,
            "unique_hashes_per_page": {
                "1": [PAGE_ONE_HASH],
                "2": [PAGE_TWO_HASH],
            },
        },
    }


def make_profile_evidence(tmp_path: Path) -> dict[str, Any]:
    production = tmp_path / "production"
    isolated = tmp_path / "isolated"
    root = tmp_path / "profile"
    production.mkdir()
    isolated.mkdir()
    root.mkdir()
    page_one = isolated / "one.png"
    page_two = isolated / "two.png"
    page_one.write_bytes(b"one")
    page_two.write_bytes(b"two")
    manifest = isolated / "manifest.json"
    manifest.write_text('{"schema_version":1}', encoding="utf-8")
    selected_pages = [
        {
            "index": 1,
            "route": "normal",
            "path": str(page_one.resolve()),
            "sha256": sha256_file(page_one),
            "bytes": page_one.stat().st_size,
        },
        {
            "index": 2,
            "route": "sharper",
            "path": str(page_two.resolve()),
            "sha256": sha256_file(page_two),
            "bytes": page_two.stat().st_size,
        },
    ]
    input_snapshot = snapshot_inputs(manifest, selected_pages)
    models = benchmark_models(tmp_path)
    benchmark_script = tmp_path / "benchmark.py"
    benchmark_script.write_text("# benchmark\n", encoding="utf-8")
    benchmark_script_record = {
        "path": str(benchmark_script.resolve()),
        "sha256": sha256_file(benchmark_script),
        "bytes": benchmark_script.stat().st_size,
    }
    configuration = {
        **benchmark_configuration(),
        "rocprof_selected_regions": True,
        "selected_region_scope": "all steady rounds only",
        "selected_region_name": "real_hat_steady_rounds",
        "timeout_seconds": 3600.0,
    }
    isolation_proof = validate_isolation(
        production_root=production,
        manifest=manifest,
        run_root=root,
        selected_pages=selected_pages,
    )
    command = [
        "/opt/rocm/bin/rocprofv3",
        "--kernel-trace",
        "--marker-trace",
        "--selected-regions",
        "--",
        "/venv/bin/python",
        "benchmark_manifest_eager.py",
        "--rocprof-selected-regions",
    ]
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "real_hat_native_rocprof_plan",
        "status": "running",
        "environment": {
            "rocprofv3_version": {
                "stdout": "version: 1.1.0\nrocm_version: 7.2.1",
                "stderr": "",
                "exit_code": 0,
                "rocm_version": "7.2.1",
            },
            "roctx_paths": {
                "rocm_root": "/opt/rocm-7.2.1",
                "roctx_site_packages": "/opt/rocm-7.2.1/lib/python3.12/site-packages",
            },
            "roctx_preflight": {
                "stdout": "roctx-control-ok",
                "stderr": "",
                "exit_code": 0,
            },
        },
        "production_root": str(production.resolve()),
        "profile_output": str(root.resolve()),
        "benchmark_script": benchmark_script_record,
        "manifest": {
            "path": str(manifest.resolve()),
            "sha256": sha256_file(manifest),
            "selected_pages": selected_pages,
        },
        "input_snapshot_before": input_snapshot,
        "input_snapshot_before_sha256": canonical_sha256(input_snapshot),
        "models": models,
        "configuration": configuration,
        "command": command,
        "safety": {"path_overlap_checks": isolation_proof},
    }
    plan_path = root / "profile_plan.json"
    write_json(plan_path, plan)

    trace_root = root / "trace"
    trace_root.mkdir()
    kernel_trace = trace_root / "1_kernel_trace.csv"
    write_trace(kernel_trace)
    marker_trace = trace_root / "1_marker_api_trace.csv"
    with marker_trace.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Function",
                "Process_Id",
                "Start_Timestamp",
                "End_Timestamp",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Function": "roctxProfilerResume",
                "Process_Id": 123,
                "Start_Timestamp": 0,
                "End_Timestamp": 0,
            }
        )
        writer.writerow(
            {
                "Function": "real_hat_steady_rounds",
                "Process_Id": 123,
                "Start_Timestamp": 0,
                "End_Timestamp": 100_000_000,
            }
        )
        writer.writerow(
            {
                "Function": "roctxProfilerPause",
                "Process_Id": 123,
                "Start_Timestamp": 100_000_000,
                "End_Timestamp": 100_000_000,
            }
        )
    profiler_stdout = root / "rocprof.stdout.log"
    profiler_stderr = root / "rocprof.stderr.log"
    profiler_stdout.write_text("profile complete\n", encoding="utf-8")
    profiler_stderr.write_text("", encoding="utf-8")

    profiled_summary = benchmark_summary(
        manifest=manifest, models=models, selected_regions=True
    )
    profiled_summary["runtime_code"] = [benchmark_script_record]
    benchmark_path = root / "benchmark" / "profiled" / "batch_summary.json"
    write_json(benchmark_path, profiled_summary)

    result = dict(plan)
    result.update(
        {
            "kind": "real_hat_native_rocprof_result",
            "status": "complete",
            "profile_plan": relative_artifact_record(root, plan_path),
            "input_snapshot_after": input_snapshot,
            "input_snapshot_after_sha256": canonical_sha256(input_snapshot),
            "roctx_control_calls": {
                "roctxProfilerResume": 1,
                "roctxProfilerPause": 1,
                "process_id": 123,
                "range_start_ns": 0,
                "range_end_ns": 100_000_000,
            },
            "kernel_traces": [relative_artifact_record(root, kernel_trace)],
            "marker_traces": [relative_artifact_record(root, marker_trace)],
            "profiler_logs": [
                relative_artifact_record(root, profiler_stdout),
                relative_artifact_record(root, profiler_stderr),
            ],
            "profiler_diagnostics": {
                "data_loss_detected": False,
                "manual_review_required": False,
                "warnings": [],
            },
            "benchmark_summary": relative_artifact_record(root, benchmark_path),
        }
    )
    result["artifacts"] = artifact_inventory(root)
    result["artifacts_sha256"] = canonical_sha256(result["artifacts"])
    result_path = root / "profile_result.json"
    write_json(result_path, result)
    completion = {
        "schema_version": SCHEMA_VERSION,
        "kind": "real_hat_native_rocprof_completion",
        "status": "complete",
        "profile_result": relative_artifact_record(root, result_path),
        "profile_plan": result["profile_plan"],
        "artifacts_sha256": result["artifacts_sha256"],
        "kernel_traces": result["kernel_traces"],
        "marker_traces": result["marker_traces"],
        "profiler_logs": result["profiler_logs"],
        "profiler_diagnostics": result["profiler_diagnostics"],
        "benchmark_summary": result["benchmark_summary"],
        "input_snapshot_before_sha256": result["input_snapshot_before_sha256"],
        "input_snapshot_after_sha256": result["input_snapshot_after_sha256"],
    }
    completion_path = root / "completion.json"
    write_json(completion_path, completion)
    baseline = benchmark_summary(
        manifest=manifest, models=models, selected_regions=False
    )
    baseline["runtime_code"] = [benchmark_script_record]
    return {
        "root": root,
        "production": production,
        "kernel_trace": kernel_trace,
        "completion_path": completion_path,
        "profiled_summary": profiled_summary,
        "result": result,
        "baseline": baseline,
    }


def test_profile_evidence_binds_completion_result_artifacts_and_configuration(
    tmp_path: Path,
) -> None:
    built = make_profile_evidence(tmp_path)
    evidence = validate_profile_evidence(built["root"])
    assert evidence["result"]["configuration"]["tile"] == 256
    assert evidence["kernel_records"][0]["sha256"] == sha256_file(built["kernel_trace"])
    metrics, hashes = validate_baseline_summary(
        built["baseline"], built["profiled_summary"], built["result"]
    )
    assert metrics == {"end_to_end_seconds": 10.0, "forward_service_seconds": 8.0}
    assert hashes == {"1": PAGE_ONE_HASH, "2": PAGE_TWO_HASH}


def test_profile_evidence_rejects_a_changed_trace(tmp_path: Path) -> None:
    built = make_profile_evidence(tmp_path)
    with built["kernel_trace"].open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="hash or size changed"):
        validate_profile_evidence(built["root"])


def test_profile_evidence_rejects_a_changed_completion_binding(tmp_path: Path) -> None:
    built = make_profile_evidence(tmp_path)
    completion = json.loads(built["completion_path"].read_text(encoding="utf-8"))
    completion["profile_result"]["sha256"] = "0" * 64
    write_json(built["completion_path"], completion)
    with pytest.raises(ValueError, match="hash or size changed"):
        validate_profile_evidence(built["root"])


def test_baseline_binding_rejects_configuration_drift(tmp_path: Path) -> None:
    built = make_profile_evidence(tmp_path)
    baseline = copy.deepcopy(built["baseline"])
    baseline["configuration"]["overlap"] = 16
    with pytest.raises(ValueError, match="configurations differ"):
        validate_baseline_summary(baseline, built["profiled_summary"], built["result"])


def test_baseline_binding_rejects_round_hash_order_and_environment_drift(
    tmp_path: Path,
) -> None:
    built = make_profile_evidence(tmp_path)
    wrong_order = copy.deepcopy(built["baseline"])
    wrong_order["rounds"][0]["pages"].reverse()
    with pytest.raises(ValueError, match="page order"):
        validate_baseline_summary(
            wrong_order, built["profiled_summary"], built["result"]
        )
    wrong_hash = copy.deepcopy(built["baseline"])
    wrong_hash["rounds"][0]["pages"][0]["pixel_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="output hash"):
        validate_baseline_summary(
            wrong_hash, built["profiled_summary"], built["result"]
        )
    wrong_environment = copy.deepcopy(built["baseline"])
    wrong_environment["environment"]["torch_hip"] = "7.1.0"
    with pytest.raises(ValueError, match="environments differ"):
        validate_baseline_summary(
            wrong_environment, built["profiled_summary"], built["result"]
        )


def write_prototype_run_summary(
    path: Path,
    *,
    role: str,
    built: dict[str, Any],
    samples: list[float],
) -> dict[str, str]:
    output_root = path.parent / f"{path.stem}-actual"
    output_root.mkdir(parents=True)
    output_payloads = {
        "1": PAGE_ONE_PIXELS,
        "2": PAGE_TWO_PIXELS,
    }
    workload, output_shapes = profiler.profile_workload_binding(
        built["result"], built["profiled_summary"]
    )
    records: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for identifier, content in output_payloads.items():
        output = output_root / f"{identifier}.raw"
        output.write_bytes(content)
        digest = sha256_file(output)
        hashes[identifier] = digest
        records.append(
            {
                "id": identifier,
                "path": output.relative_to(path.parent).as_posix(),
                "sha256": digest,
                "bytes": output.stat().st_size,
                "format": "raw-uint8-contiguous",
                "shape": output_shapes[identifier],
            }
        )
    implementation = output_root / "implementation.py"
    implementation.write_text(f"IMPLEMENTATION = {role!r}\n", encoding="utf-8")
    write_json(
        path,
        {
            "schema_version": 1,
            "kind": "real_hat_prototype_run",
            "status": "complete",
            "role": role,
            "environment": built["profiled_summary"]["environment"],
            "models": built["profiled_summary"]["models"],
            "configuration": built["profiled_summary"]["configuration"],
            "workload": workload,
            "target_segment": {
                "kind": "rocprof-kernel-regex-set",
                "kernel_patterns": ["attention"],
            },
            "implementation": {
                "name": f"{role}-implementation",
                "artifacts": [
                    {
                        "path": implementation.relative_to(path.parent).as_posix(),
                        "sha256": sha256_file(implementation),
                        "bytes": implementation.stat().st_size,
                    }
                ],
            },
            "performance": {
                "warmup_excluded": True,
                "single_stream": True,
                "iterations": len(samples),
                "segment_wall_seconds": samples,
                "median_segment_wall_seconds": sorted(samples)[len(samples) // 2],
            },
            "outputs": records,
        },
    )
    return hashes


def test_prototype_run_evidence_hashes_outputs_code_and_performance(
    tmp_path: Path,
) -> None:
    built = make_profile_evidence(tmp_path)
    summary_path = tmp_path / "candidate-run.json"
    expected_hashes = write_prototype_run_summary(
        summary_path,
        role="candidate",
        built=built,
        samples=[1.0, 1.01, 0.99, 1.0, 1.0],
    )
    hashes, evidence = load_prototype_run_summary(
        summary_path,
        label="prototype candidate",
        expected_role="candidate",
        profile_result=built["result"],
        profiled_summary=built["profiled_summary"],
        kernel_patterns=["attention"],
    )
    assert hashes == expected_hashes
    assert evidence["output_count"] == 2
    assert len(evidence["output_files"]) == 2
    assert evidence["implementation"]["artifacts"]
    assert evidence["performance"]["iterations"] == 5
    Path(evidence["output_files"][0]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="(?:size|hash) changed"):
        load_prototype_run_summary(
            summary_path,
            label="prototype candidate",
            expected_role="candidate",
            profile_result=built["result"],
            profiled_summary=built["profiled_summary"],
            kernel_patterns=["attention"],
        )


def test_prototype_run_rejects_workload_and_target_segment_drift(
    tmp_path: Path,
) -> None:
    built = make_profile_evidence(tmp_path)
    summary_path = tmp_path / "candidate-run.json"
    write_prototype_run_summary(
        summary_path,
        role="candidate",
        built=built,
        samples=[1.0, 1.01, 0.99, 1.0, 1.0],
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["workload"]["pages"][0]["input_sha256"] = "f" * 64
    write_json(summary_path, payload)
    with pytest.raises(ValueError, match="workload differs"):
        load_prototype_run_summary(
            summary_path,
            label="prototype candidate",
            expected_role="candidate",
            profile_result=built["result"],
            profiled_summary=built["profiled_summary"],
            kernel_patterns=["attention"],
        )

    payload["workload"], _ = profiler.profile_workload_binding(
        built["result"], built["profiled_summary"]
    )
    payload["target_segment"]["kernel_patterns"] = ["convolution"]
    write_json(summary_path, payload)
    with pytest.raises(ValueError, match="target segment differs"):
        load_prototype_run_summary(
            summary_path,
            label="prototype candidate",
            expected_role="candidate",
            profile_result=built["result"],
            profiled_summary=built["profiled_summary"],
            kernel_patterns=["attention"],
        )


def test_prototype_run_rejects_actual_pixels_that_differ_from_eager(
    tmp_path: Path,
) -> None:
    built = make_profile_evidence(tmp_path)
    summary_path = tmp_path / "candidate-run.json"
    write_prototype_run_summary(
        summary_path,
        role="candidate",
        built=built,
        samples=[1.0, 1.01, 0.99, 1.0, 1.0],
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    output_record = payload["outputs"][0]
    output_path = summary_path.parent / output_record["path"]
    output_path.write_bytes(bytes(reversed(PAGE_ONE_PIXELS)))
    output_record["sha256"] = sha256_file(output_path)
    output_record["bytes"] = output_path.stat().st_size
    write_json(summary_path, payload)
    with pytest.raises(ValueError, match="raw output pixels differ"):
        load_prototype_run_summary(
            summary_path,
            label="prototype candidate",
            expected_role="candidate",
            profile_result=built["result"],
            profiled_summary=built["profiled_summary"],
            kernel_patterns=["attention"],
        )


def test_full_analysis_rejects_same_implementation_with_a_different_name(
    tmp_path: Path,
) -> None:
    built = make_profile_evidence(tmp_path)
    baseline_path = tmp_path / "baseline" / "batch_summary.json"
    write_json(baseline_path, built["baseline"])
    prototype_baseline = tmp_path / "prototype-baseline.json"
    prototype_candidate = tmp_path / "prototype-candidate.json"
    write_prototype_run_summary(
        prototype_baseline,
        role="baseline",
        built=built,
        samples=[1.3, 1.3, 1.3, 1.3, 1.3],
    )
    write_prototype_run_summary(
        prototype_candidate,
        role="candidate",
        built=built,
        samples=[1.0, 1.0, 1.0, 1.0, 1.0],
    )
    baseline_payload = json.loads(prototype_baseline.read_text(encoding="utf-8"))
    candidate_payload = json.loads(prototype_candidate.read_text(encoding="utf-8"))
    baseline_implementation = (
        prototype_baseline.parent
        / baseline_payload["implementation"]["artifacts"][0]["path"]
    )
    candidate_record = candidate_payload["implementation"]["artifacts"][0]
    candidate_implementation = prototype_candidate.parent / candidate_record["path"]
    candidate_implementation.write_bytes(baseline_implementation.read_bytes())
    candidate_record["sha256"] = sha256_file(candidate_implementation)
    candidate_record["bytes"] = candidate_implementation.stat().st_size
    candidate_payload["implementation"]["name"] = "renamed-but-identical"
    write_json(prototype_candidate, candidate_payload)
    args = argparse.Namespace(
        profile_root=built["root"],
        baseline_summary=baseline_path,
        kernel_pattern=["attention"],
        prototype_baseline_summary=prototype_baseline,
        prototype_candidate_summary=prototype_candidate,
        output=None,
    )
    with pytest.raises(ValueError, match="implementations are identical"):
        analyze_profile(args)


def test_full_analysis_binds_baseline_and_prototype_files(tmp_path: Path) -> None:
    built = make_profile_evidence(tmp_path)
    baseline_path = tmp_path / "baseline" / "batch_summary.json"
    write_json(baseline_path, built["baseline"])
    prototype_baseline = tmp_path / "prototype-baseline.json"
    prototype_candidate = tmp_path / "prototype-candidate.json"
    write_prototype_run_summary(
        prototype_baseline,
        role="baseline",
        built=built,
        samples=[1.3, 1.3, 1.3, 1.3, 1.3],
    )
    write_prototype_run_summary(
        prototype_candidate,
        role="candidate",
        built=built,
        samples=[1.0, 1.0, 1.0, 1.0, 1.0],
    )
    args = argparse.Namespace(
        profile_root=built["root"],
        baseline_summary=baseline_path,
        kernel_pattern=["attention"],
        prototype_baseline_summary=prototype_baseline,
        prototype_candidate_summary=prototype_candidate,
        output=None,
    )
    assert analyze_profile(args) == 0
    analysis_path = built["root"] / "analysis.json"
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert payload["analysis"]["custom_operator_eligible"] is True
    assert payload["prototype"]["outputs_match"] is True
    assert payload["baseline"]["unprofiled"] is True


def test_benchmark_output_hashes_reject_non_determinism(tmp_path: Path) -> None:
    payload = benchmark_summary(
        manifest=tmp_path / "manifest.json",
        models=benchmark_models(tmp_path),
        selected_regions=False,
    )
    payload["steady_state"]["pixel_deterministic"] = False
    with pytest.raises(ValueError, match="deterministic"):
        benchmark_output_hashes(payload, label="baseline")
