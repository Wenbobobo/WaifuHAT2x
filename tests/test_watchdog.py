from __future__ import annotations

from pathlib import Path
import os
import pytest
import sys
import time


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_with_watchdog import _run_command, _worker_command, build_parser  # noqa: E402


def test_watchdog_forwards_optional_metrics_directory() -> None:
    args = build_parser().parse_args(
        ["--config", "daily.toml", "--metrics-dir", "Benchmark/Metrics"]
    )

    assert _worker_command(args.config, args.metrics_dir) == [
        sys.executable,
        "-u",
        "-m",
        "waifuhat2x",
        "--config",
        "daily.toml",
        "--metrics-dir",
        "Benchmark/Metrics",
    ]


def test_watchdog_passes_through_normal_exit() -> None:
    returncode, stalled = _run_command(
        [sys.executable, "-u", "-c", "print('completed')"],
        stall_seconds=2.0,
    )

    assert returncode == 0
    assert not stalled


def test_watchdog_terminates_silent_stalled_child() -> None:
    started = time.monotonic()
    returncode, stalled = _run_command(
        [sys.executable, "-u", "-c", "import time; time.sleep(30)"],
        stall_seconds=0.2,
    )

    assert returncode == 124
    assert stalled
    assert time.monotonic() - started < 5.0


def test_watchdog_terminates_worker_descendants(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    child_code = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); "
        "time.sleep(30)"
    )

    returncode, stalled = _run_command(
        [sys.executable, "-u", "-c", child_code],
        stall_seconds=0.5,
    )

    assert returncode == 124
    assert stalled
    grandchild_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 2.0
    while Path(f"/proc/{grandchild_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not Path(f"/proc/{grandchild_pid}").exists()
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild_pid, 0)


def test_watchdog_cleans_descendant_after_worker_exits(tmp_path: Path) -> None:
    pid_file = tmp_path / "orphan.pid"
    child_code = (
        "import pathlib,subprocess,sys; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))"
    )

    returncode, stalled = _run_command(
        [sys.executable, "-u", "-c", child_code],
        stall_seconds=2.0,
    )

    assert returncode == 0
    assert not stalled
    orphan_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 2.0
    while Path(f"/proc/{orphan_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not Path(f"/proc/{orphan_pid}").exists()
