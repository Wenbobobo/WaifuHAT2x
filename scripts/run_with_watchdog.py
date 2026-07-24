from __future__ import annotations

import argparse
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def _stream_output(process: subprocess.Popen[str], output: queue.Queue[str | None]) -> None:
    assert process.stdout is not None
    try:
        for line in process.stdout:
            output.put(line)
    finally:
        output.put(None)


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False


def _stop(process: subprocess.Popen[str], grace_seconds: float = 15.0) -> None:
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and _process_group_exists(process_group):
        process.poll()
        time.sleep(0.1)
    if _process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_command(command: list[str], stall_seconds: float) -> tuple[int, bool]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    output: queue.Queue[str | None] = queue.Queue()
    reader = threading.Thread(target=_stream_output, args=(process, output), daemon=True)
    reader.start()
    last_output = time.monotonic()

    try:
        while True:
            try:
                line = output.get(timeout=1.0)
            except queue.Empty:
                line = ""
            if line is None:
                returncode = process.wait()
                if _process_group_exists(process.pid):
                    _stop(process)
                return returncode, False
            if line:
                print(line, end="", flush=True)
                last_output = time.monotonic()
            if process.poll() is not None and output.empty():
                returncode = process.returncode
                if _process_group_exists(process.pid):
                    _stop(process)
                return returncode, False
            if time.monotonic() - last_output >= stall_seconds:
                print(
                    f"\n[WATCHDOG] No worker output for {stall_seconds:.0f}s; "
                    "terminating it so the transaction journal can recover.",
                    file=sys.stderr,
                    flush=True,
                )
                _stop(process)
                return 124, True
    except BaseException:
        _stop(process)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Restart WaifuHAT2x after a stalled worker.")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        help="Forward optional versioned pipeline metrics to every worker attempt.",
    )
    parser.add_argument("--stall-seconds", type=float, default=600.0)
    parser.add_argument("--max-restarts", type=int, default=2)
    return parser


def _worker_command(config: str, metrics_dir: Path | None) -> list[str]:
    command = [sys.executable, "-u", "-m", "waifuhat2x", "--config", config]
    if metrics_dir is not None:
        command.extend(["--metrics-dir", str(metrics_dir)])
    return command


def main() -> None:
    args = build_parser().parse_args()
    if args.stall_seconds < 60:
        raise SystemExit("--stall-seconds must be at least 60")
    if args.max_restarts < 0:
        raise SystemExit("--max-restarts cannot be negative")

    command = _worker_command(args.config, args.metrics_dir)
    try:
        for attempt in range(args.max_restarts + 1):
            if attempt:
                print(
                    f"[WATCHDOG] Recovery restart {attempt}/{args.max_restarts}...",
                    flush=True,
                )
            returncode, stalled = _run_command(command, args.stall_seconds)
            if not stalled:
                raise SystemExit(returncode)
            if attempt == args.max_restarts:
                print(
                    "[WATCHDOG] Restart limit reached; state/worklist were retained.",
                    file=sys.stderr,
                )
                raise SystemExit(124)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
