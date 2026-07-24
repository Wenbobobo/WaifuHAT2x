from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any
import uuid


METRICS_SCHEMA_VERSION = 1


def _duration_seconds(interval: tuple[int, int]) -> float:
    return max(0, interval[1] - interval[0]) / 1_000_000_000


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    valid = sorted((start, end) for start, end in intervals if end >= start)
    if not valid:
        return []
    merged = [valid[0]]
    for start, end in valid[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _union_seconds(intervals: list[tuple[int, int]]) -> float:
    return sum(_duration_seconds(interval) for interval in _merge_intervals(intervals))


def _overlap_seconds(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> float:
    first = _merge_intervals(left)
    second = _merge_intervals(right)
    left_index = 0
    right_index = 0
    overlap_ns = 0
    while left_index < len(first) and right_index < len(second):
        left_start, left_end = first[left_index]
        right_start, right_end = second[right_index]
        overlap_ns += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return overlap_ns / 1_000_000_000


@dataclass
class PageTelemetry:
    run: RunTelemetry
    source: str
    index: int
    total: int
    destination: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    spans: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    services: dict[str, float] = field(default_factory=dict)
    _raw_intervals: dict[str, list[tuple[int, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _finished: bool = False

    @contextmanager
    def span(self, name: str, *, clock: str = "cpu_monotonic") -> Iterator[None]:
        if not self.run.enabled:
            yield
            return
        started_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            self.add_interval(name, (started_ns, time.perf_counter_ns()), clock=clock)

    def add_interval(
        self,
        name: str,
        interval: tuple[int, int] | None,
        *,
        clock: str = "cpu_monotonic",
    ) -> None:
        if not self.run.enabled or interval is None:
            return
        start_ns, end_ns = interval
        if end_ns < start_ns:
            return
        self._raw_intervals[name].append((start_ns, end_ns))
        self.spans[name].append(self.run.format_interval(start_ns, end_ns, clock=clock))

    def set_detail(self, name: str, value: Any) -> None:
        if self.run.enabled:
            self.details[name] = value

    def set_service_seconds(self, name: str, seconds: float | int | None) -> None:
        if self.run.enabled and seconds is not None:
            self.services[name] = max(0.0, float(seconds))

    def finish(self, status: str, *, error: BaseException | None = None) -> None:
        if not self.run.enabled or self._finished:
            return
        self._finished = True
        record: dict[str, Any] = {
            "type": "waifuhat2x-page-metrics",
            "schema_version": METRICS_SCHEMA_VERSION,
            "run_id": self.run.run_id,
            "index": self.index,
            "total": self.total,
            "source": self.source,
            "status": status,
            "details": self.details,
            "timing": {
                "clock": "perf_counter_ns",
                "spans": dict(self.spans),
                "cumulative_service_seconds": self.services,
            },
        }
        if self.destination is not None:
            record["destination"] = self.destination
        if error is not None:
            record["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        self.run.record_page(record, self._raw_intervals, self.services)


class RunTelemetry:
    """Best-effort versioned metrics that never participate in output transactions."""

    def __init__(
        self,
        *,
        enabled: bool,
        started_ns: int,
        run_id: str,
        started_utc: str,
        run_dir: Path | None = None,
        pages_handle: Any = None,
    ) -> None:
        self.enabled = enabled
        self.started_ns = started_ns
        self.run_id = run_id
        self.started_utc = started_utc
        self.run_dir = run_dir
        self._pages_handle = pages_handle
        self._lock = threading.Lock()
        self._stage_spans: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._service_totals: dict[str, float] = defaultdict(float)
        self._pages_written = 0
        self._write_errors: list[str] = []

    @classmethod
    def create(cls, root: Path | None, *, started_ns: int) -> RunTelemetry:
        run_id = uuid.uuid4().hex
        started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if root is None:
            return cls(
                enabled=False,
                started_ns=started_ns,
                run_id=run_id,
                started_utc=started_utc,
            )

        metrics_root = root.expanduser().resolve()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_dir = metrics_root / f"{stamp}-{run_id}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            pages_handle = (run_dir / "pages.jsonl").open(
                "x", encoding="utf-8", newline="\n", buffering=1
            )
        except Exception as exc:
            raise RuntimeError(f"Cannot initialize metrics directory {run_dir}: {exc}") from exc
        return cls(
            enabled=True,
            started_ns=started_ns,
            run_id=run_id,
            started_utc=started_utc,
            run_dir=run_dir,
            pages_handle=pages_handle,
        )

    @property
    def write_error_count(self) -> int:
        return len(self._write_errors)

    @property
    def output_path(self) -> str | None:
        return str(self.run_dir) if self.run_dir is not None else None

    def page(
        self,
        source: str,
        index: int,
        total: int,
        *,
        destination: str | None = None,
    ) -> PageTelemetry:
        return PageTelemetry(self, source, index, total, destination)

    def format_interval(
        self, start_ns: int, end_ns: int, *, clock: str
    ) -> dict[str, Any]:
        return {
            "clock": clock,
            "start_offset_ns": max(0, start_ns - self.started_ns),
            "end_offset_ns": max(0, end_ns - self.started_ns),
            "duration_seconds": _duration_seconds((start_ns, end_ns)),
        }

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            ended_ns = time.perf_counter_ns()
            self._stage_spans[name].append(
                self.format_interval(started_ns, ended_ns, clock="cpu_monotonic")
            )
            self._intervals[f"stage:{name}"].append((started_ns, ended_ns))

    def record_page(
        self,
        record: Mapping[str, Any],
        intervals: Mapping[str, list[tuple[int, int]]],
        services: Mapping[str, float],
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            for name, values in intervals.items():
                self._intervals[name].extend(values)
            for name, seconds in services.items():
                self._service_totals[name] += seconds
            if self._pages_handle is None:
                return
            try:
                self._pages_handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                self._pages_handle.flush()
                self._pages_written += 1
            except Exception as exc:
                self._record_write_error("pages.jsonl", exc)
                try:
                    self._pages_handle.close()
                except Exception:
                    pass
                self._pages_handle = None

    def _record_write_error(self, target: str, error: BaseException) -> None:
        try:
            message = f"{target}: {type(error).__name__}: {error}"
        except Exception:
            message = f"{target}: unprintable telemetry error"
        try:
            self._write_errors.append(message)
        except Exception:
            return
        try:
            print(
                f"Telemetry WARNING (processing results remain authoritative): {message}",
                file=sys.stderr,
            )
        except Exception:
            pass

    def _interval_summary(self, wall_seconds: float) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for name, intervals in sorted(self._intervals.items()):
            if name.startswith("stage:"):
                continue
            cumulative = sum(_duration_seconds(interval) for interval in intervals)
            union = _union_seconds(intervals)
            summary[name] = {
                "count": len(intervals),
                "cumulative_seconds": cumulative,
                "union_seconds": union,
                "critical_path_percent": (union / wall_seconds * 100) if wall_seconds else 0.0,
            }

        jxl = self._intervals.get("jxl_service", [])
        for left_name, relationship_name in (
            ("engine_path", "engine_jxl_relationship"),
            ("gpu_inference", "gpu_jxl_relationship"),
        ):
            left = self._intervals.get(left_name, [])
            overlap = _overlap_seconds(left, jxl)
            busy_union = _union_seconds([*left, *jxl])
            summary[relationship_name] = {
                "left_interval": left_name,
                "overlap_seconds": overlap,
                "busy_union_seconds": busy_union,
                "busy_union_critical_path_percent": (
                    busy_union / wall_seconds * 100 if wall_seconds else 0.0
                ),
            }
        return summary

    def finalize(
        self,
        *,
        status: str,
        wall_seconds: float,
        summary: Mapping[str, Any] | None,
        context: Mapping[str, Any],
        error: BaseException | None = None,
    ) -> int:
        try:
            return self._finalize(
                status=status,
                wall_seconds=wall_seconds,
                summary=summary,
                context=context,
                error=error,
            )
        except Exception as exc:
            self._record_write_error("job.json finalize", exc)
            return self.write_error_count

    def _finalize(
        self,
        *,
        status: str,
        wall_seconds: float,
        summary: Mapping[str, Any] | None,
        context: Mapping[str, Any],
        error: BaseException | None = None,
    ) -> int:
        if not self.enabled:
            return 0
        with self._lock:
            if self._pages_handle is not None:
                try:
                    self._pages_handle.flush()
                    self._pages_handle.close()
                except Exception as exc:
                    self._record_write_error("pages.jsonl close", exc)
                finally:
                    self._pages_handle = None

        report_summary = dict(summary) if summary is not None else None
        if report_summary is not None:
            report_summary["metrics_write_errors"] = self.write_error_count
        ended_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        report: dict[str, Any] = {
            "type": "waifuhat2x-job-metrics",
            "schema_version": METRICS_SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": status,
            "started_utc": self.started_utc,
            "ended_utc": ended_utc,
            "wall_seconds": wall_seconds,
            "context": dict(context),
            "summary": report_summary,
            "pages_written": self._pages_written,
            "timing": {
                "clock": "perf_counter_ns",
                "stage_spans": dict(self._stage_spans),
                "cumulative_service_seconds": dict(sorted(self._service_totals.items())),
                "interval_summary": self._interval_summary(wall_seconds),
                "semantics": {
                    "cumulative_service_seconds": (
                        "Sum of service durations; concurrent work can make this exceed wall time."
                    ),
                    "union_seconds": (
                        "Wall-clock occupancy after merging overlapping monotonic intervals."
                    ),
                    "critical_path_percent": "Interval union divided by job wall time.",
                    "gpu_phase_service_seconds": (
                        "HIP Event phase proportions scaled to the synchronized perf_counter "
                        "engine wall interval; raw device-event values and the scale factor "
                        "remain in each page's details."
                    ),
                    "gpu_inference_interval": (
                        "Synchronized perf_counter window from the first GPU transfer through "
                        "the final device-to-host completion; it is occupancy, not kernel busy time."
                    ),
                },
            },
            "write_errors": list(self._write_errors),
        }
        if error is not None:
            report["error"] = {"type": type(error).__name__, "message": str(error)}

        assert self.run_dir is not None
        destination = self.run_dir / "job.json"
        temporary = destination.with_suffix(".json.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except Exception as exc:
            self._record_write_error("job.json", exc)
            try:
                temporary.unlink(missing_ok=True)
            except Exception as cleanup_exc:
                self._record_write_error("job.json cleanup", cleanup_exc)
        return self.write_error_count
