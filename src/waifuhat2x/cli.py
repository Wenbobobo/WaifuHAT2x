from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import load_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Incremental manga super-resolution on ROCm")
    parser.add_argument("--config", default="config.toml", help="Path to TOML configuration")
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        help=(
            "Write versioned pages.jsonl and job.json into a new run directory "
            "under this path"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        summary = run_pipeline(load_config(args.config), metrics_dir=args.metrics_dir)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(
        "Done: "
        f"processed={summary.processed}, skipped={summary.skipped}, "
        f"metadata={summary.copied}, failed={summary.failed}, "
        f"ignored={summary.ignored}, jxl_skipped={summary.jxl_skipped}, sr={summary.sr_pages}, "
        f"transcoded={summary.transcoded_pages}, replaced={summary.replaced_sources}, "
        f"legacy_jxl_adopted={summary.existing_jxl_adopted}, "
        f"existing_jxl_replaced={summary.existing_jxl_replaced}, "
        f"external_jxl_recoveries={summary.external_jxl_recoveries}, "
        f"deferred={summary.deferred}, "
        f"target_unmet={summary.target_unmet}, inference={summary.inference_seconds:.1f}s, "
        f"postprocess={summary.postprocess_seconds:.1f}s, "
        f"encoding={summary.encoding_seconds:.1f}s, output={summary.output_bytes / 1024**2:.1f}MiB, "
        f"wall={summary.wall_seconds:.1f}s, "
        f"metrics={summary.metrics_directory or 'disabled'}, "
        f"metrics_write_errors={summary.metrics_write_errors}"
    )
    if summary.failed or summary.deferred:
        raise SystemExit(2)
