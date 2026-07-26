# Release Checklist

Use this checklist before publishing a branch, tag, or release artifact. It is
designed for the public repository: production manga folders, model weights,
raw telemetry, and local machine details stay outside Git.

## Safety gate

1. Confirm no production watchdog or worker is running before changing code,
   configuration, model pins, or Git refs.
2. Keep `config.toml`, model files, official manga input/output roots, recovery
   journals, `.part` files, and raw metrics out of the release.
3. Do not run release validation against a real library. Use synthetic fixtures
   or copied disposable pages only.

Suggested process gate: run a WSL process listing for `run_with_watchdog.py`
and `python -m waifuhat2x`; it must print no active worker or watchdog lines.

## Local checks

Run from a clean checkout:

- `git status --short --branch`
- `uv lock --check` from the configured WSL runtime environment
- `./scripts/project_python.sh scripts/check_public_tree.py`
- `./scripts/project_python.sh -m ruff check src tests scripts`
- `./scripts/project_python.sh -m compileall -q src scripts tests`
- `./scripts/project_python.sh -m pytest -q`
- `git diff --check`

Before committing, scan for local paths, credentials, private image names, and
large tracked artifacts. The public-tree check covers the common cases; use an
additional search when documentation or workflow files change.

## Dependency and security triage

1. Review Dependabot alerts before publishing.
2. Patch optional helper dependencies promptly when the update does not change
   production inference behavior.
3. Keep runtime-bound ROCm PyTorch alerts open unless AMD provides a compatible
   patched wheel tuple and the full quality/performance gate passes.
4. Do not dismiss runtime-bound alerts merely to make the repository appear
   clean; document the constraint instead.

## Remote checks

After pushing:

- Inspect the latest GitHub Actions runs with `gh run list`.
- Wait for the latest CI run to finish with `gh run watch`.
- Recheck Dependabot alerts with the GitHub API or repository security UI.

The latest CI run must pass. Expected remaining alerts must be limited to
documented runtime-bound ROCm wheel issues.

## Tag and release hygiene

1. Tag only commits that passed local and remote checks.
2. Do not move public tags unless a published tag is provably unsafe.
3. Do not include model checkpoints, sample pages, raw benchmark output,
   profiler traces, hashes of private pages, or screenshots in release notes.
4. Confirm `git status --short --branch` is clean after pushing.
