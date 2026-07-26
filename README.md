# WaifuHAT2x

Local, transaction-safe manga super-resolution and JPEG XL conversion for an
AMD ROCm PyTorch runtime. It is designed for repeatable personal libraries,
not for cloud processing or a generic image enhancement service.

[中文说明](README.zh-CN.md) | [Operations](docs/OPERATIONS.md) |
[ROCm runtime notes](docs/ROCM_RUNTIME.md) | [Performance protocol](docs/PERFORMANCE.md)

## What it does

1. Reads non-JXL image pages from a local input tree and applies EXIF
   orientation before planning.
2. Uses the Real-HAT x4 **normal** checkpoint when the original short edge is
   below 1000 pixels, or **sharper** at 1000 pixels and above.
3. Resizes the x4 result in linear light to the configured target short edge.
4. Encodes JPEG XL and verifies it before making it visible.

The default `mirror` configuration preserves all source files and writes to a
separate output directory. JPEG XL is never fed back into super-resolution, so
the pipeline does not accidentally apply a second enhancement pass.

## Scope and prerequisites

- A Python 3.12 environment with a ROCm-compatible AMD PyTorch build and a
  BF16-capable GPU exposed to PyTorch.
- `uv`, ROCm, and JPEG XL tools inside the Linux environment that will run the
  workload. The Windows launchers use WSL, while the shell scripts also work in
  a suitably configured Linux environment.
- Enough free storage for isolated output and temporary JPEG XL candidates.

This repository does not include image data, model weights, credentials, or
machine-specific configuration. It has been exercised on a narrow AMD ROCm
reference stack; treat any other OS, driver, GPU, or library combination as a
new validation target rather than a supported guarantee.

## Quick start

Clone the repository, then create an ignored local configuration:

```powershell
Copy-Item config.example.toml config.toml
# Edit config.toml: choose disjoint relative or absolute input/output/model roots.

# Optional when the default WSL distribution is not the one with ROCm.
$env:WAIFUHAT_WSL_DISTRO = "YourDistro"

.\install.bat
.\inspect_workload.bat
.\run_upscale.bat
```

`inspect_workload.bat` is read-only: it scans headers, reports routing and
workload counts, and does not load a model or start GPU work. The first real
run should remain in `mirror` mode and be checked visually before any
destructive setting is considered.

To keep several local configurations, copy the example in full and select one
explicitly:

```powershell
$env:WAIFUHAT_CONFIG = "config.local.toml"
.\inspect_workload.bat
```

The launchers honor `WAIFUHAT_WSL_DISTRO` and `WAIFUHAT_CONFIG`; neither value
is committed. On a runtime that needs a vendor-specific shell setup, point
`WAIFUHAT_ROCM_ENV` at that setup script before installation or execution.

## Model routing

| EXIF-normalized original short edge | Model | Output behavior |
| --- | --- | --- |
| `< 1000` | Real-HAT-GAN x4 normal | x4 then linear-light resize |
| `>= 1000` and needs SR | Real-HAT-GAN x4 sharper | x4 then linear-light resize |
| target already met or a safety limit applies | no SR | format policy only |

The 999/1000 boundary is intentional. Both Real-HAT checkpoints are validated
before work begins and stay resident so page ordering does not cause repeated
loads. HAT-S x2/x4 remains an explicit fallback, not part of the automatic
route.

## Models and provenance

Weights are intentionally absent from Git. `install.bat` downloads the four
declared checkpoints using [model_sources.toml](model_sources.toml) and verifies
their SHA-256 values. The manifest includes the Real-HAT normal/sharper pair
and HAT-S x2/x4 fallback pair.

Download weights only from the listed upstream sources, verify every checksum,
and review the upstream terms yourself. A PyTorch checkpoint is executable
deserialization input; never substitute an untrusted file. See
[models/README.md](models/README.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Replace mode is opt-in

`replace` deletes the original non-JXL source only after the candidate JPEG XL
has passed decoding, dimension, hash, and final-file verification. It is still
destructive. Use it only after a successful mirror run and a backup.

Change all of these local configuration settings deliberately; do not commit
the resulting file:

```toml
[output]
mode = "replace"
existing_jxl_policy = "replace"
allow_lossy_replace = true
allow_metadata_loss = true
```

If a run exits nonzero, retain the state journal, worklist, `.part` files,
source image, and candidate. Confirm no worker is still running, then rerun the
same configuration. The recovery system prefers retaining both files over
guessing which one is safe to delete. Full procedure: [Operations](docs/OPERATIONS.md).

## Performance notes

The sustained bottleneck is model forward work, not JPEG XL or file transfers.
The production settings are BF16 eager, adaptive tiles `[256, 320]`, overlap
16, batch 1, and one GPU execution lane. Prior experiments rejected compile,
cross-page batching, dual streams, GPU resize, ONNX/DirectML, MIGraphX, and a
hipBLASLt preference under their quality or full-page wall-clock gates.

The public tree intentionally keeps production code and the small isolated soak
attestation tool only. Private research runners, blind-review builders,
profiler wrappers, and raw benchmark tooling were removed after the accepted
configuration stabilized. For a new runtime or backend candidate, use your own
copied pages outside Git, screen with a small representative set, and promote
only after the acceptance criteria in [Performance protocol](docs/PERFORMANCE.md).

## Development

Hosted CI is CPU-only and deliberately does not download weights, run ROCm, or
exercise `replace`. Before submitting a change, run:

```bash
uv lock --check
python -m ruff check src tests scripts
python -m compileall -q src scripts tests
python scripts/check_public_tree.py
python -m pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for privacy rules, performance gates, and
pull-request expectations. Report security or data-safety concerns privately as
described in [SECURITY.md](SECURITY.md).

## License

Project-authored code is licensed under [Apache-2.0](LICENSE). Dependencies and
model weights keep their own licenses and terms; this repository does not grant
rights to redistribute the weights.
