# ROCm Runtime Notes

## Support boundary

WaifuHAT2x pins a Python 3.12 AMD ROCm PyTorch wheel in `uv.lock`. The project
is intentionally conservative: it supports a runtime only after that runtime
can expose a compatible AMD GPU to PyTorch, run the BF16 smoke check in
`scripts/install_wsl.sh`, and pass the image-quality and throughput gates.

The public repository does not claim that every ROCm release, driver, WSL
distribution, or AMD GPU will work. Do not change a working production runtime
in place merely because a newer library is available.

Official references:

- [ROCm documentation](https://rocm.docs.amd.com/)
- [ROCm precision support](https://rocm.docs.amd.com/en/latest/reference/precision-support.html)
- [Radeon on WSL documentation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)

## Installation model

The Windows wrappers start a selected WSL distribution. Set this only in your
shell or local launch environment:

```powershell
$env:WAIFUHAT_WSL_DISTRO = "YourDistro"
```

Inside Linux, the installer requires `uv` and `rocminfo`, runs `uv lock --check`,
creates the locked virtual environment, verifies PyTorch GPU visibility, runs
small FP16/BF16 matrix smoke checks, installs the JPEG XL tools, and executes
the unit suite. It does not rewrite `uv.lock`.

If a vendor installation requires a shell setup script, set
`WAIFUHAT_ROCM_ENV` to that script's path before invoking the installer or a
project launcher. The project does not assume a specific distribution-local
setup file.

Keep the virtual environment and uv cache on a Linux filesystem with hardlink
support. Input and output may be on a mounted filesystem, but it is usually
slower for environment creation, cache management, and build artifacts.

## Runtime validation

Before a new stack is trusted for real images:

1. Run installation and unit tests.
2. Run a mirror-mode single-page check with synthetic or copied disposable
   input.
3. Confirm JXL encode/decode verification and source preservation.
4. Run a small copied quality screen and the 30-page performance gate described
   in [PERFORMANCE.md](PERFORMANCE.md).

Use a separate environment for a runtime upgrade. Record only redacted,
aggregate results in public discussions; raw logs can reveal machine paths,
device details, and image metadata.

## Profiler and kernel work

Installing a profiler does not prove that its trace or counters are accurate.
Before any profiler-led optimization, verify a nonzero GPU trace, stable
top-kernel ordering, counter availability, and agreement with HIP event timing.
If that capability gate fails, do not infer a kernel bottleneck or begin a
custom operator project. Prefer native Linux profiling for that investigation.
