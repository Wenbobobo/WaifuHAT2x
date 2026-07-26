# Performance Protocol

## Current baseline

The accepted Real-HAT path is BF16 eager with adaptive tile candidates
`[256, 320]`, overlap 16, batch 1, device-side page assembly, and one GPU
execution lane. Both normal and sharper x4 models stay resident. JPEG XL work
is allowed to overlap with model work, but full-page wall time remains the
decision metric.

Private representative runs showed that forward work dominates the critical
path; JPEG XL encoding, transfer, resize, storage, and extra host-side
parallelism do not justify invasive optimization by themselves. This is a
design conclusion, not a guarantee for every GPU or runtime.

## What has already failed its gate

The following directions were evaluated and are not production defaults:

| Direction | Reason it remains disabled |
| --- | --- |
| `torch.compile` | Regressed wall time or violated output correctness. |
| Cross-page batching and dual streams | Did not meet throughput and deterministic-output gates. |
| GPU resize and decode prefetch | Full-page upside was below the acceptance threshold. |
| ONNX/DirectML and MIGraphX | Did not meet correctness or throughput requirements. |
| hipBLASLt preference | Regressed the tested route/tile cells and changed hashes. |
| WSL profiler-led custom kernels | No trustworthy trace/counter evidence was available. |
| FP8, INT4, KNOD, handwritten convolution/GEMM | Not justified by the hardware path or Amdahl analysis. |

Do not remove a rejection merely because a microbenchmark improves. A new
runtime, driver, model revision, or native profiler result may reopen one
candidate, but it starts as a new experiment.

## Experiment stages

Use only copied, isolated images that can remain private.

1. **Micro:** normal/sharper x tile 256/320, two warmups and five steady runs.
   Stop if a key cell regresses by more than 2% or projected full-page gain is
   below 3%.
2. **Canary:** 12 representative pages in AB/BA order for three paired rounds.
   Stop after two slower candidate rounds. Report paired ratios and route/tile
   strata, not only an aggregate mean.
3. **Final:** fixed 30-page set, three paired rounds after warmup. Both sides
   need coefficient of variation below 3%, with peak reserved VRAM under the
   chosen device limit.

This public tree no longer ships the private research harnesses used for the
original sweep. Keep any new runner outside Git until it has a clear production
purpose, and always include a production-process gate that refuses to run while
a watchdog or worker is active.

## Acceptance gates

An environment-variable or low-maintenance backend change needs at least 3%
full-page wall-clock improvement in the final gate. A graph, runtime migration,
precision change, or custom operator needs at least 5%. No route/tile stratum
may regress by more than 2%.

Cache, graph, or scheduling changes must produce exactly the same final uint8
hashes as eager. A library/layout change may differ only when results are
repeatable, p95 absolute pixel difference is zero, maximum difference is at
most one, and PSNR is at least 90 dB. Precision or fused-attention changes also
need blind review of text, screentones, diagonal lines, and at least 60 tile
boundary ROIs. Any visible defect rejects the candidate.

## Custom kernels and profiling

Do not write convolution, GEMM, or general linear kernels for this project.
Before considering a fused attention path, a trusted native Linux profiler must
show that the target accounts for at least 15% of end-to-end time or 20% of GPU
time. The order of investigation is: existing ATen/Inductor behavior, a
semantically equivalent high-level attention path, a narrow Triton prototype,
then HIP/CK only if the projected page-level gain remains material. Production
code does not import Triton or ship custom Triton kernels; the pinned ROCm
Triton wheel remains only because the ROCm PyTorch wheel set needs it to
resolve reproducibly.

Never publish copied pages, raw profiler traces, model files, output hashes, or
machine fingerprints with a performance report. Publish normalized aggregate
metrics and the exact acceptance conditions instead.
