# Security Policy

## Supported version

Security fixes are made against the latest `main` branch.

## Report privately

Use the repository's Private Security Advisories feature for vulnerabilities.
Do not put an exploit, a source image, a local configuration, or unredacted
logs in a public issue.

Report any issue that could:

- Delete, replace, or expose source images unexpectedly.
- Bypass JPEG XL verification, transaction recovery, or model integrity checks.
- Write outside the configured input/output/model roots.
- Expose credentials, local paths, system metadata, or private image data.

Include the smallest reproducible synthetic fixture and redacted command output.
Maintainers will acknowledge a report, assess impact, and coordinate a fix
before public disclosure where practical.

## Dependency alerts on pinned runtimes

The ROCm PyTorch runtime is pinned as a tested GPU stack. Do not dismiss or
silently override GitHub security alerts for `torch`, `torchvision`, or related
ROCm wheels merely because a newer upstream release exists. A runtime upgrade
is accepted only after a compatible AMD wheel tuple is available and the image
quality and throughput gates pass.

Alerts in optional helper extras, such as model-download tooling, should be
patched promptly when the update does not alter production inference behavior.
