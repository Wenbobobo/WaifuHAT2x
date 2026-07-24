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
