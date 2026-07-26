# Contributing

Thanks for helping make the pipeline safer, clearer, and more reproducible.

## Scope

This project is a local AMD ROCm workflow. Changes must preserve its two hard
boundaries: source images remain untouched in `mirror` mode, and a `replace`
transaction may remove a source only after a verified JPEG XL output exists.

Do not commit any of the following:

- Manga pages, screenshots, model checkpoints, or downloaded archives.
- Local `config.toml`, `.env` files, recovery journals, metrics, or benchmark
  output.
- Absolute paths, account names, hostnames, hardware serials, raw profiler
  traces, or unredacted terminal logs.

Use synthetic fixtures or a copied, isolated representative set that remains
outside Git.

## Development checks

Run these from a configured project environment before opening a pull request:

```bash
uv lock --check
python -m ruff check src tests scripts
python -m compileall -q src scripts tests
python scripts/check_public_tree.py
python -m pytest -q
```

Hosted CI is CPU-only. It verifies unit behavior and public-tree hygiene; it
does not download weights, run ROCm, or execute source-deleting workflows.

## Performance changes

Do not promote a faster microbenchmark by itself. Use a copied representative
screen first and a fixed 30-page final gate. Report paired full-page wall time,
route-by-tile breakdown, peak reserved VRAM, deterministic output checks, and
blind ROI review for any numerical change.

The public repository does not carry the retired private benchmark harnesses or
profiler wrappers. If a new performance direction is worth pursuing, keep raw
artifacts outside Git and add only the smallest reviewed production surface
after the gate passes. Do not add custom kernels unless a trusted native
profiler first proves an Amdahl-worthy target.

## Pull requests

Keep pull requests narrow, describe user-visible behavior, and add a regression
test for a bug fix. Configuration defaults must remain safe for a new clone:
`mirror` output, a separate output root, and no bundled media or checkpoints.
