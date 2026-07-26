from __future__ import annotations

from pathlib import Path
import tomllib

from waifuhat2x import __version__
from waifuhat2x.config import load_config


ROOT = Path(__file__).resolve().parents[1]
RETIRED_RESEARCH_FILES = {
    "scripts/attest_e2e_representative.py",
    "scripts/benchmark_manifest_eager.py",
    "scripts/benchmark_pipeline_e2e.py",
    "scripts/build_blind_boundary_rois.py",
    "scripts/build_blind_comparison.py",
    "scripts/build_overlap_boundary_annotations.py",
    "scripts/build_representative_manifest.py",
    "scripts/profile_native_rocm.py",
    "scripts/render_tile_boundary_grids.py",
    "scripts/research_runtime.py",
    "scripts/research_utils.py",
    "scripts/reveal_overlap_quality_gate.py",
    "scripts/verify_gpu_phase_timing.py",
}


def test_public_example_configuration_preserves_sources() -> None:
    config = load_config(ROOT / "config.example.toml")

    assert config.paths.input == (ROOT / "Input").resolve()
    assert config.paths.output == (ROOT / "Output").resolve()
    assert config.output.mode == "mirror"
    assert config.output.existing_jxl_policy == "error"
    assert config.output.allow_lossy_replace is False
    assert config.output.allow_metadata_loss is False


def test_package_and_public_metadata_share_a_version() -> None:
    with (ROOT / "pyproject.toml").open("rb") as source:
        metadata = tomllib.load(source)

    assert metadata["project"]["version"] == __version__ == "1.0.0"


def test_public_release_omits_retired_research_surface() -> None:
    for relative in RETIRED_RESEARCH_FILES:
        assert not (ROOT / relative).exists()

    with (ROOT / "pyproject.toml").open("rb") as source:
        metadata = tomllib.load(source)

    dependencies = "\n".join(metadata["project"]["dependencies"])
    assert "triton" in dependencies.lower()
    for package_file in (ROOT / "src").rglob("*.py"):
        assert "import triton" not in package_file.read_text(encoding="utf-8")
