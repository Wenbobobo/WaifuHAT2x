from __future__ import annotations

from pathlib import Path
import tomllib

from waifuhat2x import __version__
from waifuhat2x.config import load_config


ROOT = Path(__file__).resolve().parents[1]


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
