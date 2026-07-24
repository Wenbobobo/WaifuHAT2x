from pathlib import Path

import pytest

from waifuhat2x.config import ProcessingConfig, load_config


def test_load_config_resolves_relative_paths(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[paths]
input = "in"
output = "out"
models = "weights"
[processing]
tile = 256
overlap = 32
""",
        encoding="utf-8",
    )
    loaded = load_config(config)
    assert loaded.paths.input == (tmp_path / "in").resolve()
    assert loaded.paths.output == (tmp_path / "out").resolve()
    assert loaded.processing.tile == 256
    assert loaded.processing.hat_tile == 256
    assert loaded.processing.hat_tile_candidates == (256,)
    assert loaded.processing.profile == "hat-auto"
    assert loaded.processing.real_hat_sharper_min_short_edge == 1000
    assert loaded.processing.batch_tiles == 1
    assert loaded.processing.device_assembly is True
    assert loaded.processing.model_cache_size == 2


def test_hat_tile_candidates_are_normalized_to_an_immutable_tuple(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[processing]
hat_tile = 256
hat_tile_candidates = [256, 320]
hat_overlap = 32
""",
        encoding="utf-8",
    )

    loaded = load_config(config)

    assert loaded.processing.hat_tile_candidates == (256, 320)
    assert isinstance(loaded.processing.hat_tile_candidates, tuple)


def test_direct_processing_config_keeps_legacy_hat_tile_as_the_fixed_candidate() -> None:
    processing = ProcessingConfig(hat_tile=320)

    assert processing.hat_tile == 320
    assert processing.hat_tile_candidates == (320,)


@pytest.mark.parametrize(
    ("candidate_value", "message"),
    [
        ("[]", "non-empty TOML array"),
        ("256", "non-empty TOML array"),
        ("[320, 256]", "must start with processing.hat_tile"),
        ("[256, 256]", "must contain unique values"),
        ("[256, 318]", "positive multiples of 16"),
        ("[256, 320.0]", "contain only integers"),
        ("[256, true]", "contain only integers"),
    ],
)
def test_hat_tile_candidates_reject_ambiguous_or_invalid_values(
    tmp_path: Path, candidate_value: str, message: str
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[processing]
hat_tile = 256
hat_tile_candidates = {candidate_value}
hat_overlap = 32
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_config(config)


def test_windows_drive_path_is_converted_under_wsl(tmp_path: Path) -> None:
    from waifuhat2x import config as module

    assert module._resolve(tmp_path, "Z:/sample/input") == Path("/mnt/z/sample/input")


def test_replace_mode_requires_explicit_lossy_consent(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[output]
mode = "replace"
format = "jxl"
[jxl]
distance = 0.3
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Lossy JXL replacement"):
        load_config(config)


def test_lossless_replace_mode_is_accepted(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[output]
mode = "replace"
format = "jxl"
[jxl]
distance = 0
""",
        encoding="utf-8",
    )
    assert load_config(config).output.mode == "replace"


def test_replace_mode_cannot_disable_decode_verification(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[output]
mode = "replace"
format = "jxl"
[jxl]
distance = 0
verify_decode = false
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires jxl.verify_decode = true"):
        load_config(config)


def test_existing_jxl_replace_policy_is_replace_mode_only(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[output]
mode = "replace"
format = "jxl"
existing_jxl_policy = "replace"
[jxl]
distance = 0
""",
        encoding="utf-8",
    )
    assert load_config(config).output.existing_jxl_policy == "replace"

    config.write_text(
        """
[output]
mode = "mirror"
existing_jxl_policy = "replace"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only available in replace mode"):
        load_config(config)


@pytest.mark.parametrize("threshold", [0, 1600, 1601])
def test_real_hat_threshold_must_be_below_target(
    tmp_path: Path, threshold: int
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[processing]
profile = "real-hat-auto"
target_short_edge = 1600
real_hat_sharper_min_short_edge = {threshold}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="real_hat_sharper_min_short_edge"):
        load_config(config)


def test_real_hat_requires_x4_to_remain_available(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[processing]
profile = "real-hat-auto"
max_upscale_factor = 2
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_upscale_factor must be 4"):
        load_config(config)


def test_real_hat_requires_both_models_to_remain_resident(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[processing]
profile = "real-hat-auto"
model_cache_size = 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_cache_size must be at least 2"):
        load_config(config)
