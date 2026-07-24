from pathlib import Path

import pytest

from waifuhat2x.models import available_scales, choose_model, real_hat_variant


def _install_hat_s(root: Path) -> tuple[Path, Path]:
    hat_root = root / "hat"
    hat_root.mkdir()
    x2 = hat_root / "HAT-S_SRx2.pth"
    x4 = hat_root / "HAT-S_SRx4.pth"
    x2.touch()
    x4.touch()
    return x2, x4


@pytest.mark.parametrize("profile", ["hat-auto", "hat-fast", "hat-s"])
@pytest.mark.parametrize("native_scale", [2, 4])
@pytest.mark.parametrize("grayscale", [False, True])
def test_hat_s_rollback_aliases_choose_matching_native_scale(
    tmp_path: Path, profile: str, native_scale: int, grayscale: bool
) -> None:
    x2, x4 = _install_hat_s(tmp_path)

    choice = choose_model(
        tmp_path,
        profile,
        image_height=1200,
        grayscale=grayscale,
        native_scale=native_scale,
    )

    assert choice.path == {2: x2, 4: x4}[native_scale]
    assert choice.label == f"HAT-S-x{native_scale}"


def test_hat_s_available_scales_only_reports_installed_weights(tmp_path: Path) -> None:
    hat_root = tmp_path / "hat"
    hat_root.mkdir()
    (hat_root / "HAT-S_SRx2.pth").touch()

    for profile in ("hat-auto", "hat-fast", "hat-s"):
        assert available_scales(tmp_path, profile, grayscale=True) == (2,)
    with pytest.raises(FileNotFoundError, match="HAT-S_SRx4.pth"):
        choose_model(tmp_path, "hat-auto", 1200, grayscale=True, native_scale=4)


def test_model_checkpoints_are_not_discovered_recursively(tmp_path: Path) -> None:
    nested_hat = tmp_path / "hat" / "archive"
    nested_hat.mkdir(parents=True)
    (nested_hat / "HAT-S_SRx2.pth").touch()
    (nested_hat / "Real_HAT_GAN_SRx4.pth").touch()
    (nested_hat / "Real_HAT_GAN_SRx4_sharper.pth").touch()

    assert available_scales(tmp_path, "hat-auto", grayscale=False) == ()
    with pytest.raises(FileNotFoundError, match="HAT-S_SRx2.pth"):
        choose_model(tmp_path, "hat-auto", 1200, grayscale=False, native_scale=2)
    with pytest.raises(FileNotFoundError, match="Real_HAT_GAN_SRx4.pth"):
        available_scales(tmp_path, "real-hat-auto", grayscale=False)


@pytest.mark.parametrize("legacy_profile", ["manga-auto", "hat", "hat-balanced", "hat-max", "hat-l"])
def test_removed_profiles_are_rejected(tmp_path: Path, legacy_profile: str) -> None:
    with pytest.raises(ValueError, match="Unknown processing profile"):
        available_scales(tmp_path, legacy_profile, grayscale=False)
    with pytest.raises(ValueError, match="Unknown processing profile"):
        choose_model(tmp_path, legacy_profile, 1200, grayscale=False, native_scale=2)


def test_custom_checkpoint_paths_are_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "custom_SRx4.pth"
    checkpoint.touch()

    with pytest.raises(ValueError, match="Unknown processing profile"):
        available_scales(tmp_path, str(checkpoint), grayscale=False)
    with pytest.raises(ValueError, match="Unknown processing profile"):
        choose_model(tmp_path, str(checkpoint), 1200, grayscale=False, native_scale=4)


def _install_real_hat_pair(root: Path) -> tuple[Path, Path]:
    hat_root = root / "hat"
    hat_root.mkdir()
    normal = hat_root / "Real_HAT_GAN_SRx4.pth"
    sharper = hat_root / "Real_HAT_GAN_SRx4_sharper.pth"
    normal.touch()
    sharper.touch()
    return normal, sharper


@pytest.mark.parametrize(
    ("source_short_edge", "variant"),
    [(999, "normal"), (1000, "sharper"), (1001, "sharper")],
)
@pytest.mark.parametrize("grayscale", [False, True])
def test_real_hat_auto_routes_by_source_short_edge(
    tmp_path: Path, source_short_edge: int, variant: str, grayscale: bool
) -> None:
    normal, sharper = _install_real_hat_pair(tmp_path)

    assert available_scales(tmp_path, "real-hat-auto", grayscale) == (4,)
    choice = choose_model(
        tmp_path,
        "real-hat-auto",
        image_height=1200,
        grayscale=grayscale,
        native_scale=4,
        source_short_edge=source_short_edge,
        real_hat_sharper_min_short_edge=1000,
    )

    assert choice.path == {"normal": normal, "sharper": sharper}[variant]
    assert choice.label == f"Real-HAT-GAN-x4-{variant}"
    assert real_hat_variant(source_short_edge, 1000) == variant


@pytest.mark.parametrize(
    "missing_filename",
    ["Real_HAT_GAN_SRx4.pth", "Real_HAT_GAN_SRx4_sharper.pth"],
)
def test_real_hat_auto_requires_both_weights(
    tmp_path: Path, missing_filename: str
) -> None:
    _install_real_hat_pair(tmp_path)
    (tmp_path / "hat" / missing_filename).unlink()

    with pytest.raises(FileNotFoundError, match=missing_filename):
        available_scales(tmp_path, "real-hat-auto", grayscale=False)


def test_real_hat_auto_rejects_missing_route_context_and_x2(tmp_path: Path) -> None:
    _install_real_hat_pair(tmp_path)

    with pytest.raises(ValueError, match="requires source_short_edge"):
        choose_model(tmp_path, "real-hat-auto", 1200, False, 4)
    with pytest.raises(ValueError, match="supports native scale 4"):
        choose_model(
            tmp_path,
            "real-hat-auto",
            1200,
            False,
            2,
            source_short_edge=999,
        )
