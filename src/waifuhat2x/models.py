from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_HAT_S_FILENAMES = {
    2: "HAT-S_SRx2.pth",
    4: "HAT-S_SRx4.pth",
}
_HAT_S_ROLLBACK_PROFILES = frozenset({"hat-auto", "hat-fast", "hat-s"})
_REAL_HAT_FILENAMES = {
    "normal": "Real_HAT_GAN_SRx4.pth",
    "sharper": "Real_HAT_GAN_SRx4_sharper.pth",
}


@dataclass(frozen=True)
class ModelChoice:
    path: Path
    label: str


def _required_checkpoint(root: Path, filename: str) -> Path:
    """Return the production checkpoint at its single declared location."""
    path = root / "hat" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required model not found: {path}")
    return path


def _hat_s_model(root: Path, native_scale: int) -> ModelChoice:
    try:
        filename = _HAT_S_FILENAMES[native_scale]
    except KeyError as error:
        raise ValueError(f"HAT-S supports native scales 2 and 4, not {native_scale}") from error
    return ModelChoice(_required_checkpoint(root, filename), f"HAT-S-x{native_scale}")


def _hat_s_available_scales(root: Path) -> tuple[int, ...]:
    return tuple(
        scale
        for scale, filename in sorted(_HAT_S_FILENAMES.items())
        if (root / "hat" / filename).is_file()
    )


def _real_hat_models(root: Path) -> dict[str, Path]:
    """Resolve the atomic normal/sharper model set required by real-hat-auto."""
    return {
        variant: _required_checkpoint(root, filename)
        for variant, filename in _REAL_HAT_FILENAMES.items()
    }


def real_hat_variant(source_short_edge: int, sharper_min_short_edge: int = 1000) -> str:
    if source_short_edge < 1:
        raise ValueError("source_short_edge must be positive")
    if sharper_min_short_edge < 1:
        raise ValueError("sharper_min_short_edge must be positive")
    return "normal" if source_short_edge < sharper_min_short_edge else "sharper"


def available_scales(root: Path, profile: str, grayscale: bool) -> tuple[int, ...]:
    """Return native scales whose production checkpoint files are present."""
    del grayscale
    normalized = profile.lower()
    if normalized == "real-hat-auto":
        _real_hat_models(root)
        return (4,)
    if normalized in _HAT_S_ROLLBACK_PROFILES:
        return _hat_s_available_scales(root)
    raise ValueError(f"Unknown processing profile: {profile}")


def choose_model(
    root: Path,
    profile: str,
    image_height: int,
    grayscale: bool,
    native_scale: int = 2,
    *,
    source_short_edge: int | None = None,
    real_hat_sharper_min_short_edge: int = 1000,
) -> ModelChoice:
    del image_height, grayscale
    normalized = profile.lower()
    if normalized == "real-hat-auto":
        if native_scale != 4:
            raise ValueError(f"Real-HAT supports native scale 4, not {native_scale}")
        if source_short_edge is None:
            raise ValueError("real-hat-auto requires source_short_edge")
        models = _real_hat_models(root)
        variant = real_hat_variant(source_short_edge, real_hat_sharper_min_short_edge)
        return ModelChoice(models[variant], f"Real-HAT-GAN-x4-{variant}")
    if normalized in _HAT_S_ROLLBACK_PROFILES:
        return _hat_s_model(root, native_scale)
    raise ValueError(f"Unknown processing profile: {profile}")
