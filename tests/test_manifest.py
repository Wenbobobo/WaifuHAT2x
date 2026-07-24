from pathlib import Path
import tomllib


def test_daily_manifest_contains_real_hat_pair_and_hat_s_rollback() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "model_sources.toml").open("rb") as handle:
        manifest = tomllib.load(handle)

    assert set(manifest) == {"models"}
    assert set(manifest["models"]) == {
        "hat_s_x2",
        "hat_s_x4",
        "real_hat_gan_x4",
        "real_hat_gan_sharper_x4",
    }
    assert manifest["models"]["hat_s_x2"]["filename"] == "hat/HAT-S_SRx2.pth"
    assert manifest["models"]["hat_s_x4"]["filename"] == "hat/HAT-S_SRx4.pth"
    assert manifest["models"]["hat_s_x2"]["sha256"]
    assert manifest["models"]["hat_s_x4"]["sha256"]
    assert (
        manifest["models"]["real_hat_gan_x4"]["filename"]
        == "hat/Real_HAT_GAN_SRx4.pth"
    )
    assert (
        manifest["models"]["real_hat_gan_sharper_x4"]["filename"]
        == "hat/Real_HAT_GAN_SRx4_sharper.pth"
    )
    assert manifest["models"]["real_hat_gan_x4"]["sha256"] == (
        "f5b1e3bbbb05147ca2beefcc715279cb647d7976cbda67d62ea7e6e20d5ffcc7"
    )
    assert manifest["models"]["real_hat_gan_sharper_x4"]["sha256"] == (
        "5800b67136006eb8cab3b4ed7c8d73b6a195bb18e6cc709b674f9aa069c00271"
    )
