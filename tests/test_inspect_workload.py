from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image

from scripts.inspect_workload import main


def test_inspection_reports_real_hat_routes_and_switches(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    inputs = tmp_path / "input"
    models = tmp_path / "models" / "hat"
    inputs.mkdir()
    models.mkdir(parents=True)
    (models / "Real_HAT_GAN_SRx4.pth").touch()
    (models / "Real_HAT_GAN_SRx4_sharper.pth").touch()

    dimensions = [(999, 1200), (1000, 1200), (1001, 1200), (999, 1200)]
    for index, size in enumerate(dimensions, start=1):
        Image.new("L", size, 255).save(inputs / f"{index:02d}.png")

    config = tmp_path / "config.toml"
    config.write_text(
        """
[paths]
input = "input"
output = "output"
models = "models"

[processing]
profile = "real-hat-auto"
target_short_edge = 1600
real_hat_sharper_min_short_edge = 1000
max_upscale_factor = 4
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["inspect_workload.py", "--config", str(config), "--workers", "1"],
    )

    main()

    result = json.loads(capsys.readouterr().out)
    assert result["sr_total"] == 4
    assert result["hat_x2"] == 0
    assert result["hat_x4"] == 4
    assert result["real_hat_sharper_min_short_edge"] == 1000
    assert result["real_hat_normal"] == 2
    assert result["real_hat_sharper"] == 2
    assert result["real_hat_threshold_exact"] == 1
    assert result["real_hat_model_switches"] == 2
