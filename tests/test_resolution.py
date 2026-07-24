from waifuhat2x.images import plan_resolution


def plan(width: int, height: int, scales: tuple[int, ...] = (2, 4)):
    return plan_resolution(width, height, 1600, 3200, scales, 4, 6400, 24.0)


def test_selects_smallest_native_scale_then_hits_exact_short_edge() -> None:
    two = plan(1000, 1500)
    assert two.upscale and two.native_scale == 2
    assert (two.output_width, two.output_height) == (1600, 2400)

    four = plan(799, 1200)
    assert four.upscale and four.native_scale == 4
    assert four.output_width == 1600


def test_real_hat_x4_is_always_run_before_resizing_to_target_short_edge() -> None:
    result = plan(1000, 1500, (4,))

    assert result.upscale and result.native_scale == 4
    assert (result.output_width, result.output_height) == (1600, 2400)
    assert result.reason == "4x then resize to target short edge"


def test_real_hat_extremely_small_input_stops_at_x4_and_reports_target_unmet() -> None:
    result = plan(300, 450, (4,))
    assert result.upscale and result.native_scale == 4
    assert (result.output_width, result.output_height) == (1200, 1800)
    assert "remains below target" in result.reason


def test_already_large_or_abnormally_long_image_skips_sr() -> None:
    assert plan(1599, 2399).upscale
    assert not plan(1600, 2400).upscale
    result = plan(600, 3201)
    assert not result.upscale
    assert "safety limit" in result.reason


def test_hat_x2_reports_target_unmet_instead_of_fake_interpolation() -> None:
    result = plan(600, 900, (2,))
    assert result.native_scale == 2
    assert (result.output_width, result.output_height) == (1200, 1800)
    assert "remains below target" in result.reason


def test_planned_extreme_output_is_transcode_only() -> None:
    result = plan(500, 3200)
    assert not result.upscale
    assert "planned output exceeds safety limit" in result.reason
