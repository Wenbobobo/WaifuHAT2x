from pathlib import Path

import numpy as np
from PIL import Image

from waifuhat2x.images import IMAGE_EXTENSIONS, is_grayscale, output_path_for, resize_linear_light


def test_grayscale_detection_accepts_small_chroma_noise() -> None:
    pixels = np.full((32, 32, 3), 127, dtype=np.uint8)
    pixels[0, 0] = [127, 129, 126]
    assert is_grayscale(Image.fromarray(pixels), tolerance=3)


def test_grayscale_detection_accepts_sparse_webp_edge_fringe() -> None:
    pixels = np.full((100, 100, 3), 127, dtype=np.uint8)
    pixels[:2, :, :] = [124, 134, 126]
    assert is_grayscale(Image.fromarray(pixels), tolerance=3)


def test_color_is_not_grayscale() -> None:
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[:, :, 0] = 255
    assert not is_grayscale(Image.fromarray(pixels), tolerance=3)


def test_colored_region_is_not_mistaken_for_codec_noise() -> None:
    pixels = np.full((100, 100, 3), 127, dtype=np.uint8)
    pixels[:10, :, :] = [255, 0, 0]
    assert not is_grayscale(Image.fromarray(pixels), tolerance=3)


def test_one_percent_saturated_stamp_is_not_grayscale() -> None:
    pixels = np.full((100, 100, 3), 255, dtype=np.uint8)
    pixels[45:55, 45:55, :] = [220, 20, 20]

    assert not is_grayscale(Image.fromarray(pixels), tolerance=3)


def test_tiny_saturated_stamp_on_large_page_is_not_grayscale() -> None:
    pixels = np.full((1500, 1000, 3), 255, dtype=np.uint8)
    pixels[730:770, 480:520, :] = [220, 20, 20]

    assert not is_grayscale(Image.fromarray(pixels), tolerance=3)


def test_dense_moderate_chroma_detail_is_not_grayscale() -> None:
    pixels = np.full((200, 200, 3), 127, dtype=np.uint8)
    pixels[50:53, 50:53, :] = [100, 140, 105]

    assert not is_grayscale(Image.fromarray(pixels), tolerance=3)


def test_four_extreme_chroma_pixels_are_not_grayscale() -> None:
    pixels = np.full((200, 200, 3), 127, dtype=np.uint8)
    pixels[50:52, 50:52, :] = [220, 20, 20]

    assert not is_grayscale(Image.fromarray(pixels), tolerance=3)


def test_sparse_strong_codec_chroma_noise_is_tolerated() -> None:
    pixels = np.full((100, 100, 3), 127, dtype=np.uint8)
    pixels[::20, ::20, :] = [112, 145, 124]

    assert is_grayscale(Image.fromarray(pixels), tolerance=3)


def test_output_path_preserves_tree() -> None:
    source = Path("/in/title/chapter/001.png")
    assert output_path_for(source, Path("/in"), Path("/out"), "webp") == Path(
        "/out/title/chapter/001.webp"
    )


def test_linear_light_resize_shape() -> None:
    source = np.arange(64, dtype=np.uint8).reshape(8, 8)
    result = resize_linear_light(source, 4, 4)
    assert result.shape == (4, 4)
    assert result.dtype == np.uint8


def test_only_supported_raster_extensions_enter_pipeline() -> None:
    assert {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"} == IMAGE_EXTENSIONS
    assert ".jxl" not in IMAGE_EXTENSIONS
    assert ".cbz" not in IMAGE_EXTENSIONS
