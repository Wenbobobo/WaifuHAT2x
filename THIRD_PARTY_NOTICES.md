# Third-Party Notices

WaifuHAT2x project-authored source code is licensed under Apache-2.0. This file
does not replace the licenses or terms of its dependencies and model weights.

- [HAT](https://github.com/XPixelGroup/HAT) provides the upstream HAT and
  Real-HAT architecture and publishes its source under Apache-2.0.
- [Spandrel](https://github.com/chaiNNer-org/spandrel), [PyTorch](https://pytorch.org/),
  [TorchVision](https://pytorch.org/vision/), [Triton](https://github.com/triton-lang/triton),
  [NumPy](https://numpy.org/), [Pillow](https://python-pillow.org/), and
  [JPEG XL](https://github.com/libjxl/libjxl) are used under their respective
  upstream licenses.
- The Real-HAT and HAT-S checkpoint files are **not** included, mirrored, or
  licensed by this repository. `model_sources.toml` records download locations
  and SHA-256 values only. Downloading and using a weight is subject to its
  upstream terms; users must review those terms and verify the checksum.

PyTorch checkpoint formats can be unsafe when loaded from an untrusted source.
Only download models from a trusted upstream and verify them before use.
