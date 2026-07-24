# Model directory

No model file is stored in this repository. `install.bat` obtains the declared
files from the sources in `../model_sources.toml`, verifies SHA-256, and writes
them under this directory:

- `hat/Real_HAT_GAN_SRx4.pth` (automatic normal route)
- `hat/Real_HAT_GAN_SRx4_sharper.pth` (automatic sharper route)
- `hat/HAT-S_SRx2.pth` (explicit fallback)
- `hat/HAT-S_SRx4.pth` (explicit fallback)

`checksums.sha256` is generated locally and ignored by Git. Do not commit a
model, archive, partial download, or checksum file.

The HAT project is linked from [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
The upstream code license does not automatically grant checkpoint redistribution
rights. Review the terms for each weight yourself, download only from trusted
sources, and verify the expected hash before allowing PyTorch to deserialize it.
