#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
if [[ -n "${WAIFUHAT_ROCM_ENV:-}" ]]; then
  if [[ ! -r "$WAIFUHAT_ROCM_ENV" ]]; then
    echo "WAIFUHAT_ROCM_ENV is not readable: $WAIFUHAT_ROCM_ENV" >&2
    exit 1
  fi
  source "$WAIFUHAT_ROCM_ENV"
fi
source scripts/runtime_env.sh

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed in WSL. Install it from https://docs.astral.sh/uv/ first." >&2
  exit 1
fi
if ! command -v rocminfo >/dev/null 2>&1; then
  echo "ROCm is not installed in WSL; see docs/ROCM_RUNTIME.md." >&2
  exit 1
fi
echo "Creating the locked ROCm environment with uv hardlinks..."
mkdir -p "$WAIFUHAT_RUNTIME_ROOT"
uv lock --check
uv sync --frozen --extra dev --extra download --python 3.12.13 --link-mode hardlink

PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
TORCH_LIB="$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/torch/lib"
if compgen -G "$TORCH_LIB/libhsa-runtime64.so*" >/dev/null; then
  echo "Disabling the Torch-bundled native HSA runtime for WSL ROCDXG..."
  mkdir -p "$TORCH_LIB/disabled-for-rocdxg"
  for runtime in "$TORCH_LIB"/libhsa-runtime64.so*; do
    [[ -e "$runtime" ]] || continue
    mv -f "$runtime" "$TORCH_LIB/disabled-for-rocdxg/"
  done
fi

echo "Verifying PyTorch and ROCm..."
"$PYTHON" - <<'PY'
import torch
print("torch:", torch.__version__)
print("ROCm:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access ROCm GPU")
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()
print("FP16 smoke test:", tuple(y.shape), float(y[0, 0]))
b = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
c = b @ b
torch.cuda.synchronize()
print("BF16 smoke test:", tuple(c.shape), float(c[0, 0]))
PY

echo "Installing pinned JPEG XL tools..."
"$PYTHON" scripts/install_jxl.py --runtime-root "$WAIFUHAT_RUNTIME_ROOT"

if [[ "${WAIFUHAT_SKIP_MODELS:-0}" != "1" ]]; then
  echo "Downloading official model weights..."
  "$PYTHON" scripts/download_models.py
fi

echo "Running unit tests..."
"$PYTHON" -m pytest -q

TORCH_CPU="$TORCH_LIB/libtorch_cpu.so"
if [[ -e "$TORCH_CPU" ]]; then
  LINKS="$(stat -c '%h' "$TORCH_CPU")"
  echo "Torch hardlink count: $LINKS"
  if [[ "$LINKS" -lt 2 ]]; then
    echo "Torch was not hardlinked from the uv cache." >&2
    exit 1
  fi
fi

echo "Installation complete."
