#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -n "${WAIFUHAT_ROCM_ENV:-}" ]]; then
  if [[ ! -r "$WAIFUHAT_ROCM_ENV" ]]; then
    echo "[ERROR] WAIFUHAT_ROCM_ENV is not readable: $WAIFUHAT_ROCM_ENV" >&2
    exit 1
  fi
  source "$WAIFUHAT_ROCM_ENV"
fi
source scripts/runtime_env.sh
if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
  echo "[ERROR] The WSL environment is missing. Run install.bat first." >&2
  exit 1
fi
exec "$UV_PROJECT_ENVIRONMENT/bin/python" "$@"
