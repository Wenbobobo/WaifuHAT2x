#!/usr/bin/env bash
# Keep the environment and uv cache on the same Linux filesystem. This lets uv
# hardlink the large ROCm Torch wheel and avoids mounted-filesystem metadata costs.
export WAIFUHAT_RUNTIME_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/waifuhat2x"
export UV_PROJECT_ENVIRONMENT="$WAIFUHAT_RUNTIME_ROOT/venv"
# Reuse a Linux-resident uv cache so the project environment can hardlink the
# pinned Torch wheel instead of storing a second copy.
export UV_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/uv"
export UV_LINK_MODE=hardlink
