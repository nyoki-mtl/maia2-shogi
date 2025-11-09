#!/usr/bin/env bash
set -euo pipefail

# 作業ディレクトリをワークスペースに揃える
cd "${containerWorkspaceFolder:-/workspaces/maia2-shogi}" || exit 1

# echo "[postCreate] Running apt-get update..."
# sudo apt-get update

# echo "[postCreate] Installing build dependencies..."
# sudo apt-get install -y \
#   clang \
#   python3-dev \
#   default-libmysqlclient-dev \
#   build-essential \
#   pkg-config

# echo "[postCreate] Installing Node CLI tools (claude-code, gemini-cli)..."
# npm install -g @anthropic-ai/claude-code @google/gemini-cli @openai/codex

echo "[postCreate] Syncing Python environment with uv..."
uv sync --all-extras

echo "[postCreate] Done."
