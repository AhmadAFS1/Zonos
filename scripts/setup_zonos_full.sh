#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/AhmadAFS1/Zonos.git"
REPO_DIR="Zonos"

log() {
  printf "[setup] %s\n" "$@"
}

clone_repo() {
  if [[ -d "$REPO_DIR/.git" ]]; then
    log "Repo already exists at $REPO_DIR."
    return
  fi
  log "Cloning repo from $REPO_URL..."
  git clone "$REPO_URL" "$REPO_DIR"
}

clone_repo
cd "$REPO_DIR"

log "Installing base dependencies..."
pip install -e .

log "Installing compile extras..."
pip install --no-build-isolation -e .[compile]

log "Installing flash-attn 2.7.4.post1 from source..."
PIP_NO_CACHE_DIR=1 python -m pip install --no-build-isolation --no-deps --no-binary flash-attn "flash-attn==2.7.4.post1"
