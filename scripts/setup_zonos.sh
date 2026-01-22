#!/usr/bin/env bash

#sudo apt-get update
#sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
#python3.12 --version

#python3.12 -m venv ./zonos-venv
#source ./zonos-venv/bin/activate
#python -m pip install -U pip setuptools wheel


set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SCAN_ONLY=0
WITH_COMPILE=0
RUN_CHECK=0
PYTHON_BIN=""
TORCH_VERSION="${TORCH_VERSION:-2.4.1+cu118}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.1+cu118}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.4.1+cu118}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"

usage() {
  cat <<'USAGE'
Usage: scripts/setup_zonos.sh [options]

Options:
  --scan-only   Print install-related instructions found in the repo and exit.
  --compile     Install optional "compile" extras (needed for hybrid).
  --check       Run sample.py after install.
  -h, --help    Show this help.
USAGE
}

log() {
  printf "[setup] %s\n" "$@"
}

scan_repo() {
  log "Scanning repo for install instructions (excluding models/)..."
  if command -v rg >/dev/null 2>&1; then
    rg -n "install|dependencies|requirements|setup|uv sync|pip install|espeak|phonem" -S "$ROOT_DIR" --glob '!models/**'
  else
    grep -RInE "install|dependencies|requirements|setup|uv sync|pip install|espeak|phonem" "$ROOT_DIR" --exclude-dir=models
  fi
}

install_espeak() {
  if command -v espeak-ng >/dev/null 2>&1; then
    log "eSpeak already installed."
    return
  fi

  if command -v apt-get >/dev/null 2>&1; then
    log "Installing eSpeak (apt-get)..."
    sudo apt-get update
    sudo apt-get install -y espeak-ng
    return
  fi

  if command -v apt >/dev/null 2>&1; then
    log "Installing eSpeak (apt)..."
    sudo apt update
    sudo apt install -y espeak-ng
    return
  fi

  if command -v brew >/dev/null 2>&1; then
    log "Installing eSpeak (brew)..."
    brew install espeak-ng
    return
  fi

  log "Could not detect a supported package manager. Install eSpeak manually:"
  log "  Ubuntu: sudo apt install -y espeak-ng"
  log "  macOS:  brew install espeak-ng"
}

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    log "uv already installed."
    return
  fi

  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    log "Installing uv via pip..."
    "$PYTHON_BIN" -m pip install -U uv
    return
  fi

  log "Python not found; install Python and then run: pip install -U uv"
  exit 1
}

install_python_deps() {
  log "Installing Zonos without dependency resolution to keep pinned torch..."
  uv pip install -e "$ROOT_DIR" --no-deps
  if [[ "$WITH_COMPILE" -eq 1 ]]; then
    log "Installing compile extras (hybrid) without pulling torch..."
    uv pip install psutil
    uv pip install --no-build-isolation --no-deps flash-attn causal-conv1d mamba-ssm
  fi
}

run_check() {
  log "Running sample.py..."
  (cd "$ROOT_DIR" && "$PYTHON_BIN" sample.py)
}

ensure_python_312() {
  if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
    log "Using python3.12."
    return
  fi

  if command -v pyenv >/dev/null 2>&1; then
    log "Installing Python 3.12 via pyenv..."
    pyenv install -s 3.12.0
    pyenv local 3.12.0
    PYTHON_BIN="$(pyenv which python)"
    return
  fi

  if command -v brew >/dev/null 2>&1; then
    log "Installing Python 3.12 via Homebrew..."
    brew install python@3.12
    if command -v python3.12 >/dev/null 2>&1; then
      PYTHON_BIN="python3.12"
      return
    fi
  fi

  log "Python 3.12 not found. Please install it and rerun:"
  log "  macOS (brew): brew install python@3.12"
  log "  pyenv: pyenv install 3.12.0 && pyenv local 3.12.0"
  exit 1
}

install_torch_stack() {
  log "Installing pinned torch stack to avoid CUDA/NCCL mismatches..."
  uv pip uninstall torch torchvision torchaudio triton || true
  uv pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

  if ! "$PYTHON_BIN" -c "import torch; print(torch.__version__, torch.version.cuda)" >/dev/null 2>&1; then
    log "Torch failed to import after install."
    log "If you see NCCL symbol errors, clear LD_LIBRARY_PATH or install a matching CUDA/NCCL toolchain."
    exit 1
  fi
}

install_audio_deps() {
  log "Installing audio decode dependencies (torchcodec + ffmpeg)..."
  if command -v brew >/dev/null 2>&1; then
    brew install ffmpeg
  else
    log "Homebrew not found. Install ffmpeg manually."
  fi
  "$PYTHON_BIN" -m pip install torchcodec
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scan-only) SCAN_ONLY=1 ;;
    --compile) WITH_COMPILE=1 ;;
    --check) RUN_CHECK=1 ;;
    -h|--help) usage; exit 0 ;;
    *) log "Unknown option: $1"; usage; exit 1 ;;
  esac
  shift
done

scan_repo
if [[ "$SCAN_ONLY" -eq 1 ]]; then
  exit 0
fi

ensure_python_312
install_espeak
install_uv
install_torch_stack
install_python_deps
install_audio_deps

if [[ "$RUN_CHECK" -eq 1 ]]; then
  run_check
fi
