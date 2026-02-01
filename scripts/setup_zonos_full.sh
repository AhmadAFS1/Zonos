#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/AhmadAFS1/Zonos.git"
REPO_DIR="Zonos"
ROOT_DIR=""

SCAN_ONLY=0
WITH_COMPILE=0
RUN_CHECK=0
PYTHON_BIN="python"
TORCH_VERSION="${TORCH_VERSION:-2.5.1+cu121}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1+cu121}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1+cu121}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
SUDO="sudo"
if [[ "$(id -u)" -eq 0 ]] || ! command -v sudo >/dev/null 2>&1; then
  SUDO=""
fi

usage() {
  cat <<'USAGE'
Usage: ./setup_zonos_full.sh [options]

Options:
  --scan-only   Print install-related instructions found in the repo and exit.
  --compile     Install optional "compile" extras (needed for hybrid).
  --check       Run sample.py after install.
  -h, --help    Show this help.
USAGE
}

log() {
  printf "[full-setup] %s\n" "$@"
}

install_system_tools() {
  if command -v apt-get >/dev/null 2>&1; then
    log "Installing system tools..."
    $SUDO apt-get update
    $SUDO apt-get install -y git curl ca-certificates libsndfile1
  fi
}

clone_repo() {
  if [[ -d "$REPO_DIR/.git" ]]; then
    log "Repo already exists at $REPO_DIR."
    return
  fi
  log "Cloning repo from $REPO_URL..."
  git clone "$REPO_URL"
}

scan_repo() {
  log "Scanning repo for install instructions (excluding models/)..."
  if command -v rg >/dev/null 2>&1; then
    rg -n "install|dependencies|requirements|setup|uv sync|pip install|espeak|phonem" -S . --glob '!models/**'
  else
    grep -RInE "install|dependencies|requirements|setup|uv sync|pip install|espeak|phonem" . --exclude-dir=models
  fi
}

install_python_312_and_venv() {
  if command -v apt-get >/dev/null 2>&1; then
    log "Installing Python 3.12 (apt-get)..."
    $SUDO apt-get update
    $SUDO apt-get install -y software-properties-common
    $SUDO add-apt-repository ppa:deadsnakes/ppa
    $SUDO apt-get update
    $SUDO apt-get install -y python3.12 python3.12-venv python3.12-dev
  else
    log "apt-get not found. Ensure Python 3.12 is installed manually."
  fi

  if ! command -v python3.12 >/dev/null 2>&1; then
    log "python3.12 not found in PATH. Install it and rerun."
    exit 1
  fi

  python3.12 --version
  log "Creating venv at ./zonos-venv..."
  if [[ ! -d "./zonos-venv" ]]; then
    python3.12 -m venv "./zonos-venv"
  fi
  # shellcheck disable=SC1091
  source "./zonos-venv/bin/activate"
  python -m pip install -U pip setuptools wheel
}

install_espeak() {
  if command -v espeak-ng >/dev/null 2>&1; then
    log "eSpeak already installed."
    return
  fi

  if command -v apt-get >/dev/null 2>&1; then
    log "Installing eSpeak (apt-get)..."
    $SUDO apt-get update
    $SUDO apt-get install -y espeak-ng
    return
  fi

  if command -v apt >/dev/null 2>&1; then
    log "Installing eSpeak (apt)..."
    $SUDO apt update
    $SUDO apt install -y espeak-ng
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

install_torch_stack() {
  log "Installing pinned torch stack to avoid CUDA/NCCL mismatches..."
  uv pip uninstall torch torchvision torchaudio triton || true
  uv pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

  if ! python -c "import torch; print(torch.__version__, torch.version.cuda)" >/dev/null 2>&1; then
    log "Torch failed to import after install."
    log "If you see NCCL symbol errors, clear LD_LIBRARY_PATH or install a matching CUDA/NCCL toolchain."
    exit 1
  fi
}

install_python_deps() {
  log "Installing Zonos without dependency resolution to keep pinned torch..."
  uv pip install -e . --no-deps
  log "Installing Zonos dependencies (excluding torch)..."
  deps=$(python - <<'PY'
import tomllib
from pathlib import Path

data = tomllib.loads(Path("pyproject.toml").read_text())
deps = data.get("project", {}).get("dependencies", [])
skip = {"torch", "torchaudio", "torchvision"}
filtered = []
for dep in deps:
    name = dep.split(";")[0].strip().split()[0]
    if name in skip:
        continue
    filtered.append(dep)
print(" ".join(filtered))
PY
)
  if [[ -n "$deps" ]]; then
    # shellcheck disable=SC2086
    uv pip install $deps
  fi
  if [[ "$WITH_COMPILE" -eq 1 ]]; then
    if command -v apt-get >/dev/null 2>&1; then
      log "Installing build tools for compile extras..."
      $SUDO apt-get update
      $SUDO apt-get install -y build-essential ninja-build
    fi
    if ! command -v nvcc >/dev/null 2>&1; then
      log "nvcc not found. Compile extras may fail without a CUDA devel toolkit."
    fi
    log "Installing compile extras (hybrid) without pulling torch..."
    uv pip install psutil
    uv pip install einops
    uv pip install --no-build-isolation --no-deps flash-attn causal-conv1d mamba-ssm
  fi
}

install_audio_deps() {
  log "Installing audio decode dependencies (torchcodec + ffmpeg)..."
  if command -v apt-get >/dev/null 2>&1; then
    $SUDO apt-get update
    $SUDO apt-get install -y ffmpeg libsndfile1
  elif command -v brew >/dev/null 2>&1; then
    brew install ffmpeg
  else
    log "Homebrew not found. Install ffmpeg manually."
  fi
  "$PYTHON_BIN" -m pip install torchcodec
}

run_check() {
  log "Running sample.py..."
  python sample.py
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

install_system_tools
clone_repo
cd "$REPO_DIR"
ROOT_DIR="$(pwd)"

scan_repo
if [[ "$SCAN_ONLY" -eq 1 ]]; then
  exit 0
fi

install_python_312_and_venv
install_espeak
install_uv
install_torch_stack
install_python_deps
install_audio_deps

if [[ "$RUN_CHECK" -eq 1 ]]; then
  run_check
fi
