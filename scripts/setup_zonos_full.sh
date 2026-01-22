cat <<'SETUP_ZONOS_FULL' > setup_zonos_full.sh
#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/AhmadAFS1/Zonos.git"
REPO_DIR="Zonos"

SCAN_ONLY=0
WITH_COMPILE=0
RUN_CHECK=0
PYTHON_BIN="python"

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
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
  else
    log "apt-get not found. Ensure Python 3.12 is installed manually."
  fi

  if ! command -v python3.12 >/dev/null 2>&1; then
    log "python3.12 not found in PATH. Install it and rerun."
    exit 1
  fi

  python3.12 --version
  log "Creating venv at ./zonos-venv..."
  python3.12 -m venv "./zonos-venv"
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
  log "Installing into the current active environment using uv..."
  uv pip install -e .
  if [[ "$WITH_COMPILE" -eq 1 ]]; then
    uv pip install -e .[compile]
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

clone_repo
cd "$REPO_DIR"

scan_repo
if [[ "$SCAN_ONLY" -eq 1 ]]; then
  exit 0
fi

install_python_312_and_venv
install_espeak
install_uv
install_python_deps
install_audio_deps

if [[ "$RUN_CHECK" -eq 1 ]]; then
  run_check
fi
SETUP_ZONOS_FULL
