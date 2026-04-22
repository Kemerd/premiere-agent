#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# video-use-premiere bootstrap (Linux / macOS).
#
# Idempotent. Honors TORCH_INDEX env var so CPU-only / ROCm users can swap:
#   TORCH_INDEX=https://download.pytorch.org/whl/cpu     ./install.sh
#   TORCH_INDEX=https://download.pytorch.org/whl/rocm6.0 ./install.sh
#
# Default is CUDA 12.1 wheels which match the cuDNN bundled with the
# faster-whisper / CTranslate2 wheel matrix as of late 2025.
# ---------------------------------------------------------------------------
set -euo pipefail

# Resolve the script's own directory so the script works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[video-use-premiere] python: $(command -v python3 || command -v python)"
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"

if [ -z "$PYTHON" ]; then
  echo "ERROR: no python interpreter on PATH" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. Pip itself first. An old pip refuses modern wheel selectors and you'll
#    end up downloading a years-old transformers that can't load Florence-2.
# ---------------------------------------------------------------------------
echo "[video-use-premiere] upgrading pip"
"$PYTHON" -m pip install --upgrade pip

# ---------------------------------------------------------------------------
# 2. PyTorch from the right index. Done explicitly because pip can't pick
#    CUDA vs CPU vs ROCm from PyPI alone.
# ---------------------------------------------------------------------------
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
echo "[video-use-premiere] installing torch from $TORCH_INDEX"
"$PYTHON" -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"

# ---------------------------------------------------------------------------
# 3. The package itself + the heavy preprocess + fcpxml extras. We don't pull
#    [diarize] by default because pyannote pulls a lot and most users won't
#    need speaker IDs.
# ---------------------------------------------------------------------------
echo "[video-use-premiere] installing package + preprocess + fcpxml extras"
"$PYTHON" -m pip install -e ".[preprocess,fcpxml]"

# ---------------------------------------------------------------------------
# 4. ffmpeg PATH check. Not fatal — user might have it in a non-standard
#    location and override via env elsewhere — but warn loudly.
# ---------------------------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo ""
  echo "WARN: ffmpeg not found on PATH."
  echo "      brew install ffmpeg     (macOS)"
  echo "      sudo apt install ffmpeg (Debian/Ubuntu)"
  echo ""
fi

# ---------------------------------------------------------------------------
# 5. CUDA smoke test. Doesn't fail the install if CUDA is missing — there is
#    a working CPU fallback path. Just informs the user what they'll get.
# ---------------------------------------------------------------------------
echo "[video-use-premiere] CUDA smoke test:"
"$PYTHON" - <<'PY'
import torch
ok = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if ok else "n/a"
total_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if ok else 0.0
print(f"  cuda available : {ok}")
print(f"  device         : {name}")
print(f"  total VRAM     : {total_gb:.1f} GB")
PY

echo ""
echo "[video-use-premiere] install complete."
echo "  next: cp .env.example .env   (only needed for --diarize)"
echo "        python helpers/preprocess_batch.py /path/to/your/videos"
