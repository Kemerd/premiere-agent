#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# video-use-premiere bootstrap (Linux / macOS).
#
# Idempotent. As of the Florence-2 ONNX port, the default [preprocess]
# install is torch-FREE — both the speech lane (Parakeet ONNX) and the
# visual lane (Florence-2 ONNX, helpers/florence_onnx.py) run on
# ONNX Runtime via the bundled CUDA / TensorRT EPs.
#
# TORCH_INDEX is now LEGACY — only consulted if the user opts into the
# [diarize] extra (which pulls pyannote.audio, the only remaining
# torch-dependent component) via INSTALL_DIARIZE=1.
#
#   INSTALL_DIARIZE=1 ./install.sh                                # default torch index
#   INSTALL_DIARIZE=1 TORCH_INDEX=https://download.pytorch.org/whl/cpu \
#       ./install.sh                                              # CPU-only diarize
#
# Default ONNX Runtime CUDA EP target is CUDA 12.x via the cuDNN
# bundled with onnxruntime-gpu>=1.22 — no separate torch CUDA install
# is required for the speech / visual lanes.
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
#    end up downloading years-old wheels that can't load Florence-2 ONNX.
# ---------------------------------------------------------------------------
echo "[video-use-premiere] upgrading pip"
"$PYTHON" -m pip install --upgrade pip

# ---------------------------------------------------------------------------
# 2. The package itself + the heavy preprocess + fcpxml extras. We don't
#    pull [diarize] by default because pyannote pulls torch (the ONLY
#    remaining torch-dependent component) and most users won't need
#    speaker IDs.
#
#    The [preprocess] extra now installs:
#      - onnx-asr[gpu,hub] + onnxruntime-gpu (speech + visual ONNX runtime)
#      - tensorrt-cu12-libs (opt-in TensorRT EP for both lanes)
#      - tokenizers + huggingface_hub (Florence-2 BART tokenizer + weights)
#      - imageio-ffmpeg (frame extraction for visual_lane)
#      - transformers (CLAP audio lane processor only — no torch needed
#        for transformers' tokenizer/feature-extractor code paths)
#      - soundfile + soxr (WAV I/O + fast resample)
# ---------------------------------------------------------------------------
echo "[video-use-premiere] installing package + preprocess + fcpxml extras"
"$PYTHON" -m pip install -e ".[preprocess,fcpxml]"

# ---------------------------------------------------------------------------
# 3. Optional torch install for the [diarize] extra. Skipped by default.
#    User opts in via INSTALL_DIARIZE=1 (then the script pulls torch from
#    TORCH_INDEX and adds [diarize] to the pip install).
# ---------------------------------------------------------------------------
if [ "${INSTALL_DIARIZE:-0}" = "1" ]; then
  TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
  echo "[video-use-premiere] [diarize] opt-in: installing torch from $TORCH_INDEX"
  "$PYTHON" -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"
  echo "[video-use-premiere] installing [diarize] extra (pyannote.audio)"
  "$PYTHON" -m pip install -e ".[diarize]"
fi

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
# 5. ONNX Runtime providers smoke test. Doesn't fail the install if no GPU
#    EP is available — the CPU EP is a working fallback. Reports which
#    EPs the freshly-installed onnxruntime-gpu wheel can actually load
#    on this host. Useful for spotting cuDNN / CUDA version mismatches
#    BEFORE the user kicks off a 30-min preprocess.
# ---------------------------------------------------------------------------
echo "[video-use-premiere] ONNX Runtime providers:"
"$PYTHON" - <<'PY'
import onnxruntime as ort
eps = ort.get_available_providers()
print("  available EPs :", eps)
print(f"  CUDA EP       : {'CUDAExecutionProvider' in eps}")
print(f"  TensorRT EP   : {'TensorrtExecutionProvider' in eps}")
print(f"  CoreML EP     : {'CoreMLExecutionProvider' in eps}")
print(f"  DirectML EP   : {'DmlExecutionProvider' in eps}")
print(f"  ORT version   : {ort.__version__}")
PY

echo ""
echo "[video-use-premiere] install complete."
echo "  next: cp .env.example .env   (only needed for --diarize)"
echo "        python helpers/preprocess_batch.py /path/to/your/videos"
echo ""
echo "  To opt into speaker diarization (adds torch + pyannote.audio):"
echo "        INSTALL_DIARIZE=1 ./install.sh"
