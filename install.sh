#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# video-use-premiere bootstrap (Linux / macOS).
#
# Idempotent. Auto-picks a sensible torch wheel index per OS/arch. Override
# with the TORCH_INDEX env var if you want something else:
#
#   TORCH_INDEX=https://download.pytorch.org/whl/cpu     ./install.sh   # CPU
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128   ./install.sh   # RTX 50xx
#   TORCH_INDEX=https://download.pytorch.org/whl/rocm6.0 ./install.sh   # AMD ROCm
#
# Default per platform:
#   Linux  x86_64  : CUDA 12.1 wheels (matches cuDNN bundled with the ONNX
#                    Runtime CUDA EP shipped in onnxruntime-gpu>=1.22).
#   Linux  aarch64 : PyPI default (CPU wheels). Jetson / SBSA users with a
#                    custom torch index can override via TORCH_INDEX.
#   macOS  any     : PyPI default. On Apple Silicon that's the MPS-enabled
#                    universal wheel; on Intel Macs it's CPU. There is NO
#                    CUDA on macOS — Apple dropped NVIDIA support in 2018.
# ---------------------------------------------------------------------------
set -euo pipefail

# Resolve the script's own directory so the script works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Identify OS / arch up front. `uname -s` -> Darwin/Linux/MINGW*/CYGWIN*,
# `uname -m` -> x86_64/arm64/aarch64. Drives both the default torch index
# below AND the OS-specific ffmpeg hint at the end. POSIX uname is on
# every shell that can run this script (bash >= 3 ships it on macOS too).
# ---------------------------------------------------------------------------
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"

PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"

echo "[video-use-premiere] python: ${PYTHON:-<none>}"
echo "[video-use-premiere] os:     ${OS_NAME} (${ARCH_NAME})"

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
# 2. PyTorch from the right index.
#
#    pip can't pick CUDA-vs-CPU-vs-ROCm-vs-MPS automatically — the wheel
#    name (e.g. torch-2.4.1+cu121-cp311-...whl) lives on a per-variant
#    index. So we choose a default per platform here and let the user
#    override via env var when they know better (Blackwell needs cu128,
#    AMD wants ROCm, etc).
# ---------------------------------------------------------------------------
if [ -z "${TORCH_INDEX:-}" ]; then
  case "$OS_NAME" in
    Darwin)
      # macOS: torch's MPS-enabled universal2 wheel lives on PyPI proper.
      # If we passed --index-url=<pytorch CUDA index>, pip would 404 since
      # there are no Mac wheels there at all.
      TORCH_INDEX=""
      ;;
    Linux)
      if [ "$ARCH_NAME" = "x86_64" ]; then
        # cu121 keeps lockstep with the cuDNN ABI that onnxruntime-gpu
        # 1.22+ was compiled against. Override to cu128 for Blackwell
        # (RTX 50-series, sm_120) or to /cpu for headless boxes.
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
      else
        # aarch64 / ppc64le: PyPI ships CPU wheels here. Jetson users
        # set TORCH_INDEX themselves to NVIDIA's L4T index.
        TORCH_INDEX=""
      fi
      ;;
    *)
      # MSYS / Cygwin / unknown — let PyPI default sort it out, the user
      # is presumably running a Windows-flavoured shell on Windows and
      # should really be using install.bat, but don't break them.
      TORCH_INDEX=""
      ;;
  esac
fi

if [ -n "$TORCH_INDEX" ]; then
  echo "[video-use-premiere] installing torch from $TORCH_INDEX"
  "$PYTHON" -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"
else
  echo "[video-use-premiere] installing torch from PyPI default (CPU/MPS wheel for this platform)"
  "$PYTHON" -m pip install torch torchvision torchaudio
fi

# ---------------------------------------------------------------------------
# 3. The package itself. Base deps now include the full preprocess stack
#    (Parakeet ONNX + transformers + spaCy + soxr + soundfile) AND the
#    OpenTimelineIO FCPXML / xmeml adapters — every real run needs both,
#    so there's no point hiding them behind extras. Opt-in extras still
#    exist for the genuinely-niche bits ([diarize], [animations], [flash],
#    [parakeet] for the NeMo fallback).
#
#    The pyproject markers handle the OS split for ONNX Runtime:
#      * Win/Linux  -> onnxruntime-gpu (CUDA + TRT + DML EPs)
#      * macOS      -> onnxruntime     (CoreML + CPU EPs; no CUDA on Mac)
#      * Linux/Win x86_64 only -> tensorrt-cu12-libs
# ---------------------------------------------------------------------------
echo "[video-use-premiere] installing package"
"$PYTHON" -m pip install -e "."

# ---------------------------------------------------------------------------
# 4. ffmpeg PATH check. Not fatal — user might have it in a non-standard
#    location and override via env elsewhere — but warn loudly with the
#    install hint matching the actual host OS.
# ---------------------------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo ""
  echo "WARN: ffmpeg not found on PATH."
  case "$OS_NAME" in
    Darwin)
      echo "      brew install ffmpeg"
      ;;
    Linux)
      echo "      sudo apt install ffmpeg          (Debian / Ubuntu)"
      echo "      sudo dnf install ffmpeg          (Fedora / RHEL)"
      echo "      sudo pacman -S ffmpeg            (Arch)"
      ;;
    *)
      echo "      install ffmpeg via your package manager"
      ;;
  esac
  echo ""
fi

# ---------------------------------------------------------------------------
# 5. Accelerator smoke test. Doesn't fail the install if no GPU is present —
#    every lane has a working CPU fallback. Just informs the user which
#    backend is actually going to drive inference, so a missing GPU
#    surfaces here instead of mid-render at 4am.
#
#    Reports:
#      * CUDA  : Linux / Windows + NVIDIA driver
#      * MPS   : macOS on Apple Silicon (M1/M2/M3/M4...)
#      * CPU   : everything else (Intel Mac, headless server, no driver)
# ---------------------------------------------------------------------------
echo "[video-use-premiere] accelerator smoke test:"
"$PYTHON" - <<'PY'
import platform
import torch

cuda_ok = torch.cuda.is_available()
mps_ok  = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

if cuda_ok:
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"  backend     : CUDA")
    print(f"  device      : {name}")
    print(f"  total VRAM  : {total_gb:.1f} GB")
elif mps_ok:
    # Apple Silicon: Florence-2 + CLAP run on MPS where the op set is
    # supported and silently fall back to CPU otherwise. The speech
    # lane uses ONNX Runtime, which on macOS drives the CoreML EP
    # (Neural Engine + Metal) instead of CUDA.
    print(f"  backend     : MPS (Apple Silicon)")
    print(f"  device      : {platform.machine()}")
    print(f"  note        : speech lane uses ONNX Runtime CoreML/CPU EP on Mac.")
elif platform.system() == "Darwin":
    # Intel Mac. CPU-only is fine but slow.
    print(f"  backend     : CPU (Intel Mac)")
    print(f"  note        : no Metal / MPS on Intel Macs — all lanes run on CPU.")
else:
    print(f"  backend     : CPU only")
    print(f"  note        : install GPU drivers + the matching torch index")
    print(f"                (TORCH_INDEX=https://download.pytorch.org/whl/cu121")
    print(f"                 for NVIDIA, /rocm6.0 for AMD) and rerun install.sh.")
PY

echo ""
echo "[video-use-premiere] install complete."
echo "  next: cp .env.example .env   (only needed for --diarize)"
echo "        $PYTHON helpers/preprocess_batch.py /path/to/your/videos"
