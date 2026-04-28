#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# premiere-video-editor-agent — cloud-sandbox bootstrap.
#
# This script runs INSIDE the cloud sandbox (Claude Code on the web,
# Codex Cloud, GitHub Actions, devcontainer, Docker) on the very first
# session-start, before the agent runs. It:
#
#   1. Installs ffmpeg + ffprobe via apt (NOT in either cloud's base image).
#   2. Installs PyTorch CPU wheels (cloud sandboxes have no GPU; this
#      avoids a multi-GB CUDA wheel that would never get used and may
#      not even fit in the sandbox's disk quota).
#   3. Installs the skill's [preprocess,fcpxml] extras — Parakeet ONNX
#      (CPU EP), Florence-2, CLAP, OpenTimelineIO, etc. on top of the
#      CPU torch wheels above.
#   4. Pulls the spaCy English model used by the caveman-compression
#      pass over Florence-2 captions.
#   5. Pre-warms the skill health-check cache so the first agent call
#      doesn't pay the ~3s smoke-test cost.
#
# Idempotent. Safe to re-run. Cheap on warm caches (most steps no-op).
#
# Designed to be invoked from each cloud's "environment setup script"
# field as a single one-liner:
#
#   Claude Code on the web (env settings -> Setup script):
#     bash .claude/skills/premiere-video-editor-agent/scripts/cloud_setup.sh
#
#   Codex Cloud (environment -> Setup script):
#     bash .agents/skills/premiere-video-editor-agent/scripts/cloud_setup.sh
#
# (Adjust the path if the user mounted the skill at a different
# subdirectory of their videos repo.)
#
# Setup-phase internet access is required (both clouds default to a
# Trusted allowlist that covers apt, PyPI, and Hugging Face). If the
# environment is set to "None" network access this script will fail
# and the agent won't start.
# ---------------------------------------------------------------------------
set -euo pipefail

# Resolve the skill root from the script's own location so we don't
# care which cwd the cloud spawned us in. BASH_SOURCE handles symlinks
# correctly via `readlink -f`-style resolution one level deep, which
# is enough for both `.claude/skills/<name>` and `.agents/skills/<name>`
# layouts (whether the skill is a real directory, a git submodule, or
# a relative-path symlink).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "[cloud_setup] skill root : $SKILL_ROOT"
echo "[cloud_setup] script dir : $SCRIPT_DIR"
echo "[cloud_setup] cwd        : $(pwd)"

# ---------------------------------------------------------------------------
# 1. ffmpeg / ffprobe. Hard requirement of every helper that touches
#    media (extract_audio.py, render-once-deprecated, export_fcpxml's
#    timecode snap, the visual lane's frame extraction). NOT bundled in
#    Claude Code Web's image, NOT in codex-universal — we install via
#    apt because the setup script runs as root on Ubuntu 24.04 in both
#    sandboxes.
#
#    `command -v` keeps this idempotent across re-runs / cached containers.
# ---------------------------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "[cloud_setup] installing ffmpeg via apt"
  # Single apt call; -y for unattended; -qq to quiet the progress noise
  # so the cloud's setup-script log stays readable.
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg
else
  echo "[cloud_setup] ffmpeg already present, skipping apt install"
fi

# ---------------------------------------------------------------------------
# 2. PyTorch CPU wheels.
#
#    Cloud sandboxes have NO GPU, so we explicitly pin the CPU index.
#    Skipping this and letting [preprocess] resolve torch transitively
#    would land the default PyPI wheel — which is currently the CUDA
#    build on Linux x86_64 (~2.5 GB). That's wasted disk and download
#    time in a CPU-only environment.
#
#    --upgrade-strategy only-if-needed avoids re-downloading torch on a
#    cached container where it's already at the right version.
# ---------------------------------------------------------------------------
echo "[cloud_setup] installing torch (CPU wheel)"
python3 -m pip install --upgrade --quiet pip
python3 -m pip install --quiet \
  --index-url https://download.pytorch.org/whl/cpu \
  --upgrade-strategy only-if-needed \
  torch torchvision torchaudio

# ---------------------------------------------------------------------------
# 3. The skill itself + the [preprocess,fcpxml] extras.
#
#    `pip install -e` on a path lets the skill's helper scripts import
#    by name (`from helpers._onnx_pool import ...`) without a sys.path
#    hack. The extras pull onnx-asr, onnxruntime (CPU build is what gets
#    selected when onnxruntime-gpu fails to find CUDA libs at runtime),
#    transformers, optimum, accelerate, soxr, einops, timm, spacy,
#    soundfile, opentimelineio, otio-fcpx-xml-adapter, otio-fcp-adapter.
#
#    onnxruntime-gpu is in the [preprocess] extra but is harmless on a
#    CPU-only host: the provider ladder in helpers/_onnx_providers.py
#    falls through to the CPU EP when CUDA / DirectML / TRT can't load.
# ---------------------------------------------------------------------------
echo "[cloud_setup] installing skill + [preprocess,fcpxml] extras"
python3 -m pip install --quiet -e "${SKILL_ROOT}[preprocess,fcpxml]"

# ---------------------------------------------------------------------------
# 4. spaCy English model. Pulled by helpers/caveman_compress.py the
#    first time the visual lane lands a caption; pre-fetching here keeps
#    the agent's first preprocess call from blocking on a 12 MB model
#    download mid-run.
# ---------------------------------------------------------------------------
if ! python3 -c "import en_core_web_sm" >/dev/null 2>&1; then
  echo "[cloud_setup] downloading spaCy en_core_web_sm"
  python3 -m spacy download en_core_web_sm --quiet || \
    echo "[cloud_setup] WARN: spaCy model download failed — caveman pass will fall back at runtime"
else
  echo "[cloud_setup] spaCy en_core_web_sm already present, skipping"
fi

# ---------------------------------------------------------------------------
# 5. Health-check pre-warm.
#
#    The agent's parent_rules.md mandates `python helpers/health.py
#    --json` as step 0 of every session. Cached for 7 days at
#    ~/.premiere-video-editor-agent/health.json so subsequent calls return in
#    <500 ms. Running it here means the agent's very first invocation
#    hits the cache instead of paying the smoke-test cost on top of
#    the model-download cost it ALSO can't avoid.
#
#    `|| true` so a transient health failure (e.g. HF Hub flaky)
#    doesn't block session start; the agent will just re-run health
#    when it actually starts.
# ---------------------------------------------------------------------------
echo "[cloud_setup] pre-warming health check cache"
python3 "$SKILL_ROOT/helpers/health.py" --json >/dev/null 2>&1 || \
  echo "[cloud_setup] WARN: health pre-warm failed — agent will retry on first call"

# ---------------------------------------------------------------------------
# 6. Final summary so the cloud's setup-script log gives the user a
#    quick confirmation that everything landed.
# ---------------------------------------------------------------------------
echo ""
echo "[cloud_setup] install summary:"
printf "  ffmpeg   : %s\n" "$(ffmpeg  -version 2>/dev/null | head -n1 || echo MISSING)"
printf "  ffprobe  : %s\n" "$(ffprobe -version 2>/dev/null | head -n1 || echo MISSING)"
printf "  python   : %s\n" "$(python3 --version 2>/dev/null || echo MISSING)"
printf "  torch    : %s\n" "$(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null || echo MISSING)"
printf "  ort      : %s\n" "$(python3 -c 'import onnxruntime as o;print(o.__version__)' 2>/dev/null || echo MISSING)"
printf "  otio     : %s\n" "$(python3 -c 'import opentimelineio as o;print(o.__version__)' 2>/dev/null || echo MISSING)"
echo ""
echo "[cloud_setup] done. Agent is ready."
