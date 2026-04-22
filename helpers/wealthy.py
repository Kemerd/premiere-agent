"""Wealthy-mode resolver — pure speed knob, no quality changes.

`--wealthy` (CLI flag) or `VIDEO_USE_WEALTHY=1` (env var) tells every lane
"the user has a 24GB+ card, throw bigger batches at the GPU." It does NOT
swap models, change beam counts, or alter sampling strategy — outputs are
bit-for-bit identical to the default tier, just faster.

Per-lane wealthy overrides:

    Whisper :  batch_size 16 →  32
               (Turbo + word-timestamp DTW. With turbo's 4-layer decoder
                the cross-attn map cost dropped 8x vs large-v3, so batch
                32 fits comfortably in 32 GB while staying inside 24 GB
                if the user runs --wealthy on a 4090 with no other lane.)

    Florence:  batch_size 8  →  32   (caption batch)
    PANNs   :  windows-per-call 1 → 64
               (genuine speedup — CNN14 is fast per call but per-call
                overhead dominates; batching collapses it.)

Whisper sizing rationale:
    The HF Whisper pipeline with `return_timestamps="word"` runs DTW
    over the decoder's cross-attention weights — a memory cost that
    scales LINEARLY with decoder layer count. We default to
    whisper-large-v3-turbo (4 decoder layers) instead of large-v3
    (32 decoder layers); that single swap collapses the DTW working
    set by ~8x at the same batch size for ~equivalent English quality.
    See https://github.com/huggingface/transformers/issues/27834 for
    the upstream discussion of the word-timestamp memory profile.

    On turbo + fp16 + word timestamps + sdpa attention:
        * 24 GB card (4090) wealthy : batch=32 → ~11 GB peak
        * 32 GB card (5090) wealthy : batch=32 → ~11 GB peak (same number,
          headroom is for per-video allocator fragmentation across long
          batches; pushing to 48 occasionally OOMs the 5th video in a
          24-clip run as fragments accumulate).

    Wall-clock impact: ~2x faster than default tier on the same hardware.

Env var resolution lives here so the orchestrator can set it once and
every subprocess inherits it without plumbing flags through every call.

Usage:
    from wealthy import is_wealthy, WHISPER_BATCH, FLORENCE_BATCH, PANNS_WINDOWS_PER_BATCH

    bs = WHISPER_BATCH if is_wealthy(cli_flag) else DEFAULT_BATCH_SIZE
"""

from __future__ import annotations

import os


# Public env var. Set by preprocess.py when --wealthy is on; subprocesses
# inherit it. Truthy values: "1", "true", "yes", "on" (case-insensitive).
ENV_VAR = "VIDEO_USE_WEALTHY"
_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


# Per-lane wealthy batch knobs. Tuned for a 32 GB Blackwell (RTX 5090)
# with headroom for the desktop compositor and per-video pipeline reload
# fragmentation. Whisper sized assuming the turbo model (4 decoder
# layers, ~8x less DTW cross-attn cost than large-v3) — see the module
# docstring above. Florence and PANNs are unchanged because their memory
# profile didn't shift.
WHISPER_BATCH = 32
FLORENCE_BATCH = 32
PANNS_WINDOWS_PER_BATCH = 64


# ---------------------------------------------------------------------------
# Parakeet ONNX session-pool sizing.
#
# The ONNX speech lane (helpers/parakeet_onnx_lane.py) loads N independent
# `onnxruntime.InferenceSession` handles in one process — each session is
# its own native CUDA stream / TensorRT engine cache slot. ORT releases
# the GIL during native Run() so a `ThreadPoolExecutor(max_workers=N)`
# fans out to N truly-parallel native inferences on a single GPU.
#
# VRAM math, Parakeet TDT 0.6B, fp16 ONNX (encoder+joint+decoder
# resident, plus a small per-session activation working set):
#
#   per-session resident      : ~1.2 GB (fp16) / ~0.6 GB (int8)
#   per-session transient peak: ~0.4 GB (encoder activations on a 30s clip)
#   total at N=8 (fp16)       : 8 × 1.2  GB =  9.6 GB resident
#   total at N=8 + transients : 8 × 1.6  GB = 12.8 GB peak
#
#   On a 32 GB 5090 that leaves ~19 GB free for the visual + audio lanes
#   if they're co-tenanted (PARALLEL_3 schedule), or ~22 GB free in
#   sequential mode. Either way the pool fits comfortably with margin.
#
# Default tier (most users) gets N=4 — a safe number on 12 GB cards
# (4 × 1.6 GB peak ≈ 6.4 GB) which still gives a 4x throughput multiplier
# vs. single-session inference. Wealthy tier (24 GB+) gets N=8 for the
# full ~8x multiplier.
#
# Override at runtime:
#   VIDEO_USE_PARAKEET_POOL_SIZE=<int>    (forces a specific N, ignores tier)
#
# The `OnnxSessionPool` in `helpers/_onnx_pool.py` ALSO clamps N down at
# load time if `vram.detect_gpu().free_gb` is too small to fit the pool —
# so passing 8 on an 8 GB card will silently degrade to whatever fits,
# rather than crashing on the 5th session's cudaMalloc.
# ---------------------------------------------------------------------------
PARAKEET_POOL_SIZE = 4
PARAKEET_POOL_SIZE_WEALTHY = 8

# Quantization knob. fp16 is the default — onnx-asr's stock fp16 export
# of Parakeet TDT 0.6B benchmarks within rounding noise of fp32 on the
# librispeech-clean / common-voice eval suites. int8 cuts VRAM footprint
# in half (and on Blackwell with the int8 tensor cores, runs ~30% faster)
# but loses ~0.3 WER points on noisy / accented audio.
#
# Set via env var: VIDEO_USE_PARAKEET_QUANT=int8
PARAKEET_QUANTIZATION_DEFAULT = "fp16"
PARAKEET_QUANTIZATION_ENV = "VIDEO_USE_PARAKEET_QUANT"


def parakeet_pool_size(cli_flag: bool = False) -> int:
    """Resolve the Parakeet ONNX session-pool size for this process.

    Resolution order (first match wins):
        1. VIDEO_USE_PARAKEET_POOL_SIZE env var (explicit override)
        2. PARAKEET_POOL_SIZE_WEALTHY if `is_wealthy(cli_flag)` else PARAKEET_POOL_SIZE

    The pool itself further clamps the returned value against available
    VRAM at session-construction time — this function only resolves the
    *intended* size, not the *achievable* one.
    """
    raw = os.environ.get("VIDEO_USE_PARAKEET_POOL_SIZE", "").strip()
    if raw:
        try:
            n = int(raw)
            if n >= 1:
                return n
        except ValueError:
            pass
    return PARAKEET_POOL_SIZE_WEALTHY if is_wealthy(cli_flag) else PARAKEET_POOL_SIZE


def parakeet_quantization() -> str | None:
    """Resolve the Parakeet ONNX quantization knob.

    Returns None for the default fp16 path (which is what onnx-asr's
    `load_model(...)` call expects when `quantization` is unset), or
    a string like "int8" if the user opted into a smaller variant.
    """
    raw = os.environ.get(PARAKEET_QUANTIZATION_ENV, "").strip().lower()
    if not raw or raw == PARAKEET_QUANTIZATION_DEFAULT:
        return None
    return raw


def is_wealthy(cli_flag: bool = False) -> bool:
    """True if the user is in wealthy mode.

    Resolution order:
        1. Explicit CLI flag (truthy wins immediately)
        2. VIDEO_USE_WEALTHY env var

    Idempotent + side-effect free — safe to call from anywhere.
    """
    if cli_flag:
        return True
    raw = os.environ.get(ENV_VAR, "").strip().lower()
    return raw in _TRUTHY


def propagate_to_env(cli_flag: bool) -> None:
    """Mirror the CLI flag into the env var so subprocesses inherit it.

    Called by the orchestrator once at startup. Idempotent.
    """
    if cli_flag:
        os.environ[ENV_VAR] = "1"
