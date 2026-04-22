"""Wealthy-mode resolver — pure speed knob, no quality changes.

`--wealthy` (CLI flag) or `VIDEO_USE_WEALTHY=1` (env var) tells every lane
"the user has a 24GB+ card, throw bigger batches at the GPU." It does NOT
swap models, change beam counts, or alter sampling strategy — outputs are
bit-for-bit identical to the default tier, just faster.

Per-lane wealthy overrides:

    Speech (Parakeet ONNX) : pool size scales via parakeet_pool_size().
                Default tier picks 4 sessions on consumer cards, 8 on
                wealthy / datacenter cards. Pool size dominates
                throughput here, not a "batch_size" knob — each
                session runs one clip at a time and the orchestrator
                fans clips across the pool.

    Speech (Parakeet NeMo) : batch_size 16 → 32 via SPEECH_BATCH.
                Only consulted by the NeMo torch fallback in
                `parakeet_lane.py`; the primary ONNX lane does not
                read this constant. Parakeet's RNNT decoder is small
                relative to encoder-decoder ASR stacks so batch=32
                fits comfortably in 24 GB+ envelopes.

    Florence:  batch_size 8  →  32   (caption batch)

    CLAP audio lane :  windows_per_batch 16 → 64   (audio encoder batch)
               + model tier "base" → "large" (Xenova/larger_clap_general)
               (CLAP's audio encoder is small (~150 MB base / ~600 MB
                large) so batch size scales linearly with VRAM until
                the encoder's intermediate activations crowd out other
                lanes. 64 windows × 10s @ 48kHz = ~30 MB raw audio per
                batch, comfortably inside any 24 GB+ card. The larger
                CLAP variant (`larger_clap_general`) sharpens fine-grain
                discrimination on environmental classes that the base
                HTSAT-unfused encoder confuses (e.g. crow vs raven,
                hammer vs mallet) at the cost of ~4x text-encoding
                cold-start time and ~2x audio-encoder steady-state
                throughput.)

Env var resolution lives here so the orchestrator can set it once and
every subprocess inherits it without plumbing flags through every call.

Usage:
    from wealthy import (
        is_wealthy,
        SPEECH_BATCH, FLORENCE_BATCH,
        CLAP_WINDOWS_PER_BATCH, CLAP_WINDOWS_PER_BATCH_WEALTHY,
        CLAP_MODEL_TIER_DEFAULT, CLAP_MODEL_TIER_WEALTHY,
    )

    bs = SPEECH_BATCH if is_wealthy(cli_flag) else DEFAULT_BATCH_SIZE
"""

from __future__ import annotations

import os


# Public env var. Set by preprocess.py when --wealthy is on; subprocesses
# inherit it. Truthy values: "1", "true", "yes", "on" (case-insensitive).
ENV_VAR = "VIDEO_USE_WEALTHY"
_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


# Per-lane wealthy batch knobs. Tuned for a 32 GB Blackwell (RTX 5090)
# with headroom for the desktop compositor and per-video pipeline reload
# fragmentation. Florence is unchanged because its memory profile is
# dominated by the vision encoder, not the captioning batch.
#
# Speech (Parakeet) knob:
#   - SPEECH_BATCH               : NeMo fallback batch size at --wealthy.
#                                  The primary ONNX lane scales via
#                                  parakeet_pool_size() instead.
#
# CLAP audio lane knobs:
#   - CLAP_WINDOWS_PER_BATCH     : audio-encoder batch size (default tier)
#   - CLAP_WINDOWS_PER_BATCH_WEALTHY: same, larger when --wealthy
#   - CLAP_MODEL_TIER_DEFAULT    : "base" (Xenova/clap-htsat-unfused, ~150 MB)
#   - CLAP_MODEL_TIER_WEALTHY    : "large" (Xenova/larger_clap_general, ~600 MB)
#
# Sliding 10s windows at hop 5s over a 5-minute video produce ~60 windows;
# at batch=16 that's 4 audio-encoder forward passes per video, at batch=64
# it's 1 pass. The batch-size choice is bounded by the encoder's transient
# activation footprint (~70 MB per window at fp32 for HTSAT) — 64 is well
# inside a 24 GB envelope even with Parakeet + Florence co-resident.
SPEECH_BATCH = 32
FLORENCE_BATCH = 32
CLAP_WINDOWS_PER_BATCH = 16
CLAP_WINDOWS_PER_BATCH_WEALTHY = 64
CLAP_MODEL_TIER_DEFAULT = "base"
CLAP_MODEL_TIER_WEALTHY = "large"

# Backwards-compat alias for in-flight call sites that still import
# the old name. New code should use SPEECH_BATCH; this can be removed
# once all references are migrated.
WHISPER_BATCH = SPEECH_BATCH


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
