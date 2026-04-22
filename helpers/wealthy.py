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
