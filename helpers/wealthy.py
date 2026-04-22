"""Wealthy-mode resolver — pure speed knob, no quality changes.

`--wealthy` (CLI flag) or `VIDEO_USE_WEALTHY=1` (env var) tells every lane
"the user has a 24GB+ card, throw bigger batches at the GPU." It does NOT
swap models, change beam counts, or alter sampling strategy — outputs are
bit-for-bit identical to the default tier, just faster.

Per-lane wealthy overrides:

    Whisper :  batch_size 24  →  48
    Florence:  batch_size  8  →  32   (caption batch)
    PANNs   :  windows-per-call 1 → 64
               (genuine speedup — CNN14 is fast per call but per-call
                overhead dominates; batching collapses it.)

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


# Per-lane wealthy batch knobs. Tuned for an RTX 4090 (24 GB) with
# headroom for a desktop compositor and incidental allocations. A 5090
# with 32 GB has even more room — bump these when you want.
WHISPER_BATCH = 48
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
