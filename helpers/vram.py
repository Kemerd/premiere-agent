"""VRAM detection + lane scheduling.

The three-lane preprocessor wants to run faster-whisper, PANNs CNN14, and
Florence-2-base in parallel when there's room — and gracefully fall back when
there isn't. This module owns that decision.

Approximate steady-state model footprints with conservative batch sizes:

  - faster-whisper large-v3-turbo, fp16:        ~2.5 GB
  - PANNs CNN14:                                ~0.6 GB
  - Florence-2-base, fp16, batch 8:             ~2.5 GB

A single all-three-parallel run peaks around 5.6 GB in steady state, plus a
~1.5 GB transient spike when each model loads its weights. The 8 GB threshold
in `pick_schedule` leaves headroom for that spike + display compositor + any
background processes the user has running.

Detection strategy (in order):

  1. torch.cuda.mem_get_info(device=0)      — fast, accurate, no subprocess
  2. nvidia-smi --query-gpu=memory.free     — fallback when torch isn't built
                                              with CUDA but smi is on PATH
  3. None                                   — no GPU, fall through to CPU tier

Returning a `Schedule` enum keeps the orchestrator's branching readable and
makes the schedule trivially loggable / overridable from the CLI.
"""

from __future__ import annotations

import enum
import os
import shutil
import subprocess
from dataclasses import dataclass


class Schedule(enum.Enum):
    """How to dispatch the three preprocessing lanes.

    Members are ordered from most to least parallel so callers can compare
    or sort them, and the string value doubles as the CLI flag value.
    """

    PARALLEL_3 = "parallel"        # all three lanes concurrent
    PARALLEL_2_SEQ_1 = "mixed"     # whisper || panns, then florence solo
    SEQUENTIAL = "sequential"      # one lane at a time, GPU
    CPU_FALLBACK = "cpu"           # no CUDA — whisper int8 + PANNs CPU


@dataclass(frozen=True)
class GpuInfo:
    """Snapshot of the primary GPU's memory state at scheduling time."""

    available: bool
    device_name: str
    total_gb: float
    free_gb: float

    def __str__(self) -> str:
        if not self.available:
            return "no CUDA device"
        return f"{self.device_name}  ({self.free_gb:.1f}/{self.total_gb:.1f} GB free)"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _try_torch() -> GpuInfo | None:
    """Use torch.cuda directly. Returns None if torch isn't installed or
    CUDA isn't available — the caller will try the nvidia-smi path next.
    """
    try:
        import torch  # local import: torch is in [preprocess], not always there
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    # device 0 is the convention; multi-GPU users can set CUDA_VISIBLE_DEVICES.
    free_b, total_b = torch.cuda.mem_get_info(0)
    props = torch.cuda.get_device_properties(0)
    return GpuInfo(
        available=True,
        device_name=props.name,
        total_gb=total_b / (1024 ** 3),
        free_gb=free_b / (1024 ** 3),
    )


def _try_nvidia_smi() -> GpuInfo | None:
    """Fallback for environments where torch wasn't built with CUDA but
    nvidia-smi is on PATH (e.g. someone installed the CPU torch wheel by
    accident). Returns None if smi can't tell us anything useful.
    """
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None

    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None

    if not out:
        return None

    # Take the first GPU; a multi-GPU box should set CUDA_VISIBLE_DEVICES.
    parts = [p.strip() for p in out[0].split(",")]
    if len(parts) < 3:
        return None

    try:
        total_mb = float(parts[1])
        free_mb = float(parts[2])
    except ValueError:
        return None

    return GpuInfo(
        available=True,
        device_name=parts[0],
        total_gb=total_mb / 1024.0,
        free_gb=free_mb / 1024.0,
    )


def detect_gpu() -> GpuInfo:
    """Single entry point for callers. Always returns a GpuInfo; callers
    should branch on `info.available` rather than handling None.
    """
    for fn in (_try_torch, _try_nvidia_smi):
        info = fn()
        if info is not None:
            return info
    return GpuInfo(available=False, device_name="cpu", total_gb=0.0, free_gb=0.0)


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

# Boundaries chosen from empirical model footprints above + 1.5 GB headroom
# for the transient weight-load spike. Tuned for an 8 GB / 12 GB / 24 GB
# tiering that matches consumer NVIDIA cards (3060 / 3060 12GB / 3090).
_PARALLEL_3_MIN_GB = 8.0
_PARALLEL_2_MIN_GB = 4.0
_SEQUENTIAL_MIN_GB = 2.0


# ---------------------------------------------------------------------------
# Known-broken parallel combinations.
#
# Some hardware + library combos surface as "fake OOMs" when multiple
# Python processes share a single CUDA device — the real failure is a
# Blackwell driver / kernel selection bug, not memory pressure, but it
# manifests as a `RuntimeError: CUDA error: out of memory` raised on a
# tiny `inputs.to(device)` call with the "CUDA kernel errors might be
# asynchronously reported" preamble. The retry-loop in the lanes can't
# rescue this because every retry hits the same poisoned context.
#
# When we detect such a combo, we transparently downgrade PARALLEL_3 →
# SEQUENTIAL so the lanes never co-tenant the device. Slower but
# correct — and the lane-stagger feature stays intact for its other
# benefit (smoother log interleaving).
#
# The user can opt out with `VIDEO_USE_FORCE_PARALLEL=1` if they've
# verified their driver/torch/transformers stack is fixed.
# ---------------------------------------------------------------------------

def _is_known_broken_parallel_combo(device_name: str) -> tuple[bool, str]:
    """Detect hardware + library combinations where multi-process GPU
    sharing produces silent context corruption.

    Returns (is_broken, reason) — `reason` is a human-readable string
    we surface to the orchestrator log so the downgrade isn't
    mysterious.

    Currently flagged:
        * NVIDIA Blackwell (RTX 50-series, sm_120) + transformers >= 5
          when transformers is importable. The bug surfaces in
          Whisper's pipeline path even with `attn_implementation="eager"`
          set — the issue is not just sdpa, it's the *concurrent CUDA
          context* interaction with Blackwell's kernel scheduler.
          See https://github.com/huggingface/transformers/issues/38662
          plus follow-ups for the underlying instability.

    Best-effort: if we can't import transformers (no [preprocess]
    extra installed) or can't probe the device, we return
    (False, "unknown") rather than guess pessimistically.
    """
    name_lower = device_name.lower()
    is_blackwell = (
        "rtx 50" in name_lower or
        "rtx 51" in name_lower or
        "blackwell" in name_lower or
        "rtx pro 6000" in name_lower
    )
    if not is_blackwell:
        return False, "not Blackwell"
    try:
        import transformers as _tf
        major = int(_tf.__version__.split(".", 1)[0])
    except (ImportError, ValueError):
        # transformers not installed → no Whisper lane to break → fine
        # to leave parallel scheduling on for whatever IS running.
        return False, "transformers not importable"
    if major < 5:
        # Supported / pinned config — sdpa works on Blackwell here, no
        # context corruption observed, parallel is fine. This is the
        # path users SHOULD be on (see pyproject.toml).
        return False, f"transformers {_tf.__version__} < 5 (supported config)"
    return True, (
        f"Blackwell ({device_name}) + transformers {_tf.__version__} — "
        f"known to produce fake OOMs under multi-process CUDA contention "
        f"(downgrade to transformers<5 for the supported parallel path)"
    )


def pick_schedule(info: GpuInfo) -> Schedule:
    """Map a GpuInfo to a Schedule. See module docstring for rationale.

    Conservative — when in doubt, drop a tier. OOM during preprocessing
    crashes the whole batch and forces the user to re-run; a slower
    schedule that completes is strictly better than a fast one that doesn't.

    On hardware + library combinations where we know multi-process GPU
    sharing is broken (see `_is_known_broken_parallel_combo`), we
    transparently downgrade the picked schedule to SEQUENTIAL. Override
    via `VIDEO_USE_FORCE_PARALLEL=1` if you know better.
    """
    if not info.available:
        return Schedule.CPU_FALLBACK

    # Pure VRAM-driven choice first — the historical behavior.
    if info.free_gb >= _PARALLEL_3_MIN_GB:
        chosen = Schedule.PARALLEL_3
    elif info.free_gb >= _PARALLEL_2_MIN_GB:
        chosen = Schedule.PARALLEL_2_SEQ_1
    elif info.free_gb >= _SEQUENTIAL_MIN_GB:
        chosen = Schedule.SEQUENTIAL
    else:
        chosen = Schedule.CPU_FALLBACK

    # Now apply the known-broken-combo override, but only if the user
    # hasn't explicitly opted out. We don't touch SEQUENTIAL or
    # CPU_FALLBACK choices — they're already non-parallel.
    if chosen in (Schedule.PARALLEL_3, Schedule.PARALLEL_2_SEQ_1):
        force_parallel = os.environ.get("VIDEO_USE_FORCE_PARALLEL", "").strip()
        if force_parallel not in ("1", "true", "yes"):
            broken, reason = _is_known_broken_parallel_combo(info.device_name)
            if broken:
                # Surface the downgrade reason via stderr so the
                # orchestrator log explains why we didn't pick
                # PARALLEL_3 even though VRAM permits it. Using stderr
                # keeps the [PROGRESS] machine-readable lines on stdout
                # uncontaminated for downstream parsing.
                import sys as _sys
                print(
                    f"[scheduler] downgrading {chosen.name} -> SEQUENTIAL: "
                    f"{reason}. Override with VIDEO_USE_FORCE_PARALLEL=1.",
                    file=_sys.stderr,
                )
                return Schedule.SEQUENTIAL

    return chosen


def parse_force_schedule(value: str | None) -> Schedule | None:
    """Convert a `--force-schedule {parallel,mixed,sequential,cpu}` CLI
    string to a Schedule, or None if the user didn't override.

    Raises ValueError on garbage so argparse can surface a friendly error.
    """
    if value is None:
        return None
    try:
        return Schedule(value.lower())
    except ValueError as e:
        valid = ", ".join(s.value for s in Schedule)
        raise ValueError(
            f"unknown schedule '{value}' (valid: {valid})"
        ) from e


# ---------------------------------------------------------------------------
# Smoke test — `python helpers/vram.py`
# ---------------------------------------------------------------------------


def main() -> None:
    info = detect_gpu()
    sched = pick_schedule(info)
    print(f"GPU      : {info}")
    print(f"Schedule : {sched.name}  (--force-schedule {sched.value})")


if __name__ == "__main__":
    main()
