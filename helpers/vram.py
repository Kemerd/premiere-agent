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


def pick_schedule(info: GpuInfo) -> Schedule:
    """Map a GpuInfo to a Schedule. See module docstring for rationale.

    Conservative — when in doubt, drop a tier. OOM during preprocessing
    crashes the whole batch and forces the user to re-run; a slower
    schedule that completes is strictly better than a fast one that doesn't.
    """
    if not info.available:
        return Schedule.CPU_FALLBACK
    if info.free_gb >= _PARALLEL_3_MIN_GB:
        return Schedule.PARALLEL_3
    if info.free_gb >= _PARALLEL_2_MIN_GB:
        return Schedule.PARALLEL_2_SEQ_1
    if info.free_gb >= _SEQUENTIAL_MIN_GB:
        return Schedule.SEQUENTIAL
    return Schedule.CPU_FALLBACK


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
