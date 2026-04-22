"""VRAM detection + lane scheduling.

The three-lane preprocessor wants to run the speech / audio / visual
lanes in parallel when there's room — and gracefully fall back when
there isn't. This module owns that decision.

Approximate steady-state model footprints with conservative settings:

  ── Speech lanes (one runs at a time per orchestrator dispatch) ──
  - parakeet-tdt-0.6b-v2 ONNX, fp16, 1 session:     ~1.6 GB peak
  - parakeet-tdt-0.6b-v2 ONNX, fp16, 4-session pool: ~6.4 GB peak
  - parakeet-tdt-0.6b-v2 ONNX, fp16, 8-session pool: ~12.8 GB peak
  - parakeet-tdt-0.6b-v2 ONNX, int8, 4-session pool: ~3.2 GB peak
  - parakeet-tdt-0.6b NeMo torch, fp16 (fallback):  ~3-4 GB

  ── Other lanes ──
  - CLAP audio lane (Xenova/clap-htsat-unfused, ONNX):     ~1.5 GB peak
    (audio + text encoders resident, batch=16 windows × 10s × 48kHz
     transient mel + activation working set; ~3 GB peak with the
     "large" tier under --wealthy)
  - Florence-2-base ONNX, fp16, batch 8 + KV cache:        ~1.8 GB peak
    (4 ONNX subgraphs co-resident: vision_encoder + embed_tokens +
     encoder_model + decoder_model_merged; B*beams=24 KV cache;
     measured via nvidia-smi during a 50-frame test batch on a 5090.
     Multi-instance pool scales linearly: 2 captioners ≈ 3.2 GB peak.)

The default speech lane is the multi-session ONNX path (see
helpers/parakeet_onnx_lane.py / helpers/_onnx_pool.py) which auto-clamps
its pool size to fit available VRAM. The visual lane uses the same
multi-session pattern via helpers/_florence_pool.py — also VRAM-clamped.

Parallel-mode opt-in viability:
the audio lane uses a small CLAP encoder (~1.5 GB) and Florence is now
on a leaner ONNX path (~1.8 GB), so PARALLEL_3 (speech + audio + visual)
is viable on any 12 GB+ card — Parakeet-pool ~6 GB + CLAP ~1.5 GB +
Florence ~1.8 GB fits with headroom. The 8 GB / 4 GB / 2 GB thresholds
in `pick_schedule` are unchanged. The default schedule is still
SEQUENTIAL for the unrelated multi-context CUDA allocator fragmentation
issues called out below; users on multi-GPU rigs or datacenter cards
opt into parallel via VIDEO_USE_PARALLEL_LANES=1.

Detection strategy (in order):

  1. pynvml.nvmlDeviceGetMemoryInfo()       — fastest path when the
                                              `pynvml` package is on
                                              the system (often a
                                              transitive dep of various
                                              CUDA toolkits / nvidia
                                              ML stacks).  No subprocess
                                              spawn, no PYTHONPATH stunts.
  2. nvidia-smi --query-gpu=memory.free     — primary expectation now
                                              that torch is no longer
                                              guaranteed to be installed.
                                              Works on any host with the
                                              NVIDIA driver, no python
                                              deps.  Subprocess cost is
                                              ~50ms which is fine for a
                                              one-shot probe at startup.
  3. torch.cuda.mem_get_info(device=0)      — opportunistic fast-path,
                                              only fires if torch was
                                              installed alongside the
                                              [diarize] extra.  Demoted
                                              from primary because the
                                              default preprocess install
                                              is now torch-FREE thanks
                                              to the Florence-2 ONNX port.
  4. None                                   — no GPU, fall through to CPU tier

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
    PARALLEL_2_SEQ_1 = "mixed"     # speech || audio, then florence solo
    SEQUENTIAL = "sequential"      # one lane at a time, GPU
    CPU_FALLBACK = "cpu"           # no CUDA — int8 speech, CLAP CPU EP, Florence CPU


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


def _try_pynvml() -> GpuInfo | None:
    """Use NVIDIA's NVML library directly via the ``pynvml`` Python wrapper.

    Fastest path when available -- no subprocess spawn, no string
    parsing, just two C calls into libnvidia-ml.so / nvml.dll.
    Returns None when the package isn't installed (the common case on
    a default install -- pynvml is not in our dependency list, only
    occasionally pulled transitively by CUDA toolkits or other ML
    libraries the user might have around).

    Init/shutdown are paired in a try/finally so a partially-initialized
    NVML state can't leak across processes -- the orchestrator calls
    ``detect_gpu()`` repeatedly during its scheduling pass and we want
    each call to be a pure function.
    """
    try:
        import pynvml  # type: ignore  # optional dep; fail-soft on missing
    except ImportError:
        return None

    # NVMLError can be raised by every nvml* call below if the driver
    # isn't loaded or the GPU is in an inaccessible state (e.g. ECC
    # error recovery). We catch broadly rather than enumerate every
    # NVML error code -- the next probe in the chain handles fallback.
    try:
        pynvml.nvmlInit()
    except Exception:
        return None

    try:
        # Device 0 is the convention; multi-GPU users set CUDA_VISIBLE_DEVICES
        # to limit which GPU we see (NVML respects the env var when set).
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # nvmlDeviceGetName returns bytes on older bindings, str on newer.
        # decode() on a str raises AttributeError; .decode if bytes else str.
        raw_name = pynvml.nvmlDeviceGetName(handle)
        name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        return GpuInfo(
            available=True,
            device_name=name,
            total_gb=mem.total / (1024 ** 3),
            free_gb=mem.free / (1024 ** 3),
        )
    except Exception:
        return None
    finally:
        # NVML init refcounts internally so this is cheap on every call;
        # always shut down so the next probe has a clean state if it
        # also wants to call NVML (currently only this probe does).
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _try_nvidia_smi() -> GpuInfo | None:
    """Probe via the ``nvidia-smi`` CLI shipped with the NVIDIA driver.

    Primary expectation now that the default preprocess install is
    torch-FREE (Florence-2 ONNX port). Works on any host with the
    NVIDIA driver -- doesn't require pynvml, doesn't require torch,
    doesn't require any python-side CUDA package.

    Subprocess cost is ~50 ms which is fine for a one-shot probe at
    orchestrator startup. Returns None if nvidia-smi isn't on PATH or
    its output can't be parsed.
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


def _try_torch() -> GpuInfo | None:
    """Use torch.cuda directly. Opportunistic fast-path.

    Demoted from primary detection because the default preprocess
    install is now torch-FREE -- torch only appears when the user has
    opted into the [diarize] extra (or installed it manually for some
    other reason). Still kept in the chain because it's the lowest-
    overhead probe when torch happens to be around: no subprocess, no
    NVML init, just one C call into the cached CUDA driver handle.

    Returns None if torch isn't installed or CUDA isn't available --
    the caller will have already tried pynvml and nvidia-smi above.
    """
    try:
        import torch  # local import: torch is opt-in via [diarize]
    except ImportError:
        return None

    # torch.cuda probes can raise if libcudart resolution fails on a
    # mismatched driver -- catch defensively so we can still fall
    # through to None and let the caller handle CPU fallback.
    try:
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
    except Exception:
        return None


def detect_gpu() -> GpuInfo:
    """Single entry point for callers. Always returns a GpuInfo; callers
    should branch on `info.available` rather than handling None.

    Probe order: pynvml -> nvidia-smi -> torch -> CPU fallback.
    See module docstring for the rationale on each tier.
    """
    for fn in (_try_pynvml, _try_nvidia_smi, _try_torch):
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
# Default policy: SEQUENTIAL.
#
# Why: real-world testing on a Blackwell RTX 5090 (32 GB) showed that
# even with comfortable VRAM headroom, multi-process CUDA on a single
# GPU produces hard-to-diagnose OOMs and context corruption when three
# heavy ML lanes (Parakeet ONNX pool, CLAP, Florence-2) load and infer concurrently.
# Each lane is its own subprocess with its own CUDA context; the
# driver-level allocator fragments badly across contexts, and one lane's
# transient inference spike can starve another's resident weights
# mid-forward. nvidia-smi snapshots at the OOM moment show 31 GB used
# across all procs in a 32 GB card — not because any single lane needs
# that much, but because PyTorch's caching allocator in three contexts
# simultaneously claims more than the sum of their working sets. This
# is a CUDA-runtime / multi-process issue, NOT a per-lane footprint
# issue: even with the lightweight CLAP audio lane (~1.5 GB) the
# fragmentation pattern persists.
#
# Sequential schedule sidesteps the entire class of problems: each lane
# loads, runs, releases, then the next lane starts. End-to-end wall
# time is competitive with parallel in practice because the parallel
# schedule's OOM-retry-backoff loop wastes more time than the
# serialization saves.
#
# We KEEP the PARALLEL_3 / PARALLEL_2_SEQ_1 code paths for power users
# with multi-GPU rigs, datacenter cards (A100, H100, RTX PRO 6000) where
# the driver allocator handles concurrent contexts cleanly, or anyone
# who's measured their setup and knows parallel is faster for them.
# Activate via:
#
#    --force-schedule parallel        (CLI flag, single run)
#    --force-schedule mixed           (speech||audio then florence solo)
#    VIDEO_USE_PARALLEL_LANES=1       (env var, persists across runs)
#
# When opt-in is set, we still apply the historical VRAM-driven tier
# selection (so a 4 GB card opting in still gets PARALLEL_2_SEQ_1
# instead of crashing in PARALLEL_3).
# ---------------------------------------------------------------------------

# Truthy values for any boolean env var we read here. Centralized so
# every check uses the same parse rules.
_TRUTHY_ENV = ("1", "true", "yes", "on", "y", "t")


def _user_opted_into_parallel() -> bool:
    """True if the user has explicitly opted into parallel scheduling
    via the env var. CLI `--force-schedule` is handled separately by
    `parse_force_schedule` and bypasses `pick_schedule` entirely, so
    this only governs the auto-pick path.
    """
    raw = os.environ.get("VIDEO_USE_PARALLEL_LANES", "").strip().lower()
    return raw in _TRUTHY_ENV


def pick_schedule(info: GpuInfo) -> Schedule:
    """Pick a Schedule for the auto path (no `--force-schedule` flag).

    Default is SEQUENTIAL on any GPU — see the policy comment block
    above for the full rationale. The historical VRAM-tiered parallel
    path is preserved but only fires when the user opts in via the
    `VIDEO_USE_PARALLEL_LANES` env var (or the CLI `--force-schedule`
    flag, which bypasses this function entirely).
    """
    if not info.available:
        return Schedule.CPU_FALLBACK

    # Power-user opt-in path: re-enable the original VRAM-driven tier
    # selection. Same thresholds as the historical defaults — kept so
    # multi-GPU / datacenter users aren't surprised.
    if _user_opted_into_parallel():
        if info.free_gb >= _PARALLEL_3_MIN_GB:
            return Schedule.PARALLEL_3
        if info.free_gb >= _PARALLEL_2_MIN_GB:
            return Schedule.PARALLEL_2_SEQ_1
        # Fallthrough to sequential below — even if the user opted in,
        # we won't run parallel under the 4 GB threshold.

    # Default path: sequential on any GPU with enough VRAM for the
    # smallest single lane (Parakeet 1-session ~1.6 GB or Florence-2
    # ~2.5 GB; we keep the historical 3 GB lower bound for headroom).
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
