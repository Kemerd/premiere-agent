"""Three-lane preprocessing orchestrator.

Coordinates the speech / audio / visual lanes for one or more source
videos. Picks a VRAM-aware schedule (3-parallel, 2+1 mixed, sequential,
or CPU fallback) and dispatches each lane via a child python process so
that:

  - GPU memory is fully released between lanes when running sequentially
    (important on 8 GB cards where Florence + Whisper barely co-fit)

  - Output streams from parallel lanes don't garble each other — every
    line a child emits gets prefixed with [whisper] / [audio] / [visual]
    by helpers/progress.install_lane_prefix(), and the orchestrator
    streams stdout/stderr line-by-line into its own log

  - The structured PROGRESS lines from helpers/progress.py pass straight
    through unchanged, so a downstream Claude / TUI / CI dashboard can
    aggregate them across all three lanes

The orchestrator itself does NOT import torch / transformers / panns —
those are heavy and we want `python helpers/preprocess.py --help` to
return in <100ms. All ML is in child processes.

CLI:
    python helpers/preprocess.py <video1> [<video2> ...] \\
        [--edit-dir <dir>] \\
        [--force-schedule {parallel,mixed,sequential,cpu}] \\
        [--wealthy] [--diarize] [--language en] [--force]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Sibling helpers — vram + extract_audio + progress are pure-stdlib so we
# can import freely without dragging torch into the orchestrator.
from extract_audio import extract_audio_for
from progress import child_env
from vram import (
    GpuInfo, Schedule, detect_gpu, parse_force_schedule, pick_schedule,
)
from wealthy import is_wealthy, propagate_to_env


HELPERS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Lane launch staggering
# ---------------------------------------------------------------------------
#
# When PARALLEL_3 (or any multi-lane parallel schedule) fires, all child
# subprocesses race to cuda.cudaMalloc their model weights in the same
# ~100ms window. Each lane's STEADY-STATE footprint may fit comfortably,
# but the TRANSIENT peak (weights materializing alongside intermediate
# activations + the CUDA caching allocator's slab reservations) can push
# the device past its VRAM ceiling and trigger an OOM — especially for
# Whisper-large-v3, whose KV cache grows during long-form generation
# right when Florence-2 is still committing its weights.
#
# The fix is cheap: stagger the thread starts so each child has time to
# settle its initial weight allocation before the next one fires.
#
# Why 8 seconds:
#   - Whisper-large-v3 weight load: ~3-5s on a fast NVMe + cu128 wheel
#   - Florence-2-base   weight load: ~2-3s
#   - PANNs CNN14       weight load: ~1-2s
#   - 8s gives comfortable headroom for the slowest lane to commit its
#     weights to VRAM before the next allocator hits the device
#   - Cost on the wall clock: 2 gaps × 8s = 16s extra at the start of a
#     3-15 minute preprocess run. Negligible vs. the OOM-and-retry path.
#
# Override at runtime with the env var VIDEO_USE_LANE_STAGGER_S (float,
# seconds). Set to 0 to disable staggering entirely, or crank it higher
# on slow NVMe / cold model caches where weight loads take longer.
LANE_STAGGER_S = 8.0


# ---------------------------------------------------------------------------
# Lane spec
# ---------------------------------------------------------------------------

@dataclass
class LaneJob:
    """One lane × N videos. Encapsulates the subprocess to spawn."""

    name: str                      # "whisper" | "audio" | "visual"
    script: str                    # filename in helpers/
    videos: list[Path]
    edit_dir: Path
    extra_args: list[str] = field(default_factory=list)
    # Populated after run() — used by the orchestrator's summary line.
    returncode: int = -1
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Subprocess runner with line-streamed output
# ---------------------------------------------------------------------------

def _stream_to(prefix_label: str, src, dst) -> None:
    """Copy lines from `src` (subprocess stdout/err) to `dst` (parent
    stderr) without buffering them up. The child has already prefixed
    its lines with [<lane>], so we just pass through.
    """
    try:
        for raw in iter(src.readline, b""):
            if not raw:
                break
            try:
                line = raw.decode("utf-8", errors="replace")
            except Exception:
                line = repr(raw) + "\n"
            dst.write(line)
            dst.flush()
    except Exception as e:
        dst.write(f"[orchestrator] stream from {prefix_label} died: {e}\n")
        dst.flush()


def _run_lane(job: LaneJob) -> int:
    """Spawn a child python that runs one batch of one lane.

    Each lane's batch entry point is exposed via its `__main__` block,
    but those take a SINGLE video path. To run a batch we use a tiny
    `-c` shim that imports the lane and calls its run_*_lane_batch.

    This avoids needing a separate `--videos a.mp4 b.mp4 c.mp4` flag on
    every lane CLI and keeps the lane scripts usable standalone.
    """
    # Build the shim. Keep it short enough to fit comfortably in argv.
    # Newlines are kept for readability; subprocess passes the whole
    # string to python as one -c argument.
    if job.name == "whisper":
        # ── Speech lane backend selection ─────────────────────────────
        # The lane is named "whisper" for historical reasons (progress
        # bars, cache contracts, log prefixes all key off this string).
        # Behind that name we dispatch to one of three implementations:
        #
        #   onnx    (default) - parakeet_onnx_lane (multi-session ORT
        #                       pool, ~10x faster than legacy whisper,
        #                       no transformers / DTW dependency)
        #   nemo              - parakeet_lane (NeMo torch-mode Parakeet,
        #                       fallback when ORT can't import e.g. on
        #                       exotic CPUs / corporate proxies)
        #   whisper           - whisper_lane (legacy HF transformers
        #                       pipeline; multilingual escape hatch when
        #                       Parakeet TDT v3 doesn't speak the user's
        #                       language)
        #
        # The output JSON shape is byte-identical across all three so
        # downstream consumers (pack_timelines, render, health) don't
        # notice the swap.
        backend = os.environ.get(
            "VIDEO_USE_SPEECH_LANE", "onnx"
        ).strip().lower() or "onnx"
        if backend == "onnx":
            import_line = (
                "from parakeet_onnx_lane import "
                "run_parakeet_onnx_lane_batch as fn"
            )
        elif backend == "nemo":
            import_line = (
                "from parakeet_lane import run_parakeet_lane_batch as fn"
            )
        elif backend == "whisper":
            import_line = (
                "from whisper_lane import run_whisper_lane_batch as fn"
            )
        else:
            raise ValueError(
                f"unknown VIDEO_USE_SPEECH_LANE={backend!r} "
                f"(valid: onnx, nemo, whisper)"
            )
    elif job.name == "audio":
        import_line = "from audio_lane import run_audio_lane_batch as fn"
    elif job.name == "visual":
        import_line = "from visual_lane import run_visual_lane_batch as fn"
    else:
        raise ValueError(f"unknown lane: {job.name}")

    # Serialize args via env to dodge quoting hell in the -c string.
    env = child_env(job.name)
    env["VIDEO_USE_LANE_VIDEOS"] = os.pathsep.join(str(v) for v in job.videos)
    env["VIDEO_USE_LANE_EDIT_DIR"] = str(job.edit_dir)
    env["VIDEO_USE_LANE_KWARGS_JSON"] = _kwargs_to_json(job.extra_args)
    # PYTHONPATH so the child can `from xxx import` siblings.
    env["PYTHONPATH"] = (
        str(HELPERS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    )

    shim = f"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, r"{HELPERS_DIR}")
from progress import install_lane_prefix
install_lane_prefix()
{import_line}
videos = [Path(p) for p in os.environ['VIDEO_USE_LANE_VIDEOS'].split(os.pathsep)]
edit_dir = Path(os.environ['VIDEO_USE_LANE_EDIT_DIR'])
kwargs = json.loads(os.environ.get('VIDEO_USE_LANE_KWARGS_JSON') or '{{}}')
fn(videos, edit_dir, **kwargs)
""".strip()

    cmd = [PYTHON, "-u", "-c", shim]
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(HELPERS_DIR),
        bufsize=0,
    )

    # Pump both pipes concurrently into the parent's stderr (so they
    # don't compete with any captured stdout from the orchestrator).
    t_out = threading.Thread(
        target=_stream_to, args=(job.name, proc.stdout, sys.stderr), daemon=True,
    )
    t_err = threading.Thread(
        target=_stream_to, args=(job.name, proc.stderr, sys.stderr), daemon=True,
    )
    t_out.start(); t_err.start()

    rc = proc.wait()
    t_out.join(timeout=2); t_err.join(timeout=2)

    job.returncode = rc
    job.elapsed_s = time.monotonic() - t0
    return rc


# ---------------------------------------------------------------------------
# kwargs serialization — tiny JSON encoder that handles the lane kwargs we
# actually pass (str/int/float/bool/None). Keeps the dependency surface flat.
# ---------------------------------------------------------------------------

def _kwargs_to_json(extra_args: list[str]) -> str:
    """Convert a flat ['--key', 'value', '--flag'] list into JSON kwargs.
    Handles bool flags (no value) by setting them True.
    """
    import json
    out: dict = {}
    i = 0
    while i < len(extra_args):
        tok = extra_args[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:].replace("-", "_")
        if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
            val: object = extra_args[i + 1]
            # Coerce common types so the lane signature lines up.
            if val == "None":
                val = None
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
            out[key] = val
            i += 2
        else:
            out[key] = True
            i += 1
    return json.dumps(out)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _shared_audio_extraction(videos: list[Path], edit_dir: Path) -> None:
    """Pre-extract 16kHz mono WAVs for all videos before the lanes run.

    Doing this up front (instead of inside whisper_lane / audio_lane)
    means we get a clean progress bar over the ffmpeg work AND we avoid
    a race when whisper + audio lanes run in parallel and both try to
    extract the same WAV at the same time.
    """
    from progress import lane_progress

    with lane_progress(
        "extract",
        total=len(videos),
        unit="video",
        desc="ffmpeg → 16kHz mono WAV",
    ) as bar:
        for v in videos:
            bar.start_item(v.name)
            extract_audio_for(v, edit_dir, verbose=True)
            bar.update(advance=1)


def _spawn_parallel(jobs: list[LaneJob]) -> list[int]:
    """Run jobs concurrently in their own threads. Returns return codes
    in the same order. Each job uses its own subprocess for GPU isolation
    (CUDA contexts are per-process), so this is safe even with 3 GPU
    lanes simultaneously — they share the device but not the allocator.

    Launch staggering
    -----------------
    To avoid a simultaneous cuda.cudaMalloc spike across all children
    (which can OOM even when steady-state VRAM would fit), thread starts
    are spaced by `LANE_STAGGER_S` seconds. The FIRST thread starts
    immediately; only the 2nd / 3rd / Nth get delayed. The delay is read
    once from the env var `VIDEO_USE_LANE_STAGGER_S` (float, seconds),
    falling back to the module constant `LANE_STAGGER_S`. Set the env
    var to 0 to disable staggering entirely (e.g. on multi-GPU rigs
    where each lane gets its own device).
    """
    # ---------------------------------------------------------------
    # Resolve the per-launch delay once, up front. We parse defensively
    # because env vars are user-supplied strings and a typo here would
    # otherwise crash the orchestrator mid-dispatch.
    # ---------------------------------------------------------------
    raw = os.environ.get("VIDEO_USE_LANE_STAGGER_S")
    if raw is None or raw == "":
        stagger_s = LANE_STAGGER_S
    else:
        try:
            stagger_s = float(raw)
        except ValueError:
            print(
                f"[orchestrator] WARN: VIDEO_USE_LANE_STAGGER_S={raw!r} is "
                f"not a float; falling back to {LANE_STAGGER_S}s"
            )
            stagger_s = LANE_STAGGER_S
    # Negative values are nonsense — clamp to 0 (i.e. no stagger).
    if stagger_s < 0:
        stagger_s = 0.0

    results: dict[int, int] = {}
    threads: list[threading.Thread] = []

    def runner(idx: int, job: LaneJob) -> None:
        results[idx] = _run_lane(job)

    # ---------------------------------------------------------------
    # Launch loop. We sleep BETWEEN starts (not before the first, not
    # after the last) so the wall-clock cost is exactly
    # (N - 1) * stagger_s and the first lane can begin loading weights
    # the instant we get here.
    # ---------------------------------------------------------------
    for idx, job in enumerate(jobs):
        if idx > 0 and stagger_s > 0:
            print(
                f"[orchestrator] staggering {stagger_s:.0f}s before "
                f"launching {job.name}"
            )
            time.sleep(stagger_s)
        t = threading.Thread(target=runner, args=(idx, job), daemon=False)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    return [results.get(i, -1) for i in range(len(jobs))]


def _dispatch(jobs: list[LaneJob], schedule: Schedule) -> list[LaneJob]:
    """Run lane jobs according to schedule. Returns the same job list
    (mutated with returncode + elapsed_s) so the caller can summarize.
    """
    by_name = {j.name: j for j in jobs}
    whisper = by_name.get("whisper")
    audio = by_name.get("audio")
    visual = by_name.get("visual")

    # A null lane (job==None) means the user filtered it out via flags.
    # Build the actual run plan from whichever lanes exist.

    if schedule == Schedule.PARALLEL_3:
        present = [j for j in (whisper, audio, visual) if j is not None]
        print(f"[orchestrator] PARALLEL_3: dispatching {[j.name for j in present]}")
        _spawn_parallel(present)

    elif schedule == Schedule.PARALLEL_2_SEQ_1:
        # Whisper + PANNs share the audio WAV and have a small combined
        # footprint (~3 GB) — pair them. Florence is the memory hog so
        # it gets the GPU to itself.
        pair = [j for j in (whisper, audio) if j is not None]
        if pair:
            print(f"[orchestrator] PARALLEL_2: dispatching {[j.name for j in pair]}")
            _spawn_parallel(pair)
        if visual is not None:
            print("[orchestrator] SEQ:        dispatching visual")
            _run_lane(visual)

    elif schedule == Schedule.SEQUENTIAL:
        for j in (whisper, audio, visual):
            if j is None:
                continue
            print(f"[orchestrator] SEQ:        dispatching {j.name}")
            _run_lane(j)

    elif schedule == Schedule.CPU_FALLBACK:
        # CPU is bottlenecked by core count — running parallel lanes
        # just thrashes the cache. Strict sequential, and we propagate
        # device=cpu to each lane via extra_args set up by the caller.
        for j in (whisper, audio, visual):
            if j is None:
                continue
            print(f"[orchestrator] CPU SEQ:    dispatching {j.name}")
            _run_lane(j)

    return jobs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_preprocess(
    videos: list[Path],
    edit_dir: Path,
    *,
    schedule: Schedule | None = None,
    skip_whisper: bool = False,
    skip_audio: bool = False,
    skip_visual: bool = False,
    wealthy: bool = False,
    diarize: bool = False,
    language: str | None = None,
    force: bool = False,
) -> list[LaneJob]:
    """Top-level orchestration. Returns finished LaneJob list."""
    if not videos:
        raise ValueError("no videos given")
    edit_dir = edit_dir.resolve()
    edit_dir.mkdir(parents=True, exist_ok=True)

    # Wealthy mode is global: propagate to env so child lanes see it
    # without us having to add --wealthy to every shim invocation.
    propagate_to_env(wealthy)

    # 1) GPU detection + schedule resolution
    info = detect_gpu()
    auto_picked = schedule is None
    if auto_picked:
        schedule = pick_schedule(info)
    print(f"[orchestrator] GPU      : {info}")
    print(f"[orchestrator] Schedule : {schedule.name}  "
          f"(--force-schedule {schedule.value})")
    # When the auto-picker landed on SEQUENTIAL but the card has the
    # VRAM headroom for parallel, surface the opt-in path once. This
    # is the new default policy as of the post-Florence-community
    # refactor — see vram.py module docstring for the rationale.
    if (
        auto_picked
        and schedule == Schedule.SEQUENTIAL
        and info.available
        and info.free_gb >= 8.0
        and not os.environ.get("VIDEO_USE_PARALLEL_LANES", "").strip()
    ):
        print(
            "[orchestrator] (parallel lanes available — set "
            "VIDEO_USE_PARALLEL_LANES=1 or pass --force-schedule parallel "
            "to opt in; sequential is the safer default on single-GPU rigs)"
        )
    print(f"[orchestrator] Wealthy  : {is_wealthy(False)}")
    print(f"[orchestrator] Videos   : {len(videos)}")

    # 2) Shared audio extraction (cheap, single-threaded ffmpeg)
    _shared_audio_extraction(videos, edit_dir)

    # 3) Build lane jobs
    use_cpu = (schedule == Schedule.CPU_FALLBACK)
    device_arg = "cpu" if use_cpu else "cuda:0"
    audio_device_arg = "cpu" if use_cpu else "cuda"

    jobs: list[LaneJob] = []
    if not skip_whisper:
        wargs = ["--device", device_arg]
        if language:
            wargs += ["--language", language]
        if diarize:
            wargs += ["--diarize"]
        if force:
            wargs += ["--force"]
        # Note: don't pass --wealthy here — it's already in env via
        # propagate_to_env, and is_wealthy(False) inside the lane will
        # read it from the env automatically.
        jobs.append(LaneJob("whisper", "whisper_lane.py", videos, edit_dir, wargs))

    if not skip_audio:
        aargs = ["--device", audio_device_arg]
        if force:
            aargs += ["--force"]
        jobs.append(LaneJob("audio", "audio_lane.py", videos, edit_dir, aargs))

    if not skip_visual:
        vargs = ["--device", device_arg]
        if force:
            vargs += ["--force"]
        jobs.append(LaneJob("visual", "visual_lane.py", videos, edit_dir, vargs))

    if not jobs:
        print("[orchestrator] no lanes selected — nothing to do.")
        return []

    # 4) Dispatch
    t0 = time.monotonic()
    _dispatch(jobs, schedule)
    total_s = time.monotonic() - t0

    # 5) Summary
    print("\n[orchestrator] ===== preprocessing summary =====")
    for j in jobs:
        status = "OK" if j.returncode == 0 else f"FAIL (rc={j.returncode})"
        print(f"  {j.name:<8}  {status:<20}  {j.elapsed_s:6.1f}s")
    print(f"  total wall: {total_s:.1f}s")

    failed = [j for j in jobs if j.returncode != 0]
    if failed:
        print(f"[orchestrator] {len(failed)} lane(s) failed: "
              f"{[j.name for j in failed]}", file=sys.stderr)
    else:
        print("[orchestrator] all lanes complete.")

    return jobs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Three-lane preprocessor (speech / audio / visual)",
    )
    ap.add_argument("videos", nargs="+", type=Path,
                    help="Source video files (one or many)")
    ap.add_argument("--edit-dir", type=Path, default=None,
                    help="Edit output dir (default: <first video parent>/edit)")
    ap.add_argument("--force-schedule",
                    choices=[s.value for s in Schedule],
                    default=None,
                    help="Override the auto VRAM-based schedule")
    ap.add_argument("--skip-whisper", action="store_true")
    ap.add_argument("--skip-audio", action="store_true")
    ap.add_argument("--skip-visual", action="store_true")
    ap.add_argument("--wealthy", action="store_true",
                    help="Speed knob for 24GB+ GPUs. Bigger batches in all "
                         "lanes; same models, same outputs. Also reads "
                         "VIDEO_USE_WEALTHY=1.")
    ap.add_argument("--diarize", action="store_true",
                    help="Enable pyannote speaker diarization (needs HF_TOKEN)")
    ap.add_argument("--language", default=None,
                    help="ISO language code passed to Whisper (default: auto)")
    ap.add_argument("--force", action="store_true",
                    help="Bypass per-lane caches, always re-run")
    args = ap.parse_args()

    # ffmpeg is a hard prereq for both extract_audio and visual_lane.
    if not shutil.which("ffmpeg"):
        sys.exit("[orchestrator] FATAL: ffmpeg not on PATH. "
                 "Install via winget/choco/apt/brew. See README.")

    videos = [v.resolve() for v in args.videos]
    missing = [v for v in videos if not v.exists()]
    if missing:
        sys.exit(f"[orchestrator] FATAL: not found: "
                 f"{', '.join(str(m) for m in missing)}")

    edit_dir = (args.edit_dir or (videos[0].parent / "edit")).resolve()

    schedule = parse_force_schedule(args.force_schedule)

    jobs = run_preprocess(
        videos=videos,
        edit_dir=edit_dir,
        schedule=schedule,
        skip_whisper=args.skip_whisper,
        skip_audio=args.skip_audio,
        skip_visual=args.skip_visual,
        wealthy=args.wealthy,
        diarize=args.diarize,
        language=args.language,
        force=args.force,
    )
    sys.exit(0 if all(j.returncode == 0 for j in jobs) else 1)


if __name__ == "__main__":
    main()
