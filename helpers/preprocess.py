"""Three-lane preprocessing orchestrator.

Coordinates the speech / audio / visual lanes for one or more source
videos. Picks a VRAM-aware schedule (3-parallel, 2+1 mixed, sequential,
or CPU fallback) and dispatches each lane via a child python process so
that:

  - GPU memory is fully released between lanes when running sequentially
    (important on 8 GB cards where Florence + Parakeet barely co-fit)

  - Output streams from parallel lanes don't garble each other — every
    line a child emits gets prefixed with [speech] / [audio] / [visual]
    by helpers/progress.install_lane_prefix(), and the orchestrator
    streams stdout/stderr line-by-line into its own log

  - The structured PROGRESS lines from helpers/progress.py pass straight
    through unchanged, so a downstream Claude / TUI / CI dashboard can
    aggregate them across all three lanes

The orchestrator itself does NOT import torch / transformers / onnxruntime
— those are heavy and we want `python helpers/preprocess.py --help` to
return in <100ms. All ML is in child processes.

CLI:
    python helpers/preprocess.py <video1> [<video2> ...] \\
        [--edit-dir <dir>] \\
        [--force-schedule {parallel,mixed,sequential,cpu}] \\
        [--wealthy] [--diarize] [--language en] [--force]
"""

from __future__ import annotations

import argparse
import json
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
from wealthy import FLORENCE_BATCH, is_wealthy, propagate_to_env


HELPERS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Audio-only source classification
# ---------------------------------------------------------------------------
#
# The pipeline is happy to accept audio-only sources (e.g. a voiceover
# .wav recorded after the fact, a podcast .mp3, a music bed) alongside
# the usual video files. Speech (Parakeet ONNX) and the optional CLAP
# audio lane both consume only the 16kHz mono WAV cache produced by
# extract_audio.py — they don't care whether the source had video. The
# visual lane (Florence-2) does need frames, so audio-only sources are
# filtered out of the visual job below before dispatch.
#
# Keep this list in sync with preprocess_batch.AUDIO_ONLY_EXTS. We
# duplicate it here (instead of importing) because preprocess.py is the
# more upstream module — preprocess_batch imports it, not the other
# way round, and pulling the constant from a sibling would create a
# circular-feeling dependency for a 8-element set.
AUDIO_ONLY_EXTS = frozenset({
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
})


def _is_audio_only(path: Path) -> bool:
    """True if `path` is an audio-only container (no video stream by
    convention). Classification is suffix-only — see the AUDIO_ONLY_EXTS
    docstring for rationale.
    """
    return path.suffix.lower() in AUDIO_ONLY_EXTS


def _has_audio_stream(path: Path) -> bool:
    """Return True when ffprobe sees at least one audio stream.

    Some B-roll cameras / transfer tools produce perfectly valid video
    files with no audio stream. Those should still go to the visual lane,
    but they must not be sent through extraction, speech, or CLAP.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        # The health check enforces ffprobe, but keep direct library use
        # from crashing here. Extraction will raise a clearer ffmpeg error.
        return True
    proc = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-print_format", "json",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        return False
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return False
    return any(
        isinstance(s, dict) and s.get("codec_type") == "audio"
        for s in data.get("streams", [])
    )


# ---------------------------------------------------------------------------
# Lane launch staggering
# ---------------------------------------------------------------------------
#
# When PARALLEL_3 (or any multi-lane parallel schedule) fires, all child
# subprocesses race to cuda.cudaMalloc their model weights in the same
# ~100ms window. Each lane's STEADY-STATE footprint may fit comfortably,
# but the TRANSIENT peak (weights materializing alongside intermediate
# activations + the CUDA caching allocator's slab reservations) can push
# the device past its VRAM ceiling and trigger an OOM — especially when
# Florence-2 is still committing its weights right as the Parakeet ONNX
# pool spins up its N sessions in parallel.
#
# The fix is cheap: stagger the thread starts so each child has time to
# settle its initial weight allocation before the next one fires.
#
# Why 8 seconds:
#   - Parakeet ONNX pool     weight load: ~2-4s for 4-8 sessions
#   - Florence-2-base        weight load: ~2-3s
#   - CLAP ONNX (audio+text) weight load: ~1-2s (small graphs, fast)
#   - 8s gives comfortable headroom for any single lane to commit its
#     weights before the next allocator hits the device, with margin
#     for cold model downloads on the first run of each lane.
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

    name: str                      # "speech" | "audio" | "visual"
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
    if job.name == "speech":
        # ── Speech lane backend selection ─────────────────────────────
        # Two implementations, both Parakeet TDT, byte-identical JSON
        # output:
        #
        #   onnx (default)  - parakeet_onnx_lane (multi-session ORT
        #                     pool, default everywhere — ~10x the
        #                     throughput of the encoder-decoder ASR
        #                     stacks we used to run, no transformers
        #                     dependency, no DTW alignment).
        #   nemo            - parakeet_lane (NeMo torch-mode Parakeet,
        #                     fallback when ORT can't import e.g. on
        #                     exotic CPUs / corporate proxies that
        #                     block the ONNX repo entirely).
        #
        # Whisper is intentionally NOT a backend here — see
        # parakeet_onnx_lane.py's module docstring for the post-mortem
        # on why we ripped it out. Don't add a 'whisper' branch.
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
        else:
            raise ValueError(
                f"unknown VIDEO_USE_SPEECH_LANE={backend!r} "
                f"(valid: onnx, nemo)"
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

def _shared_audio_extraction(sources: list[Path], edit_dir: Path) -> None:
    """Pre-extract 16kHz mono WAVs for all audio-capable sources.

    Doing this up front (instead of inside the speech / audio lanes)
    means we get a clean progress bar over the ffmpeg work AND we avoid
    a race when speech + audio lanes run in parallel and both try to
    extract the same WAV at the same time. The CLAP audio lane reads
    the same 16kHz mono PCM cache and resamples to 48kHz on the fly
    (CLAP's native rate) — no second WAV cache on disk.
    """
    from progress import lane_progress

    with lane_progress(
        "extract",
        total=len(sources),
        unit="source",
        desc="ffmpeg → 16kHz mono WAV",
    ) as bar:
        for v in sources:
            bar.start_item(v.name)
            try:
                extract_audio_for(v, edit_dir, verbose=True)
            except Exception as exc:
                print(
                    f"  audio_16k skipped: {v.name} "
                    f"({type(exc).__name__}: {exc})",
                    file=sys.stderr,
                )
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
    speech = by_name.get("speech")
    audio = by_name.get("audio")
    visual = by_name.get("visual")

    # A null lane (job==None) means the user filtered it out via flags.
    # Build the actual run plan from whichever lanes exist.
    #
    # Note: the audio lane is OFF by default in the standard preprocess
    # workflow — it's a Phase B step the agent invokes separately with
    # a custom CLAP vocabulary derived from the speech + visual
    # timelines. `--include-audio` opts into running it inline here
    # with the baked-in baseline vocabulary, mostly for smoke tests
    # and offline / agent-less usage.

    if schedule == Schedule.PARALLEL_3:
        present = [j for j in (speech, audio, visual) if j is not None]
        print(f"[orchestrator] PARALLEL_3: dispatching {[j.name for j in present]}")
        _spawn_parallel(present)

    elif schedule == Schedule.PARALLEL_2_SEQ_1:
        # Pairing: Parakeet + the small CLAP audio lane share the WAV
        # cache with a tiny combined footprint (~6 GB Parakeet pool +
        # ~1.5 GB CLAP ≈ 7.5 GB), while Florence (~2.5 GB) gets the
        # GPU solo afterward. Comfortably fits an 8 GB card. The
        # default SEQUENTIAL schedule is still recommended on
        # single-GPU rigs because of the multi-context CUDA allocator
        # fragmentation issue documented in vram.py — but PARALLEL_2
        # is now genuinely viable again on consumer hardware after
        # the CLAP migration.
        pair = [j for j in (speech, audio) if j is not None]
        if pair:
            print(f"[orchestrator] PARALLEL_2: dispatching {[j.name for j in pair]}")
            _spawn_parallel(pair)
        if visual is not None:
            print("[orchestrator] SEQ:        dispatching visual")
            _run_lane(visual)

    elif schedule == Schedule.SEQUENTIAL:
        for j in (speech, audio, visual):
            if j is None:
                continue
            print(f"[orchestrator] SEQ:        dispatching {j.name}")
            _run_lane(j)

    elif schedule == Schedule.CPU_FALLBACK:
        # CPU is bottlenecked by core count — running parallel lanes
        # just thrashes the cache. Strict sequential, and we propagate
        # device=cpu to each lane via extra_args set up by the caller.
        for j in (speech, audio, visual):
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
    skip_speech: bool = False,
    include_audio: bool = False,
    skip_visual: bool = False,
    wealthy: bool = False,
    diarize: bool = False,
    language: str | None = None,
    force: bool = False,
    visual_fps: float | None = None,
) -> list[LaneJob]:
    """Top-level orchestration. Returns finished LaneJob list.

    Args:
        videos: All sources to preprocess. Despite the legacy name,
            this list may MIX video files (.mp4, .mov, ...) and
            audio-only files (.wav, .mp3, ...). Audio-only sources
            are auto-detected by extension (see AUDIO_ONLY_EXTS) and
            routed to the speech / CLAP lanes only — they are filtered
            out of the visual lane because Florence-2 needs frames.
            Use cases for audio-only sources: voiceover .wav recorded
            after the shoot, music bed, podcast .mp3 being repurposed,
            etc.
        edit_dir: Output directory for caches + lane outputs.

    Lanes:
        speech  (Parakeet ONNX)  - ON by default. Runs on ALL sources
                                   (video and audio-only).
        visual  (Florence-2)     - ON by default. Runs on VIDEO sources
                                   only; if the batch is entirely
                                   audio-only the lane is skipped
                                   gracefully with a log line.
        audio   (CLAP zero-shot) - OFF by default; opt in with
                                   `include_audio=True` to run the
                                   baseline-vocab Phase A path inline.
                                   Runs on ALL sources when enabled.
                                   The agent-curated Phase B path is
                                   the recommended workflow and runs
                                   as a separate audio_lane.py
                                   invocation after preprocess
                                   completes.
    """
    if not videos:
        raise ValueError("no sources given")
    edit_dir = edit_dir.resolve()
    edit_dir.mkdir(parents=True, exist_ok=True)

    # ── Classify sources: video vs audio-only ──
    # The `videos` parameter is a misnomer kept for backward-compat —
    # in practice it's "all sources, video and audio-only mixed". We
    # split them here so each lane gets the right scope:
    #   speech / audio (CLAP) lanes  -> all sources (they read the
    #                                   16kHz mono WAV cache, which
    #                                   extract_audio_for() builds for
    #                                   any media file ffmpeg can read)
    #   visual (Florence-2)  lane    -> video sources ONLY (it needs
    #                                   frames; audio-only sources
    #                                   would crash the prefetch)
    video_sources = [v for v in videos if not _is_audio_only(v)]
    audio_only_sources = [v for v in videos if _is_audio_only(v)]
    audio_sources = [v for v in videos if _has_audio_stream(v)]
    audio_source_set = set(audio_sources)
    visual_only_sources = [
        v for v in video_sources
        if v not in audio_source_set
    ]

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
    print(
        f"[orchestrator] Sources  : {len(video_sources)} video, "
        f"{len(audio_only_sources)} audio-only "
        f"({len(videos)} total)"
    )
    if visual_only_sources:
        print(
            f"[orchestrator] Visual-only: {len(visual_only_sources)} video "
            "source(s) have no audio stream; skipping extraction/speech/audio "
            "for them, keeping them for visual captions."
        )
        for src in visual_only_sources[:10]:
            print(f"  - [visual-only] {src.name}")
        if len(visual_only_sources) > 10:
            print(f"  ... +{len(visual_only_sources) - 10} more")

    # 2) Shared audio extraction (cheap, single-threaded ffmpeg)
    _shared_audio_extraction(audio_sources, edit_dir)

    # 3) Build lane jobs
    use_cpu = (schedule == Schedule.CPU_FALLBACK)
    device_arg = "cpu" if use_cpu else "cuda:0"
    audio_device_arg = "cpu" if use_cpu else "cuda"

    jobs: list[LaneJob] = []
    if not skip_speech and audio_sources:
        wargs = ["--device", device_arg]
        if language:
            wargs += ["--language", language]
        if diarize:
            wargs += ["--diarize"]
        if force:
            wargs += ["--force"]
        # Note: don't pass --wealthy here — it's already in env via
        # propagate_to_env, and is_wealthy(False) inside the lane will
        # read it from the env automatically. The actual backend
        # script (parakeet_onnx_lane vs parakeet_lane) is selected
        # inside _run_lane() based on VIDEO_USE_SPEECH_LANE; the
        # `script` field below is just a label kept around for log
        # diagnostics and the legacy LaneJob shape.
        jobs.append(LaneJob("speech", "parakeet_onnx_lane.py", audio_sources, edit_dir, wargs))
    elif not skip_speech and not audio_sources:
        print("[orchestrator] speech lane skipped - no audio-capable sources")

    if include_audio:
        # Phase A audio with the baked-in baseline vocabulary. Off by
        # default — the recommended workflow is for the editor agent
        # to read the speech + visual timelines, generate a project-
        # specific CLAP vocabulary, and then invoke audio_lane.py
        # directly (Phase B). This branch exists so smoke tests and
        # agent-less batch runs still produce SOMETHING in audio_tags/.
        aargs = ["--device", audio_device_arg]
        if force:
            aargs += ["--force"]
        if audio_sources:
            jobs.append(LaneJob("audio", "audio_lane.py", audio_sources, edit_dir, aargs))
        else:
            print("[orchestrator] audio lane skipped - no audio-capable sources")

    # ── Visual lane scoping ───────────────────────────────────────────
    # Florence-2 needs frames, so we hand it ONLY the video sources.
    # If the entire batch is audio-only (e.g. user is running through
    # a folder of voiceover .wav files for a scripted assembly), we
    # skip the visual lane entirely — appending an empty-videos
    # LaneJob would just print "0 cache hits" and waste a model load.
    if not skip_visual and video_sources:
        vargs = ["--device", device_arg]
        if force:
            vargs += ["--force"]
        # ── Wealthy → Florence batch propagation ─────────────────────
        # Speech (Parakeet pool) and CLAP both read VIDEO_USE_WEALTHY
        # straight from the env inside their own run_*_lane_batch
        # entrypoints, so propagate_to_env() above is enough for them.
        # Florence is the odd one out: visual_lane.run_visual_lane_batch
        # takes batch_size as an explicit kwarg with a conservative
        # `DEFAULT_BATCH_SIZE = 8` default, and the wealthy → 32 bump
        # only happens inside visual_lane.main()'s CLI argparse path —
        # which we bypass entirely via the -c shim in _run_lane().
        # Without this branch, --wealthy was silently a no-op for the
        # visual lane in the orchestrated pipeline (the standalone CLI
        # path was fine). Forwarding --batch-size here closes the gap.
        if is_wealthy(wealthy):
            vargs += ["--batch-size", str(FLORENCE_BATCH)]
        # ── Visual fps override ─────────────────────────────────────
        # Forwarded only when the caller asked for a non-default value;
        # leaving it absent lets visual_lane.py's argparse default
        # (DEFAULT_FPS = 1.0) apply. `:g` keeps "1" / "0.5" / "0.25"
        # clean on the wire instead of "1.0" / "0.50".
        if visual_fps is not None:
            vargs += ["--fps", f"{float(visual_fps):g}"]
        jobs.append(LaneJob("visual", "visual_lane.py", video_sources, edit_dir, vargs))
    elif not skip_visual and not video_sources:
        # Audio-only batch — surface this so the user isn't confused
        # by the missing visual_caps/ output and the absence of any
        # visual_lane log lines.
        print(
            "[orchestrator] visual lane skipped — no video sources "
            "in this batch (all inputs are audio-only)"
        )

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
                    help="Source media files (one or many). Video files "
                         "and/or audio-only files (.wav/.mp3/...) — "
                         "audio-only sources auto-skip the visual lane.")
    ap.add_argument("--edit-dir", type=Path, default=None,
                    help="Edit output dir (default: <first video parent>/edit)")
    ap.add_argument("--force-schedule",
                    choices=[s.value for s in Schedule],
                    default=None,
                    help="Override the auto VRAM-based schedule")
    ap.add_argument("--skip-speech", action="store_true",
                    help="Skip the Parakeet ONNX speech lane")
    ap.add_argument("--include-audio", action="store_true",
                    help="Run the CLAP audio lane inline with the baked-in "
                         "baseline vocabulary. Off by default — the "
                         "recommended workflow is to invoke audio_lane.py "
                         "as a separate step with an agent-curated vocab.")
    ap.add_argument("--skip-visual", action="store_true",
                    help="Skip the Florence-2 visual lane")
    ap.add_argument(
        "--visual-fps", type=float, default=None,
        help="Sample rate for the visual lane in frames/sec "
             "(default: 1.0). Fractional values are accepted: 0.5 = "
             "one frame every 2 s, 0.25 = every 4 s. Useful for slow / "
             "static / long-form content where 1 fps over-samples and "
             "bloats audiovisual_timeline.md. Cost scales linearly.",
    )
    ap.add_argument("--wealthy", action="store_true",
                    help="Speed knob for 24GB+ GPUs. Bigger batches in all "
                         "lanes; same models, same outputs. Also reads "
                         "VIDEO_USE_WEALTHY=1.")
    ap.add_argument("--diarize", action="store_true",
                    help="Enable pyannote speaker diarization (needs HF_TOKEN)")
    ap.add_argument("--language", default=None,
                    help="ISO language code for the speech lane "
                         "(en -> Parakeet v2; otherwise Parakeet v3). "
                         "Default: auto / English.")
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
        skip_speech=args.skip_speech,
        include_audio=args.include_audio,
        skip_visual=args.skip_visual,
        wealthy=args.wealthy,
        diarize=args.diarize,
        language=args.language,
        force=args.force,
        visual_fps=args.visual_fps,
    )
    sys.exit(0 if all(j.returncode == 0 for j in jobs) else 1)


if __name__ == "__main__":
    main()
