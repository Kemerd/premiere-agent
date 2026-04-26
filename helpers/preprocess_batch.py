"""Batch wrapper around the preprocess orchestrator.

This is the script the SKILL calls. Difference vs preprocess.py:

  - Takes a DIRECTORY of source media (not an explicit file list)
  - Auto-discovers video AND audio-only files in that directory
  - Skips files whose lane outputs are already cache-fresh
  - Defaults edit-dir to <videos>/edit

Everything else delegates to preprocess.run_preprocess() so the schedule
selection, progress display, and lane dispatch stay in one place.

Mixed batches (video + audio-only)
----------------------------------
Audio-only sources (e.g. a recorded voiceover .wav, a music bed, a
podcast .mp3) are first-class citizens in the pipeline. They get the
speech lane (Parakeet ONNX runs on the same 16kHz mono WAV cache, no
matter whether the source was video or audio) and the optional CLAP
audio lane. They are EXCLUDED from the visual lane — Florence-2 needs
frames, so audio-only sources are filtered out of the visual job by
run_preprocess(). A typical mixed batch is "12 video clips + 1
voiceover.wav for a scripted assembly".

CLI:
    python helpers/preprocess_batch.py <sources_dir> \\
        [--edit-dir <dir>] [--wealthy] [--diarize] [--language en]
        [--force] [--force-schedule {parallel,mixed,sequential,cpu}]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from preprocess import run_preprocess
from vram import Schedule, parse_force_schedule


# ---------------------------------------------------------------------------
# Recognised source extensions
# ---------------------------------------------------------------------------
#
# Two buckets — video containers (frames + audio, all three lanes apply)
# and audio-only containers (just speech / optional CLAP, NO visual).
# The classification is by suffix only; that's enough for ffmpeg's input
# probe to do the right thing downstream. If someone has an audio-only
# .mp4 (rare, but legal) it'll still be sent to the visual lane and the
# Florence prefetch will fail loudly — that's the user's bug, not ours.
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm"}
AUDIO_ONLY_EXTS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
}
MEDIA_EXTS = VIDEO_EXTS | AUDIO_ONLY_EXTS


def _discover_sources(sources_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return (videos, audio_only) sorted lists under sources_dir.

    Non-recursive on purpose — recursion picks up B-roll and proxy
    folders the user may not want preprocessed. Users with a tree can
    pass `python preprocess.py file1 file2 ...` directly.

    Returns a 2-tuple so the caller can print a clean summary and pass
    them as a single combined list to run_preprocess() (which then
    re-classifies internally to scope each lane). We don't return one
    flat list because the summary line is genuinely useful — "12
    videos + 1 voiceover.wav" tells the user immediately whether they
    forgot to drop their VO file in.
    """
    videos: list[Path] = []
    audio: list[Path] = []
    for p in sources_dir.iterdir():
        if not p.is_file():
            continue
        sfx = p.suffix.lower()
        if sfx in VIDEO_EXTS:
            videos.append(p.resolve())
        elif sfx in AUDIO_ONLY_EXTS:
            audio.append(p.resolve())
    videos.sort()
    audio.sort()
    return videos, audio


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Discover & preprocess all media in a directory "
                    "(speech / audio / visual lanes; audio-only sources "
                    "are auto-detected and routed to speech-only lanes).",
    )
    ap.add_argument("videos_dir", type=Path,
                    help="Directory containing source media (video files "
                         "and/or audio-only files like voiceover .wav)")
    ap.add_argument("--edit-dir", type=Path, default=None,
                    help="Edit output dir (default: <videos_dir>/edit)")
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
                         "as a separate Phase B step with an agent-curated "
                         "vocab derived from the speech + visual timelines.")
    ap.add_argument("--skip-visual", action="store_true",
                    help="Skip the Florence-2 visual lane")
    ap.add_argument("--wealthy", action="store_true",
                    help="Speed knob for 24GB+ GPUs. Bigger batches, same "
                         "models, same outputs. Also reads VIDEO_USE_WEALTHY=1.")
    ap.add_argument("--diarize", action="store_true",
                    help="Enable pyannote speaker diarization (needs HF_TOKEN)")
    ap.add_argument("--language", default=None,
                    help="ISO language code for the speech lane "
                         "(en -> Parakeet v2; otherwise Parakeet v3). "
                         "Default: auto / English.")
    ap.add_argument("--force", action="store_true",
                    help="Bypass per-lane caches, always re-run")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("[preprocess_batch] FATAL: ffmpeg not on PATH. See README.")

    videos_dir = args.videos_dir.resolve()
    if not videos_dir.is_dir():
        sys.exit(f"[preprocess_batch] FATAL: not a directory: {videos_dir}")

    videos, audio_only = _discover_sources(videos_dir)
    if not videos and not audio_only:
        sys.exit(
            f"[preprocess_batch] no source media in {videos_dir}\n"
            f"  recognised video extensions: {sorted(VIDEO_EXTS)}\n"
            f"  recognised audio extensions: {sorted(AUDIO_ONLY_EXTS)}"
        )

    # ── Discovery summary ──
    # Print video and audio-only counts separately so the user can spot
    # at a glance whether ffmpeg picked up their voiceover.wav. The
    # combined list is what we pass downstream — run_preprocess()
    # re-classifies it internally to scope the visual lane.
    print(
        f"[preprocess_batch] discovered {len(videos)} video(s) + "
        f"{len(audio_only)} audio-only source(s) in {videos_dir}"
    )
    for v in videos:
        size_mb = v.stat().st_size / (1024 * 1024)
        print(f"  - [video] {v.name}  ({size_mb:.0f} MB)")
    for a in audio_only:
        size_mb = a.stat().st_size / (1024 * 1024)
        print(f"  - [audio] {a.name}  ({size_mb:.0f} MB)  [speech-only, no visual lane]")

    edit_dir = (args.edit_dir or (videos_dir / "edit")).resolve()
    schedule = parse_force_schedule(args.force_schedule)

    # The orchestrator accepts a single flat list and re-classifies by
    # extension to scope the visual lane. Concatenating here keeps the
    # public API of run_preprocess() unchanged (still `videos=`).
    all_sources = videos + audio_only

    jobs = run_preprocess(
        videos=all_sources,
        edit_dir=edit_dir,
        schedule=schedule,
        skip_speech=args.skip_speech,
        include_audio=args.include_audio,
        skip_visual=args.skip_visual,
        wealthy=args.wealthy,
        diarize=args.diarize,
        language=args.language,
        force=args.force,
    )
    sys.exit(0 if all(j.returncode == 0 for j in jobs) else 1)


if __name__ == "__main__":
    main()
