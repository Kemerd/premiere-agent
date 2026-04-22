"""Batch wrapper around the preprocess orchestrator.

This is the script the SKILL calls. Difference vs preprocess.py:

  - Takes a DIRECTORY of source videos (not an explicit file list)
  - Auto-discovers .mp4/.mov/.mkv/.m4v files in that directory
  - Skips files whose lane outputs are already cache-fresh
  - Defaults edit-dir to <videos>/edit

Everything else delegates to preprocess.run_preprocess() so the schedule
selection, progress display, and lane dispatch stay in one place.

CLI:
    python helpers/preprocess_batch.py <videos_dir> \\
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


# Recognised source extensions. We DON'T include audio-only (.wav, .mp3,
# .m4a) here because the visual lane needs frames; mixed batches would
# fail mid-run. If a future "speech-only" mode lands, add a flag.
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm"}


def _discover_videos(videos_dir: Path) -> list[Path]:
    """Return sorted source videos under videos_dir (non-recursive).

    Non-recursive on purpose — recursion picks up B-roll and proxy
    folders the user may not want preprocessed. Users with a tree can
    pass `python preprocess.py file1 file2 ...` directly.
    """
    out: list[Path] = []
    for p in videos_dir.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            out.append(p.resolve())
    out.sort()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Discover & preprocess all videos in a directory "
                    "(speech / audio / visual lanes).",
    )
    ap.add_argument("videos_dir", type=Path,
                    help="Directory containing source video files")
    ap.add_argument("--edit-dir", type=Path, default=None,
                    help="Edit output dir (default: <videos_dir>/edit)")
    ap.add_argument("--force-schedule",
                    choices=[s.value for s in Schedule],
                    default=None,
                    help="Override the auto VRAM-based schedule")
    ap.add_argument("--skip-whisper", action="store_true")
    ap.add_argument("--skip-audio", action="store_true")
    ap.add_argument("--skip-visual", action="store_true")
    ap.add_argument("--wealthy", action="store_true",
                    help="Speed knob for 24GB+ GPUs. Bigger batches, same "
                         "models, same outputs. Also reads VIDEO_USE_WEALTHY=1.")
    ap.add_argument("--diarize", action="store_true",
                    help="Enable pyannote speaker diarization (needs HF_TOKEN)")
    ap.add_argument("--language", default=None,
                    help="ISO language code passed to Whisper (default: auto)")
    ap.add_argument("--force", action="store_true",
                    help="Bypass per-lane caches, always re-run")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("[preprocess_batch] FATAL: ffmpeg not on PATH. See README.")

    videos_dir = args.videos_dir.resolve()
    if not videos_dir.is_dir():
        sys.exit(f"[preprocess_batch] FATAL: not a directory: {videos_dir}")

    videos = _discover_videos(videos_dir)
    if not videos:
        sys.exit(f"[preprocess_batch] no source videos in {videos_dir} "
                 f"(extensions: {sorted(VIDEO_EXTS)})")

    print(f"[preprocess_batch] discovered {len(videos)} source video(s) "
          f"in {videos_dir}")
    for v in videos:
        size_mb = v.stat().st_size / (1024 * 1024)
        print(f"  - {v.name}  ({size_mb:.0f} MB)")

    edit_dir = (args.edit_dir or (videos_dir / "edit")).resolve()
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
