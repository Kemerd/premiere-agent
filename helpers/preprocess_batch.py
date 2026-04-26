"""Batch wrapper around the preprocess orchestrator.

This is the script the SKILL calls. Difference vs preprocess.py:

  - Takes a DIRECTORY of source media (not an explicit file list)
  - Auto-discovers video AND audio-only files in that directory
  - Detects dual-mic / paired-audio pairs (X.mp4 + X.wav same stem)
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

Paired-audio detection (dual-mic recordings)
--------------------------------------------
Some camera + recorder combos produce a video file AND a stand-alone
audio file with the same stem — e.g. a Sony body recording H.264 to
`SHOT_0042.mp4` while a Zoom F2 records a lav mic to `SHOT_0042.wav`.
The convention is universal across rigs (DJI, Zoom, GoPro, Tascam,
camera + recorder rigs in general). When this script sees a stem
collision between a video file and an audio file, it has two valid
interpretations:

  1. **dual_mic** — the .wav is a second-mic recording of the SAME
     shot, usually a lav with cleaner audio than the on-camera mic.
     Both should be transcribed; the editor picks whichever
     transcript is higher quality on each cut.

  2. **ignore** — the .wav is a redundant backup of the camera audio
     (some rigs do this) or a copy the user dragged in by mistake.
     Drop it from preprocessing entirely.

These are mutually exclusive and the wrong choice silently corrupts
the cut, so this script REFUSES TO PROCEED when pairs are detected
without an explicit `--paired-audio-mode={dual_mic,ignore}`. The
parent agent must ask the user, then re-invoke with the chosen mode.
The dry-run sibling flag `--detect-pairs` lets the parent inventory
pairs without committing to a mode (prints JSON to stdout, exits 0).

In `dual_mic` mode the paired .wav is hardlinked to
`<edit>/.paired_audio/<stem>.audio<ext>` so its cache outputs land
under a unique stem and don't collide with the video sibling's cache
(same-volume hardlinks are zero-cost on NTFS / ext4 / APFS; we fall
back to a copy on cross-volume rigs). The aliasing is invisible to the
lane scripts — they just see two unrelated files. The mapping is
recorded in `<edit>/source_pairs.json` so the editor sub-agent can
join transcripts back to their video source at cut time.

CLI:
    python helpers/preprocess_batch.py <sources_dir> \\
        [--edit-dir <dir>] [--wealthy] [--diarize] [--language en]
        [--force] [--force-schedule {parallel,mixed,sequential,cpu}] \\
        [--paired-audio-mode {dual_mic,ignore}] [--detect-pairs]
"""

from __future__ import annotations

import argparse
import json
import os
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


# ---------------------------------------------------------------------------
# Paired-audio detection + aliasing
# ---------------------------------------------------------------------------
#
# A "pair" is one video file and one audio-only file that share the same
# stem under case-insensitive comparison (NTFS is case-insensitive by
# default; some users on macOS/Linux still expect the same — and the
# Sony / Zoom / DJI rigs that produce these pairs always emit matching
# stems character-for-character anyway, so case-fold is just a safety
# net, not a heuristic). One video + many .wavs of the same stem is
# theoretically possible (e.g. .wav and .flac dumped from two
# recorders) but I've never seen it in the wild — if it happens we
# report it as N pairs sharing the video, and the parent can ask.

PAIRED_AUDIO_DIR = ".paired_audio"
PAIRED_STEM_SUFFIX = ".audio"


def _detect_pairs(
    videos: list[Path], audio: list[Path],
) -> list[tuple[Path, Path]]:
    """Return [(video_path, audio_path)] for every stem collision.

    Stem comparison is case-insensitive — the camera/recorder rigs
    that actually produce these pairs emit identical stems anyway,
    but case-folding makes the detection robust to a user dragging a
    `Shot_001.MP4` next to `shot_001.wav` from two different file
    managers. Sorted by video stem for deterministic output.
    """
    by_stem_lower: dict[str, Path] = {v.stem.lower(): v for v in videos}
    out: list[tuple[Path, Path]] = []
    for a in audio:
        v = by_stem_lower.get(a.stem.lower())
        if v is not None:
            out.append((v, a))
    out.sort(key=lambda pa: pa[0].stem.lower())
    return out


def _alias_paired_audio(audio_path: Path, edit_dir: Path) -> Path:
    """Hardlink (or copy fallback) the audio file to a disambiguated
    path under <edit_dir>/.paired_audio/<stem>.audio<ext> so its cache
    outputs land under a unique stem.

    Why hardlink: the speech / audio lanes derive every output filename
    from `source_path.stem`. If we passed the original `myShot.wav`
    in alongside `myShot.mp4`, every cache file would collide:
        <edit>/audio_16k/myShot.wav         (overwritten)
        <edit>/transcripts/myShot.json      (overwritten)
        <edit>/audio_tags/myShot.json       (overwritten — if CLAP runs)
    The fix is to give the .wav a unique stem on disk. Renaming the
    user's source file is not on the table, so we hardlink it. On the
    same volume this is a single inode operation — no data copy, no
    extra disk usage, mtime preserved (which matters for the cache-
    fresh checks in extract_audio.py and friends). On cross-volume rigs
    where Path.link errors out we fall back to shutil.copy2 which
    preserves mtime, at the cost of one extra copy of the audio.

    Why under <edit_dir> rather than next to the source: the edit dir
    is the canonical "scratch space" the skill writes to (per Hard
    Rule 12 — never modify the source dir). We isolate the alias under
    a hidden `.paired_audio/` subdir so the discovery glob in
    _discover_sources() never re-picks them up if the user re-runs
    with the edit dir inside videos_dir (which is the default).
    """
    alias_dir = edit_dir / PAIRED_AUDIO_DIR
    alias_dir.mkdir(parents=True, exist_ok=True)
    alias_path = alias_dir / f"{audio_path.stem}{PAIRED_STEM_SUFFIX}{audio_path.suffix}"

    # Re-use an existing alias when it points at the same inode as the
    # source. Comparing st_ino on Windows works for hardlinks within a
    # volume; on cross-volume rigs the alias is a copy, so we instead
    # check mtime + size as a fast equality probe.
    if alias_path.exists():
        try:
            src_stat = audio_path.stat()
            dst_stat = alias_path.stat()
            same_inode = (
                getattr(src_stat, "st_ino", 0) != 0
                and src_stat.st_ino == dst_stat.st_ino
                and src_stat.st_dev == dst_stat.st_dev
            )
            same_content_probe = (
                src_stat.st_size == dst_stat.st_size
                and abs(src_stat.st_mtime - dst_stat.st_mtime) < 1.0
            )
            if same_inode or same_content_probe:
                return alias_path
        except OSError:
            pass
        # Stale alias — wipe and recreate so the cache check below
        # picks up the fresh source mtime correctly.
        try:
            alias_path.unlink()
        except OSError:
            pass

    try:
        os.link(audio_path, alias_path)
        link_kind = "hardlink"
    except OSError:
        # Cross-volume, FS without hardlink support, permissions issue —
        # any of these → fall back to copy. shutil.copy2 preserves
        # mtime which our cache-fresh checks rely on.
        shutil.copy2(audio_path, alias_path)
        link_kind = "copy"

    print(
        f"  [paired-audio] aliased {audio_path.name} -> "
        f"{alias_path.relative_to(edit_dir)} ({link_kind})"
    )
    return alias_path


def _write_source_pairs_json(
    edit_dir: Path,
    mode: str,
    pairs: list[tuple[Path, Path]],
    aliases: dict[Path, Path],
) -> Path:
    """Persist the paired-audio decisions for the editor sub-agent.

    Schema (one place, one definition):
        {
          "mode": "dual_mic" | "ignore",
          "pairs": [
            {
              "stem": "<video stem>",
              "video": "<abs path>",
              "audio": "<abs path of original .wav>",
              # dual_mic only:
              "audio_alias_stem": "<stem>.audio",
              "audio_alias_path": "<abs path under .paired_audio/>"
            },
            ...
          ]
        }

    The editor reads this to know that transcripts/<stem>.json and
    transcripts/<stem>.audio.json belong to the SAME shot and it
    should pick the higher-confidence one (or use both for
    cross-checks).
    """
    out_path = edit_dir / "source_pairs.json"
    items: list[dict] = []
    for video, audio in pairs:
        entry: dict = {
            "stem": video.stem,
            "video": str(video),
            "audio": str(audio),
        }
        if mode == "dual_mic":
            alias = aliases.get(audio)
            if alias is not None:
                entry["audio_alias_stem"] = alias.stem
                entry["audio_alias_path"] = str(alias)
        items.append(entry)

    payload = {"mode": mode, "pairs": items}
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[preprocess_batch] wrote {out_path.relative_to(edit_dir.parent)} "
          f"({mode}, {len(items)} pair(s))")
    return out_path


def _print_pairs_dry_run(
    pairs: list[tuple[Path, Path]],
    videos_dir: Path,
) -> None:
    """Print the detected pairs as JSON to stdout for the parent agent.

    The parent runs this BEFORE preprocessing to inventory pairs and
    ask the user whether they're dual-mic or backup files. Output goes
    to stdout (not stderr) so the parent can `--detect-pairs` and pipe
    straight into json parsing if needed; informational messages stay
    on stderr.
    """
    payload = {
        "videos_dir": str(videos_dir),
        "pair_count": len(pairs),
        "pairs": [
            {
                "stem": v.stem,
                "video": str(v),
                "audio": str(a),
            }
            for v, a in pairs
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


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
    # ── Paired-audio handling ────────────────────────────────────────
    # See module docstring for the full design. Two flags:
    #   --detect-pairs       inventory pairs and exit (dry run, JSON)
    #   --paired-audio-mode  decide what to do with detected pairs
    # When pairs are detected and the latter is missing, the script
    # refuses to proceed (rc=2) so the user is never silently bitten
    # by a wrong default.
    ap.add_argument(
        "--detect-pairs", action="store_true",
        help="Dry run: list video+audio stem-collision pairs as JSON "
             "to stdout and exit. Use this to inventory pairs before "
             "asking the user how to handle them.",
    )
    ap.add_argument(
        "--paired-audio-mode",
        choices=["dual_mic", "ignore"],
        default=None,
        help="What to do when a video+audio pair shares a stem. "
             "dual_mic = transcribe both (the .wav is hardlinked to "
             "<edit>/.paired_audio/ under a disambiguated stem so its "
             "cache outputs don't collide with the video sibling); "
             "ignore = drop the .wav from preprocessing entirely "
             "(use this when the .wav is just a backup of the camera "
             "audio). REQUIRED if any pairs are detected.",
    )
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

    # ── Pair detection ──
    # Compute pairs before the discovery summary so we can flag them
    # in the per-file print and so the dry-run flag can short-circuit.
    pairs = _detect_pairs(videos, audio_only)
    paired_audio_paths: set[Path] = {a for _, a in pairs}

    # --detect-pairs: print JSON to stdout for the parent and exit.
    # We deliberately DON'T print the discovery summary in this mode —
    # the parent wants clean machine-parseable JSON.
    if args.detect_pairs:
        _print_pairs_dry_run(pairs, videos_dir)
        sys.exit(0)

    # ── Discovery summary ──
    # Print video and audio-only counts separately so the user can spot
    # at a glance whether ffmpeg picked up their voiceover.wav. The
    # combined list is what we pass downstream — run_preprocess()
    # re-classifies it internally to scope the visual lane.
    print(
        f"[preprocess_batch] discovered {len(videos)} video(s) + "
        f"{len(audio_only)} audio-only source(s) "
        f"({len(pairs)} stem-pair(s)) in {videos_dir}"
    )
    for v in videos:
        size_mb = v.stat().st_size / (1024 * 1024)
        tag = " [+paired audio]" if v.stem.lower() in {p[0].stem.lower() for p in pairs} else ""
        print(f"  - [video] {v.name}  ({size_mb:.0f} MB){tag}")
    for a in audio_only:
        size_mb = a.stat().st_size / (1024 * 1024)
        if a in paired_audio_paths:
            tag = "  [PAIRED with video sibling]"
        else:
            tag = "  [speech-only, no visual lane]"
        print(f"  - [audio] {a.name}  ({size_mb:.0f} MB){tag}")

    # ── Pair-mode gate ──
    # If we found pairs, the user MUST have decided what they are.
    # We refuse to silently default — guessing wrong corrupts the cut
    # (transcript collisions in dual_mic, or wasted compute + a
    # confusing extra "voiceover" stem in ignore).
    if pairs and args.paired_audio_mode is None:
        msg_lines = [
            "[preprocess_batch] FATAL: paired audio detected, "
            "--paired-audio-mode is required.",
            "",
            f"  Found {len(pairs)} stem-pair(s):",
        ]
        for v, a in pairs:
            msg_lines.append(f"    - {v.name}  +  {a.name}  (stem: {v.stem})")
        msg_lines += [
            "",
            "  These look like a camera + external recorder rig where",
            "  the .wav is a second-mic recording of the same shot.",
            "  Re-run with one of:",
            "",
            "    --paired-audio-mode dual_mic",
            "        The .wav is a SECOND MIC recording. Both will be",
            "        transcribed; the editor picks the higher-quality",
            "        transcript per cut. The .wav is hardlinked under a",
            "        disambiguated stem so the caches don't collide.",
            "",
            "    --paired-audio-mode ignore",
            "        The .wav is just a BACKUP of the camera audio (or",
            "        a stray copy). Drop it from preprocessing.",
            "",
            "  Tip: parent agents should run with --detect-pairs first,",
            "  then ASK THE USER, then re-invoke with the chosen mode.",
        ]
        print("\n".join(msg_lines), file=sys.stderr)
        sys.exit(2)

    edit_dir = (args.edit_dir or (videos_dir / "edit")).resolve()
    edit_dir.mkdir(parents=True, exist_ok=True)
    schedule = parse_force_schedule(args.force_schedule)

    # ── Apply pair mode ──
    # `effective_audio_only` is the audio-only source list AFTER the
    # pair-mode resolution. Unpaired audio-only sources (e.g. a
    # standalone voiceover.wav with no video sibling) pass through
    # unchanged in either mode.
    aliases: dict[Path, Path] = {}
    if not pairs:
        effective_audio_only = list(audio_only)
    elif args.paired_audio_mode == "ignore":
        # Drop only the .wavs that are paired. Unpaired audio-only
        # files (true voiceovers without a video sibling) stay in.
        effective_audio_only = [a for a in audio_only if a not in paired_audio_paths]
        n_dropped = len(audio_only) - len(effective_audio_only)
        print(f"[preprocess_batch] paired-audio mode: ignore — "
              f"dropping {n_dropped} paired .wav(s) from preprocess input")
    else:  # dual_mic
        effective_audio_only = []
        for a in audio_only:
            if a in paired_audio_paths:
                alias = _alias_paired_audio(a, edit_dir)
                aliases[a] = alias
                effective_audio_only.append(alias)
            else:
                effective_audio_only.append(a)
        print(f"[preprocess_batch] paired-audio mode: dual_mic — "
              f"aliased {len(aliases)} paired .wav(s) under "
              f"{edit_dir / PAIRED_AUDIO_DIR}")

    # Persist the decision so the editor sub-agent can join the
    # transcripts back to their video sibling at cut time.
    if pairs:
        _write_source_pairs_json(edit_dir, args.paired_audio_mode, pairs, aliases)

    # The orchestrator accepts a single flat list and re-classifies by
    # extension to scope the visual lane. Concatenating here keeps the
    # public API of run_preprocess() unchanged (still `videos=`).
    all_sources = videos + effective_audio_only

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
