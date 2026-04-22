"""Render a video from an EDL.

Implements the HEURISTICS render pipeline in the correct order:

  1. Per-segment extract with color grade + 30ms audio fades baked in
  2. Lossless -c copy concat into base.mp4
  3. If overlays or subtitles: single filter graph that overlays animations
     (with PTS shift so frame 0 lands at the overlay window start)
     and applies `subtitles` filter LAST → final.mp4

Optionally builds a master SRT from the per-source transcripts + EDL
output-timeline offsets, applies the proven force_style (2-word
UPPERCASE chunks, Helvetica 18 Bold, MarginV=35).

Usage:
    python helpers/render.py <edl.json> -o final.mp4
    python helpers/render.py <edl.json> -o preview.mp4 --preview
    python helpers/render.py <edl.json> -o final.mp4 --build-subtitles
    python helpers/render.py <edl.json> -o final.mp4 --no-subtitles
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Force UTF-8 on stdout/stderr so Unicode arrows / em-dashes / ellipses in our
# log output don't crash on Windows consoles still defaulting to cp1252.
# Cheap, idempotent, only runs once at import time. Without this, a stray
# `→` in a `print(...)` aborts the entire render *after* the work is done.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

try:
    from grade import get_preset, auto_grade_for_clip  # same directory
except Exception:
    def get_preset(name: str) -> str:
        return ""

    def auto_grade_for_clip(video, start=0.0, duration=None, verbose=False):  # type: ignore
        return "eq=contrast=1.03:saturation=0.98", {}


# -------- Subtitle style (proven at 1920×1080, from HEURISTICS §5) -----------

SUB_FORCE_STYLE = (
    "FontName=Helvetica,FontSize=18,Bold=1,"
    "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H00000000,"
    "BorderStyle=1,Outline=2,Shadow=0,"
    "Alignment=2,MarginV=35"
)

# -------- Helpers ------------------------------------------------------------


def run(cmd: list[str], quiet: bool = False) -> None:
    if not quiet:
        print(f"  $ {' '.join(str(c) for c in cmd[:6])}{' …' if len(cmd) > 6 else ''}")
    subprocess.run(cmd, check=True)


def resolve_grade_filter(grade_field: str | None) -> str:
    """The EDL's 'grade' field can be a preset name, a raw ffmpeg filter, or 'auto'.

    Returns the filter string to embed into the per-segment -vf chain.
    For 'auto', returns the sentinel "__AUTO__" which is resolved per-segment.
    """
    if not grade_field:
        return ""
    if grade_field == "auto":
        return "__AUTO__"
    # Preset names are short identifiers, filter strings contain '=' or ','.
    if re.fullmatch(r"[a-zA-Z0-9_\-]+", grade_field):
        try:
            return get_preset(grade_field)
        except KeyError:
            print(f"warning: unknown preset '{grade_field}', using as raw filter")
            return grade_field
    return grade_field


def resolve_path(maybe_path: str, base: Path) -> Path:
    """Resolve a path that may be absolute or relative to `base`."""
    p = Path(maybe_path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


# ---------------------------------------------------------------------------
# J/L cut + transition support — schema-only in this renderer.
#
# An EDL range may carry these optional fields:
#   - "audio_lead":    seconds the audio of THIS range starts BEFORE its video
#                      → produces a J-cut from the previous range
#   - "video_tail":    seconds the audio of THIS range continues AFTER its video
#                      → produces an L-cut into the next range
#   - "transition_in": seconds of cross-dissolve at the head of this range
#
# These fields are FAITHFULLY honored by helpers/export_fcpxml.py because
# Premiere/Resolve/FCP have native multi-track support for split edits.
# This ffmpeg renderer is single-track — implementing J/L cuts here would
# require a multi-input filter graph that defeats the lossless per-segment
# concat (Hard Rule 2). So we warn and treat them as hard cuts here, and
# strongly recommend FCPXML export for any timeline that uses them.
# ---------------------------------------------------------------------------

JL_CUT_FIELDS = ("audio_lead", "video_tail", "transition_in")


def _has_jl_cuts(edl: dict) -> tuple[bool, int]:
    """Return (has_any_jl_cut, count_of_ranges_with_jl_fields)."""
    n = 0
    for r in edl.get("ranges", []):
        if any(float(r.get(f, 0) or 0) > 0 for f in JL_CUT_FIELDS):
            n += 1
    return n > 0, n


# ---------------------------------------------------------------------------
# Time-squeezing (timelapse) support — flat MP4 path.
#
# Mirrors helpers/export_fcpxml.py:
#   - Per-range `speed` (float, default 1.0), clamped to [1.0, MAX_SPEED].
#   - Per-range `audio_strategy` ("drop" default at speed != 1.0, else "keep").
# At speed > 1.0 we apply `setpts=PTS/{speed}` to the video and either:
#   - `atempo={speed}` to the audio    (audio_strategy="keep", pitch-preserving
#     chipmunk effect — ffmpeg's atempo handles 0.5..100 in one stage so a 10x
#     cap fits without chaining)
#   - replace audio with `anullsrc`    (audio_strategy="drop", silent gap; the
#     editor can drop a music bed under the timelapse in post)
# Subtitles for retimed ranges are SKIPPED — by editor convention timelapses
# never contain speech (otherwise it'd be unintelligible at 4x+), so there's
# nothing to caption. The seg_offset accumulator uses OUTPUT duration so any
# captions in adjacent normal-speed ranges land on the right output time.
# ---------------------------------------------------------------------------

# Same ceiling as the FCPXML exporter — see that module's MAX_SPEED block
# for why 1000% is the practical maximum for clean retime without frame
# blending. Beyond it the LLM should be CUTTING, not squeezing harder.
MAX_SPEED = 10.0
MIN_SPEED = 1.0


def _read_speed(r: dict, range_idx: int) -> float:
    """Pull the per-range speed; clamp to [MIN_SPEED, MAX_SPEED] with warn."""
    raw = r.get("speed")
    if raw is None:
        return 1.0
    try:
        spd = float(raw)
    except (TypeError, ValueError):
        print(f"  warn: range[{range_idx}] non-numeric speed={raw!r}; "
              "treating as 1.0.", file=sys.stderr)
        return 1.0
    if spd < MIN_SPEED:
        print(f"  warn: range[{range_idx}] speed={spd:g} < {MIN_SPEED:g} "
              "(slow-mo not supported); clamped to 1.0.", file=sys.stderr)
        return 1.0
    if spd > MAX_SPEED:
        print(f"  warn: range[{range_idx}] speed={spd:g} > {MAX_SPEED:g} "
              f"({int(MAX_SPEED * 100)}% retime ceiling); clamped to "
              f"{MAX_SPEED:g}. Cut the range instead of squeezing harder.",
              file=sys.stderr)
        return MAX_SPEED
    return spd


def _read_audio_strategy(r: dict, speed: float, range_idx: int) -> str:
    """Pick "drop" / "keep" with sensible defaults — see exporter."""
    raw = r.get("audio_strategy")
    if raw is None:
        return "keep" if abs(speed - 1.0) < 1e-9 else "drop"
    s = str(raw).strip().lower()
    if s in ("drop", "keep"):
        return s
    print(f"  warn: range[{range_idx}] unknown audio_strategy={raw!r}; "
          "valid: 'drop' | 'keep'.", file=sys.stderr)
    return "keep" if abs(speed - 1.0) < 1e-9 else "drop"


def warn_if_jl_cuts(edl: dict) -> None:
    """Print a one-time, loud-but-not-fatal warning when J/L cuts are
    present in the EDL but we're rendering via the ffmpeg path.

    Idempotent — safe to call multiple times in a single run; only the
    first call prints because the EDL doesn't change.
    """
    has, n = _has_jl_cuts(edl)
    if not has:
        return
    print(
        f"\n  WARN: {n} range(s) carry non-zero J/L-cut or dissolve fields "
        f"({', '.join(JL_CUT_FIELDS)}).\n"
        "       Split edits and cross-dissolves are currently DEFERRED\n"
        "       (see SKILL.md 'Split edits (DEFERRED)' / Hard Rule 14):\n"
        "       the multi-track audio-drift bug in the FCPXML pipeline\n"
        "       has not yet been fixed. This renderer flattens them to\n"
        "       hard cuts; the FCPXML exporter would also drift the audio\n"
        "       on a long timeline. Re-emit the EDL with audio_lead /\n"
        "       video_tail / transition_in all set to 0.0.\n",
        file=sys.stderr,
    )


# -------- Per-segment extraction (Rule 2 + Rule 3) --------------------------


def extract_segment(
    source: Path,
    seg_start: float,
    duration: float,
    grade_filter: str,
    out_path: Path,
    preview: bool = False,
    draft: bool = False,
    speed: float = 1.0,
    audio_strategy: str = "keep",
) -> None:
    """Extract a cut range as its own MP4 with grade + 30ms audio fades baked in.

    `-ss` before `-i` for fast accurate seeking. Scale to 1080p from 4K.

    Quality ladder:
      - final (default): 1080p libx264 fast CRF 20
      - preview:         1080p libx264 medium CRF 22 (evaluable for QC)
      - draft:           720p libx264 ultrafast CRF 28 (cut-point check only)

    Time-squeezing (speed > 1.0):
      Video: `setpts=PTS/{speed}` compresses N source seconds into N/speed
             output seconds with no frame-blending (fine up to 10x on 24/30fps
             source — beyond that the editor should cut, not squeeze).
      Audio: `audio_strategy="keep"` runs `atempo={speed}` (one-stage filter
             that handles 0.5..100; pitch-preserving — avoids chipmunk).
             `audio_strategy="drop"` swaps the audio for `anullsrc` of the
             output duration (silent gap; editor adds a music bed in post).
      The 30ms boundary fades (Hard Rule 3) still apply, computed on the
      OUTPUT duration so we never read past the rendered end.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if draft:
        scale = "scale=1280:-2"
    else:
        scale = "scale=1920:-2"

    vf_parts = [scale]
    if grade_filter:
        vf_parts.append(grade_filter)
    # When retiming, append the setpts BEFORE encode. Order matters here:
    # scale + grade run on every input frame, THEN setpts re-stamps each
    # frame's PTS so the encoder writes them at the new cadence. Doing
    # setpts first would still work but wastes filter cycles on frames
    # we'll then drop — scale-then-stamp is the cheaper graph.
    if speed != 1.0:
        vf_parts.append(f"setpts=PTS/{speed:g}")
    vf = ",".join(vf_parts)

    # Output duration after retime — used for fades AND for anullsrc.
    out_dur = duration / speed if speed != 1.0 else duration
    fade_out_start = max(0.0, out_dur - 0.03)

    if draft:
        preset, crf = "ultrafast", "28"
    elif preview:
        preset, crf = "medium", "22"
    else:
        preset, crf = "fast", "20"

    # ── Audio handling ────────────────────────────────────────────────
    # Three branches:
    #   1. speed == 1.0          → standard -af path (existing behaviour).
    #   2. speed != 1.0, "keep"  → atempo=N pitch-preserving stretch +
    #                              boundary fades on the output cadence.
    #   3. speed != 1.0, "drop"  → swap audio source for anullsrc lavfi
    #                              of out_dur seconds; no fades needed
    #                              (it's already digital silence).
    if speed == 1.0:
        af = (f"afade=t=in:st=0:d=0.03,"
              f"afade=t=out:st={fade_out_start:.3f}:d=0.03")
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{seg_start:.3f}",
            "-i", str(source),
            "-t", f"{duration:.3f}",
            "-vf", vf,
            "-af", af,
            "-c:v", "libx264", "-preset", preset, "-crf", crf,
            "-pix_fmt", "yuv420p", "-r", "24",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            str(out_path),
        ]
    elif audio_strategy == "drop":
        # Build a silent stereo 48k stream of the OUTPUT duration. We
        # use `-shortest` so any tiny mismatch between the retimed
        # video duration and the lavfi audio duration ends on the
        # video edge (which is the authoritative timeline).
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{seg_start:.3f}",
            "-i", str(source),
            "-t", f"{duration:.3f}",
            "-f", "lavfi",
            "-t", f"{out_dur:.3f}",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-vf", vf,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", preset, "-crf", crf,
            "-pix_fmt", "yuv420p", "-r", "24",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-shortest",
            "-movflags", "+faststart",
            str(out_path),
        ]
    else:  # speed != 1.0, audio_strategy == "keep"
        # atempo handles 0.5..100 in one stage; our speed is bounded to
        # MAX_SPEED (10.0) so a single-stage filter is always enough.
        # Fades stay at 30ms on the OUTPUT timeline.
        af = (f"atempo={speed:g},"
              f"afade=t=in:st=0:d=0.03,"
              f"afade=t=out:st={fade_out_start:.3f}:d=0.03")
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{seg_start:.3f}",
            "-i", str(source),
            "-t", f"{duration:.3f}",
            "-vf", vf,
            "-af", af,
            "-c:v", "libx264", "-preset", preset, "-crf", crf,
            "-pix_fmt", "yuv420p", "-r", "24",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            str(out_path),
        ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def extract_all_segments(
    edl: dict,
    edit_dir: Path,
    preview: bool,
    draft: bool = False,
) -> list[Path]:
    """Extract every EDL range into edit_dir/clips_graded/seg_NN.mp4.
    Returns the ordered list of segment paths.

    If the EDL `grade` is "auto", analyze each segment range with
    `auto_grade_for_clip` and apply a per-segment subtle correction.
    Otherwise, apply the same preset/raw filter to every segment.
    """
    resolved = resolve_grade_filter(edl.get("grade"))
    is_auto = resolved == "__AUTO__"
    clips_dir = edit_dir / (
        "clips_draft" if draft else ("clips_preview" if preview else "clips_graded")
    )
    clips_dir.mkdir(parents=True, exist_ok=True)

    ranges = edl["ranges"]
    sources = edl["sources"]

    seg_paths: list[Path] = []
    print(f"extracting {len(ranges)} segment(s) → {clips_dir.name}/")
    if is_auto:
        print("  (auto-grade per segment: analyzing each range)")
    for i, r in enumerate(ranges):
        src_name = r["source"]
        src_path = resolve_path(sources[src_name], edit_dir)
        start = float(r["start"])
        end = float(r["end"])
        duration = end - start
        # Per-range retime — defaults to 1.0 (no-op). Audio strategy
        # defaults flip to "drop" automatically when speed > 1.0; see
        # _read_audio_strategy() for the rule.
        speed = _read_speed(r, i)
        audio_strategy = _read_audio_strategy(r, speed, i)
        out_dur = duration / speed if speed != 1.0 else duration
        out_path = clips_dir / f"seg_{i:02d}_{src_name}.mp4"

        if is_auto:
            seg_filter, _stats = auto_grade_for_clip(src_path, start=start, duration=duration, verbose=False)
        else:
            seg_filter = resolved

        note = r.get("beat") or r.get("note") or ""
        # Append a retime tag to the per-segment log line so the user
        # can immediately see which ranges are squeezed and by how much.
        retime_tag = ""
        if speed != 1.0:
            retime_tag = (f"  [⚡{speed:g}x → {out_dur:.2f}s, "
                          f"audio={audio_strategy}]")
        print(f"  [{i:02d}] {src_name}  {start:7.2f}-{end:7.2f}  "
              f"({duration:5.2f}s){retime_tag}  {note}")
        if is_auto:
            print(f"        grade: {seg_filter or '(none)'}")
        extract_segment(
            src_path, start, duration, seg_filter, out_path,
            preview=preview, draft=draft,
            speed=speed, audio_strategy=audio_strategy,
        )
        seg_paths.append(out_path)

    return seg_paths


# -------- Lossless concat ----------------------------------------------------


def concat_segments(segment_paths: list[Path], out_path: Path, edit_dir: Path) -> None:
    """Lossless concat via the concat demuxer. No re-encode."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat_list = edit_dir / "_concat.txt"
    concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in segment_paths))

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    print(f"concat → {out_path.name}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    concat_list.unlink(missing_ok=True)


# -------- Master SRT (Rule 5) ------------------------------------------------


PUNCT_BREAK = set(".,!?;:")


def _srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _words_in_range(transcript: dict, t_start: float, t_end: float) -> list[dict]:
    out: list[dict] = []
    for w in transcript.get("words", []):
        if w.get("type") != "word":
            continue
        ws = w.get("start")
        we = w.get("end")
        if ws is None or we is None:
            continue
        if we <= t_start or ws >= t_end:
            continue
        out.append(w)
    return out


def build_master_srt(edl: dict, edit_dir: Path, out_path: Path) -> None:
    """Build an output-timeline SRT from per-source transcripts.

    - 2-word chunks (break on any punctuation in between)
    - UPPERCASE text
    - Output times computed as word.start - segment_start + segment_offset
    """
    transcripts_dir = edit_dir / "transcripts"
    sources = edl["sources"]

    entries: list[tuple[float, float, str]] = []
    seg_offset = 0.0

    for i, r in enumerate(edl["ranges"]):
        src_name = r["source"]
        seg_start = float(r["start"])
        seg_end = float(r["end"])
        seg_duration = seg_end - seg_start
        # Retime-aware OUTPUT duration. SRT timestamps live on the
        # output timeline, so the accumulator advances by what the
        # viewer actually sees, not the source range.
        seg_speed = _read_speed(r, i)
        seg_out_dur = seg_duration / seg_speed if seg_speed != 1.0 else seg_duration

        # Skip caption synthesis on retimed ranges — by editor convention
        # timelapses contain no speech (Hard Rule below; see SKILL.md
        # "Time-squeezing"). Even if the source had stray dialogue,
        # speech sped up 4x+ is unintelligible and shouldn't be
        # captioned. We still advance seg_offset by the OUTPUT duration
        # so subsequent ranges' captions land on the right output time.
        if seg_speed != 1.0:
            seg_offset += seg_out_dur
            continue

        # Transcripts are cached by the *source file stem* (see parakeet_onnx_lane.py:
        # `transcripts_dir / f"{video_path.stem}.json"`), NOT by the EDL short label.
        # The EDL `src_name` is just a human-friendly key (e.g. "C0303") that maps,
        # via `sources[src_name]`, to the real filename / path on disk
        # (e.g. "DJI_20250217172514_0303_D.mp4"). We resolve through that mapping so
        # the SRT lookup matches whatever Parakeet actually wrote.
        src_ref = sources.get(src_name)
        if not src_ref:
            print(f"  no source mapping for {src_name}, skipping captions for this segment")
            seg_offset += seg_out_dur
            continue
        src_stem = Path(src_ref).stem

        tr_path = transcripts_dir / f"{src_stem}.json"
        if not tr_path.exists():
            # Fallback: also try the bare label, in case an older preprocess run
            # keyed transcripts by the short EDL name instead of the file stem.
            legacy = transcripts_dir / f"{src_name}.json"
            if legacy.exists():
                tr_path = legacy
            else:
                print(f"  no transcript for {src_name} (looked for {src_stem}.json), skipping captions for this segment")
                seg_offset += seg_out_dur
                continue

        # Parakeet writes transcripts as UTF-8; be explicit so Windows hosts
        # don't fall back to cp1252 and choke on smart-quotes / em-dashes.
        transcript = json.loads(tr_path.read_text(encoding="utf-8"))
        words_in_seg = _words_in_range(transcript, seg_start, seg_end)

        # Group into 2-word chunks, break on punctuation
        chunks: list[list[dict]] = []
        current: list[dict] = []
        for w in words_in_seg:
            text = (w.get("text") or "").strip()
            if not text:
                continue
            current.append(w)
            # Break if the current text ends in punctuation or we hit 2 words
            ends_in_punct = bool(text) and text[-1] in PUNCT_BREAK
            if len(current) >= 2 or ends_in_punct:
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)

        for chunk in chunks:
            local_start = max(seg_start, chunk[0].get("start", seg_start))
            local_end = min(seg_end, chunk[-1].get("end", seg_end))
            out_start = max(0.0, local_start - seg_start) + seg_offset
            out_end = max(0.0, local_end - seg_start) + seg_offset
            if out_end <= out_start:
                out_end = out_start + 0.4
            text = " ".join((w.get("text") or "").strip() for w in chunk)
            text = re.sub(r"\s+", " ", text).strip()
            # Strip trailing punctuation for cleaner uppercase look
            text = text.rstrip(",;:")
            text = text.upper()
            entries.append((out_start, out_end, text))

        seg_offset += seg_out_dur

    # Sort and write as SRT
    entries.sort(key=lambda e: e[0])
    lines: list[str] = []
    for i, (a, b, t) in enumerate(entries, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(a)} --> {_srt_timestamp(b)}")
        lines.append(t)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"master SRT → {out_path.name} ({len(entries)} cues)")


# -------- Loudness normalization (social-ready audio) -----------------------


# Social-media standard: -14 LUFS integrated, -1 dBTP peak, LRA 11 LU.
# Matches YouTube / Instagram / TikTok / X / LinkedIn normalization targets.
LOUDNORM_I = -14.0
LOUDNORM_TP = -1.0
LOUDNORM_LRA = 11.0


def measure_loudness(video_path: Path) -> dict[str, str] | None:
    """Run ffmpeg loudnorm first pass and parse the JSON measurement.

    Returns a dict with measured_i, measured_tp, measured_lra, measured_thresh,
    target_offset, or None if measurement failed.
    """
    filter_str = (
        f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}:print_format=json"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-i", str(video_path),
        "-af", filter_str,
        "-vn", "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # loudnorm prints the JSON to stderr at the end of the run
    stderr = proc.stderr

    # Find the JSON block — loudnorm output contains a `{ ... }` block
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(stderr[start : end + 1])
    except json.JSONDecodeError:
        return None
    needed = {"input_i", "input_tp", "input_lra", "input_thresh", "target_offset"}
    if not needed.issubset(data.keys()):
        return None
    return data


def apply_loudnorm_two_pass(
    input_path: Path,
    output_path: Path,
    preview: bool = False,
) -> bool:
    """Run two-pass loudnorm on input_path, write normalized copy to output_path.

    Returns True on success, False if measurement failed (caller should fall
    back to copying the input unchanged).

    In preview mode, skips the measurement pass and uses a one-pass approximation
    for speed. Final mode always does the proper two-pass.
    """
    if preview:
        # One-pass approximation — faster, slightly less accurate.
        filter_str = f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-i", str(input_path),
            "-c:v", "copy",
            "-af", filter_str,
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            str(output_path),
        ]
        print(f"  loudnorm (1-pass preview) → {output_path.name}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True

    # Full two-pass
    print(f"  loudnorm pass 1: measuring {input_path.name}")
    measurement = measure_loudness(input_path)
    if measurement is None:
        print("  loudnorm measurement failed — falling back to 1-pass")
        return apply_loudnorm_two_pass(input_path, output_path, preview=True)

    print(f"    measured: I={measurement['input_i']} LUFS  "
          f"TP={measurement['input_tp']}  LRA={measurement['input_lra']}")

    filter_str = (
        f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}"
        f":measured_I={measurement['input_i']}"
        f":measured_TP={measurement['input_tp']}"
        f":measured_LRA={measurement['input_lra']}"
        f":measured_thresh={measurement['input_thresh']}"
        f":offset={measurement['target_offset']}"
        f":linear=true"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-i", str(input_path),
        "-c:v", "copy",
        "-af", filter_str,
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        str(output_path),
    ]
    print(f"  loudnorm pass 2: normalizing → {output_path.name}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return True


# -------- Final compositing (Rule 1 + Rule 4) -------------------------------


def build_final_composite(
    base_path: Path,
    overlays: list[dict],
    subtitles_path: Path | None,
    out_path: Path,
    edit_dir: Path,
) -> None:
    """Final pass: base → overlays (PTS-shifted) → subtitles LAST → out.

    If there are no overlays and no subtitles, just copy base to out.
    """
    has_overlays = bool(overlays)
    has_subs = subtitles_path is not None and subtitles_path.exists()

    if not has_overlays and not has_subs:
        # Nothing to do — just rename/copy base to final name
        run(["ffmpeg", "-y", "-i", str(base_path), "-c", "copy", str(out_path)], quiet=True)
        return

    inputs: list[str] = ["-i", str(base_path)]
    for ov in overlays:
        ov_path = resolve_path(ov["file"], edit_dir)
        inputs += ["-i", str(ov_path)]

    filter_parts: list[str] = []
    # PTS-shift every overlay so its frame 0 lands at start_in_output
    for idx, ov in enumerate(overlays, start=1):
        t = float(ov["start_in_output"])
        filter_parts.append(f"[{idx}:v]setpts=PTS-STARTPTS+{t}/TB[a{idx}]")

    # Chain overlays on top of base
    current = "[0:v]"
    for idx, ov in enumerate(overlays, start=1):
        t = float(ov["start_in_output"])
        dur = float(ov["duration"])
        end = t + dur
        next_label = f"[v{idx}]"
        filter_parts.append(
            f"{current}[a{idx}]overlay=enable='between(t,{t:.3f},{end:.3f})'{next_label}"
        )
        current = next_label

    # Subtitles LAST — Rule 1
    if has_subs:
        subs_abs = str(subtitles_path.resolve()).replace(":", r"\:").replace("'", r"\'")
        filter_parts.append(
            f"{current}subtitles='{subs_abs}':force_style='{SUB_FORCE_STYLE}'[outv]"
        )
        out_label = "[outv]"
    else:
        # Rename the last overlay output to [outv] for consistency
        if has_overlays:
            filter_parts.append(f"{current}null[outv]")
            out_label = "[outv]"
        else:
            out_label = "[0:v]"

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", out_label,
        "-map", "0:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    print(f"compositing → {out_path.name}")
    print(f"  overlays: {len(overlays)}, subtitles: {'yes' if has_subs else 'no'}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


# -------- Main ---------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a video from an EDL")
    ap.add_argument("edl", type=Path, help="Path to edl.json")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output video path")
    ap.add_argument(
        "--preview",
        action="store_true",
        help="Preview mode: 1080p, medium, CRF 22 — evaluable for QC, faster than final.",
    )
    ap.add_argument(
        "--draft",
        action="store_true",
        help="Draft mode: 720p, ultrafast, CRF 28 — cut-point verification only.",
    )
    ap.add_argument(
        "--build-subtitles",
        action="store_true",
        help="Build master.srt from transcripts + EDL offsets before compositing",
    )
    ap.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Skip subtitles even if the EDL references one",
    )
    ap.add_argument(
        "--no-loudnorm",
        action="store_true",
        help="Skip audio loudness normalization. Default is on (-14 LUFS, -1 dBTP, LRA 11).",
    )
    args = ap.parse_args()

    edl_path = args.edl.resolve()
    if not edl_path.exists():
        sys.exit(f"edl not found: {edl_path}")

    edl = json.loads(edl_path.read_text())
    edit_dir = edl_path.parent
    out_path = args.output.resolve()

    # Surface J/L cut fields up front so the user understands they'll be
    # flattened to hard cuts in the ffmpeg path. FCPXML export honors them.
    warn_if_jl_cuts(edl)

    # 1. Extract per-segment (auto-grade per range if EDL grade is "auto")
    segment_paths = extract_all_segments(
        edl, edit_dir, preview=args.preview, draft=args.draft
    )

    # 2. Concat → base
    if args.draft:
        base_name = "base_draft.mp4"
    elif args.preview:
        base_name = "base_preview.mp4"
    else:
        base_name = "base.mp4"
    base_path = edit_dir / base_name
    concat_segments(segment_paths, base_path, edit_dir)

    # 3. Subtitles: build if requested, resolve final path
    subs_path: Path | None = None
    if not args.no_subtitles:
        if args.build_subtitles:
            subs_path = edit_dir / "master.srt"
            build_master_srt(edl, edit_dir, subs_path)
        elif edl.get("subtitles"):
            subs_path = resolve_path(edl["subtitles"], edit_dir)
            if not subs_path.exists():
                print(f"warning: subtitles path in EDL does not exist: {subs_path}")
                subs_path = None

    # 4. Composite (overlays + subtitles LAST) → intermediate (pre-loudnorm) path
    overlays = edl.get("overlays") or []
    if args.no_loudnorm:
        # Composite directly to final output
        build_final_composite(base_path, overlays, subs_path, out_path, edit_dir)
    else:
        # Composite to a temp file, then run loudnorm → final output
        tmp_composite = out_path.with_suffix(".prenorm.mp4")
        build_final_composite(base_path, overlays, subs_path, tmp_composite, edit_dir)
        print("loudness normalization → social-ready (-14 LUFS / -1 dBTP / LRA 11)")
        apply_loudnorm_two_pass(tmp_composite, out_path, preview=args.draft)
        tmp_composite.unlink(missing_ok=True)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\ndone: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
