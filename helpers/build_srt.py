"""Build a master SRT from per-source transcripts + an EDL.

`helpers/export_fcpxml.py` calls `build_master_srt()` automatically as
part of every export so the NLE handoff always ships a captions
sidecar alongside `cut.fcpxml` / `cut.xml`. This module is also wired
as a standalone CLI for the case where you tweaked the EDL and only
want to regenerate the SRT without re-walking the timeline.

Hard Rule 5 binds here: SRT timestamps live on the OUTPUT timeline,
computed as `word.start - segment_start + segment_offset`, because
the editor sub-agent's EDL ranges are in SOURCE time but the captions
attach to the OUTPUT timeline. Mis-applying this turns captions into
a slow-drifting train wreck after the first cut.

Output format — Premiere-friendly:
  * UTF-8 (no BOM) — modern Premiere / Resolve / FCP X all read this
    cleanly; the BOM was historically required by Premiere CC 2018
    and earlier and we don't target those.
  * CRLF line endings — what Notepad-class editors expect on Windows
    so the file is human-readable when the operator double-clicks it,
    and no NLE we target chokes on \r\n.
  * Sequential numbering from 1, exact `HH:MM:SS,mmm --> HH:MM:SS,mmm`
    timestamp shape, blank line between cues — the SRT spec proper.

Style hint (not applied here — the NLE decides): the proven launch-
video look is 2-word UPPERCASE chunks, Helvetica 18 Bold, MarginV=35.
See `references/subtitles.md` for the chunking / case / placement
discussion. We just emit the SRT — restyle in your NLE.

Usage:
    python helpers/build_srt.py <edl.json> -o <out.srt>

The output path is optional. When omitted the SRT is written next to
the EDL as `<edl_dir>/master.srt` — the path the FCPXML exporter and
the editor sub-agent's documentation both expect by default.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Force UTF-8 on stdout/stderr so Unicode arrows / em-dashes / ellipses
# in our log output don't crash on Windows consoles still defaulting
# to cp1252. Cheap, idempotent, only runs once at import time.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Time-squeezing (timelapse) support — output-timeline aware.
#
# Mirrors the speed handling in helpers/export_fcpxml.py: per-range
# `speed` (float, default 1.0). When a range is retimed, by editor
# convention the timelapse contains no speech (otherwise it'd be
# unintelligible at 4x+), so caption synthesis is SKIPPED for that
# range. We still advance the seg_offset accumulator by the OUTPUT
# duration so subsequent ranges' captions land on the right output
# time. Same ceiling as the FCPXML exporter — beyond 1000% the LLM
# should be CUTTING, not squeezing harder.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# SRT writer
# ---------------------------------------------------------------------------

# Punctuation set that forces an early chunk break — keeps caption
# rhythm aligned with sentence flow rather than the rigid 2-word grid.
PUNCT_BREAK = set(".,!?;:")


def _srt_timestamp(seconds: float) -> str:
    """Format a float-seconds value as HH:MM:SS,mmm — the SRT standard."""
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _words_in_range(transcript: dict, t_start: float, t_end: float) -> list[dict]:
    """Slice a Parakeet transcript down to words overlapping [t_start, t_end]."""
    out: list[dict] = []
    for w in transcript.get("words", []):
        if w.get("type") != "word":
            continue
        ws = w.get("start")
        we = w.get("end")
        if ws is None or we is None:
            continue
        # Overlap test on the half-open interval [ws, we) vs [t_start, t_end).
        if we <= t_start or ws >= t_end:
            continue
        out.append(w)
    return out


def resolve_path(maybe_path: str, base: Path) -> Path:
    """Resolve a path that may be absolute or relative to `base`."""
    p = Path(maybe_path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def build_master_srt(edl: dict, edit_dir: Path, out_path: Path) -> None:
    """Build an output-timeline SRT from per-source transcripts.

    - 2-word chunks (break early on any punctuation in the chunk)
    - UPPERCASE text (the bold-overlay style — restyle in NLE if needed)
    - Output times are translated into the cut timeline:
      `output_time = word.start - segment_start + segment_offset`
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
        # viewer actually sees, not the raw source range.
        seg_speed = _read_speed(r, i)
        seg_out_dur = seg_duration / seg_speed if seg_speed != 1.0 else seg_duration

        # Skip caption synthesis on retimed ranges — by editor convention
        # timelapses contain no speech (see SKILL.md "Time-squeezing").
        # Even if the source had stray dialogue, speech sped up 4x+ is
        # unintelligible and shouldn't be captioned. We still advance
        # seg_offset by the OUTPUT duration so subsequent ranges' cues
        # land on the right output time.
        if seg_speed != 1.0:
            seg_offset += seg_out_dur
            continue

        # Transcripts are cached by the *source file stem* (see
        # parakeet_onnx_lane.py: `transcripts_dir / f"{video_path.stem}.json"`),
        # NOT by the EDL short label. The EDL `src_name` is just a human-
        # friendly key (e.g. "C0303") that maps, via `sources[src_name]`,
        # to the real filename / path on disk. We resolve through that
        # mapping so the SRT lookup matches whatever Parakeet wrote.
        src_ref = sources.get(src_name)
        if not src_ref:
            print(f"  no source mapping for {src_name}, skipping captions for this segment")
            seg_offset += seg_out_dur
            continue
        src_stem = Path(src_ref).stem

        tr_path = transcripts_dir / f"{src_stem}.json"
        if not tr_path.exists():
            # Fallback: also try the bare label, for older preprocess runs
            # that keyed transcripts by the short EDL name instead of the
            # file stem on disk.
            legacy = transcripts_dir / f"{src_name}.json"
            if legacy.exists():
                tr_path = legacy
            else:
                print(f"  no transcript for {src_name} (looked for {src_stem}.json), skipping captions for this segment")
                seg_offset += seg_out_dur
                continue

        # Parakeet writes transcripts as UTF-8 — be explicit so Windows
        # hosts don't fall back to cp1252 and choke on smart-quotes /
        # em-dashes the speech model occasionally emits.
        transcript = json.loads(tr_path.read_text(encoding="utf-8"))
        words_in_seg = _words_in_range(transcript, seg_start, seg_end)

        # Group into 2-word chunks, break early on punctuation so caption
        # rhythm tracks sentence flow rather than mechanically counting words.
        chunks: list[list[dict]] = []
        current: list[dict] = []
        for w in words_in_seg:
            text = (w.get("text") or "").strip()
            if not text:
                continue
            current.append(w)
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
            # Defensive minimum cue duration — protects against zero-length
            # cues when a single-word chunk falls on a tight word boundary.
            if out_end <= out_start:
                out_end = out_start + 0.4
            text = " ".join((w.get("text") or "").strip() for w in chunk)
            text = re.sub(r"\s+", " ", text).strip()
            # Strip trailing punctuation for a cleaner all-caps look.
            text = text.rstrip(",;:")
            text = text.upper()
            entries.append((out_start, out_end, text))

        seg_offset += seg_out_dur

    # No captions to write — bail without creating a stub file. Tells
    # the operator something is structurally wrong (e.g. transcripts
    # missing) instead of silently shipping an empty SRT to the NLE.
    if not entries:
        print(f"  warn: no captions emitted (no transcripts found for any "
              f"range in {len(edl['ranges'])} ranges); skipping {out_path.name}",
              file=sys.stderr)
        return

    # Sort by output start time and serialize as SRT. CRLF line endings
    # so Notepad-class editors render the file legibly on Windows; SRT
    # consumers (Premiere, Resolve, FCP X, ffmpeg) all read either flavour.
    # `newline=""` on write_text disables Python's universal-newline
    # translation — without it, write_text would re-encode \n -> \r\n on
    # Windows AND re-encode \r\n -> \r\r\n inside our string, doubling.
    entries.sort(key=lambda e: e[0])
    lines: list[str] = []
    for i, (a, b, t) in enumerate(entries, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(a)} --> {_srt_timestamp(b)}")
        lines.append(t)
        lines.append("")
    out_path.write_text("\r\n".join(lines), encoding="utf-8", newline="")
    print(f"master SRT -> {out_path.name} ({len(entries)} cues)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build master.srt from an EDL + per-source transcripts."
    )
    ap.add_argument("edl", type=Path, help="Path to edl.json")
    ap.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output SRT path. Defaults to <edl_dir>/master.srt.",
    )
    args = ap.parse_args()

    edl_path = args.edl.resolve()
    if not edl_path.exists():
        sys.exit(f"edl not found: {edl_path}")

    edl = json.loads(edl_path.read_text(encoding="utf-8"))
    edit_dir = edl_path.parent
    out_path = (args.output.resolve()
                if args.output is not None
                else edit_dir / "master.srt")
    build_master_srt(edl, edit_dir, out_path)


if __name__ == "__main__":
    main()
