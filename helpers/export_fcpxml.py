"""Export an EDL to FCPXML for Premiere Pro / DaVinci Resolve / Final Cut Pro.

Reads the same `edl.json` shape that helpers/render.py reads, but instead
of producing a flattened MP4 it produces an FCPXML timeline file that
opens natively in:

  - Adobe Premiere Pro          (File → Import → cut.fcpxml)
  - DaVinci Resolve             (File → Import Timeline → AAF/EDL/XML)
  - Apple Final Cut Pro         (File → Import → XML)

Why FCPXML and not EDL/AAF/CMX 3600:
  - FCPXML is the only widely-supported format that natively encodes
    SPLIT EDITS (J-cuts and L-cuts) — separate audio + video edges per
    clip. EDL / CMX 3600 is single-track and would force flattening.
  - OpenTimelineIO (the standard) has a maintained FCPXML adapter.
  - Round-trips between all three majors with zero massaging.

How J/L cuts map:
  - audio_lead  → the clip's AUDIO source_range starts (audio_lead) seconds
                  EARLIER than its VIDEO source_range. Audio bleeds in
                  under the previous clip's video. (J-cut)
  - video_tail  → the clip's AUDIO source_range ends (video_tail) seconds
                  LATER than its VIDEO source_range. Audio lingers under
                  the next clip's video. (L-cut)
  - transition_in → an otio.schema.Transition placed BEFORE this clip on
                    both tracks; OTIO's FCPXML adapter writes it as a
                    cross-dissolve.

Caveat: NLEs handle frame-aligned cuts. Whisper word timestamps land on
arbitrary milliseconds. The exporter snaps every cut edge to the nearest
frame at the EDL's `frame_rate` (default 24) so the import is clean.

Usage:
    python helpers/export_fcpxml.py <edl.json> -o cut.fcpxml
    python helpers/export_fcpxml.py <edl.json> -o cut.fcpxml --frame-rate 30

Dependencies (install via `pip install -e .[fcpxml]`):
    opentimelineio>=0.17
    otio-fcpx-xml-adapter>=0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# OTIO is heavy — defer the import so `--help` is fast and we can give a
# clean error if the optional extra wasn't installed.
# ---------------------------------------------------------------------------

def _import_otio():
    try:
        import opentimelineio as otio
    except ImportError as e:
        sys.exit(
            "FCPXML export requires the optional 'fcpxml' extra:\n"
            "  pip install -e .[fcpxml]\n"
            f"(import error: {e})"
        )
    return otio


# ---------------------------------------------------------------------------
# Frame-snapping. NLEs operate on integer frame counts; if we hand them
# 4.27593s they'll quietly round and complain about audio/video drift.
# ---------------------------------------------------------------------------

def _snap_to_frame(t_seconds: float, frame_rate: float) -> float:
    """Round a float-second timestamp to the nearest whole frame."""
    return round(t_seconds * frame_rate) / frame_rate


def _rt(seconds: float, frame_rate: float):
    """Build an otio.opentime.RationalTime at the given frame rate."""
    otio = _import_otio()
    return otio.opentime.RationalTime(
        value=round(seconds * frame_rate),
        rate=frame_rate,
    )


def _range(start_s: float, dur_s: float, frame_rate: float):
    """Build an otio.opentime.TimeRange (start_time + duration)."""
    otio = _import_otio()
    return otio.opentime.TimeRange(
        start_time=_rt(start_s, frame_rate),
        duration=_rt(dur_s, frame_rate),
    )


# ---------------------------------------------------------------------------
# Core build — produces an OTIO Timeline with two tracks (V1 + A1).
#
# Why we build two parallel tracks instead of relying on a single video
# track with attached audio: split edits (J/L cuts) require independent
# clip extents per track. OTIO's clip-with-attached-audio model would
# force matching extents, defeating the whole point.
# ---------------------------------------------------------------------------

def build_timeline(edl: dict, frame_rate: float):
    """Build and return an otio.schema.Timeline from the EDL.

    Schema fields honored:
      - sources           : map source_id → absolute path
      - ranges[]          : the cut list (also accepts top-level "edl")
        - source          : key into sources
        - start, end      : float seconds in the SOURCE clip
        - audio_lead      : J-cut offset (seconds, optional, default 0)
        - video_tail      : L-cut offset (seconds, optional, default 0)
        - transition_in   : dissolve seconds before this clip (optional)
        - beat / quote    : copied to clip metadata for editor reference
    """
    otio = _import_otio()

    sources = edl.get("sources") or {}
    ranges = edl.get("ranges") or edl.get("edl") or []
    if not ranges:
        raise ValueError("EDL has no ranges")

    timeline = otio.schema.Timeline(name=edl.get("name") or "video-use-premiere cut")
    # Timeline rate sets the granularity of TimeRanges in the file. NLEs
    # auto-convert on import but giving them the same rate the user
    # eventually delivers at avoids any subtle re-quantization.
    timeline.global_start_time = otio.opentime.RationalTime(0, frame_rate)

    v_track = otio.schema.Track(
        name="V1", kind=otio.schema.TrackKind.Video,
    )
    a_track = otio.schema.Track(
        name="A1", kind=otio.schema.TrackKind.Audio,
    )
    timeline.tracks.append(v_track)
    timeline.tracks.append(a_track)

    # Track current playhead on the OUTPUT timeline. This is what the
    # NLE will see — it's strictly increasing as we walk the EDL.
    cur_v = 0.0
    cur_a = 0.0

    for i, r in enumerate(ranges):
        src_name = r["source"]
        if src_name not in sources:
            raise KeyError(f"range[{i}].source '{src_name}' not in sources map")
        src_path = sources[src_name]

        start = float(r["start"])
        end = float(r["end"])
        dur = end - start
        if dur <= 0:
            print(f"  skip range[{i}] {src_name}: zero/negative duration",
                  file=sys.stderr)
            continue

        # ── J/L cut offsets ───────────────────────────────────────────
        a_lead = float(r.get("audio_lead", 0) or 0)   # J: audio starts earlier
        v_tail = float(r.get("video_tail", 0) or 0)   # L: audio ends later
        trans_in = float(r.get("transition_in", 0) or 0)

        # Snap everything to whole frames so NLE imports are clean.
        v_start = _snap_to_frame(start, frame_rate)
        v_end = _snap_to_frame(end, frame_rate)
        v_dur = max(0.0, v_end - v_start)

        # Audio source range is independently snapped — could be different
        # from the video range by ±half a frame after rounding.
        a_src_start = _snap_to_frame(start - a_lead, frame_rate)
        a_src_end = _snap_to_frame(end + v_tail, frame_rate)
        a_dur = max(0.0, a_src_end - a_src_start)

        # ── External media reference (one per clip — file path only) ──
        # OTIO's ExternalReference resolves to file:// URIs at write time.
        # NLEs follow the path on import; if the user moves the masters,
        # they'll get the standard "missing media" relink dialog, same as
        # for any imported XML.
        media_ref = otio.schema.ExternalReference(
            target_url=Path(src_path).resolve().as_uri(),
        )

        # Video clip
        v_clip = otio.schema.Clip(
            name=f"{src_name}_v_{i:02d}",
            media_reference=media_ref,
            source_range=_range(v_start, v_dur, frame_rate),
        )
        # Stash editorial metadata so the user can see WHY this cut was
        # chosen when they hover the clip in the NLE's clip inspector.
        v_clip.metadata["video-use-premiere"] = {
            "beat": r.get("beat"),
            "quote": r.get("quote"),
            "reason": r.get("reason"),
        }

        # Audio clip — independent source_range to support split edits.
        # Same media reference (NLE will only pull the audio track from it).
        a_clip = otio.schema.Clip(
            name=f"{src_name}_a_{i:02d}",
            media_reference=otio.schema.ExternalReference(
                target_url=Path(src_path).resolve().as_uri(),
            ),
            source_range=_range(a_src_start, a_dur, frame_rate),
        )

        # ── Cross-dissolve (transition_in) ────────────────────────────
        # Place a Transition BEFORE this clip on both tracks. OTIO's
        # FCPXML adapter writes it as a cross-dissolve. Cannot precede
        # the first clip on a track (no clip to dissolve from), so we
        # silently drop transition_in on i=0.
        if trans_in > 0 and i > 0:
            half = _snap_to_frame(trans_in / 2.0, frame_rate)
            half_rt = _rt(half, frame_rate)
            v_track.append(otio.schema.Transition(
                name=f"xfade_{i:02d}",
                in_offset=half_rt, out_offset=half_rt,
                transition_type=otio.schema.TransitionTypes.SMPTE_Dissolve,
            ))
            a_track.append(otio.schema.Transition(
                name=f"xfade_a_{i:02d}",
                in_offset=half_rt, out_offset=half_rt,
                transition_type=otio.schema.TransitionTypes.SMPTE_Dissolve,
            ))

        # ── J-cut: audio leads video ──────────────────────────────────
        # If audio leads by `a_lead`, we need the audio track to be
        # `a_lead` seconds AHEAD on the timeline. Insert a negative-
        # duration gap... no, gaps must be non-negative. Instead we
        # simply give the audio clip an EARLIER timeline position by
        # leaving a Gap of (cur_v - cur_a - a_lead) before it (which can
        # be zero) and letting its longer source_range absorb the lead.
        #
        # Concretely: if cur_a is currently behind cur_v due to no
        # previous L-cut, a J-cut here means we want audio to START at
        # cur_v - a_lead. Pad audio track with a Gap to reach that.
        target_a_start = cur_v - a_lead
        a_gap = target_a_start - cur_a
        if a_gap > 1e-6:
            a_track.append(otio.schema.Gap(
                source_range=_range(0.0, a_gap, frame_rate),
            ))

        v_track.append(v_clip)
        a_track.append(a_clip)

        cur_v += v_dur
        cur_a = target_a_start + a_dur  # inherits both lead AND tail

    return timeline


# ---------------------------------------------------------------------------
# Write — defers to OTIO's adapter registry which dispatches by extension.
# ---------------------------------------------------------------------------

def write_fcpxml(timeline, out_path: Path) -> None:
    """Write the timeline to disk as FCPXML.

    OTIO discovers the FCPXML adapter via the otio_fcpx_xml_adapter
    package (declared in pyproject.toml's [fcpxml] extra). If it's
    missing, OTIO raises a clean error here.
    """
    otio = _import_otio()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        otio.adapters.write_to_file(timeline, str(out_path))
    except otio.exceptions.NoKnownAdapterForExtensionError:
        sys.exit(
            "OTIO has no FCPXML adapter installed. Add the extra:\n"
            "  pip install -e .[fcpxml]\n"
            "(this pulls in otio-fcpx-xml-adapter)."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export an edl.json to FCPXML for Premiere/Resolve/FCP",
    )
    ap.add_argument("edl", type=Path, help="Path to edl.json")
    ap.add_argument("-o", "--output", type=Path, required=True,
                    help="Output FCPXML file path")
    ap.add_argument("--frame-rate", type=float, default=24.0,
                    help="Timeline frame rate. Snap all cuts to whole frames "
                         "at this rate. Default 24. Common: 23.976, 25, 29.97, 30, 60.")
    args = ap.parse_args()

    edl_path = args.edl.resolve()
    if not edl_path.exists():
        sys.exit(f"edl not found: {edl_path}")

    edl = json.loads(edl_path.read_text(encoding="utf-8"))
    timeline = build_timeline(edl, frame_rate=args.frame_rate)
    write_fcpxml(timeline, args.output.resolve())

    n_clips = sum(
        1 for t in timeline.tracks
        for c in t
        if c.__class__.__name__ == "Clip"
    )
    n_trans = sum(
        1 for t in timeline.tracks
        for c in t
        if c.__class__.__name__ == "Transition"
    )
    kb = args.output.stat().st_size / 1024
    print(f"FCPXML written: {args.output}  ({kb:.1f} KB)")
    print(f"  timeline rate: {args.frame_rate} fps")
    print(f"  clips: {n_clips}  transitions: {n_trans}")
    print(f"  open in: Premiere Pro / DaVinci Resolve / Final Cut Pro")


if __name__ == "__main__":
    main()
