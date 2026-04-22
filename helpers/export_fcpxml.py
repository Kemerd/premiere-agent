"""Export an EDL to NLE-native interchange for Premiere Pro / Resolve / FCP X.

Reads the same `edl.json` shape that helpers/render.py reads, but instead
of producing a flattened MP4 it produces editor-ready timeline file(s).

Two flavors, picked by the receiving NLE:

  * .fcpxml  — Final Cut Pro X / FCPXML 1.10+. Native to:
      - Apple Final Cut Pro   (File → Import → XML)
      - DaVinci Resolve       (File → Import → Timeline → AAF/EDL/XML)
    Premiere Pro does NOT read this directly — Adobe documents the
    XtoCC translator workflow for it.

  * .xml     — Final Cut Pro 7 xmeml. Native to:
      - Adobe Premiere Pro    (File → Import → cut.xml)
    No XtoCC, no extra tooling. This is the Premiere handoff path.

Default behaviour is to emit BOTH from a single timeline build so the
recipient picks whichever NLE they live in without us having to re-run
anything. Override with `--targets {fcpxml,premiere,both}`.

Why split-edit-friendly XML and not EDL/AAF/CMX 3600:
  - Both XML dialects natively encode SPLIT EDITS (J-cuts and L-cuts)
    via independent audio + video extents per clip. CMX 3600 is
    single-track and would force flattening.
  - OpenTimelineIO ships maintained adapters for both dialects.
  - Round-trips across the three majors with zero massaging.

How J/L cuts map:
  - audio_lead  → the clip's AUDIO source_range starts (audio_lead) seconds
                  EARLIER than its VIDEO source_range. Audio bleeds in
                  under the previous clip's video. (J-cut)
  - video_tail  → the clip's AUDIO source_range ends (video_tail) seconds
                  LATER than its VIDEO source_range. Audio lingers under
                  the next clip's video. (L-cut)
  - transition_in → an otio.schema.Transition placed BEFORE this clip on
                    both tracks; OTIO's adapters write it as a
                    cross-dissolve in either dialect.

Caveat: NLEs handle frame-aligned cuts. Whisper / Parakeet word
timestamps land on arbitrary milliseconds. The exporter snaps every cut
edge to the nearest frame at the EDL's `frame_rate` (default 24) so the
import is clean.

Usage:
    # Default — emit both cut.fcpxml AND cut.xml side-by-side
    python helpers/export_fcpxml.py <edl.json> -o cut.fcpxml

    # Resolve / FCP X only
    python helpers/export_fcpxml.py <edl.json> -o cut.fcpxml --targets fcpxml

    # Premiere Pro only (FCP7 xmeml)
    python helpers/export_fcpxml.py <edl.json> -o cut.xml --targets premiere

Dependencies (install via `pip install -e .[fcpxml]`):
    opentimelineio>=0.17
    otio-fcpx-xml-adapter>=0.2     # .fcpxml writer (Resolve / FCP X)
    otio-fcp-adapter>=0.2          # .xml writer    (Premiere Pro native)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Source-media probing — the FCPXML adapter requires every external media
# reference to have an `available_range` so it can write the asset's full
# duration into the FCPXML <asset>/<format> declaration. NLEs use this to
# pre-allocate the timeline view and to reject obviously corrupt imports.
#
# We ffprobe each unique source ONCE and cache the result so a 60-clip EDL
# referencing 4 source files only costs 4 ffprobe calls.
# ---------------------------------------------------------------------------

# Be generous on ffprobe failure — a long fake duration is preferable to
# refusing to write the file. NLEs will still relink on the actual source
# and use that source's real duration at conform time.
_FFPROBE_FALLBACK_DURATION_S = 24 * 60 * 60.0  # 24 h sentinel

_PROBE_CACHE: dict[str, float] = {}


def _probe_source_duration_s(path: Path) -> float:
    """Return the source media duration in seconds, ffprobe + cached.

    On any failure (missing ffprobe, unreadable file, malformed output)
    return a 24h sentinel so the FCPXML still writes — Premiere will
    reconcile against the real file at relink time.
    """
    key = str(path)
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        dur = float(out)
        if dur <= 0:
            raise ValueError("non-positive duration")
    except Exception as e:
        # Don't crash the export — the alternative is no FCPXML at all.
        # Mention the fallback in stderr so the user can investigate if
        # they care; most won't, the 24h sentinel works in every NLE
        # we've tested.
        print(
            f"  warn: ffprobe failed for {path.name} ({type(e).__name__}: "
            f"{e}); using {_FFPROBE_FALLBACK_DURATION_S/3600:.0f}h sentinel "
            "available_range. NLE will relink to actual source duration.",
            file=sys.stderr,
        )
        dur = _FFPROBE_FALLBACK_DURATION_S
    _PROBE_CACHE[key] = dur
    return dur


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
        #
        # available_range is REQUIRED by the FCPXML adapter — it writes
        # this into the <asset> declaration. We ffprobe the source once
        # and cache. See _probe_source_duration_s().
        src_path_resolved = Path(src_path).resolve()
        media_dur_s = _probe_source_duration_s(src_path_resolved)
        media_avail_range = otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, frame_rate),
            duration=_rt(media_dur_s, frame_rate),
        )

        def _new_media_ref():
            """Each clip gets its own ExternalReference because the FCPXML
            adapter sometimes writes per-asset state; sharing one ref
            across clips has caused 'asset already declared' errors in
            past adapter versions. Cheap enough to construct per-clip."""
            return otio.schema.ExternalReference(
                target_url=src_path_resolved.as_uri(),
                available_range=media_avail_range,
            )

        # Video clip
        v_clip = otio.schema.Clip(
            name=f"{src_name}_v_{i:02d}",
            media_reference=_new_media_ref(),
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
        # Same source media (NLE will only pull the audio track from it).
        a_clip = otio.schema.Clip(
            name=f"{src_name}_a_{i:02d}",
            media_reference=_new_media_ref(),
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
# Writers — one per dialect. OTIO dispatches by file extension under the
# hood so the surface API stays trivial; each writer just owns the
# "missing adapter" diagnostic for its dialect.
#
# Both writers accept the SAME otio.schema.Timeline instance, so a single
# build_timeline() pass feeds both outputs. There's no duplication of the
# expensive ffprobe / frame-snapping work between them.
# ---------------------------------------------------------------------------

# Friendly target -> (extension, adapter pip package, NLE list) mapping.
# Used both at write time (for diagnostics) and at CLI parse time (to
# derive sibling output paths).
_TARGET_INFO = {
    "fcpxml": {
        "ext": ".fcpxml",
        "adapter_pkg": "otio-fcpx-xml-adapter",
        "opens_in": "DaVinci Resolve / Final Cut Pro X",
    },
    "premiere": {
        "ext": ".xml",
        "adapter_pkg": "otio-fcp-adapter",
        "opens_in": "Adobe Premiere Pro (File -> Import, native xmeml)",
    },
}


def write_fcpxml(timeline, out_path: Path) -> None:
    """Write the timeline as FCPXML 1.10+ (.fcpxml) — Resolve / FCP X path.

    OTIO discovers the writer via the `otio_fcpx_xml_adapter` package
    (declared in pyproject.toml's [fcpxml] extra). If it's missing we
    raise a clean install hint instead of letting OTIO's generic
    "no adapter for extension" message reach the user.
    """
    otio = _import_otio()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        otio.adapters.write_to_file(timeline, str(out_path))
    except otio.exceptions.NoKnownAdapterForExtensionError:
        sys.exit(
            "OTIO has no FCPXML (.fcpxml) adapter installed. Fix:\n"
            "  pip install -e .[fcpxml]\n"
            "(this pulls in otio-fcpx-xml-adapter for Resolve / FCP X "
            "and otio-fcp-adapter for Premiere Pro)."
        )


def write_premiere_xml(timeline, out_path: Path) -> None:
    """Write the timeline as Final Cut Pro 7 xmeml (.xml) — Premiere path.

    Why a separate writer: Premiere Pro does NOT natively read FCPXML
    1.10+ (the .fcpxml extension). It reads the older Final Cut Pro 7
    xmeml flavor (.xml). OTIO ships an `fcp_xml` adapter for that
    dialect via the `otio-fcp-adapter` PyPI package — it lands the
    same split-edit / dissolve fidelity as the .fcpxml path because
    xmeml supports both natively. End result: Premiere Pro users get
    a one-click File -> Import experience and skip the XtoCC step
    Adobe documents.
    """
    otio = _import_otio()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        otio.adapters.write_to_file(timeline, str(out_path))
    except otio.exceptions.NoKnownAdapterForExtensionError:
        sys.exit(
            "OTIO has no FCP7 xmeml (.xml) adapter installed. Fix:\n"
            "  pip install -e .[fcpxml]\n"
            "(this pulls in otio-fcp-adapter for Premiere Pro's native "
            "xmeml import — no XtoCC required)."
        )


# ---------------------------------------------------------------------------
# Output-path resolution. `-o` is treated as a basename: we strip the
# extension and re-attach the canonical one per target. That way the user
# can pass `-o cut.fcpxml`, `-o cut.xml`, or just `-o cut` and we DTRT.
# ---------------------------------------------------------------------------

def _resolve_output_paths(
    user_output: Path, targets: str
) -> tuple[Path | None, Path | None]:
    """Return (fcpxml_path, premiere_xml_path) per the --targets choice.

    Either entry is None when that target is disabled. The basename
    (parent + stem) is taken from `user_output`; we always re-attach
    the canonical extension so we never collide (cut.fcpxml + cut.xml).
    """
    parent = user_output.parent
    stem = user_output.stem
    fcpx = parent / f"{stem}{_TARGET_INFO['fcpxml']['ext']}"
    prxml = parent / f"{stem}{_TARGET_INFO['premiere']['ext']}"
    if targets == "both":
        return fcpx, prxml
    if targets == "fcpxml":
        return fcpx, None
    if targets == "premiere":
        return None, prxml
    raise ValueError(f"unknown --targets value: {targets!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export an edl.json to NLE-native interchange XML "
                    "(FCPXML for Resolve / FCP X, FCP7 xmeml for Premiere).",
    )
    ap.add_argument("edl", type=Path, help="Path to edl.json")
    ap.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Output basename. Extension is normalized per target — pass "
             "`cut.fcpxml`, `cut.xml`, or just `cut` and we'll attach the "
             "right suffix(es) per --targets.",
    )
    ap.add_argument(
        "--targets", choices=["both", "fcpxml", "premiere"], default="both",
        help="Which dialect(s) to emit. Default: both. "
             "`fcpxml` -> .fcpxml only (Resolve / FCP X). "
             "`premiere` -> .xml only (Premiere native). "
             "`both` writes side-by-side from a single timeline build.",
    )
    ap.add_argument(
        "--frame-rate", type=float, default=24.0,
        help="Timeline frame rate. Snap all cuts to whole frames at this "
             "rate. Default 24. Common: 23.976, 25, 29.97, 30, 60.",
    )
    args = ap.parse_args()

    edl_path = args.edl.resolve()
    if not edl_path.exists():
        sys.exit(f"edl not found: {edl_path}")

    edl = json.loads(edl_path.read_text(encoding="utf-8"))

    # Build ONCE — both writers consume the same otio.schema.Timeline.
    # ffprobe + frame-snapping costs are paid here, not per-dialect.
    timeline = build_timeline(edl, frame_rate=args.frame_rate)

    fcpx_out, prxml_out = _resolve_output_paths(args.output.resolve(), args.targets)

    n_clips = sum(
        1 for t in timeline.tracks for c in t
        if c.__class__.__name__ == "Clip"
    )
    n_trans = sum(
        1 for t in timeline.tracks for c in t
        if c.__class__.__name__ == "Transition"
    )
    print(f"timeline built: {n_clips} clips, {n_trans} transitions, "
          f"{args.frame_rate} fps")

    # Emit each requested dialect. Failures in one writer don't prevent
    # the other from running — the user shouldn't lose the Premiere file
    # because, say, the Resolve adapter hit a bug on their OTIO version.
    if fcpx_out is not None:
        try:
            write_fcpxml(timeline, fcpx_out)
            kb = fcpx_out.stat().st_size / 1024
            print(f"  [fcpxml]   {fcpx_out}  ({kb:.1f} KB)  "
                  f"-> {_TARGET_INFO['fcpxml']['opens_in']}")
        except SystemExit:
            # Re-raise install-hint exits unchanged so the user sees the fix.
            raise
        except Exception as e:
            print(f"  [fcpxml]   FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr)

    if prxml_out is not None:
        try:
            write_premiere_xml(timeline, prxml_out)
            kb = prxml_out.stat().st_size / 1024
            print(f"  [premiere] {prxml_out}  ({kb:.1f} KB)  "
                  f"-> {_TARGET_INFO['premiere']['opens_in']}")
        except SystemExit:
            raise
        except Exception as e:
            print(f"  [premiere] FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
