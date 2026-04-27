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

Time-squeezing (timelapse) support:
  Every range may carry an optional `speed` field (float, default 1.0).
  When `speed > 1.0` the source segment plays back compressed by that
  factor on the output timeline — e.g. 60s of source at speed=12.0 lands
  as a 5s timelapse on the cut. Both XML dialects encode this natively:
    - FCPXML 1.10+ : a <timeMap> element inside the <clip>, mapping
                     timeline-clip-time → source-clip-time linearly.
    - FCP7 xmeml   : a <filter> block carrying the standard "Time Remap"
                     effect (effectid=timeremap) with a constant speed
                     percentage. Premiere imports this without prompting.
  An optional `audio_strategy` field per range chooses the audio
  behaviour for sped-up clips:
    - "drop" (default for speed != 1.0): the audio track is a silent gap
      over the output duration. The intended workflow for timelapses
      where the raw audio (sustained tool noise, ambience) is going to
      be replaced by music or sound design in the NLE.
    - "keep" : the audio clip is retimed alongside the video. NLEs that
      expose a "preserve pitch on retime" toggle (Premiere, FCP X) will
      let the editor decide whether the audio sounds natural or chipmunk.
  See "Time-squeezing" in SKILL.md for when to actually reach for this.

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

Captions sidecar:
  Every export ALSO emits `<edl_dir>/master.srt` by default — UTF-8
  with CRLF line endings, sequential cues, exact `HH:MM:SS,mmm -->`
  timestamps. Premiere Pro, DaVinci Resolve, and Final Cut Pro X all
  import this onto a captions track via File -> Import. Pass
  `--no-srt` to skip, or `--srt-output <path>` to override the path.
  See helpers/build_srt.py for the chunking + style notes.

Usage:
    # Default — emit cut.fcpxml + cut.xml + master.srt side-by-side
    python helpers/export_fcpxml.py <edl.json> -o cut.fcpxml

    # Resolve / FCP X only (still emits master.srt)
    python helpers/export_fcpxml.py <edl.json> -o cut.fcpxml --targets fcpxml

    # Premiere Pro only (FCP7 xmeml, still emits master.srt)
    python helpers/export_fcpxml.py <edl.json> -o cut.xml --targets premiere

    # Captions out of scope — skip the SRT
    python helpers/export_fcpxml.py <edl.json> -o cut.fcpxml --no-srt

Dependencies (install via `pip install -e .[fcpxml]`):
    opentimelineio>=0.17
    otio-fcpx-xml-adapter>=0.2     # .fcpxml writer (Resolve / FCP X)
    otio-fcp-adapter>=0.2          # .xml writer    (Premiere Pro native)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Source-media probing — the FCPXML / xmeml adapters require every external
# media reference to have an `available_range` so they can write the asset's
# full duration into the asset declaration. NLEs use this to pre-allocate
# the timeline view and to reject obviously corrupt imports.
#
# We also need the AUDIO channel count + sample rate per source. Premiere
# strict-validates "Cannot Link Media: file has N audio channel(s) and the
# clip was created with M audio channel(s) with a different channel type."
# when the asset declaration lies about the source's audio shape — so we
# probe it and write the truth.
#
# We ffprobe each unique source ONCE and cache the result so a 60-clip EDL
# referencing 4 source files only costs 4 ffprobe calls.
# ---------------------------------------------------------------------------

# Be generous on ffprobe failure — a long fake duration is preferable to
# refusing to write the file. NLEs will still relink on the actual source
# and use that source's real duration at conform time.
_FFPROBE_FALLBACK_DURATION_S = 24 * 60 * 60.0  # 24 h sentinel

# Sensible defaults when ffprobe can't read a stream (rare). Stereo 48kHz
# is what 99% of camera/phone footage ships — guessing this matches more
# reality than guessing mono, AND matches what Premiere bins assume by
# default, so the relink dialog doesn't fire on the fallback path either.
_DEFAULT_AUDIO_CHANNELS = 2
_DEFAULT_AUDIO_SAMPLERATE = 48000
_DEFAULT_VIDEO_WIDTH = 1920
_DEFAULT_VIDEO_HEIGHT = 1080
_DEFAULT_VIDEO_FPS = 24.0  # only used when no source has a probe-able fps

# FCPXML 1.10+ colorSpace string table. Maps ffprobe's
# (color_primaries, color_transfer, color_matrix) triple to the exact
# string Apple expects in <format colorSpace="..."/>. The legacy strings
# (no parenthesised name) are also accepted by Resolve and Premiere; the
# parenthesised form is what FCP X writes itself, so we use it for max
# round-trip cleanliness.
#
# We key on (transfer, primaries) — the matrix usually mirrors primaries
# for these well-known spaces, and ffprobe occasionally omits one but
# rarely both. Anything unrecognised falls through to Rec. 709, which is
# the safe assumption for SDR camera footage.
# Format is "<primaries>-<transfer>-<matrix> (Pretty Name)" using the
# ISO/IEC 23091-2 numeric code points Apple expects:
#   primaries: 1=Rec.709, 9=Rec.2020, 12=DCI-P3, 6=NTSC, 5=PAL
#   transfer:  1=Rec.709 SDR, 13=sRGB, 16=HLG, 17=ST.428, 18=PQ
#   matrix:    1=Rec.709, 9=Rec.2020 NCL, 6=Rec.601 525, 5=Rec.601 625
# Anything missing falls through to Rec. 709 — safe SDR default.
_FCPXML_COLORSPACE_TABLE: dict[tuple[str, str], str] = {
    # SDR Rec. 709 (99% of HD camera footage)
    ("bt709", "bt709"):           "1-1-1 (Rec. 709)",
    ("bt709", "unknown"):         "1-1-1 (Rec. 709)",
    ("iec61966-2-1", "bt709"):    "1-13-1 (Rec. 709 sRGB)",
    ("smpte170m", "smpte170m"):   "1-1-1 (Rec. 709)",
    # SD NTSC / PAL — mapped to Rec. 709 by FCP X for SD masters too
    ("smpte170m", "bt470bg"):     "1-1-1 (Rec. 709)",
    ("bt470bg", "bt470bg"):       "1-1-1 (Rec. 709)",
    # Rec. 2020 SDR — matrix is 9 (BT.2020 NCL), NOT 1 (was a bug)
    ("bt2020-10", "bt2020"):      "9-1-9 (Rec. 2020)",
    ("bt2020-10", "bt2020nc"):    "9-1-9 (Rec. 2020)",
    ("bt2020-12", "bt2020"):      "9-1-9 (Rec. 2020)",
    ("bt2020-12", "bt2020nc"):    "9-1-9 (Rec. 2020)",
    # Rec. 2020 HDR — HLG (DJI / GoPro / iPhone HLG output)
    ("arib-std-b67", "bt2020"):   "9-16-9 (Rec. 2020 HLG)",
    ("arib-std-b67", "bt2020nc"): "9-16-9 (Rec. 2020 HLG)",
    # Rec. 2020 HDR — PQ / HDR10 / Dolby Vision Profile 5/8
    ("smpte2084", "bt2020"):      "9-18-9 (Rec. 2020 PQ)",
    ("smpte2084", "bt2020nc"):    "9-18-9 (Rec. 2020 PQ)",
    # DCI-P3 (Apple ProRes XQ + iPhone "Wide Color" HEVC)
    ("bt709", "smpte432"):        "12-1-1 (Display P3)",
    ("iec61966-2-1", "smpte432"): "12-13-1 (Display P3 sRGB)",
}
_DEFAULT_FCPXML_COLORSPACE = "1-1-1 (Rec. 709)"


# ---------------------------------------------------------------------------
# Timestretch diagnostics — former per-export `.timestretch.log` file was
# removed; call sites stay as no-ops so build/patch code stays readable.
# ---------------------------------------------------------------------------

def _ts_log(_msg: str) -> None:
    """No-op; timestretch file logging was removed."""
    pass


def _ts_log_section(_title: str) -> None:
    """No-op; timestretch file logging was removed."""
    pass


def _classify_colorspace(transfer: str, primaries: str) -> str:
    """Map ffprobe color tags to an FCPXML 1.10+ colorSpace string.

    Defaults to Rec. 709 — Premiere and Resolve will accept that for any
    SDR source even when the camera tagged things weirdly, and HDR
    metadata gets a proper mapping when ffprobe surfaces it.
    """
    t = (transfer or "unknown").lower()
    p = (primaries or "unknown").lower()
    return _FCPXML_COLORSPACE_TABLE.get((t, p), _DEFAULT_FCPXML_COLORSPACE)


# Canonical FPS values that OTIO's fcpx-xml adapter knows how to encode
# (FRAMERATE_FRAMEDURATION dict keys in otio_fcpx_xml_adapter/fcpx_xml.py).
# The adapter does an exact-equality lookup against these keys; any rate
# that doesn't match exactly gets an empty <format frameDuration=""/>,
# which the same adapter then crashes on while reading back during write
# (`split("/")` on an empty string -> "expected 2, got 1" ValueError).
#
# ffprobe returns NTSC rates as their EXACT rational fractions:
#     60000/1001  = 59.94005994005994...    (NOT 59.94)
#     30000/1001  = 29.97002997002997...    (NOT 29.97)
#     24000/1001  = 23.976023976023976...   (NOT 23.98)
# We snap any rate within a tight tolerance of a canonical key onto that
# key, so the adapter's lookup hits and the .fcpxml writes cleanly.
# Integer rates pass through unchanged.
_FCPXML_CANONICAL_FPS: tuple[float, ...] = (
    23.98, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0,
)


def _snap_fps_for_fcpxml(fps: float, tolerance: float = 0.05) -> float:
    """Snap a probed fps onto the nearest OTIO-known FCPXML rate.

    Returns the original value unchanged when no canonical rate is
    within `tolerance` (i.e. for genuinely unusual rates like 48 or
    120 — the adapter will still produce an empty frameDuration but
    that's an OTIO-side limitation we can't paper over here, and the
    Premiere xmeml writer doesn't have this issue at all).
    """
    if fps <= 0:
        return fps
    best_key = min(_FCPXML_CANONICAL_FPS, key=lambda k: abs(k - fps))
    if abs(best_key - fps) <= tolerance:
        return best_key
    return fps


def _parse_fps(rate_str: str) -> float | None:
    """Parse ffprobe rational rate strings like '30000/1001' or '24/1'.

    Returns None for empty / malformed / zero-denominator inputs so
    callers can fall back without exception spam.
    """
    if not rate_str or rate_str == "0/0":
        return None
    try:
        if "/" in rate_str:
            num, den = rate_str.split("/", 1)
            n, d = float(num), float(den)
            if d <= 0:
                return None
            return n / d
        return float(rate_str)
    except (TypeError, ValueError):
        return None


# SourceMeta cache — one entry per absolute source path. Values are the
# probed shape of the source; see _probe_source_meta() for field semantics.
_PROBE_CACHE: dict[str, dict] = {}


def _probe_source_meta(path: Path) -> dict:
    """Return source media shape dict, ffprobe + cached.

    Keys:
      duration_s     : float — full media duration in seconds
      has_video      : bool  — at least one video stream present
      has_audio      : bool  — at least one audio stream present
      audio_channels : int   — channel count of the FIRST audio stream
      audio_rate     : int   — sample rate (Hz) of the FIRST audio stream
      video_width    : int   — pixel width of the first video stream
      video_height   : int   — pixel height of the first video stream

    On any failure (missing ffprobe, unreadable file, malformed output)
    we fall back to a stereo-48k / 1080p / 24h-duration shape so the
    export still writes and Premiere/Resolve relinks against the real
    file at conform time.
    """
    key = str(path)
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]

    # Single ffprobe call returns BOTH format-level (duration) and
    # stream-level (audio channels, sample rate, video dims, fps, color)
    # as JSON — cheaper than two probes and atomically consistent.
    meta = {
        "duration_s": _FFPROBE_FALLBACK_DURATION_S,
        "has_video": True,
        "has_audio": True,
        "audio_channels": _DEFAULT_AUDIO_CHANNELS,
        "audio_rate": _DEFAULT_AUDIO_SAMPLERATE,
        "video_width": _DEFAULT_VIDEO_WIDTH,
        "video_height": _DEFAULT_VIDEO_HEIGHT,
        # Frame rate stays None until we positively read it from the
        # source — that lets _resolve_sequence_settings() distinguish
        # "no fps in this source" from "this source is genuinely 24fps"
        # when picking the timeline rate.
        "video_fps": None,
        # ffprobe color tags, kept raw so we can map per-format later.
        "color_primaries": "unknown",
        "color_transfer": "unknown",
        "color_space": "unknown",
        "pixel_aspect_ratio": "1:1",  # square pixels, the modern default
    }
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_format",
             "-show_streams",
             "-of", "json",
             str(path)],
            capture_output=True, text=True, check=True, timeout=10,
        )
        info = json.loads(proc.stdout or "{}")
        # Format-level duration is the most reliable single number
        # across containers (per-stream duration drifts on VFR sources).
        fmt = info.get("format") or {}
        try:
            dur = float(fmt.get("duration", 0.0))
            if dur > 0:
                meta["duration_s"] = dur
        except (TypeError, ValueError):
            pass

        # Walk streams once — first audio stream wins (NLEs only see
        # one audio shape per file anyway), first video stream wins.
        a_seen, v_seen = False, False
        for s in info.get("streams") or []:
            kind = s.get("codec_type")
            if kind == "audio" and not a_seen:
                a_seen = True
                try:
                    meta["audio_channels"] = int(s.get("channels") or
                                                 _DEFAULT_AUDIO_CHANNELS)
                except (TypeError, ValueError):
                    pass
                try:
                    meta["audio_rate"] = int(s.get("sample_rate") or
                                             _DEFAULT_AUDIO_SAMPLERATE)
                except (TypeError, ValueError):
                    pass
            elif kind == "video" and not v_seen:
                v_seen = True
                try:
                    meta["video_width"] = int(s.get("width") or
                                              _DEFAULT_VIDEO_WIDTH)
                    meta["video_height"] = int(s.get("height") or
                                               _DEFAULT_VIDEO_HEIGHT)
                except (TypeError, ValueError):
                    pass
                # Prefer avg_frame_rate (mean across the file) over
                # r_frame_rate (the encoder's nominal cadence). For VFR
                # sources avg is more honest; for CFR they're identical.
                fps = (_parse_fps(s.get("avg_frame_rate") or "")
                       or _parse_fps(s.get("r_frame_rate") or ""))
                if fps and fps > 0:
                    meta["video_fps"] = fps
                # Color tags — ffprobe sometimes leaves these blank for
                # camera files that didn't write VUI bits; we keep the
                # "unknown" sentinel so the colorspace classifier can
                # fall back to Rec. 709.
                if s.get("color_primaries"):
                    meta["color_primaries"] = str(s["color_primaries"])
                if s.get("color_transfer"):
                    meta["color_transfer"] = str(s["color_transfer"])
                if s.get("color_space"):
                    meta["color_space"] = str(s["color_space"])
                # Pixel aspect ratio: ffprobe gives "1:1", "10:11", etc.
                # We only override the default when ffprobe reports a
                # non-square value — otherwise Resolve sometimes warns
                # about implicit anamorphic interpretation.
                par = s.get("sample_aspect_ratio")
                if par and par not in ("0:1", "1:1", "N/A"):
                    meta["pixel_aspect_ratio"] = str(par)
        meta["has_audio"] = a_seen
        meta["has_video"] = v_seen
        if not a_seen and not v_seen:
            # File had no streams ffprobe could classify; treat as a
            # failure for the warning path below but still write.
            raise ValueError("no audio or video streams found")
    except Exception as e:
        print(
            f"  warn: ffprobe failed for {path.name} ({type(e).__name__}: "
            f"{e}); using stereo-48k/{_DEFAULT_VIDEO_WIDTH}x"
            f"{_DEFAULT_VIDEO_HEIGHT}/{_FFPROBE_FALLBACK_DURATION_S/3600:.0f}h "
            "sentinel asset shape. NLE will relink to the actual source.",
            file=sys.stderr,
        )

    _PROBE_CACHE[key] = meta
    return meta


# ---------------------------------------------------------------------------
# Sequence-shape resolution.
#
# A timeline that doesn't match its source footage trips Premiere's
# "this clip does not match the sequence's settings — change to match?"
# dialog on every clip and (worse) silently downscales high-res sources
# to whatever default the user's NLE happens to have. Worse still, a
# sequence frame rate LOWER than any source's native rate causes
# Premiere to reinterpret the clipitem's <in>/<out>/<duration> frame
# counts (which are always written at the SEQUENCE rate by xmeml
# convention) against the source file's NATIVE rate — the math
# silently expands the source range past the file's actual end and
# Premiere paints diagonal "out of bounds" stripes over the back
# portion of any retimed clip (the (out - in) clip-rate ratio we
# encode for "[N%] Speed/Duration" requires the source range to
# resolve in the source's own frame count to stay in bounds).
#
# To avoid all of the above we take the OVERHEAD across every source
# used in the EDL:
#   • frame rate  -> MAX native fps    (sequence ≥ every source)
#   • resolution  -> MAX pixel area    (sequence ≥ every source)
# Color space / audio rate still inherit from the runtime-dominant
# "primary" source — those are aesthetic picks where "max" makes no
# physical sense, and the dominant clip's settings are the right
# baseline for round-tripped color metadata.
#
# The frame_rate argument from the CLI is honored as an override when
# explicit; "auto" / None means "match the highest source rate".
# ---------------------------------------------------------------------------

def _pick_primary_source(edl: dict) -> str | None:
    """Return the source key whose ranges sum to the largest cut duration.

    Ties are broken by the order of first appearance in the EDL, which
    mirrors how an editor would naturally think of the "primary" clip.
    Returns None for an empty EDL (caller falls back to defaults).
    """
    totals: dict[str, float] = {}
    order: dict[str, int] = {}
    ranges = edl.get("ranges") or edl.get("edl") or []
    for i, r in enumerate(ranges):
        src = r.get("source")
        if not src:
            continue
        try:
            dur = float(r.get("end", 0)) - float(r.get("start", 0))
        except (TypeError, ValueError):
            dur = 0.0
        if dur <= 0:
            continue
        totals[src] = totals.get(src, 0.0) + dur
        order.setdefault(src, i)
    if not totals:
        return None
    # Sort by (-total, first-seen index) so longest wins, ties go to
    # the earliest source — deterministic and obvious in the output.
    return sorted(totals.items(),
                  key=lambda kv: (-kv[1], order[kv[0]]))[0][0]


def _collect_used_sources(edl: dict) -> list[str]:
    """Return the unique source keys that any range in the EDL references.

    Order is "first appearance in the EDL" so probe ordering is stable
    across runs (and so warning lines come out in a predictable order).
    Sources declared in `edl["sources"]` but never referenced by a range
    are deliberately ignored — we don't want unused B-roll bloating the
    sequence-shape decision (and slowing the export with extra ffprobes).
    """
    seen: dict[str, None] = {}
    ranges = edl.get("ranges") or edl.get("edl") or []
    for r in ranges:
        src = r.get("source")
        if src and src not in seen:
            seen[src] = None
    return list(seen.keys())


def _resolve_sequence_settings(edl: dict, frame_rate_arg) -> dict:
    """Collapse the EDL's source set into one sequence-shape dict.

    Returns a dict matching the _probe_source_meta() shape PLUS an
    "fcpxml_color_space" string and a final "video_fps" guaranteed > 0.
    The caller hands this dict to build_timeline / the writers / the
    post-write patchers, all of which need the same numbers to agree.

    Selection rules:
      * video_fps        — MAX native fps across every source the EDL
                           actually uses. A higher-fps sequence cleanly
                           contains lower-fps clips (Premiere conforms
                           them on import); the reverse causes
                           reinterpretation bugs documented in the
                           file-header comment for this section.
      * video_width/height — pulled together from whichever source has
                           the LARGEST pixel area. Using one source's
                           width with another's height would risk a
                           non-physical aspect ratio.
      * color / pixel_aspect / audio_channels / audio_rate
                         — inherited from the runtime-dominant
                           ("primary") source. These are aesthetic /
                           compatibility picks, not strict bounds.

    `frame_rate_arg` semantics:
      - None or "auto"      -> use the MAX-fps result above
      - numeric (float/int) -> hard override; auto-resolved fps ignored
        (warns to stderr if the override is BELOW any source's native
         fps, since that's the exact configuration that breaks time
         remap on import — see file-header comment)
    """
    primary_key = _pick_primary_source(edl)
    sources = edl.get("sources") or {}
    used_keys = _collect_used_sources(edl)

    # Default sequence-shape — matches the _probe defaults so the caller
    # can rely on every key being present even with no sources at all.
    seq = {
        "primary_source": primary_key,
        "video_width": _DEFAULT_VIDEO_WIDTH,
        "video_height": _DEFAULT_VIDEO_HEIGHT,
        "video_fps": _DEFAULT_VIDEO_FPS,
        "audio_channels": _DEFAULT_AUDIO_CHANNELS,
        "audio_rate": _DEFAULT_AUDIO_SAMPLERATE,
        "color_primaries": "unknown",
        "color_transfer": "unknown",
        "color_space": "unknown",
        "pixel_aspect_ratio": "1:1",
        "fcpxml_color_space": _DEFAULT_FCPXML_COLORSPACE,
    }

    # Probe every used source ONCE, cache the metadata so build_timeline
    # doesn't pay for a second probe pass when it laces clips later
    # (_probe_source_meta has its own LRU cache by absolute-path key).
    probed: dict[str, dict] = {}
    for key in used_keys:
        if key not in sources:
            continue
        try:
            probed[key] = _probe_source_meta(Path(sources[key]).resolve())
        except Exception as e:
            print(f"  warn: source probe failed for {key} "
                  f"({type(e).__name__}: {e}); excluding from sequence-shape.",
                  file=sys.stderr)

    # ── Color / audio / pixel-aspect — inherit from the primary source ──
    # These keys are about *interpretation* not bounds, so taking the
    # max across sources makes no semantic sense (max color primaries?).
    # The runtime-dominant clip is the right baseline.
    primary_meta = probed.get(primary_key) if primary_key else None
    if primary_meta:
        for k in ("audio_channels", "audio_rate", "color_primaries",
                  "color_transfer", "color_space", "pixel_aspect_ratio"):
            if primary_meta.get(k) is not None:
                seq[k] = primary_meta[k]

    # ── Resolution — MAX pixel area across all used sources ──
    # We pick width AND height from the same winning source to preserve
    # its aspect ratio rather than mix-and-match (which could end up
    # 3840x1080 or other Frankenstein shapes). Ties resolve to the
    # earliest-appearing source, matching primary-source tie semantics.
    best_area = -1
    best_key = None
    for key in used_keys:
        meta = probed.get(key)
        if not meta:
            continue
        w, h = meta.get("video_width"), meta.get("video_height")
        if not w or not h:
            continue
        area = int(w) * int(h)
        if area > best_area:
            best_area = area
            best_key = key
    if best_key is not None:
        seq["video_width"] = probed[best_key]["video_width"]
        seq["video_height"] = probed[best_key]["video_height"]

    # ── Frame rate — MAX native fps across all used sources ──
    # Sources without a usable fps probe (live recordings, image
    # sequences, etc.) are skipped; we never let a missing probe drag
    # the sequence rate DOWN to the default.
    #
    # We snap each probed rate onto OTIO's canonical FCPXML key set
    # BEFORE taking the max — ffprobe returns NTSC rates as exact
    # rationals (60000/1001 = 59.94005994...) but OTIO's lookup uses
    # exact-equality against {59.94, 60, 29.97, ...} and silently
    # writes an empty frameDuration on any miss, which crashes the
    # adapter on read-back. Snapping at probe time keeps the rest of
    # the pipeline (timeline rate, frame snapping) on the canonical
    # value too, so all downstream math stays internally consistent.
    max_fps = 0.0
    for meta in probed.values():
        fps = meta.get("video_fps") or 0.0
        if fps:
            fps = _snap_fps_for_fcpxml(float(fps))
            if fps > max_fps:
                max_fps = fps
    if max_fps > 0:
        seq["video_fps"] = max_fps

    # ── CLI override ──
    # "auto" / None means "use what we just resolved" (the MAX above).
    # Any positive numeric value wins, but we LOUDLY warn if it's lower
    # than a source's native rate — that's the exact configuration that
    # breaks time-remapped clipitems on import to Premiere (the bug
    # users hit before this auto-pick logic was tightened).
    if frame_rate_arg is not None and frame_rate_arg != "auto":
        try:
            user_fps = float(frame_rate_arg)
            if user_fps > 0:
                # Snap the override too — same reasoning as the auto
                # path above. A user typing "--frame-rate=59.94" should
                # get the canonical key, not float(59.94)=59.93999...
                # which still misses OTIO's exact-equality lookup.
                user_fps = _snap_fps_for_fcpxml(user_fps)
                if max_fps and user_fps + 1e-3 < max_fps:
                    print(f"  warn: --frame-rate={user_fps} is BELOW the "
                          f"highest source fps ({max_fps:g}); time-remapped "
                          f"clips may render with diagonal 'out of bounds' "
                          f"stripes in Premiere. Pass --frame-rate=auto "
                          f"(or omit) to inherit the source rate.",
                          file=sys.stderr)
                seq["video_fps"] = user_fps
        except (TypeError, ValueError):
            pass

    seq["fcpxml_color_space"] = _classify_colorspace(
        seq["color_transfer"], seq["color_primaries"]
    )
    return seq


# Backward-compat shim for older callers (and tests) that only need the
# duration. New code should prefer `_probe_source_meta()` directly.
def _probe_source_duration_s(path: Path) -> float:
    """Return only the source duration in seconds — convenience wrapper."""
    return _probe_source_meta(path)["duration_s"]


# ---------------------------------------------------------------------------
# File-URL construction. Premiere Pro mis-parses RFC 8089 `file:///A:/...`
# Windows URLs (the three-slash form) — it interprets the leading slash run
# as a UNC share prefix and the imported clip ends up showing as
# `\\\A:\YouTube\...` (three backslashes, can't relink).
#
# The `file://localhost/A:/...` form (explicit `localhost` host) is the
# unambiguous variant Adobe's importer parses correctly. POSIX paths are
# fine either way; we only special-case Windows drive-letter paths.
#
# Path.as_uri() alone is wrong for our use; this helper supersedes it for
# every URL we hand to the OTIO adapters.
# ---------------------------------------------------------------------------

# Match Windows drive-letter paths in any flavor (forward or back slashes).
_WIN_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/]")


def _safe_file_url(path: Path) -> str:
    """Return a file:// URL that Premiere AND Resolve AND FCP X parse correctly.

    - Windows drive paths       -> file://localhost/A:/path/to/file.mp4
    - Windows UNC paths         -> file://server/share/path/to/file.mp4
    - POSIX absolute paths      -> file:///abs/path/to/file.mp4
    - Anything else             -> Path.as_uri() (RFC 8089 default)

    The localhost form is documented by Adobe for Premiere imports and
    round-trips cleanly through XtoCC, FCP 7, FCP X, and Resolve.
    """
    s = str(path)
    # UNC: "\\server\share\..." — keep server as the host segment.
    if s.startswith("\\\\") or s.startswith("//"):
        # Strip leading slashes and split server/share/rest.
        rest = s.lstrip("\\/").replace("\\", "/")
        # No percent-encoding here — NLEs are forgiving and our paths
        # come from the EDL which was written by us / the user; spaces
        # are the only realistic concern, and NLEs handle raw spaces
        # in pathurl/src attributes fine in practice.
        return f"file://{rest}"

    # Windows drive letter: "A:\foo\bar" or "A:/foo/bar".
    m = _WIN_DRIVE_RE.match(s)
    if m:
        # Normalize to forward slashes; the localhost host removes any
        # ambiguity about how many leading slashes follow `file:`.
        normalized = s.replace("\\", "/")
        return f"file://localhost/{normalized}"

    # POSIX / fallback — Path.as_uri() does the right thing here.
    try:
        return path.as_uri()
    except ValueError:
        # as_uri() rejects relative paths; resolve and retry.
        return path.resolve().as_uri()


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
# Time-squeezing (timelapse) — speed handling.
#
# The EDL allows each range to set `speed` (float, default 1.0). speed > 1
# compresses the source segment on the output timeline (8.0 = 8x timelapse).
#
# We CLAMP to [1.0, MAX_SPEED]. The cap reflects two practical realities:
#   * Both NLE retime UIs (Premiere "Speed/Duration", FCP X "Custom Speed")
#     express speed as a percentage and start showing visible decimation
#     artifacts above ~1000% on standard 24/30fps source. Beyond that the
#     editor really wants frame-blending or optical flow, which is best
#     left to the colourist post-import — we ship the clean retime.
#   * Anything faster than 10x is almost always a sign the agent should
#     have CUT the boring stretch instead of squeezing it. The cap nudges
#     the editor toward the right tool.
# Sub-1.0x slow-mo is intentionally NOT supported here — it would expand
# the timeline and turn 30s of source into N minutes, which fights every
# downstream assumption about pacing and SRT placement. If the user asks
# for it, decline and explain why.
# ---------------------------------------------------------------------------

# Hard upper bound on retime — see module-comment rationale above.
MAX_SPEED = 10.0
# Floor at 1.0 — slow-mo expansion is a separate feature with its own
# cost model (frame interpolation, audio drop-or-pitch, output-timeline
# explosion). Until we design that, anything < 1.0 is treated as 1.0.
MIN_SPEED = 1.0


def _read_speed(r: dict, range_idx: int) -> float:
    """Pull and validate the per-range `speed` field.

    Returns a float in [MIN_SPEED, MAX_SPEED]. Missing / null / non-numeric
    values resolve to 1.0. Out-of-range values are clamped with a warning
    so a typo can't silently produce a 9-frame "timelapse" of a 4-hour
    source (which would happen at speed=1000) or stretch the cut.
    """
    raw = r.get("speed")
    if raw is None:
        return 1.0
    try:
        spd = float(raw)
    except (TypeError, ValueError):
        print(f"  warn: range[{range_idx}] has non-numeric speed={raw!r}; "
              "treating as 1.0 (no time-remap).", file=sys.stderr)
        return 1.0
    if spd < MIN_SPEED:
        print(f"  warn: range[{range_idx}] speed={spd:g} < {MIN_SPEED:g} "
              "(slow-mo not supported by this exporter); clamped to 1.0.",
              file=sys.stderr)
        return 1.0
    if spd > MAX_SPEED:
        print(f"  warn: range[{range_idx}] speed={spd:g} > {MAX_SPEED:g} "
              f"(beyond the {int(MAX_SPEED * 100)}% retime ceiling); "
              f"clamped to {MAX_SPEED:g}. Consider CUTTING the range "
              "instead of squeezing it harder.", file=sys.stderr)
        return MAX_SPEED
    return spd


def _read_audio_strategy(r: dict, speed: float, range_idx: int) -> str:
    """Pick the audio behaviour for a range.

    Defaults:
      - speed == 1.0  → "keep" (audio plays normally — the standard cut)
      - speed != 1.0  → "drop" (silent gap; timelapses almost always want
                        the raw shop noise replaced by music in the NLE)
    Honors an explicit `audio_strategy` if the EDL sets one, with a soft
    warning when the value is unrecognised so we don't silently mis-handle
    the audio track on a typo.
    """
    raw = r.get("audio_strategy")
    if raw is None:
        return "keep" if abs(speed - 1.0) < 1e-9 else "drop"
    s = str(raw).strip().lower()
    if s in ("drop", "keep"):
        return s
    print(f"  warn: range[{range_idx}] has unknown audio_strategy={raw!r}; "
          "valid: 'drop' | 'keep'. Falling back to "
          f"{'keep' if abs(speed - 1.0) < 1e-9 else 'drop'}.",
          file=sys.stderr)
    return "keep" if abs(speed - 1.0) < 1e-9 else "drop"


# ---------------------------------------------------------------------------
# Core build — produces an OTIO Timeline with two tracks (V1 + A1).
#
# Why we build two parallel tracks instead of relying on a single video
# track with attached audio: split edits (J/L cuts) require independent
# clip extents per track. OTIO's clip-with-attached-audio model would
# force matching extents, defeating the whole point.
# ---------------------------------------------------------------------------

def build_timeline(edl: dict, frame_rate: float, sequence_settings: dict | None = None):
    """Build and return an otio.schema.Timeline from the EDL.

    Schema fields honored:
      - sources           : map source_id → absolute path
      - ranges[]          : the cut list (also accepts top-level "edl")
        - source          : key into sources
        - start, end      : float seconds in the SOURCE clip
        - audio_lead      : J-cut offset (seconds, optional, default 0)
        - video_tail      : L-cut offset (seconds, optional, default 0)
        - transition_in   : dissolve seconds before this clip (optional)
        - speed           : timelapse multiplier (float, default 1.0,
                            clamped to [1.0, 10.0]); >1 compresses the
                            source segment on the output timeline
        - audio_strategy  : "drop" (silent gap) or "keep" (audio retimed
                            with video). Default "keep" at speed=1.0,
                            "drop" at speed!=1.0.
        - beat / quote    : copied to clip metadata for editor reference

    `sequence_settings` (optional) carries the resolved sequence-shape
    dict from _resolve_sequence_settings() — width/height/fps/color
    space/audio rate. We stash it on the Timeline's metadata so the
    post-write patchers can find it without re-probing. When None we
    skip the stash; the patchers will fall back to per-asset probing.

    Returns the otio.schema.Timeline. The retime side-channel needed by
    the post-write XML patchers is stashed on
    `timeline.metadata['premiere-agent']['speed_map']` — see
    `_speed_map_from_timeline()` for the read side.
    """
    otio = _import_otio()

    sources = edl.get("sources") or {}
    ranges = edl.get("ranges") or edl.get("edl") or []
    if not ranges:
        raise ValueError("EDL has no ranges")

    # Sequence name defaults to "VU_Sequence" — keeps Premiere's
    # project window tidy ("VU_Sequence" beside a "Footage" bin
    # courtesy of the post-write _patch_xmeml_bin_layout pass) and
    # matches the layout convention documented in SKILL.md. The EDL
    # may override with its own `name` if the editor has a preference.
    timeline = otio.schema.Timeline(name=edl.get("name") or "VU_Sequence")
    # Timeline rate sets the granularity of TimeRanges in the file. NLEs
    # auto-convert on import but giving them the same rate the user
    # eventually delivers at avoids any subtle re-quantization.
    timeline.global_start_time = otio.opentime.RationalTime(0, frame_rate)

    # Stash the resolved sequence-shape dict on the timeline so the
    # post-write patchers (_patch_fcpxml_sequence_format /
    # _patch_xmeml_sequence_format) can read width/height/colorSpace/
    # audio shape without re-resolving everything from the EDL.
    if sequence_settings:
        timeline.metadata["premiere-agent"] = {
            "sequence": dict(sequence_settings),
        }

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

    # ── Retime side-channel ───────────────────────────────────────────
    # Both XML adapters silently DROP otio.schema.LinearTimeWarp
    # (verified against opentimelineio 0.18.x + otio-fcpx-xml-adapter
    # 0.2.x + otio-fcp-adapter 0.2.x — no <timeMap> in fcpxml, no
    # timeremap <filter> in xmeml). So we record every retime in this
    # side-channel and inject the dialect-correct retime element in a
    # post-write patch (see _patch_fcpxml_speed / _patch_xmeml_speed).
    # Keyed by the unique clip name we mint below.
    speed_map: dict[str, dict] = {}

    # ── Optional timestretch trace points (no-op unless _ts_log is wired) ─
    _ts_log_section("build_timeline")
    _ts_log(f"  sequence_fps        : {frame_rate}")
    if sequence_settings:
        _ts_log(
            f"  sequence_resolution : "
            f"{sequence_settings.get('video_width')}x"
            f"{sequence_settings.get('video_height')}"
        )
        _ts_log(
            f"  sequence_audio      : "
            f"{sequence_settings.get('audio_channels')}ch @ "
            f"{sequence_settings.get('audio_rate')}Hz"
        )
        _ts_log(
            f"  primary_source      : "
            f"{sequence_settings.get('primary_source')}"
        )
    _ts_log(f"  total_ranges        : {len(ranges)}")

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

        # ── Time-squeezing (timelapse) ────────────────────────────────
        # Per-range speed multiplier, clamped to [1.0, 10.0]. Defaults
        # to 1.0 (no retime) so untouched EDLs behave exactly as before.
        speed = _read_speed(r, i)
        audio_strategy = _read_audio_strategy(r, speed, i)

        # Refuse to combine retime with split edits / dissolves — the
        # timeline math (audio offsets vs. retimed video) is ambiguous
        # and we don't want to silently produce drift on a 30-clip cut.
        # Because Hard Rule 14 forces these fields to 0 in the editor
        # sub-agent's output today, this is mostly defensive against a
        # hand-written EDL that uses both at once.
        if speed != 1.0 and (a_lead != 0.0 or v_tail != 0.0 or trans_in != 0.0):
            print(f"  warn: range[{i}] combines speed={speed:g} with "
                  f"audio_lead/video_tail/transition_in; the split-edit "
                  "fields are IGNORED on retimed ranges (undefined math). "
                  "Hard-cut + retime only.", file=sys.stderr)
            a_lead = v_tail = trans_in = 0.0

        # ── Source-bounds clamp (BEFORE frame snapping & OTIO build) ─
        # We probe the source media here (before snapping start/end to
        # frame boundaries) so we can clamp the EDL range against the
        # file's physical end. EDL agents routinely write ranges where
        # `end` lands past the source's last decodable frame — they
        # work from rough timestamps, not precise media bookkeeping —
        # and Premiere's renderer paints the diagonal "media offline"
        # stripe over any timeline portion that decodes to a missing
        # frame. The post-write patcher has its own clamp as belt-and-
        # braces, but clamping HERE means OTIO sees in-bounds numbers
        # too, the asset's <file><duration> declaration matches what
        # the clipitem actually consumes, and the (in + src_dur) math
        # downstream stays internally consistent across both writers.
        #
        # SAFETY MARGIN — we trim 6 frames (~100ms at 60fps) from the
        # file's reported duration before clamping. This absorbs:
        #   • ffprobe vs decoder duration disagreements (container
        #     duration is sometimes rounded up past the last decodable
        #     frame, especially on DJI / GoPro action-cam MP4s)
        #   • Premiere's internal probe rounding (it sometimes sees
        #     fewer frames than ffprobe reports)
        #   • B-frame reordering tail latency on long-GOP codecs
        # The cost of a too-conservative trim is ~100ms of unused
        # source tail; the cost of a too-aggressive trim is the
        # diagonal stripe pattern. Easy trade-off.
        #
        # _probe_source_meta is LRU-cached by absolute path so this is
        # a free lookup if the source has been seen before this loop
        # iteration (which it almost always has, courtesy of
        # _resolve_sequence_settings probing every used source up
        # front to pick the sequence shape).
        src_path_resolved = Path(src_path).resolve()
        src_meta = _probe_source_meta(src_path_resolved)
        media_dur_s = src_meta["duration_s"]
        src_fps = src_meta.get("video_fps") or 0.0
        # Snapshot the EDL's intent BEFORE we mutate start/end so the
        # log can show the delta between "what the editor agent asked
        # for" and "what we actually wrote". Only emit the per-range
        # block at all when the range carries a non-trivial speed
        # (1.0× ranges are the boring path and would 100x the log).
        edl_start = start
        edl_end = end
        if speed != 1.0:
            _ts_log_section(
                f"range[{i}] {src_name}  (speed={speed:g})"
            )
            _ts_log(
                f"  edl              : start={edl_start:.6f}s "
                f"end={edl_end:.6f}s dur={edl_end - edl_start:.6f}s"
            )
            _ts_log(
                f"  edl_split_edit   : a_lead={a_lead:.6f}s "
                f"v_tail={v_tail:.6f}s trans_in={trans_in:.6f}s"
            )
            _ts_log(
                f"  source           : path={src_path_resolved}"
            )
            # media_dur_s can be None on probe failure — print as
            # "None" rather than letting the format spec crash. Same
            # for src_fps which defaults to 0.0 above when missing.
            _md_str = (
                f"{float(media_dur_s):.6f}"
                if media_dur_s is not None else "None"
            )
            _ts_log(
                f"  source_probe     : media_dur_s={_md_str}s "
                f"src_fps={float(src_fps):g} "
                f"video={src_meta.get('video_width')}x"
                f"{src_meta.get('video_height')} "
                f"audio={src_meta.get('audio_channels')}ch@"
                f"{src_meta.get('audio_rate')}Hz"
            )
        if media_dur_s and media_dur_s > 0:
            safety_s = 6.0 / float(frame_rate)
            max_end = max(0.0, media_dur_s - safety_s)
            # The widest source-side window we ever need is the union
            # of the video range AND the audio split-edit extension
            # (start - a_lead, end + v_tail). Clamp that union, then
            # back out start/end so all downstream math stays within
            # the file. We DON'T proportionally scale a_lead/v_tail —
            # those got zeroed out above for retimed clips, and on
            # 1.0× clips the EDL agent is responsible for keeping
            # them inside the source.
            need_end = end + v_tail
            need_start = start - a_lead
            clamp_fired = (need_end > max_end or need_start < 0.0)
            if clamp_fired:
                print(
                    f"  warn: range[{i}] {src_name} source window "
                    f"[{need_start:.3f}s, {need_end:.3f}s] exceeds "
                    f"media bounds [0, {max_end:.3f}s] "
                    f"(file is {media_dur_s:.3f}s, 6-frame safety "
                    f"margin applied) — clamping. Tighten the EDL "
                    f"range to silence this warning.",
                    file=sys.stderr,
                )
            if need_end > max_end:
                end = max(start + (1.0 / float(frame_rate)),
                          max_end - max(0.0, v_tail))
            if need_start < 0.0:
                start = max(0.0, a_lead)
            if speed != 1.0:
                _ts_log(
                    f"  bounds_clamp     : safety_s={safety_s:.6f}s "
                    f"max_end={max_end:.6f}s "
                    f"need_window=[{need_start:.6f}, {need_end:.6f}]s "
                    f"fired={clamp_fired} "
                    f"post_clamp_start={start:.6f}s "
                    f"post_clamp_end={end:.6f}s "
                    f"post_clamp_dur={end - start:.6f}s"
                )
            # If clamping collapsed the range to ≤0, drop it entirely.
            # Better than emitting a sub-frame stub the NLE will
            # complain about on import.
            if end - start <= 0:
                print(
                    f"  skip range[{i}] {src_name}: clamp collapsed "
                    f"the range to zero duration — likely the entire "
                    f"requested range falls past the source's end.",
                    file=sys.stderr,
                )
                continue

        # Snap everything to whole frames so NLE imports are clean.
        v_start = _snap_to_frame(start, frame_rate)
        v_end_pre_floor = _snap_to_frame(end, frame_rate)
        v_end = v_end_pre_floor
        # Belt-and-braces: _snap_to_frame() rounds to the NEAREST frame,
        # which can push v_end past the safety boundary even after the
        # clamp above. Floor it to the largest whole-frame timestamp
        # ≤ (media_dur_s - safety_s) so the snap can never round us
        # back over the cliff.
        snap_floor_fired = False
        if media_dur_s and media_dur_s > 0:
            safety_s = 6.0 / float(frame_rate)
            max_end_floor = (
                int((media_dur_s - safety_s) * frame_rate)
                / float(frame_rate)
            )
            if v_end > max_end_floor:
                snap_floor_fired = True
                v_end = max(v_start + 1.0 / float(frame_rate),
                            max_end_floor)
        v_src_dur = max(0.0, v_end - v_start)
        if speed != 1.0:
            _ts_log(
                f"  snap_to_frame    : v_start={v_start:.6f}s "
                f"v_end_pre_floor={v_end_pre_floor:.6f}s "
                f"v_end={v_end:.6f}s "
                f"snap_floor_fired={snap_floor_fired} "
                f"v_src_dur={v_src_dur:.6f}s "
                f"(at seq fps {frame_rate}: "
                f"v_start_f={int(round(v_start * frame_rate))}f "
                f"v_end_f={int(round(v_end * frame_rate))}f "
                f"v_src_dur_f={int(round(v_src_dur * frame_rate))}f)"
            )

        # Output-timeline duration of this clip after retime. At
        # speed=1.0 this is identical to the source duration; at
        # speed=8.0 a 40-second source range lands as 5 timeline
        # seconds. We snap the output duration to a whole frame so the
        # NLE's frame-aligned import path stays drift-free.
        v_out_dur = _snap_to_frame(v_src_dur / speed, frame_rate)
        if v_out_dur <= 0.0:
            # A range so short it disappears at the chosen speed (e.g.
            # 80ms at 10x = 8ms < 1 frame at 24fps). Better to drop
            # than to write an empty clip.
            print(f"  skip range[{i}] {src_name}: speed={speed:g} would "
                  f"produce {v_out_dur*1000:.1f}ms output; below frame.",
                  file=sys.stderr)
            if speed != 1.0:
                _ts_log(
                    f"  SKIP             : v_out_dur collapsed to "
                    f"{v_out_dur:.6f}s — below 1 frame at "
                    f"speed={speed:g}"
                )
            continue
        v_dur = v_out_dur  # what the timeline / OTIO layout sees
        if speed != 1.0:
            # Effective speed = v_src_dur / v_out_dur. After snapping
            # both to whole frames this is usually a hair off the
            # requested speed (e.g. requested 8.0× landing as 7.97×).
            # The drift is intentional — frame-quantization is
            # mandatory for clean NLE imports — but it's worth logging
            # so a future "why does my 8x clip play at 7.97x" question
            # has an immediate answer.
            eff_speed = (v_src_dur / v_out_dur) if v_out_dur > 0 else 0.0
            _ts_log(
                f"  output_dur       : v_out_dur={v_out_dur:.6f}s "
                f"(v_src_dur/{speed:g} snapped) "
                f"effective_speed={eff_speed:g}× "
                f"(requested {speed:g}×, drift "
                f"{(eff_speed - speed):+.4f}×) "
                f"v_out_dur_f={int(round(v_out_dur * frame_rate))}f"
            )

        # Audio source range is independently snapped — could be different
        # from the video range by ±half a frame after rounding. Same
        # max_end_floor logic as v_end above so a J-cut tail can't
        # sneak past the safety boundary either. (For retimed clips
        # a_lead / v_tail were already zeroed; this matters for 1.0×
        # clips with split edits.)
        a_src_start = max(0.0, _snap_to_frame(start - a_lead, frame_rate))
        a_src_end_pre_floor = _snap_to_frame(end + v_tail, frame_rate)
        a_src_end = a_src_end_pre_floor
        a_snap_floor_fired = False
        if media_dur_s and media_dur_s > 0:
            safety_s = 6.0 / float(frame_rate)
            max_end_floor = (
                int((media_dur_s - safety_s) * frame_rate)
                / float(frame_rate)
            )
            if a_src_end > max_end_floor:
                a_snap_floor_fired = True
                a_src_end = max(a_src_start + 1.0 / float(frame_rate),
                                max_end_floor)
        if speed != 1.0:
            _ts_log(
                f"  audio_range      : a_src_start={a_src_start:.6f}s "
                f"a_src_end_pre_floor={a_src_end_pre_floor:.6f}s "
                f"a_src_end={a_src_end:.6f}s "
                f"a_snap_floor_fired={a_snap_floor_fired}"
            )
        # When retime is on, the audio's TIMELINE duration is the
        # video's output duration regardless of audio_strategy; the
        # patcher injects the matching audio retime when "keep" is
        # selected, or we emit a silent Gap when "drop" is selected.
        if speed != 1.0:
            a_dur = v_out_dur
        else:
            a_dur = max(0.0, a_src_end - a_src_start)

        # ── External media reference (one per clip — file path only) ──
        # OTIO's ExternalReference resolves to file:// URIs at write time.
        # NLEs follow the path on import; if the user moves the masters,
        # they'll get the standard "missing media" relink dialog, same as
        # for any imported XML.
        #
        # available_range is REQUIRED by both adapters — it writes into
        # the asset declaration. `src_path_resolved` / `src_meta` /
        # `media_dur_s` were already populated by the bounds-clamp
        # block above (and _probe_source_meta is LRU-cached anyway).
        #
        # The audio shape (channel count + sample rate) matters: Premiere
        # validates "the clip was created with N audio channel(s)" against
        # the actual file on link, and refuses if they disagree. We pre-
        # populate the fcp_xml metadata so the FCP7 xmeml writer emits
        # <channelcount>/<samplerate> matching the real source.
        media_avail_range = otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, frame_rate),
            duration=_rt(media_dur_s, frame_rate),
        )

        # Use a Premiere-safe file URL (file://localhost/... on Windows)
        # in place of Path.as_uri()'s file:///A:/... which Premiere
        # mis-parses as a UNC path and shows back as \\\A:\... .
        safe_url = _safe_file_url(src_path_resolved)

        # Build the per-source fcp_xml media descriptor ONCE — both the
        # video and audio clip refs share the same source file, so they
        # share the same shape. The dict layout mirrors the FCP7 xmeml
        # element tree exactly (see otio_fcp_adapter._dict_to_xml_tree).
        fcp_xml_media: dict = {}
        if src_meta["has_video"]:
            fcp_xml_media["video"] = {
                "samplecharacteristics": {
                    "width": src_meta["video_width"],
                    "height": src_meta["video_height"],
                },
            }
        if src_meta["has_audio"]:
            fcp_xml_media["audio"] = {
                "channelcount": src_meta["audio_channels"],
                "samplecharacteristics": {
                    "depth": 16,
                    "samplerate": src_meta["audio_rate"],
                },
            }

        def _new_media_ref():
            """Each clip gets its own ExternalReference because the FCPXML
            adapter sometimes writes per-asset state; sharing one ref
            across clips has caused 'asset already declared' errors in
            past adapter versions. Cheap enough to construct per-clip."""
            ref = otio.schema.ExternalReference(
                target_url=safe_url,
                available_range=media_avail_range,
            )
            # The fcp_xml adapter reads media_reference.metadata['fcp_xml']
            # and serializes it verbatim into the <file> element. Setting
            # the `media` subtree pre-empts the adapter's empty-defaults
            # fallback (no channelcount, no samplerate) which is what
            # triggers Premiere's "Cannot Link Media" mismatch error.
            if fcp_xml_media:
                ref.metadata["fcp_xml"] = {"media": dict(fcp_xml_media)}
            # Also stash on the canonical schema fields for any adapter
            # that introspects them — and pass through to fcpx_xml's
            # post-write step (see _patch_fcpxml_audio_shape).
            ref.metadata["premiere-agent"] = {
                "audio_channels": src_meta["audio_channels"],
                "audio_rate": src_meta["audio_rate"],
                "has_audio": src_meta["has_audio"],
                "has_video": src_meta["has_video"],
            }
            return ref

        # Video clip — source_range carries the SOURCE in-point (v_start)
        # but the OUTPUT duration (v_out_dur). At speed=1.0 these match
        # the source duration, so we lay out exactly as before; at
        # speed>1.0 OTIO sees the compressed timeline length and offsets
        # subsequent clips correctly. The patcher injects the actual
        # retime element so the NLE plays the FULL source range
        # (v_src_dur) inside that compressed window.
        #
        # Clip name uses the SOURCE FILE BASENAME (e.g. "DJI_2025…_0137_D")
        # rather than the EDL's shorthand source key (e.g. "C0137"), so
        # Premiere's project-bin / timeline labels match what the editor
        # sees on disk. Suffix `_v_NN` keeps each cut individually
        # addressable for the side-channel speed_map lookup.
        src_file_stem = Path(src_path).stem
        v_clip_name = f"{src_file_stem}_v_{i:02d}"
        v_clip = otio.schema.Clip(
            name=v_clip_name,
            media_reference=_new_media_ref(),
            source_range=_range(v_start, v_out_dur, frame_rate),
        )
        # Stash editorial metadata so the user can see WHY this cut was
        # chosen when they hover the clip in the NLE's clip inspector.
        v_clip.metadata["premiere-agent"] = {
            "beat": r.get("beat"),
            "quote": r.get("quote"),
            "reason": r.get("reason"),
            "speed": speed,
        }
        # Belt-and-braces: also attach OTIO's LinearTimeWarp effect for
        # any future adapter version that DOES honor it. Today both
        # adapters drop it silently — the post-write patcher is what
        # actually carries the retime through to the .fcpxml / .xml.
        if speed != 1.0:
            v_clip.effects.append(otio.schema.LinearTimeWarp(
                name=f"timelapse_{i:02d}", time_scalar=float(speed),
            ))
            speed_map[v_clip_name] = {
                "kind": "video",
                "speed": float(speed),
                "src_start_s": v_start,
                "src_dur_s": v_src_dur,
                "out_dur_s": v_out_dur,
                # Full source-FILE duration. NOT written into the
                # clipitem's <duration> (that field is per the xmeml
                # spec the source-range frame count, == out - in,
                # not the file-wide count — bin masters own the
                # file-wide value). We carry it here purely so the
                # patcher's bounds clamp can verify (in + src_dur)
                # never reads past the file's last decodable frame.
                "media_dur_s": float(media_dur_s),
                "frame_rate": float(frame_rate),
            }
            # Pre-compute the patcher's intended XML values right here
            # (at sequence rate) so the log captures the math BOTH
            # writers will perform downstream. Anything weird shows up
            # as a delta between this line and the patcher's later
            # "patch_xmeml" / "patch_fcpxml" lines for the same clip.
            in_f = int(round(v_start * frame_rate))
            src_dur_f = int(round(v_src_dur * frame_rate))
            media_dur_f = int(round(media_dur_s * frame_rate))
            max_src_dur_f = max(1, media_dur_f - in_f - 6)
            patch_clamp_will_fire = src_dur_f > max_src_dur_f
            final_src_dur_f = min(src_dur_f, max_src_dur_f)
            _ts_log(
                f"  speed_map[v]     : name={v_clip_name} "
                f"clip_id_v={i:02d}"
            )
            _ts_log(
                f"  patcher_preview  : in_f={in_f} src_dur_f={src_dur_f} "
                f"media_dur_f={media_dur_f} "
                f"max_src_dur_f={max_src_dur_f} "
                f"will_clamp={patch_clamp_will_fire} "
                f"target_out_f={in_f + final_src_dur_f} "
                f"target_dur_f={final_src_dur_f}  "
                f"(invariant: out - in == duration == "
                f"{final_src_dur_f}f)"
            )

        # Audio handling — three paths:
        #   1. speed == 1.0  → standard A1 clip (existing behaviour).
        #   2. speed != 1.0 + audio_strategy=="drop" → silent Gap on A1
        #      of the same OUTPUT duration as the video. This is the
        #      sane default for timelapses (raw shop noise sped up
        #      sounds awful; the editor will replace it with music).
        #   3. speed != 1.0 + audio_strategy=="keep" → A1 clip with the
        #      SAME shape as the video, marked in the side-channel so
        #      the patcher applies the matching audio retime. Will
        #      sound chipmunk-y in the NLE unless the editor toggles
        #      "Maintain Audio Pitch" (Premiere) / "Preserve Pitch"
        #      (FCP X), which is a one-click property on the clip.
        a_clip = None
        a_clip_name = f"{src_file_stem}_a_{i:02d}"
        if speed == 1.0:
            a_clip = otio.schema.Clip(
                name=a_clip_name,
                media_reference=_new_media_ref(),
                source_range=_range(a_src_start, a_dur, frame_rate),
            )
        elif audio_strategy == "keep":
            a_clip = otio.schema.Clip(
                name=a_clip_name,
                media_reference=_new_media_ref(),
                # Same OUTPUT duration as the video; patcher rewrites
                # in/out (xmeml) and timeMap (fcpxml) to consume the
                # full source range under that timeline window.
                source_range=_range(v_start, v_out_dur, frame_rate),
            )
            a_clip.effects.append(otio.schema.LinearTimeWarp(
                name=f"timelapse_a_{i:02d}", time_scalar=float(speed),
            ))
            speed_map[a_clip_name] = {
                "kind": "audio",
                "speed": float(speed),
                "src_start_s": v_start,
                "src_dur_s": v_src_dur,
                "out_dur_s": v_out_dur,
                # See video branch above — full source-FILE duration so
                # the xmeml <duration> patch can match Premiere's spec.
                "media_dur_s": float(media_dur_s),
                "frame_rate": float(frame_rate),
            }
        # else: drop branch — fall through and append a Gap below.

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
        if a_clip is not None:
            a_track.append(a_clip)
        else:
            # audio_strategy=="drop" path on a retimed range — silent
            # filler covering the same OUTPUT span as the video clip.
            a_track.append(otio.schema.Gap(
                name=f"silence_{i:02d}",
                source_range=_range(0.0, v_out_dur, frame_rate),
            ))

        prev_cur_v = cur_v
        prev_cur_a = cur_a
        cur_v += v_dur
        cur_a = target_a_start + a_dur  # inherits both lead AND tail
        if speed != 1.0:
            _ts_log(
                f"  timeline_place   : "
                f"v_track_in={prev_cur_v:.6f}s "
                f"v_track_out={cur_v:.6f}s "
                f"v_track_span={cur_v - prev_cur_v:.6f}s "
                f"a_track_in={prev_cur_a:.6f}s "
                f"a_track_target_in={target_a_start:.6f}s "
                f"a_track_out={cur_a:.6f}s "
                f"a_gap_inserted={a_gap if a_gap > 1e-6 else 0.0:.6f}s"
            )

    # Stash the speed map on the timeline so the post-write patchers can
    # find it. Existing 'premiere-agent' bucket is preserved (it
    # already carries 'sequence' from the sequence-settings stash).
    if speed_map:
        bucket = timeline.metadata.setdefault("premiere-agent", {})
        bucket["speed_map"] = dict(speed_map)

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


# ---------------------------------------------------------------------------
# FCPXML 1.10+ post-write patch.
#
# The otio-fcpx-xml-adapter hardcodes `hasAudio="0"` / `hasVideo="0"` on
# every asset and never emits `audioSources` / `audioChannels` / `audioRate`
# attributes. Premiere Pro and DaVinci Resolve both reject the file with
# "Cannot Link Media" or silently skip the audio track when those are
# absent or wrong.
#
# We don't fork the adapter; we patch the file in place after write,
# walking every <asset src="..."> and looking the source's true shape
# back up via the same _probe_source_meta() cache that fed build_timeline.
# That keeps the patch authoritative (always matches the actual file) and
# adapter-version-agnostic (works regardless of how OTIO upgrades).
# ---------------------------------------------------------------------------

def _path_from_safe_url(url: str) -> Path | None:
    """Inverse of _safe_file_url — recover the local Path for cache lookup.

    Handles file://localhost/A:/..., file:///A:/..., file:///abs/...,
    and bare paths. Returns None when nothing usable can be parsed.
    """
    if not url:
        return None
    s = url
    # Strip the scheme + host segment in any of the documented forms.
    for prefix in ("file://localhost/", "file:///", "file://"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    # On Windows a leading / before the drive letter is a leftover from
    # the file:// stripping and must go (`/A:/foo` -> `A:/foo`).
    if len(s) >= 3 and s[0] == "/" and s[2] == ":":
        s = s[1:]
    try:
        return Path(s).resolve()
    except Exception:
        return None


def _read_sequence_meta_from_timeline_xml(root) -> dict | None:
    """Extract our stashed sequence-settings dict from the timeline XML.

    OTIO serializes Timeline.metadata under <metadata><k>...</k></metadata>
    in FCPXML, and as adapter-specific element trees in xmeml. We don't
    actually rely on that round-trip here — instead the writers receive
    the dict in-process and pass it to the patchers directly. This
    function exists for symmetry / standalone re-patch invocations and
    currently always returns None; future work could parse it out.
    """
    return None


def _patch_fcpxml_audio_shape(out_path: Path, sequence_meta: dict | None = None) -> int:
    """Rewrite asset + format attributes in a freshly-written .fcpxml so:

    1. Each <asset> declares the audio / video shape that matches its
       real source (hasAudio, audioSources, audioChannels, audioRate),
       so Premiere and Resolve don't trip "Cannot Link Media" dialogs.
    2. Each <format> the adapter emitted gets `width`, `height`, and
       `colorSpace` attributes pulled from the sequence settings (or
       the primary asset, when sequence_meta is None). Without these
       Premiere drops the import into a default 1080p Rec.709 sequence
       and warns about every clip not matching the sequence settings.

    Returns the count of <asset> elements patched. Errors are non-fatal
    — the .fcpxml is left in a valid (if unpatched) state and the NLE's
    own import logic still handles it, just less cleanly.
    """
    try:
        # Local import keeps cElementTree out of the cold-start path.
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(out_path))
        root = tree.getroot()
    except Exception as e:
        print(f"  warn: could not parse {out_path.name} for audio-shape "
              f"patch ({type(e).__name__}: {e}); leaving as-is.",
              file=sys.stderr)
        return 0

    patched = 0
    # FCPXML 1.10+ schema: <fcpxml><resources><asset src="..." .../></resources>
    # Tolerate any nesting by walking the whole tree — costs nothing on
    # the small XML files we generate.
    for asset in root.iter("asset"):
        src = asset.get("src")
        local_path = _path_from_safe_url(src)
        if local_path is None:
            continue
        meta = _PROBE_CACHE.get(str(local_path))
        if meta is None:
            # Not in cache (e.g. the user re-ran the patcher standalone).
            # Probe on demand — the cache will absorb the cost.
            try:
                meta = _probe_source_meta(local_path)
            except Exception:
                continue

        # hasVideo / hasAudio drive whether NLEs even attempt to link the
        # respective track. Set them to match probed reality.
        asset.set("hasVideo", "1" if meta["has_video"] else "0")
        asset.set("hasAudio", "1" if meta["has_audio"] else "0")

        if meta["has_audio"]:
            # FCPXML's audio shape: 1 source with N channels at R Hz.
            # Premiere validates these against the linked file at conform
            # time; mismatched values trigger the "Cannot Link Media"
            # dialog the user reported.
            asset.set("audioSources", "1")
            asset.set("audioChannels", str(int(meta["audio_channels"])))
            asset.set("audioRate", str(int(meta["audio_rate"])))

        patched += 1

    # ── <format> sequence-shape patch ──────────────────────────────────
    # Decide what to advertise as the sequence resolution / colorSpace.
    # Prefer the explicit sequence_meta passed by the writer (resolved
    # from the EDL primary source); fall back to the FIRST patched
    # asset's shape as a sensible last resort so a standalone re-patch
    # still does the right thing.
    fmt_w = fmt_h = fmt_cs = None
    if sequence_meta:
        fmt_w = sequence_meta.get("video_width")
        fmt_h = sequence_meta.get("video_height")
        fmt_cs = sequence_meta.get("fcpxml_color_space")
    if fmt_w is None or fmt_h is None or fmt_cs is None:
        # Fall back to the dominant probed source in the cache.
        for cached in _PROBE_CACHE.values():
            if cached.get("has_video"):
                fmt_w = fmt_w or cached.get("video_width")
                fmt_h = fmt_h or cached.get("video_height")
                fmt_cs = fmt_cs or _classify_colorspace(
                    cached.get("color_transfer", "unknown"),
                    cached.get("color_primaries", "unknown"),
                )
                break

    if fmt_w and fmt_h:
        for fmt in root.iter("format"):
            # Don't clobber existing values — the adapter normally
            # leaves these blank, but if a future version starts
            # writing them we want to respect that.
            if not fmt.get("width"):
                fmt.set("width", str(int(fmt_w)))
            if not fmt.get("height"):
                fmt.set("height", str(int(fmt_h)))
            if fmt_cs and not fmt.get("colorSpace"):
                fmt.set("colorSpace", fmt_cs)

    if patched == 0 and not (fmt_w and fmt_h):
        return 0

    try:
        # Preserve the XML declaration the adapter emitted; ET writes a
        # fresh one when xml_declaration=True. Resolve and Premiere both
        # require the declaration, so we keep it.
        tree.write(str(out_path), encoding="UTF-8", xml_declaration=True)
    except Exception as e:
        print(f"  warn: audio-shape patch built but failed to write back "
              f"to {out_path.name} ({type(e).__name__}: {e}); the file is "
              "still valid FCPXML, just missing the audio attrs.",
              file=sys.stderr)
        return 0

    return patched


# ---------------------------------------------------------------------------
# FCP7 xmeml sequence-shape patch.
#
# The otio-fcp-adapter writes a <sequence> with proper rate + duration but
# emits an empty <media><video/><audio/></media> shell. Premiere's xmeml
# importer falls back to its New Sequence default (1080p / 48k) when the
# format characteristics are missing — same painful "doesn't match"
# dialog as FCPXML, plus an unwanted resolution downgrade.
#
# We patch the sequence to declare the correct width / height / pixel
# aspect / field dominance / audio rate / channel count, matching the
# primary source's shape. The per-clip <file> elements already carry
# their real audio/video shape via media_reference.metadata['fcp_xml'],
# which the adapter serializes for free.
# ---------------------------------------------------------------------------

def _patch_xmeml_sequence_format(out_path: Path, sequence_meta: dict | None) -> int:
    """Inject sequence-level format characteristics into a written .xml.

    Returns the number of <sequence> elements patched (typically 1).
    Non-fatal: any failure leaves the file untouched and prints a warn.
    """
    if not sequence_meta:
        return 0

    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(out_path))
        root = tree.getroot()
    except Exception as e:
        print(f"  warn: could not parse {out_path.name} for xmeml "
              f"sequence patch ({type(e).__name__}: {e}); leaving as-is.",
              file=sys.stderr)
        return 0

    def _ensure(parent, tag):
        """Return an existing direct child with `tag`, or create a fresh one."""
        existing = parent.find(tag)
        if existing is not None:
            return existing
        return ET.SubElement(parent, tag)

    def _set_text(parent, tag, value):
        """Idempotently set <tag>value</tag> as a child of `parent`."""
        elt = _ensure(parent, tag)
        elt.text = str(value)
        return elt

    width = int(sequence_meta.get("video_width") or _DEFAULT_VIDEO_WIDTH)
    height = int(sequence_meta.get("video_height") or _DEFAULT_VIDEO_HEIGHT)
    fps = float(sequence_meta.get("video_fps") or _DEFAULT_VIDEO_FPS)
    a_channels = int(sequence_meta.get("audio_channels") or _DEFAULT_AUDIO_CHANNELS)
    a_rate = int(sequence_meta.get("audio_rate") or _DEFAULT_AUDIO_SAMPLERATE)
    par = sequence_meta.get("pixel_aspect_ratio") or "1:1"

    # FCP7 xmeml encodes "is this an NTSC drop-frame rate" as <ntsc>TRUE</ntsc>
    # alongside the integer <timebase>. Round to nearest integer for
    # timebase, mark NTSC for 23.976 / 29.97 / 59.94 / 119.88.
    timebase = int(round(fps))
    ntsc_flag = "TRUE" if abs(fps - timebase * 1000.0 / 1001.0) < 0.01 else "FALSE"

    # FCP7 pixel aspect ratio names (square is by far the most common).
    par_name = "square"
    if par == "10:11":
        par_name = "NTSC-601"
    elif par == "59:54":
        par_name = "PAL-601"
    elif par == "40:33":
        par_name = "NTSC-CCIR601"

    patched = 0
    for sequence in root.iter("sequence"):
        media_e = _ensure(sequence, "media")

        # ── Video format ───────────────────────────────────────────────
        video_e = _ensure(media_e, "video")
        format_e = _ensure(video_e, "format")
        sc_e = _ensure(format_e, "samplecharacteristics")
        _set_text(sc_e, "width", width)
        _set_text(sc_e, "height", height)
        _set_text(sc_e, "pixelaspectratio", par_name)
        _set_text(sc_e, "fielddominance", "none")
        _set_text(sc_e, "colordepth", 24)
        # Nested <rate> on the samplecharacteristics so the format
        # element itself carries fps, not just the parent <sequence>.
        rate_e = _ensure(sc_e, "rate")
        _set_text(rate_e, "ntsc", ntsc_flag)
        _set_text(rate_e, "timebase", timebase)

        # ── Audio format ───────────────────────────────────────────────
        audio_e = _ensure(media_e, "audio")
        _set_text(audio_e, "numOutputChannels", a_channels)
        # Some Premiere versions read <samplerate>/<depth> directly on
        # <audio>; FCP X uses a <format><samplecharacteristics> child.
        # Provide BOTH so either importer is satisfied.
        _set_text(audio_e, "samplerate", a_rate)
        _set_text(audio_e, "depth", 16)
        a_format_e = _ensure(audio_e, "format")
        a_sc_e = _ensure(a_format_e, "samplecharacteristics")
        _set_text(a_sc_e, "depth", 16)
        _set_text(a_sc_e, "samplerate", a_rate)

        patched += 1

    if patched == 0:
        return 0

    try:
        tree.write(str(out_path), encoding="UTF-8", xml_declaration=True)
    except Exception as e:
        print(f"  warn: xmeml sequence patch built but failed to write back "
              f"to {out_path.name} ({type(e).__name__}: {e}); the file is "
              "still valid xmeml, just missing the sequence attrs.",
              file=sys.stderr)
        return 0

    return patched


def _sequence_meta_from_timeline(timeline) -> dict | None:
    """Pull the sequence-shape dict we stashed on the Timeline (if any).

    build_timeline() puts the resolved settings under
    timeline.metadata['premiere-agent']['sequence']; both writers
    forward that to their post-write patchers without having to
    re-resolve the EDL.
    """
    try:
        return (timeline.metadata.get("premiere-agent") or {}).get("sequence")
    except Exception:
        return None


def _speed_map_from_timeline(timeline) -> dict | None:
    """Pull the per-clip retime side-channel stashed by build_timeline().

    Same lookup pattern as _sequence_meta_from_timeline — the speed map
    is keyed by clip name, so the post-write patchers can find each
    retimed element in the XML without re-walking the EDL.
    """
    try:
        return (timeline.metadata.get("premiere-agent") or {}).get("speed_map")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Time-squeezing post-write patchers — one per dialect.
#
# Both XML adapters (otio-fcpx-xml-adapter, otio-fcp-adapter) silently
# DROP otio.schema.LinearTimeWarp during serialization (verified against
# OTIO 0.18.x). So the timeline we wrote describes the OUTPUT durations
# correctly (we set source_range.duration = out_dur in build_timeline),
# but it does NOT carry any retime element — every clip would play at
# 1x and the sped-up segments would be silently truncated by the NLE.
#
# We fix that here by injecting the dialect-correct retime element into
# each clip whose name appears in the speed_map side-channel. The patch
# is idempotent (re-running it on an already-patched file is a no-op
# because we check for existing <timeMap> / timeremap <filter> nodes
# before adding) and adapter-version-agnostic.
# ---------------------------------------------------------------------------

def _fmt_seconds_for_fcpxml(seconds: float, frame_rate: float) -> str:
    """Format a float-seconds value as the rational-seconds string FCPXML wants.

    FCPXML 1.x expresses time as `<num>/<den>s` (whole frames over the
    timeline rate), with `0s` as the special zero literal. The denominator
    must match the timeline's frameDuration so the importer doesn't have
    to re-fraction.
    """
    if abs(seconds) < 1e-9:
        return "0s"
    # Snap to whole frames so the numerator stays an integer at the
    # timeline rate. (We already snap in build_timeline, but a stray
    # caller / variable-speed extension might pass arbitrary values.)
    frames = int(round(seconds * frame_rate))
    den = int(round(frame_rate))
    if den <= 0:
        den = 24
    return f"{frames}/{den}s"


def _patch_fcpxml_speed(out_path: Path, speed_map: dict | None) -> int:
    """Inject FCPXML 1.10+ <timeMap> elements for every retimed clip.

    Each entry in `speed_map` is keyed by the OTIO clip name we minted
    in build_timeline (e.g. "C0103_v_05" / "C0103_a_05"). FCPXML emits
    those names verbatim onto the spine's <clip name="..."> elements,
    so we can find each retimed element with a name lookup.

    For every match we prepend a <timeMap> child like:

        <timeMap>
          <timept time="0s"           value="0s"           interp="linear"/>
          <timept time="<out_dur>s"   value="<src_dur>s"   interp="linear"/>
        </timeMap>

    which tells FCP X / Resolve "play <src_dur> seconds of source media
    inside <out_dur> seconds of the timeline, linearly" — i.e. constant
    speed = src_dur / out_dur. This is the canonical FCPXML retime form
    and round-trips through both NLEs cleanly.

    Returns the count of clips patched. Non-fatal on any error — the
    file is left as a valid (un-retimed) FCPXML and the warning prints
    to stderr.
    """
    if not speed_map:
        return 0

    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(out_path))
        root = tree.getroot()
    except Exception as e:
        print(f"  warn: could not parse {out_path.name} for FCPXML "
              f"speed patch ({type(e).__name__}: {e}); leaving as-is.",
              file=sys.stderr)
        return 0

    _ts_log_section(f"_patch_fcpxml_speed → {out_path.name}")
    patched = 0
    # Walk every <clip> element in the file (typically one per timeline
    # entry). The OTIO FCPXML adapter places retime-eligible clips
    # under <spine> as <clip name="..."> with nested <video>/<audio>;
    # iterating root-wide is robust to future nesting changes.
    for clip_e in root.iter("clip"):
        name = clip_e.get("name")
        if not name or name not in speed_map:
            continue
        info = speed_map[name]
        fr = float(info.get("frame_rate") or 24.0)
        src_dur = float(info["src_dur_s"])
        out_dur = float(info["out_dur_s"])

        # Idempotence — if a <timeMap> is already there, leave it.
        if clip_e.find("timeMap") is not None:
            _ts_log(f"  skip {name}: already has <timeMap>")
            continue

        # Build the retime element. FCPXML wants children in a specific
        # order (timeMap goes before any conform-rate sibling typically
        # appears in real-world files), but Resolve / FCP X are tolerant
        # of leading-position children, so prepending is safe.
        tm = ET.Element("timeMap")
        ET.SubElement(tm, "timept", {
            "time":   "0s",
            "value":  "0s",
            "interp": "linear",
        })
        time_str = _fmt_seconds_for_fcpxml(out_dur, fr)
        value_str = _fmt_seconds_for_fcpxml(src_dur, fr)
        ET.SubElement(tm, "timept", {
            "time":   time_str,
            "value":  value_str,
            "interp": "linear",
        })
        # Prepend so the importer reads the retime before the inner
        # <video>/<audio> element it modifies.
        clip_e.insert(0, tm)
        patched += 1
        _ts_log(
            f"  patch {name}: <timeMap> "
            f"timept(0s→0s, {time_str}→{value_str}) "
            f"out_dur_s={out_dur:.6f} src_dur_s={src_dur:.6f} "
            f"effective_speed={(src_dur / out_dur if out_dur > 0 else 0):g}×"
        )

    if patched == 0:
        return 0

    try:
        tree.write(str(out_path), encoding="UTF-8", xml_declaration=True)
    except Exception as e:
        print(f"  warn: FCPXML speed patch built {patched} timeMap(s) but "
              f"failed to write back to {out_path.name} "
              f"({type(e).__name__}: {e}); the file is still valid FCPXML, "
              "just without the retime elements.", file=sys.stderr)
        return 0

    return patched


def _patch_xmeml_speed(out_path: Path, speed_map: dict | None) -> int:
    """Inject Premiere "Time Remap" filters for every retimed clipitem.

    Apple's xmeml spec — the FCPXML / FinalCutPro_XML "Elements
    Catalog" entry for the Time Remap Effect — is the source of
    truth here. Constant-speed remapping is encoded as a <filter>
    block on the clipitem carrying the standard timeremap effect
    PLUS an explicit <graphdict> parameter with two virtual
    keyframes spanning the clipitem:

        <filter>
          <effect>
            <name>Time Remap</name>
            <effectid>timeremap</effectid>
            <effectcategory>motion</effectcategory>
            <effecttype>motion</effecttype>
            <mediatype>video</mediatype>     (or "audio" on A1 clips)

            <!-- Scalar params: drive the timeline name annotation
                 ("[N%]") and pick constant-vs-spline mode. -->
            <parameter>
              <parameterid>variablespeed</parameterid>
              <value>0</value>               (0=constant, 1=keyframed)
            </parameter>
            <parameter>
              <parameterid>speed</parameterid>
              <value>SPEED_PCT</value>       (200 = 2x, 1000 = 10x)
            </parameter>
            <parameter>
              <parameterid>reverse</parameterid>
              <value>FALSE</value>
            </parameter>
            <parameter>
              <parameterid>frameblending</parameterid>
              <value>FALSE</value>
            </parameter>

            <!-- The canonical Apple-spec encoding: explicit map
                 from sequence-absolute timeline frame (`when`)
                 to absolute source-media frame (`value`). -->
            <parameter>
              <parameterid>graphdict</parameterid>
              <name>graphdict</name>
              <interpolation><name>Linear</name></interpolation>
              <keyframe>
                <when>START_FRAME</when>     (== <start>)
                <value>IN_FRAME</value>      (== <in>)
                <speedvirtualkf>TRUE</speedvirtualkf>
                <speedkfin>TRUE</speedkfin>
                <speedkfstart>TRUE</speedkfstart>
              </keyframe>
              <keyframe>
                <when>END_FRAME</when>       (== <end>)
                <value>OUT_FRAME</value>     (== <out> = <in> + SRC_DUR)
                <speedvirtualkf>TRUE</speedvirtualkf>
                <speedkfout>TRUE</speedkfout>
                <speedkfend>TRUE</speedkfend>
              </keyframe>
            </parameter>
          </effect>
        </filter>

    The filter (specifically the scalar <speed> parameter) is what
    produces the "[N%]" indicator beside the clip name on
    Premiere's timeline. Without it, the clip plays at 1.0× no
    matter what (out - in) vs (end - start) say — Premiere does
    NOT infer "Speed/Duration" from a clipitem-span ratio in
    xmeml. The filter is mandatory for the squish to render.

    Why graphdict matters for the diagonal-stripe bug: Premiere's
    timeremap engine has TWO ways to compute the source frame at a
    given timeline position — (a) the implicit clip-rate ratio
    (out - in) / (end - start) combined with the scalar speed%,
    and (b) the explicit graphdict virtual keyframes. Without (b),
    Premiere falls back to (a), and any rounding mismatch between
    the integer (out - in) frame count and the integer
    (end - start) timeline span at the requested speed gets
    painted as the diagonal "out of bounds" stripe over the
    fractional-frame discrepancy zone. The graphdict explicitly
    pins the mapping at both endpoints so there's no ambiguity:
    timeline frame <start> → source frame <in>, timeline frame
    <end> → source frame <out> (= <in> + SRC_DUR), linearly
    interpolated. Premiere uses the explicit map and never falls
    back to the implicit-ratio code path.

    The `when` field on graphdict virtual keyframes uses the SAME
    coordinate space as <start>/<end> (sequence-absolute timeline
    frames at the parent rate), NOT clipitem-local 0..span. The
    speedkfstart/speedkfend flags on the virtual keyframes
    explicitly say they sit "at the setting for `start`/`end`" —
    so kf0.when must equal <start> and kf1.when must equal <end>,
    or Premiere can't locate the keyframes within the clipitem's
    timeline range and falls back to drawing the entire clip with
    the diagonal "out of bounds" stripe pattern.

    Companion source-side rewrite — for the filter+graphdict to
    consume the correct source range we extend the clipitem's
    playhead bookkeeping:
      * <out>      goes from src_start + out_dur   →  src_start + src_dur
        so the source range matches what the graphdict's keyframe
        endpoints declare.
      * <duration> goes from out_dur_frames        →  src_dur_frames
        so the canonical xmeml invariant
            <out> - <in> == <duration>
        is preserved. This is the SAME invariant every other
        clipitem in the file (and every bin master) obeys:
        <duration> is the frame count THIS clipitem consumes from
        its source media, NOT the file-wide frame count. The
        file-wide value lives on the bin master
        (`<clip><name>foo.mp4</name><duration>FILE</duration>…`);
        the per-use clipitem only ever owns its own slice.

    The bounds clamp below (src_dur_frames vs max_src_dur_frames)
    is an additional safety net — it keeps (in + src_dur_frames)
    inside the file's physical frame count even when the EDL
    range itself landed past the end (e.g. an editor agent
    rounding up).

    <start>, <end>, <in> stay as-is — they're already correct.

    Returns the count of clipitems patched. Non-fatal on any
    error.
    """
    if not speed_map:
        return 0

    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(out_path))
        root = tree.getroot()
    except Exception as e:
        print(f"  warn: could not parse {out_path.name} for xmeml "
              f"speed patch ({type(e).__name__}: {e}); leaving as-is.",
              file=sys.stderr)
        return 0

    _ts_log_section(f"_patch_xmeml_speed → {out_path.name}")
    patched = 0
    for ci in root.iter("clipitem"):
        # In xmeml the OTIO clip name lives in the clipitem's <name>
        # CHILD, not as an attribute. (The `id` attribute is a synthetic
        # adapter-generated counter we can't rely on for lookup.)
        name_e = ci.find("name")
        if name_e is None or not name_e.text:
            continue
        name = name_e.text.strip()
        if name not in speed_map:
            continue
        info = speed_map[name]
        kind = info.get("kind", "video")
        requested_speed = float(info["speed"])
        
        # The scalar <speed> parameter is Premiere's timeline-name
        # annotation value ("[N%]"). It MUST match the slope encoded
        # in the graphdict virtual keyframes or Premiere will show
        # the wrong speed indicator. The graphdict slope is
        # (value_delta / when_delta) = (src_dur / out_dur) = speed,
        # so the scalar parameter needs the same: speed × 100.
        speed_pct = requested_speed * 100.0
        
        fr = float(info.get("frame_rate") or 24.0)

        # Snapshot the timeline-side bookkeeping FIRST so the log can
        # show what OTIO actually wrote vs what we're about to rewrite.
        # Any disagreement with the speed_map's intent (e.g. <start> /
        # <end> not landing where the cur_v cursor said) shows up here
        # before we mutate anything.
        start_e = ci.find("start")
        end_e = ci.find("end")
        rate_e = ci.find("rate")
        clip_rate_str = "?"
        if rate_e is not None:
            tb = rate_e.find("timebase")
            ntsc = rate_e.find("ntsc")
            if tb is not None:
                tb_t = (tb.text or "?").strip()
                ntsc_t = (ntsc.text or "?").strip() if ntsc is not None else "?"
                clip_rate_str = f"timebase={tb_t} ntsc={ntsc_t}"
        # Resolve the linked <file> (asset) and capture its declared
        # duration / pathurl. If the clipitem references a master via
        # `id` we have to walk the file list to find the body — but the
        # OTIO writer always emits an inline <file> on first occurrence
        # and refs after that. We grab whichever is present.
        file_e = ci.find("file")
        file_dur_str = "<no-inline-file>"
        file_url_str = "<no-inline-file>"
        file_id_str = "?"
        if file_e is not None:
            file_id_str = file_e.get("id") or "?"
            fdur = file_e.find("duration")
            if fdur is not None:
                file_dur_str = (fdur.text or "?").strip()
            furl = file_e.find("pathurl")
            if furl is not None:
                file_url_str = (furl.text or "?").strip()
        _ts_log("")
        _ts_log(
            f"  ── {name} (kind={kind}, speed={info['speed']:g}× / "
            f"{speed_pct:g}%) ──"
        )
        _ts_log(
            f"    pre-patch xml    : "
            f"<rate>{clip_rate_str}</rate> "
            f"<start>{(start_e.text if start_e is not None else '?')}</start> "
            f"<end>{(end_e.text if end_e is not None else '?')}</end> "
            f"<in>{ci.findtext('in', '?')}</in> "
            f"<out>{ci.findtext('out', '?')}</out> "
            f"<duration>{ci.findtext('duration', '?')}</duration>"
        )
        _ts_log(
            f"    asset file       : id={file_id_str} "
            f"<duration>{file_dur_str}</duration> url={file_url_str}"
        )
        # Pull the floats with explicit defaults so a malformed
        # speed_map entry (future caller bug) logs gracefully instead
        # of crashing the patcher with a TypeError on the format spec.
        _ss = float(info.get("src_start_s") or 0.0)
        _sd = float(info.get("src_dur_s") or 0.0)
        _od = float(info.get("out_dur_s") or 0.0)
        _md = float(info.get("media_dur_s") or 0.0)
        _ts_log(
            f"    speed_map info   : src_start_s={_ss:.6f} "
            f"src_dur_s={_sd:.6f} "
            f"out_dur_s={_od:.6f} "
            f"media_dur_s={_md:.6f} "
            f"frame_rate={fr}"
        )

        # Idempotence — skip if a timeremap filter is already present.
        # We check effectid because Premiere ships several motion-y
        # filter blocks and we only want to skip on a real prior patch.
        already_patched = False
        for filt in ci.findall("filter"):
            eff = filt.find("effect")
            if eff is None:
                continue
            eid = eff.find("effectid")
            if eid is not None and (eid.text or "").strip() == "timeremap":
                already_patched = True
                break
        if already_patched:
            _ts_log(
                f"    SKIP             : <filter><effectid>timeremap"
                f"</effectid> already present (idempotent re-run)"
            )
            continue

        # ── Compute the frame numbers we need, from speed_map info ────
        #
        # Premiere's NATIVE time-remap xmeml structure (verified by
        # importing our cut.xml, fixing the time-remaps in Premiere,
        # and re-exporting as fixed_seq.xml) is fundamentally
        # different from non-remapped clipitems:
        #
        #   <duration>   = clipitem extent in clipitem-local frames
        #                  (typically == timeline span <end>-<start>,
        #                  but can be larger to provide trim handles)
        #   <in>         = CLIPITEM-LOCAL timeline frame where the
        #                  visible section begins (NOT a source frame!)
        #   <out>        = CLIPITEM-LOCAL timeline frame where the
        #                  visible section ends   (NOT a source frame!)
        #
        # The graphdict carries the actual source-frame mapping:
        #
        #   when         = clipitem-local timeline frame (0..duration)
        #   value        = ABSOLUTE source-media frame number
        #                  (0..media_duration_in_frames)
        #
        # OTIO writes the clipitem with <in> = src_start_frame and
        # <out> = src_start_frame + out_dur_frames (treating in/out as
        # source frames, which is correct for non-remapped clips).
        # For time-remapped clips we MUST rewrite these to be
        # clipitem-local instead, and emit the source-frame mapping
        # in the graphdict.
        src_in_frame = int(round(
            float(info.get("src_start_s", 0.0)) * fr
        ))
        src_dur_frames_intended = int(round(
            float(info["src_dur_s"]) * fr
        ))
        src_dur_frames = src_dur_frames_intended

        # Used by older log lines that still reference in_frames as
        # "the source-IN frame number". Keeping the alias avoids
        # touching every log statement below.
        in_frames = src_in_frame

        in_e = ci.find("in")
        out_e = ci.find("out")
        dur_e = ci.find("duration")
        # Snapshot the pre-patch in/out/duration so the log can show
        # what OTIO wrote vs. what we rewrote.
        pre_in_str = (in_e.text or "?").strip() if in_e is not None else "?"
        pre_out_str = (out_e.text or "?").strip() if out_e is not None else "?"
        pre_dur_str = (dur_e.text or "?").strip() if dur_e is not None else "?"

        # Compute the timeline span of this clipitem from <start>/<end>
        # (which OTIO wrote in sequence-absolute frames at the
        # sequence rate). This is the visible portion of the clip
        # on the timeline, and equals the clipitem-local end coord.
        try:
            start_frame = int(float((start_e.text or "0").strip())) \
                if start_e is not None else 0
            end_frame = int(float((end_e.text or "0").strip())) \
                if end_e is not None else (
                    start_frame +
                    int(round(float(info["out_dur_s"]) * fr))
                )
        except (TypeError, ValueError):
            start_frame = 0
            end_frame = int(round(float(info["out_dur_s"]) * fr))
        timeline_dur_frames = max(1, end_frame - start_frame)

        # Bounds-clamp source consumption to the file's actual frame
        # count, with a 6-frame safety margin to absorb ffprobe-vs-
        # decoder duration disagreements on DJI / GoPro footage.
        media_dur_s = info.get("media_dur_s")
        if media_dur_s is not None:
            media_dur_frames = int(round(float(media_dur_s) * fr))
            media_src_str = f"{media_dur_s:.6f}s × {fr}fps"
        else:
            media_dur_frames = src_in_frame + src_dur_frames
            media_src_str = f"FALLBACK in+src_dur ({media_dur_frames}f)"
        max_src_dur_frames = max(
            1, media_dur_frames - src_in_frame - 6
        )
        clamp_fired_here = src_dur_frames > max_src_dur_frames
        if clamp_fired_here:
            print(
                f"  warn: clip '{name}' source range exceeds "
                f"media bounds — clamping src_dur from "
                f"{src_dur_frames} to {max_src_dur_frames} "
                f"frames (in={src_in_frame}, "
                f"media_dur={media_dur_frames}). "
                f"Edit the EDL range to fit within the source "
                f"file to avoid this clamp.",
                file=sys.stderr,
            )
            src_dur_frames = max_src_dur_frames

        # Source-frame absolute coords for the visible section's
        # endpoints — these are what Premiere stores in the
        # graphdict's `value` field at the in/out keyframes.
        src_out_frame = src_in_frame + src_dur_frames

        # ── Rewrite clipitem in/out/duration to clipitem-local ────────
        # For our minimal "no head/tail trim" case the visible section
        # spans the full clipitem extent: <in>=0, <out>=duration.
        if in_e is not None:
            in_e.text = "0"
        if out_e is not None:
            out_e.text = str(timeline_dur_frames)
        if dur_e is not None:
            dur_e.text = str(timeline_dur_frames)

        _ts_log(
            f"    frame_math       : src_in_f={src_in_frame} "
            f"src_dur_f_intended={src_dur_frames_intended} "
            f"media_dur_f={media_dur_frames} "
            f"(from {media_src_str}) "
            f"max_src_dur_f={max_src_dur_frames} "
            f"clamp_fired={clamp_fired_here} "
            f"src_dur_f_final={src_dur_frames} "
            f"src_out_f={src_out_frame} "
            f"timeline_span_f={timeline_dur_frames}"
        )
        _ts_log(
            f"    written xml      : <in>0</in> "
            f"<out>{timeline_dur_frames}</out> "
            f"<duration>{timeline_dur_frames}</duration> "
            f"(was in={pre_in_str} out={pre_out_str} dur={pre_dur_str}) "
            f"— now CLIPITEM-LOCAL, source mapping is in graphdict"
        )

        # The display speed_pct is now derived from the ACTUAL graphdict
        # slope (src_dur / timeline_dur), not from the requested
        # speed multiplier — this guarantees the [N%] timeline label
        # matches what Premiere computes from the graphdict, even when
        # frame-snap rounding makes the actual ratio drift slightly
        # from the requested multiplier (e.g. 800.21% instead of 800%).
        actual_slope = src_dur_frames / timeline_dur_frames
        speed_pct = actual_slope * 100.0
        _ts_log(
            f"    speed_consistency: actual_slope={actual_slope:.4f}× "
            f"display_speed_pct={speed_pct:g}% "
            f"requested={info['speed']:g}× "
            f"snap_drift={(actual_slope - float(info['speed'])):+.4f}×"
        )

        # Build the timeremap filter. We hand-build the element tree
        # so nothing depends on the OTIO adapter's filter
        # serialization.
        filt = ET.SubElement(ci, "filter")
        eff = ET.SubElement(filt, "effect")
        ET.SubElement(eff, "name").text = "Time Remap"
        ET.SubElement(eff, "effectid").text = "timeremap"
        ET.SubElement(eff, "effectcategory").text = "motion"
        ET.SubElement(eff, "effecttype").text = "motion"
        ET.SubElement(eff, "mediatype").text = (
            "audio" if kind == "audio" else "video"
        )

        def _add_param(eff_e, pid, value, vmin=None, vmax=None):
            """Append a simple <parameter> with id/name/value/(min/max).

            Used for the scalar Time Remap parameters
            (variablespeed, speed, reverse, frameblending). The
            graphdict parameter has its own builder below because it
            additionally needs <interpolation> + a pair of
            <keyframe> children with the speed-virtual-keyframe
            metadata Apple's spec mandates.
            """
            p = ET.SubElement(eff_e, "parameter")
            p.set("authoringApp", "PremierePro")
            ET.SubElement(p, "parameterid").text = pid
            ET.SubElement(p, "name").text = pid
            if vmin is not None:
                ET.SubElement(p, "valuemin").text = str(vmin)
            if vmax is not None:
                ET.SubElement(p, "valuemax").text = str(vmax)
            ET.SubElement(p, "value").text = str(value)

        # ── Scalar parameters (display indicator + sanity defaults) ──
        # variablespeed=0 says "this is a constant-speed remap". Even
        # with explicit graphdict keyframes Premiere still reads this
        # to decide whether to draw the speed band straight or as a
        # spline in the Effect Controls panel.
        _add_param(eff, "variablespeed", 0, vmin=0, vmax=1)
        # Premiere's hard limits on speed are -100000..100000 (%) —
        # the MAX_SPEED clamp upstream keeps us inside the sane
        # window. The <speed> scalar is what produces the "[N%]"
        # indicator on the timeline name strip; without it Premiere
        # shows the clip name with no speed annotation.
        _add_param(eff, "speed", f"{speed_pct:g}",
                   vmin=-100000, vmax=100000)
        _add_param(eff, "reverse", "FALSE")
        _add_param(eff, "frameblending", "FALSE")

        # ── graphdict: Premiere's canonical Time Remap encoding ───────
        # Verified by reverse-engineering Premiere's own export of a
        # time-remapped clipitem (see fixed_seq.xml in the test
        # corpus). Apple's spec told us about graphdict but was
        # ambiguous about coordinate spaces — Premiere's actual
        # implementation is:
        #
        #   <parameter authoringApp="PremierePro">
        #     <parameterid>graphdict</parameterid>
        #     <name>graphdict</name>
        #     <valuemin>0</valuemin>
        #     <valuemax>MEDIA_DURATION_FRAMES</valuemax>  ← whole file
        #     <value>0</value>
        #     <keyframe>                       ← speedkfstart (extent IN)
        #       <when>0</when>
        #       <value>SRC_IN_FRAME</value>
        #       <speedvirtualkf>TRUE</speedvirtualkf>
        #       <speedkfstart>TRUE</speedkfstart>
        #     </keyframe>
        #     <keyframe>                       ← speedkfin (visible IN)
        #       <when>0</when>                 (== <in>)
        #       <value>SRC_IN_FRAME</value>
        #       <speedvirtualkf>TRUE</speedvirtualkf>
        #       <speedkfin>TRUE</speedkfin>
        #     </keyframe>
        #     <keyframe>                       ← speedkfout (visible OUT)
        #       <when>TIMELINE_DUR</when>      (== <out>)
        #       <value>SRC_OUT_FRAME</value>
        #       <speedvirtualkf>TRUE</speedvirtualkf>
        #       <speedkfout>TRUE</speedkfout>
        #     </keyframe>
        #     <keyframe>                       ← speedkfend (extent END)
        #       <when>TIMELINE_DUR</when>
        #       <value>SRC_OUT_FRAME</value>
        #       <speedvirtualkf>TRUE</speedvirtualkf>
        #       <speedkfend>TRUE</speedkfend>
        #     </keyframe>
        #     <interpolation>
        #       <name>FCPCurve</name>          ← NOT "Linear"!
        #     </interpolation>
        #   </parameter>
        #
        # Coordinate spaces:
        #   when  → CLIPITEM-LOCAL TIMELINE FRAMES at the clip's rate
        #           (range: 0..duration). The visible section is
        #           the [<in>..<out>] sub-range of [0..duration].
        #   value → ABSOLUTE SOURCE-MEDIA FRAMES
        #           (range: 0..media_duration_in_frames). Tells
        #           Premiere which source frame to play at the
        #           corresponding timeline-local moment.
        #
        # The slope (Δvalue / Δwhen) is the speed multiplier.
        # For our constant-speed remap with no head/tail trim, the
        # start+in keyframes coincide at when=0/value=src_in, and
        # the out+end keyframes coincide at when=timeline_dur/
        # value=src_out. Premiere requires all four virtual-keyframe
        # markers to be present even when their positions coincide.
        #
        # Verified slope example from fixed_seq.xml v_42 (10× speed):
        #   kf(in):  when=554,  value=5541
        #   kf(out): when=1973, value=19736
        #   slope = (19736-5541) / (1973-554) = 14195/1419 = 10.003× ✓
        graphdict = ET.SubElement(eff, "parameter")
        graphdict.set("authoringApp", "PremierePro")
        ET.SubElement(graphdict, "parameterid").text = "graphdict"
        ET.SubElement(graphdict, "name").text = "graphdict"
        # The valuemin/valuemax bounds on the parent <parameter> are
        # what Premiere uses to know the FULL source-media frame range
        # available for trim-handle dragging. Without these, dragging
        # the trim handles in Premiere can't reveal more source
        # frames beyond what's already exposed by the keyframes.
        ET.SubElement(graphdict, "valuemin").text = "0"
        ET.SubElement(graphdict, "valuemax").text = str(media_dur_frames)
        ET.SubElement(graphdict, "value").text = "0"
        # ── Four virtual keyframes (start, in, out, end) ──
        # In our minimal case start coincides with in, and out
        # coincides with end (no head/tail trim handles), so kf0/kf1
        # share coordinates and kf2/kf3 share coordinates. Premiere
        # still wants all four flag markers present and distinct.
        kf_start = ET.SubElement(graphdict, "keyframe")
        ET.SubElement(kf_start, "when").text = "0"
        ET.SubElement(kf_start, "value").text = str(src_in_frame)
        ET.SubElement(kf_start, "speedvirtualkf").text = "TRUE"
        ET.SubElement(kf_start, "speedkfstart").text = "TRUE"

        kf_in = ET.SubElement(graphdict, "keyframe")
        ET.SubElement(kf_in, "when").text = "0"
        ET.SubElement(kf_in, "value").text = str(src_in_frame)
        ET.SubElement(kf_in, "speedvirtualkf").text = "TRUE"
        ET.SubElement(kf_in, "speedkfin").text = "TRUE"

        kf_out = ET.SubElement(graphdict, "keyframe")
        ET.SubElement(kf_out, "when").text = str(timeline_dur_frames)
        ET.SubElement(kf_out, "value").text = str(src_out_frame)
        ET.SubElement(kf_out, "speedvirtualkf").text = "TRUE"
        ET.SubElement(kf_out, "speedkfout").text = "TRUE"

        kf_end = ET.SubElement(graphdict, "keyframe")
        ET.SubElement(kf_end, "when").text = str(timeline_dur_frames)
        ET.SubElement(kf_end, "value").text = str(src_out_frame)
        ET.SubElement(kf_end, "speedvirtualkf").text = "TRUE"
        ET.SubElement(kf_end, "speedkfend").text = "TRUE"

        # FCPCurve is Apple's standard piecewise interpolation —
        # this is what Premiere's own export uses for time remap.
        # "Linear" produces a DIFFERENT (broken) playback path that
        # Premiere mis-renders as the diagonal "out of bounds" stripe.
        gd_interp = ET.SubElement(graphdict, "interpolation")
        ET.SubElement(gd_interp, "name").text = "FCPCurve"

        _ts_log(
            f"    graphdict        : "
            f"4 keyframes (start/in/out/end), "
            f"interp=FCPCurve, "
            f"valuemax={media_dur_frames}  "
            f"kf_start+kf_in (when=0, value={src_in_frame})  "
            f"kf_out+kf_end (when={timeline_dur_frames}, "
            f"value={src_out_frame})  "
            f"slope = {(src_out_frame - src_in_frame) / max(1, timeline_dur_frames):.4f}×  "
            f"(timeline-local 0..{timeline_dur_frames} → "
            f"source-absolute {src_in_frame}..{src_out_frame})"
        )

        patched += 1

    if patched == 0:
        return 0

    try:
        tree.write(str(out_path), encoding="UTF-8", xml_declaration=True)
    except Exception as e:
        print(f"  warn: xmeml speed patch built {patched} timeremap "
              f"filter(s) but failed to write back to {out_path.name} "
              f"({type(e).__name__}: {e}); the file is still valid "
              "xmeml, just without the retime filters.",
              file=sys.stderr)
        return 0

    return patched


def _patch_xmeml_bin_layout(
    out_path: Path,
    bin_name: str = "Footage",
) -> int:
    """Reorganize the xmeml so source clips live in a project-bin folder.

    The OTIO `fcp_xml` adapter emits a flat layout — `<project>/<children>`
    contains a single `<sequence>` element and Premiere derives the
    project-bin clips implicitly from the sequence's `<clipitem>`s.
    Implicit derivation means:

      1. Every source clip lands at the bin's ROOT (no folder), which
         is messy when an EDL pulls from many files.
      2. None of the sequence clipitems carry a `<masterclipid>`, which
         is the official xmeml linkage between a timeline clipitem and
         its bin "master clip" entry. Without it, Premiere treats every
         clipitem as a one-off phantom asset — and for time-remapped
         clipitems that's exactly the configuration that triggers the
         "Replace with clip > From Bin" workaround (which restores the
         masterclipid linkage Premiere wanted in the first place).

    This pass fixes both:

      * Builds one `<clip>` master per unique `<file id>` and wraps
        them all in a single `<bin name="{bin_name}">` (default
        "Footage"). Master clips carry the file's full duration / rate
        / pathurl so Premiere has the canonical asset metadata.
      * Inserts a `<masterclipid>` reference into every sequence
        `<clipitem>` so it links back to its bin master.
      * Inserts the bin as the FIRST child of `<project>/<children>`
        so it sorts above the sequence in the project window.

    Idempotent — skips re-running if a `<bin>` is already present at
    the project root. Non-fatal on any error: the file is left as-is
    and a warning is printed.

    Returns the number of master clips written into the bin.
    """
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(out_path))
        root = tree.getroot()
    except Exception as e:
        print(f"  warn: could not parse {out_path.name} for bin "
              f"layout patch ({type(e).__name__}: {e}); leaving as-is.",
              file=sys.stderr)
        return 0

    # ── Locate the project's <children> container ─────────────────────
    # OTIO's xmeml output is <xmeml><project><children><sequence/>…
    # We look for that exact shape; if the user hand-edited the file
    # into something else we bail rather than guess.
    project = root.find("project")
    children = project.find("children") if project is not None else None
    sequence = children.find("sequence") if children is not None else None
    if project is None or children is None or sequence is None:
        return 0

    # ── Idempotence — skip if a bin is already at the root ────────────
    if children.find("bin") is not None:
        return 0

    # ── Collect every UNIQUE <file id="…"> across the whole document ──
    # The OTIO adapter writes the FULL file definition (name, pathurl,
    # duration, rate, media descriptor) inside the FIRST clipitem that
    # references each file id; subsequent clipitems carry a bare
    # `<file id="file-N" />` reference. We snapshot the full def for
    # each id so we can hoist it into the master clip below.
    full_files: dict[str, ET.Element] = {}
    for f in root.iter("file"):
        fid = f.get("id")
        if not fid or fid in full_files:
            continue
        # Only the "full" definition has child elements (name, pathurl,
        # duration, rate, …). Reference-only forms are empty.
        if len(list(f)) == 0:
            continue
        full_files[fid] = f
    if not full_files:
        return 0

    # ── Build a deterministic file_id -> masterclip_id mapping ────────
    # Sort by the numeric suffix when present ("file-7" -> 7) so the
    # masterclip ids come out in source-declaration order. Falls back
    # to lexicographic for non-conforming ids.
    def _id_sort_key(fid: str):
        try:
            return (0, int(fid.rsplit("-", 1)[-1]))
        except (ValueError, IndexError):
            return (1, fid)
    file_ids_sorted = sorted(full_files.keys(), key=_id_sort_key)
    file_to_master: dict[str, str] = {
        fid: f"masterclip-{i + 1}" for i, fid in enumerate(file_ids_sorted)
    }

    # ── Build the bin <children> with one master clip per source file ──
    # Master-clip structure follows Premiere's own xmeml export shape
    # (the minimum subset Premiere requires to recognise the bin entry
    # AND treat sequence clipitems as instances of that master):
    #
    #   <clip id="masterclip-N" frameBlend="FALSE" explodedTracks="true">
    #     <uuid>...</uuid>
    #     <masterclipid>masterclip-N</masterclipid>
    #     <ismasterclip>TRUE</ismasterclip>
    #     <duration>...</duration>            (file duration in frames)
    #     <rate>...</rate>                    (file rate)
    #     <name>...</name>                    (file basename)
    #     <media>
    #       <video><track><clipitem .../></track></video>
    #       <audio><track><clipitem .../></track></audio>
    #     </media>
    #     <file id="file-N">...full def...</file>
    #   </clip>
    import uuid as _uuid

    bin_e = ET.Element("bin")
    ET.SubElement(bin_e, "name").text = bin_name
    bin_children = ET.SubElement(bin_e, "children")

    masters_written = 0
    for fid in file_ids_sorted:
        f_full = full_files[fid]
        master_id = file_to_master[fid]

        # Pull canonical fields off the file def so the master mirrors
        # the underlying asset's intrinsic shape. .find() may return
        # None for sparsely populated files; we fall back to safe
        # defaults rather than skip (a missing duration master is
        # still a valid bin entry, just with 0-length scrubber).
        file_name_e = f_full.find("name")
        file_dur_e = f_full.find("duration")
        file_rate_e = f_full.find("rate")
        file_media_e = f_full.find("media")
        # `<file>` declares whether each media stream exists via
        # `<media><video>` and `<media><audio>` children — we only
        # build a master-side <video>/<audio> branch when the source
        # actually has that stream, so we don't fabricate a phantom
        # audio track on a video-only file (or vice versa).
        has_video = (file_media_e is not None and
                     file_media_e.find("video") is not None)
        has_audio = (file_media_e is not None and
                     file_media_e.find("audio") is not None)

        clip_e = ET.SubElement(bin_children, "clip", {
            "id": master_id,
            "frameBlend": "FALSE",
            "explodedTracks": "true",
        })
        # Stable UUID per file id so reimports don't churn Premiere's
        # internal asset registry — UUIDv5 over the file id keeps it
        # deterministic across runs.
        ET.SubElement(clip_e, "uuid").text = str(
            _uuid.uuid5(_uuid.NAMESPACE_URL, f"premiere-agent/{fid}")
        )
        ET.SubElement(clip_e, "masterclipid").text = master_id
        ET.SubElement(clip_e, "ismasterclip").text = "TRUE"
        if file_dur_e is not None and file_dur_e.text:
            ET.SubElement(clip_e, "duration").text = file_dur_e.text
        if file_rate_e is not None:
            # Deep-copy rate so subsequent edits to the file's rate
            # element don't mutate the master's rate by reference.
            from copy import deepcopy as _deepcopy
            clip_e.append(_deepcopy(file_rate_e))
        if file_name_e is not None and file_name_e.text:
            ET.SubElement(clip_e, "name").text = file_name_e.text

        # Master-side media branch — one clipitem per stream, each
        # spanning the FULL file (in=0, out=duration). Premiere uses
        # this to populate the bin clip's source view scrubber.
        media_e = ET.SubElement(clip_e, "media")
        try:
            file_dur_frames = int(float((file_dur_e.text or "0").strip())) \
                if file_dur_e is not None else 0
        except (TypeError, ValueError):
            file_dur_frames = 0

        def _add_master_track(parent_tag: str, track_idx_text: str) -> None:
            """Build <{parent_tag}><track><clipitem/></track></{parent_tag}>
            with the master's full-range clipitem inside."""
            stream_e = ET.SubElement(media_e, parent_tag)
            track_e = ET.SubElement(stream_e, "track")
            ci = ET.SubElement(track_e, "clipitem", {
                "id": f"{master_id}-{parent_tag[0]}",
            })
            ET.SubElement(ci, "masterclipid").text = master_id
            if file_name_e is not None and file_name_e.text:
                ET.SubElement(ci, "name").text = file_name_e.text
            if file_dur_e is not None and file_dur_e.text:
                ET.SubElement(ci, "duration").text = file_dur_e.text
            if file_rate_e is not None:
                from copy import deepcopy as _deepcopy
                ci.append(_deepcopy(file_rate_e))
            ET.SubElement(ci, "in").text = "0"
            ET.SubElement(ci, "out").text = str(file_dur_frames)
            ET.SubElement(ci, "file", {"id": fid})
            if parent_tag == "audio":
                # Required for Premiere to wire the master's audio
                # scrubber to the file's first audio stream.
                st = ET.SubElement(ci, "sourcetrack")
                ET.SubElement(st, "mediatype").text = "audio"
                ET.SubElement(st, "trackindex").text = track_idx_text

        if has_video:
            _add_master_track("video", "1")
        if has_audio:
            _add_master_track("audio", "1")

        # Hoist the FULL <file> definition out of its current
        # (sequence-clipitem) parent and re-attach it to the master
        # clip. Sequence clipitems then link via the `<file id=...>`
        # reference shape, which is what Premiere expects when a bin
        # master is present.
        #
        # CRITICAL: When the full def was sitting inside a sequence
        # clipitem (the FIRST clipitem per file id — that's where OTIO
        # writes the populated form, with subsequent clipitems carrying
        # bare references), we MUST leave a `<file id="…"/>` reference
        # behind in the same position. Otherwise Premiere's xmeml
        # importer hits a clipitem with no <file> element at all,
        # can't resolve which asset to play, and aborts the entire
        # import with the unhelpful "The importer reported a generic
        # error." dialog. We track the original parent's children
        # list to preserve the insertion ORDER too (Premiere is
        # picky about element order inside <clipitem>).
        original_parent = None
        original_index = -1
        for parent in root.iter():
            kids = list(parent)
            if f_full in kids:
                original_parent = parent
                original_index = kids.index(f_full)
                break
        if original_parent is not None:
            original_parent.remove(f_full)
            # Only replace with a reference when the original parent
            # was a SEQUENCE clipitem (parent.tag == "clipitem") —
            # the full def could in theory have lived inside a
            # <resources>-style block on some adapter version, in
            # which case there's nothing to replace.
            if original_parent.tag == "clipitem":
                ref = ET.Element("file", {"id": fid})
                original_parent.insert(original_index, ref)
        clip_e.append(f_full)

        masters_written += 1

    if masters_written == 0:
        return 0

    # ── Wire every sequence <clipitem> back to its bin master ─────────
    # We add a <masterclipid> child (Premiere's official linkage) and
    # ensure the clipitem's <file> is a bare reference (`<file id="…"/>`)
    # rather than a duplicate of the full def we just hoisted. OTIO
    # already writes refs after the first clipitem; this is just
    # defensive in case a future adapter version regresses.
    for ci in sequence.iter("clipitem"):
        f_e = ci.find("file")
        if f_e is None:
            continue
        fid = f_e.get("id")
        if not fid or fid not in file_to_master:
            continue
        master_id = file_to_master[fid]
        # Only add a masterclipid if one isn't already present —
        # Premiere errors on duplicates.
        if ci.find("masterclipid") is None:
            mid_e = ET.Element("masterclipid")
            mid_e.text = master_id
            # Insert near the top so the ordering matches Premiere's
            # own xmeml exports (masterclipid before <name>).
            ci.insert(0, mid_e)
        # Strip any stray children from the file ref so it's a clean
        # `<file id="…"/>` self-closing reference. The full def now
        # lives on the bin master.
        for child in list(f_e):
            f_e.remove(child)
        f_e.text = None

    # ── Insert the bin BEFORE the sequence in <project>/<children> ────
    # ElementTree's Element supports `insert(index, element)`; we use
    # 0 so the bin sorts to the top of the project window.
    children.insert(0, bin_e)

    try:
        tree.write(str(out_path), encoding="UTF-8", xml_declaration=True)
    except Exception as e:
        print(f"  warn: xmeml bin layout patch built {masters_written} "
              f"master clip(s) but failed to write back to "
              f"{out_path.name} ({type(e).__name__}: {e}); the file is "
              "still valid xmeml, just without the bin reorganisation.",
              file=sys.stderr)
        return 0

    return masters_written


def write_fcpxml(timeline, out_path: Path) -> None:
    """Write the timeline as FCPXML 1.10+ (.fcpxml) — Resolve / FCP X path.

    OTIO discovers the writer via the `otio_fcpx_xml_adapter` package
    (declared in pyproject.toml's [fcpxml] extra). If it's missing we
    raise a clean install hint instead of letting OTIO's generic
    "no adapter for extension" message reach the user.

    After the adapter writes the file we run _patch_fcpxml_audio_shape()
    to inject:
      - per-asset hasAudio / audioSources / audioChannels / audioRate
        so Premiere and Resolve link the audio track without complaining
        about a channel-count mismatch.
      - per-format width / height / colorSpace pulled from the sequence
        settings the timeline was built against, so the imported
        sequence inherits the source's resolution and color space
        instead of dropping into the NLE's default 1080p / Rec.709.
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
    # Best-effort post-write patches. Each helper swallows its own
    # exceptions and only prints warnings, so the .fcpxml is always
    # left in a usable state even if a patch can't run.
    seq_meta = _sequence_meta_from_timeline(timeline)
    _patch_fcpxml_audio_shape(out_path, sequence_meta=seq_meta)
    # Inject <timeMap> retime elements for any range whose `speed` was
    # > 1.0. Idempotent and a no-op when speed_map is empty.
    speed_map = _speed_map_from_timeline(timeline)
    n_speed = _patch_fcpxml_speed(out_path, speed_map)
    if n_speed:
        print(f"  retime: patched {n_speed} timeMap element(s) "
              f"in {out_path.name}")


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

    After the adapter writes the file we run _patch_xmeml_sequence_format
    to inject the sequence's resolution / fps / pixel-aspect / audio
    shape, so Premiere creates a sequence that matches the source
    footage instead of falling back to its New Sequence default.
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
    seq_meta = _sequence_meta_from_timeline(timeline)
    _patch_xmeml_sequence_format(out_path, sequence_meta=seq_meta)
    # Inject Premiere's "Time Remap" filter for every retimed
    # clipitem. The filter is what makes Premiere display the "[N%]"
    # indicator and actually compress the source range; without it
    # the clip plays at 1.0× regardless of in/out vs start/end ratio.
    # The patcher also bounds-clamps the source range against the
    # file's actual frame count so timelapses can't run past the
    # media end and trip the diagonal "out of bounds" stripe.
    speed_map = _speed_map_from_timeline(timeline)
    n_speed = _patch_xmeml_speed(out_path, speed_map)
    if n_speed:
        print(f"  retime: patched {n_speed} timeremap filter(s) "
              f"in {out_path.name}")
    # Reorganise the project layout — hoist source files into a
    # "Footage" bin and link sequence clipitems to bin master clips
    # via <masterclipid>. Beyond the obvious organisational win, the
    # masterclipid linkage is what Premiere's "Replace with clip >
    # From Bin" workaround restores when users hit it manually to
    # repair time-remapped clipitems; running it at export time means
    # the user never needs to.
    n_masters = _patch_xmeml_bin_layout(out_path, bin_name="Footage")
    if n_masters:
        print(f"  layout: hoisted {n_masters} source file(s) into a "
              f"'Footage' bin in {out_path.name}")


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
        "--frame-rate", type=str, default="auto",
        help="Timeline frame rate. Default 'auto' inherits from the EDL's "
             "primary source (the one with the most runtime). Pass a "
             "number (23.976, 24, 25, 29.97, 30, 60) to override. All "
             "cuts are snapped to whole frames at the chosen rate.",
    )
    # Captions sidecar — generated by default so every NLE handoff
    # ships with master.srt next to the timeline XML. The flag exists
    # so air-gapped runs without transcripts (or experiments where
    # subtitles are explicitly out of scope) can opt out.
    ap.add_argument(
        "--no-srt", action="store_true",
        help="Skip the master.srt sidecar (default: always emit it next "
             "to the EDL so the NLE imports captions on a track).",
    )
    ap.add_argument(
        "--srt-output", type=Path, default=None,
        help="Override SRT path. Default: <edl_dir>/master.srt — the "
             "path subtitles.md and parent_rules.md both reference.",
    )
    args = ap.parse_args()

    edl_path = args.edl.resolve()
    if not edl_path.exists():
        sys.exit(f"edl not found: {edl_path}")

    # Read the EDL as raw text so we can normalize a single recurring
    # editor-agent typo before json.loads sees it: the canonical retime
    # key is `speed` (see SKILL.md "Time-squeezing"), but agents who
    # named their beats `..._TIMELAPSE` routinely write `timelapse_speed`
    # instead. The exporter would otherwise read `r.get("speed")` as
    # None and silently drop every retime, leaving the operator with a
    # 1x cut and no obvious indication of why. A textual rename here
    # is bulletproof — it can't accidentally hit JSON values because
    # both alternates are bare keys without quote characters of their
    # own, and the surrounding `"..."` makes the match unambiguous.
    edl_text = edl_path.read_text(encoding="utf-8")
    edl_text = edl_text.replace('"timelapse_speed"', '"speed"')
    edl = json.loads(edl_text)

    # ── Resolve the sequence shape from the EDL's primary source ──────
    # Picks the source with the most runtime in the cut, ffprobes it for
    # resolution / fps / color space / audio shape. The result is fed to
    # build_timeline (so the cut snap-rate matches), to the post-write
    # patchers (so the sequence settings in the XML match the source),
    # and to the CLI banner (so the user can see what they got).
    seq_meta = _resolve_sequence_settings(edl, args.frame_rate)
    timeline_fps = float(seq_meta["video_fps"])

    # Build ONCE — both writers consume the same otio.schema.Timeline.
    # ffprobe + frame-snapping costs are paid here, not per-dialect.
    timeline = build_timeline(
        edl, frame_rate=timeline_fps, sequence_settings=seq_meta,
    )

    fcpx_out, prxml_out = _resolve_output_paths(args.output.resolve(), args.targets)

    n_clips = sum(
        1 for t in timeline.tracks for c in t
        if c.__class__.__name__ == "Clip"
    )
    n_trans = sum(
        1 for t in timeline.tracks for c in t
        if c.__class__.__name__ == "Transition"
    )
    primary_label = (seq_meta.get("primary_source")
                     or "<no primary — using defaults>")
    print(f"timeline built: {n_clips} clips, {n_trans} transitions")
    print(f"  sequence:    {seq_meta['video_width']}x{seq_meta['video_height']} "
          f"@ {timeline_fps:.3f} fps, {seq_meta['fcpxml_color_space']}")
    print(f"  audio:       {seq_meta['audio_channels']}ch @ "
          f"{seq_meta['audio_rate']} Hz")
    # Resolution + fps come from MAX-across-sources (so the sequence
    # never under-fits any clip, see _resolve_sequence_settings); color
    # / audio shape inherit from the runtime-dominant primary source.
    print(f"  fps + res:   max-across-sources (sequence >= every clip)")
    print(f"  color/audio: inherited from primary source: {primary_label}")

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
            # Full traceback to stderr — generic ValueErrors etc. coming
            # out of the OTIO fcpxml adapter are useless without the
            # frame that fired them. The Premiere writer still runs
            # below either way, so the user gets ONE working export
            # plus enough info to file a bug on the dud one.
            import traceback as _tb
            print(f"  [fcpxml]   FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr)
            _tb.print_exc(file=sys.stderr)

    if prxml_out is not None:
        try:
            write_premiere_xml(timeline, prxml_out)
            kb = prxml_out.stat().st_size / 1024
            print(f"  [premiere] {prxml_out}  ({kb:.1f} KB)  "
                  f"-> {_TARGET_INFO['premiere']['opens_in']}")
        except SystemExit:
            raise
        except Exception as e:
            import traceback as _tb
            print(f"  [premiere] FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr)
            _tb.print_exc(file=sys.stderr)

    # ── Captions sidecar (always-on unless --no-srt) ──────────────────
    # Premiere / Resolve / FCP X all import SRT cleanly onto a captions
    # track that the editor restyles in their own caption panel. We
    # write Premiere-friendly SRT (UTF-8 + CRLF) per `helpers/build_srt.py`.
    # A lane import here, deferred to keep the FCPXML/Premiere paths
    # standalone if some operator deletes build_srt.py from a slim
    # deployment (the captions branch just won't run, the XML still ships).
    if not args.no_srt:
        try:
            from build_srt import build_master_srt  # local sibling
            srt_out = (args.srt_output.resolve()
                       if args.srt_output is not None
                       else edl_path.parent / "master.srt")
            build_master_srt(edl, edl_path.parent, srt_out)
            if srt_out.exists():
                kb = srt_out.stat().st_size / 1024
                print(f"  [srt]      {srt_out}  ({kb:.1f} KB)  "
                      f"-> Premiere / Resolve / FCP X captions track")
        except ImportError:
            print("  [srt]      SKIPPED: helpers/build_srt.py not "
                  "importable (deploy missing the captions helper).",
                  file=sys.stderr)
        except Exception as e:
            import traceback as _tb
            print(f"  [srt]      FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr)
            _tb.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
