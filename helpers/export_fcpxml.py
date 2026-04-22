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
_FCPXML_COLORSPACE_TABLE: dict[tuple[str, str], str] = {
    # SDR Rec. 709 (the 99% case for HD camera footage)
    ("bt709", "bt709"):           "1-1-1 (Rec. 709)",
    ("bt709", "unknown"):         "1-1-1 (Rec. 709)",
    ("smpte170m", "smpte170m"):   "1-1-1 (Rec. 709)",
    # SD NTSC / PAL — mapped to Rec. 709 by FCP X for SD masters too
    ("smpte170m", "bt470bg"):     "1-1-1 (Rec. 709)",
    ("bt470bg", "bt470bg"):       "1-1-1 (Rec. 709)",
    # Rec. 2020 SDR / HDR variants
    ("bt2020-10", "bt2020"):      "9-1-1 (Rec. 2020)",
    ("bt2020-10", "bt2020nc"):    "9-1-1 (Rec. 2020)",
    ("bt2020-12", "bt2020"):      "9-1-1 (Rec. 2020)",
    ("arib-std-b67", "bt2020"):   "9-16-9 (Rec. 2020 HLG)",
    ("arib-std-b67", "bt2020nc"): "9-16-9 (Rec. 2020 HLG)",
    ("smpte2084", "bt2020"):      "9-18-9 (Rec. 2020 PQ)",
    ("smpte2084", "bt2020nc"):    "9-18-9 (Rec. 2020 PQ)",
}
_DEFAULT_FCPXML_COLORSPACE = "1-1-1 (Rec. 709)"


def _classify_colorspace(transfer: str, primaries: str) -> str:
    """Map ffprobe color tags to an FCPXML 1.10+ colorSpace string.

    Defaults to Rec. 709 — Premiere and Resolve will accept that for any
    SDR source even when the camera tagged things weirdly, and HDR
    metadata gets a proper mapping when ffprobe surfaces it.
    """
    t = (transfer or "unknown").lower()
    p = (primaries or "unknown").lower()
    return _FCPXML_COLORSPACE_TABLE.get((t, p), _DEFAULT_FCPXML_COLORSPACE)


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
# to whatever default the user's NLE happens to have. We avoid both by
# picking the source that contributes the MOST runtime to the EDL and
# inheriting its resolution / fps / color space onto the sequence.
#
# The frame_rate argument from the CLI is honored as an override when
# explicit; "auto" / None means "match the dominant source".
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


def _resolve_sequence_settings(edl: dict, frame_rate_arg) -> dict:
    """Collapse the EDL's source set into one sequence-shape dict.

    Returns a dict matching the _probe_source_meta() shape PLUS an
    "fcpxml_color_space" string and a final "video_fps" guaranteed > 0.
    The caller hands this dict to build_timeline / the writers / the
    post-write patchers, all of which need the same numbers to agree.

    `frame_rate_arg` semantics:
      - None or "auto"      -> inherit from primary source (then default)
      - numeric (float/int) -> hard override; primary's fps is ignored
    """
    primary_key = _pick_primary_source(edl)
    sources = edl.get("sources") or {}

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
    primary_meta = None
    if primary_key and primary_key in sources:
        try:
            primary_meta = _probe_source_meta(Path(sources[primary_key]).resolve())
            for k in ("video_width", "video_height", "audio_channels",
                      "audio_rate", "color_primaries", "color_transfer",
                      "color_space", "pixel_aspect_ratio"):
                if primary_meta.get(k) is not None:
                    seq[k] = primary_meta[k]
            if primary_meta.get("video_fps"):
                seq["video_fps"] = primary_meta["video_fps"]
        except Exception as e:
            print(f"  warn: primary source probe failed for "
                  f"{primary_key} ({type(e).__name__}: {e}); using defaults.",
                  file=sys.stderr)

    # CLI override — "auto" / None means "use what we just resolved".
    # Any positive numeric value wins over the source's native fps.
    if frame_rate_arg is not None and frame_rate_arg != "auto":
        try:
            user_fps = float(frame_rate_arg)
            if user_fps > 0:
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
        - beat / quote    : copied to clip metadata for editor reference

    `sequence_settings` (optional) carries the resolved sequence-shape
    dict from _resolve_sequence_settings() — width/height/fps/color
    space/audio rate. We stash it on the Timeline's metadata so the
    post-write patchers can find it without re-probing. When None we
    skip the stash; the patchers will fall back to per-asset probing.
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

    # Stash the resolved sequence-shape dict on the timeline so the
    # post-write patchers (_patch_fcpxml_sequence_format /
    # _patch_xmeml_sequence_format) can read width/height/colorSpace/
    # audio shape without re-resolving everything from the EDL.
    if sequence_settings:
        timeline.metadata["video-use-premiere"] = {
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
        # available_range is REQUIRED by both adapters — it writes into
        # the asset declaration. We probe the source ONCE for duration +
        # audio/video shape and cache it; see _probe_source_meta().
        #
        # The audio shape (channel count + sample rate) matters: Premiere
        # validates "the clip was created with N audio channel(s)" against
        # the actual file on link, and refuses if they disagree. We pre-
        # populate the fcp_xml metadata so the FCP7 xmeml writer emits
        # <channelcount>/<samplerate> matching the real source.
        src_path_resolved = Path(src_path).resolve()
        src_meta = _probe_source_meta(src_path_resolved)
        media_dur_s = src_meta["duration_s"]
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
            ref.metadata["video-use-premiere"] = {
                "audio_channels": src_meta["audio_channels"],
                "audio_rate": src_meta["audio_rate"],
                "has_audio": src_meta["has_audio"],
                "has_video": src_meta["has_video"],
            }
            return ref

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
    timeline.metadata['video-use-premiere']['sequence']; both writers
    forward that to their post-write patchers without having to
    re-resolve the EDL.
    """
    try:
        return (timeline.metadata.get("video-use-premiere") or {}).get("sequence")
    except Exception:
        return None


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
    # Best-effort post-write patches. Both helpers swallow their own
    # exceptions and only print warnings, so the .fcpxml is always left
    # in a usable state even if the patches can't run.
    seq_meta = _sequence_meta_from_timeline(timeline)
    _patch_fcpxml_audio_shape(out_path, sequence_meta=seq_meta)


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
    args = ap.parse_args()

    edl_path = args.edl.resolve()
    if not edl_path.exists():
        sys.exit(f"edl not found: {edl_path}")

    edl = json.loads(edl_path.read_text(encoding="utf-8"))

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
    print(f"  inherited from primary source: {primary_label}")

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
