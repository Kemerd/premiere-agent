"""Pack the three lane outputs into Claude-readable timeline markdowns.

After preprocess[_batch].py finishes, every video has THREE JSON files:

    <edit>/transcripts/<stem>.json   — parakeet_onnx_lane (speech, word level)
    <edit>/audio_tags/<stem>.json    — audio_lane         (CLAP vocab events)
    <edit>/visual_caps/<stem>.json   — visual_lane        (Florence-2 captions)

This script fans those out into FOUR markdown timelines that the SKILL
points Claude at:

    <edit>/speech_timeline.md   — phrase-grouped transcripts (per file)
    <edit>/audio_timeline.md    — vocab-scored sound events  (per file)
    <edit>/visual_timeline.md   — 1-second captions, dedup'd (per file)
    <edit>/merged_timeline.md   — all three lanes interleaved by timestamp

`merged_timeline.md` is the **default reading surface** for the editor
sub-agent: one file, every event in chronological order, so the agent
can plan cuts in a single pass instead of cross-referencing three lanes
by hand. The per-lane files remain on disk as drill-down references
(use them when you need only one lane, or when an ambiguity in the
merged view warrants zooming in on the raw per-lane data).

The interleaved merged view looks like:

    [00:12:04] "okay now we're going to drill the pilot holes"
    [00:12:09] (audio: cordless drill 0.42, drill press 0.31)
    [00:12:09] visual: a person holding a cordless drill above a metal panel
    [00:12:18] "good, pass me the deburring tool"

The phrase-grouping logic for speech_timeline.md is reused verbatim from
the old pack_transcripts.py so existing prompts that reference packed
transcripts continue to work — only the filename changed.

The audio lane emits PANNs-shape (label, score) events scored by a CLAP
encoder against a vocabulary list (see helpers/audio_lane.py). For
backward compatibility with caches still on disk from the abandoned
Audio Flamingo 3 migration, a separate caption-shape renderer is kept
behind a `model` sniff and removed once those caches age out.

CLI:
    python helpers/pack_timelines.py --edit-dir <dir> [--no-merge]

By default emits all four files. Pass `--no-merge` to skip the unified
view (rare — only useful when the per-lane files are all you want).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Time formatting helpers (inlined from the deprecated pack_transcripts.py)
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """Format a time in seconds as "NNN.NN" with fixed 6-char width.

    The fixed width keeps `[start-end]` columns aligned in the markdown
    so the editor sub-agent can scan vertically.
    """
    return f"{seconds:06.2f}"


def format_duration(seconds: float) -> str:
    """Format a duration as "Ms" (sub-minute) or "Mm SSs" (longer)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m}m {s:04.1f}s"


# ---------------------------------------------------------------------------
# Phrase grouper — walks the canonical word stream and breaks into phrases
# on any silence >= silence_threshold OR a speaker change. Word-boundary
# precision falls out of the Parakeet TDT decoder's native per-token
# timestamps (see helpers/parakeet_onnx_lane.py); Hard Rule 8 + 6 still
# apply.
# ---------------------------------------------------------------------------

def group_into_phrases(
    words: list[dict],
    silence_threshold: float = 0.5,
) -> list[dict]:
    """Walk the canonical word list, break into phrases. Returns
    [{start, end, text, speaker_id}, ...].

    Word entries have type 'word', 'spacing', or 'audio_event'. We keep
    'word' / 'audio_event' in phrase text; 'spacing' carries the silence
    info via its start/end gap.
    """
    phrases: list[dict] = []
    current_words: list[dict] = []
    current_start: float | None = None
    current_speaker: str | None = None

    def flush() -> None:
        nonlocal current_words, current_start, current_speaker
        if not current_words:
            return
        text_parts: list[str] = []
        for w in current_words:
            t = w.get("type", "word")
            raw = (w.get("text") or "").strip()
            if not raw:
                continue
            if t == "audio_event" and not raw.startswith("("):
                raw = f"({raw})"
            text_parts.append(raw)
        if not text_parts:
            current_words = []
            current_start = None
            current_speaker = None
            return
        text = " ".join(text_parts)
        # Pull punctuation back against its preceding word — " ," → ","
        # without dragging in a real tokenizer.
        text = (text.replace(" ,", ",").replace(" .", ".")
                    .replace(" ?", "?").replace(" !", "!"))
        end_time = current_words[-1].get(
            "end", current_words[-1].get("start", current_start or 0.0),
        )
        phrases.append({
            "start": current_start,
            "end": end_time,
            "text": text,
            "speaker_id": current_speaker,
        })
        current_words = []
        current_start = None
        current_speaker = None

    prev_end: float | None = None

    for w in words:
        t = w.get("type", "word")
        if t == "spacing":
            start = w.get("start")
            end = w.get("end")
            if start is not None and end is not None:
                if (end - start) >= silence_threshold:
                    flush()
            continue

        # 'word' or 'audio_event'
        start = w.get("start")
        if start is None:
            continue
        speaker = w.get("speaker_id")

        if (current_speaker is not None and speaker is not None
                and speaker != current_speaker):
            flush()

        if prev_end is not None and start - prev_end >= silence_threshold:
            flush()

        if current_start is None:
            current_start = start
            current_speaker = speaker
        current_words.append(w)
        prev_end = w.get("end", start)

    flush()
    return phrases


# ---------------------------------------------------------------------------
# Time format helper for merged_timeline (HH:MM:SS not seconds)
# ---------------------------------------------------------------------------

def _hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS for the merged timeline."""
    s = max(0.0, float(seconds))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ---------------------------------------------------------------------------
# Lane 1 — Speech timeline (phrase-grouped from words)
# ---------------------------------------------------------------------------

def _pack_speech(transcripts_dir: Path, silence_threshold: float) -> str:
    """Render speech_timeline.md content from <edit>/transcripts/*.json."""
    json_files = sorted(transcripts_dir.glob("*.json"))
    lines: list[str] = []
    lines.append("# Speech timeline (Whisper)")
    lines.append("")
    lines.append(
        f"Phrase-level transcript, broken on silence ≥ {silence_threshold:.1f}s "
        f"or speaker change. `[start-end]` ranges are seconds from clip start."
    )
    lines.append("")

    if not json_files:
        lines.append("_no whisper output found in transcripts/_")
        return "\n".join(lines)

    for p in json_files:
        data = json.loads(p.read_text(encoding="utf-8"))
        words = data.get("words", [])
        phrases = group_into_phrases(words, silence_threshold)
        duration = data.get("duration") or (
            (phrases[-1]["end"] - phrases[0]["start"]) if phrases else 0.0
        )
        lines.append(
            f"## {p.stem}  (duration: {format_duration(duration)}, "
            f"{len(phrases)} phrases)"
        )
        if not phrases:
            lines.append("  _no speech detected_")
            lines.append("")
            continue
        for ph in phrases:
            spk = ph.get("speaker_id")
            if spk is not None:
                spk_str = str(spk)
                if spk_str.startswith("speaker_"):
                    spk_str = spk_str[len("speaker_"):]
                spk_tag = f" S{spk_str}"
            else:
                spk_tag = ""
            lines.append(
                f"  [{format_time(ph['start'])}-{format_time(ph['end'])}]"
                f"{spk_tag} {ph['text']}"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lane 2 — Audio events timeline
#
# Canonical CLAP audio_lane JSON shape:
#   {
#     model:      "Xenova/clap-htsat-unfused",
#     vocab_sha:  "<hex>",
#     vocab_size: <int>,
#     window_s:   10.0,
#     hop_s:      5.0,
#     threshold:  0.10,
#     top_k:      5,
#     duration:   <seconds>,
#     events: [{start, end, label, score}, ...]
#   }
#
# Legacy AF3 caption shape (`captions` key, free-form text per chunk) is
# detected via the `model` field and rendered through a fallback path so
# stale caches don't blow up the packer. Once those caches age out the
# fallback can be deleted.
# ---------------------------------------------------------------------------

def _render_audio_events(events: list[dict], lines: list[str]) -> None:
    """Render CLAP (label, score) events grouped by (start, end) range.

    Co-occurring labels in the same window collapse onto one line so a
    busy 10s window with 4 simultaneous tags reads as
        [start-end] (drill 0.42, hammer 0.31, sandpaper 0.28, ...)
    rather than 4 separate rows. Within each row labels are sorted by
    descending score so the strongest match leads.
    """
    by_range: dict[tuple[float, float], list[tuple[float, str]]] = {}
    for ev in events:
        key = (round(float(ev["start"]), 2), round(float(ev["end"]), 2))
        by_range.setdefault(key, []).append(
            (float(ev.get("score", 0.0)), str(ev.get("label", "?")))
        )
    for (s, e) in sorted(by_range.keys()):
        labels = sorted(by_range[(s, e)], key=lambda x: -x[0])
        label_str = ", ".join(
            f"{lab} {sc:.2f}" for sc, lab in labels[:5]
        )
        lines.append(
            f"  [{format_time(s)}-{format_time(e)}] ({label_str})"
        )


def _render_audio_captions_legacy(captions: list[dict], lines: list[str]) -> None:
    """Render free-form per-chunk captions from a stale AF3 cache.

    Tagged `[legacy AF3]` so the agent can tell at a glance the cache
    is from the abandoned Audio Flamingo 3 migration and re-running
    the audio lane will refresh it into the canonical CLAP shape.
    """
    for c in sorted(captions, key=lambda x: float(x.get("start", 0.0))):
        s = float(c.get("start", 0.0))
        e = float(c.get("end", s))
        text = (c.get("text") or "").strip().replace("\n", " ")
        if not text:
            text = "_(no caption)_"
        lines.append(f"  [{format_time(s)}-{format_time(e)}] {text} [legacy AF3]")


def _pack_audio(audio_tags_dir: Path) -> str:
    """Render audio_timeline.md content from <edit>/audio_tags/*.json."""
    json_files = sorted(audio_tags_dir.glob("*.json"))
    lines: list[str] = []
    lines.append("# Audio timeline (CLAP zero-shot events)")
    lines.append("")
    lines.append(
        "Per-window sound events scored by LAION CLAP against a vocabulary "
        "list. Each line: `[start-end] (label score, label score, ...)`. "
        "Scores are cosine similarities (typically 0.10-0.45 — higher is "
        "more confident). CLAP is much sharper than the abandoned PANNs "
        "ontology but still noisy on edge classes; if labels look wrong "
        "for THIS video, write a curated `<edit>/audio_vocab.txt` and "
        "re-run `python helpers/audio_lane.py <video> --vocab "
        "<edit>/audio_vocab.txt --force` for sharper, content-aware tags."
    )
    lines.append("")

    if not json_files:
        lines.append("_no audio_tags found — did audio_lane run?_")
        return "\n".join(lines)

    for p in json_files:
        data = json.loads(p.read_text(encoding="utf-8"))
        duration = data.get("duration", 0.0)
        model = str(data.get("model", ""))

        # Shape detection by `model` field. The CLAP lane writes
        # `Xenova/clap-htsat-unfused` or `Xenova/larger_clap_general`;
        # the abandoned AF3 lane wrote `nvidia/audio-flamingo-3-hf`.
        # Anything else with a `captions` key gets the legacy renderer
        # too (defensive: third-party forks may have other captioning
        # writers).
        if "captions" in data and isinstance(data["captions"], list) and not data.get("events"):
            captions = data["captions"]
            n = len(captions)
            lines.append(
                f"## {p.stem}  (duration: {format_duration(duration)}, "
                f"{n} legacy AF3 caption{'s' if n != 1 else ''} — "
                f"re-run audio_lane to refresh)"
            )
            if not captions:
                lines.append("  _no captions emitted_")
                lines.append("")
                continue
            _render_audio_captions_legacy(captions, lines)
            lines.append("")
            continue

        events = data.get("events", [])
        vocab_size = data.get("vocab_size")
        header_extra = (
            f", vocab={vocab_size}" if isinstance(vocab_size, int) else ""
        )
        lines.append(
            f"## {p.stem}  (duration: {format_duration(duration)}, "
            f"{len(events)} event{'s' if len(events) != 1 else ''}"
            f"{header_extra})"
        )
        if not events:
            lines.append("  _no audio events above threshold_")
            lines.append("")
            continue
        _render_audio_events(events, lines)
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lane 3 — Visual captions timeline
# ---------------------------------------------------------------------------

def _pack_visual(visual_caps_dir: Path) -> str:
    """Render visual_timeline.md from <edit>/visual_caps/*.json.

    Uses captions_dedup so consecutive identical frames collapse to (same).
    Falls back to raw captions if dedup absent (older JSON shape).
    """
    json_files = sorted(visual_caps_dir.glob("*.json"))
    lines: list[str] = []
    lines.append("# Visual timeline (Florence-2 detailed captions @ 1 fps)")
    lines.append("")
    lines.append(
        "One caption per second; consecutive identical captions collapse to "
        "`(same)`. Use timestamps to find shots / B-roll candidates / "
        "match cuts."
    )
    lines.append("")

    if not json_files:
        lines.append("_no visual_caps found — did visual_lane run?_")
        return "\n".join(lines)

    for p in json_files:
        data = json.loads(p.read_text(encoding="utf-8"))
        captions = data.get("captions_dedup") or data.get("captions") or []
        duration = data.get("duration", 0.0)
        fps = data.get("fps", 1)
        lines.append(
            f"## {p.stem}  (duration: {format_duration(duration)}, "
            f"{len(captions)} caps @ {fps} fps)"
        )
        if not captions:
            lines.append("  _no visual captions emitted_")
            lines.append("")
            continue
        for c in captions:
            t = float(c.get("t", 0.0))
            text = (c.get("text") or "").strip().replace("\n", " ")
            lines.append(f"  [{format_time(t)}] {text}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional merge — interleave all three by timestamp into one stream.
# ---------------------------------------------------------------------------

def _build_merged(edit_dir: Path) -> str:
    """Walk all three caches in lock-step per video and emit one
    timestamp-sorted stream per video.

    The merged view is for moments when the editor wants the full
    multi-modal context at a glance — e.g. "what's happening at 12:04?".
    For pure speech-driven editing the speech_timeline.md alone is
    usually denser and easier to scan.
    """
    transcripts = edit_dir / "transcripts"
    audio_tags = edit_dir / "audio_tags"
    visual_caps = edit_dir / "visual_caps"

    # Use the union of stems found in any directory. Lanes can be skipped
    # individually so we don't require all three to be present.
    stems: set[str] = set()
    for d in (transcripts, audio_tags, visual_caps):
        if d.is_dir():
            stems.update(p.stem for p in d.glob("*.json"))
    if not stems:
        return "# Merged timeline\n\n_no lane outputs to merge_\n"

    out: list[str] = []
    out.append("# Merged timeline")
    out.append("")
    out.append(
        "All three lanes interleaved by timestamp (HH:MM:SS). "
        "Speech lines are quoted; audio events use `(audio: ...)`; "
        "visual captions use `visual:` prefix."
    )
    out.append("")

    for stem in sorted(stems):
        out.append(f"## {stem}")
        events: list[tuple[float, str]] = []  # (t, line)

        # ── Speech: take phrases (already grouped) at their start time ──
        sp = transcripts / f"{stem}.json"
        if sp.exists():
            data = json.loads(sp.read_text(encoding="utf-8"))
            phrases = group_into_phrases(data.get("words", []), 0.5)
            for ph in phrases:
                events.append((
                    float(ph["start"]),
                    f"[{_hms(ph['start'])}] \"{ph['text']}\"",
                ))

        # ── Audio: one line per merged CLAP event range. Co-occurring   ──
        # labels in the same window collapse to one comma-separated tag  ──
        # so the merged stream stays scannable.                          ──
        ap = audio_tags / f"{stem}.json"
        if ap.exists():
            data = json.loads(ap.read_text(encoding="utf-8"))
            audio_events = data.get("events")
            if isinstance(audio_events, list) and audio_events:
                by_range: dict[tuple[float, float], list[tuple[float, str]]] = {}
                for ev in audio_events:
                    key = (round(float(ev["start"]), 2),
                           round(float(ev["end"]), 2))
                    by_range.setdefault(key, []).append(
                        (float(ev.get("score", 0.0)), str(ev.get("label", "?")))
                    )
                for (s, _e), labels in by_range.items():
                    labels.sort(key=lambda x: -x[0])
                    label_str = ", ".join(lab for _sc, lab in labels[:5])
                    events.append((
                        s, f"[{_hms(s)}] (audio: {label_str})",
                    ))
            else:
                # Legacy AF3 cache fallback — render free-form chunk
                # captions inline so a half-migrated edit folder stays
                # readable. Once caches age out this branch can be
                # removed alongside the legacy renderer in _pack_audio.
                for c in data.get("captions", []) or []:
                    s = float(c.get("start", 0.0))
                    text = (c.get("text") or "").strip().replace("\n", " ")
                    if not text:
                        continue
                    events.append((
                        s, f"[{_hms(s)}] (audio: {text})",
                    ))

        # ── Visual: dedup'd captions ──
        vp = visual_caps / f"{stem}.json"
        if vp.exists():
            data = json.loads(vp.read_text(encoding="utf-8"))
            caps = data.get("captions_dedup") or data.get("captions") or []
            for c in caps:
                t = float(c.get("t", 0.0))
                text = (c.get("text") or "").strip().replace("\n", " ")
                if text == "(same)":
                    continue   # don't pollute merged view with dedup markers
                events.append((t, f"[{_hms(t)}] visual: {text}"))

        events.sort(key=lambda x: x[0])
        if not events:
            out.append("  _no events_")
        else:
            for _t, line in events:
                out.append(f"  {line}")
        out.append("")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pack the three lane outputs into Claude-readable "
                    "timeline markdowns.",
    )
    ap.add_argument("--edit-dir", type=Path, required=True,
                    help="Edit directory containing transcripts/, "
                         "audio_tags/, visual_caps/")
    ap.add_argument("--silence-threshold", type=float, default=0.5,
                    help="Break speech phrases on silences >= this (default 0.5s)")
    # merged_timeline.md is the editor sub-agent's default reading
    # surface (one file, all three lanes interleaved by timestamp), so
    # we emit it by default. `--no-merge` is the opt-out for the rare
    # case where only the per-lane files are wanted.
    ap.add_argument("--no-merge", dest="merge", action="store_false",
                    default=True,
                    help="Skip merged_timeline.md (the unified, "
                         "timestamp-interleaved view of all 3 lanes). "
                         "Default is to emit it.")
    ap.add_argument("--merge", dest="merge", action="store_true",
                    help="(deprecated, no-op) merged_timeline.md is now "
                         "emitted by default; this flag is kept for "
                         "backward compatibility.")
    args = ap.parse_args()

    edit_dir = args.edit_dir.resolve()
    if not edit_dir.is_dir():
        sys.exit(f"[pack_timelines] not a directory: {edit_dir}")

    transcripts = edit_dir / "transcripts"
    audio_tags = edit_dir / "audio_tags"
    visual_caps = edit_dir / "visual_caps"

    # Speech is the only lane considered "required" for a basic edit.
    # The other two are advisory context — if either is missing we still
    # emit the markdown with a "no data" stub so the editor sub-agent
    # gets a consistent file layout.
    if not transcripts.is_dir():
        print(f"[pack_timelines] WARN: no transcripts/ at {transcripts}",
              file=sys.stderr)

    out_speech = edit_dir / "speech_timeline.md"
    out_audio = edit_dir / "audio_timeline.md"
    out_visual = edit_dir / "visual_timeline.md"

    out_speech.write_text(
        _pack_speech(transcripts, args.silence_threshold), encoding="utf-8",
    )
    out_audio.write_text(_pack_audio(audio_tags), encoding="utf-8")
    out_visual.write_text(_pack_visual(visual_caps), encoding="utf-8")

    written = [out_speech, out_audio, out_visual]

    if args.merge:
        out_merged = edit_dir / "merged_timeline.md"
        out_merged.write_text(_build_merged(edit_dir), encoding="utf-8")
        written.append(out_merged)

    for p in written:
        kb = p.stat().st_size / 1024
        print(f"  wrote {p.name}  ({kb:.1f} KB)")


if __name__ == "__main__":
    main()
