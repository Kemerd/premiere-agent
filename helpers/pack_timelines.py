"""Pack the three lane outputs into Claude-readable timeline markdowns.

After preprocess[_batch].py finishes, every video has THREE JSON files:

    <edit>/transcripts/<stem>.json   — parakeet_onnx_lane (speech, word level)
    <edit>/audio_tags/<stem>.json    — audio_lane         (CLAP vocab events)
    <edit>/visual_caps/<stem>.json   — visual_lane        (Florence-2 captions)

This script fans those out into FOUR markdown timelines that the SKILL
points the agent at:

    <edit>/merged_timeline.md       — DEFAULT reading surface; speech +
                                      audio + visual interleaved by
                                      timestamp into one file (the
                                      "single triangulated picture" the
                                      agent reasons over)
    <edit>/speech_timeline.md       — phrase-grouped transcripts (lane 1, drill-down)
    <edit>/audio_timeline.md        — vocab-scored sound events  (lane 2, drill-down)
    <edit>/visual_timeline.md       — 1-second captions, dedup'd (lane 3, drill-down)

The agent reads **`merged_timeline.md` end-to-end** every session — one
file, every event in chronological order, all three lanes triangulated.
The per-lane drill-down files stay on disk for the moments the merged
view is ambiguous and the agent zooms in on one lane (word-level
timing detail in speech, the full 1-fps caption stream including
`(same)` repeats in visual, full per-window CLAP scoring in audio).

For sub-second word-precise EDL anchors, the agent uses
`helpers/find_quote.py` against the underlying transcript JSON — the
merged view is intentionally rounded to whole seconds for token
economy, and `find_quote.py` returns first/last/prev/next word
boundaries with millisecond precision plus a `cut_window` (safe_in_s,
safe_out_s) ready for the pacing-preset margin clamp.

The merged view looks like:

    0:02 [Workbench hand tools laid brown wooden surface.]
    0:03-0:18 "okay now we're going to drill the pilot holes"
    0:12 (cordless drill 0.42, drill press 0.31)
    0:12 [a person holding a cordless drill above a metal panel]
    0:13 + [hand reaching for a deburring tool on the workbench]
    0:19-0:21 "good, pass me the deburring tool"

Bracket conventions are the type marker — `"..."` for speech, `(...)`
for audio events, `[...]` for visual captions — so the agent can scan
without per-line `speech:` / `audio:` / `visual:` prefixes burning
tokens. Speech phrases carry an outer-aligned `M:SS-M:SS` range
(start floor, end ceil) — the integer span always fully encloses the
underlying float [start, end] in `transcripts/<stem>.json`, so any
range read out of the merged file can be passed verbatim to
`find_quote.py --range` without off-by-one fences. Audio and visual
events sit on point timestamps (their windows / 1-fps stride don't
need range display).

The phrase-grouping logic for `speech_timeline.md` is reused verbatim
from the old pack_transcripts.py so existing prompts that reference
packed transcripts continue to work — only the filename changed.

The audio lane emits PANNs-shape (label, score) events scored by a
CLAP encoder against a vocabulary list (see helpers/audio_lane.py).
For backward compatibility with caches still on disk from the
abandoned Audio Flamingo 3 migration, a separate caption-shape
renderer is kept behind a `model` sniff and removed once those caches
age out.

CLI:
    python helpers/pack_timelines.py --edit-dir <dir> [--no-merge]

By default emits all four files. Pass `--no-merge` to skip the merged
interleaved view (rare — only useful when the per-lane files are all
you want).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path


# ---------------------------------------------------------------------------
# Time formatting helpers
#
# Compact M:SS / H:MM:SS form across all four timeline files. The hour
# slot is omitted whenever it would be zero (a 90-second clip reads as
# `1:30`, not `0:01:30`) and the leading minute is unpadded so 1- and
# 2-digit minute values both look natural (`5:42` not `05:42`,
# `10:32` not `10:32`). Surrounding `[…]` brackets that the older
# pack_timelines used are dropped on purpose now: every event line
# already wears a type-disambiguating wrapper (`"…"` for speech,
# `(…)` for audio, `[…]` for visual), so an extra bracket pair on the
# timestamp would just be noise the editor sub-agent has to skip.
#
# All timeline writers below funnel through `_fmt_ts` for a single
# event timestamp and `_fmt_range` for `start-end` spans. Use whole
# seconds throughout — sub-second precision was used by the legacy
# format but the editor works in whole-second cut planning anyway, and
# whole seconds round-trip cleanly to the M:SS display without ugly
# decimal padding.
# ---------------------------------------------------------------------------

def _fmt_ts(seconds: float) -> str:
    """Compact `M:SS` or `H:MM:SS` (drops the hour when zero).

    Uses banker's rounding via `int(round(...))` — used for visual /
    audio events whose timestamps already land on integer seconds (1
    fps caption stride, 5/10s CLAP windows). For speech phrase ranges
    use `_fmt_floor` / `_fmt_ceil` instead so the integer range fully
    encloses the underlying float span.
    """
    s = max(0, int(round(float(seconds))))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


# ── Outer-aligned formatters for speech phrase spans ──
# Speech phrase boundaries come out of Parakeet at sub-second precision
# (e.g. 2.16s start, 17.84s end). Rendering those as `2:02-0:18` via
# `_fmt_ts` would lose end information either way (round-down loses the
# last 0.84s of speech, round-half is non-monotonic). The editor sub-
# agent and `helpers/find_quote.py` need a CONTAINS relationship: the
# integer M:SS-M:SS span shown in `speech_timeline.md` must fully
# enclose the float [start, end] span in `transcripts/<stem>.json`,
# so any quote located by integer range is guaranteed to be inside the
# transcript word stream within those same seconds. Floor the start,
# ceil the end → integer range always wraps the float range, no
# off-by-one fences when scanning the JSON for word-precise timing.

def _fmt_floor(seconds: float) -> str:
    """`M:SS` (or `H:MM:SS`) with `seconds` rounded DOWN to the integer.

    Used as the START anchor of a speech phrase span — `2.16s -> 0:02`.
    """
    s = max(0, int(math.floor(float(seconds))))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def _fmt_ceil(seconds: float) -> str:
    """`M:SS` (or `H:MM:SS`) with `seconds` rounded UP to the integer.

    Used as the END anchor of a speech phrase span — `17.04s -> 0:18`.
    """
    s = max(0, int(math.ceil(float(seconds))))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def _fmt_range(start: float, end: float) -> str:
    """`M:SS-M:SS` span (or `H:MM:SS-H:MM:SS` once any side crosses 1h).

    Banker's-rounded; for speech use `_fmt_range_outer` instead so the
    integer span fully encloses the underlying float [start, end].
    """
    return f"{_fmt_ts(start)}-{_fmt_ts(end)}"


def _fmt_range_outer(start: float, end: float) -> str:
    """Speech phrase span — `floor(start)..ceil(end)` so the integer
    range is a guaranteed superset of the float range. See `_fmt_floor`
    / `_fmt_ceil` for the reasoning.
    """
    return f"{_fmt_floor(start)}-{_fmt_ceil(end)}"


# Legacy alias kept around for any external callers / scripts that
# imported `format_time` from this module. Routes to the new compact
# format so output stays consistent across all entry points.
def format_time(seconds: float) -> str:  # pragma: no cover — back-compat shim
    return _fmt_ts(seconds)


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
# Time format helper for merged_timeline — alias to the shared compact
# `M:SS` / `H:MM:SS` formatter above. Kept as a thin alias because the
# merged-timeline call sites read more clearly with `_hms(t)` than with
# `_fmt_ts(t)`, and renaming all of them would balloon the diff for no
# semantic gain.
# ---------------------------------------------------------------------------

def _hms(seconds: float) -> str:
    return _fmt_ts(seconds)


# ---------------------------------------------------------------------------
# Visual caption cleanup + fuzzy sentence-level delta dedup.
#
# Florence-2's `<MORE_DETAILED_CAPTION>` task emits a 5-7 sentence paragraph
# per frame at 1 fps. Two consecutive frames in static / slow-moving footage
# describe almost the same scene — same wires, same room, same colours —
# but vary by a sentence or two ("yellow panel on the top right" vs
# "yellow panel on the top corner"). The visual_lane dedup is byte-exact,
# so it only collapses literally identical captions; near-duplicates all
# survive into the audiovisual timeline and bloat it by 5-10x.
#
# The fix is sentence-level delta dedup with fuzzy matching:
#
#   * Split each caption into sentences (clean period-space split — no
#     real tokenizer, Florence captions don't use abbreviations).
#   * For each new caption, compare its sentences (normalized) against
#     the previous caption's sentences using difflib SequenceMatcher.
#     A sentence with ratio >= 0.85 vs any prior sentence counts as
#     "already seen" — colour swaps, minor word reorderings, and
#     phrasing drift don't count as new content.
#   * If every sentence is already seen → render as `(same)`.
#   * If no sentences are shared → likely shot change, full caption.
#   * Otherwise → emit only the new sentences with a `+ ` prefix
#     (think `git diff` additions).
#
# The "previous" baseline always tracks the FULL current caption's
# sentence set, not just the delta we emitted — so deltas don't compound
# (a 1-sentence drift per frame doesn't accumulate into a misleading
# claim that nothing has changed in 60 frames).
#
# Defensive `<pad>` strip is applied on every caption read so legacy
# visual_caps caches from before the visual_lane.py fix pack cleanly
# without forcing a re-preprocess.
# ---------------------------------------------------------------------------

# Mirror of visual_lane._FLORENCE_SPECIAL_TOKEN_RE — duplicated here so
# pack_timelines doesn't have to import visual_lane (which pulls in torch
# and transformers via _hf_env on import). Keeping them in sync is cheap;
# the regex is two lines and the leakage shape is fixed by Florence-2
# itself, not by our code.
_FLORENCE_SPECIAL_TOKEN_RE = re.compile(
    r"<\s*/?\s*(?:pad|s|unk|mask|bos|eos|sep|cls)\s*>",
    flags=re.IGNORECASE,
)
_WS_RUN_RE = re.compile(r"\s{2,}")

# Sentence boundaries — split on `.`/`!`/`?` followed by whitespace. The
# lookbehind keeps the punctuation attached to the preceding sentence so
# the rendered output stays grammatical when we re-join only the new
# sentences after a delta filter.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Sentence-normalization regex — strip non-word/non-space chars before
# the SequenceMatcher comparison so trailing periods, commas, and
# colour-list punctuation don't disturb the similarity ratio.
_PUNCT_STRIP_RE = re.compile(r"[^\w\s]")

# Similarity threshold for "this sentence already appeared in the previous
# caption". 0.85 was picked empirically against the reference video: at
# 0.85 a "red, white, and blue" → "red, white, and yellow" colour swap
# correctly counts as the same sentence (one token diff out of ~7), while
# a real shot change ("interior of a car" → "close-up of a drill bit")
# correctly registers as new. Lower the threshold (0.75) for noisier
# captioners; raise it (0.92) for very stable ones.
_SENT_SIMILARITY_THRESH = 0.80


def _clean_caption_text(text: str) -> str:
    """Strip Florence-2 special-token leakage from a cached caption.

    Defensive — old visual_caps/*.json from before the visual_lane.py
    decode-time strip carry runs of <pad><pad><pad>... that bloat the
    audiovisual timeline by 30-40%. Apply on read so legacy caches pack
    cleanly without forcing a re-preprocess.
    """
    if not text:
        return ""
    text = _FLORENCE_SPECIAL_TOKEN_RE.sub("", text)
    text = _WS_RUN_RE.sub(" ", text).strip()
    return text


def _split_sentences(text: str) -> list[str]:
    """Split a caption into sentences. Florence-2 outputs use clean
    period-space delimiters and no abbreviations, so a regex split is
    robust enough — no need for spaCy / nltk (would dominate the pack
    runtime on a multi-thousand-frame project).
    """
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _norm_sentence(s: str) -> str:
    """Lowercase + drop punctuation + collapse whitespace. Used purely
    for the SequenceMatcher comparison — never displayed to the user.
    """
    s = s.lower()
    s = _PUNCT_STRIP_RE.sub("", s)
    s = _WS_RUN_RE.sub(" ", s).strip()
    return s


def _sentence_seen_in(needle_norm: str, prev_norms: list[str]) -> bool:
    """True if `needle_norm` is near-identical (>= _SENT_SIMILARITY_THRESH)
    to ANY sentence in `prev_norms`. O(|prev_norms|) per call; with
    ~5-7 sentences per caption this stays well under 50 ratio() calls
    per adjacent-frame comparison — negligible against the JSON parse
    cost of even a small visual_caps file.
    """
    if not needle_norm:
        return True   # empty needle never adds anything new
    for h in prev_norms:
        if SequenceMatcher(None, needle_norm, h).ratio() >= _SENT_SIMILARITY_THRESH:
            return True
    return False


def _delta_caption(curr_text: str, prev_norms: list[str]) -> tuple[str, list[str], str]:
    """Compute the visible representation of a caption given the prior
    caption's normalized sentence list.

    Returns (mode, new_prev_norms, display_text):
        mode = "same" | "delta" | "full" | "empty"
        new_prev_norms = the sentence-norm list to carry into the next
            comparison. ALWAYS the FULL current caption's normalized
            sentences, never the delta — otherwise small sentence-level
            drifts would compound and we'd lose track of what's actually
            on screen after a few frames.
        display_text = the markdown string to render. For "delta" this
            is the new sentences joined with "+ " prefix; for "full"
            it is the entire caption; for "same" it is "(same)";
            for "empty" it is "" (caller should skip the row entirely).
    """
    curr_sents = _split_sentences(curr_text)
    if not curr_sents:
        return ("empty", [], "")

    curr_norms = [_norm_sentence(s) for s in curr_sents]
    new_idxs = [
        i for i, n in enumerate(curr_norms)
        if not _sentence_seen_in(n, prev_norms)
    ]

    if not new_idxs:
        # Every sentence already seen in the prior caption — full overlap.
        return ("same", curr_norms, "(same)")

    if len(new_idxs) == len(curr_sents):
        # Zero overlap — treat as a real shot change, render full caption.
        return ("full", curr_norms, " ".join(curr_sents))

    # Partial overlap — only emit what's new, with a `+ ` prefix so the
    # editor can see at a glance "this row is a delta against the prior
    # full caption, not a fresh description".
    delta_sents = [curr_sents[i] for i in new_idxs]
    return ("delta", curr_norms, "+ " + " ".join(delta_sents))


# ---------------------------------------------------------------------------
# Lane 1 — Speech timeline (phrase-grouped from words)
# ---------------------------------------------------------------------------

def _pack_speech(transcripts_dir: Path, silence_threshold: float) -> str:
    """Render speech_timeline.md content from <edit>/transcripts/*.json."""
    json_files = sorted(transcripts_dir.glob("*.json"))
    lines: list[str] = []
    lines.append("# Speech timeline (Parakeet ONNX)")
    lines.append("")
    lines.append(
        f"Phrase-level transcript, broken on silence ≥ {silence_threshold:.1f}s "
        f"or speaker change. Each line: `M:SS-M:SS [Sn] \"phrase text\"`. "
        f"The double quotes are the speech type marker. `[Sn]` is the "
        f"speaker tag if diarization emitted one, omitted otherwise. "
        f"START rounds DOWN to the nearest second, END rounds UP — the "
        f"integer range is a guaranteed SUPERSET of the underlying "
        f"sub-second word span in `transcripts/<stem>.json`, so any "
        f"integer range from this file can be handed to "
        f"`helpers/find_quote.py` (or scanned by hand) without "
        f"off-by-one fences. Drill-down only — the agent's primary "
        f"reading surface is `merged_timeline.md`, which carries the "
        f"same phrase ranges interleaved with audio + visual events."
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
                spk_tag = f" [S{spk_str}]"
            else:
                spk_tag = ""
            # Format: `M:SS-M:SS [Sn] "phrase text"` — quotes are the
            # speech type marker. Outer-aligned rounding (start floor /
            # end ceil) so the integer range fully encloses the float
            # [start, end] coming out of Parakeet — see _fmt_range_outer.
            lines.append(
                f"  {_fmt_range_outer(ph['start'], ph['end'])}"
                f"{spk_tag} \"{ph['text']}\""
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

    Format: `M:SS-M:SS (label score, label score, ...)`. The
    surrounding parens identify this as audio across all timeline
    files (matches the merged_timeline.md convention); the
    timestamp is bare, no `[…]` wrapper.

    Co-occurring labels in the same window collapse onto one line so a
    busy 10s window with 4 simultaneous tags reads as
        0:00-0:10 (drill 0.42, hammer 0.31, sandpaper 0.28, ...)
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
            f"  {_fmt_range(s, e)} ({label_str})"
        )


def _render_audio_captions_legacy(captions: list[dict], lines: list[str]) -> None:
    """Render free-form per-chunk captions from a stale AF3 cache.

    Tagged `[legacy AF3]` so the agent can tell at a glance the cache
    is from the abandoned Audio Flamingo 3 migration and re-running
    the audio lane will refresh it into the canonical CLAP shape.
    Wraps the caption text in parens (audio type marker) for
    consistency with the canonical CLAP renderer above.
    """
    for c in sorted(captions, key=lambda x: float(x.get("start", 0.0))):
        s = float(c.get("start", 0.0))
        e = float(c.get("end", s))
        text = (c.get("text") or "").strip().replace("\n", " ")
        if not text:
            text = "no caption"
        lines.append(f"  {_fmt_range(s, e)} ({text}) [legacy AF3]")


def _pack_audio(audio_tags_dir: Path) -> str:
    """Render audio_timeline.md content from <edit>/audio_tags/*.json."""
    json_files = sorted(audio_tags_dir.glob("*.json"))
    lines: list[str] = []
    lines.append("# Audio timeline (CLAP zero-shot events)")
    lines.append("")
    lines.append(
        "Per-window sound events scored by LAION CLAP against a vocabulary "
        "list. Each line: `M:SS-M:SS (label score, label score, ...)` — "
        "the parens are the audio type marker (consistent with "
        "`merged_timeline.md`). Scores are cosine similarities "
        "(typically 0.10-0.45 — higher is more confident). CLAP is "
        "much sharper than the abandoned PANNs ontology but still "
        "noisy on edge classes; if labels look wrong for THIS video, "
        "write a curated `<edit>/audio_vocab.txt` and re-run "
        "`python helpers/audio_lane.py <video> --vocab "
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

# ---------------------------------------------------------------------------
# Visual caps source resolver — picks comp_visual_caps/ over visual_caps/
# when caveman compression is enabled and the cache is populated.
# ---------------------------------------------------------------------------

VISUAL_CAPS_SUBDIR = "visual_caps"
COMP_VISUAL_CAPS_SUBDIR = "comp_visual_caps"


def _resolve_visual_caps_dir(edit_dir: Path, *, prefer_caveman: bool) -> Path:
    """Pick which visual caps directory to read from.

    When caveman is enabled, prefer comp_visual_caps/ if it exists and is
    non-empty. Falls back to visual_caps/ if comp/ is missing (e.g. the
    user passed --no-caveman the first time and now wants a re-pack
    without re-running spaCy). When caveman is disabled, always read
    visual_caps/.
    """
    raw = edit_dir / VISUAL_CAPS_SUBDIR
    comp = edit_dir / COMP_VISUAL_CAPS_SUBDIR
    if prefer_caveman and comp.is_dir() and any(comp.glob("*.json")):
        return comp
    return raw


def _pack_visual(visual_caps_dir: Path) -> str:
    """Render visual_timeline.md from <edit>/visual_caps/*.json.

    Uses sentence-level fuzzy delta dedup (see _delta_caption) so static
    or slow-moving footage collapses to `(same)` and slowly-evolving
    footage emits compact `+ <new sentences>` deltas instead of
    re-dumping the same 5-sentence paragraph every second. We always
    work off `captions` (the raw, per-frame list) so the dedup runs
    on actual content — `captions_dedup` is the legacy byte-exact
    collapse from visual_lane and is only used as a fallback for old
    caches that somehow lost the raw list.
    """
    json_files = sorted(visual_caps_dir.glob("*.json"))
    lines: list[str] = []
    lines.append("# Visual timeline (Florence-2 detailed captions @ 1 fps)")
    lines.append("")
    lines.append(
        "Format: `M:SS [caption]` per second (or `H:MM:SS [caption]` "
        "once a clip exceeds 1h). Square brackets are the type marker "
        "— consistent with `merged_timeline.md` so the agent "
        "can scan both files with the same parser in its head. "
        "Static/slow scenes collapse to `(same)`. Slowly-evolving "
        "scenes emit only the NEW sentences with a `+ ` prefix "
        "OUTSIDE the brackets (`M:SS + [delta sentences]`) — think "
        "`git diff` additions. A line shown WITHOUT `+ ` is a full "
        "re-description (treat as a likely shot change). Use "
        "timestamps to find shots / B-roll candidates / match cuts."
    )
    lines.append("")

    if not json_files:
        lines.append("_no visual_caps found — did visual_lane run?_")
        return "\n".join(lines)

    for p in json_files:
        data = json.loads(p.read_text(encoding="utf-8"))
        # Prefer raw captions so the fuzzy delta runs on actual content;
        # fall back to captions_dedup for very old caches that may have
        # lost the raw list (defensive — current visual_lane writes both).
        captions = data.get("captions") or data.get("captions_dedup") or []
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

        # Per-source running window of "what the prior caption's
        # sentences were". Resets at each source so deltas don't bleed
        # across clip boundaries.
        prev_norms: list[str] = []
        for c in captions:
            t = float(c.get("t", 0.0))
            raw = (c.get("text") or "").strip().replace("\n", " ")
            raw = _clean_caption_text(raw)
            # Legacy byte-exact `(same)` markers from old captions_dedup
            # caches: render verbatim and DON'T let them poison the
            # prev_norms baseline (we have no sentence content to track).
            if raw == "(same)":
                lines.append(f"  {_fmt_ts(t)} (same)")
                continue
            mode, prev_norms, display = _delta_caption(raw, prev_norms)
            if mode == "empty":
                continue
            # Wrap in `[]` (visual type marker) and pull the `+ `
            # delta prefix outside the brackets so it sits at the
            # left margin where the editor can scan it.
            if mode == "same":
                lines.append(f"  {_fmt_ts(t)} (same)")
            elif mode == "delta" and display.startswith("+ "):
                lines.append(f"  {_fmt_ts(t)} + [{display[2:]}]")
            else:
                lines.append(f"  {_fmt_ts(t)} [{display}]")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Merged interleave — speech + audio + visual lanes by timestamp.
#
# This is the agent's DEFAULT reading surface. One file, one full read,
# every event in chronological order across all three lanes — speech as
# the editorial spine (`"..."` quoted phrases with outer-aligned
# `M:SS-M:SS` ranges), audio events as `(label, label, ...)` per CLAP
# window, visual captions as `[caption text]` per 1fps Florence frame
# (delta-dedup'd so visually static stretches collapse out).
#
# The per-lane drill-down files (`speech_timeline.md`, `audio_timeline.md`,
# `visual_timeline.md`) stay on disk for moments the merged view is
# ambiguous and the agent zooms in on one lane (word-level timing,
# `(same)` repeats, full per-window CLAP scoring). For sub-second
# word-precise EDL anchors the agent uses `helpers/find_quote.py`
# against `transcripts/<stem>.json` — never grep / hand-parse those
# JSONs directly.
# ---------------------------------------------------------------------------

def _build_merged(
    edit_dir: Path,
    silence_threshold: float,
    *,
    prefer_caveman: bool = True,
    visual_stride: int = 1,
) -> str:
    """Walk transcripts + audio_tags + visual_caps in lock-step per
    video and emit one timestamp-sorted (speech + audio + visual)
    stream per video — `merged_timeline.md`.

    `prefer_caveman` controls whether we read the caveman-compressed
    visual caps (default) or the raw paragraphs.

    `visual_stride` keeps every Nth post-dedup visual caption (1 = keep
    all, 2 = keep every other → effectively 0.5 fps if the source ran
    at 1 fps, etc.). The dedup walk still updates its `prev_norms`
    baseline against EVERY caption so the delta lines that DO survive
    stride-skipping stay tight against the most recently observed
    visual state — the agent loses density, not delta accuracy. Used
    by the auto-trim path in `main()` to keep `merged_timeline.md`
    under a soft byte budget on visually dense projects (hours of
    1-fps Florence on top of speech + audio is the worst case). When
    stride > 1, a header banner is emitted so the agent knows visuals
    are thinned and should drill into `visual_timeline.md` for the
    full-density caption stream when a visual moment is ambiguous.
    """
    transcripts = edit_dir / "transcripts"
    audio_tags = edit_dir / "audio_tags"
    visual_caps = _resolve_visual_caps_dir(edit_dir, prefer_caveman=prefer_caveman)

    # Union the stems contributing to ANY lane so audio-only or
    # visual-skipped clips (e.g. voiceover .wav with no Florence pass)
    # still get a `## <stem>` header — keeps the merged file
    # structurally aligned with the per-lane drill-downs.
    stems: set[str] = set()
    for d in (transcripts, audio_tags, visual_caps):
        if d.is_dir():
            stems.update(p.stem for p in d.glob("*.json"))
    if not stems:
        return "# Merged timeline\n\n_no lane outputs to merge_\n"

    out: list[str] = []
    out.append("# Merged timeline (speech + audio + visual)")
    out.append("")
    out.append(
        "All three lanes interleaved chronologically by timestamp. "
        "Each line is `M:SS` (or `H:MM:SS` once a clip exceeds 1h) "
        "followed by ONE of three event types, distinguishable by the "
        "bracket style (no `speech:` / `audio:` / `visual:` prefixes — "
        "the brackets are the type marker, save tokens):\n"
        "  - `\"...\"`   speech phrase (Parakeet, outer-aligned `M:SS-M:SS` "
        "range — start floors, end ceils, integer span fully encloses "
        "the underlying float [start, end] in `transcripts/<stem>.json`)\n"
        "  - `(...)`   audio event(s) (CLAP zero-shot labels, comma-sep, "
        "scored against the agent-curated vocabulary)\n"
        "  - `[...]`   visual caption (Florence-2, caveman-compressed)\n"
        "\n"
        "**This is the default reading surface — read end-to-end, every "
        "spawn.** The per-lane files (`speech_timeline.md`, "
        "`audio_timeline.md`, `visual_timeline.md`) stay on disk as "
        "drill-down references for moments the merged view is ambiguous "
        "and you need to zoom in on one lane.\n"
        "\n"
        "**For sub-second word-precise cut anchors, use "
        "`helpers/find_quote.py`** — pass the clip stem, the integer "
        "`M:SS-M:SS` range read straight off the `\"...\"` line "
        "(guaranteed superset of the float span), and a quote substring "
        "to get back word-precise `{first_word, last_word, prev_word, "
        "next_word, lead_silence_s, trail_silence_s, cut_window}` JSON "
        "ready for the pacing-preset margin clamp. Never grep / "
        "hand-parse `transcripts/<stem>.json` directly.\n"
        "\n"
        "A visual line prefixed with `+ ` is a delta — only the NEW "
        "sentences vs the prior caption are shown (sentences that "
        "didn't change are dropped to keep the file scannable). A "
        "visual line WITHOUT `+ ` is a full re-description (treat as "
        "a likely shot change). Visually identical frames are dropped "
        "from this view entirely — drill into `visual_timeline.md` "
        "for per-second `(same)` markers."
    )
    out.append("")

    # ── Visual auto-trim banner ──
    # `main()` checks the size of `visual_timeline.md` (a strong proxy
    # for how big the visual lane will land in the merged file) and,
    # if the lane is dense enough that the merged file would blow past
    # the soft byte budget, passes `visual_stride > 1` here. Surface
    # that to the agent at the top of the file so the lower visual
    # density is OBVIOUS on first read — the agent should still drill
    # into `visual_timeline.md` for any visually ambiguous moment, but
    # now they know to do it more often than usual on this project.
    if visual_stride > 1:
        kept_pct = int(round(100.0 / visual_stride))
        out.append(
            f"> **VISUAL LANE THINNED {visual_stride}x — keeping every "
            f"{visual_stride}th post-dedup caption (~{kept_pct}% of the "
            f"full visual stream).** Auto-applied because the full-"
            f"density `visual_timeline.md` exceeded the soft size "
            f"budget for this view. Audio events and speech phrases "
            f"are UNAFFECTED — only visual `[...]` lines are sparser. "
            f"Drill into `visual_timeline.md` whenever a visual moment "
            f"in this file is ambiguous; you'll find the full per-"
            f"second caption stream there with `(same)` markers."
        )
        out.append("")

    # ── Sort stability for same-timestamp collisions ──
    # Within identical start times we want a predictable visual order:
    # visual context first (sets the scene), then audio events
    # (soundscape colour), then speech (the editorial spine — readers
    # naturally land on the quote last). Python's sort is stable, so
    # we append in (visual, audio, speech) order and the tuple sort
    # by start time alone preserves it within ties.
    for stem in sorted(stems):
        out.append(f"## {stem}")
        events: list[tuple[float, str]] = []  # (t, line)

        # ── Visual: raw captions, run through the sentence-level fuzzy ──
        # delta dedup. We work off `captions` (the raw per-frame list) so
        # the dedup runs on actual content, not on the byte-exact pre-
        # collapse from visual_lane. Frames that fully overlap the prior
        # caption are dropped entirely (mode == "same" / "empty"); deltas
        # carry a `+ ` prefix so the agent can see at a glance which
        # rows are partial vs full re-descriptions.
        # Format: `M:SS [caption text]` for full re-descriptions, or
        # `M:SS + [delta sentences]` when the caption is a partial
        # delta against the prior frame. The square brackets are the
        # type marker; the leading `+ ` lives OUTSIDE the brackets so
        # the delta nature is visible at the left margin without the
        # agent having to peek inside the bracket text. The
        # `_delta_caption` helper emits its display text already
        # prefixed with `+ ` in delta mode — strip that prefix so we
        # can re-attach it cleanly outside the brackets.
        vp = visual_caps / f"{stem}.json"
        if vp.exists():
            data = json.loads(vp.read_text(encoding="utf-8"))
            caps = data.get("captions") or data.get("captions_dedup") or []
            prev_norms: list[str] = []
            # `kept_idx` counts only POST-DEDUP survivors per stem. The
            # stride filter (visual_stride > 1) keeps indices 0, S, 2S,
            # ... and drops the rest. The dedup baseline `prev_norms`
            # is updated INSIDE `_delta_caption` regardless of whether
            # we emit, so the next surviving caption's delta still
            # measures against the most recently observed visual state
            # — stride costs density, not delta accuracy.
            kept_idx = 0
            for c in caps:
                t = float(c.get("t", 0.0))
                raw = (c.get("text") or "").strip().replace("\n", " ")
                raw = _clean_caption_text(raw)
                # Legacy byte-exact `(same)` survivors: skip — they don't
                # add information and we don't want them mucking with the
                # prev_norms baseline.
                if raw == "(same)":
                    continue
                mode, prev_norms, display = _delta_caption(raw, prev_norms)
                if mode in ("same", "empty"):
                    continue
                # Stride drop happens AFTER the dedup walk so we count
                # surviving captions, not raw frames. Index 0 is always
                # kept so every stem starts with a visual anchor.
                if visual_stride > 1 and (kept_idx % visual_stride) != 0:
                    kept_idx += 1
                    continue
                kept_idx += 1
                if mode == "delta" and display.startswith("+ "):
                    body = display[2:]
                    line = f"{_hms(t)} + [{body}]"
                else:
                    line = f"{_hms(t)} [{display}]"
                events.append((t, line))

        # ── Audio: one line per merged CLAP event range. Co-occurring  ──
        # labels in the same window collapse to one comma-separated tag ──
        # so the merged stream stays scannable.                         ──
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
                # Format: `M:SS (label1, label2, ...)` — the parens are
                # the type marker; no `audio:` prefix because the
                # brackets already disambiguate vs visual `[...]` and
                # speech `"..."`.
                for (s, _e), labels in by_range.items():
                    labels.sort(key=lambda x: -x[0])
                    label_str = ", ".join(lab for _sc, lab in labels[:5])
                    events.append((
                        s, f"{_hms(s)} ({label_str})",
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
                        s, f"{_hms(s)} ({text})",
                    ))

        # ── Speech: phrase-grouped Parakeet, outer-aligned ranges ──
        # Phrases break on silence ≥ silence_threshold OR speaker
        # change (same logic as `_pack_speech`, reused via the shared
        # `group_into_phrases`). Each phrase emits one line at its
        # start timestamp. Format mirrors `speech_timeline.md` so the
        # agent reads the same shape across both files:
        #   `M:SS-M:SS [Sn] "phrase text"`
        # The `[Sn]` speaker tag is omitted on undiarized inputs to
        # save tokens. Sort key is `phrase['start']` (float) — speech
        # appended last so when two phrases share the same integer
        # second with a visual / audio event, speech sorts after them
        # for visual-first / audio-mid / speech-last ordering within
        # ties.
        tp = transcripts / f"{stem}.json"
        if tp.exists():
            data = json.loads(tp.read_text(encoding="utf-8"))
            words = data.get("words", [])
            phrases = group_into_phrases(words, silence_threshold)
            for ph in phrases:
                spk = ph.get("speaker_id")
                if spk is not None:
                    spk_str = str(spk)
                    if spk_str.startswith("speaker_"):
                        spk_str = spk_str[len("speaker_"):]
                    spk_tag = f" [S{spk_str}]"
                else:
                    spk_tag = ""
                start = float(ph["start"])
                end = float(ph["end"])
                # Outer-aligned `M:SS-M:SS` so the integer span always
                # fully encloses the float [start, end] — `find_quote.py
                # --range <span>` is then guaranteed to enclose every
                # word of the phrase without off-by-one fences.
                events.append((
                    start,
                    f"{_fmt_range_outer(start, end)}{spk_tag} \"{ph['text']}\"",
                ))

        # Stable sort by start time. Within identical timestamps the
        # append order (visual → audio → speech) is preserved by
        # Python's stable timsort, giving the desired visual-first /
        # speech-last ordering on collisions.
        events.sort(key=lambda x: x[0])
        if not events:
            out.append("  _no events on any lane_")
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
    # `merged_timeline.md` is the agent's DEFAULT reading surface —
    # speech + audio + visual interleaved chronologically into one
    # file. We emit it by default. `--no-merge` (legacy alias
    # `--no-audiovisual`) is the opt-out for the rare case where only
    # the per-lane drill-down files are wanted.
    ap.add_argument("--no-merge", "--no-audiovisual",
                    dest="merge", action="store_false",
                    default=True,
                    help="Skip merged_timeline.md (the speech + audio + "
                         "visual interleaved view). `--no-audiovisual` "
                         "is kept as a legacy alias. Default is to emit.")
    ap.add_argument("--merge", "--audiovisual",
                    dest="merge", action="store_true",
                    help="(default, no-op flag for symmetry with "
                         "--no-merge / legacy --no-audiovisual.)")
    # ── Caveman compression (default ON) ──
    # Strips stop words / determiners / auxiliaries / weak adverbs from
    # every Florence-2 caption via spaCy before packing. Caches output
    # in <edit>/comp_visual_caps/ keyed by source mtime + caveman
    # version + lang. Saves 40-60% of merged_timeline.md tokens
    # with zero loss of editorial signal. See helpers/caveman_compress.py
    # for the full filter list.
    ap.add_argument("--no-caveman", dest="caveman", action="store_false",
                    default=True,
                    help="Skip caveman compression — read the raw "
                         "Florence-2 paragraphs from visual_caps/ "
                         "instead of the compressed comp_visual_caps/. "
                         "Bigger files, slower agent reads; only useful "
                         "for debugging what Florence actually said.")
    ap.add_argument("--caveman", dest="caveman", action="store_true",
                    help="(default, no-op flag for symmetry with "
                         "--no-caveman) Run caveman compression before "
                         "packing.")
    ap.add_argument("--caveman-lang", default="en",
                    help="ISO 639-1 language code for the spaCy model "
                         "the caveman pass uses (default: en). Auto-"
                         "downloads the matching model on first use.")
    ap.add_argument("--caveman-procs", type=int, default=None,
                    help="Worker process count for the caveman pass "
                         "(default: min(n_files, cpu_count // 2)).")
    ap.add_argument("--force-caveman", action="store_true",
                    help="Re-compress every visual_caps file even if "
                         "the cached comp output is fresh.")
    # ── Merged-file visual auto-trim ──
    # `merged_timeline.md` is the agent's default reading surface and
    # gets read end-to-end every spawn — keeping it under a soft byte
    # budget matters for both context window and read latency. Visual
    # captions dominate the size on dense projects (long shoots, lots
    # of motion → few `(same)` collapses). We use the size of the
    # full-density `visual_timeline.md` as a CHEAP up-front proxy:
    # write it first, stat it, and if it's over the threshold pass a
    # `visual_stride` to `_build_merged` that drops every Nth post-
    # dedup visual caption. `visual_timeline.md` itself is ALWAYS
    # written at full density so the agent can drill into it for the
    # complete caption stream when a visual moment is ambiguous.
    ap.add_argument("--max-visual-kb", type=float, default=500.0,
                    help="Soft byte budget (KB) for the visual lane in "
                         "the merged file. When `visual_timeline.md` "
                         "exceeds this on disk, every other post-dedup "
                         "visual caption is dropped from "
                         "`merged_timeline.md` (visual_stride=2). "
                         "`visual_timeline.md` itself is ALWAYS full "
                         "density. Default 500. Pass 0 to disable.")
    args = ap.parse_args()

    edit_dir = args.edit_dir.resolve()
    if not edit_dir.is_dir():
        sys.exit(f"[pack_timelines] not a directory: {edit_dir}")

    transcripts = edit_dir / "transcripts"
    audio_tags = edit_dir / "audio_tags"
    raw_visual_caps = edit_dir / VISUAL_CAPS_SUBDIR

    # Speech is the only lane considered "required" for a basic edit.
    # The other two are advisory context — if either is missing we still
    # emit the markdown with a "no data" stub so the editor sub-agent
    # gets a consistent file layout.
    if not transcripts.is_dir():
        print(f"[pack_timelines] WARN: no transcripts/ at {transcripts}",
              file=sys.stderr)

    # ── Pre-pack: caveman compression on visual_caps ──
    # Has to happen BEFORE _pack_visual / _build_merged so the
    # downstream readers see the freshly-compressed comp_visual_caps/.
    # Skipped silently when --no-caveman OR when there are no
    # visual_caps to compress (e.g. visual lane never ran).
    if args.caveman and raw_visual_caps.is_dir():
        try:
            # Sibling-import pattern — see SKILL.md / preprocess.py for
            # why we avoid `from helpers.x import y` here.
            from caveman_compress import compress_visual_caps_dir
            compress_visual_caps_dir(
                raw_visual_caps,
                edit_dir / COMP_VISUAL_CAPS_SUBDIR,
                lang=args.caveman_lang,
                force=args.force_caveman,
                n_procs=args.caveman_procs,
            )
        except Exception as exc:
            # Non-fatal: caveman is a token-budget optimization, not a
            # correctness requirement. If spaCy isn't installed or the
            # model download fails, fall back to the raw paragraphs and
            # warn the user once. The downstream _resolve_visual_caps_dir
            # naturally falls back to visual_caps/ when comp_visual_caps/
            # is missing.
            print(
                f"[pack_timelines] WARN: caveman compression failed ({exc}); "
                f"falling back to raw visual_caps/. Install with "
                f"`pip install -e .[preprocess]` and ensure the spaCy "
                f"model is available (`python -m spacy download "
                f"en_core_web_sm`).",
                file=sys.stderr,
            )

    visual_caps = _resolve_visual_caps_dir(edit_dir, prefer_caveman=args.caveman)

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
        # ── Decide visual stride for the merged file ──
        # Use the full-density `visual_timeline.md` (just written above)
        # as a cheap proxy for how big the visual lane will land in the
        # merged file. If it busts the soft byte budget, halve the
        # visual density in the merged view (every other post-dedup
        # caption). The drill-down `visual_timeline.md` stays at full
        # density on disk regardless — the agent zooms in there when a
        # visual moment in the merged view is ambiguous near a cut.
        visual_stride = 1
        if args.max_visual_kb > 0 and out_visual.exists():
            v_kb = out_visual.stat().st_size / 1024
            if v_kb > args.max_visual_kb:
                visual_stride = 2
                print(
                    f"  visual_timeline.md is {v_kb:.1f} KB (> "
                    f"{args.max_visual_kb:.0f} KB budget) — thinning "
                    f"merged_timeline.md visuals to every {visual_stride}nd "
                    f"caption. Full density preserved in "
                    f"visual_timeline.md."
                )
        out_merged = edit_dir / "merged_timeline.md"
        out_merged.write_text(
            _build_merged(
                edit_dir,
                args.silence_threshold,
                prefer_caveman=args.caveman,
                visual_stride=visual_stride,
            ),
            encoding="utf-8",
        )
        written.append(out_merged)

        # Legacy `audiovisual_timeline.md` cleanup — the rename
        # happened mid-conversation; old caches in long-lived edit
        # folders would otherwise leave stale AV-only files alongside
        # the new merged file and confuse the agent (which is now
        # told to read `merged_timeline.md`). Best-effort delete;
        # ignore failures (read-only mounts, locked files).
        legacy = edit_dir / "audiovisual_timeline.md"
        if legacy.exists():
            try:
                legacy.unlink()
            except OSError:
                pass

    for p in written:
        kb = p.stat().st_size / 1024
        print(f"  wrote {p.name}  ({kb:.1f} KB)")


if __name__ == "__main__":
    main()
