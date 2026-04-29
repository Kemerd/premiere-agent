"""Word-precise quote / range crawler for Parakeet transcripts.

Why this exists
===============

The agent reads `<edit>/merged_timeline.md` end-to-end every session —
all three lanes (speech, audio events, visual captions) interleaved
chronologically by timestamp into one file. Speech phrases on those
lines wear an outer-aligned `M:SS-M:SS` range (start floor, end
ceil), so the integer span always fully encloses the underlying float
[start, end] of the phrase in `<edit>/transcripts/<stem>.json`. The
merged view is deliberately **token-cheap**: time is rounded to whole
seconds, words are concatenated into phrases, and the per-word
boundary information that lives in `transcripts/<stem>.json` is
dropped on the way into the markdown. That trade-off is correct for
*reading* the timeline but useless when the agent has to set an
actual EDL in/out point — Hard Rule 6 demands cuts land on word
boundaries, and those boundaries are sub-second.

Historically, the agent's only path was "open the transcripts JSON
and grep / hand-walk the words[] array." That works but is slow,
error-prone, off-by-one prone, and burns context on JSON noise the
agent doesn't need (`type: "spacing"` rows, full duration headers,
etc.). This helper replaces that step with a single CLI call:

    python helpers/find_quote.py \\
        --edit-dir <edit> \\
        --clip DJI_20250214214221_0176_D \\
        --range 0:02-0:18 \\
        --quote "fabricate a hook"

Returns one JSON blob per match, with the bracketing words pinned to
the millisecond:

    {
      "clip": "DJI_20250214214221_0176_D",
      "query": {"quote": "fabricate a hook", "range_s": [2.0, 18.0]},
      "match_count": 1,
      "matches": [
        {
          "first_word": {"text": "fabricate", "start": 1.166, "end": 1.822},
          "last_word":  {"text": "hook.",     "start": 2.508, "end": 3.198},
          "prev_word":  {"text": "to",        "start": 1.021, "end": 1.166},
          "next_word":  {"text": "But",       "start": 4.418, "end": 5.246},
          "words":      [...sequence the agent matched...],
          "text":       "fabricate A hook.",
          "speaker_id": null
        }
      ]
    }

The agent reads `first_word.start` as the in-point candidate and
`last_word.end` as the out-point candidate, then applies the pacing
preset's lead/trail margins (clamped against `cut_window.safe_in_s` /
`cut_window.safe_out_s` so the lead/trail pads leave at least 60ms of
true silence for the `afade` pair) and emits the EDL range. No greps.
No off-by-one. No partial JSON parses.

Calling shapes
==============

    # 1. Quote-only (no range filter, hits all matches across the clip).
    find_quote.py --edit-dir <edit> --clip <stem> --quote "lock in"

    # 2. Quote bounded by an integer range read straight off the
    #    `"..."` line in merged_timeline.md (or speech_timeline.md
    #    drill-down). The range is OUTER-ALIGNED — start floors, end
    #    ceils — so it always contains the actual float span. Pass it
    #    verbatim.
    find_quote.py --edit-dir <edit> --clip <stem> \\
                  --range 0:02-0:18 --quote "lock in"

    # 3. Range-only — return every word inside the range, plus the
    #    bracketing prev/next. Useful when the agent wants to see the
    #    full sub-second word stream for a phrase.
    find_quote.py --edit-dir <edit> --clip <stem> --range 0:02-0:18

    # 4. Default clip — when --clip is omitted, picks the most recently
    #    modified transcript JSON in <edit>/transcripts/. Convenient
    #    for the "last clip we talked about" case.
    find_quote.py --edit-dir <edit> --quote "lock in"

    # 5. Batched — one Python startup, N queries answered. The agent
    #    builds a JSON document with a `queries` array (each row is the
    #    same shape as the single-query CLI: `clip` / `quote` / `range`
    #    / `start` / `end` / `max_matches`, all optional except that
    #    every row must carry at least a quote or a range), and pipes
    #    it in via `--batch -` or hands over a file path. Queries are
    #    grouped by clip internally so each transcript is loaded
    #    exactly once no matter how many queries reference it — turns
    #    "40 tool calls" into one. Per-row failures (bad clip stem,
    #    empty quote, etc.) come back as `{error, detail}` rows
    #    instead of crashing the whole batch:
    #
    #    echo '{
    #      "queries": [
    #        {"id":"a","clip":"DJI_..._0176_D","range":"0:02-0:18",
    #         "quote":"fabricate a hook"},
    #        {"id":"b","clip":"DJI_..._0176_D","quote":"lock in"},
    #        {"id":"c","clip":"DJI_..._0182_D","range":"0:12-0:18"}
    #      ]
    #    }' | python helpers/find_quote.py --edit-dir <edit> --batch -

Time arguments accept `M:SS`, `MM:SS`, `H:MM:SS`, integer seconds, or
floating-point seconds (e.g. `1.234`). The integer-rounded ranges out
of `merged_timeline.md` (and the `speech_timeline.md` drill-down) are
always safe to pass through.

Performance
===========

A typical 80-second clip transcript (~700 words) is under 60KB on
disk; `json.load` plus the sliding-window match runs in well under
5ms on cold cache, sub-millisecond hot. The script avoids regex
backtracking entirely — quote matching is a deterministic word-by-
word sweep with cheap punctuation/casefold normalisation. Memory is
linear in the number of words[] entries (no copies of the whole
JSON object) and bounded by one transcript at a time.

The helper is self-contained: no third-party dependencies, no model
loads, no network calls. Safe to invoke from any subprocess context
without warming up a venv.
"""

# ──────────────────────────────────────────────────────────────────────
# Imports — stdlib only on purpose. This script must run from any
# fresh subprocess without paying torch/onnx import cost. Even
# `pathlib` over raw os.path is a deliberate choice: it's free at
# import time and keeps the path-massage code legible on Windows
# where backslashes love to bite.
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────────────────────
# Console UTF-8 reconfiguration (Windows safety)
# ----------------------------------------------------------------------
# Python on Windows defaults stdout/stderr to the legacy console
# codepage (cp1252-ish), which mangles fancy quotes, em-dashes, and
# anything outside Latin-1 in the JSON output. Reconfigure both
# streams to UTF-8 before any print so the agent (which pipes our
# output back through json.loads) never has to worry about
# encoding-induced parse failures.
# ──────────────────────────────────────────────────────────────────────
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    # Some captured-stdout environments (notebooks, certain CI shims)
    # don't expose .reconfigure; the legacy encoding is fine for
    # ASCII-only quotes which is the common case anyway.
    pass


# ──────────────────────────────────────────────────────────────────────
# Time parsing — accept everything the speech_timeline.md emits
# ----------------------------------------------------------------------
# `speech_timeline.md` ranges look like `0:02-0:18`, `1:23-1:45`,
# or in long-form `1:02:33-1:02:48` for hour-spanning sources. The
# CLI also accepts plain integers and floats so the editor can pass
# `--start 3.12 --end 4.04` after a previous round-trip.
# ──────────────────────────────────────────────────────────────────────

# Single-component validator — every colon-separated piece must be a
# non-negative number, with optional fractional part on the seconds
# slot only. The components are normalised to a uniform H:MM:SS
# triple before parsing, so this regex fires three times max.
_TIME_PIECE = re.compile(r"^\d+(?:\.\d+)?$")


def _parse_time(value: str) -> float:
    """Parse `H:MM:SS` / `MM:SS` / `SS(.fff)` into float seconds.

    Strategy: pad-left with `"00"` chunks until the input has three
    colon-separated components, then read them as hours, minutes,
    seconds. This makes the rightmost piece ALWAYS seconds (and the
    speech_timeline.md `M:SS` convention falls out for free):

        "32"       -> "00:00:32"  -> 32.0s
        "2:32"     -> "00:02:32"  -> 152.0s
        "1:23:45"  -> "01:23:45"  -> 5025.0s
        "2:32.5"   -> "00:02:32.5" -> 152.5s

    Raises ValueError on garbage so argparse surfaces the bad token
    rather than silently zeroing it.
    """
    raw = value.strip()
    if not raw:
        raise ValueError(f"unrecognised time format: {value!r}")

    # Pad-left with "00" placeholders until we hit the canonical
    # three-piece H:MM:SS shape. Anything beyond three colons is
    # malformed input and trips the explicit error below.
    pieces = raw.split(":")
    if len(pieces) > 3:
        raise ValueError(f"unrecognised time format: {value!r}")
    while len(pieces) < 3:
        pieces.insert(0, "00")

    # Validate every component is a clean number (the seconds slot
    # may carry a fractional tail; hours / minutes integer-only).
    # Any failure here is bad input — surface it.
    if not all(_TIME_PIECE.match(p) for p in pieces):
        raise ValueError(f"unrecognised time format: {value!r}")
    if "." in pieces[0] or "." in pieces[1]:
        raise ValueError(
            f"hours and minutes must be integers in {value!r}"
        )

    h = int(pieces[0])
    minutes = int(pieces[1])
    s = float(pieces[2])
    return h * 3600.0 + minutes * 60.0 + s


def _parse_range(value: str) -> tuple[float, float]:
    """Parse a `START-END` pair (e.g. `0:02-0:18`).

    The dash separates two _parse_time-compatible halves. Negative
    spans error out — they're almost always typos in agent-generated
    args.
    """
    if "-" not in value:
        raise ValueError(
            f"--range must be START-END, got {value!r}"
        )
    # Split on the LAST dash so a stray sign in the seconds value
    # (impossible today, but cheap to guard) doesn't break parsing.
    a, b = value.rsplit("-", 1)
    start = _parse_time(a)
    end = _parse_time(b)
    if end < start:
        raise ValueError(
            f"--range END ({end}) precedes START ({start})"
        )
    return start, end


# ──────────────────────────────────────────────────────────────────────
# Quote tokenisation — punctuation/case-insensitive sliding window
# ----------------------------------------------------------------------
# Parakeet emits raw transcribed text per word, including trailing
# punctuation glued to the token (`hook.`). The editor's quote will
# typically be punctuation-free human-typed text. To match cleanly we
# normalise both sides identically: lower-case, strip leading/
# trailing non-alphanumerics, drop empties.
#
# We DON'T touch internal characters (apostrophes inside `I'm`, hyphens
# inside `re-tighten`) — those carry meaning and the speech-timeline
# render preserves them too, so the editor's quote will mirror.
# ──────────────────────────────────────────────────────────────────────

# Strip leading and trailing punctuation only. Internal characters
# like `'` (in `I'm`) and `-` (in `re-tighten`) survive; this matches
# how the speech_timeline.md groups words back into phrases.
_TRIM_PUNCT = re.compile(r"^[^\w']+|[^\w']+$")


def _normalise_token(text: str) -> str:
    """Casefold + strip surrounding punctuation. Returns lower-case
    token suitable for an equality test against another normalised
    token. Empty tokens (e.g. punctuation-only artifacts) are
    returned as `""` and the caller filters them out.
    """
    # casefold() handles edge-case Unicode (German ß etc.) better than
    # plain .lower() and is no slower on ASCII.
    return _TRIM_PUNCT.sub("", text).casefold()


def _tokenise_quote(quote: str) -> list[str]:
    """Split user quote into a normalised token list.

    Whitespace-split is enough — the user types human text, not raw
    transcript output. Empty tokens after normalisation are dropped.
    """
    return [t for t in (_normalise_token(p) for p in quote.split()) if t]


# ──────────────────────────────────────────────────────────────────────
# Transcript loader + word-only filter
# ----------------------------------------------------------------------
# Parakeet's words[] array carries both `type: "word"` and `type:
# "spacing"` rows. The spacing rows mark inter-word silence and we
# explicitly DON'T want them — matching against them confuses the
# token sweep and they have no editorial meaning. Filter once at
# load and never touch them again.
# ──────────────────────────────────────────────────────────────────────


def _load_transcript(path: Path) -> dict[str, Any]:
    """Load the transcript JSON, raising a friendly error if missing."""
    if not path.is_file():
        raise FileNotFoundError(
            f"transcript not found: {path}\n"
            "  (re-run helpers/preprocess.py or preprocess_batch.py)"
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _word_rows(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """Filter words[] down to type==word entries with start/end set.

    Spacing rows and any future row types (Parakeet has occasionally
    added diarisation markers in beta builds) are dropped here so
    the rest of the pipeline can assume "every entry is a real word."
    """
    out = []
    for w in transcript.get("words", []):
        if w.get("type") != "word":
            continue
        # Defensive: a malformed row missing start/end would break the
        # range filter downstream; skip it and keep going rather than
        # crash the whole search.
        if "start" not in w or "end" not in w:
            continue
        out.append(w)
    return out


# ──────────────────────────────────────────────────────────────────────
# Range filtering — half-open by intent, inclusive by data
# ----------------------------------------------------------------------
# A word "is in the range" if any part of it overlaps [start, end].
# That matches editorial intuition: if the speaker started saying
# `fabricate` at 1.166s and the user passes `--range 0:02-0:18` from
# `speech_timeline.md`, we want `fabricate` included even though its
# start is before 2.0s — the outer-aligned floor on the speech
# timeline already accounts for this overlap.
# ──────────────────────────────────────────────────────────────────────


def _in_range(word: dict[str, Any], start: float, end: float) -> bool:
    """True if `word`'s [start, end] overlaps the query [start, end].

    Uses interval-overlap semantics (`a.start < b.end and b.start <
    a.end`) — a word touching either bound at exactly the boundary
    is included, which matches how speech_timeline.md ranges are
    rendered (start floors, end ceils, both endpoints are wider than
    the float reality).
    """
    return word["start"] < end and start < word["end"]


# ──────────────────────────────────────────────────────────────────────
# Quote matcher — sliding window over the (range-filtered) word list
# ----------------------------------------------------------------------
# Classic O(n*m) sweep where n = number of words and m = number of
# tokens in the quote. For typical inputs (n ≲ 1000, m ≲ 10) that's
# microseconds; we don't bother with KMP. The simplicity also means
# the matcher is trivially auditable when an editor screenshots its
# output for verification.
#
# Returned matches are *windows* — slices of the original word list,
# not copies — so the caller can pull prev/next words off the same
# list without re-indexing.
# ──────────────────────────────────────────────────────────────────────


def _find_quote_windows(
    words: list[dict[str, Any]],
    quote_tokens: list[str],
) -> list[tuple[int, int]]:
    """Return a list of (start_idx, end_idx_inclusive) windows in
    `words` whose normalised text matches `quote_tokens` exactly.

    Match is contiguous and exact post-normalisation. No fuzzy /
    substring fallback — fuzzy matching belongs in the editor's head
    where it can quote the user's intent in the EDL `reason` field;
    this helper's job is "give me the words I asked for, with
    millisecond timestamps" or nothing.
    """
    if not quote_tokens:
        return []

    n = len(words)
    m = len(quote_tokens)
    if m > n:
        return []

    # Precompute the normalised form of every word once — saves a
    # constant factor when the same words[] array is swept multiple
    # times (rare today, but the function is reusable).
    norms = [_normalise_token(w["text"]) for w in words]

    # Pre-pull the first quote token for the cheap-skip optimisation
    # (start the inner loop only when norms[i] could match the head).
    head = quote_tokens[0]

    matches: list[tuple[int, int]] = []
    i = 0
    while i <= n - m:
        if norms[i] == head:
            ok = True
            # Inner loop avoids slicing — we step token-by-token and
            # bail at the first mismatch. Best case `m` checks; worst
            # case `m` checks; either way bounded.
            for k in range(1, m):
                if norms[i + k] != quote_tokens[k]:
                    ok = False
                    break
            if ok:
                matches.append((i, i + m - 1))
                # Step past the matched window so overlapping matches
                # don't double-count. Editorial use case never wants
                # overlapping hits.
                i += m
                continue
        i += 1
    return matches


# ──────────────────────────────────────────────────────────────────────
# Match-result assembly — JSON shape the editor consumes
# ----------------------------------------------------------------------
# Every match is rendered as a self-contained dict carrying:
#   - first_word: the word the editor's IN-point should snap to.
#   - last_word:  the word the editor's OUT-point should snap to.
#   - prev_word:  the word just before first_word, used for the
#                 lead-margin clamp (don't leak into prior speech).
#   - next_word:  the word just after last_word, used for the
#                 trail-margin clamp (don't leak into next speech).
#   - words:      the contiguous run between first and last (inclusive),
#                 raw Parakeet rows so the editor can sanity-check the
#                 transcription against the user's quote.
#   - text:       a re-joined human-readable string of the matched run.
#   - speaker_id: lifted from first_word; useful for diarised inputs
#                 where the editor wants to confirm the cut is in the
#                 right speaker's run.
# ──────────────────────────────────────────────────────────────────────


def _assemble_match(
    matched_words: list[dict[str, Any]],
    start_idx: int,
    end_idx: int,
    all_words: list[dict[str, Any]],
    duration_s: float | None,
) -> dict[str, Any]:
    """Render one (start_idx, end_idx) window into the public match
    dict shape. `end_idx` is INCLUSIVE.

    `matched_words` is the (possibly range-filtered) list the matcher
    swept; `all_words` is the full unfiltered transcript word list.
    `prev_word` / `next_word` are looked up in `all_words` so the
    editor's lead/trail clamp still has the bracketing words even
    when the matched span sits at the edge of the requested range.

    `duration_s` is the clip's full duration from the transcript
    header — used to compute trailing silence when the match is the
    last spoken phrase in the clip (no `next_word` to clamp against).
    """
    run = matched_words[start_idx : end_idx + 1]
    first = run[0]
    last = run[-1]

    # Locate first/last inside the FULL transcript word list so the
    # bracketing words are real neighbours from the timeline, not
    # just the nearest range-included neighbours. We anchor by
    # object identity (the matcher operates on the same dict refs)
    # which is O(n) worst case but n is bounded by ~1000 words per
    # clip — call it a microsecond.
    try:
        full_first_idx = all_words.index(first)
        full_last_idx = all_words.index(last)
    except ValueError:
        # Defensive fallback — shouldn't happen since matched_words
        # is always a subset of all_words by reference, but a
        # mutated input could break the identity link.
        full_first_idx = -1
        full_last_idx = len(all_words)

    prev_w = (
        all_words[full_first_idx - 1]
        if full_first_idx > 0
        else None
    )
    next_w = (
        all_words[full_last_idx + 1]
        if 0 <= full_last_idx + 1 < len(all_words)
        else None
    )

    # ── Silence accounting — the cut-budget the editor needs ────
    # `lead_silence_s` / `trail_silence_s` are the seconds of true
    # silence framing the matched span. They tell the editor how
    # much room there is before the in-point and after the out-point
    # without bumping into adjacent speech, which is exactly the
    # clamp ceiling for the pacing-preset lead/trail margins (Hard
    # Rule 7 — 30-200ms padding window).
    #
    # Edge cases:
    #   • No prev_word (match starts at the very first spoken word)
    #     → lead_silence_s = first.start (silence from 0 to start).
    #   • No next_word (match ends at the very last spoken word)
    #     → trail_silence_s = duration_s - last.end if duration_s
    #       is known, else None (we don't fabricate clip lengths).
    #   • Negative gaps — should never happen on a healthy
    #     transcript but Parakeet has been seen emitting sub-ms
    #     overlap on consecutive words; clamp at 0.0.
    #
    # The editor reads these values directly into its trail-margin
    # clamp formula:
    #
    #     trail_margin_used = min(target_trail_margin, trail_silence_s - 0.06)
    #
    # leaving 60ms of true silence for the 30ms `afade` pair on
    # each boundary (per the "Word-boundary verification" worked
    # example in SKILL.md). Same on the lead side.
    lead_silence_s: float | None
    if prev_w is not None:
        lead_silence_s = max(0.0, float(first["start"]) - float(prev_w["end"]))
    else:
        lead_silence_s = max(0.0, float(first["start"]))

    trail_silence_s: float | None
    if next_w is not None:
        trail_silence_s = max(0.0, float(next_w["start"]) - float(last["end"]))
    elif duration_s is not None:
        trail_silence_s = max(0.0, float(duration_s) - float(last["end"]))
    else:
        # No next word AND no clip duration → we genuinely don't know
        # how much trailing silence the source has. Returning None
        # forces the editor to either drill into the source manually
        # or treat the cut as "safe to extend up to the clip end."
        trail_silence_s = None

    # ── Cut-window — the absolute outer bounds for safe trimming ──
    # `safe_in_s`  := the earliest second the in-point can land at
    # without eating into prior speech (= prev_word.end, or 0.0 when
    # there's no prev_word).
    # `safe_out_s` := the latest second the out-point can land at
    # without bleeding into next speech (= next_word.start, or the
    # clip duration when there's no next_word).
    # The editor uses these as clamp ceilings rather than re-deriving
    # them from prev/next every time. Saves three lines of brittle
    # arithmetic per range and surfaces "you can cut anywhere from
    # X to Y" cleanly in the JSON.
    safe_in_s = float(prev_w["end"]) if prev_w is not None else 0.0
    if next_w is not None:
        safe_out_s: float | None = float(next_w["start"])
    elif duration_s is not None:
        safe_out_s = float(duration_s)
    else:
        safe_out_s = None

    # Re-joining with single spaces is good enough for verification.
    # The original transcript text is in the top-level "text" field
    # if the editor needs the exact spacing.
    text = " ".join(w["text"] for w in run)

    return {
        "first_word": first,
        "last_word": last,
        "prev_word": prev_w,
        "next_word": next_w,
        "lead_silence_s": lead_silence_s,
        "trail_silence_s": trail_silence_s,
        "cut_window": {
            "safe_in_s": safe_in_s,
            "safe_out_s": safe_out_s,
        },
        "words": run,
        "text": text,
        "speaker_id": first.get("speaker_id"),
    }


# ──────────────────────────────────────────────────────────────────────
# Default-clip resolver — "the last one we talked about"
# ----------------------------------------------------------------------
# When the editor doesn't pass --clip, we pick the most recently
# modified file in <edit>/transcripts/. This matches the user's
# colloquial "find the quote in the last clip" — preprocess wrote
# transcripts in batch order so the latest mtime is the latest clip
# the parent processed.
# ──────────────────────────────────────────────────────────────────────


def _default_clip_stem(transcripts_dir: Path) -> str:
    """Return the stem of the most-recently-modified transcript JSON.

    Errors clearly when the directory is empty so the caller knows
    preprocess hasn't run yet.
    """
    files = list(transcripts_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"no transcripts found in {transcripts_dir}\n"
            "  (run helpers/preprocess.py or preprocess_batch.py first)"
        )
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0].stem


# ──────────────────────────────────────────────────────────────────────
# Per-query worker — one transcript search, returns the JSON envelope
# ----------------------------------------------------------------------
# Pulled out of main() so the batch path can call it N times against a
# shared transcript cache. Returns the same envelope shape the
# single-query CLI emits today, OR an `{error, detail, ...}` dict on
# any per-query failure (missing transcript, bad range, empty quote,
# no query at all). Never raises — batch callers want one bad row to
# turn into one error row, not a whole-batch SystemExit.
#
# `transcript_cache` is a dict[stem -> (transcript_dict, words_list,
# duration_s)]. The caller owns it; we just memoise loads through it.
# Pass an empty dict for single-query mode and the cache is a no-op.
# ──────────────────────────────────────────────────────────────────────


def _resolve_query_range(
    range_str: str | None,
    start: float | str | None,
    end: float | str | None,
) -> tuple[float, float] | None:
    """Same precedence as the CLI (`range` > `start`/`end`), but
    accepts already-parsed numeric `start`/`end` from a batch row as
    well as the string form a CLI arg would carry. Returns None when
    no range was specified.
    """
    if range_str:
        return _parse_range(range_str)
    if start is None and end is None:
        return None
    if start is None or end is None:
        raise ValueError("`start` and `end` must be provided together")

    # Coerce numeric input through float() but funnel string input
    # through _parse_time so both shapes ("0:02" and 2.0) work.
    s = float(start) if not isinstance(start, str) else _parse_time(start)
    e = float(end) if not isinstance(end, str) else _parse_time(end)
    if e < s:
        raise ValueError(f"`end` ({e}) precedes `start` ({s})")
    return (s, e)


def _run_query(
    query: dict[str, Any],
    transcripts_dir: Path,
    transcript_cache: dict[str, tuple[dict[str, Any], list[dict[str, Any]], float | None]],
    default_clip_stem: str | None = None,
) -> dict[str, Any]:
    """Execute one query against the transcripts directory.

    `query` keys (all optional unless noted):
      - clip          : transcript stem (falls back to default_clip_stem,
                        then to most-recently-modified transcript)
      - quote         : substring to match (must have quote OR range)
      - range         : "M:SS-M:SS" string
      - start, end    : numeric seconds OR M:SS string (alternative to range)
      - max_matches   : int cap on returned matches (0 = unbounded)
      - id            : opaque token echoed back in `query.id` so a
                        batch caller can correlate inputs to outputs
                        without zipping by index

    Returns the same envelope shape the single-query CLI emits, or an
    `{error, detail, query}` dict on per-query failure. The `query`
    echo on errors carries the `id` if one was passed, plus whichever
    inputs we managed to read before failing.
    """
    # ── Echo block — built up as we resolve inputs so any error path
    #    returns as much context as we have at the point of failure.
    query_echo: dict[str, Any] = {}
    if "id" in query:
        query_echo["id"] = query.get("id")
    query_echo["quote"] = query.get("quote")

    # ── Resolve clip stem (per-query > batch default > newest mtime) ──
    clip_stem = query.get("clip") or default_clip_stem
    if clip_stem is None:
        try:
            clip_stem = _default_clip_stem(transcripts_dir)
        except FileNotFoundError as e:
            return {
                "error": "no_transcripts",
                "detail": str(e),
                "query": query_echo,
            }

    # Tolerate "<stem>.json" — same defensive trim as the CLI path.
    if isinstance(clip_stem, str) and clip_stem.endswith(".json"):
        clip_stem = clip_stem[:-5]
    query_echo["clip"] = clip_stem

    # ── Load transcript via cache ──────────────────────────────────
    # Loads cost a json.load + a words[] filter. Doing it once per
    # unique clip across the whole batch is the whole point of
    # batching — without the cache a 40-query batch on 12 clips
    # would still pay 40 disk reads.
    cached = transcript_cache.get(clip_stem)
    if cached is None:
        transcript_path = transcripts_dir / f"{clip_stem}.json"
        try:
            transcript = _load_transcript(transcript_path)
        except FileNotFoundError as e:
            return {
                "error": "transcript_not_found",
                "detail": str(e),
                "query": query_echo,
            }
        words = _word_rows(transcript)
        duration_s = transcript.get("duration")
        transcript_cache[clip_stem] = (transcript, words, duration_s)
    else:
        transcript, words, duration_s = cached

    transcript_path = transcripts_dir / f"{clip_stem}.json"

    # ── Resolve range (string OR numeric pair) ─────────────────────
    try:
        time_range = _resolve_query_range(
            query.get("range"),
            query.get("start"),
            query.get("end"),
        )
    except ValueError as e:
        return {"error": "bad_range", "detail": str(e), "query": query_echo}

    if time_range is not None:
        rstart, rend = time_range
        ranged_words = [w for w in words if _in_range(w, rstart, rend)]
        query_echo["range_s"] = [rstart, rend]
    else:
        ranged_words = words
        query_echo["range_s"] = None

    # ── Match quote / range-only / nothing ─────────────────────────
    quote = query.get("quote")
    max_matches = int(query.get("max_matches", 0) or 0)
    matches: list[dict[str, Any]] = []

    if quote:
        tokens = _tokenise_quote(quote)
        if not tokens:
            return {
                "error": "empty_quote",
                "detail": "`quote` tokenised to nothing",
                "query": query_echo,
            }
        windows = _find_quote_windows(ranged_words, tokens)
        if max_matches > 0:
            windows = windows[:max_matches]
        matches = [
            _assemble_match(ranged_words, s, e, words, duration_s)
            for s, e in windows
        ]
    elif time_range is not None:
        if ranged_words:
            matches = [
                _assemble_match(
                    ranged_words, 0, len(ranged_words) - 1,
                    words, duration_s,
                )
            ]
    else:
        return {
            "error": "no_query",
            "detail": "pass `quote`, `range`, or both",
            "query": query_echo,
        }

    return {
        "clip": clip_stem,
        "transcript": str(transcript_path),
        "duration_s": duration_s,
        "query": query_echo,
        "match_count": len(matches),
        "matches": matches,
    }


# ──────────────────────────────────────────────────────────────────────
# CLI plumbing
# ──────────────────────────────────────────────────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    """Construct the argparse spec. Pulled into its own function so
    tests / docs can introspect it without invoking main()."""
    p = argparse.ArgumentParser(
        prog="find_quote.py",
        description=(
            "Word-precise quote / range crawler over Parakeet "
            "transcript JSON. Used by the agent for cut anchor "
            "verification — see SKILL.md 'Word-boundary verification' "
            "for the workflow."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--edit-dir",
        type=Path,
        required=True,
        help=(
            "Path to the project's <edit>/ directory. Transcripts "
            "are read from <edit>/transcripts/<stem>.json."
        ),
    )
    p.add_argument(
        "--clip",
        type=str,
        default=None,
        help=(
            "Stem of the transcript file to search (e.g. "
            "DJI_20250214214221_0176_D). When omitted, the most "
            "recently modified transcript in <edit>/transcripts/ "
            "is used — the colloquial 'last clip' shortcut."
        ),
    )
    p.add_argument(
        "--quote",
        type=str,
        default=None,
        help=(
            "Quote substring to search for, in human-typed prose. "
            "Punctuation and case are ignored. Internal apostrophes "
            "/ hyphens preserved. Match is contiguous and exact "
            "post-normalisation; no fuzzy fallback."
        ),
    )
    # Range can be passed as a single string (matching speech_timeline.md
    # rendering) OR as an explicit start/end pair (matching post-EDL
    # round-trip use cases). Either way a tuple is computed by the
    # parsing layer.
    p.add_argument(
        "--range",
        type=str,
        default=None,
        dest="range_str",
        help=(
            "Time range filter, formatted as START-END (e.g. "
            "0:02-0:18). Pass values verbatim from "
            "speech_timeline.md — they're outer-aligned and always "
            "contain the actual word span. Omit to search the whole "
            "clip."
        ),
    )
    p.add_argument(
        "--start",
        type=_parse_time,
        default=None,
        help="Range start (alternative to --range).",
    )
    p.add_argument(
        "--end",
        type=_parse_time,
        default=None,
        help="Range end (alternative to --range).",
    )
    p.add_argument(
        "--max-matches",
        type=int,
        default=0,
        help=(
            "Cap the number of returned quote matches. 0 (default) "
            "returns all of them. Useful when a short quote like "
            "'okay' would otherwise hit dozens of times."
        ),
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help=(
            "Emit compact single-line JSON instead of pretty-printed. "
            "Default is pretty-printed for human readability — agents "
            "consuming the output can json.loads either form."
        ),
    )
    p.add_argument(
        "--batch",
        type=str,
        default=None,
        dest="batch_path",
        help=(
            "Run a batch of queries from a JSON document. Pass a file "
            "path or `-` to read from stdin. The document must be "
            "either {\"queries\": [...]} or a bare [...] array of "
            "query objects. Each query object accepts the same fields "
            "as the single-query CLI (`clip`, `quote`, `range` or "
            "`start`/`end`, `max_matches`) plus an optional `id` that "
            "is echoed back so the caller can correlate results "
            "without zipping by index. Queries are grouped by clip "
            "internally so each transcript JSON is loaded exactly "
            "once. Per-query failures come back as `{error, detail, "
            "query}` rows; the batch never aborts mid-stream."
        ),
    )
    return p


def _load_batch_doc(path: str) -> list[dict[str, Any]]:
    """Read the batch JSON document from a file path or stdin (`-`).

    Accepts two top-level shapes for ergonomics:
      • `{"queries": [...]}` — explicit, leaves room to add batch-
        level options later (defaults, per-batch flags, etc.) without
        a breaking change.
      • bare `[...]` — convenient for hand-rolled batches piped in
        via heredoc; we wrap it into the canonical shape internally.

    Returns the list of per-query dicts. Raises ValueError on any
    structural problem so the caller can convert to a clean error
    envelope.
    """
    # `-` means stdin — standard CLI convention. Read the whole stream
    # rather than streaming JSON; batches are bounded by agent context
    # (well under a megabyte) so memory is a non-issue.
    if path == "-":
        raw = sys.stdin.read()
    else:
        try:
            with Path(path).open("r", encoding="utf-8") as f:
                raw = f.read()
        except FileNotFoundError as e:
            raise ValueError(f"batch file not found: {path}") from e

    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"batch JSON parse error: {e}") from e

    # Normalise both accepted shapes down to a list of dicts.
    if isinstance(doc, dict) and "queries" in doc:
        queries = doc["queries"]
    elif isinstance(doc, list):
        queries = doc
    else:
        raise ValueError(
            "batch document must be {\"queries\": [...]} or a bare list"
        )

    if not isinstance(queries, list):
        raise ValueError("`queries` must be a list")

    # Every row must be a dict — anything else (string, number, list)
    # is almost certainly a typo and would explode deep inside
    # _run_query with a confusing AttributeError otherwise.
    for i, q in enumerate(queries):
        if not isinstance(q, dict):
            raise ValueError(
                f"queries[{i}] must be an object, got {type(q).__name__}"
            )

    return queries


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code: 0 on any successful
    parse + search (even when no matches found — the empty result IS
    the answer), non-zero only on argument / IO errors.

    Two modes:
      • Single-query (default) — uses --clip / --quote / --range etc.
        from the CLI flags. Output is one envelope dict.
      • Batch (--batch <path|->) — reads N queries from a JSON
        document. Output is `{batch_count, results:[...]}` with one
        result row per input query. Single Python startup, single
        transcript load per unique clip, per-row error isolation.
    """
    parser = _build_argparser()
    args = parser.parse_args(argv)

    # ── Resolve paths ──────────────────────────────────────────────
    # Edit-dir is the well-known root holding `transcripts/`,
    # `audio_tags/`, `visual_caps/`, etc. Resolve once so the rest
    # of the function works on canonical paths.
    edit_dir = args.edit_dir.resolve()
    transcripts_dir = edit_dir / "transcripts"
    if not transcripts_dir.is_dir():
        print(
            json.dumps(
                {
                    "error": "transcripts_dir_missing",
                    "edit_dir": str(edit_dir),
                    "transcripts_dir": str(transcripts_dir),
                    "hint": (
                        "run helpers/preprocess.py or "
                        "preprocess_batch.py against the source files"
                    ),
                }
            ),
            file=sys.stderr,
        )
        return 2

    # ── Shared transcript cache for the duration of this process ──
    # Single-query mode only ever populates one entry; batch mode
    # populates one per unique clip across the whole queries[] array.
    # Cleared on process exit (we don't bother holding it longer).
    transcript_cache: dict[
        str, tuple[dict[str, Any], list[dict[str, Any]], float | None]
    ] = {}

    # ────────────────────────────────────────────────────────────────
    # Batch path
    # ────────────────────────────────────────────────────────────────
    if args.batch_path:
        try:
            queries = _load_batch_doc(args.batch_path)
        except ValueError as e:
            print(
                json.dumps({"error": "bad_batch", "detail": str(e)}),
                file=sys.stderr,
            )
            return 2

        # Reorder queries by clip so each transcript load happens once
        # and consecutively (cache-friendly memory access too — the
        # words[] list stays warm between back-to-back queries on the
        # same clip). We DON'T reorder the output; results carry their
        # original input index so the agent can zip-by-index OR by
        # `query.id` whichever it prefers.
        order: list[int] = sorted(
            range(len(queries)),
            key=lambda i: (
                queries[i].get("clip") or args.clip or "",
                i,  # stable secondary key preserves intra-clip order
            ),
        )
        results: list[dict[str, Any] | None] = [None] * len(queries)
        for i in order:
            row = _run_query(
                queries[i],
                transcripts_dir,
                transcript_cache,
                default_clip_stem=args.clip,
            )
            # Echo the original input index so the agent can correlate
            # without trusting iteration order, even after reordering.
            row.setdefault("query", {})["index"] = i
            results[i] = row

        envelope = {
            "batch_count": len(queries),
            "clips_loaded": len(transcript_cache),
            "results": results,
        }

        if args.compact:
            sys.stdout.write(json.dumps(envelope, ensure_ascii=False))
        else:
            sys.stdout.write(json.dumps(envelope, ensure_ascii=False, indent=2))
        sys.stdout.write("\n")
        return 0

    # ────────────────────────────────────────────────────────────────
    # Single-query path — synthesise a one-row "batch" and dispatch
    # through the same worker so the two modes can never drift.
    # ────────────────────────────────────────────────────────────────
    single_query: dict[str, Any] = {
        "clip": args.clip,
        "quote": args.quote,
        "range": args.range_str,
        "start": args.start,
        "end": args.end,
        "max_matches": args.max_matches,
    }
    result = _run_query(
        single_query,
        transcripts_dir,
        transcript_cache,
        default_clip_stem=None,
    )

    # In single-query mode we keep the historical exit-code contract:
    # IO / arg errors return non-zero so shell scripts can branch on
    # `$?`. Match-zero is still success (the empty `matches: []` is
    # the answer).
    if "error" in result:
        print(json.dumps(result, ensure_ascii=False), file=sys.stderr)
        return 2

    if args.compact:
        sys.stdout.write(json.dumps(result, ensure_ascii=False))
    else:
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
