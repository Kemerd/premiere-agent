"""Speaker diarization on the canonical word list.

Diarization is model-agnostic: it operates on a list of `{type, text,
start, end, ...}` word entries (the canonical schema every speech lane
emits) and a 16 kHz mono WAV. It assigns `speaker_id` to each `word`
entry by majority overlap with pyannote's diarized turns.

This module exists so the speech lane stack (currently
`parakeet_onnx_lane.py` + the optional `parakeet_lane.py` NeMo
fallback) can call into a single source of truth without depending on
any particular acoustic-model module.

Public API:

    from diarize import diarize_and_assign, load_hf_token

    token = load_hf_token()
    if token:
        words = diarize_and_assign(words, wav_path, token,
                                   num_speakers=num_speakers)

`pyannote.audio` is gated behind the `[diarize]` extra in
`pyproject.toml`; missing imports surface a friendly stderr line and
return the words unmodified rather than crashing the lane.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# .env loader
#
# Single-purpose dotenv reader so users don't have to install
# python-dotenv just to expose HF_TOKEN. We check the project-root
# .env (sibling of `helpers/`) first, then a CWD-relative .env, then
# the process environment.
# ---------------------------------------------------------------------------

def load_hf_token() -> str | None:
    """Read HF_TOKEN from .env or environment. Returns None if absent.

    Diarization needs it (pyannote/speaker-diarization-3.1 is gated on
    the HF Hub); raw transcription does not. The caller decides how
    to behave when None — typically by skipping diarization with a
    one-line warning.
    """
    # Sibling-of-helpers .env first (project root), then CWD .env.
    candidates = [
        Path(__file__).resolve().parent.parent / ".env",
        Path(".env"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "HF_TOKEN":
                    v = v.strip().strip('"').strip("'")
                    if v:
                        return v
        except OSError:
            continue
    v = os.environ.get("HF_TOKEN", "").strip()
    return v or None


# Backwards-compatible alias for code that imported the underscore-
# prefixed legacy name. Both names point at the same function so we
# don't have to ripple a one-shot rename through every caller.
_load_hf_token = load_hf_token


# ---------------------------------------------------------------------------
# Diarize + assign
#
# pyannote.audio direct (no whisperx wrapper). Works on the canonical
# word list any speech lane produces — completely orthogonal to which
# acoustic model wrote those words.
# ---------------------------------------------------------------------------

def diarize_and_assign(
    words: list[dict],
    wav_path: Path,
    hf_token: str,
    *,
    num_speakers: int | None = None,
) -> list[dict]:
    """Run pyannote diarization on the WAV and tag each word with `speaker_id`.

    The assignment rule is "majority overlap": for each `{type:'word'}`
    entry we find the diarized turn with the largest overlap with the
    word's [start, end] span. Words with zero overlap (rare — usually
    inter-segment ASR artifacts at the very edge of the audio) keep
    `speaker_id=None`. Downstream phrase grouping handles None
    gracefully.

    Args:
        words: canonical word list (mutated in place; also returned for
            convenience). Non-`word` entries (`spacing`, `audio_event`,
            ...) are skipped — they don't carry a speaker.
        wav_path: 16 kHz mono PCM WAV that the words came from. Must
            be the same audio that produced the timestamps so the
            overlap arithmetic is meaningful.
        hf_token: HuggingFace access token with the
            `pyannote/speaker-diarization-3.1` model card accepted on
            the user's HF account. Pass via `load_hf_token()`.
        num_speakers: optional hard speaker count. When provided,
            pyannote skips clustering and assigns exactly N labels;
            useful for known-cardinality interviews and panels.

    Returns the same `words` list (mutated). On any pyannote import or
    runtime failure we log a one-line warning to stderr and return the
    words unchanged so the lane still completes.
    """
    try:
        # Local import: pyannote pulls torch + a ~600 MB model and we
        # only want that hit when diarization is actually requested.
        from pyannote.audio import Pipeline as PyannotePipeline
    except ImportError:
        print(
            "  diarize: pyannote.audio not installed. "
            "Run `pip install -e .[diarize]` to enable speaker IDs.",
            file=sys.stderr,
        )
        return words

    print("  diarize: loading pyannote/speaker-diarization-3.1")
    pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    diarize_kwargs = {}
    if num_speakers is not None:
        # Hard speaker-count hint. pyannote bypasses agglomerative
        # clustering when this is set, which is faster AND avoids
        # the 1-vs-2-speaker oscillation that single-speaker monologues
        # sometimes trigger in the auto path.
        diarize_kwargs["num_speakers"] = num_speakers

    diarization = pipeline(str(wav_path), **diarize_kwargs)

    # Flatten + sort by start so the per-word overlap loop below can
    # walk segments linearly (O(N+M)) instead of binary-searching
    # every word against every segment (O(N*M)).
    segments = sorted(
        ((float(seg.start), float(seg.end), label)
         for seg, _, label in diarization.itertracks(yield_label=True)),
        key=lambda s: s[0],
    )
    if not segments:
        return words

    print(
        f"  diarize: {len(segments)} speaker turns, "
        f"{len(set(s[2] for s in segments))} distinct speakers"
    )

    # Moving-window assignment. `si` is the index of the first segment
    # whose end > current word's start; we never need to look earlier
    # because words come in time order.
    si = 0
    for w in words:
        if w.get("type") != "word":
            continue
        ws = float(w.get("start", 0.0))
        we = float(w.get("end", ws))

        # Skip past segments that ended strictly before this word starts.
        while si < len(segments) and segments[si][1] <= ws:
            si += 1

        # Scan forward for the segment with the largest overlap. Bail
        # the inner loop the moment a segment STARTS after this word
        # ends — by sort order, no later segment can overlap either.
        best_overlap = 0.0
        best_label: str | None = None
        for j in range(si, len(segments)):
            ss, se, label = segments[j]
            if ss >= we:
                break
            overlap = min(we, se) - max(ws, ss)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label

        if best_label is not None:
            # Normalize "SPEAKER_00" -> "speaker_0" so downstream
            # display logic (which strips "speaker_" -> "0") renders
            # "S0", "S1", etc. consistently regardless of pyannote's
            # zero-padding choice in the underlying model.
            normalized = best_label.lower()
            if normalized.startswith("speaker_"):
                tail = normalized[len("speaker_"):].lstrip("0") or "0"
                normalized = f"speaker_{tail}"
            w["speaker_id"] = normalized

    return words


# Backwards-compatible alias for code that imported the underscore-
# prefixed legacy name.
_diarize_and_assign = diarize_and_assign
