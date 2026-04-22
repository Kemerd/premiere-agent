"""Speech lane (PRIMARY): NVIDIA Parakeet TDT via ONNX Runtime.

This module is the default speech lane as of 2026-04, replacing the
HuggingFace `transformers` Whisper pipeline that previously held the
slot. The output JSON shape is BYTE-FOR-BYTE identical to both
`whisper_lane.py` and the legacy `parakeet_lane.py` (NeMo torch path),
so every downstream consumer (`pack_timelines.py`, `render.py`,
`export_fcpxml.py`) is genuinely lane-agnostic — they don't and
shouldn't know which acoustic model produced the words.

╔══════════════════════════════════════════════════════════════════╗
║  WHY ONNX, WHY A POOL, WHY PARAKEET — the long-form rationale    ║
╚══════════════════════════════════════════════════════════════════╝

The original Whisper-large-v3-turbo pipeline pinned a 32 GB Blackwell
GPU at the device limit while transcribing a SINGLE 4-minute clip
(see https://github.com/huggingface/transformers/issues/27834 — the
`return_timestamps="word"` codepath copies `encoder_attentions` to
CPU at the end of every batch, and the working set grows linearly
with batch size). At batch=16 + word timestamps + transformers
>= 4.43 we measured 26-30 GB peak instead of the ~7-8 GB the
docstring claimed. That meant ~1x real-time on a card that should do
30-40x. Unworkable for an editor that wants to grind through hours
of footage in minutes.

Three architectural changes get us back to the speed envelope we
actually want:

  1) MODEL: Parakeet TDT 0.6B in place of Whisper. NVIDIA's TDT
     (Token-Duration-Transducer) decoder is purpose-built for native
     word-level timestamps — no DTW alignment over cross-attention,
     no separate forced-alignment step. v2 is English-only and tops
     the Open ASR Leaderboard at ~320x RTFx; v3 is multilingual at
     ~200x. Both are ~600M params (vs Whisper-turbo's 809M).

  2) RUNTIME: ONNX Runtime in place of `transformers` + PyTorch.
     ORT ships native C++ bindings with multiple execution
     providers (TensorRT, CUDA, DirectML, CPU) and releases the
     GIL during `Run()`. The `transformers` pipeline runs through
     the Python-side trainer infrastructure, which spends a bunch
     of cycles on autograd graph teardown and dtype casts even at
     inference time. ORT skips all of that.

  3) PARALLELISM: a session pool, not a single big batch. Whisper's
     "bigger batch = more parallel" model has a memory-bandwidth
     ceiling (the KV cache + cross-attn maps grow with batch). ORT
     sessions are independent — N sessions with their own CUDA
     streams ⇒ the GPU's hardware scheduler overlaps them as long
     as we have unused SMs. Parakeet only saturates ~30% of a 5090's
     SMs per inference, so 8 parallel sessions ≈ 8x throughput.

╔══════════════════════════════════════════════════════════════════╗
║  ARCHITECTURE — what calls what                                  ║
╚══════════════════════════════════════════════════════════════════╝

    run_parakeet_onnx_lane_batch(videos, edit_dir, ...)
            │
            ├──► language router ─► nemo-parakeet-tdt-0.6b-v2 (en)
            │                       nemo-parakeet-tdt-0.6b-v3 (else)
            │
            ├──► OnnxSessionPool(model_id, desired_size=N)
            │       ├── _onnx_providers.resolve_providers()
            │       │       ├── try TensorRT EP    (gated by env var)
            │       │       ├── try CUDA EP
            │       │       ├── try DirectML EP    (Windows non-NV)
            │       │       └── CPU EP             (always)
            │       └── N x onnx_asr.load_model(...).with_timestamps()
            │
            ├──► for each video:
            │       wav = extract_audio_for(video)             (ffmpeg → 16k mono PCM)
            │       result = pool.transcribe_batch([wav])      (parallel under the hood)
            │       words = _onnx_to_canonical_words(result)   (project schema)
            │       words = _diarize_and_assign(...)           (optional, pyannote)
            │       json.dump(payload, transcripts/<stem>.json) (atomic write)
            │
            └──► return list[Path]  (one JSON per input video)

╔══════════════════════════════════════════════════════════════════╗
║  CANONICAL OUTPUT SHAPE — preserved exactly                       ║
╚══════════════════════════════════════════════════════════════════╝

    {
      "model": "nvidia/parakeet-tdt-0.6b-v2",
      "language": "en",
      "duration": 254.3,
      "text": "...full plain transcript...",
      "words": [
        {"type": "word",    "text": "Hello", "start": 0.12, "end": 0.34, "speaker_id": null},
        {"type": "spacing", "text": " ",     "start": 0.34, "end": 0.41},
        ...
      ],
      "sample_rate": 16000
    }

╔══════════════════════════════════════════════════════════════════╗
║  CLI                                                              ║
╚══════════════════════════════════════════════════════════════════╝

    Full lane (one or more videos, writes to <edit>/transcripts/):
        python helpers/parakeet_onnx_lane.py video.mp4 [--diarize]

    Smoke test (one wav, prints chosen EP + RTFx + JSON, writes nothing):
        python helpers/parakeet_onnx_lane.py --smoke-test clip.wav
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Local sibling imports — work whether invoked via `python helpers/<this>.py`
# or via the orchestrator's `-c` shim that adds helpers/ to sys.path.
from extract_audio import SAMPLE_RATE, extract_audio_for
from progress import install_lane_prefix, lane_progress
from wealthy import (
    is_wealthy,
    parakeet_pool_size,
    parakeet_quantization,
)

# Multi-session pool + EP ladder live in private siblings (underscore-
# prefixed) because they're implementation details — the lane is the
# public surface this module exports.
from _onnx_pool import OnnxSessionPool

# Diarizer is single-source-of-truth in whisper_lane — it operates on
# the canonical word list, has zero acoustic-model knowledge.
from whisper_lane import _diarize_and_assign, _load_hf_token


# ---------------------------------------------------------------------------
# Model identifiers
#
# onnx-asr accepts either:
#   * one of its short aliases ("nemo-parakeet-tdt-0.6b-v2")
#   * an HF Hub repo id ("istupakov/parakeet-tdt-0.6b-v2-onnx")
#   * a local directory containing the encoder/decoder/joint .onnx files
#
# We use the short aliases by default because they hit the istupakov/
# pre-converted ONNX repos which include the bundled mel preprocessor.
# ---------------------------------------------------------------------------

# Default model exposed in metadata + log lines. Carries the canonical
# `nvidia/...` name even though the ONNX bytes come from istupakov's
# converted repo — this is the model the user sees, not the binary
# format under the hood.
DEFAULT_MODEL_ID_EN = "nvidia/parakeet-tdt-0.6b-v2"   # English-only, ~320x RTFx
DEFAULT_MODEL_ID_MULTI = "nvidia/parakeet-tdt-0.6b-v3"  # multilingual, ~200x RTFx

# What we actually pass to onnx_asr.load_model() — the short aliases
# resolve to the istupakov ONNX repos via onnx-asr's built-in registry.
_ONNX_ASR_ALIAS_EN = "nemo-parakeet-tdt-0.6b-v2"
_ONNX_ASR_ALIAS_MULTI = "nemo-parakeet-tdt-0.6b-v3"

# Map ISO codes Parakeet v3 supports. Anything outside this set still
# falls through to v3 (it's better than nothing) but we log a warning
# so the user knows quality may degrade. The set is curated from
# NVIDIA's published v3 model card + their NeMo docs.
_V3_LANGS = frozenset({
    "en", "bg", "hr", "cs", "da", "nl", "et", "fi", "fr", "de",
    "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
    "sl", "es", "sv", "ru", "uk",
})

# Where we drop the per-clip JSON outputs. Identical to the directory
# whisper_lane and parakeet_lane (NeMo) write to, on purpose: the
# downstream `pack_timelines.py` reads from one canonical location
# regardless of which lane produced the data, so swapping lanes
# between sessions is a no-op.
TRANSCRIPTS_SUBDIR = "transcripts"

# Env var escape hatch for fully-air-gapped networks: if set, we point
# onnx_asr.load_model() at a local directory containing the ONNX files
# (encoder/decoder/joint + tokenizer.json + preprocessor config), and
# the loader skips the HuggingFace download path entirely.
PARAKEET_ONNX_DIR_ENV = "PARAKEET_ONNX_DIR"


# ---------------------------------------------------------------------------
# Language routing — pick v2 for English, v3 for everything else
# ---------------------------------------------------------------------------

def _resolve_model_for_language(
    language: str | None,
    *,
    explicit_model_id: str | None = None,
) -> tuple[str, str]:
    """Pick the (display name, onnx-asr alias) for a target language.

    Resolution order:
        1. `explicit_model_id` if the caller passed --model: respected
           verbatim. We don't try to be clever about cross-checking
           the alias against the language hint — power users who pin
           a model know what they're doing.
        2. None / "en" / unset                  → v2 English (fastest)
        3. ISO code in v3 supported set         → v3 multilingual
        4. Anything else (warn)                 → v3 multilingual
                                                  (still best free option)

    Returns:
        (display_id, onnx_asr_alias) — e.g. ("nvidia/parakeet-tdt-0.6b-v2",
        "nemo-parakeet-tdt-0.6b-v2"). The display id is what we record
        in the JSON's "model" field; the alias is what onnx_asr loads.
    """
    # Explicit override always wins. We try to recover the alias by
    # mapping known display names; otherwise pass the user's string
    # straight through (onnx-asr accepts repo ids and local paths).
    if explicit_model_id:
        if explicit_model_id == DEFAULT_MODEL_ID_EN:
            return DEFAULT_MODEL_ID_EN, _ONNX_ASR_ALIAS_EN
        if explicit_model_id == DEFAULT_MODEL_ID_MULTI:
            return DEFAULT_MODEL_ID_MULTI, _ONNX_ASR_ALIAS_MULTI
        # Unknown id — trust the user, use it for both sides.
        return explicit_model_id, explicit_model_id

    # Default-tier auto-routing. Empty / None / "en" → v2; otherwise v3.
    lang = (language or "en").strip().lower()
    if lang == "en" or lang == "":
        return DEFAULT_MODEL_ID_EN, _ONNX_ASR_ALIAS_EN

    # Non-English. v3 is the right model. Warn once if it's outside
    # v3's published language set so the user knows quality may dip.
    if lang not in _V3_LANGS:
        print(
            f"  [parakeet_onnx] WARN: language '{lang}' is not in "
            f"Parakeet v3's published supported set "
            f"({len(_V3_LANGS)} languages). Falling back to v3 anyway "
            f"— quality may degrade. Pin a different model with "
            f"--model or VIDEO_USE_SPEECH_LANE=whisper for full "
            f"multilingual coverage.",
            file=sys.stderr,
        )
    return DEFAULT_MODEL_ID_MULTI, _ONNX_ASR_ALIAS_MULTI


# ---------------------------------------------------------------------------
# onnx-asr result → canonical word list
#
# onnx_asr's `with_timestamps()` adapter returns a result object whose
# exact attribute layout has shifted across releases. We handle multiple
# shapes defensively so we don't break on a minor version bump:
#
#   * 0.10-0.11: result.timestamps -> list[Segment]
#                where Segment has .word/.text + .start + .end
#   * Some adapters: result.tokens -> list[(text, start, end)] tuples
#   * Some adapters: result.words  -> list of dicts
#
# When all attribute probes fail, we fall back to text-only mode (the
# transcript still gets written, just without word-level timing).
# ---------------------------------------------------------------------------

def _onnx_to_canonical_words(result, *, fallback_text: str = "") -> list[dict]:
    """Convert an onnx-asr timestamped result to the canonical list.

    Output format matches `parakeet_lane._parakeet_to_canonical_words`
    AND `whisper_lane._words_from_chunks` exactly:

        [
          {"type": "word",    "text": "Hello", "start": 0.12, "end": 0.34,
           "speaker_id": None},
          {"type": "spacing", "text": " ",     "start": 0.34, "end": 0.41},
          ...
        ]

    `speaker_id` is always None at this stage; the diarizer pass fills
    it in if --diarize was passed AND pyannote + HF_TOKEN are available.

    Synthetic spacing entries are emitted between consecutive words
    whenever there's a gap in the timestamps — Hard Rule 7 padding
    math (in pack_timelines.py) depends on these existing.
    """
    out: list[dict] = []

    # Probe the result shape. We accept the first one that yields
    # iterable timestamp entries with start/end/text — order matters,
    # `.timestamps` is the documented public attribute on current
    # onnx-asr releases so we try it first.
    raw_entries = None
    for attr in ("timestamps", "tokens", "words"):
        candidate = getattr(result, attr, None)
        if candidate:
            raw_entries = candidate
            break

    # Last-ditch: result might be a dict (old adapter), or a tuple
    # of (text, segments). Walk the few known shapes.
    if raw_entries is None and isinstance(result, dict):
        raw_entries = (
            result.get("timestamps")
            or result.get("tokens")
            or result.get("words")
        )

    if not raw_entries:
        # No timestamps available — return empty list. The caller
        # writes the JSON with text-only and an empty words[] which
        # downstream code tolerates (the phrase grouper just skips
        # empty videos).
        return out

    # Walk the entries in order, emitting word + spacing pairs.
    prev_end: float | None = None
    for entry in raw_entries:
        # Each entry can be: a dict, a dataclass, or a 3-tuple
        # (text, start, end). Handle all three with one helper.
        text, start, end = _extract_entry_fields(entry)
        if text is None or start is None or end is None:
            continue

        text = text.strip()
        if not text:
            continue

        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue

        # Synthesize a spacing entry covering any gap since the last
        # word. Mirrors what whisper_lane and parakeet_lane both do —
        # the phrase grouper uses these to detect long pauses for
        # cut candidate suggestions.
        if prev_end is not None and start_f > prev_end:
            out.append({
                "type": "spacing",
                "text": " ",
                "start": float(prev_end),
                "end": float(start_f),
            })

        out.append({
            "type": "word",
            "text": text,
            "start": start_f,
            "end": end_f,
            # Diarizer fills this in later; phrase grouper tolerates None.
            "speaker_id": None,
        })
        prev_end = end_f

    return out


def _extract_entry_fields(entry) -> tuple[str | None, float | None, float | None]:
    """Pull (text, start, end) from a single timestamp entry.

    Defensive against the four shapes onnx-asr has shipped across
    minor versions:
        * dataclass with .text/.word + .start + .end
        * dict with the same keys (or "start_time"/"end_time")
        * 3-tuple (text, start, end)
        * 4-tuple (text, start, end, score)  -- score ignored
    """
    # Tuple/list shape — common when adapters return raw decoder output.
    if isinstance(entry, (tuple, list)):
        if len(entry) >= 3:
            return str(entry[0]), entry[1], entry[2]
        return None, None, None

    # Object/dict shape — pull each field by trying multiple names.
    def _get(obj, *names):
        for n in names:
            v = getattr(obj, n, None) if not isinstance(obj, dict) else obj.get(n)
            if v is not None:
                return v
        return None

    text = _get(entry, "text", "word", "token")
    start = _get(entry, "start", "start_time", "begin")
    end = _get(entry, "end", "end_time", "stop")
    return (str(text) if text is not None else None, start, end)


def _result_text(result) -> str:
    """Pull the plain-text transcript out of an onnx-asr result.

    Defensive against attribute name drift; falls back to joining the
    timestamp entries' text if `.text` is missing entirely.
    """
    txt = getattr(result, "text", None)
    if txt is None and isinstance(result, dict):
        txt = result.get("text") or result.get("transcript")
    if txt:
        return str(txt).strip()

    # Fallback — stitch words from the timestamp list. Less ideal
    # (we lose punctuation reconstruction) but keeps the JSON valid.
    canon = _onnx_to_canonical_words(result)
    return " ".join(w["text"] for w in canon if w.get("type") == "word")


# ---------------------------------------------------------------------------
# Per-video processing — wraps cache check, transcribe, diarize, JSON write
# ---------------------------------------------------------------------------

def _process_one(
    pool: OnnxSessionPool,
    video_path: Path,
    edit_dir: Path,
    *,
    display_model_id: str,
    language: str | None,
    diarize: bool,
    num_speakers: int | None,
    force: bool,
) -> Path:
    """Transcribe one video using a pre-loaded session pool.

    Mirrors `parakeet_lane._process_one` and `whisper_lane._process_one`
    cache contracts byte-for-byte — that's how we keep the swap between
    lanes invisible to downstream consumers.

    Returns the path to the written JSON (cache-hit or fresh).
    """
    transcripts_dir = (edit_dir / TRANSCRIPTS_SUBDIR).resolve()
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video_path.stem}.json"

    # mtime cache check — same rule as the other speech lanes:
    # JSON newer than source video? Reuse the JSON. Otherwise rebuild.
    # `force` flag bypasses the check entirely (useful when the user
    # changes diarize / language / model and wants to re-run).
    if not force and out_path.exists():
        try:
            if out_path.stat().st_mtime >= video_path.stat().st_mtime:
                print(f"  parakeet_onnx_lane cache hit: {out_path.name}")
                return out_path
        except OSError:
            # Stat failed — fall through and re-transcribe defensively.
            pass

    # Pull the 16 kHz mono WAV out of the source video. extract_audio
    # itself caches under <edit>/audio_16k/<stem>.wav so this is a
    # no-op when running side-by-side with the audio (PANNs) lane.
    wav_path = extract_audio_for(video_path, edit_dir, verbose=True)

    t0 = time.time()
    print(f"  parakeet_onnx: transcribing {wav_path.name} via pool size {pool.size}")

    # Single-clip dispatch through the pool. We pass [wav_path] because
    # the pool's transcribe_batch is built to fan out across multiple
    # WAVs in parallel — when called with N=1 it just runs serially
    # in one session but still respects the timeout / error semantics.
    results = pool.transcribe_batch([wav_path])
    if not results:
        raise RuntimeError(f"pool returned no results for {wav_path}")

    result = results[0]

    # Worker may have stuffed an Exception into the slot rather than
    # raising it directly (so one bad clip doesn't abort the batch).
    # Surface it here as a real exception — the outer retry loop will
    # decide whether to retry, fall back, or give up.
    if isinstance(result, BaseException):
        raise result

    words = _onnx_to_canonical_words(result)
    text = _result_text(result)

    # ── Optional diarization ──────────────────────────────────────────
    # We re-use whisper_lane's diarizer verbatim. It reads the WAV,
    # runs pyannote/speaker-diarization-3.1, and assigns speaker_id
    # to each word by majority overlap — completely orthogonal to
    # which model produced the words.
    if diarize:
        token = _load_hf_token()
        if not token:
            print(
                "  diarize: HF_TOKEN not set in .env or environment, "
                "skipping speaker diarization.",
                file=sys.stderr,
            )
        else:
            words = _diarize_and_assign(
                words, wav_path, token, num_speakers=num_speakers
            )

    # Derive duration from the last timestamp. Falls back to 0.0 on
    # empty / silent / decoder-bailed inputs — still produces valid JSON.
    duration = 0.0
    for w in reversed(words):
        end = w.get("end")
        if end is not None:
            duration = float(end)
            break

    payload = {
        "model": display_model_id,
        # Honor the caller's hint; default "en" since v2 (the most
        # common case) is English-only. v3 sessions are still labeled
        # with whatever the user requested — the model itself doesn't
        # emit detected language.
        "language": (language or "en"),
        "duration": duration,
        "text": text,
        "words": words,
        "sample_rate": SAMPLE_RATE,
    }

    # Atomic write: rename of an already-flushed file is atomic on
    # both POSIX and Windows, so a Ctrl-C between write and rename
    # leaves the cache in a coherent state (either old JSON or no
    # JSON, never a half-written one).
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)

    dt = time.time() - t0
    n_words = sum(1 for w in words if w.get("type") == "word")
    kb = out_path.stat().st_size / 1024
    rtfx = (duration / dt) if dt > 0 else 0.0
    print(
        f"  parakeet_onnx_lane done: {n_words} words, {duration:.1f}s "
        f"audio, {dt:.1f}s wall ({rtfx:.1f}x RTFx), {kb:.1f} KB → "
        f"{out_path.name}"
    )
    return out_path


# ---------------------------------------------------------------------------
# OOM-resilient pool builder
#
# Pool construction can fail mid-way through if our VRAM clamp guessed
# wrong (driver fragmentation, leftover allocations from a prior lane,
# etc.). Halve and retry rather than crash — a 4-session pool is still
# 4x faster than no pool at all.
# ---------------------------------------------------------------------------

def _build_pool_with_retry(
    *,
    onnx_alias: str,
    desired_size: int,
    quantization: str | None,
    prefer_tensorrt: bool,
) -> OnnxSessionPool:
    """Build the OnnxSessionPool, halving the pool on CUDA OOM.

    The pool already does VRAM-aware clamping at construction, but that
    clamp is based on `nvidia-smi` reports which can lag behind real
    allocations (especially right after another lane has finished and
    its CUDA context hasn't fully torn down yet). This wrapper catches
    the cudaMalloc failure path and tries again at half size.
    """
    n = max(1, int(desired_size))
    while True:
        try:
            return OnnxSessionPool(
                model_id=onnx_alias,
                desired_size=n,
                quantization=quantization,
                prefer_tensorrt=prefer_tensorrt,
            )
        except (MemoryError, RuntimeError) as e:
            # OOM signatures vary by driver version. We match on the
            # error message text since neither MemoryError nor the
            # opaque ORT RuntimeError exposes a structured code.
            msg = str(e).lower()
            is_oom = (
                "out of memory" in msg
                or "cudamalloc" in msg
                or "cuda_error_out_of_memory" in msg
                or "cublas_status_alloc_failed" in msg
            )
            if not is_oom or n <= 1:
                raise
            new_n = max(1, n // 2)
            print(
                f"  [parakeet_onnx] OOM building pool of {n} session(s); "
                f"halving to {new_n} and retrying.",
                file=sys.stderr,
            )
            n = new_n


# ---------------------------------------------------------------------------
# Batch entry point — the function preprocess.py dispatches to
# ---------------------------------------------------------------------------

def run_parakeet_onnx_lane_batch(
    video_paths: list[Path],
    edit_dir: Path,
    *,
    # Same kwargs as parakeet_lane.run_parakeet_lane_batch — caller
    # interchangeable. Defaults are wired so passing nothing produces
    # the right behavior on a typical 12-24 GB consumer card.
    model_id: str | None = None,
    device: str = "cuda:0",
    dtype_name: str = "fp16",
    batch_size: int | None = None,         # accepted for signature compat; we
                                           # use pool size, not batch size
    chunk_length_s: int = 30,              # accepted for signature compat; ignored
    language: str | None = None,
    diarize: bool = False,
    num_speakers: int | None = None,
    force: bool = False,
    # ONNX-specific knobs (not present on the other lanes' signatures):
    pool_size: int | None = None,
    prefer_tensorrt: bool | None = None,
) -> list[Path]:
    """Run Parakeet ONNX on N videos with the session pool loaded ONCE.

    The pool stays warm across all videos in the batch — this is the
    central performance win over the legacy Whisper lane, which
    rebuilt its pipeline per process invocation.

    Args:
        video_paths: Source videos. Will be ffmpeg-extracted to
            16 kHz mono WAV (cached under edit/audio_16k/).
        edit_dir: Output directory. JSON drops under
            edit_dir/transcripts/<stem>.json.
        model_id: Optional display model id; if None we auto-route by
            language (en→v2, else→v3).
        device: Carried through for signature compatibility with the
            other lanes. The actual device is decided by the EP ladder
            in `_onnx_providers.py` — we don't need a per-call device
            string because ORT handles that internally per provider.
        dtype_name: "fp16" / "fp32" / "int8". fp16 is the default and
            recommended; int8 cuts VRAM in half at small WER cost.
        batch_size: Accepted for signature compat with the other lanes.
            ONNX uses pool_size, not batch_size — the value is ignored.
        chunk_length_s: Accepted for signature compat. onnx-asr's
            VAD-aware chunker handles long-form audio internally.
        language: ISO code hint (e.g. "en", "fr"). Drives the v2/v3
            routing decision when `model_id` is None.
        diarize: Run pyannote diarization after transcription.
        num_speakers: Optional fixed speaker count for the diarizer.
        force: Bypass the per-clip mtime cache.
        pool_size: Override the auto-resolved pool size (default reads
            from VIDEO_USE_PARAKEET_POOL_SIZE env var or wealthy.py
            constants).
        prefer_tensorrt: Override TRT preference. None = honor the
            VIDEO_USE_PARAKEET_TRT env var; True = force TRT attempt;
            False = skip TRT entirely (use CUDA/DML/CPU only).

    Returns:
        list[Path] — one transcript JSON path per input video, in
        the same order as `video_paths`.
    """
    if not video_paths:
        return []

    # Resolve dtype → onnx-asr quantization argument. fp32 maps to
    # `None` (the loader's default), fp16 also maps to None (since
    # onnx-asr's stock fp16 export is the default), int8 forwards
    # explicitly.
    quant = parakeet_quantization() if dtype_name in (None, "fp16") else (
        None if dtype_name == "fp32" else dtype_name.lower()
    )

    # ── Pre-flight cache check ──────────────────────────────────────
    # Skip the (expensive) pool construction entirely when every
    # video is already cache-fresh. Same optimization the other
    # speech lanes have. ~3-4s saved per session for cache-hit runs.
    transcripts_dir = (edit_dir / TRANSCRIPTS_SUBDIR).resolve()
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    if not force:
        all_fresh = True
        for v in video_paths:
            out = transcripts_dir / f"{v.stem}.json"
            try:
                if not out.exists() or out.stat().st_mtime < v.stat().st_mtime:
                    all_fresh = False
                    break
            except OSError:
                all_fresh = False
                break
        if all_fresh:
            print(
                f"  parakeet_onnx_lane: all {len(video_paths)} cache "
                f"hits, skipping pool construction"
            )
            return [transcripts_dir / f"{v.stem}.json" for v in video_paths]

    # ── Language → model routing ────────────────────────────────────
    display_id, onnx_alias = _resolve_model_for_language(
        language, explicit_model_id=model_id,
    )

    # Honor the air-gap escape hatch — overrides BOTH model selections
    # because the user has put the .onnx files at the path themselves.
    local_dir = os.environ.get(PARAKEET_ONNX_DIR_ENV, "").strip()
    if local_dir:
        print(
            f"  parakeet_onnx: loading model from local dir "
            f"{PARAKEET_ONNX_DIR_ENV}={local_dir} (zero network)"
        )
        onnx_alias = local_dir

    # ── Pool sizing ─────────────────────────────────────────────────
    desired_n = pool_size if pool_size is not None else parakeet_pool_size()

    # ── TRT preference ──────────────────────────────────────────────
    # Honor explicit kwarg first, otherwise let resolve_providers()
    # do its env-var probe internally. We pass True/True so the
    # ladder is built with TRT in scope; the env var inside
    # resolve_providers is what actually decides whether it appears.
    use_trt = True if prefer_tensorrt is None else bool(prefer_tensorrt)

    # ── Build the pool (with OOM retry) ─────────────────────────────
    print(
        f"  parakeet_onnx: lane batch — model={display_id} "
        f"language={language or 'auto'} desired_pool={desired_n} "
        f"quant={quant or 'fp16'}"
    )
    pool = _build_pool_with_retry(
        onnx_alias=onnx_alias,
        desired_size=desired_n,
        quantization=quant,
        prefer_tensorrt=use_trt,
    )

    # ── Process every video, with progress bar ──────────────────────
    out_paths: list[Path] = []
    try:
        with lane_progress(
            "parakeet_onnx",
            total=len(video_paths),
            unit="video",
            desc="speech transcription (parakeet onnx)",
        ) as bar:
            for v in video_paths:
                bar.start_item(v.name)
                try:
                    out_paths.append(_process_one(
                        pool, v, edit_dir,
                        display_model_id=display_id,
                        language=language,
                        diarize=diarize,
                        num_speakers=num_speakers,
                        force=force,
                    ))
                except Exception as e:
                    # One bad clip shouldn't kill the batch. Log it,
                    # write a sentinel JSON so cache logic doesn't try
                    # again next run, and move on. Hard Rule 7 padding
                    # (in pack_timelines.py) tolerates empty word lists.
                    print(
                        f"  parakeet_onnx: video {v.name} failed: "
                        f"{type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    sentinel = transcripts_dir / f"{v.stem}.json"
                    sentinel.write_text(json.dumps({
                        "model": display_id,
                        "language": language or "en",
                        "duration": 0.0,
                        "text": "",
                        "words": [],
                        "sample_rate": SAMPLE_RATE,
                        "_error": f"{type(e).__name__}: {e}",
                    }, indent=2), encoding="utf-8")
                    out_paths.append(sentinel)
                bar.update(advance=1, item=v.name)
    finally:
        # Always release pool sessions — otherwise a downstream lane
        # in the same process would see N x 1.2 GB still pinned.
        pool.close()

    return out_paths


def run_parakeet_onnx_lane(
    video_path: Path,
    edit_dir: Path,
    **kwargs,
) -> Path:
    """Single-video convenience wrapper."""
    return run_parakeet_onnx_lane_batch([video_path], edit_dir, **kwargs)[0]


# ---------------------------------------------------------------------------
# Smoke test — load + transcribe + report. Writes nothing to disk.
# ---------------------------------------------------------------------------

def _smoke_test(
    wav_path: Path,
    *,
    language: str | None,
    pool_size: int,
    prefer_tensorrt: bool,
) -> int:
    """Verify the install end-to-end on a single WAV.

    Prints (in order):
        a) the resolved EP ladder (the actual provider chosen by ORT)
        b) achieved RTFx (audio_seconds / wall_seconds)
        c) word count + duration
        d) the full canonical JSON

    Returns 0 on success, 1 on failure — usable as a CLI exit code.
    """
    import wave

    if not wav_path.exists():
        print(f"smoke-test: WAV not found: {wav_path}", file=sys.stderr)
        return 1

    # Probe audio duration up front so we can compute RTFx without
    # waiting for the result. soundfile would also work but `wave` is
    # stdlib, no extra import cost.
    try:
        with wave.open(str(wav_path), "rb") as wf:
            audio_dur = wf.getnframes() / float(wf.getframerate())
    except Exception as e:
        print(f"smoke-test: could not read WAV header: {e}", file=sys.stderr)
        audio_dur = 0.0

    display_id, onnx_alias = _resolve_model_for_language(language)
    quant = parakeet_quantization()

    print(f"[smoke-test] WAV         : {wav_path}")
    print(f"[smoke-test] duration    : {audio_dur:.2f}s")
    print(f"[smoke-test] model       : {display_id}  ({onnx_alias})")
    print(f"[smoke-test] quant       : {quant or 'fp16'}")
    print(f"[smoke-test] pool size   : {pool_size}")
    print(f"[smoke-test] prefer TRT  : {prefer_tensorrt}")

    # Build a single-session pool — the smoke test only ever runs one
    # WAV so multi-session would just waste VRAM during the load.
    pool = _build_pool_with_retry(
        onnx_alias=onnx_alias,
        desired_size=max(1, pool_size),
        quantization=quant,
        prefer_tensorrt=prefer_tensorrt,
    )
    try:
        # Probe the actual EP that ORT picked. Each session in the
        # pool is wrapped by onnx-asr — the underlying ORT session
        # is exposed via various private attributes depending on
        # version. We try a couple of likely paths.
        chosen_ep = _probe_chosen_ep(pool)
        print(f"[smoke-test] chosen EP   : {chosen_ep}")

        t0 = time.time()
        results = pool.transcribe_batch([wav_path])
        wall = time.time() - t0

        if not results or isinstance(results[0], BaseException):
            err = results[0] if results else "empty result list"
            print(f"[smoke-test] FAIL: {err}", file=sys.stderr)
            return 1

        result = results[0]
        words = _onnx_to_canonical_words(result)
        text = _result_text(result)
        n_words = sum(1 for w in words if w.get("type") == "word")
        rtfx = (audio_dur / wall) if wall > 0 else 0.0

        print(f"[smoke-test] wall        : {wall:.2f}s")
        print(f"[smoke-test] RTFx        : {rtfx:.1f}x")
        print(f"[smoke-test] words       : {n_words}")
        print(f"[smoke-test] text        : {text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"[smoke-test] full JSON below:")
        print(json.dumps({
            "model": display_id,
            "language": language or "en",
            "duration": audio_dur,
            "text": text,
            "words": words,
            "sample_rate": SAMPLE_RATE,
            "_meta": {
                "wall_seconds": wall,
                "rtfx": rtfx,
                "chosen_ep": chosen_ep,
                "pool_size": pool.size,
            },
        }, indent=2))
        return 0
    finally:
        pool.close()


def _probe_chosen_ep(pool: OnnxSessionPool) -> str:
    """Best-effort: ask the first session what EP it actually loaded.

    onnx-asr wraps each ONNX file in its own InferenceSession; we
    grab the encoder session (the heavy one) and call
    `get_providers()` on it. If the wrapper doesn't expose the raw
    session we just report what `resolve_providers` would have
    returned — still useful diagnostic info.
    """
    try:
        # Pull one session out of the pool's queue. We don't care
        # about putting it back here — the smoke test is single-shot.
        if pool.size > 0 and pool._sessions:
            wrapper = pool._sessions[0]
            # onnx-asr stores the underlying ORT sessions on the
            # adapter; the encoder is the largest and the most
            # informative. Common attribute names across versions:
            for attr in ("_encoder", "encoder", "_session", "session"):
                ort_sess = getattr(wrapper, attr, None)
                if ort_sess is not None and hasattr(ort_sess, "get_providers"):
                    return ort_sess.get_providers()[0]
    except Exception:
        pass
    # Fallback to whatever the ladder says is at the top.
    from _onnx_providers import resolve_providers
    ladder = resolve_providers(prefer_tensorrt=True)
    head = ladder[0] if ladder else "unknown"
    return head[0] if isinstance(head, tuple) else str(head)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Speech lane (PRIMARY): NVIDIA Parakeet TDT via ONNX Runtime",
    )
    ap.add_argument(
        "video", type=Path, nargs="?", default=None,
        help="Source video (full lane mode) or WAV (--smoke-test mode)",
    )
    ap.add_argument(
        "--smoke-test", action="store_true",
        help="Verify install end-to-end on one WAV; print EP, RTFx, JSON. "
             "Writes nothing to disk.",
    )
    ap.add_argument("--edit-dir", type=Path, default=None,
                    help="Edit output dir (default: <video parent>/edit)")
    ap.add_argument("--model", default=None,
                    help="Display model id override (auto-routed by --language otherwise)")
    ap.add_argument("--device", default="cuda:0",
                    help="Carried for signature compat; EP ladder owns the real device.")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "int8"])
    ap.add_argument("--language", default=None,
                    help="ISO language hint; drives v2 (en) vs v3 (multilingual).")
    ap.add_argument("--diarize", action="store_true",
                    help="Run pyannote speaker diarization (needs HF_TOKEN).")
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--pool-size", type=int, default=None,
                    help="Override session pool size (default: env / wealthy.py).")
    ap.add_argument("--no-tensorrt", action="store_true",
                    help="Force-skip TensorRT EP even if VIDEO_USE_PARAKEET_TRT=1.")
    ap.add_argument("--wealthy", action="store_true",
                    help="Speed knob for 24GB+ cards. Also reads VIDEO_USE_WEALTHY=1.")
    ap.add_argument("--force", action="store_true", help="Bypass cache.")
    args = ap.parse_args()

    install_lane_prefix()

    if args.video is None:
        ap.error("video / wav path is required")

    target = args.video.resolve()
    if not target.exists():
        sys.exit(f"path not found: {target}")

    prefer_tensorrt = not args.no_tensorrt

    # ── Smoke test branch ──────────────────────────────────────────
    if args.smoke_test:
        # Pool size 1 by default in smoke mode — we're not benchmarking
        # parallelism, we're verifying the install. Caller can still
        # override with --pool-size.
        n = args.pool_size if args.pool_size is not None else 1
        rc = _smoke_test(
            target,
            language=args.language,
            pool_size=n,
            prefer_tensorrt=prefer_tensorrt,
        )
        sys.exit(rc)

    # ── Full lane branch ───────────────────────────────────────────
    edit_dir = (args.edit_dir or (target.parent / "edit")).resolve()

    # Wealthy mode: caller may pass --wealthy on the CLI; we propagate
    # to env so the pool-size resolver picks it up. is_wealthy() also
    # honors VIDEO_USE_WEALTHY=1 if the orchestrator already set it.
    if args.wealthy:
        os.environ["VIDEO_USE_WEALTHY"] = "1"

    n = args.pool_size if args.pool_size is not None else parakeet_pool_size(args.wealthy)

    run_parakeet_onnx_lane(
        video_path=target,
        edit_dir=edit_dir,
        model_id=args.model,
        device=args.device,
        dtype_name=args.dtype,
        language=args.language,
        diarize=args.diarize,
        num_speakers=args.num_speakers,
        force=args.force,
        pool_size=n,
        prefer_tensorrt=prefer_tensorrt,
    )


if __name__ == "__main__":
    main()
