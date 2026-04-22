"""Speech lane: insanely-fast-whisper-style transcription.

We don't shell out to the `insanely-fast-whisper` CLI — that subprocess
re-loads the model on every call and we want the model resident across all
sources in a batch. Instead we use the same components IFW does internally
(transformers' Whisper pipeline + fp16 + batched chunking + FA2 when
available) directly. See https://github.com/Vaibhavs10/insanely-fast-whisper
for the canonical recipe; this file is the in-process equivalent.

Output JSON shape — matched to the existing render.py `build_master_srt`
contract so SRT generation works without modification:

    {
      "language": "en",
      "duration": 43.0,
      "text": "...full plain transcript...",
      "words": [
        {"type": "word", "text": "Ninety", "start": 2.52, "end": 2.78,
         "speaker_id": "speaker_0"},
        {"type": "spacing", "text": " ", "start": 2.78, "end": 2.81},
        {"type": "word", "text": "percent", "start": 2.81, "end": 3.09,
         "speaker_id": "speaker_0"},
        ...
      ]
    }

`speaker_id` is None unless --diarize was passed AND pyannote.audio +
HF_TOKEN are available. The packed-transcript phrase grouper handles
missing speaker IDs gracefully.

Usage:
    python helpers/whisper_lane.py <video> [--diarize] [--language en]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Local sibling imports — work whether invoked via -m helpers.whisper_lane
# or directly as a script (helpers/ is on sys.path either way).
from extract_audio import SAMPLE_RATE, extract_audio_for
from progress import install_lane_prefix, lane_progress
from wealthy import WHISPER_BATCH, is_wealthy


# ---------------------------------------------------------------------------
# Tunables — picked from the IFW benchmark table for an A100 / 3060 sweet
# spot. batch_size=24 is what IFW recommends for large-v3 + FA2; drop to 8
# on smaller cards via the CLI flag.
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "openai/whisper-large-v3"
DEFAULT_BATCH_SIZE = 24
DEFAULT_CHUNK_LENGTH_S = 30   # Whisper's native receptive field
TRANSCRIPTS_SUBDIR = "transcripts"


# ---------------------------------------------------------------------------
# Parakeet fallback plumbing
#
# When a corporate proxy blocks HuggingFace (NVIDIA intranet, restricted
# enterprise networks, etc.) the Whisper model download will fail at the
# `_build_pipeline` step. We catch that specific failure mode and silently
# swap in NVIDIA Parakeet via `parakeet_lane.run_parakeet_lane_batch`.
#
# Sentinel file: once we've confirmed Whisper is blocked on this machine
# we write `~/.video-use-premiere/whisper_blocked` with a timestamp. For
# the next BLOCKED_TTL_DAYS days we skip the Whisper attempt entirely
# and go straight to Parakeet — saves the ~3-5s download timeout per
# session on a chronically blocked network.
# ---------------------------------------------------------------------------

BLOCKED_SENTINEL = Path.home() / ".video-use-premiere" / "whisper_blocked"
BLOCKED_TTL_DAYS = 7


def _whisper_blocked_recently() -> bool:
    """True if the sentinel exists and is younger than BLOCKED_TTL_DAYS.

    The TTL exists so a temporary network blip doesn't permanently pin
    the user to Parakeet — a week later we re-attempt Whisper and update
    the sentinel based on the new outcome. The sentinel lives outside
    the per-session edit/ dir (Hard Rule 12 exception, same as the
    health.json cache) because network blockage is a per-machine
    property, not a per-project one.
    """
    if not BLOCKED_SENTINEL.exists():
        return False
    try:
        age_days = (time.time() - BLOCKED_SENTINEL.stat().st_mtime) / 86400.0
        return age_days < BLOCKED_TTL_DAYS
    except OSError:
        return False


def _mark_whisper_blocked(reason: str) -> None:
    """Touch the sentinel so the next session skips the Whisper attempt.

    The reason string is written to the sentinel for forensic value —
    user can `cat` it to see WHY their session was on Parakeet.
    """
    try:
        BLOCKED_SENTINEL.parent.mkdir(parents=True, exist_ok=True)
        BLOCKED_SENTINEL.write_text(
            f"whisper_blocked_at={time.time():.0f}\n"
            f"reason={reason}\n",
            encoding="utf-8",
        )
    except OSError as e:
        # Non-fatal — fallback still works, we just won't remember it
        # next session. Log and move on.
        print(f"  whisper_lane: could not write block sentinel: {e}",
              file=sys.stderr)


def _clear_whisper_blocked() -> None:
    """Remove the sentinel — called when Whisper succeeds after we
    previously thought it was blocked. Network changed, proxy lifted,
    user joined a different VPN, whatever."""
    try:
        if BLOCKED_SENTINEL.exists():
            BLOCKED_SENTINEL.unlink()
    except OSError:
        pass


def _is_blocked_exception(exc: BaseException) -> bool:
    """Heuristic: does this exception look like 'HF download was blocked'
    rather than 'something else broke'?

    Caught families (most specific to most general):
        * huggingface_hub.errors.HfHubHTTPError    — HTTP 4xx/5xx from HF
        * huggingface_hub.errors.OfflineModeIsEnabled  — HF_HUB_OFFLINE=1
        * urllib3 / requests ConnectionError        — proxy ate the request
        * socket.timeout / TimeoutError             — proxy stalled
        * OSError with a network-flavored message   — transformers wraps lots

    We deliberately do NOT match generic OSError unless the message
    smells like network failure — otherwise a missing file or a CUDA OOM
    would silently downgrade us to Parakeet and the user would never
    know their GPU is broken.
    """
    # Fast-path: known HuggingFace exception types if the lib is present.
    try:
        from huggingface_hub.errors import (
            HfHubHTTPError, OfflineModeIsEnabled, LocalEntryNotFoundError,
        )
        if isinstance(exc, (HfHubHTTPError, OfflineModeIsEnabled,
                            LocalEntryNotFoundError)):
            return True
    except ImportError:
        pass

    # Generic network errors any HTTP client could raise.
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True

    # OSError + network-flavored message. transformers wraps a LOT of
    # download failures in OSError("We couldn't connect to ..."), so we
    # peek at the message rather than swallow every OSError blindly.
    if isinstance(exc, OSError):
        msg = str(exc).lower()
        network_smells = (
            "couldn't connect", "could not connect",
            "connection error", "connection reset", "connection refused",
            "proxy", "ssl", "certificate", "tls",
            "name or service not known", "name resolution",
            "max retries exceeded", "remote disconnected",
            "403", "407", "451",  # forbidden / proxy-auth / unavailable-legal
            "offline mode",
        )
        if any(s in msg for s in network_smells):
            return True

    return False


# ---------------------------------------------------------------------------
# .env loader — same one-liner the old transcribe.py used, kept so users
# don't have to install python-dotenv just for HF_TOKEN.
# ---------------------------------------------------------------------------

def _load_hf_token() -> str | None:
    """Read HF_TOKEN from .env or environment. None if absent.

    Diarization needs it; raw transcription does not. Caller decides what
    to do with None.
    """
    for candidate in [Path(__file__).resolve().parent.parent / ".env", Path(".env")]:
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "HF_TOKEN":
                v = v.strip().strip('"').strip("'")
                return v or None
    v = os.environ.get("HF_TOKEN", "").strip()
    return v or None


# ---------------------------------------------------------------------------
# Pipeline construction — one-time per process, model stays resident.
# ---------------------------------------------------------------------------

def _build_pipeline(
    model_id: str,
    device: str,
    dtype_name: str,
):
    """Build the transformers ASR pipeline with the IFW recipe.

    Resolved attention implementation (in priority order):
        1. flash_attention_2 — IFW's headline path, ~3-5x faster
        2. sdpa             — PyTorch native, ~1.5-2x faster than eager
        3. eager            — fallback (won't reach here on torch >=2.1)
    """
    import torch
    from transformers import pipeline

    # Resolve the dtype here so callers can pass "fp16" / "fp32" / "bf16"
    # strings without importing torch themselves.
    dtype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"unknown dtype '{dtype_name}' (use fp16/fp32/bf16)")
    torch_dtype = dtype_map[dtype_name]

    # Probe FA2 by attempting the import. transformers exposes a helper but
    # it's been moved across versions; doing the import ourselves is more
    # robust to version drift.
    attn_impl = "sdpa"
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        pass

    # MPS doesn't support FA2 yet (transformers will raise) — force SDPA.
    if device.startswith("mps") and attn_impl == "flash_attention_2":
        attn_impl = "sdpa"

    print(f"  whisper: model={model_id}  device={device}  dtype={dtype_name}  attn={attn_impl}")

    return pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs={"attn_implementation": attn_impl},
    )


# ---------------------------------------------------------------------------
# Word-level transcription
# ---------------------------------------------------------------------------

def _transcribe_words(
    asr_pipeline,
    wav_path: Path,
    *,
    language: str | None,
    batch_size: int,
    chunk_length_s: int,
) -> dict:
    """Run ASR with word-level timestamps. Returns the raw HF dict.

    `chunks` in the result is a list of {"text": str, "timestamp": (start, end)}
    where each chunk corresponds to a single word when
    return_timestamps="word".
    """
    generate_kwargs: dict = {}
    if language:
        # Whisper expects a lower-case language code without country suffix.
        generate_kwargs["language"] = language.lower()
        generate_kwargs["task"] = "transcribe"

    # transformers' Whisper pipeline emits a UserWarning about the input
    # being longer than the receptive field even when chunk_length_s is
    # set. Silence it — we know.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Whisper.*long-form transcription.*",
            category=UserWarning,
        )
        result = asr_pipeline(
            str(wav_path),
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            return_timestamps="word",
            generate_kwargs=generate_kwargs or None,
        )
    return result


def _to_canonical_words(hf_result: dict) -> list[dict]:
    """Convert HF `chunks` to the canonical word/spacing list the rest of
    the pipeline expects (matches the old Scribe shape).

    HF gives us `[{"text": " hello", "timestamp": (0.0, 0.42)}, ...]`. We
    split each chunk into (optional spacing) + (word) and flatten.
    """
    out: list[dict] = []
    prev_end: float | None = None

    for chunk in hf_result.get("chunks", []) or []:
        text = chunk.get("text") or ""
        ts = chunk.get("timestamp") or (None, None)
        start, end = ts

        # Pipeline can emit None on the trailing chunk of a long file when
        # the model bails on a partial timestamp. Skip — phrase grouper
        # tolerates gaps.
        if start is None or end is None:
            continue

        # Insert a synthetic spacing entry covering the gap from prev word.
        # The pack_transcripts code uses these to detect long silences.
        if prev_end is not None and start > prev_end:
            out.append({
                "type": "spacing",
                "text": " ",
                "start": float(prev_end),
                "end": float(start),
            })

        # HF chunks include leading whitespace; strip for the word entry
        # but emit the actual visible text (punctuation included).
        word_text = text.lstrip() or text
        out.append({
            "type": "word",
            "text": word_text,
            "start": float(start),
            "end": float(end),
            # speaker_id filled in later by the diarization pass when on.
            "speaker_id": None,
        })
        prev_end = end

    return out


# ---------------------------------------------------------------------------
# Optional speaker diarization — pyannote.audio direct (no whisperx needed)
# ---------------------------------------------------------------------------

def _diarize_and_assign(
    words: list[dict],
    wav_path: Path,
    hf_token: str,
    *,
    num_speakers: int | None = None,
) -> list[dict]:
    """Run pyannote diarization on the WAV, then assign speaker_id to every
    'word' entry by majority overlap with the diarized segments.

    Words with zero overlap (rare — usually inter-segment whisper artifacts)
    keep speaker_id=None. The phrase grouper handles None gracefully.
    """
    try:
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

    # Feed pyannote a {"audio": ...} dict so it doesn't re-decode the file.
    diarize_kwargs = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers
    diarization = pipeline(str(wav_path), **diarize_kwargs)

    # Build a flat list of (start, end, label) sorted by start. Linear scan
    # below assumes order, which lets us assign in O(N+M) instead of O(N*M).
    segments = sorted(
        ((float(seg.start), float(seg.end), label)
         for seg, _, label in diarization.itertracks(yield_label=True)),
        key=lambda s: s[0],
    )
    if not segments:
        return words

    print(f"  diarize: {len(segments)} speaker turns, "
          f"{len(set(s[2] for s in segments))} distinct speakers")

    # For each word, find the diarized segment with the most overlap.
    # We walk segments in order alongside words (also in time order) so
    # we keep a moving window instead of binary-searching every word.
    si = 0
    for w in words:
        if w.get("type") != "word":
            continue
        ws = float(w.get("start", 0.0))
        we = float(w.get("end", ws))

        # Advance si past any segments that ended before this word starts.
        while si < len(segments) and segments[si][1] <= ws:
            si += 1

        # Find best-overlap segment in a small forward window.
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
            # Normalize "SPEAKER_00" → "speaker_0" so pack_transcripts'
            # display rule (strip "speaker_" → "0") still works unchanged.
            normalized = best_label.lower()
            if normalized.startswith("speaker_"):
                tail = normalized[len("speaker_"):].lstrip("0") or "0"
                normalized = f"speaker_{tail}"
            w["speaker_id"] = normalized

    return words


# ---------------------------------------------------------------------------
# Main lane entry point
# ---------------------------------------------------------------------------

def _process_one(
    asr,
    video_path: Path,
    edit_dir: Path,
    *,
    model_id: str,
    language: str | None,
    batch_size: int,
    chunk_length_s: int,
    diarize: bool,
    num_speakers: int | None,
    force: bool,
) -> Path:
    """Transcribe one video with an already-built pipeline.

    Split out so the batch entry point can amortize the ~3-5s pipeline
    construction across many videos in one Python process.
    """
    transcripts_dir = (edit_dir / TRANSCRIPTS_SUBDIR).resolve()
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video_path.stem}.json"

    if not force and out_path.exists():
        try:
            if out_path.stat().st_mtime >= video_path.stat().st_mtime:
                print(f"  whisper_lane cache hit: {out_path.name}")
                return out_path
        except OSError:
            pass

    wav_path = extract_audio_for(video_path, edit_dir, verbose=True)

    t0 = time.time()
    print(f"  whisper: transcribing {wav_path.name} "
          f"(batch={batch_size}, chunk={chunk_length_s}s)")
    hf_result = _transcribe_words(
        asr, wav_path,
        language=language,
        batch_size=batch_size,
        chunk_length_s=chunk_length_s,
    )
    words = _to_canonical_words(hf_result)

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

    duration = 0.0
    for w in reversed(words):
        end = w.get("end")
        if end is not None:
            duration = float(end)
            break

    payload = {
        "model": model_id,
        "language": (hf_result.get("language") or language or "auto"),
        "duration": duration,
        "text": (hf_result.get("text") or "").strip(),
        "words": words,
        "sample_rate": SAMPLE_RATE,
    }
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)

    dt = time.time() - t0
    n_words = sum(1 for w in words if w.get("type") == "word")
    kb = out_path.stat().st_size / 1024
    print(f"  whisper_lane done: {n_words} words, {duration:.1f}s audio, "
          f"{dt:.1f}s wall, {kb:.1f} KB → {out_path.name}")
    return out_path


def run_whisper_lane_batch(
    video_paths: list[Path],
    edit_dir: Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda:0",
    dtype_name: str = "fp16",
    batch_size: int = DEFAULT_BATCH_SIZE,
    chunk_length_s: int = DEFAULT_CHUNK_LENGTH_S,
    language: str | None = None,
    diarize: bool = False,
    num_speakers: int | None = None,
    force: bool = False,
) -> list[Path]:
    """Run the speech lane on N videos with the model loaded ONCE.

    Pre-flight check: if EVERY video is cache-fresh, skip the (slow)
    pipeline build entirely. Saves 3-5s + a model download check on
    repeat runs.
    """
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
            print(f"  whisper_lane: all {len(video_paths)} cache hits, skipping model load")
            return [transcripts_dir / f"{v.stem}.json" for v in video_paths]

    # ------------------------------------------------------------------
    # Parakeet fallback dispatch.
    #
    # Two paths flip us to Parakeet:
    #   1. The sentinel says Whisper was recently confirmed blocked on
    #      this machine. Skip the slow attempt entirely.
    #   2. _build_pipeline() raises a network-flavored exception while
    #      trying to fetch the Whisper weights. Catch, mark sentinel,
    #      re-dispatch to Parakeet.
    # ------------------------------------------------------------------
    if _whisper_blocked_recently():
        print(
            "  whisper_lane: WHISPER previously detected as BLOCKED on this "
            "machine (sentinel age < 7d). Going straight to NVIDIA Parakeet "
            "(local, ~10x faster, English/EU only). Delete "
            f"{BLOCKED_SENTINEL} to re-attempt Whisper."
        )
        from parakeet_lane import run_parakeet_lane_batch
        return run_parakeet_lane_batch(
            video_paths, edit_dir,
            device=device,
            dtype_name=("bf16" if dtype_name == "fp32" else dtype_name),
            batch_size=batch_size,
            chunk_length_s=chunk_length_s,
            language=language,
            diarize=diarize,
            num_speakers=num_speakers,
            force=force,
        )

    try:
        asr = _build_pipeline(model_id, device, dtype_name)
    except BaseException as exc:
        # We catch BaseException (not just Exception) because some HF
        # download failures bubble up as KeyboardInterrupt-adjacent
        # signal hijacking on Windows. Re-raise non-blocking ones so a
        # real CUDA OOM or syntax error still surfaces loudly.
        if not _is_blocked_exception(exc):
            raise

        reason = f"{type(exc).__name__}: {exc}"
        print(
            f"  whisper_lane: WHISPER BLOCKED ({reason[:200]}). "
            f"Falling back to NVIDIA Parakeet (local, ~10x faster, "
            f"English/EU only). Caching this decision for "
            f"{BLOCKED_TTL_DAYS}d at {BLOCKED_SENTINEL}.",
            file=sys.stderr,
        )
        _mark_whisper_blocked(reason[:500])

        from parakeet_lane import run_parakeet_lane_batch
        return run_parakeet_lane_batch(
            video_paths, edit_dir,
            device=device,
            dtype_name=("bf16" if dtype_name == "fp32" else dtype_name),
            batch_size=batch_size,
            chunk_length_s=chunk_length_s,
            language=language,
            diarize=diarize,
            num_speakers=num_speakers,
            force=force,
        )

    # Whisper loaded fine — clear any stale block sentinel from a
    # previous network state. Idempotent / safe if the sentinel doesn't
    # exist (most common case on a clean machine).
    _clear_whisper_blocked()
    out_paths: list[Path] = []
    # Outer bar: one tick per video. Each call to _process_one is opaque
    # work — Whisper's pipeline doesn't expose internal callbacks cheaply,
    # so we frame progress at the video granularity which is the unit
    # users actually care about (and which Claude Code can summarize).
    try:
        with lane_progress(
            "whisper",
            total=len(video_paths),
            unit="video",
            desc="speech transcription",
        ) as bar:
            for v in video_paths:
                bar.start_item(v.name)
                out_paths.append(_process_one(
                    asr, v, edit_dir,
                    model_id=model_id,
                    language=language,
                    batch_size=batch_size,
                    chunk_length_s=chunk_length_s,
                    diarize=diarize,
                    num_speakers=num_speakers,
                    force=force,
                ))
                bar.update(advance=1, item=v.name)
    finally:
        try:
            import torch
            del asr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return out_paths


def run_whisper_lane(
    video_path: Path,
    edit_dir: Path,
    **kwargs,
) -> Path:
    """Single-video convenience wrapper around run_whisper_lane_batch."""
    return run_whisper_lane_batch([video_path], edit_dir, **kwargs)[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Speech lane: insanely-fast-whisper-style transcription",
    )
    ap.add_argument("video", type=Path, help="Path to source video file")
    ap.add_argument(
        "--edit-dir", type=Path, default=None,
        help="Edit output dir (default: <video parent>/edit)",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL_ID,
                    help=f"HF model id (default: {DEFAULT_MODEL_ID})")
    ap.add_argument("--device", default="cuda:0",
                    help="Torch device: cuda:0, mps, cpu (default: cuda:0)")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--batch-size", type=int, default=None,
                    help=f"Inference batch size (default: {DEFAULT_BATCH_SIZE}, "
                         f"or {WHISPER_BATCH} with --wealthy)")
    ap.add_argument("--chunk-length-s", type=int, default=DEFAULT_CHUNK_LENGTH_S)
    ap.add_argument("--wealthy", action="store_true",
                    help="Speed knob for 24GB+ cards (4090/5090). Bigger batch, "
                         "same model, identical outputs. Also reads VIDEO_USE_WEALTHY=1.")
    ap.add_argument("--language", default=None,
                    help="ISO language code (e.g. 'en'). Omit to auto-detect.")
    ap.add_argument("--diarize", action="store_true",
                    help="Enable pyannote speaker diarization (needs HF_TOKEN).")
    ap.add_argument("--num-speakers", type=int, default=None,
                    help="Optional fixed number of speakers for diarize.")
    ap.add_argument("--force", action="store_true",
                    help="Bypass cache, always re-transcribe.")
    args = ap.parse_args()

    # If we were spawned by the orchestrator, tag every line of our
    # stdout/stderr with [whisper] so the parent can demux parallel lanes.
    install_lane_prefix()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    # Resolve batch size: explicit CLI value wins, else wealthy mode picks the
    # tier, else the conservative default.
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif is_wealthy(args.wealthy):
        batch_size = WHISPER_BATCH
    else:
        batch_size = DEFAULT_BATCH_SIZE

    run_whisper_lane(
        video_path=video,
        edit_dir=edit_dir,
        model_id=args.model,
        device=args.device,
        dtype_name=args.dtype,
        batch_size=batch_size,
        chunk_length_s=args.chunk_length_s,
        language=args.language,
        diarize=args.diarize,
        num_speakers=args.num_speakers,
        force=args.force,
    )


if __name__ == "__main__":
    main()
