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


def _is_cuda_oom(exc: BaseException) -> bool:
    """Return True if `exc` is a CUDA out-of-memory error.

    torch ships the OOM exception under three different names depending
    on version:
        * torch 2.6+         : `torch.OutOfMemoryError` (top-level)
        * torch 2.4 - 2.5    : `torch.cuda.OutOfMemoryError`
        * torch < 2.4        : plain `RuntimeError` with "out of memory"
                               in the message

    We accept all three so the retry logic works regardless of which
    torch wheel the user has installed. Pure exception inspection — no
    torch state poked, safe to call from any thread / context.
    """
    try:
        import torch
        # torch 2.6+: top-level OutOfMemoryError class
        oom_top = getattr(torch, "OutOfMemoryError", None)
        if oom_top is not None and isinstance(exc, oom_top):
            return True
        # torch 2.4 - 2.5: cuda-namespaced OutOfMemoryError
        oom_cuda = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if oom_cuda is not None and isinstance(exc, oom_cuda):
            return True
    except ImportError:
        pass
    # Fallback string match — covers torch < 2.4 and any wrapped/chained
    # OOMs that didn't inherit cleanly. We check a few synonymous phrases
    # because different CUDA layers (driver, runtime, allocator) word it
    # slightly differently.
    msg = str(exc).lower()
    return any(s in msg for s in (
        "out of memory",
        "cuda error: out of memory",
        "cudnn error",  # CUDNN_STATUS_NOT_INITIALIZED is OOM-adjacent
        "cublas",       # CUBLAS_STATUS_ALLOC_FAILED
    )) and ("memory" in msg or "alloc" in msg)


def _vram_snapshot() -> str:
    """Return a one-line summary of current GPU memory state, or '' if
    CUDA isn't available. Used to give OOM logs context — without this
    every OOM looks identical, so a user can't tell whether they hit
    fragmentation, model-too-big, or co-tenant pressure.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return ""
        device = torch.cuda.current_device()
        free_b, total_b = torch.cuda.mem_get_info(device)
        # Caching allocator's view: how much torch *thinks* it owns vs.
        # how much it's actively using right now. Big delta => fragmentation.
        reserved_b = torch.cuda.memory_reserved(device)
        allocated_b = torch.cuda.memory_allocated(device)
        gb = lambda b: b / (1024 ** 3)
        return (
            f"VRAM dev{device}: free={gb(free_b):.2f} GB / "
            f"total={gb(total_b):.2f} GB, "
            f"torch_reserved={gb(reserved_b):.2f} GB, "
            f"torch_allocated={gb(allocated_b):.2f} GB"
        )
    except Exception:
        return ""


def _release_cuda_cache() -> None:
    """Drop torch's caching allocator + sync the device. Used between
    videos and after an OOM so growth on clip N doesn't haunt clip N+1.

    Best-effort: never raises — if torch isn't built with CUDA, or the
    device is already in a bad state, we just return.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # ipc_collect releases shared-memory blocks held by other
            # CUDA processes (the visual + audio lanes) that have
            # exited. Without this, fragments stick around for the rest
            # of the parent's lifetime.
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
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

def _resolve_whisper_attn(device: str) -> str:
    """Pick the safest attention implementation for Whisper.

    Background — why eager is the default:
        transformers 5.x + sdpa + Whisper-large-v3 + Blackwell (sm_120)
        is a known-broken combination. Symptoms: cryptic "CUDA error:
        out of memory" raised on `inputs.to(device)` of a tiny mel-spec
        tensor, with the "CUDA kernel errors might be asynchronously
        reported" preamble — i.e. an earlier kernel failed silently and
        poisoned the CUDA context. The retry-loop sees identical fake
        OOMs at every batch size from 24 down to 1 and bails. See:
            https://github.com/huggingface/transformers/issues/38662
        and r/CUDA threads on PyTorch sm_120 + Whisper interactions.

        Florence-2 in the same process runs fine on the same card
        because its broken sdpa/flash flags are already monkey-patched
        off in visual_lane.py — meaning Florence runs eager too. We're
        just making Whisper consistent with that.

    Performance tradeoff: eager vs sdpa is roughly 1.5-2x slower on
    Whisper-large-v3 long-form. On a 5090 that's still ~50x realtime,
    which is fine for batch preprocessing. Once HF / PyTorch ships a
    fix for the sdpa-on-Blackwell path, set `VIDEO_USE_WHISPER_ATTN=sdpa`
    (or `flash_attention_2`) to opt back in without code changes.

    MPS path is unchanged: MPS doesn't support FA2 and our eager
    default is already valid there.
    """
    # Env var override comes first — power users / future-us can flip
    # this once the underlying bug is fixed without redeploying code.
    forced = os.environ.get("VIDEO_USE_WHISPER_ATTN", "").strip().lower()
    if forced in ("eager", "sdpa", "flash_attention_2"):
        return forced

    # Default policy: eager. Safe on every device + transformers combo
    # we've seen in the wild. The perf hit is a deliberate tradeoff for
    # not silently poisoning CUDA contexts on Blackwell.
    return "eager"


def _build_pipeline(
    model_id: str,
    device: str,
    dtype_name: str,
):
    """Build the transformers ASR pipeline with the IFW recipe.

    Attention implementation is resolved by `_resolve_whisper_attn` —
    see that function's docstring for why we default to `eager` rather
    than the IFW-recommended `flash_attention_2` / `sdpa` paths. The
    short version: those paths are broken on transformers 5.x +
    Blackwell, silently corrupting CUDA state and surfacing as fake
    OOMs that no batch-size retry can fix.
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

    attn_impl = _resolve_whisper_attn(device)
    print(
        f"  whisper: model={model_id}  device={device}  "
        f"dtype={dtype_name}  attn={attn_impl}"
        + ("  (override via VIDEO_USE_WHISPER_ATTN env)" if attn_impl == "eager" else "")
    )

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

    # ------------------------------------------------------------------
    # OOM-resilient transcription.
    #
    # Long-form audio with high batch sizes (--wealthy bumps batch from
    # 24 to 48) drives Whisper's KV cache to grow during generation. On
    # a 4+ minute clip with 32 decoder layers in fp16, the cache can
    # easily push past 30 GB of free VRAM — especially when the other
    # two lanes briefly co-exist on the same GPU during PARALLEL_3
    # startup. The crash happens deep inside transformers' cache_utils
    # at `torch.cat([self.values, value_states], dim=-2)`.
    #
    # Strategy: catch the OOM, free the caching allocator, halve the
    # batch size, and retry the SAME clip. Halving converges quickly
    # (48 -> 24 -> 12 -> 6 -> 3 -> 1) and outputs are bit-identical
    # across batch sizes — just slower at smaller batches. We bail when
    # we hit batch_size=1 and STILL OOM, because at that point the model
    # itself doesn't fit and the user has a real config problem.
    # ------------------------------------------------------------------
    cur_batch = max(1, batch_size)
    hf_result: dict | None = None
    while True:
        t0 = time.time()
        print(f"  whisper: transcribing {wav_path.name} "
              f"(batch={cur_batch}, chunk={chunk_length_s}s)")
        try:
            hf_result = _transcribe_words(
                asr, wav_path,
                language=language,
                batch_size=cur_batch,
                chunk_length_s=chunk_length_s,
            )
            break
        # We catch BaseException (then filter via _is_cuda_oom) instead of
        # RuntimeError because torch 2.6+ raises `torch.OutOfMemoryError`
        # from the top-level torch namespace; depending on which call
        # site allocates, it may or may not inherit from RuntimeError. The
        # filter ensures we still re-raise unrelated errors immediately.
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            # Snapshot memory BEFORE the cache flush so the log shows
            # what the failure actually looked like, not the post-recovery
            # state. Then flush, so the next attempt has clean allocator.
            vram_at_fail = _vram_snapshot()
            _release_cuda_cache()
            new_batch = cur_batch // 2
            if new_batch < 1:
                # Already at the floor — model is too big for this card
                # in this lane configuration. Surface the real error
                # plus the VRAM context so the user can act on it.
                print(
                    f"  whisper: CUDA OOM at batch=1 — model does not "
                    f"fit on this device with the current dtype/chunk.\n"
                    f"  whisper: at-fail snapshot: {vram_at_fail}\n"
                    f"  whisper: actionable fixes:\n"
                    f"    * re-run with --force-schedule sequential\n"
                    f"      (loads lanes one-at-a-time, no co-tenancy)\n"
                    f"    * re-run with --chunk-length-s 20 (smaller KV)\n"
                    f"    * re-run with --skip-visual to free Florence\n"
                    f"      (~5 GB) for Whisper",
                    file=sys.stderr,
                )
                raise
            print(
                f"  whisper: CUDA OOM at batch={cur_batch}, "
                f"retrying at batch={new_batch}\n"
                f"  whisper: at-fail snapshot: {vram_at_fail}",
                file=sys.stderr,
            )
            cur_batch = new_batch
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
          f"{dt:.1f}s wall, {kb:.1f} KB -> {out_path.name}")

    # Drop the KV cache + any transient allocator blocks before the next
    # clip starts. Without this, generation buffers from a long clip
    # remain reserved (not released to CUDA) and the very next clip can
    # OOM on its own KV growth even though steady-state usage is fine.
    _release_cuda_cache()
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

    # Snapshot VRAM right before model load so the diff between this and
    # the post-load snapshot makes whisper's footprint legible in the log.
    vram_pre = _vram_snapshot()
    if vram_pre:
        print(f"  whisper: pre-load {vram_pre}")
    try:
        asr = _build_pipeline(model_id, device, dtype_name)
    except BaseException as exc:
        # CUDA OOM during model load means the *static* model weights
        # don't fit alongside whatever the visual/audio lanes are
        # holding. The per-clip retry loop downstream can't help here —
        # we never even got past `from_pretrained`. Give the user the
        # same actionable fixes early, then re-raise.
        if _is_cuda_oom(exc):
            print(
                f"  whisper: CUDA OOM during model load — Whisper weights "
                f"could not be allocated alongside co-tenant lanes.\n"
                f"  whisper: at-fail snapshot: {_vram_snapshot()}\n"
                f"  whisper: actionable fixes:\n"
                f"    * re-run with --force-schedule sequential\n"
                f"      (loads lanes one-at-a-time, no co-tenancy)\n"
                f"    * raise VIDEO_USE_LANE_STAGGER_S env var (currently "
                f"~8s) so visual lane finishes loading first\n"
                f"    * re-run with --skip-visual to free Florence\n"
                f"      (~5 GB) for Whisper",
                file=sys.stderr,
            )
            raise
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
    # Post-load VRAM snapshot — the diff against the pre-load line
    # tells the user (and us, debugging) exactly how much weight memory
    # Whisper claimed and how much headroom is left for KV cache growth
    # during long-form generation. Roughly: if free < 6 GB after this
    # line, the first 4-min clip at default batch will likely OOM and
    # the retry loop will halve down to a working batch.
    vram_post = _vram_snapshot()
    if vram_post:
        print(f"  whisper: post-load {vram_post}")
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
