"""Speech lane (fallback): NVIDIA Parakeet TDT v3 transcription via NeMo.

This module is the torch / NeMo fallback for `parakeet_onnx_lane.py`,
the project's primary speech lane. It exists for the (rare) hosts
where ONNX Runtime can't load a working execution provider — ancient
glibc, no compatible GPU stack, etc. — but `nemo_toolkit` happens to
work via plain PyTorch. It produces the EXACT SAME output JSON shape
as the ONNX lane so `pack_timelines.py`, `render.py`, and the rest
of the pipeline are completely unaware that a different runtime
produced the words.

Why Parakeet TDT 0.6B v3:
  * ~2-3x the throughput of comparable encoder-decoder models on the
    same GPU thanks to the RNNT decoder.
  * Native word-level timestamps (no extra forced-alignment step).
  * English + 24 European languages (sufficient for most editing work).
  * Distributed via NeMo's HuggingFace AND NGC mirrors — even if HF is
    walled off, NVIDIA's own NGC catalog is reachable from inside the
    NVIDIA corporate network.
  * Robust on silence and music-only segments (no "hallucination on
    silence" failure mode common to encoder-decoder ASR).

The lazy NeMo install is handled in `_lazy_nemo.py` so this module's
top-level import surface stays cheap (no `import nemo` until we
actually need to build the model).

Output JSON shape (identical to `parakeet_onnx_lane.py` — see that
file's docstring for the canonical schema):

    {
      "model": "nvidia/parakeet-tdt-0.6b-v3",
      "language": "en",
      "duration": 43.0,
      "text": "...full plain transcript...",
      "words": [
        {"type": "word", "text": "Ninety", "start": 2.52, "end": 2.78,
         "speaker_id": null},
        {"type": "spacing", "text": " ", "start": 2.78, "end": 2.81},
        ...
      ],
      "sample_rate": 16000
    }

Diarization: this lane re-uses the shared `helpers/diarize.py` helper.
Speaker attribution operates on the canonical word list, not the
model — so the moment we have words on a shared timeline, the
diarizer doesn't care which backend produced them.

CLI:
    python helpers/parakeet_lane.py <video> [--language en]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Local sibling imports — work whether invoked via -m helpers.parakeet_lane
# or as a script directly (helpers/ is on sys.path either way).
from extract_audio import SAMPLE_RATE, extract_audio_for
from progress import install_lane_prefix, lane_progress
from wealthy import WHISPER_BATCH, is_wealthy

# Diarizer lives in `helpers/diarize.py` — single source of truth that
# operates on the canonical word list and has zero acoustic-model
# knowledge, so this NeMo fallback shares it byte-for-byte with the
# primary ONNX lane.
from diarize import diarize_and_assign as _diarize_and_assign
from diarize import load_hf_token as _load_hf_token


# ---------------------------------------------------------------------------
# Defaults — tuned for an RTX 3060+ baseline. Wealthy mode (24 GB+) uses
# WHISPER_BATCH (currently 32, see helpers/wealthy.py) which Parakeet
# absorbs comfortably; the RNNT decoder is small relative to
# Whisper-large-v3-turbo so batch headroom is plentiful. We re-use the
# same constant so the user's --wealthy flag means the same thing across
# the fallback chain — Parakeet COULD safely run higher batches, but
# matching Whisper's number keeps logs / VRAM expectations consistent.
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_BATCH_SIZE = 8         # conservative — Parakeet RNNT is small
TRANSCRIPTS_SUBDIR = "transcripts"

# Env var escape hatch for fully-air-gapped networks: if set, we load the
# Parakeet weights from a local .nemo file via `ASRModel.restore_from(...)`
# instead of `ASRModel.from_pretrained(...)` (which would hit HuggingFace).
#
# Workflow for users behind a proxy that blocks HF entirely:
#   1. On a machine with HF access (or via NGC: catalog.ngc.nvidia.com →
#      nvidia/nemo/parakeet_tdt_0_6b_v3), download the .nemo file.
#   2. Copy the .nemo to the air-gapped machine.
#   3. set PARAKEET_MODEL_PATH=C:\path\to\parakeet-tdt-0.6b-v3.nemo
#   4. Run the skill as normal — fallback path picks the local file up
#      with zero network calls.
PARAKEET_MODEL_PATH_ENV = "PARAKEET_MODEL_PATH"


# ---------------------------------------------------------------------------
# Model construction — single load amortized across N videos in one batch
# ---------------------------------------------------------------------------

def _build_parakeet_model(
    model_id: str,
    device: str,
    dtype_name: str,
):
    """Load the Parakeet ASR model via NeMo.

    Lazy-installs `nemo_toolkit[asr]` on the very first call if it's not
    importable yet (see `_lazy_nemo.ensure_nemo_installed`). After that
    every subsequent invocation hits the import cache instantly.

    Args:
        model_id:   HF / NGC model identifier (e.g. nvidia/parakeet-tdt-0.6b-v3).
        device:     Torch device string — "cuda:0", "cuda", "cpu".
        dtype_name: "fp16" / "fp32" / "bf16". Parakeet is trained in
                    bf16; fp16 inference works on Ampere+ but bf16 is the
                    safest default. We honor whatever the caller passed.

    Returns:
        A loaded NeMo ASRModel ready for `.transcribe(...)`.
    """
    # One-time NeMo install if needed. Prints a clear status line so the
    # user sees what's happening during the ~600 MB download.
    from _lazy_nemo import ensure_nemo_installed
    ensure_nemo_installed()

    # Defer torch + nemo imports until AFTER the lazy install — otherwise
    # the first call would crash before we ever got a chance to install.
    import torch
    import nemo.collections.asr as nemo_asr  # type: ignore

    # Map the dtype name to a torch dtype. We accept the same vocabulary
    # as the primary ONNX lane so callers don't have to special-case
    # the fallback.
    dtype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"unknown dtype '{dtype_name}' (use fp16/fp32/bf16)")
    torch_dtype = dtype_map[dtype_name]

    # Local .nemo escape hatch — for networks that block HuggingFace
    # entirely (worst-case: ONNX lane unavailable AND Parakeet would
    # also have to download from HF, leaving us stuck). User
    # pre-downloads the .nemo file via NGC or a colleague, sets the
    # env var, and we skip the network path completely.
    local_path = os.environ.get(PARAKEET_MODEL_PATH_ENV, "").strip()

    print(f"  parakeet: model={model_id}  device={device}  dtype={dtype_name}"
          + (f"  local={local_path}" if local_path else ""))

    # NeMo emits a wall of INFO logs on model construction. Quiet them so
    # the lane output stays readable in the orchestrator's tagged stream.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            if local_path:
                # `restore_from` reads a .nemo archive from disk — zero
                # network access. Honors the env-var path verbatim so the
                # user can drop it anywhere convenient.
                if not Path(local_path).exists():
                    raise FileNotFoundError(
                        f"{PARAKEET_MODEL_PATH_ENV}={local_path} does not "
                        f"exist. Either fix the path or unset the env var."
                    )
                model = nemo_asr.models.ASRModel.restore_from(
                    restore_path=local_path,
                )
            else:
                model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=model_id,
                )
        except Exception as e:
            # Surface a clear, actionable message for the worst-case
            # fallback-failed-too scenario (HF blocked AND no local file).
            # We re-raise so the caller (preprocess.py speech dispatch)
            # still propagates the real failure, but the user gets a
            # paste-ready next step instead of a NeMo stack trace.
            raise RuntimeError(
                f"Parakeet model load failed: {type(e).__name__}: {e}\n\n"
                f"  If your network blocks HuggingFace entirely, the "
                f"fallback also can't reach the model. Workaround:\n"
                f"    1. On a machine with HF or NGC access, download "
                f"the .nemo file for {model_id}\n"
                f"       (NGC: https://catalog.ngc.nvidia.com → "
                f"nvidia/nemo/{model_id.split('/')[-1].replace('-', '_')})\n"
                f"    2. Copy it to this machine.\n"
                f"    3. Set the env var: "
                f"{PARAKEET_MODEL_PATH_ENV}=/path/to/parakeet-tdt-0.6b-v3.nemo\n"
                f"    4. Re-run the skill — it will load locally with "
                f"zero network calls."
            ) from e

    # Move to the requested device + dtype. NeMo models inherit nn.Module
    # so the standard .to() chain works. We do it in two calls because
    # some NeMo builds choke on .to(device, dtype=...) signature.
    if device and device != "cpu":
        model = model.to(device)
    if dtype_name != "fp32":
        # bf16/fp16 inference. Parakeet was trained bf16 so this is lossless
        # for bf16 and within rounding noise for fp16 on Ampere+.
        try:
            model = model.to(torch_dtype)
        except Exception as e:
            # Some NeMo decoder modules don't survive a global .to(dtype).
            # Fall back to fp32 rather than crashing — it's slower but works.
            print(f"  parakeet: dtype cast to {dtype_name} failed ({e!r}); "
                  f"continuing in fp32.", file=sys.stderr)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Transcription — NeMo returns Hypothesis objects when timestamps=True
# ---------------------------------------------------------------------------

def _transcribe_parakeet(
    model,
    wav_path: Path,
    *,
    batch_size: int,
) -> object:
    """Run Parakeet on one WAV with word-level timestamps enabled.

    Returns the raw Hypothesis-like object NeMo emits. Caller is
    responsible for extracting the canonical word list via
    `_parakeet_to_canonical_words`.
    """
    # Silence NeMo's per-batch progress bars — we have our own.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        outputs = model.transcribe(
            [str(wav_path)],
            batch_size=batch_size,
            timestamps=True,
            # Greedy decoding is faster AND benchmarks show it strictly
            # beats beam search for Parakeet TDT — the RNNT decoder
            # already scores its own paths internally.
            verbose=False,
        )

    if not outputs:
        raise RuntimeError(f"parakeet returned no outputs for {wav_path}")

    # NeMo's transcribe() returns a list per input; with timestamps=True
    # each entry is a Hypothesis object exposing .text and .timestamp.
    # On older NeMo versions (<2.0) it might be a tuple; handle both.
    first = outputs[0]
    if isinstance(first, (list, tuple)) and first:
        first = first[0]
    return first


# ---------------------------------------------------------------------------
# Hypothesis -> canonical word list
# ---------------------------------------------------------------------------

def _parakeet_to_canonical_words(hyp) -> list[dict]:
    """Convert a Parakeet hypothesis into the canonical word/spacing list.

    The canonical shape is what `pack_timelines.py` consumes. Every
    `word` entry includes a `speaker_id` slot (default None) so the
    diarizer pass can fill it in later without restructuring the list.

    Parakeet's word timestamp shape (NeMo >= 2.0):

        hyp.timestamp = {
            "word":    [{"word": "Hello", "start": 0.12, "end": 0.34}, ...],
            "segment": [...],
            "char":    [...],
        }

    Robust to:
      * `hyp.timestamp` missing entirely (returns empty list).
      * Word entries using "start_time"/"end_time" instead of "start"/"end"
        (older NeMo dialects).
      * Word entries using "text" instead of "word" (defensive).
    """
    out: list[dict] = []

    # Parakeet hypotheses can be plain dicts in some adapters or
    # objects in others — handle both shapes.
    ts = getattr(hyp, "timestamp", None)
    if ts is None and isinstance(hyp, dict):
        ts = hyp.get("timestamp") or hyp.get("timestamps")
    if not ts:
        return out

    # `ts` is a dict keyed by granularity. We want the word-level slice.
    word_entries = ts.get("word") if isinstance(ts, dict) else None
    if not word_entries:
        return out

    prev_end: float | None = None

    for entry in word_entries:
        # Field-name normalization — different NeMo versions disagree.
        text = (
            entry.get("word")
            or entry.get("text")
            or entry.get("token")
            or ""
        )
        text = str(text).strip()
        if not text:
            continue

        start = entry.get("start", entry.get("start_time"))
        end = entry.get("end", entry.get("end_time"))
        if start is None or end is None:
            continue
        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            continue

        # Synthetic spacing so the phrase grouper can detect long pauses
        # the same way it does for Whisper output. Hard Rule 7 padding
        # math depends on these gap entries existing.
        if prev_end is not None and start > prev_end:
            out.append({
                "type": "spacing",
                "text": " ",
                "start": float(prev_end),
                "end": float(start),
            })

        out.append({
            "type": "word",
            "text": text,
            "start": float(start),
            "end": float(end),
            # Diarizer pass fills this in if --diarize was passed AND
            # pyannote + HF_TOKEN are available. Phrase grouper tolerates
            # None speaker IDs (renders as a single-speaker block).
            "speaker_id": None,
        })
        prev_end = float(end)

    return out


def _hypothesis_text(hyp) -> str:
    """Extract the plain-text transcript from a hypothesis. Defensive
    against shape drift between NeMo versions."""
    # NeMo >= 2.0 puts the joined text on .text directly.
    text = getattr(hyp, "text", None)
    if text is None and isinstance(hyp, dict):
        text = hyp.get("text") or hyp.get("pred_text")
    return (text or "").strip()


# ---------------------------------------------------------------------------
# Per-video processing — wraps cache check, ASR, optional diarize, JSON write
# ---------------------------------------------------------------------------

def _process_one(
    model,
    video_path: Path,
    edit_dir: Path,
    *,
    model_id: str,
    language: str | None,
    batch_size: int,
    diarize: bool,
    num_speakers: int | None,
    force: bool,
) -> Path:
    """Transcribe one video with an already-loaded Parakeet model.

    Mirrors the cache contract of `parakeet_onnx_lane._process_one`
    exactly so `pack_timelines.py` sees a single coherent transcripts/
    directory regardless of which speech backend produced the JSON.
    """
    transcripts_dir = (edit_dir / TRANSCRIPTS_SUBDIR).resolve()
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video_path.stem}.json"

    # mtime cache — same rule as the ONNX lane: WAV / source unchanged
    # since the JSON was written? Reuse it.
    if not force and out_path.exists():
        try:
            if out_path.stat().st_mtime >= video_path.stat().st_mtime:
                print(f"  parakeet_lane cache hit: {out_path.name}")
                return out_path
        except OSError:
            pass

    wav_path = extract_audio_for(video_path, edit_dir, verbose=True)

    t0 = time.time()
    print(f"  parakeet: transcribing {wav_path.name} (batch={batch_size})")
    hyp = _transcribe_parakeet(model, wav_path, batch_size=batch_size)
    words = _parakeet_to_canonical_words(hyp)
    text = _hypothesis_text(hyp)

    # Optional diarization — shared `helpers/diarize.py` helper. Same
    # HF token sourcing, same pyannote pipeline, same overlap
    # algorithm as the primary ONNX lane.
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

    # Derive duration from the last timestamp we have. Falls back to 0.0
    # if the hypothesis was empty (silent / 1s clip / decoder bailed).
    duration = 0.0
    for w in reversed(words):
        end = w.get("end")
        if end is not None:
            duration = float(end)
            break

    payload = {
        "model": model_id,
        # Parakeet doesn't return language detection — it's a fixed multi-
        # lingual model. Honor the caller's hint, default to "en" since
        # the v3 model's primary training data is English.
        "language": (language or "en"),
        "duration": duration,
        "text": text,
        "words": words,
        "sample_rate": SAMPLE_RATE,
    }

    # Atomic write so a Ctrl-C mid-write doesn't leave a corrupt JSON
    # that the cache check would later mistake for a finished result.
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)

    dt = time.time() - t0
    n_words = sum(1 for w in words if w.get("type") == "word")
    kb = out_path.stat().st_size / 1024
    print(f"  parakeet_lane done: {n_words} words, {duration:.1f}s audio, "
          f"{dt:.1f}s wall, {kb:.1f} KB → {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Batch entry point — drop-in fallback for run_parakeet_onnx_lane_batch
# ---------------------------------------------------------------------------

def run_parakeet_lane_batch(
    video_paths: list[Path],
    edit_dir: Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda:0",
    dtype_name: str = "bf16",
    batch_size: int = DEFAULT_BATCH_SIZE,
    # Accept (and ignore) chunk_length_s so any orchestrator that
    # passes encoder-decoder kwargs verbatim still works. Parakeet
    # doesn't need a chunk size — its RNNT decoder streams natively.
    chunk_length_s: int = 30,
    language: str | None = None,
    diarize: bool = False,
    num_speakers: int | None = None,
    force: bool = False,
) -> list[Path]:
    """Run Parakeet on N videos with the model loaded ONCE.

    Same signature as `run_parakeet_onnx_lane_batch` so the
    orchestrator can swap to this NeMo fallback without changing
    callers.
    """
    # Pre-flight: skip the (slow) model load entirely when every video
    # is cache-fresh. Same optimization the ONNX lane has.
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
            print(f"  parakeet_lane: all {len(video_paths)} cache hits, "
                  f"skipping model load")
            return [transcripts_dir / f"{v.stem}.json" for v in video_paths]

    model = _build_parakeet_model(model_id, device, dtype_name)
    out_paths: list[Path] = []

    # Outer bar: one tick per video. Parakeet's transcribe() is a single
    # opaque call per file from our perspective — finer-grained progress
    # would require hooking into NeMo's internal data loader.
    try:
        with lane_progress(
            "parakeet",
            total=len(video_paths),
            unit="video",
            desc="speech transcription (parakeet fallback)",
        ) as bar:
            for v in video_paths:
                bar.start_item(v.name)
                out_paths.append(_process_one(
                    model, v, edit_dir,
                    model_id=model_id,
                    language=language,
                    batch_size=batch_size,
                    diarize=diarize,
                    num_speakers=num_speakers,
                    force=force,
                ))
                bar.update(advance=1, item=v.name)
    finally:
        # Free the GPU allocator so subsequent lanes (visual / audio)
        # see the full VRAM. Parakeet TDT 0.6B is small (~1.5 GB in
        # fp16) but in tight 8 GB scheduling that headroom matters.
        try:
            import torch
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return out_paths


def run_parakeet_lane(
    video_path: Path,
    edit_dir: Path,
    **kwargs,
) -> Path:
    """Single-video convenience wrapper around `run_parakeet_lane_batch`."""
    return run_parakeet_lane_batch([video_path], edit_dir, **kwargs)[0]


# ---------------------------------------------------------------------------
# CLI — for manual invocation / debugging the fallback path in isolation
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Speech lane (fallback): NVIDIA Parakeet TDT v3",
    )
    ap.add_argument("video", type=Path, help="Path to source video file")
    ap.add_argument(
        "--edit-dir", type=Path, default=None,
        help="Edit output dir (default: <video parent>/edit)",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL_ID,
                    help=f"NeMo model id (default: {DEFAULT_MODEL_ID})")
    ap.add_argument("--device", default="cuda:0",
                    help="Torch device: cuda:0, cuda, cpu (default: cuda:0)")
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--batch-size", type=int, default=None,
                    help=f"Inference batch size "
                         f"(default: {DEFAULT_BATCH_SIZE}, "
                         f"or {WHISPER_BATCH} with --wealthy)")
    ap.add_argument("--wealthy", action="store_true",
                    help="Speed knob for 24GB+ cards. Same model, identical "
                         "outputs, just bigger batches.")
    ap.add_argument("--language", default=None,
                    help="ISO language code hint (e.g. 'en'). Parakeet v3 "
                         "supports English + 24 European languages.")
    ap.add_argument("--diarize", action="store_true",
                    help="Enable pyannote speaker diarization (needs HF_TOKEN).")
    ap.add_argument("--num-speakers", type=int, default=None,
                    help="Optional fixed number of speakers for diarize.")
    ap.add_argument("--force", action="store_true",
                    help="Bypass cache, always re-transcribe.")
    args = ap.parse_args()

    # Tag every line with [parakeet] when spawned by the orchestrator.
    install_lane_prefix()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    # Resolve batch size — explicit CLI value wins; otherwise wealthy
    # mode bumps to WHISPER_BATCH (Parakeet absorbs it easily); else the
    # conservative default that fits on a 3060.
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif is_wealthy(args.wealthy):
        batch_size = WHISPER_BATCH
    else:
        batch_size = DEFAULT_BATCH_SIZE

    run_parakeet_lane(
        video_path=video,
        edit_dir=edit_dir,
        model_id=args.model,
        device=args.device,
        dtype_name=args.dtype,
        batch_size=batch_size,
        language=args.language,
        diarize=args.diarize,
        num_speakers=args.num_speakers,
        force=args.force,
    )


if __name__ == "__main__":
    main()
