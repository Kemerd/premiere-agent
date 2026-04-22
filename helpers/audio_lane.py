"""Audio lane: LAION CLAP via ONNX Runtime -> vocab-scored sound events.

Why CLAP, why ONNX, why an agent-curated vocabulary
---------------------------------------------------
The two predecessors of this module both lived in the same repo at one
time or another and both had the same fundamental problem: they shipped
with a fixed answer space that was either too small or too generic for
the kind of footage the rest of this skill targets.

  * `panns_inference.AudioTagger` (CNN14 trained on AudioSet) gave us a
    closed 527-class ontology where workshop / friction textures sit in
    the long tail of the published 43% mAP. Real failures: a clean
    sanding pass surfaced as "Goat bleating" because the spectro-temporal
    envelope of coarse abrasion lined up uncomfortably well with one
    weakly-labelled livestock sample. The ontology also literally cannot
    represent things that matter for editing (no "cordless drill driving
    a screw", no "router on plywood"). Discontinued.

  * `nvidia/audio-flamingo-3-hf` (a 7B audio-LLM that GENERATES natural
    language captions) solved the vocabulary problem, but at the cost of
    a ~14-18 GB resident weight footprint per chunk and ~3-8 s of
    decoder time PER 30s window. Co-tenanting it with Whisper + Florence
    on a 24 GB card became impossible; even sequentially it dominated
    end-to-end wall time for a feature whose output is supposed to be
    advisory context, not the editor's primary signal. Discontinued.

CLAP (Contrastive Language-Audio Pretraining, LAION) is the natural
middle ground. It's a dual-encoder zero-shot model: an audio tower
embeds an audio chunk into a 512-d vector, a text tower embeds an
arbitrary phrase into the same space, and cosine similarity scores
the match. We pre-encode a vocabulary of label phrases ONCE per video
inventory and then every per-window inference is a single ~50ms audio
encoder forward pass + a dot product against the cached text matrix.

The vocabulary itself comes from one of two sources:

  1. The baked-in default in `audio_vocab_default.py` (~250 cross-domain
     labels covering speech-adjacent / workshop / nature / urban /
     household / music / sports / SFX). Good first cut for any video.
  2. A per-project vocabulary list authored by the Claude Code agent
     after Phase A preprocessing produces `speech_timeline.md` and
     `visual_timeline.md`. The agent reads those, infers what kinds
     of sounds plausibly appear in THIS specific video, and writes a
     hand-curated `<edit>/audio_vocab.txt` (one label per line, blank
     lines and `#` comments allowed). Re-running the audio lane with
     `--vocab <edit>/audio_vocab.txt --force` produces sharper labels
     because the score competition only happens between phrases the
     agent has reason to believe are in the audio.

Model tiers
-----------
  * Default ("base"):   `Xenova/clap-htsat-unfused`     (~150 MB ONNX,
                        LAION CLAP HTSAT-unfused, mAP 0.463 zero-shot
                        on AudioSet). Fast, fits on every consumer GPU.
  * Wealthy ("large"):  `Xenova/larger_clap_general`    (~600 MB ONNX,
                        sharper text encoder, slightly better fine-grain
                        discrimination). Selected automatically when
                        --wealthy / VIDEO_USE_WEALTHY=1.

Both are pre-converted ONNX exports of the original LAION torch weights;
they ship audio + text encoders as separate ONNX graphs so we can run
the text tower exactly once per vocab change and amortize that cost
across every video in the batch.

Sliding-window cadence
----------------------
HTSAT was pre-trained on 10s mel windows. We use:

    WINDOW_S = 10.0   # native HTSAT receptive field
    HOP_S    = 5.0    # 50% overlap; gives 5s temporal grain

Anything shorter (PANNs used 1s windows) requires zero-padding the
spectrogram, which the CLAP encoder reads as silence between transient
events and silently degrades discrimination on short bursts. 10s/5s is
the sweet spot for cross-domain editorial context — fine enough to
localize a tool burst, coarse enough that the encoder sees its native
input distribution.

ONNX execution provider ladder
------------------------------
Reuses `_onnx_providers.resolve_providers(prefer_tensorrt=False)`. We
explicitly skip TensorRT here because CLAP's audio + text graphs are
small enough that ORT-CUDA hits the GPU's compute roof at the first
batch — TRT's 2-5 minute engine compile cost dwarfs any per-window
speedup. Power users on long-running services can flip
VIDEO_USE_PARAKEET_TRT=1 (the same env var the speech lane reads) and
the ladder will include TRT, but it's NOT recommended for one-shot
preprocess runs.

JSON shape (per video)
----------------------
    {
      "model":      "Xenova/clap-htsat-unfused",
      "vocab_sha":  "0a1b2c3d4e5f6789",
      "vocab_size": 247,
      "window_s":   10.0,
      "hop_s":      5.0,
      "threshold":  0.10,
      "top_k":      5,
      "duration":   263.4,
      "events": [
        {"start": 12.0, "end": 22.0, "label": "drill",    "score": 0.421},
        {"start": 18.5, "end": 28.5, "label": "laughter", "score": 0.387}
      ]
    }

Cache invalidation
------------------
A cached JSON is reused only when ALL of:
  (a) JSON mtime is fresh vs the source video
  (b) JSON `model` field equals the current model id
  (c) JSON `vocab_sha` field equals the current sha256(model_id + vocab)

That third clause is what makes the agent vocab workflow safe: re-run
with a different `--vocab` file and every cache invalidates cleanly,
no `--force` needed. Old AF3 caches (`"nvidia/audio-flamingo-3-hf"`)
and orphan PANNs caches (`"PANNs CNN14 / AudioSet"`) both fail check
(b) and re-tag transparently on the first run after this lane lands.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# CRITICAL: this MUST come before anything that pulls `transformers` in
# (directly OR transitively). Sets USE_TF=0 / USE_FLAX=0 so transformers'
# eager backend probe doesn't crash on a stale TF install. See
# `_hf_env.py` for the full rationale.
from _hf_env import HF_ENV_GUARDS_INSTALLED  # noqa: F401  - import for side effect

from extract_audio import SAMPLE_RATE as INPUT_SAMPLE_RATE, extract_audio_for
from progress import install_lane_prefix, lane_progress
from wealthy import (
    is_wealthy,
    CLAP_MODEL_TIER_DEFAULT,
    CLAP_MODEL_TIER_WEALTHY,
    CLAP_WINDOWS_PER_BATCH,
    CLAP_WINDOWS_PER_BATCH_WEALTHY,
)
from audio_vocab_default import load_vocab
from _onnx_providers import resolve_providers


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# CLAP HTSAT was pre-trained on 48 kHz audio. Our shared `extract_audio`
# cache produces 16 kHz mono PCM (matches Whisper/Parakeet's native rate);
# we resample on the fly inside this lane rather than maintain a second
# WAV cache on disk. Resample cost via `soxr` is sub-second per minute
# of audio and is dominated by the encoder pass anyway.
CLAP_SAMPLE_RATE = 48_000

# Native HTSAT receptive field. Anything shorter zero-pads the mel
# spectrogram and degrades discrimination on transient bursts; anything
# longer triggers the model's internal chunking and loses fine-grain
# temporal info across the chunk boundary.
DEFAULT_WINDOW_S = 10.0
DEFAULT_HOP_S = 5.0

# ReCLAP-style template prompt. Wrapping every label in "the sound of
# {}" before text-encoding gives a measured ~5-10% accuracy lift over
# the bare label text — see ReCLAP (2024). This is a CLAP property,
# not just a CLAP-HTSAT one; the larger_clap_general checkpoint
# benefits identically.
LABEL_TEMPLATE = "the sound of {}"

# Cosine-similarity floor for keeping a (window, label) hit. CLAP's
# raw cosine values are NOT bounded to [0, 1] like a softmax classifier
# — typical positive matches sit in the 0.15-0.45 band, with strong
# matches above 0.30. 0.10 is intentionally permissive so the agent
# vocab workflow can use a tight whitelist without losing weak-but-
# real hits; users with broad vocabs may want to bump this up to
# 0.20 to reduce noise.
DEFAULT_THRESHOLD = 0.10

# Per-window top-K. We keep at most this many label hits per window;
# CLAP's score distribution rolls off quickly past the top 3-5 even
# on busy chunks, so 5 is plenty without flooding the timeline.
DEFAULT_TOP_K = 5

# Coalescer gap: consecutive same-label hits whose end->start gap is
# at most this many seconds get merged into one continuous event. With
# HOP_S=5.0 and overlapping 10s windows, two adjacent windows of the
# same label naturally have a 5s overlap (gap of -5s, well below
# threshold), so this primarily merges sustained sounds across many
# windows into one big block.
COALESCE_MAX_GAP_S = 2.0

# Output directory name kept as `audio_tags` for backwards-compat with
# the existing edit-folder layout that pack_timelines and the SKILL.md
# both reference. Data inside is event lists; folder name is part of
# the on-disk contract.
AUDIO_TAGS_SUBDIR = "audio_tags"

# Per-vocab text-embedding cache. Lives next to the per-video event
# JSONs so wiping the edit folder also wipes the cached embeddings —
# no risk of a stale embedding cache silently surviving a cache flush.
VOCAB_EMBEDS_CACHE_NAME = "audio_vocab_embeds.npz"

# Model tier -> HF Hub id. Both repos are Xenova's pre-converted ONNX
# exports of the original LAION CLAP torch weights, hosted on the Hub.
# `huggingface_hub.snapshot_download` is the documented fetch path.
MODEL_TIERS: dict[str, str] = {
    "base":  "Xenova/clap-htsat-unfused",
    "large": "Xenova/larger_clap_general",
}


# ---------------------------------------------------------------------------
# Atomic-rename retry helper
#
# The per-video JSON cache files are tiny (a few KB), but on Windows
# Defender / Search Indexer / third-party AV briefly hold a scan handle
# on freshly-closed files. A sharing violation here would corrupt the
# cache and re-trigger a full re-tag on the next run for no reason.
# Same primitive the legacy lanes used; kept verbatim.
# ---------------------------------------------------------------------------

_WINERROR_SHARING_VIOLATION = 32  # ERROR_SHARING_VIOLATION
_WINERROR_LOCK_VIOLATION = 33     # ERROR_LOCK_VIOLATION


def _is_sharing_violation(exc: OSError) -> bool:
    """True iff `exc` looks like a transient Windows file-lock conflict."""
    if getattr(exc, "winerror", None) in (
        _WINERROR_SHARING_VIOLATION,
        _WINERROR_LOCK_VIOLATION,
    ):
        return True
    if exc.errno == errno.EACCES:
        return True
    return False


def _atomic_replace_with_retry(
    tmp_path: Path,
    final_path: Path,
    *,
    max_attempts: int = 5,
    initial_delay_s: float = 0.05,
) -> None:
    """`tmp_path.replace(final_path)` with backoff for AV scan handles.

    Backoff schedule: 50ms, 100ms, 200ms, 400ms, 800ms — total ceiling
    ~1.5s. Adequate for the small JSON files we write here.
    """
    delay = initial_delay_s
    last_exc: OSError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            tmp_path.replace(final_path)
            return
        except OSError as exc:
            if not _is_sharing_violation(exc):
                raise
            last_exc = exc
            if attempt == max_attempts:
                break
            time.sleep(delay)
            delay *= 2.0
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Vocab hashing — drives both cache invalidation AND text-embed cache key
# ---------------------------------------------------------------------------

def _hash_vocab(model_id: str, vocab: list[str]) -> str:
    """Stable 16-hex-char hash of (model_id + sorted vocab labels).

    Sorting before hashing means re-ordering the same labels in the
    vocab file does NOT invalidate caches. Including the model id
    means switching tiers (base <-> large) does invalidate, because
    text embeddings are NOT cross-compatible across CLAP variants.
    """
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    for label in sorted(vocab):
        h.update(b"\x00")
        h.update(label.encode("utf-8"))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Model + processor download (HF snapshot)
# ---------------------------------------------------------------------------

def _download_clap_onnx(model_id: str) -> Path:
    """Snapshot Xenova's CLAP ONNX repo, return the local cache path.

    We grab only what we actually use: the two quantized ONNX graphs
    (~150 MB total for base, ~600 MB for large) and the processor /
    tokenizer config files. Skipping the unquantized ONNX exports cuts
    the cold-start download by roughly half.
    """
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=model_id,
        allow_patterns=[
            # Processor / tokenizer configs (small, all required by
            # transformers.AutoProcessor.from_pretrained).
            "*.json",
            "*.txt",
            # The two ONNX graphs we feed onnxruntime. Quantized variants
            # are within rounding noise of fp32 on AudioSet eval AND
            # ~3-4x smaller on disk, which matters for a first-run
            # download bar that doesn't have ~1 GB to spend.
            "onnx/audio_model_quantized.onnx",
            "onnx/text_model_quantized.onnx",
        ],
    )
    return Path(local)


def _build_audio_session(model_dir: Path, providers):
    """Construct the audio-encoder ORT session.

    The session is reused across every audio window and every video in
    the batch. ORT releases the GIL during native `Run()`, so even with
    a single session we get true GPU concurrency between Python-side
    pre/post processing and the kernel forward pass.
    """
    import onnxruntime as ort
    onnx_path = model_dir / "onnx" / "audio_model_quantized.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"audio ONNX missing at {onnx_path}; HF snapshot may be "
            f"incomplete — delete the HF cache for this repo and re-run"
        )
    return ort.InferenceSession(str(onnx_path), providers=providers)


def _build_text_session(model_dir: Path, providers):
    """Construct the text-encoder ORT session.

    Used exactly once per vocab change (when the text-embedding cache
    misses). Kept around as a session object rather than spun up per
    call so a `--force` re-run on the same vocab pays no extra session
    construction cost.
    """
    import onnxruntime as ort
    onnx_path = model_dir / "onnx" / "text_model_quantized.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"text ONNX missing at {onnx_path}; HF snapshot may be "
            f"incomplete — delete the HF cache for this repo and re-run"
        )
    return ort.InferenceSession(str(onnx_path), providers=providers)


def _load_processor(model_dir: Path):
    """Load the CLAP feature_extractor + tokenizer pair.

    transformers.AutoProcessor handles both: the audio side outputs the
    log-mel `input_features` tensor (with the correct n_mels / hop /
    window for whichever CLAP variant we loaded), and the text side
    handles tokenization + special tokens. Using the official processor
    instead of hand-rolling mel computation guards us against silent
    drift between this lane and the model's pretrain distribution.
    """
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(str(model_dir))


# ---------------------------------------------------------------------------
# Audio loading — read 16 kHz cache, upsample to 48 kHz on the fly
# ---------------------------------------------------------------------------

def _load_audio_48k(wav_path: Path):
    """Load `extract_audio`'s 16 kHz mono WAV, return float32 @ 48 kHz.

    Returns np.ndarray shape (n_samples_48k,), float32 in [-1, 1].

    Resampling path tries `soxr` first (fast, high-quality VHQ kernel
    via libsoxr) and falls back to `librosa.resample` if soxr isn't
    importable. Both are transitive deps of the preprocess extra so
    one of them is always available.
    """
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        # Defensive: extract_audio guarantees mono but a stray drop-in
        # WAV could be 2-channel. Mean-down rather than left-channel
        # because workshop footage's stereo mics pick up different
        # tools per channel and we want both contributions.
        audio = audio.mean(axis=1).astype(np.float32)

    if sr == CLAP_SAMPLE_RATE:
        return audio  # impossibly lucky — extract_audio uses 16k

    # Try soxr first (~5x faster than librosa for typical 16k->48k
    # upsamples on commodity CPUs). Fall through to librosa on import
    # failure so we never hard-crash when soxr's wheel is missing on
    # an exotic platform.
    try:
        import soxr
        return soxr.resample(audio, sr, CLAP_SAMPLE_RATE).astype(np.float32)
    except ImportError:
        import librosa
        return librosa.resample(
            audio, orig_sr=sr, target_sr=CLAP_SAMPLE_RATE,
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# Text-embedding compute + cache
# ---------------------------------------------------------------------------

def _processor_to_numpy_feed(inputs, session) -> dict:
    """Filter a transformers `BatchFeature` down to the keys ORT wants.

    The processor returns a dict-like with several tensors; the ONNX
    graph only declares a subset as inputs. We hand-pick by input
    name to avoid `Got invalid dimensions for input X` errors on
    accidental extras.
    """
    import numpy as np
    feed: dict[str, "np.ndarray"] = {}
    wanted = {inp.name for inp in session.get_inputs()}
    for name in wanted:
        if name not in inputs:
            # The CLAP audio graph sometimes declares `is_longer` even
            # though the processor only emits it when fed audio longer
            # than 10s. Synthesize a False flag when it's missing —
            # all our windows are exactly 10s by construction.
            if name == "is_longer":
                feed[name] = np.zeros((1, 1), dtype=bool)
                continue
            raise KeyError(
                f"CLAP processor did not emit required input {name!r}; "
                f"transformers + model version drift? processor keys: "
                f"{sorted(inputs.keys())}"
            )
        val = inputs[name]
        # transformers may return torch tensors when torch is installed;
        # cast to numpy for ORT. The `.numpy()` path is zero-copy on CPU.
        if hasattr(val, "numpy"):
            val = val.numpy()
        feed[name] = val
    return feed


def _compute_text_embeddings(text_session, processor, vocab: list[str]):
    """Encode every vocab label through the text tower in ONE pass.

    Returns L2-normalized embeddings shape (N, D) so that cosine
    similarity later collapses to a plain matrix multiply. CLAP's
    canonical projection size is 512 for HTSAT-unfused; the larger
    variant uses 1024.
    """
    import numpy as np

    prompts = [LABEL_TEMPLATE.format(label) for label in vocab]
    # padding=True pads the batch to its longest item; truncation
    # caps at the model's max_position_embeddings (77 for CLAP, same
    # as CLIP). Both are upstream-defaults so we don't override them.
    inputs = processor.tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="np",
    )
    feed = _processor_to_numpy_feed(inputs, text_session)
    outputs = text_session.run(None, feed)
    # The Xenova export returns the projected text_embeds as the first
    # output. Defensive: flatten/squeeze any leading singleton dims so
    # the downstream matmul broadcasts cleanly.
    text_embeds = outputs[0]
    if text_embeds.ndim == 3 and text_embeds.shape[0] == 1:
        text_embeds = text_embeds[0]

    norm = np.linalg.norm(text_embeds, axis=-1, keepdims=True) + 1e-10
    return (text_embeds / norm).astype(np.float32)


def _load_text_cache(edit_dir: Path, vocab_sha: str):
    """Return cached text embeddings if the sha matches, else None."""
    import numpy as np

    cache_path = edit_dir / VOCAB_EMBEDS_CACHE_NAME
    if not cache_path.exists():
        return None
    try:
        data = np.load(str(cache_path), allow_pickle=False)
        cached_sha = str(data["sha"].item()) if data["sha"].shape == () else str(data["sha"])
        if cached_sha != vocab_sha:
            return None
        return data["embeds"]
    except (OSError, ValueError, KeyError):
        # Corrupt / partial / wrong-format cache file — ignore and
        # let the caller recompute. Don't try to delete it; the next
        # write will overwrite atomically anyway.
        return None


def _persist_text_cache(edit_dir: Path, vocab_sha: str, text_embeds, vocab: list[str]) -> None:
    """Write the (sha, embeds, vocab) bundle to a single .npz file.

    `vocab` is persisted alongside for diagnostics — `np.load` it and
    you can audit what labels produced these embeddings without
    re-reading the source video's vocab text file.
    """
    import numpy as np

    cache_path = edit_dir / VOCAB_EMBEDS_CACHE_NAME
    tmp_path = cache_path.with_suffix(".npz.tmp")
    np.savez(
        str(tmp_path),
        sha=np.array(vocab_sha),
        embeds=text_embeds.astype(np.float32),
        vocab=np.array(vocab, dtype=object),
    )
    _atomic_replace_with_retry(tmp_path, cache_path)


# ---------------------------------------------------------------------------
# Audio-side inference: slide a window across the WAV and score each
# ---------------------------------------------------------------------------

def _embed_audio_batch(audio_session, processor, audio_clips_48k):
    """Encode a batch of fixed-length 48 kHz clips, return (B, D) embeds.

    All clips MUST be the same length (we enforce 10s windows upstream)
    so the processor produces a uniform batch tensor. The CLAP audio
    encoder is internally batchable, so feeding N clips at once is
    strictly cheaper than N separate forward passes.
    """
    import numpy as np

    inputs = processor.feature_extractor(
        audio_clips_48k,
        sampling_rate=CLAP_SAMPLE_RATE,
        return_tensors="np",
    )
    feed = _processor_to_numpy_feed(inputs, audio_session)
    outputs = audio_session.run(None, feed)
    audio_embeds = outputs[0]
    if audio_embeds.ndim == 3 and audio_embeds.shape[0] == 1:
        audio_embeds = audio_embeds[0]

    norm = np.linalg.norm(audio_embeds, axis=-1, keepdims=True) + 1e-10
    return (audio_embeds / norm).astype(np.float32)


def _slide_and_score(
    audio_session,
    processor,
    audio_48k,
    text_embeds,
    vocab: list[str],
    *,
    window_s: float,
    hop_s: float,
    top_k: int,
    threshold: float,
    windows_per_batch: int,
) -> list[dict]:
    """Slide a `window_s` window with `hop_s` stride across the audio
    and emit the top-K (label, score) hits per window above `threshold`.

    Short audio (< window_s) is zero-padded up to a single window so
    we never silently drop short clips. The tail of long audio gets
    one extra window anchored at `n - window_samples` so the final
    seconds are always covered even when (n - window) % hop != 0.
    """
    import numpy as np

    win_samples = int(window_s * CLAP_SAMPLE_RATE)
    hop_samples = max(1, int(hop_s * CLAP_SAMPLE_RATE))
    n = len(audio_48k)

    if n < win_samples:
        # Pad with silence to one full window. We never emit a window
        # shorter than 10s because the encoder's mel front-end pads
        # internally anyway, and explicit padding here keeps the
        # batch tensor uniform.
        padded = np.zeros(win_samples, dtype=np.float32)
        padded[:n] = audio_48k
        audio_48k = padded
        n = win_samples

    starts: list[int] = list(range(0, n - win_samples + 1, hop_samples))
    if not starts:
        starts = [0]
    # Ensure we always emit a window covering the very end of the audio.
    # Without this guard a 12s clip with hop=5 would only get windows at
    # [0, 5] -> covers 0-15s but the actual data ends at 12s; we still
    # include start=2 so the last 10s slice ends exactly at the audio end.
    tail_start = max(0, n - win_samples)
    if starts[-1] != tail_start:
        starts.append(tail_start)

    raw_events: list[dict] = []

    # Batch the windows so the audio encoder's GPU pass is fully
    # utilized. windows_per_batch is sized in `wealthy.py`; bigger on
    # 24 GB+ cards. We also clamp here against the actual window count
    # to avoid an empty trailing batch.
    for batch_off in range(0, len(starts), windows_per_batch):
        batch_starts = starts[batch_off : batch_off + windows_per_batch]
        clips = [audio_48k[s : s + win_samples] for s in batch_starts]
        audio_embeds = _embed_audio_batch(audio_session, processor, clips)

        # (B, D) @ (D, N) -> (B, N) cosine sim matrix. Both sides are
        # already L2-normalized so the dot product IS the cosine.
        sims = audio_embeds @ text_embeds.T

        for i, s in enumerate(batch_starts):
            row = sims[i]
            # argpartition would be faster for huge vocabs but vocabs
            # of a few hundred labels make argsort essentially free.
            top_idx = np.argsort(-row)[:top_k]
            start_s = s / CLAP_SAMPLE_RATE
            end_s = (s + win_samples) / CLAP_SAMPLE_RATE
            for idx in top_idx:
                score = float(row[int(idx)])
                if score < threshold:
                    # `top_idx` is sorted descending so once we drop
                    # below threshold the remainder definitely will too.
                    break
                raw_events.append({
                    "start": round(start_s, 3),
                    "end":   round(end_s, 3),
                    "label": vocab[int(idx)],
                    "score": round(score, 4),
                })

    return raw_events


def _coalesce(events: list[dict], max_gap_s: float = COALESCE_MAX_GAP_S) -> list[dict]:
    """Merge consecutive same-label events whose end->start gap is small.

    Sustained sounds (a 30s drilling pass, applause for the entire end
    of a clip) naturally produce a label hit in EVERY overlapping
    window. Without coalescing the timeline would have 6 separate
    `drill` rows for one continuous drilling — visually noisy and
    harder for the agent LLM to reason about. After coalescing the
    same drilling pass collapses to one `[start, end] drill score`
    row, where `score` is the MAX score across the merged windows.
    """
    if not events:
        return []

    # Group by label so we can merge each label's hits independently.
    by_label: dict[str, list[dict]] = {}
    for ev in events:
        by_label.setdefault(ev["label"], []).append(ev)

    merged: list[dict] = []
    for label, evs in by_label.items():
        evs.sort(key=lambda e: e["start"])
        cur = dict(evs[0])
        for nxt in evs[1:]:
            # Negative gap means overlap (50% by default with our
            # WINDOW=10s / HOP=5s cadence) — definitely merge.
            if nxt["start"] - cur["end"] <= max_gap_s:
                cur["end"] = max(cur["end"], nxt["end"])
                cur["score"] = max(cur["score"], nxt["score"])
            else:
                merged.append(cur)
                cur = dict(nxt)
        merged.append(cur)

    # Final order: ascending start, then descending score so a busy
    # window's loudest match renders first under it.
    merged.sort(key=lambda e: (e["start"], -e["score"]))
    return merged


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _is_cache_valid(out_path: Path, video_path: Path, model_id: str, vocab_sha: str) -> bool:
    """Cache hit needs (a) fresh mtime AND (b) matching model id AND (c)
    matching vocab sha. Any failure means re-tag.
    """
    try:
        if out_path.stat().st_mtime < video_path.stat().st_mtime:
            return False
    except OSError:
        return False
    try:
        cached = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if str(cached.get("model", "")) != model_id:
        return False
    if str(cached.get("vocab_sha", "")) != vocab_sha:
        return False
    if not isinstance(cached.get("events"), list):
        return False
    return True


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def _process_one(
    audio_session,
    processor,
    vocab: list[str],
    text_embeds,
    vocab_sha: str,
    video_path: Path,
    edit_dir: Path,
    *,
    model_id: str,
    window_s: float,
    hop_s: float,
    threshold: float,
    top_k: int,
    windows_per_batch: int,
    force: bool,
) -> Path:
    """Tag one video against the prepared vocab + text embeddings."""
    out_dir = (edit_dir / AUDIO_TAGS_SUBDIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}.json"

    if not force and _is_cache_valid(out_path, video_path, model_id, vocab_sha):
        print(f"  audio_lane cache hit: {out_path.name}")
        return out_path

    wav_path = extract_audio_for(video_path, edit_dir, verbose=True)
    print(f"  clap: loading audio from {wav_path.name}")
    audio_48k = _load_audio_48k(wav_path)
    duration = len(audio_48k) / CLAP_SAMPLE_RATE
    print(
        f"  clap: {duration:.1f}s @ 48kHz, "
        f"window={window_s:.1f}s hop={hop_s:.1f}s "
        f"vocab={len(vocab)} threshold={threshold} top_k={top_k} "
        f"batch={windows_per_batch}"
    )

    t0 = time.time()
    raw_events = _slide_and_score(
        audio_session, processor, audio_48k, text_embeds, vocab,
        window_s=window_s,
        hop_s=hop_s,
        top_k=top_k,
        threshold=threshold,
        windows_per_batch=windows_per_batch,
    )
    events = _coalesce(raw_events)
    dt = time.time() - t0

    payload = {
        "model": model_id,
        "vocab_sha": vocab_sha,
        "vocab_size": len(vocab),
        "window_s": window_s,
        "hop_s": hop_s,
        "threshold": threshold,
        "top_k": top_k,
        "duration": round(duration, 3),
        "events": events,
    }
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _atomic_replace_with_retry(tmp_path, out_path)

    print(
        f"  audio_lane done: {len(events)} merged event(s) "
        f"({len(raw_events)} raw hits), {dt:.1f}s wall -> {out_path.name}"
    )
    return out_path


# ---------------------------------------------------------------------------
# Public batch entry — loads model + text embeds ONCE, processes N videos
# ---------------------------------------------------------------------------

def run_audio_lane_batch(
    video_paths: list[Path],
    edit_dir: Path,
    *,
    vocab_path: Path | str | None = None,
    model_tier: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    window_s: float = DEFAULT_WINDOW_S,
    hop_s: float = DEFAULT_HOP_S,
    windows_per_batch: int | None = None,
    device: str = "cuda",
    force: bool = False,
) -> list[Path]:
    """Tag N videos against `vocab_path` (or the baked-in default).

    Cold-start cost (model download + ORT session construction + text
    encoding) is paid once and amortized across all N videos. The
    text-embedding cache further amortizes ACROSS preprocess runs that
    share the same vocab.

    All-cache-hit fast path: if every output JSON is already fresh AND
    matches the current model + vocab sha, we skip BOTH the model load
    and the vocab text encoding entirely.
    """
    # Coerce vocab_path: the orchestrator JSON-serializes it as a str.
    if vocab_path is not None and not isinstance(vocab_path, Path):
        vocab_path = Path(str(vocab_path))

    # Resolve model tier from explicit arg or wealthy mode.
    if model_tier is None:
        model_tier = CLAP_MODEL_TIER_WEALTHY if is_wealthy(False) else CLAP_MODEL_TIER_DEFAULT
    if model_tier not in MODEL_TIERS:
        raise ValueError(
            f"unknown model_tier={model_tier!r} "
            f"(valid: {list(MODEL_TIERS)})"
        )
    model_id = MODEL_TIERS[model_tier]

    # Resolve windows-per-batch from explicit arg or wealthy mode.
    if windows_per_batch is None:
        windows_per_batch = (
            CLAP_WINDOWS_PER_BATCH_WEALTHY if is_wealthy(False) else CLAP_WINDOWS_PER_BATCH
        )

    # Load + sha the vocab. If `vocab_path is None`, `load_vocab`
    # returns the baked-in default list.
    vocab = load_vocab(vocab_path)
    if not vocab:
        raise ValueError(
            f"vocab is empty (vocab_path={vocab_path!r}); the audio lane "
            f"needs at least one label to score against"
        )
    vocab_sha = _hash_vocab(model_id, vocab)
    print(
        f"  clap: model={model_id}  tier={model_tier}  "
        f"vocab={len(vocab)} labels  sha={vocab_sha}"
    )

    out_dir = (edit_dir / AUDIO_TAGS_SUBDIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # All-cache-hit fast path. Avoids the ~5-10s of model + processor
    # + session construction on a fully-cached re-run.
    if not force:
        all_fresh = all(
            _is_cache_valid(out_dir / f"{v.stem}.json", v, model_id, vocab_sha)
            for v in video_paths
        )
        if all_fresh:
            print(
                f"  audio_lane: all {len(video_paths)} cache hits "
                f"(model + vocab match), skipping model load"
            )
            return [out_dir / f"{v.stem}.json" for v in video_paths]

    # Bootstrap: download + sessions + processor.
    print(f"  clap: ensuring ONNX weights for {model_id}")
    model_dir = _download_clap_onnx(model_id)

    # CLAP doesn't benefit from TRT (small graphs, 2-5 min compile cost
    # dwarfs any per-window speedup). Skip the prefer_tensorrt branch.
    providers = resolve_providers(prefer_tensorrt=False)

    audio_session = _build_audio_session(model_dir, providers)
    text_session = _build_text_session(model_dir, providers)
    processor = _load_processor(model_dir)

    # Text-embedding cache lookup. If miss, encode once + persist.
    text_embeds = _load_text_cache(edit_dir, vocab_sha)
    if text_embeds is None:
        print(
            f"  clap: encoding {len(vocab)} vocab labels through text "
            f"tower (first run for this vocab; cache miss)"
        )
        t0 = time.time()
        text_embeds = _compute_text_embeddings(text_session, processor, vocab)
        print(f"  clap: text encoding done in {time.time() - t0:.1f}s")
        _persist_text_cache(edit_dir, vocab_sha, text_embeds, vocab)
    else:
        print(
            f"  clap: reusing cached text embeddings for sha={vocab_sha} "
            f"({text_embeds.shape[0]} labels)"
        )

    out_paths: list[Path] = []

    # Per-video progress: one tick per video. Per-window timing is
    # uniform within a video so video-level granularity is honest.
    try:
        with lane_progress(
            "audio",
            total=len(video_paths),
            unit="video",
            desc="audio event tagging (CLAP)",
        ) as bar:
            for v in video_paths:
                bar.start_item(v.name)
                out_paths.append(_process_one(
                    audio_session, processor, vocab, text_embeds, vocab_sha,
                    v, edit_dir,
                    model_id=model_id,
                    window_s=window_s,
                    hop_s=hop_s,
                    threshold=threshold,
                    top_k=top_k,
                    windows_per_batch=windows_per_batch,
                    force=force,
                ))
                bar.update(advance=1, item=v.name)
    finally:
        # Drop sessions so the visual lane can claim the VRAM if we
        # were running in SEQUENTIAL mode (orchestrator default).
        try:
            del audio_session
            del text_session
        except NameError:
            pass

    return out_paths


def run_audio_lane(
    video_path: Path,
    edit_dir: Path,
    **kwargs,
) -> Path:
    """Single-video convenience wrapper around `run_audio_lane_batch`."""
    return run_audio_lane_batch([video_path], edit_dir, **kwargs)[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audio lane: LAION CLAP via ONNX -> vocab-scored sound events",
    )
    ap.add_argument(
        "video", type=Path,
        help="Path to source video file",
    )
    ap.add_argument(
        "--edit-dir", type=Path, default=None,
        help="Edit output dir (default: <video parent>/edit)",
    )
    ap.add_argument(
        "--vocab", type=Path, default=None,
        help="Path to a per-line vocabulary file (blank lines and # "
             "comments allowed). When omitted, the baked-in default "
             "vocab from helpers/audio_vocab_default.py is used. "
             "Tip: have the Claude agent write a curated vocab from "
             "speech_timeline.md + visual_timeline.md, then re-run "
             "this lane with --vocab <edit>/audio_vocab.txt --force.",
    )
    ap.add_argument(
        "--model", choices=list(MODEL_TIERS.keys()), default=None,
        help=f"CLAP variant (default: 'base'={MODEL_TIERS['base']}, "
             f"'large'={MODEL_TIERS['large']} when --wealthy)",
    )
    ap.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Cosine sim floor for keeping a hit (default: {DEFAULT_THRESHOLD})",
    )
    ap.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Max labels per window (default: {DEFAULT_TOP_K})",
    )
    ap.add_argument(
        "--window-s", type=float, default=DEFAULT_WINDOW_S,
        help=f"Sliding window length, seconds (default: {DEFAULT_WINDOW_S})",
    )
    ap.add_argument(
        "--hop-s", type=float, default=DEFAULT_HOP_S,
        help=f"Sliding window stride, seconds (default: {DEFAULT_HOP_S})",
    )
    ap.add_argument(
        "--windows-per-batch", type=int, default=None,
        help=f"Windows per audio-encoder batch (default: "
             f"{CLAP_WINDOWS_PER_BATCH}, or {CLAP_WINDOWS_PER_BATCH_WEALTHY} "
             f"with --wealthy)",
    )
    ap.add_argument(
        "--wealthy", action="store_true",
        help="Use the larger CLAP variant + bigger batches on 24GB+ cards. "
             "Also reads VIDEO_USE_WEALTHY=1.",
    )
    ap.add_argument(
        "--device", default="cuda",
        help="cuda | cpu (default: cuda; CLAP runs on either, CPU is ~5-8x slower)",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Bypass cache, always re-tag.",
    )
    args = ap.parse_args()

    install_lane_prefix()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    # If --wealthy was passed on the CLI, mirror it to the env so the
    # tier resolution inside run_audio_lane_batch sees it. (Same
    # pattern the orchestrator uses for batch runs.)
    if args.wealthy:
        os.environ["VIDEO_USE_WEALTHY"] = "1"

    run_audio_lane(
        video_path=video,
        edit_dir=edit_dir,
        vocab_path=args.vocab,
        model_tier=args.model,
        threshold=args.threshold,
        top_k=args.top_k,
        window_s=args.window_s,
        hop_s=args.hop_s,
        windows_per_batch=args.windows_per_batch,
        device=args.device,
        force=args.force,
    )


if __name__ == "__main__":
    main()
