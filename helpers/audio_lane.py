"""Audio events lane: PANNs CNN14 → AudioSet 527-class tags.

For non-speech audio context — drill, sawing, hammer, applause, laughter,
glass breaking, music — Whisper is the wrong tool (it transcribes speech and
collapses everything else into `(applause)` style tags at best). PANNs (the
PreTrained Audio Neural Network) trained on AudioSet has 527 fine-grained
classes and gives confidence scores per second.

We use the panns_inference package which wraps a CNN14 checkpoint:
    https://github.com/qiuqiangkong/panns_inference

Scheme:
    1. Load the 16k mono WAV (already cached by extract_audio_for).
    2. Slide a WINDOW_S-second window with HOP_S-second hop.
    3. For each window, take the CNN14 sigmoid output, keep top-K classes
       with confidence >= THRESHOLD.
    4. Adjacent windows with the same dominant class get coalesced into
       one event range (saves a lot of clutter in the markdown).

JSON shape:
    {
      "model": "PANNs CNN14 / AudioSet",
      "duration": 43.0,
      "events": [
        {"start": 12.04, "end": 12.40, "label": "drill", "score": 0.87},
        {"start": 12.04, "end": 12.40, "label": "power_tool", "score": 0.71},
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from extract_audio import SAMPLE_RATE, extract_audio_for
from progress import install_lane_prefix, lane_progress
from wealthy import PANNS_WINDOWS_PER_BATCH, is_wealthy


# ---------------------------------------------------------------------------
# PANNs data bootstrap
#
# `panns_inference` reads `~/panns_data/class_labels_indices.csv` at IMPORT
# time (its config.py opens the file at module load), but the package does
# NOT auto-fetch it. Fresh installs therefore explode with FileNotFoundError
# before `AudioTagging` is even constructed — and the error message points
# at panns internals so it looks like a packaging bug rather than a missing
# data file.
#
# We fetch the canonical CSV from the upstream `audioset_tagging_cnn` repo
# (same author as panns_inference) and drop it in panns's hardcoded location
# the first time the lane runs on a machine. Cost: ~10 KB one-time download.
# Co-located with the Cnn14 checkpoint that panns will fetch on first
# inference, so the whole `~/panns_data/` dir stays self-contained.
# ---------------------------------------------------------------------------

PANNS_DATA_DIR = Path.home() / "panns_data"
PANNS_LABELS_CSV = PANNS_DATA_DIR / "class_labels_indices.csv"
PANNS_LABELS_URL = (
    "https://raw.githubusercontent.com/qiuqiangkong/"
    "audioset_tagging_cnn/master/metadata/class_labels_indices.csv"
)
PANNS_CHECKPOINT_PATH = PANNS_DATA_DIR / "Cnn14_mAP=0.431.pth"
PANNS_CHECKPOINT_URL = (
    "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth"
)


# ---------------------------------------------------------------------------
# Atomic-rename retry helper (Windows Defender workaround)
#
# On Windows, `Path.replace()` can fail with PermissionError / WinError 32
# ("being used by another process") moments after a large file is closed.
# The culprit is almost never our code — the writer handle exits cleanly
# via the `with open(...)` block. The real offender is a transient scan
# handle held by:
#   * Windows Defender real-time protection (scans newly-written files
#     before any other process can rename/move them; this is its whole job)
#   * Windows Search Indexer / SearchProtocolHost
#   * Third-party AV (Kaspersky, McAfee, etc.)
#
# Defender's scan window for a ~340 MB binary is typically 200ms - 2s on a
# modern NVMe. So we retry the rename a handful of times with exponential
# backoff. Total worst-case wait ~3.1s — invisible next to the 5+ minute
# download we just performed, but enough to outlast the scan handle.
#
# We only retry on errors that smell like a sharing violation:
#   - EACCES (errno 13)
#   - WinError 32 / 33 (sharing violation / lock violation), which Python
#     surfaces as PermissionError with .winerror set
# Any other OSError is a real problem (disk full, permission denied at the
# directory level, etc.) and re-raises immediately so we don't mask bugs.
# ---------------------------------------------------------------------------

_WINERROR_SHARING_VIOLATION = 32  # ERROR_SHARING_VIOLATION
_WINERROR_LOCK_VIOLATION = 33     # ERROR_LOCK_VIOLATION


def _is_sharing_violation(exc: OSError) -> bool:
    """True iff `exc` looks like a transient Windows file-lock conflict.

    We deliberately accept both the cross-platform `errno.EACCES` and the
    Windows-specific `winerror` codes — different Python versions surface
    sharing violations through slightly different attributes.
    """
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
    max_attempts: int = 6,
    initial_delay_s: float = 0.1,
) -> None:
    """`tmp_path.replace(final_path)` with backoff for AV scan handles.

    Backoff schedule (default): 0.1, 0.2, 0.4, 0.8, 1.6 s between attempts
    — total ceiling ~3.1s of waiting. If the rename still fails after that,
    something is genuinely wrong (file actually open by user, weird mount,
    etc.) and the original error propagates so the caller can clean up.
    """
    delay = initial_delay_s
    last_exc: OSError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            tmp_path.replace(final_path)
            return
        except OSError as exc:
            # Re-raise non-transient errors immediately. Disk full, perm
            # denied, etc. won't get better with sleep.
            if not _is_sharing_violation(exc):
                raise
            last_exc = exc
            if attempt == max_attempts:
                break
            print(
                f"  panns: rename blocked (likely AV scan), "
                f"retrying in {delay:.1f}s (attempt {attempt}/{max_attempts})",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay *= 2.0
    # Out of retries — surface the last error with context.
    assert last_exc is not None  # for type checkers; loop above guarantees
    raise last_exc


def _ensure_panns_data_csv() -> None:
    """Guarantee `~/panns_data/class_labels_indices.csv` exists before we
    import `panns_inference`. Idempotent: returns instantly when the file
    is already present and non-empty.

    Raises RuntimeError with actionable advice when the network fetch
    fails — better than the cryptic FileNotFoundError panns would emit.
    """
    if PANNS_LABELS_CSV.exists() and PANNS_LABELS_CSV.stat().st_size > 0:
        return

    PANNS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  panns: bootstrapping label CSV -> {PANNS_LABELS_CSV}")

    # Atomic-write pattern: fetch into .tmp, then rename. Avoids leaving a
    # half-written CSV on Ctrl-C — panns would happily re-import it and
    # silently produce wrong labels for half the AudioSet vocabulary.
    tmp_path = PANNS_LABELS_CSV.with_suffix(".csv.tmp")
    try:
        with urllib.request.urlopen(PANNS_LABELS_URL, timeout=30) as resp:
            data = resp.read()
        tmp_path.write_bytes(data)
        # Same Defender-aware rename used by the checkpoint download below;
        # 10 KB rarely triggers the AV scan window but the cost is zero.
        _atomic_replace_with_retry(tmp_path, PANNS_LABELS_CSV)
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"could not download PANNs labels CSV from {PANNS_LABELS_URL} "
            f"to {PANNS_LABELS_CSV}: {type(exc).__name__}: {exc}. "
            f"If your network is restricted, place the file manually at "
            f"that path and re-run."
        ) from exc


# ---------------------------------------------------------------------------
# PANNs CNN14 checkpoint bootstrap
#
# Older `panns_inference` versions auto-fetched the CNN14 weights on first
# `AudioTagging(checkpoint_path=None)` call; current releases dropped that
# behavior and just crash with a FileNotFoundError pointing at
# `~/panns_data/Cnn14_mAP=0.431.pth`. Same problem shape as the labels CSV
# above — we mirror the same fix: fetch from the upstream Zenodo record
# the first time the lane runs on a machine.
#
# The checkpoint is ~340 MB so we stream it in 1 MB chunks (don't .read()
# the whole body — that would peak at 340 MB of resident RAM with zero
# user feedback). A 10 MB progress tick keeps the console alive without
# spamming the log.
# ---------------------------------------------------------------------------

def _ensure_panns_data_checkpoint() -> None:
    """Guarantee `~/panns_data/Cnn14_mAP=0.431.pth` exists before we hand
    `panns_inference.AudioTagging` a `checkpoint_path=None`. Idempotent:
    returns instantly when the file is already present and non-empty.

    Streams the ~340 MB body in 1 MB chunks with a coarse progress print
    every ~10 MB so the user knows the process isn't hung. Atomic-write
    pattern (`.pth.tmp` -> `Path.replace`) so a Ctrl-C mid-download never
    leaves a half-written checkpoint that PANNs would later mis-load.

    Raises RuntimeError with actionable advice when the network fetch
    fails — better than the cryptic FileNotFoundError panns would emit.
    """
    if PANNS_CHECKPOINT_PATH.exists() and PANNS_CHECKPOINT_PATH.stat().st_size > 0:
        return

    PANNS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Salvage path: a previous run may have completed the download but
    # crashed during the .tmp -> .pth rename (Windows Defender holding a
    # scan handle long enough to outlast our retry budget). In that case
    # the .pth.tmp file is on disk, fully written, and just needs a
    # rename. We use the heuristic "tmp exists and is at least 300 MB"
    # to avoid mistakenly promoting a half-downloaded fragment from a
    # genuine Ctrl-C scenario. Real checkpoint is ~340 MB.
    tmp_path = PANNS_CHECKPOINT_PATH.with_suffix(".pth.tmp")
    SALVAGE_MIN_BYTES = 300 * 1024 * 1024
    if tmp_path.exists() and tmp_path.stat().st_size >= SALVAGE_MIN_BYTES:
        print(
            f"  panns: salvaging existing tmp ({tmp_path.stat().st_size // (1024*1024)} MB) "
            f"from previous run -> {PANNS_CHECKPOINT_PATH.name}"
        )
        try:
            _atomic_replace_with_retry(tmp_path, PANNS_CHECKPOINT_PATH)
            return
        except OSError as exc:
            # Salvage failed too — fall through to a fresh download but
            # warn the user so they understand why we're spending 5 min
            # on bandwidth they already paid for.
            print(
                f"  panns: salvage rename failed ({exc}); re-downloading. "
                f"You may want to add this folder to your AV scan exclusions: "
                f"{PANNS_DATA_DIR}",
                file=sys.stderr,
            )
            try:
                tmp_path.unlink()
            except OSError:
                pass

    print(f"  panns: bootstrapping CNN14 checkpoint -> {PANNS_CHECKPOINT_PATH}")
    print(f"  panns: ~340 MB one-time download from {PANNS_CHECKPOINT_URL}")

    # Atomic-write pattern matches the CSV helper above. Same reasoning:
    # an interrupted half-file would silently break later inference runs.
    # 1 MB read chunks. Small enough to keep RAM flat, large enough that
    # the per-chunk Python overhead vanishes next to the network read.
    chunk_size = 1024 * 1024
    # Print a progress line every ~10 MB. Coarse on purpose — the goal is
    # "not hung", not a precise progress bar (we said no tqdm dep).
    progress_step = 10 * 1024 * 1024
    try:
        with urllib.request.urlopen(PANNS_CHECKPOINT_URL, timeout=60) as resp:
            # Content-Length lets us print "X / Y MB"; some mirrors omit
            # it, in which case we just print the running total.
            total_bytes = 0
            try:
                total_bytes = int(resp.headers.get("Content-Length", "0") or 0)
            except (TypeError, ValueError):
                total_bytes = 0
            total_mb = total_bytes / (1024 * 1024) if total_bytes else 0.0

            downloaded = 0
            next_tick = progress_step
            with open(tmp_path, "wb") as fh:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_tick:
                        done_mb = downloaded / (1024 * 1024)
                        if total_mb:
                            print(f"  panns: downloaded {done_mb:.0f} / {total_mb:.0f} MB")
                        else:
                            print(f"  panns: downloaded {done_mb:.0f} MB")
                        next_tick += progress_step
                # Force the kernel buffer to disk *before* the file handle
                # closes. This matters on Windows: as soon as the handle
                # closes, Defender opens its own scan handle, and if the
                # data is still in the page cache the scan takes longer
                # (= longer rename block). fsync makes the scan deterministic.
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    # fsync can fail on some weird filesystems (network mounts,
                    # tmpfs, etc.). Not fatal — the OS will flush eventually.
                    pass

            # Final summary line so the user sees a clean "done" rather
            # than the loop just stopping mid-progress.
            done_mb = downloaded / (1024 * 1024)
            if total_mb:
                print(f"  panns: downloaded {done_mb:.0f} / {total_mb:.0f} MB (complete)")
            else:
                print(f"  panns: downloaded {done_mb:.0f} MB (complete)")

        # Use the retry-aware helper instead of bare replace() — Defender
        # holds a scan handle on the freshly-written 340 MB blob for a few
        # hundred ms, which would otherwise crash us with WinError 32.
        _atomic_replace_with_retry(tmp_path, PANNS_CHECKPOINT_PATH)
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"could not download PANNs CNN14 checkpoint from "
            f"{PANNS_CHECKPOINT_URL} to {PANNS_CHECKPOINT_PATH}: "
            f"{type(exc).__name__}: {exc}. If your network is restricted, "
            f"place the file manually at that path and re-run."
        ) from exc


# ---------------------------------------------------------------------------
# Tunables. WINDOW_S=1.0 + HOP_S=0.5 mirrors PANNs' own demo and keeps the
# event grid at ~2 events/sec which packs nicely into the markdown view
# without losing transient sounds (a hammer hit is ~150ms — the 0.5s hop
# guarantees it lands in at least one window).
# ---------------------------------------------------------------------------

WINDOW_S = 1.0
HOP_S = 0.5
DEFAULT_THRESHOLD = 0.30
DEFAULT_TOP_K = 5
# Default windows-per-call. The single-window (=1) loop preserves exact
# legacy behavior; bumping to 16 already gives ~5-8x speedup on GPU with
# trivial memory cost. Wealthy mode raises to PANNS_WINDOWS_PER_BATCH.
DEFAULT_WINDOWS_PER_BATCH = 16
AUDIO_TAGS_SUBDIR = "audio_tags"


# ---------------------------------------------------------------------------
# Label normalization — AudioSet labels are space + uppercase ("Power tool",
# "Hammer", "Drill"). Convert to snake_case for cleaner markdown.
# ---------------------------------------------------------------------------

def _norm_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_").replace(",", "")


# ---------------------------------------------------------------------------
# WAV loader — we want a numpy float32 array at SAMPLE_RATE. soundfile is
# fastest for PCM_16; we already extracted that format.
# ---------------------------------------------------------------------------

def _load_wav_mono_16k(wav_path: Path):
    """Returns np.ndarray shape (n_samples,), float32 in [-1, 1]."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        # extract_audio guarantees 16k but defend anyway — a stray drop-in
        # WAV could break PANNs silently otherwise.
        raise ValueError(
            f"expected {SAMPLE_RATE} Hz audio, got {sr} for {wav_path}"
        )
    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------

def _slide_and_tag_with_tagger(
    tagger,
    audio,
    sample_rate: int,
    *,
    window_s: float,
    hop_s: float,
    top_k: int,
    threshold: float,
    windows_per_batch: int,
) -> list[dict]:
    """Run CNN14 over hop-spaced windows, return per-window top-K events.

    Performance note: the per-call overhead of `tagger.inference()` (Python
    + tensor staging) dwarfs the actual GPU forward pass for a single 1s
    16k window. Batching N windows into one (N, 16000) tensor collapses
    that overhead into one round trip — typical 8-15x speedup on GPU,
    smaller but still positive on CPU. `windows_per_batch=1` reproduces
    the old per-window loop exactly.

    Memory: one batch holds N * win_samples * 4 bytes (float32). At
    N=64, win=16000 that's ~4 MB — trivial. The output tensor adds
    N * 527 * 4 = ~135 KB per batch.
    """
    import numpy as np
    from panns_inference import labels as panns_labels

    win_samples = int(window_s * sample_rate)
    hop_samples = int(hop_s * sample_rate)
    n = len(audio)

    n_starts = max(1, 1 + (n - win_samples) // hop_samples) if n >= win_samples else 1

    # Build all (start_sample, end_sample) tuples up front. Cheap, lets us
    # slice the audio into a contiguous batch buffer below.
    starts: list[int] = [w_idx * hop_samples for w_idx in range(n_starts)]

    out: list[dict] = []

    # Process in chunks of `windows_per_batch`. Each chunk becomes one
    # (B, win_samples) tensor handed to PANNs in a single call.
    for batch_off in range(0, len(starts), windows_per_batch):
        batch_starts = starts[batch_off : batch_off + windows_per_batch]
        b = len(batch_starts)

        # Pre-allocate the batch buffer once per chunk. Rows beyond the
        # actual audio length are zero-padded (silence) which is what the
        # old per-window path did one-at-a-time.
        batch = np.zeros((b, win_samples), dtype=np.float32)
        for i, s in enumerate(batch_starts):
            e = min(n, s + win_samples)
            clip = audio[s:e]
            batch[i, : len(clip)] = clip

        clipwise_output, _embedding = tagger.inference(batch)  # (B, 527)

        # Decode each row's top-K above threshold.
        for i, s in enumerate(batch_starts):
            scores = clipwise_output[i]
            e = min(n, s + win_samples)
            idx_sorted = np.argsort(-scores)[:top_k]
            for idx in idx_sorted:
                score = float(scores[idx])
                if score < threshold:
                    break
                out.append({
                    "start": s / sample_rate,
                    "end": e / sample_rate,
                    "label": _norm_label(panns_labels[int(idx)]),
                    "score": round(score, 3),
                })

    return out


# ---------------------------------------------------------------------------
# Event coalescing — fold adjacent windows with the same label into one
# longer range. Massively cuts down the markdown noise on continuous sounds
# (a 5-second drill becomes ONE event line, not ten).
# ---------------------------------------------------------------------------

def _coalesce(events: list[dict], max_gap_s: float = 0.6) -> list[dict]:
    """Merge consecutive same-label events whose gap <= max_gap_s.
    Score of the merged range is the max of the constituents.
    """
    by_label: dict[str, list[dict]] = {}
    for ev in events:
        by_label.setdefault(ev["label"], []).append(ev)

    merged: list[dict] = []
    for label, evs in by_label.items():
        evs.sort(key=lambda e: e["start"])
        cur = dict(evs[0])
        for nxt in evs[1:]:
            if nxt["start"] - cur["end"] <= max_gap_s:
                cur["end"] = max(cur["end"], nxt["end"])
                cur["score"] = max(cur["score"], nxt["score"])
            else:
                merged.append(cur)
                cur = dict(nxt)
        merged.append(cur)

    merged.sort(key=lambda e: (e["start"], -e["score"]))
    return merged


# ---------------------------------------------------------------------------
# Main lane entry point
# ---------------------------------------------------------------------------

def _build_tagger(device: str):
    """Construct the PANNs CNN14 tagger. Split out so the batch entry
    point can amortize model load across many videos.

    Bootstraps the AudioSet labels CSV first — otherwise the
    `from panns_inference import AudioTagging` line below crashes inside
    panns's own config.py with a FileNotFoundError that obscures the real
    "no data file shipped with the package" cause. Then bootstraps the
    CNN14 checkpoint (current `panns_inference` releases no longer
    auto-fetch it on first use, so `AudioTagging(checkpoint_path=None)`
    would otherwise blow up with a FileNotFoundError for the .pth).
    """
    _ensure_panns_data_csv()
    _ensure_panns_data_checkpoint()
    from panns_inference import AudioTagging
    try:
        return AudioTagging(checkpoint_path=None, device=device)
    except TypeError:
        return AudioTagging(checkpoint_path=None)


def _process_one(
    tagger,
    video_path: Path,
    edit_dir: Path,
    *,
    threshold: float,
    top_k: int,
    windows_per_batch: int,
    force: bool,
) -> Path:
    """Run audio tagging on one video with an already-built tagger."""
    out_dir = (edit_dir / AUDIO_TAGS_SUBDIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}.json"

    if not force and out_path.exists():
        try:
            if out_path.stat().st_mtime >= video_path.stat().st_mtime:
                print(f"  audio_lane cache hit: {out_path.name}")
                return out_path
        except OSError:
            pass

    wav_path = extract_audio_for(video_path, edit_dir, verbose=True)
    print(f"  panns: loading audio from {wav_path.name}")
    audio = _load_wav_mono_16k(wav_path)
    duration = len(audio) / SAMPLE_RATE
    print(f"  panns: {duration:.1f}s, sliding {WINDOW_S}s window @ {HOP_S}s hop "
          f"(threshold={threshold}, top_k={top_k}, windows_per_batch={windows_per_batch})")

    t0 = time.time()
    raw_events = _slide_and_tag_with_tagger(
        tagger, audio, SAMPLE_RATE,
        window_s=WINDOW_S, hop_s=HOP_S,
        top_k=top_k, threshold=threshold,
        windows_per_batch=windows_per_batch,
    )
    events = _coalesce(raw_events)
    dt = time.time() - t0

    payload = {
        "model": "PANNs CNN14 / AudioSet",
        "window_s": WINDOW_S,
        "hop_s": HOP_S,
        "threshold": threshold,
        "top_k": top_k,
        "windows_per_batch": windows_per_batch,
        "duration": round(duration, 3),
        "events": events,
    }
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)

    print(f"  audio_lane done: {len(events)} events ({len(raw_events)} raw), "
          f"{dt:.1f}s wall → {out_path.name}")
    return out_path


def run_audio_lane_batch(
    video_paths: list[Path],
    edit_dir: Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    windows_per_batch: int = DEFAULT_WINDOWS_PER_BATCH,
    device: str = "cuda",
    force: bool = False,
) -> list[Path]:
    """Run the audio lane on N videos with the tagger loaded ONCE."""
    out_dir = (edit_dir / AUDIO_TAGS_SUBDIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not force:
        all_fresh = True
        for v in video_paths:
            out = out_dir / f"{v.stem}.json"
            try:
                if not out.exists() or out.stat().st_mtime < v.stat().st_mtime:
                    all_fresh = False
                    break
            except OSError:
                all_fresh = False
                break
        if all_fresh:
            print(f"  audio_lane: all {len(video_paths)} cache hits, skipping model load")
            return [out_dir / f"{v.stem}.json" for v in video_paths]

    tagger = _build_tagger(device)
    out_paths: list[Path] = []
    # One tick per video. PANNs' inner sliding-window loop is uniform
    # work so per-video granularity is honest; sub-window progress would
    # add more noise than signal in the structured log.
    try:
        with lane_progress(
            "audio",
            total=len(video_paths),
            unit="video",
            desc="audio event tagging",
        ) as bar:
            for v in video_paths:
                bar.start_item(v.name)
                out_paths.append(_process_one(
                    tagger, v, edit_dir,
                    threshold=threshold, top_k=top_k,
                    windows_per_batch=windows_per_batch,
                    force=force,
                ))
                bar.update(advance=1, item=v.name)
    finally:
        try:
            import torch
            del tagger
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return out_paths


def run_audio_lane(
    video_path: Path,
    edit_dir: Path,
    **kwargs,
) -> Path:
    """Single-video convenience wrapper around run_audio_lane_batch."""
    return run_audio_lane_batch([video_path], edit_dir, **kwargs)[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audio events lane: PANNs CNN14 → AudioSet 527 tags",
    )
    ap.add_argument("video", type=Path, help="Path to source video file")
    ap.add_argument(
        "--edit-dir", type=Path, default=None,
        help="Edit output dir (default: <video parent>/edit)",
    )
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Min confidence to keep an event (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                    help=f"Top-K labels per window (default {DEFAULT_TOP_K})")
    ap.add_argument("--windows-per-batch", type=int, default=None,
                    help=f"Windows per CNN14 forward pass (default "
                         f"{DEFAULT_WINDOWS_PER_BATCH}, or {PANNS_WINDOWS_PER_BATCH} "
                         f"with --wealthy)")
    ap.add_argument("--wealthy", action="store_true",
                    help="Speed knob for 24GB+ cards (4090/5090). Bigger PANNs batch, "
                         "same model + outputs. Also reads VIDEO_USE_WEALTHY=1.")
    ap.add_argument("--device", default="cuda",
                    help="cuda | cpu (default: cuda — falls through to cpu if no CUDA)")
    ap.add_argument("--force", action="store_true",
                    help="Bypass cache, always re-tag.")
    args = ap.parse_args()

    install_lane_prefix()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    # Resolve windows-per-batch: explicit CLI value wins, else wealthy mode
    # picks the tier, else the conservative default.
    if args.windows_per_batch is not None:
        wpb = args.windows_per_batch
    elif is_wealthy(args.wealthy):
        wpb = PANNS_WINDOWS_PER_BATCH
    else:
        wpb = DEFAULT_WINDOWS_PER_BATCH

    run_audio_lane(
        video_path=video,
        edit_dir=edit_dir,
        threshold=args.threshold,
        top_k=args.top_k,
        windows_per_batch=wpb,
        device=args.device,
        force=args.force,
    )


if __name__ == "__main__":
    main()
