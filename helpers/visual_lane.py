"""Visual lane: Florence-2-base captions at 1 fps for the entire timeline.

For an LLM editor to spot match cuts, identify shots, find B-roll
candidates, or react to "show the part where they're using the drill",
it needs *describable* visual context -- not raw frames. Florence-2-base
(230M params, MIT Microsoft Research License) is the speed champion:

    RTX 4090: 50-100 fps with batching, ~5 minutes for 10k frames
    RTX 3060: ~20 fps, ~10 minutes for 10k frames

Sampling at 1 fps means a 3-hour shoot is ~10,800 frames. That's a 5-15
min preprocess on consumer hardware which is the right ballpark.

We use the `<MORE_DETAILED_CAPTION>` task -- Florence-2's most descriptive
mode. Sample output:

    "a person holding a cordless drill above a metal panel with visible
     rivet holes"

JSON shape:
    {
      "model": "onnx-community/Florence-2-base",
      "fps": 1,
      "duration": 43.0,
      "captions": [
        {"t": 12, "text": "a person holding a cordless drill ..."},
        {"t": 13, "text": "close-up of a drill bit entering metal, sparks"},
        {"t": 14, "text": "(same)"},      # dedup marker, see _dedup_consecutive
        ...
      ]
    }

Backend
-------
ONNX Runtime via :mod:`florence_onnx` (custom 4-subgraph orchestrator
with real beam=3, no_repeat_ngram=3, forced BOS/EOS -- algorithmically
identical to the previous torch path) plus an optional multi-instance
pool from :mod:`_florence_pool` for intra-batch parallelism on
high-VRAM cards.

Why not torch + transformers anymore: the ONNX path eliminates the
torch + transformers + accelerate + optimum + timm + einops + flash-attn
install set entirely (~3 GB of wheels, ~5s of cold-start eager
imports), gives us 1.3-1.8x faster per-frame latency on a CUDA EP,
and lets us opt into a TensorRT decoder for another ~1.5-2x via
``VIDEO_USE_FLORENCE_TRT=1``.  Caption text is bit-for-bit identical
within fp16 numerical noise -- same model weights, same logits
processors, same beam scorer, same prompt strings.

License note: Florence-2 ships under the MS Research License which is
non-commercial. README documents this. SigLIP / BLIP-2 are drop-in
replaceable behind the same module interface if commercial use matters.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

# NOTE: no _hf_env import here -- the visual lane no longer touches
# transformers (the previous torch path needed `USE_TF=0` guards to
# avoid eager TensorFlow probe at import time; the ONNX path doesn't
# import transformers at all).  audio_lane.py keeps the guard for CLAP.

# Sibling helpers folder is on sys.path when invoked from the orchestrator.
# extract_audio is NOT used here -- visual lane is fully independent of
# the audio extraction step.
from progress import install_lane_prefix, lane_progress
from wealthy import (
    FLORENCE_BATCH,
    florence_pool_size,
    is_wealthy,
)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# HF community ONNX export of Florence-2-base.  Same weights, same MIT
# license, same task prompts as the original microsoft repo -- but
# pre-converted to four ONNX subgraphs (vision_encoder, embed_tokens,
# encoder_model, decoder_model_merged) for ORT consumption.  This is
# the only Florence-2 repo on the Hub whose ONNX I/O contract matches
# what the :mod:`florence_onnx` orchestrator expects, which is why the
# default isn't user-configurable through the orchestrator -- changing
# repos silently produces gibberish captions.
DEFAULT_MODEL_ID = "onnx-community/Florence-2-base"

# Legacy model ids users / scripts may still pass through.  Silently
# remapped to DEFAULT_MODEL_ID with a one-time print warning so old
# tests / cached configs don't crash on the rename.  The actual weights
# are pretrained-equivalent; the ONNX repo just exports them with
# identical numerics.
_LEGACY_MODEL_REMAP: dict[str, str] = {
    "microsoft/Florence-2-base":         DEFAULT_MODEL_ID,
    "florence-community/Florence-2-base": DEFAULT_MODEL_ID,
}

DEFAULT_FPS = 1
DEFAULT_BATCH_SIZE = 8           # safe on 8 GB; orchestrator bumps via --wealthy
DEFAULT_TASK_PROMPT = "<MORE_DETAILED_CAPTION>"
VISUAL_CAPS_SUBDIR = "visual_caps"


# ---------------------------------------------------------------------------
# Frame extraction -- 1 fps via ffmpeg, decoded to in-memory raw RGB.
# For very long shoots, writing 10k PNGs to disk would be wasteful --
# we stream raw rgb24 buffers through a Popen pipe instead.
# ---------------------------------------------------------------------------

def _video_duration_s(video_path: Path) -> float:
    """Quick ffprobe to get duration. Used for progress + batch math."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    try:
        return float(out.strip())
    except ValueError:
        return 0.0


def _iter_frames_at_fps(video_path: Path, fps: int):
    """Yield (timestamp_s, ndarray uint8) for each sampled frame.

    Yields raw NumPy arrays of shape ``(768, 768, 3) uint8`` -- exactly
    the format :class:`florence_onnx.FlorenceCaptioner.caption_batch`
    expects (the ONNX path bypasses PIL entirely; the previous torch
    path needed PIL.Image instances for the HF processor's resize +
    rescale chain, but our pure-NumPy
    :class:`_florence_processor.FlorenceImageProcessor` operates on
    ndarrays directly, saving a Python-side allocation per frame).

    Uses imageio_ffmpeg's bundled ffmpeg binary to stream raw RGB out
    of ffmpeg without writing PNGs to disk.  ~3x faster than the disk
    roundtrip for long shoots and avoids leaving thousands of stale
    PNGs in the edit dir.
    """
    import numpy as np
    import imageio_ffmpeg

    # Probe size first -- imageio_ffmpeg needs explicit (w, h) for raw read.
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(video_path),
    ]
    probe = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
    try:
        w_str, h_str = probe.stdout.strip().split("x")
        width, height = int(w_str), int(h_str)
    except ValueError:
        raise RuntimeError(f"could not probe dimensions of {video_path}")

    # ------------------------------------------------------------------
    # Square center-crop + downscale to Florence's native input size,
    # baked into the ffmpeg filter chain.
    #
    # Why crop:
    #   Florence-2's vision tower has a hard assertion that the encoded
    #   feature map is square (`assert h * w == num_tokens, 'only
    #   support square feature maps for now'` -- `modeling_florence2.py`
    #   line ~2610).  With non-square pixel_values (e.g. 16:9 4K DJI
    #   footage) the embedding produces a non-square map and the
    #   assertion explodes mid-generate.  Same constraint applies to
    #   the ONNX export -- the vision_encoder.onnx graph carries the
    #   same fixed-size positional embedding lookup.
    #
    # Why scale to 768:
    #   Florence-2-base ships with a fixed-size learned positional
    #   embedding table sized for 768x768 / patch_size=32 -> 24x24=576
    #   tokens.  Hand it any other resolution and the patch embedding
    #   indexes out-of-bounds -> device-side assert (which surfaces as
    #   a misleading async CUDA error on the next op).
    #
    # Why ffmpeg-side crop+scale instead of PIL post-decode:
    #   1. ffmpeg does both ops BEFORE rgb24 conversion, so the pipe
    #      carries `768*768*3 = 1.7 MB` per frame instead of
    #      `width * height * 3` (4K = ~25 MB).  ~14x reduction in pipe
    #      bandwidth + Python-side allocator churn at 1 fps over a
    #      multi-hour shoot.
    #   2. PIL.Image.{crop,resize} would force temporary copies in
    #      user-space Python; ffmpeg's filter graph does both inside
    #      the decoder with zero extra allocation.
    #   3. ffmpeg's `lanczos` resampler is higher quality than PIL's
    #      default (bilinear) for big downscales -- meaningful for
    #      detail-rich captioning targets.
    #
    # We center-crop to `min(width, height)` so portrait, landscape,
    # and already-square footage all become square.  ffmpeg's `crop`
    # filter defaults to centered when x/y are omitted.
    # ------------------------------------------------------------------
    square_dim = min(width, height)
    target_dim = 768  # Florence-2-base native input size; see preprocessor_config.json

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_bin, "-loglevel", "error",
        "-i", str(video_path),
        # Filter chain order matters: fps decimation FIRST (cheap, drops
        # ~95% of frames before we pay the crop+scale cost), THEN crop
        # to square, THEN scale to Florence's native size with lanczos
        # for sharp downscaling.  Order of crop->scale (rather than the
        # reverse) saves ffmpeg a needless aspect-preserving resize.
        "-vf",
        f"fps={fps},crop={square_dim}:{square_dim},"
        f"scale={target_dim}:{target_dim}:flags=lanczos",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo", "-",
    ]

    # Subprocess.Popen so we can stream stdout in frame-sized chunks.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Per-frame size reflects the FINAL output of the filter chain --
    # post-crop, post-scale.  Misreading this would mis-frame every
    # chunk and yield garbage / a hang on the final partial chunk.
    frame_size = target_dim * target_dim * 3
    t = 0
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if not buf or len(buf) < frame_size:
                break
            # `frombuffer` returns a read-only view onto the bytes
            # buffer.  `.copy()` materializes our own writable buffer
            # so the next `proc.stdout.read()` doesn't reuse the same
            # underlying memory and silently mutate frames we've
            # already yielded.  np.frombuffer is the fast path here
            # (no Python-level loop, just a buffer protocol cast).
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(
                target_dim, target_dim, 3,
            ).copy()
            yield t, arr
            t += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Florence-2 ONNX captioner construction.
#
# Replaces the old `_build_florence` torch path.  The new flow:
#
#   1. download_florence_onnx() -- HF snapshot_download of the four
#      ONNX subgraphs + tokenizer.json + config.json, cached in
#      ~/.cache/huggingface.  Idempotent; subsequent calls hit cache.
#   2. FlorenceCaptionerPool(model_dir, desired_size=N, dtype=...,
#      quantized_decoder=...) -- builds N independent captioner
#      instances, each with its own four ORT sessions.  VRAM-clamped
#      so passing N=4 on an 8 GB card silently degrades to N=2.
#
# The pool is built ONCE per process and reused across every video in
# the batch.  Same pattern as the old torch model load -- amortizing
# the ~3-5s load cost is what makes batching across many videos worth
# it vs. one-process-per-video.
# ---------------------------------------------------------------------------


def _resolve_model_id(model_id: str) -> str:
    """Map legacy / torch-era model ids to the ONNX-community equivalent.

    The visual lane previously defaulted to ``microsoft/Florence-2-base``
    or ``florence-community/Florence-2-base``; both are the same weights
    in different HF repo layouts.  The ONNX path needs the
    ``onnx-community/Florence-2-base`` repo (which has the four
    pre-exported ONNX subgraphs).  We remap silently with a one-time
    notice so old test scripts and cached env vars don't crash.
    """
    if model_id in _LEGACY_MODEL_REMAP:
        new_id = _LEGACY_MODEL_REMAP[model_id]
        if model_id != new_id:
            print(
                f"  visual_lane: remapping legacy model id "
                f"{model_id!r} -> {new_id!r} (ONNX export of the same weights)"
            )
        return new_id
    return model_id


def _build_pool(
    model_id: str,
    *,
    dtype_name: str,
    quantized_decoder: bool,
    pool_size: int,
):
    """Download the ONNX snapshot + build a FlorenceCaptionerPool.

    Args:
        model_id: HF Hub repo id.  Auto-remapped via
            :func:`_resolve_model_id` if the caller passed a legacy
            torch-era id.
        dtype_name: ``"fp16"`` (default, recommended) or ``"fp32"``
            (paranoid quality reference).  ``"bf16"`` is silently
            mapped to fp16 for backward compat with the old torch
            CLI -- ORT's CUDA EP doesn't carry a bf16 path for
            Florence-2's ops at the time of writing.
        quantized_decoder: When True, use the q4f16 decoder weights
            (~1.5-2x faster decoder, very minor caption drift).
        pool_size: Number of parallel captioner instances.  Pool clamps
            this down if VRAM is tight.

    Returns:
        A ready-to-use :class:`_florence_pool.FlorenceCaptionerPool`.
        Caller is responsible for ``.close()`` (or use the returned
        object as a context manager).
    """
    # Lazy imports keep module load cheap.  The orchestrator's import
    # probe just wants to confirm `import visual_lane` works -- it
    # shouldn't pay the ORT-init cost until an actual run starts.
    from florence_onnx import download_florence_onnx
    from _florence_pool import FlorenceCaptionerPool

    resolved_id = _resolve_model_id(model_id)

    # Coerce dtype_name to the captioner's accepted set.  bf16 falls
    # back to fp16 because the ONNX exports are fp16/fp32-only -- bf16
    # would require re-exporting through onnxconverter-common, which
    # is more complexity than the bf16 path is worth on Florence-base.
    if dtype_name == "bf16":
        print(
            "  visual_lane: dtype 'bf16' is not supported by the ONNX path; "
            "falling back to fp16 (numerically equivalent for caption quality)"
        )
        dtype_name = "fp16"
    if dtype_name not in ("fp16", "fp32"):
        raise ValueError(
            f"unknown dtype '{dtype_name}'; valid: fp16, fp32"
        )

    print(
        f"  florence-onnx: model={resolved_id}  dtype={dtype_name}  "
        f"quantized_decoder={quantized_decoder}  pool_size={pool_size}"
    )

    model_dir = download_florence_onnx(
        model_id=resolved_id,
        dtype=dtype_name,
        quantized_decoder=quantized_decoder,
    )

    pool = FlorenceCaptionerPool(
        model_dir,
        desired_size=pool_size,
        dtype=dtype_name,
        quantized_decoder=quantized_decoder,
    )
    return pool


# ---------------------------------------------------------------------------
# Dedup: collapse consecutive identical (or near-identical) captions to
# "(same)" markers in the markdown view.  Saves ~30-50% on tokens for
# static / slow-moving footage.  We keep all raw text in the JSON cache
# so re-rendering with a different dedup policy is just a pack-step rerun.
# ---------------------------------------------------------------------------

def _normalize_for_compare(s: str) -> str:
    return " ".join(s.lower().split())


def _dedup_consecutive(captions: list[dict]) -> list[dict]:
    """Mark runs of identical captions with text='(same)' after the first.
    Mutates a copy; returns the new list.
    """
    out: list[dict] = []
    last_norm: str | None = None
    for c in captions:
        norm = _normalize_for_compare(c["text"])
        if last_norm is not None and norm == last_norm:
            out.append({"t": c["t"], "text": "(same)"})
        else:
            out.append(dict(c))
            last_norm = norm
    return out


# ---------------------------------------------------------------------------
# Main lane entry point
# ---------------------------------------------------------------------------

def _process_one(
    pool,
    video_path: Path,
    edit_dir: Path,
    *,
    model_id: str,
    fps: int,
    batch_size: int,
    task: str,
    num_beams: int,
    max_new_tokens: int,
    force: bool,
) -> Path:
    """Caption one video with an already-built FlorenceCaptionerPool.

    Split out so the batch entry point can amortize the ~3-5s pool load
    across many videos in one Python process.

    Memory strategy
    ---------------
    We accumulate frames into a buffer of at most
    ``batch_size * pool.size`` ndarrays before dispatching to
    ``pool.caption_batch``.  Each frame is ~1.7 MB (768*768*3 uint8)
    so even with batch=32 and pool=2 the buffer ceiling is ~110 MB --
    safe even for hour-long shoots.  Bounded buffer means hour-long
    inputs don't blow up Python heap; the trade-off is that the pool
    sits idle for ~50 ms while we extract the next chunk.  Acceptable
    overhead at our caption latency (~300-600 ms per chunk).
    """
    import numpy as np  # noqa: F401  - kept for downstream symmetry

    out_dir = (edit_dir / VISUAL_CAPS_SUBDIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}.json"

    if not force and out_path.exists():
        try:
            if out_path.stat().st_mtime >= video_path.stat().st_mtime:
                print(f"  visual_lane cache hit: {out_path.name}")
                return out_path
        except OSError:
            pass

    duration = _video_duration_s(video_path)
    expected_frames = max(1, int(math.ceil(duration * fps)))
    buffer_cap = max(1, batch_size * pool.size)
    print(
        f"  florence: {video_path.name}  duration={duration:.1f}s  "
        f"frames~{expected_frames} @ {fps}fps  batch={batch_size}  "
        f"pool={pool.size}  buffer={buffer_cap}"
    )

    captions: list[dict] = []
    # buffer_imgs / buffer_ts grow together; we never index one without
    # the other, and they get cleared in lockstep after each dispatch.
    buffer_imgs: list = []
    buffer_ts: list[int] = []
    t0 = time.time()

    # Per-frame progress is genuinely informative here -- Florence is
    # the slow lane, so users want to see frames-per-second crawl
    # forward.  We tick once per BATCH (not per frame) to keep emit
    # volume sane.  When pool.size > 1 the chunk-complete callback
    # advances by chunk size as each worker finishes, so progress
    # updates feel smoother than a single end-of-buffer jump.
    with lane_progress(
        "visual",
        total=expected_frames,
        unit="frame",
        desc=f"florence captions: {video_path.name}",
    ) as fbar:

        # Closure: pool.caption_batch fires this from worker threads
        # as each chunk finishes.  lane_progress.update is thread-safe
        # (counter increment + structured print), so calling from
        # multiple workers concurrently is fine.
        def _on_chunk_done(_chunk_idx: int, n_frames: int, _captions: list[str]) -> None:
            fbar.update(advance=n_frames)

        def _flush() -> None:
            """Send accumulated buffer through the pool, drain results, repeat."""
            if not buffer_imgs:
                return
            # `chunk_size=batch_size` slices the buffer into pool.size
            # equal chunks of <= batch_size each so every worker gets
            # a full chunk in parallel.  When pool.size == 1 this is
            # a single-chunk dispatch (one worker submit + join, ~50us
            # of overhead -- negligible vs. caption latency).
            texts = pool.caption_batch(
                buffer_imgs,
                task=task,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                chunk_size=batch_size,
                on_chunk_complete=_on_chunk_done,
            )
            # Splat results back into the per-frame caption list,
            # preserving the input order (pool guarantees this even
            # though chunks may complete out-of-order across workers).
            for tt, txt in zip(buffer_ts, texts):
                captions.append({"t": tt, "text": txt})
            buffer_imgs.clear()
            buffer_ts.clear()

        for ts, img in _iter_frames_at_fps(video_path, fps):
            buffer_imgs.append(img)
            buffer_ts.append(ts)
            if len(buffer_imgs) >= buffer_cap:
                _flush()

        # Trailing partial buffer.  Dispatch even if it's < buffer_cap
        # so we don't drop the tail of the video.
        _flush()

    dt = time.time() - t0

    captions_md = _dedup_consecutive(captions)
    payload = {
        # Use the pool's effective model_id (which encodes the
        # quantized-decoder suffix when the q4f16 variant is loaded)
        # so the JSON sidecar accurately identifies which Florence
        # variant produced these captions.  Falls back to the
        # caller-provided model_id if the pool didn't expose one.
        "model": getattr(pool, "model_id", model_id),
        "task": task,
        "fps": fps,
        "duration": round(duration, 3),
        # Generation knobs preserved in the sidecar so a downstream
        # consumer can verify the captions were produced with beam-3
        # (vs. greedy beam=1 from --num-beams 1).  Cheap to record
        # and useful for debugging caption-quality regressions.
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "captions": captions,            # raw
        "captions_dedup": captions_md,   # display copy
    }
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)

    rate = len(captions) / max(1e-3, dt)
    print(f"  visual_lane done: {len(captions)} captions, {dt:.1f}s wall "
          f"({rate:.1f} fps) -> {out_path.name}")
    return out_path


def run_visual_lane_batch(
    video_paths: list[Path],
    edit_dir: Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    fps: int = DEFAULT_FPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = "cuda:0",       # kept for backward compat (no-op on ONNX path)
    dtype_name: str = "fp16",
    task: str = DEFAULT_TASK_PROMPT,
    num_beams: int | None = None,
    max_new_tokens: int | None = None,
    quantized: bool = False,
    pool_size: int | None = None,
    wealthy: bool = False,
    force: bool = False,
) -> list[Path]:
    """Run the visual lane on N videos with Florence-2 loaded ONCE.

    Args:
        video_paths: Source videos.  All sidecar JSONs land in
            ``edit_dir / visual_caps / <stem>.json``.
        edit_dir: Per-project edit directory.
        model_id: HF Hub repo.  Defaults to the ONNX community port;
            legacy torch-era ids (``microsoft/Florence-2-base``,
            ``florence-community/Florence-2-base``) are remapped
            silently for backward compatibility.
        fps: Sampling rate.  1 fps is the recommended default.
        batch_size: Per-captioner batch size for the vision encoder
            and beam-search dispatch.  Bigger = better GPU utilization
            but more VRAM peak.  Default 8 fits on 8 GB cards;
            ``--wealthy`` bumps to 32.
        device: LEGACY no-op kwarg.  The ONNX path picks the device
            from the EP ladder (``CUDAExecutionProvider`` on NVIDIA,
            ``DmlExecutionProvider`` on Windows non-NVIDIA, etc.) --
            see :mod:`_onnx_providers`.  Kept on the signature so the
            orchestrator's ``--device cuda:0`` flag passthrough doesn't
            crash; ignored at runtime.
        dtype_name: ``"fp16"`` (default) or ``"fp32"``.  ``"bf16"``
            silently maps to fp16 (no ONNX bf16 export available).
        task: Florence task token.  ``<MORE_DETAILED_CAPTION>`` is the
            default; OD/OCR/region tasks intentionally raise.
        num_beams: Beam width override.  ``None`` -> Florence default
            (3).  Set to 1 for greedy decoding (~1.5-2x faster, slight
            quality drop on detailed captions).
        max_new_tokens: Hard generation cap override.  ``None`` ->
            Florence default (256).  Smaller caps speed up worst-case
            beams that would otherwise generate to the limit.
        quantized: Use the q4f16 decoder weight variant (~1.5-2x
            decoder speedup, very minor caption drift on long
            generations).
        pool_size: Override the auto-resolved pool size.  ``None``
            falls through to ``florence_pool_size(wealthy)``.
        wealthy: Forwarded to ``florence_pool_size`` and used as a
            second source of truth for ``is_wealthy``.  The
            orchestrator already propagates ``VIDEO_USE_WEALTHY=1``
            via env so passing ``wealthy=True`` here is rarely needed.
        force: Bypass the per-video sidecar cache and re-caption.

    Returns:
        List of output JSON paths in the same order as ``video_paths``.
    """
    # `device` is intentionally accepted but ignored.  The lane's
    # subprocess shim (`preprocess.py::_run_lane`) passes
    # `--device cuda:0` blindly for every lane; failing on it would
    # break the orchestrator without giving us anything in return.
    del device

    # Lazy imports inside this function so just `import visual_lane`
    # (the orchestrator's import probe) doesn't pay the ORT init cost.
    from florence_onnx import DEFAULT_NUM_BEAMS, DEFAULT_MAX_NEW_TOKENS

    out_dir = (edit_dir / VISUAL_CAPS_SUBDIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # All-cache-hit short-circuit: if every output is fresh we skip
    # the expensive pool load entirely.  Same logic as the old torch
    # path -- still worth it because the ONNX pool load (~3-5s for 4
    # sessions x N instances) is faster than the old torch load (~5-8s)
    # but still meaningful when running orchestrator dry-runs.
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
            print(
                f"  visual_lane: all {len(video_paths)} cache hits, "
                f"skipping model load"
            )
            return [out_dir / f"{v.stem}.json" for v in video_paths]

    # Resolve generation hyperparameters with Florence defaults if
    # caller didn't override.  Done here (not at the kwarg default)
    # so the caller can pass `num_beams=None` to get the model default
    # without us having to introspect the captioner.
    eff_beams = DEFAULT_NUM_BEAMS if num_beams is None else int(num_beams)
    eff_max_new = (
        DEFAULT_MAX_NEW_TOKENS if max_new_tokens is None else int(max_new_tokens)
    )
    if eff_beams < 1:
        raise ValueError(f"num_beams must be >= 1, got {num_beams}")
    if eff_max_new < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

    # Resolve pool size.  Explicit `pool_size` arg wins; else the
    # wealthy-tier resolver (which checks env + CLI flag).  The pool
    # itself further clamps to fit available VRAM at construction.
    eff_pool_size = (
        int(pool_size)
        if pool_size is not None and pool_size >= 1
        else florence_pool_size(wealthy)
    )

    # Build the pool ONCE per process.  Single load amortizes across
    # every video in this batch.
    pool = _build_pool(
        model_id,
        dtype_name=dtype_name,
        quantized_decoder=bool(quantized),
        pool_size=eff_pool_size,
    )
    out_paths: list[Path] = []
    try:
        # Outer bar tracks video-of-N progress; inner per-frame bar
        # (in _process_one) tracks current-video frame progress.  Both
        # emit their own structured PROGRESS lines so the orchestrator
        # / Claude can render either granularity.
        with lane_progress(
            "visual",
            total=len(video_paths),
            unit="video",
            desc="visual captioning",
        ) as vbar:
            for v in video_paths:
                vbar.start_item(v.name)
                out_paths.append(_process_one(
                    pool, v, edit_dir,
                    model_id=model_id,
                    fps=fps,
                    batch_size=batch_size,
                    task=task,
                    num_beams=eff_beams,
                    max_new_tokens=eff_max_new,
                    force=force,
                ))
                vbar.update(advance=1, item=v.name)
    finally:
        try:
            pool.close()
        except Exception:
            # Pool close is best-effort; a leak here would be cleaned
            # up by the subprocess teardown anyway.
            pass
    return out_paths


def run_visual_lane(
    video_path: Path,
    edit_dir: Path,
    **kwargs,
) -> Path:
    """Single-video convenience wrapper around run_visual_lane_batch."""
    return run_visual_lane_batch([video_path], edit_dir, **kwargs)[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visual lane: Florence-2 ONNX captions at N fps",
    )
    ap.add_argument("video", type=Path, help="Path to source video file")
    ap.add_argument(
        "--edit-dir", type=Path, default=None,
        help="Edit output dir (default: <video parent>/edit)",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL_ID,
                    help=f"HF model id (default: {DEFAULT_MODEL_ID})")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS,
                    help=f"Sample rate in frames/sec (default: {DEFAULT_FPS})")
    ap.add_argument("--batch-size", type=int, default=None,
                    help=f"Per-captioner batch size (default: {DEFAULT_BATCH_SIZE}, "
                         f"or {FLORENCE_BATCH} with --wealthy)")
    ap.add_argument("--num-beams", type=int, default=None,
                    help="Beam search width (default: 3 to match Florence-2's "
                         "training config; pass 1 for greedy decoding -- "
                         "~1.5-2x faster, slight caption-quality drop)")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="Hard cap on generated tokens per caption "
                         "(default: 256)")
    ap.add_argument("--quantized", action="store_true",
                    help="Use the q4f16 decoder weight variant. ~1.5-2x "
                         "faster decoder step on a CUDA EP, very minor "
                         "caption drift on long generations. Vision + text "
                         "encoder stay fp16.")
    ap.add_argument("--pool-size", type=int, default=None,
                    help="Number of parallel FlorenceCaptioner instances "
                         "(default: 1, or 2 with --wealthy). The pool clamps "
                         "this down if VRAM is tight; override with "
                         "VIDEO_USE_FLORENCE_POOL_SIZE=<N> too.")
    ap.add_argument("--wealthy", action="store_true",
                    help="Speed knob for 24GB+ cards (4090/5090). Bigger "
                         "batch + parallel captioner pool, same model + "
                         "outputs. Also reads VIDEO_USE_WEALTHY=1.")
    # `--device` retained for backward compat with the orchestrator's
    # subprocess shim, which passes `--device cuda:0` to every lane.
    # The ONNX path resolves the device via the EP ladder, not this
    # flag; we accept and ignore.
    ap.add_argument("--device", default=None,
                    help="LEGACY: ignored on the ONNX path. EP selection "
                         "lives in helpers/_onnx_providers.py "
                         "(VIDEO_USE_FLORENCE_TRT=1 opts into TensorRT).")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"],
                    help="Float dtype for the three non-decoder ONNX graphs. "
                         "bf16 silently maps to fp16 (no ONNX bf16 export).")
    ap.add_argument("--task", default=DEFAULT_TASK_PROMPT,
                    help="Florence task prompt (default: <MORE_DETAILED_CAPTION>)")
    ap.add_argument("--force", action="store_true",
                    help="Bypass cache, always re-caption.")
    args = ap.parse_args()

    install_lane_prefix()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    # Resolve batch size: explicit CLI value wins, else wealthy mode picks
    # the tier, else the conservative default.
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif is_wealthy(args.wealthy):
        batch_size = FLORENCE_BATCH
    else:
        batch_size = DEFAULT_BATCH_SIZE

    run_visual_lane(
        video_path=video,
        edit_dir=edit_dir,
        model_id=args.model,
        fps=args.fps,
        batch_size=batch_size,
        device=args.device,
        dtype_name=args.dtype,
        task=args.task,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        quantized=args.quantized,
        pool_size=args.pool_size,
        wealthy=args.wealthy,
        force=args.force,
    )


if __name__ == "__main__":
    main()
