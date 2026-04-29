"""Visual lane: Florence-2-base captions at N fps for the entire timeline.

For an LLM editor to spot match cuts, identify shots, find B-roll
candidates, or react to "show the part where they're using the drill",
it needs *describable* visual context — not raw frames. Florence-2-base
(230M params, MIT Microsoft Research License) is the speed champion:

    RTX 4090: 50–100 fps with batching, ~5 minutes for 10k frames
    RTX 3060: ~20 fps, ~10 minutes for 10k frames

Default sampling is 1 fps. A 3-hour shoot at 1 fps is ~10,800 frames —
a 5-15 min preprocess on consumer hardware which is the right ballpark
for typical talking-head / interview / tutorial work. For long, slow,
or static-pacing content (lecture, ambient B-roll, locked-off camera)
1 fps is overkill — every second emits a 5-7 sentence paragraph that
the dedup pass collapses to `(same)` 90% of the time, and the editor
sub-agent still has to scan past every line. Lower rates (0.5 = one
frame every 2 s, 0.25 = every 4 s, 0.1 = every 10 s) cut both
preprocess time AND merged-timeline token cost roughly in proportion.
Cost scales linearly with fps. Override via the `--fps` flag on this
CLI or `--visual-fps N` on `preprocess.py` / `preprocess_batch.py`.
Fractional values are accepted (ffmpeg's `fps=` filter handles them
natively).

We use the `<MORE_DETAILED_CAPTION>` task — Florence-2's most descriptive
mode. Sample output:

    "a person holding a cordless drill above a metal panel with visible
     rivet holes"

JSON shape:
    {
      "model": "microsoft/Florence-2-base",
      "fps": <float — actual sample rate this clip was captioned at>,
      "duration": 43.0,
      "captions": [
        # `t` is float seconds-from-start (idx / fps). At fps=1.0 it
        # coincides with the integer second; at fractional fps it
        # spaces out by `1/fps` seconds (fps=0.5 → t=0, 2, 4, ...).
        {"t": 12.0, "text": "a person holding a cordless drill ..."},
        {"t": 13.0, "text": "close-up of a drill bit entering metal, sparks"},
        {"t": 14.0, "text": "(same)"},      # dedup marker, see _dedup_consecutive
        ...
      ]
    }

License note: Florence-2 ships under the MS Research License which is
non-commercial. README documents this. SigLIP / BLIP-2 are drop-in
replaceable behind the same module interface if commercial use matters.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# CRITICAL: this import MUST come before anything that could pull in
# `transformers` (directly OR transitively via timm, einops, etc.). It
# sets `USE_TF=0` / `USE_FLAX=0` so transformers' module-load probe
# doesn't eagerly import a possibly-broken TensorFlow install (classic
# protobuf-version-mismatch crash on Windows). See `_hf_env.py` for
# the full rationale.
from _hf_env import HF_ENV_GUARDS_INSTALLED  # noqa: F401  - import for side effect

# Sibling helpers folder is on sys.path when invoked from the orchestrator.
# extract_audio is NOT used here — visual lane is fully independent of
# the audio extraction step.
from progress import install_lane_prefix, lane_progress
from wealthy import FLORENCE_BATCH, is_wealthy


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# HF community port of Florence-2-base. Same weights / same MIT license /
# same task prompts as `microsoft/Florence-2-base`, but uses the NATIVE
# `Florence2ForConditionalGeneration` class shipped in transformers 4.55+
# instead of the abandoned trust_remote_code modeling file. The original
# microsoft repo's remote code broke against transformers' new attention
# dispatcher contract (`_supports_sdpa` requirement) and HF themselves
# now point users at this checkpoint as the canonical replacement —
# see https://github.com/huggingface/transformers/issues/41622.
DEFAULT_MODEL_ID = "florence-community/Florence-2-base"
# Default sample rate. Float so callers can pass fractional values like
# 0.5 (one frame every 2 seconds) or 0.25 (every 4 seconds) for slow /
# long / static-pacing content where 1 fps over-samples and bloats the
# merged timeline. ffmpeg's `fps=` filter accepts arbitrary floats.
DEFAULT_FPS: float = 1.0
DEFAULT_BATCH_SIZE = 8           # safe on 4 GB; orchestrator can override
DEFAULT_TASK_PROMPT = "<MORE_DETAILED_CAPTION>"
VISUAL_CAPS_SUBDIR = "visual_caps"
# Florence-2-base's native vision-tower input is 768x768. The square
# crop + scale baked into the ffmpeg filter chain matches this exactly
# so the processor doesn't need to do an extra resize. Pulling the
# constant out keeps `_build_extract_cmd` and `run_visual_lane_batch`
# in sync — change in one place if you ever swap to a different
# Florence variant (e.g. Florence-2-large at 768 too, SigLIP at 384).
FLORENCE_INPUT_DIM = 768


# ---------------------------------------------------------------------------
# Frame extraction — 1 fps via ffmpeg, decoded to in-memory PNGs (or PIL
# Images via the imageio-ffmpeg generator). For very long shoots, writing
# 10k PNGs to disk would be wasteful — we stream instead.
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


def _iter_frames_at_fps(video_path: Path, fps: float):
    """Yield (timestamp_s, PIL.Image) for each sampled frame.

    Uses imageio_ffmpeg to stream raw RGB out of ffmpeg without writing
    PNGs to disk. This is ~3x faster than the disk roundtrip for long
    shoots and avoids leaving thousands of stale PNGs in the edit dir.
    """
    import numpy as np
    from PIL import Image
    import imageio_ffmpeg

    # Probe size first — imageio_ffmpeg needs explicit (w, h) for raw read.
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
    #   support square feature maps for now'` — `modeling_florence2.py`
    #   line ~2610). With non-square pixel_values (e.g. 16:9 4K DJI
    #   footage) the embedding produces a non-square map and the
    #   assertion explodes mid-generate.
    #
    # Why scale to 768:
    #   Florence-2-base ships with a fixed-size learned positional
    #   embedding table sized for 768x768 / patch_size=32 → 24x24=576
    #   tokens. Hand it any other resolution and `image_pos_embed(x)`
    #   indexes out-of-bounds → device-side assert (which surfaces as a
    #   misleading async CUDA error on the next op). transformers 5.x's
    #   CLIPImageProcessor *should* resize for us via its `do_resize`
    #   path, but the size-dict-vs-shortest-edge interpretation got
    #   reshuffled in the 5.x rewrite and the resize is silently a
    #   no-op on already-square inputs at non-default resolutions. We
    #   short-circuit the ambiguity by handing the processor exactly
    #   what its embedding table expects.
    #
    # Why ffmpeg-side crop+scale instead of PIL post-decode:
    #   1. ffmpeg does both ops BEFORE rgb24 conversion, so the pipe
    #      carries `768*768*3 = 1.7 MB` per frame instead of
    #      `width * height * 3` (4K = ~25 MB). ~14x reduction in pipe
    #      bandwidth + Python-side allocator churn at 1 fps over a
    #      multi-hour shoot.
    #   2. PIL.Image.{crop,resize} would force temporary copies in
    #      user-space Python; ffmpeg's filter graph does both inside
    #      the decoder with zero extra allocation.
    #   3. ffmpeg's `lanczos` resampler is higher quality than PIL's
    #      default (bilinear) for big downscales — meaningful for
    #      detail-rich captioning targets.
    #
    # We center-crop to `min(width, height)` so portrait, landscape,
    # and already-square footage all become square. ffmpeg's `crop`
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
        # for sharp downscaling. Order of crop->scale (rather than the
        # reverse) saves ffmpeg a needless aspect-preserving resize.
        "-vf",
        f"fps={fps},crop={square_dim}:{square_dim},"
        f"scale={target_dim}:{target_dim}:flags=lanczos",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo", "-",
    ]

    # Subprocess.Popen so we can stream stdout in frame-sized chunks.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Per-frame size reflects the FINAL output of the filter chain —
    # post-crop, post-scale. Misreading this would mis-frame every
    # chunk and yield garbage / a hang on the final partial chunk.
    frame_size = target_dim * target_dim * 3
    # Frame index counted in produced frames; the actual timestamp in
    # seconds is `idx / fps`. Computed as float so fps>1 carries
    # sub-second precision through to the JSON cache.
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if not buf or len(buf) < frame_size:
                break
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(
                target_dim, target_dim, 3
            )
            # Florence-2 takes PIL images. Conversion is cheap (no copy).
            yield (idx / float(fps)), Image.fromarray(arr, mode="RGB")
            idx += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Hardware-accelerated streaming pipe (NVDEC → ffmpeg stdout → Florence).
#
# Design intent (the version 3 rewrite):
#
#   Florence loads ONCE, sits primed and hungry on the GPU. For each
#   clip we spawn a single ffmpeg subprocess that streams 768x768 rgb24
#   frames out its stdout, into a producer thread that wraps each frame
#   as a numpy array and pushes it onto a bounded queue. The main
#   thread (consumer) drains batches off the queue, hands them to
#   Florence, writes the captions out, and DROPS the references. Peak
#   in-flight memory is `batch_size * 2 * 1.69 MB` ≈ 27-110 MB,
#   regardless of clip length. No bulk pre-extract, no disk cache,
#   no JPEG roundtrip.
#
# Why we abandoned the bulk-extract-to-disk model:
#
#   The previous design wrote 768x768 mjpeg frames (`-q:v 3`) to a
#   disk cache via N parallel ffmpeg workers, then PIL-decoded them
#   back into RAM at consume time. That cost was pure waste:
#
#     - mjpeg encode is single-threaded (~5-10 ms per 768² frame at q=3)
#     - filesystem write + fsync (variable but real, especially Windows NTFS)
#     - filesystem read + libjpeg-turbo decode in PIL (~3-8 ms per frame)
#
#   That's 10-25 ms × thousands of frames = real wall-clock for ZERO
#   information gain — the mjpeg-encoded data was about to be decoded
#   straight back to the same rgb24 we could have just kept in RAM.
#   Killing it is the dominant speedup of this rewrite.
#
# Why we abandoned the multi-clip parallel decoder pool:
#
#   The 4-worker prefetch pool assumed Florence was the bottleneck and
#   we needed to saturate NVDEC independently. In practice ffmpeg+NVDEC
#   produces 768² rgb24 frames at >200 fps even on entry-level RTX
#   cards, while Florence-2-base maxes at ~50-100 fps on a 4090. So one
#   ffmpeg producer always leads one Florence consumer — adding more
#   producers just steals NVDEC sessions and burns RAM with no win.
#
#   Trade-off: we lose the ~50-100 ms of clip-boundary overlap (next
#   clip's ffmpeg launching while Florence finishes current clip's last
#   batch). On an N-clip shoot that's N×100ms ≈ 1-2 seconds total. We'll
#   add a one-clip-lookahead later if it ever shows up in profiling.
#
# Why hwaccel + tonemap stays exactly the same:
#
#   The filter chain (`fps=N → tonemap (HDR only) → crop=square → scale=768
#   → rgb24`) is identical to the v2 design. Only the OUTPUT muxer
#   changes: `-q:v 3 %06d.jpg` becomes `-pix_fmt rgb24 -f rawvideo -`.
#   That means HDR (HLG/PQ) → SDR Rec.709 still works, NVDEC → CPU
#   tonemap → re-format still works, and the SDR fast-path still skips
#   tonemap entirely. We just stream the result instead of muxing it
#   to disk.
#
# Why python's BufferedReader.read(n) is safe here:
#
#   Each frame is exactly `target_dim * target_dim * 3` bytes (768²×3
#   = 1,769,472 bytes). Python's `read(n)` on a binary pipe blocks
#   until exactly n bytes have arrived OR the pipe hits EOF. So we
#   read frame-by-frame in a fixed-size loop, and a short read tells
#   us cleanly that ffmpeg is done. No need to parse a container or
#   sniff frame boundaries.
#
# CPU fallback (the user explicitly asked for this):
#
#   _nvdec_available() probes once at process start. If NVDEC is
#   present we try it first; if ffmpeg's exit code is non-zero AND we
#   haven't yielded any frames yet, the producer restarts the same
#   clip with software decode. Mid-stream NVDEC failures (rare) are
#   surfaced as a producer error so the consumer sees them.
#
# HDR detection: ffprobe `color_transfer`. The two HDR transfers in
# the wild are `arib-std-b67` (HLG, what DJI uses) and `smpte2084`
# (PQ, what Sony / Apple use). Anything else (`bt709`, blank, etc.)
# is treated as SDR and skips the tonemap chain.
# ---------------------------------------------------------------------------

# Cached NVDEC capability probe — we only run `ffmpeg -hwaccels` once
# per process. None means "not yet probed".
_NVDEC_PROBED: bool | None = None


def _nvdec_available() -> bool:
    """One-shot probe: does this ffmpeg build expose CUDA hwaccel?

    Memoized so repeated calls (one per clip extraction) are free.
    Defensive: any failure path (timeout, OSError, parse failure)
    returns False so we silently fall back to software decode.
    """
    global _NVDEC_PROBED
    if _NVDEC_PROBED is not None:
        return _NVDEC_PROBED
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        out = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-hwaccels"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout
        _NVDEC_PROBED = "cuda" in out.lower()
    except Exception:
        _NVDEC_PROBED = False
    return _NVDEC_PROBED


def _probe_video_meta(video_path: Path) -> dict:
    """ffprobe single-shot: width/height + HDR transfer characteristic.

    Returns a small dict the extractor uses to choose a filter chain.
    Any probe failure yields {"is_hdr": False} so we degrade to the
    SDR path rather than crashing the whole batch.
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,color_transfer,color_primaries,color_space",
        "-of", "json", str(video_path),
    ]
    try:
        out = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=10,
        ).stdout
        data = json.loads(out)
        s = data.get("streams", [{}])[0]
        transfer = (s.get("color_transfer") or "").lower()
        # The two HDR transfer functions in the wild. Everything else
        # (bt709, smpte170m, blank metadata, etc.) is treated as SDR.
        is_hdr = transfer in {"smpte2084", "arib-std-b67"}
        return {
            "width": int(s.get("width") or 0),
            "height": int(s.get("height") or 0),
            "transfer": transfer,
            "is_hdr": is_hdr,
        }
    except Exception as e:
        print(f"  extract: probe failed for {video_path.name} ({e}); assuming SDR 1920x1080")
        return {"width": 1920, "height": 1080, "transfer": "", "is_hdr": False}


def _build_extract_cmd(
    video_path: Path,
    target_dim: int,
    fps: float,
    meta: dict,
    *,
    use_nvdec: bool,
) -> list[str]:
    """Construct an ffmpeg argv that streams raw rgb24 frames to stdout.

    Filter chain order, picked for performance:

        fps=N        : decimate to N fps FIRST so every downstream filter
                       only touches our target sample rate instead of
                       all 60. By far the biggest CPU saving on 4K60.
                       N is float — fractional values like 0.5 (one
                       frame every 2 s) or 0.25 (every 4 s) are valid
                       and ffmpeg picks the nearest source frame for
                       each output time bucket.

        [tonemap]    : HDR → SDR if needed. CPU-side (no NVENC tonemap
                       in mainline ffmpeg). The chain is the canonical
                       zscale → linearise → tonemap (Hable) → BT.709
                       round-trip recommended by the FFmpeg HDR docs.
                       Skipped entirely on SDR sources.

        crop=SxS     : centre crop to the shortest side. Florence's
                       vision tower asserts a square feature map.

        scale=DxD    : lanczos downscale to Florence's native 768x768.

    Output stage: `-pix_fmt rgb24 -f rawvideo -` writes raw interleaved
    RGB to stdout, exactly `target_dim * target_dim * 3` bytes per
    frame. The producer thread reads in fixed-size chunks and wraps
    each chunk as a numpy view — no muxer, no encode, no disk.

    NVDEC path:
        Hardware-decodes the H.264 / H.265 / AV1 stream on the dedicated
        NVDEC block, then `hwdownload,format=nv12` brings the frame to
        system RAM for the SDR conversion + crop + scale. This sounds
        wasteful but a 768² rgb24 frame is ~1.7 MB and we only emit at
        the decimated `fps`, not the source rate. CUDA filters
        (`scale_cuda` etc.) don't compose cleanly with `tonemap` so
        we'd need the download anyway on HDR sources — one code path
        for both SDR and HDR keeps the filter graph predictable.
    """
    import imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    width = meta.get("width") or 1920
    height = meta.get("height") or 1080
    square = max(2, min(width, height))

    # ── Tonemap chain (HDR sources only) ──────────────────────────────
    # zscale: linearise the HDR transfer (HLG or PQ) to scene-referred
    #         linear light, normalised to a 100-nit display peak.
    # format: float planar so tonemap has headroom.
    # zscale: BT.709 primaries (chromatic adaptation).
    # tonemap=hable: shoulder-knee curve, no desaturation. Hable was
    #         developed for Uncharted 2 and is the safest default for
    #         mixed-content HDR (vs. mobius / reinhard which clip
    #         highlights or wash out shadows).
    # zscale: BT.709 transfer (sRGB-ish for SDR display).
    # format=yuv420p: standard 8-bit chroma sub for downstream filters.
    if meta.get("is_hdr"):
        tonemap = (
            "zscale=t=linear:npl=100,"
            "format=gbrpf32le,"
            "zscale=p=bt709,"
            "tonemap=tonemap=hable:desat=0,"
            "zscale=t=bt709:m=bt709:r=tv,"
            "format=yuv420p,"
        )
    else:
        tonemap = ""

    # crop+scale always. fps decimation goes first to slash the work.
    crop_scale = f"crop={square}:{square},scale={target_dim}:{target_dim}:flags=lanczos"
    vf = f"fps={fps},{tonemap}{crop_scale}"

    base = [ffmpeg_bin, "-loglevel", "error", "-nostdin"]
    if use_nvdec:
        # `-hwaccel cuda` decodes on NVDEC; we deliberately do NOT pass
        # `-hwaccel_output_format cuda` because tonemap + zscale don't
        # exist on the GPU pipeline — letting ffmpeg auto-download to
        # system RAM after decode keeps the filter graph happy on both
        # SDR and HDR paths with one code path.
        base += ["-hwaccel", "cuda"]
    base += ["-i", str(video_path), "-vf", vf]
    # Raw rgb24 to stdout. `-an` skips any audio stream from being
    # written to the output (we don't want it on the pipe). `-` is
    # ffmpeg's stdout sigil — the producer thread Popen's this and
    # reads from proc.stdout in fixed `target_dim * target_dim * 3`
    # byte chunks.
    base += ["-an", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    return base


# ---------------------------------------------------------------------------
# Florence-2 model construction.
#
# We use `florence-community/Florence-2-base` — the HF-format port of
# `microsoft/Florence-2-base` that ships with the NATIVE
# `Florence2ForConditionalGeneration` class (added to transformers in
# 4.55). No `trust_remote_code`, no remote modeling file, no monkey
# patches required for the four 5.x dispatcher / cache / tokenizer
# breakages we used to carry. HF themselves point users at the
# community port now; see issue #41622.
#
# Same weights, same MIT license, same task prompts (`<MORE_DETAILED_CAPTION>`
# etc.) as the original microsoft repo — just the loading mechanics changed.
# ---------------------------------------------------------------------------

def _build_florence(
    model_id: str,
    device: str,
    dtype_name: str,
    *,
    compile_enabled: bool = False,
):
    """Construct Florence-2 from the HF community port.

    Uses the native `Florence2ForConditionalGeneration` class shipped in
    transformers 4.55+ — no `trust_remote_code`, no remote modeling file,
    no monkey patches. Falls back to `AutoModelForImageTextToText` if the
    explicit class import fails (forward-compatible with transformers
    moving the class around in future minor releases).

    Parameters
    ----------
    dtype_name : str
        One of `"auto"`, `"fp16"`, `"bf16"`, `"fp32"`. `"auto"` resolves
        to `"bf16"` on CUDA devices that report bf16 support (Ampere or
        newer — RTX 30/40/50 series, A100, H100, etc.) and `"fp16"`
        elsewhere (Turing / Pascal / Volta CUDA, plus all non-CUDA
        backends). bf16 has the same memory footprint as fp16 but a
        wider exponent range, which sidesteps the dynamic-loss-scale
        guards transformers wraps generate() in for fp16 and is the
        native math format on Blackwell tensor cores. Numerical output
        is within rounding noise of fp16 for vision-language inference.

    compile_enabled : bool
        When True (and the device is CUDA), wrap the model in
        `torch.compile(mode=<resolved>, fullgraph=False)` after
        construction. The mode string is resolved by
        `_resolve_compile_mode` from the
        `VIDEO_USE_FLORENCE_COMPILE_MODE` env var and defaults to
        "default" — see that resolver's docstring for the full
        rationale on why "reduce-overhead" is opt-in (CUDA-Graph
        capture deadlocks on autoregressive decode with variable
        sequence lengths). "default" still buys ~10-20% on Blackwell
        from Inductor kernel fusion, without the CUDA-Graph memory
        pinning that wedges long batches.

        Any failure during compile is caught and the eager model is
        used instead — no compile bug should ever break a run. The
        orchestrator enables compile when len(videos) > 1 because the
        warmup amortizes nicely across many clips; the standalone CLI
        disables it by default to keep single-clip invocations snappy.
    """
    import torch
    from transformers import AutoProcessor

    # ------------------------------------------------------------------
    # dtype resolution: "auto" picks bf16 when the hardware reports it,
    # falling back to fp16 otherwise. Done HERE (not at the call site)
    # so every entrypoint — orchestrator shim, standalone CLI, tests —
    # gets the same hardware-aware default without each one re-implementing
    # the probe. `torch.cuda.is_bf16_supported()` requires CUDA to be
    # initialized; the `device.startswith("cuda")` guard ensures we only
    # call it on CUDA backends, otherwise we silently fall back to fp16
    # for MPS / CPU / non-CUDA accelerators.
    # ------------------------------------------------------------------
    if dtype_name == "auto":
        try:
            if device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype_name = "bf16"
                print("  florence: dtype=auto -> bf16 (Ampere+ tensor cores)")
            else:
                dtype_name = "fp16"
                print("  florence: dtype=auto -> fp16 (no bf16 support / non-cuda)")
        except Exception as _e:
            # Belt + suspenders: if the bf16 probe blows up for any reason
            # (driver glitch, exotic backend), fall back to fp16 — the
            # historically-known-safe path.
            dtype_name = "fp16"
            print(f"  florence: dtype=auto -> fp16 (bf16 probe failed: "
                  f"{type(_e).__name__})")

    # Tiny version sniff to pick the right dtype kwarg name.
    # transformers 5.x renamed `torch_dtype` -> `dtype` and emits a
    # DeprecationWarning on every load if the old name is used. 4.x
    # still requires `torch_dtype`. One cheap probe at load time
    # silences the noise on 5.x without breaking the 4.55-4.x range.
    import transformers as _tf
    _tf_major = int(_tf.__version__.split(".", 1)[0])

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    if dtype_name not in dtype_map:
        raise ValueError(f"unknown dtype '{dtype_name}'")
    torch_dtype = dtype_map[dtype_name]
    dtype_kwarg = {"dtype": torch_dtype} if _tf_major >= 5 else {"torch_dtype": torch_dtype}

    # Prefer the explicit class import — it's the path the model card
    # documents and gives us a clear ImportError if the user's
    # transformers is too old (< 4.55, before the class was added).
    # Fall back to AutoModelForImageTextToText, which is Florence-2's
    # registered Auto class per its config.json. Both paths are
    # zero-cost — they resolve to the same class object.
    try:
        from transformers import Florence2ForConditionalGeneration as _ModelCls
    except ImportError:
        from transformers import AutoModelForImageTextToText as _ModelCls

    print(f"  florence: model={model_id}  device={device}  dtype={dtype_name}")

    # No trust_remote_code — community port ships native HF code.
    # `.to(device)` after `.from_pretrained()` is fine here; we don't
    # use device_map="auto" because we want hard, predictable placement
    # for the multi-process VRAM accounting in the orchestrator.
    model = _ModelCls.from_pretrained(model_id, **dtype_kwarg).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    # ------------------------------------------------------------------
    # torch.compile wrapping (Blackwell / Ada / Ampere kernel fusion).
    #
    # The compile MODE is resolved through `_resolve_compile_mode` so
    # the env-var override surface stays in one place. Defaults to
    # "default" (Inductor kernel fusion, no CUDA Graphs) because
    # Florence-2's autoregressive decode produces variable KV-cache
    # shapes per frame, which makes the CUDA-Graphs-backed
    # "reduce-overhead" mode thrash Dynamo's per-shape cache and
    # eventually deadlock on long batches (47-clip / 4700-frame runs
    # observed wedging at GPU=0% with VRAM held). "default" mode keeps
    # ~10-20% of the speedup with none of the deadlock risk.
    #
    # Power users with stable-shape footage can opt back into
    # "reduce-overhead" or "max-autotune" with:
    #   VIDEO_USE_FLORENCE_COMPILE_MODE=reduce-overhead
    #
    # `fullgraph=False` is mandatory regardless of mode — generate()
    # has a Python control flow loop and Dynamo will refuse to compile
    # it as one graph.
    #
    # Failure handling: any exception during compile (Dynamo crash,
    # backend init failure, OOM during specialization) is caught and
    # we fall back to the eager model. The compile attempt itself is
    # essentially free — `torch.compile` is lazy and only specializes
    # on first forward, so even the FAILED-compile path doesn't cost
    # warmup time, just the print statement.
    #
    # Override knobs:
    #   VIDEO_USE_FLORENCE_COMPILE=off       — disable compile entirely
    #   VIDEO_USE_FLORENCE_COMPILE_MODE=...  — pick scheduler mode
    # ------------------------------------------------------------------
    if compile_enabled and device.startswith("cuda"):
        compile_mode = _resolve_compile_mode()
        # Mode-specific warmup-cost messaging so the user knows whether
        # to expect a fast first batch ("default") or a long one
        # ("reduce-overhead" / "max-autotune" both capture CUDA Graphs).
        warmup_hint = (
            "~5-15s first-batch warmup"
            if compile_mode == "default"
            else "~30-60s first-batch warmup"
        )
        try:
            model = torch.compile(model, mode=compile_mode, fullgraph=False)
            print(f"  florence: torch.compile enabled "
                  f"(mode={compile_mode}, {warmup_hint})")
        except Exception as _e:
            print(f"  florence: torch.compile failed "
                  f"({type(_e).__name__}: {_e}); falling back to eager mode")

    return model, processor, torch_dtype


# ---------------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Florence-2 special-token leakage cleanup.
#
# `post_process_generation(<MORE_DETAILED_CAPTION>)` is supposed to peel the
# task header / control tokens off the decoded sequence and hand back a
# clean caption string. In practice — at least on the florence-community
# port we ship — it consistently leaves the autoregressive decoder's
# trailing pad run untouched whenever the actual caption finished short
# of `max_new_tokens=128`. The result is captions like
#
#   "a person holding a drill above a metal panel.<pad><pad><pad><pad>...
#    <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>"
#
# where the first 5-7 sentences are real content and the back half is
# pure tokenizer noise. On a 6500-frame timeline that ballooned the
# merged_timeline.md file by ~30-40% and pushed it past the agent's
# reasonable single-read budget. We strip them at decode
# (so new caches stay clean) AND defensively in pack_timelines.py
# (so legacy caches that still carry the leakage pack cleanly without
# a forced re-preprocess).
#
# The regex is broad on purpose — anything that looks like an HF-shape
# special token (`<pad>` / `<s>` / `</s>` / `<unk>` / `<mask>` / `<bos>`
# / `<eos>` / `<sep>` / `<cls>`) gets nuked. Florence's real caption
# text never legitimately contains those substrings.
# ---------------------------------------------------------------------------

_FLORENCE_SPECIAL_TOKEN_RE = re.compile(
    r"<\s*/?\s*(?:pad|s|unk|mask|bos|eos|sep|cls)\s*>",
    flags=re.IGNORECASE,
)
# Whitespace collapser used after the strip — repeated ` ` runs would
# otherwise survive where the deleted tokens used to live.
_WS_RUN_RE = re.compile(r"\s{2,}")


def _strip_florence_special_tokens(text: str) -> str:
    """Strip leftover Florence-2 special tokens from a decoded caption.

    Idempotent and cheap (~0.3 us per caption on a typical CPU). Safe to
    apply both at decode time inside this module and as a defensive pass
    inside pack_timelines.py for legacy caches.
    """
    if not text:
        return ""
    text = _FLORENCE_SPECIAL_TOKEN_RE.sub("", text)
    text = _WS_RUN_RE.sub(" ", text).strip()
    return text


def _caption_batch(model, processor, images, *, device: str, torch_dtype, task: str) -> list[str]:
    """Run Florence-2 on a list of frames, return parsed captions.

    `images` may be either a list of PIL.Image (legacy callers) or a
    list of numpy `(H, W, 3)` uint8 arrays (the streaming pipe path).
    Florence-2's processor accepts both shapes natively — internally it
    normalises everything to a torch tensor of `pixel_values` before
    feeding the vision tower.
    """
    import torch

    inputs = processor(
        text=[task] * len(images),
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device, dtype=torch_dtype)

    # input_ids must remain int — only the visual / float tensors get
    # promoted to fp16. Restore them after the .to() above moved everything.
    if "input_ids" in inputs:
        inputs["input_ids"] = inputs["input_ids"].long()
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].long()

    # ------------------------------------------------------------------
    # Decode strategy: greedy + tighter token cap.
    #
    # Florence-2's stock generate() defaults are tuned for human-readable
    # benchmark prose (COCO captions, Flickr30k) where beam search at b=3
    # buys the model a tiny BLEU lift by exploring 3 hypothesis sequences
    # in parallel. That costs us 3x decoder forward passes per token AND
    # 3x KV-cache memory. For OUR consumer — a Phase-B LLM editor that
    # parses captions for entities ("drill", "rivet panel"), actions
    # ("holding", "operating"), and shot composition ("close-up",
    # "wide") — the marginal prose-quality gain is invisible. The LLM
    # extracts the same structured facts from "a person holding a drill"
    # as it does from "a man operating a power tool", so we cash in the
    # ~2.5x decoder-time savings and ship greedy.
    #
    # max_new_tokens=128 (down from 256): MORE_DETAILED_CAPTION outputs
    # cluster around 40-60 tokens. The 256 ceiling only ever fires on
    # the ~1% of frames where the model gets stuck in a repetition loop
    # ("a man a man a man ...") — capping at 128 truncates those earlier
    # without affecting any well-formed caption. Net effect: lower P99
    # latency, marginally cleaner output on the pathological tail.
    #
    # If you ever need to revert for an A/B comparison: bump num_beams
    # back to 3 and max_new_tokens back to 256 — both args live right
    # here, no other code paths depend on them.
    # ------------------------------------------------------------------
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
        )

    raw_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
    out: list[str] = []
    for raw, img in zip(raw_texts, images):
        # post_process_generation needs (width, height) for region tasks
        # like <CAPTION_TO_PHRASE_GROUNDING>; for plain caption prompts
        # the value is unused but the kwarg is mandatory. Handle both
        # shapes:
        #   - PIL.Image  -> .width, .height
        #   - numpy ndarray (H, W, 3) -> .shape[1], .shape[0]
        # Falling through to (768, 768) covers any exotic future input.
        if hasattr(img, "width") and hasattr(img, "height"):
            img_size = (img.width, img.height)
        elif hasattr(img, "shape") and len(img.shape) >= 2:
            img_size = (int(img.shape[1]), int(img.shape[0]))
        else:
            img_size = (FLORENCE_INPUT_DIM, FLORENCE_INPUT_DIM)
        parsed = processor.post_process_generation(
            raw, task=task, image_size=img_size,
        )
        # post_process_generation returns {task: caption_str} for caption tasks.
        text = parsed.get(task, "") if isinstance(parsed, dict) else str(parsed)
        # Strip any trailing <pad>... run (and stray <s> / </s> markers)
        # that the post-processor failed to clean up — see the long
        # comment on _strip_florence_special_tokens above for why this
        # is necessary on the florence-community port.
        text = _strip_florence_special_tokens(str(text).strip())
        out.append(text)
    return out


# ---------------------------------------------------------------------------
# Dedup: collapse consecutive identical (or near-identical) captions to
# "(same)" markers in the markdown view. Saves ~30-50% on tokens for
# static / slow-moving footage. We keep all raw text in the JSON cache
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
# Producer / consumer streaming pipe (ffmpeg → bounded queue → Florence).
#
# Architecture:
#   The producer thread Popen()'s a single ffmpeg subprocess per clip
#   that streams 768×768 raw rgb24 frames out its stdout. Each frame
#   is exactly `target_dim * target_dim * 3` bytes — Python's
#   BufferedReader.read(n) blocks until exactly n bytes have arrived
#   or the pipe hits EOF, so we read in fixed chunks with no muxer
#   parsing or boundary sniffing required. Each chunk is wrapped as
#   a numpy view (zero copy until consume time) and pushed onto a
#   bounded queue. The consumer (Florence) drains batches off the
#   other end, infers, writes the captions, and DROPS the references
#   so the underlying bytes become collectable immediately.
#
# Why a bounded queue:
#   maxsize = batch_size * 2 caps in-flight RAM at ~108 MB peak for a
#   wealthy batch_size=32 (2 × 32 × 1.69 MB) and ~27 MB for the default
#   batch_size=8. Keeps the producer ~1 batch ahead of the consumer
#   without letting it run away on multi-thousand-frame shoots.
#
# Why a stop_event + sentinel-on-error:
#   On consumer-side exception (OOM, KeyboardInterrupt, model crash),
#   we need the producer to stop reading from ffmpeg cleanly AND tear
#   down the subprocess so we don't leak GPU NVDEC sessions or zombie
#   processes. stop_event + the producer's try/finally handles both
#   the cooperative-cancellation poll and the proc.terminate() cleanup.
#   The error sentinel ("__error__", exc) lets producer-side errors
#   (NVDEC blowout, codec mismatch, corrupt file) propagate up to the
#   consumer's main thread instead of dying silently in the background.
#
# NVDEC fallback:
#   If `_nvdec_available()` reported True we try the hardware path
#   first. If ffmpeg's exit code is non-zero AND we haven't yielded
#   any frames yet, the producer restarts the same clip with software
#   decode — covers NVDEC session-limit exhaustion, malformed muxers
#   (corrupt PPS / missing AV1 sequence header), and codec types the
#   NVDEC build doesn't support. Mid-stream NVDEC failures (very rare)
#   are surfaced as a producer error so we don't paper over partial
#   decodes.
# ---------------------------------------------------------------------------

# Internal sentinel objects. Module-level so identity comparison works
# across thread boundaries — never wrap them in a transient object.
_FRAME_END_SENTINEL = object()
_FRAME_ERROR_TAG = "__frame_producer_error__"


def _spawn_ffmpeg_pipe(
    video_path: Path,
    fps: float,
    target_dim: int,
    meta: dict,
    *,
    use_nvdec: bool,
):
    """Popen an ffmpeg subprocess that pipes rgb24 frames to stdout.

    Returns the live `subprocess.Popen` handle. The caller owns the
    process lifecycle — they MUST eventually `.wait()` on it (or
    `.terminate()` + `.wait()` on early exit) to avoid zombies.

    bufsize is set to one full frame's worth of bytes so Python's
    BufferedReader pre-allocates a sensibly-sized userspace buffer
    instead of the default 8 KB, which would force ~200 small reads
    per 1.7 MB frame. One contiguous read = one syscall = lower
    overhead and better cache behaviour on the consumer side.
    """
    cmd = _build_extract_cmd(
        video_path, target_dim, fps, meta, use_nvdec=use_nvdec,
    )
    frame_bytes = target_dim * target_dim * 3
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=frame_bytes,
    )


def _stream_frames_from_proc(
    proc: "subprocess.Popen",
    fps: float,
    target_dim: int,
    out_q: "queue.Queue",
    stop_event: threading.Event,
) -> tuple[int, str]:
    """Pump rgb24 chunks from `proc.stdout` onto `out_q` until EOF.

    Returns (n_frames_yielded, stderr_tail). Caller is expected to
    call proc.wait() afterwards and check returncode for failure.

    Each chunk is exactly `target_dim * target_dim * 3` bytes. A short
    read means ffmpeg closed stdout (normal EOF) — we exit cleanly.
    Any in-progress partial frame is discarded: a truncated rgb24 buffer
    would alias garbage pixels into Florence's input, which is worse
    than a missing caption at the tail.
    """
    import numpy as np
    frame_bytes = target_dim * target_dim * 3
    n_frames = 0
    try:
        while True:
            # Cooperative cancellation. Polled between every produced
            # frame; worst case the consumer waits one frame's worth of
            # decode (~5 ms NVDEC, ~30 ms software) for the producer
            # to exit. Cheaper than wedging a thread on a blocked read.
            if stop_event.is_set():
                break
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                # EOF or short read — ffmpeg is done. Discard partials.
                break
            # Wrap the bytes as an (H, W, 3) uint8 numpy view. No copy:
            # the array shares memory with `buf`, and `buf` stays alive
            # via the array's `.base` reference. Florence's processor
            # will copy when it builds the pixel_values tensor anyway.
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(
                target_dim, target_dim, 3
            )
            ts = n_frames / float(fps)
            # Block on `.put()` — the bounded queue is precisely what
            # implements backpressure. If Florence is slow we want
            # the producer to throttle, NOT build an unbounded backlog
            # of 1.7 MB rgb24 buffers in RAM.
            out_q.put((ts, arr))
            n_frames += 1
    finally:
        # Drain stderr non-destructively so the caller can include it
        # in error messages on non-zero exit. We tail the last 12 lines
        # (~1 KB) which is enough to identify the failure cause.
        stderr_tail = ""
        try:
            if proc.stderr is not None:
                stderr_tail = proc.stderr.read().decode("utf-8", "replace")
                stderr_tail = "\n".join(stderr_tail.splitlines()[-12:])
        except Exception:
            pass
    return n_frames, stderr_tail


def _frame_producer(
    video_path: Path,
    fps: float,
    target_dim: int,
    out_q: "queue.Queue",
    stop_event: threading.Event,
) -> None:
    """Stream rgb24 frames straight from a per-clip ffmpeg subprocess.

    Runs in its own thread. Always emits exactly one terminal item
    (either the end sentinel or an (`__error__`, exc) tuple) so the
    consumer's blocking `.get()` never deadlocks. Re-raises nothing —
    the consumer is responsible for re-raising the captured exception
    in the main thread so the surrounding `lane_progress` context
    unwinds cleanly.

    NVDEC-with-software-fallback: if hardware decode fails BEFORE
    any frame was yielded, we restart the same clip with software
    decode. This catches the common failure modes (NVDEC session
    exhaustion, codec not in the NVDEC build, malformed muxer) without
    masking real corruption — once we've yielded N frames the consumer
    has committed to those captions and a mid-stream codec switch
    would silently produce a discontinuous timeline.
    """
    meta = _probe_video_meta(video_path)
    nvdec_first = _nvdec_available()
    proc: subprocess.Popen | None = None
    n_yielded_total = 0
    try:
        # ── First attempt: NVDEC (if available) ───────────────────────
        proc = _spawn_ffmpeg_pipe(
            video_path, fps, target_dim, meta, use_nvdec=nvdec_first,
        )
        n_yielded, stderr_tail = _stream_frames_from_proc(
            proc, fps, target_dim, out_q, stop_event,
        )
        rc = proc.wait()
        n_yielded_total = n_yielded

        # ── Software fallback path ────────────────────────────────────
        # Only viable if (a) the hardware attempt failed AND (b) we
        # haven't committed any frames yet. Both conditions matter:
        # restarting mid-stream would skip frames the consumer has
        # already received, breaking the (idx -> ts) timeline contract.
        if rc != 0 and nvdec_first and n_yielded == 0 and not stop_event.is_set():
            print(f"  visual_lane: NVDEC failed for {video_path.name} "
                  f"({stderr_tail.strip()[:120] or 'no stderr'}); "
                  f"retrying software decode")
            proc = _spawn_ffmpeg_pipe(
                video_path, fps, target_dim, meta, use_nvdec=False,
            )
            n_yielded, stderr_tail = _stream_frames_from_proc(
                proc, fps, target_dim, out_q, stop_event,
            )
            rc = proc.wait()
            n_yielded_total = n_yielded

        if rc != 0 and not stop_event.is_set():
            raise RuntimeError(
                f"ffmpeg streaming decode failed for {video_path.name} "
                f"(rc={rc}): {stderr_tail.strip()[:500] or 'no stderr'}"
            )
    except BaseException as exc:  # noqa: BLE001 - we re-raise via sentinel
        # Any failure (codec init, NVDEC blowout, OS pipe error)
        # gets handed off as a tagged error sentinel so the consumer
        # can re-raise it in its own thread context.
        try:
            out_q.put((_FRAME_ERROR_TAG, exc))
        except Exception:
            pass
    finally:
        # Belt + suspenders subprocess teardown. If the consumer cancelled
        # mid-stream the ffmpeg process is probably still alive and
        # blocked on a stdout write — terminate() + wait() with a short
        # timeout cleans that up; kill() is the last-resort hammer for
        # processes that ignored SIGTERM.
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=2.0)
                except Exception:
                    pass
        # Always emit the end sentinel last — guarantees the consumer's
        # `while True: q.get()` loop terminates regardless of which path
        # we exited through.
        try:
            out_q.put(_FRAME_END_SENTINEL)
        except Exception:
            pass
        # Best-effort completion log so the user sees per-clip extract
        # stats inline with the inference progress. Mirrors the v2
        # disk-cache path's "extract: foo -> N frames" line.
        if n_yielded_total > 0:
            print(f"  visual_lane: streamed {n_yielded_total} frames from "
                  f"{video_path.name} (hdr={meta.get('is_hdr')}, "
                  f"nvdec={nvdec_first})")


# ---------------------------------------------------------------------------
# Main lane entry point
# ---------------------------------------------------------------------------

def _process_one(
    model,
    processor,
    torch_dtype,
    video_path: Path,
    edit_dir: Path,
    *,
    target_dim: int,
    model_id: str,
    fps: float,
    batch_size: int,
    device: str,
    task: str,
    force: bool,
) -> Path:
    """Caption one video by streaming frames straight from ffmpeg into Florence.

    The producer thread Popen()'s a single ffmpeg subprocess that pipes
    rgb24 frames out stdout; we batch them on this side and feed
    Florence as soon as the queue holds `batch_size` frames. Peak
    in-flight memory is bounded by the queue size (2 batches), not by
    clip length — a 1-hour clip uses the same ~50 MB working set as a
    30-second clip.

    Split out from run_visual_lane_batch so the batch entry point can
    amortize the ~3s Florence load + the torch.compile warmup across
    many videos in one Python process.
    """
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
    # `:g` is the right format for fps because it prints `1` instead of
    # `1.0` for integer values and `0.5` / `0.25` for fractional ones —
    # log lines stay clean across the full default-and-fractional range.
    print(f"  florence: {video_path.name}  duration={duration:.1f}s  "
          f"frames~{expected_frames} @ {fps:g}fps  batch={batch_size}")

    captions: list[dict] = []
    batch_imgs: list = []
    batch_ts: list[float] = []
    t0 = time.time()

    # ------------------------------------------------------------------
    # Spin up the streaming ffmpeg producer in a daemon thread. Bounded
    # queue = 2 batches deep so the producer stays one batch ahead of
    # the consumer at most — that's enough to overlap NVDEC decode with
    # Florence inference without letting RAM grow with clip length.
    # `daemon=True` is the belt + suspenders backup to the explicit
    # join() in the `finally` block below — if anything truly
    # catastrophic kills the consumer (segfault, etc.) the daemon flag
    # prevents the producer thread from blocking interpreter exit.
    # ------------------------------------------------------------------
    frame_q: "queue.Queue" = queue.Queue(maxsize=max(2, batch_size * 2))
    stop_event = threading.Event()
    producer = threading.Thread(
        target=_frame_producer,
        args=(video_path, fps, target_dim, frame_q, stop_event),
        name=f"florence-producer-{video_path.stem}",
        daemon=True,
    )
    producer.start()

    # Per-frame progress is genuinely informative here — Florence is the
    # slow lane, so users want to see frames-per-second crawl forward.
    # We tick once per BATCH (not per frame) to keep emit volume sane,
    # advancing by `len(batch)` each time.
    try:
        with lane_progress(
            "visual",
            total=expected_frames,
            unit="frame",
            desc=f"florence captions: {video_path.name}",
        ) as fbar:
            # Drain the queue until the producer signals end-of-stream
            # (or hands us back an exception). All batching logic stays
            # IDENTICAL to the pre-pipelining loop — only the frame
            # source changed (queue vs direct generator).
            while True:
                item = frame_q.get()

                # End-of-stream sentinel: producer is done, flush any
                # trailing partial batch and exit the drain loop.
                if item is _FRAME_END_SENTINEL:
                    break

                # Producer-side exception sentinel: re-raise in this
                # thread so the surrounding context manager + outer
                # try/finally see it as a normal exception.
                if isinstance(item, tuple) and len(item) == 2 and item[0] == _FRAME_ERROR_TAG:
                    raise item[1]

                ts, img = item
                batch_imgs.append(img)
                batch_ts.append(ts)
                if len(batch_imgs) >= batch_size:
                    texts = _caption_batch(
                        model, processor, batch_imgs,
                        device=device, torch_dtype=torch_dtype, task=task,
                    )
                    for tt, txt in zip(batch_ts, texts):
                        captions.append({"t": tt, "text": txt})
                    fbar.update(advance=len(batch_imgs))
                    batch_imgs.clear()
                    batch_ts.clear()

            # Flush trailing partial batch (only reached on normal
            # end-of-stream; an exception jumps straight to `finally`).
            if batch_imgs:
                texts = _caption_batch(
                    model, processor, batch_imgs,
                    device=device, torch_dtype=torch_dtype, task=task,
                )
                for tt, txt in zip(batch_ts, texts):
                    captions.append({"t": tt, "text": txt})
                fbar.update(advance=len(batch_imgs))
    finally:
        # ------------------------------------------------------------------
        # Producer cleanup. Belt + suspenders + a third belt:
        #   1) stop_event tells the producer to bail at its next poll AND
        #      triggers proc.terminate() inside the producer's finally
        #      block so we don't leak an ffmpeg subprocess.
        #   2) drain remaining queue items so a blocked .put() can return
        #      (the producer thread might be stuck inside .put() waiting
        #       for queue space if we exited mid-iteration).
        #   3) join with a generous timeout — ffmpeg terminate+wait is
        #      capped at 2s inside the producer, so 10s here is plenty
        #      of slack.
        # ------------------------------------------------------------------
        stop_event.set()
        try:
            while not frame_q.empty():
                try:
                    frame_q.get_nowait()
                except queue.Empty:
                    break
        except Exception:
            pass
        producer.join(timeout=10.0)
        # Drop any lingering numpy frame refs early — caption_batch
        # already copied them into the pixel_values tensor on GPU and
        # the producer's queue is now drained.
        batch_imgs.clear()

    dt = time.time() - t0

    captions_md = _dedup_consecutive(captions)
    payload = {
        "model": model_id,
        "task": task,
        "fps": fps,
        "duration": round(duration, 3),
        "captions": captions,            # raw
        "captions_dedup": captions_md,   # display copy
    }
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)

    rate = len(captions) / max(1e-3, dt)
    print(f"  visual_lane done: {len(captions)} captions, {dt:.1f}s wall "
          f"({rate:.1f} fps) → {out_path.name}")
    return out_path


def _resolve_compile_enabled(num_videos: int) -> bool:
    """Resolve the torch.compile knob from env var + sensible auto rule.

    Resolution order:
        1. VIDEO_USE_FLORENCE_COMPILE env var (explicit override):
             "on" / "1" / "true" / "yes"  -> compile ON
             "off" / "0" / "false" / "no" -> compile OFF
             anything else (or unset)     -> "auto" (rule below)

        2. Auto rule: ON when batching across multiple videos, OFF for
           single-clip runs. Compile's first-batch warmup (~30-60s on
           Florence-2-base) is dead time we'd never recover on a single
           short clip; on a multi-clip batch it amortizes nicely across
           every subsequent clip's frames.

    The standalone CLI's `--force-compile` / `--no-compile` flags both
    write into this env var, so this single function is the canonical
    resolver — no policy logic duplicated across entry points.
    """
    raw = os.environ.get("VIDEO_USE_FLORENCE_COMPILE", "").strip().lower()
    if raw in {"on", "1", "true", "yes", "y", "t"}:
        return True
    if raw in {"off", "0", "false", "no", "n", "f"}:
        return False
    return num_videos > 1


# Valid torch.compile mode strings. Kept in one place so both the
# resolver below and any future help-text generation stay in sync with
# the upstream PyTorch API. "default" is intentionally first — it's the
# safe pick for autoregressive models with variable decode lengths.
_VALID_COMPILE_MODES: tuple[str, ...] = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)


def _resolve_compile_mode() -> str:
    """Resolve the torch.compile *mode* string from an env var.

    Why this exists as its own resolver:
        `_resolve_compile_enabled` answers "should we compile at all";
        this one answers "if we DO compile, with which scheduler".
        Splitting the two knobs lets users force a mode without also
        having to think about the on/off question, and keeps the
        env-var surface symmetric (one var per decision).

    Resolution order:
        1. VIDEO_USE_FLORENCE_COMPILE_MODE env var, normalised to lower.
           Hyphens / underscores are interchangeable so users don't
           have to remember which separator the upstream API uses
           (Inductor accepts "reduce-overhead", `torch.compile` accepts
           "reduce_overhead" — we feed the canonical hyphenated form).
        2. Default: "default".

    Why "default" is the safe default (changed from "reduce-overhead"):
        `reduce-overhead` uses CUDA Graphs which capture memory
        addresses statically. Florence-2's autoregressive decode
        produces a different output sequence length per frame, which
        means a different KV-cache shape per generate() call, which
        means Dynamo treats every novel shape as a fresh
        specialization. Each specialization needs a fresh CUDA Graph
        capture (~1-3 min) AND pins ~50-200 MB of VRAM. On long batches
        (47-clip / 4700-frame runs) this thrashes Dynamo's per-code-
        object cache (default size 8) and eventually wedges the
        process: GPU at 0%, VRAM held, no log output, no progress.
        See the long-form root-cause notes in `_resolve_compile_enabled`.

        `default` mode keeps Inductor's kernel fusion (still a real
        speedup, ~10-20%) but skips the CUDA-Graph capture step, so
        recompiles are 5-10× faster, no memory pinning, no Case-C
        deadlocks. Net: slightly less peak speed, dramatically more
        reliable on real-world variable-content batches.

    Power users with uniform shape footage (e.g. all-talking-head, one
    resolution, narrow caption length distribution) can opt back into
    `reduce-overhead` for the extra 10-15% with:

        VIDEO_USE_FLORENCE_COMPILE_MODE=reduce-overhead
    """
    raw = os.environ.get("VIDEO_USE_FLORENCE_COMPILE_MODE", "").strip().lower()
    if not raw:
        return "default"
    # Accept both hyphen and underscore separators for ergonomics.
    canon = raw.replace("_", "-")
    if canon in _VALID_COMPILE_MODES:
        return canon
    # Unknown mode → fall back to the safe default rather than crashing
    # late inside torch.compile with a less-helpful error. Print so the
    # user sees their typo got silently corrected.
    print(f"  florence: unknown VIDEO_USE_FLORENCE_COMPILE_MODE='{raw}'; "
          f"valid={list(_VALID_COMPILE_MODES)}; falling back to 'default'")
    return "default"


def run_visual_lane_batch(
    video_paths: list[Path],
    edit_dir: Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    fps: float = DEFAULT_FPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cuda:0",
    dtype_name: str = "auto",
    task: str = DEFAULT_TASK_PROMPT,
    force: bool = False,
) -> list[Path]:
    """Run the visual lane on N videos with Florence-2 loaded ONCE.

    Streaming pipe architecture: Florence loads first (primed and
    hungry), then for each clip we spawn ONE ffmpeg subprocess that
    pipes 768x768 rgb24 frames straight into a bounded queue. Florence
    drains batches of `batch_size` frames, captions them, drops the
    references, and waits for the next batch. Peak in-flight memory
    is ~batch_size×2×1.7 MB regardless of clip length — a 1-hour clip
    uses the same working set as a 30-second clip.

    `fps` is float so callers can pass fractional values (e.g. 0.5 for
    one frame every 2 s) for slow / static content where 1 fps is
    over-sampling. Both the streaming extract and the JSON
    timestamps honour the fractional value.
    """
    out_dir = (edit_dir / VISUAL_CAPS_SUBDIR).resolve()
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
            print(f"  visual_lane: all {len(video_paths)} cache hits, skipping model load")
            return [out_dir / f"{v.stem}.json" for v in video_paths]

    # ------------------------------------------------------------------
    # Filter out clips that already have caption JSON. No point spinning
    # up ffmpeg or invoking Florence for clips whose captions are already
    # fresher than their source. The filtered list drives Florence; the
    # original `video_paths` order is preserved in the return value.
    # ------------------------------------------------------------------
    work_videos: list[Path] = []
    for v in video_paths:
        out = out_dir / f"{v.stem}.json"
        if not force and out.exists():
            try:
                if out.stat().st_mtime >= v.stat().st_mtime:
                    continue
            except OSError:
                pass
        work_videos.append(v)

    # Compile decision is taken here (not inside _build_florence) because
    # it depends on the BATCH SHAPE — we want to enable it whenever the
    # warmup amortizes, which is precisely "more than one clip in this
    # invocation". The env var override still lets a user force one way
    # or the other regardless of clip count.
    compile_enabled = _resolve_compile_enabled(len(work_videos))

    if work_videos:
        print(f"  visual_lane: streaming {len(work_videos)} clip(s) "
              f"(nvdec={_nvdec_available()}, batch={batch_size}, "
              f"fps={fps:g})")

    out_paths: list[Path] = []

    # ------------------------------------------------------------------
    # Build Florence FIRST so it's primed and hungry by the time the
    # first frame arrives. ffmpeg+NVDEC produces 768² rgb24 frames at
    # >200 fps — Florence is universally the bottleneck — so there's
    # nothing to gain by overlapping decode with model load. Better
    # to know the model is sane (CUDA OOM, dtype mismatch, missing
    # weights) BEFORE we start burning subprocess time.
    # ------------------------------------------------------------------
    model = processor = torch_dtype = None
    try:
        if work_videos:
            model, processor, torch_dtype = _build_florence(
                model_id, device, dtype_name, compile_enabled=compile_enabled,
            )

        # Outer bar tracks video-of-N progress; inner per-frame bar
        # (in _process_one) tracks current-video frame progress. Both
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

                # Cache-hit fast path: never made it onto work_videos,
                # so no ffmpeg, no model invocation. Just return the
                # existing JSON path.
                cached_path = out_dir / f"{v.stem}.json"
                if v not in work_videos:
                    out_paths.append(cached_path)
                    vbar.update(advance=1, item=v.name)
                    continue

                # _process_one spawns the ffmpeg producer thread for
                # THIS clip and runs the streaming consumer loop. On
                # producer-side failure (codec init, NVDEC blowout)
                # the exception propagates here; we log and skip the
                # clip rather than poisoning the whole batch.
                try:
                    out_paths.append(_process_one(
                        model, processor, torch_dtype, v, edit_dir,
                        target_dim=FLORENCE_INPUT_DIM,
                        model_id=model_id, fps=fps, batch_size=batch_size,
                        device=device, task=task, force=force,
                    ))
                except Exception as e:
                    print(f"  visual_lane: streaming failed for {v.name}: {e}")
                vbar.update(advance=1, item=v.name)
    finally:
        try:
            import torch
            if model is not None:
                del model
            if processor is not None:
                del processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
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
        description="Visual lane: Florence-2 captions at N fps",
    )
    ap.add_argument("video", type=Path, help="Path to source video file")
    ap.add_argument(
        "--edit-dir", type=Path, default=None,
        help="Edit output dir (default: <video parent>/edit)",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL_ID,
                    help=f"HF model id (default: {DEFAULT_MODEL_ID})")
    ap.add_argument(
        "--fps", type=float, default=DEFAULT_FPS,
        help=(
            f"Sample rate in frames/sec (default: {DEFAULT_FPS:g}). "
            "Fractional values are accepted: 0.5 = one frame every 2 s, "
            "0.25 = one frame every 4 s, etc. Lower fps means a faster "
            "preprocess and a leaner merged_timeline.md, at the cost of "
            "coarser temporal resolution. Linear cost — halving fps "
            "roughly halves preprocess wall time."
        ),
    )
    ap.add_argument("--batch-size", type=int, default=None,
                    help=f"Inference batch size (default: {DEFAULT_BATCH_SIZE}, "
                         f"or {FLORENCE_BATCH} with --wealthy)")
    ap.add_argument("--wealthy", action="store_true",
                    help="Speed knob for 24GB+ cards (4090/5090). Bigger batch, "
                         "same model + outputs. Also reads VIDEO_USE_WEALTHY=1.")
    ap.add_argument("--device", default="cuda:0",
                    help="Torch device: cuda:0, mps, cpu (default: cuda:0)")
    # `auto` (the new default) picks bf16 on Ampere+ CUDA cards (RTX
    # 30/40/50, A100, H100) and fp16 elsewhere. Old explicit choices
    # (fp16/bf16/fp32) still work as direct overrides.
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "fp16", "fp32", "bf16"],
                    help="Model dtype. 'auto' picks bf16 on Ampere+ CUDA, "
                         "fp16 elsewhere (default: auto)")
    ap.add_argument("--task", default=DEFAULT_TASK_PROMPT,
                    help="Florence task prompt (default: <MORE_DETAILED_CAPTION>)")
    ap.add_argument("--force", action="store_true",
                    help="Bypass cache, always re-caption.")
    # ── torch.compile flags ──────────────────────────────────────────
    # These two flags are mutually exclusive — argparse enforces it. They
    # both write into the same env var that `_resolve_compile_enabled`
    # reads, so even if the user invokes _build_florence directly from
    # an embedded script, the flag still takes effect. Default behaviour
    # (neither flag set) leaves the env var alone, so the auto rule
    # (compile when batching > 1 clip) applies.
    compile_group = ap.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--force-compile", action="store_true",
        help="Force-enable torch.compile for Florence-2. Pays a "
             "~30-60s warmup but yields ~20-30%% steady-state speedup "
             "on Ampere+ CUDA. Overrides the auto-on-multi-clip rule.")
    compile_group.add_argument(
        "--no-compile", action="store_true",
        help="Disable torch.compile entirely. Useful if you hit a "
             "compile / Dynamo regression on your setup.")
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

    # ------------------------------------------------------------------
    # Wire the compile flags through the env var that
    # `_resolve_compile_enabled` reads. Setting the env var (rather than
    # plumbing yet another kwarg through run_visual_lane[_batch]) keeps
    # the resolver as the single source of truth for compile policy and
    # means the flag also takes effect for any code path that imports
    # `run_visual_lane_batch` directly. We only WRITE the env var when
    # the user passed an explicit flag — the default (no flag) leaves
    # whatever the orchestrator / parent shell already set in place,
    # which then falls through to the "auto" branch.
    # ------------------------------------------------------------------
    if args.force_compile:
        os.environ["VIDEO_USE_FLORENCE_COMPILE"] = "on"
    elif args.no_compile:
        os.environ["VIDEO_USE_FLORENCE_COMPILE"] = "off"

    run_visual_lane(
        video_path=video,
        edit_dir=edit_dir,
        model_id=args.model,
        fps=args.fps,
        batch_size=batch_size,
        device=args.device,
        dtype_name=args.dtype,
        task=args.task,
        force=args.force,
    )


if __name__ == "__main__":
    main()
