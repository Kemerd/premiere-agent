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
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
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
# Hardware-accelerated bulk frame prefetch.
#
# The legacy `_iter_frames_at_fps` above pipes raw RGB frames through a
# Popen() pipe and feeds Florence one batch at a time. That worked fine
# on tutorial / talking-head footage where ffmpeg decode was cheap. On
# 4K 60fps HDR DJI footage it falls over hard:
#
#   - Software h264 decode of 4K 60fps is CPU-bound at ~150-300 ms per
#     decoded frame on consumer CPUs. Even with `fps=1` decimation
#     applied AFTER decode, ffmpeg still has to walk the GOP and decode
#     enough frames to hit each second mark.
#   - HDR (HLG / PQ) → SDR tonemap is mandatory: Florence-2 was trained
#     on SDR Rec.709 imagery; feeding it raw HLG yields washed-out
#     desaturated captions because the colour primaries land outside
#     its training distribution.
#   - In-process pipelining (the producer thread we already have)
#     overlaps decode-of-batch-N with infer-of-batch-N-minus-1, but
#     CAN'T overlap one CLIP'S decode with another's inference because
#     ffmpeg decode is bound to the lifetime of the per-clip generator.
#
# The fix is two-fold:
#
#   1. Move ffmpeg to NVIDIA's NVDEC (`-hwaccel cuda`) so 4K H.264 / H.265
#      decode runs on the GPU's dedicated decode block — independent of
#      the CUDA SMs that Florence is using, no contention. NVDEC on
#      Blackwell does 4K60 H.265 in real time at ~1% SM cost.
#   2. Bulk-extract ALL clips' frames to disk concurrently in a
#      ThreadPoolExecutor BEFORE Florence starts inferencing. Each
#      worker is a separate ffmpeg subprocess pinned to its own clip,
#      so N workers = N parallel decoders saturating the IO + decode
#      blocks while Florence eats the per-clip results in order.
#
# Frame cache layout:
#
#   <edit_dir>/visual_caps/_frame_cache/
#     ├── DJI_..._0303_D/
#     │   ├── 000001.jpg
#     │   ├── 000002.jpg
#     │   └── .done             # sentinel: extraction completed
#     └── DJI_..._0304_D/
#         └── ...
#
# JPEG quality 3 (mjpeg `-q:v 3`, 1=best 31=worst) gives ~150-250 KB
# per 768x768 frame and is visually lossless for captioning targets.
# A 14-clip × ~150 frame-avg shoot = ~2100 frames × 200 KB = ~420 MB
# disk. Trivial. The cache is auto-deleted at end-of-run unless the
# user sets `VIDEO_USE_FRAME_CACHE_KEEP=1` for debugging.
#
# HDR detection: ffprobe `color_transfer`. The two HDR transfers in
# the wild are `arib-std-b67` (HLG, what DJI uses) and `smpte2084`
# (PQ, what Sony / Apple use). Anything else (`bt709`, blank, etc.)
# is treated as SDR and skips the tonemap chain.
# ---------------------------------------------------------------------------

# Subdirectory under visual_caps for pre-extracted frame JPEGs.
FRAME_CACHE_SUBDIR = "_frame_cache"

# Extraction worker pool size. Capped low to avoid CPU thrash + NVDEC
# session limit on consumer cards (RTX 3-series / 4-series cap at 3
# concurrent NVDEC sessions; 5090 lifts that to 5+). Override with
# VIDEO_USE_FRAME_EXTRACT_WORKERS=N if you know your card.
DEFAULT_EXTRACT_WORKERS = 4

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
    out_dir: Path,
    target_dim: int,
    fps: float,
    meta: dict,
    *,
    use_nvdec: bool,
) -> list[str]:
    """Construct an ffmpeg argv that extracts JPEGs into out_dir.

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

    NVDEC path:
        Hardware-decodes the H.264 / H.265 / AV1 stream on the dedicated
        NVDEC block, then `hwdownload,format=nv12` brings the frame to
        system RAM for the SDR conversion + crop + scale. This sounds
        wasteful but the download is ~1 MB/frame at 1 fps which is
        nothing, and CUDA filters (`scale_cuda` etc.) don't compose
        cleanly with `tonemap` so we'd need the download anyway on
        HDR sources.
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

    base = [ffmpeg_bin, "-loglevel", "error", "-y"]
    if use_nvdec:
        # `-hwaccel cuda` decodes on NVDEC; we deliberately do NOT pass
        # `-hwaccel_output_format cuda` because tonemap + zscale don't
        # exist on the GPU pipeline — letting ffmpeg auto-download to
        # system RAM after decode keeps the filter graph happy on both
        # SDR and HDR paths with one code path.
        base += ["-hwaccel", "cuda"]
    base += ["-i", str(video_path), "-vf", vf]
    # MJPEG quality. -q:v 3 is visually lossless at 768x768 (~200 KB/frame)
    # and decodes in <5 ms via PIL. Tried PNG (lossless, ~2-5 MB/frame,
    # ~30 ms decode) — total wall time was worse despite no quality loss.
    base += ["-q:v", "3", str(out_dir / "%06d.jpg")]
    return base


def _extract_frames_to_disk(
    video_path: Path,
    cache_dir: Path,
    fps: float,
    target_dim: int = 768,
) -> Path:
    """Extract all frames-of-interest for one clip into cache_dir.

    Idempotent: if `<cache_dir>/.done` exists we skip the ffmpeg call
    entirely. On NVDEC failure we automatically retry with software
    decode — some malformed muxers (corrupted PPS, AV1 with missing
    sequence header) can choke the hardware path while software handles
    them fine.

    Returns the cache_dir path so the caller can pass it directly into
    the consumer iterator.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    sentinel = cache_dir / ".done"
    if sentinel.exists():
        try:
            n_existing = sum(1 for _ in cache_dir.glob("*.jpg"))
        except Exception:
            n_existing = -1
        print(f"  extract: cache hit {video_path.name} ({n_existing} frames)")
        return cache_dir

    # Wipe any partial extraction from a prior crashed run. We can't
    # trust frame counts without the .done sentinel.
    for stale in cache_dir.glob("*.jpg"):
        try:
            stale.unlink()
        except OSError:
            pass

    meta = _probe_video_meta(video_path)
    nvdec_first = _nvdec_available()

    t0 = time.time()
    cmd = _build_extract_cmd(
        video_path, cache_dir, target_dim, fps, meta, use_nvdec=nvdec_first,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0 and nvdec_first:
        # NVDEC retry-with-software fallback. Common causes: corrupt
        # SEI, NVDEC session-limit exhaustion under heavy parallel
        # extract load, or codec the NVDEC build doesn't support.
        print(f"  extract: NVDEC failed for {video_path.name} "
              f"({proc.stderr.strip()[:120]}); retrying software")
        # Wipe any partial frames the NVDEC attempt may have written.
        for stale in cache_dir.glob("*.jpg"):
            try:
                stale.unlink()
            except OSError:
                pass
        cmd = _build_extract_cmd(
            video_path, cache_dir, target_dim, fps, meta, use_nvdec=False,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg extract failed for {video_path.name}: "
            f"{proc.stderr.strip()[:500]}"
        )

    # Atomic-ish completion marker. Written LAST so a SIGKILL mid-extract
    # leaves the dir in a "no .done -> redo from scratch" state instead
    # of "looks-done-but-truncated".
    sentinel.write_text("done")
    n = sum(1 for _ in cache_dir.glob("*.jpg"))
    print(f"  extract: {video_path.name} -> {n} frames in {time.time()-t0:.1f}s "
          f"(hdr={meta['is_hdr']}, nvdec={nvdec_first})")
    return cache_dir


def _iter_frames_from_dir(cache_dir: Path, fps: float):
    """Yield (timestamp_s, PIL.Image) for each pre-extracted JPEG.

    Drop-in replacement for `_iter_frames_at_fps` — same yield shape so
    the producer thread + batching loop don't need to know which source
    they're being fed from. Loads frames in filename order; ffmpeg's
    `%06d.jpg` template gives 1-indexed sequential numbering. We
    convert that frame index back to a float timestamp in seconds via
    `(idx - 1) / fps`, so:

        fps = 1.0   →  ts = 0, 1, 2, 3, ...   (one frame per second)
        fps = 0.5   →  ts = 0, 2, 4, 6, ...   (one frame every 2 s)
        fps = 0.25  →  ts = 0, 4, 8, 12, ...  (one frame every 4 s)
        fps = 2.0   →  ts = 0, 0.5, 1.0, ...  (sub-second precision)

    The float timestamp is preserved through the JSON cache so any
    downstream consumer (pack_timelines, build_srt, editor sub-agents)
    can render or align against the true sample time.
    """
    from PIL import Image
    # Guard against accidental fps=0 (would zero-divide); fall back to
    # 1.0 with a warning rather than crashing the entire batch.
    if not fps or fps <= 0:
        print(f"  visual_lane: invalid fps={fps!r} for {cache_dir.name}, "
              f"falling back to 1.0 for timestamp math")
        fps = 1.0
    for jpg in sorted(cache_dir.glob("*.jpg")):
        try:
            idx = int(jpg.stem)  # ffmpeg's %06d is 1-indexed
        except ValueError:
            continue
        ts = (idx - 1) / float(fps)
        # `with` + `.load()` forces the JPEG to fully decode now so we
        # can release the file handle before yielding. Otherwise PIL
        # keeps the file open lazily and we leak descriptors at scale.
        with Image.open(jpg) as fh:
            fh.load()
            img = fh.copy()
        yield ts, img


def _resolve_extract_workers(num_videos: int) -> int:
    """How many parallel ffmpeg extractor processes to spawn.

    Resolution order:
        1. VIDEO_USE_FRAME_EXTRACT_WORKERS env var (explicit override)
        2. min(DEFAULT_EXTRACT_WORKERS, num_videos) — never spin up
           more workers than there is work to do.

    The default ceiling (4) is the comfortable simultaneous-NVDEC limit
    on consumer Blackwell / Ada (5090 / 4090). Older cards (3090 and
    earlier) cap at 3 concurrent NVDEC sessions; even at the default 4
    the auto-fallback to software inside _extract_frames_to_disk handles
    overflow without crashing.
    """
    raw = os.environ.get("VIDEO_USE_FRAME_EXTRACT_WORKERS", "").strip()
    if raw:
        try:
            n = int(raw)
            if n >= 1:
                return min(n, max(1, num_videos))
        except ValueError:
            pass
    return min(DEFAULT_EXTRACT_WORKERS, max(1, num_videos))


def _safe_rmtree(path: Path) -> None:
    """Best-effort recursive delete. Never raises.

    Used for frame-cache cleanup at end-of-run. We deliberately swallow
    every exception class because cleanup failure (e.g. file locked by
    Windows Defender mid-scan) should not poison a successful run's
    return code.
    """
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


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
# audiovisual_timeline.md file by ~30-40% and pushed it past the editor
# sub-agent's reasonable single-read budget. We strip them at decode
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
    """Run Florence-2 on a list of PIL images, return parsed captions."""
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
        parsed = processor.post_process_generation(
            raw, task=task, image_size=(img.width, img.height),
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
# Producer / consumer pipelining for disk-cache JPEG decode <-> Florence
# inference.
#
# Why bother:
#   Even though all frames are pre-extracted to disk by the prefetch
#   pool (see _extract_frames_to_disk + _prefetch_all_frames below),
#   the consumer still has to (a) read the JPEG bytes, (b) decode them
#   to RGB via PIL/libjpeg-turbo (~3-8 ms per 768x768 frame), and
#   (c) marshall the resulting PIL.Image into a list ready for the
#   processor. At batch=32 that's ~100-250 ms of pure CPU work per
#   batch that would otherwise serialize behind generate(). One
#   producer thread overlaps it cleanly with Florence inference.
#
# Why a bounded queue:
#   maxsize = batch_size * 2 caps RAM at ~108 MB peak (2 batches of
#   32 frames × 768×768×3 RGB uint8 = 56 MB each + PIL header overhead).
#   Keeps the producer ~1 batch ahead of the consumer without letting it
#   run away on multi-thousand-frame shoots.
#
# Why a stop_event + sentinel-on-error:
#   On consumer-side exception (OOM, KeyboardInterrupt, model crash), we
#   need the producer to stop reading from disk cleanly. stop_event +
#   the producer's own try/finally guarantees pending file handles are
#   closed even on the abrupt exit path. The error sentinel
#   ("__error__", exc) lets producer-side errors (corrupt JPEG, missing
#   file) propagate up to the consumer's main thread instead of dying
#   silently in the background.
# ---------------------------------------------------------------------------

# Internal sentinel objects. Module-level so identity comparison works
# across thread boundaries — never wrap them in a transient object.
_FRAME_END_SENTINEL = object()
_FRAME_ERROR_TAG = "__frame_producer_error__"


def _frame_producer(
    cache_dir: Path,
    fps: float,
    out_q: "queue.Queue",
    stop_event: threading.Event,
) -> None:
    """Drain pre-extracted JPEG frames into `out_q` until done or signalled.

    Runs in its own thread. Always emits exactly one terminal item
    (either the sentinel or an ("__error__", exc) tuple) so the consumer's
    blocking `.get()` never deadlocks. Re-raises nothing — the consumer
    is responsible for re-raising the captured exception in the main
    thread so the surrounding `lane_progress` context unwinds cleanly.
    """
    try:
        for ts, img in _iter_frames_from_dir(cache_dir, fps):
            # Cooperative cancellation. Polled between every produced
            # frame; in the worst case the consumer waits one JPEG
            # decode (~5-10 ms) for the producer to exit.
            if stop_event.is_set():
                break
            # Block on `.put()` — the bounded queue is precisely what
            # implements backpressure. If the consumer is slow we want
            # the producer to throttle, NOT build an unbounded backlog
            # of decoded PIL images in RAM.
            out_q.put((ts, img))
    except BaseException as exc:  # noqa: BLE001 - we re-raise via sentinel
        # Any failure (corrupt JPEG, missing file mid-iteration, etc.)
        # gets handed off as a tagged error sentinel so the consumer
        # can re-raise it in its own thread context.
        try:
            out_q.put((_FRAME_ERROR_TAG, exc))
        except Exception:
            pass
    finally:
        # Always emit the end sentinel last — guarantees the consumer's
        # `while True: q.get()` loop terminates regardless of which path
        # we exited through.
        try:
            out_q.put(_FRAME_END_SENTINEL)
        except Exception:
            pass


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
    cache_dir: Path,
    model_id: str,
    fps: float,
    batch_size: int,
    device: str,
    task: str,
    force: bool,
) -> Path:
    """Caption one video with already-built Florence model + processor.

    Frame source is the pre-extracted JPEG cache produced by
    _extract_frames_to_disk — by the time we get here, the prefetch pool
    has already done all the ffmpeg / NVDEC / tonemap heavy-lifting and
    we just stream JPEGs off disk. See _prefetch_all_frames for the
    overlap with model load.

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
    batch_ts: list[int] = []
    t0 = time.time()

    # ------------------------------------------------------------------
    # Spin up the JPEG-decode producer in a daemon thread. Bounded queue
    # = 2 batches deep so the producer stays one batch ahead of the
    # consumer at most. `daemon=True` is the belt + suspenders backup
    # to the explicit join() in the `finally` block below — if anything
    # truly catastrophic kills the consumer (segfault, etc.) the daemon
    # flag prevents the producer thread from blocking interpreter exit.
    # ------------------------------------------------------------------
    frame_q: "queue.Queue" = queue.Queue(maxsize=max(2, batch_size * 2))
    stop_event = threading.Event()
    producer = threading.Thread(
        target=_frame_producer,
        args=(cache_dir, fps, frame_q, stop_event),
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
        #   1) stop_event tells the producer to bail at its next poll.
        #   2) drain remaining queue items so a blocked .put() can return
        #      (the producer thread might be stuck inside .put() waiting
        #       for queue space if we exited mid-iteration).
        #   3) join with a generous timeout — JPEG decode finishes
        #      quickly (~5-10 ms in flight) so the join is essentially
        #      instant in the happy path.
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
        # Drop any lingering PIL refs early — caption_batch already moved
        # them to GPU and the producer's queue is now drained.
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

    `fps` is float so callers can pass fractional values (e.g. 0.5 for
    one frame every 2 s) for slow / static content where 1 fps is
    over-sampling. Both the on-disk extract step and the JSON
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
    # Filter out clips that already have caption JSON. We don't want to
    # waste prefetch + Florence cycles on cache hits, AND we want to
    # keep the per-clip cache_dir mapping tight so the prefetch pool
    # only touches what it'll actually feed. The filtered list is what
    # we drive Florence with; the original `video_paths` is what we
    # return at the end so caller order is preserved.
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

    # ------------------------------------------------------------------
    # Kick off the bulk frame prefetch BEFORE the model load. This is
    # the whole point of the disk-cache architecture: while transformers
    # is downloading + materializing the Florence weights (~3-8s cold,
    # ~1-2s warm), N parallel ffmpeg / NVDEC processes are already
    # chewing through 4K HDR source files in the background. By the
    # time generate() wants its first batch, the JPEGs are usually
    # already on disk.
    # ------------------------------------------------------------------
    cache_root = (out_dir / FRAME_CACHE_SUBDIR).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    extract_workers = _resolve_extract_workers(len(work_videos))
    extract_pool: ThreadPoolExecutor | None = None
    extract_futures: dict[Path, "Future[Path]"] = {}
    if work_videos:
        print(f"  prefetch: starting {extract_workers} ffmpeg workers "
              f"for {len(work_videos)} clip(s) "
              f"(nvdec={_nvdec_available()})")
        extract_pool = ThreadPoolExecutor(
            max_workers=extract_workers,
            thread_name_prefix="frame-extract",
        )
        for v in work_videos:
            cache_dir = cache_root / v.stem
            extract_futures[v] = extract_pool.submit(
                _extract_frames_to_disk, v, cache_dir, fps,
            )

    out_paths: list[Path] = []

    # Build Florence in parallel with the running prefetch pool. The
    # GIL doesn't block here because both ffmpeg and the model download
    # release it during their native I/O.
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
                # so no prefetch future, no model invocation. Just
                # return the existing JSON path.
                cached_path = out_dir / f"{v.stem}.json"
                if v not in extract_futures:
                    out_paths.append(cached_path)
                    vbar.update(advance=1, item=v.name)
                    continue

                # Block on the prefetch future for THIS clip. If it's
                # already done (very likely when the model load took
                # longer than the extract), this returns instantly. If
                # not, we wait — but the GPU was idle during model load
                # anyway, so the wall-clock ceiling is just the longest
                # single-clip extract.
                try:
                    cache_dir = extract_futures[v].result()
                except Exception as e:
                    # Prefetch failed for this clip; surface clearly
                    # and skip it rather than poisoning the whole batch.
                    print(f"  visual_lane: prefetch failed for {v.name}: {e}")
                    vbar.update(advance=1, item=v.name)
                    continue

                out_paths.append(_process_one(
                    model, processor, torch_dtype, v, edit_dir,
                    cache_dir=cache_dir,
                    model_id=model_id, fps=fps, batch_size=batch_size,
                    device=device, task=task, force=force,
                ))
                vbar.update(advance=1, item=v.name)
    finally:
        # Tear down the prefetch pool. shutdown(wait=True) lets any
        # in-flight extractions finish so we don't leave half-written
        # JPEG dirs behind on early exit.
        if extract_pool is not None:
            try:
                extract_pool.shutdown(wait=True)
            except Exception:
                pass

        # Optional disk-cache cleanup. Keeps zero state by default;
        # set VIDEO_USE_FRAME_CACHE_KEEP=1 to retain JPEGs for debug.
        keep_raw = os.environ.get("VIDEO_USE_FRAME_CACHE_KEEP", "").strip().lower()
        if keep_raw not in {"1", "true", "yes", "on", "y", "t"}:
            for v in work_videos:
                cache_dir = cache_root / v.stem
                _safe_rmtree(cache_dir)
            # Try to remove the parent _frame_cache dir if it ended up empty.
            try:
                cache_root.rmdir()
            except OSError:
                pass

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
            "preprocess and a leaner audiovisual_timeline.md, at the cost of "
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
