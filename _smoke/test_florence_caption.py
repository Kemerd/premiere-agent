"""Standalone smoke test for the Florence-2 ONNX captioner.

Runs FlorenceCaptioner end-to-end on a single still image or a video file,
matching the exact production preprocess from helpers/visual_lane.py:
center-crop to square, Lanczos downscale to 768x768, uint8 RGB.  Useful
for debugging the captioner without rebuilding the visual_lane JSON
cache or waiting for a multi-minute video to fully process.

Usage:
    python _smoke/test_florence_caption.py <image_or_video_path>
    python _smoke/test_florence_caption.py <path> --beams 3 --max-new 128
    python _smoke/test_florence_caption.py <video> --frames 4 --fps 1
    python _smoke/test_florence_caption.py <path> --dtype mixed
    python _smoke/test_florence_caption.py <path> --dtype fp32   # paranoid

Exit codes:
    0 = caption(s) generated successfully
    1 = preprocess / load / inference failure (full traceback printed)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Make `helpers/` importable regardless of CWD.  This script lives at
# <repo>/_smoke/ but the prod code is one level up under <repo>/helpers.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "helpers"))

# Suppress the ~200-line TRT graph-import WARNING wall on session build.
# Setting BEFORE the onnxruntime import is the only thing that works;
# session_options.log_severity_level only affects per-Run() noise.
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

import numpy as np  # noqa: E402  (import after sys.path tweak)
from PIL import Image  # noqa: E402


# ---------------------------------------------------------------------------
# Image and video extension whitelists.  Anything not in IMAGE_EXTS gets
# treated as a video and routed through ffmpeg for frame extraction.
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# Florence-2-base native input geometry.  Hard-coded here instead of
# imported from _florence_processor so this script can be eyeballed in
# isolation without first reading the processor module.
FLORENCE_INPUT_SIZE = 768


# ---------------------------------------------------------------------------
# Preprocess:  arbitrary HxWx3 frame  ->  768x768x3 uint8
# ---------------------------------------------------------------------------

def _square_crop_resize_pillow(img: Image.Image) -> np.ndarray:
    """Center-square-crop and Lanczos-resize a PIL image to 768x768 RGB.

    Mirrors the ffmpeg filter chain from visual_lane._iter_frames_at_fps
    (``crop=min(w,h):min(w,h),scale=768:768:flags=lanczos``) but operates
    on a single in-memory PIL.Image instead of a streaming raw-video
    pipe.  Used for the still-image branch of this smoke test.

    Args:
        img: Any-size PIL image; converted to RGB up-front so RGBA /
            CMYK / palette inputs all funnel into the same code path.

    Returns:
        ``(768, 768, 3) uint8`` ndarray exactly as
        :class:`FlorenceImageProcessor` expects.
    """
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    # Centered crop box (left, upper, right, lower).
    left = (w - side) // 2
    upper = (h - side) // 2
    img = img.crop((left, upper, left + side, upper + side))
    img = img.resize(
        (FLORENCE_INPUT_SIZE, FLORENCE_INPUT_SIZE),
        Image.LANCZOS,
    )
    return np.asarray(img, dtype=np.uint8)


def _extract_video_frames(
    video_path: Path,
    *,
    n_frames: int,
    fps: int,
) -> list[np.ndarray]:
    """Pull N evenly-spaced 768x768 frames out of a video via ffmpeg.

    Uses imageio_ffmpeg's bundled ffmpeg the same way visual_lane.py
    does, so the smoke test exercises the SAME crop+scale filter chain
    that production runs.  Limits the pull to ``n_frames`` to keep the
    smoke test snappy regardless of clip length.

    Args:
        video_path: Local path to any container ffmpeg understands
            (mp4, mov, mkv, ...).
        n_frames: Maximum number of frames to caption.  We pull at
            ``fps`` Hz and stop once we have this many; this keeps a
            10-minute drone clip from generating 600 frames.
        fps: Sample rate for the ffmpeg ``fps=`` filter.  1 fps is the
            visual_lane default.

    Returns:
        List of ``(768, 768, 3) uint8`` ndarrays, length up to
        ``n_frames`` (may be shorter if the clip is shorter than
        ``n_frames / fps`` seconds).
    """
    import imageio_ffmpeg

    # Probe source dims so we can compute the square_dim center crop.
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(video_path),
    ]
    probe = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
    w_str, h_str = probe.stdout.strip().split("x")
    width, height = int(w_str), int(h_str)
    square_dim = min(width, height)

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_bin, "-loglevel", "error",
        "-i", str(video_path),
        "-vf",
        f"fps={fps},crop={square_dim}:{square_dim},"
        f"scale={FLORENCE_INPUT_SIZE}:{FLORENCE_INPUT_SIZE}:flags=lanczos",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo", "-",
    ]

    # Each frame on the pipe is exactly 768*768*3 bytes (rgb24).  We
    # read frames in this lockstep size and stop once we have enough.
    frame_bytes = FLORENCE_INPUT_SIZE * FLORENCE_INPUT_SIZE * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frames: list[np.ndarray] = []
    try:
        while len(frames) < n_frames:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                # End of stream (or short clip); whatever we have is
                # what we caption.
                break
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(
                FLORENCE_INPUT_SIZE, FLORENCE_INPUT_SIZE, 3,
            ).copy()  # copy() detaches from the pipe buffer
            frames.append(arr)
    finally:
        # Tear down the ffmpeg subprocess cleanly even on early break.
        proc.stdout.close()
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    return frames


def _load_frames(input_path: Path, *, n_frames: int, fps: int) -> list[np.ndarray]:
    """Dispatch on extension: image -> single-frame, anything else -> video."""
    ext = input_path.suffix.lower()
    if ext in IMAGE_EXTS:
        with Image.open(input_path) as img:
            return [_square_crop_resize_pillow(img)]
    return _extract_video_frames(input_path, n_frames=n_frames, fps=fps)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a still image (jpg/png/...) or any video file.",
    )
    parser.add_argument(
        "--beams", type=int, default=3,
        help="Beam search width (Florence-2 default = 3).",
    )
    parser.add_argument(
        "--max-new", type=int, default=128,
        help="Max new tokens per caption (Florence-2 default = 1024; "
             "128 is fine for MORE_DETAILED_CAPTION smoke tests).",
    )
    parser.add_argument(
        "--frames", type=int, default=2,
        help="Max number of video frames to caption (ignored for images).",
    )
    parser.add_argument(
        "--fps", type=int, default=1,
        help="Sample rate (Hz) for video frame extraction.",
    )
    parser.add_argument(
        "--dtype", choices=("mixed", "fp16", "fp32"), default="mixed",
        help="Captioner weight-precision mode.  'mixed' = fp16 weights "
             "for vision/embed/encoder + fp32 weights for the broken-"
             "upstream decoder (default, fastest working combo); "
             "'fp16' = all fp16 weights (currently broken upstream); "
             "'fp32' = all fp32 weights (paranoid quality reference).",
    )
    parser.add_argument(
        "--task", default="<MORE_DETAILED_CAPTION>",
        help="Florence-2 task token; defaults to MORE_DETAILED_CAPTION "
             "to match visual_lane.py.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[smoke] ERROR: input does not exist: {args.input}", file=sys.stderr)
        return 1

    print("=" * 72)
    print(f"[smoke] Florence-2 ONNX captioner smoke test")
    print(f"[smoke]   input    : {args.input}")
    print(f"[smoke]   dtype    : {args.dtype}")
    print(f"[smoke]   beams    : {args.beams}")
    print(f"[smoke]   max_new  : {args.max_new}")
    print(f"[smoke]   task     : {args.task}")
    print("=" * 72, flush=True)

    # Lazy imports keep the --help path snappy and let the script
    # report a clean error if onnxruntime is missing at the venv level.
    import _onnx_providers
    import florence_onnx

    print("[smoke] Resolving ONNX providers...", flush=True)
    providers = _onnx_providers.resolve_providers()
    print(f"[smoke]   providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}", flush=True)

    print(f"[smoke] Snapshotting onnx-community/Florence-2-base ({args.dtype})...", flush=True)
    snap_path = florence_onnx.download_florence_onnx(dtype=args.dtype)
    print(f"[smoke]   snapshot: {snap_path}", flush=True)

    print(f"[smoke] Building captioner ({args.dtype})...", flush=True)
    t0 = time.perf_counter()
    cap = florence_onnx.FlorenceCaptioner(
        snap_path, providers, dtype=args.dtype, intra_op_threads=2,
    )
    build_s = time.perf_counter() - t0
    print(f"[smoke]   build done in {build_s:.2f}s", flush=True)

    print(f"[smoke] Loading + preprocessing frames...", flush=True)
    t0 = time.perf_counter()
    frames = _load_frames(args.input, n_frames=args.frames, fps=args.fps)
    prep_s = time.perf_counter() - t0
    if not frames:
        print("[smoke] ERROR: no frames extracted from input.", file=sys.stderr)
        return 1
    print(f"[smoke]   {len(frames)} frame(s) ready in {prep_s:.2f}s "
          f"(shape={frames[0].shape}, dtype={frames[0].dtype})", flush=True)

    print(f"[smoke] Captioning {len(frames)} frame(s)...", flush=True)
    t0 = time.perf_counter()
    captions = cap.caption_batch(
        frames,
        task=args.task,
        num_beams=args.beams,
        max_new_tokens=args.max_new,
    )
    caption_s = time.perf_counter() - t0
    print(f"[smoke]   captioning done in {caption_s:.2f}s "
          f"({caption_s / len(frames):.2f}s/frame)", flush=True)

    print("-" * 72)
    for i, c in enumerate(captions):
        print(f"[smoke] frame {i}: {c!r}")
    print("-" * 72, flush=True)

    # Cooperative teardown so any pending CUDA streams flush before the
    # interpreter exits and prints a noisy destructor warning.
    cap.close()
    print("[smoke] Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
