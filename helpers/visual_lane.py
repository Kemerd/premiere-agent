"""Visual lane: Florence-2-base captions at 1 fps for the entire timeline.

For an LLM editor to spot match cuts, identify shots, find B-roll
candidates, or react to "show the part where they're using the drill",
it needs *describable* visual context — not raw frames. Florence-2-base
(230M params, MIT Microsoft Research License) is the speed champion:

    RTX 4090: 50–100 fps with batching, ~5 minutes for 10k frames
    RTX 3060: ~20 fps, ~10 minutes for 10k frames

Sampling at 1 fps means a 3-hour shoot is ~10,800 frames. That's a 5-15
min preprocess on consumer hardware which is the right ballpark.

We use the `<MORE_DETAILED_CAPTION>` task — Florence-2's most descriptive
mode. Sample output:

    "a person holding a cordless drill above a metal panel with visible
     rivet holes"

JSON shape:
    {
      "model": "microsoft/Florence-2-base",
      "fps": 1,
      "duration": 43.0,
      "captions": [
        {"t": 12, "text": "a person holding a cordless drill ..."},
        {"t": 13, "text": "close-up of a drill bit entering metal, sparks"},
        {"t": 14, "text": "(same)"},      # dedup marker, see _dedup_consecutive
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
import subprocess
import sys
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
DEFAULT_FPS = 1
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


def _iter_frames_at_fps(video_path: Path, fps: int):
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
    t = 0
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if not buf or len(buf) < frame_size:
                break
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(
                target_dim, target_dim, 3
            )
            # Florence-2 takes PIL images. Conversion is cheap (no copy).
            yield t, Image.fromarray(arr, mode="RGB")
            t += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait(timeout=5)


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

def _build_florence(model_id: str, device: str, dtype_name: str):
    """Construct Florence-2 from the HF community port.

    Uses the native `Florence2ForConditionalGeneration` class shipped in
    transformers 4.55+ — no `trust_remote_code`, no remote modeling file,
    no monkey patches. Falls back to `AutoModelForImageTextToText` if the
    explicit class import fails (forward-compatible with transformers
    moving the class around in future minor releases).
    """
    import torch
    from transformers import AutoProcessor

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
    return model, processor, torch_dtype


# ---------------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------------

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

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            do_sample=False,
            num_beams=3,
        )

    raw_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
    out: list[str] = []
    for raw, img in zip(raw_texts, images):
        parsed = processor.post_process_generation(
            raw, task=task, image_size=(img.width, img.height),
        )
        # post_process_generation returns {task: caption_str} for caption tasks.
        text = parsed.get(task, "") if isinstance(parsed, dict) else str(parsed)
        out.append(str(text).strip())
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
# Main lane entry point
# ---------------------------------------------------------------------------

def _process_one(
    model,
    processor,
    torch_dtype,
    video_path: Path,
    edit_dir: Path,
    *,
    model_id: str,
    fps: int,
    batch_size: int,
    device: str,
    task: str,
    force: bool,
) -> Path:
    """Caption one video with already-built Florence model + processor.

    Split out so the batch entry point can amortize the ~3s Florence load
    across many videos in one Python process.
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
    print(f"  florence: {video_path.name}  duration={duration:.1f}s  "
          f"frames~{expected_frames} @ {fps}fps  batch={batch_size}")

    captions: list[dict] = []
    batch_imgs: list = []
    batch_ts: list[int] = []
    t0 = time.time()

    # Per-frame progress is genuinely informative here — Florence is the
    # slow lane, so users want to see frames-per-second crawl forward.
    # We tick once per BATCH (not per frame) to keep emit volume sane,
    # advancing by `len(batch)` each time.
    with lane_progress(
        "visual",
        total=expected_frames,
        unit="frame",
        desc=f"florence captions: {video_path.name}",
    ) as fbar:
        for ts, img in _iter_frames_at_fps(video_path, fps):
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

        # Flush trailing partial batch.
        if batch_imgs:
            texts = _caption_batch(
                model, processor, batch_imgs,
                device=device, torch_dtype=torch_dtype, task=task,
            )
            for tt, txt in zip(batch_ts, texts):
                captions.append({"t": tt, "text": txt})
            fbar.update(advance=len(batch_imgs))

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


def run_visual_lane_batch(
    video_paths: list[Path],
    edit_dir: Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    fps: int = DEFAULT_FPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cuda:0",
    dtype_name: str = "fp16",
    task: str = DEFAULT_TASK_PROMPT,
    force: bool = False,
) -> list[Path]:
    """Run the visual lane on N videos with Florence-2 loaded ONCE."""
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

    model, processor, torch_dtype = _build_florence(model_id, device, dtype_name)
    out_paths: list[Path] = []
    try:
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
                out_paths.append(_process_one(
                    model, processor, torch_dtype, v, edit_dir,
                    model_id=model_id, fps=fps, batch_size=batch_size,
                    device=device, task=task, force=force,
                ))
                vbar.update(advance=1, item=v.name)
    finally:
        try:
            import torch
            del model, processor
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
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS,
                    help=f"Sample rate in frames/sec (default: {DEFAULT_FPS})")
    ap.add_argument("--batch-size", type=int, default=None,
                    help=f"Inference batch size (default: {DEFAULT_BATCH_SIZE}, "
                         f"or {FLORENCE_BATCH} with --wealthy)")
    ap.add_argument("--wealthy", action="store_true",
                    help="Speed knob for 24GB+ cards (4090/5090). Bigger batch, "
                         "same model + outputs. Also reads VIDEO_USE_WEALTHY=1.")
    ap.add_argument("--device", default="cuda:0",
                    help="Torch device: cuda:0, mps, cpu (default: cuda:0)")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
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
        force=args.force,
    )


if __name__ == "__main__":
    main()
