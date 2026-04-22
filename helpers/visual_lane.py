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

# Sibling helpers folder is on sys.path when invoked from the orchestrator.
# extract_audio is NOT used here — visual lane is fully independent of
# the audio extraction step.
from wealthy import FLORENCE_BATCH, is_wealthy


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "microsoft/Florence-2-base"
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

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_bin, "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo", "-",
    ]

    # Subprocess.Popen so we can stream stdout in frame-sized chunks.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_size = width * height * 3
    t = 0
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if not buf or len(buf) < frame_size:
                break
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
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
# Florence-2 model construction. trust_remote_code=True is REQUIRED — the
# model uses a custom modeling file that ships in the HF repo.
# ---------------------------------------------------------------------------

def _build_florence(model_id: str, device: str, dtype_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    if dtype_name not in dtype_map:
        raise ValueError(f"unknown dtype '{dtype_name}'")
    torch_dtype = dtype_map[dtype_name]

    print(f"  florence: model={model_id}  device={device}  dtype={dtype_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
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

def run_visual_lane(
    video_path: Path,
    edit_dir: Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    fps: int = DEFAULT_FPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cuda:0",
    dtype_name: str = "fp16",
    task: str = DEFAULT_TASK_PROMPT,
    force: bool = False,
) -> Path:
    """Run the visual lane on a single video. Returns the JSON path.

    Caching: per-source-video, mtime-based.
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
    expected_frames = int(math.ceil(duration * fps))
    print(f"  florence: {video_path.name}  duration={duration:.1f}s  "
          f"frames~{expected_frames} @ {fps}fps  batch={batch_size}")

    model, processor, torch_dtype = _build_florence(model_id, device, dtype_name)

    captions: list[dict] = []
    batch_imgs: list = []
    batch_ts: list[int] = []
    t0 = time.time()
    last_log = t0

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
            batch_imgs.clear()
            batch_ts.clear()

            now = time.time()
            if now - last_log > 5.0:
                done = len(captions)
                rate = done / max(1e-3, now - t0)
                eta = (expected_frames - done) / max(1e-3, rate)
                print(f"    florence progress: {done}/{expected_frames} "
                      f"({rate:.1f} fps, eta {eta:.0f}s)")
                last_log = now

    # Flush trailing partial batch.
    if batch_imgs:
        texts = _caption_batch(
            model, processor, batch_imgs,
            device=device, torch_dtype=torch_dtype, task=task,
        )
        for tt, txt in zip(batch_ts, texts):
            captions.append({"t": tt, "text": txt})

    dt = time.time() - t0

    # Free GPU memory.
    try:
        import torch
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

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
