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
from progress import install_lane_prefix, lane_progress
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

    # ------------------------------------------------------------------
    # Square center-crop, baked into the ffmpeg filter chain.
    #
    # Why: Florence-2's vision tower has a hard assertion that the
    # encoded feature map is square (`assert h * w == num_tokens, 'only
    # support square feature maps for now'` — `modeling_florence2.py`
    # line ~2610). With non-square pixel_values (e.g. 16:9 4K DJI
    # footage) the embedding produces a non-square map and the
    # assertion explodes mid-generate.
    #
    # Why ffmpeg-side crop instead of PIL post-decode:
    #   1. ffmpeg crops BEFORE rgb24 conversion, so the pipe carries
    #      `square^2 * 3` bytes per frame instead of `width * height * 3`.
    #      For 4K 16:9 input that's a ~33% reduction in pipe bandwidth
    #      and Python-side memory churn — meaningful at 1 fps over a
    #      multi-hour shoot.
    #   2. PIL.Image.crop() would force a temporary copy in user-space
    #      Python; ffmpeg's `crop` filter does it inside the decoder
    #      with zero extra allocation.
    #
    # We center-crop to `min(width, height)` so portrait, landscape,
    # and already-square footage all become square. ffmpeg's `crop`
    # filter defaults to centered when x/y are omitted.
    # ------------------------------------------------------------------
    square_dim = min(width, height)

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_bin, "-loglevel", "error",
        "-i", str(video_path),
        # Filter chain order matters: fps decimation FIRST (cheap, drops
        # ~95% of frames before we pay the crop cost), THEN crop. Output
        # of crop is `square_dim x square_dim` regardless of source AR.
        "-vf", f"fps={fps},crop={square_dim}:{square_dim}",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo", "-",
    ]

    # Subprocess.Popen so we can stream stdout in frame-sized chunks.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Per-frame size now reflects the post-crop dimensions, not the
    # source. Misreading this would mis-frame every chunk and yield
    # garbage / a hang on the final partial chunk.
    frame_size = square_dim * square_dim * 3
    t = 0
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if not buf or len(buf) < frame_size:
                break
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(
                square_dim, square_dim, 3
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
# transformers 5.x compatibility shim for Florence-2's trust-remote-code
# config.
#
# Florence-2 ships its own `configuration_florence2.py` (loaded via
# trust_remote_code=True) that references a handful of generation-related
# attributes on `self` during __init__:
#
#     if self.forced_bos_token_id is None and ...
#
# In transformers 4.x those attributes were initialized to None by
# `PretrainedConfig.__init__`. transformers 5.x removed several of them
# (they migrated into `GenerationConfig`), so Florence's old config code
# now crashes with:
#
#     AttributeError: 'Florence2LanguageConfig' object has no attribute
#                     'forced_bos_token_id'
#
# We restore the legacy lookup behavior by setting class-level None
# defaults on PretrainedConfig itself. Instance attribute lookup falls
# through to the class, so any subclass — including Florence's custom
# config — sees None instead of AttributeError. We do NOT re-add the
# attributes to instances, so any code that legitimately wrote them
# still wins over the class default.
#
# Idempotent: re-calling does nothing once the attributes are present.
# Scope: only patches attributes that were actually present in 4.x and
# removed in 5.x (verified against the transformers 4.45 → 5.0 changelog).
# Cheap: pure attribute writes on a class object, no model load.
# ---------------------------------------------------------------------------

_REMOVED_IN_TRANSFORMERS_5 = (
    "forced_bos_token_id",
    "forced_eos_token_id",
    "begin_suppress_tokens",
    "suppress_tokens",
    "task_specific_params",
)


def _install_legacy_pretrained_config_compat() -> None:
    """Restore the attributes Florence-2's trust-remote-code config still
    references but which transformers 5.x dropped from PretrainedConfig.

    Safe no-op when transformers isn't installed (the visual lane is
    optional via `[preprocess]`) and when the attributes already exist
    (transformers 4.x or future-5.x where the attrs were re-added).
    """
    try:
        from transformers import PretrainedConfig
    except ImportError:
        return
    for attr in _REMOVED_IN_TRANSFORMERS_5:
        if not hasattr(PretrainedConfig, attr):
            setattr(PretrainedConfig, attr, None)


# ---------------------------------------------------------------------------
# transformers 5.x compatibility shim — second layer, model side.
#
# In transformers 5.x the attention dispatcher in `modeling_utils.py`
# unconditionally reads support flags off `self` to decide which kernel
# to wire up (eager vs sdpa vs flash-attn-2). Bona-fide HF models declare
# these as class attributes on their subclass of `PreTrainedModel`. But
# Florence-2 ships its modeling file via trust_remote_code, and that file
# predates the new dispatcher contract — it never declares `_supports_sdpa`
# at all. Result, on first forward pass:
#
#     AttributeError: 'Florence2ForConditionalGeneration' object has no
#                     attribute '_supports_sdpa'
#         at transformers/modeling_utils.py:1709
#
# Why setting these to False is the correct floor (not True):
#   - False → dispatcher falls back to the eager attention path, which
#     is universally correct, just slower. Worst case: a perf hit.
#   - True without the model actually implementing the SDPA / flash-attn
#     contract → the dispatcher hands the kernel tensors it can't handle,
#     producing silently-wrong outputs (or a crash inside the kernel,
#     which is the lucky case). Correctness > throughput.
#
# Why this is safe for models that DO declare the attrs:
#   - We only `setattr` when `hasattr(...)` is False on the class. Real
#     HF models that declare `_supports_sdpa = True` on their subclass
#     are untouched — subclass attribute lookup hits their declaration
#     long before walking up to `PreTrainedModel`.
#   - Instance-level writes (`self._supports_sdpa = True` from inside a
#     custom __init__) also win over the class default, because Python
#     attribute lookup checks the instance __dict__ first. We're setting
#     the FLOOR, not overriding declared values.
#
# Scope: this matters only for trust_remote_code modules whose authors
# never updated to the 5.x dispatcher contract. First-party HF models
# always declare these flags — the patch is a no-op for them.
#
# Idempotent, cheap, no model load required.
# ---------------------------------------------------------------------------

_MISSING_MODEL_FLAGS_IN_TRANSFORMERS_5 = (
    "_supports_sdpa",
    "_supports_flash_attn_2",
    "_supports_flash_attn",          # legacy 4.x name, harmless to add
    "_supports_cache_class",
    "_supports_static_cache",
    "_supports_quantized_cache",
)


def _install_legacy_pretrained_model_compat() -> None:
    """Provide False defaults for attention-dispatch support flags that
    transformers 5.x's `PreTrainedModel` no longer declares but its
    dispatcher unconditionally reads.

    Safe no-op when transformers isn't installed (the visual lane is
    optional via `[preprocess]`) and when the attributes already exist
    (transformers 4.x, or trust_remote_code modules that *do* declare
    them). False is the correctness-preserving floor: it forces the
    eager attention path instead of risking a silently-wrong fast path.
    """
    try:
        from transformers.modeling_utils import PreTrainedModel
    except ImportError:
        return
    for attr in _MISSING_MODEL_FLAGS_IN_TRANSFORMERS_5:
        if not hasattr(PreTrainedModel, attr):
            setattr(PreTrainedModel, attr, False)


# ---------------------------------------------------------------------------
# transformers 5.x compatibility shim — fourth layer, tokenizer side.
#
# Florence-2's processor (the trust_remote_code `processing_florence2.py`)
# directly reads `tokenizer.additional_special_tokens` as an instance
# attribute during processor construction. In transformers 4.x this was
# always present (a real attribute initialized to `[]` in
# `PreTrainedTokenizerBase.__init__`). In transformers 5.x it was removed
# from the slow tokenizer's instance dict; the canonical access is now via
# `tokenizer.special_tokens_map.get("additional_special_tokens", [])`.
#
# Florence-2's processor was written before that move and crashes with:
#
#     AttributeError: RobertaTokenizer has no attribute
#                     additional_special_tokens.
#                     Did you mean: 'add_special_tokens'?
#
# We can't (and shouldn't) edit the trust_remote_code module — it lives in
# the HF cache. So we add a `property` to the tokenizer base classes that
# returns `[]` (or whatever's in `special_tokens_map`) when the underlying
# attribute is missing. Plain `setattr(cls, "additional_special_tokens",
# [])` would be wrong: the SETTER side of the same name still needs to
# work for code that *does* register additional special tokens. A property
# with both getter and setter preserves both contracts.
#
# Patch the bases (`PreTrainedTokenizerBase`, `PreTrainedTokenizer`,
# `PreTrainedTokenizerFast`) — every concrete tokenizer (Roberta, BART,
# T5, ...) inherits from one of those, so one shim covers them all.
# ---------------------------------------------------------------------------

def _install_legacy_tokenizer_compat() -> None:
    """Backfill `additional_special_tokens` on tokenizer base classes when
    transformers 5.x removed the legacy instance attribute that
    Florence-2's remote-code processor still reads as a bare attribute.

    Idempotent + best-effort: if transformers is uninstalled, if any of
    the base classes can't be imported, or if the attribute is already a
    real property/attr on the class, we leave the world alone. The check
    `attr in cls.__dict__` is intentional — we want to detect whether the
    *exact* class declares it, not whether some superclass does (which it
    will, after our first patch — that's what makes this idempotent).
    """
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except ImportError:
        return

    # Build the (getter, setter) pair once. The getter falls back through
    # three sources in priority order:
    #   1. The instance dict — if downstream code wrote a real attribute,
    #      respect it (transformers 4.x style code path).
    #   2. `special_tokens_map["additional_special_tokens"]` — the
    #      canonical transformers 5.x location.
    #   3. Empty list — safe default; Florence's processor only reads the
    #      attribute and never errors on `[]`.
    def _get_additional_special_tokens(self):
        # Instance attribute wins — preserves any explicit assignment.
        # We poke __dict__ directly to avoid triggering this very property.
        if "additional_special_tokens" in self.__dict__:
            return self.__dict__["additional_special_tokens"]
        # Fall back to the canonical 5.x location.
        try:
            stm = self.special_tokens_map
        except AttributeError:
            return []
        return list(stm.get("additional_special_tokens", []))

    def _set_additional_special_tokens(self, value):
        # Stash on the instance so the getter above will return it next
        # time, AND mirror it into the underlying tokenizer state via the
        # public API when available (so HF internals stay consistent).
        self.__dict__["additional_special_tokens"] = (
            list(value) if value is not None else []
        )
        # Best-effort sync with the special_tokens_map. add_special_tokens
        # exists on both fast and slow tokenizers across all versions, so
        # this is safe to call. Wrapped in try/except because some
        # subclasses lock the tokenizer state during init and would raise.
        try:
            self.add_special_tokens(
                {"additional_special_tokens": list(value or [])}
            )
        except Exception:
            pass

    new_property = property(
        _get_additional_special_tokens,
        _set_additional_special_tokens,
    )

    # Walk every tokenizer base class transformers exposes and inject the
    # shim where it's actually missing. We tolerate any of these imports
    # failing — different transformers versions split the class hierarchy
    # differently (PreTrainedTokenizerFast moved out of tokenization_utils
    # into tokenization_utils_fast in 4.x, etc.).
    base_classes: list[type] = [PreTrainedTokenizerBase]
    try:
        from transformers.tokenization_utils import PreTrainedTokenizer
        base_classes.append(PreTrainedTokenizer)
    except ImportError:
        pass
    try:
        from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
        base_classes.append(PreTrainedTokenizerFast)
    except ImportError:
        pass

    for cls in base_classes:
        # Only patch if THIS class doesn't declare the attr — the inheritance
        # chain will surface our patch from a base class to subclasses anyway.
        # Without this guard we'd shadow a working property with our own.
        if "additional_special_tokens" in cls.__dict__:
            existing = cls.__dict__["additional_special_tokens"]
            # If it's already a property (vanilla 4.x or working 5.x), leave
            # it alone — our shim is for the "attribute went missing" case.
            if isinstance(existing, property):
                continue
            # Plain class attr (rare) — also leave alone, it works.
            continue
        # Attr missing on this class. Patch it. Subclasses (RobertaTokenizer
        # et al.) automatically inherit the property via Python's MRO, so a
        # single setattr here covers every concrete tokenizer.
        setattr(cls, "additional_special_tokens", new_property)


# ---------------------------------------------------------------------------
# transformers 5.x compatibility shim — third layer, Florence-2 specific.
#
# `Florence2PreTrainedModel` (loaded via trust_remote_code) declares
# `_supports_sdpa` and `_supports_flash_attn_2` as `property` *objects*
# rather than plain class-attribute booleans:
#
#     @property
#     def _supports_sdpa(self):
#         return self.<some_instance_state>
#
# The getter is invoked during `PreTrainedModel.__init__`'s attention-
# dispatch resolution — at which point Florence's own __init__ has not
# yet finished setting up the instance state the getter needs. The
# getter raises AttributeError; Python's descriptor protocol swallows
# that AttributeError and falls through to nn.Module.__getattr__, which
# then raises the misleading top-level error:
#
#     AttributeError: 'Florence2ForConditionalGeneration' object has no
#                     attribute '_supports_sdpa'
#
# The previous two patches (config defaults + PreTrainedModel-class
# defaults) cannot help here, because the *subclass* property shadows
# the class attribute via Python's MRO — the property is found first,
# the broken getter runs first, the AttributeError is raised first.
#
# Fix: replace the broken `property` descriptor with a plain `False` on
# the Florence subclass itself. Plain class attributes don't trigger the
# descriptor protocol, so __getattribute__ returns False directly during
# init and the dispatcher routes to eager attention (the safe floor).
#
# We patch ONLY when the existing class attribute is actually a
# `property`, so a future Florence release that ships a fixed bool
# declaration is not silently re-broken by us.
# ---------------------------------------------------------------------------

_FLORENCE_BROKEN_FLAG_PROPERTIES = (
    "_supports_sdpa",
    "_supports_flash_attn_2",
)


def _patch_florence_support_flag_properties(model_id: str) -> None:
    """Replace Florence-2's broken `property`-typed support flags with
    plain `False` on `Florence2PreTrainedModel`.

    `model_id` is the HF repo id we'll load the trust_remote_code module
    from — same id you'll pass to `from_pretrained` shortly after. We
    fetch the class via `get_class_from_dynamic_module` which is the
    same path `from_pretrained` uses internally, so this guarantees we
    patch the exact class that will be instantiated.

    Best-effort: if transformers isn't installed, the dynamic module
    can't be resolved (network blocked, model id wrong), or the class
    doesn't actually declare these as properties (a fixed Florence
    release, or a different model entirely), we silently return. The
    subsequent `from_pretrained` call surfaces any real problem with a
    proper traceback.
    """
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except ImportError:
        return

    try:
        florence_pre = get_class_from_dynamic_module(
            "modeling_florence2.Florence2PreTrainedModel",
            model_id,
            revision=None,
        )
    except Exception:
        # Module not yet downloaded / network blocked / wrong model_id /
        # not actually a Florence model — leave it alone. The downstream
        # from_pretrained will produce a real error if anything's wrong.
        return

    for attr in _FLORENCE_BROKEN_FLAG_PROPERTIES:
        if isinstance(florence_pre.__dict__.get(attr), property):
            setattr(florence_pre, attr, False)


# ---------------------------------------------------------------------------
# Florence-2 model construction. trust_remote_code=True is REQUIRED — the
# model uses a custom modeling file that ships in the HF repo.
# ---------------------------------------------------------------------------

def _build_florence(model_id: str, device: str, dtype_name: str):
    # MUST run before from_pretrained — Florence's custom config touches
    # the removed attrs during __init__, which is invoked synchronously
    # by AutoConfig (and therefore AutoModelForCausalLM) below.
    _install_legacy_pretrained_config_compat()
    # Order matters: config patch must precede the model patch, because
    # AutoConfig resolution happens before any PreTrainedModel subclass
    # is touched. This second patch covers the attention dispatcher's
    # support-flag reads on the model side via PreTrainedModel defaults.
    _install_legacy_pretrained_model_compat()
    # Third patch: the previous two cover *missing* attrs, but Florence-2
    # actively *declares* `_supports_sdpa` as a broken `property` in its
    # subclass — that shadows our class-level defaults. We have to patch
    # the Florence subclass itself, which means resolving the dynamic
    # module first. This call also primes the trust_remote_code download
    # so the upcoming from_pretrained doesn't have to.
    _patch_florence_support_flag_properties(model_id)
    # Fourth patch: tokenizer-side. Florence's processor reads
    # `tokenizer.additional_special_tokens` as a bare attr; transformers
    # 5.x removed that. Backfill the property on the tokenizer base
    # classes so every concrete tokenizer (RobertaTokenizer in our case)
    # picks it up via MRO before AutoProcessor.from_pretrained runs.
    _install_legacy_tokenizer_compat()

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    if dtype_name not in dtype_map:
        raise ValueError(f"unknown dtype '{dtype_name}'")
    torch_dtype = dtype_map[dtype_name]

    print(f"  florence: model={model_id}  device={device}  dtype={dtype_name}")
    # transformers 5.x renamed the kwarg `torch_dtype` -> `dtype` and
    # warns on every load when you pass the old name. transformers 4.x
    # only accepts `torch_dtype`. Sniff the major version once and pick
    # the right keyword so we silence the 5.x noise without breaking
    # 4.x installs (the [preprocess] extra still pins transformers>=4.45).
    import transformers as _tf
    _tf_major = int(_tf.__version__.split(".", 1)[0])
    dtype_kwarg = {"dtype": torch_dtype} if _tf_major >= 5 else {"torch_dtype": torch_dtype}
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        **dtype_kwarg,
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
