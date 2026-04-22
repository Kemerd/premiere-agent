"""Florence-2 NumPy image processor + Rust-tokenizer wrap.

Replaces ``transformers.AutoProcessor`` for the visual lane's caption
tasks with a torch-free, transformers-free pipeline.  Three pieces:

  1. :class:`FlorenceImageProcessor` -- normalize already-768x768 RGB
     uint8 frames (the existing ffmpeg crop+scale chain in
     ``visual_lane._iter_frames_at_fps`` does the heavy lifting) into
     the ``(N, 3, 768, 768)`` float pixel_values that
     ``vision_encoder.onnx`` expects.  CLIP-style normalization values
     baked in from Florence-2's ``preprocessor_config.json`` -- we
     hard-code rather than parse because changing them silently
     destroys the visual encoder's pretrain distribution and the
     symptom (gibberish captions on the fp16 path) takes hours to
     trace back here.

  2. :class:`FlorenceTokenizer` -- thin wrap over the ``tokenizers``
     Rust library loading the BART-vocab ``tokenizer.json`` shipped
     with the ``onnx-community/Florence-2-base`` repo.  Encode prompts
     (with BOS/EOS) for the encoder side, decode generated ids (with
     bytefallback unicode handling) for the caption text.  We expose
     the special-token ids as plain attributes so the captioner's
     beam search can plug them straight into the logits processors.

  3. :func:`construct_prompts` + :func:`post_process_caption` -- the
     task-token -> human-readable text expansion (e.g. the API token
     ``<MORE_DETAILED_CAPTION>`` becomes the actual prompt
     ``"Describe with a paragraph what is shown in the image."``)
     and the post-generation cleanup (strip ``<s></s><pad>``, normalize
     punctuation spacing, return ``{task: text}``).

We deliberately do NOT replicate the full HF Florence2Processor here.
The visual lane only emits captions, not OD / OCR / region tasks --
those depend on a coordinate quantizer that has its own dependency
chain and is best ported in a follow-up if/when the lane needs them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Task prompt expansion
#
# Mirrors transformers.Florence2Processor.task_prompts_without_inputs and
# task_prompts_with_inputs.  Same wording as both the upstream HF
# implementation and the curiosity-ai/florence2-sharp C# reference -- if
# you change these strings, the visual encoder gets a DIFFERENT prompt
# than the one Florence-2 was trained against and caption quality drops.
# ---------------------------------------------------------------------------

# Tasks that take no extra text input (pure image -> text generation).
# The visual lane only ever uses MORE_DETAILED_CAPTION today, but the
# others stay here so a follow-up CLI flag can switch tasks without
# editing this module.
TASK_PROMPTS_WITHOUT_INPUTS: dict[str, str] = {
    "<OCR>":                  "What is the text in the image?",
    "<OCR_WITH_REGION>":      "What is the text in the image, with regions?",
    "<CAPTION>":              "What does the image describe?",
    "<DETAILED_CAPTION>":     "Describe in detail what is shown in the image.",
    "<MORE_DETAILED_CAPTION>": "Describe with a paragraph what is shown in the image.",
    "<OD>":                   "Locate the objects with category name in the image.",
    "<DENSE_REGION_CAPTION>": "Locate the objects in the image, with their descriptions.",
    "<REGION_PROPOSAL>":      "Locate the region proposals in the image.",
}

# Tasks that take an extra string input (the {0} slot).  Not used by the
# visual lane today; included for symmetry with the upstream reference.
TASK_PROMPTS_WITH_INPUTS: dict[str, str] = {
    "<CAPTION_TO_PHRASE_GROUNDING>":      "Locate the phrases in the caption: {0}",
    "<REFERRING_EXPRESSION_SEGMENTATION>": "Locate {0} in the image with mask",
    "<REGION_TO_SEGMENTATION>":            "What is the polygon mask of region {0}",
    "<OPEN_VOCABULARY_DETECTION>":         "Locate {0} in the image.",
    "<REGION_TO_CATEGORY>":                "What is the region {0}?",
    "<REGION_TO_DESCRIPTION>":             "What does the region {0} describe?",
    "<REGION_TO_OCR>":                     "What text is in the region {0}?",
}

# Tasks whose post-process is "strip special tokens, return raw text".
# OD / OCR / region tasks need a coordinate quantizer parser that lives
# in a follow-up; raise rather than silently emit garbage if asked.
CAPTION_TASKS: frozenset[str] = frozenset({
    "<CAPTION>",
    "<DETAILED_CAPTION>",
    "<MORE_DETAILED_CAPTION>",
})


def construct_prompts(task: str, text_input: str | None = None) -> str:
    """Resolve a Florence-2 API task token to its actual text prompt.

    Args:
        task: Task token like ``"<MORE_DETAILED_CAPTION>"``.  Must be a
            key in either :data:`TASK_PROMPTS_WITHOUT_INPUTS` or
            :data:`TASK_PROMPTS_WITH_INPUTS`.
        text_input: Required for tasks in the with-inputs dict (filled
            into the ``{0}`` slot); ignored for the without-inputs dict.

    Returns:
        The text prompt string fed to the BART tokenizer for the encoder
        input.  This is what Florence-2 was trained against -- exact
        wording matters.

    Raises:
        ValueError: ``task`` is not a known task token, or is a
            with-inputs task but ``text_input`` is None.
    """
    if task in TASK_PROMPTS_WITHOUT_INPUTS:
        return TASK_PROMPTS_WITHOUT_INPUTS[task]
    if task in TASK_PROMPTS_WITH_INPUTS:
        if text_input is None:
            raise ValueError(
                f"task {task!r} requires a text_input argument"
            )
        return TASK_PROMPTS_WITH_INPUTS[task].format(text_input)
    raise ValueError(
        f"unknown Florence-2 task {task!r}; valid tasks: "
        f"{sorted(set(TASK_PROMPTS_WITHOUT_INPUTS) | set(TASK_PROMPTS_WITH_INPUTS))}"
    )


# ---------------------------------------------------------------------------
# Image normalization
# ---------------------------------------------------------------------------

# CLIP-style normalization, exact values from Florence-2-base's
# preprocessor_config.json.  rescale_factor=1/255 maps uint8 [0,255]
# into [0,1] before mean/std subtraction.  Hard-coded because the
# visual encoder will not produce sane features against any other
# distribution -- this isn't a hyperparameter, it's part of the
# model's contract.
_FLORENCE_IMAGE_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_FLORENCE_IMAGE_STD:  tuple[float, float, float] = (0.229, 0.224, 0.225)
_FLORENCE_RESCALE: float = 1.0 / 255.0

# Florence-2-base's vision encoder ships a fixed-size positional
# embedding for 768x768 inputs (24x24 patches at patch_size=32).  Any
# other resolution indexes out-of-bounds inside the embedding and
# triggers a device-side assert.  The visual lane's ffmpeg pipeline
# already crop+scales to exactly this size, so we just assert the
# contract here rather than silently resizing on the Python side.
FLORENCE_INPUT_SIZE: int = 768


class FlorenceImageProcessor:
    """Pure-NumPy preprocessor for Florence-2's vision encoder.

    Inputs come in as uint8 RGB arrays sized exactly
    ``(FLORENCE_INPUT_SIZE, FLORENCE_INPUT_SIZE, 3)`` -- the visual
    lane's ffmpeg filter chain (``crop=min:min,scale=768:768:lanczos``)
    already produces frames in this exact shape, which is much cheaper
    than redoing the resize in PIL/NumPy.  We ONLY normalize and
    transpose to NCHW here.

    Output dtype is selectable, but in practice every onnx-community
    Florence-2 graph variant accepts FP32 pixel_values regardless of
    the internal weight precision (the "_fp16" file suffixes refer to
    weight-only quantization; I/O stays fp32 across all variants).
    The captioner always passes ``dtype=np.float32``; the parameter
    is kept for forward-compat with future graph exports that genuinely
    use FP16 I/O.
    """

    def __init__(self, dtype: np.dtype | str = np.float32) -> None:
        # Pre-compute mean/std as broadcastable (1, 3, 1, 1) buffers in
        # the target dtype so the per-frame normalize is a single
        # fused multiply-add the BLAS / SIMD path already optimizes.
        out_dtype = np.dtype(dtype)
        if out_dtype.kind != "f":
            raise ValueError(
                f"FlorenceImageProcessor needs a float dtype, got {out_dtype!r}"
            )
        self._dtype = out_dtype
        # Mean and std are stored AFTER rescale so the per-frame math
        # is `(uint8 * rescale - mean) / std` with no extra constants.
        self._mean = np.asarray(
            _FLORENCE_IMAGE_MEAN, dtype=out_dtype,
        ).reshape(1, 3, 1, 1)
        self._std = np.asarray(
            _FLORENCE_IMAGE_STD, dtype=out_dtype,
        ).reshape(1, 3, 1, 1)
        self._rescale = out_dtype.type(_FLORENCE_RESCALE)

    @property
    def dtype(self) -> np.dtype:
        """Output pixel_values dtype.  Matches the ONNX vision graph's expected input dtype."""
        return self._dtype

    def __call__(
        self,
        images: Sequence[np.ndarray] | np.ndarray,
    ) -> np.ndarray:
        """Normalize a batch of frames to NCHW pixel_values.

        Args:
            images: Either a single ``(H, W, 3) uint8`` array, a list
                of such arrays, or a stacked ``(N, H, W, 3) uint8``
                array.  All frames MUST be ``(FLORENCE_INPUT_SIZE,
                FLORENCE_INPUT_SIZE, 3)`` -- the ffmpeg pipeline
                upstream guarantees this; we re-assert because a
                silent shape mismatch produces a misleading
                CUDA-side error in the vision encoder.

        Returns:
            ``(N, 3, FLORENCE_INPUT_SIZE, FLORENCE_INPUT_SIZE)`` array
            in :attr:`dtype`, ready to feed ``vision_encoder.onnx``
            as ``pixel_values``.
        """
        # Coerce single-frame and list inputs into one stacked uint8 batch.
        # np.stack copies; np.asarray on an already-stacked input is a
        # no-op.  Either way the eventual normalize allocates a fresh
        # float buffer so we don't strictly need to avoid the copy here.
        if isinstance(images, np.ndarray) and images.ndim == 4:
            batch = images
        elif isinstance(images, np.ndarray) and images.ndim == 3:
            batch = images[None, ...]
        else:
            # Sequence of (H, W, 3) arrays; stack along new batch axis.
            batch = np.stack(list(images), axis=0)

        if batch.dtype != np.uint8:
            # Defensive -- the ffmpeg raw pipe is rgb24 = uint8.  A
            # float input here would silently double the rescale.
            raise ValueError(
                f"FlorenceImageProcessor expects uint8 frames, got {batch.dtype}"
            )
        if batch.shape[1:] != (FLORENCE_INPUT_SIZE, FLORENCE_INPUT_SIZE, 3):
            raise ValueError(
                f"FlorenceImageProcessor expects "
                f"({FLORENCE_INPUT_SIZE}, {FLORENCE_INPUT_SIZE}, 3) frames, "
                f"got {batch.shape[1:]}"
            )

        # uint8 (N, H, W, 3) -> float dtype (N, 3, H, W).  The transpose
        # is a view (zero copy); the .astype materializes the float
        # buffer once.  Doing the transpose BEFORE the multiply means
        # the subsequent (mean, std) broadcast operates on a contiguous
        # NCHW layout that NumPy's SIMD path is optimal for.
        nchw = np.transpose(batch, (0, 3, 1, 2))
        # Single fused expression so NumPy can elide the temporaries
        # under a recent BLAS / OpenBLAS arena.  Equivalent to
        # `((nchw.astype(out) * rescale) - mean) / std`.
        out = nchw.astype(self._dtype) * self._rescale
        out -= self._mean
        out /= self._std
        return out


# ---------------------------------------------------------------------------
# Tokenizer wrap
# ---------------------------------------------------------------------------

# Decoder start token id for BART-style models.  Florence-2 inherits
# this from BART: the decoder is primed with </s> and the first
# generated token gets force-routed to <s> via ForcedBOSTokenLogitsProcessor.
# We hard-code rather than read from generation_config.json because:
#   1. The repo we use (onnx-community/Florence-2-base) is frozen, so
#      these never change.
#   2. Reading config files at import time would couple this module to
#      the model dir layout, which the captioner already owns.
_FLORENCE_BOS_TOKEN_ID: int = 0          # <s>
_FLORENCE_PAD_TOKEN_ID: int = 1          # <pad>
_FLORENCE_EOS_TOKEN_ID: int = 2          # </s>
_FLORENCE_DECODER_START_TOKEN_ID: int = 2  # = EOS, BART convention


# Pre-compiled cleanup regexes.  These mirror the curiosity-ai
# CleanUpTokenization C# port, which itself mirrors the HF
# tokenization_utils_base.clean_up_tokenization function.  The Rust
# tokenizer with BPE byte-level should produce clean output already,
# but we run these as a belt-and-braces guarantee against any
# whitespace artefacts at sentence boundaries.
_CLEANUP_PATTERNS: tuple[tuple[str, str], ...] = (
    (" .", "."),
    (" ?", "?"),
    (" !", "!"),
    (" ,", ","),
    (" ' ", "'"),
    (" n't", "n't"),
    (" 'm", "'m"),
    (" 's", "'s"),
    (" 've", "'ve"),
    (" 're", "'re"),
)

# Pre-compiled multi-space collapse for the post-processor.  Florence's
# byte-level decoder occasionally emits a doubled space when a word
# straddles a special-token boundary.
_MULTISPACE_RE = re.compile(r"\s{2,}")


def clean_up_tokenization(text: str) -> str:
    """Fix common BPE-decoder whitespace artefacts in a generated string.

    Applied AFTER ``tokenizer.decode(skip_special_tokens=True)`` -- the
    decode produces clean text in 99% of cases but occasionally leaves
    a stray space before punctuation when a word ended at a special
    token boundary.  Cheap to run unconditionally.
    """
    for needle, replacement in _CLEANUP_PATTERNS:
        text = text.replace(needle, replacement)
    text = _MULTISPACE_RE.sub(" ", text)
    return text.strip()


def post_process_caption(text: str, task: str) -> dict[str, str]:
    """HF-Florence-compatible post-process for caption tasks.

    Args:
        text: The decoded text from the BART decoder.  May still
            contain ``<s>``, ``</s>`` and ``<pad>`` tokens if the
            caller used ``skip_special_tokens=False``; we strip
            defensively here regardless.
        task: The original task token, e.g. ``"<MORE_DETAILED_CAPTION>"``.
            Returned as the dict key so callers can treat the output
            uniformly across the different caption tasks.

    Returns:
        ``{task: cleaned_text}``.  Same shape as the upstream
        ``transformers.Florence2Processor.post_process_generation``
        for caption tasks; downstream code can swap from one to the
        other without other changes.

    Raises:
        ValueError: ``task`` is not a recognized caption task.  OD /
            OCR / region tasks are intentionally unsupported here --
            those need a coordinate-quantizer parser the visual lane
            doesn't currently call into.
    """
    if task not in CAPTION_TASKS:
        raise ValueError(
            f"post_process_caption only handles caption tasks; got "
            f"{task!r}.  Caption tasks: {sorted(CAPTION_TASKS)}"
        )
    cleaned = (
        text.replace("<s>", "")
            .replace("</s>", "")
            .replace("<pad>", "")
    )
    return {task: clean_up_tokenization(cleaned)}


class FlorenceTokenizer:
    """Wrap the ``tokenizers`` Rust BART tokenizer for Florence-2.

    Two reasons we don't just call ``transformers.AutoTokenizer``:

      1. Removing ``transformers`` from the visual lane's import path
         is half the point of this rewrite -- it cuts ~3-5s of cold
         start (the eager transformers backend probe) and ~50 MB of
         install size for the auto-init machinery.
      2. The Rust tokenizer is the same ``tokenizer.json`` file the
         ONNX export ships with.  Loading directly via
         ``tokenizers.Tokenizer.from_file`` keeps every BPE merge,
         every special token, and every byte-level pre-tokenizer rule
         identical to what Florence-2 was trained against.
    """

    # Special token ids exposed as class attributes so the captioner's
    # beam-search logits processors don't have to re-derive them.
    bos_token_id: int = _FLORENCE_BOS_TOKEN_ID
    pad_token_id: int = _FLORENCE_PAD_TOKEN_ID
    eos_token_id: int = _FLORENCE_EOS_TOKEN_ID
    decoder_start_token_id: int = _FLORENCE_DECODER_START_TOKEN_ID

    def __init__(self, tokenizer_json_path: str | Path) -> None:
        """Load a tokenizer from the standard ``tokenizer.json`` file.

        Args:
            tokenizer_json_path: Path to ``tokenizer.json`` -- the file
                ``huggingface_hub.snapshot_download`` writes alongside
                the ONNX subdir.  Must exist; we don't try to fall
                back to the slow tokenizer.

        Raises:
            FileNotFoundError: The path doesn't exist on disk.
        """
        from tokenizers import Tokenizer  # local import: keep module light
        path = Path(tokenizer_json_path)
        if not path.exists():
            raise FileNotFoundError(
                f"tokenizer.json missing at {path}; the ONNX snapshot "
                f"download may be incomplete -- delete the HF cache "
                f"for this repo and re-run"
            )
        self._tok = Tokenizer.from_file(str(path))
        # Configure padding so encode_batch with pad_to_multiple=None
        # produces uniform-length batches.  The pad token id is a fixed
        # constant for BART's vocab; we hard-code rather than query
        # ``self._tok.token_to_id("<pad>")`` because the tokenizer
        # config from onnx-community ships with explicit pad config
        # already and an extra setter is a no-op.
        self._tok.enable_padding(
            direction="right",
            pad_id=self.pad_token_id,
            pad_token="<pad>",
        )

    def encode_prompts(self, prompts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Batch-tokenize text prompts for the BART encoder side.

        Args:
            prompts: List of N prompt strings.  Each will get
                ``<s>`` prefix and ``</s>`` suffix added automatically
                via the tokenizer's post_processor (configured in
                ``tokenizer.json``).  The batch is padded to the
                longest member with ``<pad>``.

        Returns:
            ``(input_ids, attention_mask)`` as a pair of int64 arrays
            both of shape ``(N, max_len)``.  int64 because the ONNX
            embed_tokens / encoder graphs declare int64 inputs --
            int32 would require a per-call cast.
        """
        if not prompts:
            return (
                np.zeros((0, 0), dtype=np.int64),
                np.zeros((0, 0), dtype=np.int64),
            )
        encodings = self._tok.encode_batch(prompts, add_special_tokens=True)
        # Stack into NumPy arrays.  enable_padding above guarantees
        # all encodings share the same length, so the list comp -> stack
        # path is cheap and avoids a Python-level pad loop.
        input_ids = np.asarray(
            [enc.ids for enc in encodings], dtype=np.int64,
        )
        attention_mask = np.asarray(
            [enc.attention_mask for enc in encodings], dtype=np.int64,
        )
        return input_ids, attention_mask

    def decode_one(
        self,
        ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode a single sequence of token ids to a string.

        Args:
            ids: 1-D iterable of token ids.  Typically a ``list[int]``
                trimmed to the EOS-included prefix the beam search
                hypothesis chose.
            skip_special_tokens: When True (default), the BPE byte
                decoder skips ``<s>``, ``</s>``, ``<pad>``, ``<unk>``
                and the loc_/quad_ region tokens.  When False, the
                caller is expected to call :func:`post_process_caption`
                which strips them via string replace.

        Returns:
            The decoded string.  No additional cleanup applied --
            call :func:`clean_up_tokenization` separately if needed.
        """
        # `tokenizers` accepts any iterable of ints; cast to list for
        # the Rust binding which does exact-shape inference.
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self,
        sequences: list[list[int]],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Batch-decode N sequences with one Rust round-trip.

        Strictly faster than mapping :meth:`decode_one` over the list
        for batch >= ~4 -- the Rust binding amortizes the GIL release
        across the whole batch.
        """
        if not sequences:
            return []
        return self._tok.decode_batch(
            [list(s) for s in sequences],
            skip_special_tokens=skip_special_tokens,
        )

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including added/special tokens.

        Used by the captioner to size logits-processor scratch buffers
        (no_repeat_ngram needs an O(vocab_size) mask per beam).
        """
        # `get_vocab_size(with_added_tokens=True)` includes the
        # ``<MORE_DETAILED_CAPTION>`` etc. added tokens -- important
        # because a no-repeat-ngram mask sized to the BPE vocab would
        # silently truncate any added-token logits.
        return self._tok.get_vocab_size(with_added_tokens=True)
