"""Florence-2 ONNX Runtime captioner with real beam=3 search.

Hand-rolled orchestrator for the four ONNX subgraphs that
``onnx-community/Florence-2-base`` ships with:

  * ``vision_encoder.onnx`` -- DaViT image tower, ``(N, 3, 768, 768) ->
    (N, 577, 768)`` image-feature tokens.
  * ``embed_tokens.onnx`` -- BART input embedding lookup, ``(N, L) ->
    (N, L, 768)``.  Used twice: once for the prompt tokens before they
    flow through the encoder, and once per decoder step for the new
    token.
  * ``encoder_model.onnx`` -- BART encoder, ``(N, 577 + L_prompt, 768)
    -> (N, 577 + L_prompt, 768)`` after we've concatenated image
    features with prompt embeddings (the ``MergeInputIdsWithImageFeatures``
    operation -- see ``transformers.js`` Florence2's
    ``_merge_input_ids_with_image_features``).
  * ``decoder_model_merged.onnx`` -- BART decoder with both the
    ``use_cache_branch=False`` (full forward, computes encoder
    cross-attention KV from scratch) and ``use_cache_branch=True``
    (incremental, reuses cached encoder + decoder KV) paths in one
    graph.  Auto-regressed step by step until either every beam emits
    EOS or ``max_new_tokens`` is reached.

Why not ``optimum.ORTModelForVision2Seq`` / ``transformers.AutoModel``?
Both still depend on torch at runtime, which is exactly the dependency
this module exists to delete.  Why not ``onnxruntime-genai``?  It
doesn't model the four-subgraph Florence topology -- the genai loop
assumes a single fused decoder graph.  So we hand-roll, taking the
algorithm verbatim from Hugging Face ``transformers.generation`` and
the C# reference at ``curiosity-ai/florence2-sharp/Florence2.cs``.

Key correctness contracts (these all caused silent quality regressions
during development; the comments are the receipts):

  1. Decoder-side ``inputs_embeds`` is always shape ``(B*beams, 1,
     768)`` -- ONE token per step, NEVER the full prompt.  The merged
     decoder's ``use_cache_branch=False`` path on step 0 still takes a
     single token (the BART ``decoder_start_token_id`` = ``</s>`` =
     2) -- it doesn't take the encoder prompt.  The encoder side
     already absorbed the prompt; the decoder primes from the BART
     start token and generates from there.

  2. Past-KV initial shapes are ``(B*beams, num_heads, 0, head_dim)``
     for BOTH encoder and decoder layers.  The merged graph IGNORES
     these on step 0 (use_cache_branch=False) but still requires the
     names + dtypes match -- pass empty tensors, not ``None``.

  3. The encoder cross-attention KV cache produced on step 0 is
     IDENTICAL across all beams of the same frame (because the encoder
     output was replicated across beams BEFORE the decoder ran).  We
     therefore SKIP encoder-side beam reorder and only reorder the
     decoder self-attention KV between steps.

  4. ``ForcedBOSTokenLogitsProcessor`` fires when the input length to
     the decoder is exactly 1 (i.e. the very first generated step,
     where the only token in the running history is
     ``decoder_start_token_id``).  After this step, every beam's
     history is ``[</s>, <s>]`` and beam search proceeds normally.

  5. ``no_repeat_ngram_size=3`` runs PER BEAM PER STEP -- a banned
     token for beam i is one whose addition would create a 3-gram
     already present in beam i's history.  Using an ngram set
     spanning all beams would silently force the beams into different
     n-gram regions (which is technically a "diverse beam search"
     variant and not what Florence was tuned against).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from _florence_processor import (
    CAPTION_TASKS,
    FLORENCE_INPUT_SIZE,
    FlorenceImageProcessor,
    FlorenceTokenizer,
    construct_prompts,
    post_process_caption,
)


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Architecture constants (Florence-2-base, frozen)
#
# These are properties of the on-disk ONNX graphs we load; changing them
# would not "configure a different model size" -- it would silently feed
# wrong-shape tensors to a graph that expects exact shapes.  Hard-code
# rather than parse config.json so a corrupted config (or a wrong repo)
# fails loudly during ORT session construction instead of producing
# gibberish captions hours into a preprocess.
# ---------------------------------------------------------------------------

NUM_IMAGE_TOKENS: int = 577        # 1 CLS + 24*24 patch tokens at patch_size=32
NUM_DECODER_LAYERS: int = 6
NUM_DECODER_HEADS: int = 12
HEAD_DIM: int = 64
HIDDEN_SIZE: int = 768             # = NUM_DECODER_HEADS * HEAD_DIM


# ---------------------------------------------------------------------------
# Logits-processor + beam-scorer hyperparameters (Florence-2 defaults)
#
# Mirrors generation_config.json from onnx-community/Florence-2-base.
# These are the exact knobs Florence-2 was evaluated with -- changing
# them is a quality regression, not a configuration choice.
# ---------------------------------------------------------------------------

DEFAULT_NUM_BEAMS: int = 3
DEFAULT_NO_REPEAT_NGRAM_SIZE: int = 3
DEFAULT_LENGTH_PENALTY: float = 1.0
DEFAULT_MAX_NEW_TOKENS: int = 256  # matches the existing torch-path call site


# ---------------------------------------------------------------------------
# Per-subgraph weight-precision resolution
#
# Critical context, two intertwined facts that drove this design:
#
#   FACT 1 (the upstream bug we work around):
#     onnx-community's Florence-2 export ships with three decoder
#     variants (fp32, fp16, q4f16), but the FP16 + Q4F16 decoder
#     graphs are STRUCTURALLY INVALID -- the FP16 conversion produced
#     an If-node subgraph that returns a `logits` value defined in
#     the OUTER scope without an Identity-node bridge, violating the
#     ONNX spec.  ONNX Runtime >= 1.16 rejects it with:
#
#       "Subgraph output (logits) is an outer scope value being
#        returned directly. Please update the model to add an
#        Identity node between the outer scope value and the
#        subgraph output."
#
#     Tracked at https://huggingface.co/onnx-community/Florence-2-large-ft/discussions/7
#     (open since Sep 2025, unresolved).  The transformers.js README
#     for Florence-2-base works around this by using `dtype: 'fp32'`
#     for the whole model.  We do BETTER: only the broken decoder is
#     forced to FP32, while the vision encoder (biggest forward pass
#     per frame) stays on the fp16-weights variant for max throughput.
#
#   FACT 2 (the I/O contract that simplifies everything):
#     The "_fp16" suffix on these onnx-community files refers to the
#     INTERNAL weight precision only -- input and output tensors are
#     uniformly FP32 across ALL four graph variants.  This is the
#     standard `optimum` + `onnxconverter-common` weight-only fp16
#     quantization pattern.  So there is NO encoder->decoder dtype
#     boundary cast to worry about; we just always feed fp32 ndarrays.
#
# Mode summary:
#   "mixed" (DEFAULT, recommended):
#       Loads vision/embed/encoder from the fp16-weight variants
#       (smaller download, faster GEMMs on tensor cores) and the
#       decoder from the fp32 variant (the only loadable decoder).
#       Best speed achievable while remaining functionally correct.
#   "fp16": all-fp16-weights.  CURRENTLY BROKEN at decoder-load time
#       upstream; kept as an opt-in for the day onnx-community
#       re-exports a valid FP16 decoder.
#   "fp32": all-fp32-weights.  Matches the upstream transformers.js
#       README example exactly.  Bigger download (~745 MB vs ~620 MB
#       for "mixed"), modestly slower vision pass; useful as a
#       paranoid quality-reference baseline when debugging.
#
# Bonus complication (handled in _make_session below):
#   ORT's SimplifiedLayerNormFusion graph optimizer pass crashes on
#   the fp16 vision_encoder + fp16 encoder graphs with
#   "InsertedPrecisionFreeCast ... attempting to get index by a name
#   which does not exist".  Tracked at microsoft/onnxruntime#25692.
#   We disable that one specific fusion via `disabled_optimizers` so
#   the graphs load cleanly on every EP, including the CPU fallback.
# ---------------------------------------------------------------------------


# Per-subgraph default weight-precision map.  Keyed by the four ONNX
# subgraphs we orchestrate.  Mutating this changes what dtype="mixed"
# resolves to.
_MIXED_DTYPE_MAP: dict[str, str] = {
    "vision":  "fp16",
    "embed":   "fp16",
    "encoder": "fp16",
    "decoder": "fp32",   # forced FP32 -- see module-top docstring
}


def _resolve_dtype_map(dtype: str) -> dict[str, str]:
    """Expand a single ``dtype`` string into a per-subgraph weight map.

    Args:
        dtype: ``"mixed"`` (default), ``"fp16"`` (broken upstream),
            or ``"fp32"`` (paranoid reference).  Any unknown value
            raises ``ValueError`` so typos surface at construction
            time, not three hours into a preprocess.

    Returns:
        Dict with keys ``"vision"``, ``"embed"``, ``"encoder"``,
        ``"decoder"`` mapping to ``"fp16"`` or ``"fp32"``.  These
        select WHICH .onnx file to load -- they do NOT change the
        I/O dtype, which is uniformly fp32 across all variants.
    """
    if dtype == "mixed":
        # Copy so the caller can mutate without polluting the module
        # constant.  Cheap (4 entries).
        return dict(_MIXED_DTYPE_MAP)
    if dtype == "fp16":
        return {k: "fp16" for k in _MIXED_DTYPE_MAP}
    if dtype == "fp32":
        return {k: "fp32" for k in _MIXED_DTYPE_MAP}
    raise ValueError(
        f"dtype must be one of 'mixed' (default), 'fp16', 'fp32'; "
        f"got {dtype!r}"
    )


# ---------------------------------------------------------------------------
# File discovery: pick the right ONNX variants based on dtype + quantized
# ---------------------------------------------------------------------------

def _resolve_onnx_paths(
    model_dir: Path,
    *,
    dtype_map: dict[str, str],
    quantized_decoder: bool,
) -> dict[str, Path]:
    """Return absolute paths to the four ONNX files for this dtype combo.

    Args:
        model_dir: Root of the ``snapshot_download`` cache for the
            ``onnx-community/Florence-2-base`` repo.  Must contain an
            ``onnx/`` subdirectory.
        dtype_map: Per-subgraph dtype dict from :func:`_resolve_dtype_map`,
            with keys ``vision``, ``embed``, ``encoder``, ``decoder``
            mapping to ``"fp16"`` / ``"fp32"``.  Each subgraph is
            resolved independently so the default "mixed" mode picks
            the FP16 vision encoder + FP32 decoder.
        quantized_decoder: If True, swap the decoder for the
            ``decoder_model_merged_q4f16.onnx`` int4-weight variant.
            ~3.5x smaller decoder file, ~1.5-2x faster decoder step
            on a CUDA EP, very minor caption drift on long generations.
            **Currently broken upstream** with the same outer-scope
            subgraph bug as the FP16 decoder; kept wired for the day
            it gets re-exported cleanly.  Vision encoder + text
            encoder + embed graphs stay on whatever dtype_map says.

    Returns:
        Dict with keys ``"vision"``, ``"embed"``, ``"encoder"``,
        ``"decoder"`` mapping to existing Path objects.

    Raises:
        FileNotFoundError: A required ONNX file is missing on disk.
            Most likely cause is an incomplete HF snapshot; the
            error message names the missing file so the caller can
            re-run ``snapshot_download`` with the right ``allow_patterns``.
    """
    onnx_dir = model_dir / "onnx"
    if not onnx_dir.exists():
        raise FileNotFoundError(
            f"missing onnx/ subdir in {model_dir}; the HF snapshot may "
            f"be incomplete -- delete the local cache and re-download"
        )

    # Per-subgraph FP16 suffix.  The fp32 path is intentionally
    # unsuffixed because that's the upstream default in onnx-community's
    # exporter (`vision_encoder.onnx` is fp32; `vision_encoder_fp16.onnx`
    # is fp16).
    def _suffix(role: str) -> str:
        return "" if dtype_map[role] == "fp32" else "_fp16"

    files = {
        "vision":  onnx_dir / f"vision_encoder{_suffix('vision')}.onnx",
        "embed":   onnx_dir / f"embed_tokens{_suffix('embed')}.onnx",
        "encoder": onnx_dir / f"encoder_model{_suffix('encoder')}.onnx",
        "decoder": onnx_dir / (
            "decoder_model_merged_q4f16.onnx"
            if quantized_decoder
            else f"decoder_model_merged{_suffix('decoder')}.onnx"
        ),
    }

    for role, path in files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"missing Florence ONNX file for {role!r} at {path}; "
                f"the HF snapshot may be incomplete -- delete the local "
                f"cache and re-download with the matching allow_pattern"
            )
    return files


# ---------------------------------------------------------------------------
# ORT session construction
# ---------------------------------------------------------------------------

def _make_session(
    onnx_path: Path,
    providers,
    *,
    intra_op_threads: int = 1,
):
    """Build one ``InferenceSession`` with shared optimization flags.

    Args:
        onnx_path: Path to the .onnx file.
        providers: Resolved provider list from
            ``_onnx_providers.resolve_providers()``.  Pre-validated
            (TRT/CUDA/DML/CPU ladder).
        intra_op_threads: 1 by default -- multi-session pools want
            each session to NOT thrash CPU threads against its
            siblings.  The CUDA EP releases the GIL during native
            Run() so we still get true GPU concurrency between
            sessions even with single-threaded CPU side.

    Returns:
        An ``onnxruntime.InferenceSession`` ready for ``Run()``.
    """
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    # ORT_ENABLE_ALL = constant folding, redundant elim, fused
    # attention, etc.  Free at runtime, paid once at session
    # construction.  Florence's per-session construction is ~1-2s with
    # full opts vs. ~0.5s without -- worth it for the autoregressive
    # decoder where we re-hit the same kernels 200+ times per frame.
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Persist activations across Run() calls when shapes line up --
    # the decoder loop has a fixed B*beams batch and a slowly growing
    # seq dim, both of which the mem pattern caches happily.
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = True
    sess_options.intra_op_num_threads = intra_op_threads
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # SimplifiedLayerNormFusion crashes during init on the fp16
    # vision_encoder + fp16 encoder graphs (and any other graph that
    # has an "InsertedPrecisionFreeCast" node inside a layer-norm
    # pattern).  The fusion pass tries to look up the cast node by
    # name in a dict that doesn't contain it and asserts:
    #   "Attempting to get index by a name which does not exist:
    #    InsertedPrecisionFreeCast_/...".
    # Tracked at microsoft/onnxruntime#25692.  Disabling this one
    # specific fusion lets the fp16 graphs load on every EP, including
    # the CPU fallback we hit during smoke tests.  The cost is a
    # ~3-5% slowdown vs. the fused kernel on the layer-norm steps,
    # which is invisible next to the decoder's autoregressive cost.
    #
    # We pass the disabled_optimizers list directly to InferenceSession
    # because SessionOptions doesn't expose a setter for it -- ORT's
    # Python API takes it as a separate constructor arg.
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=providers,
        disabled_optimizers=["SimplifiedLayerNormFusion"],
    )


# ---------------------------------------------------------------------------
# No-repeat n-gram banned-token computation
#
# transformers.NoRepeatNGramLogitsProcessor maintained as a NumPy port.
# Per-beam, per-step it scans the running token sequence for n-grams
# whose prefix matches the last (n-1) tokens; the last token of each
# such n-gram becomes a banned next-token id.
# ---------------------------------------------------------------------------

def _calc_banned_ngram_tokens(
    prev_ids: list[int],
    ngram_size: int,
) -> list[int]:
    """Return the list of token ids that would create a repeat n-gram.

    Args:
        prev_ids: 1-D list of token ids generated so far for one beam.
            Includes the ``decoder_start_token_id`` at position 0.
        ngram_size: ``no_repeat_ngram_size`` -- 3 for Florence default.

    Returns:
        List of distinct banned token ids.  Empty if the sequence is
        too short to form an n-gram OR if no prefix-matching n-gram
        exists in the history.
    """
    cur_len = len(prev_ids)
    if cur_len + 1 < ngram_size:
        # Not enough history yet to form an n-gram including the next
        # token.  Returning early avoids an empty-tuple edge case in
        # the prefix comparison below.
        return []
    # The prefix the next token would extend: last (n-1) generated.
    prefix = tuple(prev_ids[-(ngram_size - 1):])
    banned: set[int] = set()
    # Walk every position in `prev_ids` where a full n-gram fits.
    # The n-gram at index i is `prev_ids[i : i + ngram_size]`.
    for i in range(cur_len - ngram_size + 1):
        if tuple(prev_ids[i : i + ngram_size - 1]) == prefix:
            banned.add(int(prev_ids[i + ngram_size - 1]))
    return list(banned)


# ---------------------------------------------------------------------------
# Stable log_softmax in float32 for the (B*beams, V) logits row
#
# We materialize fp32 here because the beam scorer adds running scores
# across many steps and the cumulative numerical drift in fp16 matters
# (fp16 has ~3 decimal digits of precision; over 200 steps the
# cumulative score loses ~2 of those digits).  Same trick HF uses in
# generation_utils.
# ---------------------------------------------------------------------------

def _log_softmax(logits_row: np.ndarray) -> np.ndarray:
    """Numerically stable log_softmax along the last axis, fp32 output.

    Args:
        logits_row: ``(B*beams, V)`` array.  Any float dtype -- we cast
            internally.

    Returns:
        ``(B*beams, V)`` array of float32 log-probabilities, max value
        per row exactly 0.0.
    """
    x = logits_row.astype(np.float32, copy=False)
    # Subtract per-row max for numerical stability.  The max op alone
    # is ~5x cheaper than the exp/log roundtrip so this is essentially
    # free relative to what follows.
    x = x - x.max(axis=-1, keepdims=True)
    # log(sum(exp(x))) computed via expm1 wouldn't help here because
    # the per-row max is now 0 so the largest exp arg is exactly 0
    # (exp = 1) and there's no underflow concern for the others.
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# Initial past-KV: zero-length tensors for use_cache_branch=False step
# ---------------------------------------------------------------------------

def _init_past_kv(
    batch_beams: int,
    *,
    dtype: np.dtype,
) -> dict[str, np.ndarray]:
    """Build the dict of empty ``past_key_values.*`` inputs for step 0.

    Args:
        batch_beams: Effective batch dimension = ``frames * num_beams``.
        dtype: Decoder I/O float dtype.  The onnx-community decoder
            graphs all use FP32 I/O regardless of internal weight
            precision, so callers should pass ``np.float32``.  Kept
            as a parameter for symmetry / future-proofing if a real
            FP16 decoder ever ships upstream.  ONNX shape-checks
            dtype on every input even for zero-length tensors.

    Returns:
        Dict of 24 entries (6 layers x {encoder,decoder} x {key,value})
        ready to splat into the decoder's feed-dict.
    """
    out: dict[str, np.ndarray] = {}
    empty_shape = (batch_beams, NUM_DECODER_HEADS, 0, HEAD_DIM)
    for i in range(NUM_DECODER_LAYERS):
        for side in ("encoder", "decoder"):
            for tk in ("key", "value"):
                out[f"past_key_values.{i}.{side}.{tk}"] = np.zeros(
                    empty_shape, dtype=dtype,
                )
    return out


def _present_to_past(
    decoder_outputs: list[np.ndarray],
    output_names: list[str],
) -> dict[str, np.ndarray]:
    """Rename ``present.*`` decoder outputs to ``past_key_values.*`` inputs.

    The merged decoder ONNX graph emits 25 outputs in the order
    ``[logits, present.0.encoder.key, present.0.encoder.value,
    present.0.decoder.key, present.0.decoder.value, ...]``.  The next
    step's input feed wants the SAME tensors but renamed
    ``past_key_values.*``.  We drive the rename off the actual
    ``output_names`` so it stays correct if a future export reorders
    the outputs.
    """
    out: dict[str, np.ndarray] = {}
    for name, arr in zip(output_names, decoder_outputs):
        if not name.startswith("present."):
            # Skip 'logits' (the only non-present output).
            continue
        out["past_key_values." + name[len("present.") :]] = arr
    return out


# ---------------------------------------------------------------------------
# The captioner
# ---------------------------------------------------------------------------

class FlorenceCaptioner:
    """Stateful Florence-2-base captioner over four ONNX subgraphs.

    Construction loads four ``InferenceSession``s + the tokenizer +
    the image processor; subsequent calls to :meth:`caption_batch`
    reuse all of these.  Sessions are NOT thread-safe in the general
    case, so use one ``FlorenceCaptioner`` per worker thread (the
    ``_florence_pool`` module provides a managed pool of these).

    Memory footprint after construction (CUDA mixed-dtype default,
    batch up to 8, beams=3, 256 max_new_tokens):
      * ONNX weights resident on GPU: ~620 MB (fp16-weight vision/
        embed/encoder + fp32-weight decoder)
      * Activation arena steady-state: ~700-900 MB
      * KV cache peak (B*beams=24, src_len~600, tgt_len~256): ~720 MB
        (fp32 KV is 2x the fp16 size; the trade for not crashing on
        the broken upstream fp16 decoder)
      * Total per-session: ~2.0-2.3 GB

    Attributes:
        model_id: Hugging Face repo id this captioner was loaded from.
            Returned in the JSON cache so cache invalidation can
            differentiate model versions.  Suffixed with the dtype
            mode so changing dtype re-tags the cache transparently.
        dtype: ``"mixed"`` (default), ``"fp16"`` (broken upstream),
            or ``"fp32"`` (paranoid reference).  See module-top docs.
            Selects WHICH .onnx file gets loaded for each subgraph;
            does NOT affect the I/O dtype which is always fp32.
        dtype_map: Per-subgraph weight-precision dict actually used
            to pick the four ONNX files.  Useful for debugging and
            for the JSON sidecar's ``model_dtype_map`` field.
        quantized_decoder: Whether the q4f16 decoder variant is loaded.
    """

    def __init__(
        self,
        model_dir: str | Path,
        providers,
        *,
        dtype: str = "mixed",
        quantized_decoder: bool = False,
        intra_op_threads: int = 1,
    ) -> None:
        """Load the four ONNX sessions + processor + tokenizer.

        Args:
            model_dir: Path to a downloaded ``onnx-community/Florence-2-base``
                snapshot.  Must contain ``tokenizer.json`` AND an ``onnx/``
                subdir with the four required ONNX files (see
                :func:`_resolve_onnx_paths`).
            providers: Resolved EP ladder from
                ``_onnx_providers.resolve_providers()``.  Each session
                gets the same ladder; ORT picks the highest-priority
                EP that supports the model's ops.
            dtype: ``"mixed"`` (default), ``"fp16"`` (broken upstream),
                or ``"fp32"`` (paranoid reference).  See the module-
                level docstring for the long story; tl;dr the upstream
                onnx-community fp16 decoder graph is structurally
                invalid and won't load on ORT >= 1.16, so the default
                "mixed" path picks the fp32 decoder file while keeping
                the rest of the graphs on the fp16-weight variants
                for max speed.  All graphs use FP32 I/O regardless of
                weight precision -- the dtype switch is purely about
                which .onnx file lives on disk.
            quantized_decoder: Swap the decoder for the q4f16 variant.
                Currently broken upstream with the same subgraph bug
                as the fp16 decoder; kept wired for the day it gets
                re-exported cleanly.  Off by default.
            intra_op_threads: Per-session intra-op thread count.  1
                is correct for the multi-session pool; bump to 4-8
                if running a single captioner standalone.
        """
        # Resolve dtype string into a per-subgraph weight-precision map
        # upfront.  _resolve_dtype_map raises ValueError on unknown
        # strings, so typos surface here instead of three hours into a
        # preprocess.  This map only drives FILE SELECTION -- the
        # actual ndarray dtype we feed at runtime is always fp32 (see
        # the long-form note in the module-top docstring).
        self.dtype = dtype
        self.dtype_map = _resolve_dtype_map(dtype)
        self.quantized_decoder = quantized_decoder
        self.model_dir = Path(model_dir)

        # Track an ID that uniquely identifies the bytes we're about to
        # encode against, for cache-invalidation in the visual lane's
        # JSON sidecar.  The dtype suffix means swapping --dtype re-tags
        # the cache without needing --force.  Quantized variant gets
        # its own suffix on top.
        self.model_id = f"onnx-community/Florence-2-base+{dtype}"
        if quantized_decoder:
            # Same pattern as audio_lane's vocab_sha -- a small textual
            # discriminator on the cache key so flipping the flag at
            # the CLI re-tags transparently.
            self.model_id += "+q4f16dec"

        paths = _resolve_onnx_paths(
            self.model_dir,
            dtype_map=self.dtype_map,
            quantized_decoder=quantized_decoder,
        )
        _log.info(
            "florence-onnx[%s]: vision=%s embed=%s encoder=%s decoder=%s",
            dtype,
            paths["vision"].name, paths["embed"].name,
            paths["encoder"].name, paths["decoder"].name,
        )

        # Build the four sessions sequentially.  Could parallelize but
        # ORT is heavy on the GPU memory allocator during construction
        # and parallel construction tends to OOM on tight cards --
        # sequential is the safe + slightly slower path.
        self._vision = _make_session(
            paths["vision"], providers, intra_op_threads=intra_op_threads,
        )
        self._embed = _make_session(
            paths["embed"], providers, intra_op_threads=intra_op_threads,
        )
        self._encoder = _make_session(
            paths["encoder"], providers, intra_op_threads=intra_op_threads,
        )
        self._decoder = _make_session(
            paths["decoder"], providers, intra_op_threads=intra_op_threads,
        )

        # Cache decoder output names so the inner loop's present->past
        # renamer doesn't pay the per-call overhead of querying ORT.
        self._decoder_output_names: list[str] = [
            o.name for o in self._decoder.get_outputs()
        ]

        # Pure-NumPy preprocessor.  All four ONNX variants have FP32
        # I/O contracts (the "_fp16" suffix in the filenames is purely
        # internal weight precision; inputs and outputs stay fp32).
        # So the preprocessor always produces fp32, regardless of
        # which graph variant is loaded.  No per-step casts needed.
        self._image_processor = FlorenceImageProcessor(dtype=np.float32)

        tokenizer_json = self.model_dir / "tokenizer.json"
        self._tokenizer = FlorenceTokenizer(tokenizer_json)
        self._vocab_size = self._tokenizer.vocab_size

    # ----------------------------------------------------------------
    # Sub-pass wrappers -- all return numpy arrays for clarity.  The
    # IOBinding optimization (keep tensors GPU-resident across passes)
    # is a follow-up; correctness-first matters because debugging
    # silent caption regressions later is much harder than now.
    # ----------------------------------------------------------------

    def _run_vision(self, pixel_values: np.ndarray) -> np.ndarray:
        """Forward the vision encoder on a batch of pixel_values.

        Args:
            pixel_values: ``(N, 3, 768, 768) float32``.  Produced by
                :class:`FlorenceImageProcessor`.  All four ONNX
                graph variants have FP32 I/O regardless of internal
                weight precision.

        Returns:
            ``(N, NUM_IMAGE_TOKENS=577, HIDDEN_SIZE=768) float32``
            image-feature tokens.
        """
        outputs = self._vision.run(
            None, {"pixel_values": pixel_values},
        )
        return outputs[0]

    def _run_embed(self, input_ids: np.ndarray) -> np.ndarray:
        """Run the BART input embedding lookup on token ids.

        Args:
            input_ids: ``(N, L) int64``.  Both the encoder side prompt
                pass (large L) and the decoder per-step pass (L=1)
                use this same graph.

        Returns:
            ``(N, L, HIDDEN_SIZE=768) float32`` token embeddings.
            All variants of the embed graph have FP32 outputs.
        """
        outputs = self._embed.run(
            None, {"input_ids": input_ids.astype(np.int64, copy=False)},
        )
        return outputs[0]

    def _run_encoder(
        self,
        merged_embeds: np.ndarray,
        merged_attn: np.ndarray,
    ) -> np.ndarray:
        """Forward the BART text encoder on merged image+prompt embeds.

        Args:
            merged_embeds: ``(N, 577 + L_prompt, 768) float32`` --
                concatenation of vision_encoder output and prompt
                embeds along axis 1.
            merged_attn: ``(N, 577 + L_prompt) int64`` -- ones for the
                image prefix concatenated with the tokenizer's prompt
                attention_mask.

        Returns:
            ``(N, 577 + L_prompt, 768) float32`` encoder hidden states.
            Fed straight to the decoder as ``encoder_hidden_states``
            for cross-attention.
        """
        outputs = self._encoder.run(
            None,
            {
                "inputs_embeds":   merged_embeds,
                "attention_mask":  merged_attn.astype(np.int64, copy=False),
            },
        )
        return outputs[0]

    # ----------------------------------------------------------------
    # The beam-search loop -- the heavy logic.
    # ----------------------------------------------------------------

    def _beam_search(
        self,
        encoder_hidden: np.ndarray,
        encoder_attn: np.ndarray,
        *,
        num_beams: int,
        max_new_tokens: int,
    ) -> list[list[int]]:
        """Run BART-style beam search over the decoder.

        Args:
            encoder_hidden: ``(B, src_len, 768)`` encoder output, ONE
                row per frame in the batch -- this method handles
                replication to ``(B*beams, src_len, 768)`` internally.
            encoder_attn: ``(B, src_len) int64`` matching attention
                mask for the encoder output.
            num_beams: Beam width, default 3 (Florence-2's training
                config); 1 reduces to greedy decoding.
            max_new_tokens: Hard cap on generated length, exclusive of
                the leading decoder_start_token.  Forced EOS fires at
                step ``max_new_tokens - 1`` if no beam has terminated.

        Returns:
            List of length B; each element is a list of token ids for
            the best (highest length-normalized score) hypothesis,
            including the leading ``decoder_start_token_id`` and the
            terminating EOS (when present).  Caller decodes via
            :meth:`FlorenceTokenizer.decode_one`.
        """
        bsz = encoder_hidden.shape[0]
        bb = bsz * num_beams
        src_len = encoder_hidden.shape[1]

        # Replicate encoder outputs across beams along axis 0.  After
        # this the BB axis is interleaved as
        #   [(frame0, beam0), (frame0, beam1), ..., (frame0, beam-1),
        #    (frame1, beam0), ...].
        # np.repeat with axis=0 + count=num_beams gives exactly this
        # interleaving (vs np.tile which would give [(f0, b0), (f1, b0), ...]).
        enc_hidden = np.repeat(encoder_hidden, num_beams, axis=0)
        enc_attn = np.repeat(
            encoder_attn.astype(np.int64, copy=False), num_beams, axis=0,
        )
        # No encoder -> decoder dtype cast needed.  All four
        # subgraphs (regardless of weight precision) have FP32 I/O
        # contracts -- the "_fp16" suffix in the filenames is purely
        # internal weight precision per the standard optimum +
        # onnxconverter-common weight-only fp16 quantization pattern.
        # See the module-top docstring for the long story.

        tok = self._tokenizer
        decoder_start = tok.decoder_start_token_id
        bos_id = tok.bos_token_id
        eos_id = tok.eos_token_id
        pad_id = tok.pad_token_id

        # Beam state.  All beam_scores in fp32; integer ids in int64.
        # Initialize beam_scores so that on step 0 the top-2*beams
        # candidates can ONLY come from beam 0 (others have score -inf,
        # so any expansion of them is dominated).  This is the standard
        # "first step deduplication" trick from transformers.
        beam_scores = np.zeros((bsz, num_beams), dtype=np.float32)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.reshape(bb)

        # Per-beam token history (Python list of ints; small).  These
        # also drive the no_repeat_ngram processor.
        generated_ids: list[list[int]] = [
            [decoder_start] for _ in range(bb)
        ]

        # Current-step input tokens for the decoder.  Always shape
        # (BB, 1) -- the merged decoder always takes single-token
        # incremental input regardless of use_cache_branch.
        cur_tokens = np.full(
            (bb, 1), decoder_start, dtype=np.int64,
        )

        # Past KV initialized to zero-len for both encoder and decoder
        # sides.  use_cache_branch=False on step 0 ignores values but
        # still shape-checks the names.  Decoder I/O is uniformly fp32
        # across all weight-precision variants, so the KV cache is
        # always fp32 too.
        past_kv = _init_past_kv(bb, dtype=np.float32)

        # Per-frame finished-hypothesis collectors.  Each entry is a
        # tuple (length_normalized_score, token_id_list).
        finished_hyps: list[list[tuple[float, list[int]]]] = [
            [] for _ in range(bsz)
        ]
        # Per-frame done flag.  early_stopping=True (Florence default):
        # frame is done as soon as num_beams hyps collected.
        done_mask = np.zeros(bsz, dtype=bool)

        use_cache_branch = False

        # The cumulative-score buffer is the largest per-step alloc
        # outside the decoder forward.  Pre-allocating once and
        # in-place writing each step avoids 200+ allocs per frame.
        # Shape (BB, V) fp32.
        cum_scores_buf = np.empty((bb, self._vocab_size), dtype=np.float32)

        for step in range(max_new_tokens):
            # 1. Embed the current single token per beam -> (BB, 1, 768).
            #    The embed graph emits fp32 regardless of weight
            #    precision, so this drops straight into the decoder
            #    feed without a cast.
            inputs_embeds = self._run_embed(cur_tokens)

            # 2. Decoder forward.  Feed includes ALL 24 past_kv tensors
            #    + encoder hidden + encoder attention + use_cache_branch.
            feed = {
                "encoder_attention_mask": enc_attn,
                "encoder_hidden_states":  enc_hidden,
                "inputs_embeds":          inputs_embeds,
                "use_cache_branch":       np.array([use_cache_branch], dtype=np.bool_),
                **past_kv,
            }
            decoder_outputs = self._decoder.run(None, feed)
            # outputs[0] is logits; rest are present.* tensors.
            logits = decoder_outputs[0]  # (BB, 1, V)
            past_kv = _present_to_past(
                decoder_outputs, self._decoder_output_names,
            )

            # 3. log_softmax on the LAST position (which is the only
            #    position when use_cache_branch=True; on step 0 the
            #    decoder still emits a single position because we fed
            #    a single token).
            log_probs = _log_softmax(logits[:, -1, :])  # (BB, V) fp32

            # 4. Logits processors -- order matters.  HF runs them in
            #    the order: NoRepeatNGram, ForcedBOS, ForcedEOS.
            #    Forced BOS/EOS are absolute overrides (set the entire
            #    row except the forced token to -inf), so when they
            #    fire the no_repeat_ngram pass on the same row is a
            #    no-op (it just zeroes already-zero entries).  Still
            #    cheap to call unconditionally.

            # NoRepeatNGram: per beam.  Loop is tight enough at BB=24
            # (8 frames * 3 beams) and seq_len <= 256 that NumPy
            # vectorization isn't worth the complexity here.
            if step + 1 >= DEFAULT_NO_REPEAT_NGRAM_SIZE:
                for bi in range(bb):
                    banned = _calc_banned_ngram_tokens(
                        generated_ids[bi], DEFAULT_NO_REPEAT_NGRAM_SIZE,
                    )
                    if banned:
                        log_probs[bi, banned] = -np.inf

            # ForcedBOSTokenLogitsProcessor: when the generated history
            # is just [decoder_start_token] (i.e. step == 0), force the
            # next token to be bos_token_id (=<s>=0 for BART).
            if step == 0:
                log_probs[:, :] = -np.inf
                log_probs[:, bos_id] = 0.0

            # ForcedEOSTokenLogitsProcessor: at the very last step we
            # could possibly take, force EOS so we always end with </s>.
            if step == max_new_tokens - 1:
                log_probs[:, :] = -np.inf
                log_probs[:, eos_id] = 0.0

            # 5. Per-frame beam scoring + selection.
            np.add(beam_scores[:, None], log_probs, out=cum_scores_buf)
            # cum_scores_buf shape (BB, V).  Reshape to (B, beams*V) for
            # per-frame top-K extraction.  reshape returns a view; safe.
            cum_scores_per_frame = cum_scores_buf.reshape(
                bsz, num_beams * self._vocab_size,
            )

            # Per-frame: top 2*num_beams candidates (the +1 over
            # num_beams gives us slack to skip EOS hypotheses without
            # dropping below num_beams active beams).
            # transformers picks 2*num_beams; we mirror that exactly.
            n_keep = 2 * num_beams

            next_beam_scores = np.full(
                (bsz, num_beams), -np.inf, dtype=np.float32,
            )
            next_beam_tokens = np.full(
                (bsz, num_beams), pad_id, dtype=np.int64,
            )
            # Global beam index in [0, BB).  Default = "this slot stays
            # parked on the source beam" so done frames don't disrupt
            # the KV reorder gather.
            next_beam_indices = np.arange(bb, dtype=np.int64).reshape(
                bsz, num_beams,
            ).copy()

            for bi in range(bsz):
                if done_mask[bi]:
                    # Frame already finalized; pad next_tokens so the
                    # next decoder step on these positions emits noise
                    # we ignore.  next_beam_indices stays identity for
                    # KV reorder (no shuffle for this frame).
                    next_beam_tokens[bi, :] = pad_id
                    next_beam_scores[bi, :] = 0.0
                    continue

                row = cum_scores_per_frame[bi]  # (beams*V,)
                # argpartition is O(K) in expectation vs O(K log K) for
                # full argsort; for K=6 the difference doesn't matter
                # but the contract is identical.
                top_idx = np.argpartition(-row, n_keep - 1)[:n_keep]
                # Sort the top n_keep by score descending so we walk in
                # the order transformers does.  Matters because EOS
                # finalization gates on rank < num_beams.
                top_idx = top_idx[np.argsort(-row[top_idx])]

                # Walk the top n_keep candidates, pick the next num_beams
                # that aren't EOS to seed the next step; finalize EOS
                # candidates whose rank < num_beams as completed hyps.
                beam_slot = 0
                for rank, flat_idx in enumerate(top_idx):
                    src_beam_within_frame = int(flat_idx) // self._vocab_size
                    token_id = int(flat_idx) % self._vocab_size
                    cand_score = float(row[flat_idx])
                    src_global_idx = bi * num_beams + src_beam_within_frame

                    if token_id == eos_id:
                        # Finalize hypothesis only if this EOS would
                        # have made it into the top num_beams.  Skipping
                        # rank>=num_beams EOS expansions matches HF's
                        # is_done short-circuit exactly.
                        if rank >= num_beams:
                            continue
                        full_seq = list(generated_ids[src_global_idx]) + [token_id]
                        # Length normalization: divide by the number of
                        # GENERATED tokens (excludes the leading
                        # decoder_start, which is provided not generated).
                        gen_len = len(full_seq) - 1
                        norm_score = cand_score / max(
                            gen_len ** DEFAULT_LENGTH_PENALTY, 1.0,
                        )
                        finished_hyps[bi].append((norm_score, full_seq))
                    else:
                        next_beam_scores[bi, beam_slot] = cand_score
                        next_beam_tokens[bi, beam_slot] = token_id
                        next_beam_indices[bi, beam_slot] = src_global_idx
                        beam_slot += 1

                    if beam_slot == num_beams:
                        # Filled all next-beam slots; remaining
                        # candidates can't improve the active set.
                        break

                # Frame done iff we have at least num_beams finished
                # hypotheses.  early_stopping=True semantics (Florence's
                # generation_config default).
                if len(finished_hyps[bi]) >= num_beams:
                    done_mask[bi] = True

            if done_mask.all():
                break

            # 6. Reorder decoder-side past_kv along axis 0 by the new
            #    beam indices.  Encoder-side past_kv is constant across
            #    beams of the same frame so we skip its reorder.
            flat_beam_indices = next_beam_indices.reshape(bb)
            past_kv = self._reorder_decoder_kv(past_kv, flat_beam_indices)

            # 7. Update beam state for the next iteration.
            new_generated: list[list[int]] = []
            for bi in range(bsz):
                for j in range(num_beams):
                    src_global = int(next_beam_indices[bi, j])
                    new_generated.append(
                        list(generated_ids[src_global])
                        + [int(next_beam_tokens[bi, j])]
                    )
            generated_ids = new_generated

            beam_scores = next_beam_scores.reshape(bb)
            cur_tokens = next_beam_tokens.reshape(bb, 1)

            # After step 0 we're in the incremental regime forever.
            use_cache_branch = True

        # Finalization: any frame that didn't reach early-stop adds its
        # remaining active beams as hypotheses (so we always have at
        # least num_beams candidates to pick from per frame).
        for bi in range(bsz):
            if done_mask[bi]:
                continue
            for j in range(num_beams):
                src_global = bi * num_beams + j
                seq = generated_ids[src_global]
                gen_len = max(len(seq) - 1, 1)
                norm_score = float(beam_scores[src_global]) / (
                    gen_len ** DEFAULT_LENGTH_PENALTY
                )
                finished_hyps[bi].append((norm_score, list(seq)))

        # Pick the best hypothesis per frame by length-normalized score.
        best_seqs: list[list[int]] = []
        for bi in range(bsz):
            best_score, best_seq = max(
                finished_hyps[bi], key=lambda h: h[0],
            )
            best_seqs.append(best_seq)
        return best_seqs

    @staticmethod
    def _reorder_decoder_kv(
        past_kv: dict[str, np.ndarray],
        beam_indices: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Gather the decoder-side past_kv tensors along axis 0 by beams.

        Args:
            past_kv: Dict produced by :func:`_present_to_past` -- has
                both encoder.* and decoder.* entries.
            beam_indices: ``(B*beams,) int64`` -- the new ordering of
                source beams for the next step.

        Returns:
            New dict where every ``past_key_values.{i}.decoder.*`` entry
            has been gathered by ``beam_indices`` along axis 0; the
            ``past_key_values.{i}.encoder.*`` entries pass through
            unchanged (they're identical across beams of the same
            frame, so reordering is a no-op).
        """
        out: dict[str, np.ndarray] = {}
        for name, arr in past_kv.items():
            if ".decoder." in name:
                # np.take with axis=0 is a single contiguous-row gather;
                # the result is a fresh contiguous buffer (important so
                # ORT's in-place ops on the next step don't alias the
                # previous step's tensor).
                out[name] = np.take(arr, beam_indices, axis=0)
            else:
                # Encoder cross-attn KV: identical across beams of the
                # same frame post-replication; skip the gather.
                out[name] = arr
        return out

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def caption_batch(
        self,
        frames: Sequence[np.ndarray],
        *,
        task: str = "<MORE_DETAILED_CAPTION>",
        num_beams: int = DEFAULT_NUM_BEAMS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> list[str]:
        """Caption a list of pre-cropped 768x768 RGB frames.

        Args:
            frames: List of ``(768, 768, 3) uint8`` ndarrays produced
                by the visual lane's ffmpeg pipeline.  An empty list
                returns ``[]`` without loading any sessions.
            task: Florence-2 task token; one of the keys in
                :data:`_florence_processor.CAPTION_TASKS` for caption
                tasks.  OD/OCR/region tasks intentionally raise here.
            num_beams: Beam width.  Default 3 matches Florence's
                generation_config; 1 reduces to greedy (~1.5-2x
                faster, slight quality drop on detailed captions).
            max_new_tokens: Hard cap on generated tokens per caption.
                Forced EOS fires at this limit if no beam terminates
                naturally.

        Returns:
            List of caption strings, same length as ``frames``.  Empty
            input -> empty output, no model load.

        Raises:
            ValueError: ``task`` is not a recognized caption task.
        """
        if task not in CAPTION_TASKS:
            # OD / OCR / region tasks are intentionally unsupported in
            # this captioner -- they need a coordinate quantizer parser
            # the visual lane doesn't currently call into.  Fail loudly.
            raise ValueError(
                f"FlorenceCaptioner only handles caption tasks today; "
                f"got {task!r}. Caption tasks: {sorted(CAPTION_TASKS)}"
            )
        if not frames:
            return []

        bsz = len(frames)

        # 1. Image normalization + vision pass.  Both batched for
        #    maximum GPU utilization on the vision encoder.
        pixel_values = self._image_processor(frames)  # (B, 3, 768, 768)
        image_features = self._run_vision(pixel_values)  # (B, 577, 768)

        # 2. Tokenize the prompt.  We use the same prompt for every
        #    frame in the batch, so encode_prompts pads to the prompt's
        #    own length (no inter-frame padding needed).
        prompt = construct_prompts(task)
        prompt_ids, prompt_attn = self._tokenizer.encode_prompts(
            [prompt] * bsz,
        )
        prompt_embeds = self._run_embed(prompt_ids)  # (B, L, 768)

        # 3. Merge image features with prompt embeds + masks (along the
        #    sequence axis).  This is _merge_input_ids_with_image_features
        #    in transformers.js Florence2 and MergeInputIdsWithImageFeatures
        #    in curiosity-ai/florence2-sharp.
        merged_embeds = np.concatenate(
            [image_features, prompt_embeds], axis=1,
        )  # (B, 577 + L, 768)
        # Image attention mask is all ones (every image token is real).
        image_attn = np.ones(
            (bsz, NUM_IMAGE_TOKENS), dtype=prompt_attn.dtype,
        )
        merged_attn = np.concatenate([image_attn, prompt_attn], axis=1)

        # 4. Encoder forward.  Produces the cross-attention source for
        #    every decoder step.
        encoder_hidden = self._run_encoder(merged_embeds, merged_attn)

        # 5. Beam-search decoder loop.
        token_seqs = self._beam_search(
            encoder_hidden, merged_attn,
            num_beams=num_beams, max_new_tokens=max_new_tokens,
        )

        # 6. Decode + post-process.
        captions: list[str] = []
        for ids in token_seqs:
            # Strip the leading decoder_start_token (which is just
            # </s>) so the decoder doesn't see a stray BOS-like token
            # at position 0.  Trailing EOS gets stripped by skip_special_tokens.
            if ids and ids[0] == self._tokenizer.decoder_start_token_id:
                ids = ids[1:]
            raw = self._tokenizer.decode_one(ids, skip_special_tokens=False)
            cleaned = post_process_caption(raw, task)[task]
            captions.append(cleaned)
        return captions

    def close(self) -> None:
        """Release the four ORT sessions.

        ORT uses ref-counted destruction; setting attrs to ``None``
        and triggering GC is the documented cleanup path.  Called
        automatically by the multi-session pool on shutdown.
        """
        # ORT InferenceSession has no explicit close; rely on refcount.
        self._vision = None
        self._embed = None
        self._encoder = None
        self._decoder = None


# ---------------------------------------------------------------------------
# Snapshot download helper -- mirrors audio_lane._download_clap_onnx
# ---------------------------------------------------------------------------

def download_florence_onnx(
    model_id: str = "onnx-community/Florence-2-base",
    *,
    dtype: str = "mixed",
    quantized_decoder: bool = False,
) -> Path:
    """Snapshot-download the Florence-2 ONNX repo, return the local path.

    Args:
        model_id: HF Hub repo id.  Defaults to the onnx-community port
            we built this captioner against; not exposed as a CLI knob
            because no other Florence-2 repo on the Hub matches the
            exact ONNX I/O contract we depend on.
        dtype: Match the captioner's :attr:`FlorenceCaptioner.dtype`.
            ``"mixed"`` (default) pulls fp16 vision/embed/encoder +
            fp32 decoder (~620 MB); ``"fp16"`` pulls all-fp16 (~250 MB,
            but the decoder is currently broken upstream and will
            fail to load); ``"fp32"`` pulls all-fp32 (~745 MB, the
            paranoid quality reference).
        quantized_decoder: When True, also pull the
            ``decoder_model_merged_q4f16.onnx`` weight (~56 MB);
            otherwise skip it.  Currently broken upstream in the same
            way as the fp16 decoder.

    Returns:
        Path to the local snapshot directory.  Subsequent calls with
        the same args are essentially free (HF's cache is content-
        addressed).
    """
    from huggingface_hub import snapshot_download

    # Resolve the per-subgraph dtype map up front so this function
    # picks exactly the same files _resolve_onnx_paths() will look
    # for at session-load time.  Matching them up is the whole point
    # of having one constant table at the top of the module.
    dtype_map = _resolve_dtype_map(dtype)

    def _suffix(role: str) -> str:
        return "" if dtype_map[role] == "fp32" else "_fp16"

    # Always pull config + tokenizer + processor JSONs (small, all
    # required).  Pull the matched-per-subgraph variants of the four
    # ONNX files plus any external_data sidecars (the .onnx_data
    # pattern catches the >2 GB-ONNX-spec external-weight files;
    # the fp16 graphs are all weight-inline but the fp32 decoder
    # uses an external data file on disk).
    allow_patterns = [
        "*.json",
        "*.txt",
        f"onnx/vision_encoder{_suffix('vision')}.onnx",
        f"onnx/vision_encoder{_suffix('vision')}.onnx_data",
        f"onnx/embed_tokens{_suffix('embed')}.onnx",
        f"onnx/embed_tokens{_suffix('embed')}.onnx_data",
        f"onnx/encoder_model{_suffix('encoder')}.onnx",
        f"onnx/encoder_model{_suffix('encoder')}.onnx_data",
        f"onnx/decoder_model_merged{_suffix('decoder')}.onnx",
        f"onnx/decoder_model_merged{_suffix('decoder')}.onnx_data",
    ]
    if quantized_decoder:
        allow_patterns.extend([
            "onnx/decoder_model_merged_q4f16.onnx",
            "onnx/decoder_model_merged_q4f16.onnx_data",
        ])

    local = snapshot_download(
        repo_id=model_id,
        allow_patterns=allow_patterns,
    )
    return Path(local)
