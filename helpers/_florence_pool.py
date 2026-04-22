"""Multi-instance Florence-2 captioner pool for parallel image captioning.

Lifts the pattern from ``_onnx_pool.py`` (which manages a pool of
``onnx_asr`` speech models) to manage a pool of
:class:`florence_onnx.FlorenceCaptioner` instances.  Each instance
holds four independent ``InferenceSession``s (vision, embed, encoder,
decoder), so an N-sized pool means 4*N concurrent sessions on the GPU.

Why a pool, not a queue feeding a single captioner:
    A single ``InferenceSession.run()`` call serializes inside ORT
    behind a per-session mutex.  Multiple sessions on the same GPU
    CAN run concurrently because each holds its own CUDA stream and
    the device's hardware scheduler overlaps them as long as we have
    free SMs and free VRAM bandwidth.  For Florence-2-base (which is
    decoder-bottlenecked and uses ~40-60% of a 5090's SMs per beam-3
    decoder step) we get ~1.5-2x stacked speedup at N=2 and modest
    diminishing returns past N=3 on the same card.

Why duplicating the four-subgraph state per worker is OK:
    In the default "mixed" dtype mode (fp16 vision/embed/encoder + fp32
    decoder, the only currently-loadable combination -- see
    florence_onnx.py for why), Florence-2-base ONNX weights are ~620
    MB on disk (vision 184 MB + embed 79 MB + encoder 87 MB + decoder
    270 MB fp32).  ORT's CUDA EP keeps these resident; a 2-instance
    pool resident-only is ~1.25 GB.  Add ~700-900 MB activation arena
    and ~720 MB peak KV cache (fp32 KV) per instance and the
    steady-state ceiling is ~4-4.5 GB for N=2.  Fits comfortably
    alongside the speech lane on a 12 GB+ card.

VRAM-aware sizing:
    The caller passes a *desired* pool size (from ``wealthy.py``).  At
    construction we probe ``vram.detect_gpu().free_gb`` and clamp the
    actual pool size down if the desired N would overflow VRAM.  The
    per-instance footprint constants live in this module rather than
    ``vram.py`` because they're Florence-ONNX-specific (the torch path
    we deleted had a different memory profile -- bigger eager-mode
    activations, no ORT arena).

Public API::

    pool = FlorenceCaptionerPool(model_dir,
                                 desired_size=2,
                                 dtype="mixed",
                                 quantized_decoder=False)
    captions_per_frame = pool.caption_batch(frames, task=...)
    pool.close()  # release captioners (or let the context manager / GC do it)

Each result is the post-processed caption string in input order.
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Any, Sequence

import numpy as np

from _onnx_providers import resolve_providers
from florence_onnx import FlorenceCaptioner, DEFAULT_NUM_BEAMS, DEFAULT_MAX_NEW_TOKENS


# ---------------------------------------------------------------------------
# Per-instance footprint estimates.
#
# Numbers come from nvidia-smi snapshots taken during a 50-frame test
# batch on a 5090 with batch_size=8, num_beams=3, max_new_tokens=256
# -- NOT theoretical param counts.  Caching allocator overhead, CUDA
# context slabs, and IOBinding KV cache all matter and don't appear in
# the parameter count.
#
# Tuned for `onnx-community/Florence-2-base`:
#   * mixed (default): ~1.95 GB per instance peak (~1.25 GB resident
#     weights with fp32 decoder + ~500 MB activations + ~200 MB KV
#     cache).  KV is fp32 so 2x the fp16 number, but per-instance
#     batch is small enough that 200 MB is the right ballpark.
#   * fp16: ~1.6 GB per instance peak -- BROKEN UPSTREAM right now.
#     Kept for the day onnx-community re-exports the decoder cleanly.
#   * fp32: roughly 2x weights + 1.5x activations -> ~2.8 GB per inst.
#   * mixed+q4f16dec: knocks ~150 MB off the resident, so ~1.8 GB
#     peak.  Also currently broken upstream alongside fp16 decoder.
# ---------------------------------------------------------------------------

_PER_INSTANCE_RESIDENT_GB: dict[str, float] = {
    "mixed":          1.25,  # default; fp16 vision/embed/encoder + fp32 decoder
    "fp16":           1.1,   # all four graphs in fp16 (broken upstream)
    "fp32":           2.2,   # all four graphs in fp32 (paranoid mode)
    "mixed+q4f16dec": 0.95,  # quantized decoder; vision/embed/encoder stay fp16
    "fp16+q4f16dec":  0.95,  # legacy alias from before the mixed-dtype mode
}

# Transient peak above resident, hit during a single decoder step's
# beam-batched forward (B*beams=24 with B=8 / beams=3) plus the KV
# cache growth.  We size the pool with this added so the CUDA
# allocator doesn't dance up to the device limit on every batch.
_PER_INSTANCE_TRANSIENT_GB: float = 0.5

# Minimum free VRAM we leave after sizing the pool -- for the desktop
# compositor, the speech lane if it's co-tenanted, and the CUDA
# driver's own bookkeeping.  2 GB matches the speech lane's pool.
_VRAM_HEADROOM_GB: float = 2.0


def _per_instance_peak_gb(dtype: str, quantized_decoder: bool) -> float:
    """Return the peak per-instance VRAM footprint for sizing decisions.

    Adds the transient working set on top of resident weights so we
    don't over-allocate instances that would be fine in steady state
    but OOM during the first decoder step.
    """
    base = dtype.lower()
    key = f"{base}+q4f16dec" if quantized_decoder else base
    resident = _PER_INSTANCE_RESIDENT_GB.get(
        key, _PER_INSTANCE_RESIDENT_GB["mixed"],
    )
    return resident + _PER_INSTANCE_TRANSIENT_GB


# ---------------------------------------------------------------------------
# Pool implementation
# ---------------------------------------------------------------------------

class FlorenceCaptionerPool:
    """N independent FlorenceCaptioner instances, one per worker thread.

    Lifecycle:
        1. ``__init__`` resolves the EP ladder, clamps pool size to fit
           VRAM, and constructs N captioner instances eagerly.
           Construction cost is paid once -- each instance does its
           own ORT init + 4-graph weight load (~2-4s on a warm
           filesystem cache).
        2. ``caption_batch(frames)`` fans out *frame-batches* across
           the N instances via a ThreadPoolExecutor.  Each worker
           takes a chunk of frames and calls
           ``FlorenceCaptioner.caption_batch`` on that chunk.
        3. ``close()`` drops the instance refs, runs gc, and (if torch
           is around) calls cuda.empty_cache to nudge the allocator.

    Thread safety:
        Each FlorenceCaptioner's underlying ORT sessions are
        single-Run-at-a-time internally, so we never share a
        captioner across threads.  The Queue-of-instances ensures
        every worker grabs one captioner, runs, returns it.
        ``caption_batch`` is itself NOT reentrant -- call it from a
        single thread.

    Frame-chunking strategy:
        The caller passes a flat list of frames.  We split into N
        chunks of roughly equal size and dispatch one chunk per
        worker.  This is cheaper than per-frame submission because
        Florence's per-batch overhead (image_processor + vision
        encoder forward + encoder forward) amortizes nicely over
        ~4-8 frames per chunk; per-frame submission would pay that
        overhead N times more often than necessary.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        desired_size: int,
        dtype: str = "mixed",
        quantized_decoder: bool = False,
        prefer_tensorrt: bool | None = None,
    ) -> None:
        """Build the pool of ``desired_size`` captioners, clamped to fit VRAM.

        Args:
            model_dir: Path to a downloaded ``onnx-community/Florence-2-base``
                snapshot.  Forwarded to every ``FlorenceCaptioner``.
            desired_size: Caller's requested pool size.  Will be
                clamped down by available VRAM; clamped up to a
                minimum of 1 so the pool is always usable.
            dtype: ``"mixed"`` (default, recommended), ``"fp16"``
                (broken upstream right now), or ``"fp32"`` (paranoid
                quality reference).  See :class:`FlorenceCaptioner`
                for the full breakdown.
            quantized_decoder: Forwarded verbatim to FlorenceCaptioner.
            prefer_tensorrt: Forwarded to ``resolve_providers``.  When
                ``None`` (default), reads ``VIDEO_USE_FLORENCE_TRT=1``
                from the environment to opt in.  TRT compile is 2-5
                min on first run for the decoder graph; default off.
        """
        # Resolve TRT preference from env if not explicit.  Same env
        # var the speech lane uses for Parakeet (VIDEO_USE_PARAKEET_TRT)
        # but Florence-specific because the trade-off is different
        # (Florence's autoregressive decoder benefits MORE from TRT
        # than Parakeet's encoder does, but compile time is similar).
        if prefer_tensorrt is None:
            import os
            prefer_tensorrt = os.environ.get(
                "VIDEO_USE_FLORENCE_TRT", "0",
            ).lower() in ("1", "true", "yes", "on")

        self._model_dir = Path(model_dir)
        self._dtype = dtype
        self._quantized_decoder = quantized_decoder
        self._prefer_tensorrt = prefer_tensorrt

        # Probe VRAM and clamp pool size BEFORE building any captioners
        # -- otherwise we'd OOM mid-construction and leave half a pool
        # alive in CUDA context limbo.
        target_n = self._clamp_to_vram(
            desired_size, dtype=dtype, quantized_decoder=quantized_decoder,
        )
        if target_n < desired_size:
            print(
                f"  [florence-pool] desired pool size {desired_size} "
                f"clamped to {target_n} by available VRAM "
                f"(per-instance peak ~"
                f"{_per_instance_peak_gb(dtype, quantized_decoder):.1f} GB)"
            )
        self._size = target_n

        # Resolve the EP ladder once and reuse for every instance.
        # Logging happens inside resolve_providers on first call.
        providers = resolve_providers(prefer_tensorrt=prefer_tensorrt)

        head_provider = providers[0] if providers else None
        head_name = (
            head_provider[0] if isinstance(head_provider, tuple)
            else head_provider
        )
        print(
            f"  [florence-pool] loading {target_n} captioner(s) from "
            f"{self._model_dir.name} (dtype={dtype}, "
            f"quantized_decoder={quantized_decoder}, "
            f"head_ep={head_name})"
        )

        t0 = time.time()
        self._instances: list[FlorenceCaptioner] = []
        for i in range(target_n):
            try:
                inst = FlorenceCaptioner(
                    self._model_dir,
                    providers,
                    dtype=dtype,
                    quantized_decoder=quantized_decoder,
                    intra_op_threads=1,  # multi-instance: don't fight over CPU
                )
            except Exception as e:
                # If we got at least one instance built, run with what
                # we have rather than failing the whole pool -- partial
                # parallelism is better than zero parallelism.
                if i == 0:
                    raise
                print(
                    f"  [florence-pool] WARN: instance {i+1}/{target_n} "
                    f"failed to load ({type(e).__name__}: {e}); "
                    f"continuing with {i} instance(s).",
                    file=sys.stderr,
                )
                self._size = i
                break
            self._instances.append(inst)

        # Worker queue -- one slot per instance.  Workers ``_acquire()``
        # an instance, do their caption_batch, then ``_release()`` it
        # back.  Bounded queue gives us natural backpressure: at most N
        # concurrent caption_batch calls, the rest wait on the queue.
        self._queue: "Queue[FlorenceCaptioner]" = Queue(maxsize=self._size)
        for inst in self._instances:
            self._queue.put(inst)

        dt = time.time() - t0
        print(
            f"  [florence-pool] loaded {self._size} captioner(s) in "
            f"{dt:.1f}s (avg {dt / max(1, self._size):.1f}s per instance)"
        )

        # Lock to make close() idempotent + thread-safe even if a
        # worker is mid-caption_batch when shutdown is requested.
        self._closed = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Sizing helper
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_to_vram(
        desired: int,
        *,
        dtype: str,
        quantized_decoder: bool,
    ) -> int:
        """Clamp ``desired`` to fit within current free VRAM minus headroom.

        Returns at least 1 -- even on a CPU-only host we want a usable
        pool.  The CUDA EP simply degrades to CPU EP via the ladder
        when no GPU is present.
        """
        if desired < 1:
            return 1

        # Single source-of-truth VRAM probe -- same one the orchestrator
        # uses to pick a schedule.  Falls back gracefully on systems
        # with no torch/no nvidia-smi.
        try:
            from vram import detect_gpu
            info = detect_gpu()
        except Exception:
            return desired  # can't probe, trust the caller

        if not info.available:
            # CPU-only path.  Keep the pool tiny; Florence is slow on CPU
            # and N>1 doesn't help much because the CPU EP is already
            # multi-threaded internally.
            return min(desired, 2)

        per_peak = _per_instance_peak_gb(dtype, quantized_decoder)
        usable_gb = max(0.0, info.free_gb - _VRAM_HEADROOM_GB)
        max_fitting = max(1, int(usable_gb / per_peak))
        return min(desired, max_fitting)

    # ------------------------------------------------------------------
    # Public: batch caption
    # ------------------------------------------------------------------

    def caption_batch(
        self,
        frames: Sequence[np.ndarray],
        *,
        task: str = "<MORE_DETAILED_CAPTION>",
        num_beams: int = DEFAULT_NUM_BEAMS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        chunk_size: int | None = None,
        on_chunk_complete=None,
    ) -> list[str]:
        """Caption a flat list of frames in parallel across pool workers.

        Args:
            frames: List of ``(768, 768, 3) uint8`` ndarrays produced
                by the visual lane's ffmpeg pipeline.  Empty list
                returns ``[]`` without any work dispatched.
            task: Florence task token; forwarded to the captioner.
            num_beams: Beam width; forwarded to the captioner.
            max_new_tokens: Hard cap on generated tokens; forwarded.
            chunk_size: Frames per dispatch chunk.  When ``None``
                (default), splits into ``size`` roughly-equal chunks
                so every worker stays busy throughout the batch.
                Force a smaller value (e.g. 4) for finer progress
                granularity at the cost of more per-chunk Python
                overhead.
            on_chunk_complete: Optional callback
                ``(chunk_idx, n_frames_in_chunk, captions)`` called
                from a worker thread as each chunk finishes -- useful
                for live progress bars in the lane caller.  Must be
                thread-safe (just print is fine; lane_progress.update
                is fine).

        Returns:
            List of caption strings, same length AND ORDER as ``frames``.
            If a chunk's captioner.caption_batch throws, every slot in
            that chunk is filled with the empty string and the
            exception is printed to stderr -- partial batch is more
            useful than zero batch when one frame is bad.
        """
        if self._closed:
            raise RuntimeError("FlorenceCaptionerPool is closed")
        if not frames:
            return []

        n = len(frames)
        # Pick a chunk size that gives every worker at least one
        # chunk if possible.  Min chunk = 1 (no point splitting
        # below that).  Max chunk = n (single-worker fallback when
        # pool is size 1).
        if chunk_size is None or chunk_size < 1:
            # Round up so we don't end with a tiny straggler chunk.
            chunk_size = max(1, (n + self._size - 1) // self._size)

        # Build (start_idx, end_idx) chunks.  end_idx exclusive.
        chunks: list[tuple[int, int]] = []
        for start in range(0, n, chunk_size):
            chunks.append((start, min(start + chunk_size, n)))

        results: list[str] = [""] * n

        with ThreadPoolExecutor(
            max_workers=self._size,
            thread_name_prefix="florence-worker",
        ) as ex:
            futures = {
                ex.submit(
                    self._run_chunk,
                    chunk_idx,
                    frames,
                    s, e,
                    task, num_beams, max_new_tokens,
                ): chunk_idx
                for chunk_idx, (s, e) in enumerate(chunks)
            }
            for fut in as_completed(futures):
                chunk_idx = futures[fut]
                s, e = chunks[chunk_idx]
                try:
                    chunk_caps = fut.result()
                except Exception as exc:
                    print(
                        f"  [florence-pool] WARN: chunk {chunk_idx} "
                        f"({e - s} frames) failed "
                        f"({type(exc).__name__}: {exc}); leaving "
                        f"empty captions for this slice",
                        file=sys.stderr,
                    )
                    chunk_caps = [""] * (e - s)
                # Splat into the results buffer in input order.
                # `chunk_caps` length == e - s by contract.
                results[s : e] = chunk_caps
                if on_chunk_complete is not None:
                    try:
                        on_chunk_complete(chunk_idx, e - s, chunk_caps)
                    except Exception as cb_err:
                        # Never let a callback error sabotage the batch.
                        print(
                            f"  [florence-pool] WARN: on_chunk_complete "
                            f"raised {type(cb_err).__name__}: {cb_err}",
                            file=sys.stderr,
                        )
        return results

    def _run_chunk(
        self,
        chunk_idx: int,
        all_frames: Sequence[np.ndarray],
        start: int,
        end: int,
        task: str,
        num_beams: int,
        max_new_tokens: int,
    ) -> list[str]:
        """Worker body: acquire an instance, caption the chunk, release.

        The chunk slice is taken from `all_frames` inside the worker so
        the dispatch layer doesn't have to materialize a per-chunk list
        copy ahead of time -- a no-cost slice when the source is a
        Python list of pre-built ndarrays.
        """
        # Discard chunk_idx -- only used by the dispatch layer for
        # logging; the worker itself doesn't need it.
        del chunk_idx
        instance = self._queue.get()
        try:
            chunk = list(all_frames[start:end])
            return instance.caption_batch(
                chunk,
                task=task,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
            )
        finally:
            # ALWAYS return the instance, even on exception, so a bad
            # frame doesn't permanently shrink the usable pool.
            self._queue.put(instance)

    # ------------------------------------------------------------------
    # Public: introspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Effective pool size (after VRAM clamp, may be < desired)."""
        return self._size

    @property
    def model_id(self) -> str:
        """The model identifier these captioners were built for.

        Includes the ``+q4f16dec`` suffix when the quantized decoder
        is in use, so visual_lane's JSON cache invalidates cleanly
        when the user toggles the ``--quantized`` flag.
        """
        # All instances share the same model_id; pick the first.  If
        # the pool failed to build any instances we'd have raised in
        # __init__, so this is always safe.
        return self._instances[0].model_id

    @property
    def dtype(self) -> str:
        """``"mixed"``, ``"fp16"``, or ``"fp32"`` -- dtype the instances were built with."""
        return self._dtype

    @property
    def quantized_decoder(self) -> bool:
        """True iff the q4f16 decoder weight variant is in use."""
        return self._quantized_decoder

    # ------------------------------------------------------------------
    # Public: shutdown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release all captioner handles + nudge the CUDA allocator.

        Idempotent + thread-safe.  Safe to call multiple times.  After
        close() the pool's ``caption_batch`` will raise.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True

            # Drain the queue -- any still-resident instances get dropped.
            try:
                while not self._queue.empty():
                    self._queue.get_nowait()
            except Exception:
                pass

            # Explicit close on each instance so its four ORT sessions
            # release their CUDA contexts BEFORE we drop the strong
            # ref.  ORT's destructors are reliable but explicit is
            # better when there are reference cycles in user code.
            for inst in self._instances:
                try:
                    inst.close()
                except Exception:
                    pass
            self._instances.clear()

            # Run a gc cycle so destructors actually fire (CPython
            # doesn't always run them eagerly when there are refcycles
            # inside ORT's wrappers).
            try:
                import gc
                gc.collect()
            except Exception:
                pass

            # Best-effort CUDA cache flush.  Doesn't matter if torch
            # isn't installed -- this whole module's only torch
            # mention is in this try block.
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def __enter__(self) -> "FlorenceCaptionerPool":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:
        # Defensive -- don't leak captioners if caller forgot to close().
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Single-instance convenience: when desired_size=1, the pool overhead is
# pure cost.  This factory bypasses the threadpool + queue and returns a
# plain FlorenceCaptioner that the visual lane can call directly.
# ---------------------------------------------------------------------------

def make_solo_captioner(
    model_dir: str | Path,
    *,
    dtype: str = "mixed",
    quantized_decoder: bool = False,
    prefer_tensorrt: bool | None = None,
    intra_op_threads: int = 4,
) -> FlorenceCaptioner:
    """Build a single :class:`FlorenceCaptioner` without the pool layers.

    Args:
        model_dir: Path to the ``onnx-community/Florence-2-base`` snapshot.
        dtype: ``"mixed"`` (default), ``"fp16"`` (broken upstream),
            or ``"fp32"`` (paranoid reference).
        quantized_decoder: Use the q4f16 decoder weight variant.
        prefer_tensorrt: Forwarded to ``resolve_providers``.  ``None``
            (default) reads ``VIDEO_USE_FLORENCE_TRT=1`` from env.
        intra_op_threads: 4 by default for the solo path -- single
            captioner has the whole CPU to itself, so we let ORT
            spread its activations across multiple threads (faster
            beam-search reorder + log_softmax host code).

    Returns:
        A ready-to-use :class:`FlorenceCaptioner`.  Caller is
        responsible for calling ``.close()`` (or use the pool's
        managed lifecycle if you want auto-cleanup).
    """
    if prefer_tensorrt is None:
        import os
        prefer_tensorrt = os.environ.get(
            "VIDEO_USE_FLORENCE_TRT", "0",
        ).lower() in ("1", "true", "yes", "on")
    providers = resolve_providers(prefer_tensorrt=prefer_tensorrt)
    return FlorenceCaptioner(
        model_dir,
        providers,
        dtype=dtype,
        quantized_decoder=quantized_decoder,
        intra_op_threads=intra_op_threads,
    )
