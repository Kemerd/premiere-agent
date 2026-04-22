"""Multi-session ONNX Runtime pool for parallel speech recognition.

Loads N independent `onnx_asr` model instances at startup, each with
its own `onnxruntime.InferenceSession` handle. ORT releases the GIL
during its native `Run()` call, so a `ThreadPoolExecutor(max_workers=N)`
dispatching to N threads ⇒ N truly-parallel native inferences on the
GPU: one session per worker, isolated CUDA streams, shared model
weights duplicated per session (Parakeet TDT 0.6B is small enough
that this is fine — at fp16 each duplicate is ~1.2 GB resident).

Why a pool, not a queue:
    A single InferenceSession can only Run() one request at a time —
    internally it serializes Run() calls behind a per-session mutex.
    Multiple sessions on the same GPU CAN run concurrently because
    each holds its own CUDA stream; the device's hardware scheduler
    overlaps them as long as we have unused SMs and unused VRAM
    bandwidth. For Parakeet (which is encoder-bottlenecked and only
    uses ~30% of a 5090's SMs per inference) we get near-linear
    scaling up to N=8.

VRAM-aware sizing:
    The caller passes a *desired* pool size (from `wealthy.py`). At
    construction we probe `vram.detect_gpu().free_gb` and clamp the
    actual pool size down if the desired N would overflow VRAM. The
    per-session footprint constants live in this module rather than
    `vram.py` because they're ONNX-specific (NeMo's torch-mode
    Parakeet has a totally different memory profile).

Public API:
    pool = OnnxSessionPool(model_id="nemo-parakeet-tdt-0.6b-v2",
                           desired_size=4)
    results = pool.transcribe_batch([wav1, wav2, wav3, ...])
    pool.close()  # release sessions explicitly (or let GC do it)

Each result is the raw `TimestampedResult` (or `TimestampedSegmentResult`
list) that `onnx_asr` returns; the lane module is responsible for
canonicalizing it into the project's word/spacing JSON shape.
"""

from __future__ import annotations

import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Sibling helpers — both are pure-stdlib so they're cheap to import even
# from inside a heavy lane subprocess.
from _onnx_providers import resolve_providers


# ---------------------------------------------------------------------------
# Per-session footprint estimates.
#
# Values tuned for Parakeet TDT 0.6B as exported by the istupakov/
# parakeet-tdt-0.6b-v2-onnx repo (and the v3 multilingual variant,
# which is the same architecture). Numbers come from nvidia-smi
# snapshots taken during a 30-clip test batch on a 5090, NOT from
# theoretical param counts — caching allocator overhead, CUDA context
# slabs, and TRT engine cache all matter and don't appear in the
# parameter count.
# ---------------------------------------------------------------------------

# Resident weights + always-allocated CUDA buffers per session.
# Doubles for fp32, halves for int8.
_PER_SESSION_RESIDENT_GB = {
    "fp16": 1.2,
    "fp32": 2.4,
    "int8": 0.6,
}

# Transient peak above resident, hit during a single Run() call's
# encoder forward. We size the pool with this added so the CUDA
# allocator doesn't dance up to the device limit on every chunk.
_PER_SESSION_TRANSIENT_GB = 0.4

# Minimum free VRAM we leave after sizing the pool — for the desktop
# compositor, the audio + visual lanes if they're co-tenanted, and
# the CUDA driver's own bookkeeping. 2 GB is the empirical floor
# below which Windows starts swapping graphics memory to system RAM.
_VRAM_HEADROOM_GB = 2.0


def _per_session_peak_gb(quantization: str | None) -> float:
    """Return the peak per-session VRAM footprint for sizing decisions.

    Adds the transient working set on top of resident weights so we
    don't over-allocate sessions that would be fine in steady state
    but OOM during the first encoder forward.
    """
    key = (quantization or "fp16").lower()
    resident = _PER_SESSION_RESIDENT_GB.get(key, _PER_SESSION_RESIDENT_GB["fp16"])
    return resident + _PER_SESSION_TRANSIENT_GB


# ---------------------------------------------------------------------------
# Pool implementation
# ---------------------------------------------------------------------------

class OnnxSessionPool:
    """N independent onnx_asr models, one per worker thread.

    Lifecycle:
        1. `__init__` resolves the EP ladder, clamps pool size to fit
           VRAM, and constructs N model instances eagerly. Construction
           cost is paid once — each session does its own ORT init +
           weight load (~1-2s on a warm filesystem cache).
        2. `transcribe_batch(wavs)` fans out across the N sessions via
           a ThreadPoolExecutor. Each worker pulls one wav at a time.
        3. `close()` drops the session refs, runs gc, and (if torch
           is around) calls cuda.empty_cache to nudge the allocator.

    Thread safety:
        Each InferenceSession is single-Run-at-a-time internally, so
        we never share a session across threads. The Queue-of-sessions
        below ensures every worker grabs one session, runs, returns
        it. `transcribe_batch` is itself NOT reentrant — call it from
        a single thread.
    """

    def __init__(
        self,
        model_id: str,
        *,
        desired_size: int,
        quantization: str | None = None,
        prefer_tensorrt: bool = True,
    ) -> None:
        """Build the pool of `desired_size` sessions, clamped to fit VRAM.

        Args:
            model_id: onnx-asr model name (e.g. "nemo-parakeet-tdt-0.6b-v2").
                Anything `onnx_asr.load_model()` accepts.
            desired_size: Caller's requested pool size. Will be clamped
                down by available VRAM; clamped up to a minimum of 1
                so the pool is always usable.
            quantization: None for fp16 default, or "int8" for the
                quantized variant. Forwarded verbatim to onnx_asr.
            prefer_tensorrt: Forwarded to `resolve_providers`. False
                if the caller knows TRT compile is too expensive for
                their workload.
        """
        # Lazy-import onnx_asr so this module's top-level `import _onnx_pool`
        # stays cheap (matters for the orchestrator's lane-dispatch shim
        # which imports many siblings to read constants).
        try:
            import onnx_asr  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "onnx-asr is not installed. Add the [onnx] extra:\n"
                "    pip install -e .[onnx]\n"
                "or pin to the older NeMo path with:\n"
                "    set VIDEO_USE_SPEECH_LANE=nemo"
            ) from e

        self._model_id = model_id
        self._quantization = quantization

        # Probe VRAM and clamp pool size BEFORE building any sessions —
        # otherwise we'd OOM mid-construction and leave half a pool
        # alive in CUDA context limbo.
        target_n = self._clamp_to_vram(desired_size, quantization)
        if target_n < desired_size:
            print(
                f"  [pool] desired pool size {desired_size} clamped to "
                f"{target_n} by available VRAM "
                f"(per-session peak ~{_per_session_peak_gb(quantization):.1f} GB)"
            )

        self._size = target_n

        # Resolve the EP ladder once and reuse for every session. The
        # logging happens inside resolve_providers on first call.
        providers = resolve_providers(prefer_tensorrt=prefer_tensorrt)

        # ── Decide whether we need VAD chunking ──────────────────────
        # VAD is a TRT-only requirement. Quick recap:
        #
        #   * The TensorRT EP compiles the Parakeet encoder with a
        #     fixed optimization profile (typically ≤30s @ 16 kHz =
        #     480000 samples). Long audio violates the profile and
        #     ORT-TRT raises EP_FAIL at Run() time.
        #   * silero VAD splits the waveform into speech-bounded
        #     windows that always fit the profile.
        #
        # The catch: onnx-asr's `.with_vad()` adapter chains AFTER
        # `.with_timestamps()` and CHANGES the result shape. Instead
        # of a single `TimestampedResult` (with token + timestamp
        # parallel lists for true word-level timing), it returns a
        # generator of `SegmentResult` objects that ONLY carry
        # segment-level (start, end, text). All word timing is lost —
        # which is unacceptable for the editor's cut/phrase logic
        # downstream.
        #
        # Therefore: only enable VAD when TRT is actually at the head
        # of the ladder. CUDA / CPU EPs handle dynamic shapes natively
        # and prefer the richer per-token timing path. This was the
        # silent root cause of the long-standing "transcripts come
        # back empty" bug — every CUDA-EP run was being fed through
        # the VAD generator path, then the canonical extractor (which
        # walks `result.timestamps`) saw `None` because the result
        # was a generator and dropped everything.
        head_provider = providers[0] if providers else None
        head_name = (
            head_provider[0] if isinstance(head_provider, tuple)
            else head_provider
        )
        use_vad = head_name == "TensorrtExecutionProvider"
        self._uses_vad = use_vad  # exposed for the lane's extractor

        print(
            f"  [pool] loading {target_n} session(s) of "
            f"{model_id} (quant={quantization or 'fp16'}, "
            f"vad={'on' if use_vad else 'off'})"
        )

        # Pre-load a single shared silero VAD instance — only when we
        # actually need it. Loading the VAD adds ~2 MB and one HF
        # round-trip; skipping it on CUDA/CPU runs is a small but
        # honest startup-time win.
        vad = onnx_asr.load_vad("silero") if use_vad else None

        t0 = time.time()
        self._sessions: list[Any] = []
        for i in range(target_n):
            try:
                # Build the recognition pipeline. `.with_timestamps()`
                # is always on — it's how we get word-level data
                # (or segment-level when VAD is also on). `.with_vad()`
                # is conditional per the comment block above.
                model = (
                    onnx_asr.load_model(
                        model_id,
                        quantization=quantization,
                        providers=providers,
                    )
                    .with_timestamps()
                )
                if vad is not None:
                    model = model.with_vad(vad)
            except Exception as e:
                # If we got at least one session built, run with what
                # we have rather than failing the whole pool — partial
                # parallelism is better than zero parallelism.
                if i == 0:
                    raise
                print(
                    f"  [pool] WARN: session {i+1}/{target_n} failed "
                    f"to load ({type(e).__name__}: {e}); continuing "
                    f"with {i} session(s).",
                    file=sys.stderr,
                )
                self._size = i
                break
            self._sessions.append(model)

        # Worker queue — one slot per session. Workers `_acquire()` a
        # session, do their Run(), then `_release()` it back. Using a
        # bounded queue gives us natural backpressure: at most N
        # concurrent recognize() calls, the rest wait on the queue.
        from queue import Queue
        self._queue: "Queue[Any]" = Queue(maxsize=self._size)
        for s in self._sessions:
            self._queue.put(s)

        dt = time.time() - t0
        print(
            f"  [pool] loaded {self._size} session(s) in {dt:.1f}s "
            f"(avg {dt / max(1, self._size):.1f}s per session)"
        )

        # Lock to make close() idempotent + thread-safe even if a
        # worker is mid-transcribe when shutdown is requested.
        self._closed = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Sizing helper
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_to_vram(desired: int, quantization: str | None) -> int:
        """Clamp `desired` to fit within current free VRAM minus headroom.

        Returns at least 1 — even on a CPU-only host we want a usable
        pool. The CUDA EP simply degrades to CPU EP via the ladder
        when no GPU is present.
        """
        if desired < 1:
            return 1

        # Single-source-of-truth VRAM probe — same one the orchestrator
        # uses to pick a schedule. Falls back gracefully on systems
        # with no torch/no nvidia-smi.
        try:
            from vram import detect_gpu
            info = detect_gpu()
        except Exception:
            return desired  # can't probe, trust the caller

        if not info.available:
            # CPU-only path. We still allow N>1 because the CPU EP
            # will run on multiple threads anyway, and onnx-asr is
            # cheap enough on CPU to make a small pool worthwhile.
            return min(desired, 4)

        per_peak = _per_session_peak_gb(quantization)
        usable_gb = max(0.0, info.free_gb - _VRAM_HEADROOM_GB)
        max_fitting = max(1, int(usable_gb / per_peak))
        return min(desired, max_fitting)

    # ------------------------------------------------------------------
    # Public: batch transcribe
    # ------------------------------------------------------------------

    def transcribe_batch(
        self,
        wav_paths: list[Path],
        *,
        on_complete=None,
    ) -> list[Any]:
        """Recognize a batch of WAVs in parallel, preserving input order.

        Args:
            wav_paths: list of WAV file paths. Empty list returns [].
            on_complete: optional callback `(idx, wav, result)` called
                from a worker thread as each WAV finishes — useful for
                live progress bars in the lane caller. Must be
                thread-safe (just print is fine; tqdm is fine).

        Returns:
            A list of onnx_asr `TimestampedResult` objects, one per
            input WAV, in the same order as `wav_paths`. If an
            individual WAV's recognize() throws, the corresponding
            slot holds the Exception object instead — caller decides
            whether to re-raise (the lane wraps each into a JSON
            "transcribe failed" entry rather than crashing the batch).
        """
        if self._closed:
            raise RuntimeError("OnnxSessionPool is closed")
        if not wav_paths:
            return []

        results: list[Any] = [None] * len(wav_paths)

        # ThreadPoolExecutor max_workers = pool size. Submitting more
        # than N work items is fine — the executor itself queues them,
        # and the session-queue inside _run_one blocks workers waiting
        # for a free session.
        with ThreadPoolExecutor(
            max_workers=self._size,
            thread_name_prefix="onnx-asr-worker",
        ) as ex:
            futures = {
                ex.submit(self._run_one, i, p): i
                for i, p in enumerate(wav_paths)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    # Capture the exception in the slot; let caller
                    # decide whether to fail the batch or skip the clip.
                    results[i] = e
                if on_complete is not None:
                    try:
                        on_complete(i, wav_paths[i], results[i])
                    except Exception as cb_err:
                        # Never let a callback error sabotage the batch.
                        print(
                            f"  [pool] WARN: on_complete callback "
                            f"raised {type(cb_err).__name__}: {cb_err}",
                            file=sys.stderr,
                        )
        return results

    def _run_one(self, idx: int, wav_path: Path) -> Any:
        """Worker body: acquire a session, recognize, release."""
        session = self._queue.get()
        try:
            # `recognize` accepts either a single path or a list. We
            # pass a single path because we want a single result back
            # (onnx-asr's batch-mode returns a list which would force
            # us to unwrap it here anyway).
            return session.recognize(str(wav_path))
        finally:
            # ALWAYS return the session, even on exception, so a bad
            # WAV doesn't permanently shrink the usable pool.
            self._queue.put(session)

    # ------------------------------------------------------------------
    # Public: introspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Effective pool size (after VRAM clamp, may be < desired)."""
        return self._size

    @property
    def model_id(self) -> str:
        """The model identifier this pool was built for."""
        return self._model_id

    @property
    def uses_vad(self) -> bool:
        """True if the pool wraps recognize() in silero VAD chunking.

        Toggled on automatically when the TensorRT EP is at the head
        of the ladder (TRT needs fixed-shape inputs). The lane's
        result extractor uses this to pick between the per-token
        path (rich word timestamps) and the per-segment path
        (segment-level start/end only).
        """
        return self._uses_vad

    # ------------------------------------------------------------------
    # Public: shutdown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release all session handles + nudge the CUDA allocator.

        Idempotent + thread-safe. Safe to call multiple times. After
        close() the pool's `transcribe_batch` will raise.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True

            # Drain the queue — any still-resident sessions get dropped.
            # We don't need to .close() each session: ORT's
            # InferenceSession releases its CUDA context in __del__.
            try:
                while not self._queue.empty():
                    self._queue.get_nowait()
            except Exception:
                pass

            # Drop strong refs and run a gc cycle so the destructors
            # actually fire (CPython doesn't always run them eagerly
            # when there are reference cycles inside ORT's wrappers).
            self._sessions.clear()
            try:
                import gc
                gc.collect()
            except Exception:
                pass

            # Best-effort CUDA cache flush. Doesn't matter if torch
            # isn't installed — onnx_asr doesn't import it either.
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def __enter__(self) -> "OnnxSessionPool":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:
        # Defensive — don't leak sessions if caller forgot to close().
        try:
            self.close()
        except Exception:
            pass
