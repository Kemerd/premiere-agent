"""ONNX Runtime execution-provider ladder.

Builds the ranked `providers=...` list passed to every
`onnxruntime.InferenceSession` we construct in the speech lane. The
contract:

  * Try the fastest backend first.
  * Each backend gets its own per-EP option dict (TRT workspace, fp16
    flags, CUDA arena enable, etc.).
  * If a backend isn't installed / can't initialise, fall through to
    the next tier WITHOUT crashing the caller.
  * Always end with `CPUExecutionProvider` so any model is at least
    runnable.

Ladder, fastest-to-most-portable:

  ┌────────────────────────┬──────────────────────────┬──────────────────┐
  │ TensorrtExecutionProv. │ gated by env var         │ ~320x RTFx       │
  │                        │ VIDEO_USE_PARAKEET_TRT=1 │ Parakeet TDT 0.6B│
  │                        │ + tensorrt_libs import   │                  │
  ├────────────────────────┼──────────────────────────┼──────────────────┤
  │ CUDAExecutionProvider  │ Ampere+ NVIDIA           │ ~57-100x RTFx    │
  ├────────────────────────┼──────────────────────────┼──────────────────┤
  │ DmlExecutionProvider   │ Windows DirectML         │ ~30-50x RTFx     │
  │                        │ (Intel Arc / AMD / NV)   │                  │
  ├────────────────────────┼──────────────────────────┼──────────────────┤
  │ CPUExecutionProvider   │ always                   │ ~17-30x RTFx     │
  └────────────────────────┴──────────────────────────┴──────────────────┘

Why TensorRT is gated rather than auto-on:
    The TRT EP compiles an engine on the FIRST forward pass for every
    new (model, input-shape) combination. That compile takes 2-5
    minutes and writes a multi-MB engine cache to disk. For one-shot
    transcribe jobs the compile dominates wall time; for long-running
    services the amortized cost is fine. We let the user opt in
    explicitly so they're not surprised by a 5-minute first-run hang
    when they thought CUDA EP was already fast enough (it usually is).

Why CUDA -> DirectML -> CPU rather than CUDA -> CPU:
    DirectML beats CPU by 2-3x on Intel Arc / Iris Xe and AMD APUs,
    which a non-trivial slice of "video editor" users have. On systems
    where DML isn't available the import probe fails cheaply and we
    fall through to CPU.

Public API:
    from _onnx_providers import resolve_providers
    providers = resolve_providers(prefer_tensorrt=True)
    # providers -> [("TensorrtExecutionProvider", {...}),
    #               ("CUDAExecutionProvider",     {...}),
    #               "CPUExecutionProvider"]
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Truthy values for env-var gating. Same vocabulary as wealthy.py /
# vram.py so users only have to learn one set of "yes" strings.
_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


# ---------------------------------------------------------------------------
# NVIDIA DLL bootstrap (Windows only)
#
# The `tensorrt-cu12-libs` and `nvidia-cudnn-cu12` pip packages ship the
# native runtime DLLs (nvinfer_10.dll, cudnn64_9.dll, etc.) inside their
# site-packages folders, but they do NOT register those folders with the
# OS loader. Result: when `onnxruntime`'s provider bridge calls
# `LoadLibraryW("onnxruntime_providers_cuda.dll")`, the OS looks for its
# transitive deps (`cudnn64_9.dll`, `nvinfer_10.dll`) on PATH and the
# Python-visible DLL search dirs — neither of which contains them — and
# fails with WinError 126 ("specified module could not be found"). ORT
# silently falls back to CPU EP, so users get ~20x RTFx instead of the
# 200-300x the CUDA/TRT EP would have delivered.
#
# Two-pronged fix is required, learned the hard way:
#
#   1. `os.add_dll_directory()` — covers DLLs the Python interpreter
#      itself loads. Necessary but not sufficient: ORT's provider
#      bridge uses `LoadLibraryW` with the default search path which
#      bypasses DLL directories added via the `AddDllDirectory` API on
#      some Windows builds (depends on whether the EP DLL is loaded
#      with `LOAD_LIBRARY_SEARCH_*` flags or default semantics).
#
#   2. PATH prepend — guarantees the OS loader sees the directories no
#      matter how the EP DLL was loaded. The classic workaround that
#      every Windows-based ORT user eventually rediscovers.
#
# Both pip packages need to be visible:
#   * tensorrt_libs/                 (nvinfer_10.dll + plugin DLLs)
#   * nvidia/cudnn/bin/              (cudnn64_9.dll + cnn/ops DLLs)
#
# We do this at module import time so any caller of `resolve_providers`
# is guaranteed to get a working EP. Idempotent — repeated import is a
# no-op via a module-level "done" flag.
# ---------------------------------------------------------------------------

_NVIDIA_DLL_BOOTSTRAP_DONE = False


def _bootstrap_nvidia_dlls() -> None:
    """Make pip-shipped NVIDIA runtime DLLs visible to the OS loader.

    Without this, ORT's CUDA/TensorRT EPs silently fail to load on
    Windows installs that rely on the `tensorrt-cu12-libs` and
    `nvidia-cudnn-cu12` pip wheels (the recommended install path —
    no system-wide CUDA / cuDNN required).

    Idempotent. Safe to call before or after `import onnxruntime` —
    ORT only loads the EP-specific DLLs at `InferenceSession()` time,
    not at module import.
    """
    global _NVIDIA_DLL_BOOTSTRAP_DONE
    if _NVIDIA_DLL_BOOTSTRAP_DONE:
        return

    # Non-Windows: POSIX dynamic linker uses LD_LIBRARY_PATH /
    # rpath semantics that the wheels handle correctly via auditwheel.
    # Nothing for us to do — bail early so we don't pollute env on
    # Linux/macOS containers where the same code might run.
    if sys.platform != "win32":
        _NVIDIA_DLL_BOOTSTRAP_DONE = True
        return

    # Walk a list of (probe_module, sub_path) candidates. Each probe is
    # an importable module whose __file__ tells us where pip extracted
    # the wheel; sub_path is the directory under it where the actual
    # DLLs live (some wheels nest under bin/, some don't).
    #
    # We probe lazily — a missing wheel is not an error here, it just
    # means that EP won't be available. resolve_providers() handles
    # the "EP not available" path gracefully by falling through.
    candidates: list[tuple[str, str]] = [
        # TensorRT runtime — nvinfer_10.dll lives at the package root.
        ("tensorrt_libs", ""),
        # cuDNN — cudnn64_9.dll lives under the bin/ subfolder.
        ("nvidia.cudnn", "bin"),
        # CUDA runtime (cudart64_12.dll) — newer ORT builds want this
        # too for arena allocator init. Some installs get it via the
        # `nvidia-cuda-runtime-cu12` wheel; others bundle it inside
        # cuDNN's bin/. Probe both names so we don't miss it.
        ("nvidia.cuda_runtime", "bin"),
        ("nvidia.cublas", "bin"),
    ]

    found_dirs: list[str] = []
    for mod_name, sub_path in candidates:
        try:
            mod = __import__(mod_name, fromlist=["__file__"])
        except ImportError:
            # Wheel not installed — skip silently. Each EP that needed
            # it will fall through to the next tier in resolve_providers.
            continue
        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            continue
        base_dir = os.path.dirname(mod_file)
        target = os.path.join(base_dir, sub_path) if sub_path else base_dir
        if os.path.isdir(target):
            found_dirs.append(target)

    if not found_dirs:
        # No NVIDIA wheels installed — user is presumably on a machine
        # with system-wide CUDA / cuDNN, or running CPU-only. Either
        # way we have nothing to add and resolve_providers() is fine.
        _NVIDIA_DLL_BOOTSTRAP_DONE = True
        return

    # Prong 1: prepend to PATH so the OS loader finds the DLLs no
    # matter which LoadLibrary variant ORT's bridge ends up using.
    # Prepend (not append) so our DLLs win against any conflicting
    # system-wide install of an older cuDNN / CUDA.
    current_path = os.environ.get("PATH", "")
    new_path_parts = found_dirs + ([current_path] if current_path else [])
    os.environ["PATH"] = os.pathsep.join(new_path_parts)

    # Prong 2: register with the Python DLL search list. Belt-and-
    # suspenders — covers the case where ORT loads a DLL via the
    # `LOAD_LIBRARY_SEARCH_USER_DIRS` flag set, which ignores PATH
    # entirely and only consults `AddDllDirectory` registrations.
    for d in found_dirs:
        try:
            os.add_dll_directory(d)
        except (OSError, AttributeError):
            # add_dll_directory missing on Python <3.8 (we require 3.10
            # so this won't fire) or the directory already registered.
            # Either way, the PATH prepend above is the load-bearing
            # mechanism — the add_dll_directory call is just insurance.
            pass

    _NVIDIA_DLL_BOOTSTRAP_DONE = True
    # One-line debug breadcrumb so users grepping for `[providers]` see
    # what got bound. Kept to a single short line — the EP ladder log
    # below is the primary signal anyway.
    print(
        f"  [providers] bootstrapped NVIDIA DLL dirs: "
        f"{', '.join(os.path.basename(d) or d for d in found_dirs)}"
    )


# Run the bootstrap at module import time. Every code path into the
# EP ladder goes through `resolve_providers()` defined below, which is
# in this same module — so by the time anything constructs an
# `InferenceSession` the DLL search has already been fixed up. We do
# this at import (rather than inside `resolve_providers`) because some
# adapters (notably onnx-asr's) construct a session before they touch
# our resolver, e.g. for the bundled VAD model.
_bootstrap_nvidia_dlls()


# ---------------------------------------------------------------------------
# Per-EP option builders.
#
# Each builder returns a (name, options-dict) tuple in the exact shape
# `onnxruntime.InferenceSession(..., providers=[...])` accepts. Options
# are deliberately conservative — onnx-asr / Parakeet works fine with
# defaults, but a few knobs (TRT workspace size, fp16 enable, CUDA
# arena) are worth setting explicitly so behavior is reproducible
# across machines that have different ORT defaults compiled in.
# ---------------------------------------------------------------------------

def _trt_options() -> dict[str, Any]:
    """Options for `TensorrtExecutionProvider`.

    Workspace 6 GB is the sweet spot for Parakeet TDT 0.6B on a 24 GB+
    card — enough for the encoder's largest intermediate tensors at
    fp16 with a 30s chunk, without starving the rest of the device.
    Engine cache lives under <tempdir>/video_use_trt_cache so repeat
    runs reuse the compiled engine (the 5-minute first-run hit becomes
    a ~50ms load on subsequent sessions).
    """
    import tempfile
    cache_dir = os.path.join(tempfile.gettempdir(), "video_use_trt_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return {
        # 6 GB workspace — fits Parakeet's transient tensors with margin
        # while leaving room for other lanes if the schedule allows.
        "trt_max_workspace_size": 6 * (1024 ** 3),
        # fp16 engine. Parakeet TDT was trained bf16 so fp16 inference
        # is within rounding noise on the librispeech-clean eval suite.
        "trt_fp16_enable": True,
        # Persist compiled engines so subsequent runs skip the 2-5
        # minute compile. Engine files are keyed by model+shape hash
        # so different audio durations get separate engines (this is
        # fine — silero VAD chunks audio to fixed-ish window sizes).
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache_dir,
    }


def _cuda_options() -> dict[str, Any]:
    """Options for `CUDAExecutionProvider`.

    Arena allocator is enabled by default — kept that way so frequent
    small allocations during chunked inference don't churn the driver
    allocator. `cudnn_conv_algo_search=DEFAULT` picks the fastest
    convolution kernel per shape, cached after the first call.
    """
    return {
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "DEFAULT",
        # do_copy_in_default_stream=true keeps host->device memcpy on
        # the same stream as compute, avoiding a stream sync per chunk.
        "do_copy_in_default_stream": True,
    }


def _dml_options() -> dict[str, Any]:
    """Options for `DmlExecutionProvider`.

    DirectML on Windows targets D3D12 — the only knob worth setting
    is `device_id`, which we leave at 0 (the primary GPU). Multi-GPU
    DML users can override via DML_DEVICE_ID env var if they ever
    surface; for now keeping it implicit.
    """
    return {}


# ---------------------------------------------------------------------------
# Probe helpers — cheap, non-throwing checks for each EP's availability.
# ---------------------------------------------------------------------------

def _ort_available_providers() -> list[str]:
    """Return ORT's view of the installed/enabled providers.

    Returns an empty list if onnxruntime isn't importable at all (which
    means the speech lane should fall back to the NeMo path long before
    we reach this code, but we handle it defensively).
    """
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except ImportError:
        return []


def _tensorrt_libs_importable() -> bool:
    """True if `tensorrt_libs` (the runtime DLLs/SO bundle) is importable.

    The TRT EP needs the libnvinfer / nvinfer.dll runtime loaded into
    the process before `InferenceSession` is constructed — `import
    tensorrt_libs` triggers that load via its module-init side-effect
    (this is the documented onnx-asr recipe; see their TensorRT usage
    page). If the import fails we silently skip TRT.
    """
    try:
        import tensorrt_libs  # noqa: F401  -- side-effect import
        return True
    except ImportError:
        return False


def _trt_enabled() -> bool:
    """True if the user opted into TensorRT EP.

    Two gates: env var must be truthy AND tensorrt_libs must be
    importable. Both must hold — opting in without the libs is a
    config error we surface via a one-line warning rather than a
    crash, because the lane still works fine on CUDA.
    """
    raw = os.environ.get("VIDEO_USE_PARAKEET_TRT", "").strip().lower()
    if raw not in _TRUTHY:
        return False
    if not _tensorrt_libs_importable():
        # User asked for TRT but the runtime libs aren't installed.
        # Single line to stderr so they know why we silently fell back
        # to CUDA — never raise, because CUDA EP is still very fast.
        print(
            "  [providers] VIDEO_USE_PARAKEET_TRT=1 set but "
            "`tensorrt_libs` is not importable. Falling back to "
            "CUDA EP. Install with:  pip install tensorrt-cu12-libs",
            file=sys.stderr,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Public entry point — `resolve_providers()` — module-cached
# ---------------------------------------------------------------------------

# Lazily-built ladder; same one is reused for every session in the pool
# so we don't re-probe + re-log on each construction. Keyed by
# `prefer_tensorrt` because the boolean changes the result.
_LADDER_CACHE: dict[bool, list] = {}


def resolve_providers(prefer_tensorrt: bool = True) -> list:
    """Return the ranked provider list for ORT InferenceSession.

    Args:
        prefer_tensorrt: If True (default), include the TRT EP at the
            top of the ladder when both VIDEO_USE_PARAKEET_TRT=1 AND
            `tensorrt_libs` is importable. If False, skip the TRT
            check entirely (used by callers who know their workload
            doesn't benefit from TRT, e.g. tiny audio clips where the
            engine compile dwarfs the inference).

    Returns:
        A list suitable for `InferenceSession(providers=...)`. Each
        entry is either a bare provider-name string (for providers
        with no per-EP options, like CPU) or a `(name, options)`
        tuple. Always ends with "CPUExecutionProvider" so the model
        is guaranteed to be runnable.

    The first call probes the environment + emits a one-line summary
    of the chosen ladder to stderr so users can see what backend is
    actually running. Subsequent calls hit the module cache.
    """
    if prefer_tensorrt in _LADDER_CACHE:
        return list(_LADDER_CACHE[prefer_tensorrt])  # defensive copy

    # Probe ORT's compiled-in provider list once. Empty = no ORT, but
    # we still build a CPU-only ladder so callers get something
    # consistent to pass through.
    available = set(_ort_available_providers())
    ladder: list = []

    # ── Tier 1: TensorRT (gated) ──────────────────────────────────────
    # Only added when the user opted in AND tensorrt_libs imports.
    # Even when added it's NOT the only provider — we keep CUDA right
    # behind it as the per-shape fallback (TRT engine compile failures
    # are silent in some ORT builds; CUDA EP picks up automatically).
    if (
        prefer_tensorrt
        and "TensorrtExecutionProvider" in available
        and _trt_enabled()
    ):
        ladder.append(("TensorrtExecutionProvider", _trt_options()))

    # ── Tier 2: CUDA ──────────────────────────────────────────────────
    if "CUDAExecutionProvider" in available:
        ladder.append(("CUDAExecutionProvider", _cuda_options()))

    # ── Tier 3: DirectML (Windows) ────────────────────────────────────
    # Sits below CUDA so an NVIDIA-on-Windows user gets CUDA, not DML;
    # but for an Intel Arc / AMD user on Windows where CUDA isn't
    # available, DML is a 2-3x speedup over CPU.
    if "DmlExecutionProvider" in available:
        ladder.append(("DmlExecutionProvider", _dml_options()))

    # ── Tier 4: CPU (always) ──────────────────────────────────────────
    # Bare string (no options dict) since we accept ORT defaults — the
    # CPU EP's intra/inter-op thread counts come from `SessionOptions`,
    # not from per-EP options, so there's nothing meaningful to set
    # here.
    ladder.append("CPUExecutionProvider")

    # ── One-time summary ──────────────────────────────────────────────
    # Print exactly the names so a user grepping for `[providers]` in
    # the lane log sees the chosen ladder without ANSI noise.
    names = [p[0] if isinstance(p, tuple) else p for p in ladder]
    print(f"  [providers] resolved EP ladder: {' -> '.join(names)}")

    _LADDER_CACHE[prefer_tensorrt] = list(ladder)
    return ladder


# ---------------------------------------------------------------------------
# Smoke test — `python helpers/_onnx_providers.py`
# ---------------------------------------------------------------------------

def main() -> None:
    """Print the resolved ladder + raw ORT availability for diagnostics."""
    print("ORT installed providers:", _ort_available_providers())
    print("VIDEO_USE_PARAKEET_TRT  :", os.environ.get("VIDEO_USE_PARAKEET_TRT", ""))
    print("tensorrt_libs importable:", _tensorrt_libs_importable())
    print("ladder (prefer_trt=True):", resolve_providers(prefer_tensorrt=True))
    print("ladder (prefer_trt=False):", resolve_providers(prefer_tensorrt=False))


if __name__ == "__main__":
    main()
