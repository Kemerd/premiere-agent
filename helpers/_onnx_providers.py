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

# Truthy / falsy values for env-var gating. Same vocabulary as
# wealthy.py / vram.py so users only have to learn one set of strings.
# We support BOTH so users can force-disable an EP that's auto-on by
# default (e.g. set VIDEO_USE_PARAKEET_TRT=0 to skip the engine compile
# step on a one-shot transcribe even though the host has TRT installed).
_TRUTHY = {"1", "true", "yes", "on", "y", "t"}
_FALSY  = {"0", "false", "no", "off", "n", "f"}


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

# Capabilities discovered by `_bootstrap_nvidia_dlls()`. Populated once at
# module import and consumed by both `_trt_enabled()` and `resolve_providers()`
# so we never claim an EP we couldn't actually instantiate. The shape stays
# tiny on purpose — booleans + a debug list of which dirs we wired up. It is
# NOT a per-EP option dict; per-EP options live in `_trt_options()` etc.
_CAPS: dict[str, Any] = {
    "platform": sys.platform,    # for diagnostic output only
    "tensorrt": False,           # nvinfer_10.dll reachable
    "cuda":     False,           # cudart64_12.dll reachable
    "cudnn":    False,           # cudnn64_9.dll reachable
    "msvc":     False,           # vcruntime140*.dll reachable in System32
    "added_dirs": [],            # dirs we prepended to PATH (debug)
}


def _bootstrap_nvidia_dlls() -> None:
    """Make NVIDIA runtime DLLs visible to the OS loader AND populate `_CAPS`.

    Walks two parallel sources for each DLL family:

        1. Pip wheels (tensorrt-cu12-libs, nvidia-cudnn-cu12, etc.) —
           the recommended path, lives under site-packages and is the
           only thing a clean `pip install -e .` produces.

        2. System CUDA Toolkit (`%CUDA_PATH%`, `%CUDA_HOME%`, and the
           globbed `C:\\Program Files\\NVIDIA GPU Computing Toolkit\\
           CUDA\\v12.*\\bin\\`). Required for `cudart64_12.dll`,
           `cufft64_11.dll`, `nvrtc64_120_0.dll`, `nvJitLink_120_0.dll`
           — these are NEVER shipped by any pip wheel.

    A given EP is only marked capable in `_CAPS` when EVERY DLL that
    EP needs at runtime resolves under the bootstrapped PATH. Cloud /
    CPU-only machines without CUDA Toolkit therefore see `cuda=False`
    and `tensorrt=False`, and `resolve_providers()` skips both tiers
    cleanly (no warning, no crash, no silent CPU fallback after a
    failed EP load).

    Idempotent. Safe to call before or after `import onnxruntime`.
    """
    global _NVIDIA_DLL_BOOTSTRAP_DONE
    if _NVIDIA_DLL_BOOTSTRAP_DONE:
        return

    # Non-Windows: POSIX dynamic linker uses LD_LIBRARY_PATH /
    # rpath semantics that the wheels handle correctly via auditwheel.
    # We still probe ORT's available providers later (since CUDA EP is
    # very real on Linux), but skip the PATH dance.
    if sys.platform != "win32":
        # Coarse probe — assume the wheels' linker rpath handles it on
        # Linux. resolve_providers() will still defer to ORT's own
        # `get_available_providers()` for the final yes/no.
        _CAPS["cuda"] = _try_import("nvidia.cudnn") or _try_import("torch")
        _CAPS["tensorrt"] = _try_import("tensorrt") or _try_import("tensorrt_libs")
        _CAPS["msvc"] = True  # not applicable; pretend OK so we don't gate on it
        _NVIDIA_DLL_BOOTSTRAP_DONE = True
        return

    # ── Phase 1: collect candidate DLL directories ──────────────────────
    # Each entry is a directory we'll add to PATH/add_dll_directory if
    # it exists. Order matters only for deterministic logging.
    candidate_dirs: list[str] = []

    # Pip-wheel sources. Each is `(import_name, optional_subdir)`.
    pip_candidates: list[tuple[str, str]] = [
        ("tensorrt_libs",       ""),    # nvinfer_10.dll + plugins at root
        ("nvidia.cudnn",        "bin"), # cudnn64_9.dll
        ("nvidia.cublas",       "bin"), # cublas64_12.dll, cublasLt64_12.dll
        ("nvidia.cuda_runtime", "bin"), # cudart64_12.dll (when wheel present)
        ("nvidia.cufft",        "bin"), # cufft64_11.dll (when wheel present)
        ("nvidia.cuda_nvrtc",   "bin"), # nvrtc64_120_0.dll (when wheel present)
        ("nvidia.nvjitlink",    "bin"), # nvJitLink_120_0.dll (when wheel present)
    ]
    for mod_name, sub_path in pip_candidates:
        d = _wheel_dir(mod_name, sub_path)
        if d is not None and d not in candidate_dirs:
            candidate_dirs.append(d)

    # System CUDA Toolkit. Many DLLs the CUDA EP needs (cudart, cufft,
    # nvrtc, nvJitLink) ship ONLY with the toolkit installer — no pip
    # wheel covers them on Windows today. Probe in priority order:
    #   1. CUDA_PATH / CUDA_HOME env vars (whatever the user picked)
    #   2. globbed install dirs, highest-version-first
    # We keep ALL matching dirs (not just the highest) so users with
    # multiple toolkit versions side-by-side don't get silently bound
    # to the wrong one.
    for cuda_root in _system_cuda_roots():
        bin_dir = os.path.join(cuda_root, "bin")
        if os.path.isdir(bin_dir) and bin_dir not in candidate_dirs:
            candidate_dirs.append(bin_dir)

    # ── Phase 2: register every candidate dir with the OS loader ────────
    if candidate_dirs:
        # Prepend (not append) so our pinned versions win against any
        # older system-wide install of cuDNN/CUDA still on PATH.
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(
            candidate_dirs + ([current_path] if current_path else [])
        )
        for d in candidate_dirs:
            try:
                os.add_dll_directory(d)
            except (OSError, AttributeError):
                # Either the dir was already registered (harmless) or
                # add_dll_directory is missing on this Python (we require
                # 3.10 so this branch won't fire in practice).
                pass

    _CAPS["added_dirs"] = list(candidate_dirs)

    # ── Phase 3: probe each EP's required DLLs against the live loader ─
    # We only mark an EP capable if EVERY DLL it needs at runtime is
    # reachable. This is what stops the "looks fine, then crashes at
    # InferenceSession" trap that bites cloud users without CUDA.
    _CAPS["cudnn"] = _can_load_dll("cudnn64_9.dll")
    _CAPS["cuda"] = (
        _can_load_dll("cudart64_12.dll")
        and _can_load_dll("cublas64_12.dll")
        and _can_load_dll("cublasLt64_12.dll")
        and _CAPS["cudnn"]
    )
    _CAPS["tensorrt"] = (
        _can_load_dll("nvinfer_10.dll")
        and _CAPS["cuda"]   # TRT EP delegates non-TRT ops to CUDA EP
    )
    _CAPS["msvc"] = _msvc_runtime_present()

    _NVIDIA_DLL_BOOTSTRAP_DONE = True

    # Single-line capability summary. Quiet enough for production logs,
    # informative enough that "TRT didn't kick in" is one grep away.
    enabled = [k for k in ("tensorrt", "cuda", "cudnn", "msvc") if _CAPS[k]]
    missing = [k for k in ("tensorrt", "cuda", "cudnn", "msvc") if not _CAPS[k]]
    print(
        f"  [providers] capabilities: "
        f"enabled=[{', '.join(enabled) or 'none'}] "
        f"missing=[{', '.join(missing) or 'none'}]"
    )


# ---------------------------------------------------------------------------
# Probe helpers used by the bootstrap. Kept module-private (underscore)
# because they're implementation details — only `_bootstrap_nvidia_dlls()`
# and `resolve_providers()` should call them.
# ---------------------------------------------------------------------------

def _try_import(mod_name: str) -> bool:
    """True iff `import mod_name` succeeds. No side-effects on failure."""
    try:
        __import__(mod_name)
        return True
    except ImportError:
        return False


def _wheel_dir(mod_name: str, sub_path: str) -> str | None:
    """Return the on-disk dir for a pip wheel's DLL bundle, or None.

    Uses the package's `__file__` to locate site-packages, then joins
    `sub_path` (commonly "bin"). Returns None if the wheel isn't
    installed OR the expected sub_path doesn't exist on disk.
    """
    try:
        mod = __import__(mod_name, fromlist=["__file__"])
    except ImportError:
        return None
    mod_file = getattr(mod, "__file__", None)
    if not mod_file:
        return None
    base_dir = os.path.dirname(mod_file)
    target = os.path.join(base_dir, sub_path) if sub_path else base_dir
    return target if os.path.isdir(target) else None


def _system_cuda_roots() -> list[str]:
    """All plausible system CUDA Toolkit roots, highest-version-first.

    Probes:
      * `CUDA_PATH`, `CUDA_HOME`, `CUDA_PATH_V12_*` env vars
      * standard installer location
        `C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*`

    Filters to versions whose `bin/cudart64_12.dll` actually exists —
    older v10/v11 toolkits would mismatch our cu12 wheels and ORT.
    Returns a deduplicated list, no order guarantees beyond
    "best candidate first".
    """
    import glob

    roots: list[str] = []
    seen: set[str] = set()

    # Env-var probes first — user's explicit choice wins.
    for env in (
        "CUDA_PATH", "CUDA_HOME",
        "CUDA_PATH_V12_0", "CUDA_PATH_V12_1", "CUDA_PATH_V12_2",
        "CUDA_PATH_V12_3", "CUDA_PATH_V12_4", "CUDA_PATH_V12_5",
        "CUDA_PATH_V12_6", "CUDA_PATH_V12_7", "CUDA_PATH_V12_8",
        "CUDA_PATH_V12_9",
    ):
        v = os.environ.get(env, "").strip()
        if v and os.path.isdir(v) and v not in seen:
            seen.add(v)
            roots.append(v)

    # Then the installer's standard layout. We sort descending so the
    # highest v12.* wins on ties (newer CUDA = more codegen kernels +
    # better Blackwell support — relevant for the user's RTX 5090).
    matches = glob.glob(
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*"
    )
    matches.sort(reverse=True)
    for m in matches:
        if m in seen:
            continue
        if not os.path.isdir(os.path.join(m, "bin")):
            continue
        seen.add(m)
        roots.append(m)

    # Filter to v12-compatible toolkits only. Older v10/v11 ship
    # cudart64_10.dll / cudart64_11.dll and ORT-cu12 will reject them.
    return [r for r in roots if os.path.isfile(
        os.path.join(r, "bin", "cudart64_12.dll")
    )]


def _can_load_dll(name: str) -> bool:
    """True iff Windows can resolve `name` via the current PATH/dll dirs.

    Uses `ctypes.WinDLL` because that calls `LoadLibraryW` with the
    same default search semantics ORT's provider bridge uses — so a
    success here is the strongest available signal that ORT will
    succeed too. We unload the test handle immediately by dropping
    the reference (Windows refcounts module loads).

    Non-Windows callers should not rely on this — returns False on
    POSIX since the DLL extension wouldn't match libfoo.so anyway.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        ctypes.WinDLL(name)
        return True
    except (OSError, FileNotFoundError):
        return False


def _msvc_runtime_present() -> bool:
    """True iff the MSVC C++ Redistributable runtime DLLs are installed.

    ORT's CUDA/TRT EP DLLs link against `vcruntime140.dll` and
    `msvcp140.dll`. On a fresh Windows Server / Datacenter image
    (common in cloud) these are NOT pre-installed, and the EPs fail
    to load with a misleading WinError 126 that names a CUDA DLL.
    Probing System32 directly is the cheapest way to give the user
    a clear "install VC++ Redistributable" hint instead.
    """
    sys32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    needed = ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")
    return all(os.path.isfile(os.path.join(sys32, dll)) for dll in needed)


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
    """True if the TensorRT EP should sit at the top of the ladder.

    Decision matrix (env var wins over capability probe, capability
    probe wins over wishful thinking):

        ┌─────────────────────────┬──────────┬──────────────────────┐
        │ VIDEO_USE_PARAKEET_TRT  │ _CAPS    │ result               │
        ├─────────────────────────┼──────────┼──────────────────────┤
        │ "1"/"true"/...          │ capable  │ True                 │
        │ "1"/"true"/...          │ NOT cap. │ False + loud warning │
        │ "0"/"false"/...         │ either   │ False (user override)│
        │ unset / empty           │ capable  │ True  (NEW default)  │
        │ unset / empty           │ NOT cap. │ False (silent skip)  │
        └─────────────────────────┴──────────┴──────────────────────┘

    Why TRT-by-default-when-capable:
        On a machine with the TRT wheels + system CUDA installed, the
        engine compile happens once and is cached on disk; subsequent
        runs load the cached engine in <1s and we get the full ~320x
        RTFx speedup instead of CUDA's ~70x. The user already paid
        the disk + install cost for `tensorrt-cu12-libs`, so it would
        be rude not to use it.

    Why we still gate on _CAPS["tensorrt"] in cloud:
        Cloud CPU-only / GPU-without-TRT machines can't compile an
        engine at all. Auto-on without the capability check would
        crash at first inference instead of the smooth fallback to
        CUDA / CPU we want. The probe is cheap (a single ctypes
        WinDLL) and runs once at module import.
    """
    raw = os.environ.get("VIDEO_USE_PARAKEET_TRT", "").strip().lower()
    capable = bool(_CAPS.get("tensorrt"))

    # Explicit user override — honor it and warn loudly if the env
    # asks for something we can't actually deliver.
    if raw in _TRUTHY:
        if not capable:
            print(
                "  [providers] VIDEO_USE_PARAKEET_TRT=1 set but "
                "TensorRT runtime not detected (missing nvinfer_10.dll "
                "or upstream CUDA dependency). Falling back to CUDA EP. "
                "Install: `pip install tensorrt-cu12-libs nvidia-cudnn-cu12` "
                "and ensure CUDA Toolkit 12.x is on PATH.",
                file=sys.stderr,
            )
            return False
        return True
    if raw in _FALSY:
        return False

    # Unset → capability-driven default. No noise either way; the
    # one-line "[providers] resolved EP ladder: ..." log already tells
    # the user which tier we picked.
    return capable


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
    # Gated on `_CAPS["cuda"]` on Windows — ORT happily lists CUDA as
    # "available" purely because the EP DLL was compiled in, but it
    # will silent-fall-back to CPU at session creation if cudart /
    # cublas / cudnn aren't actually loadable. Probing first lets us
    # skip the broken tier cleanly so the lane log shows
    # `CPUExecutionProvider` (truth) instead of CUDA -> CPU (lie).
    # On non-Windows we trust ORT's probe since rpath usually works.
    cuda_capable = _CAPS.get("cuda", True) if sys.platform == "win32" else True
    if "CUDAExecutionProvider" in available and cuda_capable:
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
    print("ORT installed providers :", _ort_available_providers())
    print("VIDEO_USE_PARAKEET_TRT  :", os.environ.get("VIDEO_USE_PARAKEET_TRT", "(unset)"))
    print("tensorrt_libs importable:", _tensorrt_libs_importable())
    # Capability matrix lifted straight from `_CAPS` so users can see
    # exactly which gate failed when TRT/CUDA didn't kick in.
    print("capabilities            :", {
        k: _CAPS[k] for k in ("tensorrt", "cuda", "cudnn", "msvc")
    })
    if _CAPS.get("added_dirs"):
        print("DLL dirs added          :")
        for d in _CAPS["added_dirs"]:
            print(f"  - {d}")
    print("ladder (prefer_trt=True ):", resolve_providers(prefer_tensorrt=True))
    print("ladder (prefer_trt=False):", resolve_providers(prefer_tensorrt=False))


if __name__ == "__main__":
    main()
