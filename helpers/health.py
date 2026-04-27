"""Cached health check for premiere-agent.

Runs the FAST tier of tests.py and caches the structured result to
`~/.premiere-agent/health.json` so subsequent invocations within a
TTL window return instantly.

Designed for Claude Code skill startup:

    On every new session, the skill calls:

        python helpers/health.py --json

    If the cache is fresh AND the env fingerprint hasn't changed, this
    returns in < 50 ms with the last-known status. If anything is stale
    or missing, it transparently re-runs the smoke tests and updates the
    cache.

    The JSON output is structured so the LLM can:
      - announce a one-line "skill ready" or "skill needs attention" status
      - if anything failed, surface the SPECIFIC failures and pre-canned
        resolution advice rather than dumping a wall of test output
      - decide whether to proceed with the session or block on a fix

Cache invalidates on:
  - older than --ttl-days (default 7)
  - any of {python, torch, transformers, platform} version-string changed
  - --force flag
  - --clear flag (deletes cache, no run)

Cache location:
  Win   : %USERPROFILE%\.premiere-agent\health.json
  Unix  : ~/.premiere-agent/health.json

CLI:
    python helpers/health.py            # human-readable summary, run if stale
    python helpers/health.py --json     # machine-readable, run if stale
    python helpers/health.py --force    # ignore cache, always run
    python helpers/health.py --clear    # delete cache, exit
    python helpers/health.py --status   # cached status only, no re-run ever
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make tests.py importable from anywhere — health.py lives in helpers/
# but tests.py lives at the project root one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "helpers"))

# CRITICAL: install the HF backend guards BEFORE any code path here can
# probe `transformers` (we do `__import__("transformers")` in
# `env_fingerprint` to read its version string, which triggers the
# eager TF/JAX import path in 4.x). Must come AFTER the sys.path setup
# above so the helpers/ folder is actually importable.
from _hf_env import HF_ENV_GUARDS_INSTALLED  # noqa: F401  - import for side effect


CACHE_VERSION = 1
DEFAULT_TTL_DAYS = 7


# ---------------------------------------------------------------------------
# Cache location — XDG-aware on Linux, ~/AppData on Windows, ~/Library on
# macOS would be ideal but we use a single unified `~/.premiere-agent/`
# for predictability across machines.
# ---------------------------------------------------------------------------

def cache_dir() -> Path:
    """Return the per-user cache directory (created on first use)."""
    home = Path.home()
    d = home / ".premiere-agent"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path() -> Path:
    return cache_dir() / "health.json"


# ---------------------------------------------------------------------------
# Environment fingerprint — what triggers automatic re-runs when libs change.
# Lightweight: only string lookups, no model loading.
# ---------------------------------------------------------------------------

def env_fingerprint() -> dict:
    """Return a dict of version strings used to invalidate the cache.

    Any change here re-runs the smoke tests. Order is stable so dict
    equality works for comparison.
    """
    fp = {
        "python": sys.version.split()[0],
        "platform": platform.system().lower(),
    }
    # Versions of the libraries most likely to break on upgrade.
    for mod_name in ("torch", "transformers", "opentimelineio"):
        try:
            mod = __import__(mod_name)
            fp[mod_name] = getattr(mod, "__version__", "?")
        except ImportError:
            fp[mod_name] = "missing"
    return fp


# ---------------------------------------------------------------------------
# Resolution advice — when the suite fails, give Claude concrete fix steps
# instead of dumping raw failure strings. Map known failure patterns to
# short actionable strings the LLM can relay verbatim.
# ---------------------------------------------------------------------------

# Each entry: (substring to match in failure name OR reason, advice line).
# Matched in order; first match wins. Substring match is case-insensitive.
ADVICE_RULES: list[tuple[str, str]] = [
    ("ffmpeg",
     "ffmpeg/ffprobe missing on PATH. Install: "
     "Win `winget install Gyan.FFmpeg`, "
     "macOS `brew install ffmpeg`, "
     "Linux `apt install ffmpeg`. Restart the shell after."),
    ("torch import",
     "PyTorch not installed. Run install.bat (Windows) or install.sh "
     "(Linux/macOS) from the project root, OR "
     "`pip install torch --index-url https://download.pytorch.org/whl/cu128` "
     "for an RTX 50-series."),
    ("cuda available",
     "PyTorch can't see your GPU. Check `nvidia-smi`. If your GPU is an "
     "RTX 50-series (sm_120), reinstall torch with the cu128 wheel: "
     "`pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch`."),
    ("import parakeet_onnx_lane",
     "Parakeet ONNX lane import failed — usually missing onnxruntime / "
     "onnx-asr. Run `pip install -e \".[preprocess]\"` from the project root."),
    ("import parakeet_lane",
     "Parakeet NeMo fallback lane import failed — pure-Python module, this "
     "should never happen. Re-clone the repo or check helpers/parakeet_lane.py "
     "exists."),
    ("nemo",
     "NeMo install failed (Parakeet ASR fallback). Manual: "
     "`pip install -e .[parakeet]`. If your network blocks PyPI too, install "
     "nemo_toolkit[asr] from a local wheel cache and re-run."),
    ("parakeet",
     "Parakeet fallback path errored. Inspect the failure reason; if it's a "
     "network issue, your proxy may also block NGC. Manual install: "
     "`pip install -e .[parakeet]`."),
    ("import audio_lane",
     "Audio lane import failed — install CLAP deps: "
     "`pip install -e \".[preprocess]\"` (pulls onnxruntime[-gpu on Win/Linux, "
     "plain on macOS] + transformers + soxr)."),
    ("import visual_lane",
     "Visual lane import failed — install Florence-2 deps: "
     "`pip install -e \".[preprocess]\"` (transformers, einops, timm)."),
    ("import export_fcpxml",
     "FCPXML/xmeml export deps missing. Install: `pip install -e \".[fcpxml]\"` "
     "(opentimelineio + otio-fcpx-xml-adapter for .fcpxml -> Resolve/FCP X, "
     "+ otio-fcp-adapter for .xml -> Premiere Pro native)."),
    ("fcpxml round-trip",
     "NLE export module loaded but failed to round-trip. "
     "Common cause: source file missing or unreadable for ffprobe, OR the "
     "Premiere xmeml adapter (otio-fcp-adapter) isn't installed. "
     "Re-run with --keep-tmp to inspect the synthetic test clip."),
    ("schedule sanity",
     "VRAM scheduler picked an unexpected tier. If you want to override, "
     "pass --force-schedule {parallel|sequential|cpu} to preprocess.py."),
    ("pack_timelines",
     "Timeline packer failed on synthetic input. Likely a JSON schema "
     "mismatch between the tests fixture and the lane output. Re-run "
     "tests.py with --keep-tmp and inspect the JSON in transcripts/."),
]


def derive_advice(failures: list[tuple[str, str]]) -> list[str]:
    """For each failure, find the best-matching advice rule. Dedupe so we
    don't tell the user "install ffmpeg" three times when three tests
    failed for the same root cause."""
    seen: set[str] = set()
    out: list[str] = []
    for name, reason in failures:
        haystack = f"{name}\n{reason}".lower()
        for needle, advice in ADVICE_RULES:
            if needle.lower() in haystack:
                if advice not in seen:
                    seen.add(advice)
                    out.append(advice)
                break
        else:
            # No rule matched — surface the raw failure so the user can act.
            generic = f"Unhandled failure in '{name}': {reason}"
            if generic not in seen:
                seen.add(generic)
                out.append(generic)
    return out


# ---------------------------------------------------------------------------
# Cache load / save / freshness
# ---------------------------------------------------------------------------

def _load_cache() -> dict | None:
    """Return the parsed cache dict, or None if missing / corrupt."""
    p = cache_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_cache(payload: dict) -> None:
    """Atomic write so a Ctrl-C mid-write doesn't leave a half-file."""
    p = cache_path()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def _is_fresh(cache: dict, ttl_days: float) -> tuple[bool, str]:
    """Return (fresh, reason). reason is empty if fresh, else why-stale."""
    if cache.get("version") != CACHE_VERSION:
        return False, f"cache schema v{cache.get('version')} != current v{CACHE_VERSION}"
    cached_ts = cache.get("cached_at_ts")
    if not isinstance(cached_ts, (int, float)):
        return False, "cached_at_ts missing or invalid"
    age_days = (time.time() - cached_ts) / 86400.0
    if age_days > ttl_days:
        return False, f"cache is {age_days:.1f}d old, TTL is {ttl_days}d"
    fp_old = cache.get("env_fingerprint") or {}
    fp_new = env_fingerprint()
    if fp_old != fp_new:
        diffs = [
            f"{k}: {fp_old.get(k, '?')} -> {fp_new.get(k, '?')}"
            for k in sorted(set(fp_old) | set(fp_new))
            if fp_old.get(k) != fp_new.get(k)
        ]
        return False, "env changed: " + "; ".join(diffs)
    return True, ""


# ---------------------------------------------------------------------------
# Run wrapper — invokes tests.run_all() and assembles a cache-able payload
# ---------------------------------------------------------------------------

def detect_active_fallbacks() -> list[str]:
    """Probe per-machine sentinels that indicate a non-default lane backend
    is in use. Read at health-check time so Claude can announce the
    fallback in one line at session start instead of the user being
    surprised mid-run.

    The speech lane runs Parakeet TDT through ONNX Runtime by default;
    when that's not available (no working EP, exotic OS, etc.) the
    NeMo torch fallback in `parakeet_lane.py` kicks in. We expose
    that via the `VIDEO_USE_SPEECH_LANE=nemo` env var rather than a
    sentinel file — there's no longer a network-blocked "I had to
    fall back" state to persist across runs.

    Returns an empty list when everything is on the default path.
    """
    active: list[str] = []
    backend = os.environ.get("VIDEO_USE_SPEECH_LANE", "").strip().lower()
    if backend and backend != "onnx":
        # Surface non-default backend names (e.g. "nemo") so the
        # session-start banner matches what's actually going to run.
        active.append(f"speech={backend}")
    return active


def run_and_build_payload(heavy: bool = False) -> dict:
    """Run the smoke suite, return a dict suitable for caching."""
    import tests as t
    R = t.run_all(heavy=heavy, keep_tmp=False)

    failures = [{"name": n, "reason": r} for (n, r) in R.failed]
    advice = derive_advice(R.failed)

    if not R.failed and not R.skipped:
        status = "ok"
    elif R.failed:
        status = "fail"
    else:
        status = "warn"  # only skips, no failures

    return {
        "version": CACHE_VERSION,
        "cached_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cached_at_ts": time.time(),
        "ttl_days": DEFAULT_TTL_DAYS,
        "env_fingerprint": env_fingerprint(),
        "tier": "heavy" if heavy else "fast",
        "status": status,
        "passed": len(R.passed),
        "failed": len(R.failed),
        "skipped": len(R.skipped),
        "elapsed_s": round(time.monotonic() - R._t0, 2),
        "failures": failures,
        "skips": [{"name": n, "reason": r} for (n, r) in R.skipped],
        "advice": advice,
        # Fallbacks listed here are NOT failures — they're informational.
        # Claude reads this at session start and announces "speech lane
        # on Parakeet (Whisper unreachable from this machine)" in one line.
        "fallbacks_active": detect_active_fallbacks(),
    }


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def print_human(payload: dict, *, from_cache: bool, why_run: str = "") -> None:
    """One-screen human summary."""
    status = payload.get("status", "?")
    icons = {"ok": "[OK]", "fail": "[FAIL]", "warn": "[WARN]"}
    icon = icons.get(status, "[?]")

    print()
    print(f"{icon} premiere-agent health: {status.upper()}")
    print(f"  cached_at:  {payload.get('cached_at')}  "
          f"({'cache hit' if from_cache else 'just ran'})")
    if why_run and not from_cache:
        print(f"  why ran:    {why_run}")
    print(f"  tier:       {payload.get('tier')}")
    print(f"  results:    {payload.get('passed')} pass / "
          f"{payload.get('failed')} fail / {payload.get('skipped')} skip "
          f"({payload.get('elapsed_s')}s)")
    fp = payload.get("env_fingerprint") or {}
    print(f"  env:        python {fp.get('python')}, "
          f"torch {fp.get('torch')}, "
          f"transformers {fp.get('transformers')}, "
          f"otio {fp.get('opentimelineio')}, "
          f"platform {fp.get('platform')}")

    fallbacks = payload.get("fallbacks_active") or []
    if fallbacks:
        print()
        print("  FALLBACKS ACTIVE:")
        for fb in fallbacks:
            if fb == "parakeet":
                print("    * speech lane: NVIDIA Parakeet "
                      "(Whisper download was blocked on this machine; "
                      "Parakeet is local-only, ~10x faster, English/EU only)")
            else:
                print(f"    * {fb}")

    if payload.get("failures"):
        print()
        print("  FAILURES:")
        for f in payload["failures"]:
            print(f"    - {f['name']}: {f['reason']}")
    if payload.get("skips"):
        print()
        print("  SKIPPED:")
        for s in payload["skips"]:
            print(f"    - {s['name']}: {s['reason']}")
    if payload.get("advice"):
        print()
        print("  HOW TO FIX:")
        for a in payload["advice"]:
            print(f"    * {a}")
    print()


def print_json(payload: dict, *, from_cache: bool, why_run: str = "") -> None:
    """Machine-readable output. Adds two transient fields the cache itself
    doesn't store (they describe THIS invocation, not the test result)."""
    out = dict(payload)
    out["from_cache"] = from_cache
    if why_run:
        out["why_ran"] = why_run
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cached health check for premiere-agent",
    )
    ap.add_argument("--force", action="store_true",
                    help="Ignore cache, always re-run the smoke suite.")
    ap.add_argument("--clear", action="store_true",
                    help="Delete the cache file and exit (no run).")
    ap.add_argument("--status", action="store_true",
                    help="Print cached status only — never run, even if stale "
                         "or missing. Returns exit 2 if no cache exists.")
    ap.add_argument("--heavy", action="store_true",
                    help="Run the heavy tier (~4.5 GB downloads on first run). "
                         "Cached separately; use sparingly.")
    ap.add_argument("--ttl-days", type=float, default=DEFAULT_TTL_DAYS,
                    help=f"Cache lifetime in days (default {DEFAULT_TTL_DAYS}).")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of human summary.")
    args = ap.parse_args()

    # ── --clear: wipe and exit ────────────────────────────────────────
    if args.clear:
        p = cache_path()
        if p.exists():
            p.unlink()
            print(f"cleared {p}")
        else:
            print("no cache to clear")
        return 0

    cache = _load_cache()

    # ── --status: cached-only, never run ──────────────────────────────
    if args.status:
        if cache is None:
            print("no cached health result. Run `python helpers/health.py` "
                  "(without --status) to populate the cache.",
                  file=sys.stderr)
            return 2
        if args.json:
            print_json(cache, from_cache=True)
        else:
            print_human(cache, from_cache=True)
        return 0 if cache.get("status") == "ok" else 1

    # ── Normal flow: run iff stale / missing / forced ─────────────────
    why_run = ""
    if args.force:
        why_run = "--force flag"
    elif cache is None:
        why_run = "no cache yet"
    else:
        fresh, reason = _is_fresh(cache, args.ttl_days)
        if not fresh:
            why_run = reason

    if why_run:
        payload = run_and_build_payload(heavy=args.heavy)
        _save_cache(payload)
        if args.json:
            print_json(payload, from_cache=False, why_run=why_run)
        else:
            print_human(payload, from_cache=False, why_run=why_run)
    else:
        if args.json:
            print_json(cache, from_cache=True)
        else:
            print_human(cache, from_cache=True)
        payload = cache

    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
