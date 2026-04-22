"""Lazy NeMo install — single source of truth for the Parakeet NeMo fallback.

Why this exists:
  We don't want `nemo_toolkit[asr]` (~600 MB of deps including its own
  pinned torch / lightning / hydra) in the default install. 99% of
  users on an open network never need it — the primary
  `parakeet_onnx_lane.py` runs on plain ONNX Runtime and the NeMo
  fallback never fires.

  But for the 1% on hosts where ONNX Runtime can't load a working
  execution provider, the fallback needs to "just work" on first run
  without making them hunt for an install command. So we shell out to
  pip the FIRST time the NeMo Parakeet lane is invoked, then cache
  the import for the rest of the process.

Called from:
  - `parakeet_lane._build_parakeet_model`  (only call site)
"""

from __future__ import annotations

import subprocess
import sys


# Module-level memo so we don't fork a subprocess on every import attempt
# in long-lived processes. None = not checked yet, True = present,
# False = install failed.
_NEMO_AVAILABLE: bool | None = None


def ensure_nemo_installed() -> None:
    """Block until `import nemo` succeeds, lazy-installing on first miss.

    Idempotent. After the first successful call this returns immediately
    via the `_NEMO_AVAILABLE` memo — subsequent invocations cost a single
    boolean check, not a subprocess spawn.

    Raises:
        RuntimeError: if pip install fails (network down, no PyPI access,
                      conflicting deps, etc.). Caller should surface the
                      error to the user verbatim — there's no automatic
                      recovery from this state.
    """
    global _NEMO_AVAILABLE

    # Fast path: already verified in this process.
    if _NEMO_AVAILABLE:
        return

    # Try the cheap import first. If NeMo is already in the env (user
    # ran `pip install -e .[parakeet]` ahead of time, or a previous
    # session did the lazy install), we're done.
    try:
        import nemo  # noqa: F401
        _NEMO_AVAILABLE = True
        return
    except ImportError:
        pass

    # Not present — pip-install it. We pin a permissive lower bound; the
    # `[asr]` extra pulls librosa, omegaconf, hydra-core, pytorch-lightning,
    # and a pile of other things. ~600 MB on a clean machine.
    print(
        "[parakeet_lane] one-time install: nemo_toolkit[asr] "
        "(~600 MB, pulls lightning + hydra). "
        "This only happens once per machine.",
        file=sys.stderr, flush=True,
    )

    # Use the SAME interpreter that's running this script — critical on
    # systems with multiple Python versions (e.g. py launcher on Windows
    # picking 3.13 while our venv is 3.11). `sys.executable -m pip`
    # avoids that whole class of bug.
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--upgrade", "nemo_toolkit[asr]>=2.0",
    ]

    try:
        # check=True so a non-zero pip exit raises CalledProcessError,
        # which we re-wrap with a clearer message for the user.
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        _NEMO_AVAILABLE = False
        raise RuntimeError(
            "Failed to install nemo_toolkit[asr] for the Parakeet "
            "fallback. The pip install exited with code "
            f"{e.returncode}. If your network blocks PyPI as well, "
            "install NeMo offline from a wheel cache and re-run. "
            "Manual command: "
            f"`{sys.executable} -m pip install -e .[parakeet]`."
        ) from e

    # Verify the install actually landed something importable. A
    # successful `pip install` doesn't always mean `import nemo` works
    # — wheel conflicts, partial installs, etc. can leave you with a
    # pip cache hit but no usable module.
    try:
        import nemo  # noqa: F401
        _NEMO_AVAILABLE = True
        print(
            "[parakeet_lane] nemo_toolkit installed successfully.",
            file=sys.stderr, flush=True,
        )
    except ImportError as e:
        _NEMO_AVAILABLE = False
        raise RuntimeError(
            "nemo_toolkit installed via pip but `import nemo` still "
            f"fails: {e}. Likely a partial install or a conflicting "
            "torch/lightning version pin. Try wiping the env and "
            "reinstalling from scratch."
        ) from e


def is_nemo_installed() -> bool:
    """Cheap check: is NeMo importable RIGHT NOW, without trying to install?

    Useful for tests and the health check — they want to know the state
    of the env without triggering a 600 MB download.
    """
    global _NEMO_AVAILABLE
    if _NEMO_AVAILABLE is not None:
        return _NEMO_AVAILABLE
    try:
        import nemo  # noqa: F401
        _NEMO_AVAILABLE = True
        return True
    except ImportError:
        # Don't memoize a False here — the user might pip-install NeMo
        # in between checks (e.g. during the same long-running session).
        # We only memoize True.
        return False
