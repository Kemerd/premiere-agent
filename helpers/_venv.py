"""Skill-local virtualenv bootstrap.

Every CLI helper calls `ensure()` before importing anything heavy.

    ┌───────────────────────────────┬──────────────────────────────────┐
    │ situation                     │ behaviour                        │
    ├───────────────────────────────┼──────────────────────────────────┤
    │ no `<skill>/.venv/`           │ no-op (legacy global install)    │
    │ already running inside it     │ no-op                            │
    │ PREMIERE_AGENT_NO_VENV=1      │ no-op (force the global env)     │
    │ venv exists, we're outside it │ re-launch same argv under the    │
    │                               │ venv python, exit with its code  │
    └───────────────────────────────┴──────────────────────────────────┘

Why re-launch instead of documenting a venv path: the documented
invocation stays `python helpers/<script>.py ...` from any shell and any
global interpreter, while the pinned dependency set (onnxruntime-gpu,
CUDA torch, ...) is guaranteed to be what actually runs. A stray
package in the global site-packages (e.g. CPU `onnxruntime` shadowing
`onnxruntime-gpu`) can no longer silently demote a lane to CPU.

Lane subprocesses spawned by preprocess.py use `sys.executable`, so once
the entry point is inside the venv every child inherits it for free.

The venv itself is created by install.bat / install.sh.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Skill root = parent of helpers/. `resolve()` follows the
# ~/.claude/skills symlink so the venv is found beside the real checkout.
SKILL_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = SKILL_ROOT / ".venv"

# Set on the re-launched child. If the child STILL doesn't look like it
# is inside the venv (broken / relocated venv), we stop instead of
# re-launching forever.
_REEXEC_GUARD = "_PREMIERE_AGENT_VENV_REEXEC"


def venv_python() -> Path:
    """Return the venv interpreter path for the current platform."""
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _same_path(a: str | os.PathLike, b: str | os.PathLike) -> bool:
    """Case / symlink-insensitive path equality. False on any OS error."""
    try:
        return (os.path.normcase(os.path.realpath(a))
                == os.path.normcase(os.path.realpath(b)))
    except (OSError, ValueError):
        return False


def in_venv() -> bool:
    """True when the running interpreter IS the skill's venv."""
    return _same_path(sys.prefix, VENV_DIR)


def ensure() -> None:
    """Re-launch the current script under the skill venv when needed.

    Returns normally when no re-launch is required (or possible);
    otherwise never returns — the process exits with the child's code.
    """
    # ── Opt-out escape hatch ──────────────────────────────────────────
    if os.environ.get("PREMIERE_AGENT_NO_VENV", "").strip().lower() in ("1", "true", "yes"):
        return

    # ── Nothing to do: no venv on disk, or already inside it ──────────
    py = venv_python()
    if not py.is_file() or in_venv():
        return

    # ── Loop guard: a re-launched child that still isn't "in" the venv
    #    means the venv is broken. Warn and run on what we have. ───────
    if os.environ.get(_REEXEC_GUARD) == "1":
        print(f"  [venv] warn: {py} did not activate {VENV_DIR}; "
              f"continuing on {sys.executable}", file=sys.stderr)
        return

    # ── Nothing to re-launch (interactive / -c invocation) ────────────
    if not sys.argv or not sys.argv[0] or sys.argv[0] == "-c":
        return

    # ── Re-launch with the identical argv + cwd, stream stdio through ─
    env = dict(os.environ)
    env[_REEXEC_GUARD] = "1"
    try:
        rc = subprocess.call([str(py), *sys.argv], env=env)
    except KeyboardInterrupt:
        rc = 130
    except OSError as exc:
        print(f"  [venv] warn: could not launch {py} ({exc}); "
              f"continuing on {sys.executable}", file=sys.stderr)
        return

    # Flush anything the parent buffered, then mirror the child's code.
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
