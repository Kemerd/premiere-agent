"""Environment guards that MUST be applied before any `transformers` import.

Why this module exists
----------------------
The `transformers` library (especially the 4.x line) probes for installed
deep-learning backends at module-load time and eagerly imports any it
finds — Torch, TensorFlow, JAX/Flax. This is fine when all three installs
are healthy. It's a disaster when ONE of them is subtly broken, because
a single `import transformers` crashes everything that depends on it,
even code paths that have zero interest in the broken backend.

The most common breakage is TensorFlow ↔ protobuf version drift on
Windows: `pip install tensorflow` years ago + a recent unrelated
`pip install protobuf` upgrade leaves TF unable to register its
descriptors at import. transformers 4.x then dies on the eager
`import tensorflow as tf` inside `image_transforms.py` and the user
sees a stack trace that has nothing to do with what they were trying
to do.

The fix
-------
HuggingFace documents three env vars that gate the eager backend
import: `USE_TORCH`, `USE_TF`, `USE_FLAX`. Setting any to "1" forces
that backend on, setting to "0" disables it entirely. We pin
`USE_TORCH=1` (we always want torch — it's the only backend we use)
and disable the other two. This means transformers will skip the eager
TF/JAX import entirely, and our lanes can't be taken down by an
unrelated TF install going stale.

We use `setdefault` rather than unconditional assignment so power users
can still override via their shell env if they really want to test a
TF or JAX path.

Usage
-----
This module performs the env-var setup as an import side-effect.
Import it BEFORE any `import transformers` (direct OR transitive)
in any process that wants the guard:

    from _hf_env import *  # noqa: F401,F403  - import for side effect
    from transformers import pipeline  # safe now

The `*` import is intentional — it guarantees the import isn't
elided by an aggressive linter that "knows" the import is unused.
"""

from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# Backend guards. Order doesn't matter — these are read independently by
# transformers' `is_X_available()` helpers at first import time. Using
# setdefault so a user shell-env override (e.g. `set USE_TF=1` to test a
# TF model) wins over our pin.
# ---------------------------------------------------------------------------
os.environ.setdefault("USE_TORCH", "1")  # we ALWAYS want torch
os.environ.setdefault("USE_TF", "0")     # skip eager TF import (broken protobuf, etc.)
os.environ.setdefault("USE_FLAX", "0")   # skip eager Flax/JAX import
os.environ.setdefault("USE_JAX", "0")    # belt-and-suspenders alias used by some HF code


# ---------------------------------------------------------------------------
# Public sentinel — re-exported so consumers can `from _hf_env import *`
# without triggering "unused import" warnings on their import line.
# Reading this attr is a no-op; the work happened at module load.
# ---------------------------------------------------------------------------

HF_ENV_GUARDS_INSTALLED = True

__all__ = ["HF_ENV_GUARDS_INSTALLED"]
