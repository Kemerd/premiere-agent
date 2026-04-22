"""Unified progress display for the three preprocessing lanes.

Two display modes, auto-selected:

  - "rich"  — interactive TTY, single-lane invocation. Uses rich.progress
              for a live-updating bar with description + percent + ETA.

  - "line"  — non-TTY (piped to Claude Code) OR multi-lane parallel mode
              (where rich's live cursor games would be garbled by
              subprocess prefix tagging in the parent). Emits a single
              structured status line every UPDATE_INTERVAL_S seconds OR
              on milestones (start, end, every 10%).

Mode is forced to "line" when:
  - sys.stderr is not a TTY                              (Claude Code, CI, log files)
  - VIDEO_USE_PROGRESS_MODE=line is set in the env       (orchestrator forces)
  - the subprocess has VIDEO_USE_LANE_PREFIX=<lane> set  (parallel-lane child)

Public API — both modes share the same shape so call sites are identical:

    from progress import lane_progress

    with lane_progress("whisper", total=12, unit="video", desc="speech lane") as bar:
        for video in videos:
            bar.start_item(video.name)
            ... do work ...
            bar.update(advance=1)
        # bar.done() called automatically on exit

    # Sub-progress for per-frame work inside one item:
    with lane_progress("visual", total=240, unit="frame",
                       desc=video.name, parent=outer_bar) as sub:
        for i, frame in enumerate(frames):
            sub.update(advance=1)

The structured "line" output is designed to be greppable. Format:

    [PROGRESS lane=whisper status=tick item="footage_03.mp4"
              done=120 total=240 pct=50.0 eta_s=134 elapsed_s=87]

So Claude can `tail -f` it and reason about progress without parsing
ANSI escape sequences.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


# Min seconds between "tick" lines in line mode — keeps log volume sane on
# fast lanes without losing responsiveness on slow ones.
UPDATE_INTERVAL_S = 3.0

# Emit a tick line at every this-percent milestone regardless of time.
PCT_MILESTONES = (0, 10, 25, 50, 75, 90, 100)


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

def _detect_mode() -> str:
    """Return "rich" or "line" based on env + TTY state.

    Order:
        1. VIDEO_USE_PROGRESS_MODE env var (explicit override)
        2. VIDEO_USE_LANE_PREFIX present (we're a child of the orchestrator)
        3. stderr is a TTY?  yes → rich, no → line
    """
    forced = os.environ.get("VIDEO_USE_PROGRESS_MODE", "").strip().lower()
    if forced in ("rich", "line"):
        return forced
    if os.environ.get("VIDEO_USE_LANE_PREFIX"):
        return "line"
    return "rich" if sys.stderr.isatty() else "line"


# ---------------------------------------------------------------------------
# Common bar state
# ---------------------------------------------------------------------------

@dataclass
class _BarState:
    """Per-bar accounting shared by both backends."""
    lane: str
    desc: str
    unit: str
    total: int
    done: int = 0
    started_at: float = field(default_factory=time.monotonic)
    last_emit: float = 0.0
    last_pct_milestone: int = -1
    current_item: str = ""

    def pct(self) -> float:
        return 100.0 * self.done / max(1, self.total)

    def eta_s(self) -> float:
        if self.done <= 0:
            return -1.0
        elapsed = time.monotonic() - self.started_at
        rate = self.done / max(1e-3, elapsed)
        remaining = max(0, self.total - self.done)
        return remaining / max(1e-3, rate)

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at


# ---------------------------------------------------------------------------
# Line-mode backend (Claude-friendly, parallel-safe)
# ---------------------------------------------------------------------------

class _LineBar:
    """Emits structured status lines on stderr at controlled cadence.

    Designed to play nicely with the orchestrator's `[whisper] ...`
    prefix tagging — every emit is one complete line, no carriage
    returns or cursor games.
    """

    def __init__(self, state: _BarState):
        self.s = state
        self._emit("start", force=True)

    def _emit(self, status: str, *, force: bool = False) -> None:
        now = time.monotonic()
        pct = self.s.pct()
        # Pick the highest milestone we've crossed since last emit.
        milestone_hit = -1
        for m in PCT_MILESTONES:
            if pct >= m and m > self.s.last_pct_milestone:
                milestone_hit = m
        if (
            not force
            and milestone_hit < 0
            and (now - self.s.last_emit) < UPDATE_INTERVAL_S
        ):
            return

        if milestone_hit >= 0:
            self.s.last_pct_milestone = milestone_hit
        self.s.last_emit = now

        # Field-tagged so it's trivially parseable. Keep order stable.
        item_q = self.s.current_item.replace('"', "'")
        line = (
            f"[PROGRESS lane={self.s.lane} status={status} "
            f'item="{item_q}" '
            f"done={self.s.done} total={self.s.total} "
            f"pct={pct:.1f} "
            f"eta_s={self.s.eta_s():.0f} elapsed_s={self.s.elapsed_s():.0f} "
            f"unit={self.s.unit} desc=\"{self.s.desc}\"]"
        )
        # stderr to stay out of any captured stdout JSON / pipe contents.
        print(line, file=sys.stderr, flush=True)

    def start_item(self, item: str) -> None:
        self.s.current_item = item
        self._emit("item", force=True)

    def update(self, advance: int = 1, item: str | None = None) -> None:
        if item is not None:
            self.s.current_item = item
        self.s.done = min(self.s.total, self.s.done + advance)
        self._emit("tick")

    def done(self) -> None:
        self.s.done = self.s.total
        self._emit("done", force=True)


# ---------------------------------------------------------------------------
# Rich-mode backend (interactive TTY, single lane)
# ---------------------------------------------------------------------------

class _RichBar:
    """Wraps rich.progress.Progress with the same surface as _LineBar."""

    # Class-level so multiple bars in the same process share one display
    # (e.g. an outer "videos" bar and inner "frames" bar).
    _shared_progress = None
    _ref_count = 0

    def __init__(self, state: _BarState):
        from rich.progress import (
            Progress, BarColumn, TextColumn, TimeRemainingColumn,
            TimeElapsedColumn, MofNCompleteColumn, SpinnerColumn,
        )
        self.s = state

        if _RichBar._shared_progress is None:
            _RichBar._shared_progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.fields[lane]:<8}[/]"),
                TextColumn("{task.description}"),
                BarColumn(bar_width=None),
                MofNCompleteColumn(),
                TextColumn("•"),
                TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("eta"),
                TimeRemainingColumn(),
                transient=False,
                refresh_per_second=8,
            )
            _RichBar._shared_progress.start()
        _RichBar._ref_count += 1

        self.task_id = _RichBar._shared_progress.add_task(
            description=state.desc,
            total=state.total,
            lane=state.lane,
        )

    def start_item(self, item: str) -> None:
        self.s.current_item = item
        if _RichBar._shared_progress is not None:
            _RichBar._shared_progress.update(
                self.task_id,
                description=f"{self.s.desc} — {item}",
            )

    def update(self, advance: int = 1, item: str | None = None) -> None:
        if item is not None:
            self.start_item(item)
        self.s.done = min(self.s.total, self.s.done + advance)
        if _RichBar._shared_progress is not None:
            _RichBar._shared_progress.update(self.task_id, advance=advance)

    def done(self) -> None:
        self.s.done = self.s.total
        if _RichBar._shared_progress is not None:
            _RichBar._shared_progress.update(
                self.task_id, completed=self.s.total
            )
        _RichBar._ref_count -= 1
        if _RichBar._ref_count <= 0 and _RichBar._shared_progress is not None:
            _RichBar._shared_progress.stop()
            _RichBar._shared_progress = None
            _RichBar._ref_count = 0


# ---------------------------------------------------------------------------
# Public context manager
# ---------------------------------------------------------------------------

@contextmanager
def lane_progress(
    lane: str,
    *,
    total: int,
    unit: str = "item",
    desc: str = "",
) -> Iterator:
    """Context manager that yields a bar object with .start_item / .update / .done.

    The bar auto-completes on context exit (whether via normal flow or
    exception). Both rich and line backends are exception-safe.
    """
    state = _BarState(
        lane=lane,
        desc=desc or lane,
        unit=unit,
        total=max(1, total),
    )

    mode = _detect_mode()
    if mode == "rich":
        try:
            bar: object = _RichBar(state)
        except ImportError:
            # rich not installed for some reason — degrade silently
            bar = _LineBar(state)
    else:
        bar = _LineBar(state)

    try:
        yield bar
    finally:
        try:
            bar.done()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Subprocess prefix tagging — parent calls this before spawning a child
# ---------------------------------------------------------------------------

def child_env(lane: str, base_env: dict | None = None) -> dict:
    """Return an env dict suitable for a subprocess running `lane`.

    Forces line-mode progress (no rich live cursor games), sets the lane
    prefix marker, and inherits the parent's env if not given.
    """
    env = dict(base_env if base_env is not None else os.environ)
    env["VIDEO_USE_PROGRESS_MODE"] = "line"
    env["VIDEO_USE_LANE_PREFIX"] = lane
    # Force UTF-8 in child Python — Windows default is sometimes cp1252
    # which mangles ▌ █ characters in the rich fallback.
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def install_lane_prefix() -> None:
    """Wrap stdout/stderr so every line emitted in this process is tagged
    with [<lane>] from VIDEO_USE_LANE_PREFIX. No-op if env var unset.

    Safe to call multiple times — second call is a no-op.
    """
    prefix = os.environ.get("VIDEO_USE_LANE_PREFIX", "").strip()
    if not prefix:
        return

    tag = f"[{prefix}]"

    class _Prefixed:
        """Line-buffered prefixer over an underlying stream."""
        def __init__(self, underlying):
            self._u = underlying
            self._buf = ""
            self._tagged = True  # set by class to avoid double-wrapping

        def write(self, s: str) -> int:
            if not s:
                return 0
            self._buf += s
            written = 0
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                # Don't double-tag lines that already start with the prefix
                # (e.g. [PROGRESS ...] from progress.py).
                if line.startswith("[") and " lane=" in line[:40]:
                    written += self._u.write(line + "\n")
                else:
                    written += self._u.write(f"{tag} {line}\n")
            return written

        def flush(self) -> None:
            if self._buf:
                self._u.write(f"{tag} {self._buf}")
                self._buf = ""
            self._u.flush()

        def isatty(self) -> bool:
            return False  # we're not a real TTY anymore

        def __getattr__(self, name):
            return getattr(self._u, name)

    if not getattr(sys.stdout, "_tagged", False):
        sys.stdout = _Prefixed(sys.stdout)
    if not getattr(sys.stderr, "_tagged", False):
        sys.stderr = _Prefixed(sys.stderr)
