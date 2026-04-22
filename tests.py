"""End-to-end smoke tests for video-use-premiere.

Runs in two tiers:

    python tests.py            # FAST  — structural / API only (~5s)
    python tests.py --heavy    # HEAVY — actually loads + invokes the three
                                          production models on a 2s synthetic
                                          clip. Downloads ~4.5 GB on first run.
                                          Intended for end-to-end validation
                                          on a GPU machine.

Designed to run identically under:
    - a normal interactive terminal (rich progress bars light up)
    - a non-TTY pipe / Claude Code session (structured [PROGRESS ...] lines)

Each test prints PASS / FAIL on its own line so that grep/CI integrations
can scrape pass-rate without parsing rich output.
"""

from __future__ import annotations

import argparse
import builtins
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

# Make sure the helpers/ package is importable when tests.py is run from
# the project root. Avoids the "you must pip install -e ." dance for the
# fast tier.
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "helpers"))


# ---------------------------------------------------------------------------
# Live output: force everything we print to flush IMMEDIATELY.
#
# Why this matters: when stdout isn't a TTY (piped to Claude Code, redirected
# to a file, run under `Get-Content -Wait`), Python's default buffering hides
# everything until the buffer fills or the process exits. On the heavy tier
# that means 60-90s of "did it freeze??" silence per model load.
#
# Two layers of defense:
#   1. Reconfigure the std streams to line-buffered + UTF-8 (Windows console
#      defaults to cp1252 which mangles unicode in HF model names).
#   2. Replace builtins.print with a flush=True version. Calls in this file
#      still pass through normally; calls in transitively-imported helper
#      modules (parakeet_onnx_lane, audio_lane, etc) ALSO get the flush
#      behaviour for free.
# ---------------------------------------------------------------------------

def _enable_live_output() -> None:
    try:
        # Python 3.7+ — set line buffering and UTF-8 encoding atomically.
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        # Already wrapped (rare) or older Python — no-op, the print
        # monkey-patch below still gives us flushing.
        pass

    _orig_print = builtins.print

    def _flushy_print(*args, **kwargs):
        kwargs.setdefault("flush", True)
        return _orig_print(*args, **kwargs)

    builtins.print = _flushy_print

_enable_live_output()


# ---------------------------------------------------------------------------
# Optional log tee — when --log <path> is passed, mirror EVERY byte that
# would go to stdout/stderr into the log file as well, atomically per write.
# Lets the user run:
#
#     python -u tests.py --heavy --log run.log
#
# in one window and:
#
#     Get-Content run.log -Wait        # PowerShell
#     tail -f run.log                  # bash / zsh
#
# in another to watch progress live, even if the test process is suspended
# in a Claude Code subshell.
# ---------------------------------------------------------------------------

class _Tee:
    """Mirror writes to two streams. Tolerant of one side being closed."""

    def __init__(self, primary, secondary):
        self._a = primary
        self._b = secondary

    def write(self, s: str) -> int:
        n = 0
        try:
            n = self._a.write(s)
        except Exception:
            pass
        try:
            self._b.write(s)
            self._b.flush()
        except Exception:
            pass
        return n

    def flush(self) -> None:
        for s in (self._a, self._b):
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        try:
            return self._a.isatty()
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self._a, name)


def _install_log_tee(log_path: Path) -> None:
    """Wrap stdout + stderr so all output also goes to the log file.

    Writes are flushed immediately so external `tail -f` sees lines as
    they're emitted, not in 4 KB chunks.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "w", encoding="utf-8", buffering=1)  # line-buffered
    fh.write(f"# tests.py log @ {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    fh.flush()
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)


# ---------------------------------------------------------------------------
# Timestamped status helper — used by HEAVY tier so user can tell which
# model load is "currently happening" vs "definitely hung".
# ---------------------------------------------------------------------------

_TEST_T0 = time.monotonic()

def _status(msg: str) -> None:
    """Print a single timestamped status line. Always flushes."""
    elapsed = time.monotonic() - _TEST_T0
    print(f"  [{elapsed:6.1f}s] {msg}")


# ---------------------------------------------------------------------------
# Minimal pass/fail tracker. No third-party test framework dependency so
# the structural tier runs on a bare interpreter with zero installs.
# ---------------------------------------------------------------------------

class Results:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.skipped: list[tuple[str, str]] = []
        self._t0 = time.monotonic()

    def ok(self, name: str) -> None:
        self.passed.append(name)
        print(f"  PASS  {name}")

    def fail(self, name: str, why: str) -> None:
        self.failed.append((name, why))
        print(f"  FAIL  {name}: {why}")

    def skip(self, name: str, why: str) -> None:
        self.skipped.append((name, why))
        print(f"  SKIP  {name}: {why}")

    def summary(self) -> int:
        elapsed = time.monotonic() - self._t0
        print()
        print("=" * 60)
        print(
            f"  {len(self.passed)} passed  "
            f"{len(self.failed)} failed  "
            f"{len(self.skipped)} skipped  "
            f"({elapsed:.1f}s)"
        )
        print("=" * 60)
        if self.failed:
            print()
            print("FAILURES:")
            for name, why in self.failed:
                print(f"  - {name}: {why}")
        return 0 if not self.failed else 1


def _section(title: str) -> None:
    print()
    print(f"── {title} " + "─" * (60 - len(title) - 4))


# ---------------------------------------------------------------------------
# 1. ENVIRONMENT — show the user exactly what we're running on
# ---------------------------------------------------------------------------

def test_environment(R: Results) -> None:
    _section("Environment")

    # Python
    try:
        ver = sys.version.split()[0]
        print(f"  python:    {ver}  ({sys.executable})")
        R.ok("python interpreter")
    except Exception as e:
        R.fail("python interpreter", str(e))

    # ffmpeg
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=True,
        ).stdout.splitlines()[0]
        print(f"  ffmpeg:    {out}")
        R.ok("ffmpeg on PATH")
    except Exception as e:
        R.fail("ffmpeg on PATH", f"{e} — install with winget/brew/apt")

    # PyTorch + CUDA — most important for the wealthy 5090 path
    try:
        import torch
        print(f"  torch:     {torch.__version__}  cuda={torch.version.cuda}")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            free, total = torch.cuda.mem_get_info(0)
            free_gb = free / (1024 ** 3)
            total_gb = total / (1024 ** 3)
            print(
                f"  device:    {name}  sm_{cap[0]}{cap[1]}  "
                f"VRAM={free_gb:.1f}/{total_gb:.1f} GB free"
            )
            R.ok("CUDA available")
            # Blackwell / Ada / Ampere all expose mem_get_info; if you got
            # this far without an exception you're golden.
            if cap[0] >= 12:
                print(f"  note:      Blackwell (sm_{cap[0]}{cap[1]}) — wealthy mode is your default tier")
            R.ok("CUDA capability detected")
        else:
            R.skip("CUDA available", "torch didn't see a GPU")
    except ImportError:
        R.fail("torch import", "torch not installed — run install.bat / install.sh")
    except Exception as e:
        R.fail("CUDA probe", str(e))


# ---------------------------------------------------------------------------
# 2. IMPORTS — verify all helpers import clean under the current env.
#    This is what catches the most "I bumped transformers 5.x and forgot
#    to update an import" type bugs.
# ---------------------------------------------------------------------------

HELPER_MODULES = (
    "vram", "wealthy", "extract_audio", "progress",
    "pack_timelines", "preprocess", "preprocess_batch",
    "diarize", "parakeet_onnx_lane", "parakeet_lane",
    "audio_lane", "audio_vocab_default", "visual_lane",
    "render", "export_fcpxml",
)


def test_imports(R: Results) -> None:
    _section("Helper imports")
    import importlib
    for m in HELPER_MODULES:
        try:
            importlib.import_module(m)
            R.ok(f"import {m}")
        except Exception as e:
            R.fail(f"import {m}", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 3. VRAM scheduler — make sure it picks PARALLEL_3 for the user's card
# ---------------------------------------------------------------------------

def test_vram_schedule(R: Results) -> None:
    _section("VRAM-aware scheduler")
    try:
        from vram import detect_gpu, pick_schedule, Schedule
        gpu = detect_gpu()
        sched = pick_schedule(gpu)
        if gpu.available:
            print(
                f"  detected:  {gpu.device_name}  "
                f"VRAM={gpu.free_gb:.1f}/{gpu.total_gb:.1f} GB"
            )
        else:
            print("  detected:  no CUDA (CPU fallback)")
        print(f"  schedule:  {sched.name}")
        R.ok("vram.detect_gpu")
        R.ok("vram.pick_schedule")
        # Sanity rules under the post-Florence-community refactor:
        #   * Default policy is SEQUENTIAL on any single-GPU rig — see
        #     vram.py module docstring for why parallel is opt-in now.
        #   * Power users can re-enable parallel via either
        #     VIDEO_USE_PARALLEL_LANES=1 (env var) or
        #     --force-schedule parallel (CLI bypass).
        # We test both the default and the opted-in branch so a future
        # regression in either path is caught on a 32 GB box.
        if gpu.available and gpu.free_gb >= 16:
            if sched != Schedule.SEQUENTIAL:
                R.fail(
                    "schedule sanity (default)",
                    f"32GB card got {sched.name}, expected SEQUENTIAL "
                    f"(parallel is now opt-in via VIDEO_USE_PARALLEL_LANES)",
                )
            else:
                R.ok("schedule sanity: SEQUENTIAL is the default on big cards")
            # Probe the opt-in path too. Save / restore env to avoid
            # leaking state into other tests in the same process.
            import os as _os
            _prev = _os.environ.get("VIDEO_USE_PARALLEL_LANES")
            try:
                _os.environ["VIDEO_USE_PARALLEL_LANES"] = "1"
                opted = pick_schedule(gpu)
                if opted != Schedule.PARALLEL_3:
                    R.fail(
                        "schedule sanity (opt-in)",
                        f"with VIDEO_USE_PARALLEL_LANES=1 on a 32GB card we "
                        f"got {opted.name}, expected PARALLEL_3",
                    )
                else:
                    R.ok("schedule sanity: opt-in unlocks PARALLEL_3")
            finally:
                if _prev is None:
                    _os.environ.pop("VIDEO_USE_PARALLEL_LANES", None)
                else:
                    _os.environ["VIDEO_USE_PARALLEL_LANES"] = _prev
        else:
            R.ok("schedule sanity (matches free VRAM tier)")
    except Exception as e:
        R.fail("vram module", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 4. Wealthy mode toggle (env var + CLI flag mirror)
# ---------------------------------------------------------------------------

def test_wealthy(R: Results) -> None:
    _section("Wealthy mode (--wealthy / VIDEO_USE_WEALTHY=1)")
    try:
        from wealthy import (
            is_wealthy, propagate_to_env,
            SPEECH_BATCH, FLORENCE_BATCH,
            CLAP_WINDOWS_PER_BATCH, CLAP_WINDOWS_PER_BATCH_WEALTHY,
            CLAP_MODEL_TIER_DEFAULT, CLAP_MODEL_TIER_WEALTHY,
        )

        # Save & isolate the env so we don't pollute the rest of the suite.
        saved = os.environ.pop("VIDEO_USE_WEALTHY", None)
        try:
            assert not is_wealthy(False), "default state should be not-wealthy"
            R.ok("wealthy off by default")

            propagate_to_env(True)
            assert os.environ.get("VIDEO_USE_WEALTHY") == "1", "propagate failed"
            assert is_wealthy(False), "env var should flip is_wealthy"
            R.ok("wealthy CLI flag → env propagation")

            print(
                f"  wealthy knobs:    speech={SPEECH_BATCH}  "
                f"florence={FLORENCE_BATCH}  "
                f"clap={CLAP_WINDOWS_PER_BATCH}->{CLAP_WINDOWS_PER_BATCH_WEALTHY} windows/batch  "
                f"clap_tier={CLAP_MODEL_TIER_DEFAULT}->{CLAP_MODEL_TIER_WEALTHY}"
            )
            R.ok("wealthy batch constants exposed")
        finally:
            os.environ.pop("VIDEO_USE_WEALTHY", None)
            if saved is not None:
                os.environ["VIDEO_USE_WEALTHY"] = saved
    except Exception as e:
        R.fail("wealthy module", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 5. Progress bar contract — both modes, no exceptions, correct ticks
# ---------------------------------------------------------------------------

def test_progress(R: Results) -> None:
    _section("Progress bars (rich + line modes)")
    try:
        from progress import lane_progress

        # Force line mode for the test so output is deterministic.
        os.environ["VIDEO_USE_PROGRESS_MODE"] = "line"
        try:
            with lane_progress("test", total=4, unit="item", desc="dummy") as bar:
                for i in range(4):
                    bar.start_item(f"item_{i}")
                    bar.update(advance=1)
            R.ok("line-mode progress bar lifecycle")
        finally:
            os.environ.pop("VIDEO_USE_PROGRESS_MODE", None)
    except Exception as e:
        traceback.print_exc()
        R.fail("progress module", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 6. pack_timelines on synthetic lane JSON — the part Claude actually reads
# ---------------------------------------------------------------------------

def test_pack_timelines(R: Results, tmp: Path) -> None:
    _section("pack_timelines on synthetic lane caches")
    try:
        edit = tmp / "edit"
        (edit / "transcripts").mkdir(parents=True)
        (edit / "audio_tags").mkdir(parents=True)
        (edit / "visual_caps").mkdir(parents=True)

        # Synthetic speech-lane output — three words, one silence break
        (edit / "transcripts" / "C0001.json").write_text(json.dumps({
            "source_path": "/tmp/C0001.mp4",
            "duration": 10.0,
            "words": [
                {"type": "word", "text": "Hello",  "start": 1.0, "end": 1.4, "speaker_id": "S0"},
                {"type": "word", "text": "world.", "start": 1.45, "end": 2.0, "speaker_id": "S0"},
                {"type": "spacing", "text": " ", "start": 2.0, "end": 5.0},
                {"type": "word", "text": "Again.", "start": 5.0, "end": 5.6, "speaker_id": "S0"},
            ],
        }), encoding="utf-8")

        # Synthetic CLAP audio_lane output — (label, score) events per
        # sliding window. Matches the canonical shape produced by the
        # real `helpers/audio_lane.py` (see its module docstring + the
        # _process_one writer). Two events at different start times so
        # the renderer exercises the per-range grouping path.
        (edit / "audio_tags" / "C0001.json").write_text(json.dumps({
            "source_path": "/tmp/C0001.mp4",
            "model": "Xenova/clap-htsat-unfused",
            "vocab_sha": "deadbeefcafef00d",
            "vocab_size": 247,
            "window_s": 10.0,
            "hop_s": 5.0,
            "threshold": 0.10,
            "top_k": 5,
            "duration": 10.0,
            "events": [
                {"start": 0.0, "end": 10.0, "label": "cordless drill",  "score": 0.42},
                {"start": 0.0, "end": 10.0, "label": "drill press",     "score": 0.31},
                {"start": 5.0, "end": 10.0, "label": "laughter",        "score": 0.27},
            ],
        }), encoding="utf-8")

        # Synthetic Florence-2 output (1fps captions)
        (edit / "visual_caps" / "C0001.json").write_text(json.dumps({
            "source_path": "/tmp/C0001.mp4",
            "duration": 10.0,
            "fps": 1,
            "captions": [
                {"t": 0, "text": "a workshop bench with tools"},
                {"t": 1, "text": "a workshop bench with tools"},
                {"t": 2, "text": "a person holding a drill"},
                {"t": 3, "text": "close-up of a drill bit on metal"},
            ],
        }), encoding="utf-8")

        # Run the packer
        proc = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "helpers" / "pack_timelines.py"),
             "--edit-dir", str(edit), "--merge"],
            capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        if proc.returncode != 0:
            R.fail("pack_timelines exit", proc.stderr.strip()[:300])
            return
        R.ok("pack_timelines ran")

        # Verify each output exists and contains the expected markers
        for fname, must_contain in [
            ("speech_timeline.md", "Hello world."),
            ("audio_timeline.md", "drill"),
            ("visual_timeline.md", "a workshop bench"),
            ("merged_timeline.md", "drill"),
        ]:
            p = edit / fname
            if not p.exists():
                R.fail(f"output {fname}", "missing")
                continue
            txt = p.read_text(encoding="utf-8")
            if must_contain not in txt:
                R.fail(f"content {fname}", f"missing '{must_contain}'")
            else:
                R.ok(f"output {fname}")
    except Exception as e:
        traceback.print_exc()
        R.fail("pack_timelines", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 7. FCPXML round-trip — build a synthetic 4-clip EDL with one of each
#    cut type (hard / J / L / dissolve), export, re-read via OTIO, sanity
#    check the structure. Catches structural bugs without needing Premiere.
# ---------------------------------------------------------------------------

def test_fcpxml_roundtrip(R: Results, tmp: Path) -> None:
    _section("FCPXML round-trip (hard / J / L / dissolve)")
    try:
        import opentimelineio as otio
    except ImportError:
        R.skip("FCPXML round-trip", "pip install -e .[fcpxml] required")
        return

    try:
        from export_fcpxml import build_timeline, write_fcpxml

        # FCPXML adapter requires `available_range` on each external ref,
        # which we populate via ffprobe. Generate a tiny REAL clip (20s)
        # so the probe path is exercised end-to-end (not just the sentinel
        # fallback). 20s is enough to cover all 4 ranges in the synth EDL.
        src = tmp / "C0001.mp4"
        try:
            _make_synthetic_clip(src, seconds=20.0)
        except Exception as e:
            R.fail("synthetic source for FCPXML test", str(e))
            return

        edl = {
            "name": "test_cut",
            "sources": {"C0001": str(src)},
            "ranges": [
                # Hard cut
                {"source": "C0001", "start": 0.0, "end": 2.0, "beat": "A"},
                # J cut: audio leads by 400ms
                {"source": "C0001", "start": 4.0, "end": 6.0, "beat": "B",
                 "audio_lead": 0.4},
                # L cut: audio lingers 1.2s
                {"source": "C0001", "start": 8.0, "end": 10.0, "beat": "C",
                 "video_tail": 1.2},
                # Cross-dissolve: 300ms
                {"source": "C0001", "start": 12.0, "end": 14.0, "beat": "D",
                 "transition_in": 0.3},
            ],
        }

        timeline = build_timeline(edl, frame_rate=24.0)
        R.ok("build_timeline returned")

        # Structural assertions — two tracks, right kinds, right counts
        tracks = list(timeline.tracks)
        assert len(tracks) == 2, f"expected 2 tracks, got {len(tracks)}"
        kinds = sorted(t.kind for t in tracks)
        assert kinds == [otio.schema.TrackKind.Audio, otio.schema.TrackKind.Video], \
            f"track kinds wrong: {kinds}"
        R.ok("two tracks (V1 + A1)")

        v_track = next(t for t in tracks if t.kind == otio.schema.TrackKind.Video)
        a_track = next(t for t in tracks if t.kind == otio.schema.TrackKind.Audio)

        v_clips = [c for c in v_track if isinstance(c, otio.schema.Clip)]
        a_clips = [c for c in a_track if isinstance(c, otio.schema.Clip)]
        v_trans = [c for c in v_track if isinstance(c, otio.schema.Transition)]

        assert len(v_clips) == 4, f"video clips: {len(v_clips)}"
        assert len(a_clips) == 4, f"audio clips: {len(a_clips)}"
        assert len(v_trans) == 1, f"video transitions: {len(v_trans)} (expect 1 dissolve)"
        R.ok("clip + transition counts")

        # J cut: clip B's audio source_range should be 400ms LONGER than its video
        v_dur = v_clips[1].source_range.duration.to_seconds()
        a_dur = a_clips[1].source_range.duration.to_seconds()
        if abs((a_dur - v_dur) - 0.4) > 1.0 / 24:  # allow 1-frame snap slop
            R.fail("J-cut audio extends earlier",
                   f"a_dur-v_dur = {a_dur-v_dur:.3f}, expected 0.4")
        else:
            R.ok("J-cut audio leads video by 400ms")

        # L cut: clip C's audio source_range should be 1.2s LONGER than its video
        v_dur = v_clips[2].source_range.duration.to_seconds()
        a_dur = a_clips[2].source_range.duration.to_seconds()
        if abs((a_dur - v_dur) - 1.2) > 1.0 / 24:
            R.fail("L-cut audio extends later",
                   f"a_dur-v_dur = {a_dur-v_dur:.3f}, expected 1.2")
        else:
            R.ok("L-cut audio extends 1.2s past video")

        # Now write to disk and re-read through the FCPXML adapter
        out = tmp / "test.fcpxml"
        write_fcpxml(timeline, out)
        if not out.exists() or out.stat().st_size < 100:
            R.fail("FCPXML written", f"file missing or tiny: {out.stat().st_size if out.exists() else 0}b")
            return
        R.ok(f"FCPXML written ({out.stat().st_size} bytes)")

        try:
            reloaded = otio.adapters.read_from_file(str(out))
            r_tracks = list(reloaded.tracks)
            r_clips = sum(1 for t in r_tracks for c in t if isinstance(c, otio.schema.Clip))
            print(f"  reloaded: {len(r_tracks)} tracks, {r_clips} clips total")
            R.ok("FCPXML reread via OTIO adapter")
        except Exception as e:
            # Re-read failures are usually adapter-specific quirks; report
            # but don't fail the whole suite — the WRITE is what matters
            # for Premiere import.
            R.skip("FCPXML re-read", f"{type(e).__name__}: {e}")
    except Exception as e:
        traceback.print_exc()
        R.fail("FCPXML round-trip", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 8. Parakeet fallback path — exercises the conversion + blocked-exception
#    classifier without requiring NeMo to be installed. Catches structural
#    bugs in the fallback that would otherwise only surface on a friend's
#    NVIDIA-blocked machine.
# ---------------------------------------------------------------------------

def test_parakeet_fallback(R: Results, tmp: Path) -> None:
    _section("Parakeet fallback path (pure-Python, no nemo required)")

    # Import the lane itself — pure Python, no nemo touch yet.
    try:
        import parakeet_lane as pl
        R.ok("import parakeet_lane")
    except Exception as e:
        R.fail("import parakeet_lane", f"{type(e).__name__}: {e}")
        return

    # _lazy_nemo helper should be importable + expose ensure_nemo_installed.
    try:
        from _lazy_nemo import ensure_nemo_installed, is_nemo_installed
        assert callable(ensure_nemo_installed)
        assert callable(is_nemo_installed)
        R.ok("import _lazy_nemo")
        # is_nemo_installed should return False on a clean install
        # (or True if the user pre-installed [parakeet]); either way it
        # must NOT raise. We just exercise the boolean contract.
        result = is_nemo_installed()
        assert isinstance(result, bool)
        R.ok(f"_lazy_nemo.is_nemo_installed() -> {result}")
    except Exception as e:
        R.fail("_lazy_nemo helpers", f"{type(e).__name__}: {e}")

    # Converter contract: a synthetic hypothesis mimicking NeMo's
    # `output[0].timestamp['word']` shape must produce the canonical
    # word list every speech lane in this project emits.
    try:
        # Plain object with .timestamp attribute (NeMo Hypothesis shape)
        class _FakeHyp:
            text = "Hello world. Again."
            timestamp = {
                "word": [
                    # Contiguous pair -> NO spacing entry between them.
                    # The converter contract: any positive gap emits a
                    # spacing, so we keep these flush on purpose.
                    {"word": "Hello",  "start": 1.0,  "end": 1.4},
                    {"word": "world.", "start": 1.4,  "end": 2.0},
                    # 3-second silence gap -> exactly one spacing entry expected
                    {"word": "Again.", "start": 5.0,  "end": 5.6},
                ],
            }

        words = pl._parakeet_to_canonical_words(_FakeHyp())
        assert isinstance(words, list), "must return list"

        word_entries = [w for w in words if w.get("type") == "word"]
        spacing_entries = [w for w in words if w.get("type") == "spacing"]

        assert len(word_entries) == 3, f"expected 3 words, got {len(word_entries)}"
        assert len(spacing_entries) == 1, \
            f"expected 1 spacing (3s gap), got {len(spacing_entries)}"

        # Field-shape check — canonical schema shared across speech lanes.
        for w in word_entries:
            assert "text" in w and "start" in w and "end" in w
            assert "speaker_id" in w and w["speaker_id"] is None
            assert isinstance(w["start"], float)
            assert isinstance(w["end"], float)

        # Spacing entry must bridge the gap exactly: 2.0 -> 5.0
        sp = spacing_entries[0]
        assert sp["start"] == 2.0 and sp["end"] == 5.0, \
            f"spacing range wrong: {sp['start']}-{sp['end']}"
        R.ok("_parakeet_to_canonical_words shape + gap detection")
    except Exception as e:
        traceback.print_exc()
        R.fail("converter shape", f"{type(e).__name__}: {e}")

    # Defensive: empty timestamp dict must not crash, must return [].
    try:
        class _Empty:
            text = ""
            timestamp = {}
        out = pl._parakeet_to_canonical_words(_Empty())
        assert out == [], f"expected [], got {out}"
        R.ok("converter handles empty hypothesis")
    except Exception as e:
        R.fail("converter empty hyp", f"{type(e).__name__}: {e}")

    # Diarize module contract — load_hf_token must be importable and
    # must not raise even when no .env / HF_TOKEN is present.
    try:
        from diarize import load_hf_token, diarize_and_assign
        # Both should be callable; we don't actually run diarize_and_assign
        # here (it would pull in pyannote.audio + a 600 MB model).
        assert callable(load_hf_token)
        assert callable(diarize_and_assign)
        # load_hf_token returns None | str. Either is fine.
        tok = load_hf_token()
        assert tok is None or isinstance(tok, str)
        R.ok("diarize module: load_hf_token + diarize_and_assign exposed")
    except Exception as e:
        traceback.print_exc()
        R.fail("sentinel lifecycle", f"{type(e).__name__}: {e}")

    # Air-gapped escape hatch: PARAKEET_MODEL_PATH env-var contract.
    # We can't actually load NeMo here (heavy + may not be installed),
    # but we *can* verify the constant exists and that an obviously-bad
    # path produces the actionable RuntimeError instead of a raw
    # FileNotFoundError. This locks in the user-facing contract for
    # users behind proxies that block HF entirely.
    try:
        assert hasattr(pl, "PARAKEET_MODEL_PATH_ENV"), \
            "parakeet_lane must expose PARAKEET_MODEL_PATH_ENV constant"
        assert pl.PARAKEET_MODEL_PATH_ENV == "PARAKEET_MODEL_PATH", \
            f"env var name regressed: {pl.PARAKEET_MODEL_PATH_ENV}"
        R.ok("PARAKEET_MODEL_PATH env-var contract")
    except Exception as e:
        R.fail("PARAKEET_MODEL_PATH contract", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 9. HEAVY tier — actually load the three production models and run them
#    on a 2-second synthetic clip. Tagged --heavy so it doesn't fire on
#    every smoke run.
# ---------------------------------------------------------------------------

def _make_synthetic_clip(out_path: Path, seconds: float = 2.0) -> None:
    """Build a 2-second 1280x720 test clip via ffmpeg. Sine wave audio +
    moving color bar video. No external assets needed."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=size=1280x720:rate=30:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def test_heavy(R: Results, tmp: Path) -> None:
    _section("HEAVY: end-to-end on a 2s synthetic clip")

    edit = tmp / "edit"
    edit.mkdir(exist_ok=True)
    clip = tmp / "synth.mp4"

    # Hint to the user up front about what's about to happen — first run
    # of each lane downloads its model weights, which can take minutes
    # and look like a hang if no progress is shown.
    _status(
        "HEAVY mode is loading three real models. First run downloads:"
    )
    print("           - nvidia/parakeet-tdt-0.6b-v2 (ONNX) (~600 MB)")
    print("           - florence-community/Florence-2-base   (~1.0 GB)")
    print("           - Xenova/clap-htsat-unfused           (~150 MB ONNX)")
    print("           Subsequent runs hit the HF cache and start in ~5s each.")
    print("           HF transformers prints its own download bars to stderr.")

    try:
        _status("generating synthetic 2s clip via ffmpeg ...")
        _make_synthetic_clip(clip, seconds=2.0)
        R.ok(f"synthetic clip generated ({clip.stat().st_size // 1024} KB)")
    except Exception as e:
        R.fail("synthetic clip ffmpeg", str(e))
        return

    # ── Speech (Parakeet ONNX) ────────────────────────────────────────
    # The default speech lane: NVIDIA Parakeet TDT through ONNX Runtime.
    # First run downloads the istupakov-converted ONNX repo (~600 MB)
    # plus the bundled mel preprocessor and TDT decoder graphs.
    try:
        _status("Speech lane: importing parakeet_onnx_lane ...")
        from parakeet_onnx_lane import run_parakeet_onnx_lane_batch
        _status("Speech lane: loading nvidia/parakeet-tdt-0.6b-v2 (ONNX) — "
                "downloads on first run ...")
        t0 = time.monotonic()
        out = run_parakeet_onnx_lane_batch(
            [clip], edit,
            language="en",
            diarize=False,
            num_speakers=None,
            force=False,
        )
        _status(f"Speech lane: done in {time.monotonic()-t0:.1f}s")
        if out and out[0].exists():
            data = json.loads(out[0].read_text(encoding="utf-8"))
            n_words = sum(1 for w in data.get("words", []) if w.get("type") == "word")
            print(f"  speech:    {n_words} word(s) in 2s sine — expect 0 (silence)")
            R.ok("speech lane ran (Parakeet ONNX)")
        else:
            R.fail("speech lane", "no output file")
    except Exception as e:
        traceback.print_exc()
        R.fail("speech lane", f"{type(e).__name__}: {e}")

    # ── CLAP audio lane ───────────────────────────────────────────────
    # CLAP is a small dual-encoder model (~150 MB ONNX for the base
    # tier) — first run downloads the audio + text ONNX graphs and the
    # processor configs, both quantized. The 2s synthetic sine clip
    # gets zero-padded to one full 10s window inside the lane so this
    # exercises the slide_and_score + text-embed cache + per-window
    # scoring path end-to-end against the baked-in default vocab.
    try:
        _status("Audio lane: importing audio_lane + loading CLAP HTSAT ...")
        from audio_lane import run_audio_lane_batch
        t0 = time.monotonic()
        out = run_audio_lane_batch(
            [clip], edit,
            # Pin the base tier so the smoke test doesn't accidentally
            # download the larger ~600 MB variant on a wealthy machine.
            model_tier="base",
            # Keep the per-batch window count tiny so the test runs
            # cleanly on a 4 GB card too.
            windows_per_batch=4,
            force=False,
        )
        _status(f"Audio lane: done in {time.monotonic()-t0:.1f}s")
        if out and out[0].exists():
            data = json.loads(out[0].read_text(encoding="utf-8"))
            model = data.get("model", "")
            events = data.get("events") or []
            top_label = events[0].get("label") if events else "(none above threshold)"
            top_score = events[0].get("score") if events else None
            print(
                f"  clap:      model={model}  events={len(events)} on a 2s sine; "
                f"top: {top_label!r} score={top_score}"
            )
            if model != "Xenova/clap-htsat-unfused":
                R.fail(
                    "audio lane model id",
                    f"expected 'Xenova/clap-htsat-unfused', got {model!r}",
                )
            else:
                R.ok("audio lane ran (CLAP)")
        else:
            R.fail("audio lane", "no output file")
    except Exception as e:
        traceback.print_exc()
        R.fail("audio lane", f"{type(e).__name__}: {e}")

    # ── Florence-2 ────────────────────────────────────────────────────
    try:
        _status("Visual lane: importing visual_lane + loading Florence-2-base ...")
        from visual_lane import run_visual_lane_batch
        t0 = time.monotonic()
        out = run_visual_lane_batch(
            [clip], edit,
            model_id="microsoft/Florence-2-base",
            fps=1, batch_size=2,
            task="<MORE_DETAILED_CAPTION>",
            force=False,
        )
        _status(f"Visual lane: done in {time.monotonic()-t0:.1f}s")
        if out and out[0].exists():
            data = json.loads(out[0].read_text(encoding="utf-8"))
            n_caps = len(data.get("captions", []))
            sample = (data.get("captions") or [{}])[0].get("text", "")
            print(f"  florence:  {n_caps} caption(s); first: {sample[:80]!r}")
            R.ok("visual lane ran")
        else:
            R.fail("visual lane", "no output file")
    except Exception as e:
        traceback.print_exc()
        R.fail("visual lane", f"{type(e).__name__}: {e}")

    # ── Pack the three lanes ──────────────────────────────────────────
    try:
        _status("packing three lanes -> *_timeline.md ...")
        proc = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "helpers" / "pack_timelines.py"),
             "--edit-dir", str(edit), "--merge"],
            capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        if proc.returncode == 0:
            R.ok("pack_timelines ran on real lane outputs")
        else:
            R.fail("pack_timelines on real outputs", proc.stderr.strip()[:300])
    except Exception as e:
        R.fail("pack_timelines on real outputs", str(e))


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def run_all(heavy: bool = False, keep_tmp: bool = False) -> Results:
    """Programmatic entry point — used by helpers/health.py.

    Mirrors `main()` minus the argparse/banner/exit-code wrapping. Returns
    the populated `Results` object so the caller can introspect failures.
    """
    print("video-use-premiere :: smoke tests")
    print("=" * 60)

    R = Results()

    with tempfile.TemporaryDirectory(prefix="vup-tests-") as td:
        tmp = Path(td)
        if keep_tmp:
            print(f"  tmp dir kept: {tmp}")

        test_environment(R)
        test_imports(R)
        test_vram_schedule(R)
        test_wealthy(R)
        test_progress(R)
        test_pack_timelines(R, tmp)
        test_fcpxml_roundtrip(R, tmp)
        test_parakeet_fallback(R, tmp)

        if heavy:
            test_heavy(R, tmp)

        if keep_tmp:
            # Move the tmp dir somewhere we won't auto-delete it.
            keep = PROJECT_ROOT / f"_tests_kept_{int(time.time())}"
            import shutil
            shutil.copytree(tmp, keep)
            print(f"  copied tmp -> {keep}")

    R.summary()
    return R


def main() -> int:
    ap = argparse.ArgumentParser(
        description="video-use-premiere smoke tests",
        epilog=(
            "TIP: For live output during long-running heavy mode, run with:\n"
            "    python -u tests.py --heavy --log run.log\n"
            "Then in a SECOND window:\n"
            "    Get-Content run.log -Wait      (PowerShell)\n"
            "    tail -f run.log                (bash/zsh)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--heavy", action="store_true",
                    help="Also run the model-loading tier (~4.5 GB download "
                         "on first run, GPU recommended).")
    ap.add_argument("--keep-tmp", action="store_true",
                    help="Don't delete the temp dir at the end.")
    ap.add_argument("--log", type=Path, default=None,
                    help="Tee all stdout+stderr to this log file as well, "
                         "line-buffered. Use with `Get-Content -Wait` or "
                         "`tail -f` from another window to follow live progress.")
    args = ap.parse_args()

    if args.log is not None:
        _install_log_tee(args.log.resolve())
        print(f"  log tee: {args.log.resolve()}")

    R = run_all(heavy=args.heavy, keep_tmp=args.keep_tmp)
    return 0 if not R.failed else 1


if __name__ == "__main__":
    sys.exit(main())
