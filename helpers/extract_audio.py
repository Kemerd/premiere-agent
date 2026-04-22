"""Extract a 16 kHz mono PCM WAV from a source video, exactly once.

Parakeet wants 16 kHz mono PCM, and the CLAP audio lane
upsamples the same cache to 48 kHz on the fly. Decoding the source video
twice is wasteful — a 4K H.265 master can take longer to demux than the
actual inference. So we extract once, cache the WAV under
<edit>/audio_16k/, and every audio-consuming lane reads the same file.

Output spec is the canonical ASR-friendly format:

    -ar 16000      sample rate
    -ac 1          mono
    -c:a pcm_s16le signed 16-bit little endian, no compression

Cache invalidation: if the WAV exists AND its mtime is newer than the
source's mtime, the cache is reused. Otherwise it's regenerated.

Usage:
    from extract_audio import extract_audio_for
    wav_path = extract_audio_for(video_path, edit_dir)

CLI:
    python helpers/extract_audio.py <video> [--edit-dir <dir>]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# Public constants — other lanes import these so the wav format contract is
# only declared in one place.
SAMPLE_RATE = 16_000
CHANNELS = 1
SUBDIR = "audio_16k"


def _is_cache_fresh(wav_path: Path, source_path: Path) -> bool:
    """True iff the cached WAV exists and is at least as new as the source.

    We use mtime rather than hashing because hashing a 50GB master is
    silly and editors typically don't edit-in-place — they write a new
    file with a new mtime. If the source is older than the WAV, the WAV
    represents the current contents.
    """
    if not wav_path.exists():
        return False
    try:
        return wav_path.stat().st_mtime >= source_path.stat().st_mtime
    except OSError:
        return False


def extract_audio_for(
    source_path: Path,
    edit_dir: Path,
    *,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """Extract (or reuse cached) 16k mono PCM WAV for the source video.

    Args:
        source_path: Absolute path to the source video file.
        edit_dir:    The session edit directory (e.g. <videos>/edit).
        force:       Bypass the cache check, always re-extract.
        verbose:     Print one line per file. Off in batch contexts.

    Returns:
        Absolute Path to the WAV inside <edit_dir>/audio_16k/.

    Raises:
        FileNotFoundError if source_path doesn't exist.
        subprocess.CalledProcessError if ffmpeg fails.
    """
    source_path = source_path.resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"source video not found: {source_path}")

    out_dir = (edit_dir / SUBDIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{source_path.stem}.wav"

    if not force and _is_cache_fresh(wav_path, source_path):
        if verbose:
            kb = wav_path.stat().st_size / 1024
            print(f"  audio_16k cache hit: {wav_path.name} ({kb:.0f} KB)")
        return wav_path

    # Atomic write: extract to .tmp then rename. Prevents a partial WAV from
    # being left around when ffmpeg crashes / the user CTRL-C's mid-extract.
    #
    # Muxer note: the temp filename ends in ".wav.tmp", which ffmpeg cannot
    # auto-detect a muxer for (it probes by extension, and ".tmp" isn't a
    # known container). Without an explicit `-f wav` ffmpeg returns
    # AVERROR(EINVAL) = -22 and refuses to open the output file. So we pin
    # the muxer here instead of renaming the temp pattern — keeps the
    # atomic-write semantics simple and matches the codec we asked for.
    tmp_path = wav_path.with_suffix(".wav.tmp")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source_path),
        "-vn",
        "-ac", str(CHANNELS),
        "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(tmp_path),
    ]
    if verbose:
        print(f"  audio_16k extract: {source_path.name}")

    # We capture stderr (rather than DEVNULL it) so that on a non-zero exit
    # we can re-raise with the actual ffmpeg diagnostic attached. Bare
    # `check=True` only surfaces the exit code, which for ffmpeg is opaque
    # (e.g. AVERROR(EINVAL) = -22 → 4294967274 on Windows) and forces the
    # caller to re-run by hand to learn what actually went wrong.
    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-12:]) if proc.stderr else "(no stderr)"
        raise subprocess.CalledProcessError(
            proc.returncode, cmd,
            output=None,
            stderr=(
                f"ffmpeg failed (exit {proc.returncode}) extracting audio from\n"
                f"  {source_path}\n"
                f"last lines of ffmpeg stderr:\n{tail}"
            ),
        )

    # On Windows you can't atomically rename onto an existing file with
    # os.rename if the destination exists. Path.replace handles that.
    tmp_path.replace(wav_path)

    if verbose:
        kb = wav_path.stat().st_size / 1024
        print(f"  audio_16k written:  {wav_path.name} ({kb:.0f} KB)")
    return wav_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract 16kHz mono PCM WAV from a video. Cached.",
    )
    ap.add_argument("video", type=Path, help="Path to source video file")
    ap.add_argument(
        "--edit-dir", type=Path, default=None,
        help="Edit output dir (default: <video parent>/edit)",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Bypass cache check, always re-extract.",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    extract_audio_for(video, edit_dir, force=args.force)


if __name__ == "__main__":
    main()
