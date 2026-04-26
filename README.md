# premiere-agent

Edit videos by conversation — runs **100% locally** and exports straight into **Premiere Pro** (and Resolve, and Final Cut) with **J cuts, L cuts, dissolves**, and the rest of the cut vocabulary you'd actually use in an NLE.

Drop raw footage in a folder, chat with your agent ([Claude Code](https://docs.claude.com/en/docs/claude-code/overview) — local CLI **or** [cloud workspaces](https://code.claude.com/docs/en/claude-code-on-the-web), or [Codex](https://developers.openai.com/codex) CLI / IDE / app), get `cut.fcpxml` + `cut.xml` back — XML-only delivery, the cut lives in your NLE. Text + on-demand visuals, no frame-dumping — three perception lanes so the LLM can reason about speech, abstract sounds, and visible objects, not just one of them.

## What it does

- **Cuts out filler words** (`umm`, `uh`, false starts) and dead space between takes
- **J/L cuts on speaker handoffs** so dialogue overlaps the visual cut like a real edit
- **Time-squeezes long activity stretches into timelapses** — 30 minutes of shop work becomes 8 seconds of compressed visual progress, with the audio dropped (or pitch-corrected if you keep it). Speed clamps at 1000%; FCPXML gets `<timeMap>` blocks, Premiere xmeml gets a `Time Remap` filter
- **Captions ambient sound** — knows when the drill is running, when something's being sanded, when there's applause
- **Captions every visible second** so the LLM can find match cuts, B-roll candidates, and identify shots without watching
- **Ships a `master.srt` captions sidecar with every export** (Premiere-friendly UTF-8 + CRLF) — Premiere / Resolve / FCP X import it via `File → Import` onto a captions track the editor restyles in their own caption panel. 2-word UPPERCASE chunks by default, fully customizable
- **Generates animation overlays** via [Manim](https://www.manim.community/), [Remotion](https://www.remotion.dev/), or PIL — spawned in parallel sub-agents, one per animation
- **Self-evaluates the cut decisions** at every EDL boundary against the source clips before handing the XML over
- **Persists session memory** in `project.md` so next week's session picks up where you left off

## How it works

The LLM never watches the video. It **reads** it — through three timestamp-aligned timelines plus an on-demand visual drill-down.

### The three lanes

Pre-processing produces three small markdown files per session, each addressable by `[start-end]` time ranges. The LLM reads them to make cut decisions. Speech + visual run by default in `preprocess.py`; the CLAP audio lane is an agent-driven Phase B step that ships its own custom vocabulary derived from the first two timelines (see the note under Lane 2).

**Lane 1 — `speech_timeline.md`** (Parakeet TDT word-level via ONNX Runtime, phrase-grouped — N parallel sessions, each with a TensorRT or CUDA execution provider):

```
## C0103  (duration: 43.0s, 8 phrases)
  [002.52-005.36] S0 Ninety percent of what a web agent does is completely wasted.
  [006.08-006.74] S0 We fixed this.
```

**Lane 2 — `audio_timeline.md`** (CLAP zero-shot vs an agent-curated vocabulary, top-K above the per-label threshold):

```
## C0103  (duration: 43.0s, 27 events)
  [012.04-012.40] drill (0.87), power_tool (0.71)
  [012.18-012.30] metal_scraping (0.62)
  [018.50-019.10] hammer (0.55)
```

**Lane 3 — `visual_timeline.md`** (Florence-2-base `<MORE_DETAILED_CAPTION>` at 1 fps, deduped):

```
## C0103  (duration: 43.0s, 38 captions)
  [012] a person holding a cordless drill above a metal panel with visible rivet holes
  [013] close-up of a drill bit entering metal, sparks visible
  [014] (same)
  [015] hands placing the drill down on a workbench
```

`pack_timelines.py` also emits a `merged_timeline.md` by default that interleaves all three lanes chronologically — this is the editor sub-agent's default reading surface, so it gets the full multimodal context for every moment from a single file. Pass `--no-merge` to skip it. The per-lane files above remain on disk as drill-down references for ambiguous moments.

```
[00:12:04] S0 "okay now we're going to drill the pilot holes"
[00:12:09] AUDIO drill (0.87), power_tool (0.71)
[00:12:09] VISUAL a person holding a cordless drill above a metal panel
[00:12:18] S0 "good, pass me the deburring tool"
[00:12:24] AUDIO metal_scraping (0.62)
```

> The CLAP audio lane is a **separate Phase B step** invoked after `preprocess.py` (or `preprocess_batch.py`) finishes. The first `pack_timelines.py` run produces a two-lane `merged_timeline.md` (speech + visual, interleaved by timestamp); the vocab agent reads that single file, writes a project-specific vocabulary to `audio_vocab.txt`, and then `helpers/audio_lane.py` scores against it. A second `pack_timelines.py` run folds the new audio events into the merged view for the editor. Pass `--include-audio` to the preprocessor to instead run CLAP inline against a baked-in baseline vocabulary (smoke tests, agent-less runs).

### On-demand visual drill-down

`timeline_view.py` still produces the filmstrip + waveform + word-labels PNG for any time range, called only at decision points (ambiguous pauses, retake comparisons, cut-point sanity checks). The visual lane gives the LLM enough context to know *when* to drill in.

> Naive approach: 30,000 frames × 1,500 tokens = **45M tokens of noise**.
> premiere-agent: **three ~12KB text files + a handful of PNGs**.

Same idea as browser-use giving an LLM a structured DOM instead of a screenshot — but for video.

## Pipeline

```
                  ┌─ Parakeet ONNX speech lane ─┐
Extract 16k WAV ──┤                              ├──> 2 timelines ──> agent gens vocab ──┐
                  └─ (audio lane runs Phase B) ──┘                                       │
Extract 1fps PNGs ──> Florence visual lane ──────────────────────────────────────────────┤
                                                                                         ▼
                                                            CLAP audio lane (Phase B) ──> 3rd timeline
                                                                                         │
                                                            LLM reasons ──> EDL ──> Export ──┬──> cut.fcpxml   (Resolve / FCP X)
                                                                                              ├──> cut.xml      (Premiere Pro, native xmeml)
                                                                                              └──> master.srt   (captions sidecar, always emitted)
```

`preprocess.py` runs the speech (Parakeet ONNX) and visual (Florence-2) lanes in **parallel** on a single GPU, with a VRAM-aware scheduler that drops to sequential or CPU fallback on smaller cards. The 16 kHz mono WAV is extracted **once** per source and reused by every audio-consuming step. The CLAP audio lane runs as a **separate, agent-driven step** afterwards so its vocabulary can match the actual project content; pass `--include-audio` if you'd rather run all three lanes inline against the baseline vocab.

The self-eval loop runs `timeline_view` against the source clips at every EDL cut boundary before handing the XML over — catches visual discontinuities, mid-word cuts, and overlay duration mismatches. You get the XML only after it passes.

## Get started

### Install

```bash
git clone https://github.com/<you>/premiere-agent
cd premiere-agent

# Linux / macOS
./install.sh

# Windows
.\install.bat
```

The install script auto-picks a sensible PyTorch wheel index per platform — CUDA 12.1 on Linux/Windows x86_64, the MPS-enabled universal wheel on Apple Silicon, and CPU on Intel Mac / Linux aarch64. Override with `TORCH_INDEX` if you want something else:

```bash
TORCH_INDEX=https://download.pytorch.org/whl/cu128   ./install.sh   # RTX 50-series (Blackwell)
TORCH_INDEX=https://download.pytorch.org/whl/cpu     ./install.sh   # headless / no GPU
TORCH_INDEX=https://download.pytorch.org/whl/rocm6.0 ./install.sh   # AMD ROCm
```

On macOS the speech lane runs ONNX Runtime against the **CoreML** EP (Neural Engine + Metal) and the Florence-2 / CLAP lanes use **MPS** where supported; there's no CUDA on Mac (Apple dropped NVIDIA in 2018) and `pip install onnxruntime-gpu` would 404, so the package's `pyproject.toml` markers install plain `onnxruntime` on Darwin instead. Win/Linux pulls `onnxruntime-gpu` (CUDA + TensorRT + DirectML + CPU EPs); the provider ladder in `helpers/_onnx_providers.py` falls through to the CPU EP gracefully on hosts without a GPU.

You'll also need `ffmpeg` on PATH. The script warns if it's missing with the right install command for your OS (`brew`, `apt`, `dnf`, `pacman`, `winget`).

### The speech lane in detail (ONNX Parakeet, multi-session pool)

Speech is the most expensive lane and the one most editors want fastest, so it gets the heaviest engineering. The default path runs **NVIDIA Parakeet TDT 0.6B in ONNX Runtime** with a pool of N independent inference sessions executing N clips in parallel. The NeMo Parakeet runtime stays wired in as a fallback for hosts where ONNX Runtime can't load a working execution provider — but it pulls a multi-gigabyte CUDA/PyTorch graph per call so the ONNX path is the day-to-day default. We deliberately do **not** ship a Hugging Face Whisper backend: the encoder-decoder Whisper pipeline has a [known word-timestamp memory regression](https://github.com/huggingface/transformers/issues/27834) that pinned a 5090 at 32 GB on a single 4-minute clip, and Whisper loves to hallucinate text on silence — both dealbreakers for an editor that grinds through hours of footage.

**Why ONNX, why a pool:**

ONNX Runtime ships native C++ bindings for several execution providers (TensorRT, CUDA, DirectML, CPU) and releases the GIL during `Run()`. That means N Python threads each holding their own `InferenceSession` = N truly parallel native inferences on the same GPU, sharing a single set of model weights mapped into device memory once per session. One CUDA stream per worker, the device's hardware scheduler overlaps them as long as we have unused SMs and unused VRAM bandwidth — Parakeet only uses ~30% of a 5090's SMs per inference, so we get near-linear scaling up to N=8. Throughput scales near-linearly until you hit the GPU's compute ceiling, and there's no cold-start cost between clips because the sessions stay warm.

**Provider ladder** (each tier only attempts the next on init failure):

```
TensorRT EP (gated by VIDEO_USE_PARAKEET_TRT=1)  ──► ~320× RTFx ceiling, fp16, engine-cached on disk
        │ failure / not enabled
        ▼
CUDA EP                                         ──► ~57-100× RTFx, default on NVIDIA
        │ failure / no CUDA
        ▼
DirectML EP                                     ──► ~30-50× RTFx, Windows AMD/Intel/NVIDIA
        │ failure / not Windows
        ▼
CPU EP                                          ──► ~17-30× RTFx, always works
```

**Language auto-routing.** Parakeet has two model variants: v2 is English-only and the fastest ASR model on Hugging Face's [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard); v3 is multilingual (English + 24 European languages) at a small speed cost. The lane picks automatically:

| Requested language | Loaded model |
|---|---|
| `None` or `en` | `nvidia/parakeet-tdt-0.6b-v2` (~320× RTFx) |
| anything else | `nvidia/parakeet-tdt-0.6b-v3` (~200× RTFx) |

Both pools coexist in memory if a session needs both — Parakeet TDT 0.6B is ~1.2 GB resident per session in fp16, ~600 MB at INT8, so even pinning both pools on a 12 GB card is fine.

**Knobs (env vars):**

```bash
# Force a specific lane (escape hatch — primary stays the ONNX one)
VIDEO_USE_SPEECH_LANE=onnx       # default — Parakeet TDT (ONNX), N parallel sessions
VIDEO_USE_SPEECH_LANE=nemo       # NeMo Parakeet TDT, single in-process model (fallback)

# Opt into the TensorRT execution provider. Off by default because the
# first run compiles a per-shape engine (~30-60s) before it pays off; the
# engines are cached under .ort_cache/ so subsequent runs are instant.
VIDEO_USE_PARAKEET_TRT=1

# Override the auto-sized pool size (default: free_vram_gb // 4, capped at 8).
VIDEO_USE_PARAKEET_POOL_SIZE=4

# Quantization — fp16 is the default and recommended on GPU. INT8 cuts
# VRAM in half at a small WER cost (~0.1 abs. WER on LibriSpeech).
VIDEO_USE_PARAKEET_QUANT=fp16    # or 'int8'

# Air-gapped? Same escape hatch as the NeMo lane, points at a local
# directory containing the ONNX files (encoder/decoder/joint + tokenizer).
PARAKEET_ONNX_DIR=/path/to/parakeet-onnx
```

**Quick verification.** The lane has a built-in smoke test that loads one model with the resolved provider ladder, transcribes a single WAV, and prints the chosen execution provider, the achieved RTFx, the word count, and the full JSON. Use it to sanity-check the install before running the full preprocess pipeline:

```bash
python helpers/parakeet_onnx_lane.py --smoke-test path/to/clip.wav
```

**Diarization** is unchanged — it operates on the canonical word list, not the model that produced it. Pass `--diarize` to `preprocess.py` exactly as before.

### Optional: Flash Attention 2 (Florence-2 speedup)

Florence-2 (the visual lane) transparently uses Flash Attention 2 when it's importable, falling back to PyTorch SDPA otherwise. Pulling FA2 isn't done by the install scripts because the wheel build is fiddly on Windows (needs MSVC + CUDA toolkit). If you actually need it:

```bash
pip install -e ".[flash]"
```

### Behind a corporate proxy? (NVIDIA / restricted networks)

The ONNX Parakeet weights are hosted on Hugging Face the same way the NeMo `.nemo` archive is. If your network blocks `huggingface.co` outright, pre-download the ONNX directory on a machine with access (one-time, ~1.5 GB) and point the lane at it via `PARAKEET_ONNX_DIR`. The session loader skips all network calls when that env var is set. If ONNX Runtime itself can't load on the host, set `VIDEO_USE_SPEECH_LANE=nemo` and the orchestrator runs the NeMo Parakeet path instead — you never end up without a transcript on a machine that can run *anything* locally.

The output JSON shape is byte-for-byte identical across both speech backends, so the rest of the pipeline (`pack_timelines.py`, `export_fcpxml.py`, `build_srt.py`) is genuinely lane-agnostic.

### Install for your agent runtime

The repo ships with both **`SKILL.md`** (Claude Code's skill format —
YAML frontmatter + body) and **`AGENTS.md`** (Codex's project-guidance
format — body only). Body content is identical; whichever runtime
loaded the skill will only read its own entry-point file. Pick the
section below that matches your runtime — they're not exclusive, you
can install for all three on the same machine.

#### Claude Code (local CLI / desktop)

Claude Code auto-discovers skills under `~/.claude/skills/`. Symlink
the repo there (don't copy — `git pull` upstream still works through
the link).

**Linux / macOS:**

```bash
mkdir -p ~/.claude/skills
ln -s "$(pwd)" ~/.claude/skills/video-use-premiere
```

**Windows (PowerShell, run as Administrator** — or enable Developer
Mode in Settings → System → For developers, which lets non-admin users
create symlinks):

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.claude\skills" | Out-Null
New-Item -ItemType SymbolicLink `
  -Path   "$env:USERPROFILE\.claude\skills\video-use-premiere" `
  -Target "C:\path\to\video-use-premiere"
```

Verify the link landed (`d----l` in the `Mode` column means symlink):

```powershell
Get-Item "$env:USERPROFILE\.claude\skills\video-use-premiere" |
  Select-Object FullName, Target, LinkType
```

#### Codex (local CLI / IDE / app)

Codex discovers skills under `~/.agents/skills/` (note: `.agents`,
not `.claude`). Same symlink pattern:

**Linux / macOS:**

```bash
mkdir -p ~/.agents/skills
ln -s "$(pwd)" ~/.agents/skills/video-use-premiere
```

**Windows (PowerShell, Administrator or Developer Mode):**

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.agents\skills" | Out-Null
New-Item -ItemType SymbolicLink `
  -Path   "$env:USERPROFILE\.agents\skills\video-use-premiere" `
  -Target "C:\path\to\video-use-premiere"
```

Codex will pick up `AGENTS.md` via the same symlink — `name` and
`description` come from the surrounding skill folder metadata; the
file body is what gets concatenated into the prompt when you invoke
the skill (explicitly via `$video-use-premiere` or implicitly when
your task matches the description).

> **Heads-up: Codex symlinks must be valid relative paths.** If the
> link target ends up broken (e.g. you moved the repo), Codex
> silently skips the skill. Re-run the `ln -s` / `New-Item` command
> with the new absolute path. Live-update on file changes inside the
> link target works on real directories; symlinked roots only refresh
> on Codex restart ([upstream issue](https://github.com/openai/codex/issues/11314)).

#### Claude Code on the web (cloud workspaces)

[Claude Code on the web](https://code.claude.com/docs/en/claude-code-on-the-web)
runs sessions on Anthropic-managed Ubuntu 24.04 sandboxes — it clones
the git repo you point it at, runs your environment setup script,
applies your network access tier, then launches the agent. Three
things have to be wired up for this skill to work in that environment.

**1. Mount the skill into your videos repo.** Cloud sessions only see
files that are part of the cloned repo — there's no `~/.claude/` to
symlink into. Either submodule (recommended, pinned version) or a
SessionStart hook that clones on every boot:

```bash
# inside your videos repo
git submodule add https://github.com/<you>/video-use-premiere.git \
  .claude/skills/video-use-premiere
git commit -am "Add video-use-premiere skill"
git push
```

Now every cloud session has the skill mounted at
`.claude/skills/video-use-premiere/` — Claude Code picks it up
automatically (the [docs page on cloud sessions](https://code.claude.com/docs/en/claude-code-on-the-web#what-claude-can-and-cant-access-in-cloud-sessions)
lists `.claude/skills/` as one of the directories included in the
clone).

**2. Configure the environment's setup script.** The cloud image
ships Python / Node / Go / Rust / Docker / Postgres / Redis but
[NOT `ffmpeg`](https://code.claude.com/docs/en/claude-code-on-the-web#installed-tools),
and it doesn't pre-install the skill's Python deps. Both are handled
by `scripts/cloud_setup.sh` in this repo. In the Claude Code on the
web environment settings (Settings → Environments → your environment
→ Setup script), paste:

```bash
#!/bin/bash
set -e
bash .claude/skills/video-use-premiere/scripts/cloud_setup.sh
```

The setup script runs as root, installs `ffmpeg` via `apt`, pulls the
CPU-only PyTorch wheels (no GPU in the cloud — never download the
2.5 GB CUDA build for nothing), runs `pip install -e .` (base deps now
include the full preprocess stack + OpenTimelineIO FCPXML / xmeml
adapters), fetches the spaCy English model, and pre-warms `health.py`'s
result cache. It's idempotent and the result is cached by the env, so
subsequent sessions skip almost all of it. Total cold-cache wall time
is roughly 2–3 minutes; warm-cache <5 seconds.

**3. Pick a network access tier that lets Hugging Face through.** The
default `Trusted` tier covers `pypi.org`, `apt`, `npm`, `crates.io`
and a handful of other registries — enough for `cloud_setup.sh` to
finish. **First-time runs** also need to fetch model weights from
`huggingface.co` (Parakeet TDT ~1.2 GB, Florence-2 ~470 MB, CLAP
~150 MB). If your environment is set to `Trusted` and HF works for
you, you're done; if it's set to `None`, downloads will hang. When in
doubt, set the environment to the broader internet-access tier just
for the first session, then drop it back to `Trusted` once weights
are cached on disk.

> **One unavoidable limitation.** Cloud sandboxes have **no GPU** —
> not pictured anywhere on the [installed-tools list](https://code.claude.com/docs/en/claude-code-on-the-web#installed-tools)
> (no CUDA, no nvidia-smi, no GPU drivers), and there's an open
> upstream feature request — [anthropics/claude-code#13108](https://github.com/anthropics/claude-code/issues/13108) —
> noting Claude Code's bubblewrap sandbox specifically doesn't pass
> through `/dev/dri`, `/dev/kfd`, or `/dev/nvidia*` device nodes, with
> a community workaround that has to wrap `bwrap` itself. So: the
> Parakeet / Florence-2 / CLAP lanes fall through to the CPU EP in
> cloud sessions — usable for short clips, painfully slow on
> hour-long footage. If you want the cloud agent for
> *strategy + EDL* but the heavy preprocess running on your local
> GPU, run `helpers/preprocess_batch.py` locally first and commit
> `edit/transcripts/` + `edit/visual_caps/` + `edit/audio_tags/`
> (the cached lane outputs) into the videos repo. The skill is fully
> resumable from those caches; the cloud agent then only does
> conversation, strategy, EDL, and FCPXML export — all CPU-cheap.

#### Codex Cloud (cloud environments)

[Codex Cloud](https://developers.openai.com/codex/cloud/environments)
follows the same pattern: clone repo, run setup script, apply network
policy, launch agent. The `codex-universal` base image ships Python,
Node, Rust, Go, Swift, Ruby, PHP, Java, bun, bazel, erlang/elixir —
[but again no `ffmpeg`](https://github.com/openai/codex-universal),
and the skill's Python deps aren't installed.

**1. Mount the skill** at `.agents/skills/video-use-premiere/` in your
videos repo — Codex CLI scans `.agents/skills/` from cwd up to the
repo root for skills:

```bash
git submodule add https://github.com/<you>/video-use-premiere.git \
  .agents/skills/video-use-premiere
```

**2. Configure the environment's setup script** (Codex Settings →
Environments → your env → Setup script). Same script, different path:

```bash
#!/bin/bash
set -e
bash .agents/skills/video-use-premiere/scripts/cloud_setup.sh
```

Codex caches the resulting container after the setup script finishes,
so subsequent tasks skip the install entirely. Use the optional
**maintenance script** field (runs on cached-container resume) for
cheap re-validation if you want, but it's not required.

**3. Network policy.** Codex setup scripts run with internet access
by default, so `apt install ffmpeg` and `pip install` work
out-of-the-box. **Agent-phase internet is OFF by default, though** —
that's fine for a fully local edit session (everything's on disk
after setup), but if a session needs to fetch model weights for the
first time you'll have to flip [agent internet access](https://developers.openai.com/codex/cloud/agent-internet-access)
on for that session, or pre-warm the HF cache by running
`python tests.py --heavy` once in your setup script (downloads ~2 GB,
caches under `~/.cache/huggingface/`).

**4. Permissions.** Codex's default sandbox mode is `workspace-write`,
which is what the skill expects (it writes to `<videos_dir>/edit/`).
If you've globally locked it down to `read-only`, you'll have to flip
it back for video sessions — the skill produces output files on disk,
that's the whole point.

**5. No GPU here either.** [`codex-universal`](https://github.com/openai/codex-universal)
is a CPU-only Linux container — same caveat as Claude Code on the web,
same workaround: pre-run `helpers/preprocess_batch.py` locally on your
GPU and commit the cached lane outputs into the videos repo so the
cloud agent only does the CPU-cheap conversation + EDL + FCPXML
export pass.

> Codex's sub-agent primitive lives in `~/.codex/agents/` (or
> `.codex/agents/` checked into the repo). The skill's parent rules
> assume parallel sub-agent dispatch — without it, the editor /
> vocab / animation sub-agents run inline and you pay their token
> cost in the parent's context window. See
> [the Codex subagents docs](https://developers.openai.com/codex/concepts/customization/)
> if you want to wire up the parallel primitive — it's optional, the
> skill degrades gracefully without it.

### Which folder do I open with the agent?

**Not the skill repo** — open the folder that contains your raw video
clips. The skill is global once installed (or submoduled into the
videos repo for cloud sessions), and it writes all output (`edit/`,
`transcripts/`, `cut.fcpxml`, `cut.xml`, …) into the current working
directory:

```powershell
cd C:\Footage\my-launch-video    # folder containing your .mp4 / .mov sources
claude        # OR: codex
```

### Optional: speaker diarization

If your footage has multiple speakers and you want `S0`/`S1`/etc tags in the speech timeline, add an HF token to `.env`:

```
HF_TOKEN=hf_...
```

Then preprocess with `--diarize`. Without the flag (or without a token) you still get word timestamps — just no speaker IDs. The phrase grouper handles missing speakers gracefully.

### Verify the install

```bash
python tests.py            # ~3s structural smoke test
python tests.py --heavy    # ~2 GB downloads on first run, exercises real models on a 2s synthetic clip
```

#### Watch heavy-mode progress live

Heavy mode loads three real models (Parakeet ONNX, Florence-2, CLAP).
First run downloads ~2 GB and looks quiet for 30–60 s per model. To see
live progress (especially when running under Claude Code, Codex, or
any non-TTY shell that buffers stdout), tee the output to a log file
and tail it from a second window:

```powershell
# window 1 — run the tests, also writing to run.log
python -u tests.py --heavy --log run.log

# window 2 — follow it live
Get-Content run.log -Wait        # PowerShell
tail -f run.log                  # bash / zsh
```

The `-u` flag forces unbuffered stdio in the parent process and the `--log`
flag installs a line-buffered tee inside `tests.py`, so every `[xx.xs]`
status line and every Hugging Face download bar shows up in the log the
moment it's emitted.

The skill itself runs `python helpers/health.py --json` on every session start — it's cached for 7 days and auto-invalidates when `torch` / `transformers` / `opentimelineio` versions change, so subsequent sessions return in <500 ms unless something actually changed. If anything fails, the agent surfaces concrete fix steps from the cache rather than re-running.

### Run

```bash
cd /path/to/your/videos
claude          # Claude Code (local CLI)
# OR
codex           # Codex (local CLI)
# OR — for cloud workspaces, push the videos repo and start a session at claude.ai/code
```

In the session:

> edit these into a launch video

It inventories the sources, runs Phase A (speech + visual) preprocess, generates a project-specific audio vocabulary and runs Phase B (CLAP audio events), proposes a strategy, waits for your OK, then produces `edit/cut.fcpxml` + `edit/cut.xml` + `edit/master.srt` (captions sidecar, always emitted) next to your sources. All outputs live in `<videos_dir>/edit/` — the skill directory stays clean. Open the XML in your NLE; that's where the cut lives.

## Hardware tiers

The preprocessor checks free VRAM on startup and picks a schedule:

| Free VRAM | Schedule | 3 hr footage wall time |
|---|---|---|
| ≥ 8 GB | Speech ‖ visual (Phase A), then CLAP audio (Phase B, separate) | ~10–15 min on RTX 3060+ |
| 4–8 GB | Speech ‖ visual still parallel; smaller pool sizes for each | ~15–25 min |
| 2–4 GB | Sequential, smaller batches | ~30–45 min |
| no CUDA / < 2 GB | CPU fallback for speech / CLAP, `--skip-visual` recommended | hours |

(With `--include-audio`, the audio lane runs inline alongside speech + visual against the baseline vocab — same scheduler decides whether to parallelize.)

Override detection with `--force-schedule {parallel|sequential|cpu}`.

### Wealthy mode (4090 / 5090 — pure speed, same outputs)

If you've got a 24 GB+ card, opt into bigger batch sizes that just throw more parallelism at the GPU. **Same model, identical outputs, just faster.** No quality changes — beam counts, sampling, models, and prompts stay exactly as-is.

```bash
# CLI flag — applies to a single invocation
python helpers/preprocess_batch.py /path/to/videos --wealthy

# OR set the env var once and every lane (and subprocess) inherits it
export VIDEO_USE_WEALTHY=1   # bash / zsh
$env:VIDEO_USE_WEALTHY=1     # PowerShell
set VIDEO_USE_WEALTHY=1      # cmd

# OR just tell Claude in the session: "I'm wealthy" — the skill sets the env var for you
```

What changes under the hood:

| Lane | Default batch | Wealthy batch | Why it works |
|---|---|---|---|
| Parakeet (ONNX) parallel sessions | 4 | 8 | Each session is its own CUDA stream; ORT releases the GIL during `Run()` so N threads = N parallel GPU inferences. 8 sessions × ~1.2 GB fp16 = ~9.6 GB resident on a 5090. |
| Parakeet (NeMo, fallback) | 8 | 32 | Only when `VIDEO_USE_SPEECH_LANE=nemo`. Same RNNT decoder, bigger torch batches; greedy decoding stays — same logits. |
| Florence-2 captions | 8 | 32 | num_beams stays at 3 — same captions, more frames per forward pass |
| CLAP audio lane | 16 windows/batch + base tier | 64 windows/batch + large tier | Per-batch overhead dominates over the audio encoder forward; switching to `larger_clap_general` sharpens fine-grain discrimination on environmental sounds |

## Output: NLE handoff

Once the LLM produces `edl.json`, export it to your NLE via [OpenTimelineIO](https://opentimeline.io/):

```bash
python helpers/export_fcpxml.py edit/edl.json -o edit/cut.fcpxml --frame-rate 24
```

This emits **three** files from a single timeline build:

- `edit/cut.fcpxml` — FCPXML 1.10+, native to **DaVinci Resolve** and **Final Cut Pro X** (`File → Import → XML`).
- `edit/cut.xml`    — Final Cut Pro 7 xmeml, native to **Adobe Premiere Pro** (`File → Import`). No XtoCC, no extra tooling — Premiere reads this dialect directly.
- `edit/master.srt` — captions sidecar (UTF-8, CRLF, sequential cues). Premiere / Resolve / FCP X all import this onto a captions track via `File → Import`.

This is deliberate: Premiere does **not** import `.fcpxml` natively (Adobe's docs route you through the third-party [XtoCC](https://www.intelligentassistance.com/xtocc.html) translator), but it does read FCP7 xmeml out of the box. Emitting both XML dialects lets the recipient pick whichever NLE they live in. Override with `--targets {both,fcpxml,premiere}` if you only want one. Pass `--no-srt` to skip the captions sidecar.

Either XML file lands clips on V1/A1 with split offsets honoring `audio_lead` (J cut), `video_tail` (L cut), and `transition_in` (cross-dissolve) per range. Ranges with `speed > 1.0` are emitted as native NLE retime: FCPXML gets a `<timeMap>` element, Premiere xmeml gets a `Time Remap` filter, and the audio is silenced or pitch-corrected per `audio_strategy`. The exporter snaps every cut to whole frames at the timeline rate so there's no audio/video drift on import.

The captions are built straight from the cached Parakeet transcripts on the OUTPUT timeline (Hard Rule 5) — 2-word UPPERCASE chunks by default, fully customizable in the NLE's caption panel. Need to regenerate just the SRT after a hand-tweak to the EDL?

```bash
python helpers/build_srt.py edit/edl.json
```

Color is the colorist's job; the skill never emits a grade.

**Why XML-only?** The cut lives in the NLE. There is no flat-MP4 renderer in this skill — your editor / colorist owns the final pixels, the AI owns the cut decisions.

## Project layout

```
<videos_dir>/
├── <source files, untouched>
└── edit/
    ├── project.md               ← memory; appended every session
    ├── merged_timeline.md       ← DEFAULT reading surface (all 3 lanes
    │                              interleaved chronologically by timestamp)
    ├── speech_timeline.md       ← lane 1, drill-down
    ├── audio_timeline.md        ← lane 2, drill-down
    ├── visual_timeline.md       ← lane 3, drill-down
    ├── edl.json                 ← cut decisions
    ├── transcripts/<name>.json  ← raw Parakeet ONNX words (cached)
    ├── audio_tags/<name>.json   ← raw CLAP zero-shot events (cached)
    ├── visual_caps/<name>.json  ← raw Florence-2 output (cached)
    ├── audio_vocab.txt          ← agent-curated CLAP vocabulary (Phase B)
    ├── audio_vocab_embeds.npz   ← cached CLAP text embeddings for that vocab
    ├── audio_16k/<name>.wav     ← shared 16kHz mono PCM (cached)
    ├── animations/slot_<id>/    ← per-animation source + render + reasoning
    ├── master.srt               ← output-timeline subtitles (build_srt.py)
    ├── verify/                  ← debug frames / timeline PNGs
    ├── cut.fcpxml               ← NLE export, FCPXML 1.10+ (Resolve / FCP X)
    └── cut.xml                  ← NLE export, FCP7 xmeml (Premiere Pro)
```

## Design principles

1. **Text + on-demand visuals.** No frame-dumping. The three timelines are the surface.
2. **Speech is primary, visuals are secondary, audio events are tertiary.** Cuts come from speech word boundaries and silence gaps; the visual + audio-event lanes inform context, not cut points directly. When the visual lane and the audio-event lane disagree about what's on screen, trust the visual lane.
3. **Local-first.** No API calls during pre-processing. Your footage doesn't leave the machine.
4. **Ask → confirm → execute → self-eval → persist.** Never touch the cut without strategy approval.
5. **Zero assumptions about content type.** Look, ask, then edit.
6. **12 hard rules, artistic freedom elsewhere.** Production-correctness is non-negotiable. Taste isn't.

See [`SKILL.md`](./SKILL.md) for the full production rules and editing craft.

## Credits

- Speech (primary): NVIDIA [Parakeet TDT 0.6B v2 (English)](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) and [v3 (multilingual)](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) running on [ONNX Runtime](https://onnxruntime.ai/) via [`onnx-asr`](https://github.com/istupakov/onnx-asr) (bundles the NeMo-compatible mel preprocessor + TDT decoder + Silero VAD). Multi-session pool with one CUDA stream per worker thread; TensorRT EP gated by `VIDEO_USE_PARAKEET_TRT=1`.
- Speech (fallback): NeMo Parakeet via `parakeet_lane.py` for hosts where ONNX Runtime can't load a working execution provider
- Audio events: [LAION CLAP](https://github.com/LAION-AI/CLAP) (HTSAT-unfused / larger_clap_general) via [Xenova's quantized ONNX exports](https://huggingface.co/Xenova/clap-htsat-unfused) on ONNX Runtime, scored zero-shot against an agent-curated vocabulary
- Visual captions: [Florence-2](https://huggingface.co/microsoft/Florence-2-base) (Microsoft Research License — non-commercial use only)
- NLE interchange: [OpenTimelineIO](https://opentimeline.io/) + `otio-fcpx-xml-adapter`
