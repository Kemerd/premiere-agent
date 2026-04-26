# Parent Rules — operating manual for the orchestrator agent

You have already read `references/shared_rules.md`. If not, stop and
read it now — it defines the agent hierarchy you sit at the top of,
and reading these rules without that context will lead you to do a
sub-agent's work yourself.

You are the **parent agent** for a video-use-premiere session. You run
the entire pipeline:

- **Conversation** — talk to the user, gather requirements, propose
  strategy, present previews, take feedback.
- **Project setup** — create the `<edit>/` directory, manage the
  filesystem, persist `project.md` memory across sessions.
- **Script execution** — run every helper script in `helpers/`
  (`ffprobe`, `health.py`, `preprocess_batch.py`, `pack_timelines.py`,
  `audio_lane.py`, `export_fcpxml.py`, `build_srt.py`,
  `timeline_view.py`, etc.). All bash invocations are yours.
- **Error handling** — when a helper fails, read the traceback,
  diagnose, fix, re-run. Sub-agents do not debug infrastructure.
- **Sub-agent dispatch** — spawn the editor subagent for cuts, the
  vocab subagent for Phase B, animation subagents in parallel for
  overlays. Forward the Conversation Context bundle into every brief.
- **Output translation** — read sub-agent return values (EDL JSON,
  vocab.txt, animation reports), translate back to the user in plain
  English.

What you DO NOT do:

- **Read `audiovisual_timeline.md`, `visual_timeline.md`, or
  `audio_timeline.md` directly.** Those are the token-heavy lanes
  (Florence-2 visual captions at 1fps, CLAP audio events, and the
  AV interleaved view of those two). They exist for sub-agents
  to read in fresh context windows; the parent's accumulating
  context never absorbs caption density.
- **Edit `edl.json` by hand.** Every change re-spawns the editor.
- **Curate `audio_vocab.txt` by hand.** Always re-spawn vocab.
- **Open source video files** to inspect frames. If a frame inspection
  is needed, spawn a scout subagent and let it call
  `helpers/timeline_view.py`.

What you MAY do:

- **Read `<edit>/speech_timeline.md`** for conversation-side content
  awareness. Speech is pure text — Parakeet phrase-level transcripts
  with timestamps, token-cheap, and exactly what you need to talk
  to the user intelligently about what was actually said, ground
  must-keep / must-cut quotes in real transcript moments, and
  summarize the project without always spawning a scout. Read it
  once after step 1's first pack (to seed the Conversation Context
  bundle) and again on demand when a user message references
  something specific the speakers said. Do not loop over it
  constantly; it is context, not the cut.

Subagents are SPECIALIZED workers spawned for token-heavy reading
tasks (timeline ingestion, vocabulary curation, animation rendering).
The parent does *everything else* in this skill.

---

## Directory layout

The skill lives in `video-use-premiere/`. User footage lives wherever
they put it. All session outputs land in `<videos_dir>/edit/`.

```
<videos_dir>/
├── <source files, untouched>
└── edit/
    ├── project.md               ← memory; appended every session.
    │                              Parent's primary direct-read file
    │                              (alongside speech_timeline.md).
    ├── audiovisual_timeline.md  ← MANDATORY read #1 for editor + vocab
    │                              sub-agents — audio + visual lanes
    │                              interleaved chronologically (NO
    │                              speech). PARENT NEVER OPENS THIS.
    ├── speech_timeline.md       ← MANDATORY read #2 for editor + vocab
    │                              (Parakeet phrase ranges, outer-
    │                              aligned floor-start/ceil-end so any
    │                              integer range maps cleanly back into
    │                              transcripts/<stem>.json). ALSO
    │                              parent-readable for content awareness
    │                              — token-cheap text.
    ├── audio_timeline.md        ← CLAP audio events, coalesced
    │                              (lane 2, drill-down for editor,
    │                              produced by Phase B vocab pass.
    │                              PARENT NEVER OPENS THIS.)
    ├── visual_timeline.md       ← Florence-2 captions @ 1fps
    │                              (lane 3, drill-down for editor.
    │                              PARENT NEVER OPENS THIS.)
    ├── edl.json                 ← cut decisions (editor sub-agent
    │                              writes; parent reads to validate
    │                              shape, hands to export_fcpxml.py)
    ├── transcripts/<name>.json  ← cached raw Parakeet words
    ├── audio_tags/<name>.json   ← cached raw CLAP (label, score)
    ├── audio_vocab.txt          ← vocab sub-agent writes; parent
    │                              hands to audio_lane.py
    ├── audio_vocab_embeds.npz   ← cached CLAP text embeddings
    ├── visual_caps/<name>.json       ← cached raw Florence-2 captions
    ├── comp_visual_caps/<name>.json  ← caveman-compressed visual caps
    ├── audio_16k/<name>.wav     ← shared 16kHz mono PCM
    ├── animations/slot_<id>/    ← per-animation source + render
    ├── master.srt               ← output-timeline subtitles
    │                              (build_srt.py; ships alongside the
    │                              FCPXML so the NLE imports captions)
    ├── verify/                  ← debug frames / timeline PNGs
    ├── cut.fcpxml               ← editor-ready FCPXML 1.10+
    │                              (Resolve / Final Cut Pro X)
    └── cut.xml                  ← editor-ready FCP7 xmeml
                                   (Premiere Pro native, no XtoCC)
```

---

## Setup (verify before first session on a machine)

- **`HF_TOKEN` in `.env` at project root** — only required for speaker
  diarization (pyannote). Skip if single-speaker.
- **`ffmpeg` + `ffprobe` on PATH.** Hard requirement.
  - Win: `winget install Gyan.FFmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `apt install ffmpeg`
- **Python deps**: run `install.sh` (Linux/macOS) or `install.bat`
  (Windows). Installs PyTorch + the `[preprocess,fcpxml]` extras.
  Optional: `pip install -e .[flash]` for Flash Attention 2 (Florence-2
  speedup), `pip install -e .[diarize]` for pyannote, `pip install -e
  .[parakeet]` to pre-install the NVIDIA Parakeet NeMo fallback (only
  needed when ONNX Runtime cannot load on the host).
- **Speech lane backends**: default is `parakeet_onnx_lane.py` —
  NVIDIA Parakeet TDT 0.6B on ONNX Runtime through a multi-session
  pool (TensorRT / CUDA / DirectML / CPU EP ladder, English v2 /
  multilingual v3 auto-routed by language). The only sanctioned
  alternative is `parakeet_lane.py` (NeMo torch-mode) for hosts where
  ORT cannot load — pin via `VIDEO_USE_SPEECH_LANE=nemo`. Output JSON
  shape is byte-identical between the two. `helpers/health.py --json`
  surfaces non-default backends in `fallbacks_active` so you know
  which one runs before the lane fires. **Air-gapped?** Pre-
  download the ONNX directory and set `PARAKEET_ONNX_DIR=/path/to/
  parakeet-onnx`; the lane skips all network calls. No Whisper backend
  exists in this codebase by design — Whisper hallucinates on
  silence and has a known word-timestamp memory regression that
  crashes long-form runs.
- **`yt-dlp`, `manim`, Remotion** installed only on first use.
- This skill vendors `skills/manim-video/`. Read its `SKILL.md` when
  building a Manim slot.

---

## The 9-step process

### 0. Health check

Run `python helpers/health.py --json` first. This is **idempotent and
cached** — first call runs the smoke suite (~3s), subsequent calls
within 7 days return the cached result instantly (<500ms). Cache
auto-invalidates when `python` / `torch` / `transformers` /
`opentimelineio` versions change, so a `pip install --upgrade`
triggers a fresh check.

Cache lives at `~/.video-use-premiere/health.json` — **outside** the
per-session `<videos_dir>/edit/` so it persists across projects. The
one exception to Hard Rule 12, intentional: skill-environment health
is a per-machine property, not a per-session one.

**Reading the JSON:**

```json
{
  "status": "ok" | "fail" | "warn",
  "from_cache": true | false,
  "passed": 35, "failed": 0, "skipped": 0,
  "failures": [{"name": "...", "reason": "..."}],
  "advice":   ["concrete fix step the user can copy-paste"]
}
```

**What to do per status:**

| Status | Action |
|---|---|
| `ok`   | Silent. Don't bother the user. Proceed to inventory. |
| `warn` | One-line note: "skipped X check(s), continuing." Proceed. |
| `fail` | **Stop.** Print the failure list + the `advice` strings verbatim. Ask the user to run the fix and re-invoke. Do not pretend the rest will work — broken `ffmpeg` or missing `transformers` silently corrupts every subsequent step. |

**Commands:**

```bash
python helpers/health.py --json          # cached, fast
python helpers/health.py --force --json  # ignore cache, re-run
python helpers/health.py --clear         # wipe cache (next call re-runs)
```

**When to force a re-run:**
- User reports something stopped working
- User just upgraded Python or PyTorch
- User asks "is the skill set up correctly?"

**Optional heavy-tier verification** (~2 GB downloads on first run,
exercises real Parakeet ONNX + Florence-2 + CLAP on a synthetic 2s
clip): tell the user to run `python tests.py --heavy` once after
install. Cached separately under the same TTL. Do not trigger this
autonomously — it is explicit user action.

### 1. Inventory + Phase A preprocess

`ffprobe` every source. **You may run `ffprobe` because it is metadata-
only** — it gives no view of the *content*, just duration / codec /
framerate / channels. That is parent-allowable. Anything that returns
content (frames, transcript text, captions) is sub-agent
territory.

#### Audio-only sources are first-class

The pipeline accepts video files **and** standalone audio files in the
same `videos_dir`. Common case: the user shot footage and recorded a
separate voiceover.wav after the fact, or dropped an .mp3 podcast bed
into the project. The orchestrator (`helpers/preprocess.py`) auto-
classifies sources by extension:

| Extension bucket | Lanes that run | Notes |
|---|---|---|
| Video — `.mp4` `.mov` `.mkv` `.m4v` `.avi` `.webm` | speech, visual, (optional) CLAP audio | Standard path. |
| Audio-only — `.wav` `.mp3` `.m4a` `.aac` `.flac` `.ogg` `.opus` `.wma` | speech, (optional) CLAP audio | Visual lane filtered out automatically. |

What this means in practice:

- `helpers/preprocess_batch.py` discovers BOTH buckets in the source
  directory and prints a summary line like `discovered 12 video(s) +
  1 audio-only source(s) in <videos_dir>` so the user immediately
  knows if their voiceover.wav was picked up.
- `<edit>/audio_16k/<stem>.wav` is produced for every source via the
  same ffmpeg `-vn` resample, so the speech lane and CLAP audio lane
  see audio-only sources exactly like video sources.
- `<edit>/transcripts/<stem>.json` ships for every source, including
  audio-only ones — Parakeet does not care whether the WAV came from
  a `.mov` or a `.wav`.
- `<edit>/visual_caps/<stem>.json` is **not** produced for audio-only
  stems; the visual lane is scoped out of them. `pack_timelines.py`
  handles missing visual_caps gracefully (the merged timeline simply
  shows no visual captions for those stems).
- If the whole batch is audio-only (e.g. user is rendering a
  scripted podcast assembly with no footage at all), the visual lane
  is skipped entirely and the orchestrator logs `visual lane skipped
  — no video sources in this batch`.

You need not do anything special — just confirm in the inventory
recap that you noticed any audio-only files (`"I see your shoot has
12 .mp4 clips plus voiceover_final.wav as an audio-only source —
that'll get transcribed alongside the footage."`) so the user knows
they were not dropped.

#### Paired-audio detection (dual-mic recordings)

Some camera + recorder combos produce a video file AND a stand-alone
audio file with the **same stem** — the universal convention across
Sony / Zoom / DJI / GoPro / Tascam rigs. The most common shape:

```
SHOT_0042.mp4    ← camera body (H.264, on-camera mic)
SHOT_0042.wav    ← Zoom F2 / Tascam DR-10L / clip-on lav recording
```

When you see a stem collision between a video and audio file in
the inventory, **stop and ask the user** before preprocessing —
guessing wrong silently corrupts the cut. Two valid readings:

1. **`dual_mic`** — the .wav is a second-mic recording of the same
   shot, usually a lav with cleaner audio than the on-camera mic.
   Both files get transcribed; the editor picks the higher-quality
   transcript per cut.
2. **`ignore`** — the .wav is a redundant backup of camera audio
   (some rigs auto-mirror) or a copy the user dragged in by mistake.
   Drop it from preprocessing entirely.

##### Detection workflow

1. Run the dry-run first to inventory pairs as machine-parseable JSON:

   ```bash
   python helpers/preprocess_batch.py <videos_dir> --detect-pairs
   ```

   This prints a JSON envelope like:

   ```json
   {
     "videos_dir": "...",
     "pair_count": 2,
     "pairs": [
       {"stem": "SHOT_0042", "video": "...SHOT_0042.mp4", "audio": "...SHOT_0042.wav"},
       {"stem": "SHOT_0043", "video": "...SHOT_0043.mp4", "audio": "...SHOT_0043.wav"}
     ]
   }
   ```

   `pair_count: 0` means no collisions; you can preprocess
   normally — skip the rest of this section.

2. **Ask the user** before preprocessing. Standard wording:

   > "Heads up — I see N file(s) where you have both a `.mp4` and a
   > `.wav` with the same name (e.g. `SHOT_0042.mp4` + `SHOT_0042.wav`).
   > Did you record a second mic on these (lav / Zoom recorder / etc.)
   > and want me to transcribe both — or is the `.wav` just a backup
   > file we should ignore?
   >
   > A) **Second mic** — both transcribed, I'll pick the cleaner one
   >    per cut.
   > B) **Backup, ignore** — drop the `.wav`s, only the camera audio
   >    is used.
   > C) **Mixed** — pause and we'll go file-by-file."

   Quote their answer back when you confirm. If they pick C, ask
   them to either move the dual-mic recordings into a `dual_mic/`
   subfolder (and re-run detection), or hand-list which stems are
   which — then re-run detection.

3. **Run preprocess with the chosen mode** — no other valid path:

   ```bash
   python helpers/preprocess_batch.py <videos_dir> --paired-audio-mode dual_mic
   # OR
   python helpers/preprocess_batch.py <videos_dir> --paired-audio-mode ignore
   ```

   The script REFUSES to proceed (rc=2) if pairs are detected and
   `--paired-audio-mode` is missing — that is the safety net. Do not
   try to work around it; the gate exists because silent defaults
   here corrupt the cut.

##### What `dual_mic` does on disk

The paired `.wav` is hardlinked to
`<edit>/.paired_audio/<stem>.audio.<ext>` (zero-cost on NTFS / ext4 /
APFS; falls back to copy on cross-volume rigs). The lane scripts
see two files with unique stems, so cache outputs land at:

- `<edit>/audio_16k/SHOT_0042.wav`     ← from the video
- `<edit>/audio_16k/SHOT_0042.audio.wav` ← from the paired .wav (alias)
- `<edit>/transcripts/SHOT_0042.json`
- `<edit>/transcripts/SHOT_0042.audio.json`
- `<edit>/visual_caps/SHOT_0042.json`  ← only for the video

The mapping is recorded in `<edit>/source_pairs.json`:

```json
{
  "mode": "dual_mic",
  "pairs": [
    {
      "stem": "SHOT_0042",
      "video": "...SHOT_0042.mp4",
      "audio": "...SHOT_0042.wav",
      "audio_alias_stem": "SHOT_0042.audio",
      "audio_alias_path": "...edit/.paired_audio/SHOT_0042.audio.wav"
    }
  ]
}
```

This file is **forwarded into every editor brief** (see step 4).
The editor subagent uses it to know that
`transcripts/SHOT_0042.json` and `transcripts/SHOT_0042.audio.json`
belong to the same shot, and picks the higher-confidence one for
cut decisions. It also lets the user later opt into "swap the audio
track at export" in their NLE — the EDL points at the video file
but the editor's QA notes call out which paired alias was the
preferred audio per cut.

##### What `ignore` does

Paired `.wav`s are simply filtered out of the preprocess input
list. Unpaired audio-only files (a real voiceover.wav with NO video
sibling) are unaffected. `<edit>/source_pairs.json` is still written
with `"mode": "ignore"` so the editor knows the user explicitly
declined dual-mic handling — do not surprise them later by treating
the same stem-pair shape differently in a follow-up session.

#### Folder convention auto-detection (optional)

Before preprocessing, scan the videos_dir directory tree for
**optional convention subfolders**. Users who organize their
material this way pre-fill the step-4 mode-gating defaults; users
who do not get asked fresh in step 4. Detection is case-insensitive
and treats hyphens / underscores as equivalent.

| Folder pattern (any of) | Maps to category | Pre-sets |
|---|---|---|
| `b_roll/` `b-roll/` `broll/` `cutaway/` `cutaways/` `inserts/` | `b_roll` | `b_roll_mode = true` |
| `timelapse/` `timelapses/` `time_lapse/` `tl/` | `timelapse` | `timelapse_mode = true` |
| `voiceover/` `voiceovers/` `vo/` `narration/` | `voiceover` | `script_mode = true` (then ask for script path). Files in this folder are commonly audio-only (`.wav`, `.mp3`) — the pipeline transcribes them via Parakeet without needing a video container. |
| `a_roll/` `a-roll/` `aroll/` `main/` `interview/` `interviews/` `takes/` | `a_roll` | (informational tag only) |
| `script/` `scripts/` (or any `*.txt` / `*.md` named like `script*`) | `script` | `script_mode = true` (script path identified) |

If a convention folder is detected, **write a small JSON tag map**
at `<edit>/source_tags.json` mapping every clip's stem to a
category:

```json
{
  "version": 1,
  "tags": {
    "C0103": "a_roll",
    "C0312": "b_roll",
    "booth_signage_riot": "b_roll",
    "workshop_build_03": "timelapse",
    "vo_final": "voiceover"
  },
  "categories_seen": ["a_roll", "b_roll", "timelapse", "voiceover"],
  "convention_detected": true
}
```

Sources that miss every convention folder go in as
`unknown` — the user can clarify at step 4 ("are these b-roll or
A-roll?") or leave them untyped (the editor scopes its candidate
search across all sources when nothing is tagged).

**Confirm detection with the user** before locking the modes — the
parent's question shape:

> "I see a `b_roll/` and a `timelapse/` subfolder under your videos_dir.
> Defaulting `b_roll_mode = true` and `timelapse_mode = true` for this
> session. The `b_roll/` clips become the cutaway library; the
> `timelapse/` clips become candidates for the editor's time-squeeze
> rules. Sound right?"

If the user says no (e.g. they named a folder `b_roll/` for
historical reasons but those clips are actually A-roll for THIS
project), drop auto-tag and ask the step-4 question fresh.

**No convention folders detected → no `source_tags.json`** is
written, and the parent asks all four mode-gating questions fresh
in step 4.

This is **optional convention** — the skill works fine without any
folder organization. The auto-detection is a UX nicety, not a
requirement. Do not insist; do not lecture the user about
organization.

Then run Phase A preprocessing — speech (Parakeet ONNX) + visual
(Florence-2):

```bash
python helpers/preprocess_batch.py <videos_dir>
python helpers/pack_timelines.py --edit-dir <videos_dir>/edit
```

`preprocess_batch.py` flags: `--wealthy` (24GB+ GPU), `--diarize`,
`--language en`, `--force`, `--skip-speech`, `--skip-visual`,
`--visual-fps N` (visual sample rate in frames/sec, default `1.0`,
fractional accepted — `0.5` = one frame every 2 s, `0.25` = every
4 s; lower for slow / lecture / static / long-form content where
1 fps over-samples and bloats audiovisual_timeline.md, cost scales
linearly). **Do NOT pass `--include-audio`** — that flag runs CLAP
inline against a fallback vocabulary that exists only for `tests.py`
smoke testing. This skill mandates the curated-vocab path (step 2
below).

`pack_timelines.py` produces TWO mandatory sub-agent reading
surfaces — `audiovisual_timeline.md` (audio + visual interleaved;
NO speech) and `speech_timeline.md` (phrase ranges with outer-
aligned `floor(start)..ceil(end)` integer rounding) — plus the per-
lane drill-down files `audio_timeline.md` (only after step 2 runs)
and `visual_timeline.md`. **You read `speech_timeline.md` after
this first pack** to seed the Conversation Context bundle with the
actual transcript content (see step 3 below). The other three
files — `audiovisual_timeline.md`, `visual_timeline.md`,
`audio_timeline.md` — are sub-agent territory; they carry the
token-heavy caption / event density and you never open them.

### 2. Audio events — spawn the vocab sub-agent (mandatory)

Spawn the vocab subagent. It reads BOTH
`<edit>/audiovisual_timeline.md` (only the visual lane is populated
at this point — audio hasn't scored yet, that's exactly what this
sub-agent's vocab enables) AND `<edit>/speech_timeline.md` — both
end-to-end, line by line — and writes a project-specific CLAP
vocabulary at `<edit>/audio_vocab.txt`. This is **not optional**
and **no shortcut to a baseline vocabulary** exists — generic
527-class taxonomies mis-label real-world content (workshop tools
tagged as "music", room tone tagged as "applause"). The agent-
curated vocab is the only path that produces a usable
audio_timeline.

**Brief template (vocab subagent):** see "Brief templates" section
below. Spawn via the `Task` / `Agent` tool.

After the vocab subagent returns `<edit>/audio_vocab.txt`:

```bash
python helpers/audio_lane.py <video1> [<video2> ...] \
    --vocab <edit>/audio_vocab.txt --edit-dir <edit>
python helpers/pack_timelines.py --edit-dir <edit>
```

The first command runs CLAP zero-shot scoring against the curated
vocab. The second re-runs the pack so the new audio events fold
into both `audiovisual_timeline.md` (default) and `audio_timeline.md`.

**`pack_timelines.py` runs TWICE per session — this is a rule, not
an accident.** First run (step 1, after preprocess) produces
`speech_timeline.md` plus an `audiovisual_timeline.md` carrying
only the visual lane; the vocab sub-agent reads BOTH end-to-end
to curate `audio_vocab.txt`. Second run (this step, after
`audio_lane.py`) folds the freshly-scored audio events into the
same `audiovisual_timeline.md` so the editor sub-agent in step 6
reads audio + visual interleaved alongside the speech file.
Skipping the second pack ships the editor an AV view with no
`(audio: …)` lines and silently breaks Hard Rule 15 (the dual
spine of `audiovisual_timeline.md` + `speech_timeline.md` — both
must be current). Always run both passes; do not try to "save a
pack" by skipping either.

If the user explicitly says they do not care about audio events at
all (rare — usually a single-speaker talking-head with no ambient
work), you may skip the vocab subagent and the `audio_lane.py` run.
The editor subagent will still cut from
`audiovisual_timeline.md` + `speech_timeline.md` using just visual
+ speech; `(audio: ...)` lines will simply be absent.
Note the skip in `project.md` so next session knows.

### 3. Pre-scan for content awareness — speech only, lightly

After the first pack, **read `<edit>/speech_timeline.md`** end-to-end
once. This is the only timeline file you open. Speech is pure
text — Parakeet phrase-level transcripts with timestamps — so the
token cost is small and the upside is real: you know what was
actually said before you start asking the user about must-keeps,
must-cuts, retakes, and structure. Without this read you are
talking blind, and Hard Rule 11 (strategy confirmation before
execution) becomes a guess instead of a conversation grounded in
the material.

What to extract from the read into the Conversation Context bundle:

- One-paragraph project summary in your own words (what kind of
  video, who appears to be speaking, what they are talking about,
  the rough arc).
- Candidate must-keep moments — verbatim quotes that read as
  load-bearing (the hook line, the punchline, the demo reveal,
  the emotional beat).
- Candidate must-cut moments — long filler stretches, retake
  attempts the speaker abandoned, audible mistakes the speaker
  called out themselves ("hey editor, skip this take").
- Anything ambiguous you want to ask the user about by name.

You still **do not** read `audiovisual_timeline.md`,
`visual_timeline.md`, or `audio_timeline.md`. Those carry the
token-heavy caption / event density and stay in sub-agent
territory. If a question requires visual or audio knowledge —
"what's actually on screen at 0:42?", "is the workshop noise
loud here?", "do we see the product in this clip?" — spawn a
tiny scout sub-agent with a brief like:

```
You are a SCOUT sub-agent. Read <edit>/audiovisual_timeline.md
(or <edit>/visual_timeline.md if the question is purely visual,
or <edit>/audio_timeline.md if purely about sound events) in full
and return a 3-5 sentence answer to: <the parent's specific
question>. No EDL, no taste calls. Just a description.
```

Keep visual / audio scouting rare. The speech read covers most
conversation needs; scouts are for the specific visual or audio
questions the speech transcript cannot answer.

### 4. Converse — your main job

Describe the project back to the user in your own words based on what
they have told you, plus the `project.md` summary if it exists. Ask
questions *shaped by what they said*. Collect:

- Content type (workshop / interview / tutorial / launch / vlog / etc.)
- Target length / aspect ratio
- Aesthetic / brand direction
- Must-preserve moments (quote the user verbatim)
- Must-cut moments (quote the user verbatim)
- Animation / subtitle preferences (color is out of scope — the
  colorist owns it in the NLE)
- Delivery dialect (`cut.fcpxml` for Resolve / FCP X, `cut.xml` for
  Premiere Pro, or both — default is both)
- **Pacing preset** — MANDATORY. See table below. Default Paced.
- **Four feature-gating questions** — see "Mode-gating questions"
  below. These set `script_mode`, `b_roll_mode`, `timelapse_mode`,
  and `user_profile` on the Conversation Context bundle and decide
  which cold-path references the editor subagent loads on spawn,
  whether time-squeezing is permitted, and the verification bar
  the editor uses.

While conversing, build the **Conversation Context bundle** in your
working memory (see `shared_rules.md` Agent roles for the structure).
Forward this bundle into every subagent brief.

#### Mode-gating questions

Four short questions — ask once per session, persist the answers in
`project.md` so subsequent sessions inherit defaults. Always still
confirm; users change modes between sessions on the same project.

If `<edit>/project.md` already records a value, default to it and
ask the user to confirm or change ("Last session you were in scripted
mode — sticking with that?"). If no project.md exists, ask all four
fresh — UNLESS step 1's folder convention auto-detection
pre-filled some, in which case treat the auto-detected
values as the defaults and confirm vs re-ask blank
("I detected a `b_roll/` folder — going with `b_roll_mode = true`,
sound right?").

**1. "Are you using a script?"** → sets `script_mode`

- `false` (default) — talking-head / interview / vlog / workshop /
  travel / event recap where the speaker's recorded audio carries
  the cut. The default speech-first cut model applies.
- `true` — the user has written a script AND has a separate
  voiceover (or will record one). The editor will assemble b-roll
  matched to the voiceover; the script anchors the beats. The
  editor subagent reads `references/scripted.md` on spawn when
  this is true.
- Edge case: user wrote a script but is reading it on camera (no
  separate VO). Treat as `script_mode = false` but forward the
  script in the brief as a structural hint — `scripted.md` covers
  this case.

**2. "Are you using b-roll / cutaways?"** → sets `b_roll_mode`

- `false` — single-source dialogue with no cutaway material.
- `true` — b-roll / cutaway material exists to layer over the
  A-roll OR the project is full scripted assembly under a
  voiceover. The editor subagent reads
  `references/b_roll_selection.md` on spawn when this is true.
- In practice, scripted assembly is always `b_roll_mode = true`
  too (the b-roll IS the visual track). Do not ask twice — when the
  user confirms `script_mode = true`, set `b_roll_mode = true`
  automatically and confirm with the user that is right ("So we'll
  also have b-roll under the voiceover?").

**3. "Does this project use timelapses?"** → sets `timelapse_mode`

- `false` (default for safety) — the editor emits NO
  `speed > 1.0` ranges. Even visually-continuous activity stays
  1x or gets cut. This shields b-roll-heavy projects from the
  editor accidentally retiming a stretch the user wanted at
  normal speed.
- `true` — the editor's time-squeezing rules in
  `subagent_editor_rules.md` apply normally. Long visually-
  continuous activity stretches (workshop builds, packing,
  cooking, painting, prep, walking, driving) become candidates
  for 5-30s output timelapses. Hard ceiling stays `speed = 10.0`
  (Hard Rule on retime clamp).
- If step 1 detected a `timelapse/` folder, default this to `true`
  and confirm with the user (they organized for a reason).
- The user can also use this per-project — many
  scripted assemblies want zero timelapses (the b-roll IS the
  visual track at 1x); workshop / build vlogs want explicit
  timelapse permission.

**4. "Is this personal, a creator project, or a professional /
client deliverable?"** → sets `user_profile`

- `personal` — own use, family, hobby. Default verification bar.
- `creator` — own channel / social / portfolio. Default
  verification bar; QA notes in EDL `reason` fields stay terse.
- `professional` — working for a company, client, sponsor, agency,
  or paid deliverable. **Verification bar goes up:** the editor
  subagent's brief carries this flag, the editor must do
  top-candidate review on every named-subject b-roll beat,
  QA notes in `reason` fields list rejected candidates with
  reasons, every specific-mention fallback (e.g. could not find the
  exact game, used closest verifiable match) is explicitly flagged
  so the parent surfaces it.

These four flags + `<edit>/source_tags.json` (when present) travel
with the Conversation Context bundle into every subagent
brief. The editor uses `script_mode` and `b_roll_mode` to pick
cold-path references; `timelapse_mode` to decide
whether to even consider time-squeezing; `user_profile` to set
verification / QA-note discipline. The b-roll scout subagent (when
spawned by the editor) uses `source_tags.json` to scope which clips
qualify as candidates.

### 5. Propose strategy

Write 4-8 sentences describing the editorial shape, take direction,
chosen pacing preset (name + four expanded ms values), animation plan,
subtitle style (NLE captions track), length estimate, NLE delivery
dialect. **Wait for confirmation.** No subagent runs until the user
says yes.

You are writing the strategy in the user's terms — what they said the
video is, what they said they want. You are NOT writing it from a read
of the timeline. The editor subagent will translate the strategy into
specific cut decisions when it reads the timeline in step 6.

### 6. Execute — spawn the editor sub-agent (and animation sub-agents)

Build the editor brief (see "Brief templates" below) and spawn the
editor subagent via the `Task` / `Agent` tool. If animations are
planned, spawn one subagent per slot in parallel — see
`references/animations.md`.

The editor subagent returns `<edit>/edl.json`. The animation sub-
agents return rendered overlay clips in `<edit>/animations/slot_*/`.

### 7. Self-eval — QA gate BEFORE export

The deliverable is the FCPXML / xmeml. Don't write it until the
cut is sane. Self-eval is the gate; export (step 8) only runs once
this passes. **No XML before QA.** Eval reads `edl.json` plus the
source clips — it does not need an exported XML, so producing one
just to QA it would be wasted work that has to be redone if the
EDL changes.

Run `helpers/timeline_view.py` against the **source clips at every
EDL cut boundary** (+/- 1.5s window) and inspect each image for:

- Visual discontinuity / flash / jump at the cut boundary
- Waveform spike at the boundary on the SOURCE side — the NLE will
  honor whatever crossfade you ask, but the cut has to land in
  a place where a crossfade can actually save it
- Animation overlay landing on visible content (not on a hard cut
  or a black frame)

Also sample first 2s, last 2s, and 2-3 mid-points — check shot
selection, animation placement, and overall coherence on the
source side. The NLE is where a colorist / editor sees the final
result; your job is to hand them an EDL whose cut decisions are
already sane.

Then validate the EDL structurally — folded in from what the
exporter would otherwise crash on, so you catch arithmetic /
schema problems here instead of after a wasted export pass:

- Sum of effective range durations matches `total_duration_s` in
  the EDL (catches arithmetic mistakes from the editor subagent)
- Required keys present on every range and animation entry
- No negative or zero-length ranges; no overlapping ranges on the
  same track; no gaps that weren't explicitly placed
- Every animation `start_in_output + duration` stays inside the
  timeline, and `duration` matches what the animation sub-agent
  was asked to render (mismatch silently desyncs the overlay)
- Every clip path referenced by a range exists on disk

If anything fails: re-spawn the editor sub-agent with a brief that
quotes the specific eval failure, get a new `edl.json`, re-eval.
Cap at 3 self-eval passes — if issues remain after 3, flag them
to the user vs looping forever. Only proceed to step 8 once
self-eval passes.

### 8. Export to the NLE — publish on pass

XML-only delivery — no flat-MP4 path exists in this skill anymore.
The cut lives in the NLE and the editor finishes it there. This
step runs ONCE, after step 7 passes, and produces the deliverable
artifacts. If the user later requests a change (step 9), eval
gates the next export the same way.

```bash
python helpers/export_fcpxml.py <edit>/edl.json -o <edit>/cut.fcpxml
```

Default emits BOTH `cut.fcpxml` (Resolve / FCP X) AND `cut.xml`
(Premiere Pro native xmeml) side-by-side from a single timeline build.
Recipient picks whichever NLE they use. Tell Premiere users to
`File -> Import -> cut.xml`. Override with `--targets {both,fcpxml,
premiere}`. `--frame-rate 24` (default), 25, 29.97, 30, 60.

**The captions sidecar is always emitted.** `export_fcpxml.py` calls
`build_master_srt` automatically on every run, writing
`<edit>/master.srt` next to the XML on the OUTPUT timeline (Hard
Rule 5) straight from cached Parakeet transcripts. The SRT is
Premiere-friendly (UTF-8, CRLF, sequential cues, `HH:MM:SS,mmm -->`
timestamps) and Premiere Pro / DaVinci Resolve / Final Cut Pro X
all import it via `File -> Import` onto a captions track that the
editor restyles in their own caption panel.

If subtitles are explicitly out of scope for a session (rare —
silent-cinema, music video without speech), you can opt out:

```bash
python helpers/export_fcpxml.py <edit>/edl.json -o <edit>/cut.fcpxml --no-srt
```

Or regenerate only the SRT after a hand-tweak to the EDL without
re-walking the timeline:

```bash
python helpers/build_srt.py <edit>/edl.json
```

### 9. Iterate + persist — every change request = fresh editor spawn

User feedback comes in. **Do not edit `edl.json` by hand.** Append the
user's verbatim quote to the change-request history in your
Conversation Context bundle, then spawn a fresh editor subagent with
an updated brief that includes:

- The full Conversation Context bundle (now including the new quote)
- The full change-request history (chronological, all prior revisions
  + their diffs)
- The prior `edl.json` (so the subagent can diff)
- An explicit description of THIS revision: what specifically the user
  asked to change, what to keep

Re-self-eval. Re-export only on pass. Show. Loop until the user is
happy — every revision repeats the same gate (eval before XML).

Final export on confirmation. Then **append to `project.md`** — see
"project.md memory format" below. Your only persistent state
across sessions; do not skip it.

---

## Pacing presets — required, asked in step 4

Every session must have a pacing preset (Hard Rule 13). Ask the user
up-front. Default is **Paced**. Each preset expands to four numbers
the editor subagent applies when picking cut points.

| Preset       | min_silence_to_remove | min_talk_to_keep | lead_margin | trail_margin | Vibe |
|--------------|----------------------:|-----------------:|------------:|-------------:|------|
| Calm         |                500 ms |           500 ms |      500 ms |       500 ms | Cinematic, contemplative, breathing room. Documentary, interview, narrative. |
| Measured     |                350 ms |           350 ms |      350 ms |       350 ms | Conversational, professional talking-head, podcast-style. |
| **Paced** *(default)* | **200 ms** |   **200 ms** |  **200 ms** |   **200 ms** | Balanced and modern. Tech demos, launch videos. |
| Energetic    |                100 ms |           100 ms |      100 ms |       100 ms | Tight and punchy. Social, fast tutorials, hype reels. |
| Jumpy        |                 50 ms |            50 ms |       50 ms |        50 ms | Ultra-tight every-breath cut. Montage, trailer, vlog supercut. |

Present the five options with one-line descriptions and tell the user
the default is Paced. They can pick a name or just say "use the
default." The detailed application algorithm lives in
`subagent_editor_rules.md` — you do not need it; just pick the
preset name + four numbers and forward them in the editor brief.

Persist the choice in `project.md` so subsequent sessions inherit a
sensible default — but still ask if the user wants to keep it.

---

## Brief templates

The parent's main editorial output is briefs. The shape below is load-
bearing — subagents need every section to do their job without
guessing. Fill in the placeholders with verbatim user quotes wherever
possible.

### Brief: Vocab sub-agent

Spawn this in step 2.

```
You are the VOCAB sub-agent for a video-use-premiere session. Your job
is to produce a project-specific CLAP zero-shot vocabulary file.

STEP 0 (mandatory before anything else):
  Read references/shared_rules.md IN FULL.
  Read references/subagent_vocab_rules.md IN FULL.
  Those two files are your operating manual.

CONVERSATION CONTEXT (from parent):
  Project summary: <complete description in parent's words>

  Verbatim user quotes (chronological):
    [t=session_start] "<exact quote>"
    [t=after_inventory] "<exact quote>"
    ...

  Things user explicitly asked to keep:
    - "<quote>" (context: ...)
  Things user explicitly rejected:
    - "<quote>" (context: ...)

INPUTS (BOTH MANDATORY — read each end-to-end, line by line):
  - <edit>/audiovisual_timeline.md  (audio + visual interleaved by
    timestamp; at vocab time only the visual lane is populated —
    audio hasn't scored yet, that's exactly what your vocab enables)
  - <edit>/speech_timeline.md       (phrase-grouped transcripts, the
    editorial spine, outer-aligned floor-start/ceil-end ranges)
  - <edit>/visual_timeline.md       (drill-down only — open only on
    a specific ambiguity, never as a substitute for the dual read)

OUTPUT:
  Write <edit>/audio_vocab.txt — 200-1000 short labels, one per line,
  lowercase. Include a healthy negative set (~15-20%). Categories per
  subagent_vocab_rules.md.

RETURN:
  A complete report describing:
    - What kind of soundscape you inferred from the AV + speech
      timelines.
    - Categories you covered (tools, materials, ambience, music,
      animals, vehicles, environments, negatives) with counts.
    - Specific labels you debated and why you chose what you chose.
    - Anything ambiguous you want the parent to confirm with the user.
  No artificial length limit; be thorough.
```

### Brief: Editor sub-agent

Spawn this in step 6 for the initial cut, re-spawn in step 9 for
every revision.

```
You are the EDITOR sub-agent for a video-use-premiere session. Your job
is to produce <edit>/edl.json — the cut decisions for this video.

STEP 0 (mandatory before anything else):
  Read references/shared_rules.md IN FULL.
  Read references/subagent_editor_rules.md IN FULL.
  Those two files are your operating manual. The ABSOLUTE READ
  MANDATE in subagent_editor_rules.md binds you specifically.

  Mode-gated cold-path reads (read each file IN FULL only if the
  matching flag below is true; skip silently if false):
    - script_mode    = <true | false>   -> references/scripted.md
    - b_roll_mode    = <true | false>   -> references/b_roll_selection.md
  These bind in addition to your default rules; they do not replace
  the dual-spine read in STEP 1.

  Time-squeeze permission flag (binds the time-squeezing section of
  your operating manual):
    - timelapse_mode = <true | false>
      false  -> emit NO ranges with speed > 1.0; no timelapses.
      true   -> the time-squeezing rules in your operating manual
                apply normally (5-30s output, speed <= 10.0,
                visually-continuous stretches only).

STEP 1:
  Read BOTH <edit>/audiovisual_timeline.md AND
  <edit>/speech_timeline.md END-TO-END. EVERY LINE. (Per the
  ABSOLUTE READ MANDATE.) No first-N-lines, no grep-and-cut, no
  "I have enough." If either file exceeds one Read call, issue
  sequential Reads with offset/limit until every line is covered.
  The two files are aligned by `## <stem>` headers — scroll them
  in parallel when reasoning about a clip. Same full-coverage
  rule applies to the prior edl.json on revisions.

  If script_mode = true: also read the script itself end-to-end
  (path forwarded below or at <edit>/script.md / <edit>/script.txt).
  If b_roll_mode = true and a clip index exists, the parent will
  list <edit>/clip_index/index.json below; you may use it as a
  shortlisting aid in scripted-mode step 4. Verification (stage 2)
  still binds.

CONVERSATION CONTEXT (from parent):
  Project summary: <complete description in parent's words>

  Verbatim user quotes (chronological):
    [t=session_start] "<exact quote>"
    [t=after_strategy_propose] "<exact quote>"
    [t=after_first_preview] "<exact quote>"
    ...

  Things user explicitly asked to keep:
    - "<quote>" (context: ...)
  Things user explicitly rejected:
    - "<quote>" (context: ...)

  Feature mode flags (asked & confirmed in step 4):
    script_mode    = <true | false>
    b_roll_mode    = <true | false>
    timelapse_mode = <true | false>
    user_profile   = <personal | creator | professional>

  Source tags (from step 1 folder convention auto-detection, if any):
    Source tags JSON: <edit>/source_tags.json (or "(not present)")
    Categories seen:  [<a_roll, b_roll, timelapse, voiceover, ...>]
    When tags exist, restrict b-roll candidate searches to clips
    tagged b_roll / cutaway. A-roll-tagged clips are the primary
    audio bed for talking-head sessions.

  Paired audio (from step 1 stem-pair detection, if any):
    Source pairs JSON: <edit>/source_pairs.json (or "(not present)")
    Pair mode:         <dual_mic | ignore | "(no pairs detected)">
    Pair count:        <N>
    When mode = dual_mic, transcripts/<stem>.json (camera audio) and
    transcripts/<stem>.audio.json (paired-mic audio) BOTH exist for
    each pair. Pick the higher-confidence transcript per cut and
    record which one you used in QA notes — see
    subagent_editor_rules.md "Dual-mic pair handling".

  When script_mode = true:
    Script path:    <edit>/script.md (or absolute path)
    Voiceover path: <abs path to VO file>
    Voiceover transcript (cached): <edit>/transcripts/<vo_stem>.json
  When b_roll_mode = true and a clip index exists:
    Clip index path: <edit>/clip_index/index.json (shortlisting aid only)
  When b_roll_mode = true and you choose to spawn b-roll scout
  sub-agents:
    See references/subagent_editor_rules.md "B-roll scout spawn
    protocol" for when and how. Brief template at
    references/subagent_broll_scout_rules.md.

STRATEGY (parent locked in):
  Beats / structure: <archetype + beat list>
  Pacing preset: <name>
    min_silence_to_remove: <ms>
    min_talk_to_keep:      <ms>
    lead_margin:           <ms>
    trail_margin:          <ms>
  Target runtime: <seconds>
  Delivery: <flat mp4 / fcpxml / both>
  Verbal slips to avoid (parent's note from any prior scout): <list>

CHANGE-REQUEST HISTORY (chronological, all revisions so far):
  Revision 0 (initial spawn):
    User asked: "<verbatim quote>"
    Strategy locked: <as above>
    EDL produced: <reference to revision-0 edl.json or "this is rev 0">

  Revision 1:
    User asked: "<verbatim quote>"
    Diff applied: "<short description of what changed>"

  Revision N (THIS SPAWN):
    User asks: "<verbatim quote>"
    Specific changes to apply: <description>
    Prior EDL (input — diff against this): <pasted edl.json or path>

OUTPUT:
  Write <edit>/edl.json. Format per subagent_editor_rules.md.

  Per Hard Rule 14: emit audio_lead = video_tail = transition_in = 0.0
  on every range. No J-cuts, no L-cuts, no dissolves.

RETURN:
  - One-line runtime check (sum of range durations vs target)
  - Per-beat rationale (one line per beat, why this take, why these
    boundaries)
  - Any beat where you had to compromise (note the compromise)
```

### Brief: Animation sub-agent

See `references/animations.md`. One subagent per slot, all spawned in
parallel via the `Task` / `Agent` tool (Hard Rule 10).

---

## Output spec

Match the source unless the user asked for something specific. Common
targets: `1920x1080@24` cinematic, `1920x1080@30` screen content,
`1080x1920@30` vertical social, `3840x2160@24` 4K cinema,
`1080x1080@30` square. The XML carries a timeline frame rate, not a
canvas resolution — the NLE inherits resolution from source clips.
Pass `--frame-rate` matching the source (or user's intended
deliverable) so cuts snap to whole frames. Worth asking the user which
NLE they use so the timeline rate matches.

---

## EDL format (parent reads sub-agent output)

The editor subagent emits this; the parent only needs to validate
high-level shape (range count, total duration, all `audio_lead` /
`video_tail` / `transition_in` are 0.0).

```json
{
  "version": 1,
  "sources": {"C0103": "/abs/path/C0103.MP4"},
  "ranges": [
    {"source": "C0103", "start": 2.42, "end": 6.85,
     "beat": "HOOK", "quote": "...", "reason": "Cleanest delivery.",
     "audio_lead": 0.0, "video_tail": 0.0, "transition_in": 0.0}
  ],
  "pacing_preset": "Paced",
  "pacing": {"min_silence_to_remove_ms": 200,
             "min_talk_to_keep_ms": 200,
             "lead_margin_ms": 200, "trail_margin_ms": 200},
  "overlays": [{"file": "edit/animations/slot_1/render.mp4",
                "start_in_output": 0.0, "duration": 5.0}],
  "subtitles": "edit/master.srt",
  "total_duration_s": 87.4
}
```

The `subtitles` field points at the SRT sidecar emitted by
`helpers/build_srt.py` so the NLE imports it on a captions track.
Color is out of scope — no `grade` field exists; the colorist
owns it end-to-end in the NLE.

If the editor subagent returns an EDL with non-zero split-edit fields,
reject it and re-spawn — that violates Hard Rule 14.

---

## project.md memory format

Append one section per session at `<edit>/project.md`:

```markdown
## Session N — YYYY-MM-DD

**Strategy:** one paragraph describing the approach.
**Pacing:** preset name + the four expanded ms values.
**Mode flags:**
  script_mode    = <true | false>
  b_roll_mode    = <true | false>
  timelapse_mode = <true | false>
  user_profile   = <personal | creator | professional>
**Source tags:** present | not present (e.g. categories = a_roll, b_roll, timelapse)
**Paired audio:** mode = dual_mic | ignore | none (e.g. "dual_mic, 12 pairs from Sony+Zoom rig")
**Decisions:** take choices, cuts, grades, animations + why.
**Reasoning log:** one-line rationale for non-obvious decisions.
**Outstanding:** deferred items.

**Conversation Context snapshot (final state):**
  Project summary: <as written this session>
  Key user quotes: <3-5 most load-bearing verbatim quotes>
```

On startup, read `project.md` if it exists and summarize the last
session in one sentence before asking whether to continue. The
Conversation Context snapshot at the bottom lets next session's parent
rebuild the bundle without re-asking the user every preference.

The **Mode flags** block persists across sessions for the
three step-4 questions: when the user opens a project tomorrow that
was scripted today, the parent reads this block and defaults the
gating questions to last session's answers (still asks to confirm
— users change projects mid-stream). If the block is missing
(legacy `project.md` from before this format), treat as "ask all
three fresh."

---

## Helpers (parent invokes via Bash)

> **All helpers live in `helpers/`.** Always invoke from the skill
> root as `python helpers/<script>.py ...` (the sibling-import pattern
> they use depends on `helpers/` being the script's own directory,
> which `sys.path` resolves automatically when run by path). Never
> `cd helpers/` first — `cwd` semantics differ across shells
> (PowerShell, bash, agentic shells that do not persist `cd`), and the
> cache layout assumes the project root is the cwd.

### Phase A — speech + visual (default)

- **`helpers/preprocess_batch.py <videos_dir>`** — auto-discover
  videos, run the speech (Parakeet ONNX) + visual (Florence-2) lanes
  with VRAM-aware scheduling. Default entry point.
  - Flags: `--wealthy` (24GB+ GPU), `--diarize`, `--language en`,
    `--force`, `--skip-speech`, `--skip-visual`.
  - **Do NOT use `--include-audio`.** That flag runs CLAP inline
    against a baked-in fallback vocabulary that exists only for
    `tests.py` smoke-testing. The mandated workflow: speech +
    visual finish first, then the vocab subagent generates a
    project-specific `audio_vocab.txt`, then `audio_lane.py` runs
    against THAT. No "skip the vocab subagent" shortcut exists.

- **`helpers/preprocess.py <video1> [<video2> ...]`** — same
  orchestrator with explicit file list. Use when you want a subset.

- **`helpers/pack_timelines.py --edit-dir <dir>`** — read the
  available lane caches (`transcripts/`, `audio_tags/`,
  `visual_caps/`) and produce TWO mandatory sub-agent reading
  surfaces — `audiovisual_timeline.md` (audio + visual lanes
  interleaved by timestamp; NO speech) and `speech_timeline.md`
  (phrase ranges, outer-aligned `floor(start)..ceil(end)` integer
  rounding) — plus the per-lane drill-down views
  `audio_timeline.md` (only if Phase B has run) and
  `visual_timeline.md`. Pass `--no-audiovisual` (legacy alias
  `--no-merge`) to skip the AV view (rare). Safe to call multiple
  times — re-running after Phase B folds the new audio events into
  both `audiovisual_timeline.md` and `audio_timeline.md`.

  **Run this exactly TWICE per session.** First call after Phase A
  preprocess produces `speech_timeline.md` plus an
  `audiovisual_timeline.md` carrying only the visual lane (audio
  hasn't scored yet); the vocab sub-agent reads BOTH end-to-end —
  that's why the AV file exists at vocab time even though it has
  no `(audio: ...)` lines. Second call after `audio_lane.py`
  (Phase B) folds the new audio events into the same
  `audiovisual_timeline.md` so the editor sub-agent's dual-spine
  read covers audio + visual + speech. Skipping the second pack
  hands the editor a stale AV file and silently violates Hard Rule
  15. See step 2 of the 9-step process for the exact ordering.

  **Caveman compression on visual captions is ON by default** — a
  spaCy NLP pass strips stop words / determiners / auxiliaries / weak
  adverbs from every Florence-2 caption before packing, cutting
  `audiovisual_timeline.md` token cost by ~55-60% on detailed-caption
  footage with zero loss of editorial signal (entities, actions,
  colours, shot composition all survive). Cached in
  `<edit>/comp_visual_caps/` keyed by source mtime + caveman version
  + lang; subsequent re-packs run instantly.

  Pass `--no-caveman` to read the raw Florence paragraphs (slower,
  bigger, only useful for debugging what Florence actually said).
  `--caveman-lang en` (default) picks the spaCy model;
  `--caveman-procs N` overrides the worker count (default
  `min(n_files, cpu_count // 2)`); `--force-caveman` re-runs even
  cached files.

  Sentence-level fuzzy delta dedup also applies at pack time:
  visually static frames collapse to `(same)` in
  `visual_timeline.md` and disappear entirely from
  `audiovisual_timeline.md`; slowly-evolving frames emit only the
  NEW sentences with a `+ ` prefix (think `git diff` additions).

- **`helpers/caveman_compress.py`** — standalone CLI for the caveman
  pass. Useful for debugging the compression on a single caption
  (`python helpers/caveman_compress.py "verbose text"`) or for
  manually batching a `visual_caps/` directory (`python
  helpers/caveman_compress.py --visual-caps <edit>/visual_caps/`).
  The pack helper calls it automatically — only use this directly
  when iterating on filter rules.

### Audio events (CLAP) — agent-curated vocabulary, mandatory

The audio workflow has only one path: **spawn the vocab subagent**
(it reads BOTH `audiovisual_timeline.md` AND `speech_timeline.md`
from step 1's first pack, end-to-end, line by line, and writes
`<edit>/audio_vocab.txt`), then run `audio_lane.py` against that
vocab, then re-pack timelines. No smoke-test / agent-less fallback
exists in the parent's playbook; the baked-in default vocab in
`audio_vocab_default.py` exists only for `tests.py`. See "Brief
templates" below for the vocab subagent brief.

- **`helpers/audio_lane.py <video1> [<video2> ...] --vocab
  <edit>/audio_vocab.txt --edit-dir <edit>`** — run CLAP zero-shot
  scoring against the curated vocabulary. **`--vocab` is mandatory**;
  the lane has a baked-in fallback for `tests.py` smoke testing but
  the parent never relies on it. Caches text embeddings in
  `audio_vocab_embeds.npz` so subsequent runs are fast.
  - Flags: `--device {cuda,cpu}`, `--model-tier {base,large}`,
    `--windows-per-batch N`, `--force`.
  - If `<edit>/audio_vocab.txt` does not exist yet, the vocab sub-
    agent has not run yet — go back to step 2 and spawn it.

- After Phase B finishes, **re-run `pack_timelines.py`** to fold the
  new audio events into both `audiovisual_timeline.md` (default) and
  `audio_timeline.md`.

**Individual lanes** (rarely needed — the orchestrator wraps them):
`helpers/parakeet_onnx_lane.py`, `helpers/parakeet_lane.py` (NeMo
fallback), `helpers/audio_lane.py`, `helpers/visual_lane.py`. Each
accepts `--wealthy` and runs standalone.

- **`helpers/extract_audio.py <video>`** — manually extract 16kHz
  mono WAV. Cached. Mainly for debugging.
- **`helpers/vram.py`** — print detected GPU + the schedule that
  would be chosen. Useful sanity check.

### Editing

- **`helpers/timeline_view.py <video> <start> <end>`** — filmstrip +
  waveform PNG for visual drill-down. **The editor subagent invokes
  this** at decision points; the parent does not. (If a scout sub-
  agent is needed for parent-level questions, the scout calls it.)
  Not a scan tool — use it at decision points, not constantly. The
  `visual_timeline.md` replaces 90% of the old "scan with
  timeline_view" workflow.

- **`helpers/export_fcpxml.py <edl.json> -o cut.fcpxml`** — emit
  editor-ready timeline files. **The only delivery path in
  the skill** — no flat-MP4 renderer exists. Hard-cut delivery
  only right now (Hard Rule 14): the EDL's `audio_lead` /
  `video_tail` / `transition_in` fields are still consumed by the
  code path but the editor subagent must emit `0` for all three.

  **Default emits BOTH `cut.fcpxml` AND `cut.xml`** side-by-side
  from a single timeline build, because Premiere Pro and Resolve /
  FCP X want different XML dialects: `.fcpxml` (FCPXML 1.10+) is
  native to DaVinci Resolve and Final Cut Pro X, `.xml` (Final Cut
  Pro 7 xmeml) is native to Premiere Pro. The recipient picks
  whichever NLE they use — no XtoCC conversion required for
  Premiere. Override with `--targets {both,fcpxml,premiere}`.
  `--frame-rate 24` (default), 25, 29.97, 30, 60.

- **`helpers/build_srt.py <edl.json>`** — emit `<edit>/master.srt`
  on the OUTPUT timeline (Hard Rule 5) from the cached Parakeet
  transcripts in `<edit>/transcripts/`. Premiere-friendly format
  (UTF-8, CRLF, sequential cues, `HH:MM:SS,mmm -->` timestamps);
  imports cleanly via `File -> Import` in Premiere / Resolve / FCP X.
  2-word UPPERCASE chunks, breaks on punctuation. Skips retimed
  (timelapse) ranges by editor convention — see SKILL.md
  "Time-squeezing". **`export_fcpxml.py` calls this automatically**
  on every export; the standalone CLI exists for when you
  hand-tweaked the EDL and only want to regenerate the captions
  without rebuilding the XML timeline.

For animations, create `<edit>/animations/slot_<id>/` with `Bash`
and spawn a subagent via the `Task` / `Agent` tool per
`references/animations.md`.

---

## Cold-path references — load on demand

Cold-path features the **editor subagent** loads only when the
matching mode flag is true in its brief, plus a couple of
parent-side feature references. Read the matching file before
proposing strategy for that feature; do not read pre-emptively.

**Editor cold-path (gated by step-4 questions):**

- `references/scripted.md` — script + voiceover assembly
  procedure, beat segmentation, vo-anchored timing, source-in-point
  sync on named subjects. Editor reads when
  `script_mode = true`.
- `references/b_roll_selection.md` — b-roll selection preference
  order (signage / product / gameplay / booth / stage / people),
  rejection rules, stability bias, diversification, optimized
  matching philosophy (caching / two-stage / clip index). Editor
  reads when `b_roll_mode = true`. Common combo: scripted assembly
  triggers both.

**Sub-sub-agent role spawned by the editor (not by you):**

- `references/subagent_broll_scout_rules.md` — b-roll scout
  subagent's operating manual. Spawned by the **editor subagent**
  (not by the parent) on demand, one per beat or one per cluster
  of beats, in parallel per Hard Rule 10. Reads
  `<edit>/visual_timeline.md` for in-scope sources and returns
  ranked candidate shortlists. Editor picks / verifies / writes the
  EDL range. The parent never spawns scouts directly — but the
  parent's editor brief lists `source_tags.json` so the editor
  can pass the in-scope source list down to scouts when spawning
  them.

**Parent-side cold-path (read when the feature is in scope):**

- `references/subtitles.md` — chunking / case / placement,
  `bold-overlay` and `natural-sentence` worked styles, FCPXML
  delivery and the always-emitted `master.srt` sidecar.
- `references/animations.md` — animation subagent brief template,
  PIL / Manim / Remotion timing and easing, parallel-spawn
  discipline (Hard Rule 10).

The Hard Rules that bind these features stay in `shared_rules.md`
— output-timeline SRT (Rule 5) and parallel subagents for
animations (Rule 10) are the live ones for XML delivery; the
ffmpeg-pipeline rules (1-4) are dormant since the flat-MP4 path
was removed. Color grade is out of scope — the colorist owns it
in the NLE.

---

## Editor return rationale — extra blocks to echo to the user

Two editor subagent features detect **user intent baked into the
source recording itself** and produce dedicated blocks in the
editor's return rationale that you must surface to the user when
describing the cut. Detection happens autonomously inside the
editor's dual-timeline read; you do not gate them with flags,
you do not read the timeline yourself. Your job is echo +
clarification.

### `In-clip editor notes` block

The editor detects verbal directives the user recorded into the
source clips themselves — *"hey to the AI editing this, skip the
first take"*, *"editor's note: this clip is a throwaway"*,
countdown markers (*"three two one"*) excluded from the cut, etc.
Full detection / application rules live in
`subagent_editor_rules.md` "In-clip editor notes".

The editor returns a list shaped like:

```
In-clip editor notes detected:
  - C0312 t=0.4s "skip the first take, the second is the keeper"
    — APPLIED. ...
  - C0418 t=0.8s "this whole clip is a throwaway" — APPLIED. ...
  - C0507 t=12.3s "hey editor I think we should..." — IGNORED.
    No clear directive followed.
```

What you do with it:

1. **Echo every APPLIED note back to the user**, quoting the
   editor's transcribed phrasing verbatim. *"In your C0312 you
   said 'skip the first take, the second is the keeper' — went
   with the second, like you asked."*
2. **Surface ambiguous notes the editor flagged** (*"cut around
   the embarrassing bit"*, *"the audio is bad here"*). Ask one
   clarifying question, capture the user's answer as a
   verbatim conversation quote, re-spawn the editor with the
   quote in the bundle.
3. **Persist applied notes in `project.md`** — one-liner per
   session is enough.

If the user wants to disable the feature for a session (e.g. *"my
brother yells 'hey AI' as a joke, ignore those"*), capture the
exact quote in the bundle. The editor honors that override.

### `Retake decisions` block

The editor detects when the user re-recorded a line — frustration
marker (*"fuck, again"*), restart phrase (*"let me try that
again"*), long-pause-then-restart, slate / clap between takes, or
a fresh clip starting with the same content as the last — and
picks the cleaner take, dropping the rest. Full rules live in
`subagent_editor_rules.md` "Retake detection".

The editor returns a list shaped like:

```
Retake decisions:
  - INTRO beat: kept C0312 14.2-22.5 (later take); dropped
    C0312 4.1-12.0 — speaker said "fuck, again" at 11.4s.
  - DEMO beat: kept C0312 first delivery 32.0-41.5; later
    delivery at 45.0-54.2 had three "uh" fillers vs zero on
    the first, so the earlier take won.
  - PUNCHLINE: kept BOTH "we'll never ship" instances —
    rhetorical emphasis, not a retake.
```

What you do with it:

1. **Echo retake calls when they are load-bearing** — the user
   wants to know their second take landed. Need not be
   exhaustive on every micro-retake; surface the meaningful ones.
2. **Surface override calls** explicitly — when the editor used
   the EARLIER take vs the later (because the later was
   worse), tell the user that judgement was made and why.
3. **Surface "kept both" calls on emphatic repetition** so the
   user knows nothing was dropped by mistake.

This block is informational; the editor already made the call.
You are translating to the user, not re-deciding.

### When the user overrides a retake decision

If the user replies *"actually use the first take of the intro,
the second is too rushed"*, capture verbatim and re-spawn the
editor with that quote in the bundle. The editor's "explicit
user quote" priority overrides its retake heuristic.

---

## Parent-specific anti-patterns

- **Reading `audiovisual_timeline.md`, `visual_timeline.md`, or
  `audio_timeline.md`.** See shared_rules.md "Agent roles" — the
  heavy caption / event density stays in sub-agent windows so the
  parent's accumulating context does not bloat. Spawn a scout for
  visual / audio questions. Reading `speech_timeline.md` for
  conversation-side content awareness IS allowed and expected
  (step 3) — speech is text-only and token-cheap.
- **Skipping the step-3 speech read.** Without it you are talking
  to the user blind about a video you have no content awareness
  of. Strategy confirmation (Hard Rule 11) becomes guesswork
  instead of a conversation grounded in what was actually said.
- **Editing `edl.json` by hand for "trivial" tweaks.** Always re-spawn.
- **Curating `audio_vocab.txt` by hand.** Always re-spawn.
- **Skipping the second `pack_timelines.py` run after
  `audio_lane.py`.** The pack runs TWICE per session by design —
  first after Phase A preprocess (so the vocab sub-agent has a
  AV + speech timelines to read), second after Phase B audio scoring
  (so the editor sub-agent's AV file carries audio + visual). Skipping
  the second pack ships the editor an AV view with no
  `(audio: …)` lines, silently violating Hard Rule 15 (the dual
  spine of `audiovisual_timeline.md` + `speech_timeline.md` — both
  must be current). See step 2.
- **Skipping the pacing prompt.** Hard Rule 13.
- **Skipping the four mode-gating questions in step 4.** They set
  `script_mode`, `b_roll_mode`, `timelapse_mode`, `user_profile` —
  the editor subagent's cold-path reads, retime permission, and
  verification bar all depend on them. Default-guessing the flags
  ships the wrong cut.
- **Forgetting to forward the mode flags or `source_tags.json`
  into the editor brief.** Without them the editor falls back to
  non-scripted, non-b-roll, no-tags, default-bar behaviour even
  if the user is on a scripted client deliverable with organized
  folders.
- **Running `preprocess_batch.py` without `--detect-pairs` first
  on inventories that look like camera + recorder rigs (matched
  stems for `.mp4`/`.mov` and `.wav`).** The script will refuse
  with rc=2 and you will have to re-ask the user anyway — but if you
  miss the failure on a long preprocess invocation it wastes their
  GPU time. Detect first, ask, then run with the chosen mode.
- **Picking the paired-audio mode for the user.** Do not assume
  "they probably wanted second mic" — some rigs auto-mirror the
  camera audio as a `.wav` backup and the user genuinely wants it
  ignored. Always ask, always quote the answer back, persist the
  choice in `project.md` so next session's default is right.
- **Forgetting to forward `source_pairs.json` into the editor
  brief when it exists.** Without it the editor does not know that
  `transcripts/SHOT_0042.json` and `transcripts/SHOT_0042.audio.json`
  are the same shot; it will happily place the same beat twice with
  different transcriptions.
- **Skipping the step-1 folder convention scan.** Even when the
  user only mentions "I have a script and some b-roll" in passing,
  scan the videos_dir for convention folders before asking — the
  user organized for a reason; pre-filling beats re-asking.
- **Forcing folder convention.** Auto-detection is opt-in
  organization. Do not lecture users who use a flat folder layout;
  ask the four questions fresh and move on.
- **Inventing ad-hoc cut-padding numbers.** Pacing preset is the
  contract.
- **Paraphrasing user quotes in the brief.** Quote verbatim. Vibes
  matter.
- **Forgetting to forward the change-request history on revisions.**
  The subagent has no memory across spawns; the brief is its memory.
- **Spawning subagents sequentially when they could be parallel.**
  Animations especially. Hard Rule 10.
- **Exporting XML before self-eval passes, or presenting the
  preview before self-eval passes.** Step 7 gates step 8 — no
  `cut.fcpxml` / `cut.xml` / `master.srt` written until the EDL
  passes QA. Self-eval is mandatory.
- **Skipping the `project.md` append at session end.** That is how
  next session knows where this one stopped.
- **Burying or paraphrasing an in-clip editor note the subagent
  flagged.** If the editor's return rationale contains
  `In-clip editor notes detected`, surface every APPLIED entry
  to the user and quote the transcribed phrasing verbatim. Hiding
  or summarizing them violates the user's expectation that their
  recorded directives reach them.
- **Skipping the `Retake decisions` block in the cut summary.**
  The user wants to know which take landed. At minimum, surface
  any decision where the editor went against the default
  (later-take wins) or kept both takes of repeated content.
