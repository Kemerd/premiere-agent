# Parent Rules — operating manual for the orchestrator agent

You have already read `references/shared_rules.md`. If you have not,
stop and read it now — it defines the agent hierarchy you sit at the
top of, and reading these rules without that context will lead you to
do a sub-agent's work yourself.

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
- **Error handling** — when a helper fails, you read the traceback,
  diagnose, fix, and re-run. Sub-agents do not debug infrastructure.
- **Sub-agent dispatch** — spawn the editor sub-agent for cuts, the
  vocab sub-agent for Phase B, animation sub-agents in parallel for
  overlays. Forward the Conversation Context bundle into every brief.
- **Output translation** — read sub-agent return values (EDL JSON,
  vocab.txt, animation reports), translate back to the user in plain
  English.

What you DO NOT do:

- **Read `merged_timeline.md`, `speech_timeline.md`,
  `visual_timeline.md`, or `audio_timeline.md` directly.** Token-heavy
  reads of caption / transcript content are sub-agent territory; that
  is the entire reason the architecture exists. The parent dispatches
  reads, never performs them.
- **Edit `edl.json` by hand.** Every change re-spawns the editor.
- **Curate `audio_vocab.txt` by hand.** Always re-spawn vocab.
- **Open source video files** to look at frames. If a frame inspection
  is needed, spawn a scout sub-agent and let it call
  `helpers/timeline_view.py`.

Sub-agents are SPECIALIZED workers spawned for token-heavy reading
tasks (timeline ingestion, vocabulary curation, animation rendering).
The parent does *everything else* in this skill.

---

## Directory layout

The skill lives in `video-use-premiere/`. User footage lives wherever
they put it. All session outputs go into `<videos_dir>/edit/`.

```
<videos_dir>/
├── <source files, untouched>
└── edit/
    ├── project.md               ← memory; appended every session.
    │                              The ONE timeline-adjacent file the
    │                              parent reads directly.
    ├── merged_timeline.md       ← editor sub-agent's default reading
    │                              surface — all 3 lanes interleaved
    │                              chronologically by timestamp.
    │                              PARENT NEVER OPENS THIS.
    ├── speech_timeline.md       ← Parakeet phrase-level transcripts
    │                              (lane 1, drill-down for editor)
    ├── audio_timeline.md        ← CLAP audio events, coalesced
    │                              (lane 2, drill-down for editor,
    │                              produced by Phase B vocab pass)
    ├── visual_timeline.md       ← Florence-2 captions @ 1fps
    │                              (lane 3, drill-down for editor)
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
  needed when ONNX Runtime can't load on the host).
- **Speech lane backends**: default is `parakeet_onnx_lane.py` —
  NVIDIA Parakeet TDT 0.6B on ONNX Runtime through a multi-session
  pool (TensorRT / CUDA / DirectML / CPU EP ladder, English v2 /
  multilingual v3 auto-routed by language). The only sanctioned
  alternative is `parakeet_lane.py` (NeMo torch-mode) for hosts where
  ORT can't load — pin via `VIDEO_USE_SPEECH_LANE=nemo`. Output JSON
  shape is byte-identical between the two. `helpers/health.py --json`
  surfaces non-default backends in `fallbacks_active` so you know
  which one is running before the lane fires. **Air-gapped?** Pre-
  download the ONNX directory and set `PARAKEET_ONNX_DIR=/path/to/
  parakeet-onnx`; the lane skips all network calls. There is no
  Whisper backend in this codebase by design — Whisper hallucinates on
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
per-session `<videos_dir>/edit/` so it persists across projects. This
is the one exception to Hard Rule 12, and it's intentional: skill-
environment health is a per-machine property, not a per-session one.

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
| `fail` | **Stop.** Print the failure list + the `advice` strings verbatim. Ask the user to run the fix and re-invoke. Don't pretend the rest of the skill will work — broken `ffmpeg` or missing `transformers` will silently corrupt every subsequent step. |

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
install. Cached separately under the same TTL. Don't trigger this
autonomously — it's an explicit user action.

### 1. Inventory + Phase A preprocess

`ffprobe` every source. **You may run `ffprobe` because it is metadata-
only** — it does not give you a view of the *content*, just duration /
codec / framerate / channels. That is parent-allowable. Anything that
returns content (frames, transcript text, captions) is sub-agent
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
  knows whether their voiceover.wav was picked up.
- `<edit>/audio_16k/<stem>.wav` is produced for every source by the
  same ffmpeg `-vn` resample, so the speech lane and CLAP audio lane
  see audio-only sources exactly the same as video sources.
- `<edit>/transcripts/<stem>.json` ships for every source, including
  audio-only ones — Parakeet doesn't care whether the WAV came from
  a `.mov` or a `.wav`.
- `<edit>/visual_caps/<stem>.json` is **not** produced for audio-only
  stems; the visual lane is scoped out of them. `pack_timelines.py`
  handles missing visual_caps gracefully (the merged timeline simply
  shows no visual captions for those stems).
- If the entire batch is audio-only (e.g. user is rendering a
  scripted podcast assembly with no footage at all), the visual lane
  is skipped entirely and the orchestrator logs `visual lane skipped
  — no video sources in this batch`.

You don't have to do anything special — just confirm in the inventory
recap that you noticed any audio-only files (`"I see your shoot has
12 .mp4 clips plus voiceover_final.wav as an audio-only source —
that'll get transcribed alongside the footage."`) so the user knows
they weren't dropped.

#### Folder convention auto-detection (optional)

Before preprocessing, look at the videos_dir directory tree for
**optional convention subfolders**. Users who organize their
material this way pre-fill the step-4 mode-gating defaults; users
who don't get asked fresh in step 4. Detection is case-insensitive
and treats hyphens / underscores as equivalent.

| Folder pattern (any of) | Maps to category | Pre-sets |
|---|---|---|
| `b_roll/` `b-roll/` `broll/` `cutaway/` `cutaways/` `inserts/` | `b_roll` | `b_roll_mode = true` |
| `timelapse/` `timelapses/` `time_lapse/` `tl/` | `timelapse` | `timelapse_mode = true` |
| `voiceover/` `voiceovers/` `vo/` `narration/` | `voiceover` | `script_mode = true` (then ask for script path). Files in this folder are commonly audio-only (`.wav`, `.mp3`) — the pipeline transcribes them via Parakeet without needing a video container. |
| `a_roll/` `a-roll/` `aroll/` `main/` `interview/` `interviews/` `takes/` | `a_roll` | (informational tag only) |
| `script/` `scripts/` (or any `*.txt` / `*.md` named like `script*`) | `script` | `script_mode = true` (script path identified) |

If a convention folder is detected, **write a small JSON tag map**
at `<edit>/source_tags.json` mapping every clip's stem to its
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

Sources that don't fall under any convention folder go in as
`unknown` — the user can clarify at step 4 ("are these b-roll or
A-roll?") or leave them untyped (the editor scopes its candidate
search across all sources when nothing's tagged).

**Confirm detection with the user** before locking the modes — the
parent's question shape:

> "I see a `b_roll/` and a `timelapse/` subfolder under your videos_dir.
> Defaulting `b_roll_mode = true` and `timelapse_mode = true` for this
> session. The `b_roll/` clips become the cutaway library; the
> `timelapse/` clips become candidates for the editor's time-squeeze
> rules. Sound right?"

If the user says no (e.g. they named a folder `b_roll/` for
historical reasons but those clips are actually A-roll for THIS
project), drop the auto-tag and ask the step-4 question fresh.

**No convention folders detected → no `source_tags.json`** is
written, and the parent asks all four mode-gating questions fresh
in step 4.

This is **optional convention** — the skill works fine without any
folder organization. The auto-detection is a UX nicety, not a
requirement. Don't insist; don't lecture the user about
organization.

Then run Phase A preprocessing — speech (Parakeet ONNX) + visual
(Florence-2):

```bash
python helpers/preprocess_batch.py <videos_dir>
python helpers/pack_timelines.py --edit-dir <videos_dir>/edit
```

`preprocess_batch.py` flags: `--wealthy` (24GB+ GPU), `--diarize`,
`--language en`, `--force`, `--skip-speech`, `--skip-visual`. **Do
NOT pass `--include-audio`** — that flag runs CLAP inline against a
fallback vocabulary that exists only for `tests.py` smoke testing.
This skill mandates the curated-vocab path (step 2 below).

`pack_timelines.py` produces `merged_timeline.md` (the editor's
default reading surface) plus the per-lane drill-down files:
`speech_timeline.md`, `audio_timeline.md` (only after step 2 runs),
`visual_timeline.md`. **You don't read any of these.** They exist
for sub-agents.

### 2. Audio events — spawn the vocab sub-agent (mandatory)

Spawn the vocab sub-agent. It reads `speech_timeline.md` +
`visual_timeline.md` and produces a project-specific CLAP vocabulary
at `<edit>/audio_vocab.txt`. This is **not optional** and there is
**no shortcut to a baseline vocabulary** — generic 527-class
taxonomies mis-label real-world content (workshop tools tagged as
"music", room tone tagged as "applause"). The agent-curated vocab is
the only path that produces a usable audio_timeline.

**Brief template (vocab sub-agent):** see "Brief templates" section
below. Spawn via the `Task` / `Agent` tool.

After the vocab sub-agent returns `<edit>/audio_vocab.txt`:

```bash
python helpers/audio_lane.py <video1> [<video2> ...] \
    --vocab <edit>/audio_vocab.txt --edit-dir <edit>
python helpers/pack_timelines.py --edit-dir <edit>
```

The first command runs CLAP zero-shot scoring against the curated
vocab. The second re-runs the pack so the new audio events are folded
into both `merged_timeline.md` (default) and `audio_timeline.md`.

If the user explicitly says they don't care about audio events at
all (rare — usually a single-speaker talking-head with no ambient
work), you may skip the vocab sub-agent and the `audio_lane.py` run.
The editor sub-agent will still cut from the merged_timeline using
just speech + visual; `(audio: ...)` lines will simply be absent.
Note the skip in `project.md` so next session knows.

### 3. Pre-scan for problems — DO NOT DO THIS YOURSELF

In the previous design, the parent would read `merged_timeline.md`
end-to-end before talking to the user. **That is no longer the parent's
job.** You don't read it. The editor sub-agent does that on every
spawn, in a fresh context window.

If you genuinely need a one-paragraph summary of the project to talk
intelligently to the user (e.g. user is being vague), spawn a tiny
"scout" sub-agent with a brief like:

```
You are a SCOUT sub-agent. Read <edit>/merged_timeline.md in full and
return a 3-5 sentence summary of: what kind of video this appears to
be, who appears to be in it, what activity dominates, what unique
moments stand out. No EDL, no taste calls. Just a description.
```

Keep scouting rare. Most sessions, the user describes the project
clearly enough that you can write a strategy without scouting.

### 4. Converse — your main job

Describe the project back to the user in your own words based on what
they've told you, plus the `project.md` summary if it exists. Ask
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
  which cold-path references the editor sub-agent loads on spawn,
  whether time-squeezing is permitted, and what verification bar
  the editor uses.

While conversing, build the **Conversation Context bundle** in your
working memory (see `shared_rules.md` Agent roles for the structure).
You will forward this bundle into every sub-agent brief.

#### Mode-gating questions

Four short questions — ask once per session, persist the answers in
`project.md` so subsequent sessions inherit defaults. Always still
confirm; users change modes between sessions on the same project.

If `<edit>/project.md` already records a value, default to it and
ask the user to confirm or change ("Last session you were in scripted
mode — sticking with that?"). If no project.md exists, ask all four
fresh — UNLESS step 1's folder convention auto-detection
pre-filled some of them, in which case treat the auto-detected
values as the defaults and confirm rather than re-ask blank
("I detected a `b_roll/` folder — going with `b_roll_mode = true`,
sound right?").

**1. "Are you using a script?"** → sets `script_mode`

- `false` (default) — talking-head / interview / vlog / workshop /
  travel / event recap where the speaker's recorded audio carries
  the cut. The default speech-first cut model applies.
- `true` — the user has written a script AND has a separate
  voiceover (or will record one). The editor will assemble b-roll
  matched to the voiceover; the script anchors the beats. The
  editor sub-agent reads `references/scripted.md` on spawn when
  this is true.
- Edge case: user wrote a script but is reading it on camera (no
  separate VO). Treat as `script_mode = false` but forward the
  script in the brief as a structural hint — `scripted.md` covers
  this case.

**2. "Are you using b-roll / cutaways?"** → sets `b_roll_mode`

- `false` — single-source dialogue with no cutaway material.
- `true` — there is b-roll / cutaway material to layer over the
  A-roll OR the project is full scripted assembly under a
  voiceover. The editor sub-agent reads
  `references/b_roll_selection.md` on spawn when this is true.
- In practice, scripted assembly is always `b_roll_mode = true`
  too (the b-roll IS the visual track). Don't ask twice — when the
  user confirms `script_mode = true`, set `b_roll_mode = true`
  automatically and confirm with the user that's right ("So we'll
  also have b-roll under the voiceover?").

**3. "Does this project use timelapses?"** → sets `timelapse_mode`

- `false` (default for safety) — the editor will NOT emit any
  `speed > 1.0` ranges. Even visually-continuous activity stays
  1x or gets cut. This protects b-roll-heavy projects from the
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
- The user can also use this on a per-project basis — many
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
  sub-agent's brief carries this flag, the editor must do
  top-candidate review on every named-subject b-roll beat,
  QA notes in `reason` fields list rejected candidates with
  reasons, every specific-mention fallback (e.g. couldn't find the
  exact game, used closest verifiable match) is explicitly flagged
  so the parent surfaces it.

These four flags + `<edit>/source_tags.json` (when present) travel
together with the Conversation Context bundle into every sub-agent
brief. The editor uses `script_mode` and `b_roll_mode` to decide
which cold-path references to read; `timelapse_mode` to decide
whether to even consider time-squeezing; `user_profile` to set the
verification / QA-note discipline. The b-roll scout sub-agent (when
spawned by the editor) uses `source_tags.json` to scope which clips
are eligible candidates.

### 5. Propose strategy

Write 4-8 sentences describing the editorial shape, take direction,
chosen pacing preset (name + four expanded ms values), animation plan,
subtitle style (NLE captions track), length estimate, NLE delivery
dialect. **Wait for confirmation.** No sub-agent runs until the user
says yes.

You are writing the strategy in the user's terms — what they said the
video is, what they said they want. You are NOT writing it from a read
of the timeline. The editor sub-agent will translate the strategy into
specific cut decisions when it reads the timeline in step 6.

### 6. Execute — spawn the editor sub-agent (and animation sub-agents)

Build the editor brief (see "Brief templates" below) and spawn the
editor sub-agent via the `Task` / `Agent` tool. If animations are
planned, spawn one sub-agent per slot in parallel — see
`references/animations.md`.

The editor sub-agent returns `<edit>/edl.json`. The animation sub-
agents return rendered overlay clips in `<edit>/animations/slot_*/`.

### 7. Export to the NLE

XML-only delivery — there is no flat-MP4 path in this skill anymore.
The cut lives in the NLE and the editor finishes it there.

```bash
python helpers/export_fcpxml.py <edit>/edl.json -o <edit>/cut.fcpxml
```

Default emits BOTH `cut.fcpxml` (Resolve / FCP X) AND `cut.xml`
(Premiere Pro native xmeml) side-by-side from a single timeline build.
Recipient picks whichever NLE they live in. Tell Premiere users to
`File -> Import -> cut.xml`. Override with `--targets {both,fcpxml,
premiere}`. `--frame-rate 24` (default), 25, 29.97, 30, 60.

**The captions sidecar is always emitted.** `export_fcpxml.py` calls
`build_master_srt` automatically as part of every run, writing
`<edit>/master.srt` next to the XML on the OUTPUT timeline (Hard
Rule 5) straight from the cached Parakeet transcripts. The SRT is
Premiere-friendly (UTF-8, CRLF, sequential cues, `HH:MM:SS,mmm -->`
timestamps) and Premiere Pro / DaVinci Resolve / Final Cut Pro X
all import it via `File -> Import` onto a captions track that the
editor restyles in their own caption panel.

If subtitles are explicitly out of scope for a session (rare —
silent-cinema, music video without speech), you can opt out:

```bash
python helpers/export_fcpxml.py <edit>/edl.json -o <edit>/cut.fcpxml --no-srt
```

Or regenerate just the SRT after a hand-tweak to the EDL without
re-walking the timeline:

```bash
python helpers/build_srt.py <edit>/edl.json
```

### 8. Self-eval (before showing the user)

Run `helpers/timeline_view.py` against the **source clips at every
EDL cut boundary** (+/- 1.5s window). XML delivery means there is no
rendered MP4 to inspect — but the cut decisions still need to be sane
on the source side. Check each image for:

- Visual discontinuity / flash / jump at the cut boundary
- Waveform spike at the boundary on the SOURCE side — the NLE will
  honor whatever crossfade you ask it to, but the cut has to land in
  a place where a crossfade can actually save it
- Animation overlay durations match `start_in_output + duration` from
  the EDL (a duration mismatch silently desyncs the overlay)
- Sum of effective range durations matches `total_duration_s` in the
  EDL (catches arithmetic mistakes from the editor sub-agent)

Also sample first 2s, last 2s, and 2-3 mid-points — check shot
selection, animation placement, and overall coherence. The NLE is
where a colorist / editor sees the final result; your job is to
deliver an XML they can drop in without rebuilding the cut.

If anything fails: fix -> re-export -> re-eval. Cap at 3 self-eval
passes — if issues remain after 3, flag them to the user rather than
looping forever. Only hand the XML over once the self-eval passes.

### 9. Iterate + persist — every change request = fresh editor spawn

User feedback comes in. **Do not edit `edl.json` by hand.** Append the
user's verbatim quote to the change-request history in your
Conversation Context bundle, then spawn a fresh editor sub-agent with
an updated brief that includes:

- The full Conversation Context bundle (now including the new quote)
- The full change-request history (chronological, all prior revisions
  + their diffs)
- The prior `edl.json` (so the sub-agent can diff)
- An explicit description of THIS revision: what specifically the user
  asked to change, what to keep

Re-export. Re-self-eval. Show. Loop until user is happy.

Final export on confirmation. Then **append to `project.md`** — see
"project.md memory format" below. This is your only persistent state
across sessions; do not skip it.

---

## Pacing presets — required, asked in step 4

Every session must have a pacing preset (Hard Rule 13). Ask the user
up-front. Default is **Paced**. Each preset expands to four numbers
the editor sub-agent applies when picking cut points.

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
`subagent_editor_rules.md` — you don't need it; you just pick the
preset name + four numbers and forward them in the editor brief.

Persist the choice in `project.md` so subsequent sessions inherit a
sensible default — but still ask if the user wants to keep it.

---

## Brief templates

The parent's main editorial output is briefs. The shape below is load-
bearing — sub-agents need every section to do their job without
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

INPUTS:
  - <edit>/speech_timeline.md
  - <edit>/visual_timeline.md
  Read both end-to-end, in full, in your fresh context window.

OUTPUT:
  Write <edit>/audio_vocab.txt — 200-1000 short labels, one per line,
  lowercase. Include a healthy negative set (~15-20%). Categories per
  subagent_vocab_rules.md.

RETURN:
  A complete report describing:
    - What kind of soundscape you inferred from the speech + visual
      timelines.
    - Categories you covered (tools, materials, ambience, music,
      animals, vehicles, environments, negatives) with counts.
    - Specific labels you debated and why you chose what you chose.
    - Anything ambiguous you want the parent to confirm with the user.
  No artificial length limit; be thorough.
```

### Brief: Editor sub-agent

Spawn this in step 6 for the initial cut, and re-spawn in step 9 for
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
  the merged-view spine read in STEP 1.

  Time-squeeze permission flag (binds the time-squeezing section of
  your operating manual):
    - timelapse_mode = <true | false>
      false  -> emit NO ranges with speed > 1.0; no timelapses.
      true   -> the time-squeezing rules in your operating manual
                apply normally (5-30s output, speed <= 10.0,
                visually-continuous stretches only).

STEP 1:
  Read <edit>/merged_timeline.md END-TO-END. EVERY LINE. (Per the
  ABSOLUTE READ MANDATE.) No first-N-lines, no grep-and-cut, no
  "I have enough." If the file exceeds one Read call, issue
  sequential Reads with offset/limit until every line is covered.
  Same applies to the prior edl.json on revisions.

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

See `references/animations.md`. One sub-agent per slot, all spawned in
parallel via the `Task` / `Agent` tool (Hard Rule 10).

---

## Output spec

Match the source unless the user asked for something specific. Common
targets: `1920x1080@24` cinematic, `1920x1080@30` screen content,
`1080x1920@30` vertical social, `3840x2160@24` 4K cinema,
`1080x1080@30` square. The XML carries a timeline frame rate, not a
canvas resolution — the NLE inherits resolution from the source clips.
Pass `--frame-rate` matching the source (or the user's intended
deliverable) so cuts snap to whole frames. Worth asking the user which
NLE they live in so the timeline rate matches.

---

## EDL format (parent reads sub-agent output)

The editor sub-agent emits this; the parent only needs to validate
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
Color is out of scope — there is no `grade` field; the colorist
owns it end-to-end in the NLE.

If the editor sub-agent returns an EDL with non-zero split-edit fields,
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
re-build the bundle without re-asking the user every preference.

The **Mode flags** block is what persists across sessions for the
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
> (PowerShell, bash, agentic shells that don't persist `cd`), and the
> cache layout assumes the project root is the cwd.

### Phase A — speech + visual (default)

- **`helpers/preprocess_batch.py <videos_dir>`** — auto-discover
  videos, run the speech (Parakeet ONNX) + visual (Florence-2) lanes
  with VRAM-aware scheduling. Default entry point.
  - Flags: `--wealthy` (24GB+ GPU), `--diarize`, `--language en`,
    `--force`, `--skip-speech`, `--skip-visual`.
  - **Do NOT use `--include-audio`.** That flag runs CLAP inline
    against a baked-in fallback vocabulary that exists only for
    `tests.py` smoke-testing. The mandated workflow is: speech +
    visual finish first, then the vocab sub-agent generates a
    project-specific `audio_vocab.txt`, then `audio_lane.py` runs
    against THAT. There is no "skip the vocab sub-agent" shortcut.

- **`helpers/preprocess.py <video1> [<video2> ...]`** — same
  orchestrator with explicit file list. Use when you want a subset.

- **`helpers/pack_timelines.py --edit-dir <dir>`** — read the
  available lane caches (`transcripts/`, `audio_tags/`,
  `visual_caps/`) and produce `merged_timeline.md` (the editor sub-
  agent's default reading surface, all three lanes interleaved by
  timestamp) plus the three per-lane drill-down views:
  `speech_timeline.md`, `audio_timeline.md` (only if Phase B has
  run), `visual_timeline.md`. Pass `--no-merge` to skip the merged
  view (rare). Safe to call multiple times — re-running after Phase
  B picks up the new audio events into both the merged file and
  `audio_timeline.md`.

  **Caveman compression on visual captions is ON by default** — a
  spaCy NLP pass strips stop words / determiners / auxiliaries / weak
  adverbs from every Florence-2 caption before packing, cutting
  `merged_timeline.md` token cost by ~55-60% on detailed-caption
  footage with zero loss of editorial signal (entities, actions,
  colours, shot composition all survive). Cached in
  `<edit>/comp_visual_caps/` keyed by source mtime + caveman version
  + lang; subsequent re-packs are instant.

  Pass `--no-caveman` to read the raw Florence paragraphs (slower,
  bigger, only useful for debugging what Florence actually said).
  `--caveman-lang en` (default) picks the spaCy model;
  `--caveman-procs N` overrides the worker count (default
  `min(n_files, cpu_count // 2)`); `--force-caveman` re-runs even
  cached files.

  Sentence-level fuzzy delta dedup is also applied at pack time:
  visually static frames collapse to `(same)` in
  `visual_timeline.md` and disappear entirely from
  `merged_timeline.md`; slowly-evolving frames emit only the NEW
  sentences with a `+ ` prefix (think `git diff` additions).

- **`helpers/caveman_compress.py`** — standalone CLI for the caveman
  pass. Useful for debugging the compression on a single caption
  (`python helpers/caveman_compress.py "verbose text"`) or for
  manually batching a `visual_caps/` directory (`python
  helpers/caveman_compress.py --visual-caps <edit>/visual_caps/`).
  The pack helper calls it automatically — only use this directly
  when iterating on the filter rules.

### Audio events (CLAP) — agent-curated vocabulary, mandatory

The audio workflow has only one path: **spawn the vocab sub-agent**
(it reads `speech_timeline.md` + `visual_timeline.md` and writes
`<edit>/audio_vocab.txt`), then run `audio_lane.py` against that
vocab, then re-pack timelines. There is no smoke-test / agent-less
fallback in the parent's playbook; the baked-in default vocab in
`audio_vocab_default.py` exists only for `tests.py`. See "Brief
templates" below for the vocab sub-agent brief.

- **`helpers/audio_lane.py <video1> [<video2> ...] --vocab
  <edit>/audio_vocab.txt --edit-dir <edit>`** — run CLAP zero-shot
  scoring against the curated vocabulary. **`--vocab` is mandatory**;
  the lane has a baked-in fallback for `tests.py` smoke testing but
  the parent never relies on it. Caches text embeddings in
  `audio_vocab_embeds.npz` so subsequent runs are fast.
  - Flags: `--device {cuda,cpu}`, `--model-tier {base,large}`,
    `--windows-per-batch N`, `--force`.
  - If `<edit>/audio_vocab.txt` doesn't exist yet, the vocab sub-
    agent has not run yet — go back to step 2 and spawn it.

- After Phase B finishes, **re-run `pack_timelines.py`** to fold the
  new audio events into both `merged_timeline.md` (default) and
  `audio_timeline.md`.

**Individual lanes** (rarely needed — the orchestrator wraps them):
`helpers/parakeet_onnx_lane.py`, `helpers/parakeet_lane.py` (NeMo
fallback), `helpers/audio_lane.py`, `helpers/visual_lane.py`. Each
accepts `--wealthy` and runs standalone.

- **`helpers/extract_audio.py <video>`** — manually extract 16kHz
  mono WAV. Cached. Mainly for debugging.
- **`helpers/vram.py`** — print detected GPU + the schedule that
  would be picked. Useful sanity check.

### Editing

- **`helpers/timeline_view.py <video> <start> <end>`** — filmstrip +
  waveform PNG for visual drill-down. **The editor sub-agent invokes
  this** at decision points; the parent does not. (If a scout sub-
  agent is needed for parent-level questions, the scout calls it.)
  Not a scan tool — use it at decision points, not constantly. The
  `visual_timeline.md` replaces 90% of the old "scan with
  timeline_view" workflow.

- **`helpers/export_fcpxml.py <edl.json> -o cut.fcpxml`** — emit
  editor-ready timeline files. **This is the only delivery path in
  the skill** — there is no flat-MP4 renderer. Hard-cut delivery
  only right now (Hard Rule 14): the EDL's `audio_lead` /
  `video_tail` / `transition_in` fields are still consumed by the
  code path but the editor sub-agent must emit `0` for all three.

  **Default emits BOTH `cut.fcpxml` AND `cut.xml`** side-by-side
  from a single timeline build, because Premiere Pro and Resolve /
  FCP X want different XML dialects: `.fcpxml` (FCPXML 1.10+) is
  native to DaVinci Resolve and Final Cut Pro X, `.xml` (Final Cut
  Pro 7 xmeml) is native to Premiere Pro. The recipient picks
  whichever NLE they live in — no XtoCC conversion required for
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
  on every export; the standalone CLI exists for the case where you
  hand-tweaked the EDL and only want to regenerate the captions
  without rebuilding the XML timeline.

For animations, create `<edit>/animations/slot_<id>/` with `Bash`
and spawn a sub-agent via the `Task` / `Agent` tool per
`references/animations.md`.

---

## Cold-path references — load on demand

Cold-path features the **editor sub-agent** loads only when the
matching mode flag is true in its brief, plus a couple of
parent-side feature references. Read the matching file before
proposing strategy for that feature; do not read pre-emptively.

**Editor cold-path (gated by step-4 questions):**

- `references/scripted.md` — script + voiceover assembly
  procedure, beat segmentation, vo-anchored timing, source-in-point
  synchronisation on named subjects. Editor reads when
  `script_mode = true`.
- `references/b_roll_selection.md` — b-roll selection preference
  order (signage / product / gameplay / booth / stage / people),
  rejection rules, stability bias, diversification, optimized
  matching philosophy (caching / two-stage / clip index). Editor
  reads when `b_roll_mode = true`. Common combo: scripted assembly
  triggers both.

**Sub-sub-agent role spawned by the editor (not by you):**

- `references/subagent_broll_scout_rules.md` — b-roll scout
  sub-agent's operating manual. Spawned by the **editor sub-agent**
  (not by the parent) on demand, one per beat or one per cluster
  of beats, in parallel per Hard Rule 10. Reads
  `<edit>/visual_timeline.md` for in-scope sources and returns
  ranked candidate shortlists. Editor picks / verifies / writes the
  EDL range. The parent never spawns scouts directly — but the
  parent's editor brief mentions `source_tags.json` so the editor
  can pass the in-scope source list down to scouts when it spawns
  them.

**Parent-side cold-path (read when the feature is in scope):**

- `references/subtitles.md` — chunking / case / placement,
  `bold-overlay` and `natural-sentence` worked styles, FCPXML
  delivery and the always-emitted `master.srt` sidecar.
- `references/animations.md` — animation sub-agent brief template,
  PIL / Manim / Remotion timing and easing, parallel-spawn
  discipline (Hard Rule 10).

The Hard Rules that bind these features stay in `shared_rules.md`
— output-timeline SRT (Rule 5) and parallel sub-agents for
animations (Rule 10) are the live ones for XML delivery; the
ffmpeg-pipeline rules (1-4) are dormant since the flat-MP4 path
was removed. Color grade is out of scope — the colorist owns it
in the NLE.

---

## Parent-specific anti-patterns

- **Reading any timeline file.** See shared_rules.md "Agent roles"
  — the parent never opens timeline files. Always re-spawn a sub-
  agent if a question requires timeline knowledge.
- **Editing `edl.json` by hand for "trivial" tweaks.** Always re-spawn.
- **Curating `audio_vocab.txt` by hand.** Always re-spawn.
- **Skipping the pacing prompt.** Hard Rule 13.
- **Skipping the four mode-gating questions in step 4.** They set
  `script_mode`, `b_roll_mode`, `timelapse_mode`, `user_profile` —
  the editor sub-agent's cold-path reads, retime permission, and
  verification bar all depend on them. Default-guessing the flags
  ships the wrong cut.
- **Forgetting to forward the mode flags or `source_tags.json`
  into the editor brief.** Without them the editor falls back to
  non-scripted, non-b-roll, no-tags, default-bar behaviour even
  if the user is on a scripted client deliverable with organized
  folders.
- **Skipping the step-1 folder convention scan.** Even when the
  user only mentions "I have a script and some b-roll" in passing,
  scan the videos_dir for convention folders before asking — the
  user organized for a reason; pre-filling beats re-asking.
- **Forcing folder convention.** Auto-detection is opt-in
  organization. Don't lecture users who use a flat folder layout;
  ask the four questions fresh and move on.
- **Inventing ad-hoc cut-padding numbers.** Pacing preset is the
  contract.
- **Paraphrasing user quotes in the brief.** Quote verbatim. Vibes
  matter.
- **Forgetting to forward the change-request history on revisions.**
  The sub-agent has no memory across spawns; the brief is its memory.
- **Spawning sub-agents sequentially when they could be parallel.**
  Animations especially. Hard Rule 10.
- **Presenting the preview before self-eval passes.** Step 8 is
  mandatory.
- **Skipping the `project.md` append at session end.** That's how
  next session knows where this one stopped.
