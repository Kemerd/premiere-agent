> **Agent: read ONE entry point, not both.** This file (`AGENTS.md`)
> is for **Codex**'s project-guidance loader — Codex concatenates
> `AGENTS.md` files from `~/.codex/` and the repo root into your
> system prompt at session start. A byte-identical copy of this body
> lives at `SKILL.md` (with a YAML `name`/`description` frontmatter on
> top) for **Claude Code**'s skill loader. Whichever runtime loaded
> you already gave you the right one. **Do not read the other** —
> it's the same content and would just burn tokens.

# Video Use Premiere — Entry point

You are reading this because you have just been invoked on a video-use-
premiere session. Welcome.

The skill's operating manual is **split across four rule files** so
that each agent in the hierarchy reads only what binds it — no token
spent on rules that don't apply to the current role. Read the files in
the exact order below; do not skim, do not skip.

## You are the parent agent (orchestrator + conversation manager)

If you were spawned as a sub-agent, your spawn prompt told you so
explicitly and pointed you at the right rule files for your role. If
no one told you which role you are, **you are the parent.**

Read these two files now, in this order, before doing anything else:

1. **`references/shared_rules.md`** — universal rules: the Principle,
   the Agent Roles section (defines the boundary between you and the
   sub-agents you will spawn), the numbered Hard Rules block, and
   universal anti-patterns. Binds every agent in the hierarchy.

2. **`references/parent_rules.md`** — your specific operating manual:
   directory layout, setup checklist, skill health check workflow,
   the 9-step process you run, the helper scripts (`ffprobe`,
   `health.py`, `preprocess_batch.py`, `pack_timelines.py`,
   `audio_lane.py`, `export_fcpxml.py`, `build_srt.py`, etc.), pacing
   presets, brief templates for spawning sub-agents, EDL format,
   `project.md` memory format, and parent-specific anti-patterns.

After those two files you have everything you need to run a session.

## Sub-agent rule files (you do NOT read these — sub-agents do)

You point spawned sub-agents at these via their briefs (templates in
`parent_rules.md`):

- **`references/subagent_editor_rules.md`** — editor sub-agent's
  manual. Spawned to read `merged_timeline.md` and produce
  `edl.json`. Re-spawned for every user-requested change.

- **`references/subagent_vocab_rules.md`** — vocab sub-agent's
  manual. Spawned once after Phase A speech + visual finishes, to
  read `speech_timeline.md` + `visual_timeline.md` and produce a
  project-specific `audio_vocab.txt`. Mandatory step — there is no
  baseline-vocab shortcut in the parent's playbook.

- **`references/animations.md`** — animation sub-agent's manual.
 One sub-agent spawned per animation slot, all in parallel
 (Hard Rule 10).

- **`references/subagent_broll_scout_rules.md`** — b-roll scout
 sub-agent's manual. **Spawned by the editor sub-agent**, not by
 you (the parent). Sub-sub-agents — they read
 `<edit>/visual_timeline.md` for in-scope sources only and return
 ranked b-roll candidate shortlists. The editor decides whether to
 delegate per-beat shortlisting based on library size,
 `user_profile`, and beat count. Architectural ceiling stays at
 two levels below parent: `parent -> editor -> scout`.

## Cold-path features (load on demand when the user asks for them)

- **`references/subtitles.md`** — chunking / case / placement
 reasoning, the `bold-overlay` and `natural-sentence` worked
 styles, FCPXML delivery (ship `master.srt` alongside via
 `helpers/build_srt.py`).

- **`references/scripted.md`** — script + voiceover assembly
 procedure (beat segmentation, vo-anchored timing, source-in-point
 sync on named subjects). The **editor sub-agent** loads this when
 the parent's brief says `script_mode = true`. The parent gates
 this in step 4 of the 9-step process by asking "are you using a
 script?"

- **`references/b_roll_selection.md`** — b-roll selection
 preferences (signage / product / gameplay / booth / stage / people),
 rejection rules, stability bias, diversification, optimized
 matching (caching / two-stage / clip index). The **editor sub-
 agent** loads this when the parent's brief says
 `b_roll_mode = true`. Common combo: scripted assembly triggers
 both.

The parent collects four flags in step 4: `script_mode`,
`b_roll_mode`, `timelapse_mode` (default `false` — explicit opt-in
to retime / time-squeeze, otherwise the editor stays at 1x), and
`user_profile` (`personal | creator | professional` — sets the
editor's verification bar). All four are persisted in
`<edit>/project.md` so subsequent sessions inherit defaults.

The parent also runs a **folder convention auto-detection** in
step 1: case-insensitive subfolders like `b_roll/`, `timelapse/`,
`voiceover/`, `a_roll/` pre-fill the step-4 mode-gating defaults
and write `<edit>/source_tags.json` mapping clip stems to
categories. Sub-agents respect these tags when shortlisting.

Step 1 also runs **paired-audio detection** for dual-mic rigs — a
video and audio file with the same stem (`SHOT_0042.mp4` +
`SHOT_0042.wav`, the universal Sony / Zoom / DJI / Tascam camera +
recorder convention). When pairs are detected the parent ASKS the
user whether the .wav is a second-mic recording (`dual_mic`, both
transcribed, editor picks the cleaner per cut) or a backup file
(`ignore`, the .wav is filtered out). The decision is recorded in
`<edit>/source_pairs.json` and travels into every editor brief.
The pipeline also accepts standalone audio-only files (e.g.
voiceover .wav with no video sibling) as first-class sources —
they get the speech lane, skip the visual lane.

See `parent_rules.md` for the exact question templates and the
folder-convention / pair-mode tables.

## What this architecture buys you

- **Token economy.** Token-heavy timeline reads happen in fresh sub-
  agent context windows, never in the parent's accumulating one.
  Long iteration sessions (revision 5+) cost the same per spawn as
  revision 1 — the parent's context grows linearly with conversation,
  not with file size.

- **Specialization.** Each sub-agent reads only what binds it, so
  the editor isn't paying tokens on vocab curation guidance and the
  vocab sub-agent isn't paying tokens on FCPXML internals.

- **Reproducibility.** A change request becomes a brief diff, not
  a hand-edit on the EDL. Every revision is a fresh spawn with the
  full Conversation Context bundle and the chronological change-
  request history forwarded by the parent.

- **The parent stays light.** Its job is `listen -> summarize ->
  quote -> dispatch -> translate -> run scripts -> handle errors`.
  Conversation, orchestration, helper-script execution, error
  handling, filesystem management — all parent work. Timeline
  reading and cut decisions — sub-agent work. The boundary is
  load-bearing; do not cross it.

Now go read `references/shared_rules.md` first, then
`references/parent_rules.md`. After that, you're ready to start the
session.
