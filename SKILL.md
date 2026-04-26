---
name: video-use-premiere
description: Edit any video by conversation. Local two-phase preprocessing — Phase A runs Parakeet ONNX speech + Florence-2 visual captions in parallel; Phase B runs CLAP zero-shot audio events against an agent-curated vocabulary derived from the speech + visual timelines. Cut, color grade, generate overlay animations, burn subtitles, OR export FCPXML to Premiere/Resolve/FCP with split edits. For talking heads, montages, tutorials, travel, interviews, workshop / shop footage. No presets, no menus, no cloud transcription. Ask questions, confirm the plan, execute, iterate, persist. Production-correctness rules are hard; everything else is artistic freedom.
---

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
   `audio_lane.py`, `render.py`, `export_fcpxml.py`, etc.), pacing
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

## Cold-path features (load on demand when the user asks for them)

- **`references/color-grade.md`** — ASC CDL mental model, shipped
  filter chain presets, FCPXML "don't bake the grade" rule.

- **`references/subtitles.md`** — chunking / case / placement
  reasoning, the `bold-overlay` and `natural-sentence` worked
  styles, FCPXML delivery (ship `master.srt` alongside, don't burn).

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
