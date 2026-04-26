# Shared Rules — every agent in this skill reads this first

Every agent in `video-use-premiere` — the parent conversation manager AND every sub-agent it spawns (editor, vocab, animations) — reads this file as step 0. Rules here are **universal**: they bind production correctness, the agent hierarchy, and the philosophy that keeps the skill working without silent failures.

After reading this file, continue to your role-specific rules:

- Parent agent: `references/parent_rules.md`
- Editor sub-agent: `references/subagent_editor_rules.md`
- Vocab sub-agent: `references/subagent_vocab_rules.md`
- Animation sub-agent: `references/animations.md`
- B-roll scout sub-sub-agent: `references/subagent_broll_scout_rules.md` (spawned by the editor sub-agent, not by the parent)

If you do not know your role, you are the parent. Subagents are told their role explicitly by the parent (or, for scouts, by the editor) in their spawn brief.

**Cold-path references** — feature-specific rules loaded on demand, not at session start:

- `references/subtitles.md` — caption styling, chunking, FCPXML delivery (parent reads when subtitle styling is in scope).
- `references/animations.md` — animation sub-agent brief template (also the animation sub-agent's own operating manual).
- `references/scripted.md` — script + voiceover assembly procedure (editor reads when the parent's brief says `script_mode = true`).
- `references/b_roll_selection.md` — b-roll selection rules + optimized matching philosophy (editor reads when the parent's brief says `b_roll_mode = true`).
- `references/subagent_broll_scout_rules.md` — b-roll scout sub-agent's full operating manual (read by scouts at spawn time; the editor reads it once when writing a scout brief).

The two editor cold-path files are gated by mode flags the parent collects in step 4 of the 9-step process — see `parent_rules.md` for the question templates and `subagent_editor_rules.md` for the conditional-read mandate. The scout role file is read by scouts on spawn (their own STEP 0) and consulted by the editor for scout briefs.

---

## Principle

1. **LLM reasons from two mandatory markdown views + on-demand drill-down.** The editor and vocab sub-agents read TWO files end-to-end, line by line, every spawn: `audiovisual_timeline.md` (audio events + visual captions interleaved by timestamp) and `speech_timeline.md` (phrase-grouped transcripts with outer-aligned `floor(start)..ceil(end)` integer ranges). Speech is INTENTIONALLY split out of the AV file — phrase-long blocks rendered against per-second visual captions made the merged view harder to scan without buying anything (the editor still had to cross-reference `speech_timeline.md` for accurate spans). Two sibling files, both mandatory, no ambiguity. The remaining per-lane views (`audio_timeline.md`, `visual_timeline.md`) are kept on disk as drill-down references for the moments where the AV view is ambiguous and the editor zooms in on one lane. Everything else — filler tagging, retake detection, shot classification, B-roll spotting, emphasis scoring — is derived at decision time.

2. **Speech is primary, visuals are secondary, audio events are tertiary.** Cut candidates come from Parakeet ONNX speech boundaries and silence gaps — that lane is highly accurate, the editorial spine. Visual captions (Florence-2) are the second source of truth: they answer "what's actually on screen here?" and resolve ambiguous decision points (B-roll spotting, shot continuity, action beats). Audio events (CLAP, zero-shot scoring against an agent-curated vocabulary) tag non-speech sounds per ~10s window. When audio and visual disagree about *what is happening on screen*, **trust the visual lane.**

3. **Ask -> confirm -> execute -> iterate -> persist.** No agent ever touches a cut, a vocabulary, or a render until the user has confirmed the strategy in plain English. The parent handles the conversation; subagents only run after the parent has dispatched them.

4. **Generalize.** Do not assume what kind of video this is. Look at the material, ask the user, then edit.

5. **Artistic freedom is the default.** Every specific value, preset, font, color, duration, pitch structure, and technique in this skill is a *worked example* from one proven video — not a mandate. Read them to understand what's possible and why each worked. Then make taste calls based on what the material is and what the user wants. **The only things you MUST do are in the Hard Rules block below.** Everything else is yours.

6. **Invent freely.** If the material calls for a technique not described in the role files — split-screen, picture-in-picture, lower-third identity cards, reaction cuts, freeze frames, match cuts, speed ramps over breath, whatever — build it. The helpers are ffmpeg and PIL; the FCPXML exporter handles hard-cut delivery to NLEs. They can do anything the format supports. Do not wait for permission. (Note: J-cuts, L-cuts, and cross-dissolves are currently DEFERRED — see Hard Rule 14.)

7. **Verify your own output before showing it upstream.** If you wouldn't ship it, don't return it. The editor sub-agent verifies its own EDL against word-boundary discipline before returning. The vocab sub-agent verifies its own vocabulary covers the project's actual content before returning. The parent verifies the renderer's output against the self-eval checklist before showing the user.

---

## Agent roles (this binds harder than the Hard Rules below)

Three kinds of agent make this skill work. **They are not interchangeable.** Token economy AND quality of output both depend on the boundary holding. The parent agent must NEVER do a sub-agent's job, and a sub-agent must never assume facts the parent didn't put in its brief.

### Parent agent — conversation manager, content awareness via speech only

- **Talks to the user.** Asks questions, confirms strategy, presents previews, takes feedback. This is the parent's only persistent state.
- **Reads `project.md`** at session start (compressed memory from prior sessions). The parent's most-frequent direct read.
- **MAY read `speech_timeline.md` directly** to ground itself in what was actually said. This is the only timeline file the parent is allowed to open. Speech is pure text (Parakeet phrase-level transcripts with timestamps) — token-cheap, conversationally useful, and lets the parent quote-match the user's must-keeps / must-cuts against real transcript moments without a scout spawn. Read it once after step 1's first pack to seed the Conversation Context bundle, and again on demand when a user message references something specific the speakers said. Do not loop over it constantly — it is context, not the cut.
- **Maintains a Conversation Context bundle** in its own working memory throughout the session:
    - A *complete* summary of the project in the parent's own words — what kind of video, who's in it, what it's for, target deliverable, aesthetic direction, must-keeps, must-cuts. **Comprehensive. No artificial word cap.** Length is whatever it needs to be.
    - A *list of verbatim user quotes*, chronological, capturing the original wording of every meaningful request, rejection, or preference. Direct quotes preserve nuance the parent cannot reliably paraphrase ("make it punchy" != "make it tight" != "cut it shorter"). When in doubt, quote. Tag each quote with a short context note (when in the session, what it responded to).
    - Things the user *explicitly rejected* — also as quotes with context.
    - Things the user *explicitly asked to keep* — also as quotes with context.
- **Writes briefs. Spawns subagents.** The Conversation Context bundle is forwarded verbatim into every sub-agent brief — so iterative subagents have memory across spawns.
- **Reads sub-agent return values** (EDL JSON, vocab.txt, animation slot reports) and translates them to the user in plain English.
- **NEVER reads `audiovisual_timeline.md`, `visual_timeline.md`, or `audio_timeline.md` directly.** Those are the token-heavy lanes — Florence-2 visual captions at 1fps, CLAP audio events per ~10s window, and the AV view that interleaves them. They exist for sub-agents to read in fresh context windows so the parent's accumulating context never bloats with caption density. If the user asks "what's actually on screen in C0103?" or "what does the audio sound like at 0:42?", the parent spawns a scout or asks the editor sub-agent — never opens `audiovisual_timeline.md` / `visual_timeline.md` / `audio_timeline.md` itself. (Speech is the exception per the bullet above — text transcripts only, parent-readable.)
- **NEVER edits `edl.json` by hand**, even for a "trivial" tweak the user asks for. Every change request, no matter how small, re-spawns the editor sub-agent with an updated brief. Hand-edits skip the word-boundary / pacing / filler discipline the editor's role rules enforce.
- **NEVER curates `audio_vocab.txt` by hand.** Vocab work belongs to the vocab sub-agent.
- **NEVER opens the source video files.** Not even with `ffprobe`-via-eyeball-of-a-frame. That is a sub-agent's job if needed at all.

### Vocab sub-agent — runs once per Phase B (and again after a vocab miss)

- Spawned by the parent in step 2 of the process.
- Reads BOTH `audiovisual_timeline.md` AND `speech_timeline.md` end-to-end, line by line, with its own fresh context window. At vocab time the AV file carries only the visual lane (audio hasn't scored yet — that's exactly what this sub-agent's vocab enables); the speech file carries the editorial spine. Drilling into `audio_timeline.md` / `visual_timeline.md` is allowed only at ambiguous moments, never as a substitute for the mandatory dual end-to-end read.
- Receives the parent's Conversation Context bundle in its brief so the vocabulary matches what the user said the video is.
- Writes `<edit>/audio_vocab.txt` — 200-1000 short labels, project-specific, with a healthy negative set.
- Returns a complete report of what it inferred about the soundscape, the chosen vocabulary categories, and any ambiguous calls — no artificial length limit. The parent forwards salient bits to the user.

### Editor sub-agent — runs once per cut decision (every revision = a fresh spawn)

- Spawned by the parent in step 6 for the initial cut, and re-spawned in step 9 for every user-requested change without exception.
- Reads BOTH `audiovisual_timeline.md` AND `speech_timeline.md` end-to-end, line by line (per the ABSOLUTE READ MANDATE in `subagent_editor_rules.md` and the spine principle of Hard Rule 15) in its own fresh context window. Drills into the remaining per-lane files (`audio_timeline.md`, `visual_timeline.md`) only at ambiguous moments. For word-precise quote → range lookup, calls `helpers/find_quote.py` against `transcripts/<stem>.json` rather than re-reading the JSON in full.
- Receives from the parent in its brief:
    1. The full Conversation Context bundle (project summary + verbatim user quotes + rejections + must-keeps).
    2. Strategy: beats / structure, pacing preset values, target runtime, delivery target.
    3. Change-request history — chronological list of every cut revision the user has asked for so far, each as a verbatim quote with what was changed in the prior EDL in response.
    4. On revisions: the prior `edl.json` plus the user's specific change request quote for THIS revision.
- Writes `<edit>/edl.json`. Returns a one-line runtime check and a short rationale per beat.

### Animation sub-agents — one per slot, all spawned in parallel

- Spawned by the parent in step 6 alongside the editor sub-agent.
- See `references/animations.md` for the full brief template.

### B-roll scout sub-agents — spawned by the EDITOR (sub-sub-agents)

- Spawned by the **editor sub-agent**, not by the parent. The editor decides whether to delegate per-beat shortlisting based on library size, `user_profile`, and beat count — see `subagent_editor_rules.md` "B-roll scout spawn protocol".
- One scout per beat (or one per cluster of beats), all spawned in parallel per Hard Rule 10.
- Reads `<edit>/visual_timeline.md` for in-scope sources only, in its own fresh context window. Returns ranked candidate shortlists with evidence; the editor picks / verifies / writes the EDL range.
- See `references/subagent_broll_scout_rules.md` for the full brief template + return format.

### Architectural ceiling — two levels of sub-agents below the parent

`parent → editor → b-roll scout` is the deepest the hierarchy goes. Scouts do not spawn sub-sub-sub-agents. If a scout exhausts its context, it returns BUDGET_EXHAUSTED and the editor re-shapes the brief — never deeper recursion.

### What this means in practice

- A long iteration session does NOT bloat the parent's context with the heavy timeline data (visual captions, audio events, AV view), because the parent never reads those lanes. Speech transcripts the parent does read are text-only and token-cheap. The parent's context grows linearly with conversation, not with caption density.
- A sub-agent spawned on revision #5 has the same fresh window as one spawned on revision #1, plus the full change history in its brief. It is NOT operating with less context — it has curated context plus a clean read budget.
- The parent's job is: **listen -> summarize -> quote -> dispatch -> translate.** Nothing else. If you catch yourself reading a timeline file as the parent, stop — write the question into a sub-agent brief instead.

---

## Hard Rules (production correctness — non-negotiable)

These are where deviation produces silent failures or broken output. They are not taste, they are correctness. Memorize them.

1. **Subtitles are applied LAST in the filter chain**, after every overlay. Otherwise overlays hide captions. Silent failure.

2. **Per-segment extract -> lossless `-c copy` concat**, not single-pass filtergraph. Otherwise every segment is double-encoded when overlays are added.

3. **30ms audio fades at every segment boundary** (`afade=t=in:st=0:d=0.03,afade=t=out:st={dur-0.03}:d=0.03`). Otherwise audible pops at every cut.

4. **Overlays use `setpts=PTS-STARTPTS+T/TB`** to shift the overlay's frame 0 to its window start. Otherwise the middle of the animation shows during the overlay window.

5. **Master SRT uses output-timeline offsets**: `output_time = word.start - segment_start + segment_offset`. Otherwise captions misalign after segment concat.

6. **Never cut inside a word.** Snap every cut edge to a word boundary from the Parakeet word-level transcript.

7. **Pad every cut edge.** Working window: 30-200ms. ASR timestamps drift 50-100ms — padding absorbs the drift. Tighter for fast-paced, looser for cinematic.

8. **Word-level verbatim ASR only.** Parakeet TDT emits per-token timestamps natively — keep them; never collapse to phrase / SRT shape on the lane output (that loses sub-second gap data). Never normalize fillers either (loses editorial signal — the editor sub-agent uses `umm` / `uh` / false starts to find candidate cuts).

9. **Cache lane outputs per source.** Never re-run a lane unless the source file itself changed (mtime). The orchestrator handles this; do not pass `--force` reflexively.

10. **Parallel subagents for multiple animations.** Never sequential. Spawn N at once via the `Agent` / `Task` tool; total wall time approx slowest one.

11. **Strategy confirmation before execution.** No sub-agent runs until the user has approved the plain-English plan via the parent.

12. **All session outputs in `<videos_dir>/edit/`.** Never write inside the `video-use-premiere/` project directory.

13. **Pacing preset is REQUIRED before strategy.** Every session must have a pacing preset confirmed by the user (Calm / Measured / Paced / Energetic / Jumpy — default Paced). The preset defines four numbers used by the editor sub-agent: `min_silence_to_remove`, `min_talk_to_keep`, `lead_margin`, and `trail_margin`. See `parent_rules.md` for the value table. Never skip the prompt; never invent ad-hoc values.

14. **No split edits (J/L cuts) and no cross-dissolves until further notice.** The editor sub-agent MUST emit `audio_lead = video_tail = transition_in = 0` on every range. They are deferred because the OTIO single-track audio model + per-clip independent frame-snapping causes cumulative audio drift across long timelines (visible as the audio sliding further out of sync with each subsequent cut). Audio at cut boundaries is protected by the 30ms `afade` pair from Hard Rule 3 — the current "small crossfade" story.

15. **TWO mandatory reading surfaces (`audiovisual_timeline.md` + `speech_timeline.md`) are the spine for the editor and the vocab sub-agents — never reason from one alone, never reason from a single lane in isolation.** Cut decisions (editor) and vocabulary curation (vocab) both read BOTH files end-to-end, line by line, so every lane visible at that step informs the output. For the editor at step 6 the AV file carries audio + visual interleaved (visuals for shot continuity / B-roll, audio events as soundscape hints) and the speech file carries the editorial spine (phrase ranges, outer-aligned integer rounding so any range maps cleanly back into `transcripts/<stem>.json`). For the vocab sub-agent at step 2 the AV file carries only the visual lane (audio hasn't scored yet — that's exactly what this sub-agent's vocab enables) and the speech file again carries the spine; the per-clip `## <stem>` headers are aligned across both files so the two views can be scrolled in parallel. A cut chosen blind to the other file lands mid-shot, mid-action, or on a CLAP mis-label; a vocabulary curated blind to the visuals misses every silently-shown sound source. Drill into the remaining per-lane files (`audio_timeline.md`, `visual_timeline.md`) only at ambiguous moments. The mandatory full-coverage read discipline for each sub-agent lives in its own rules file (`subagent_editor_rules.md` Pre-flight, `subagent_vocab_rules.md` Pre-flight) — those files are authoritative for what "read in full" means inside each role.

16. **`pack_timelines.py` runs TWICE per session — once before vocab, once after audio.** The parent's first pack (after Phase A preprocess) produces `speech_timeline.md` plus an `audiovisual_timeline.md` carrying only the visual lane; the vocab sub-agent reads BOTH files end-to-end as its dual spine. The parent's second pack (after `audio_lane.py` finishes Phase B) folds the freshly-scored audio events into the same `audiovisual_timeline.md` so the editor sub-agent in step 6 sees audio + visual interleaved alongside the speech file. Skipping the second pack ships the editor a stale AV view with no `(audio: …)` lines and silently breaks Hard Rule 15. Helpers and per-lane caches (`transcripts/`, `audio_tags/`, `visual_caps/`) are reused across both passes — the cost of the second pack is small; the cost of skipping it is editorial blindness on the audio lane.

Everything outside the Hard Rules block is taste. Deviate whenever the material calls for it.

---

## Universal anti-patterns

These fail regardless of role:

- **Hierarchical pre-computed codec formats** with USABILITY / tone tags / shot layers. Over-engineering. Derive from the timelines at decision time.
- **Hand-tuned moment-scoring functions.** The LLM picks better than any heuristic you can write.
- **SRT / phrase-level lane output.** Loses sub-second gap data. Always word-level verbatim from the speech lane.
- **Re-running `helpers/preprocess_batch.py --force` reflexively.** The mtime-based cache is correct; bypass only when the source changed or a model was upgraded.
- **Reading `transcripts/*.json` directly for general timeline scanning.** Use the timeline markdowns; same data, 1/10 the tokens, phrase-aligned. The carve-out: the editor sub-agent reads `transcripts/<stem>.json` *surgically* at cut-verification time — the markdown views drop per-word boundaries (phrase-grouped concatenation) so they cannot satisfy Hard Rule 6 alone. See `subagent_editor_rules.md` "Word-boundary verification".
- **Editing before confirming the strategy.** Never.
- **Re-preprocessing cached sources.** Immutable outputs of immutable inputs.
- **Assuming what kind of video it is.** Look first, ask second, edit last.
- **Sub-agent reading from one timeline file when both are required.** `audiovisual_timeline.md` AND `speech_timeline.md` are BOTH the default reading surfaces for the editor and the vocab sub-agents (Hard Rule 15). Per-lane files are drill-down only; reading just one of the two mandatory files is the same correctness violation as reading neither.
- **Skipping the second `pack_timelines.py` run after `audio_lane.py`.** Hard Rule 16. The pack runs twice by design; the second pass is the only way audio events reach the AV view the editor reads.
- **Parent agent reading `audiovisual_timeline.md`, `visual_timeline.md`, or `audio_timeline.md`.** See "Agent roles" above — the heavy timeline reads happen in fresh sub-agent windows, not the parent's accumulating one. Reading `speech_timeline.md` for conversation-side content awareness is fine (speech is text-only and token-cheap); reading the visual / audio / AV views is not.
