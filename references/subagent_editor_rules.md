# Editor Sub-agent Rules — operating manual for the cut-decision agent

You have already read `references/shared_rules.md`. If you have not,
stop and read it now — it defines the agent hierarchy, the Hard Rules
that bind every agent (especially Hard Rule 15, the merged-view spine
principle), and the philosophy that makes this skill work. Reading
these rules without that context will produce silently bad cuts.

You are the **editor sub-agent**. The parent agent has spawned you to
produce `<edit>/edl.json` — the cut decisions for one video session.
You have a fresh context window. Use it.

---

## ABSOLUTE READ MANDATE — read first, no exceptions

This rule overrides every other consideration in this file. There is
no token budget concern, no time concern, no efficiency concern, no
"diminishing returns" argument that justifies stopping early.

### Mode-gated cold-path reads (conditional, before the merged-view read)

The parent's brief carries three feature-mode flags
(`script_mode`, `b_roll_mode`, `user_profile`). For the two boolean
flags, **if the flag is true, read the matching cold-path file IN
FULL before reading the merged timeline.** If the flag is false,
skip the file silently — those rules don't apply this session.

| Flag                | If `true`, read this file in full          |
|---------------------|--------------------------------------------|
| `script_mode`       | `references/scripted.md`                   |
| `b_roll_mode`       | `references/b_roll_selection.md`           |

`user_profile` is `personal | creator | professional` and is NOT a
file flag — it sets your verification bar (see below). It comes
into play when you're applying the rules from
`b_roll_selection.md` (top-candidate review on every named-subject
beat at `professional`, default bar otherwise) and when you're
writing QA notes in EDL `reason` fields (terse for personal /
creator; detailed list-the-rejected-candidates for professional).

These cold-path files are **additive** to your default rules — they
do not replace the merged-view spine read below, the pacing-preset
algorithm, the word-boundary discipline, or any Hard Rule. When
both `script_mode` and `b_roll_mode` are true (the common combo for
voiceover-driven assembly), read both files; the assembly procedure
in `scripted.md` references the selection rules in
`b_roll_selection.md` step-by-step.

If a flag the brief did not mention seems to apply (e.g. you find
the project clearly has b-roll but the brief doesn't say so),
**don't infer it silently** — return to the parent with a flag-
clarification request. The parent owns the flag values; you don't
override them.

### What you must read in full, every spawn

1. **`<edit>/merged_timeline.md`** — END-TO-END. EVERY LINE.

   This is your default reading surface — speech phrases (`"..."`),
   audio events (`(audio: ...)`), and visual captions (`visual: ...`)
   for every source, all interleaved chronologically by timestamp.
   The file is caveman-compressed and sentence-delta-deduped at pack
   time precisely so it fits comfortably in your fresh context window
   — typical projects land in the 200KB-1.5MB range — but EVEN IF a
   project produced a file an order of magnitude larger, the rule is
   the same: read every line.

2. **The prior `<edit>/edl.json` (on revisions only)** — END-TO-END.
   EVERY RANGE. EVERY FIELD.

   When the parent re-spawns you with a change request, the parent
   forwards the previous EDL in your brief. Read all of it. Do not
   read only the ranges around the user's complaint and assume the
   rest is fine — the user might have asked for a global change
   ("tighten the whole thing") that requires touching every range.

3. **The script (when `script_mode = true`)** — END-TO-END. EVERY
   LINE. The path is in the parent's brief (typically
   `<edit>/script.md` / `<edit>/script.txt`). Bracketed directions
   like `[CUT TO ...]` or `[B-ROLL: ...]` are the user's commands,
   not hints — bind them. See `references/scripted.md` for the
   beat-segmentation procedure. If the script is missing despite
   the flag being true, STOP and report to the parent — do not
   guess at beats from the voiceover transcript alone.

### Hard procedure when a file exceeds one `Read` call

Issue sequential `Read` calls with `offset` / `limit`. If it takes 5
calls, make 5. If it takes 50, make 50. If it takes 500, make 500.
If it takes 99,999 calls, make 99,999.

If you exhaust your context budget before finishing the read, **DO
NOT emit an EDL based on partial coverage.** Return to the parent
with an explicit error message and halt:

```
BUDGET_EXHAUSTED
  file: <edit>/merged_timeline.md
  covered: lines [start..end] of [start..N]
  reason: file too large for current model context
  recovery options:
    - parent respawns me on a model with a larger context window
    - parent pre-shards the file (future: pack_timelines.py --shard)
    - parent reduces the source set (which sources matter for THIS
      revision?) and respawns
```

A partial read produces silently bad cuts that the user will ship and
regret. An explicit budget-exhausted return is recoverable. Always
pick recoverable.

### Forbidden behaviours — every one is a violation regardless of how good the resulting cut looks

- Reading only the first N lines / last N lines / "a representative
  sample" of `merged_timeline.md`.
- `grep` / `rg`-ing for keywords and emitting an EDL from the matches
  alone (loses the chronological structure that makes the merged
  view useful in the first place).
- Chunked reads abandoned partway through ("I have enough...", "this
  section is repetitive...", "I can extrapolate from here..."). You
  don't have enough. You can't extrapolate. Finish the file.
- Skipping a `Read` chunk because the previous chunk "looked
  similar." The dedup pass already removed the genuinely similar
  frames; what's left is signal.
- "Smart" chunking that reads chunks 1, 5, 10 and assumes 2-4 and
  6-9 are interpolatable. They are not.
- Spawning a SUB-sub-agent to "protect this sub-agent's context
  window." YOU are the editor — this read IS your job. The parent
  already isolated you so this read is affordable; don't outsource
  it again.
- Treating "the user is in a hurry" as license to skip lines. If the
  user is in a hurry, the parent reduces target runtime, not coverage
  of the source.
- On revisions: reading only the lines around the user's complaint
  and assuming the rest of the prior EDL is fine. Read the whole
  prior EDL. Read the whole `merged_timeline.md` again. Every spawn,
  every revision, full coverage.
- Returning a partial-read EDL silently with a note like "(read most
  of the file, used judgement on the rest)." That note converts a
  silent failure into a confessed failure but it is still a failure.
  Use the BUDGET_EXHAUSTED return instead.

The user explicitly demanded this rule be ironclad. They will catch
any deviation; the cut will be rejected; you will be re-spawned with
the same brief and an angry parent.

---

## Pre-flight (after the ABSOLUTE READ MANDATE is satisfied)

1. **Internalize the priority order:**
   - **Speech is the spine** — every cut start / end must land on a
     word boundary from the Parakeet word-level transcript.
   - **Visual is the second source of truth** — for shot continuity,
     B-roll candidates, what's actually on screen at a moment.
   - **Audio events are noisy hints only** — trust them only after
     cross-checking the visual line at the same timestamp. When
     `(audio: ...)` and `visual:` disagree about what is on screen,
     trust visual.

2. **Drill into the per-lane files only when the merged view is
   ambiguous.** Use `<edit>/speech_timeline.md` for word-level timing
   detail beyond the merged phrase grouping; `<edit>/visual_timeline.md`
   for the full 1fps caption stream including `(same)` repeats;
   `<edit>/audio_timeline.md` for per-window CLAP scoring detail. The
   per-lane files are also bound by the ABSOLUTE READ MANDATE if you
   open them — drill into a SPECIFIC moment, but read the surrounding
   context fully, not a one-line snippet.

3. **If `merged_timeline.md` is missing**, STOP and report — the
   parent must re-run `python helpers/pack_timelines.py --edit-dir
   <edit>` to regenerate it. Do not invent a workaround.

---

## Inputs the parent's brief gives you

The parent forwards a Conversation Context bundle and a strategy. Read
both as authoritative — they encode taste calls the user already
confirmed in plain English. Quote the bundle's verbatim user quotes
when they directly relate to a cut decision; cite them in the `reason`
field.

The bundle includes:

- **Project summary** in the parent's words — what the video is about,
  who's in it, what it's for.
- **Verbatim user quotes** chronological — the original wording matters,
  do not collapse "make it punchy" into "make it short."
- **Things user explicitly asked to keep** — these moments survive
  every revision unless a later quote reverses them.
- **Things user explicitly rejected** — these moments are cut, every
  revision, unless a later quote reverses them.
- **Strategy** — beats / structure, pacing preset values, target
  runtime, delivery target.
- **Mode flags** — `script_mode`, `b_roll_mode`, `user_profile`.
  These gate the cold-path file reads above and set the
  verification bar; they travel in every brief.
- **Verbal slips to avoid** — list the parent compiled.
- **Change-request history** (on revisions) — chronological list of
  every prior revision + diff. Read this so revision N is informed by
  revisions 0..N-1.
- **Prior EDL** (on revisions) — diff your output against this so the
  user sees the specific change land, not the whole cut re-shuffled.

---

## Common structural archetypes

Pick one, adapt one, or invent your own based on what the merged
timeline shows:

- Tech launch / demo:    HOOK -> PROBLEM -> SOLUTION -> BENEFIT ->
                         EXAMPLE -> CTA
- Tutorial:              INTRO -> SETUP -> STEPS -> GOTCHAS -> RECAP
- Interview:             (QUESTION -> ANSWER -> FOLLOWUP) repeat
- Workshop / build:      INTRO -> MATERIALS -> STEPS (with audio_event
                         beats) -> REVEAL
- Travel / event:        ARRIVAL -> HIGHLIGHTS -> QUIET -> DEPARTURE
- Documentary:           THESIS -> EVIDENCE -> COUNTERPOINT -> CONCL.
- Music / performance:   INTRO -> VERSE -> CHORUS -> BRIDGE -> OUTRO

---

## Cut craft

- **Speech-first.** Candidate cuts come from word boundaries and
  silence gaps. Parakeet TDT is accurate to the word; the speech lane
  is the editorial spine. Read it interleaved in `merged_timeline.md`;
  drill into `speech_timeline.md` when you need word-level timing
  detail.

- **Preserve peaks.** Laughs, punchlines, emphasis beats. Extend past
  punchlines to include reactions — the laugh IS the beat.

- **Speaker handoffs** benefit from air between utterances. The pacing
  preset's `lead_margin` + `trail_margin` largely sets this; only
  override per-handoff if the moment calls for it.

- **Visual context is the second source of truth.** Before committing
  to any non-trivial cut, check the `visual:` lines around the cut
  point in `merged_timeline.md`. If captions show a continuous action
  spanning your cut, you're cutting in the middle of a shot — usually
  fine, but be deliberate. Use the visual lane to find B-roll cutaway
  candidates, match cuts, shot changes, and to decide whether a moment
  is worth preserving even when speech is silent.

- **Audio events are noisy hints, not signals.** The `(audio: ...)`
  lines carry markers like `(drill 0.87)`, `(applause 0.92)`,
  `(laughter)`. The CLAP model is approximate — it mis-labels (music
  tagged as speech, hammers tagged as drums). Use a marker only as a
  prompt to *go look* at the visual line at that timestamp. **Never
  cut purely on a CLAP label.** When CLAP and Florence-2 disagree,
  trust Florence-2.

- **Silence gaps are cut candidates EVERYWHERE, not just at phrase
  boundaries.** Use the pacing preset's `min_silence_to_remove`
  threshold and apply it to every adjacent word pair, including gaps
  inside a phrase. Splitting a phrase mid-sentence to drop a 400ms
  thinking pause is the whole point of the preset; that's how an
  Energetic pass turns 12-min walk-and-talk into 7 min without losing
  any words. Anything shorter than the threshold stays as the natural
  rhythm. <30ms is always unsafe — mid-phoneme.

- **Cut out filler words and disfluencies by default.** Treat each
  occurrence as an inline cut candidate exactly like a silence gap —
  split the EDL range around it so the kept words concatenate cleanly:

      "uh", "um", "umm", "uhh", "er", "erm", "ah", "ahh", "hmm", "mm",
      "like" (verbal-tic usage only — keep it when it's the verb / a
              preposition / a simile),
      "you know" (filler usage),
      "I mean" (false-start usage, not when correcting meaning),
      "so yeah", "kinda", "sorta" (filler usage),
      single-syllable false starts ("th-", "wh-", "the the", "we we",
              "I I", "and and"),
      repeated stutter words (same word twice while collecting thought),
      abandoned sentence fragments where the speaker restarts.

  The Parakeet lane preserves these verbatim so you can find and
  remove them. **Do NOT leave them in out of "respect for the natural
  voice."** A clean tight delivery IS the speaker's voice with the
  friction removed.

  Exceptions (keep the filler, note in `reason`):
    (a) the filler IS the punchline / joke / emotional beat,
    (b) removing it would break a load-bearing rhythm the user
        explicitly asked for (check the verbatim quotes),
    (c) every surrounding take is worse and this one is genuinely
        best.

  When cutting a filler, the resulting two adjacent same-source
  ranges must each still snap to word boundaries (Hard Rule 6) and
  the combined-pad clamp from the silence-removal pass applies (so
  the lead/trail margins don't re-introduce the filler you just cut).
  For zero-gap repeated words, snap to the END of the first instance
  / the START of the second — never mid-word.

- **Cut padding comes from the pacing preset**, not per-cut taste.
  Expand each range by `lead_margin` at the head and `trail_margin`
  at the tail. Hard Rule 7's 30-200ms working window still bounds
  anything outside the preset table — never go below 30ms.

- **Never reason audio and video independently.** Every cut must work
  on both tracks.

---

## Pacing preset application algorithm

The parent's brief gives you four numbers from a preset name. Apply
them like this:

```
Per-source pre-pass (run BEFORE picking takes across sources):

  1. Walk the word-level transcript for the source.
  2. Compute gap_i = word[i+1].start - word[i].end for every adjacent
     word pair.
  3. Mark every gap_i >= min_silence_to_remove as a "cut here" point.
     This INCLUDES intra-phrase gaps — a thinking pause inside a
     phrase splits the phrase into two adjacent ranges.
  4. The kept-speech runs are the spans between consecutive cut points
     (plus the head before the first cut and the tail after the last).
  5. Drop any run whose total speech duration is < min_talk_to_keep
     (filters orphan single-syllable false starts that survived the
     silence pass).
  6. Each surviving run becomes one EDL range, padded with lead_margin
     at the head and trail_margin at the tail.
  7. Boundary clamp: when two surviving runs come from the same source
     and are separated by a cut silence of `gap_ms`, clamp the trailing
     margin of the first range and the leading margin of the second so
     their combined padding never exceeds `gap_ms - 60ms` (leave at
     least 60ms of true silence so the 30ms afade pair on each side
     has room to breathe). Concretely:

        combined_pad_ms = min(trail_margin + lead_margin,
                              max(0, gap_ms - 60))
        prev.trail_pad  = combined_pad_ms * trail_margin /
                          (trail_margin + lead_margin)
        next.lead_pad   = combined_pad_ms - prev.trail_pad

     This split-evenly-by-ratio rule keeps the head/tail balance the
     user picked while keeping aggressive silence removal aggressive.
```

EDL range expansion formula:

```
range.start = max(0, kept_first_word.start - lead_margin / 1000)
range.end   = min(src_duration, kept_last_word.end + trail_margin / 1000)
```

Stay inside Hard Rule 7's 30-200ms working window. Never go below
30ms regardless of preset.

---

## Time-squeezing (timelapse)

Real-world footage is often "1 minute of explanation, then 25 minutes
of silently doing the work, then 2 minutes of wrap-up." Cutting the 25
minutes throws away the visual story; keeping it 1x bores the viewer.
The third option is **time-squeezing**: compress the work segment into
a 5-30s timelapse on the output timeline.

### When to reach for it

Look for stretches in `merged_timeline.md` where BOTH are true:

1. **Visually continuous activity** — long runs of `(same)` collapses
   in `visual_timeline.md` OR successive `visual:` lines describing
   the same scene with mild variation. Pure dead-air (camera abandoned
   on a tripod, nothing moving) is a CUT candidate, not a timelapse
   candidate.
2. **A coherent story-of-progress** the viewer benefits from seeing
   compressed: assembly, packing, walking, driving, cooking, painting,
   prep, teardown. If the squeezed result wouldn't read as "watch them
   do this thing fast," cut instead.

### Speech inside the stretch is a judgement call

The real test is "does the viewer need to hear this?", not "is anyone
talking?":

- **Load-bearing speech** (instruction, explanation, narration that
  carries the cut, the punchline that lands the beat): split AROUND
  it. Emit a 1x range for the words, then a `speed > 1.0` range for
  the silent / no-words-that-matter middle, then another 1x range for
  whatever talks next.
- **Filler speech** (mumbling, swearing at a misplaced screw, idle
  narration of "okay ... there we go ... hmm"; 30 minutes of casual
  chatter while building that isn't actually teaching anything):
  squeeze right over it. With `audio_strategy="drop"` (the default at
  `speed != 1.0`) the words vanish along with the room tone, the
  visual story plays compressed, and the viewer thanks you.

When in doubt: lean toward squeezing over filler rather than splitting
into a hundred tiny 1x ranges. The video is for the viewer.

### How to size the squeeze

Pick `speed` so the OUTPUT segment lands between **5-30 seconds**:

| Source stretch | Speed | Output | Reads as |
|----------------|-------|--------|----------|
|  30 s          | 4x    |  7.5s  | quick montage |
|  2 min         | 8x    |  15s   | "they assembled it" |
|  5 min         | 10x   |  30s   | full build sequence |
| 10 min         | 10x   |  60s   | over budget — split into two squeezes with a beat between, or cut |
| 30 min         | 10x   | 180s   | far over — pick the visually richest 5-min sub-stretch and squeeze that; cut the rest |

**Hard ceiling: `speed = 10.0` (1000%).** Both helpers clamp to it
with a warning. Beyond that the retime starts decimating frames and
looks broken.

### Audio strategy

Two values, picked automatically from `speed`:

- `audio_strategy = "drop"` (default at `speed != 1.0`): audio is
  silenced over the squeezed range. Right answer for ~95% of timelapses
  — sustained shop noise / room tone sped up 5-10x sounds awful.
- `audio_strategy = "keep"`: audio retimed alongside video. Use only
  when there's a specific reason to keep source audio (recognisable
  voice in the background, distinctive ambient texture). FCPXML / xmeml
  output gets a matching retime; editor must toggle "Maintain Audio
  Pitch" in the NLE.

### Discipline

- **Decide per-stretch: is the speech worth keeping?** Load-bearing
  earns a 1x split around it; filler gets squeezed over with `drop`.
- **Cut FIRST, squeeze SECOND.** Apply the silence-removal pass first;
  then identify surviving long stretches that fit the criteria; then
  squeeze. Squeezing dead air is just slower nothing.
- **Word-boundary discipline still applies on adjacent 1x ranges**
  (Hard Rule 6 / 7). The squeezed range itself doesn't need word-
  boundary alignment when `audio_strategy="drop"`, but pad it
  generously (~1-2s on each side of the activity).
- **`speed` field is OPTIONAL and defaults to 1.0.** Untouched EDLs
  behave exactly as before. Only emit `speed` when actively squeezing.
- **The retime key is `speed` — NOT `timelapse_speed`, NOT
  `clip_speed`, NOT `retime`.** Recurring agent footgun: beats named
  `*_TIMELAPSE` invite an autocomplete-style `"timelapse_speed": 8`
  the exporter cannot recognise. The export pipeline does a defensive
  textual rename of `timelapse_speed -> speed` before parsing, but do
  NOT rely on it — write the canonical key the first time. The
  `notes_for_editor` block at the end of the EDL is an especially
  common offender.

### Example EDL with a timelapse

```json
"ranges": [
  {"source": "C0210", "start":   2.40, "end":  62.30, "beat": "INTRO",
   "audio_lead": 0.0, "video_tail": 0.0, "transition_in": 0.0,
   "quote": "today we're going to build a bench from scratch"},
  {"source": "C0210", "start":  68.10, "end": 1248.40, "beat": "BUILD",
   "audio_lead": 0.0, "video_tail": 0.0, "transition_in": 0.0,
   "speed": 10.0, "audio_strategy": "drop",
   "reason": "19.6 min of cutting/sanding/assembly with no speech and continuous visual activity -> 118s timelapse; editor adds music in NLE"},
  {"source": "C0210", "start": 1255.20, "end": 1310.80, "beat": "REVEAL",
   "audio_lead": 0.0, "video_tail": 0.0, "transition_in": 0.0,
   "quote": "and that's the finished bench"}
]
```

---

## Split edits — DEFERRED (Hard Rule 14)

J-cuts (`audio_lead`), L-cuts (`video_tail`), and cross-dissolves
(`transition_in`) are deferred. Emit `audio_lead = video_tail =
transition_in = 0.0` on EVERY range. Do not use J-cuts, L-cuts, or
cross-dissolves under any circumstances.

The EDL schema still accepts these fields and the FCPXML exporter
still consumes them — but you must emit `0.0` for all three. The
30ms `afade` pair at every boundary (Hard Rule 3) is the only audio
crossfade available right now and is sufficient to suppress boundary
pops.

If the user explicitly asks for J/L cuts or dissolves: note it in
your return rationale so the parent can explain the deferral honestly
and offer to log it in `project.md` as an outstanding item.

---

## EDL output format

Write to `<edit>/edl.json`:

```json
{
  "version": 1,
  "sources": {"C0103": "/abs/path/C0103.MP4",
              "C0108": "/abs/path/C0108.MP4"},
  "ranges": [
    {"source": "C0103", "start": 2.42, "end": 6.85,
     "beat": "HOOK",
     "audio_lead": 0.0, "video_tail": 0.0, "transition_in": 0.0,
     "quote": "...", "reason": "Cleanest delivery, stops before slip at 38.46."},

    {"source": "C0108", "start": 14.30, "end": 28.90,
     "beat": "SOLUTION",
     "audio_lead": 0.0, "video_tail": 0.0, "transition_in": 0.0,
     "quote": "...", "reason": "Only take without the false start."},

    {"source": "C0210", "start": 68.10, "end": 1248.40,
     "beat": "BUILD",
     "audio_lead": 0.0, "video_tail": 0.0, "transition_in": 0.0,
     "speed": 10.0, "audio_strategy": "drop",
     "reason": "19.6 min of silent assembly -> 118s timelapse"}
  ],
  "pacing_preset": "Paced",
  "pacing": {"min_silence_to_remove_ms": 200,
             "min_talk_to_keep_ms":      200,
             "lead_margin_ms":           200,
             "trail_margin_ms":          200},
  "overlays": [
    {"file": "edit/animations/slot_1/render.mp4",
     "start_in_output": 0.0, "duration": 5.0}
  ],
  "subtitles": "edit/master.srt",
  "total_duration_s": 87.4
}
```

`overlays` are rendered animation clips, played as picture-in-picture
on a higher video track in the NLE. `subtitles` is optional and
points at an SRT the parent emits via `helpers/build_srt.py`.
`pacing_preset` + `pacing` record the user's chosen preset.
`audio_lead` / `video_tail` / `transition_in` per range are DEFERRED
(Hard Rule 14) and must always be `0.0`. `speed` and `audio_strategy`
are OPTIONAL and only appear on time-squeezed ranges.

There is **no `grade` field** — color is out of scope. The skill
emits XML for the NLE; the colorist applies the grade there.

---

## Return format (what you give back to the parent)

Return:

1. **One-line runtime check** — sum of effective range durations vs
   target. e.g. `"Total: 87.4s vs target 90s — under budget by 2.6s"`.
2. **Per-beat rationale** — one line per beat:
   `HOOK [C0103 2.42-6.85] cleanest delivery, no slips`
   `PROBLEM [C0108 14.30-28.90] only take without false start`
3. **Compromises** — any beat where you had to keep a verbal slip or
   compromise on take selection because no better option existed.
   Note explicitly so the parent can flag to the user.

The parent will translate this into plain English for the user. Be
factual, be terse on the report, but be thorough on the EDL itself.

---

## Editor-specific anti-patterns

- **Skipping the merged_timeline read** to "save time." Violates the
  ABSOLUTE READ MANDATE at the top of this file. The cut is silently
  bad and the user will catch it.
- **Cutting purely on a CLAP label.** Always cross-check visual.
- **Leaving "uh" / "um" / "like" / repeated stutters in.** Default is
  to cut them; only the three named exceptions stay.
- **Ignoring intra-phrase silence gaps** because "the speaker is still
  talking." The threshold applies to every word-pair gap, not just
  phrase boundaries.
- **Emitting non-zero `audio_lead` / `video_tail` / `transition_in`.**
  Hard Rule 14.
- **Using a synonym for `speed`** (`timelapse_speed`, `clip_speed`,
  `retime`). The exporter only knows `speed`.
- **Squeezing pure dead air** instead of cutting it. Time-squeezing
  is for visually continuous activity, not for empty rooms.
- **Picking `speed` so the squeezed result lands < 5s or > 30s.**
  Re-pick to land in the sweet spot, OR split into multiple squeezes
  with beats between, OR cut some of it.
- **Splitting around every word of filler speech in an otherwise-
  squeezable stretch.** If the speech isn't load-bearing, squeeze
  right over it with `audio_strategy="drop"`.
- **Ignoring the mode flags in the brief.** If `script_mode = true`
  and you skip `references/scripted.md`, you'll cut as if the
  voiceover were a talking head — wrong assembly model. Same for
  `b_roll_mode = true` and `references/b_roll_selection.md`. The
  flags exist so the rules apply when they're meant to.
- **Inferring a mode flag the brief didn't set.** If you discover
  scripted-mode-shaped material when the brief says
  `script_mode = false`, return to the parent with a flag-
  clarification request, don't silently switch modes mid-spawn.
- **Defaulting the verification bar to "creator" on a
  `professional` brief.** Top-candidate review and detailed QA
  notes are mandatory at the professional bar — see
  `b_roll_selection.md` "How `user_profile` shapes the bar."
