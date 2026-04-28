# Scripted assembly — script + voiceover cuts

> Cold-path feature. Loaded on demand by the **editor sub-agent** when the
> parent's brief says `script_mode = true` (i.e. user confirmed they
> have a written script, an already-recorded voiceover, or both, and they
> want b-roll matched to it). Parent gates this in step 4 of the
> 9-step process — see `parent_rules.md`. If the user is editing a
> talking-head / interview / vlog / workshop with no separate VO, this
> file is **not** in scope and the editor never reads it.
>
> Read this file in full when spawned with `script_mode = true`.
> It binds the assembly model — script is the source of truth, the
> voiceover provides the timing, and b-roll is matched per beat against
> the visual lane. Do not improvise around these rules.

---

## What "scripted assembly" actually is

In the default (non-scripted) workflow, editor cuts dialogue *out
of* the recorded speech — the speaker says everything, you trim
filler / silence / bad takes from a single talking-head source.

In **scripted assembly**, something different happens:

1. The user wrote a **script** (a text document — paragraphs, beats,
   sometimes timing notes).
2. They (or someone else) recorded a separate **voiceover** reading
   that script.
3. You are assembling **b-roll** (and possibly cutaways from the
   speaker's footage) **timed to the voiceover**, so the visible
   subject matches what the narrator says.

The output is a multi-source EDL where ranges from many b-roll clips
sit under a single continuous voiceover audio bed. This is the
common shape for product launches, sponsorship reads, recaps,
explainers, branded content, sizzle reels, and event recaps.

### Voiceover container format

The voiceover source can be either:

- **An audio-only file** (`.wav`, `.mp3`, `.m4a`, `.flac`, ...) — the
  common case when the user recorded VO at a desk post-shoot.
  Parent runs `preprocess.py <voiceover.wav>` (or via
  `preprocess_batch.py` mixed with video sources) and the
  speech lane produces `<edit>/transcripts/<voiceover_stem>.json` just
  like any other source. The visual lane is
  auto-skipped for audio-only sources.
- **A video file with the VO baked into the audio track** — e.g. the
  user re-recorded VO in their NLE and exported a `.mov` reference.
  Treated like any other video; the speech lane transcribes the audio
  and you ignore the visual lane output for the VO clip
  (Florence-2 captions of a black-screen VO are useless).

Treatment is the same either way. The transcript
shape is identical; the path is `<edit>/transcripts/<voiceover_stem>
.json`. Verify the file via `source_tags.json`
(if present) for clips tagged `voiceover` — that's the parent's
declaration of which file is the timing spine.

Because both script and voiceover are fixed, **the cut
problem inverts**: instead of "find the best take in this footage,"
the question becomes "for each named beat in the script, which clip
in my library shows the thing the narrator is saying right when
they're saying it?"

---

## The source of truth hierarchy (scripted mode)

This overrides the default speech-first / visual-second / audio-third
priority (still applies to the voiceover transcription).
Per cut decision in scripted mode:

1. **The script is the editorial spine.** Read it first. Segment it
   into beats. Every cut is anchored to a beat.
2. **The voiceover word timestamps set the timing.** Use the fresh
   Parakeet ONNX transcription of the VO file — not the
   script's prose timing. The script tells you *what* is said;
   the voiceover transcript tells you *when*.
3. **Visual captions decide which clip lands on each beat.** The
   `[…]` visual lines in `audiovisual_timeline.md` (Florence-2
   captions @ 1 fps) are how you verify a clip shows the named
   subject.
4. **Audio events are noisy hints only**, same as the default rule
   (Hard Rule on CLAP cross-checks).

## What to ignore — explicitly

When parent's brief says `script_mode = true` and the user has
provided a script + separate VO, **ignore these unless user
explicitly asks for them:**

- Old `cut.fcpxml` / `cut.xml` from a previous session.
- Old `master.srt` (it's stale relative to the new script + VO).
- Pre-existing transcripts cached against an OLD voiceover file.
- Caption files the user provided from a different cut.
- The previous EDL — except as a diff target on revisions, per the
  change-request flow.

If you re-use any of those, you'll match b-roll to the wrong words.
Always start from the **fresh** Parakeet transcription of the
**current** voiceover file. Parent will have re-run
`preprocess.py <voiceover.wav>` (or equivalent) so a fresh
`<edit>/transcripts/<voiceover>.json` exists; trust that, ignore the
stale ones.

---

## In-clip notes + retake detection apply to the VO too

The voiceover file is the editorial spine in scripted mode, but the
user often records VO with same friction as A-roll:
*"…welcome to the- ugh, let me try that again. Welcome to the
show…"*. The two source-side detection rules from your default
operating manual — **in-clip editor notes** (per-clip verbal
directives) and **retake detection** (frustration markers,
restarts, paraphrased redos) — apply to the VO transcription
exactly as for A-roll. Re-read those sections in
`subagent_editor_rules.md` if unclear.

Concretely on the VO:

- Trigger phrases like *"hey editor, scrap this take"* embedded in
  the VO file → exclude the preamble from the VO audio bed, apply
  the directive.
- Frustration markers in the VO followed by a restart of the same
  script line → use the cleaner take, drop the earlier one.
- Multiple takes of the same paragraph in the same VO file → pick
  the keeper per the retake-selection rules (default: later).
- Cross-VO retakes (user re-recorded the entire VO into a fresh
  file): parent re-runs preprocess on whichever VO is
  authoritative; cut from THAT, ignore older.

Surface every VO-side note / retake decision in the return
rationale, same shape as A-roll. The script-beat alignment
(steps 3-5 below) runs against the **cleaned** VO timeline — i.e.
what's left AFTER you've excluded preambles and rejected takes.

---

## The 7-step assembly procedure

Run in order every spawn with `script_mode = true`. Do not
shuffle steps; later steps depend on earlier ones completing fully.

### 1. Read the script in full

Parent forwards the script in the brief (or as
`<edit>/script.md` / `<edit>/script.txt`). Read every line. Do not
skim. Do not paraphrase. The script's exact wording determines which
named subjects you'll be matching against — "the new RTX 5090" and
"a graphics card" are different beats and need different clips.

If the script is not in plain prose (paragraphed
beats, scene headings, timing markers, bracketed stage directions),
honour the structure — bracketed directions like `[CUT TO BOOTH]` or
`[B-ROLL: assembly line]` are user instructions and bind the
matching for that beat.

### 2. Segment the script into beats

A **beat** is one continuous span of voiceover where one b-roll clip
should be on screen. Beats are typically 2-8 seconds each. Boundaries
fall on:

- Sentence breaks where the topic shifts ("...the RTX 5090 was on
  display. *Across the floor, ASUS had its own line-up.*")
- Named-subject transitions ("...Riot Games announced. *Then Valve
  came out with...*")
- Punchline / reveal landings (the b-roll changes ON the punchline,
  not before)
- Director-style instructions in brackets (`[CUT TO ...]` —
  authoritative)

Each beat gets a short label and a 1-line description of what should
be visible. Examples:

```
BEAT 01  [0.00-3.20s VO]  "Welcome back to PAX East"
         visible: PAX East entrance signage / banner / show-floor wide
BEAT 02  [3.20-7.85s VO]  "Riot Games had a massive booth"
         visible: Riot Games booth signage, logo, banners
BEAT 03  [7.85-11.40s VO]  "with Valorant on every screen"
         visible: monitors showing Valorant gameplay, agent select, HUD
```

The label / description is for your reasoning; you'll use them in
the `reason` field of the EDL range you emit per beat.

### 3. Anchor every beat to voiceover timestamps

Use the fresh Parakeet transcript of the voiceover (cached in
`<edit>/transcripts/<voiceover_stem>.json` — read directly for
word-level `start` / `end`). Walk the transcript word-by-word and
match it to the script beats:

- Find the first word of each beat in the transcript. Its `start` is
  the beat's `vo_start`.
- Find the last word of the beat. Its `end` is the beat's `vo_end`.
- The beat's runtime is `vo_end - vo_start`.

If the script and the voiceover **diverge** (the narrator skipped a
sentence, ad-libbed an extra clause, mis-pronounced a brand name) —
trust the voiceover. The script is what *should* have been said;
the voiceover is what *was* said. Re-anchor the beat boundary to the
voiceover's word timing and note the divergence in the
`reason` field for that range.

If the script names a brand / product / game / person and the
voiceover skips that name, drop the beat (or merge it with
its neighbour). Don't put a "Riot Games booth" b-roll over a
voiceover that doesn't mention Riot.

### 4. Shortlist b-roll candidates per beat

For each beat, you need a clip whose `[…]` visual captions in
`audiovisual_timeline.md` describe what the script says is visible.

For named-subject beats, shortlist **source ranges**, not just clips.
The candidate range must make the named subject the visual subject:
front/center/readable signage, product label, screen, booth, character,
or stage. A caption hit where the named thing is only in the
background is a fallback, not a match. Within the valid subject-forward
shortlist, prefer smoother ranges: repeated or consistent captions,
static shots, and low-motion camera work beat jerky walk-bys.
Smoothness never outranks subject accuracy. On revisions to an already
liked cut, preserve the existing subject-correct clip unless the
replacement is at least as subject-forward and only improves stability,
framing, or rhythm.

The default reading surface is the dual spine
(`audiovisual_timeline.md` + `speech_timeline.md`) — read BOTH
end-to-end (per the ABSOLUTE READ MANDATE in
`subagent_editor_rules.md`) before this step. The visual lane lines
in `audiovisual_timeline.md` are your search target; the
`speech_timeline.md` cross-check tells you whether the clip's
on-camera speech aligns with or contradicts the script beat.

**Three shortlisting paths**, pick whichever fits the library size
and the user_profile bar:

1. **In-context scan (default for small libraries).** Walk
   the `[…]` visual lines in `audiovisual_timeline.md` per beat
   for matches. Build a shortlist
   of 3-8 candidate clips per beat by:
   - Captions containing the beat's named subject ("Riot Games sign"
     → captions mentioning Riot, sign, banner, booth signage, logo).
   - Cross-checking the speech lane on the same source — if a clip's
     speaker is *talking about* the subject, that doesn't help; you
     want the clip *showing* the subject.
   - Preferring clips with multiple consecutive matching `[…]`
     visual lines (>= 3 seconds of stable visual evidence) over a
     single one-frame hit (which might be a fast pan-through).

2. **Spawn b-roll scout subagents (recommended for large libraries
   or `professional` bar).** Per `subagent_editor_rules.md` "B-roll
   scout spawn protocol", delegate per-beat shortlisting to
   parallel scout subagents. Pass them `source_tags.json` (when
   present) so they only consider b-roll-tagged clips. They return
   ranked shortlists with evidence; you pick / verify / write the
   EDL range. This keeps your context for cut decisions
   vs caption re-scans.

3. **Use a clip index (when available, as a shortlist aid).** If
   the brief mentions `<edit>/clip_index/index.json`, you
   may text-search it for shortlisting. Whether you use it
   in-context or hand it to a scout, **verification still binds in
   step 5** — the index suggests; the visual-lane drill-down decides.

When `source_tags.json` is in the brief, restrict candidate
searches (whether in-context, scout-delegated, or index-queried) to
clips tagged `b_roll` / `cutaway` / `unknown`. A-roll-tagged clips
are the speech bed in talking-head mode; in scripted assembly the
A-roll tag is rare since the VO carries audio — but if it appears,
respect user organization.

### 5. Verify the top candidate against the visual evidence

Pick the highest-evidence clip from the shortlist. Then **verify**
before committing:

- Read the surrounding `[…]` visual lines in
  `audiovisual_timeline.md` (the range you're proposing to cut from
  + 1-2 seconds before/after).
  Continuous matching captions = good. A single matching frame
  surrounded by something else = a fast pan, reject.
- Verify subject dominance: the named thing should be the frame's
  subject, not incidental background. Look for words like
  `foreground`, `center`, `large sign`, `screen displays`, `logo`,
  `banner`, `close-up`, `booth`, or product text. Treat
  `background`, `distant`, `blurred`, and people-heavy descriptions
  as downgrades.
- Verify smoothness: prefer ranges where adjacent captions keep
  describing the same subject. Reject shaky/jerky walk-by footage
  when a stable subject-forward option exists.
- Drill into `<edit>/visual_timeline.md` for the full 1fps caption
  stream (including `(same)` repeats — those are gold for stability).
  A long run of `(same)` near the candidate range means the shot is
  static and editorially safe.
- If parent told you frames matter (rare but happens for
  high-stakes brand work), invoke `helpers/timeline_view.py
  <source.mp4> <start> <end>` for a filmstrip + waveform PNG of the
  candidate range and verify visually before committing.

If the top candidate fails verification, descend to candidate 2, 3,
... until one passes. Note in the `reason` field which candidate you
picked and (if relevant) which higher-ranked candidates you rejected
and why.

### 6. Set source in-points to land the visual on the spoken word

The naive cut is `range.start = candidate.first_match_t` — the clip
starts at the first frame the subject appears. That's usually wrong.

The right cut sets the **in-point inside the candidate** so the
visible subject is **on screen at the moment the narrator says its
name**, not seconds before or after. Concretely:

```
beat.vo_subject_word_start  = the VO timestamp where the named subject
                              is spoken (e.g. when the narrator says
                              "Riot Games", not the start of the
                              sentence)
candidate.subject_visible_t = the source timestamp where the subject
                              first becomes clearly visible in the
                              candidate clip (from visual lane)

# The clip's in-point is offset so the subject appears in sync:
range.start = candidate.subject_visible_t
              - (beat.vo_subject_word_start - beat.vo_start)

# Pad with the pacing preset's lead_margin and clamp.
```

If the subject name lands at the start of the beat ("Riot Games
had a massive booth..."), the clip in-point is just
`subject_visible_t - lead_margin`. If the name lands mid-beat
("...and right next to it was the *Riot Games* booth..."), shift the
clip in-point earlier so the visual lands on the spoken name.

For named subjects (brands / products / games / people / venues),
**this synchronisation is non-negotiable**. The user will catch a
"Riot Games" b-roll that lands two seconds late on the wrong word —
scripted-mode users mean this when asking for "tighter
sync" or "land the booth on the booth."

The sync target is the subject-forward moment, not the first weak
caption hit. If the source has a background glimpse at 0:02 but the
subject becomes front/center/readable at 0:05, align the 0:05 moment
to the spoken name. A slightly earlier background flash does not count
as `candidate.subject_visible_t`.

### 7. Emit a QA note per b-roll decision

For every important b-roll decision (especially named-subject beats),
the EDL range's `reason` field should be a compact QA note in this
shape:

```
beat: <BEAT_NN>  vo: "<the 4-8 word phrase that triggers the cut>"
  vo_subject_word_t: <s>     visible_at: <s in source>
  candidate: <stem>          rejected: <stems> (why)
  evidence: <2-3 word visual caption excerpt>
```

Example:

```json
{
  "source": "C0312", "start": 18.40, "end": 22.85,
  "beat": "RIOT_BOOTH",
  "audio_lead": 0.0, "video_tail": 0.0, "transition_in": 0.0,
  "quote": "Riot Games had a massive booth",
  "reason": "vo_subject 'Riot Games' at 4.85s; C0312 booth signage stable 18.4-22.9; rejected C0188 (banner partially blocked), C0204 (people-heavy walk-by)"
}
```

Revision conversations stay cheap — parent can read your
reasons aloud to the user, the user says "the rejected C0188 actually
had the better banner shot," and the next spawn knows which mismatch
mattered.

---

## Cross-references that bind in scripted mode

- The **B-Roll Selection Rules** in `references/b_roll_selection.md`
  bind every decision in step 4 and 5 above. If parent set both
  `script_mode = true` and `b_roll_mode = true`, read both files.
  (Practically always both — scripted assembly is a b-roll workflow.)
- The **pacing preset** still applies to the voiceover side: silence-
  removal on the VO transcript, lead/trail margins on each range,
  word-boundary discipline (Hard Rule 6) on every cut edge.
- **Hard Rule 14** still defers J/L cuts and dissolves to `0.0` —
  scripted mode does not unlock split edits.
- **Hard Rule 15** (dual-spine read) still applies — read BOTH
  `audiovisual_timeline.md` AND `speech_timeline.md` end-to-end,
  drill into per-lane files at ambiguous moments. The script does
  not replace the dual-spine read; it *adds* a structural anchor on
  top of it.

---

## When the user has a script but no separate voiceover

Edge case: user wrote a script but talking-head footage IS
the voiceover (they're reading the script on camera). In that case:

- The script is still the source of truth for *what should have been
  said* — useful for catching skipped lines or re-takes.
- But the cut model collapses back to default talking-head behaviour:
  one source, cut for filler / silence / bad takes against the
  pacing preset.
- B-roll cutaways still benefit from the rules in
  `b_roll_selection.md` — the named-subject rule still applies when
  inserting cutaways.

Parent picks the mode by asking user up-front
("do you have a separate voiceover, or is the speaker reading the
script on camera?"). If on-camera, it sets `script_mode = false` but
keeps `b_roll_mode = true` and forwards the script as a "structural
hint" in the brief — you read it for context, not as the assembly
spine.

---

## When the user has a voiceover but no script

Inverse edge case. Treat the fresh Parakeet transcript of the VO as
the source of truth — nothing else. Segment beats
from the transcript (sentence breaks, named-subject
mentions). Match b-roll per the same procedure. No script to verify
against.

---

## Anti-patterns specific to scripted mode

- **Using stale transcripts for the voiceover.** Every scripted
  session starts with a fresh Parakeet pass on the VO file
  the user provided. Parent re-runs preprocessing for that file;
  trust the freshly cached transcript, not anything older.
- **Matching by filename / metadata instead of visual captions.**
  Filenames lie ("riot_booth.mp4" might be a wide of the
  show floor where Riot is barely visible). Always verify against
  the `[…]` visual lines in `audiovisual_timeline.md`.
- **Picking the first index hit without verifying.** Two-stage
  matching is the rule (shortlist + verify). Skipping verification
  is how brand b-roll lands on the wrong booth.
- **Letting beats slip out of sync because "the words still happen
  during this clip."** Named subjects (brands / products / people)
  must land **on** their spoken name, not "during the same sentence."
- **Treating the script as approximate.** The script is exact text.
  Bracketed directions (`[CUT TO ...]`) are commands, not hints.
- **Putting people-heavy crowd shots on subject-named beats.** When
  the script says "Riot Games booth," the visible thing is a Riot
  Games booth — not a crowd in front of an unknown booth. See
  `b_roll_selection.md` for the full preference rule.
- **Re-using last session's EDL as a starting point on a script
  rewrite.** If the script changed, the b-roll mapping changed.
  Start fresh; diff old EDL only on parent's request, not as a
  default.
- **Skipping the QA note on a named-subject beat.** Those notes are
  load-bearing on revisions. Skip the note → next revision the
  parent has nothing to forward → user repeats same
  feedback.

---

## Scaling shortlisting — scouts, indexes, in-context

Three escalation tiers as your library grows:

- **Tier 1 — in-context scan (small libraries).** Read both
  `audiovisual_timeline.md` and `speech_timeline.md` end-to-end
  once, then scan the `[…]` visual lines in the AV file per beat.
  Cheap when library fits your read budget and the number of beats
  is small (<= 8).
- **Tier 2 — clip index (medium-large libraries, no scouts yet).**
  When parent has built `<edit>/clip_index/index.json` (a
  per-clip text-searchable record from cached captions + speech),
  query the index per beat for shortlisting, then verify the
  top candidate(s) against `audiovisual_timeline.md` /
  `visual_timeline.md`. The index is a parent-managed helper
  (`helpers/clip_index.py`-style); it's optional. The index
  accelerates step 4; it doesn't replace verification in step 5.
- **Tier 3 — b-roll scout subagents (large libraries OR
  professional bar OR many beats).** Per `subagent_editor_rules.md`
  "B-roll scout spawn protocol", spawn parallel scout subagents
  (Hard Rule 10) — one per beat or one per cluster of beats.
  Scouts read `<edit>/visual_timeline.md` for in-scope sources in
  their own fresh context windows, return ranked shortlists with
  evidence, and you pick / verify / write the EDL range. This keeps
  your context for editorial decisions vs caption-re-
  scanning, and it stacks with the index when both are available
  (you can pass the index path to scouts for faster shortlisting).

For ALL tiers: the dual-spine pre-flight read
(`audiovisual_timeline.md` + `speech_timeline.md`) is still
mandatory; the verification step (step 5) still binds; and source
in-points (step 6) are computed by you, not by scouts or the index.
