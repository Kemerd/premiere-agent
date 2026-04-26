# Editor Sub-agent Rules — operating manual for the cut-decision agent

You have already read `references/shared_rules.md`. If not, stop and
read it now — it defines the agent hierarchy, the Hard Rules binding
every agent (especially Hard Rule 15, the merged-view spine
principle), and the philosophy making this skill work. Reading these
rules without that context produces silently bad cuts.

You are the **editor sub-agent**. The parent spawned you to produce
`<edit>/edl.json` — the cut decisions for one video session. You
have a fresh context window. Use it.

---

## ABSOLUTE READ MANDATE — read first, no exceptions

This rule overrides everything else in this file. No token budget
concern, no time concern, no efficiency concern, no "diminishing
returns" argument justifies stopping early.

### Mode-gated cold-path reads (conditional, before the merged-view read)

The parent's brief carries four feature-mode flags
(`script_mode`, `b_roll_mode`, `timelapse_mode`, `user_profile`)
plus an optional `source_tags.json` path. For the file-bound
boolean flags, **if true, read the matching cold-path file IN FULL
before reading the merged timeline.** If false, skip the file
silently — those rules don't apply this session.


| Flag                | If `true`, read this file in full          |
|---------------------|--------------------------------------------|
| `script_mode`       | `references/scripted.md`                   |
| `b_roll_mode`       | `references/b_roll_selection.md`           |

`timelapse_mode` is a permission flag, NOT a file flag:

- `timelapse_mode = false` (safe default) — **emit zero ranges
  with `speed != 1.0`**. The "Time-squeezing (timelapse)" section
  later in this file is fully overridden — even when you find a
  long visually-continuous activity stretch that's a textbook
  timelapse candidate, you DO NOT emit `speed > 1.0`. Cut the
  stretch (drop dead air; pick highlights at 1x), or keep selected
  1x ranges, but no retime. The user explicitly opted out of
  timelapses this session — silent retime would damage their
  material.
- `timelapse_mode = true` — the time-squeezing rules in this file
  apply normally. Hard ceiling stays `speed = 10.0`.

`user_profile` is `personal | creator | professional` and is NOT a
file flag — it sets your verification bar. It applies when you're
using the rules from `b_roll_selection.md` (top-candidate review
on every named-subject beat at `professional`, default bar
otherwise) and when writing QA notes in EDL `reason` fields (terse
for personal / creator; detailed list-the-rejected-candidates for
professional).

`source_tags.json` (path may be in the brief) is a small JSON map
of clip stems to categories (`a_roll`, `b_roll`, `timelapse`,
`voiceover`, `unknown`). When the brief lists this path,
**respect the categorization** for candidate searches:

- B-roll candidate searches (whether in-context or via spawned
  b-roll scout subagents) should restrict to clips tagged
  `b_roll` / `cutaway`, OR `unknown` if the user didn't tag
  everything.
- A-roll-tagged clips are the primary speech / audio bed in
  talking-head mode; do not treat them as cutaways unless the
  user explicitly asked for the speaker's footage as cutaways.
- `timelapse`-tagged clips are pre-organized timelapse source
  material; if `timelapse_mode = true`, prefer them as timelapse
  retime candidates over discovering retime stretches in arbitrary
  footage.
- `voiceover`-tagged clips are the VO source in scripted mode; the
  parent will have re-preprocessed it and will list the cached
  transcript path in the brief.

If `source_tags.json` is absent from the brief, all sources are
eligible across all roles (the user didn't organize, so you don't
pre-filter).

`source_pairs.json` (path may be present in the brief) records the
parent's resolution of dual-mic / paired-audio detection from step 1.
Schema:

```json
{
  "mode": "dual_mic" | "ignore",
  "pairs": [
    {
      "stem": "SHOT_0042",
      "video": "...SHOT_0042.mp4",
      "audio": "...SHOT_0042.wav",
      "audio_alias_stem": "SHOT_0042.audio",
      "audio_alias_path": ".../edit/.paired_audio/SHOT_0042.audio.wav"
    }
  ]
}
```

When `mode = dual_mic`, two transcripts exist for the same shot:
`transcripts/<stem>.json` (camera audio) and
`transcripts/<stem>.audio.json` (lav / external recorder). Both
appear as separate entries in `audiovisual_timeline.md` and
`speech_timeline.md` with the suffixed stem visible in the source
label. **Treat them as the same shot for
cut purposes** — don't place the same beat twice. The full handling
procedure is below in "Dual-mic pair handling"; that section is
mandatory reading whenever `source_pairs.json` is present with
`mode = dual_mic` in your brief.

When `mode = ignore`, the paired `.wav` was filtered from
preprocessing; only the camera-audio transcript exists. No special
handling — proceed normally. The file is still recorded so a future
session knows the user explicitly declined dual-mic on this project.

If `source_pairs.json` is absent from the brief, no stem pairs were
detected (or the batch predates pair detection). Proceed normally.


These cold-path files are **additive** to your default rules — they
don't replace the dual-spine read below, the pacing-preset
algorithm, the word-boundary discipline, or any Hard Rule. When
both `script_mode` and `b_roll_mode` are true (common combo for
voiceover-driven assembly), read both files; the assembly procedure
in `scripted.md` references the selection rules in
`b_roll_selection.md` step-by-step.

If a flag the brief didn't mention seems to apply (e.g. you find
the project clearly has b-roll but the brief doesn't say so),
**don't infer it silently** — return to the parent with a flag-
clarification request. The parent owns the flag values; you don't
override them.

### What you must read in full, every spawn

1. **TWO mandatory reading surfaces — read BOTH end-to-end, every
   line, every spawn.**

   1a. **`<edit>/audiovisual_timeline.md`** — END-TO-END. EVERY LINE.

       Audio events (`(...)`) and visual captions (`[...]`) for
       every source, interleaved chronologically by timestamp.
       Speech is intentionally NOT in this file — it lives in 1b
       below. The file is caveman-compressed and sentence-delta-
       deduped at pack time so it fits comfortably in your fresh
       context window — typical projects land in the 200KB-1.5MB
       range — but EVEN IF a project produced a file an order of
       magnitude larger, the rule is the same: read every line.

   1b. **`<edit>/speech_timeline.md`** — END-TO-END. EVERY LINE.

       Phrase-grouped Parakeet transcripts with `M:SS-M:SS [Sn]
       "phrase text"` per line. Integer ranges are **outer-aligned**:
       start rounds DOWN, end rounds UP, so the integer range is a
       guaranteed SUPERSET of the underlying float [start, end] in
       `transcripts/<stem>.json`. This means any integer range you
       read here can be handed straight to `helpers/find_quote.py`
       (see "Word-boundary verification" below) without off-by-one
       fences.

   The two files are aligned by `## <stem>` headers — when reasoning
   about a clip, scroll them in parallel: AV gives you the soundscape
   + on-screen action, speech gives you the editorial spine.

2. **The prior `<edit>/edl.json` (on revisions only)** — END-TO-END.
   EVERY RANGE. EVERY FIELD.

   When the parent re-spawns you with a change request, the parent
   forwards the previous EDL in your brief. Read all of it. Don't
   read only the ranges around the user's complaint and assume the
   rest is fine — the user might have asked for a global change
   ("tighten the whole thing") requiring every range to be touched.

3. **The script (when `script_mode = true`)** — END-TO-END. EVERY
   LINE. The path is in the parent's brief (typically
   `<edit>/script.md` / `<edit>/script.txt`). Bracketed directions
   like `[CUT TO ...]` or `[B-ROLL: ...]` are the user's commands,
   not hints — bind them. See `references/scripted.md` for the
   beat-segmentation procedure. If the script is missing despite
   the flag being true, STOP and report to the parent — don't
   guess at beats from the voiceover transcript alone.

### Hard procedure when a file exceeds one `Read` call

Issue sequential `Read` calls with `offset` / `limit`. If it takes 5
calls, make 5. If it takes 50, make 50. If it takes 500, make 500.
If it takes 99,999 calls, make 99,999.

If you exhaust your context budget before finishing the read, **DO
NOT emit an EDL from partial coverage.** Return to the parent with
an explicit error message and halt:

```
BUDGET_EXHAUSTED
  files: <edit>/audiovisual_timeline.md AND/OR <edit>/speech_timeline.md
  covered: lines [start..end] of [start..N] per file
  reason: combined size too large for current model context
  recovery options:
    - parent respawns me on a model with a larger context window
    - parent pre-shards the file(s) (future: pack_timelines.py --shard)
    - parent reduces the source set (which sources matter for THIS
      revision?) and respawns
```

A partial read produces silently bad cuts the user will ship and
regret. An explicit budget-exhausted return is recoverable. Always
pick recoverable.

### Forbidden behaviours — every one is a violation regardless of how good the resulting cut looks

- Reading only the first N lines / last N lines / "a representative
  sample" of `audiovisual_timeline.md` or `speech_timeline.md`.
- Reading only one of the two mandatory files (`audiovisual_timeline.md`
  alone is missing the editorial spine; `speech_timeline.md` alone is
  blind to the soundscape and on-screen action). BOTH or neither.
- `grep` / `rg`-ing for keywords in the timelines and emitting an EDL
  from matches alone (loses the chronological structure making the
  AV + speech views useful at all). For sub-second, word-precise
  quote → time lookup, use `helpers/find_quote.py` against
  `transcripts/<stem>.json` — never grep transcripts.
- Chunked reads abandoned partway through ("I have enough...", "this
  section is repetitive...", "I can extrapolate from here..."). You
  don't have enough. You can't extrapolate. Finish the file.
- Skipping a `Read` chunk because the previous chunk "looked
  similar." The dedup pass already removed genuinely similar
  frames; what's left is signal.
- "Smart" chunking reading chunks 1, 5, 10 and assuming 2-4 and
  6-9 are interpolatable. They aren't.
- Spawning a SUB-sub-agent to "protect this sub-agent's context
  window." YOU are the editor — this read IS your job. The parent
  already isolated you so this read is affordable; don't outsource
  it again.
- Treating "the user is in a hurry" as license to skip lines. If the
  user is in a hurry, the parent reduces target runtime, not source
  coverage.
- On revisions: reading only the lines around the user's complaint
  and assuming the rest of the prior EDL is fine. Read the whole
  prior EDL. Read both `audiovisual_timeline.md` and
  `speech_timeline.md` again. Every spawn, every revision, full
  coverage.
- Returning a partial-read EDL silently with a note like "(read most
  of the file, used judgement on the rest)." That converts a silent
  failure into a confessed failure but it's still a failure. Use
  the BUDGET_EXHAUSTED return instead.

The user explicitly demanded this rule be ironclad. They will catch
any deviation; the cut will be rejected; you'll be re-spawned with
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
     `(audio: ...)` and `visual:` disagree about what's on screen,
     trust visual.

2. **Drill into the remaining per-lane files only when the dual
   spine is ambiguous.** `<edit>/visual_timeline.md` for the full
   1fps caption stream including `(same)` repeats;
   `<edit>/audio_timeline.md` for per-window CLAP scoring detail.
   These per-lane files are bound by the ABSOLUTE READ MANDATE if
   you open them — drill into a SPECIFIC moment, but read
   surrounding context fully, not a one-line snippet. Note that
   `speech_timeline.md` is no longer a "drill-down" file — it is
   one of the two MANDATORY reads from step 1; do not relegate it.

   **Per-word timestamps live ONLY in `<edit>/transcripts/<stem>.json`**
   (raw Parakeet `words[]` array with `{text, start, end}` per token).
   Neither AV nor speech timelines carry word boundaries — phrase
   grouping in `speech_timeline.md` concatenates words for readability
   and drops the per-token times. When you need to verify a cut at
   a specific word (which is every cut — see "Word-boundary
   verification" below), use **`helpers/find_quote.py`** to crawl
   `transcripts/<stem>.json` programmatically — it accepts a clip
   stem + integer time range (which you read directly out of
   `speech_timeline.md`) and/or a quote substring, and returns
   word-precise `{start, end}` timestamps as JSON. Do NOT `grep` /
   hand-parse `transcripts/<stem>.json` — the helper is 50-100x
   faster and bounds-checks the result against the speech-timeline
   integer range so off-by-one errors are impossible. Reading
   `transcripts/<stem>.json` directly is reserved for cases the
   helper genuinely cannot answer (e.g. inspecting a stem's
   diarization metadata).

3. **If `audiovisual_timeline.md` OR `speech_timeline.md` is
   missing**, STOP and report — the parent must re-run
   `python helpers/pack_timelines.py --edit-dir <edit>` to
   regenerate them. Don't invent a workaround. Both files are
   mandatory; reasoning from one alone is a Hard Rule 15 violation.

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
- **Verbatim user quotes** chronological — original wording matters,
  don't collapse "make it punchy" into "make it short."
- **Things user explicitly asked to keep** — these survive every
  revision unless a later quote reverses them.
- **Things user explicitly rejected** — these are cut, every
  revision, unless a later quote reverses them.
- **Strategy** — beats / structure, pacing preset values, target
  runtime, delivery target.
- **Mode flags** — `script_mode`, `b_roll_mode`, `user_profile`.
  These gate the cold-path file reads above and set the
  verification bar; they travel in every brief.
- **Source tags** (when `<edit>/source_tags.json` exists) — folder-
  convention categorization of clips. Constrains b-roll candidate
  searches per the rules in the cold-path read section.
- **Source pairs** (when `<edit>/source_pairs.json` exists) — the
  parent's resolution of dual-mic / paired-audio detection. When
  `mode = dual_mic`, two transcripts exist per paired shot; follow
  the "Dual-mic pair handling" procedure below.
- **Verbal slips to avoid** — list the parent compiled.
- **Change-request history** (on revisions) — chronological list of
  every prior revision + diff. Read this so revision N is informed
  by revisions 0..N-1.
- **Prior EDL** (on revisions) — diff your output against it so the
  user sees the specific change land, not the whole cut re-shuffled.

---

## Word-boundary verification — every cut, every spawn

This is the rule that prevents the mid-sentence / mid-word cuts the
user has been catching. Hard Rule 6 says "never cut inside a word";
this section is the procedure that enforces it. **No EDL range gets
emitted until both `start` and `end` have been verified against the
WORD-level timestamps in `<edit>/transcripts/<stem>.json`.**

The two-file dual spine exists for beat planning, not for setting
cut times. Phrase ranges in `speech_timeline.md` are outer-aligned
to whole seconds (`floor(start)..ceil(end)`); they fully ENCLOSE the
underlying float spans but they do NOT tell you where individual
words start and end. The Parakeet word boundaries are sub-second
precise and live in `transcripts/<stem>.json`. Use them — through
`helpers/find_quote.py`, not through `grep` / hand-parsing.

### What each file actually carries

| File                            | Granularity      | Use for cuts? |
|---------------------------------|------------------|---------------|
| `audiovisual_timeline.md`       | per-second visual + per-window audio (no speech) | NO — beat planning + soundscape only |
| `speech_timeline.md`            | phrase START-END range, outer-aligned (start floor / end ceil) | NO — phrase-span scoping; integer range fully encloses the float span |
| `transcripts/<stem>.json` (via `helpers/find_quote.py`) | per-word `{text, start, end}` sub-second | **YES — canonical source for every cut anchor** |

### Verification workflow (run on every range you write)

For each candidate range you decide on while reading the dual spine:

1. **Identify the words you want to keep.** From the phrase quoted
   in `speech_timeline.md`, pick the first and last word the kept
   range should contain — e.g. *"Geared to lock in the uplock hook"*
   with the user wanting to keep through *"lock in"* gives
   `first_word = "Geared"`, `last_word = "in"`.

2. **Call `helpers/find_quote.py`** to crawl
   `<edit>/transcripts/<S>.json` programmatically. Pass the clip
   stem, the integer time range you read straight off
   `speech_timeline.md` (which is a guaranteed superset of the
   actual word span), and either a quote substring or a pair of
   word anchors. The helper returns word-precise `{start, end}`
   in JSON, plus the matched word run for sanity-checking. It is
   bounds-checked, lightning fast (sequential JSON walk under the
   integer range — typically under 5ms), and removes the off-by-one
   class of bugs entirely. Example invocation:

   ```bash
   python helpers/find_quote.py \
       --edit-dir <edit> \
       --clip <S> \
       --range 0:02-0:18 \
       --quote "lock in"
   ```

   Reading `transcripts/<stem>.json` by hand or with `grep` is
   forbidden — the helper is the supported interface.

3. **Snap the cut anchors to the helper's word boundaries.**
   - In-point: `result.first_word.start`
   - Out-point: `result.last_word.end`
   - Never let the cut land between a word's `start` and `end`.

4. **Apply pacing-preset margins** (per the formula in "Pacing
   preset application algorithm"):

   ```
   range.start = max(0, kept_first_word.start - lead_margin / 1000)
   range.end   = min(src_duration, kept_last_word.end + trail_margin / 1000)
   ```

5. **Clamp adjacent same-source ranges** with the combined-pad rule
   so the trail_margin of range N + the lead_margin of range N+1
   don't re-introduce the silence you removed (and leave at least
   60ms of true silence for the 30ms `afade` pair on each side).

### Worked example — "cut after 'lock in'"

Phrase visible in `speech_timeline.md`:
`0:03-0:05 [S0] "Geared to lock in the uplock hook."`
Editor goal: keep through *"lock in"*, drop the rest.

`helpers/find_quote.py` invocation:

```bash
python helpers/find_quote.py --edit-dir <edit> --clip <S> \
    --range 0:03-0:05 --quote "Geared to lock in"
```

Helper output (relevant fields):

```json
{
  "first_word": {"text": "Geared", "start": 3.12, "end": 3.45},
  "last_word":  {"text": "in",     "start": 3.80, "end": 3.95},
  "next_word":  {"text": "the",    "start": 4.10, "end": 4.18}
}
```

- Out-point snaps to `last_word.end = 3.95` — the last word kept.
- Naive Paced trail_margin (200ms) would give `range.end = 4.15`,
  but the helper also returns `next_word.start = 4.10`. The
  combined-pad clamp binds: `gap_ms = 4.10 − 3.95 = 150ms`; clamp
  ceiling `max(0, 150 − 60) = 90ms`. Final `range.end = 3.95 +
  0.09 = 4.04` so 60ms of true silence remains for the `afade` pair.
- In-point uses the same procedure on the FIRST word of the kept
  range (here `Geared.start = 3.12`, padded by lead_margin with
  the same clamp against the prior range's tail).

The speech-timeline range `0:03-0:05` was the input to the lookup;
the EDL number is `3.12` / `4.04`. Sub-second precision came from
the helper crawling `transcripts/<S>.json`, never from the integer
ranges in the markdown.

### What counts as a verification miss (forbidden)

- Emitting `range.end` from a phrase START shown in merged plus a
  guessed duration. The phrase's actual end is in transcripts; look
  it up.
- Treating `speech_timeline.md`'s phrase-end timestamp as your cut
  point without confirming it matches `last_word.end` in the
  transcript. Speech_timeline rounds to whole seconds; the
  underlying word end is sub-second.
- Setting `range.end` to the timestamp of the NEXT line in merged
  (the next phrase's start, or a visual caption second). The
  speaker stopped earlier — Florence-2 sampled at 1fps, the visual
  line has nothing to do with where the speech ended.
- Rounding word timestamps to whole seconds because merged shows
  whole-second times. The merged display is for the editor's eye;
  the EDL writes the precise word-boundary value.
- Skipping the lookup on "obviously safe" cuts at the start / end
  of a phrase. Even those snap to `first_word.start` /
  `last_word.end` — never to `phrase.start` / `phrase.end` from
  the markdown views (which are rounded).

### When `transcripts/<stem>.json` is missing

Phase A preprocess didn't finish for that source — STOP and return
to the parent with a specific error naming the missing stem. Don't
guess at boundaries from the markdown views; that's exactly the
failure mode this section exists to prevent.

---

## Sentence integrity — splice grammar at every cut point

Word-boundary verification (above) makes sure cuts land on real word
edges. This section makes sure the resulting *splice* still reads as
language when the kept words on either side concatenate. A cut that
snaps cleanly to `word.end` and `word.start` per Hard Rule 6 can
still ship a broken sentence — *"…lets do the th- | w- we are
doing…"* satisfies the word-boundary rule and reads like garbage.
Both rules bind together.

### Default bias: keep sentences whole

Prefer cut anchors at **sentence boundaries** (a `.` / `?` / `!` in
the transcript text, OR a gap ≥ `min_silence_to_remove` from the
pacing preset between adjacent words). The cleanest cut is the one
between two complete sentences — no splice grammar to worry about,
no trailing filler to trim, no stutter to skip past.

A mid-sentence cut is **only** justified when at least one of these
holds:

- A pacing-preset silence gap inside the sentence is removable
  (intra-phrase `gap_ms ≥ min_silence_to_remove`). The cut just
  drops dead air; the words on either side were already adjacent in
  the speaker's intent.
- A filler / disfluency / false-start spans the cut zone (everything
  in the "Cut filler words" list below).
- A retake decision drops the earlier take and joins to a cleaner
  take of the same content (see "Retake detection").
- A time-squeeze stretch separates two 1x ranges of load-bearing
  speech (see "Time-squeezing").
- The user explicitly asked to remove the spanned content (verbatim
  conversation-bundle quote).

If none of those hold, **keep the sentence whole even if it's
slightly loose.** Loose-but-coherent reads better than tight-but-
broken every time.

### Splice-grammar test (mandatory before emitting any mid-sentence cut)

For every range whose IN-point is mid-sentence OR whose OUT-point is
mid-sentence (the previous / next concatenated range joins on a
mid-sentence boundary), read the kept words on each side together as
if no cut existed and verify:

1. **Does the concatenation read as English?** *"lets do the thing"*
   + *"now we are doing thing"* → fine. *"lets do the th-"* + *"w we
   are doing"* → broken (you cut into / out of a partial word).
2. **Is the FIRST word on the IN side a content-carrying word?**
   Land on a noun / verb / adverb / adjective, OR on a sentence-
   starting article / conjunction the speaker actually used. Don't
   land on a stutter onset (`"w-"`), a half-word fragment (`"th-"`),
   or a filler the speaker barely emitted (`"uh-"`, `"um-"`).
3. **Does the LAST word on the OUT side complete its phrase?** If
   the speaker said *"lets do the thing… uh"* and you want to cut
   here, the last KEPT word is *"thing"*, not the trailing *"uh"*.
4. **Is the speaker's intent preserved?** A cut that drops a
   sentence's subject and lands on a bare verb is not a successful
   cut, even when the words individually exist.

If any check fails, move the anchor one or two words inward (later
on the IN side, earlier on the OUT side) and re-verify.

### Trailing-filler trim (out-point side)

When picking `kept_last_word.end`, scan ~1-2s of transcript past your
candidate out-point for any of these — if found inside the window,
the last word **before** that material is the keeper:

- Filler tokens from the "Cut filler words" list below (`uh`, `um`,
  `like`, `you know`, `I mean`, `so yeah`, ...).
- Single-syllable false starts / abandoned partial words (entries
  whose `text` ends in `-`, or is a single phoneme-shaped fragment
  like `"th"`, `"wh"`, `"s"`).
- Aspirated breath sounds Parakeet sometimes captures as `"hh"` /
  `"hm"`.
- Trail-off mumble (sequences of 3+ very short words separated by
  < 100ms gaps; the speaker is fading, not landing).

The trail is the cut zone — pull `range.end` back to the last
content word's `end` and let the trail vanish in the splice.

### Stutter / false-start skip (in-point side)

When picking `kept_first_word.start`, scan ~1-2s before your
candidate in-point for the same patterns — if found, the first
word **after** that material is the keeper:

- Repeated stutter words: `["the", "the", "drill"]` → in-point at
  `drill.start`, not `the.start`.
- Cut-off partial words then redo: `["th-", "the", "drill"]` →
  in-point at `the.start` (or `drill.start` if the speaker's intent
  was to start with the noun).
- Abandoned fragment then restart: `["I", "uh", "we", "walked"]` →
  in-point at `we.start`; the splice drops the abandoned `"I uh"`.
- Filler-led sentence: `["um", "now", "we", ...]` → in-point at
  `now.start`.

### Worked example — splice grammar around a trailing filler

Speaker recording: *"lets do the thing… uh… now we are doing
thing"*. Editor wants the trail-off and the gap removed.

`transcripts/<S>.json` for the join zone:

```json
{"text": "thing",  "start":  8.10, "end":  8.42}
{"text": "uh",     "start":  9.05, "end":  9.18}
{"text": "now",    "start": 12.20, "end": 12.38}
{"text": "we",     "start": 12.40, "end": 12.50}
{"text": "are",    "start": 12.52, "end": 12.65}
```

PASS — kept_last_word = `"thing"` (not `"uh"`), kept_first_word = `"now"`:

```
range A end   = 8.42 + trail_margin   (clamped against the gap)
range B start = 12.20 - lead_margin   (clamped against the gap)
splice reads: "...lets do the thing | now we are doing thing"
```

FAIL — out-point ran past the trail and kept the `"uh"`:

```
range A end   = 9.18 + trail_margin   ← keeps the filler trail
splice reads: "...lets do the thing uh | now we are doing thing"
```

FAIL — in-point started inside the speaker's `"w-"` stutter (when
transcripts show a partial-word entry before `"we"`):

```
range B start = 12.10 (the "w-" stutter onset)
splice reads: "...lets do the thing | w- we are doing thing"
```

The fix in both failure cases is the same: walk one or two words
further into the transcript toward the cleanest content-word
boundary and re-snap the anchor to that word's `end` (out-point) or
`start` (in-point).

### Citation in the EDL `reason` field

When the splice-grammar pass shaped a range, note it tersely:

```
"reason": "Mid-sentence out-point at 'thing'; trimmed trailing 'uh' (9.05-9.18) before splice."
"reason": "Mid-sentence in-point at 'now'; skipped 'w-' stutter onset (12.10-12.18)."
```

Mandatory at `user_profile = professional`, recommended at
`creator`, optional at `personal` — but the splice-grammar
verification itself is mandatory at every profile. The note is the
audit trail; the verification is the rule.

### Reading transcripts efficiently

`transcripts/<stem>.json` can be large for long clips, but you only
need the words bracketing each candidate cut. Three approaches, any
of them fine:

- **Targeted Read** with `offset` / `limit` near the merged
  timestamp's byte range (binary search by reading a chunk and
  walking until `start` exceeds your target).
- **Grep** for the exact word text (`"text": "Geared"`) when the
  word is rare enough that the matches are unambiguous within the
  phrase span.
- **Full Read** when the file is small (< ~10k tokens); simpler
  and still cheap.

Whichever you pick, the rule is: read enough to confirm the words
that bracket your cut, then write the EDL with the exact `start` /
`end` values you found.

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
  is the editorial spine. Read `speech_timeline.md` end-to-end (per
  the dual-spine read mandate) for beat planning; **verify every cut
  anchor against `transcripts/<stem>.json` via `helpers/find_quote.py`
  per "Word-boundary verification" above before writing the EDL.**
  `speech_timeline.md` carries
  phrase-level start/end ranges — useful for scoping a phrase span,
  not for setting a sub-phrase cut point (no per-word times in that
  file).

- **Preserve peaks.** Laughs, punchlines, emphasis beats. Extend past
  punchlines to include reactions — the laugh IS the beat.

- **Speaker handoffs** benefit from air between utterances. The pacing
  preset's `lead_margin` + `trail_margin` largely sets this; only
  override per-handoff if the moment calls for it.

- **Visual context is the second source of truth.** Before committing
  to any non-trivial cut, check `[...]` visual caption lines around
  the cut point in `audiovisual_timeline.md`. If captions show
  continuous action
  spanning your cut, you're cutting mid-shot — usually fine, but be
  deliberate. Use the visual lane to find B-roll cutaway candidates,
  match cuts, shot changes, and to decide whether a moment is worth
  preserving even when speech is silent.

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
  words. Anything shorter than the threshold stays as natural rhythm.
  <30ms is always unsafe — mid-phoneme.

- **Cut filler words and disfluencies by default.** Treat each
  occurrence as an inline cut candidate exactly like a silence gap —
  split the EDL range around it so kept words concatenate cleanly:

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
  voice."** A clean tight delivery IS the speaker's voice with
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
  lead/trail margins don't re-introduce the filler you just cut).
  For zero-gap repeated words, snap to the END of the first instance
  / the START of the second — never mid-word.

- **Cut padding comes from the pacing preset**, not per-cut taste.
  Expand each range by `lead_margin` at the head and `trail_margin`
  at the tail. Hard Rule 7's 30-200ms working window still bounds
  anything outside the preset table — never go below 30ms.

- **Never reason audio and video independently.** Every cut must work
  on both tracks.

---

## In-clip editor notes — talkback baked into the source

Users sometimes record verbal directives **into the clip itself**,
addressed to a downstream editor before, between, or after takes.
Common shapes:

- **Preamble note before a take.** *"hey to the AI editing this,
  skip the first attempt — the second one's the keeper. Three,
  two, one…"* then the actual take begins.
- **Mid-clip note between takes.** *"…ugh, that was awful. Editor,
  just use the next one. Okay, take two — three, two, one…"*
- **End-of-clip note.** *"…and that's that. Note for the editor:
  if my hands shake on the close-up, cut to the wide."*
- **Pickup directive.** *"editor's note: skip ahead until I clap"*
  followed by the user clapping, the visual lane registering a
  hand motion at t=N, and the take starting after.

These are first-class user instructions — the user spoke them into
the recording so a downstream editor would find them. **Detect
them, honour them, exclude the preamble from the EDL, and surface
every one in your return rationale** so the parent echoes them
back to the user in plain English.

### Trigger-phrase detection

Walk `speech_timeline.md` (the editorial spine you read end-to-end
in step 1b) for any phrase the speaker uses to **address an editor
or AI**. For sub-second word timing on a matched phrase, hand the
phrase + speech-timeline range to `helpers/find_quote.py`. Match
liberally — case-insensitive,
tolerant of mis-transcription — but require **imperative or
instructional content after the address** before treating it as
a note.

Common opening phrasings (non-exhaustive — match the *intent*, not
a fixed list):

- *"hey, [to the] editor[s]"*, *"hey editor"*, *"yo editor"*
- *"hey, AI"*, *"hey, Claude"*, *"hey, the AI editing this"*
- *"note to [the] editor"*, *"note to AI"*, *"note to self"*
- *"editor's note"*, *"memo to editor"*, *"dear editor"*
- *"for the editor"*, *"for whoever's editing"*,
  *"to whoever's cutting this"*
- *"AI listen up"*, *"AI take note"*

Parakeet may mis-hear these (`"note the editor"` vs
`"note to the editor"`, `"hey AI editing"` vs `"hey AI editor"`).
Read for *intent*: if the speaker is clearly addressing a downstream
editor / AI vs the on-camera audience, the following content is a
note candidate.

### Boundary detection — where does the directive end?

The directive runs from the trigger to **the first of**:

1. **A take-start countdown.** *"three two one"*, *"3 2 1"*,
   *"in three two one"*, *"and three… two… one… action"*,
   *"and… action"*, *"okay rolling"*, *"okay take one"*,
   *"take two"*, etc. The countdown / call itself is also EXCLUDED
   from the EDL — nobody wants *"three two one"* in the cut.
2. **A clap or slate.** Look for an `(audio: clap …)` /
   `(audio: slate …)` event within ~3s of the trigger. Visual
   confirmation (`visual: …hands clapping…` on the same window)
   strengthens the signal but isn't required.
3. **A long silence gap** (≥ 1.5s) that clearly separates the
   address from the rest of the clip.
4. **An obvious topic / register shift** — speaker stops addressing
   the editor and starts addressing the audience (*"Hey everyone,
   today we're going to…"*).

If none of those land within ~10s of the trigger, the trigger was
probably a false positive (*"hey editor"* said rhetorically). Don't
treat as a note; let the words ride as content (still subject to
filler-cut rules).

### Exclusion from the EDL

Everything from the trigger through the take-start marker
(inclusive of countdown / *"action"* / clap window) is excluded.
Concretely: don't start an EDL range inside the preamble; place
the in-point at or after the take-start marker, snapped to a word
boundary per Hard Rule 6. The pacing preset's `lead_margin` still
applies on the chosen in-point — but never let it pull the in-point
back INTO the preamble. If `lead_margin` would re-include
*"…three two one…"*, clamp the in-point so the preamble stays out.

### Application — directive ranks alongside conversation quotes

Treat each detected directive as a first-class instruction:

1. **Things the user explicitly asked to keep / reject** in the
   parent's Conversation Context bundle (always wins — these are
   post-hoc, deliberate, may explicitly reverse an in-clip note).
2. **In-clip editor notes** detected in this read.
3. Default editorial rules (pacing, filler removal, etc.).

If an in-clip note conflicts with a conversation-bundle quote, the
conversation quote wins; note the override in the return rationale
(*"In-clip note from C0312 said 'use the first take' but the user's
later quote 'use the cleanest delivery' wins"*).

Common directive shapes and how to apply them:

- *"skip the first take, use the second"* → exclude first-take
  ranges, prefer second-take words.
- *"this take is bad"* / *"don't use this one"* → exclude that
  take's words entirely from candidates.
- *"cut after I say <X>"* / *"end on <X>"* → out-point on that word.
- *"start when I clap"* → in-point at the clap audio event /
  visual hand-clap frame.
- *"the wide shot is better than the close-up here"* → bias toward
  the wide-shot source for that beat.
- *"cut around me coughing at minute four"* → split the EDL range
  to drop the cough span.
- *"speed this up"* — only honour when `timelapse_mode = true`
  (else note the deferral; gating wins).

### Citation in the EDL `reason` field

When an in-clip note shaped a range, cite it in `reason` with
source stem and timestamp, quoting verbatim:

```json
{"source": "C0312", "start": 8.20, "end": 22.45,
 "beat": "DEMO",
 "audio_lead": 0.0, "video_tail": 0.0, "transition_in": 0.0,
 "quote": "...",
 "reason": "Second take per in-clip note (C0312 t=0.4s): 'skip the first take, the second one is the keeper.' In-point at first word after countdown."}
```

Mandatory at `user_profile = professional`, recommended at
`creator`, optional at `personal` — but the return-rationale report
below is mandatory regardless of profile.

### Return rationale — surface every detected note

Include a dedicated `In-clip editor notes` block listing every
detected directive, what you did, and any deviation:

```
In-clip editor notes detected:
  - C0312 t=0.4s "skip the first take, the second one is the
    keeper" — APPLIED. First-take range (C0312 0.0-7.8s) excluded;
    used 8.2-22.5s instead.
  - C0312 t=24.1s "if my hands shake on the close-up, cut to the
    wide" — APPLIED conditionally. Hands appeared steady on the
    close-up frames (visual: t=30s hands holding object steady),
    no cut to wide needed.
  - C0418 t=0.8s "for the editor, this whole clip is a throwaway"
    — APPLIED. C0418 excluded from all ranges.
  - C0507 t=12.3s "hey editor I think we should…" — IGNORED.
    Trigger fired but no clear directive followed; speaker
    transitioned into normal narration. Treated as content.
```

The parent will translate this into plain English for the user.

### Conservative handling — when in doubt, surface, don't act

If a directive is **ambiguous, contradictory, or impossible to
verify** against the timeline, **don't silently act on it.**

- *"editor cut around the embarrassing bit"* — what counts as
  embarrassing? Don't guess. Flag in rationale; preserve the take.
- *"the audio is bad on this one"* — you can't measure SNR from
  the timeline (CLAP / Florence-2 don't surface it). Flag; preserve.
- Two in-clip notes that contradict each other across takes
  (*"use the first"* in C0312 vs *"use the second"* in C0312
  pickup) — pick the **later** note (the user updated their
  preference) and flag the conflict.

Surfacing > silent guessing. The parent will ask one clarifying
question and re-spawn you with the clarification in the bundle.

### When the user wants to disable the feature

If a verbatim conversation-bundle quote says *"ignore any 'hey
editor' notes in the source"* / *"my brother yells 'hey AI' as a
joke, don't act on those"*, **respect the override** — treat all
in-clip notes as normal content this session and note the override
at the top of your return rationale.

---

## Retake detection — pick the cleanest take of repeated content

Real recordings contain **multiple takes of the same line**. The
speaker flubs, swears, restarts; or just naturally re-says
something a beat later because they didn't like how it landed.
Sometimes retakes happen across clip boundaries — one source ends
right before another picks up the same line. Your job: detect the
repetition, pick the cleanest take, drop the rest. The user
recorded the better take so you'd use it.

This is distinct from filler removal (per-word *"uh"* / *"um"*) and
distinct from in-clip editor notes (explicit verbal directives).
Retakes are an **implicit pattern**: the same words, twice, the
later one usually better.

### Signals that a retake just happened

Walk the speech lane looking for these patterns:

1. **Frustration marker followed by similar content.** A curse word
   (*"fuck"*, *"shit"*, *"damn"*, *"crap"*, *"bollocks"*), a
   self-disgust noise (*"ugh"*, *"argh"*, *"god"*, *"jesus"*), or a
   self-correction phrase (*"no no no"*, *"nope"*, *"hold on"*,
   *"wait"*, *"sorry"*, *"let me try that again"*, *"one more
   time"*, *"start over"*, *"take two"*, *"again"*) **followed
   within ~10s by speech that paraphrases or repeats the
   immediately preceding utterance.** The frustration marker is the
   strongest single retake signal — when you see one, look both
   ways for the matching pair.

2. **Semantic repetition without an explicit marker.** Two
   utterances within ~30s that say substantially the same thing in
   different words (or the same words). Compute rough similarity by
   hand: shared content nouns / verbs / named entities, similar
   sentence shape, similar information density. Bag-of-content-
   words overlap ≥ 50% with no intervening topic shift is a strong
   cue.

3. **Cross-clip retakes.** Two adjacent sources starting with
   similar content. Source ordering in `audiovisual_timeline.md`
   and `speech_timeline.md` is the parent's argv order; if
   filenames suggest sequence (`C0312` then
   `C0313` recorded back-to-back, or `intro_take1.MP4` then
   `intro_take2.MP4`), the second is almost always the keeper for
   any overlap.

4. **An explicit slate / clap separating the takes.**
   `(audio: clap …)` or `(audio: slate …)` between two utterances
   of the same content — that's a deliberate retake marker.

5. **Long pause followed by restart.** A silence gap ≥ 2s followed
   by speech restating what came before the pause (*"…welcome to
   the show. [3.4s silence] Welcome to the show, today we're…"*).

### How to pick the keeper

Default heuristic: **prefer the LATER take.** The later take exists
because the speaker decided the earlier wasn't good enough —
respect that decision. Concretely, when you have two semantically
matched ranges, exclude the earlier from the EDL and use the later.

Override the default when:

- An in-clip editor note explicitly says *"use the first take"* —
  in-clip notes win (per the application priority above).
- The later take is *worse* on objective signals: more fillers
  than the earlier, more silences, more false starts, OR the
  speaker visibly / audibly dropped energy. Compare on:
    - filler-word count after silence-pass
    - shortest false-start span
    - total speech duration vs target word count (longer ≠ better;
      tighter delivery wins)
  If the EARLIER take wins decisively on these signals, use it
  instead and note the override in `reason`.
- The takes diverge in *meaning*: one ends with a punchline the
  other lacks; one introduces a named subject the script needs;
  one is the *"with the joke"* version the user explicitly asked
  to keep. Trust the conversation bundle on this — verbatim quotes
  saying *"keep the punchline take"* override the default.
- The keeper take fails the structural test (cuts off mid-thought;
  the speaker walks out of frame; visual continuity breaks). Drop
  to the alternate, note in `reason`.

### When repetition is INTENTIONAL — keep both

Not every repetition is a retake. Watch for:

- **Rhetorical / emphatic repetition.** *"Buy now. Buy now. Buy
  NOW."* — rhythm IS the beat. Keep all three.
- **Comedic repetition / callback.** *"…and then he says 'no'.
  No. Just no."* — beat structure depends on it.
- **List repetition.** *"It's fast, it's faster, it's the
  fastest."* — kept for the ladder.
- **Speaker quoting another speaker.** *"He said: 'we'll never
  ship.' We'll never ship."* — voice / tone change distinguishes.

Disambiguation cues:

| Cue                              | Retake | Intentional |
|----------------------------------|:------:|:-----------:|
| Frustration marker between       |   X    |             |
| Long pause (≥ 2s) between        |   X    |             |
| Slate / clap between             |   X    |             |
| Visible camera reset / re-frame  |   X    |             |
| Identical wording, no pause      |        |     X       |
| Escalating pitch / energy        |        |     X       |
| Speaker tone shift (joke beat)   |        |     X       |
| Explicit *"again"* / *"take two"*|   X    |             |

When ambiguous, **lean toward keeping both** — losing intentional
emphasis is a more visible bug than keeping a single redundant
sentence. Note the call in `reason`.

### Cutting mechanics for retakes

When you've identified an earlier-take range to drop and a later-
take range to keep, the mechanics match any other inline cut (see
"Cut craft" above):

- Both surviving / dropped boundaries snap to word boundaries
  (Hard Rule 6).
- The frustration marker, the curse word, the *"let me try that
  again"* — **all excluded** from the EDL. They were connective
  tissue between takes; nobody wants them in the cut.
- The combined-pad clamp from the silence-removal pass applies on
  the gap between the kept earlier audio (before the dropped
  retake) and the kept later audio (the keeper take), so margins
  don't bleed back into the dropped span.
- If the retake straddles a clip boundary (one source ended, next
  picked up), emit two adjacent EDL ranges from different sources;
  the FCPXML exporter handles same-track concatenation natively.

### Citation in the EDL `reason` field

When a retake decision shaped a range, cite the rejection reason:

```json
{"source": "C0312", "start": 14.20, "end": 22.45,
 "beat": "INTRO",
 "audio_lead": 0.0, "video_tail": 0.0, "transition_in": 0.0,
 "quote": "Welcome back to PAX East today we're at...",
 "reason": "Second take of intro (C0312 14.2-22.5s); first take at C0312 4.1-12.0s rejected — speaker said 'fuck, again' at 11.4s and restarted with cleaner delivery."}
```

### Return rationale — surface every retake call

Include a dedicated `Retake decisions` block in your return:

```
Retake decisions:
  - INTRO beat: kept C0312 14.2-22.5 (later take); dropped
    C0312 4.1-12.0 (first take) — speaker said "fuck, again"
    at 11.4s indicating a deliberate restart.
  - DEMO beat: kept C0312 first delivery 32.0-41.5; later
    delivery at 45.0-54.2 had three "uh" fillers vs zero on
    the first, so the EARLIER take won (override on default).
  - PUNCHLINE: kept BOTH instances of "we'll never ship"
    (C0418 8.0-9.5 and 9.5-10.8); rhetorical emphasis,
    no frustration marker between, escalating pitch.
  - Cross-clip: C0420 begins with the same line that ends
    C0419 — kept C0420's version (later, cleaner).
```

The parent surfaces these to the user.

### Conservative handling for retakes

- **If you can't decide which take is cleaner**, default to the
  later one and flag the call in the rationale so the parent knows
  you guessed.
- **If both takes contain unique content** (one has a sentence
  the other lacks), keep BOTH and let the speech ride — better to
  over-include than silently drop a beat the speaker actually
  meant to land.
- **Never cut around a frustration marker without confirming a
  matching restart within 10s.** A standalone *"fuck"* with no
  retake might just be the speaker's natural reaction to something
  on camera — that's content, not retake noise. Filler-word rules
  don't list curse words as default-cut for that reason.
- **Cross-clip retake detection requires temporal evidence.** Two
  clips with similar speech aren't necessarily takes of each
  other — they might be different days, different scenes. Use clip
  stem ordering (numeric or `_take1` / `_take2` suffixes), an
  explicit user quote about retakes, or a slate / clap audio event
  before excluding an entire clip as a "rejected take."

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

## Dual-mic pair handling (when `source_pairs.json` mode = `dual_mic`)

When the parent's brief lists `source_pairs.json` with `mode =
dual_mic`, you have **two transcripts per paired shot** capturing
the same speech via two different microphones. Your job: treat the
pair as one shot for cut purposes, pick the better transcript per
cut, and record which one you used so the parent can surface it in
the user-facing summary.

### What appears in the timeline files

For a pair with stem `SHOT_0042`, both `audiovisual_timeline.md`
and `speech_timeline.md` contain TWO source sections:

```
=== SHOT_0042 (00:01:23 — 00:01:48) ===
  00:01:24.10  "we're walking past the Riot booth now"
  ...
  visual: ...

=== SHOT_0042.audio (00:01:23 — 00:01:48) ===
  00:01:24.10  "we're walking past the Riot booth now"
  ...
```

Both sections cover the SAME duration on the SAME wall-clock
timeline. The `.audio` suffix tells you which one came from the lav /
external recorder. The video sibling has visual captions; the
`.audio` sibling doesn't (no visual lane).

### Picking the better transcript

For each pair, decide which transcript to trust per cut. Decision
order:

1. **Confidence** — Parakeet emits per-word confidence in
   `transcripts/<stem>.json` (`words[].conf`). Compute a windowed
   mean confidence across the words inside the candidate cut range
   for both transcripts. Higher mean wins.

2. **Word-error proxies** — if both confidences are similar (within
   ~0.05), prefer the transcript with FEWER repeated single-letter
   tokens, fewer `[UNK]` placeholders, and stable spelling on
   technical / proper nouns mentioned elsewhere in the timeline.
   These are typical artefacts of the noisier mic.

3. **User explicit override** — if the user said in the brief
   "always use the lav" or "the camera audio is unusable on
   `SHOT_0042`", that quote wins regardless of confidence math.

4. **Tie-breaker** — prefer the lav (`.audio` sibling). External
   recorders are almost always cleaner than on-camera mics; this
   matches the user's hardware intent for setting up the rig that
   way in the first place.

### What goes in the EDL

The EDL range still points at the **video file** for both video and
audio (a paired shot is still one shot at the NLE level). The
`source` field is the video stem, NOT the `.audio` alias — the alias
is a preprocessing fiction so caches don't collide; the user never
wants the alias path showing up in their NLE bin.

**Do** record which transcript you trusted for each range, in a new
EDL field:

```json
{
  "source": "SHOT_0042.mp4",
  "in":  82.31,
  "out": 89.04,
  "speech": "we're walking past the Riot booth now",
  "preferred_transcript": "SHOT_0042.audio",
  "preferred_transcript_reason": "lav 0.93 vs camera 0.71 mean conf"
}
```

The `preferred_transcript` field is OPTIONAL on unpaired shots and
MUST be present on paired shots. The parent forwards it to the user
in the cut summary so the user knows whether to manually patch the
audio source in their NLE for that range.

### When to call out a swap-the-audio recommendation

If you preferred the lav transcript on a range AND the confidence
gap was wide (camera mean conf < 0.6 or > 0.20 below the lav), add
a QA note recommending the user replace the audio track on that
range with the paired `.wav` in their NLE. Premiere / Resolve /
FCPX all support per-clip audio source replacement; the user's
preprocessed lav is at `audio_alias_path` from `source_pairs.json`.
Example QA note:

```
QA: SHOT_0042 [82.31 - 89.04] — camera audio is muddy
    (mean conf 0.58, transcript: "wal'in past the riid booth").
    Lav transcript is clean (mean conf 0.93). RECOMMEND: in your
    NLE, replace audio for this range with
    edit/.paired_audio/SHOT_0042.audio.wav.
```

Don't swap unilaterally — the EDL stays pointing at the video file,
and you tell the user. They may have a specific reason to keep the
camera audio (room tone match with surrounding cuts, etc.).

### Anti-patterns specific to dual-mic

- **Placing both members of a pair as separate cuts.** They're the
  same shot. If you find yourself emitting two ranges for `SHOT_0042`
  and `SHOT_0042.audio` covering overlapping wall-clock spans, you've
  treated the alias as a real source — go back, merge, pick one
  transcript, write one range pointing at the video file.
- **Pointing the EDL `source` at the `.audio` alias.** The alias
  lives under `<edit>/.paired_audio/` and isn't a real user-facing
  asset. Always point at the video file.
- **Picking by transcript length / vibes.** Use the confidence math
  first. Length proxies for confidence only weakly and you'll pick
  the wrong one on shots where the lav had a wind hit but the
  camera mic was fine.
- **Forgetting `preferred_transcript` on paired shots.** It's
  required output on those ranges; the parent's user-facing summary
  is built from it.

---

## B-roll scout spawn protocol (when `b_roll_mode = true`)

You may delegate the per-beat b-roll shortlisting work to a **b-roll
scout sub-agent**. Scouts are sub-subagents — you spawn them, they
return ranked candidate shortlists, you pick / verify / write the
EDL ranges. The scout's operating manual is
`references/subagent_broll_scout_rules.md`; it reads
`<edit>/visual_timeline.md` for in-scope sources and returns
shortlists with evidence.

This is **optional** — for small libraries you do the work yourself
in your own context window. The protocol below tells you when
scouts are worth it.

### When to spawn scouts

Spawn a scout (or batch of parallel scouts) when ANY of:

- **Library is large.** `>50` b-roll-eligible clips, or the dual
  spine (`audiovisual_timeline.md` + `speech_timeline.md`) strained
  your read budget. Per-beat re-scanning of visual lines is wasteful
  at this size.
- **`user_profile = professional`.** Top-candidate review on every
  named-subject beat is mandatory; offloading shortlisting to scouts
  + you doing verification/selection yields better QA notes.
- **Many beats need shortlisting.** When scripted mode has 8+ beats
  with named-subject b-roll, parallel scouts (Hard Rule 10) finish
  faster than sequential in-context scanning.
- **Ambiguous beat.** A specific beat where the visual evidence in
  the AV view didn't decide it — a scout's fresh-context
  visual_timeline read can surface candidates you missed.

For small libraries (`<= 30` clips) and `personal` / `creator` bar,
do the work in-context; spawn overhead isn't worth it.

### Spawning N scouts in parallel

Per Hard Rule 10, when spawning multiple scouts in one batch (e.g.
one per beat), spawn them **in parallel** via the agent / Task
tool, not sequentially. Total wall time approximates the slowest
scout, not the sum.

### Scout brief shape

Build the scout brief with these sections (the scout's rules file
documents the full input shape; this is the fill-in-the-blanks
template):

```
You are the B-ROLL SCOUT sub-agent for a video-use-premiere session.
You have a fresh context window. Use it.

STEP 0 (mandatory):
  Read references/shared_rules.md IN FULL.
  Read references/subagent_broll_scout_rules.md IN FULL.

CONVERSATION CONTEXT (forwarded from editor):
  Project summary: <as forwarded by parent to me>
  Verbatim user quotes (relevant to b-roll selection):
    [t=...] "<quote>"
    ...
  Things the user explicitly asked to keep:
    - "<quote>"
  Things the user explicitly rejected:
    - "<quote>"

MODE FLAGS:
  script_mode    = <true|false>
  b_roll_mode    = true                # otherwise we wouldn't be here
  timelapse_mode = <true|false>        # binds whether you may
                                       # suggest timelapse-shaped ranges
  user_profile   = <personal|creator|professional>

SOURCE CONSTRAINTS:
  source_tags.json:  <edit>/source_tags.json (or "(not present)")
  In-scope source stems for THIS shortlist:
    [<list of stems eligible for b-roll candidates>]
  Out-of-scope (explicitly excluded): [<list>]

INPUTS YOU MAY READ:
  - <edit>/visual_timeline.md (your default reading surface; read
    in full for in-scope sources)
  - <edit>/visual_caps/<stem>.json (drill-down for raw frame
    captions on a candidate)
  - <edit>/transcripts/<stem>.json (confirm low speech if a clip
    looks like A-roll mis-filed)
  - <edit>/clip_index/index.json (if path forwarded; shortlist aid)

BEATS TO SHORTLIST:
  Beat 1:
    label:                 <BEAT_NAME>
    subject:               <"Riot Games booth signage">
    target_duration_s:     <seconds>
    vo_start_s:            <vo timestamp on output timeline> (scripted-mode only)
    vo_end_s:              <...>
    vo_subject_word_t_s:   <...>      (when the named subject is spoken)
    keywords:              [<riot, games, booth, sign, banner, logo>]
    notes:                 <"user explicitly asked to land on the booth on the booth">
  Beat 2:
    ...

TOP-N PER BEAT: <typically 3-8>

OUTPUT:
  Return the JSON block per references/subagent_broll_scout_rules.md
  "Return format" plus a 3-6 sentence rationale. Include
  subject_visible_t_s on every candidate so I can compute the
  in-point per scripted.md step 6.
```

### When to NOT spawn scouts

- Tiny libraries (<= 5 clips) — read both timelines once, decide.
- Pure talking-head with one A-roll source and 1-2 cutaways — same.
- The user explicitly said "I want this fast, don't over-engineer
  it" — note in your return that scouts were skipped per the user's
  quote.

### Using scout returns

When the scout returns:

1. Read the scout's JSON shortlist + rationale.
2. For each beat, pick the top-ranked candidate passing YOUR
   verification (drill into `audiovisual_timeline.md` +
   `speech_timeline.md` around the candidate range; check there's
   no audio-event or speech conflict you care about).
3. If the top candidate fails verification, descend to candidate 2,
   3, ... — that's why you got a shortlist.
4. Compute the source in-point per `scripted.md` step 6 using the
   scout's `subject_visible_t_s`.
5. Write the EDL range with a QA note in `reason` listing the
   scout's rejected alternatives (when `user_profile =
   professional`) — that's detailed-QA discipline.
6. Note in your final return rationale that scouts were used and
   how many; the parent surfaces this to the user if they ask
   how the cut was made.

If a scout returns `BUDGET_EXHAUSTED`, narrow its in-scope source
list (drop sources clearly out of category) and re-spawn. If it
still exhausts, fall back to in-context scanning for that beat —
scouts are an optimization, not a hard requirement.

---

## Time-squeezing (timelapse)

> **Gated by `timelapse_mode` in the parent's brief.** This entire
> section applies ONLY when `timelapse_mode = true`. When
> `timelapse_mode = false` (the default), emit zero ranges with
> `speed != 1.0` — even if a stretch looks textbook timelapse-
> shaped. The user opted out for a reason (often: "the b-roll IS
> the visual track at 1x; don't compress it"). Skip this section
> when the flag is false; don't invent retime decisions the user
> didn't authorize. If `source_tags.json` exists and a clip is
> tagged `timelapse`, that clip is OBVIOUSLY pre-meant for retime —
> but `timelapse_mode = false` still wins (the user might want the
> clip in 1x for THIS session). Honour the flag.

Real-world footage is often "1 minute of explanation, then 25 minutes
of silently doing the work, then 2 minutes of wrap-up." Cutting the 25
minutes throws away the visual story; keeping it 1x bores the viewer.
The third option is **time-squeezing**: compress the work segment into
a 5-30s timelapse on the output timeline.

### When to reach for it

Look for stretches in `audiovisual_timeline.md` (cross-checking
`speech_timeline.md` for any speech you'd be silencing) where BOTH
are true:

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

- **Load-bearing speech** (instruction, explanation, narration
  carrying the cut, the punchline landing the beat): split AROUND
  it. Emit a 1x range for the words, then a `speed > 1.0` range for
  the silent / no-words-that-matter middle, then another 1x range for
  whatever talks next.
- **Filler speech** (mumbling, swearing at a misplaced screw, idle
  narration of "okay ... there we go ... hmm"; 30 minutes of casual
  chatter while building that isn't actually teaching anything):
  squeeze right over it. With `audio_strategy="drop"` (the default at
  `speed != 1.0`) the words vanish along with room tone, the visual
  story plays compressed, and the viewer thanks you.

When in doubt: lean toward squeezing over filler vs splitting into a
hundred tiny 1x ranges. The video is for the viewer.

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
  with a specific reason to keep source audio (recognisable voice
  in the background, distinctive ambient texture). FCPXML / xmeml
  output gets a matching retime; editor must toggle "Maintain Audio
  Pitch" in the NLE.

### Discipline

- **Decide per-stretch: is the speech worth keeping?** Load-bearing
  earns a 1x split around it; filler gets squeezed over with `drop`.
- **Cut FIRST, squeeze SECOND.** Apply the silence-removal pass first;
  then identify surviving long stretches fitting the criteria; then
  squeeze. Squeezing dead air is just slower nothing.
- **Word-boundary discipline still applies on adjacent 1x ranges**
  (Hard Rule 6 / 7). The squeezed range itself doesn't need word-
  boundary alignment when `audio_strategy="drop"`, but pad it
  generously (~1-2s each side of the activity).
- **`speed` field is OPTIONAL and defaults to 1.0.** Untouched EDLs
  behave exactly as before. Only emit `speed` when actively squeezing.
- **The retime key is `speed` — NOT `timelapse_speed`, NOT
  `clip_speed`, NOT `retime`.** Recurring agent footgun: beats named
  `*_TIMELAPSE` invite an autocomplete-style `"timelapse_speed": 8`
  the exporter can't recognise. The export pipeline does a defensive
  textual rename of `timelapse_speed -> speed` before parsing, but
  do NOT rely on it — write the canonical key first time. The
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
transition_in = 0.0` on EVERY range. Don't use J-cuts, L-cuts, or
cross-dissolves under any circumstances.

The EDL schema still accepts these fields and the FCPXML exporter
still consumes them — but you must emit `0.0` for all three. The
30ms `afade` pair at every boundary (Hard Rule 3) is the only audio
crossfade available right now and suffices to suppress boundary
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

`preferred_transcript` and `preferred_transcript_reason` are OPTIONAL
on unpaired shots and **REQUIRED on paired shots** (whose `source`
stem appears in `<edit>/source_pairs.json` with `mode = dual_mic`).
Schema:

```json
{"source": "SHOT_0042", "start": 82.31, "end": 89.04,
 "beat": "HOOK",
 "preferred_transcript": "SHOT_0042" | "SHOT_0042.audio",
 "preferred_transcript_reason": "lav 0.93 vs camera 0.71 mean conf",
 "quote": "...", "reason": "..."}
```

The `source` always points at the video stem (never the `.audio`
alias — that's a preprocessing fiction, not a user-facing asset).
The `preferred_transcript` field tells the parent which mic was
trusted so the user-facing summary can recommend an in-NLE audio
swap when the camera audio was the loser. See "Dual-mic pair
handling" above for the decision rules.

**No `grade` field** — color is out of scope. The skill emits XML
for the NLE; the colorist applies the grade there.

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
   Note explicitly so the parent can flag it to the user.

The parent will translate this into plain English for the user. Be
factual, be terse on the report, but thorough on the EDL itself.

---

## Editor-specific anti-patterns

- **Skipping either timeline read** to "save time." Violates the
  ABSOLUTE READ MANDATE at the top of this file. BOTH
  `audiovisual_timeline.md` AND `speech_timeline.md` are mandatory
  end-to-end reads, every spawn. The cut is silently bad if you skip
  one, and the user will catch it.
- **Emitting an EDL range without word-boundary verification via
  `helpers/find_quote.py` against `transcripts/<stem>.json`.** This
  is the mid-sentence-cut footgun — the AV view has no speech, the
  speech timeline rounds outward to whole seconds (it CONTAINS the
  word span but doesn't pin it), only `transcripts/<stem>.json`
  carries the per-word `{start, end}` Hard Rule 6 binds you to, and
  `find_quote.py` is the supported interface for crawling it. Verify
  the words bracketing
  every cut anchor; snap `range.start` to `kept_first_word.start`
  and `range.end` to `kept_last_word.end` before applying margins.
  See "Word-boundary verification" above.
- **Snapping a cut to a phrase end shown in `speech_timeline.md`
  without confirming it matches the last word's `end` in transcripts.**
  Speech_timeline rounds to whole seconds; the underlying
  `last_word.end` is sub-second. Look it up.
- **Setting `range.end` to the next merged-line timestamp** (next
  phrase start, or a Florence visual caption second). The speaker
  stopped earlier; visual captions sample at 1fps and have nothing
  to do with where speech ended.
- **Cutting mid-sentence without splice-grammar verification.** A
  word-boundary-clean cut that splices *"…lets do the th-"* to
  *"w- we are doing…"* still ships a broken sentence. Read the
  kept words on each side together; verify the concatenation reads
  as English; trim trailing filler / skip leading stutter per
  "Sentence integrity".
- **Defaulting to mid-sentence cuts when a sentence boundary is
  available.** The default bias is whole sentences; mid-sentence
  cuts are reserved for removable silence, fillers, retakes,
  time-squeeze gaps, or explicit user requests. Loose-but-coherent
  reads better than tight-but-broken.
- **Keeping a trailing `"uh"` / `"um"` / partial-word at the
  out-point of a mid-sentence cut.** Trail-off material is the cut
  zone — pull `range.end` back to the last content word.
- **Starting a mid-sentence in-point inside a stutter onset / on a
  redundant `"the the"` / on `"w- we"`.** The first kept word is
  the one that carries the sentence forward; skip the stutter and
  in-point on the content word.
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
- **Squeezing pure dead air** vs cutting it. Time-squeezing is for
  visually continuous activity, not for empty rooms.
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
- **Emitting `speed > 1.0` on a `timelapse_mode = false` brief.**
  Even one retime range silently violates the user's opt-out.
  Cut the stretch or keep 1x; never retime without permission.
- **Spawning b-roll scouts sequentially.** When you spawn N
  scouts, spawn them in parallel (Hard Rule 10). Sequential
  spawning forfeits the parallelism making scouts worthwhile.
- **Trusting a scout's top candidate without verification.** The
  scout shortlists; you decide. Verification (drilling
  `audiovisual_timeline.md` + `speech_timeline.md` around the
  candidate range) still binds.
- **Ignoring `source_tags.json` when proposing b-roll candidates.**
  A-roll-tagged clips don't become cutaways unless the user
  explicitly authorized it. The user organized for a reason.
- **Spawning scouts on tiny libraries or single-source projects.**
  Spawn overhead exceeds savings. Do the work in-context.
- **Including a *"hey editor"* preamble in the EDL.** Trigger
  phrases + countdown / clap markers are exclusion zones; the
  in-point belongs at or after the take-start marker.
- **Acting silently on an in-clip note.** Every note you applied
  (or chose to skip) goes in the `In-clip editor notes` block of
  your return rationale. The user spoke it into the recording so
  they'd hear it land back in the cut.
- **Acting on an ambiguous in-clip directive.** *"Cut around the
  embarrassing bit"* / *"the audio is bad here"* — flag, don't
  guess. The parent can ask one question and re-spawn you with
  the clarification.
- **Cutting the LATER take when the speaker swore at the
  earlier one.** Frustration markers are restart signals; the
  later take is the keeper unless objective signals say otherwise.
- **Treating rhetorical / emphatic repetition as a retake.**
  *"Buy now. Buy now. Buy NOW."* is a beat, not three takes of
  one line. When in doubt, keep both.
- **Cutting around a curse word without confirming a matching
  restart.** A standalone *"fuck"* may be content, not retake
  noise. Look both ways before cutting.
- **Cross-clip retake decisions without temporal evidence.** Two
  clips with similar speech might be different scenes entirely.
  Confirm via stem ordering / recording metadata / explicit user
  quote before excluding a whole clip as rejected.
- **Failing to surface retake decisions in the return rationale.**
  The user wants to know which take landed and why; the
  `Retake decisions` block is mandatory whenever a retake call
  influenced the cut.
- **Treating a `dual_mic` paired shot as two independent sources.**
  `SHOT_0042` and `SHOT_0042.audio` are the same shot captured by
  two mics; they share a wall-clock timeline and you must NOT place
  both as separate cuts. See "Dual-mic pair handling" — read
  `source_pairs.json` whenever its path is in the brief.
- **Pointing the EDL `source` at a `.audio` alias.** The alias
  exists only to keep preprocess caches separate; it lives under
  `<edit>/.paired_audio/` and isn't a user-facing asset. EDL ranges
  always point at the video stem.
- **Omitting `preferred_transcript` on paired shots.** Required on
  every range whose source stem appears in `source_pairs.json` with
  `mode = dual_mic`. Without it the parent can't tell the user
  which mic landed in the cut, and the recommendation to swap audio
  in their NLE never reaches them.
- **Picking the dual-mic transcript by vibes / length.** Use the
  windowed mean per-word confidence first; lean on the lav as
  tie-breaker. Length isn't a quality signal.
