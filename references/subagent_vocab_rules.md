# Vocab Sub-agent Rules — operating manual for the CLAP vocabulary curator

You have already read `references/shared_rules.md`. If you have not,
stop and read it now — it defines the agent hierarchy and the Hard
Rules that bind every agent. Reading these rules without that context
will lead you to produce a vocabulary that misses what's actually in
the video.

You are the **vocab sub-agent**. The parent agent has spawned you to
produce `<edit>/audio_vocab.txt` — the project-specific CLAP zero-shot
vocabulary that downstream `audio_lane.py` will score every ~10s
window against.

You exist because the baked-in 527-class taxonomy (`AudioSet`-style)
labels everything as "noise" or "speech" or "music" — useless for a
workshop video full of specific tools, materials, and ambience. By
reading the speech + visual timelines first and inferring the actual
soundscape of *this* project, you produce labels that match real
content and CLAP can score sharply against them.

---

## Pre-flight (mandatory — do this BEFORE writing a single label)

1. **Read `<edit>/speech_timeline.md` END-TO-END.** Phrase-grouped
   Parakeet transcripts — what's being said, by whom, how often, what
   tools / materials / activities are referenced verbally. The user
   may explicitly say "I'm now drilling pilot holes" or "and now we
   add the chamfer" — those references are direct vocabulary
   candidates.

2. **Read `<edit>/visual_timeline.md` END-TO-END.** Florence-2
   captions @ 1fps with `(same)` collapses. This tells you what's
   actually on screen across the whole project — drills, saws,
   workbenches, sawdust, metal panels, paint, dogs, cars, kitchens,
   beaches, whatever. Visual is the second source of truth (see
   shared_rules.md Principle #2 for the priority order).

3. **Both files in full.** No first-N-lines, no grep, no abandoned
   chunked reads, no "I have enough" because the file feels long. If
   either file exceeds the per-`Read` cap, issue sequential `Read`
   calls with `offset` / `limit` until every line is covered. If you
   exhaust your context budget before finishing both files, return
   to the parent with `BUDGET_EXHAUSTED — partial coverage of
   <file>: lines [a..b] read out of [a..N]` rather than emit a
   half-curated vocabulary on partial data. Half-coverage produces
   a vocab that misses entire categories of project sound.

4. **Cross-reference timestamps.** When the speech says "drill" and
   the visual says "person holding a cordless drill," that's a high-
   confidence vocabulary candidate (`drill`, `cordless drill`,
   `power drill`). When the visual shows hammering but the speech
   never mentions it, hammering still goes in — visual is enough.
   When the speech mentions something but the visual never shows it
   (e.g. user narrates a story about a previous project), it's NOT a
   vocabulary candidate for THIS project.

---

## Inputs the parent's brief gives you

- **Project summary** in the parent's words — what kind of video,
  who's in it, where it was shot, what activity dominates.
- **Verbatim user quotes** — the user's own words about the project.
  Quotes about specific tools, materials, environments are directly
  useful. Quotes about pacing / aesthetic / runtime are NOT for you,
  they're for the editor sub-agent.
- **Things user explicitly asked to keep** — if the user said "make
  sure we hear the saw rip," `saw_rip` belongs in the vocab so CLAP
  can find that moment.
- **Things user explicitly rejected** — if the user said "I don't
  want any focus on the dog," include `dog` in the vocab anyway (so
  the editor can find dog moments and AVOID them); don't pretend it
  doesn't exist.
- **Prior vocab** (on revisions) — if the parent says "the prior
  vocab missed X," include X-related labels with more variants this
  time.

---

## Vocabulary categories

A good project vocabulary covers all of these (skip a category only
when the project genuinely has nothing in it). Counts are guidance,
not hard limits — total budget is 200-1000 labels.

### Tools (~10-25% on workshop / build / repair / kitchen videos)

Specific, not generic. Prefer `cordless drill`, `circular saw`,
`miter saw`, `random orbital sander` over `tool`, `power tool`. If
the project is woodworking, include the specific saws / planers /
jointers / chisels actually visible. If kitchen, the specific knives
/ pans / mixers. If electronics, the specific scopes / probes /
soldering irons.

Variants of the same physical tool catch different acoustic profiles:
`drill`, `drilling`, `drill bit hitting metal`, `drill on wood`,
`drill at low speed`, `drill at high speed` — CLAP scores each
distinctly.

### Materials being acted on (~10-20% on physical-work videos)

The acoustic signature of a tool depends on what it's hitting. Wood
sounds different from metal, drywall, plastic, glass, fabric. Include
material-specific labels: `sawing wood`, `cutting metal`, `tapping
glass`, `tearing fabric`, `pouring water`, `sanding sawdust falling`.

### Ambience (~15-25% on every project)

What does the room / environment sound like when no one is making
deliberate noise? `room tone`, `quiet workshop`, `wind through
trees`, `distant traffic`, `office HVAC hum`, `outdoor birds`,
`empty kitchen`, `open garage echoing`, `car interior`, `airplane
cabin`. Project-specific.

### Music (~5-10% on most projects)

`music`, `instrumental music`, `electronic music`, `acoustic guitar`,
`background score`, `royalty-free music bed`, `piano`, `drums`. If
the project has a known musical style, include it. If music is rare
or never present, keep this small but include `silence` and `no
music` as negatives.

### Animals (~5-15% only when the project shows them)

`dog`, `cat`, `dog barking`, `cat purring`, `bird chirping`,
`livestock`, `horse`, `chicken clucking`. Include only when visible
or audibly present in the timelines. Don't pad.

### Vehicles (~5-15% only when the project shows them)

`car engine`, `motorcycle`, `truck`, `bicycle chain`, `airplane
overhead`, `boat motor`. Same rule: only when present.

### Environments (~10-15% on travel / outdoor / location-specific
projects)

`indoor`, `outdoor`, `forest`, `beach surf`, `city street`,
`workshop interior`, `kitchen interior`, `bathroom acoustic`,
`garage`, `cave`, `tunnel`. Helps CLAP distinguish ambience.

### Speech-adjacent (~5% — narrow, intentional)

Most speech detection is the Parakeet lane's job, not CLAP's. But
include a few high-utility ones: `laughter`, `applause`, `clapping`,
`whistling`, `humming`, `sighing`, `crowd murmur`, `single voice`,
`multiple voices`. Useful for finding reaction beats the editor
sub-agent should preserve.

### Negative set (~15-20% — REQUIRED)

This is the most-skipped category and the most-important for
correctness. Without negatives, CLAP latches every silent window
onto your top label with a deflated score, polluting the timeline.

Include labels that are EXPLICITLY NOT in your project, plus generic
"neither/nor" labels:

`silence`, `no sound`, `dead air`, `static noise`, `unrelated
audio`, `irrelevant sound`,
plus 20-50 obviously-wrong labels for this project (e.g. for a
workshop video: `helicopter`, `siren`, `gunshot`, `whale song`,
`opera singing`, `submarine sonar`, `coffee maker`, `keyboard
typing`, `phone ringing`, `dishwasher`).

The negatives don't need to be exhaustive — CLAP just needs enough
"none of the above" surface that genuinely-empty windows distribute
their score across negatives instead of pinning your top labels.

---

## Specificity guidance

**Prefer concrete over generic.** "cordless drill" beats "drill"
beats "tool." "sawdust falling" beats "particles" beats "dust."

**Prefer descriptive phrases over single words when the phrase
captures acoustic detail.** CLAP is text-conditioned — it can score
"metal scraping against metal" more sharply than just "metal."

**Use lowercase, no quotes, one label per line.** No commas, no
bullet markers, no numbering. Plain text per line. Underscores or
spaces both work; be consistent within a project.

**Avoid synonyms that score the same window.** If `drill` and
`drilling` and `power drill` all score >0.8 on the same window,
they're competing for the same evidence. Keep one canonical label
per acoustic concept; use variants only when they describe genuinely
different acoustic profiles (`drill into wood` vs `drill into metal`
DO sound different — keep both).

**Length budget: 200-1000 labels total.** Below 200 you under-cover
the project; above 1000 you overload the embedding cache and slow
the lane without adding signal. Most workshop / kitchen / studio
projects land at 400-600. Travel videos with multiple environments
land higher (600-900). Single-speaker talking-heads land lower
(200-300).

---

## Output format

Write to `<edit>/audio_vocab.txt`. One label per line. UTF-8.
Lowercase recommended. No quotes, no commas, no bullets, no comments.

Example structure (illustrative, not prescriptive — every project is
different):

```
# (no comments — this is a real plain text file; example below shows
#  what the file CONTENT looks like when opened. Don't include the
#  hashes literally.)

cordless drill
drilling into wood
drilling into metal
hammer on wood
hammering nails
sawing wood
circular saw
miter saw
sanding wood
random orbital sander
sawdust falling
wood being cut
metal scraping
workshop ambience
workshop room tone
quiet shop
power tool whine
shop vacuum
compressor cycling
... (200-1000 lines total) ...
silence
no sound
unrelated audio
helicopter
opera singing
phone ringing
keyboard typing
... (negatives at end, ~15-20% of total) ...
```

Layout convention: group similar labels together visually (tools,
then materials, then ambience, etc.) so the file is human-readable
when the user opens it. CLAP doesn't care about order, but the user
might.

---

## Return format (what you give back to the parent)

Return a complete report. No artificial length cap.

1. **Vocabulary summary**: total label count + per-category counts.

   ```
   Total: 487 labels
     Tools:        102
     Materials:     63
     Ambience:      78
     Music:         24
     Animals:        9
     Vehicles:      18
     Environments:  41
     Speech-adj:    27
     Negatives:    125
   ```

2. **Project soundscape inference**: what kind of video you read out
   of the timelines, what the dominant acoustic environment is, what
   activities make sound. 2-5 paragraphs.

3. **Specific calls you debated**: 5-15 entries describing labels
   you considered including or excluding and why. e.g. "Considered
   `chainsaw` — visual timeline mentions a saw at [t=423-456] but
   visuals show a hand saw, not a chainsaw. Excluded."

4. **Ambiguities for the parent to confirm**: anything where you
   needed to make a judgement call the user might want to override.
   e.g. "User mentioned 'the dog incident' verbally but no dog is
   visible in the timeline — included `dog`, `dog barking`, `dog
   running` defensively in case the dog is off-screen audibly. Flag
   for user."

5. **Cache note**: if `<edit>/audio_vocab_embeds.npz` exists from a
   prior vocab version, mention that re-running `audio_lane.py` will
   trigger a fresh embedding computation (the cache invalidates
   automatically on vocab text change). The parent will hand this to
   `audio_lane.py` next.

---

## Vocab-specific anti-patterns

- **Returning the baked-in `audio_vocab_default.py` taxonomy
  verbatim, or any thin variation of it.** That list exists only as
  a `tests.py` smoke-test fallback — the parent's rules forbid using
  it on real projects. Your job IS curation. If your output looks
  like it could have been a generic AudioSet-style taxonomy with the
  filenames swapped, you have not done the job.

- **Using only single-word labels.** `drill` alone catches drills but
  conflates them with electric drills used elsewhere. Pair with
  acoustic-context phrases.

- **Skipping the negative set.** Without negatives, every silent
  window pins onto your top label at a deflated score and pollutes
  the audio_timeline. Hard requirement: 15-20% negatives.

- **Padding the vocab with synonyms.** `drill`, `drilling`, `power
  drill`, `electric drill`, `cordless drill`, `cordless power drill`
  competing for the same evidence wastes CLAP cycles and inflates
  per-window output. One canonical label per acoustic concept.

- **Excluding labels for things the user said NOT to feature.** Wrong
  intuition. Include them so the editor sub-agent can FIND those
  moments and avoid them. CLAP doesn't decide what's in the cut;
  it labels.

- **Inventing vocab from imagination instead of from the timelines.**
  Always ground the vocabulary in what the speech + visual timelines
  actually contain. If neither timeline references a thing, it
  doesn't go in the vocab even if the user mentioned it casually.

- **Skipping the verbatim user quotes in the brief.** The user's own
  words about specific tools / materials / locations are direct
  vocabulary candidates. Quote them, include them.

- **Sub-spawning a sub-sub-agent to "save context."** YOU are the
  vocab sub-agent. The parent isolated you for this exact read.
  Don't outsource it again. (Same rule as the editor sub-agent.)
