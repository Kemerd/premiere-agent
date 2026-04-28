# B-roll selection — rules + optimized matching

> Cold-path. **The agent** loads this on demand when
> `b_roll_mode = true` — the user has b-roll / cutaway material,
> either as a dedicated library for scripted assembly or cutaways
> over a single talking-head A-roll. If the project is single-source
> dialogue with no cutaways, this file is **not** in scope and the
> agent never reads it.
>
> Read in full when entering b-roll mode. Selection rules bind every
> cutaway choice; the matching philosophy (caching / two-stage /
> index-first) binds how you spend your time.

---

## What "b-roll" means in this skill

Two flavours; rules here apply to both:

1. **Scripted assembly** — many b-roll clips assembled under a
   continuous voiceover bed. Every visible range is from a
   different source than the audio. See `scripted.md` for
   the procedure; rules below decide *which clip* to pick
   per beat.
2. **Cutaway b-roll on talking-head A-roll** — a primary source
   (interview / explainer / vlog) carries the audio, and short
   cutaway ranges from secondary sources hide behind speaker on
   a higher video track. The audio underneath stays from
   the A-roll. Cutaways might be 2-6 seconds; the A-roll might
   run continuously underneath.

Either way, pick a clip whose visual content reinforces
what the audio says. The selection rules don't care
which flavour you're in.

---

## The cardinal rule

**The visible thing on screen must match what the audio names.**

If the audio says "Riot Games booth," b-roll shows a Riot Games
booth. Not a generic gaming-event crowd. Not a different company's
booth. Not the Riot booth from a fast pan-by where the logo is
half-visible for 8 frames. A Riot Games booth, clearly visible,
stable, for at least the named-subject phrase's duration.

Everything else here is the procedure for *reliably*
satisfying that rule.

---

## Selection preference order (named-subject beats)

When the audio (script or transcript) names a specific subject, rank
candidates by **what's actually visible**, in this preference order:

1. **Readable signage / logos / banners** — text the camera resolves
   well, identifying the named subject directly. A Riot Games sign
   reading "Riot Games" is the highest-evidence shot for a "Riot
   Games" beat. Visual captions like *"sign reads Riot Games"*,
   *"banner with logo"*, *"booth signage"* score top.
2. **Product detail / hardware close-ups** — the named product /
   GPU / hardware visibly on screen. *"GeForce RTX 5090 graphics
   card"*, *"close-up of the graphics card"*. Beats a wide shot of
   the same booth where the product is barely visible.
3. **Game screens / gameplay / UI** — for game / app / software
   beats, the actual game UI / agent select / HUD / menu /
   in-game footage. *"Valorant gameplay on monitor"*, *"agent select
   screen"*. Beats a wide of the booth where the game is on a
   distant monitor.
4. **Booth detail / setup / hardware on display** — when the named
   subject is the booth / area / installation itself. Mid-shot of
   the booth with the brand visible on multiple surfaces beats a
   wide showing many booths at once.
5. **Stage / arena / competition area** — for esports / event-stage
   beats. The actual stage with the named match / panel / talk on
   it.
6. **People-heavy shots (crowd, walkby, talking-head, attendees)** —
   **lowest** priority for named-subject beats. People shots fit
   when the script is *about* the people (audience,
   crowd reaction, attendees), not when it's about the named
   product / brand / venue.

The preference order isn't absolute — a stunningly composed crowd
shot can beat a flat sign shot when the moment calls for it — but
the default for named-subject beats is "show the named subject."

## Rejection rules — drop the candidate, no exceptions

A candidate is **rejected** (drop and pick next in the
shortlist) if any of these are true:

- **Visual captions contradict the script beat.** The
  `merged_timeline.md` `[…]` visual line for the candidate range says
  one thing, the script says another, and they're not the same
  subject. Older filenames or path-based hints don't override the
  caption — Florence-2 saw what's there; trust it. Example: file
  is named `c1_riot_setup.mp4` but captions say *"AMD signage
  visible"* — that's an AMD shot, reject for a Riot beat.
- **The named subject is named-but-not-visible.** Captions say
  *"distant booths in the background"* — the Riot booth might be
  one of those distant booths but you can't tell. Reject; find a
  closer shot.
- **The match is a single-frame flash from a fast pan.** One matching
  caption surrounded by completely different ones means the camera
  whipped past. Reject; the cut won't read.
- **Stability: shaky walking handheld where a stable static exists.**
  See "Stability bias" below.
- **The clip is already used on a different recent beat with a
  distinctive visual.** See "Diversification" below.

## Stability bias — prefer steady shots

At the same subject-match level, **prefer stable / static /
smooth source ranges over shaky handheld walking-and-talking
b-roll.** Reasons:

- Stable shots cut cleaner — boundaries don't betray the camera move.
- Captions are more reliable on stable footage (Florence-2 confidence
  drops on motion blur).
- A stable sign / screen / product shot reads as deliberate; a shaky
  walk-by reads as noise even when it's technically the right
  subject.

Practical signals from the visual lane:

- Long runs of `(same)` in `visual_timeline.md` (10+ consecutive
  frames identical) → highly stable. Top candidate.
- Captions that mention motion words explicitly (*"walking past"*,
  *"camera moving"*, *"panning across"*) → motion shot. De-prioritise.
- Captions that change every frame on the candidate range with no
  `(same)` collapses → camera moving fast or scene is
  chaotic. De-prioritise unless the chaos IS the beat.

If only motion shots exist for a beat (no stable alternative in the
shortlist), use the motion shot but prefer the **smoothest** one —
look for runs where consecutive captions describe the same subject
with mild variation vs completely different things on each
frame.

If `<edit>/clip_index/index.json` exists with stability scores
attached (an optional optimization — see "Optimized matching"
below), use them as a tiebreaker; otherwise reason from the
captions.

## Diversification — don't repeat distinctive visuals

When one distinctive visual appears more than once in the
output (same cosplay character, same distinctive booth, same
distinctive person, same memorable hand gesture), **diversify by
default**: pick the second-best candidate that shows a different
visual. Reasons:

- Viewers notice repeats. A repeated cosplay shot 30 seconds
  apart reads as "the editor only had one of these."
- B-roll variety is a quality signal the user doesn't usually
  articulate but always notices.

Exception: when the user explicitly asks for the repeat (a callback,
a montage returning to the same character, a recurring motif),
keep it. Note in the `reason` field that the repeat is intentional
and quote the user's request.

The diversification rule binds within ~30s of the
output timeline. Same clip 3 minutes apart in a 7-minute video
is fine.

## Specific-mention fallback

When the script names a *specific* game / product / character / venue
not present in any candidate's visual captions:

1. Use the **closest visually verifiable** alternative — gameplay
   from the same game series, a different angle on the same product
   line, the next-best booth shot. Captions still bind: pick what's
   actually visible, don't bend the truth.
2. **Note the limitation** in the `reason` field. The QA note
   pattern from `scripted.md` step 7 covers this — list rejected
   candidates with reasons, name the compromise.
3. **Do not** silently substitute a wrong-but-similar shot
   (a different game's gameplay tagged as the right game's
   gameplay). The user will spot it; revisions multiply.

The named-game-beat example (e.g. "Quake III Arena RTX") is an
*illustration* of this rule, not a hard-coded exclusion. Prefer
footage that visibly matches the named subject; only reject loose
matches when a better verified option
exists.

---

## Optimized matching — spend GPU time on evidence, not re-reads

The skill is built around caching: lane outputs (`transcripts/`,
`visual_caps/`, `audio_tags/`) are immutable products of immutable
inputs and reused across sessions. B-roll matching gets cheaper
the longer a clip library has been processed; rules below
describe how to spend the saved time on quality, not lower
standards.

### 1. Build the index once per library, search the index per beat

The reusable model:

- **Phase A preprocessing** runs once when new clips arrive (or
  changed clips arrive). The parent runs
  `helpers/preprocess_batch.py <videos_dir>` (or `--recursive` for
  nested trees); the speech / visual lanes drop their JSON into
  `<edit>/transcripts/` and `<edit>/visual_caps/`. Cached lane files
  are part of the workflow, not an optional shortcut.
- **`pack_timelines.py`** rolls those lane files into the merged
  view + per-lane drill-down files. Re-runs are fast on cached
  inputs.
- **Folder convention auto-detection** (parent's step 1) writes
  `<edit>/source_tags.json` mapping clip stems → categories
  (`a_roll`, `b_roll`, `timelapse`, `voiceover`, `unknown`) when
  the user organizes by convention folder. **Respect these tags**
  for candidate searches: only `b_roll` / `cutaway` / `unknown`
  clips are eligible cutaway candidates; A-roll is the speech bed
  in talking-head mode; `timelapse` clips are pre-organized retime
  source material (still gated by `timelapse_mode` — see below).
  When tags are absent, all sources are eligible.
- **(Optional) clip index** is a helper that walks `transcripts/`
  + `visual_caps/` and builds a per-clip searchable record. Without
  an index, scan `merged_timeline.md` per beat — slower but
  correct.

If `<edit>/clip_index/index.json` exists, treat it as a
shortlisting aid only — the full `merged_timeline.md` read still
binds before any beat-level matching.

### 2. Two-stage matching — shortlist first, verify second

The non-negotiable structure:

- **Stage 1 (shortlist):** fast text-overlap query against cached
  evidence (clip-index search, or careful read of
  `merged_timeline.md` `[…]` visual lines) → top 3-8
  candidates per beat.
- **Stage 2 (verify):** drill into the surrounding visual context
  in `merged_timeline.md` (and `visual_timeline.md` for the full
  1-fps stream including `(same)` repeats) on the top candidate.
  If verification fails, descend to the next candidate. Optionally
  invoke `helpers/timeline_view.py` for filmstrip / waveform PNG
  inspection at high-stakes moments.

Stage 2 is where quality lives. The shortlist doesn't decide the
cut; verification does. Skipping stage 2 to "save time" is the
single most common scripted-b-roll quality regression.

### 3. Top-candidate review — when quality matters more than speed

For high-stakes work (professional / client deliverables, branded
content, named-subject beats with sponsor implications), spend the
extra time **reviewing the top 2-3 candidates explicitly** vs
accepting the first verified hit. Compare:

- Which candidate has the longest stable visual evidence?
- Which lands cleanest at the boundaries (no continuity break against
  adjacent ranges)?
- Which is least re-used relative to other recent beats?

Note the comparison in the `reason` field. The user (especially in
professional / company / client contexts — see the `user_profile`
question in step 4 of the SKILL.md process) reads those notes as
confidence.

### 4. Cache discipline

- **Don't pass `--force` reflexively.** The orchestrator caches
  lanes per source mtime — re-running with `--force` discards
  evidence the index relies on for shortlisting and re-burns GPU
  time on unchanged media. Pass `--force` only when source file,
  model settings, dependencies, or matching rules changed.
- **Rebuild the index only when clip-set / captions / inventory /
  matching rules change.** Note in your rationale if matching
  uncovered new clips not in the index — that's the trigger for a
  rebuild.
- **Network / OneDrive paths:** if the project lives on a synced
  volume, local scratch / cache may be used for speed, but final
  deliverables and editable timelines are copied back to the
  project / edit folder.

### 5. Spend GPU time on evidence, not on re-runs

The general principle behind every cache rule: GPU time should buy
**better matching evidence** — better visual captions, more accurate
voiceover timing, stability checks, top-candidate review — not
brute-force reanalysis of unchanged media. If you find yourself
re-preprocessing clips that haven't changed, reframe: what evidence
do you actually need, and is it already cached?

---

## How `timelapse_mode` interacts with b-roll selection

`timelapse_mode` binds the time-squeezing section in SKILL.md, but
also shapes b-roll selection in two ways:

- **`timelapse_mode = false` (default).** Any visually-continuous
  long-stretch clip in the b-roll library — workshop builds,
  packing montages, walking shots — is treated as 1x source
  material. You **may** select short ranges from those clips (a
  4-second cut from a 10-minute build sequence is fine) but
  **may not** retime any range. If the only candidate for a beat
  is a long stretch that would only "work" as a timelapse, drop
  the beat or pick a different shorter clip; do not retime to fit.
- **`timelapse_mode = true`.** Long-stretch clips in the b-roll
  library are also valid timelapse retime candidates per the
  time-squeezing rules. If `source_tags.json` tagged a clip as
  `timelapse`, prefer it for retime over discovering retime
  stretches in arbitrary footage — the user pre-organized the
  retime source.

This matters most for workshop / build / travel / vlog projects
where the user has long-running source material that could go
either way. Asking the timelapse question explicitly in step 4 of
the SKILL.md process disambiguates; your job is to honour the
answer.

---

## How `user_profile` shapes the bar

`user_profile` is one of `personal`, `creator`, or `professional`
— ask the user during step 4 of the SKILL.md process. Use it to
set the verification bar:

- **`personal` / `creator`** — default rules apply; one verified
  candidate per beat is fine, QA notes terse.
- **`professional` (working for a company / client / sponsor / agency
  deliverable)** — top-candidate review is mandatory on every
  named-subject beat, QA notes are detailed (list rejected
  candidates with reasons), every compromise on a specific-mention
  fallback is flagged explicitly. The user is shipping this to a
  client and will need to defend the cut.

Don't conflate `user_profile` with target runtime or pacing — those
remain user-confirmed independently. `user_profile` is purely the
QA / verification dial.

---

## Cross-references

- The **assembly procedure** (script + voiceover, beat segmentation,
  vo-anchored timing, source in-points on spoken words) lives in
  `references/scripted.md`. When both `script_mode = true` and
  `b_roll_mode = true`, read both files. (The common combo.)
- The **pacing preset** binds every range edge — word-boundary
  discipline (Hard Rule 6), 30-200ms padding window (Hard Rule 7),
  silence-removal threshold (the four numbers from the chosen preset).
- **Hard Rule 6** — every cut anchor is verified with
  `helpers/find_quote.py` against the cached transcripts. B-roll
  cutaway in/out times still snap to word boundaries on whichever
  source carries the audio bed.
- **Hard Rule 14** still defers J/L cuts and dissolves. B-roll
  cutaways are hard cuts only; the NLE's caption track and the 30ms
  `afade` pair on each boundary handle the soundscape.
- **Hard Rule 15 / 16** — read `merged_timeline.md` end-to-end before
  any beat-level work; that's your default reading surface.

---

## Anti-patterns specific to b-roll selection

- **Picking by filename / path / metadata vs visual
  captions.** Florence-2 saw what's there; the filename was a guess.
- **Accepting the first index hit without verifying.** Two-stage
  matching is the rule. Shortlist suggests; verification decides.
- **Ignoring stability differences when several candidates match.**
  Same subject + steadier shot wins.
- **Repeating distinctive visuals within a 30-second window.**
  Diversify unless the user explicitly asked for the callback.
- **Substituting "close enough" for a named subject silently.** Note
  the compromise in the `reason` field so the user can flag it.
- **Putting people-heavy crowd shots on subject-named beats.** Crowd
  shots belong on crowd-named beats.
- **Cutting the b-roll *before* the named word lands.** Land the
  visual on the spoken name, not before. (Scripted-mode synced
  in-points; see `scripted.md` step 6.)
- **Skipping QA notes on named-subject beats.** Revisions need them.
- **Re-running preprocessing with `--force` to "refresh" matching.**
  The cache is correct; bypass only on real input change.
- **Ignoring `source_tags.json` and proposing A-roll clips as
  cutaways.** The user organized their footage; respect the tags.
  A-roll-tagged clips are the speech bed, not the cutaway library.
- **Retiming a b-roll candidate to fit a beat when
  `timelapse_mode = false`.** The user opted out of timelapses for
  this session. Pick a different clip or shorter range; never
  silently retime.
- **Skipping the clip-index build on a 100-clip professional
  project.** That's where the index pays off most — text-overlap
  shortlisting against cached captions cuts beat-matching time by
  ~10x on large libraries, and the verification + detailed QA notes
  the professional bar requires are easier to write when shortlisting
  is fast.
