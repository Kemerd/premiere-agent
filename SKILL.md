---
name: video-use-premiere
description: Edit any video by conversation. Local three-lane preprocessing (Whisper speech + CLAP zero-shot audio events + Florence-2 visual captions), cut, color grade, generate overlay animations, burn subtitles, OR export FCPXML to Premiere/Resolve/FCP with split edits. For talking heads, montages, tutorials, travel, interviews, workshop / shop footage. No presets, no menus, no cloud transcription. Ask questions, confirm the plan, execute, iterate, persist. Production-correctness rules are hard; everything else is artistic freedom.
---

# Video Use Premiere

## Principle

1. **LLM reasons from raw transcript + sound captions + visual captions + on-demand drill-down.** Three lightweight markdown views (`speech_timeline.md`, `audio_timeline.md`, `visual_timeline.md`) are the entire reading surface. Everything else — filler tagging, retake detection, shot classification, B-roll spotting, emphasis scoring — you derive at decision time.
2. **Speech is primary, visuals are secondary, audio events are tertiary.** Cut candidates come from Whisper speech boundaries and silence gaps — that lane is highly accurate and is the editorial spine. Visual captions (Florence-2) are the second source of truth: they answer "what's actually on screen here?" and resolve ambiguous decision points (B-roll spotting, shot continuity, action beats). Audio events (CLAP, zero-shot scoring against a vocabulary) tag non-speech sounds per ~10s window (tools, materials, ambience, music, animals, vehicles). Vocabulary is editable per project — see Phase B below for the agent-curated workflow that makes labels much sharper. When audio and visual disagree about *what is happening on screen*, **trust the visual lane.**
3. **Ask → confirm → execute → iterate → persist.** Never touch the cut until the user has confirmed the strategy in plain English.
4. **Generalize.** Do not assume what kind of video this is. Look at the material, ask the user, then edit.
5. **Artistic freedom is the default.** Every specific value, preset, font, color, duration, pitch structure, and technique in this document is a *worked example* from one proven video — not a mandate. Read them to understand what's possible and why each worked. Then make your own taste calls based on what the material actually is and what the user actually wants. **The only things you MUST do are in the Hard Rules section below.** Everything else is yours.
6. **Invent freely.** If the material calls for a technique not described here — split-screen, picture-in-picture, lower-third identity cards, reaction cuts, speed ramps, freeze frames, crossfades, match cuts, L-cuts, J-cuts, speed ramps over breath, whatever — build it. The helpers are ffmpeg, PIL, and (for split edits / dissolves) the FCPXML exporter. They can do anything the format supports. Do not wait for permission.
7. **Verify your own output before showing it to the user.** If you wouldn't ship it, don't present it.

## Hard Rules (production correctness — non-negotiable)

These are the things where deviation produces silent failures or broken output. They are not taste, they are correctness. Memorize them.

1. **Subtitles are applied LAST in the filter chain**, after every overlay. Otherwise overlays hide captions. Silent failure.
2. **Per-segment extract → lossless `-c copy` concat**, not single-pass filtergraph. Otherwise you double-encode every segment when overlays are added.
3. **30ms audio fades at every segment boundary** (`afade=t=in:st=0:d=0.03,afade=t=out:st={dur-0.03}:d=0.03`). Otherwise audible pops at every cut.
4. **Overlays use `setpts=PTS-STARTPTS+T/TB`** to shift the overlay's frame 0 to its window start. Otherwise you see the middle of the animation during the overlay window.
5. **Master SRT uses output-timeline offsets**: `output_time = word.start - segment_start + segment_offset`. Otherwise captions misalign after segment concat.
6. **Never cut inside a word.** Snap every cut edge to a word boundary from the Whisper word-level transcript.
7. **Pad every cut edge.** Working window: 30–200ms. Whisper timestamps drift 50–100ms — padding absorbs the drift. Tighter for fast-paced, looser for cinematic.
8. **Word-level verbatim ASR only.** Whisper is invoked with `return_timestamps="word"` — never SRT/phrase mode (loses sub-second gap data). Never normalized fillers (loses editorial signal).
9. **Cache lane outputs per source.** Never re-run a lane unless the source file itself changed (mtime). The orchestrator handles this; do not pass `--force` reflexively.
10. **Parallel sub-agents for multiple animations.** Never sequential. Spawn N at once via the `Agent` tool; total wall time ≈ slowest one.
11. **Strategy confirmation before execution.** Never touch the cut until the user has approved the plain-English plan.
12. **All session outputs in `<videos_dir>/edit/`.** Never write inside the `video-use-premiere/` project directory.

Everything else in this document is a worked example. Deviate whenever the material calls for it.

## Directory layout

The skill lives in `video-use-premiere/`. User footage lives wherever they put it. All session outputs go into `<videos_dir>/edit/`.

```
<videos_dir>/
├── <source files, untouched>
└── edit/
    ├── project.md               ← memory; appended every session
    ├── speech_timeline.md       ← Whisper phrase-level transcripts (lane 1)
    ├── audio_timeline.md        ← CLAP audio events, coalesced   (lane 2)
    ├── visual_timeline.md       ← Florence-2 captions @ 1fps     (lane 3)
    ├── merged_timeline.md       ← optional, all three interleaved by ts
    ├── edl.json                 ← cut decisions
    ├── transcripts/<name>.json  ← cached raw Whisper words
    ├── audio_tags/<name>.json   ← cached raw CLAP (label, score) events
    ├── audio_vocab.txt          ← optional: agent-curated vocab for Phase B
    ├── audio_vocab_embeds.npz   ← cached CLAP text embeddings for that vocab
    ├── visual_caps/<name>.json  ← cached raw Florence-2 captions
    ├── audio_16k/<name>.wav     ← shared 16kHz mono PCM (speech lane + CLAP)
    ├── animations/slot_<id>/    ← per-animation source + render + reasoning
    ├── clips_graded/            ← per-segment extracts with grade + fades
    ├── master.srt               ← output-timeline subtitles
    ├── verify/                  ← debug frames / timeline PNGs
    ├── preview.mp4
    ├── final.mp4                ← flattened deliverable (ffmpeg path)
    └── cut.fcpxml               ← editor-ready timeline (NLE path)
```

## Setup

- **`HF_TOKEN` in `.env` at project root** — only required for speaker diarization (pyannote). Skip if single-speaker.
- **`ffmpeg` + `ffprobe` on PATH.** Hard requirement. Win: `winget install Gyan.FFmpeg`. macOS: `brew install ffmpeg`. Linux: `apt install ffmpeg`.
- **Python deps**: run `install.sh` (Linux/macOS) or `install.bat` (Windows). Installs PyTorch + the `[preprocess,fcpxml]` extras. Optional: `pip install -e .[flash]` for Flash Attention 2 (~3-5x Whisper speedup), `pip install -e .[diarize]` for pyannote, `pip install -e .[parakeet]` to pre-install the NVIDIA Parakeet fallback (auto-installs on first blocked-Whisper detection otherwise).
- **Speech lane fallback**: if HuggingFace is unreachable from this machine (corporate proxy / NVIDIA intranet / restricted network), `whisper_lane.py` auto-falls-back to NVIDIA Parakeet TDT v3 — local, ~10x faster, English + 24 European languages. Decision cached 7 days at `~/.video-use-premiere/whisper_blocked`. The output JSON shape is identical, so cuts, diarization, FCPXML export all work unchanged. `helpers/health.py --json` surfaces `"fallbacks_active": ["parakeet"]` so you know which backend is running before the lane fires. **Fully air-gapped?** Set `PARAKEET_MODEL_PATH=/path/to/parakeet-tdt-0.6b-v3.nemo` (download from NGC) and the lane skips network entirely via `ASRModel.restore_from()`.
- **`yt-dlp`, `manim`, Remotion** installed only on first use.
- This skill vendors `skills/manim-video/`. Read its SKILL.md when building a Manim slot.

## Skill health check (run on EVERY session start)

Before doing anything else in a session, run:

```bash
python helpers/health.py --json
```

This is **idempotent and cached** — first call runs the smoke suite (~3s), subsequent calls within 7 days return the cached result instantly (<500ms). Cache auto-invalidates when `python` / `torch` / `transformers` / `opentimelineio` versions change, so a `pip install --upgrade` triggers a fresh check.

Cache lives at `~/.video-use-premiere/health.json` — **outside** the per-session `<videos_dir>/edit/` so it persists across projects. This is the one exception to Hard Rule 12, and it's intentional: skill-environment health is a per-machine property, not a per-session one.

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

**When to force a re-run:**
- User reports something stopped working
- User just upgraded Python or PyTorch
- User asks "is the skill set up correctly?"

```bash
python helpers/health.py --force --json    # ignore cache, run now
python helpers/health.py --clear           # wipe cache (next call re-runs)
```

**Optional heavy-tier verification** (~4.5 GB downloads on first run, exercises real Whisper-large-v3 + Florence-2 + PANNs on a synthetic 2s clip): tell the user to run `python tests.py --heavy` once after install. Cached separately under the same TTL. Don't trigger this autonomously — it's an explicit user action.

## Helpers

### Preprocessing (the three lanes)

> **All helpers live in `helpers/`.** Always invoke them from the skill root as `python helpers/<script>.py …` (the sibling-import pattern they use depends on `helpers/` being the script's own directory, which `sys.path` resolves automatically when you run them by path). Never `cd helpers/` first — `cwd` semantics differ across shells (PowerShell, bash, agentic shells that don't persist `cd`), and the cache layout assumes the project root is the cwd.

- **`helpers/preprocess_batch.py <videos_dir>`** — auto-discover videos, run all three lanes with VRAM-aware scheduling. Default entry point. Flags: `--wealthy` (24GB+ GPU), `--diarize`, `--language en`, `--force`, `--skip-{whisper,audio,visual}`.
- **`helpers/preprocess.py <video1> [<video2> ...]`** — same orchestrator with explicit file list. Use when you want a subset.
- **`helpers/pack_timelines.py --edit-dir <dir>`** — read the three lane caches (`transcripts/`, `audio_tags/`, `visual_caps/`) and produce `speech_timeline.md`, `audio_timeline.md`, `visual_timeline.md`. Add `--merge` for `merged_timeline.md`.
- **Individual lanes** (rarely needed — the orchestrator wraps them): `helpers/whisper_lane.py`, `helpers/audio_lane.py`, `helpers/visual_lane.py`. Each accepts `--wealthy` and runs standalone.
- **`helpers/extract_audio.py <video>`** — manually extract 16kHz mono WAV. Cached. Mainly for debugging.
- **`helpers/vram.py`** — print detected GPU + the schedule that would be picked. Useful sanity check.

### Editing

- **`helpers/timeline_view.py <video> <start> <end>`** — filmstrip + waveform PNG. On-demand visual drill-down. **Not a scan tool** — use it at decision points, not constantly. The visual_timeline.md replaces 90% of the old "scan with timeline_view" workflow.
- **`helpers/render.py <edl.json> -o <out>`** — per-segment extract → concat → overlays (PTS-shifted) → subtitles LAST → loudness norm → final.mp4. `--preview` for 1080p fast, `--draft` for 720p ultrafast, `--build-subtitles` to generate master.srt inline. **Flattens J/L cuts to hard cuts** — see Cut Techniques below.
- **`helpers/grade.py <in> -o <out>`** — ffmpeg filter chain grade. Presets + `--filter '<raw>'` for custom.
- **`helpers/export_fcpxml.py <edl.json> -o cut.fcpxml`** — emit an editor-ready FCPXML timeline. Honors `audio_lead` / `video_tail` (J/L cuts) and `transition_in` (cross-dissolves) natively. Opens in Premiere Pro, DaVinci Resolve, Final Cut Pro. `--frame-rate 24` (default), 25, 29.97, 30, 60.

For animations, create `<edit>/animations/slot_<id>/` with `Bash` and spawn a sub-agent via the `Agent` tool.

## The process

0. **Health check.** Run `python helpers/health.py --json` first. Cached for 7 days; usually returns instantly. If `status != "ok"`, surface the `advice` strings to the user verbatim and stop. See the "Skill health check" section above.
1. **Inventory.** `ffprobe` every source. `python helpers/preprocess_batch.py <videos_dir>` to run all three lanes (Whisper + PANNs + Florence-2) — cached by mtime, so this is one-time per source. Then `python helpers/pack_timelines.py --edit-dir <edit>` to produce the three timeline markdowns (and `--merge` if the material is action-heavy and you want all three context streams interleaved).
2. **Pre-scan for problems.** One pass over `speech_timeline.md` to note verbal slips, mis-speaks, or phrasings to avoid. Then scan `visual_timeline.md` for shot variety, B-roll candidates, and visually continuous actions you'll want to keep whole. `audio_timeline.md` is a *last* pass and only as a rough hint at where non-speech beats might live — verify any PANNs label against the visual lane at the same timestamp before trusting it (the model is noisy, especially on tools, music, and crowd sounds).
3. **Converse.** Describe what you see in plain English. Ask questions *shaped by the material*. Collect: content type, target length/aspect, aesthetic/brand direction, pacing feel, must-preserve moments, must-cut moments, animation and grade preferences, subtitle needs, **delivery target (flattened mp4 vs FCPXML to NLE)**. Do not use a fixed checklist — the right questions are different every time.
4. **Propose strategy.** 4–8 sentences: shape, take choices, cut direction, animation plan, grade direction, subtitle style, length estimate, **delivery format**. **Wait for confirmation.**
5. **Execute.** Produce `edl.json` via the editor sub-agent brief. Drill into `timeline_view` at ambiguous moments where the visual_timeline caption alone isn't enough. Build animations in parallel sub-agents. Apply grade per-segment.
   - **Flat MP4 path:** Compose via `render.py`.
   - **NLE handoff path:** Export via `export_fcpxml.py`. Recipient finishes in Premiere/Resolve/FCP.
   - **Both:** run them both — they consume the same EDL.
6. **Preview.** `render.py --preview` (or hand the `cut.fcpxml` to the user to open and scrub).
7. **Self-eval (before showing the user).** Run `timeline_view` on the **rendered output** (not the sources) at every cut boundary (±1.5s window). Check each image for:
   - Visual discontinuity / flash / jump at the cut
   - Waveform spike at the boundary (audio pop that slipped past the 30ms fade)
   - Subtitle hidden behind an overlay (Rule 1 violation)
   - Overlay misaligned or showing wrong frames (Rule 4 violation)

   Also sample: first 2s, last 2s, and 2–3 mid-points — check grade consistency, subtitle readability, overall coherence. Run `ffprobe` on the output to verify duration matches the EDL expectation.

   If anything fails: fix → re-render → re-eval. **Cap at 3 self-eval passes** — if issues remain after 3, flag them to the user rather than looping forever. Only present the preview once the self-eval passes.
8. **Iterate + persist.** Natural-language feedback, re-plan, re-render. Never re-preprocess unchanged sources. Final render on confirmation. Append to `project.md`.

## Cut craft (techniques)

- **Speech-first.** Candidate cuts from word boundaries and silence gaps in `speech_timeline.md`. Whisper is accurate to the word; this lane is the editorial spine.
- **Preserve peaks.** Laughs, punchlines, emphasis beats. Extend past punchlines to include reactions — the laugh IS the beat.
- **Speaker handoffs** benefit from air between utterances. Common values: 400–600ms. Less for fast-paced, more for cinematic. Taste call.
- **Visual context is the second source of truth.** Before committing to *any* non-trivial cut, read `visual_timeline.md` around the cut point. If captions show a continuous action ("person holding drill") spanning your cut, you're cutting in the middle of a shot — usually fine, but be deliberate. Use the visual lane to find B-roll cutaway candidates, match cuts, shot changes, and to decide whether a moment is worth preserving even when speech is silent.
- **Audio events are noisy hints, not signals.** `audio_timeline.md` carries `(drill 0.87)`, `(applause 0.92)`, `(laughter)`, `(power_tool)` markers from PANNs. **The model is approximate** — it mis-labels frequently (music tagged as speech, hammers tagged as drums, room tone tagged as applause). Use a marker only as a prompt to *go look* at the visual lane (and if needed `timeline_view`) at that timestamp. **Never cut purely on a PANNs label.** When PANNs and Florence-2 disagree about what's happening, trust Florence-2.
- **Silence gaps are cut candidates.** Silences ≥400ms are usually the cleanest. 150–400ms phrase boundaries are usable with a visual check. <150ms is unsafe (mid-phrase).
- **Example cut padding** (the launch video shipped with this): 50ms before the first kept word, 80ms after the last. Tighter for montage energy, looser for documentary. Stay in the 30–200ms working window (Hard Rule 7).
- **Never reason audio and video independently.** Every cut must work on both tracks.

### Split edits (J/L cuts) and dissolves

Modern NLE-style cuts that don't render cleanly in flat single-track ffmpeg but absolutely shine in FCPXML:

- **J-cut** (`audio_lead` field) — the next clip's audio bleeds in BEFORE its video appears. Classic for interviews ("you hear the answer start while still seeing the question"). Typical: 200–800ms.
- **L-cut** (`video_tail` field) — the previous clip's audio lingers UNDER the next clip's video. Used for B-roll cutaways while the speaker keeps talking. Typical: 500–2000ms.
- **Cross-dissolve** (`transition_in` field) — symmetric video+audio dissolve into this clip. Use sparingly: scene changes, time jumps, montage transitions. Typical: 250–500ms.

**EDL fields** (all optional, all default to 0 = hard cut):

```json
{"source": "C0103", "start": 12.20, "end": 18.45, "beat": "ANSWER",
 "audio_lead": 0.4,        // J-cut: audio leads video by 400ms
 "video_tail": 1.2,        // L-cut: audio lingers 1.2s past video end
 "transition_in": 0.3}     // 300ms cross-dissolve into this clip
```

**Render path matrix:**

| Output            | Hard cuts | J/L cuts            | Dissolves          |
|-------------------|-----------|---------------------|--------------------|
| `render.py` → mp4 | ✓         | flattened, warned   | flattened, warned  |
| `export_fcpxml.py` → fcpxml | ✓ | native split edit | native cross-dissolve |

**Workflow:** if the user wants J/L cuts or dissolves, build the EDL with those fields populated and run BOTH `render.py` (gives them a flattened preview MP4 to evaluate cuts) and `export_fcpxml.py` (gives them the editor file with the split edits intact). Tell them the MP4 is for reviewing cut points, the FCPXML is for finishing.

## The three timelines (primary reading view)

`pack_timelines.py` reads each lane's JSON cache and produces three markdowns. They share an addressing scheme: every line carries `[start-end]` (or `[t]` for visual frames) in seconds-from-clip-start, so a line read out of any timeline can be directly addressed in `edl.json` cut ranges.

**`speech_timeline.md`** — phrase-grouped Whisper transcript. Phrases break on silence ≥0.5s OR speaker change. The artifact the editor sub-agent reads to pick cuts.

```
## C0103  (duration: 43.0s, 8 phrases)
  [002.52-005.36] S0 Ninety percent of what a web agent does is completely wasted.
  [006.08-006.74] S0 We fixed this.
```

**`audio_timeline.md`** — Audio Flamingo 3 (NVIDIA, 7B audio-language model) natural-language captions of non-speech audio, one caption per ~30s chunk. Open-vocabulary — the model describes what it actually hears (specific tools, materials, ambience, music character) instead of mapping into a fixed 527-class taxonomy. Use it to find action beats, sync points, ambient transitions, and sounds the visual lane can't see (off-screen tools, room tone changes). When AF3 and Florence-2 disagree about what's on screen, trust Florence-2 — AF3 is the authority on the **soundscape**, not the picture.

```
## C0108  (duration: 87.4s, 3 chunks @ 30.0s)
  [000.00-030.00] A power drill running in short bursts on metal, with a faint shop fan in the background.
  [030.00-060.00] Hammer strikes on sheet metal, then a man laughs briefly off-mic. Drill resumes at the end.
  [060.00-087.40] Sandpaper rubbing on wood in long even strokes; quiet otherwise.
```

**`visual_timeline.md`** — Florence-2 detailed captions @ 1fps. Consecutive identical captions collapse to `(same)`. Use to spot shots, B-roll candidates, match cuts, action. **This is the second source of truth after speech** — when classifying *what is happening* in a moment, prefer this over the audio events lane.

```
## C0108  (duration: 87.4s, 87 caps @ 1 fps)
  [000.00] a workbench with hand tools laid out on a brown wooden surface
  [001.00] (same)
  [002.00] (same)
  [003.00] a person holding a cordless drill above a metal panel with rivet holes
  [004.00] close-up of a drill bit entering metal, sparks visible
  [005.00] (same)
```

## Editor sub-agent brief (for multi-take selection)

When the task is "pick the best take of each beat across many clips," spawn a dedicated sub-agent with a brief shaped like this. The structure is load-bearing; the pitch-shape example is not.

```
You are editing a <type> video. Pick the best take of each beat and 
assemble them chronologically by beat, not by source clip order.

INPUTS (in priority order — trust them in this order when they disagree):
  - speech_timeline.md  (phrase-level Whisper transcripts; ACCURATE, the spine)
  - visual_timeline.md  (1fps Florence-2 captions; second source of truth for
                         what's on screen / what's happening)
  - audio_timeline.md   (Audio Flamingo 3 natural-language sound captions per
                         ~30s chunk; describes the soundscape — tools,
                         materials, ambience, music. Trust for non-speech
                         audio; defer to visual_timeline for what's on screen)
  - Product/narrative context: <2 sentences from the user>
  - Speaker(s): <name, role, delivery style note>
  - Expected structure: <pick an archetype or invent one>
  - Verbal slips to avoid: <list from the pre-scan pass>
  - Target runtime: <seconds>
  - Delivery: <flat mp4 / fcpxml / both>

Common structural archetypes (pick, adapt, or invent):
  - Tech launch / demo:   HOOK → PROBLEM → SOLUTION → BENEFIT → EXAMPLE → CTA
  - Tutorial:             INTRO → SETUP → STEPS → GOTCHAS → RECAP
  - Interview:            (QUESTION → ANSWER → FOLLOWUP) repeat
  - Workshop / build:     INTRO → MATERIALS → STEPS (with audio_event beats) → REVEAL
  - Travel / event:       ARRIVAL → HIGHLIGHTS → QUIET MOMENTS → DEPARTURE
  - Documentary:          THESIS → EVIDENCE → COUNTERPOINT → CONCLUSION
  - Music / performance:  INTRO → VERSE → CHORUS → BRIDGE → OUTRO
  - Or invent your own.

RULES:
  - Start/end times must fall on word boundaries from speech_timeline.md.
  - Pad cut boundaries (working window 30–200ms).
  - Prefer silences ≥ 400ms as cut targets.
  - Cross-reference visual_timeline.md before committing to a cut whose
    audio looks clean — make sure you're not cutting in the middle of a
    visually continuous action you wanted to keep whole. The visual lane
    classifies the moment; the audio events lane only suggests where to look.
  - Use J/L cuts (audio_lead / video_tail) for interview answers and
    B-roll cutaways. Cross-dissolves (transition_in) for scene changes.
  - Unavoidable slips are kept if no better take exists. Note them in "reason".
  - If over budget, revise: drop a beat or trim tails. Report total and self-correct.

OUTPUT (JSON array, no prose):
  [{"source": "C0103", "start": 2.42, "end": 6.85, "beat": "HOOK",
    "audio_lead": 0.0, "video_tail": 0.0, "transition_in": 0.0,
    "quote": "...", "reason": "..."}, ...]

Return the final EDL and a one-line total runtime check.
```

## Color grade (when requested)

Your job is to **reason about the image**, not apply a preset. Look at a frame (via `timeline_view`), decide what's wrong, adjust one thing, look again.

Mental model is ASC CDL. Per channel: `out = (in * slope + offset) ** power`, then global saturation. `slope` → highlights, `offset` → shadows, `power` → midtones.

**Example filter chains** (`grade.py` has `--list-presets`; use them as starting points or mix your own):

- **`warm_cinematic`** — retro/technical, subtle teal/orange split, desaturated. Shipped in a real launch video. Safe for talking heads.
- **`neutral_punch`** — minimal corrective: contrast bump + gentle S-curve. No hue shifts.
- **`none`** — straight copy. Default when the user hasn't asked.

For anything else — portraiture, nature, product, music video, documentary — invent your own chain. `grade.py --filter '<raw ffmpeg>'` accepts any filter string.

Hard rules: apply **per-segment during extraction** (not post-concat, which re-encodes twice). Never go aggressive without testing skin tones. **For FCPXML delivery, do NOT bake the grade** — leave the cut clean and let the colorist do the grade in the NLE. Mention the grade direction in the FCPXML clip metadata if you have one in mind.

## Subtitles (when requested)

Subtitles have three dimensions worth reasoning about: **chunking** (1/2/3/sentence per line), **case** (UPPER/Title/Natural), and **placement** (margin from bottom). The right combo depends on content.

**Worked styles** — pick, adapt, or invent:

**`bold-overlay`** — short-form tech launch, fast-paced social. 2-word chunks, UPPERCASE, break on punctuation, Helvetica 18 Bold, white-on-outline, `MarginV=35`. `render.py` ships with this as `SUB_FORCE_STYLE`.

```
FontName=Helvetica,FontSize=18,Bold=1,
PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H00000000,
BorderStyle=1,Outline=2,Shadow=0,
Alignment=2,MarginV=35
```

**`natural-sentence`** (if you invent this mode) — narrative, documentary, education. 4–7 word chunks, sentence case, break on natural pauses, `MarginV=60–80`, larger font for readability, slightly wider max-width. No shipped force_style — design one if you need it.

Invent a third style if neither fits. Hard rules: subtitles LAST (Rule 1), output-timeline offsets (Rule 5).

For FCPXML delivery: ship `master.srt` alongside `cut.fcpxml`. Most NLEs import SRT as a captions track that the editor can restyle.

## Animations (when requested)

Animations match the content and the brand. **Get the palette, font, and visual language from the conversation** — never assume a default. If the user hasn't told you, propose a palette in the strategy phase and wait for confirmation before building anything.

**Tool options:**

- **PIL + PNG sequence + ffmpeg** — simple overlay cards: counters, typewriter text, single bar reveals, progressive draws. Fast to iterate, any aesthetic you want. The launch video used this.
- **Manim** — formal diagrams, state machines, equation derivations, graph morphs. Read `skills/manim-video/SKILL.md` and its references for depth.
- **Remotion** — typography-heavy, brand-aligned, web-adjacent layouts. React/CSS-based.

None is mandatory. Invent hybrids if useful (e.g., PIL background with a Remotion layer on top).

**Duration rules of thumb, context-dependent:**

- **Sync-to-narration explanations.** A viewer needs to parse the content at 1×. Rough floor 3s, typical 5–7s for simple cards, 8–14s for complex diagrams. The launch video shipped at 5–7s per simple card.
- **Beat-synced accents** (music video, fast montage). 0.5–2s is fine — they're visual accents, not information. The "readable at 1×" rule becomes *"recognizable at 1×"*, not *"fully parseable."*
- **Hold the final frame ≥ 1s** before the cut (universal).
- **Over voiceover:** total duration ≥ `narration_length + 1s` (universal).
- **Never parallel-reveal independent elements** — the eye can't track two new things at once. One thing, pause, next thing.

**Animation payoff timing (rule for sync-to-narration):** get the payoff word's timestamp from `speech_timeline.md`. Start the overlay `reveal_duration` seconds earlier so the landing frame coincides with the spoken payoff word. Without this sync the animation feels disconnected.

**Easing** (universal — never `linear`, it looks robotic):

```python
def ease_out_cubic(t):    return 1 - (1 - t) ** 3
def ease_in_out_cubic(t):
    if t < 0.5: return 4 * t ** 3
    return 1 - (-2 * t + 2) ** 3 / 2
```

`ease_out_cubic` for single reveals (slow landing). `ease_in_out_cubic` for continuous draws.

**Typing text anchor trick:** center on the FULL string's width, not the partial-string width — otherwise text slides left during reveal.

**Example palette** (the launch video — one aesthetic among infinite):
- Background `(10, 10, 10)` near-black
- Accent `#FF5A00` / `(255, 90, 0)` orange
- Labels `(110, 110, 110)` dim gray
- Font: Menlo Bold at `/System/Library/Fonts/Menlo.ttc` (index 1)
- ≤ 2 accent colors, ~40% empty space, minimal chrome
- Result: terminal / retro tech feel

This is one style. If the brand is warm and serif, use that. If it's colorful and playful, use that. If the user handed you a style guide, follow it. If they didn't, propose one and confirm.

**Parallel sub-agent brief** — each animation is one sub-agent spawned via the `Agent` tool. Each prompt is self-contained (sub-agents have no parent context). Include:

1. One-sentence goal: *"Build ONE animation: [spec]. Nothing else."*
2. Absolute output path (`<edit>/animations/slot_<id>/render.mp4`)
3. Exact technical spec: resolution, fps, codec, pix_fmt, CRF, duration
4. Style palette as concrete values (RGB tuples, hex, or reference to a design system)
5. Font path with index
6. Frame-by-frame timeline (what happens when, with easing)
7. Anti-list ("no chrome, no extras, no titles unless specified")
8. Code pattern reference (copy helpers inline, don't import across slots)
9. Deliverable checklist (script, render, verify duration via ffprobe, report)
10. **"Do not ask questions. If anything is ambiguous, pick the most obvious interpretation and proceed."**

One sub-agent = one file (unique filenames, parallel agents don't overwrite each other).

## Output spec

Match the source unless the user asked for something specific. Common targets: `1920×1080@24` cinematic, `1920×1080@30` screen content, `1080×1920@30` vertical social, `3840×2160@24` 4K cinema, `1080×1080@30` square. `render.py` defaults the scale to 1080p from any source; pass `--filter` or edit the extract command for other targets. For FCPXML delivery, pass `--frame-rate` matching the source (or the user's intended deliverable) so cuts snap to whole frames. Worth asking the user which delivery format matters.

## EDL format

```json
{
  "version": 1,
  "sources": {"C0103": "/abs/path/C0103.MP4", "C0108": "/abs/path/C0108.MP4"},
  "ranges": [
    {"source": "C0103", "start": 2.42, "end": 6.85,
     "beat": "HOOK", "quote": "...", "reason": "Cleanest delivery, stops before slip at 38.46."},
    {"source": "C0108", "start": 14.30, "end": 28.90,
     "beat": "SOLUTION", "quote": "...", "reason": "Only take without the false start.",
     "audio_lead": 0.4, "video_tail": 1.2, "transition_in": 0.3}
  ],
  "grade": "warm_cinematic",
  "overlays": [
    {"file": "edit/animations/slot_1/render.mp4", "start_in_output": 0.0, "duration": 5.0}
  ],
  "subtitles": "edit/master.srt",
  "total_duration_s": 87.4
}
```

`grade` is a preset name or raw ffmpeg filter (ignored by FCPXML export — colorist's job). `overlays` are rendered animation clips (ffmpeg path only). `subtitles` is optional and applied LAST (ffmpeg path) or imported as a captions track (FCPXML path). `audio_lead` / `video_tail` / `transition_in` per range are optional split-edit / dissolve fields, all default 0 (hard cut). See "Cut Techniques → Split edits" above.

## Memory — `project.md`

Append one section per session at `<edit>/project.md`:

```markdown
## Session N — YYYY-MM-DD

**Strategy:** one paragraph describing the approach
**Decisions:** take choices, cuts, grades, animations + why
**Reasoning log:** one-line rationale for non-obvious decisions
**Outstanding:** deferred items
```

On startup, read `project.md` if it exists and summarize the last session in one sentence before asking whether to continue.

## Anti-patterns

Things that consistently fail regardless of style:

- **Hierarchical pre-computed codec formats** with USABILITY / tone tags / shot layers. Over-engineering. Derive from the timelines at decision time.
- **Hand-tuned moment-scoring functions.** The LLM picks better than any heuristic you'll write.
- **Whisper SRT / phrase-level output.** Loses sub-second gap data. Always word-level verbatim.
- **Re-running `helpers/preprocess_batch.py --force` reflexively.** The mtime-based cache is correct; bypass only when the source file actually changed or you've upgraded a model.
- **Reading `transcripts/*.json` directly.** Use `speech_timeline.md`. Same data, 1/10 the tokens, phrase-aligned.
- **Burning subtitles into base before compositing overlays.** Overlays hide them. (Hard Rule 1.)
- **Single-pass filtergraph when you have overlays.** Double re-encodes. Use per-segment extract → concat.
- **Linear animation easing.** Looks robotic. Always cubic.
- **Hard audio cuts at segment boundaries.** Audible pops. (Hard Rule 3.)
- **Typing text centered on the partial string.** Text slides left as it grows.
- **Sequential sub-agents for multiple animations.** Always parallel.
- **Editing before confirming the strategy.** Never.
- **Re-preprocessing cached sources.** Immutable outputs of immutable inputs.
- **Assuming what kind of video it is.** Look first, ask second, edit last.
- **Using `render.py` for J/L cuts or dissolves.** It flattens them. Use `export_fcpxml.py` for split-edit work and finish in the NLE.
