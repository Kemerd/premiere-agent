---
name: video-use-premiere
description: Edit any video by conversation. Local two-phase preprocessing — Phase A runs Parakeet ONNX speech + Florence-2 visual captions in parallel; Phase B runs CLAP zero-shot audio events against an agent-curated vocabulary derived from the speech + visual timelines. Cut, color grade, generate overlay animations, burn subtitles, OR export FCPXML to Premiere/Resolve/FCP with split edits. For talking heads, montages, tutorials, travel, interviews, workshop / shop footage. No presets, no menus, no cloud transcription. Ask questions, confirm the plan, execute, iterate, persist. Production-correctness rules are hard; everything else is artistic freedom.
---

# Video Use Premiere

## Principle

1. **LLM reasons from one interleaved markdown view + on-demand drill-down.** `merged_timeline.md` is the editor's default reading surface — speech phrases, audio events, and visual captions for every source, all interleaved chronologically by timestamp in a single file. The three per-lane views (`speech_timeline.md`, `audio_timeline.md`, `visual_timeline.md`) are kept on disk as drill-down references for the moments where the merged view is ambiguous and you need to zoom in on one lane. Everything else — filler tagging, retake detection, shot classification, B-roll spotting, emphasis scoring — you derive at decision time.
2. **Speech is primary, visuals are secondary, audio events are tertiary.** Cut candidates come from Parakeet ONNX speech boundaries and silence gaps — that lane is highly accurate and is the editorial spine. Visual captions (Florence-2) are the second source of truth: they answer "what's actually on screen here?" and resolve ambiguous decision points (B-roll spotting, shot continuity, action beats). Audio events (CLAP, zero-shot scoring against a vocabulary) tag non-speech sounds per ~10s window (tools, materials, ambience, music, animals, vehicles). Vocabulary is **agent-curated per project** by reading the speech + visual timelines first — see Phase B below. When audio and visual disagree about *what is happening on screen*, **trust the visual lane.**
3. **Ask → confirm → execute → iterate → persist.** Never touch the cut until the user has confirmed the strategy in plain English.
4. **Generalize.** Do not assume what kind of video this is. Look at the material, ask the user, then edit.
5. **Artistic freedom is the default.** Every specific value, preset, font, color, duration, pitch structure, and technique in this document is a *worked example* from one proven video — not a mandate. Read them to understand what's possible and why each worked. Then make your own taste calls based on what the material actually is and what the user actually wants. **The only things you MUST do are in the Hard Rules section below.** Everything else is yours.
6. **Invent freely.** If the material calls for a technique not described here — split-screen, picture-in-picture, lower-third identity cards, reaction cuts, speed ramps, freeze frames, match cuts, speed ramps over breath, whatever — build it. The helpers are ffmpeg and PIL; the FCPXML exporter handles hard-cut delivery to NLEs. They can do anything the format supports. Do not wait for permission. (Note: J-cuts, L-cuts, and cross-dissolves are currently **deferred** — see "Split edits (DEFERRED)" below.)
7. **Verify your own output before showing it to the user.** If you wouldn't ship it, don't present it.

## Hard Rules (production correctness — non-negotiable)

These are the things where deviation produces silent failures or broken output. They are not taste, they are correctness. Memorize them.

1. **Subtitles are applied LAST in the filter chain**, after every overlay. Otherwise overlays hide captions. Silent failure.
2. **Per-segment extract → lossless `-c copy` concat**, not single-pass filtergraph. Otherwise you double-encode every segment when overlays are added.
3. **30ms audio fades at every segment boundary** (`afade=t=in:st=0:d=0.03,afade=t=out:st={dur-0.03}:d=0.03`). Otherwise audible pops at every cut.
4. **Overlays use `setpts=PTS-STARTPTS+T/TB`** to shift the overlay's frame 0 to its window start. Otherwise you see the middle of the animation during the overlay window.
5. **Master SRT uses output-timeline offsets**: `output_time = word.start - segment_start + segment_offset`. Otherwise captions misalign after segment concat.
6. **Never cut inside a word.** Snap every cut edge to a word boundary from the Parakeet word-level transcript.
7. **Pad every cut edge.** Working window: 30–200ms. ASR timestamps drift 50–100ms — padding absorbs the drift. Tighter for fast-paced, looser for cinematic.
8. **Word-level verbatim ASR only.** Parakeet TDT emits per-token timestamps natively — keep them; never collapse to phrase / SRT shape on the lane output (that loses sub-second gap data). Never normalize fillers either (loses editorial signal — the editor uses `umm` / `uh` / false starts to find candidate cuts).
9. **Cache lane outputs per source.** Never re-run a lane unless the source file itself changed (mtime). The orchestrator handles this; do not pass `--force` reflexively.
10. **Parallel sub-agents for multiple animations.** Never sequential. Spawn N at once via the `Agent` tool; total wall time ≈ slowest one.
11. **Strategy confirmation before execution.** Never touch the cut until the user has approved the plain-English plan.
12. **All session outputs in `<videos_dir>/edit/`.** Never write inside the `video-use-premiere/` project directory.
13. **Pacing preset is REQUIRED before strategy.** Every session must have a pacing preset confirmed by the user (Calm / Measured / Paced / Energetic / Jumpy — default Paced). The preset defines four numbers used by the editor sub-agent: `min_silence_to_remove`, `min_talk_to_keep`, `lead_margin`, and `trail_margin`. See "Pacing presets" below. Never skip the prompt; never invent ad-hoc values.
14. **No split edits (J/L cuts) and no cross-dissolves until further notice.** The editor sub-agent MUST emit `audio_lead = video_tail = transition_in = 0` on every range. They are deferred because the OTIO single-track audio model + per-clip independent frame-snapping causes cumulative audio drift across long timelines (visible as the audio sliding further out of sync with each subsequent cut). Audio at cut boundaries is protected by the 30ms `afade` pair from Hard Rule 3 — that's the current "small crossfade" story. See "Split edits (DEFERRED)".
15. **Editor sub-agent must read `merged_timeline.md` end-to-end before emitting a single EDL range.** The merged view interleaves all three lanes by timestamp — speech (the spine), visual captions (shot continuity / B-roll), and audio events (soundscape hints) — so a single full-file read gives the editor the same triangulated picture it would get from reading each lane separately, in one pass. Never edit from a single lane in isolation. A cut chosen blind to the other lanes will land mid-shot, mid-action, or on a CLAP mis-label. If something in the merged view is ambiguous (e.g. you need word-level timing detail not captured in the phrase grouping, or a denser CLAP scoring than the merged stream shows), drill into the corresponding per-lane file (`speech_timeline.md`, `visual_timeline.md`, `audio_timeline.md`) for that specific moment. The brief in "Editor sub-agent brief" enforces this with a mandatory PRE-FLIGHT block; do not strip it when spawning.
16. **Read `merged_timeline.md` IN FULL. Do not try to be clever about tokens.** The file is caveman-compressed and sentence-delta-deduped at pack time precisely so it fits comfortably in context — typical projects land in the 200KB–1.5MB range, well under any modern model's window. **Forbidden behaviours**, all of which produce silently bad cuts:
    - Reading only the first N lines / last N lines / "a representative sample."
    - `grep`/`rg`-ing for keywords and editing from the matches alone (you lose the chronological structure that makes the merged view useful in the first place).
    - Chunked reads that you abandon partway through ("I have enough…") because the file feels long. You don't have enough. Finish the file.
    - Delegating the read to a sub-agent "to protect the parent context window." The editor IS the agent making the taste calls; outsourcing the read means outsourcing the judgement to something with strictly less context than you. Read it yourself, in full, in the parent agent.
    - Skipping a `Read` chunk because the previous chunk "looked similar." The dedup pass already removed the genuinely similar frames; what's left is signal.
    If the file is genuinely too large for one `Read` call (hits the per-call cap), issue *sequential* `Read` calls with `offset`/`limit` until you have covered every line — not a sample. Treat partial-read shortcuts as a Hard Rule 16 violation regardless of how good the resulting cut looks; the user will catch it and you will be slapped.

Everything else in this document is a worked example. Deviate whenever the material calls for it.

## Directory layout

The skill lives in `video-use-premiere/`. User footage lives wherever they put it. All session outputs go into `<videos_dir>/edit/`.

```
<videos_dir>/
├── <source files, untouched>
└── edit/
    ├── project.md               ← memory; appended every session
    ├── merged_timeline.md       ← DEFAULT reading surface — all 3 lanes
    │                              interleaved chronologically by timestamp
    ├── speech_timeline.md       ← Parakeet phrase-level transcripts  (lane 1, drill-down)
    ├── audio_timeline.md        ← CLAP audio events, coalesced       (lane 2, drill-down, Phase B)
    ├── visual_timeline.md       ← Florence-2 captions @ 1fps         (lane 3, drill-down)
    ├── edl.json                 ← cut decisions
    ├── transcripts/<name>.json  ← cached raw Parakeet words
    ├── audio_tags/<name>.json   ← cached raw CLAP (label, score) events
    ├── audio_vocab.txt          ← agent-curated CLAP vocabulary (Phase B)
    ├── audio_vocab_embeds.npz   ← cached CLAP text embeddings for that vocab
    ├── visual_caps/<name>.json       ← cached raw Florence-2 captions
    ├── comp_visual_caps/<name>.json  ← caveman-compressed visual caps
    │                                   (NLP/spaCy pass; default reading
    │                                   surface for the timelines below)
    ├── audio_16k/<name>.wav     ← shared 16kHz mono PCM (speech lane + CLAP)
    ├── animations/slot_<id>/    ← per-animation source + render + reasoning
    ├── clips_graded/            ← per-segment extracts with grade + fades
    ├── master.srt               ← output-timeline subtitles
    ├── verify/                  ← debug frames / timeline PNGs
    ├── preview.mp4
    ├── final.mp4                ← flattened deliverable (ffmpeg path)
    ├── cut.fcpxml               ← editor-ready timeline, FCPXML 1.10+
    │                              (Resolve / Final Cut Pro X)
    └── cut.xml                  ← editor-ready timeline, FCP7 xmeml
                                   (Premiere Pro native, no XtoCC)
```

## Setup

- **`HF_TOKEN` in `.env` at project root** — only required for speaker diarization (pyannote). Skip if single-speaker.
- **`ffmpeg` + `ffprobe` on PATH.** Hard requirement. Win: `winget install Gyan.FFmpeg`. macOS: `brew install ffmpeg`. Linux: `apt install ffmpeg`.
- **Python deps**: run `install.sh` (Linux/macOS) or `install.bat` (Windows). Installs PyTorch + the `[preprocess,fcpxml]` extras. Optional: `pip install -e .[flash]` for Flash Attention 2 (Florence-2 speedup), `pip install -e .[diarize]` for pyannote speaker diarization, `pip install -e .[parakeet]` to pre-install the NVIDIA Parakeet NeMo fallback (only needed when ONNX Runtime can't load on the host).
- **Speech lane backends**: the default is `parakeet_onnx_lane.py` — NVIDIA Parakeet TDT 0.6B running on ONNX Runtime through a multi-session pool (TensorRT / CUDA / DirectML / CPU EP ladder, English v2 / multilingual v3 auto-routed by language). The only sanctioned alternative is `parakeet_lane.py` (NeMo torch-mode Parakeet) for hosts where ORT can't load — pin via `VIDEO_USE_SPEECH_LANE=nemo`. Output JSON shape is byte-identical between the two so cuts, diarization, and FCPXML export are lane-agnostic. `helpers/health.py --json` surfaces non-default backends in `fallbacks_active` so you know which one is running before the lane fires. **Fully air-gapped?** Pre-download the ONNX directory and set `PARAKEET_ONNX_DIR=/path/to/parakeet-onnx`; the lane skips all network calls. There is no Whisper backend in this codebase by design — Whisper hallucinates on silence and has a known word-timestamp memory regression that crashes long-form runs.
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

**Optional heavy-tier verification** (~2 GB downloads on first run, exercises real Parakeet ONNX + Florence-2 + CLAP on a synthetic 2s clip): tell the user to run `python tests.py --heavy` once after install. Cached separately under the same TTL. Don't trigger this autonomously — it's an explicit user action.

## Helpers

### Preprocessing (Phase A: speech + visual; Phase B: agent-driven CLAP audio events)

> **All helpers live in `helpers/`.** Always invoke them from the skill root as `python helpers/<script>.py …` (the sibling-import pattern they use depends on `helpers/` being the script's own directory, which `sys.path` resolves automatically when you run them by path). Never `cd helpers/` first — `cwd` semantics differ across shells (PowerShell, bash, agentic shells that don't persist `cd`), and the cache layout assumes the project root is the cwd.

**Phase A — speech + visual (default):**

- **`helpers/preprocess_batch.py <videos_dir>`** — auto-discover videos, run the speech (Parakeet ONNX) + visual (Florence-2) lanes with VRAM-aware scheduling. Default entry point. Flags: `--wealthy` (24GB+ GPU), `--diarize`, `--language en`, `--force`, `--skip-speech`, `--skip-visual`, `--include-audio` (opt into running CLAP inline against the baseline vocab — see Phase B for the recommended path instead).
- **`helpers/preprocess.py <video1> [<video2> ...]`** — same orchestrator with explicit file list. Use when you want a subset.
- **`helpers/pack_timelines.py --edit-dir <dir>`** — read the available lane caches (`transcripts/`, `audio_tags/`, `visual_caps/`) and produce `merged_timeline.md` (the editor's default reading surface, all three lanes interleaved by timestamp) plus the three per-lane drill-down views: `speech_timeline.md`, `audio_timeline.md` (only if Phase B has run), `visual_timeline.md`. Pass `--no-merge` to skip the merged view (rare). Safe to call multiple times — re-running after Phase B picks up the new audio events into both the merged file and `audio_timeline.md`. **Caveman compression on visual captions is ON by default** — a spaCy NLP pass strips stop words / determiners / auxiliaries / weak adverbs from every Florence-2 caption before packing, cutting `merged_timeline.md` token cost by ~55-60% on detailed-caption footage with zero loss of editorial signal (entities, actions, colours, shot composition all survive). Cached in `<edit>/comp_visual_caps/` keyed by source mtime + caveman version + lang; subsequent re-packs are instant. Pass `--no-caveman` to read the raw Florence paragraphs (slower, bigger, only useful for debugging what Florence actually said). `--caveman-lang en` (default) picks the spaCy model; `--caveman-procs N` overrides the worker count (default `min(n_files, cpu_count // 2)`); `--force-caveman` re-runs even cached files. Sentence-level fuzzy delta dedup is also applied at pack time: visually static frames collapse to `(same)` in `visual_timeline.md` and disappear entirely from `merged_timeline.md`; slowly-evolving frames emit only the NEW sentences with a `+ ` prefix (think `git diff` additions).
- **`helpers/caveman_compress.py`** — standalone CLI for the caveman pass. Useful for debugging the compression on a single caption (`python helpers/caveman_compress.py "verbose text"`) or for manually batching a `visual_caps/` directory (`python helpers/caveman_compress.py --visual-caps <edit>/visual_caps/`). The pack helper calls it automatically — you only need this directly when iterating on the filter rules.

**Phase B — CLAP audio events with an agent-curated vocabulary (recommended):**

The default audio workflow is: read `speech_timeline.md` + `visual_timeline.md` first, then write a project-specific vocabulary to `<edit>/audio_vocab.txt` (one label per line, 200–1000 entries — broad coverage of the actual content + a healthy "negative" set so silence and unrelated sounds don't latch onto a label), then invoke the audio lane against it. This produces dramatically sharper labels than any baked-in 527-class taxonomy because the vocabulary actually matches what's on screen.

- **`helpers/audio_lane.py <video1> [<video2> ...] --vocab <edit>/audio_vocab.txt --edit-dir <edit>`** — run CLAP zero-shot scoring against your custom vocabulary. Caches text embeddings in `audio_vocab_embeds.npz` so subsequent runs are fast. Flags: `--device {cuda,cpu}`, `--model-tier {base,large}`, `--windows-per-batch N`, `--force`. Without `--vocab`, the lane uses the baked-in baseline vocab from `audio_vocab_default.py` — that's the smoke-test / agent-less fallback.
- After Phase B finishes, re-run `pack_timelines.py` to fold the new audio events into both `merged_timeline.md` (default) and `audio_timeline.md`.

**Individual lanes** (rarely needed — the orchestrator wraps them): `helpers/parakeet_onnx_lane.py`, `helpers/parakeet_lane.py` (NeMo fallback), `helpers/audio_lane.py`, `helpers/visual_lane.py`. Each accepts `--wealthy` and runs standalone.

- **`helpers/extract_audio.py <video>`** — manually extract 16kHz mono WAV. Cached. Mainly for debugging.
- **`helpers/vram.py`** — print detected GPU + the schedule that would be picked. Useful sanity check.

### Editing

- **`helpers/timeline_view.py <video> <start> <end>`** — filmstrip + waveform PNG. On-demand visual drill-down. **Not a scan tool** — use it at decision points, not constantly. The visual_timeline.md replaces 90% of the old "scan with timeline_view" workflow.
- **`helpers/render.py <edl.json> -o <out>`** — per-segment extract → concat → overlays (PTS-shifted) → subtitles LAST → loudness norm → final.mp4. `--preview` for 1080p fast, `--draft` for 720p ultrafast, `--build-subtitles` to generate master.srt inline. Bakes a 30ms `afade` at every boundary (Hard Rule 3) — that's the current "small audio crossfade." J/L cuts and dissolves are DEFERRED (Hard Rule 14); if any are present in the EDL the renderer warns and flattens them.
- **`helpers/grade.py <in> -o <out>`** — ffmpeg filter chain grade. Presets + `--filter '<raw>'` for custom.
- **`helpers/export_fcpxml.py <edl.json> -o cut.fcpxml`** — emit editor-ready timeline files. Hard-cut delivery only right now (Hard Rule 14): the EDL's `audio_lead` / `video_tail` / `transition_in` fields are still consumed by the code path but the editor sub-agent must emit `0` for all three. **Default emits BOTH `cut.fcpxml` AND `cut.xml`** side-by-side from a single timeline build, because Premiere Pro and Resolve/FCP X want different XML dialects: `.fcpxml` (FCPXML 1.10+) is native to DaVinci Resolve and Final Cut Pro X, `.xml` (Final Cut Pro 7 xmeml) is native to Premiere Pro. The recipient picks whichever NLE they live in — no XtoCC conversion required for Premiere. Override with `--targets {both,fcpxml,premiere}`. `--frame-rate 24` (default), 25, 29.97, 30, 60.

For animations, create `<edit>/animations/slot_<id>/` with `Bash` and spawn a sub-agent via the `Agent` tool.

## The process

0. **Health check.** Run `python helpers/health.py --json` first. Cached for 7 days; usually returns instantly. If `status != "ok"`, surface the `advice` strings to the user verbatim and stop. See the "Skill health check" section above.
1. **Inventory + Phase A preprocess.** `ffprobe` every source. `python helpers/preprocess_batch.py <videos_dir>` to run the speech + visual lanes (Parakeet ONNX + Florence-2) — cached by mtime, so this is one-time per source. Then `python helpers/pack_timelines.py --edit-dir <edit>` to produce `merged_timeline.md` (the default reading surface) plus the per-lane drill-down views `speech_timeline.md` and `visual_timeline.md`.
2. **Phase B audio (agent-curated CLAP).** Read `merged_timeline.md` yourself (or `speech_timeline.md` + `visual_timeline.md` if you want the per-lane view), infer what kinds of sounds will plausibly appear in this footage (tools, materials, ambience, music, animals, vehicles, environments — be specific to *this* project), and write a vocabulary list of ~200–1000 short labels to `<edit>/audio_vocab.txt`. Include a healthy negative / unrelated set too so silence and out-of-domain sounds don't all latch onto your top labels. Then run `python helpers/audio_lane.py <videos> --vocab <edit>/audio_vocab.txt --edit-dir <edit>` and re-run `pack_timelines.py` to fold the new events into `merged_timeline.md` and `audio_timeline.md`. Skip this step only if the user explicitly says they don't care about audio events, or pass `--include-audio` to `preprocess_batch.py` upstream to use the baked-in baseline vocab instead (smoke tests, agent-less batch runs).
3. **Pre-scan for problems.** One pass over `merged_timeline.md` end-to-end — every speech phrase, every audio event, every visual caption, all interleaved by timestamp. Note verbal slips, mis-speaks, or phrasings to avoid (from the `"..."` lines). Note shot variety, B-roll candidates, and visually continuous actions you'll want to keep whole (from the `visual:` lines). Treat `(audio: ...)` lines as the lowest-priority hints — verify any CLAP label against the visual line at the same timestamp before trusting it (the model is approximate, especially when the vocabulary is too small or too generic). Drill into the per-lane files only when the merged view leaves you guessing about word-level timing or denser audio scoring than the merged stream shows.
4. **Converse.** Describe what you see in plain English. Ask questions *shaped by the material*. Collect: content type, target length/aspect, aesthetic/brand direction, must-preserve moments, must-cut moments, animation and grade preferences, subtitle needs, **delivery target (flattened mp4 vs FCPXML to NLE)**. Do not use a fixed checklist — the right questions are different every time. **One question is mandatory and not skippable: pacing preset** — present the five options (Calm / Measured / Paced / Energetic / Jumpy) with one-line descriptions and tell the user the default is **Paced**. They can pick a name or just say "use the default." See "Pacing presets" below for the value table you'll feed to the editor sub-agent.
5. **Propose strategy.** 4–8 sentences: shape, take choices, cut direction, **chosen pacing preset (by name + the four numbers it expands to)**, animation plan, grade direction, subtitle style, length estimate, **delivery format**. **Wait for confirmation.**
6. **Execute.** Produce `edl.json` via the editor sub-agent brief. Drill into `timeline_view` at ambiguous moments where the visual_timeline caption alone isn't enough. Build animations in parallel sub-agents. Apply grade per-segment.
   - **Flat MP4 path:** Compose via `render.py`.
   - **NLE handoff path:** Export via `export_fcpxml.py`. Default emits both `cut.fcpxml` (Resolve / FCP X) and `cut.xml` (Premiere Pro native xmeml) from one build — recipient picks. Tell Premiere users to `File → Import → cut.xml` (the `.fcpxml` does **not** work natively in Premiere — that's the file Adobe wants you to run through XtoCC, which we sidestep entirely).
   - **Both:** run them both — they consume the same EDL.
7. **Preview.** `render.py --preview` (or hand `cut.fcpxml` / `cut.xml` to the user to open and scrub in their NLE — the `.xml` for Premiere, the `.fcpxml` for Resolve / FCP X).
8. **Self-eval (before showing the user).** Run `timeline_view` on the **rendered output** (not the sources) at every cut boundary (±1.5s window). Check each image for:
   - Visual discontinuity / flash / jump at the cut
   - Waveform spike at the boundary (audio pop that slipped past the 30ms fade)
   - Subtitle hidden behind an overlay (Rule 1 violation)
   - Overlay misaligned or showing wrong frames (Rule 4 violation)

   Also sample: first 2s, last 2s, and 2–3 mid-points — check grade consistency, subtitle readability, overall coherence. Run `ffprobe` on the output to verify duration matches the EDL expectation.

   If anything fails: fix → re-render → re-eval. **Cap at 3 self-eval passes** — if issues remain after 3, flag them to the user rather than looping forever. Only present the preview once the self-eval passes.
9. **Iterate + persist.** Natural-language feedback, re-plan, re-render. Never re-preprocess unchanged sources. Final render on confirmation. Append to `project.md`.

## Pacing presets

Every session must have a pacing preset (Hard Rule 13). Ask the user up-front in step 4. Default is **Paced**. The preset expands to four numbers that the editor sub-agent applies when picking cut points and trimming silences:

| Preset    | `min_silence_to_remove` | `min_talk_to_keep` | `lead_margin` | `trail_margin` | Vibe |
|-----------|------------------------:|-------------------:|--------------:|---------------:|------|
| Calm      | 500 ms                  | 500 ms             | 500 ms        | 500 ms         | Cinematic, contemplative, breathing room. Long silences are kept; only obvious dead air is trimmed. Documentary, interview, narrative. |
| Measured  | 350 ms                  | 350 ms             | 350 ms        | 350 ms         | Conversational and considered. Professional talking-head, podcast-style, unhurried tutorial. |
| **Paced** *(default)* | **200 ms** | **200 ms**     | **200 ms**    | **200 ms**     | Balanced and modern. Retains rhythm without dragging. Default for tech demos, launch videos, mid-form content. |
| Energetic | 100 ms                  | 100 ms             | 100 ms        | 100 ms         | Tight and punchy. Social-friendly, fast tutorials, hype reels. |
| Jumpy     |  50 ms                  |  50 ms             |  50 ms        |  50 ms         | Ultra-tight "every breath cut" style. Montage, trailer, vlog supercuts. Risks audible artifacts on poor source audio — verify on preview. |

**What each number means** (apply at decision time inside the editor sub-agent):

- **`min_silence_to_remove`** — silences (gaps between words from the speech lane) shorter than this are *kept*; longer ones are candidates to cut out entirely. A Jumpy preset chops out anything ≥50ms; a Calm preset only chops ≥500ms gaps so natural pauses survive.
- **`min_talk_to_keep`** — speech segments shorter than this are *not* worth retaining as standalone clips. Used to filter out single-syllable false starts ("uh-", "th-") that survived the silence filter. Tighter presets keep shorter fragments because the editing rhythm absorbs them.
- **`lead_margin`** — silent padding *before* the first kept word of a clip. Absorbs ASR drift (50–100ms typical) and gives the listener a beat of air before the talker comes in.
- **`trail_margin`** — silent padding *after* the last kept word of a clip. Same purpose at the tail. Together with `lead_margin`, replaces the old "30–200ms working window" guidance with a preset-driven number — but Hard Rule 7's working window still bounds the legal range (so a Calm preset's 500ms margin is the upper end, not infinity).

**Translating to the EDL:** when you build each range, expand the kept word boundary by the margins:

```
range.start = max(0, kept_first_word.start - lead_margin / 1000)
range.end   = min(src_duration, kept_last_word.end + trail_margin / 1000)
```

And while picking ranges, only consider silence gaps `>= min_silence_to_remove` as legitimate cut targets, and discard any candidate kept clip whose net speech duration is `< min_talk_to_keep`.

**Aggressive intra-phrase silence removal (this is the whole point of the preset).** `min_silence_to_remove` applies to **every word-to-word gap in the speech lane**, not just to phrase boundaries or speaker handoffs. If a single phrase like *"today we're going to ⟨640ms gap⟩ drill the pilot holes"* contains a gap ≥ the threshold, the editor MUST split that phrase into two adjacent ranges from the same source — `[..., "going", "to"]` then `[..., "drill", "the", "pilot", "holes"]` — so the dead air is dropped from the timeline. This is how a Paced preset turns a 12-minute walking-talking-head into 7 minutes without losing a word: by deleting hundreds of small breath gaps, hesitations, and thinking pauses scattered inside otherwise-kept speech. Do not romanticize the "natural rhythm of how someone talks" — the preset *is* the rhythm decision. If the user wants those pauses kept, they pick Calm.

**Algorithm to apply per source clip** (run this before picking takes across clips):

```
1. Walk the word-level transcript for the source.
2. Compute gap_i = word[i+1].start - word[i].end  for every adjacent pair.
3. Mark every gap_i >= min_silence_to_remove as a "cut here" point.
4. The kept-speech runs are the spans between consecutive cut points
   (plus the head before the first cut and the tail after the last).
5. Drop any run whose total speech duration is < min_talk_to_keep
   (filters orphan single-syllable false starts).
6. Each surviving run becomes one EDL range, padded with lead_margin
   at the head and trail_margin at the tail (clamped so adjacent ranges
   from the same source don't re-overlap into the silence you just cut).
```

**Boundary clamp** (important — otherwise the margins re-introduce the silence you just removed): when two surviving runs come from the same source and are separated by a cut silence of `gap_ms`, clamp the trailing margin of the first range and the leading margin of the second so their combined padding never exceeds `gap_ms - 60ms` (leave at least 60ms of true silence so the 30ms `afade` pair on each side has room to breathe). Concretely:

```
combined_pad_ms = min(trail_margin + lead_margin, max(0, gap_ms - 60))
prev.trail_pad  = combined_pad_ms * trail_margin / (trail_margin + lead_margin)
next.lead_pad   = combined_pad_ms - prev.trail_pad
```

This split-evenly-by-ratio rule keeps the head/tail balance the user picked while making sure aggressive silence removal stays aggressive.

**Persist the choice.** Record the preset name (and the four expanded values) in `project.md` under "Strategy" so subsequent sessions inherit a sensible default — but still ask if the user wants to keep it.

## Cut craft (techniques)

- **Speech-first.** Candidate cuts from word boundaries and silence gaps. Parakeet TDT is accurate to the word; the speech lane is the editorial spine. Read it interleaved in `merged_timeline.md`; drill into `speech_timeline.md` when you need word-level timing detail.
- **Preserve peaks.** Laughs, punchlines, emphasis beats. Extend past punchlines to include reactions — the laugh IS the beat.
- **Speaker handoffs** benefit from air between utterances. The pacing preset's `lead_margin` + `trail_margin` largely sets this; only override per-handoff if the moment calls for it (e.g. a punchline beat that earns extra silence).
- **Visual context is the second source of truth.** Before committing to *any* non-trivial cut, check the `visual:` lines around the cut point in `merged_timeline.md`. If captions show a continuous action ("person holding drill") spanning your cut, you're cutting in the middle of a shot — usually fine, but be deliberate. Use the visual lane to find B-roll cutaway candidates, match cuts, shot changes, and to decide whether a moment is worth preserving even when speech is silent. Drill into `visual_timeline.md` when you need the full 1-fps caption stream (the merged view drops `(same)` repeats).
- **Audio events are noisy hints, not signals.** The `(audio: ...)` lines in `merged_timeline.md` carry `(drill 0.87)`, `(applause 0.92)`, `(laughter)`, `(power_tool)` markers from CLAP scored against the agent-curated vocab. **The model is approximate** — it mis-labels (music tagged as speech, hammers tagged as drums, room tone tagged as applause), especially when the vocabulary is too small or too generic. Use a marker only as a prompt to *go look* at the visual line (and if needed `timeline_view`) at that timestamp. **Never cut purely on a CLAP label.** When CLAP and Florence-2 disagree about what's happening, trust Florence-2. Drill into `audio_timeline.md` when you want the full per-window scoring instead of the collapsed merged form.
- **Silence gaps are cut candidates — EVERYWHERE, not just at phrase boundaries.** Use the pacing preset's `min_silence_to_remove` as your threshold (Calm 500ms → Jumpy 50ms) and apply it to every adjacent word pair in the speech lane, including gaps that sit *inside* a phrase as the speaker pauses to breathe or think. Splitting a phrase mid-sentence to drop a 400ms thinking pause is the whole point of the preset; it's how you cut runtime without cutting content. The user picked Energetic because they want every breath gone — give them every breath. (Anything shorter than the preset threshold stays as the natural rhythm of the speech. <30ms is always unsafe — mid-phoneme.)
- **Cut padding comes from the pacing preset**, not from per-cut taste. Expand each range by `lead_margin` at the head and `trail_margin` at the tail (see "Pacing presets"). Hard Rule 7's 30–200ms working window still bounds anything outside the preset table — never go below 30ms.
- **Never reason audio and video independently.** Every cut must work on both tracks.

### Split edits (DEFERRED — do not emit)

J-cuts (`audio_lead`), L-cuts (`video_tail`), and cross-dissolves (`transition_in`) are **deferred** until further notice (Hard Rule 14). The EDL schema still accepts these fields and the FCPXML exporter still consumes them — but the editor sub-agent must emit `0` for all three on every range, and reviewers must reject any EDL that doesn't.

**Why deferred:** the current FCPXML build uses an OTIO single-track audio model with per-clip independent frame-snapping. When `audio_lead` or `video_tail` is non-zero, the math `cur_a = target_a_start + a_dur` (in `helpers/export_fcpxml.py`) drifts away from `cur_v` because:

1. Snapping `a_src_start` and `a_src_end` independently to the frame grid doesn't always preserve `a_dur == v_dur` (sub-frame rounding error per clip).
2. A single audio track can't actually overlap clips, so an L-cut tail forces the *next* clip's audio backward — but the gap-padding only handles the positive case (audio LATER than video). Negative gaps silently collapse to zero, so the next J-cut's audio starts at the wrong timeline position.
3. Errors compound across every cut. On a 50-cut timeline the audio is visibly sliding out of alignment under the video by the end (the symptom the user reported).

**Path forward when we revisit this** (not now): switch to two audio tracks (A1 carries the speech, A2 carries the lead/tail spillover) so overlaps are legal; lock the audio source range to the same snapped frame edges as the video; advance both `cur_v` and `cur_a` from a single canonical "next timeline position" rather than independent counters.

**For now, the only legal split-edit story is the 30ms `afade` pair** at every cut boundary (Hard Rule 3) — that's the "small audio crossfade" the renderer bakes in to prevent boundary pops. It's not a J/L cut; it's a click-suppression fade. Good enough until the multi-track path lands.

**EDL fields the editor MUST emit as zero:**

```json
{"source": "C0103", "start": 12.20, "end": 18.45, "beat": "ANSWER",
 "audio_lead": 0.0,        // DEFERRED — must be 0
 "video_tail": 0.0,        // DEFERRED — must be 0
 "transition_in": 0.0}     // DEFERRED — must be 0
```

**Render path matrix (current state):**

| Output                       | Hard cuts | 30ms boundary afade | J/L cuts | Dissolves | Time-squeeze (≤10x) |
|------------------------------|-----------|---------------------|----------|-----------|---------------------|
| `render.py` → mp4            | ✓         | ✓ (Hard Rule 3)     | DEFERRED | DEFERRED  | ✓ (`setpts` + `atempo`/`anullsrc`) |
| `export_fcpxml.py` → fcpxml  | ✓         | colorist's job      | DEFERRED | DEFERRED  | ✓ (`<timeMap>` + Premiere `timeremap`) |

If the user explicitly asks for J/L cuts or dissolves: explain the deferral honestly, ship hard cuts, and offer to log it in `project.md` as an outstanding item for the day the multi-track path lands.

### Time-squeezing (timelapse) — when the source has long no-speech stretches

Real-world footage is often "1 minute of explanation, then 25 minutes of silently doing the work, then 2 minutes of wrap-up." Cutting the 25 minutes entirely throws away the visual story; keeping it 1× bores the viewer. The third option is **time-squeezing**: compress the work segment into a 5–30s timelapse on the output timeline so the viewer sees the whole arc in seconds. Both delivery paths (`render.py` and `export_fcpxml.py`) support this natively — see the matrix above.

**When to reach for it.** Look for stretches in `merged_timeline.md` where BOTH of these are true:

1. **Visually continuous activity** — long runs of `(same)` collapses in `visual_timeline.md` OR successive `visual:` lines describing the same scene with mild variation ("a person sanding a board" → "hand pushing a sander across wood" → "sawdust accumulating"). Pure dead-air (camera abandoned on a tripod, nothing moving) is a CUT candidate, not a timelapse candidate.
2. **A coherent story-of-progress** the viewer benefits from seeing compressed: assembly, packing, walking, driving, cooking, painting, prep, teardown. If the squeezed result wouldn't read as "watch them do this thing fast," cut instead.

**Speech inside the stretch is a judgement call, not a blocker.** The real test is "**does the viewer need to hear this?**" not "is anyone talking?":

- **Load-bearing speech** (instruction, explanation, narration that carries the cut, the punchline that lands the beat): split AROUND it. Emit a 1× range for the words, then a `speed > 1.0` range for the silent / no-words-that-matter middle, then another 1× range for whatever talks next. This is the cleaner edit when the language is doing real work.
- **Filler speech** (mumbling, swearing at a misplaced screw, idle narration of "okay … there we go … hmm"; rambling backstory the viewer doesn't need; 30 minutes of casual chatter while building that isn't actually teaching anything): squeeze right over it. With `audio_strategy="drop"` (the default at `speed != 1.0`) the words vanish along with the room tone, the visual story plays compressed, and the viewer thanks you for the 28 minutes of their life back. The decision is editorial — would keeping the words make the video better, or just longer?

When in doubt: lean toward squeezing over filler speech rather than splitting into a hundred tiny 1× ranges. The video is for the viewer.

If the stretch fails the two criteria above (no continuous activity, or no story-of-progress), just CUT it — squeezing nothing into less nothing is wasted budget.

**How to size the squeeze.** Pick `speed` so the resulting OUTPUT segment lands between **5–30 seconds** (the sweet spot where the viewer registers the activity without it overstaying). Examples:

| Source stretch | Speed | Output | Read as |
|----------------|-------|--------|---------|
|  30 s          | 4x    |  7.5s  | quick montage |
|  2 min         | 8x    |  15s   | "they assembled it" |
|  5 min         | 10x   |  30s   | full build sequence |
| 10 min         | 10x   |  60s   | over budget — split into two squeezes with a beat between, or cut |
| 30 min         | 10x   | 180s   | far over — pick the visually richest 5-min sub-stretch and squeeze that; cut the rest |

**Hard ceiling: `speed = 10.0` (1000%).** Both helpers clamp to it with a warning. Beyond that the retime starts decimating frames and looks broken on standard 24/30fps source — and besides, if you wanted >10x you should have CUT.

**Audio strategy.** Two values, picked automatically from `speed`:

- `audio_strategy = "drop"` (default at `speed != 1.0`): the audio track is silenced over the squeezed range. Both helpers emit a silent gap. This is the right answer for ~95% of timelapses — sustained shop noise / room tone sped up 5–10× sounds awful, and the editor will drop a music bed under the squeeze in the NLE.
- `audio_strategy = "keep"`: the audio is retimed alongside the video. In `render.py` we use `atempo` (pitch-preserving — sounds natural at moderate speeds). In FCPXML / xmeml the audio gets a matching retime element and will be chipmunk-y unless the editor toggles "Maintain Audio Pitch" (Premiere) / "Preserve Pitch" (FCP X) on the clip — both NLEs offer this as a one-click clip property. Use this only when there's a specific reason to keep the source audio (recognisable voice in the background, distinctive ambient texture).

**Editorial discipline for time-squeezing** (not in the numbered Hard Rules block — these are taste calls, not silent-failure issues):

- **Decide per-stretch: is the speech worth keeping?** Load-bearing speech earns a 1× split around it; filler speech gets squeezed over with `audio_strategy="drop"`. See the "judgement call, not a blocker" paragraph above. There is no universal rule — read the words and ask whether the viewer benefits from hearing them.
- **Cut FIRST, squeeze SECOND.** Apply the pacing preset's silence-removal pass first (drop dead air ≥ `min_silence_to_remove`); then identify the surviving long stretches that fit the "coherent activity + story-of-progress" criteria; then squeeze those ranges. Squeezing dead air is just slower nothing.
- **Word-boundary discipline still applies on adjacent 1× ranges** (Hard Rule 6 / 7). The squeezed range itself doesn't need word-boundary alignment when `audio_strategy="drop"` (the audio's gone anyway), but pad it generously (~1–2s on each side of the activity) so the viewer's eye registers the speed change cleanly.
- **`speed` field is OPTIONAL and defaults to 1.0.** Untouched EDLs behave exactly as before. Only emit a `speed` value when you're actively squeezing.
- **The retime key is `speed` — NOT `timelapse_speed`, NOT `clip_speed`, NOT `retime`.** Recurring agent footgun: beats named `*_TIMELAPSE` invite an autocomplete-style `"timelapse_speed": 8` that the exporter cannot recognise. The export pipeline does a defensive textual rename of `timelapse_speed` → `speed` before parsing, but do NOT rely on it — write the canonical key the first time and don't invent synonyms. The `notes_for_editor` block at the end of the EDL is an especially common offender; if you mention the field by name there, call it `speed`.

**EDL example with a timelapse:**

```json
"ranges": [
  {"source": "C0210", "start":   2.40, "end":  62.30, "beat": "INTRO",
   "quote": "today we're going to build a bench from scratch"},
  {"source": "C0210", "start":  68.10, "end": 1248.40, "beat": "BUILD",
   "speed": 10.0, "audio_strategy": "drop",
   "reason": "19.6 min of cutting/sanding/assembly with no speech and continuous visual activity → 118s timelapse; editor adds music in NLE"},
  {"source": "C0210", "start": 1255.20, "end": 1310.80, "beat": "REVEAL",
   "quote": "and that's the finished bench"}
]
```

## The timelines (primary reading view)

`pack_timelines.py` reads each lane's JSON cache and produces four markdowns: one unified view (`merged_timeline.md`, the editor's default reading surface) plus three per-lane drill-down files. They share an addressing scheme: every line carries `[start-end]` (or `[t]` for visual frames) in seconds-from-clip-start (the per-lane files) or `[HH:MM:SS]` (the merged file), so a line read out of any timeline can be directly addressed in `edl.json` cut ranges.

**`merged_timeline.md`** — the **default reading surface for the editor sub-agent.** All three lanes interleaved chronologically by timestamp into a single per-source section. Speech phrases as `"..."`, audio events as `(audio: label1, label2, ...)`, visual captions as `visual: ...`. One file, one full read, every event in order — the editor gets the same triangulated picture it would get from reading three lanes in parallel, without the three-way cross-reference cost.

Visual captions are **caveman-compressed** by default (NLP pass over the Florence-2 paragraphs — stop words / determiners / auxiliaries / weak adverbs stripped, entities / actions / colours / shot composition kept). Reads like a telegram but every fact survives, and the LLM editor reconstructs the grammar effortlessly. Lines prefixed with `+ ` are sentence-level deltas — only the NEW sentences vs the prior caption are shown (think `git diff` additions); lines without `+ ` are full re-descriptions (treat as a likely shot change). Frames whose caption fully overlaps the prior frame are dropped from the merged view entirely.

```
## C0108  (duration: 87.4s, ...)
  [00:00:02] visual: Workbench hand tools laid brown wooden surface.
  [00:00:03] "okay so today we're going to drill the pilot holes"
  [00:00:12] (audio: drill 0.87, power_tool 0.71)
  [00:00:12] visual: Person holding cordless drill metal panel rivet holes.
  [00:00:13] visual: + Close - drill bit entering metal sparks visible.
  [00:00:18] "good, pass me the deburring tool"
```

The three per-lane files below remain on disk for **drill-down only** — read them when the merged view is ambiguous and you need word-level timing, the dedup'd 1-fps caption stream, or the full per-window CLAP scoring.

**`speech_timeline.md`** — phrase-grouped Parakeet transcript. Phrases break on silence ≥0.5s OR speaker change. Drill in here when you need word-level timing detail beyond what the phrase grouping in the merged view shows.

```
## C0103  (duration: 43.0s, 8 phrases)
  [002.52-005.36] S0 Ninety percent of what a web agent does is completely wasted.
  [006.08-006.74] S0 We fixed this.
```

**`audio_timeline.md`** — CLAP zero-shot scoring against the agent-curated vocabulary in `audio_vocab.txt`, one row per ~10s sliding window with the top-K labels above the per-label threshold. Adaptive vocabulary — the labels match the actual project content (specific tools, materials, ambience, music character, animals, vehicles, environments) instead of mapping into a fixed 527-class taxonomy. Drill in here when you want every per-window CLAP row instead of the collapsed "(audio: ...)" lines in the merged view, or to find sounds the visual lane can't see (off-screen tools, room tone changes). When CLAP and Florence-2 disagree about what's on screen, trust Florence-2 — CLAP is the authority on the **soundscape**, not the picture.

```
## C0108  (duration: 87.4s, 27 events)
  [012.04-012.40] drill (0.87), power_tool (0.71)
  [012.18-012.30] metal_scraping (0.62)
  [018.50-019.10] hammer (0.55)
```

If `audio_timeline.md` doesn't exist or looks coarse, you haven't run Phase B yet — see step 2 of "The process" below for the workflow.

**`visual_timeline.md`** — Florence-2 detailed captions @ 1fps. Consecutive identical captions collapse to `(same)`. Drill in here when you need the full 1-fps caption stream (the merged view drops the `(same)` repeats) or to spot shots, B-roll candidates, match cuts, action with no surrounding speech. **This is the second source of truth after speech** — when classifying *what is happening* in a moment, prefer this over the audio events lane.

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

PRE-FLIGHT (mandatory — do this before writing a single range):
  1. Read merged_timeline.md  END-TO-END.  IN FULL.  EVERY LINE.  This is
     your default reading surface — speech phrases ("..."), audio events
     ((audio: ...)), and visual captions (visual: ...) for every source,
     all interleaved chronologically by timestamp. One file, one full
     read; you get the same triangulated picture you would from reading
     the three per-lane files separately, in a single pass.

     Do not try to be clever about tokens (Hard Rule 16). The file is
     caveman-compressed and sentence-delta-deduped at pack time so it
     fits in context — typical projects are 200KB–1.5MB, comfortably
     within any modern window. Forbidden:
       * Reading only the first/last N lines or a "representative sample"
       * grep/rg-for-keywords and editing from the matches alone
       * Stopping a chunked read early because "I have enough"
       * Delegating the read to a SUB-sub-agent "to save context" — YOU
         are the editor, you make the taste calls, you read the file
       * Skipping a chunk because the prior chunk "looked similar"
     If the file exceeds the per-Read cap, issue sequential Read calls
     with offset/limit until every line is covered. Cover the WHOLE file.

  2. Internalize the priority order: speech is the spine (every cut
     start/end must land on a word boundary), visual is the second source
     of truth for shot continuity / what's on screen, audio events are
     noisy hints only. When (audio: ...) and visual: disagree about what's
     happening on screen at the same timestamp, trust visual.
  3. Drill into the per-lane files (speech_timeline.md / visual_timeline.md
     / audio_timeline.md) ONLY for moments where the merged view is
     ambiguous and you need word-level timing, the full 1-fps caption
     stream (with `(same)` repeats), or per-window CLAP scoring detail.
  If merged_timeline.md is missing from <edit>/, STOP and report — re-run
  `python helpers/pack_timelines.py --edit-dir <edit>` to regenerate it.
  Skipping the merged read and editing from a single per-lane file alone
  is a Hard Rule violation (#15). Partial-read shortcuts are a Hard Rule
  16 violation regardless of how the resulting cut looks.

INPUTS (in priority order — trust them in this order when they disagree):
  - merged_timeline.md  (DEFAULT reading surface; all 3 lanes interleaved
                         by timestamp; one file, full read)
  - speech_timeline.md  (drill-down: phrase-level Parakeet transcripts;
                         ACCURATE, the spine. Word-boundary precision)
  - visual_timeline.md  (drill-down: 1fps Florence-2 captions, full stream
                         including `(same)` repeats; second source of truth
                         for what's on screen / what's happening)
  - audio_timeline.md   (drill-down: CLAP zero-shot scoring against an
                         agent-curated vocabulary, top-K labels per ~10s
                         window. Describes the soundscape — tools,
                         materials, ambience, music. Trust for non-speech
                         audio; defer to visual_timeline for what's on
                         screen)
  - Product/narrative context: <2 sentences from the user>
  - Speaker(s): <name, role, delivery style note>
  - Expected structure: <pick an archetype or invent one>
  - Verbal slips to avoid: <list from the pre-scan pass>
  - Target runtime: <seconds>
  - Delivery: <flat mp4 / fcpxml / both>
  - Pacing preset: <Calm | Measured | Paced | Energetic | Jumpy>
      min_silence_to_remove: <ms>     // skip silence gaps shorter than this
      min_talk_to_keep:      <ms>     // drop kept clips shorter than this
      lead_margin:           <ms>     // pad before first kept word
      trail_margin:          <ms>     // pad after  last  kept word

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
  - Pad each range using the pacing preset:
        range.start = max(0, first_kept_word.start - lead_margin/1000)
        range.end   = min(src_dur, last_kept_word.end + trail_margin/1000)
    Stay inside Hard Rule 7's 30-200ms working window.
  - Apply min_silence_to_remove to EVERY adjacent word-to-word gap in
    the speech lane, including gaps that sit INSIDE a phrase. If a phrase
    contains a >= threshold gap, split it into two adjacent EDL ranges
    from the same source so the dead air drops out of the timeline.
    This is the whole point of the preset — it's how an Energetic pass
    chops a 12-min walk-and-talk into 7 min without losing any words.
    Anything shorter than the threshold stays as the speech's natural
    rhythm.
  - Per-source pre-pass before take selection:
       1. walk word transcript, compute every gap_i = w[i+1].start - w[i].end
       2. mark gap_i >= min_silence_to_remove as cut points
       3. surviving runs (between cut points) become candidate ranges
       4. drop any run with net speech duration < min_talk_to_keep
       5. pad each surviving run with lead_margin / trail_margin
       6. clamp adjacent same-source pads so combined pad <= gap_ms - 60ms
          (leaves room for the 30ms afade pair at each boundary)
  - Discard any kept clip whose net speech duration is < min_talk_to_keep
    (filters single-syllable false starts that survived silence trimming).
  - Cross-reference visual_timeline.md before committing to a cut whose
    audio looks clean — make sure you're not cutting in the middle of a
    visually continuous action you wanted to keep whole. The visual lane
    classifies the moment; the audio events lane only suggests where to look.
  - HARD RULE 14: split edits and dissolves are DEFERRED.
    Emit audio_lead = video_tail = transition_in = 0.0 on EVERY range.
    Do not use J-cuts, L-cuts, or cross-dissolves under any circumstances.
    The 30ms afade pair at every boundary (baked into render.py) is the
    only "audio crossfade" available right now and is sufficient to
    suppress boundary pops.
  - TIME-SQUEEZING (timelapse) is available — see "Time-squeezing" in
    SKILL.md for the full editorial rules. Short version: when you find
    a stretch with visually continuous activity (assembly, walking,
    prep, teardown) AND a coherent story-of-progress, emit a single
    range with "speed": <2.0..10.0> over that stretch. Pick speed so
    the OUTPUT lands in the 5-30s sweet spot. Default audio_strategy
    is "drop" (silent gap) — accept the default unless there's a
    specific reason to keep the source audio.

    Speech inside the stretch is a JUDGEMENT CALL, not a blocker. The
    real question is "does the viewer need to hear this?":
      * Load-bearing speech (instruction, explanation, narration that
        carries the cut, the punchline) → SPLIT around it: 1× for the
        words, speed>1 for the silent middle, 1× for the next words.
      * Filler speech (mumbling, swearing at a screw, "okay there we
        go", rambling that doesn't teach anything) → SQUEEZE right
        over it; "drop" makes the words vanish with the room tone and
        the visual story plays compressed.
    When in doubt, lean toward squeezing over filler rather than
    splitting into many tiny 1× ranges. The video is for the viewer.

    Pure dead-air (camera abandoned, nothing moving) is a CUT, not a
    squeeze. The `speed` field is OPTIONAL and defaults to 1.0; omit
    it on every normal range.

    THE KEY IS LITERALLY `"speed"`. Not `"timelapse_speed"`, not
    `"clip_speed"`, not `"retime"`. If you name your beats with a
    `_TIMELAPSE` suffix the autocomplete tug toward `timelapse_speed`
    is real — resist it. The exporter does a textual rename of
    `timelapse_speed` → `speed` as a safety net, but every other
    invented synonym is silently dropped and the range plays at 1×.
  - Unavoidable slips are kept if no better take exists. Note them in "reason".
  - If over budget, revise: drop a beat or trim tails, OR squeeze a
    qualifying long no-speech stretch (per "Time-squeezing" above)
    instead of cutting it entirely. Report total and self-correct.

OUTPUT (JSON array, no prose — split-edit fields forced to 0.0 per
Hard Rule 14; `speed` and `audio_strategy` are OPTIONAL and only
appear on time-squeezed ranges):
  [{"source": "C0103", "start": 2.42, "end": 6.85, "beat": "HOOK",
    "audio_lead": 0.0, "video_tail": 0.0, "transition_in": 0.0,
    "quote": "...", "reason": "..."},
   {"source": "C0210", "start": 68.10, "end": 1248.40, "beat": "BUILD",
    "audio_lead": 0.0, "video_tail": 0.0, "transition_in": 0.0,
    "speed": 10.0, "audio_strategy": "drop",
    "reason": "19.6 min of silent assembly → 118s timelapse"}, ...]

Return the final EDL and a one-line total runtime check.
```

## Color grade, Subtitles, Animations — load on demand

These three feature areas live in **`references/`** and are **not** loaded into context by default. Each is cold-path — only ~10% of sessions touch them — so paying tokens for them on every invocation is wasted. When the user asks for one, read the matching file in full **before** proposing strategy for that feature:

- **Color grade** → `references/color-grade.md` — ASC CDL mental model, shipped filter chain presets (`warm_cinematic`, `neutral_punch`, `none`), per-segment-during-extraction discipline, FCPXML "don't bake the grade" rule, decision flow at a frame.
- **Subtitles** → `references/subtitles.md` — chunking / case / placement reasoning, the `bold-overlay` and `natural-sentence` worked styles with full `force_style` strings, FCPXML delivery (ship `master.srt` alongside, don't burn), content → style decision shortcuts.
- **Animations** → `references/animations.md` — tool options (PIL / Manim / Remotion), duration heuristics, animation-payoff timing against `speech_timeline.md`, `ease_out_cubic` / `ease_in_out_cubic` snippets, typing-text anchor trick, parallel sub-agent brief template.

The relevant **Hard Rules** for these features stay in the numbered block above so they bind even if you forget to read the reference: subtitles LAST (Rule 1), per-segment grade during extraction (Rule 2), 30ms boundary afade (Rule 3), overlay `setpts=PTS-STARTPTS+T/TB` (Rule 4), output-timeline SRT offsets (Rule 5), parallel sub-agents for multiple animations (Rule 10). Read the reference for the *taste-call* depth on top of those non-negotiables.

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
     "audio_lead": 0.0, "video_tail": 0.0, "transition_in": 0.0},
    {"source": "C0210", "start": 68.10, "end": 1248.40,
     "beat": "BUILD", "reason": "19.6 min of silent assembly → 118s timelapse",
     "speed": 10.0, "audio_strategy": "drop",
     "audio_lead": 0.0, "video_tail": 0.0, "transition_in": 0.0}
  ],
  "pacing_preset": "Paced",
  "pacing": {"min_silence_to_remove_ms": 200,
             "min_talk_to_keep_ms": 200,
             "lead_margin_ms": 200,
             "trail_margin_ms": 200},
  "grade": "warm_cinematic",
  "overlays": [
    {"file": "edit/animations/slot_1/render.mp4", "start_in_output": 0.0, "duration": 5.0}
  ],
  "subtitles": "edit/master.srt",
  "total_duration_s": 87.4
}
```

`grade` is a preset name or raw ffmpeg filter (ignored by FCPXML export — colorist's job). `overlays` are rendered animation clips (ffmpeg path only). `subtitles` is optional and applied LAST (ffmpeg path) or imported as a captions track (FCPXML path). `pacing_preset` + `pacing` record the user's chosen preset and its expanded values for traceability (the editor sub-agent already applied them when picking ranges; downstream tools may re-read them for reporting). `audio_lead` / `video_tail` / `transition_in` per range are DEFERRED (Hard Rule 14) and must always be `0.0`.

## Memory — `project.md`

Append one section per session at `<edit>/project.md`:

```markdown
## Session N — YYYY-MM-DD

**Strategy:** one paragraph describing the approach
**Pacing:** preset name + the four expanded ms values (so next session can default to it)
**Decisions:** take choices, cuts, grades, animations + why
**Reasoning log:** one-line rationale for non-obvious decisions
**Outstanding:** deferred items
```

On startup, read `project.md` if it exists and summarize the last session in one sentence before asking whether to continue.

## Anti-patterns

Things that consistently fail regardless of style:

- **Hierarchical pre-computed codec formats** with USABILITY / tone tags / shot layers. Over-engineering. Derive from the timelines at decision time.
- **Hand-tuned moment-scoring functions.** The LLM picks better than any heuristic you'll write.
- **SRT / phrase-level lane output.** Loses sub-second gap data. Always word-level verbatim from the speech lane (Parakeet TDT emits per-token timestamps natively — keep them).
- **Re-running `helpers/preprocess_batch.py --force` reflexively.** The mtime-based cache is correct; bypass only when the source file actually changed or you've upgraded a model.
- **Reading `transcripts/*.json` directly.** Use `merged_timeline.md` (or `speech_timeline.md` for a speech-only drill-down). Same data, 1/10 the tokens, phrase-aligned.
- **Reading the three per-lane timelines separately when `merged_timeline.md` exists.** The merged view is the editor's default reading surface — one file, all three lanes interleaved by timestamp. Open the per-lane files only as drill-down references for ambiguous moments (Hard Rule 15).
- **"Saving tokens" by partial-reading `merged_timeline.md`.** First-N-lines, last-N-lines, "representative sample," grep-and-edit-from-matches, abandoning a chunked read because "I have enough," delegating the full read to a sub-agent to "protect the parent context window" — all forbidden (Hard Rule 16). The file is compressed and dedup'd at pack time so it fits; if it exceeds one `Read` call, issue sequential `Read` calls with `offset`/`limit` until every line is covered. The editor IS the agent making the taste calls; outsourcing the read outsources the judgement.
- **Burning subtitles into base before compositing overlays.** Overlays hide them. (Hard Rule 1.)
- **Single-pass filtergraph when you have overlays.** Double re-encodes. Use per-segment extract → concat.
- **Linear animation easing.** Looks robotic. Always cubic.
- **Hard audio cuts at segment boundaries.** Audible pops. (Hard Rule 3.)
- **Typing text centered on the partial string.** Text slides left as it grows.
- **Sequential sub-agents for multiple animations.** Always parallel.
- **Editing before confirming the strategy.** Never.
- **Re-preprocessing cached sources.** Immutable outputs of immutable inputs.
- **Assuming what kind of video it is.** Look first, ask second, edit last.
- **Skipping the pacing prompt or inventing ad-hoc cut-padding numbers.** Hard Rule 13 — every session uses one of the five presets; default is Paced.
- **Emitting non-zero `audio_lead` / `video_tail` / `transition_in`.** Hard Rule 14 — split edits and dissolves are deferred. The current FCPXML pipeline drifts the audio across long timelines under non-zero values; until the multi-track rebuild lands, hard cuts only.
- **Squeezing pure dead air instead of cutting it.** A camera abandoned on a tripod with nothing moving is not a timelapse candidate — it's a CUT candidate. Time-squeezing is for visually continuous *activity* (assembly, walking, prep, teardown). Compressing 30 minutes of nothing into 3 seconds of nothing is just slower nothing.
- **Picking `speed` so the squeezed result lands < 5s or > 30s.** Under 5s the viewer doesn't register the activity; over 30s it overstays its welcome. Re-pick speed to land in the 5–30s sweet spot, OR split a long source stretch into multiple squeezes with a beat between, OR cut some of it.
- **Setting `speed > 10.0` and expecting it to apply.** Both helpers clamp to 10.0 (1000%) with a warning. Beyond that retime decimates frames and looks broken; if you wanted >10x you should have cut.
- **Splitting around every word of filler speech inside an otherwise-squeezable stretch.** If the speech isn't load-bearing, squeeze right over it with `audio_strategy="drop"`. A hundred 1× micro-ranges interleaved with a hundred speed=8 micro-ranges is worse cut than one honest squeeze that drops the rambling. The video is for the viewer.
