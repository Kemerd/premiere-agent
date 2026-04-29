---
name: premiere-agent
description: Edit any video by conversation. Local two-phase preprocessing — Phase A runs Parakeet ONNX speech + Florence-2 visual captions in parallel; Phase B runs CLAP zero-shot audio events against an agent-curated vocabulary derived from the speech + visual timelines. Reasons over a single interleaved `merged_timeline.md` and exports `cut.fcpxml` + `cut.xml` + `master.srt` to Premiere / Resolve / Final Cut Pro X — XML-only delivery, the cut lives in the NLE. For talking heads, montages, tutorials, travel, interviews, workshop / shop footage. No presets, no menus, no cloud transcription. Ask questions, confirm the plan, execute, iterate, persist. Production-correctness rules are hard; everything else is artistic freedom.
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

1. **Master SRT uses output-timeline offsets**: `output_time = word.start - segment_start + segment_offset`. Otherwise captions misalign across the cut. `helpers/build_srt.py` does this for you — call it (or let `helpers/export_fcpxml.py` call it) instead of hand-rolling timestamps.
2. **Never cut inside a word. Verify every EDL anchor with `helpers/find_quote.py`.** Snap every cut edge to a word boundary from the Parakeet word-level transcript. The supported interface is `helpers/find_quote.py` — pass the clip stem, the integer `M:SS-M:SS` range read straight off the `"..."` line in `merged_timeline.md` (guaranteed superset of the underlying float span), and a quote substring; the helper returns word-precise `{first_word, last_word, prev_word, next_word, lead_silence_s, trail_silence_s, cut_window}` JSON. Read `first_word.start` as the in-point and `last_word.end` as the out-point, then apply the pacing preset's lead/trail margins clamped against `cut_window.safe_in_s` / `safe_out_s` so each side leaves ≥60ms of true silence for the NLE's audio crossfade to breathe in. **Never grep / hand-parse `transcripts/<stem>.json`** — the helper is 50-100x faster, bounds-checks the result against the integer range, and removes the off-by-one class of bugs entirely. See "Word-boundary verification" below for the worked example.
3. **Pad every cut edge.** Working window: 30–200ms. ASR timestamps drift 50–100ms — padding absorbs the drift. Tighter for fast-paced, looser for cinematic.
4. **Word-level verbatim ASR only.** Parakeet TDT emits per-token timestamps natively — keep them; never collapse to phrase / SRT shape on the lane output (that loses sub-second gap data). Never normalize fillers either (loses editorial signal — the agent uses `umm` / `uh` / false starts to find candidate cuts).
5. **Cache lane outputs per source.** Never re-run a lane unless the source file itself changed (mtime). The orchestrator handles this; do not pass `--force` reflexively.
6. **Strategy confirmation before execution.** Never touch the cut until the user has approved the plain-English plan.
7. **All session outputs in `<videos_dir>/edit/`.** Never write inside the `premiere-agent/` project directory.
8. **Pacing preset is REQUIRED before strategy.** Every session must have a pacing preset confirmed by the user (Calm / Measured / Paced / Energetic / Jumpy — default Paced). The preset defines four numbers the agent applies when picking cut points: `min_silence_to_remove`, `min_talk_to_keep`, `lead_margin`, and `trail_margin`. See "Pacing presets" below. Never skip the prompt; never invent ad-hoc values.
9. **No split edits (J/L cuts) and no cross-dissolves until further notice.** Emit `audio_lead = video_tail = transition_in = 0` on every range. They are deferred because the OTIO single-track audio model + per-clip independent frame-snapping causes cumulative audio drift across long timelines (visible as the audio sliding further out of sync with each subsequent cut). The NLE adds its own audio crossfade on import — that's the current "small crossfade" story until the multi-track rebuild lands. See "Split edits (DEFERRED)".
10. **Read `merged_timeline.md` end-to-end before emitting a single EDL range.** The merged view interleaves all three lanes by timestamp — speech (the spine), visual captions (shot continuity / B-roll), and audio events (soundscape hints) — so a single full-file read gives you the same triangulated picture you would get from reading each lane separately, in one pass. Never edit from a single lane in isolation. A cut chosen blind to the other lanes will land mid-shot, mid-action, or on a CLAP mis-label. If something in the merged view is ambiguous (e.g. you need word-level timing detail not captured in the phrase grouping, or a denser CLAP scoring than the merged stream shows), drill into the corresponding per-lane file (`speech_timeline.md`, `visual_timeline.md`, `audio_timeline.md`) for that specific moment.
11. **Read `merged_timeline.md` IN FULL. Do not try to be clever about tokens.** The file is caveman-compressed and sentence-delta-deduped at pack time precisely so it fits comfortably in context — typical projects land in the 200KB–1.5MB range, well under any modern model's window. **Forbidden behaviours**, all of which produce silently bad cuts:
    - Reading only the first N lines / last N lines / "a representative sample."
    - `grep`/`rg`-ing for keywords and editing from the matches alone (you lose the chronological structure that makes the merged view useful in the first place).
    - Chunked reads that you abandon partway through ("I have enough…") because the file feels long. You don't have enough. Finish the file.
    - Delegating the read to another agent "to protect context window." YOU are the agent making the taste calls; outsourcing the read means outsourcing the judgement.
    - Skipping a `Read` chunk because the previous chunk "looked similar." The dedup pass already removed the genuinely similar frames; what's left is signal.
    If the file is genuinely too large for one `Read` call (hits the per-call cap), issue *sequential* `Read` calls with `offset`/`limit` until you have covered every line — not a sample. Treat partial-read shortcuts as a Hard Rule 11 violation regardless of how good the resulting cut looks; the user will catch it and you will be slapped.

Everything else in this document is a worked example. Deviate whenever the material calls for it.

## Directory layout

The skill lives in `premiere-agent/`. User footage lives wherever they put it. All session outputs go into `<videos_dir>/edit/`.

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
    ├── transcripts/<name>.json  ← cached raw Parakeet words (read via
    │                              helpers/find_quote.py — never grep)
    ├── audio_tags/<name>.json   ← cached raw CLAP (label, score) events
    ├── audio_vocab.txt          ← agent-curated CLAP vocabulary (Phase B)
    ├── audio_vocab_embeds.npz   ← cached CLAP text embeddings for that vocab
    ├── visual_caps/<name>.json       ← cached raw Florence-2 captions
    ├── comp_visual_caps/<name>.json  ← caveman-compressed visual caps
    │                                   (NLP/spaCy pass; default reading
    │                                   surface for the timelines below)
    ├── audio_16k/<name>.wav     ← shared 16kHz mono PCM (speech lane + CLAP)
    ├── master.srt               ← output-timeline captions sidecar (build_srt.py;
    │                              shipped automatically by export_fcpxml.py)
    ├── verify/                  ← debug frames / timeline PNGs
    ├── cut.fcpxml               ← editor-ready timeline, FCPXML 1.10+
    │                              (Resolve / Final Cut Pro X)
    └── cut.xml                  ← editor-ready timeline, FCP7 xmeml
                                   (Premiere Pro native, no XtoCC)
```

**Color is the colorist's job; this skill does not emit a grade.** **There is no flat-MP4 renderer** — the cut lives in your NLE, this skill ships the XML and the captions sidecar only.

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

Cache lives at `~/.premiere-agent/health.json` — **outside** the per-session `<videos_dir>/edit/` so it persists across projects. This is the one exception to Hard Rule 7, and it's intentional: skill-environment health is a per-machine property, not a per-session one.

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

- **`helpers/find_quote.py --edit-dir <edit> --clip <stem> --range <M:SS-M:SS> --quote "..."`** — word-precise quote / range crawler over `transcripts/<stem>.json`. **The supported interface for every EDL anchor (Hard Rule 2).** Returns first/last/prev/next word boundaries pinned to the millisecond plus `lead_silence_s` / `trail_silence_s` / `cut_window` ready for the pacing-preset margin clamp. Stdlib-only, no model loads, sub-millisecond hot. With `--clip` omitted, picks the most-recently-modified transcript (the colloquial "last clip we talked about" shortcut). **Batch mode (`--batch <path|->`)** answers N queries in one process — pass a JSON `{"queries":[{clip, quote, range|start|end, max_matches, id}, ...]}` doc as a file path or via stdin and get a `{batch_count, clips_loaded, results:[...]}` envelope back; queries are grouped by clip internally so each transcript loads exactly once, per-row failures come back as `{error, detail, query}` rows instead of aborting the batch, and every result carries `query.index` + the optional `query.id` you passed in for correlation. Use it whenever you'd otherwise spin up the helper more than ~3 times in a row (boundary self-eval, verifying every range in a freshly drafted EDL, etc.) — one Python startup beats forty. Never grep / hand-parse `transcripts/<stem>.json` directly — the helper is the only sanctioned path.
- **`helpers/timeline_view.py <video> <start> <end>`** — filmstrip + waveform PNG. On-demand visual drill-down. **Not a scan tool** — use it at decision points, not constantly. The visual_timeline.md replaces 90% of the old "scan with timeline_view" workflow.
- **`helpers/build_srt.py <edl.json> -o master.srt`** — standalone captions sidecar generator. Reads each EDL range's `transcripts/<stem>.json`, applies the output-timeline offset math from Hard Rule 1, and emits a single `master.srt` aligned to the cut. Called automatically by `export_fcpxml.py` so the SRT lands next to the XML; invoke directly only when you need to regenerate captions without re-exporting the timeline.
- **`helpers/export_fcpxml.py <edl.json> -o cut.fcpxml`** — emit editor-ready timeline files. **The single delivery path.** Hard-cut only right now (Hard Rule 9): the EDL's `audio_lead` / `video_tail` / `transition_in` fields are still consumed by the code path but every range must emit `0` for all three. **Default emits BOTH `cut.fcpxml` AND `cut.xml`** side-by-side from a single timeline build, because Premiere Pro and Resolve/FCP X want different XML dialects: `.fcpxml` (FCPXML 1.10+) is native to DaVinci Resolve and Final Cut Pro X, `.xml` (Final Cut Pro 7 xmeml) is native to Premiere Pro. The recipient picks whichever NLE they live in — no XtoCC conversion required for Premiere. Also calls `build_srt.py` to drop `master.srt` next to the XML for caption import. Override with `--targets {both,fcpxml,premiere}`. `--frame-rate 24` (default), 25, 29.97, 30, 60.

## The process

0. **Health check.** Run `python helpers/health.py --json` first. Cached for 7 days; usually returns instantly. If `status != "ok"`, surface the `advice` strings to the user verbatim and stop. See the "Skill health check" section above.
1. **Inventory + Phase A preprocess.** `ffprobe` every source. While walking, note folder names + root docs as semantic tags: `b_roll/` → load `references/b_roll_selection.md`; `script.*` / `narration.*` / `voiceover/` folder / standalone `vo*.wav` → load `references/scripted.md`. Folder names ARE the categories (`a_roll/` = speech bed, `timelapse/` = retime source) — keep in working memory, no on-disk file. Then `python helpers/preprocess_batch.py <videos_dir>` runs the speech + visual lanes (Parakeet ONNX + Florence-2), cached by mtime; `python helpers/pack_timelines.py --edit-dir <edit>` produces `merged_timeline.md` plus per-lane drill-downs `speech_timeline.md` and `visual_timeline.md`.
2. **Phase B audio (agent-curated CLAP).** Read `merged_timeline.md` yourself (or `speech_timeline.md` + `visual_timeline.md` if you want the per-lane view), infer what kinds of sounds will plausibly appear in this footage (tools, materials, ambience, music, animals, vehicles, environments — be specific to *this* project), and write a vocabulary list of ~200–1000 short labels to `<edit>/audio_vocab.txt`. Include a healthy negative / unrelated set too so silence and out-of-domain sounds don't all latch onto your top labels. Then run `python helpers/audio_lane.py <videos> --vocab <edit>/audio_vocab.txt --edit-dir <edit>` and re-run `pack_timelines.py` to fold the new events into `merged_timeline.md` and `audio_timeline.md`. Skip this step only if the user explicitly says they don't care about audio events, or pass `--include-audio` to `preprocess_batch.py` upstream to use the baked-in baseline vocab instead (smoke tests, agent-less batch runs).
3. **Pre-scan for problems.** One pass over `merged_timeline.md` end-to-end — every speech phrase, every audio event, every visual caption, all interleaved by timestamp. Note verbal slips, mis-speaks, or phrasings to avoid (from the `"..."` lines). Note shot variety, B-roll candidates, and visually continuous actions you'll want to keep whole (from the `visual:` lines). Treat `(audio: ...)` lines as the lowest-priority hints — verify any CLAP label against the visual line at the same timestamp before trusting it (the model is approximate, especially when the vocabulary is too small or too generic). Drill into the per-lane files only when the merged view leaves you guessing about word-level timing or denser audio scoring than the merged stream shows.
4. **Converse.** Describe what you see in plain English. Ask questions *shaped by the material*. Collect: content type, target length/aspect, aesthetic/brand direction, must-preserve moments, must-cut moments, subtitle needs (style, on/off, language). Do not use a fixed checklist — the right questions are different every time. **One question is mandatory and not skippable: pacing preset** — present the five options (Calm / Measured / Paced / Energetic / Jumpy) with one-line descriptions and tell the user the default is **Paced**. They can pick a name or just say "use the default." See "Pacing presets" below for the value table you'll apply when picking cut edges. Color is the colorist's job inside the NLE; do not ask grade preferences. There is only one delivery format — FCPXML + xmeml + master.srt sidecar — so do not ask about delivery either.
5. **Propose strategy.** 4–8 sentences: shape, take choices, cut direction, **chosen pacing preset (by name + the four numbers it expands to)**, subtitle style, length estimate. **Wait for confirmation.**
6. **Execute.** Produce `edl.json` directly. Drill into `timeline_view` at ambiguous moments where the `visual_timeline` caption alone isn't enough; verify every word-boundary anchor with `helpers/find_quote.py` (Hard Rule 2). Then run `python helpers/export_fcpxml.py <edit>/edl.json -o <edit>/cut.fcpxml` — that emits `cut.fcpxml` (Resolve / FCP X), `cut.xml` (Premiere Pro xmeml), and `master.srt` (output-timeline captions) all in one shot. Tell Premiere users to `File → Import → cut.xml` (the `.fcpxml` does **not** work natively in Premiere — that's the file Adobe wants you to run through XtoCC, which we sidestep entirely). Tell Resolve / FCP X users to import `cut.fcpxml`. The SRT goes in as a captions track if they want burned-in subtitles — both NLEs accept it natively.
7. **Preview is in the NLE.** Hand `cut.fcpxml` / `cut.xml` to the user to open and scrub in their NLE of choice — the `.xml` for Premiere, the `.fcpxml` for Resolve / FCP X. There is no flat-MP4 preview from this skill; the cut lives in the NLE from now on.
8. **Self-eval (before handing off the XML).** Sample-check the EDL itself, since there is no rendered file to scrub. For each cut boundary in `edl.json`:
    - Re-run `find_quote.py` on the in-point and out-point clips and confirm `first_word.start` / `last_word.end` match the EDL's `in` / `out` to the millisecond after the pacing-preset margin clamp (a mid-word cut here is a Hard Rule 2 violation). **Use `--batch` here** — build one `{queries:[...]}` doc with two rows per EDL range (an in-anchor and an out-anchor, each carrying the range's `id` so you can correlate failures back to the EDL row) and submit it once. A 40-cut EDL is one tool call, not eighty.
    - Spot-check `merged_timeline.md` around the boundary timestamps — does the speech context flow, does the visual lane confirm shot continuity (or an intentional shot change)?
    - Run `ffprobe` on each source referenced by the EDL to confirm `out` is within the source's actual duration (no off-the-end ranges that crash NLE import).
    - Open `master.srt` and skim the first 5 / last 5 cues — confirm they line up with the cut's first and last spoken words and that the timestamps are monotonically increasing across the whole file (Hard Rule 1 sanity check).

   If anything fails: fix the EDL, re-export, re-eval. **Cap at 3 self-eval passes** — if issues remain after 3, flag them to the user rather than looping forever. Only hand off the XML once the self-eval passes.
9. **Iterate + persist.** Natural-language feedback, re-plan, re-export the XML. Never re-preprocess unchanged sources. Append the session's decisions to `project.md`.

## Pacing presets

Every session must have a pacing preset (Hard Rule 8). Ask the user up-front in step 4. Default is **Paced**. The preset expands to four numbers you apply when picking cut points and trimming silences:

| Preset    | `min_silence_to_remove` | `min_talk_to_keep` | `lead_margin` | `trail_margin` | Vibe |
|-----------|------------------------:|-------------------:|--------------:|---------------:|------|
| Calm      | 500 ms                  | 500 ms             | 500 ms        | 500 ms         | Cinematic, contemplative, breathing room. Long silences are kept; only obvious dead air is trimmed. Documentary, interview, narrative. |
| Measured  | 350 ms                  | 350 ms             | 350 ms        | 350 ms         | Conversational and considered. Professional talking-head, podcast-style, unhurried tutorial. |
| **Paced** *(default)* | **200 ms** | **200 ms**     | **200 ms**    | **200 ms**     | Balanced and modern. Retains rhythm without dragging. Default for tech demos, launch videos, mid-form content. |
| Energetic | 100 ms                  | 100 ms             | 100 ms        | 100 ms         | Tight and punchy. Social-friendly, fast tutorials, hype reels. |
| Jumpy     |  50 ms                  |  50 ms             |  50 ms        |  50 ms         | Ultra-tight "every breath cut" style. Montage, trailer, vlog supercuts. Risks audible artifacts on poor source audio — verify on preview. |

**What each number means** (apply at decision time when picking EDL ranges):

- **`min_silence_to_remove`** — silences (gaps between words from the speech lane) shorter than this are *kept*; longer ones are candidates to cut out entirely. A Jumpy preset chops out anything ≥50ms; a Calm preset only chops ≥500ms gaps so natural pauses survive.
- **`min_talk_to_keep`** — speech segments shorter than this are *not* worth retaining as standalone clips. Used to filter out single-syllable false starts ("uh-", "th-") that survived the silence filter. Tighter presets keep shorter fragments because the editing rhythm absorbs them.
- **`lead_margin`** — silent padding *before* the first kept word of a clip. Absorbs ASR drift (50–100ms typical) and gives the listener a beat of air before the talker comes in.
- **`trail_margin`** — silent padding *after* the last kept word of a clip. Same purpose at the tail. Together with `lead_margin`, replaces the old "30–200ms working window" guidance with a preset-driven number — but Hard Rule 3's working window still bounds the legal range (so a Calm preset's 500ms margin is the upper end, not infinity).

**Translating to the EDL:** when you build each range, expand the kept word boundary by the margins:

```
range.start = max(0, kept_first_word.start - lead_margin / 1000)
range.end   = min(src_duration, kept_last_word.end + trail_margin / 1000)
```

And while picking ranges, only consider silence gaps `>= min_silence_to_remove` as legitimate cut targets, and discard any candidate kept clip whose net speech duration is `< min_talk_to_keep`.

**Aggressive intra-phrase silence removal (this is the whole point of the preset).** `min_silence_to_remove` applies to **every word-to-word gap in the speech lane**, not just to phrase boundaries or speaker handoffs. If a single phrase like *"today we're going to ⟨640ms gap⟩ drill the pilot holes"* contains a gap ≥ the threshold, you MUST split that phrase into two adjacent ranges from the same source — `[..., "going", "to"]` then `[..., "drill", "the", "pilot", "holes"]` — so the dead air is dropped from the timeline. This is how a Paced preset turns a 12-minute walking-talking-head into 7 minutes without losing a word: by deleting hundreds of small breath gaps, hesitations, and thinking pauses scattered inside otherwise-kept speech. Do not romanticize the "natural rhythm of how someone talks" — the preset *is* the rhythm decision. If the user wants those pauses kept, they pick Calm.

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

- **Speech-first.** Candidate cuts from word boundaries and silence gaps. Parakeet TDT is accurate to the word; the speech lane is the editorial spine. Read it interleaved in `merged_timeline.md`; drill into `speech_timeline.md` when you need word-level timing detail. **Verify every cut anchor with `helpers/find_quote.py` before writing the EDL** (Hard Rule 2 + the worked example below).
- **Preserve peaks.** Laughs, punchlines, emphasis beats. Extend past punchlines to include reactions — the laugh IS the beat.
- **Speaker handoffs** benefit from air between utterances. The pacing preset's `lead_margin` + `trail_margin` largely sets this; only override per-handoff if the moment calls for it (e.g. a punchline beat that earns extra silence).
- **Visual context is the second source of truth.** Before committing to *any* non-trivial cut, check the `visual:` lines around the cut point in `merged_timeline.md`. If captions show a continuous action ("person holding drill") spanning your cut, you're cutting in the middle of a shot — usually fine, but be deliberate. Use the visual lane to find B-roll cutaway candidates, match cuts, shot changes, and to decide whether a moment is worth preserving even when speech is silent. Drill into `visual_timeline.md` when you need the full 1-fps caption stream (the merged view drops `(same)` repeats).
- **Audio events are noisy hints, not signals.** The `(audio: ...)` lines in `merged_timeline.md` carry `(drill 0.87)`, `(applause 0.92)`, `(laughter)`, `(power_tool)` markers from CLAP scored against the agent-curated vocab. **The model is approximate** — it mis-labels (music tagged as speech, hammers tagged as drums, room tone tagged as applause), especially when the vocabulary is too small or too generic. Use a marker only as a prompt to *go look* at the visual line (and if needed `timeline_view`) at that timestamp. **Never cut purely on a CLAP label.** When CLAP and Florence-2 disagree about what's happening, trust Florence-2. Drill into `audio_timeline.md` when you want the full per-window scoring instead of the collapsed merged form.
- **Silence gaps are cut candidates — EVERYWHERE, not just at phrase boundaries.** Use the pacing preset's `min_silence_to_remove` as your threshold (Calm 500ms → Jumpy 50ms) and apply it to every adjacent word pair in the speech lane, including gaps that sit *inside* a phrase as the speaker pauses to breathe or think. Splitting a phrase mid-sentence to drop a 400ms thinking pause is the whole point of the preset; it's how you cut runtime without cutting content. The user picked Energetic because they want every breath gone — give them every breath. (Anything shorter than the preset threshold stays as the natural rhythm of the speech. <30ms is always unsafe — mid-phoneme.)
- **Cut out filler words and disfluencies by default.** "uh", "um", "umm", "uhh", "er", "erm", "ah", "ahh", "hmm", "mm", "like" (when used as a verbal tic, not as a verb / preposition / simile), "you know" (filler usage), "I mean" (false-start usage), "so yeah", "kinda", "sorta" (filler usage), single-syllable false starts ("th-", "wh-", "the the", "we we", "I I"), repeated stutter words (the speaker says the same word twice while collecting their thought), and trailing "..."s where the speaker abandons a sentence and restarts. **Treat each as a cut candidate equivalent to a silence gap** — split the EDL range around the filler so the kept words concatenate cleanly. The Parakeet lane preserves them verbatim precisely so you can find them and remove them; do not leave them in out of "respect for the speaker's natural voice." A clean tight delivery is the speaker's voice with the friction removed. **Exceptions** (keep the filler): (a) the filler IS the punchline / the joke / the emotional beat ("…uhhhh, that's not what I expected"), (b) removing it would break a load-bearing rhythm the user explicitly asked for, (c) the surrounding take is so much worse that the filler-version is genuinely the best option — note it in `reason` when this happens. When you cut a filler, the resulting two adjacent EDL ranges from the same source must each still satisfy word-boundary alignment (Hard Rule 6) and pacing-preset margin clamping (so the lead/trail pads don't re-introduce the filler you just removed; the same `combined_pad_ms <= gap_ms - 60` rule from the silence-removal pass applies). On rare doubled-word repeats where Parakeet emits zero gap between the two instances, snap the cut to the END of the first instance / the START of the second; do not cut mid-word.
- **Cut padding comes from the pacing preset**, not from per-cut taste. Expand each range by `lead_margin` at the head and `trail_margin` at the tail (see "Pacing presets"). Hard Rule 3's 30–200ms working window still bounds anything outside the preset table — never go below 30ms.
- **Never reason audio and video independently.** Every cut must work on both tracks.

### Word-boundary verification — `helpers/find_quote.py` (mandatory, every range)

Hard Rule 2 binds you to word-boundary cuts. The merged timeline is rounded to whole seconds for token economy — those rounded ranges are NOT cut anchors, they're scoping windows. Sub-second word boundaries live in `transcripts/<stem>.json`, and the only sanctioned way to reach them is `helpers/find_quote.py`.

**The workflow per range:**

1. **Identify the words you want to keep.** From the `"..."` phrase line in `merged_timeline.md`, pick the first and last word the kept range should contain — e.g. *"Geared to lock in the uplock hook"* with the user wanting to keep through *"lock in"* gives `first_word = "Geared"`, `last_word = "in"`.

2. **Call `helpers/find_quote.py`** with the clip stem, the integer `M:SS-M:SS` range read straight off the `"..."` line (it's outer-aligned — start floors, end ceils — so it's a guaranteed superset of the float span), and a quote substring covering the words you want:

   ```bash
   python helpers/find_quote.py --edit-dir <edit> --clip <stem> \
       --range 0:03-0:05 --quote "Geared to lock in"
   ```

   Returns (relevant fields):

   ```json
   {
     "first_word":     {"text": "Geared", "start": 3.12, "end": 3.45},
     "last_word":      {"text": "in",     "start": 3.80, "end": 3.95},
     "prev_word":      {"text": "and",    "start": 2.85, "end": 3.05},
     "next_word":      {"text": "the",    "start": 4.10, "end": 4.18},
     "lead_silence_s": 0.07,
     "trail_silence_s": 0.15,
     "cut_window":     {"safe_in_s": 3.05, "safe_out_s": 4.10}
   }
   ```

3. **Snap the cut anchors.**
   - In-point: `result.first_word.start` (e.g. `3.12`)
   - Out-point: `result.last_word.end` (e.g. `3.95`)
   - Never let the cut land between a word's `start` and `end`.

4. **Apply pacing-preset margins, clamped against the cut window.** The `cut_window.safe_in_s` / `safe_out_s` are the absolute outer bounds — outside them you're eating into the previous / next word. Leave at least 60ms on each side so the NLE has room to add its own audio crossfade on import:

   ```
   range.start = max(safe_in_s + 0.06, first_word.start - lead_margin/1000)
   range.end   = min(safe_out_s - 0.06, last_word.end  + trail_margin/1000)
   ```

5. **Emit the EDL range.** `range.start` and `range.end` are now word-boundary-pinned, margin-padded, and silence-clamped. No off-by-one. No mid-word cut. No re-introduction of a silence you wanted to drop.

**Reading `transcripts/<stem>.json` by hand or with `grep` is forbidden.** The helper is bounds-checked, sub-millisecond hot, and returns a self-contained match envelope the EDL generator consumes verbatim. Direct JSON access is reserved for cases the helper genuinely cannot answer (e.g. inspecting a stem's diarization metadata).

**Batching when you have many ranges to verify.** When the EDL has more than two or three ranges, do not call `find_quote.py` once per range — submit them all in one batch. The helper accepts a JSON document of N queries via `--batch <path|->` and answers them in a single process:

```bash
python helpers/find_quote.py --edit-dir <edit> --batch verify.json --compact
```

```json
{
  "queries": [
    {"id": "r0_in",  "clip": "DJI_..._0317_D", "range": "0:02-0:08", "quote": "Geared to lock"},
    {"id": "r0_out", "clip": "DJI_..._0317_D", "range": "0:14-0:20", "quote": "uplock hook"},
    {"id": "r1_in",  "clip": "DJI_..._0320_D", "range": "0:00-0:06", "quote": "Beautiful day"},
    {"id": "r1_out", "clip": "DJI_..._0320_D", "range": "0:50-0:58", "quote": "permanently"}
  ]
}
```

The result envelope is `{batch_count, clips_loaded, results:[...]}` with one row per input query, each carrying back the `query.id` you passed in plus a `query.index` matching the input order. Per-row failures (`transcript_not_found`, `bad_range`, `empty_quote`, `no_query`) come back as `{error, detail, query}` rows — the batch never aborts mid-stream. Queries are grouped by clip internally so each transcript loads exactly once: a 40-row batch across 12 clips is 12 disk reads. **This is the right shape for the Hard Rule 8 self-eval pass — one tool call, every boundary verified.** Pass `-` instead of a file path to stream the JSON in over stdin.

### Split edits (DEFERRED — do not emit)

J-cuts (`audio_lead`), L-cuts (`video_tail`), and cross-dissolves (`transition_in`) are **deferred** until further notice (Hard Rule 9). The EDL schema still accepts these fields and the FCPXML exporter still consumes them — but you must emit `0` for all three on every range, and any EDL that doesn't is broken.

**Why deferred:** the current FCPXML build uses an OTIO single-track audio model with per-clip independent frame-snapping. When `audio_lead` or `video_tail` is non-zero, the math `cur_a = target_a_start + a_dur` (in `helpers/export_fcpxml.py`) drifts away from `cur_v` because:

1. Snapping `a_src_start` and `a_src_end` independently to the frame grid doesn't always preserve `a_dur == v_dur` (sub-frame rounding error per clip).
2. A single audio track can't actually overlap clips, so an L-cut tail forces the *next* clip's audio backward — but the gap-padding only handles the positive case (audio LATER than video). Negative gaps silently collapse to zero, so the next J-cut's audio starts at the wrong timeline position.
3. Errors compound across every cut. On a 50-cut timeline the audio is visibly sliding out of alignment under the video by the end (the symptom the user reported).

**Path forward when we revisit this** (not now): switch to two audio tracks (A1 carries the speech, A2 carries the lead/tail spillover) so overlaps are legal; lock the audio source range to the same snapped frame edges as the video; advance both `cur_v` and `cur_a` from a single canonical "next timeline position" rather than independent counters.

**For now, the only legal "audio crossfade" is whatever the receiving NLE applies on import** — Premiere and Resolve both add a short default crossfade at every audio cut boundary, which suppresses pops without needing a J/L cut. Good enough until the multi-track FCPXML path lands.

**EDL fields the editor MUST emit as zero:**

```json
{"source": "C0103", "start": 12.20, "end": 18.45, "beat": "ANSWER",
 "audio_lead": 0.0,        // DEFERRED — must be 0
 "video_tail": 0.0,        // DEFERRED — must be 0
 "transition_in": 0.0}     // DEFERRED — must be 0
```

**Render path matrix (current state):**

| Output                       | Hard cuts | Audio crossfade   | J/L cuts | Dissolves | Time-squeeze (≤10x) |
|------------------------------|-----------|-------------------|----------|-----------|---------------------|
| `export_fcpxml.py` → fcpxml + xml + master.srt | ✓         | NLE's job on import | DEFERRED (Hard Rule 9) | DEFERRED (Hard Rule 9) | ✓ (`<timeMap>` + Premiere `timeremap`) |

If the user explicitly asks for J/L cuts or dissolves: explain the deferral honestly, ship hard cuts, and offer to log it in `project.md` as an outstanding item for the day the multi-track path lands.

### Time-squeezing (timelapse) — when the source has long no-speech stretches

Real-world footage is often "1 minute of explanation, then 25 minutes of silently doing the work, then 2 minutes of wrap-up." Cutting the 25 minutes entirely throws away the visual story; keeping it 1× bores the viewer. The third option is **time-squeezing**: compress the work segment into a 5–30s timelapse on the output timeline so the viewer sees the whole arc in seconds. `export_fcpxml.py` writes the retime element into both the FCPXML `<timeMap>` and the Premiere xmeml `timeremap` block, so the squeeze travels into either NLE intact.

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

- `audio_strategy = "drop"` (default at `speed != 1.0`): the audio track is silenced over the squeezed range — the exporter emits a silent gap. This is the right answer for ~95% of timelapses; sustained shop noise / room tone sped up 5–10× sounds awful, and the editor will drop a music bed under the squeeze in the NLE.
- `audio_strategy = "keep"`: the audio gets the same retime element as the video. It will be chipmunk-y unless the editor toggles "Maintain Audio Pitch" (Premiere) / "Preserve Pitch" (FCP X) on the clip — both NLEs offer this as a one-click clip property. Use this only when there's a specific reason to keep the source audio (recognisable voice in the background, distinctive ambient texture).

**Editorial discipline for time-squeezing** (not in the numbered Hard Rules block — these are taste calls, not silent-failure issues):

- **Decide per-stretch: is the speech worth keeping?** Load-bearing speech earns a 1× split around it; filler speech gets squeezed over with `audio_strategy="drop"`. See the "judgement call, not a blocker" paragraph above. There is no universal rule — read the words and ask whether the viewer benefits from hearing them.
- **Cut FIRST, squeeze SECOND.** Apply the pacing preset's silence-removal pass first (drop dead air ≥ `min_silence_to_remove`); then identify the surviving long stretches that fit the "coherent activity + story-of-progress" criteria; then squeeze those ranges. Squeezing dead air is just slower nothing.
- **Word-boundary discipline still applies on adjacent 1× ranges** (Hard Rule 2 / 3). The squeezed range itself doesn't need word-boundary alignment when `audio_strategy="drop"` (the audio's gone anyway), but pad it generously (~1–2s on each side of the activity) so the viewer's eye registers the speed change cleanly.
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

**`merged_timeline.md`** — the **default reading surface.** All three lanes interleaved chronologically by timestamp into a single per-source section. Speech phrases as `"..."`, audio events as `(audio: label1, label2, ...)`, visual captions as `visual: ...`. One file, one full read, every event in order — you get the same triangulated picture you would get from reading three lanes in parallel, without the three-way cross-reference cost.

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

## Multi-take cuts — picking the best take of each beat

When the source pile contains multiple takes of the same beats (the typical "we shot it five times, pick the best one" job), assemble the EDL **chronologically by BEAT, not by source clip order**. Walk every take of each beat across all the merged-timeline sections, pick the cleanest delivery, and concatenate the chosen ranges in narrative order — even when that means jumping from `C0108` back to `C0103` and forward to `C0210`. The Hard Rules + Cut craft + Pacing presets sections above are everything you need; the rest of this section is a pair of editorial conveniences that don't fit anywhere else.

### Common structural archetypes (pick, adapt, or invent)

A skeleton to test "what beats do I actually need" against. Don't force the material into a template — invent your own when the footage calls for it.

- **Tech launch / demo**: HOOK → PROBLEM → SOLUTION → BENEFIT → EXAMPLE → CTA
- **Tutorial**: INTRO → SETUP → STEPS → GOTCHAS → RECAP
- **Interview**: (QUESTION → ANSWER → FOLLOWUP) repeat
- **Workshop / build**: INTRO → MATERIALS → STEPS (with audio-event beats) → REVEAL
- **Travel / event**: ARRIVAL → HIGHLIGHTS → QUIET MOMENTS → DEPARTURE
- **Documentary**: THESIS → EVIDENCE → COUNTERPOINT → CONCLUSION
- **Music / performance**: INTRO → VERSE → CHORUS → BRIDGE → OUTRO

### Picking takes in practice

Walk the merged timeline section by section, identify every take of each beat, pick the cleanest delivery. For each chosen range, call `helpers/find_quote.py` (Hard Rule 2) — it returns word-precise `first_word.start` / `last_word.end` plus `lead_silence_s` / `trail_silence_s` / `cut_window` so you have everything you need to apply the pacing preset's margins and confirm the cut isn't eating into the previous / next word. The silence-removal pass from "Pacing presets" then tells you whether to split the chosen range further around any in-phrase silence gaps the user's preset wants gone. No separate pre-pass is needed; the merged timeline + `find_quote.py` cover it.

### Retake detection — drop the flubbed take, keep the cleaner one

Real recordings contain multiple takes of the same line — the speaker flubs, swears, restarts, or naturally re-says something a beat later because they didn't like how it landed. Retakes also straddle clip boundaries. **The user recorded the better take so you'd use it** — detect the repetition, pick the cleaner take, drop the rest. Distinct from filler removal (per-word *"uh"* / *"um"*) and from in-clip editor notes (explicit verbal directives); retakes are an *implicit pattern*: the same words, twice, the later one usually better.

**Signals** (walk the speech lane in `merged_timeline.md`; drill into `speech_timeline.md` for word timing): a **frustration marker** within ~10s of similar content — curse (*"fuck"*, *"shit"*, *"damn"*), self-disgust noise (*"ugh"*, *"argh"*), or self-correction (*"no no no"*, *"hold on"*, *"sorry"*, *"let me try that again"*, *"one more time"*, *"take two"*, *"again"*) — is the strongest single cue, look both ways for the matching pair. Also: **semantic repetition** within ~30s (≥50% content-word overlap, no intervening topic shift); **cross-clip retakes** when adjacent sources start with similar content (filenames like `intro_take1.MP4` / `intro_take2.MP4` or numerically adjacent stems make the second the keeper for any overlap); a **slate / clap** between two utterances of the same content (`(audio: clap …)` / `(audio: slate …)`); a **long pause ≥2s followed by restart** of what came before.

**Pick the keeper — prefer the LATER take by default.** The later take exists because the speaker decided the earlier wasn't good enough; respect that. Override when (a) an in-clip note says *"use the first take"* (notes win), (b) the later take is *worse* on objective signals — more fillers / silences / false starts / dropped energy — note the override in `reason`, (c) the takes diverge in *meaning* (one has a punchline the other lacks; one introduces a named subject the script needs), or (d) the keeper fails the structural test (cuts off mid-thought, speaker walks out of frame, visual continuity breaks).

**When repetition is INTENTIONAL — keep both.** Rhetorical / emphatic (*"Buy now. Buy now. Buy NOW."*), comedic callback, list ladder (*"fast, faster, fastest"*), speaker quoting another speaker. Disambiguator: frustration marker / long pause / slate / explicit *"again"* / *"take two"* ⇒ retake; identical wording with no pause, escalating pitch, or a joke-beat tone shift ⇒ intentional. **When ambiguous, keep both** — losing emphasis is a more visible bug than keeping one redundant sentence.

**Mechanics.** Standard inline-cut rules: word-boundary alignment (Hard Rule 2), the frustration marker / curse / *"let me try that again"* **all excluded** from the EDL (connective tissue between takes — nobody wants them in the cut), and the same `combined_pad_ms <= gap_ms - 60` clamp from the silence-removal pass governs the gap between kept earlier audio and kept later audio. Cross-clip retakes emit two adjacent ranges from different sources — the FCPXML exporter handles same-track concatenation natively. Cite the rejection in `reason` (*"Second take of intro; first take C0312 4.1–12.0s rejected — speaker said 'fuck, again' at 11.4s and restarted cleaner."*) and surface every retake call in a dedicated `Retake decisions` block in your strategy / handoff so the user can audit what got dropped.

**Conservative.** Never cut around a frustration marker without confirming a matching restart within 10s (a standalone *"fuck"* may be the speaker's reaction to something on camera — content, not retake noise; that's why filler-word rules don't list curses as default-cut). Cross-clip decisions require temporal evidence — similar speech across two clips isn't proof; require stem ordering, an in-clip note, or a slate / clap event before dropping a whole earlier source.

## In-clip editor notes — AI guidance baked into the source ("hey editor")

Users sometimes record verbal directives **into the clip itself**, addressed to a downstream editor (you) before, between, or after takes. These are first-class user instructions — the user spoke them into the recording precisely so a downstream editor would find and honour them. **Detect, honour, exclude the preamble from the EDL, and surface every one in your handoff** in plain English so the user can audit what you applied.

**Common shapes:**

- **Preamble before a take.** *"hey to the AI editing this, skip the first attempt — the second one's the keeper. Three, two, one…"* then the take begins.
- **Mid-clip note between takes.** *"…ugh, that was awful. Editor, just use the next one. Okay, take two — three, two, one…"*
- **End-of-clip note.** *"…and that's that. Note for the editor: if my hands shake on the close-up, cut to the wide."*
- **Pickup directive.** *"editor's note: skip ahead until I clap"* followed by a clap (audio + visual confirms).

**Trigger-phrase detection.** Walk `merged_timeline.md` (drill into `speech_timeline.md` via `helpers/find_quote.py` for sub-second word timing on a matched phrase) for any phrase the speaker uses to **address an editor or AI**. Match liberally — case-insensitive, tolerant of mis-transcription — but require **imperative or instructional content after the address** before treating it as a note. Common openings (non-exhaustive — match the *intent*): *"hey editor"*, *"hey, to the editor"*, *"hey AI"*, *"hey Claude"*, *"note to the editor"*, *"editor's note"*, *"for whoever's editing this"*, *"AI listen up"*. Parakeet may mis-hear (*"note the editor"* vs *"note to the editor"*) — read for intent: if the speaker is clearly addressing a downstream editor / AI vs the on-camera audience, the following content is a note candidate.

**Boundary detection — where does the directive end?** From the trigger to **the first of**: (1) a take-start countdown (*"three two one"*, *"and… action"*, *"okay rolling"*, *"take two"* — countdown / call itself ALSO excluded), (2) a clap or slate (`(audio: clap …)` / `(audio: slate …)` within ~3s, visual confirmation strengthens), (3) a long silence gap ≥1.5s, or (4) a register shift to addressing the audience (*"Hey everyone, today we're going to…"*). If none lands within ~10s, the trigger was probably rhetorical — let the words ride as content.

**Exclusion from the EDL.** Everything from the trigger through the take-start marker (inclusive of countdown / *"action"* / clap window) is excluded. Place the in-point at or after the take-start marker, snapped to a word boundary per Hard Rule 2. The pacing preset's `lead_margin` still applies — but **never let it pull the in-point back INTO the preamble.** Clamp.

**Application priority** (highest first): (1) things the user explicitly asked to keep / reject in the live conversation always win — those are post-hoc and may explicitly reverse an in-clip note; (2) in-clip editor notes; (3) default editorial rules. If an in-clip note conflicts with a live-conversation request, the live request wins; note the override in your rationale.

**Common directive shapes:** *"skip the first take, use the second"* → exclude first-take ranges, prefer second. *"this take is bad"* / *"don't use this one"* → exclude that take entirely. *"cut after I say <X>"* / *"end on <X>"* → out-point on that word. *"start when I clap"* → in-point at the clap event / visual hand-clap frame. *"the wide shot is better than the close-up here"* → bias toward the wide source for that beat. *"cut around me coughing at minute four"* → split the EDL range to drop the cough span. *"speed this up"* → only honour when timelapsing fits the material (else note the deferral). When an in-clip note shaped a range, cite it in `reason` with source stem and timestamp, quoting verbatim (*"Second take per in-clip note (C0312 t=0.4s): 'skip the first take, the second one is the keeper.' In-point at first word after countdown."*).

**Conservative — when in doubt, surface, don't act.** Ambiguous directives (*"cut around the embarrassing bit"* — what counts as embarrassing?), unverifiable claims (*"the audio is bad on this one"* — you can't measure SNR from the timeline), or two contradictory notes across takes — **don't silently guess.** For the contradiction case, pick the **later** note (the user updated their preference) and flag the conflict; for the ambiguous / unverifiable case, preserve the take and flag in rationale. Surfacing > silent guessing — the user will clarify in the next turn. **When the user wants the feature off** (*"ignore any 'hey editor' notes — my brother yells that as a joke"*), respect the override: treat all in-clip notes as normal content this session and note the override at the top of your rationale.

## Subtitles — load on demand

The default delivery already drops `master.srt` next to the XML (built by `helpers/build_srt.py`, called automatically by `export_fcpxml.py`) — output-timeline timestamps, one cue per spoken phrase, ready for the NLE's caption import. That covers most sessions.

When the user asks for *styled* subtitles (bold-overlay burn, natural-sentence chunking, custom `force_style` strings, language-specific placement reasoning), read **`references/subtitles.md`** in full before proposing the style. It's cold-path — ~10% of sessions touch it — so it stays out of the default context to save tokens. The output-timeline offset math from Hard Rule 1 binds whether or not you read the reference.

Color grading is **not** part of this skill — that's the colorist's job inside the NLE, on the imported XML. Don't ask the user about grade preferences and don't try to bake one in.

## Output spec

Match the source unless the user asked for something specific. Common targets: `1920×1080@24` cinematic, `1920×1080@30` screen content, `1080×1920@30` vertical social, `3840×2160@24` 4K cinema, `1080×1080@30` square. Pass `--frame-rate` to `export_fcpxml.py` matching the source (or the user's intended deliverable) so cuts snap to whole frames in the NLE. Resolution is set inside the NLE on the imported sequence — this skill only emits the cut decisions and the captions sidecar.

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
  "total_duration_s": 87.4
}
```

`pacing_preset` + `pacing` record the user's chosen preset and its expanded values for traceability (you already applied them when picking ranges; downstream tools re-read them for reporting). `audio_lead` / `video_tail` / `transition_in` per range are DEFERRED (Hard Rule 9) and must always be `0.0`. `speed` and `audio_strategy` are OPTIONAL and only appear on time-squeezed ranges (see "Time-squeezing").

Color grade, overlays, and the subtitles file path are NOT part of the EDL — color is the colorist's job inside the NLE, there is no overlay compositor in this skill, and `master.srt` is generated automatically by `export_fcpxml.py` and dropped next to the XML without needing an EDL field.

## Memory — `project.md`

Append one section per session at `<edit>/project.md`:

```markdown
## Session N — YYYY-MM-DD

**Strategy:** one paragraph describing the approach
**Pacing:** preset name + the four expanded ms values (so next session can default to it)
**Decisions:** take choices, cuts, time-squeezes, subtitle style + why
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
- **Reading `transcripts/*.json` directly for general scanning.** Use `merged_timeline.md` (or `speech_timeline.md` for a speech-only drill-down). Same data, 1/10 the tokens, phrase-aligned.
- **Grepping / hand-parsing `transcripts/<stem>.json` for word-precise cut anchors.** Use `helpers/find_quote.py` — same data, sub-second word boundaries pinned, off-by-one errors impossible, sub-millisecond hot. The helper is the only sanctioned interface for word-level lookup (Hard Rule 2). Direct JSON access is reserved for cases the helper genuinely cannot answer (e.g. diarization metadata).
- **Reading the three per-lane timelines separately when `merged_timeline.md` exists.** The merged view is the default reading surface — one file, all three lanes interleaved by timestamp. Open the per-lane files only as drill-down references for ambiguous moments (Hard Rule 10).
- **"Saving tokens" by partial-reading `merged_timeline.md`.** First-N-lines, last-N-lines, "representative sample," grep-and-edit-from-matches, abandoning a chunked read because "I have enough," delegating the full read to another agent to "protect context" — all forbidden (Hard Rule 11). The file is compressed and dedup'd at pack time so it fits; if it exceeds one `Read` call, issue sequential `Read` calls with `offset`/`limit` until every line is covered. YOU make the taste calls; outsourcing the read outsources the judgement.
- **Editing before confirming the strategy.** Never.
- **Re-preprocessing cached sources.** Immutable outputs of immutable inputs.
- **Assuming what kind of video it is.** Look first, ask second, edit last.
- **Skipping the pacing prompt or inventing ad-hoc cut-padding numbers.** Hard Rule 8 — every session uses one of the five presets; default is Paced.
- **Emitting non-zero `audio_lead` / `video_tail` / `transition_in`.** Hard Rule 9 — split edits and dissolves are deferred. The current FCPXML pipeline drifts the audio across long timelines under non-zero values; until the multi-track rebuild lands, hard cuts only.
- **Asking the user about color grade or render format.** Color is the colorist's job inside the NLE; there is no flat-MP4 renderer. The only delivery is `cut.fcpxml` + `cut.xml` + `master.srt`. Don't invent options that don't exist.
- **Building overlay animations or expecting an overlay compositor.** Removed from the skill. Overlays / motion graphics live in the NLE on top of the imported XML.
- **Squeezing pure dead air instead of cutting it.** A camera abandoned on a tripod with nothing moving is not a timelapse candidate — it's a CUT candidate. Time-squeezing is for visually continuous *activity* (assembly, walking, prep, teardown). Compressing 30 minutes of nothing into 3 seconds of nothing is just slower nothing.
- **Picking `speed` so the squeezed result lands < 5s or > 30s.** Under 5s the viewer doesn't register the activity; over 30s it overstays its welcome. Re-pick speed to land in the 5–30s sweet spot, OR split a long source stretch into multiple squeezes with a beat between, OR cut some of it.
- **Setting `speed > 10.0` and expecting it to apply.** The exporter clamps to 10.0 (1000%) with a warning. Beyond that retime decimates frames and looks broken; if you wanted >10x you should have cut.
- **Splitting around every word of filler speech inside an otherwise-squeezable stretch.** If the speech isn't load-bearing, squeeze right over it with `audio_strategy="drop"`. A hundred 1× micro-ranges interleaved with a hundred speed=8 micro-ranges is worse cut than one honest squeeze that drops the rambling. The video is for the viewer.
- **Leaving "uh" / "um" / "like" / "you know" / repeated-word stutters in the cut.** They are inline cut candidates by default — split the EDL range around each one so the kept words concatenate cleanly. The Parakeet lane preserves them verbatim precisely so you can find and remove them; do not preserve them out of "respect for the speaker's natural voice." A tight delivery IS the speaker's voice with the friction removed. Exceptions (filler as punchline, load-bearing rhythm the user asked for, every other take is worse) get kept with a one-line note in `reason`. See "Cut craft" for the full list and the cut-snap rules for zero-gap repeats.
- **Acting silently on an in-clip "hey editor" note, or including the preamble in the EDL.** Every in-clip directive you applied (or chose to skip) gets a line in your handoff rationale — silent application is the same class of error as silently inventing pacing. And the trigger phrase + content directive + countdown / clap window all sit on the cutting-room floor; EDL ranges start at or after the take-start marker.
- **Cutting around a frustration marker without confirming a matching restart within 10s.** A standalone *"fuck"* / *"ugh"* may be the speaker's reaction to something on camera (content), not retake noise. Confirm the matching repeat before dropping the earlier take, and surface every retake call in the `Retake decisions` block of your handoff so the user can audit what got dropped.
- **Treating rhetorical / emphatic repetition as a retake.** *"Buy now. Buy now. Buy NOW."* is the beat — the rhythm IS the content. Use the disambiguation cues (frustration marker / long pause / slate ⇒ retake; identical wording, no pause, escalating pitch ⇒ intentional). When ambiguous, keep both.
