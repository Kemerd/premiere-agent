# Subtitles (always on)

> Loaded on demand from `SKILL.md`. The SRT sidecar is **always emitted** by `helpers/export_fcpxml.py` — read this when the user asks about caption styling, chunking, or how the NLE picks it up. Skip the SRT only when subtitles are explicitly out of scope (silent cinema, lyric-only music video) by passing `--no-srt`.

Subtitles have three dimensions worth reasoning about:

- **Chunking** — 1 / 2 / 3 / sentence-per-line. Tight chunks for fast-paced social, longer chunks for narrative.
- **Case** — UPPERCASE / Title Case / Natural sentence case. Uppercase reads as urgency; sentence case reads as documentary.
- **Placement** — `MarginV` (margin from bottom). Higher `MarginV` lifts subtitles into the frame, away from device-UI safe areas.

The right combo depends on content. Pick deliberately, don't default.

## Worked styles — pick, adapt, or invent

### `bold-overlay` — short-form tech launch, fast-paced social

2-word chunks, UPPERCASE, break on punctuation, Helvetica 18 Bold, white-on-outline, `MarginV=35`. `helpers/build_srt.py` emits the SRT in this rhythm — apply the style in the NLE's caption panel.

```
FontName=Helvetica,FontSize=18,Bold=1,
PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H00000000,
BorderStyle=1,Outline=2,Shadow=0,
Alignment=2,MarginV=35
```

### `natural-sentence` — narrative, documentary, education

4–7 word chunks, sentence case, break on natural pauses, `MarginV=60–80`, larger font for readability, slightly wider max-width. No shipped force_style — design one if you need it. Suggested starting point:

```
FontName=Inter,FontSize=22,Bold=0,
PrimaryColour=&H00FFFFFF,OutlineColour=&HC0000000,BackColour=&H00000000,
BorderStyle=1,Outline=1,Shadow=0,
Alignment=2,MarginV=70
```

Invent a third style if neither fits.

## Hard rules

1. **Master SRT uses output-timeline offsets** (Hard Rule 5): `output_time = word.start - segment_start + segment_offset`. Otherwise captions drift across the cut timeline.
2. **Word-level boundaries from the speech lane.** Never invent chunk timing — the Parakeet word timestamps already give you exact in/out. Group N words into a line; the line's `start` is `words[0].start` and its `end` is `words[-1].end`.

## FCPXML / xmeml delivery

`helpers/export_fcpxml.py` automatically runs `build_master_srt` after the timeline XML is written, so every export drops `<edit>/master.srt` next to `cut.fcpxml` / `cut.xml` in one shot:

```bash
python helpers/export_fcpxml.py edit/edl.json -o edit/cut.fcpxml
# writes cut.fcpxml + cut.xml + master.srt
```

Need to regenerate just the SRT after a hand-tweak to the EDL without rebuilding the XML?

```bash
python helpers/build_srt.py edit/edl.json
```

The SRT is written **on the OUTPUT timeline** straight from the cached Parakeet transcripts in `<edit>/transcripts/`. Format is Premiere-friendly:

- UTF-8 (no BOM) — modern Premiere / Resolve / FCP X all parse this cleanly.
- CRLF line endings — Notepad-class editors render the file legibly on Windows; every NLE we target accepts \r\n.
- Sequential numbering from 1, exact `HH:MM:SS,mmm --> HH:MM:SS,mmm` timestamp shape, blank line between cues.
- Retimed (timelapse) ranges are SKIPPED — by editor convention they contain no speech (see SKILL.md "Time-squeezing"); the offset accumulator advances by the OUTPUT duration so subsequent cues still land correctly.

Importing into the NLE — every major NLE accepts SRT through plain `File → Import`:

| NLE | Path | What you get |
|---|---|---|
| Adobe Premiere Pro | `File → Import → master.srt` | A captions clip on the project bin; drag to a captions track. Restyle via the Captions panel. |
| DaVinci Resolve | `File → Import → Subtitle…` | A subtitle track on the timeline; restyle via the Captions inspector. |
| Final Cut Pro X | `File → Import → Captions…` | A connected captions clip on the primary storyline; restyle in the inspector. |

The skill never burns subtitles into a flat output — XML-only delivery means the NLE owns the final pixels and the editor controls caption style end-to-end.

## Decision shortcuts

| Content                          | Chunking      | Case      | MarginV |
|----------------------------------|---------------|-----------|---------|
| TikTok / Reels / Shorts          | 2 words       | UPPER     | 35–80   |
| Tech launch / explainer          | 3–4 words     | UPPER     | 40–60   |
| Tutorial / how-to                | sentence      | Sentence  | 60      |
| Documentary / interview          | sentence      | Sentence  | 70      |
| Music video / lyric              | line per beat | varies    | varies  |
