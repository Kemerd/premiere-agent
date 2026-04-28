# Subtitles (when requested)

> Loaded on demand from `SKILL.md`. Read this only when the user asks for burned or imported subtitles.

Subtitles have three dimensions worth reasoning about:

- **Chunking** — 1 / 2 / 3 / sentence-per-line. Tight chunks for fast-paced social, longer chunks for narrative.
- **Case** — UPPERCASE / Title Case / Natural sentence case. Uppercase reads as urgency; sentence case reads as documentary.
- **Placement** — `MarginV` (margin from bottom). Higher `MarginV` lifts subtitles into the frame, away from device-UI safe areas.

The right combo depends on content. Pick deliberately, don't default.

## Worked styles — pick, adapt, or invent

### `bold-overlay` — short-form tech launch, fast-paced social

2-word chunks, UPPERCASE, break on punctuation, Helvetica 18 Bold, white-on-outline, `MarginV=35`. `render.py` ships with this as `SUB_FORCE_STYLE`.

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

1. **Subtitles are applied LAST in the filter chain**, after every overlay (Hard Rule 1 in main SKILL.md). Otherwise overlays cover captions. Silent failure.
2. **Master SRT uses output-timeline offsets** (Hard Rule 5): `output_time = word.start - segment_start + segment_offset`. Otherwise captions drift after segment concat.
3. **Word-level boundaries from the speech lane.** Never invent chunk timing — the Parakeet word timestamps already give you exact in/out. Group N words into a line; the line's `start` is `words[0].start` and its `end` is `words[-1].end`.

## FCPXML / xmeml delivery

Ship `master.srt` alongside `cut.fcpxml` and `cut.xml`. Most NLEs (Premiere, Resolve, FCP X) import SRT as a captions track that the editor can restyle in their own caption panel. Don't burn subtitles into the segments for the NLE path — the editor will want to control style themselves.

For the flat MP4 path (`render.py`), subtitles are baked into the final pass via the `subtitles=…:force_style=…` filter, applied LAST after every overlay.

## Decision shortcuts

| Content                          | Chunking      | Case      | MarginV |
|----------------------------------|---------------|-----------|---------|
| TikTok / Reels / Shorts          | 2 words       | UPPER     | 35–80   |
| Tech launch / explainer          | 3–4 words     | UPPER     | 40–60   |
| Tutorial / how-to                | sentence      | Sentence  | 60      |
| Documentary / interview          | sentence      | Sentence  | 70      |
| Music video / lyric              | line per beat | varies    | varies  |
