# Subtitles (when requested)

> Loaded on demand from `SKILL.md`. Read this only when the user asks for burned or imported subtitles.

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

```bash
python helpers/build_srt.py edit/edl.json
```

This writes `<edit>/master.srt` straight from the cached Parakeet transcripts in `<edit>/transcripts/`, on the OUTPUT timeline. Ship it alongside `cut.fcpxml` and `cut.xml`. Most NLEs (Premiere, Resolve, FCP X) import SRT as a captions track the editor can restyle in their own caption panel. The skill never burns subtitles into a flat MP4 — XML-only delivery means the NLE owns the final pixels and the editor controls caption style end-to-end.

## Decision shortcuts

| Content                          | Chunking      | Case      | MarginV |
|----------------------------------|---------------|-----------|---------|
| TikTok / Reels / Shorts          | 2 words       | UPPER     | 35–80   |
| Tech launch / explainer          | 3–4 words     | UPPER     | 40–60   |
| Tutorial / how-to                | sentence      | Sentence  | 60      |
| Documentary / interview          | sentence      | Sentence  | 70      |
| Music video / lyric              | line per beat | varies    | varies  |
