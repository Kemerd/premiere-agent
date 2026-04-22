# Animations (when requested)

> Loaded on demand from `SKILL.md`. Read this only when the user asks for overlay animations / motion graphics.

Animations match the content and the brand. **Get the palette, font, and visual language from the conversation** — never assume a default. If the user hasn't told you, propose a palette in the strategy phase and wait for confirmation before building anything.

## Tool options

- **PIL + PNG sequence + ffmpeg** — simple overlay cards: counters, typewriter text, single bar reveals, progressive draws. Fast to iterate, any aesthetic you want. The launch video used this.
- **Manim** — formal diagrams, state machines, equation derivations, graph morphs. Read `skills/manim-video/SKILL.md` and its references for depth.
- **Remotion** — typography-heavy, brand-aligned, web-adjacent layouts. React/CSS-based.

None is mandatory. Invent hybrids if useful (e.g. PIL background with a Remotion layer on top, or Manim diagram exported as PNG sequence and composited as an overlay).

## Duration — context-dependent rules of thumb

- **Sync-to-narration explanations.** A viewer needs to parse the content at 1×. Rough floor 3s, typical 5–7s for simple cards, 8–14s for complex diagrams. The launch video shipped at 5–7s per simple card.
- **Beat-synced accents** (music video, fast montage). 0.5–2s is fine — they're visual accents, not information. The "readable at 1×" rule becomes *"recognizable at 1×"*, not *"fully parseable."*
- **Hold the final frame ≥ 1s** before the cut (universal).
- **Over voiceover:** total duration ≥ `narration_length + 1s` (universal).
- **Never parallel-reveal independent elements** — the eye can't track two new things at once. One thing, pause, next thing.

## Animation payoff timing (sync-to-narration)

Get the payoff word's timestamp from `speech_timeline.md`. Start the overlay `reveal_duration` seconds earlier so the landing frame coincides with the spoken payoff word. Without this sync the animation feels disconnected from the narration.

```
overlay.start_in_output = payoff_word_output_time - reveal_duration
```

## Easing — universal, never `linear` (looks robotic)

```python
def ease_out_cubic(t):
    return 1 - (1 - t) ** 3

def ease_in_out_cubic(t):
    if t < 0.5:
        return 4 * t ** 3
    return 1 - (-2 * t + 2) ** 3 / 2
```

`ease_out_cubic` for single reveals (slow landing). `ease_in_out_cubic` for continuous draws (gentle in, gentle out).

## Typing-text anchor trick

When animating text that grows letter-by-letter, **center on the FULL string's width**, not the partial-string width — otherwise the text slides left during reveal. Compute the full bounding box once, then draw the partial string left-aligned inside that box (or use the box's center as the anchor and offset the partial string appropriately).

## Example palette — the launch video (one aesthetic among infinite)

- Background `(10, 10, 10)` near-black
- Accent `#FF5A00` / `(255, 90, 0)` orange
- Labels `(110, 110, 110)` dim gray
- Font: Menlo Bold at `/System/Library/Fonts/Menlo.ttc` (index 1)
- ≤ 2 accent colors, ~40% empty space, minimal chrome
- Result: terminal / retro tech feel

This is one style. If the brand is warm and serif, use that. If it's colorful and playful, use that. If the user handed you a style guide, follow it. If they didn't, propose one and confirm before building.

## Hard rules — production correctness

1. **Overlays use `setpts=PTS-STARTPTS+T/TB`** to shift the overlay's frame 0 to its window start (Hard Rule 4 in main SKILL.md). Without this you see the middle of the animation during the overlay window.
2. **Subtitles still LAST in the filter chain** (Hard Rule 1) — overlays come before subtitle burn-in.
3. **Parallel sub-agents for multiple animations** (Hard Rule 10) — never sequential. Spawn N at once via the `Agent` tool; total wall time ≈ slowest one.

## Parallel sub-agent brief

Each animation is one sub-agent spawned via the `Agent` tool. Each prompt is **self-contained** — sub-agents have no parent context, so the prompt must include everything needed to build the slot in isolation. Include:

1. **One-sentence goal:** *"Build ONE animation: [spec]. Nothing else."*
2. **Absolute output path** (`<edit>/animations/slot_<id>/render.mp4`)
3. **Exact technical spec:** resolution, fps, codec, pix_fmt, CRF, duration
4. **Style palette** as concrete values (RGB tuples, hex, or reference to a design system)
5. **Font path with index**
6. **Frame-by-frame timeline:** what happens when, with easing
7. **Anti-list:** ("no chrome, no extras, no titles unless specified")
8. **Code pattern reference:** copy helpers inline, don't import across slots
9. **Deliverable checklist:** script, render, verify duration via `ffprobe`, report
10. **"Do not ask questions. If anything is ambiguous, pick the most obvious interpretation and proceed."**

One sub-agent = one file. Use unique filenames so parallel agents don't overwrite each other.

## Output spec — overlay container

- Codec `libx264` or `prores_ks` (alpha) depending on whether you need transparency
- `pix_fmt yuv420p` (no alpha) or `yuva444p10le` (with alpha, ProRes 4444)
- CRF 16–18 for x264 overlays (visually lossless against a near-black background)
- Frame rate matches the final timeline (24/30/60 — ask the user)
- Resolution matches the final output canvas (1920×1080, 1080×1920, 3840×2160…)

Verify the rendered overlay's duration with `ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 <render.mp4>` before reporting back. A duration mismatch silently desyncs the overlay against the timeline.
