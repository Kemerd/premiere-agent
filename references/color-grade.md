# Color grade (when requested)

> Loaded on demand from `SKILL.md`. Read this only when the user asks for grading work.

Your job is to **reason about the image**, not apply a preset. Look at a frame (via `helpers/timeline_view.py`), decide what's wrong, adjust one thing, look again. Iterate.

## Mental model — ASC CDL

Per channel: `out = (in * slope + offset) ** power`, then global saturation.

- `slope`  → highlights
- `offset` → shadows
- `power`  → midtones (gamma)

Most ffmpeg grades collapse to combinations of `eq`, `curves`, `colorbalance`, `colorchannelmixer`, and `colortemperature`. Build the chain from those primitives; don't reach for LUTs unless the user supplied one.

## Worked filter chains

`helpers/grade.py --list-presets` enumerates what ships in code. Use them as starting points or mix your own:

- **`warm_cinematic`** — retro/technical, subtle teal/orange split, desaturated. Shipped in a real launch video. Safe for talking heads.
- **`neutral_punch`** — minimal corrective: contrast bump + gentle S-curve. No hue shifts.
- **`none`** — straight copy. Default when the user hasn't asked.

For anything else — portraiture, nature, product, music video, documentary — invent your own chain. `helpers/grade.py --filter '<raw ffmpeg>'` accepts any filter string, so prototype freely.

## Hard rules for grading

1. **Apply per-segment during extraction**, not post-concat. Post-concat grading re-encodes the whole timeline twice (once concatenating, once grading). Per-segment keeps the lossless `-c copy` concat path intact (Hard Rule 2).
2. **Test skin tones before going aggressive.** Teal/orange splits and heavy saturation pushes destroy skin first. Look at a frame with a face in it, not just landscape plates.
3. **Never bake the grade for FCPXML delivery.** The colorist does the grade in the NLE — your job there is the cut. Leave the segments clean. If you have a grade direction in mind, mention it in the FCPXML clip metadata or in `project.md` so the colorist starts from your taste call.
4. **One adjustment at a time.** Slope, then offset, then power, then saturation — in that order. Adjusting two channels simultaneously while eyeballing a frame gives you no signal about which knob did what.

## Decision flow at a frame

1. Pull a frame at a representative timestamp via `timeline_view`.
2. Diagnose: are the **shadows muddy** (lift offset), **highlights blown** (pull slope down), **midtones flat** (raise power slightly), **white balance off** (`colortemperature` warm/cool), **saturation lifeless** (`eq=saturation=1.1`)?
3. Apply ONE change. Re-render the segment. Look again.
4. Cap at 3–4 iterations per segment — past that you're chasing taste, not fixing problems. Ship and move on.
