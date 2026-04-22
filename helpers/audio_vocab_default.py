"""Baked-in default audio vocabulary for the CLAP audio lane.

Why this list exists
--------------------
The CLAP audio lane (`helpers/audio_lane.py`) scores each audio window
against a vocabulary of natural-language sound labels. With NO vocab
file from the user, we still need SOMETHING to score against — that's
this module.

The list below is intentionally cross-domain: ~250 labels covering the
sound categories most likely to show up in the kind of footage the
project is used for (talking heads, montages, tutorials, workshop /
shop reels, travel, interviews, sports, gaming, podcasts). It is NOT
trying to be exhaustive (LAION-Audio-630k has tens of thousands of
plausible English sound phrases); it's trying to be a competent first
cut so a one-shot preprocess run produces useful events without any
agent intervention.

Phase B (better) workflow
-------------------------
The agent invoking this skill is encouraged to:

    1. Read `<edit>/speech_timeline.md` and `<edit>/visual_timeline.md`
       after the initial Phase A preprocess run.
    2. From those, infer what kinds of sounds plausibly appear in THIS
       specific video (e.g. workshop -> drill / hammer / sandpaper;
       cooking show -> sizzle / chopping / boiling; outdoor sports ->
       cheering crowd / whistle / ball impact).
    3. Write a curated list of 100-500 labels to `<edit>/audio_vocab.txt`,
       one label per line. Blank lines and `#` comments are stripped.
    4. Re-run the audio lane:
            python helpers/audio_lane.py <video> --vocab <edit>/audio_vocab.txt --force

       Cache invalidates automatically on vocab change; no `--force`
       needed in principle, but it's the convention so the user knows
       a re-run is happening.

A targeted vocab beats this default list every time because score
competition only happens between phrases the agent has reason to
believe are in the audio. Generic labels in this default ("ambient
hum", "footsteps") are noisy attractors that absorb confidence away
from the specific events actually present.

Format conventions
------------------
- Lowercase, English, plain noun phrases. The audio lane wraps each
  label in `"the sound of {}"` before encoding (ReCLAP-style prompt),
  so labels like "drill driving a screw into plywood" work well —
  natural prose, no need to be terse.
- Avoid duplicates. CLAP's text encoder will produce nearly identical
  embeddings for "dog bark" and "barking dog"; pick one, drop the
  other. Duplicate-style entries crowd the top-K and add no signal.
- Avoid "no sound" / "silence" — there's no positive contrastive
  signal for the absence of sound and the encoder produces a near-zero
  cosine for it against any window with content, which is fine, but
  the entry then crowds out a real label in the top-K race.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# The default vocabulary list (~250 labels).
#
# Grouped into thematic sections for editing convenience; the audio lane
# treats it as a flat list, sorts it for hashing, and never sees the
# group structure.
# ---------------------------------------------------------------------------

DEFAULT_VOCAB: list[str] = [
    # ── Speech-adjacent vocalizations (15) ─────────────────────────
    # Things humans do with their voices that aren't speech proper.
    # The whisper lane covers actual words; these labels surface the
    # surrounding wordless vocal events that change a scene's mood.
    "laughter",
    "applause",
    "crying",
    "coughing",
    "sneezing",
    "heavy breathing",
    "snoring",
    "whispering",
    "shouting",
    "singing",
    "humming",
    "gasping",
    "sigh",
    "clearing throat",
    "yawning",

    # ── Workshop / hand & power tools (35) ─────────────────────────
    # Heavy bias toward the maker / DIY footage this skill runs on.
    # These are the labels PANNs CNN14 most often got wrong (sandpaper
    # -> goat bleating, etc.); CLAP with a tight vocab handles them
    # much better.
    "cordless drill",
    "drill press",
    "hammer striking nail",
    "sledgehammer",
    "circular saw",
    "table saw",
    "jigsaw cutting wood",
    "miter saw",
    "router on wood",
    "wood plane",
    "wood lathe",
    "chainsaw",
    "nail gun",
    "screwdriver",
    "ratchet wrench clicking",
    "wrench tightening bolt",
    "pliers",
    "vise grip",
    "metal file scraping",
    "sandpaper rubbing wood",
    "orbital sander",
    "belt sander",
    "bench grinder",
    "angle grinder",
    "soldering iron",
    "welding arc",
    "blowtorch",
    "mallet hitting wood",
    "chisel cutting wood",
    "clamp tightening",
    "shop vacuum",
    "air compressor running",
    "pneumatic nail gun",
    "dust collector",
    "shop fan hum",

    # ── Kitchen / cooking (25) ──────────────────────────────────────
    # Recipe / cooking-show beats. The action-density of a kitchen
    # scene rivals a workshop, and the labels are equally distinct.
    "blender running",
    "food processor",
    "stand mixer",
    "microwave beep",
    "oven timer beeping",
    "kettle whistling",
    "water boiling",
    "frying sizzle",
    "chopping vegetables",
    "knife on cutting board",
    "refrigerator hum",
    "dishwasher running",
    "coffee grinder",
    "espresso machine",
    "ice clinking in glass",
    "pouring liquid",
    "opening can",
    "bacon sizzling",
    "popcorn popping",
    "bread toasting",
    "eggs frying",
    "washing dishes",
    "running water tap",
    "garbage disposal",
    "range hood fan",

    # ── Nature / animals / weather (30) ─────────────────────────────
    # Outdoor footage. Birds and weather alone cover ~half of all
    # B-roll over outdoor scenes.
    "birds chirping",
    "rooster crow",
    "dog barking",
    "cat meowing",
    "cow mooing",
    "horse neighing",
    "sheep bleating",
    "frog croaking",
    "crickets chirping",
    "bees buzzing",
    "fly buzzing",
    "mosquito",
    "owl hooting",
    "woodpecker",
    "ducks quacking",
    "geese honking",
    "wolf howling",
    "lion roar",
    "monkey chattering",
    "elephant trumpet",
    "whale song",
    "river flowing",
    "ocean waves",
    "waterfall",
    "rain falling",
    "thunder",
    "wind through trees",
    "leaves rustling",
    "hail on roof",
    "gentle breeze",

    # ── Vehicles / urban (25) ──────────────────────────────────────
    # Street and travel footage. Sirens + horns are the most editorial
    # markers — they signal a beat change in any city scene.
    "car engine idling",
    "car horn",
    "motorcycle revving",
    "truck engine",
    "bus passing",
    "train horn",
    "train passing",
    "subway train",
    "tram bell",
    "ambulance siren",
    "police siren",
    "fire truck siren",
    "helicopter overhead",
    "airplane flying",
    "jet engine",
    "propeller plane",
    "bicycle bell",
    "skateboard rolling",
    "electric scooter",
    "car door slam",
    "tire screech",
    "brake squeal",
    "traffic noise",
    "construction site",
    "jackhammer",

    # ── Music / instruments (20) ────────────────────────────────────
    # Instrument identification on the audio side complements visual
    # labels (visual lane sees the instrument, audio lane confirms
    # it's actually being played).
    "acoustic guitar",
    "electric guitar",
    "piano",
    "drum kit",
    "snare drum",
    "kick drum",
    "cymbal crash",
    "hi-hat",
    "bass guitar",
    "violin",
    "cello",
    "flute",
    "saxophone",
    "trumpet",
    "harmonica",
    "accordion",
    "synthesizer",
    "ukulele",
    "choir singing",
    "orchestral music",

    # ── Household / appliances / electronic UI (25) ─────────────────
    # Indoor scenes. Doorbell / phone / alarm sounds are common
    # narrative beats in vlogs and home tutorials.
    "doorbell ringing",
    "phone ringing",
    "phone notification",
    "alarm clock",
    "smoke alarm",
    "washing machine",
    "clothes dryer",
    "vacuum cleaner",
    "ceiling fan",
    "air conditioner",
    "space heater",
    "clock ticking",
    "footsteps on wood floor",
    "footsteps on tile",
    "footsteps on carpet",
    "walking on gravel",
    "door opening",
    "door closing",
    "door knock",
    "glass breaking",
    "dishes clattering",
    "drawer opening",
    "light switch click",
    "paper rustling",
    "plastic crinkling",

    # ── Sports / activity (15) ──────────────────────────────────────
    # Bias toward team sports + gym + cycling — the most-shot
    # athletic content categories.
    "basketball dribble",
    "ball hitting bat",
    "soccer ball kicked",
    "tennis racket hitting ball",
    "swimming splash",
    "person running",
    "person jogging",
    "jumping rope",
    "gym weights clanking",
    "treadmill running",
    "bicycle chain",
    "ice skating",
    "ping pong ball",
    "golf club swing",
    "bowling pins",

    # ── Office / digital (15) ───────────────────────────────────────
    # Screencasts, podcasts, productivity content.
    "keyboard typing",
    "mouse clicking",
    "mouse scroll wheel",
    "printer printing",
    "photocopier",
    "paper shredder",
    "pen writing on paper",
    "pencil sketching",
    "page turning",
    "stapler",
    "hole puncher",
    "calculator buttons",
    "fax machine",
    "video game controller",
    "electronic beep",

    # ── Body / impact (10) ──────────────────────────────────────────
    # Foley-grade events. Rare in talking-head footage but common in
    # action / sports / comedy reels.
    "hand clapping",
    "finger snapping",
    "slap",
    "punch impact",
    "kick impact",
    "footfall thump",
    "person jumping",
    "person falling",
    "slip and fall",
    "body hitting floor",

    # ── Ambient / room tone (10) ────────────────────────────────────
    # The "background of the background". Surfacing these helps the
    # editor identify rooms / acoustic spaces consistently across cuts.
    "room tone silence",
    "white noise",
    "hvac hum",
    "fluorescent light buzz",
    "refrigerator compressor",
    "computer fan",
    "server hum",
    "distant traffic",
    "distant chatter",
    "library quiet",

    # ── Children (5) ────────────────────────────────────────────────
    # Family / vlog content. Distinct from adult-vocal labels above
    # because the encoder uses age-cued spectral envelopes.
    "baby crying",
    "baby laughing",
    "child screaming",
    "kids playing",
    "infant babbling",

    # ── UI / SFX / alarms (10) ──────────────────────────────────────
    # Edit-bay favorites. These often sneak into source footage from
    # off-camera devices and signal scene transitions.
    "camera shutter",
    "button click",
    "error beep",
    "success chime",
    "notification ping",
    "alarm beep",
    "alert tone",
    "phone dial tone",
    "busy signal",
    "fireworks explosion",

    # ── Outdoor maintenance / yard (8) ──────────────────────────────
    # Suburban + landscaping content.
    "lawn mower",
    "leaf blower",
    "weed trimmer",
    "snow shovel",
    "broom sweeping",
    "rake on leaves",
    "garden hose spraying",
    "hedge trimmer",

    # ── Misc / discriminative (5) ───────────────────────────────────
    # Specific labels that cleanly disambiguate ambiguous mid-band
    # textures CLAP otherwise smears together.
    "gunshot",
    "balloon popping",
    "bottle uncorking",
    "zipper",
    "velcro tearing",
]


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_vocab(path: Path | str | None) -> list[str]:
    """Return the vocabulary to score against.

    Resolution:
        - `path is None`               -> the baked-in DEFAULT_VOCAB list.
        - `path` points to a file      -> read it, strip blank lines and
                                          `#`-prefixed comments, return
                                          the remaining non-empty stripped
                                          lines as the vocab.

    Trailing whitespace and Windows CRLF line endings are handled. A
    file with zero non-comment lines returns an empty list — the caller
    (audio_lane) treats that as a hard error so the user gets a clear
    message rather than silently scoring against nothing.

    Per-line `#` mid-line comments are NOT stripped (a label like
    `door knock #2` would be preserved verbatim). Only lines whose first
    non-whitespace character is `#` count as comment lines.
    """
    if path is None:
        # Defensive copy so a caller mutating the returned list can't
        # corrupt the module-level constant for other callers.
        return list(DEFAULT_VOCAB)

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"vocab file not found: {p}")

    out: list[str] = []
    # `errors="replace"` so a stray non-UTF8 byte in a hand-edited
    # file doesn't crash the whole lane; the user sees the offending
    # line as garbled text and can fix it.
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        out.append(line)
    return out
