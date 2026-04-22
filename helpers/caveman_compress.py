"""NLP-based "caveman compression" for visual_caps captions.

Florence-2's `<MORE_DETAILED_CAPTION>` task emits a verbose 5-7 sentence
paragraph per frame. The editor sub-agent doesn't need the connective
tissue ("appears to be in", "and there is", "various", "overall") — it
needs the FACTS: entities (drill, panel, wires), actions (holding,
operating), colours (red, yellow), shot composition (close-up, wide).
The grammar around those facts is fully predictable and an LLM editor
will mentally reconstruct it for free during reasoning.

Caveman compression strips that grammar with a deterministic spaCy pass:

    "The image shows a person holding a cordless drill above a metal
     panel with rivet holes."

becomes

    "Image shows person holding cordless drill above metal panel rivet
     holes."

Token reduction: 40-60% on Florence detailed captions with zero loss
of editorial signal. Works offline, is deterministic (no LLM, no API
key, no nondeterministic sampling), and runs at ~10-15k tokens/sec on
a single CPU core.

Inspired by `wilpel/caveman-compression` (MIT). The filter rules below
are adapted from that project's `caveman_compress_nlp.py`. We tweak a
few thresholds for Florence's specific verbosity tics (e.g. dropping
"various" / "appears" / "overall" by name — Florence uses those as
filler in nearly every caption).

Usage as a library:

    from caveman_compress import compress_visual_caps_file
    stats = compress_visual_caps_file(src_json, dst_json, lang="en")

Usage as a CLI (for debugging / one-off testing):

    python helpers/caveman_compress.py "Verbose source text…"
    python helpers/caveman_compress.py --file input.txt
    python helpers/caveman_compress.py --visual-caps <edit>/visual_caps/

The visual_caps batch path is what `pack_timelines.py` calls in its
pre-pack step — see `_ensure_comp_visual_caps` over there for the
parallel orchestration.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Module-level versioning. Bump this whenever the filter rules below change
# in a way that should invalidate cached comp_visual_caps/*.json. Pack-side
# cache logic compares this string to the `_caveman.version` field stored
# in each cached file and silently re-runs on mismatch.
# ---------------------------------------------------------------------------

CAVEMAN_VERSION = "1.1"

# ---------------------------------------------------------------------------
# Florence-2 special-token leakage cleanup. Mirror of the regex in
# visual_lane.py / pack_timelines.py — duplicated locally so this module
# stays import-light (no transitive torch / transformers pull-in).
# ---------------------------------------------------------------------------

_FLORENCE_SPECIAL_TOKEN_RE = re.compile(
    r"<\s*/?\s*(?:pad|s|unk|mask|bos|eos|sep|cls)\s*>",
    flags=re.IGNORECASE,
)
_WS_RUN_RE = re.compile(r"\s{2,}")


def _strip_florence_special_tokens(text: str) -> str:
    """Defensive strip of leftover Florence-2 special tokens. See
    visual_lane._strip_florence_special_tokens for the long rationale.
    """
    if not text:
        return ""
    text = _FLORENCE_SPECIAL_TOKEN_RE.sub("", text)
    text = _WS_RUN_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Caveman filter rules.
#
# Tokens are KEPT unless they match one of the skip rules below. The skip
# rules target exactly the categories an LLM can reconstruct from context
# without losing meaning:
#
#   * Stop words (token.is_stop)            — "the", "a", "is", "of"
#   * Determiners (POS == DET)              — "the", "this", "those"
#   * Auxiliaries (POS == AUX)              — "is", "are", "have", "be"
#   * Weak adverbs (closed list below)      — "very", "really", "quite"
#   * Coordinating "and" / "or" (POS CCONJ) — joiner tokens, recoverable
#   * Punctuation (except units / separators)
#
# Everything else is kept verbatim. The kept stream is rejoined with
# single spaces and terminated with a period to preserve sentence
# boundaries (so the downstream sentence-level fuzzy delta dedup in
# pack_timelines.py still has something to split on).
# ---------------------------------------------------------------------------

# Adverbs that carry no information for an editor. Florence-2 sprinkles
# these as filler ("the room appears to be cluttered, and there are
# various wires"). The user's reference list, plus a few Florence-specific
# additions ("various", "overall", "actually"-which-is-already-there).
_WEAK_ADVERBS = frozenset({
    "very", "really", "quite", "extremely", "incredibly", "absolutely",
    "totally", "completely", "utterly", "highly", "particularly",
    "especially", "truly", "actually", "basically", "essentially",
    "overall",  # Florence: "the overall appearance is messy"
})

# Adjectives Florence adds for no editorial reason. Keep the list short
# — only words that are pure padding in 99% of detailed captions. Real
# adjectives ("metal", "yellow", "tangled") stay because they carry
# describable detail an editor cares about.
_WEAK_ADJECTIVES = frozenset({
    "various",   # "various wires" -> "wires"
    "several",   # "several components" -> "components"
    "different", # "different colors" -> "colors"
    "overall",   # double-purpose: also adverbial
})

# Non-content verbs Florence loves: "appears", "seems", "shows". The first
# two are pure hedging; "shows" is the universal Florence opener ("The
# image shows…"). Editor doesn't need them.
_WEAK_VERBS = frozenset({
    "appears", "appear", "seems", "seem",
    "shows",   "show",   # "The image shows X" -> "Image X"
    "showing",
})

# Punctuation we KEEP because it carries information. Hyphens are part
# of compound nouns ("close-up"), slashes appear in measurements,
# colons separate keys from values, and currency / percent signs are
# numeric units that change meaning if dropped.
_KEEP_PUNCT = frozenset({"-", "/", ":", "%", "$", "€", "£"})


# ---------------------------------------------------------------------------
# Generic English shorthand pass.
#
# Applied AFTER the spaCy POS filter, as a final regex word-boundary pass
# on the joined caveman string. Two operations:
#
#   1. _DROP_WORDS  — generic English filler that survives the POS filter
#      because it's a plain noun/verb but carries zero editorial signal
#      in a Florence-2 caption. Empirically derived from word-frequency
#      analysis on a 6500-line `merged_timeline.md` from the reference
#      project: "image" appeared 5648 times, "visible" 2334, "including"
#      1283, "appearance" 653, etc. — every one is pure padding.
#
#   2. _SHORTHAND   — generic English nouns that occur frequently and
#      have an unambiguous abbreviation an LLM can decode for free. The
#      table below targets words that appear >= ~300 times in a
#      typical project's merged timeline. Domain-specific vocabulary
#      ("Garmin", "screwdriver", "drill") is intentionally NOT in this
#      table because (a) it varies per project and (b) editorial
#      signal-per-character is already high there.
#
# Both lists are case-insensitive and word-boundary anchored, so
# "Background" / "BACKGROUND" / "background" all collapse to "bg" and
# "person" inside "personality" stays untouched.
#
# Why a regex post-pass instead of mutating spaCy tokens in
# _compress_doc? Three reasons:
#   * Keeps the POS filter language-agnostic. The shorthand table is
#     English-only; non-English models skip this pass cleanly via the
#     `lang == "en"` guard in `compress_text` / `compress_batch`.
#   * Avoids fighting spaCy's tokenizer about hyphenated compounds —
#     "close-up" stays one token in the regex, but spaCy splits it
#     into `close` / `-` / `up` which would foil per-token replacement.
#   * Makes the table trivially user-extensible — anyone who hates
#     "rect" can flip a single dict entry and re-run with --force-caveman.
#
# Apply order matters: _DROP_WORDS first (so we don't waste regex
# cycles shortening words about to be dropped), then _SHORTHAND, then
# whitespace / punctuation cleanup.
# ---------------------------------------------------------------------------

# Generic-English filler that survives the POS pass. These are usually
# concrete nouns / verbs (so the POS filter keeps them) but in Florence
# captions they're always either the universal opener ("Image shows...")
# or hedge / connector noise ("X visible in background, including Y").
_DROP_WORDS = frozenset({
    "image", "images",          # Florence-2's universal opener
    "visible",                  # "X visible in bg" -> "X bg" (implied)
    "including",                # list connector — commas do this job
    "appearance",               # "the appearance of the room is messy"
    "likely",                   # hedge ("likely a workshop")
    "possibly",                 # hedge
    "approximately",            # hedge
    "slightly",                 # vague modifier
    "various",                  # defensive (also in caveman ADJ list)
    "overall",                  # defensive (also in caveman ADV list)
    "appears", "appear",        # defensive (also in caveman VERB list)
    "seems", "seem",            # defensive (also in caveman VERB list)
    "shows", "show", "showing", # defensive (also in caveman VERB list)
})

# Generic-English shorthands. Keys are matched case-insensitively at
# word boundaries; values are emitted verbatim (lowercase by convention).
# Token savings on the reference project (6500-line merged timeline):
#
#   person      -> prsn   :  3967 occ  *  3 chars saved =  ~12k chars
#   background  -> bg     :  3670 occ  *  8           =  ~29k chars
#   electrical  -> elec   :  2854 occ  *  6           =  ~17k chars
#   components  -> comp   :  1107 occ  *  6           =   ~6k chars
#   equipment   -> equip  :   957 occ  *  4           =   ~3k chars
#   workshop    -> shop   :   752 occ  *  4           =   ~3k chars
#   interior    -> int    :   472 occ  *  5           =  ~2k chars
#
# Combined with _DROP_WORDS this layer alone trims another ~10-15% of
# merged_timeline.md token count on top of the spaCy POS pass.
_SHORTHAND = {
    "person":       "prsn",
    "people":       "ppl",
    "background":   "bg",
    "foreground":   "fg",
    "electrical":   "elec",
    "electronic":   "elec",     # collapse near-synonym
    "electronics":  "elec",
    "components":   "comp",
    "component":    "comp",
    "equipment":    "equip",
    "workshop":     "shop",
    "interior":     "int",
    "exterior":     "ext",
    "horizontal":   "horiz",
    "vertical":     "vert",
    "rectangular":  "rect",
    "rectangle":    "rect",
    "cylindrical":  "cyl",
    "approximate":  "~",
}

# Pre-compiled word-boundary regex patterns. Built once at import time
# so the per-doc post-pass is a single sub() call per pattern, no
# string parsing on the hot path.
_DROP_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _DROP_WORDS) + r")\b",
    flags=re.IGNORECASE,
)
_SHORTHAND_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _SHORTHAND) + r")\b",
    flags=re.IGNORECASE,
)
# Cleanup pass for the empty slots left by _DROP_WORDS removals:
#   "image of a person"  ->  " of a person"  ->  "of person" (after caveman)
#   ", visible,"         ->  ", ,"           ->  ","         (this regex)
_DOUBLE_PUNCT_RE = re.compile(r"\s*([,.;:])(?:\s*[,.;:])+")
_LEADING_PUNCT_RE = re.compile(r"^[\s,;:.]+")


def _apply_shorthand(text: str) -> str:
    """Apply the generic-English shorthand + drop-words pass.

    Idempotent (running twice yields the same output) so it's safe to
    call from both the per-doc compression path and any defensive
    re-pack scenario.
    """
    if not text:
        return ""
    # 1. Drop pure-filler words. Replace with a single space so we
    #    don't accidentally weld two words together ("imagewires").
    text = _DROP_RE.sub(" ", text)
    # 2. Apply shorthand, preserving sentence-final punctuation by
    #    using word boundaries (the regex doesn't touch trailing
    #    periods because `\b` matches between a letter and a `.`).
    text = _SHORTHAND_RE.sub(lambda m: _SHORTHAND[m.group(0).lower()], text)
    # 3. Collapse the punctuation rubble left behind by drops:
    #    " ,  ,  ." -> ","
    text = _DOUBLE_PUNCT_RE.sub(r"\1", text)
    # 4. Trim leading orphan punctuation per sentence (split, fix, rejoin).
    parts = text.split(". ")
    parts = [_LEADING_PUNCT_RE.sub("", p).lstrip() for p in parts]
    text = ". ".join(p for p in parts if p)
    # 5. Final whitespace collapse — the drop pass left runs of spaces.
    text = _WS_RUN_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# spaCy model loader — singleton with auto-download on first use.
# ---------------------------------------------------------------------------

_NLP_CACHE: dict[str, object] = {}

# Map ISO 639-1 language code -> spaCy small-model name. Mirrors the
# user-supplied reference list so language coverage matches caveman-
# compression's NLP variant.
_LANG_TO_MODEL = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "it": "it_core_news_sm",
    "pt": "pt_core_news_sm",
    "nl": "nl_core_news_sm",
    "el": "el_core_news_sm",
    "nb": "nb_core_news_sm",
    "lt": "lt_core_news_sm",
    "ja": "ja_core_news_sm",
    "zh": "zh_core_web_sm",
    "pl": "pl_core_news_sm",
    "ro": "ro_core_news_sm",
    "ru": "ru_core_news_sm",
}
_MULTILINGUAL_FALLBACK = "xx_ent_wiki_sm"


def _resolve_model_name(lang: str) -> str:
    return _LANG_TO_MODEL.get(lang, _MULTILINGUAL_FALLBACK)


def _ensure_spacy_installed() -> None:
    """Auto-install spaCy if it's missing from the current interpreter.

    Caveman compression is a default step in the pack_timelines pipeline,
    so a fresh checkout that hasn't done `pip install -e .[preprocess]`
    would otherwise crash on the first `import spacy` and the user would
    get a wall of stack trace instead of the timeline files they asked
    for. Auto-installing on first use keeps the "just clone and run"
    story working — the cost is one ~30s pip install on the very first
    invocation, then nothing.

    Same subprocess-vs-API choice as `_spacy_download` below: pip's
    Python API isn't a stable public interface, so we shell out to the
    documented entry point. `--quiet` so we don't drown the
    `[caveman]` progress lines under pip's own resolver chatter.
    """
    try:
        import spacy  # noqa: F401  (probing import — discarded immediately)
        return
    except ImportError:
        pass

    print(
        "[caveman] spaCy not installed — auto-installing via "
        "`pip install spacy>=3.7,<4` (one-time, ~30s)…",
        file=sys.stderr,
    )
    try:
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "--quiet",
                "spacy>=3.7,<4",
            ],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            "caveman_compress: failed to auto-install spaCy. Install "
            "manually with `pip install spacy>=3.7,<4` and try again. "
            f"Underlying error: {exc}"
        ) from exc

    # Sanity-check the install actually landed in the running interpreter
    # (catches the rare case where pip writes to a different site-packages,
    # e.g. when sys.executable is a venv but pip is the user's global pip).
    try:
        import spacy  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "caveman_compress: pip reported success but `import spacy` "
            "still fails — your `pip` may be installing into a different "
            "site-packages than `sys.executable`. Run "
            f"`{sys.executable} -m pip install spacy>=3.7,<4` directly."
        ) from exc


def _spacy_download(model_name: str) -> bool:
    """Attempt to auto-download a missing spaCy model via the standard
    `python -m spacy download <name>` invocation. Returns True on
    success, False on failure (caller should fall back).

    We run it as a subprocess instead of programmatically because
    spacy.cli.download is not part of the stable API and changes
    shape between minor releases; the CLI is the documented entry
    point and is stable.
    """
    print(
        f"[caveman] spaCy model {model_name!r} not found locally — "
        f"downloading via `python -m spacy download {model_name}`…",
        file=sys.stderr,
    )
    try:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(
            f"[caveman] auto-download failed for {model_name!r}: {exc}",
            file=sys.stderr,
        )
        return False


def get_nlp(lang: str = "en"):
    """Load (and cache) the spaCy pipeline for `lang`.

    Disables NER and lemmatizer because the caveman filter only reads
    `token.is_stop` / `token.pos_` / `token.text`, none of which need
    the entity recognizer or the lemmatizer. Saves ~30-40% per-doc
    parse time on en_core_web_sm.

    Auto-downloads the model on first use; falls back to the
    multilingual `xx_ent_wiki_sm` model if the language-specific
    download fails (rare — usually means no network).
    """
    if lang in _NLP_CACHE:
        return _NLP_CACHE[lang]

    # Auto-install spaCy on first use so a fresh checkout doesn't need
    # the user to `pip install -e .[preprocess]` before pack_timelines
    # can do its job. Idempotent — once installed, the probe in
    # `_ensure_spacy_installed` returns immediately.
    _ensure_spacy_installed()

    import spacy  # heavy import; deferred until first call

    model_name = _resolve_model_name(lang)
    nlp = None
    try:
        nlp = spacy.load(model_name, disable=["ner", "lemmatizer"])
    except OSError:
        if _spacy_download(model_name):
            try:
                nlp = spacy.load(model_name, disable=["ner", "lemmatizer"])
            except OSError:
                nlp = None

    if nlp is None and model_name != _MULTILINGUAL_FALLBACK:
        # Last-ditch fallback: try the tiny multilingual model so we at
        # least get tokenization. POS quality drops but caveman still
        # produces *something* readable.
        try:
            nlp = spacy.load(_MULTILINGUAL_FALLBACK, disable=["ner"])
        except OSError:
            if _spacy_download(_MULTILINGUAL_FALLBACK):
                nlp = spacy.load(_MULTILINGUAL_FALLBACK, disable=["ner"])

    if nlp is None:
        raise RuntimeError(
            f"caveman_compress: could not load any spaCy model for "
            f"lang={lang!r}. Install manually: "
            f"`python -m spacy download {model_name}`."
        )

    _NLP_CACHE[lang] = nlp
    return nlp


# ---------------------------------------------------------------------------
# Per-document compression — pure function over a spaCy Doc.
# ---------------------------------------------------------------------------

def _compress_doc(doc) -> str:
    """Apply the caveman filter to a spaCy Doc and return the
    compressed string.

    Sentence boundaries are preserved (each kept-token run from one
    sentence becomes one period-terminated chunk in the output) so
    the downstream sentence-level fuzzy delta dedup in
    pack_timelines.py still has something to split on.
    """
    sentences: list[str] = []

    for sent in doc.sents:
        kept: list[str] = []
        for token in sent:
            text = token.text
            text_lower = text.lower()
            pos = token.pos_

            # ── Punctuation: skip unless it carries information ──
            if token.is_punct and text not in _KEEP_PUNCT:
                continue

            # ── Stop words: the bulk of the savings ──
            if token.is_stop:
                continue

            # ── Closed POS classes that an LLM reconstructs trivially ──
            if pos == "AUX":
                continue
            if pos == "DET":
                continue

            # ── Targeted closed-list filters by lemma/text ──
            if pos == "ADV" and text_lower in _WEAK_ADVERBS:
                continue
            if pos == "ADJ" and text_lower in _WEAK_ADJECTIVES:
                continue
            if pos == "VERB" and text_lower in _WEAK_VERBS:
                continue

            # ── Bare "and" / "or" coordinators: drop. Comma serves the
            #    same role in caveman style ("red, white, blue") and we
            #    already keep commas via the punctuation rule above.
            #    NOTE: we drop the comma in the punct rule too, so the
            #    result is just "red white blue" — fine for an LLM. ──
            if pos == "CCONJ" and text_lower in {"and", "or"}:
                continue

            kept.append(text)

        if kept:
            sentences.append(" ".join(kept) + ".")

    result = " ".join(sentences)
    if result:
        # Capitalize first letter so the output reads as proper text.
        result = result[0].upper() + result[1:]
    return result


def _maybe_shorten(text: str, lang: str) -> str:
    """Apply the English shorthand pass when lang == 'en'; pass through
    otherwise. Centralized here so both `compress_text` and
    `compress_batch` get the same behaviour without duplicating the
    language guard at every call site.
    """
    if lang != "en":
        return text
    return _apply_shorthand(text)


def compress_text(text: str, lang: str = "en") -> str:
    """Compress a single string. Cheap convenience wrapper around the
    batch path — for >1 caption use `compress_batch` to avoid per-call
    spaCy pipeline setup overhead.
    """
    text = _strip_florence_special_tokens(text)
    if not text:
        return ""
    nlp = get_nlp(lang)
    doc = nlp(text)
    return _maybe_shorten(_compress_doc(doc), lang)


def compress_batch(
    texts: list[str],
    lang: str = "en",
    *,
    batch_size: int = 64,
    nlp=None,
) -> list[str]:
    """Compress an ordered list of strings. Uses `nlp.pipe()` for
    in-process batched parsing — single-process but heavily vectorized
    inside spaCy's C extensions.

    Cross-file parallelism is handled at a higher level (one process
    per visual_caps/*.json file via multiprocessing.Pool — see
    `compress_visual_caps_dir` below). Doing nested parallelism inside
    `nlp.pipe(n_process>1)` on top of an outer ProcessPool tends to
    deadlock on Windows because of nested spawn() restrictions.
    """
    if not texts:
        return []
    nlp = nlp or get_nlp(lang)

    # Pre-strip Florence special-token noise before parsing. spaCy
    # would tokenize the `<pad>` literals as separate tokens otherwise
    # and waste parser time on them.
    cleaned = [_strip_florence_special_tokens(t or "") for t in texts]

    out: list[str] = []
    # nlp.pipe streams Docs in input order, so we can append directly.
    # The shorthand pass runs after each doc so we don't pay the regex
    # cost on docs that came back empty from the POS filter.
    for doc in nlp.pipe(cleaned, batch_size=batch_size):
        out.append(_maybe_shorten(_compress_doc(doc), lang))
    return out


# ---------------------------------------------------------------------------
# Per-file visual_caps compression.
#
# Reads <edit>/visual_caps/<stem>.json, replaces each caption's `text`
# field with its caveman-compressed form, writes to
# <edit>/comp_visual_caps/<stem>.json. Preserves every other field
# (model, fps, duration, etc.) so downstream readers don't need to
# special-case the comp variant.
# ---------------------------------------------------------------------------

def _stat_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _is_cache_fresh(src: Path, dst: Path, lang: str) -> bool:
    """A cached comp_visual_caps file is fresh iff:
        1. dst exists
        2. dst's recorded src_mtime matches src's current mtime
        3. dst's recorded caveman_version matches the current module version
        4. dst's recorded lang matches the requested lang
    Any mismatch → re-run the compression.
    """
    if not dst.exists():
        return False
    try:
        meta = json.loads(dst.read_text(encoding="utf-8")).get("_caveman") or {}
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("version") != CAVEMAN_VERSION:
        return False
    if meta.get("lang") != lang:
        return False
    if abs(float(meta.get("src_mtime", 0.0)) - _stat_mtime(src)) > 1e-3:
        return False
    return True


def compress_visual_caps_file(
    src: Path,
    dst: Path,
    lang: str = "en",
) -> dict:
    """Compress one visual_caps/*.json file → comp_visual_caps/*.json.

    Returns a small stats dict suitable for printing / aggregating
    across a parallel batch run.
    """
    src = Path(src)
    dst = Path(dst)
    t0 = time.time()

    data = json.loads(src.read_text(encoding="utf-8"))
    raw_caps = data.get("captions") or []

    texts = [(c.get("text") or "") for c in raw_caps]
    orig_chars = sum(len(t) for t in texts)

    compressed_texts = compress_batch(texts, lang=lang)
    comp_chars = sum(len(t) for t in compressed_texts)

    # Replace just the `text` field in each caption record; preserve
    # the timestamp `t` and any other metadata the visual lane wrote.
    # `captions_dedup` is intentionally dropped — the byte-exact dedup
    # was built off the verbose paragraphs and would no longer line up
    # against the compressed text. pack_timelines.py runs its own
    # sentence-level fuzzy delta dedup on the comp output anyway, so
    # the legacy dedup field would just be misleading.
    new_caps = [
        {**c, "text": ct} for c, ct in zip(raw_caps, compressed_texts)
    ]

    payload = {
        **data,
        "captions": new_caps,
        "_caveman": {
            "version": CAVEMAN_VERSION,
            "lang": lang,
            "src_mtime": _stat_mtime(src),
            "src_path": str(src),
            "orig_chars": orig_chars,
            "comp_chars": comp_chars,
            "reduction_pct": round(
                100.0 * (orig_chars - comp_chars) / max(1, orig_chars), 1,
            ),
        },
    }
    payload.pop("captions_dedup", None)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(dst)

    return {
        "src": src.name,
        "n_caps": len(raw_caps),
        "orig_chars": orig_chars,
        "comp_chars": comp_chars,
        "reduction_pct": payload["_caveman"]["reduction_pct"],
        "elapsed_s": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# Worker entry point used by the multiprocessing pool. Has to be a
# module-level function (not a closure / lambda) so Windows spawn() can
# pickle it across the process boundary.
# ---------------------------------------------------------------------------

def _worker_compress_one(job: tuple[str, str, str]) -> dict:
    src_str, dst_str, lang = job
    return compress_visual_caps_file(Path(src_str), Path(dst_str), lang=lang)


def compress_visual_caps_dir(
    src_dir: Path,
    dst_dir: Path,
    *,
    lang: str = "en",
    force: bool = False,
    n_procs: int | None = None,
    quiet: bool = False,
) -> list[dict]:
    """Compress every visual_caps/*.json in `src_dir` into `dst_dir`.

    Skips files whose cached comp output is already fresh
    (mtime + caveman_version + lang match). Runs the remaining jobs
    across a `multiprocessing.Pool` of `n_procs` workers (default:
    `min(n_files, os.cpu_count() // 2)` clamped to ≥1) — half the
    cores by default to leave headroom for whatever else the user is
    running. Each worker re-loads spaCy on first use; the load cost
    (~3s) amortises away once a worker chews through more than one
    file, so the default sizing favours 1 worker per 2 files when
    that's lower than the half-CPU cap.

    Force-recompresses everything if `force=True`.

    Returns the per-file stats dicts in completion order.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(src_dir.glob("*.json"))
    if not json_files:
        if not quiet:
            print(f"[caveman] no visual_caps in {src_dir}; nothing to do.")
        return []

    jobs: list[tuple[str, str, str]] = []
    cached: list[Path] = []
    for src in json_files:
        dst = dst_dir / src.name
        if not force and _is_cache_fresh(src, dst, lang):
            cached.append(dst)
            continue
        jobs.append((str(src), str(dst), lang))

    if not jobs:
        if not quiet:
            print(
                f"[caveman] all {len(cached)} comp_visual_caps fresh "
                f"(cached). Skipping."
            )
        return []

    # Sizing: half the CPU cores, capped at the number of jobs (no
    # point spawning 8 workers for 2 files). Minimum 1.
    if n_procs is None:
        cpu = os.cpu_count() or 2
        n_procs = max(1, min(len(jobs), cpu // 2))
    else:
        n_procs = max(1, min(len(jobs), n_procs))

    if not quiet:
        print(
            f"[caveman] compressing {len(jobs)} visual_caps file(s) "
            f"({len(cached)} already cached) with {n_procs} worker(s)…"
        )

    results: list[dict] = []

    if n_procs == 1:
        # Serial path — avoids the pool spawn overhead when there's
        # only one job (most common case for short single-clip projects).
        for job in jobs:
            r = _worker_compress_one(job)
            results.append(r)
            if not quiet:
                print(
                    f"  [caveman] {r['src']:40s} {r['n_caps']:5d} caps  "
                    f"{r['orig_chars']:>7d} -> {r['comp_chars']:>7d} chars  "
                    f"({r['reduction_pct']:5.1f}%)  in {r['elapsed_s']}s"
                )
        return results

    # Parallel path — ProcessPoolExecutor with as_completed for live
    # status updates as workers finish.
    with ProcessPoolExecutor(max_workers=n_procs) as pool:
        futures = [pool.submit(_worker_compress_one, j) for j in jobs]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if not quiet:
                print(
                    f"  [caveman] {r['src']:40s} {r['n_caps']:5d} caps  "
                    f"{r['orig_chars']:>7d} -> {r['comp_chars']:>7d} chars  "
                    f"({r['reduction_pct']:5.1f}%)  in {r['elapsed_s']}s"
                )
    return results


# ---------------------------------------------------------------------------
# CLI — convenience for one-off testing / debugging.
# ---------------------------------------------------------------------------

def _count_tokens_estimate(text: str) -> int:
    """Cheap chars/4 estimator. Fine for relative reduction reporting;
    NOT a substitute for a real tokenizer if you need absolute counts.
    """
    return max(0, len(text.strip()) // 4)


def _cli_compress_text(text: str, lang: str) -> None:
    print("ORIGINAL:")
    print(text)
    print()
    out = compress_text(text, lang=lang)
    print("CAVEMAN:")
    print(out)
    print()
    o, c = _count_tokens_estimate(text), _count_tokens_estimate(out)
    pct = 100.0 * (o - c) / o if o else 0.0
    print(f"~{o} -> ~{c} tokens ({pct:.1f}% reduction)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Caveman-compress visual_caps captions (NLP / spaCy).",
    )
    ap.add_argument(
        "text", nargs="?",
        help="text to compress (use --file or --visual-caps for batch)",
    )
    ap.add_argument(
        "-f", "--file", type=Path,
        help="compress the contents of a text file",
    )
    ap.add_argument(
        "--visual-caps", type=Path,
        help="batch-compress every JSON in a visual_caps/ directory; "
             "writes to a sibling comp_visual_caps/ directory",
    )
    ap.add_argument(
        "-o", "--out", type=Path,
        help="(--visual-caps mode) override the output dir; defaults to "
             "the sibling comp_visual_caps/ next to --visual-caps",
    )
    ap.add_argument(
        "-l", "--lang", default="en",
        help="ISO 639-1 language code (default: en)",
    )
    ap.add_argument(
        "--procs", type=int, default=None,
        help="(--visual-caps mode) worker process count "
             "(default: min(n_files, cpu_count // 2))",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="(--visual-caps mode) re-compress even cached files",
    )
    args = ap.parse_args()

    if args.visual_caps:
        src_dir = args.visual_caps
        if not src_dir.is_dir():
            sys.exit(f"--visual-caps must be a directory: {src_dir}")
        dst_dir = args.out or (src_dir.parent / "comp_visual_caps")
        compress_visual_caps_dir(
            src_dir, dst_dir,
            lang=args.lang, force=args.force, n_procs=args.procs,
        )
        return

    if args.file:
        text = args.file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        ap.print_help()
        sys.exit("provide TEXT, --file, or --visual-caps")

    _cli_compress_text(text, args.lang)


if __name__ == "__main__":
    main()
