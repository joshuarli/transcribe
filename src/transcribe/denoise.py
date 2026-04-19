"""Heuristic noise removal for rendered transcripts."""

import re

from word2number import w2n as _w2n

_TS = r"\[\d+:\d{2}(?::\d{2})?(?:\s*\|\s*[^\]]+)?\]"
_TS_PARSE = re.compile(r"\[(\d+):(\d{2})(?::(\d{2}))?")
_TS_PREFIX_RE = re.compile(rf"^{_TS}\s*")


def _seconds(para: str) -> int:
    m = _TS_PARSE.match(para)
    if not m:
        return -1
    a, b, c = m.group(1), m.group(2), m.group(3)
    return int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)


def _content(para: str) -> str:
    return _TS_PREFIX_RE.sub("", para)


# Spoken digit sequences → digit strings (e.g. "seven one eight" → "718").
# Must run before phone detection so formatted numbers can be matched and stripped.
_DIGIT_WORD_TO_CHAR = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_SINGLE_DIGIT_WORDS = frozenset(_DIGIT_WORD_TO_CHAR)
_digit_pat = "|".join(_DIGIT_WORD_TO_CHAR)
_SPOKEN_DIGIT_SEQ_RE = re.compile(rf"\b({_digit_pat})(?:\s+({_digit_pat}))+\b", re.IGNORECASE)

# Words that form compound numbers — used by normalize_numbers.
_ALL_NUM_WORDS = _SINGLE_DIGIT_WORDS | {
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
}
# Multiplier-only words are meaningless without a preceding number; skip when standalone.
_MULTIPLIER_ONLY = frozenset({"hundred", "thousand", "million"})


def _collapse_spoken_digits(text: str) -> str:
    def _replace(m: re.Match) -> str:
        return "".join(_DIGIT_WORD_TO_CHAR[w.lower()] for w in m.group(0).split())

    return _SPOKEN_DIGIT_SEQ_RE.sub(_replace, text)


def normalize_numbers(text: str) -> str:
    """Convert word-spelled numbers to digits (e.g. 'sixty three' → '63')."""
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        raw = words[i]
        clean = raw.lower().rstrip(".,!?;:")
        if clean not in _ALL_NUM_WORDS:
            out.append(raw)
            i += 1
            continue
        # Collect the longest contiguous run of number words.
        j = i + 1
        while j < len(words) and words[j].lower().rstrip(".,!?;:") in _ALL_NUM_WORDS:
            j += 1
        span_raw = words[i:j]
        span_clean = [w.lower().rstrip(".,!?;:") for w in span_raw]
        trailing = span_raw[-1][len(span_raw[-1].rstrip(".,!?;:")) :]
        # Skip standalone multipliers — they only make sense attached to a preceding number.
        if len(span_clean) == 1 and span_clean[0] in _MULTIPLIER_ONLY:
            out.append(raw)
            i += 1
            continue
        try:
            out.append(str(_w2n.word_to_num(" ".join(span_clean))) + trailing)
            i = j
        except ValueError:
            out.append(raw)
            i += 1
    return " ".join(out)


# normalize_numbers converts "three" → "3" before these run, so three-quarters
# appears as either "three quarters" or "3 quarters" depending on context.
_FRAC_THREE_QUARTERS_RE = re.compile(r"\b(\d+)\s+and\s+(?:three|3)\s+quarters?\b", re.IGNORECASE)
_FRAC_HALF_RE = re.compile(r"\b(\d+)\s+and\s+a\s+half\b", re.IGNORECASE)
_FRAC_QUARTER_RE = re.compile(r"\b(\d+)\s+and\s+a\s+quarter\b", re.IGNORECASE)
_FRAC_THIRD_RE = re.compile(r"\b(\d+)\s+and\s+a\s+third\b", re.IGNORECASE)

# F/C patterns must run before bare-degrees to avoid double-matching.
_TEMP_F_RE = re.compile(r"\b(\d+)\s*degrees?\s+(?:Fahrenheit|F)\b", re.IGNORECASE)
_TEMP_C_RE = re.compile(r"\b(\d+)\s*degrees?\s+(?:Celsius|Centigrade|C)\b", re.IGNORECASE)
_TEMP_BARE_RE = re.compile(r"\b(\d+)\s*degrees?\b")

_PERCENT_RE = re.compile(r"\b(\d+)\s+percent\b", re.IGNORECASE)


def _normalize_fractions(text: str) -> str:
    text = _FRAC_THREE_QUARTERS_RE.sub(r"\1¾", text)
    text = _FRAC_HALF_RE.sub(r"\1½", text)
    text = _FRAC_QUARTER_RE.sub(r"\1¼", text)
    return _FRAC_THIRD_RE.sub(r"\1⅓", text)


def _normalize_temperatures(text: str) -> str:
    text = _TEMP_F_RE.sub(r"\1°F", text)
    text = _TEMP_C_RE.sub(r"\1°C", text)
    return _TEMP_BARE_RE.sub(r"\1°", text)


def _normalize_percentages(text: str) -> str:
    return _PERCENT_RE.sub(r"\1%", text)


_MEASUREMENT_RE = re.compile(
    r"\b(?:tablespoons?|teaspoons?|ounces?|pounds?|grams?|milliliters?|liters?|kilograms?)\b",
    re.IGNORECASE,
)
_MEASUREMENT_ABBREV = {
    "tablespoon": "tbsp",
    "tablespoons": "tbsp",
    "teaspoon": "tsp",
    "teaspoons": "tsp",
    "ounce": "oz",
    "ounces": "oz",
    "pound": "lb",
    "pounds": "lb",
    "gram": "g",
    "grams": "g",
    "milliliter": "ml",
    "milliliters": "ml",
    "liter": "L",
    "liters": "L",
    "kilogram": "kg",
    "kilograms": "kg",
}


def _normalize_measurements(text: str) -> str:
    return _MEASUREMENT_RE.sub(lambda m: _MEASUREMENT_ABBREV[m.group(0).lower()], text)


# 3+ space-separated single letters → collapse ("f n y c" → "fnyc")
_LETTER_SEQ_RE = re.compile(r"\b[a-zA-Z](?:\s+[a-zA-Z]){2,}\b")
_SPOKEN_DOT_RE = re.compile(r"\b(\w+)\s+dot\s+(\w+)\b", re.IGNORECASE)


def _normalize_urls(text: str) -> str:
    text = _LETTER_SEQ_RE.sub(lambda m: m.group(0).replace(" ", ""), text)
    prev = None
    while prev != text:
        prev = text
        text = _SPOKEN_DOT_RE.sub(r"\1.\2", text)
    return text


_REPEAT_PUNCT_RE = re.compile(r"([.!?])\1+")


def _normalize_punctuation(text: str) -> str:
    return _REPEAT_PUNCT_RE.sub(r"\1", text)


_PAIN_RE = re.compile(r"\bpain in the (?:butt|ass)\b", re.IGNORECASE)
_SOME_OF_THEM_RE = re.compile(r"\bsome of (?:them|those|these|it)\b", re.IGNORECASE)


def _normalize_phrases(text: str) -> str:
    text = _PAIN_RE.sub("pain", text)
    return _SOME_OF_THEM_RE.sub("some", text)


# Matches XXX-XXX-XXXX, XXXXXXXXXX, XXXXXX-XXXX (fused area code), etc.
_PHONE_RE = re.compile(r"\b\d{3}[.-]?\d{3}[.-]?\d{4}\b")
# also matches fused (718497-2128) and no-separator (7184972128) formats
_PHONE_REPEAT_RE = re.compile(r"[.,]?\s*[Tt]hat(?:'s| is)\s+\d{3}[.-]?\d{3}[.-]?\d{4}[,.]?")


def _strip_phones(para: str) -> str:
    p = _collapse_spoken_digits(para)
    p = _PHONE_REPEAT_RE.sub("", p)
    p = _PHONE_RE.sub("", p)
    p = re.sub(r",\s*\.", ".", p)
    p = re.sub(r" {2,}", " ", p)
    p = re.sub(r" ([.,])", r"\1", p)
    return p.strip()


# Tag questions: replaced with "." to create a proper sentence break.
_RIGHT_TAG_RE = re.compile(r",\s*right\?\s*", re.IGNORECASE)
_WMEAN_TAG_RE = re.compile(r",\s*(?:you\s+know\s+)?what\s+I\s+mean\?\s*", re.IGNORECASE)

# Parenthetical filler phrases — stripped with surrounding comma/space consumed.
# "you know" lookbehind guards against "do/if you know" (semantic uses).
_YOU_KNOW_RE = re.compile(r",?\s*(?<!\bdo )(?<!\bif )\byou know\b\s*,?", re.IGNORECASE)
# At sentence start: strip "I mean" whether comma-followed or space-followed,
# but NOT when followed by "it" (semantic: "I mean it when I say...").
_I_MEAN_RE = re.compile(r"^,?\s*I mean\b(?:,\s*|\s+(?!it\b))|,\s*I mean\b,\s*", re.IGNORECASE)
_BY_THE_WAY_RE = re.compile(r"^by the way\b,?\s*|,\s*by the way\s*,\s*", re.IGNORECASE)
# "you see" as parenthetical — only stripped when comma-fenced or at end of clause
_YOU_SEE_RE = re.compile(r",\s*you see\b,?\s*|\byou see\s*,\s*", re.IGNORECASE)
# Sentence-opening discourse fillers — stripped in a loop to handle stacked openers ("So like...").
# "like" and "so" only stripped at sentence start; mid-sentence they carry meaning.
_OPENER_FILLER_START_RE = re.compile(
    r"^,?\s*\b(?:basically|essentially|like|so|yeah|anyway|well|now|listen|look)\b,?\s*",
    re.IGNORECASE,
)
# "nice" only stripped as opener when comma-fenced (avoids "nice recipe" being gutted).
_NICE_OPENER_RE = re.compile(r"^nice\s*,\s*", re.IGNORECASE)
# "Let me just say/tell you" — sentence start and parenthetical mid-sentence.
_LET_ME_JUST_RE = re.compile(
    r"^let me just (?:tell you|say)\b,?\s*|,\s*let me just (?:tell you|say)\b,?\s*", re.IGNORECASE
)
# "very, very" / "very very" → "very"
_VERY_VERY_RE = re.compile(r"\bvery,?\s+very\b", re.IGNORECASE)
# "basically/essentially" also stripped as mid-sentence parenthetical when comma-fenced.
_BASICALLY_MID_RE = re.compile(r",\s*\b(?:basically|essentially)\b,?\s*", re.IGNORECASE)
# Sentence-start "The X is" → "X is" (strips the definite article from set phrases).
_OPENING_THE_RE = re.compile(r"^the\s+(problem|thing|issue)\s+is\b,?\s*", re.IGNORECASE)
# "I think" as a hedge — strip when followed by a clause, not "about/of" (semantic).
_I_THINK_RE = re.compile(r"\bI think,?\s+(?!about\b|of\b|so\b)", re.IGNORECASE)
# Trailing hedge — strip before end-of-clause punctuation or end of string.
_OR_SOMETHING_RE = re.compile(r"\bor something(?:\s+like\s+that)?\b(?=\s*[.,!?]|\s*$)", re.IGNORECASE)

_DISFLUENCY_RE = re.compile(r"\s*(?<![a-zA-Z-])(uh+|um+|ah+|mm+|hmm+|eh|er)(?![a-zA-Z-])\s*,?", re.IGNORECASE)
# stutter dedup: consecutive identical tokens (supports contractions like we'll, he's)
_STUTTER_RE = re.compile(r"\b((?:\w|')+)\s+\1\b", re.IGNORECASE)
# contraction false-start: "it's a it's" → "it's"; "it's it" at end of clause → "it's"
_ITS_STUTTER_RE = re.compile(r"\bit's\s+\w+\s+it's\b", re.IGNORECASE)
_ITS_IT_RE = re.compile(r"\bit's\s+it\b(?=\s*[.,!?]|\s*$)", re.IGNORECASE)

# sentence splitting: break on .!? followed by whitespace + uppercase/digit/quote
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z\d"])')

_FILLER_SENTENCE_RE = re.compile(
    r"^(?:yeah|yep|yup|okay|ok|right|alright|sure|exactly|absolutely|"
    r"totally|nice|great|anyway|said|mm+|hmm+|ah+|oh+|uh-?huh|mm-?hmm|got\s+it|gotcha|"
    r"what\s+do\s+you\s+think|"
    r"etc\.?|and\s+so\s+on\.?|blah(?:[\s-]+blah(?:[\s-]+blah)?)?\.?|"
    r"cooking\s+issues|"
    r"you\s+know\s+what\s+I\s+mean|what\s+I\s+mean|I\s+think)[.!?]?$",
    re.IGNORECASE,
)
# sentences about the show's call-in line — logistical noise, not content
_CALL_IN_SENTENCE_RE = re.compile(r"\bcall[- ]?in\b", re.IGNORECASE)
# show intro / bumper sentences at the sentence level
_INTRO_SENTENCE_RE = re.compile(
    r"\bwelcome\s+(?:back\s+)?to\b"
    r"|\bhost\s+of\b|\bhosted\s+by\b"
    r"|\bheritage\s+radio\b"
    r"|\bcoming\s+to\s+you\s+live\b"
    r"|\bthank\s+our\s+sponsors\b",
    re.IGNORECASE,
)
# known broadcast-transition fragments — anchored to full sentence
_CONNECTOR_FRAGMENT_RE = re.compile(
    r"^(?:and\s+so|but\s+then|and\s+then|or\s+so|"
    r"go\s+to\s+break|back\s+after\s+this|stay\s+tuned)[.!?]?$",
    re.IGNORECASE,
)


def _clean_sentence(text: str) -> str:
    """Word-level disfluency and filler removal within a single sentence."""
    # tag questions → sentence break
    p = _RIGHT_TAG_RE.sub(".", text)
    p = _WMEAN_TAG_RE.sub(".", p)
    # contraction false-start stutters
    p = _ITS_STUTTER_RE.sub("it's", p)
    p = _ITS_IT_RE.sub("it's", p)
    # parenthetical / hedge fillers
    p = _YOU_KNOW_RE.sub(" ", p)
    p = _YOU_SEE_RE.sub(" ", p)
    p = _I_MEAN_RE.sub(" ", p)
    p = _BY_THE_WAY_RE.sub(" ", p)
    p = _LET_ME_JUST_RE.sub(" ", p)
    p = _I_THINK_RE.sub("", p)
    p = _VERY_VERY_RE.sub("very", p)
    # sentence-opening fillers — loop handles stacked openers ("So like basically...")
    prev = None
    while prev != p:
        prev = p
        p = _OPENER_FILLER_START_RE.sub("", p)
    p = _NICE_OPENER_RE.sub("", p)
    p = _OPENING_THE_RE.sub(lambda m: m.group(1).capitalize() + " is ", p)
    p = _BASICALLY_MID_RE.sub(" ", p)
    # trailing hedge
    p = _OR_SOMETHING_RE.sub("", p)
    # word-level disfluencies
    p = _DISFLUENCY_RE.sub(" ", p)
    # stutter dedup
    prev = None
    while prev != p:
        prev = p
        p = _STUTTER_RE.sub(r"\1", p)
    p = re.sub(r" {2,}", " ", p)
    p = re.sub(r" ([.,!?])", r"\1", p)
    p = p.strip()
    if not re.search(r"\w", p):
        return ""
    # re-capitalize only when the sentence originally opened with a capital letter
    # (filler was stripped from the front), not when it was always lowercase
    if p[0].islower() and text[0].isupper():
        p = p[0].upper() + p[1:]
    return p


# Parakeet produces very long run-on sentences; a single noise phrase inside one
# shouldn't delete real content. Above this word count, excise the match inline
# instead of dropping the whole sentence.
_INLINE_EXCISE_THRESHOLD = 25


def _excise_noise(s: str) -> str:
    """Remove noise phrase matches from s in-place; return '' if nothing substantial remains."""
    s = _CALL_IN_SENTENCE_RE.sub("", s)
    s = _INTRO_SENTENCE_RE.sub("", s)
    s = re.sub(r" {2,}", " ", s).strip()
    return s if len(s.split()) >= 4 else ""


def strip_fillers(text: str) -> str:
    """Remove disfluencies, filler sentences, and noise phrases from a block of text."""
    kept = []
    for s in _SENTENCE_RE.split(text):
        s = _clean_sentence(s)
        if not s or _FILLER_SENTENCE_RE.match(s) or _CONNECTOR_FRAGMENT_RE.match(s):
            continue
        if _CALL_IN_SENTENCE_RE.search(s) or _INTRO_SENTENCE_RE.search(s):
            if len(s.split()) > _INLINE_EXCISE_THRESHOLD:
                s = _excise_noise(s)
                if not s:
                    continue
            else:
                continue
        kept.append(s)
    result = normalize_numbers(" ".join(kept))
    result = _normalize_fractions(result)
    result = _normalize_temperatures(result)
    result = _normalize_percentages(result)
    result = _normalize_measurements(result)
    result = _normalize_urls(result)
    result = _normalize_phrases(result)
    return _normalize_punctuation(result)


def strip_fillers_rendered(text: str) -> str:
    """Apply strip_fillers to each paragraph, stripping timestamp prefixes from output."""
    parts = []
    for para in text.split("\n\n"):
        cleaned = strip_fillers(_content(para))
        if cleaned:
            parts.append(cleaned)
    return "\n\n".join(parts)


_FILLER_RE = re.compile(
    rf"^{_TS} (Yeah\.?|Mm+\.?|Ah+\.?|Right\.?|Alright\.?|Uh-?huh\.?|M?hm\.?|Sure\.?|Okay\.?|Yep\.?|Ha+\.?|Hmm+\.?)$",
    re.IGNORECASE,
)
_BUMPER_RE = re.compile(
    r"(?:you are|you'?re) listening to.{0,80}(?:cooking issues|heritage radio)"
    r"|I'?m Dave Arnold.{0,20}Cooking Issues",
    re.IGNORECASE,
)
_SPONSOR_RE = re.compile(r"\bbrought to you by\b|\bsponsored by\b", re.IGNORECASE)
_BOILERPLATE = ("Heritage Radio Network", "Thank you for listening")
_PREROLL_KEYWORDS = ("cooking", "food", "heritage radio", "french culinary")
_SHORT_THRESHOLD = 300

# Music detection: runs of consecutive short-content paragraphs with dense timestamps
_MUSIC_MAX_CHARS = 80
_MUSIC_MIN_RUN = 5
_MUSIC_MAX_AVG_GAP_S = 15
# Allow isolated long paragraphs (garbled ASR of lyrics) within a music run.
_MUSIC_MIN_SHORT_RATIO = 0.75


def _music_block_indices(paras: list[str]) -> set[int]:
    n = len(paras)
    times = [_seconds(p) for p in paras]
    is_short = [len(_content(p)) <= _MUSIC_MAX_CHARS for p in paras]

    remove: set[int] = set()
    i = 0
    while i < n:
        if not is_short[i]:
            i += 1
            continue
        # Extend run, tolerating up to 1 consecutive long paragraph at a time.
        j = i + 1
        long_streak = 0
        while j < n:
            if is_short[j]:
                long_streak = 0
                j += 1
            elif long_streak < 1:
                long_streak += 1
                j += 1
            else:
                break
        # Trim trailing long paragraphs so content after the run is not removed.
        while j > i and not is_short[j - 1]:
            j -= 1
        run = j - i
        short_count = sum(is_short[i:j])
        if run >= _MUSIC_MIN_RUN and short_count / run >= _MUSIC_MIN_SHORT_RATIO:
            run_times = times[i:j]
            gaps = [
                run_times[k + 1] - run_times[k] for k in range(run - 1) if run_times[k] >= 0 and run_times[k + 1] >= 0
            ]
            pos_gaps = [g for g in gaps if g > 0]
            if pos_gaps and sum(pos_gaps) / len(pos_gaps) < _MUSIC_MAX_AVG_GAP_S:
                remove.update(range(i, j))
        i = j
    return remove


def denoise(text: str) -> str:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    kept = []
    for para in paras:
        if _FILLER_RE.match(para):
            continue
        if len(para) < _SHORT_THRESHOLD and any(b in para for b in _BOILERPLATE):
            continue
        if _BUMPER_RE.search(para) or _SPONSOR_RE.search(para):
            continue
        t = _seconds(para)
        if 0 <= t < 30 and not any(kw in _content(para).lower() for kw in _PREROLL_KEYWORDS):
            continue
        para = _collapse_spoken_digits(para)
        if _PHONE_RE.search(para):
            para = _strip_phones(para)
            if not _content(para).strip():
                continue
        kept.append(para)

    music = _music_block_indices(kept)
    return "\n\n".join(p for i, p in enumerate(kept) if i not in music)
