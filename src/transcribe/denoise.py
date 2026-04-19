"""Heuristic noise removal for rendered transcripts."""

import re

_TS = r"\[\d+:\d{2}(?::\d{2})?(?:\s*\|\s*[^\]]+)?\]"
_TS_PARSE = re.compile(r"\[(\d+):(\d{2})(?::(\d{2}))?")

_FILLER_RE = re.compile(
    rf"^{_TS} (Yeah\.?|Mm+\.?|Ah+\.?|Right\.?|Alright\.?|Uh-?huh\.?|M?hm\.?|Sure\.?|Okay\.?|Yep\.?|Ha+\.?|Hmm+\.?)$",
    re.IGNORECASE,
)
_BUMPER_RE = re.compile(
    r"(?:you are|you'?re) listening to.{0,80}(?:cooking issues|heritage radio)",
    re.IGNORECASE,
)
_SPONSOR_RE = re.compile(r"\bbrought to you by\b|\bsponsored by\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"\d{3}[.-]\d{3}[.-]\d{4}")


def _strip_phones(para: str) -> str:
    # Remove "That's/That is <phone>" repetitions before stripping the primary number
    p = re.sub(r"[.,]?\s*[Tt]hat(?:'s| is)\s+\d{3}[.-]\d{3}[.-]\d{4}[,.]?", "", para)
    p = _PHONE_RE.sub("", p)
    p = re.sub(r",\s*\.", ".", p)  # ", ." → "."
    p = re.sub(r" {2,}", " ", p)  # collapse runs of spaces
    p = re.sub(r" ([.,])", r"\1", p)  # remove space before punctuation
    return p.strip()


_BOILERPLATE = [
    "Heritage Radio Network",
    "Thank you for listening",
]
_SHORT_THRESHOLD = 300
_PREROLL_KEYWORDS = ("cooking", "food", "heritage radio", "french culinary")

# Music detection: runs of consecutive short-content paragraphs with dense timestamps
_MUSIC_MAX_CHARS = 80
_MUSIC_MIN_RUN = 5
_MUSIC_MAX_AVG_GAP_S = 15


def _seconds(para: str) -> int:
    m = _TS_PARSE.match(para)
    if not m:
        return -1
    a, b, c = m.group(1), m.group(2), m.group(3)
    if c:
        return int(a) * 3600 + int(b) * 60 + int(c)
    return int(a) * 60 + int(b)


def _content(para: str) -> str:
    return re.sub(rf"^{_TS}\s*", "", para)


def denoise(text: str) -> str:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    kept = []
    for para in paras:
        if _FILLER_RE.match(para):
            continue
        if len(para) < _SHORT_THRESHOLD and any(phrase in para for phrase in _BOILERPLATE):
            continue
        if _BUMPER_RE.search(para):
            continue
        if _SPONSOR_RE.search(para):
            continue
        # Pre-roll: very early content unrelated to the show (ads for other podcasts, etc.)
        t = _seconds(para)
        if 0 <= t < 30:
            low = _content(para).lower()
            if not any(kw in low for kw in _PREROLL_KEYWORDS):
                continue
        if _PHONE_RE.search(para):
            para = _strip_phones(para)
            if not _content(para).strip():
                continue
        kept.append(para)

    # Music block detection: ≥5 consecutive paragraphs with short content and dense timestamps
    contents = [_content(p) for p in kept]
    is_short = [len(c) <= _MUSIC_MAX_CHARS for c in contents]

    remove: set[int] = set()
    i = 0
    while i < len(kept):
        if not is_short[i]:
            i += 1
            continue
        j = i + 1
        while j < len(kept) and is_short[j]:
            j += 1
        run = j - i
        if run >= _MUSIC_MIN_RUN:
            times = [_seconds(kept[k]) for k in range(i, j)]
            gaps = [times[k + 1] - times[k] for k in range(run - 1) if times[k] >= 0 and times[k + 1] >= 0]
            pos = [g for g in gaps if g > 0]
            if pos and sum(pos) / len(pos) < _MUSIC_MAX_AVG_GAP_S:
                remove.update(range(i, j))
        i = j

    result = [p for idx, p in enumerate(kept) if idx not in remove]
    return "\n\n".join(result)
