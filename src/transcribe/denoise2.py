"""High-quality NLP denoise pipeline.

Step 3 (TF-IDF information scoring) is intentionally omitted: within a single podcast
episode the corpus is too small and topically homogeneous for TF-IDF to discriminate
meaningfully. Steps 1, 2, 4, and 5 provide sufficient quality filtering.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import spacy
    from sentence_transformers import SentenceTransformer
    from spacy.matcher import Matcher
    from spacy.tokens import Doc

from transcribe.denoise import (
    _BOILERPLATE,
    _BUMPER_RE,
    _DISFLUENCY_RE,
    _FILLER_RE,
    _ITS_IT_RE,
    _ITS_STUTTER_RE,
    _PREROLL_KEYWORDS,
    _SHORT_THRESHOLD,
    _SPONSOR_RE,
    _STUTTER_RE,
    _content,
    _music_block_indices,
    _normalize_fractions,
    _normalize_measurements,
    _normalize_percentages,
    _normalize_phrases,
    _normalize_punctuation,
    _normalize_temperatures,
    _normalize_urls,
    _seconds,
    _strip_phones,
    normalize_numbers,
)

DEFAULT_SIMILARITY_THRESHOLD = 0.90

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z\d"])')
_MIN_TURN_TOKENS = 5

_CONTRAST_WORDS = frozenset(
    {
        "but",
        "however",
        "although",
        "yet",
        "while",
        "whereas",
        "nevertheless",
        "though",
        "still",
    }
)
# Single-consonant false-start: "f floral" → "floral", "s something" → "something".
# Vowels excluded to avoid stripping articles ("a apple") or pronouns ("I imagine").
_LETTER_STUTTER_RE = re.compile(r"\b([b-df-hj-np-tv-z])\s+(\1[a-z]+)\b", re.IGNORECASE)

_CAUSAL_WORDS = frozenset({"because", "since", "so", "therefore", "thus", "hence"})
# Following tokens that make "you know"/"I mean"/"I think" semantic rather than filler
_SEMANTIC_FOLLOWS = frozenset({"it", "it's", "this", "that", "him", "her", "them", "what", "why", "how"})
# Preceding tokens that make "you know"/"you see" a real question rather than backchannel
_SEMANTIC_PRECEDES = frozenset({"do", "did", "does", "if", "whether"})


def _prepass(text: str) -> list[str]:
    """Para-level boilerplate gating + timestamp stripping → plain-text paragraph list."""
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
        kept.append(para)

    music = _music_block_indices(kept)
    return [_content(p) for i, p in enumerate(kept) if i not in music]


def _normalize_para(text: str) -> str:
    """Regex normalization: disfluencies, stutters, phones, numbers, units."""
    text = _DISFLUENCY_RE.sub(" ", text)
    text = _LETTER_STUTTER_RE.sub(r"\2", text)
    text = _ITS_STUTTER_RE.sub("it's", text)
    text = _ITS_IT_RE.sub("it's", text)
    prev = None
    while prev != text:
        prev = text
        text = _STUTTER_RE.sub(r"\1", text)
    text = _strip_phones(text)
    text = normalize_numbers(text)
    text = _normalize_fractions(text)
    text = _normalize_temperatures(text)
    text = _normalize_percentages(text)
    text = _normalize_measurements(text)
    text = _normalize_urls(text)
    text = _normalize_phrases(text)
    text = _normalize_punctuation(text)
    return re.sub(r" {2,}", " ", text).strip()


def _build_filler_matcher(nlp: spacy.Language) -> Matcher:
    from spacy.matcher import Matcher

    m = Matcher(nlp.vocab)
    # Longer patterns registered first so overlap resolution prefers them
    m.add(
        "YOU_KNOW_WHAT_I_MEAN",
        [
            [
                {"LOWER": "you"},
                {"LOWER": "know"},
                {"LOWER": "what"},
                {"LOWER": "i"},
                {"LOWER": "mean"},
            ]
        ],
    )
    m.add(
        "LET_ME_JUST_TELL_YOU",
        [
            [
                {"LOWER": "let"},
                {"LOWER": "me"},
                {"LOWER": "just"},
                {"LOWER": "tell"},
                {"LOWER": "you"},
            ]
        ],
    )
    m.add(
        "OR_SOMETHING_LIKE_THAT",
        [
            [
                {"LOWER": "or"},
                {"LOWER": "something"},
                {"LOWER": "like"},
                {"LOWER": "that"},
            ]
        ],
    )
    m.add(
        "DOES_THAT_MAKE_SENSE",
        [
            [
                {"LOWER": "does"},
                {"LOWER": "that"},
                {"LOWER": "make"},
                {"LOWER": "sense"},
            ]
        ],
    )
    m.add("BY_THE_WAY", [[{"LOWER": "by"}, {"LOWER": "the"}, {"LOWER": "way"}]])
    m.add("LET_ME_JUST_SAY", [[{"LOWER": "let"}, {"LOWER": "me"}, {"LOWER": "just"}, {"LOWER": "say"}]])
    m.add("YOU_KNOW", [[{"LOWER": "you"}, {"LOWER": "know"}]])
    m.add("YOU_SEE", [[{"LOWER": "you"}, {"LOWER": "see"}]])
    m.add("I_MEAN", [[{"LOWER": "i"}, {"LOWER": "mean"}]])
    m.add("I_THINK", [[{"LOWER": "i"}, {"LOWER": "think"}]])
    m.add("I_GUESS", [[{"LOWER": "i"}, {"LOWER": "guess"}]])
    m.add("SORT_OF", [[{"LOWER": "sort"}, {"LOWER": "of"}]])
    m.add("KIND_OF", [[{"LOWER": "kind"}, {"LOWER": "of"}]])
    m.add("OR_SOMETHING", [[{"LOWER": "or"}, {"LOWER": "something"}]])
    return m


def _clean_filler(doc: Doc, matcher: Matcher) -> str:
    """Step 1: remove matched filler phrases and POS-identified discourse tokens."""
    matches = matcher(doc)

    # Prefer longer spans when resolving overlaps
    matches_sorted = sorted(matches, key=lambda m: -(m[2] - m[1]))
    remove: set[int] = set()
    covered: set[int] = set()
    for match_id, start, end in matches_sorted:
        span = set(range(start, end))
        if span & covered:
            continue
        label: str = doc.vocab.strings[match_id]
        # "do/did/does/if/whether you know/see" → semantic question, not backchannel
        if label in {"YOU_KNOW", "YOU_SEE"} and start > 0 and doc[start - 1].lower_ in _SEMANTIC_PRECEDES:
            continue
        # "I mean/think/guess it/this/that/..." → semantic predicate
        if label in {"I_MEAN", "I_THINK", "I_GUESS"} and end < len(doc) and doc[end].lower_ in _SEMANTIC_FOLLOWS:
            continue
        # "I think about/of/so" → deliberate mental action
        if label == "I_THINK" and end < len(doc) and doc[end].lower_ in {"about", "of", "so"}:
            continue
        remove |= span
        covered |= span

    # POS-based: discourse "like" (INTJ), sentence-opening interjections, parenthetical adverbs
    for token in doc:
        if token.i in remove:
            continue
        if (
            (token.lower_ == "like" and token.pos_ == "INTJ")
            or (token.lower_ in {"well", "okay", "ok", "alright", "anyway"} and token.pos_ == "INTJ")
            or (
                token.lower_ in {"basically", "essentially", "literally", "actually"}
                and token.dep_
                in {
                    "advmod",
                    "discourse",
                }
            )
        ):
            remove.add(token.i)

    result = "".join(t.text_with_ws for i, t in enumerate(doc) if i not in remove).strip()
    result = re.sub(r" {2,}", " ", result)
    result = re.sub(r" ([.,!?;:])", r"\1", result)
    # Strip leading punctuation left behind by filler removal
    result = re.sub(r"^[,;:\s]+", "", result)
    # Catch stutters created by filler removal (pre-pass already ran before NLP)
    result = _ITS_STUTTER_RE.sub("it's", result)
    result = _ITS_IT_RE.sub("it's", result)
    prev = None
    while prev != result:
        prev = result
        result = _STUTTER_RE.sub(r"\1", result)
    if result and result[0].islower() and doc.text[0].isupper():
        result = result[0].upper() + result[1:]
    return result


def _has_discourse_marker(doc: Doc) -> bool:
    for token in doc:
        if token.dep_ == "neg":
            return True
        if token.lower_ in _CONTRAST_WORDS or token.lower_ in _CAUSAL_WORDS:
            return True
    return False


def _should_keep(doc: Doc, cleaned: str) -> bool:
    """Step 2: drop short sentences with no predicate; exempt those with discourse markers or NEs."""
    words = [w for w in cleaned.split() if re.search(r"[a-zA-Z0-9]", w)]
    if len(words) >= _MIN_TURN_TOKENS:
        return True
    if not any(t.pos_ in {"NOUN", "VERB", "PROPN"} for t in doc):
        return False
    return bool(doc.ents) or _has_discourse_marker(doc)


def _deduplicate(sentences: list[str], embedder: SentenceTransformer, threshold: float) -> list[int]:
    """Step 5: cluster by cosine similarity, return indices of sentences to keep (longest per cluster)."""
    n = len(sentences)
    if n < 2:
        return list(range(n))

    vecs: np.ndarray = embedder.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
    sim: np.ndarray = vecs @ vecs.T

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                rx, ry = find(i), find(j)
                if rx != ry:
                    parent[rx] = ry

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    keep: set[int] = {max(members, key=lambda idx: len(sentences[idx].split())) for members in clusters.values()}
    return sorted(keep)


def denoise2(
    text: str,
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    nlp: spacy.Language | None = None,
    embedder: SentenceTransformer | None = None,
) -> str:
    """Full pipeline: boilerplate gating → normalization → NLP cleanup → embedding deduplication."""
    if nlp is None:
        import spacy as _spacy

        nlp = _spacy.load("en_core_web_trf")
    if embedder is None:
        from sentence_transformers import SentenceTransformer as _ST

        embedder = _ST("all-mpnet-base-v2")

    paras = _prepass(text)

    # Track (para_idx, sentence) so paragraph structure survives deduplication
    indexed: list[tuple[int, str]] = []
    for para_idx, para in enumerate(paras):
        normalized = _normalize_para(para)
        if not normalized:
            continue
        # NOTE (LLM ASR post-correction): a Haiku pass would fit here before spaCy.
        # Prompt: fix transcription errors only (truncated words like "vacu"→"vacuum",
        # "licen"→"license", "thous"→"thousands"; ASR proper-noun errors like
        # "Frenchulinary"→"French Culinary Institute"; consonant false-starts the regex
        # misses like "impossi to re redo"→"impossible to redo"). Batch all paragraphs
        # into one call. Guard against hallucination with a Levenshtein ratio check: if
        # corrected text diverges >~15% from input, fall back to uncorrected.
        # transcribe.extract is the right scaffold.
        for s in _SENTENCE_RE.split(normalized):
            s = s.strip()
            if s:
                indexed.append((para_idx, s))

    if not indexed:
        return ""

    matcher = _build_filler_matcher(nlp)
    kept: list[tuple[int, str]] = []
    for (para_idx, _), doc in zip(indexed, nlp.pipe(s for _, s in indexed)):
        cleaned = _clean_filler(doc, matcher)
        if cleaned and _should_keep(doc, cleaned):
            kept.append((para_idx, cleaned))

    # Step 4: discourse-aware exemptions are enforced inside _should_keep —
    # negation, contrast, and causal markers prevent short sentences from being dropped.

    if not kept:
        return ""

    # Step 5: deduplicate globally across the transcript, then reconstruct paragraphs
    kept_sentences = [s for _, s in kept]
    keep_indices = _deduplicate(kept_sentences, embedder, similarity_threshold)

    para_groups: dict[int, list[str]] = {}
    for i in keep_indices:
        para_idx, sentence = kept[i]
        para_groups.setdefault(para_idx, []).append(sentence)

    return "\n\n".join(" ".join(para_groups[p]) for p in sorted(para_groups))
