"""Podcast registry — add new shows here."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Podcast:
    slug: str
    name: str
    feed_urls: list[str]
    extraction_prompt: str = ""

    @property
    def cache_dir(self) -> Path:
        return Path("cache") / self.slug

    @property
    def audio_dir(self) -> Path:
        return self.cache_dir / "audio"

    @property
    def speakers_path(self) -> Path:
        return self.cache_dir / "speakers.json"


PODCASTS: dict[str, Podcast] = {
    "cooking-issues": Podcast(
        slug="cooking-issues",
        name="Cooking Issues",
        feed_urls=[
            "https://rss.art19.com/cooking-issues",
            "https://feeds.acast.com/public/shows/cooking-issues-with-dave-arnold",
        ],
        extraction_prompt="""\
You are extracting structured notes from a Cooking Issues podcast transcript. \
Cooking Issues is hosted by Dave Arnold, a food scientist at the French Culinary \
Institute, who discusses modern cooking techniques and answers listener questions.

Extract all substantive culinary information. Include:
- Techniques with specific parameters (temperatures, times, ratios, equipment settings)
- Equipment discussed: names, models, tradeoffs, recommendations
- Ingredients and their functional roles or properties
- Personal anecdotes and demonstrations that contain culinary technique information \
(e.g. a story about cooking a specific dish that reveals technique)
- Caller Q&A: only for questions that actually occur in the transcript — restate \
each question briefly, then capture the full answer
- Strong opinions or recommendations Dave offers

Preserve step sequences and measurements exactly as described. Do not convert units \
between measurement systems. When a specific value in the transcript appears garbled \
or nonsensical, omit it rather than guessing — do not include internal reasoning or \
caveats in the output.

Omit: banter, pleasantries, sponsor mentions, show logistics, music, \
contact info, phone numbers, websites, caller names, anything unrelated to food or cooking.

Format as structured markdown grouped by topic. Be terse — preserve precision over prose.\
""",
    ),
}

DEFAULT_PODCAST = "cooking-issues"
