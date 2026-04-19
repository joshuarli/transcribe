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
- Caller Q&A: restate each question briefly, then capture the full answer
- Strong opinions or recommendations Dave offers

Omit: banter, pleasantries, sponsor mentions, show logistics, music, \
contact info, phone numbers, websites, caller names, anything unrelated to food or cooking.

Format as structured markdown grouped by topic. Be terse — preserve precision over prose.\
""",
    ),
}

DEFAULT_PODCAST = "cooking-issues"
