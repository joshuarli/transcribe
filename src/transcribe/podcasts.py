"""Podcast registry — add new shows here."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Podcast:
    slug: str
    name: str
    feed_urls: list[str]

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
    ),
}

DEFAULT_PODCAST = "cooking-issues"
