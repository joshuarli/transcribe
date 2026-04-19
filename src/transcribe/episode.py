"""Episode data model."""

import re
from pathlib import Path

from transcribe.types import Episode, FeedEpisode


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).rstrip("-")


def make_episode(
    index: int,
    ep: FeedEpisode,
    *,
    audio_dir: Path,
    transcript_dir: Path,
    text_dir: Path,
) -> Episode:
    n = index + 1
    slug = f"{n:03d}-{slugify(ep['title'])}"
    return {
        "number": n,
        "slug": slug,
        "title": ep["title"],
        "audio_url": ep["audio_url"],
        "audio": audio_dir / f"{slug}.mp3",
        "transcript": transcript_dir / f"{slug}.json",
        "text": text_dir / f"{slug}.txt",
        "diarized_text": text_dir / f"{slug}.diarized.txt",
    }
