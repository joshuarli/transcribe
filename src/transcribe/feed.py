"""RSS feed fetching and parsing."""

import time
import xml.etree.ElementTree as ET
from pathlib import Path

from transcribe.http import fetch
from transcribe.podcasts import Podcast
from transcribe.types import FeedEpisode

FEED_TTL = 3600  # seconds


def _is_fresh(path: Path, ttl: int = FEED_TTL) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl


def _fetch_feed(url: str, xml_path: Path) -> None:
    etag_path = xml_path.with_suffix(".etag")
    headers = {"If-None-Match": etag_path.read_text().strip()} if etag_path.exists() else {}
    status, body, resp_headers = fetch(url, headers=headers)
    if status == 304:
        xml_path.touch()  # refresh mtime so TTL resets
    else:
        xml_path.write_bytes(body)
        if etag := resp_headers.get("ETag"):
            etag_path.write_text(etag)


def fetch_feeds(podcast: Podcast) -> None:
    podcast.cache_dir.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(podcast.feed_urls):
        xml_path = podcast.cache_dir / f"feed_{i}.xml"
        if not _is_fresh(xml_path):
            _fetch_feed(url, xml_path)


def parse_feed(path: Path) -> list[FeedEpisode]:
    episodes: list[FeedEpisode] = []
    for item in ET.parse(path).findall(".//item"):
        title = item.findtext("title", "")
        enclosure = item.find("enclosure")
        if enclosure is not None:
            url = enclosure.get("url")
            if url:
                episodes.append({"title": title, "audio_url": url})
    return list(reversed(episodes))


def load_episodes(podcast: Podcast) -> list[FeedEpisode]:
    fetch_feeds(podcast)
    episodes: list[FeedEpisode] = []
    for i in range(len(podcast.feed_urls)):
        episodes += parse_feed(podcast.cache_dir / f"feed_{i}.xml")
    return episodes
