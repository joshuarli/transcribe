import time

import pytest
import urllib3.exceptions
from conftest import SAMPLE_RSS, mock_fetch_sequence

from transcribe.feed import _is_fresh, fetch_feeds, parse_feed
from transcribe.podcasts import Podcast

TEST_PODCAST = Podcast(
    slug="test-show",
    name="Test Show",
    feed_urls=["https://example.com/feed0.xml", "https://example.com/feed1.xml"],
)


def test_parse_feed_order(sample_rss_file):
    # RSS lists newest-first; parse_feed reverses to chronological order
    eps = parse_feed(sample_rss_file)
    assert len(eps) == 2
    assert eps[0]["title"] == "Episode 2: Follow-up"
    assert eps[1]["title"] == "Episode 1: Pilot"


def test_parse_feed_urls(sample_rss_file):
    eps = parse_feed(sample_rss_file)
    assert eps[0]["audio_url"] == "https://example.com/ep2.mp3"
    assert eps[1]["audio_url"] == "https://example.com/ep1.mp3"


def test_parse_feed_skips_missing_enclosure(tmp_path):
    xml = """\
<rss><channel>
  <item><title>No audio</title></item>
  <item><title>Has audio</title><enclosure url="https://x.com/ep.mp3"/></item>
</channel></rss>"""
    feed = tmp_path / "feed.xml"
    feed.write_text(xml)
    eps = parse_feed(feed)
    assert len(eps) == 1
    assert eps[0]["title"] == "Has audio"


def test_is_fresh_missing(tmp_path):
    assert not _is_fresh(tmp_path / "nonexistent.xml")


def test_is_fresh_new_file(tmp_path):
    p = tmp_path / "feed.xml"
    p.write_text("x")
    assert _is_fresh(p, ttl=3600)


def test_is_fresh_expired(tmp_path):
    p = tmp_path / "feed.xml"
    p.write_text("x")
    past = time.time() - 7200
    import os

    os.utime(p, (past, past))
    assert not _is_fresh(p, ttl=3600)


def test_fetch_feeds_skips_fresh(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = TEST_PODCAST.cache_dir
    cache.mkdir(parents=True)
    (cache / "feed_0.xml").write_text(SAMPLE_RSS)
    (cache / "feed_1.xml").write_text(SAMPLE_RSS)

    called = []

    def fake_fetch(url, *, headers=None, retries=3):
        called.append(url)
        raise AssertionError("should not fetch fresh feeds")

    monkeypatch.setattr("transcribe.feed.fetch", fake_fetch)
    fetch_feeds(TEST_PODCAST)
    assert called == []


def test_fetch_feeds_writes_feeds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    body = SAMPLE_RSS.encode()
    monkeypatch.setattr(
        "transcribe.feed.fetch",
        mock_fetch_sequence((200, body, {}), (200, body, {})),
    )
    fetch_feeds(TEST_PODCAST)
    assert (TEST_PODCAST.cache_dir / "feed_0.xml").read_bytes() == body
    assert (TEST_PODCAST.cache_dir / "feed_1.xml").read_bytes() == body


def test_fetch_feeds_writes_etag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    body = SAMPLE_RSS.encode()
    monkeypatch.setattr(
        "transcribe.feed.fetch",
        mock_fetch_sequence((200, body, {}), (200, body, {"ETag": '"abc"'})),
    )
    fetch_feeds(TEST_PODCAST)
    assert (TEST_PODCAST.cache_dir / "feed_1.etag").read_text() == '"abc"'


def test_fetch_feeds_304_uses_etag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    body = SAMPLE_RSS.encode()
    cache = TEST_PODCAST.cache_dir
    cache.mkdir(parents=True)
    (cache / "feed_1.etag").write_text('"old"')
    monkeypatch.setattr(
        "transcribe.feed.fetch",
        mock_fetch_sequence((200, body, {}), (304, b"", {})),
    )
    fetch_feeds(TEST_PODCAST)


def test_fetch_feeds_error_reraises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    body = SAMPLE_RSS.encode()
    monkeypatch.setattr(
        "transcribe.feed.fetch",
        mock_fetch_sequence((200, body, {}), urllib3.exceptions.HTTPError("HTTP 500")),
    )
    with pytest.raises(urllib3.exceptions.HTTPError, match="500"):
        fetch_feeds(TEST_PODCAST)
