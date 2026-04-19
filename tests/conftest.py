import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest_plugins = ["pytest_parallel"]

SAMPLE_RSS = """\
<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Episode 1: Pilot</title>
      <enclosure url="https://example.com/ep1.mp3" type="audio/mpeg"/>
    </item>
    <item>
      <title>Episode 2: Follow-up</title>
      <enclosure url="https://example.com/ep2.mp3" type="audio/mpeg"/>
    </item>
  </channel>
</rss>"""

SAMPLE_TRANSCRIPT = {
    "text": "Hello world.",
    "language": "en",
    "segments": [
        {"start": 0.0, "end": 1.0, "text": " Hello", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "text": " world.", "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 4.0, "text": " Goodbye.", "speaker": "SPEAKER_01"},
    ],
}


@pytest.fixture
def sample_rss_file(tmp_path: Path) -> Path:
    p = tmp_path / "feed.xml"
    p.write_text(SAMPLE_RSS)
    return p


@pytest.fixture
def sample_transcript_file(tmp_path: Path) -> Path:
    p = tmp_path / "transcript.json"
    p.write_text(json.dumps(SAMPLE_TRANSCRIPT))
    return p


@pytest.fixture
def ep(tmp_path: Path) -> dict[str, Any]:
    """A minimal episode dict with paths rooted in tmp_path."""
    return {
        "number": 1,
        "slug": "001-episode-1-pilot",
        "title": "Episode 1: Pilot",
        "audio_url": "https://example.com/ep1.mp3",
        "audio": tmp_path / "audio" / "001-episode-1-pilot.mp3",
        "transcript": tmp_path / "transcripts" / "001-episode-1-pilot.json",
        "text": tmp_path / "text" / "001-episode-1-pilot.txt",
        "diarized_text": tmp_path / "text" / "001-episode-1-pilot.diarized.txt",
    }


def mock_fetch_sequence(*responses: Any) -> Callable[..., Any]:
    """
    Returns a fake fetch() callable that yields responses in order.
    Each response is (status, bytes, headers_dict) for success, or an exception to raise.
    """
    it = iter(responses)

    def fake_fetch(url: str, *, headers: dict[str, str] | None = None, retries: int = 3) -> Any:
        resp = next(it)
        if isinstance(resp, BaseException):
            raise resp
        return resp  # caller passes (status, body, headers_dict)

    return fake_fetch
