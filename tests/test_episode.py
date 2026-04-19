from transcribe.episode import make_episode, slugify


def test_slugify_basic():
    assert slugify("Episode 42: Raw Vegan Diet") == "episode-42-raw-vegan-diet"


def test_slugify_collapses_punctuation():
    assert slugify("Dave & Friends!!!") == "dave-friends"


def test_slugify_strips_trailing_dash():
    assert slugify("Hello---") == "hello"


def test_slugify_lowercases():
    assert slugify("ALL CAPS") == "all-caps"


def test_make_episode_number(tmp_path):
    ep = make_episode(
        0,
        {"title": "Episode 1: Pilot", "audio_url": "https://x.com/ep.mp3"},
        audio_dir=tmp_path / "audio",
        transcript_dir=tmp_path / "transcripts",
        text_dir=tmp_path / "text",
    )
    assert ep["number"] == 1


def test_make_episode_slug(tmp_path):
    ep = make_episode(
        41,
        {"title": "Episode 42: Raw Vegan Diet", "audio_url": "https://x.com/ep.mp3"},
        audio_dir=tmp_path / "audio",
        transcript_dir=tmp_path / "transcripts",
        text_dir=tmp_path / "text",
    )
    assert ep["slug"] == "042-episode-42-raw-vegan-diet"


def test_make_episode_paths(tmp_path):
    ep = make_episode(
        0,
        {"title": "Episode 1: Pilot", "audio_url": "https://x.com/ep.mp3"},
        audio_dir=tmp_path / "audio",
        transcript_dir=tmp_path / "transcripts",
        text_dir=tmp_path / "text",
    )
    assert ep["audio"] == tmp_path / "audio" / "001-episode-1-pilot.mp3"
    assert ep["transcript"] == tmp_path / "transcripts" / "001-episode-1-pilot.json"
    assert ep["text"] == tmp_path / "text" / "001-episode-1-pilot.txt"
