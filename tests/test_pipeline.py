import json

from conftest import SAMPLE_TRANSCRIPT

from transcribe.pipeline import (
    _apply_mapping,
    _first_appearance_mapping,
    _flush,
    _fmt,
    download_missing,
    render,
    run_episode,
)


def test_fmt_seconds_only():
    assert _fmt(90) == "1:30"


def test_fmt_with_hours():
    assert _fmt(3661) == "1:01:01"


def test_flush_with_speaker():
    segs = [{"start": 0.0, "end": 1.0, "text": " Hello", "speaker": "SPEAKER_00"}]
    assert _flush(segs) == "[0:00 | SPEAKER_00] Hello"


def test_flush_without_speaker():
    segs = [{"start": 62.0, "end": 63.0, "text": " Hi", "speaker": "UNKNOWN"}]
    assert _flush(segs) == "[1:02] Hi"


def test_flush_joins_multiple_segments():
    segs = [
        {"start": 0.0, "end": 1.0, "text": " Hello", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "text": " world.", "speaker": "SPEAKER_00"},
    ]
    assert _flush(segs) == "[0:00 | SPEAKER_00] Hello world."


def test_first_appearance_mapping():
    segs = [
        {"speaker": "SPEAKER_00"},
        {"speaker": "SPEAKER_01"},
        {"speaker": "SPEAKER_00"},
    ]
    m = _first_appearance_mapping(segs, ["Dave", "Nastassia"])
    assert m == {"SPEAKER_00": "Dave", "SPEAKER_01": "Nastassia"}


def test_first_appearance_mapping_fewer_names():
    segs = [{"speaker": "SPEAKER_00"}, {"speaker": "SPEAKER_01"}]
    m = _first_appearance_mapping(segs, ["Dave"])
    assert m == {"SPEAKER_00": "Dave"}  # SPEAKER_01 not mapped (no name available)


def test_first_appearance_mapping_skips_unknown():
    segs = [{"speaker": "UNKNOWN"}, {"speaker": "SPEAKER_00"}]
    m = _first_appearance_mapping(segs, ["Dave"])
    assert "UNKNOWN" not in m
    assert m["SPEAKER_00"] == "Dave"


def test_apply_mapping():
    segs = [{"start": 0.0, "text": " Hi", "speaker": "SPEAKER_00"}]
    result = _apply_mapping(segs, {"SPEAKER_00": "Dave"})
    assert result[0]["speaker"] == "Dave"


def test_render_writes_paragraphs(ep, sample_transcript_file):
    ep["transcript"].parent.mkdir(parents=True)
    ep["text"].parent.mkdir(parents=True)
    ep["transcript"].write_text(sample_transcript_file.read_text())

    render(ep)

    paragraphs = ep["text"].read_text().split("\n\n")
    # 1s gap between seg[1] (end=2.0) and seg[2] (start=3.0) → two paragraphs
    assert len(paragraphs) == 2


def test_render_paragraph_content(ep, sample_transcript_file):
    ep["transcript"].parent.mkdir(parents=True)
    ep["text"].parent.mkdir(parents=True)
    ep["transcript"].write_text(sample_transcript_file.read_text())

    render(ep)

    text = ep["text"].read_text()
    assert "[0:00 | SPEAKER_00] Hello world." in text
    assert "[0:03 | SPEAKER_01] Goodbye." in text


def test_render_uses_supplied_segments(ep, sample_transcript_file):
    ep["transcript"].parent.mkdir(parents=True)
    ep["text"].parent.mkdir(parents=True)
    ep["transcript"].write_text(sample_transcript_file.read_text())

    segs = [{"start": 0.0, "end": 1.0, "text": " Injected", "speaker": "SPEAKER_00"}]
    render(ep, segs)

    assert "Injected" in ep["text"].read_text()


def test_render_applies_speaker_names(ep, sample_transcript_file):
    ep["transcript"].parent.mkdir(parents=True)
    ep["text"].parent.mkdir(parents=True)
    ep["transcript"].write_text(sample_transcript_file.read_text())

    render(ep, speakers=["Dave", "Nastassia"])

    text = ep["text"].read_text()
    assert "Dave" in text
    assert "Nastassia" in text
    assert "SPEAKER_00" not in text
    assert "SPEAKER_01" not in text


def test_render_uses_stored_speaker_mapping(ep, tmp_path):
    ep["transcript"].parent.mkdir(parents=True)
    ep["text"].parent.mkdir(parents=True)
    transcript_data = {
        **SAMPLE_TRANSCRIPT,
        "speaker_mapping": {"SPEAKER_00": "Dave", "SPEAKER_01": "Nastassia"},
    }
    ep["transcript"].write_text(json.dumps(transcript_data))

    render(ep)

    text = ep["text"].read_text()
    assert "Dave" in text
    assert "Nastassia" in text


def test_render_explicit_speakers_override_stored_mapping(ep, tmp_path):
    ep["transcript"].parent.mkdir(parents=True)
    ep["text"].parent.mkdir(parents=True)
    transcript_data = {
        **SAMPLE_TRANSCRIPT,
        "speaker_mapping": {"SPEAKER_00": "Dave", "SPEAKER_01": "Nastassia"},
    }
    ep["transcript"].write_text(json.dumps(transcript_data))

    render(ep, speakers=["Alice", "Bob"])

    text = ep["text"].read_text()
    assert "Alice" in text
    assert "Dave" not in text


def test_render_custom_gap(ep, sample_transcript_file):
    ep["transcript"].parent.mkdir(parents=True)
    ep["text"].parent.mkdir(parents=True)
    ep["transcript"].write_text(sample_transcript_file.read_text())

    # gap=2.0 — the 1s gap is below threshold → one paragraph
    render(ep, gap=2.0)
    assert ep["text"].read_text().count("\n\n") == 0


def test_render_no_stray_tmp(ep, sample_transcript_file):
    ep["transcript"].parent.mkdir(parents=True)
    ep["text"].parent.mkdir(parents=True)
    ep["transcript"].write_text(sample_transcript_file.read_text())

    render(ep)

    assert not list(ep["text"].parent.glob("*.tmp"))


def test_run_episode_all_cached(ep, capsys):
    ep["audio"].parent.mkdir(parents=True)
    ep["transcript"].parent.mkdir(parents=True)
    ep["text"].parent.mkdir(parents=True)
    ep["audio"].write_bytes(b"audio")
    ep["transcript"].write_text(json.dumps(SAMPLE_TRANSCRIPT))
    ep["text"].write_text("cached text")

    run_episode(ep)

    out = capsys.readouterr().out
    assert "audio cached" in out
    assert "transcript cached" in out
    assert "text cached" in out


def test_run_episode_dry_run(ep, capsys):
    run_episode(ep, dry_run=True)

    out = capsys.readouterr().out
    assert "would download audio" in out
    assert "would transcribe" in out
    assert "would render" in out
    assert not ep["audio"].exists()
    assert not ep["transcript"].exists()
    assert not ep["text"].exists()


def test_run_episode_downloads_if_missing(ep, monkeypatch, capsys):
    ep["transcript"].parent.mkdir(parents=True)
    ep["text"].parent.mkdir(parents=True)
    ep["transcript"].write_text(json.dumps(SAMPLE_TRANSCRIPT))
    ep["text"].write_text("cached text")

    def fake_download(url, dest, cancel=None, *, retries=5):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"audio")

    monkeypatch.setattr("transcribe.pipeline.download", fake_download)
    monkeypatch.setattr("transcribe.pipeline._transcode_to_opus", lambda p: None)
    run_episode(ep)

    assert ep["audio"].exists()
    assert "downloading audio" in capsys.readouterr().out


def test_run_episode_transcribes_if_missing(ep, monkeypatch, capsys):
    ep["audio"].parent.mkdir(parents=True)
    ep["text"].parent.mkdir(parents=True)
    ep["audio"].write_bytes(b"audio")
    ep["text"].write_text("cached text")

    called = []

    def fake_do_transcribe(ep, *, backend, speakers, learn, diarize, speakers_path):
        called.append(True)
        ep["transcript"].parent.mkdir(parents=True, exist_ok=True)
        ep["transcript"].write_text(json.dumps(SAMPLE_TRANSCRIPT))
        return SAMPLE_TRANSCRIPT["segments"], {}

    monkeypatch.setattr("transcribe.pipeline.do_transcribe", fake_do_transcribe)
    run_episode(ep)

    assert called
    assert "transcribing" in capsys.readouterr().out


def test_run_episode_passes_segments_to_render(ep, monkeypatch):
    """Fresh segments from do_transcribe are passed directly to render (no re-read)."""
    ep["audio"].parent.mkdir(parents=True)
    ep["audio"].write_bytes(b"audio")

    fresh = SAMPLE_TRANSCRIPT["segments"]

    def fake_do_transcribe(ep, *, backend, speakers, learn, diarize, speakers_path):
        ep["transcript"].parent.mkdir(parents=True, exist_ok=True)
        ep["transcript"].write_text(json.dumps(SAMPLE_TRANSCRIPT))
        return fresh, {}

    captured = {}

    def fake_render(ep, segments=None, speaker_mapping=None, *, gap, speakers, diarized):
        captured["segments"] = segments
        ep["text"].parent.mkdir(parents=True, exist_ok=True)
        ep["text"].write_text("rendered")

    monkeypatch.setattr("transcribe.pipeline.do_transcribe", fake_do_transcribe)
    monkeypatch.setattr("transcribe.pipeline.render", fake_render)
    run_episode(ep)

    assert captured["segments"] is fresh


def test_run_episode_renders_if_missing(ep, monkeypatch, sample_transcript_file):
    ep["audio"].parent.mkdir(parents=True)
    ep["transcript"].parent.mkdir(parents=True)
    ep["audio"].write_bytes(b"audio")
    ep["transcript"].write_text(sample_transcript_file.read_text())

    run_episode(ep)

    assert ep["text"].exists()
    assert ep["text"].read_text() != ""


def test_run_episode_backend_param(ep, monkeypatch, capsys):
    ep["audio"].parent.mkdir(parents=True)
    ep["text"].parent.mkdir(parents=True)
    ep["audio"].write_bytes(b"audio")
    ep["text"].write_text("cached")

    received = {}

    def fake_do_transcribe(ep, *, backend, speakers, learn, diarize, speakers_path):
        received["backend"] = backend
        ep["transcript"].parent.mkdir(parents=True, exist_ok=True)
        ep["transcript"].write_text(json.dumps(SAMPLE_TRANSCRIPT))
        return SAMPLE_TRANSCRIPT["segments"], {}

    monkeypatch.setattr("transcribe.pipeline.do_transcribe", fake_do_transcribe)
    run_episode(ep, backend="parakeet-tdt-0.6b-v3")

    assert received["backend"] == "parakeet-tdt-0.6b-v3"


def test_run_episode_learn_param(ep, monkeypatch, capsys):
    ep["audio"].parent.mkdir(parents=True)
    ep["text"].parent.mkdir(parents=True)
    ep["audio"].write_bytes(b"audio")
    ep["text"].write_text("cached")

    received = {}

    def fake_do_transcribe(ep, *, backend, speakers, learn, diarize, speakers_path):
        received["learn"] = learn
        received["speakers"] = speakers
        ep["transcript"].parent.mkdir(parents=True, exist_ok=True)
        ep["transcript"].write_text(json.dumps(SAMPLE_TRANSCRIPT))
        return SAMPLE_TRANSCRIPT["segments"], {}

    monkeypatch.setattr("transcribe.pipeline.do_transcribe", fake_do_transcribe)
    run_episode(ep, speakers=["Dave", "Nastassia"], learn=True)

    assert received["learn"] is True
    assert received["speakers"] == ["Dave", "Nastassia"]


def test_download_missing_parallel(ep, monkeypatch, tmp_path):
    ep2 = {
        **ep,
        "slug": "002-ep2",
        "audio_url": "https://example.com/ep2.mp3",
        "audio": tmp_path / "audio" / "002-ep2.mp3",
    }

    downloaded = []

    def fake_download(url, dest, cancel=None, *, retries=5):
        dest.write_bytes(b"audio")
        downloaded.append(url)

    monkeypatch.setattr("transcribe.pipeline.download", fake_download)
    monkeypatch.setattr("transcribe.pipeline._transcode_to_opus", lambda p: None)
    download_missing([ep, ep2])

    assert len(downloaded) == 2
    assert ep["audio"].exists()
    assert ep2["audio"].exists()


def test_download_missing_skips_existing(ep, monkeypatch):
    ep["audio"].parent.mkdir(parents=True)
    ep["audio"].write_bytes(b"existing")

    called = []
    monkeypatch.setattr("transcribe.pipeline.download", lambda *a: called.append(a))
    download_missing([ep])
    assert called == []
