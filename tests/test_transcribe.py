import json

import numpy as np
import pytest

from transcribe.transcribe import (
    _apply_silence_breaks,
    _fmt_ts,
    _optimal_chunk_s,
    _run_chunks,
    _silence_split_points,
    _silence_windows,
    _text_to_segments,
    assign_speakers,
)

SR = 16_000  # samples/second used throughout silence tests


def _audio(spec: list[tuple[float, str]], sr: int = SR) -> np.ndarray:
    """Build a float32 waveform from a list of (duration_s, 'speech'|'silence') pairs."""
    rng = np.random.default_rng(0)
    parts = []
    for dur, kind in spec:
        n = int(dur * sr)
        if kind == "speech":
            parts.append(rng.standard_normal(n).astype(np.float32) * 0.3)
        else:
            parts.append(np.full(n, 1e-7, dtype=np.float32))
    return np.concatenate(parts)


def test_silence_splits_land_in_silence():
    # speech 2 s | silence 0.5 s | speech 2 s | silence 0.5 s | speech 2 s  → 7.5 s
    data = _audio(
        [
            (2, "speech"),
            (0.5, "silence"),
            (2, "speech"),
            (0.5, "silence"),
            (2, "speech"),
        ]
    )
    splits = _silence_split_points(data, SR, target_s=2.5)

    assert len(splits) == 2
    # Each split sample must fall within a known silence window (±0.3 s tolerance)
    silence_windows = [(2.0, 2.5), (4.5, 5.0)]
    for sample, (t0, t1) in zip(splits, silence_windows, strict=True):
        t = sample / SR
        assert t0 - 0.3 <= t <= t1 + 0.3, f"split at {t:.2f}s is outside silence {t0}-{t1}s"


def test_silence_splits_never_exceed_audio_length():
    data = _audio([(2, "speech"), (0.5, "silence"), (2, "speech")])
    splits = _silence_split_points(data, SR, target_s=2.5)
    for s in splits:
        assert 0 < s < len(data)


def test_silence_splits_short_audio_returns_empty():
    # Audio shorter than one target chunk → no splits needed
    data = _audio([(1, "speech")])
    splits = _silence_split_points(data, SR, target_s=60.0)
    assert splits == []


def test_silence_splits_no_silence_falls_back_to_target():
    # Continuous speech: no silence regions to snap to, splits at raw target boundary
    data = _audio([(4, "speech")])
    splits = _silence_split_points(data, SR, target_s=2.0)
    assert len(splits) == 1
    t = splits[0] / SR
    # Falls back to target position (±1 frame = 20 ms)
    assert abs(t - 2.0) < 0.02


def test_silence_splits_multiple_chunks():
    # 10 s of speech with three 0.5 s silences → should produce 3 splits for target=2.5 s
    data = _audio(
        [
            (2, "speech"),
            (0.5, "silence"),
            (2, "speech"),
            (0.5, "silence"),
            (2, "speech"),
            (0.5, "silence"),
            (2, "speech"),
        ]
    )
    splits = _silence_split_points(data, SR, target_s=2.5)
    assert len(splits) == 3


class _Turn:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _Diarization:
    def __init__(self, turns):
        self._turns = turns  # [(start, end, speaker), ...]

    def itertracks(self, yield_label=False):
        for start, end, speaker in self._turns:
            yield _Turn(start, end), None, speaker


def test_assign_speakers_basic():
    segments = [{"start": 0.0, "end": 2.0, "text": " Hello"}]
    diarization = _Diarization([(0.0, 2.0, "SPEAKER_00")])
    result = assign_speakers(segments, diarization)
    assert result[0]["speaker"] == "SPEAKER_00"


def test_assign_speakers_unknown_when_no_overlap():
    segments = [{"start": 5.0, "end": 6.0, "text": " Hi"}]
    diarization = _Diarization([(0.0, 2.0, "SPEAKER_00")])
    result = assign_speakers(segments, diarization)
    assert result[0]["speaker"] == "UNKNOWN"


def test_assign_speakers_picks_max_overlap():
    segments = [{"start": 0.0, "end": 4.0, "text": " Mixed"}]
    diarization = _Diarization(
        [
            (0.0, 1.0, "SPEAKER_00"),  # 1s overlap
            (1.0, 4.0, "SPEAKER_01"),  # 3s overlap — should win
        ]
    )
    result = assign_speakers(segments, diarization)
    assert result[0]["speaker"] == "SPEAKER_01"


def test_assign_speakers_preserves_other_fields():
    segments = [{"start": 0.0, "end": 1.0, "text": " Hi", "words": []}]
    diarization = _Diarization([(0.0, 1.0, "SPEAKER_00")])
    result = assign_speakers(segments, diarization)
    assert result[0]["words"] == []
    assert result[0]["text"] == " Hi"


def test_assign_speakers_multiple_segments():
    segments = [
        {"start": 0.0, "end": 2.0, "text": " Hello"},
        {"start": 3.0, "end": 5.0, "text": " World"},
    ]
    diarization = _Diarization(
        [
            (0.0, 2.0, "SPEAKER_00"),
            (3.0, 5.0, "SPEAKER_01"),
        ]
    )
    result = assign_speakers(segments, diarization)
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[1]["speaker"] == "SPEAKER_01"


def test_assign_speakers_two_pointer_skips_expired_turns():
    # Turn at [0,1] should not be reconsidered for the second segment at [5,6].
    segments = [
        {"start": 0.0, "end": 1.0, "text": " A"},
        {"start": 5.0, "end": 6.0, "text": " B"},
    ]
    diarization = _Diarization(
        [
            (0.0, 1.0, "SPEAKER_00"),
            (5.0, 6.0, "SPEAKER_01"),
        ]
    )
    result = assign_speakers(segments, diarization)
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[1]["speaker"] == "SPEAKER_01"


def test_assign_speakers_many_segments_correct_order():
    n = 100
    segments = [{"start": float(i * 2), "end": float(i * 2 + 1), "text": f" {i}"} for i in range(n)]
    turns = [(float(i * 2), float(i * 2 + 1), f"SPEAKER_{i:02d}") for i in range(n)]
    result = assign_speakers(segments, _Diarization(turns))
    for i, seg in enumerate(result):
        assert seg["speaker"] == f"SPEAKER_{i:02d}"


# _text_to_segments


def test_text_to_segments_empty_returns_empty():
    assert _text_to_segments("", 0.0, 10.0) == []


def test_text_to_segments_whitespace_only_returns_empty():
    assert _text_to_segments("   ", 0.0, 10.0) == []


def test_text_to_segments_single_sentence():
    segs = _text_to_segments("Hello world.", 2.0, 5.0)
    assert len(segs) == 1
    assert segs[0]["start"] == 2.0
    assert segs[0]["end"] == 5.0
    assert segs[0]["text"] == " Hello world."


def test_text_to_segments_multiple_sentences_count():
    text = "First sentence. Second sentence. Third sentence."
    segs = _text_to_segments(text, 0.0, 9.0)
    assert len(segs) == 3


def test_text_to_segments_timestamps_are_contiguous():
    text = "Hello world. How are you? Fine thanks."
    segs = _text_to_segments(text, 0.0, 10.0)
    for i in range(1, len(segs)):
        assert segs[i]["start"] == segs[i - 1]["end"]


def test_text_to_segments_spans_full_range():
    text = "Hello world. How are you? Fine thanks."
    segs = _text_to_segments(text, 1.5, 8.5)
    assert segs[0]["start"] == 1.5
    assert segs[-1]["end"] == pytest.approx(8.5, abs=0.01)


def test_text_to_segments_proportional_to_length():
    # "AB." is 3 chars, "ABCDEFGHIJ." is 11 chars — second should get ~11/14 of duration
    text = "AB. ABCDEFGHIJ."
    segs = _text_to_segments(text, 0.0, 14.0)
    assert len(segs) == 2
    assert segs[1]["end"] - segs[1]["start"] > segs[0]["end"] - segs[0]["start"]


def test_text_to_segments_text_has_leading_space():
    segs = _text_to_segments("Hello. World.", 0.0, 1.0)
    for seg in segs:
        assert seg["text"].startswith(" ")


def test_text_to_segments_no_split_on_lowercase_after_period():
    # "e.g. something" — lowercase after period should NOT split
    segs = _text_to_segments("Use e.g. this method. Done.", 0.0, 10.0)
    # Only the capital-letter boundary triggers a split
    assert len(segs) == 2


# _silence_windows


def test_silence_windows_detects_silence():
    data = _audio([(1, "speech"), (0.6, "silence"), (1, "speech")])
    windows = _silence_windows(data, SR, min_silence_s=0.5)
    assert len(windows) == 1
    t0, t1 = windows[0]
    # Silence starts around 1.0s and ends around 1.6s
    assert 0.8 <= t0 <= 1.2
    assert 1.4 <= t1 <= 1.8


def test_silence_windows_below_threshold_not_returned():
    # 0.3s silence is below 0.5s threshold
    data = _audio([(1, "speech"), (0.3, "silence"), (1, "speech")])
    windows = _silence_windows(data, SR, min_silence_s=0.5)
    assert windows == []


def test_silence_windows_multiple():
    data = _audio([(1, "speech"), (0.6, "silence"), (1, "speech"), (0.6, "silence"), (1, "speech")])
    windows = _silence_windows(data, SR, min_silence_s=0.5)
    assert len(windows) == 2


def test_silence_windows_continuous_speech_returns_empty():
    data = _audio([(5, "speech")])
    windows = _silence_windows(data, SR, min_silence_s=0.5)
    assert windows == []


# _apply_silence_breaks


def test_apply_silence_breaks_inserts_gap():
    # Two segments, synthetic timestamps covering 0-10s; silence at 4-5s
    segs = [
        {"start": 0.0, "end": 5.0, "text": " Hello world."},
        {"start": 5.0, "end": 10.0, "text": " How are you?"},
    ]
    result = _apply_silence_breaks(segs, [(4.0, 5.0)])
    assert result[0]["end"] == 4.0
    assert result[1]["start"] == 5.0
    # Gap is now 1.0s — render will break here
    assert result[1]["start"] - result[0]["end"] == pytest.approx(1.0)


def test_apply_silence_breaks_picks_nearest_boundary():
    # Three segments; silence at 8s — should patch boundary between seg[1] and seg[2]
    segs = [
        {"start": 0.0, "end": 3.0, "text": " A"},
        {"start": 3.0, "end": 7.0, "text": " B"},
        {"start": 7.0, "end": 10.0, "text": " C"},
    ]
    result = _apply_silence_breaks(segs, [(7.8, 8.2)])
    # Boundary between seg[1] (end=7.0) and seg[2] (start=7.0) is nearest to mid=8.0
    assert result[1]["end"] == 7.8
    assert result[2]["start"] == 8.2


def test_apply_silence_breaks_empty_inputs():
    assert _apply_silence_breaks([], [(1.0, 2.0)]) == []
    segs = [{"start": 0.0, "end": 5.0, "text": " Hi"}]
    assert _apply_silence_breaks(segs, []) == segs


def test_apply_silence_breaks_does_not_mutate_input():
    segs = [
        {"start": 0.0, "end": 5.0, "text": " A"},
        {"start": 5.0, "end": 10.0, "text": " B"},
    ]
    original_end = segs[0]["end"]
    _apply_silence_breaks(segs, [(4.0, 5.0)])
    assert segs[0]["end"] == original_end


# _fmt_ts


def test_fmt_ts_seconds_only():
    assert _fmt_ts(90.0) == "1:30"


def test_fmt_ts_with_hours():
    assert _fmt_ts(3661.0) == "1:01:01"


def test_fmt_ts_zero():
    assert _fmt_ts(0.0) == "0:00"


def test_fmt_ts_exactly_one_minute():
    assert _fmt_ts(60.0) == "1:00"


# _optimal_chunk_s


def test_optimal_chunk_s_returns_int():
    result = _optimal_chunk_s()
    assert isinstance(result, int)


def test_optimal_chunk_s_within_bounds():
    result = _optimal_chunk_s()
    assert 30 <= result <= 600


# _run_chunks — checkpoint / resume


def test_run_chunks_basic(tmp_path):
    sr = 16_000
    chunks = [(0, sr), (sr, 2 * sr)]
    calls = []

    def infer(i: int, s0: int, s1: int):
        calls.append(i)
        return f"chunk{i}", [{"start": s0 / sr, "end": s1 / sr, "text": f" chunk{i}"}]

    result = _run_chunks(chunks, sr, infer, None, "test")
    assert calls == [0, 1]
    assert "chunk0" in result["text"]
    assert "chunk1" in result["text"]
    assert len(result["segments"]) == 2


def test_run_chunks_writes_checkpoint(tmp_path):
    sr = 16_000
    ckpt = tmp_path / "ckpt.json"
    chunks = [(0, sr), (sr, 2 * sr)]

    def infer(i: int, s0: int, s1: int):
        return f"t{i}", [{"start": s0 / sr, "end": s1 / sr, "text": f" t{i}"}]

    _run_chunks(chunks, sr, infer, ckpt, "test")
    # Checkpoint should be deleted after successful completion
    assert not ckpt.exists()


def test_run_chunks_resumes_from_checkpoint(tmp_path):
    sr = 16_000
    ckpt = tmp_path / "ckpt.json"
    chunks = [(0, sr), (sr, 2 * sr), (2 * sr, 3 * sr)]

    # Pre-populate checkpoint with chunk 0 done
    ckpt.write_text(
        json.dumps({"i": 0, "text": "already done", "segments": [{"start": 0.0, "end": 1.0, "text": " already done"}]})
        + "\n"
    )

    called = []

    def infer(i: int, s0: int, s1: int):
        called.append(i)
        return f"fresh{i}", [{"start": s0 / sr, "end": s1 / sr, "text": f" fresh{i}"}]

    result = _run_chunks(chunks, sr, infer, ckpt, "test")
    assert 0 not in called  # chunk 0 was already done
    assert 1 in called
    assert 2 in called
    assert "already done" in result["text"]
    assert "fresh1" in result["text"]


def test_run_chunks_keyboard_interrupt_preserves_checkpoint(tmp_path):
    sr = 16_000
    ckpt = tmp_path / "ckpt.json"
    chunks = [(0, sr), (sr, 2 * sr), (2 * sr, 3 * sr)]
    calls = []

    def infer(i: int, s0: int, s1: int):
        calls.append(i)
        if i == 1:
            raise KeyboardInterrupt
        return f"t{i}", [{"start": s0 / sr, "end": s1 / sr, "text": f" t{i}"}]

    with pytest.raises(KeyboardInterrupt):
        _run_chunks(chunks, sr, infer, ckpt, "test")

    # Checkpoint from chunk 0 should still be on disk
    assert ckpt.exists()
    lines = [json.loads(line) for line in ckpt.read_text().splitlines()]
    assert lines[0]["i"] == 0
