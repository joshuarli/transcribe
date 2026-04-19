import numpy as np

from transcribe.transcribe import _silence_split_points, assign_speakers

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
