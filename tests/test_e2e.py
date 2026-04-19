import struct
import wave

import pytest

from transcribe.transcribe import transcribe


def _write_silence(path, duration_s=1, sample_rate=16000):
    n = duration_s * sample_rate
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(struct.pack(f"<{n}h", *([0] * n)))


@pytest.mark.integration
def test_transcribe_returns_valid_structure(tmp_path):
    audio = tmp_path / "silence.wav"
    _write_silence(audio)

    result = transcribe(str(audio))

    assert "segments" in result
    assert "text" in result
    assert isinstance(result["segments"], list)
