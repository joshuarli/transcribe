"""Speaker diarization with pyannote."""

from collections.abc import Callable
from typing import Any, cast

from transcribe.types import RawSegment, Segment


def diarize(audio_path: str, hf_token: str) -> tuple[Any, Any, int, Any]:
    """Returns (annotation, waveform, sample_rate, pipeline)."""
    import soundfile as sf
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    )
    assert pipeline is not None, "Failed to load pyannote pipeline"
    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))

    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)  # (channels, samples)
    result = pipeline({"waveform": waveform, "sample_rate": sr}, hook=_diarize_hook())
    print()  # end the final \r line
    annotation = result.exclusive_speaker_diarization if hasattr(result, "exclusive_speaker_diarization") else result
    return annotation, waveform, sr, pipeline


def _diarize_hook() -> Callable[..., None]:
    """Returns a pyannote-compatible progress hook.

    pyannote calls the hook in two patterns:
      Inference.slide (segmentation chunks): hook(completed=n, total=m)  — no step name
      Embeddings batches:  hook("embeddings", batch, total=m, completed=n, file=...)
      Named completions:   hook("speaker_counting"|"discrete_diarization", ..., file=...)
    """
    _phase: list[str] = ["segmentation"]
    _last: list[int] = [-1]

    def hook(
        *args: Any,  # noqa: ANN401
        file: object = None,
        completed: int | None = None,
        total: int | None = None,
        **_: object,
    ) -> None:
        step = next((a for a in args if isinstance(a, str)), None)

        # Named-completion notifications (no progress bar needed, just a label)
        if step and step not in ("segmentation", "embeddings"):
            print(f"\r  {step}...", end="", flush=True)
            return

        if completed is None or total is None:
            return
        if step:
            _phase[0] = step
        if completed == _last[0]:
            return
        _last[0] = completed

        w = 30
        filled = round(w * completed / total) if total else 0
        bar = "█" * filled + "░" * (w - filled)
        print(f"\r  {_phase[0]}: [{bar}] {completed}/{total} ", end="", flush=True)
        if completed >= total:
            print()
            _last[0] = -1

    return hook


def assign_speakers(segments: list[RawSegment], diarization: Any) -> list[Segment]:  # noqa: ANN401
    # Materialise turns once; both segments and turns are time-ordered, so a
    # two-pointer walk is O(n+m) rather than O(n*m).
    turns: list[tuple[float, float, str]] = [
        (t.start, t.end, spk) for t, _, spk in diarization.itertracks(yield_label=True)
    ]
    labeled: list[Segment] = []
    j = 0  # lower bound: turns whose end <= seg.start can never overlap again
    for seg in segments:
        while j < len(turns) and turns[j][1] <= seg["start"]:
            j += 1
        best_speaker = None
        best_overlap = 0.0
        k = j
        while k < len(turns) and turns[k][0] < seg["end"]:
            t_start, t_end, spk = turns[k]
            overlap = min(seg["end"], t_end) - max(seg["start"], t_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = spk
            k += 1
        # cast preserves any extra fields from mlx_whisper (e.g. "words") at runtime
        labeled.append(cast("Segment", {**seg, "speaker": best_speaker or "UNKNOWN"}))
    return labeled


def extract_cluster_embeddings(
    waveform: Any,  # noqa: ANN401
    sample_rate: int,
    diarization: Any,  # noqa: ANN401
    pipeline: Any,  # noqa: ANN401
) -> dict[str, list[float]]:
    """Extract a mean embedding vector per diarization cluster.

    Uses the embedding sub-model already loaded inside the pyannote pipeline,
    so no extra model downloads are needed.
    """
    from pyannote.audio import Inference

    emb_fn = Inference(pipeline.embedding, window="whole")
    cluster_embs: dict[str, list[list[float]]] = {}

    for turn, _, cluster in diarization.itertracks(yield_label=True):
        if turn.duration < 0.5:
            continue
        start = int(turn.start * sample_rate)
        end = int(turn.end * sample_rate)
        crop = waveform[:, start:end]
        try:
            emb = emb_fn({"waveform": crop.unsqueeze(0), "sample_rate": sample_rate})
            cluster_embs.setdefault(cluster, []).append(cast("list[float]", emb.tolist()))  # ty:ignore[unresolved-attribute]
        except Exception:
            continue

    result: dict[str, list[float]] = {}
    for cluster, emb_list in cluster_embs.items():
        n = len(emb_list)
        dim = len(emb_list[0])
        mean = [sum(e[i] for e in emb_list) / n for i in range(dim)]
        result[cluster] = mean
    return result
