"""Episode pipeline: download, transcribe, render."""

import json
import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

from transcribe.denoise import strip_fillers as _strip_fillers
from transcribe.diarize import assign_speakers, diarize, extract_cluster_embeddings
from transcribe.http import download
from transcribe.podcasts import Podcast
from transcribe.speakers import load_embeddings, match_speakers, save_embeddings
from transcribe.transcribe import transcribe
from transcribe.types import (
    AnnotationTurn,
    Episode,
    Segment,
    StoredTranscript,
    TranscriptResult,
)

PARAGRAPH_GAP_S = 0.5


def dirs_for_backend(podcast: Podcast, backend: str) -> tuple[Path, Path]:
    """Returns (transcript_dir, text_dir) for the given podcast and backend."""
    return podcast.cache_dir / backend, Path("transcripts") / podcast.slug / backend


def audio_path(ep: Episode) -> Path:
    """Return the opus file if it exists, otherwise the original audio path."""
    opus = ep["audio"].with_suffix(".opus")
    return opus if opus.exists() else ep["audio"]


def download_missing(episodes: list[Episode]) -> None:
    """Download all missing audio files in parallel (up to 4 at a time)."""
    missing = []
    for ep in episodes:
        if not audio_path(ep).exists():
            ep["audio"].parent.mkdir(parents=True, exist_ok=True)
            missing.append(ep)
    if not missing:
        return

    cancel = threading.Event()
    prev_handler = signal.signal(signal.SIGINT, lambda _s, _f: cancel.set())
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(download, ep["audio_url"], ep["audio"], cancel): ep for ep in missing}
            for fut in as_completed(futures):
                ep = futures[fut]
                if cancel.is_set():
                    continue
                fut.result()  # re-raise on error
                print(f"{ep['slug']}: audio downloaded")
                _transcode_to_opus(ep["audio"])
    finally:
        signal.signal(signal.SIGINT, prev_handler)

    if cancel.is_set():
        raise KeyboardInterrupt


def intermediate_paths(ep: Episode) -> list[Path]:
    """All intermediate cache files for an episode (safe to delete on retranscribe)."""
    stem = ep["transcript"]
    return [
        stem.with_suffix(".raw.json"),
        stem.with_suffix(".annotation.json"),
        stem.with_suffix(".ckpt.json"),
    ]


def do_transcribe(
    ep: Episode,
    *,
    backend: str = "whisper-large-v3-mlx",
    speakers: list[str] | None = None,
    learn: bool = False,
    diarize: bool = False,
    speakers_path: Path = Path("cache/speakers.json"),
) -> tuple[list[Segment], dict[str, str]]:
    """Transcribe (and optionally diarize) one episode, with intermediate checkpoints.

    Steps, each with its own cache file so a ^C between any two resumes
    from that boundary on the next run:

      .raw.json        — transcription output (no speaker labels)
      .annotation.json — raw pyannote turns (SPEAKER_00/01 …)  [diarize only]
      .json            — final combined transcript

    Returns (segments, speaker_mapping).
    """
    raw = _load_or_transcribe(ep, backend)

    if diarize or learn:
        hf_token = os.environ.get("HUGGING_FACE_TOKEN")
        if not hf_token:
            sys.exit("Set HUGGING_FACE_TOKEN for diarization.")
        annotation, waveform, sr, pipeline = _load_or_diarize(
            ep, hf_token, need_full=learn or bool(load_embeddings(speakers_path))
        )
        segments = assign_speakers(raw["segments"], annotation)

        if speakers is not None:
            speaker_mapping = _first_appearance_mapping(segments, speakers)
        else:
            known = load_embeddings(speakers_path)
            if known and waveform is not None and sr and pipeline is not None:
                cluster_embs = extract_cluster_embeddings(waveform, sr, annotation, pipeline)
                speaker_mapping = match_speakers(cluster_embs, known)
            else:
                speaker_mapping = {}

        if learn:
            assert speakers  # guaranteed by CLI validation before reaching here
            _record_learned_embeddings(
                ep["slug"], segments, speakers, waveform, sr, annotation, pipeline, speakers_path=speakers_path
            )
    else:
        segments = [cast("Segment", {**s, "speaker": "UNKNOWN"}) for s in raw["segments"]]
        speaker_mapping = {}

    _atomic_write(
        ep["transcript"],
        json.dumps(
            {
                "text": raw["text"],
                "language": raw["language"],
                "backend": backend,
                "speaker_mapping": speaker_mapping,
                "segments": segments,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    return segments, speaker_mapping


def _load_or_transcribe(ep: Episode, backend: str) -> TranscriptResult:
    raw_path = ep["transcript"].with_suffix(".raw.json")
    ckpt_path = ep["transcript"].with_suffix(".ckpt.json")
    if raw_path.exists():
        print(f"{ep['slug']}: raw transcript cached, skipping to diarization")
        return cast("TranscriptResult", json.loads(raw_path.read_text()))
    raw = transcribe(str(audio_path(ep)), backend=backend, checkpoint_path=ckpt_path)
    _atomic_write(raw_path, json.dumps(raw, ensure_ascii=False, indent=2))
    ckpt_path.unlink(missing_ok=True)
    return raw


def _load_or_diarize(ep: Episode, hf_token: str, *, need_full: bool) -> tuple[Any, Any, Any, Any]:
    """Returns (annotation, waveform, sr, pipeline). waveform/sr/pipeline are None when loaded from cache."""
    ann_path = ep["transcript"].with_suffix(".annotation.json")
    if ann_path.exists() and not need_full:
        print(f"{ep['slug']}: diarization cached")
        return _annotation_from_json(cast("list[AnnotationTurn]", json.loads(ann_path.read_text()))), None, None, None
    print(f"{ep['slug']}: diarizing...")
    annotation, waveform, sr, pipeline = diarize(str(audio_path(ep)), hf_token)
    if not ann_path.exists():
        ann_path.write_text(json.dumps(_annotation_to_json(annotation)), encoding="utf-8")
    return annotation, waveform, sr, pipeline


def _record_learned_embeddings(
    slug: str,
    segments: list[Segment],
    speakers: list[str],
    waveform: Any,  # noqa: ANN401
    sr: Any,  # noqa: ANN401
    annotation: Any,  # noqa: ANN401
    pipeline: Any,  # noqa: ANN401
    *,
    speakers_path: Path,
) -> None:
    # learn=True forces need_full=True in do_transcribe, so waveform/sr/pipeline are always non-None here
    assert waveform is not None
    assert sr is not None
    assert pipeline is not None
    mapping = _first_appearance_mapping(segments, speakers)
    cluster_embs = extract_cluster_embeddings(waveform, sr, annotation, pipeline)
    named = {mapping[c]: emb for c, emb in cluster_embs.items() if c in mapping}
    save_embeddings(named, speakers_path)
    print(f"{slug}: saved embeddings for {', '.join(named)}")


def _transcode_to_opus(src: Path) -> None:
    dest = src.with_suffix(".opus")
    if dest.exists():
        return
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-i",
            str(src),
            "-c:a",
            "libopus",
            "-b:a",
            "48k",
            "-ac",
            "1",
            "-compression_level",
            "0",
            str(dest),
        ],
        check=True,
    )


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.rename(path)


def _annotation_to_json(annotation: Any) -> list[AnnotationTurn]:  # noqa: ANN401
    return [
        {"start": turn.start, "end": turn.end, "speaker": label}
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]


def _annotation_from_json(data: list[AnnotationTurn]) -> Any:  # noqa: ANN401
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for item in data:
        ann[Segment(item["start"], item["end"])] = item["speaker"]
    return ann


def render(
    ep: Episode,
    segments: list[Segment] | None = None,
    speaker_mapping: dict[str, str] | None = None,
    *,
    gap: float = PARAGRAPH_GAP_S,
    speakers: list[str] | None = None,
    diarized: bool = False,
    strip_fillers: bool = False,
) -> None:
    if segments is None:
        data = cast("StoredTranscript", json.loads(ep["transcript"].read_text()))
        segments = data["segments"]
        if speaker_mapping is None and speakers is None:
            speaker_mapping = data.get("speaker_mapping") or {}

    # Explicit --speakers override (by first-appearance) takes priority over stored mapping.
    effective_mapping = _first_appearance_mapping(segments, speakers) if speakers is not None else speaker_mapping or {}

    if effective_mapping:
        segments = _apply_mapping(segments, effective_mapping)

    paragraphs: list[str] = []
    current: list[Segment] = []
    for i, seg in enumerate(segments):
        if i > 0 and seg["start"] - segments[i - 1]["end"] >= gap:
            paragraphs.append(_flush(current, strip_fillers=strip_fillers))
            current = []
        current.append(seg)
    if current:
        paragraphs.append(_flush(current, strip_fillers=strip_fillers))
    out_path = ep["diarized_text"] if diarized else ep["text"]
    out_path.write_text("\n\n".join(paragraphs))


def _first_appearance_mapping(segments: list[Segment], names: list[str]) -> dict[str, str]:
    """Map SPEAKER_XX labels to names by order of first appearance in segments."""
    mapping: dict[str, str] = {}
    for seg in segments:
        spk = seg["speaker"]
        if spk not in mapping and spk != "UNKNOWN":
            idx = len(mapping)
            if idx < len(names):
                mapping[spk] = names[idx]
    return mapping


def _apply_mapping(segments: list[Segment], mapping: dict[str, str]) -> list[Segment]:
    # cast preserves any extra fields (e.g. "words" from mlx_whisper) at runtime
    return [cast("Segment", {**s, "speaker": mapping.get(s["speaker"], s["speaker"])}) for s in segments]


# Kept for backward compatibility with tests that mock _remap_speakers directly.
def _remap_speakers(segments: list[Segment], names: list[str]) -> list[Segment]:
    return _apply_mapping(segments, _first_appearance_mapping(segments, names))


def _flush(segs: list[Segment], *, strip_fillers: bool = False) -> str:
    ts = _fmt(segs[0]["start"])
    speaker = segs[0]["speaker"]
    text = " ".join(s["text"].strip() for s in segs)
    if strip_fillers:
        text = _strip_fillers(text)
    return f"[{ts} | {speaker}] {text}" if speaker != "UNKNOWN" else f"[{ts}] {text}"


def _fmt(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def run_episode(
    ep: Episode,
    *,
    gap: float = PARAGRAPH_GAP_S,
    dry_run: bool = False,
    speakers: list[str] | None = None,
    backend: str = "whisper-large-v3-mlx",
    learn: bool = False,
    diarize: bool = False,
    strip_fillers: bool = False,
    speakers_path: Path = Path("cache/speakers.json"),
) -> None:
    ep["transcript"].parent.mkdir(parents=True, exist_ok=True)
    ep["text"].parent.mkdir(parents=True, exist_ok=True)
    ep["diarized_text"].parent.mkdir(parents=True, exist_ok=True)
    slug = ep["slug"]

    ep["audio"].parent.mkdir(parents=True, exist_ok=True)
    if audio_path(ep).exists():
        print(f"{slug}: audio cached")
    elif dry_run:
        print(f"{slug}: would download audio")
    else:
        print(f"{slug}: downloading audio...")
        download(ep["audio_url"], ep["audio"])
        _transcode_to_opus(ep["audio"])

    fresh_segments: list[Segment] | None = None
    fresh_mapping: dict[str, str] | None = None
    if ep["transcript"].exists():
        print(f"{slug}: transcript cached")
    elif dry_run:
        print(f"{slug}: would transcribe")
    else:
        print(f"{slug}: transcribing...")
        fresh_segments, fresh_mapping = do_transcribe(
            ep, backend=backend, speakers=speakers, learn=learn, diarize=diarize, speakers_path=speakers_path
        )

    text_path = ep["diarized_text"] if diarize else ep["text"]
    if text_path.exists():
        print(f"{slug}: text cached")
    elif dry_run:
        print(f"{slug}: would render")
    elif ep["transcript"].exists() or fresh_segments is not None:
        render(
            ep, fresh_segments, fresh_mapping, gap=gap, speakers=speakers, diarized=diarize, strip_fillers=strip_fillers
        )
