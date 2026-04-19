"""Shared TypedDicts for the transcribe pipeline."""

from pathlib import Path
from typing import NotRequired, TypedDict


class FeedEpisode(TypedDict):
    title: str
    audio_url: str


class Episode(TypedDict):
    number: int
    slug: str
    title: str
    audio_url: str
    audio: Path
    transcript: Path
    text: Path
    diarized_text: Path


class RawSegment(TypedDict):
    start: float
    end: float
    text: str


class Segment(TypedDict):
    start: float
    end: float
    text: str
    speaker: str


class TranscriptResult(TypedDict):
    text: str
    language: str | None
    segments: list[RawSegment]


class StoredTranscript(TypedDict):
    text: str
    language: str | None
    backend: str
    speaker_mapping: NotRequired[dict[str, str]]
    segments: list[Segment]


class AnnotationTurn(TypedDict):
    start: float
    end: float
    speaker: str


class ParakeetCheckpoint(TypedDict):
    i: int
    text: str
    segments: list[RawSegment]
