import argparse

from transcribe.pipeline import PARAGRAPH_GAP_S
from transcribe.transcribe import BACKENDS


def parse_speakers(value: str | None) -> list[str] | None:
    return [s.strip() for s in value.split(",")] if value else None


def add_render_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--gap",
        type=float,
        default=PARAGRAPH_GAP_S,
        metavar="S",
        help=f"Paragraph gap in seconds (default: {PARAGRAPH_GAP_S})",
    )
    p.add_argument(
        "--speakers",
        metavar="NAMES",
        help="Comma-separated speaker names in order of first appearance",
    )
    p.add_argument(
        "--backend",
        choices=BACKENDS,
        default=BACKENDS[0],
        help=f"Transcription backend (default: {BACKENDS[0]})",
    )
    p.add_argument(
        "--diarize",
        action="store_true",
        help="Run speaker diarization (requires HUGGING_FACE_TOKEN)",
    )
    p.add_argument(
        "--learn",
        action="store_true",
        help="Extract and save speaker embeddings (requires --speakers and --diarize)",
    )
    p.add_argument(
        "--strip-fillers",
        action="store_true",
        help="Remove 'uh'/'um' disfluencies from rendered transcript",
    )
