import argparse
import sys

from transcribe.cli.common import parse_speakers
from transcribe.pipeline import PARAGRAPH_GAP_S, do_transcribe, render
from transcribe.podcasts import Podcast
from transcribe.transcribe import BACKENDS
from transcribe.types import Episode


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("diarize", help="Diarize an already-transcribed episode")
    p.add_argument("number", type=int)
    p.add_argument(
        "--gap",
        type=float,
        default=PARAGRAPH_GAP_S,
        metavar="S",
        help=f"Paragraph gap in seconds (default: {PARAGRAPH_GAP_S})",
    )
    p.add_argument("--speakers", metavar="NAMES", help="Comma-separated speaker names in order of first appearance")
    p.add_argument("--learn", action="store_true", help="Extract and save speaker embeddings (requires --speakers)")
    p.add_argument("--backend", choices=BACKENDS, default=BACKENDS[0])


def run(args: argparse.Namespace, podcast: Podcast, episodes: list[Episode], backend: str) -> None:
    if not 1 <= args.number <= len(episodes):
        sys.exit(f"Episode {args.number} not found.")
    ep = episodes[args.number - 1]
    speakers = parse_speakers(args.speakers)
    ep["transcript"].parent.mkdir(parents=True, exist_ok=True)
    ep["diarized_text"].parent.mkdir(parents=True, exist_ok=True)
    segments, mapping = do_transcribe(
        ep,
        backend=backend,
        speakers=speakers,
        learn=args.learn,
        diarize=True,
        speakers_path=podcast.speakers_path,
    )
    render(ep, segments, mapping, gap=args.gap, speakers=speakers, diarized=True)
