import argparse
import sys

from transcribe.cli.common import add_render_args, parse_speakers
from transcribe.pipeline import intermediate_paths, run_episode
from transcribe.podcasts import Podcast
from transcribe.types import Episode


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("transcribe", help="Transcribe episode by number")
    p.add_argument("number", type=int)
    p.add_argument("--redo", action="store_true", help="Delete cached transcript and re-transcribe from scratch")
    add_render_args(p)


def run(args: argparse.Namespace, podcast: Podcast, episodes: list[Episode], backend: str) -> None:
    if not 1 <= args.number <= len(episodes):
        sys.exit(f"Episode {args.number} not found.")
    ep = episodes[args.number - 1]
    if args.redo:
        for p in [ep["transcript"], ep["text"], ep["diarized_text"], *intermediate_paths(ep)]:
            p.unlink(missing_ok=True)
    run_episode(
        ep,
        gap=args.gap,
        speakers=parse_speakers(args.speakers),
        backend=backend,
        learn=args.learn,
        diarize=args.diarize,
        strip_fillers=args.strip_fillers,
        speakers_path=podcast.speakers_path,
    )
