import argparse

from transcribe.podcasts import Podcast
from transcribe.transcribe import BACKENDS
from transcribe.types import Episode


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("episodes", help="List all episodes")
    p.add_argument(
        "--backend",
        choices=BACKENDS,
        default=BACKENDS[0],
        help=f"Which backend's transcripts to check (default: {BACKENDS[0]})",
    )


def run(args: argparse.Namespace, podcast: Podcast, episodes: list[Episode], backend: str) -> None:
    for ep in episodes:
        t = "t" if ep["text"].exists() else " "
        d = "d" if ep["diarized_text"].exists() else " "
        print(f"[{t}{d}] {ep['slug']}  {ep['title']}")
