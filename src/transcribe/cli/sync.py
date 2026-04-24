import argparse

from transcribe.cli.common import add_render_args, parse_speakers
from transcribe.pipeline import download_missing, run_episode
from transcribe.podcasts import Podcast
from transcribe.types import Episode


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("sync", help="Transcribe all episodes")
    p.add_argument("--dry-run", action="store_true", help="Print what would be done without doing it")
    p.add_argument(
        "--download-only",
        action="store_true",
        help="Download missing audio files and exit without transcribing",
    )
    add_render_args(p)


def run(args: argparse.Namespace, podcast: Podcast, episodes: list[Episode], backend: str) -> None:
    speakers = parse_speakers(args.speakers)
    if not args.dry_run:
        download_missing(episodes)
    if args.download_only:
        return
    for ep in episodes:
        run_episode(
            ep,
            gap=args.gap,
            dry_run=args.dry_run,
            speakers=speakers,
            backend=backend,
            learn=args.learn,
            diarize=args.diarize,
            strip_fillers=args.strip_fillers,
            speakers_path=podcast.speakers_path,
        )
