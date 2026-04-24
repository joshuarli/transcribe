#!/usr/bin/env python3
"""Podcast transcript pipeline."""

import argparse

from transcribe.cli import denoise as denoise_cmd
from transcribe.cli import diarize, distill, episodes, extract, sync
from transcribe.cli import transcribe as transcribe_cmd
from transcribe.episode import make_episode
from transcribe.feed import load_episodes
from transcribe.pipeline import dirs_for_backend
from transcribe.podcasts import DEFAULT_PODCAST, PODCASTS
from transcribe.transcribe import BACKENDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--podcast",
        choices=list(PODCASTS),
        default=DEFAULT_PODCAST,
        help=f"Which podcast to operate on (default: {DEFAULT_PODCAST})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    episodes.add_parser(sub)
    sync.add_parser(sub)
    transcribe_cmd.add_parser(sub)
    denoise_cmd.add_parser(sub)
    extract.add_parser(sub)
    distill.add_parser(sub)
    diarize.add_parser(sub)

    args = parser.parse_args()

    if getattr(args, "learn", False) and not getattr(args, "speakers", None):
        parser.error("--learn requires --speakers")
    if getattr(args, "learn", False) and args.command != "diarize" and not getattr(args, "diarize", False):
        parser.error("--learn requires --diarize")

    podcast = PODCASTS[args.podcast]
    backend = getattr(args, "transcriber", None) or getattr(args, "backend", BACKENDS[0])
    transcript_dir, text_dir = dirs_for_backend(podcast, backend)

    raw_eps = load_episodes(podcast)
    all_episodes = [
        make_episode(i, ep, audio_dir=podcast.audio_dir, transcript_dir=transcript_dir, text_dir=text_dir)
        for i, ep in enumerate(raw_eps)
    ]

    match args.command:
        case "episodes":
            episodes.run(args, podcast, all_episodes, backend)
        case "sync":
            sync.run(args, podcast, all_episodes, backend)
        case "transcribe":
            transcribe_cmd.run(args, podcast, all_episodes, backend)
        case "denoise":
            denoise_cmd.run(args, podcast, all_episodes, backend)
        case "extract":
            extract.run(args, podcast, all_episodes, backend)
        case "distill":
            distill.run(args, podcast, all_episodes, backend)
        case "diarize":
            diarize.run(args, podcast, all_episodes, backend)


if __name__ == "__main__":
    main()
