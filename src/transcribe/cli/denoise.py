import argparse
import sys

from transcribe.denoise import denoise, strip_fillers_rendered
from transcribe.podcasts import Podcast
from transcribe.transcribe import BACKENDS
from transcribe.types import Episode


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("denoise", help="Clean a rendered transcript with heuristic filters")
    p.add_argument("number", type=int, nargs="?", help="Episode number (omit to process all)")
    p.add_argument(
        "--transcriber",
        choices=BACKENDS,
        default=BACKENDS[0],
        help=f"Which transcription backend's output to read (default: {BACKENDS[0]})",
    )


def run(args: argparse.Namespace, podcast: Podcast, episodes: list[Episode], backend: str) -> None:
    if args.number is not None:
        if not 1 <= args.number <= len(episodes):
            sys.exit(f"Episode {args.number} not found.")
        targets = [episodes[args.number - 1]]
    else:
        targets = [ep for ep in episodes if ep["text"].exists()]
        if not targets:
            sys.exit("No transcripts found — run 'transcribe' first.")
    for ep in targets:
        if not ep["text"].exists():
            print(f"{ep['slug']}: no transcript, skipping")
            continue
        raw = ep["text"].read_text(encoding="utf-8")
        result = strip_fillers_rendered(denoise(raw))
        out = ep["text"].with_name(ep["text"].stem + ".denoised.txt")
        out.write_text(result, encoding="utf-8")
        saved = len(raw) - len(result)
        print(f"{ep['slug']}: {len(raw)} → {len(result)} chars ({saved / len(raw):.0%} removed), written to {out}")
