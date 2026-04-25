import argparse
import sys

from transcribe.extract import extract
from transcribe.models import QWEN3_9B
from transcribe.podcasts import Podcast
from transcribe.transcribe import BACKENDS
from transcribe.types import Episode

_DEFAULT_MODEL = QWEN3_9B


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("extract", help="Extract culinary information from a transcript via LLM")
    p.add_argument("number", type=int, nargs="?", help="Episode number (omit to process all)")
    p.add_argument(
        "--transcriber",
        choices=BACKENDS,
        default=BACKENDS[0],
        help=f"Which transcription backend's output to read (default: {BACKENDS[0]})",
    )


def run(args: argparse.Namespace, podcast: Podcast, episodes: list[Episode], backend: str) -> None:
    if not podcast.extraction_prompt:
        sys.exit(f"No extraction prompt configured for '{podcast.slug}'.")
    if args.number is not None:
        if not 1 <= args.number <= len(episodes):
            sys.exit(f"Episode {args.number} not found.")
        targets = [episodes[args.number - 1]]
    else:
        targets = [ep for ep in episodes if ep["text"].exists()]
        if not targets:
            sys.exit("No transcripts found — run 'transcribe' first.")

    model = _DEFAULT_MODEL
    print(f"model: {model.repo_id}")
    for ep in targets:
        denoised = ep["text"].with_name(ep["text"].stem + ".denoised2.txt")
        if not denoised.exists():
            print(f"{ep['slug']}: no denoised2 transcript, skipping (run denoise2 first)")
            continue
        out = ep["text"].with_name(ep["text"].stem + f".extracted2-{model.id}.txt")
        if out.exists():
            print(f"{ep['slug']}: already extracted, skipping")
            continue
        text = denoised.read_text(encoding="utf-8")
        print(f"{ep['slug']}: extracting...")
        out.write_text(extract(text, podcast.extraction_prompt, model), encoding="utf-8")
        print(f"{ep['slug']}: written to {out}")
