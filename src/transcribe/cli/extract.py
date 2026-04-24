import argparse
import sys
from pathlib import Path

from transcribe.denoise import denoise, strip_fillers_rendered
from transcribe.extract import extract
from transcribe.models import QWEN3_9B, MLXModel
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


def prepare(ep: Episode, model: MLXModel) -> tuple[str, Path] | None:
    """Denoise ep's transcript if needed; return (cleaned_text, output_path) or None to skip."""
    if not ep["text"].exists():
        print(f"{ep['slug']}: no transcript, skipping")
        return None
    out = ep["text"].with_name(ep["text"].stem + f".extracted-{model.id}.txt")
    if out.exists():
        print(f"{ep['slug']}: already extracted, skipping")
        return None
    denoised = ep["text"].with_name(ep["text"].stem + ".denoised.txt")
    if denoised.exists():
        print(f"{ep['slug']}: using existing denoised transcript at {denoised}")
        return denoised.read_text(encoding="utf-8"), out
    raw = ep["text"].read_text(encoding="utf-8")
    cleaned = strip_fillers_rendered(denoise(raw))
    saved = len(raw) - len(cleaned)
    print(f"{ep['slug']}: {len(raw)} → {len(cleaned)} chars ({saved / len(raw):.0%} removed by heuristics)")
    denoised.write_text(cleaned, encoding="utf-8")
    print(f"{ep['slug']}: denoised written to {denoised}")
    return cleaned, out


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
        prepared = prepare(ep, model)
        if prepared is None:
            continue
        cleaned, out = prepared
        print(f"{ep['slug']}: extracting...")
        out.write_text(extract(cleaned, podcast.extraction_prompt, model), encoding="utf-8")
        print(f"{ep['slug']}: written to {out}")
