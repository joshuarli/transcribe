import argparse
import sys
from pathlib import Path

from transcribe.extract import extract_request, llama_server
from transcribe.llama_serve import detect_hardware
from transcribe.models import LlamaModel, default_llama_model
from transcribe.podcasts import Podcast
from transcribe.transcribe import BACKENDS
from transcribe.types import Episode


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("distill", help="Distill niche culinary knowledge from an extracted transcript via LLM")
    p.add_argument("number", type=int, nargs="?", help="Episode number (omit to process all)")
    p.add_argument(
        "--transcriber",
        choices=BACKENDS,
        default=BACKENDS[0],
        help=f"Which transcription backend's output to read (default: {BACKENDS[0]})",
    )
    p.add_argument(
        "--extractor",
        default="qwen3.5-9b-8bit",
        metavar="SLUG",
        help="Model slug used to name the extracted input files (default: qwen3.5-9b-8bit)",
    )


def prepare(ep: Episode, extractor_slug: str, model: LlamaModel) -> tuple[str, Path] | None:
    """Find the extracted file for ep; return (text, output_path) or None to skip."""
    extracted = ep["text"].with_name(ep["text"].stem + f".extracted-{extractor_slug}.txt")
    if not extracted.exists():
        print(f"{ep['slug']}: no extracted file ({extracted.name}), skipping")
        return None
    out = ep["text"].with_name(ep["text"].stem + f".extracted-{extractor_slug}.distilled-{model.id}.txt")
    if out.exists():
        print(f"{ep['slug']}: already distilled, skipping")
        return None
    return extracted.read_text(encoding="utf-8"), out


def run(args: argparse.Namespace, podcast: Podcast, episodes: list[Episode], backend: str) -> None:
    if not podcast.distillation_prompt:
        sys.exit(f"No distillation prompt configured for '{podcast.slug}'.")
    if args.number is not None:
        if not 1 <= args.number <= len(episodes):
            sys.exit(f"Episode {args.number} not found.")
        targets = [episodes[args.number - 1]]
    else:
        targets = [ep for ep in episodes if ep["text"].exists()]
        if not targets:
            sys.exit("No transcripts found — run 'transcribe' first.")

    mem_gb, _ = detect_hardware()
    model = default_llama_model(mem_gb)

    with llama_server(model) as base_url:
        for ep in targets:
            prepared = prepare(ep, args.extractor, model)
            if prepared is None:
                continue
            text, out = prepared
            print(f"{ep['slug']}: distilling...")
            out.write_text(extract_request(text, podcast.distillation_prompt, base_url), encoding="utf-8")
            print(f"{ep['slug']}: written to {out}")
