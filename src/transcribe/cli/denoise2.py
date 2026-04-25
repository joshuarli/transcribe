import argparse
import sys

from transcribe.denoise2 import DEFAULT_SIMILARITY_THRESHOLD
from transcribe.podcasts import Podcast
from transcribe.transcribe import BACKENDS
from transcribe.types import Episode


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("denoise2", help="High-quality NLP denoise pipeline (spaCy + embeddings)")
    p.add_argument("number", type=int, nargs="?", help="Episode number (omit to process all)")
    p.add_argument(
        "--transcriber",
        choices=BACKENDS,
        default=BACKENDS[0],
        help=f"Which transcription backend's output to read (default: {BACKENDS[0]})",
    )
    p.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        metavar="THRESHOLD",
        help=f"Cosine similarity threshold for deduplication clustering (default: {DEFAULT_SIMILARITY_THRESHOLD})",
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

    import spacy
    from sentence_transformers import SentenceTransformer

    from transcribe.denoise2 import denoise2, make_asr_corrector
    from transcribe.extract import llama_server
    from transcribe.models import PHI_4_MINI_INSTRUCT

    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_trf")
    print("Loading embedder...")
    embedder = SentenceTransformer("all-mpnet-base-v2")
    print(f"Starting ASR corrector ({PHI_4_MINI_INSTRUCT.id})...")
    with llama_server(PHI_4_MINI_INSTRUCT) as base_url:
        corrector = make_asr_corrector(base_url)
        for ep in targets:
            if not ep["text"].exists():
                print(f"{ep['slug']}: no transcript, skipping")
                continue
            raw = ep["text"].read_text(encoding="utf-8")
            result = denoise2(
                raw,
                similarity_threshold=args.similarity_threshold,
                nlp=nlp,
                embedder=embedder,
                corrector=corrector,
                show_progress=True,
            )
            out = ep["text"].with_name(ep["text"].stem + ".denoised2.txt")
            out.write_text(result, encoding="utf-8")
            saved = len(raw) - len(result)
            print(f"{ep['slug']}: {len(raw)} → {len(result)} chars ({saved / len(raw):.0%} removed), written to {out}")
