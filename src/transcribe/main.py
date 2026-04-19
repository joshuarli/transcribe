#!/usr/bin/env python3
"""Podcast transcript pipeline."""

import argparse
import sys

from transcribe.denoise import denoise
from transcribe.episode import make_episode
from transcribe.extract import extract
from transcribe.feed import load_episodes
from transcribe.pipeline import (
    PARAGRAPH_GAP_S,
    dirs_for_backend,
    do_transcribe,
    download_missing,
    intermediate_paths,
    render,
    run_episode,
)
from transcribe.podcasts import DEFAULT_PODCAST, PODCASTS


def _parse_speakers(value: str | None) -> list[str] | None:
    return [s.strip() for s in value.split(",")] if value else None


def _add_render_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--gap",
        type=float,
        default=PARAGRAPH_GAP_S,
        metavar="S",
        help=f"Paragraph gap in seconds (default: {PARAGRAPH_GAP_S})",
    )
    p.add_argument(
        "--speakers",
        metavar="NAMES",
        help="Comma-separated speaker names in order of first appearance",
    )
    p.add_argument(
        "--backend",
        choices=["whisper-large-v3-turbo", "parakeet-tdt-0.6b-v3"],
        default="whisper-large-v3-turbo",
        help="Transcription backend (default: whisper-large-v3-turbo)",
    )
    p.add_argument(
        "--diarize",
        action="store_true",
        help="Run speaker diarization (requires HUGGING_FACE_TOKEN)",
    )
    p.add_argument(
        "--learn",
        action="store_true",
        help="Extract and save speaker embeddings (requires --speakers and --diarize)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--podcast",
        choices=list(PODCASTS),
        default=DEFAULT_PODCAST,
        help=f"Which podcast to operate on (default: {DEFAULT_PODCAST})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("episodes", help="List all episodes")
    p.add_argument(
        "--backend",
        choices=["whisper-large-v3-turbo", "parakeet-tdt-0.6b-v3"],
        default="whisper-large-v3-turbo",
        help="Which backend's transcripts to check (default: whisper-large-v3-turbo)",
    )

    p = sub.add_parser("sync", help="Transcribe all episodes")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without doing it",
    )
    p.add_argument(
        "--download-only",
        action="store_true",
        help="Download missing audio files and exit without transcribing",
    )
    _add_render_args(p)

    p = sub.add_parser("transcribe", help="Transcribe episode by number")
    p.add_argument("number", type=int)
    p.add_argument("--redo", action="store_true", help="Delete cached transcript and re-transcribe from scratch")
    _add_render_args(p)

    p = sub.add_parser("extract", help="Extract culinary information from a transcript via LLM")
    p.add_argument("number", type=int)
    p.add_argument(
        "--transcriber",
        choices=["whisper-large-v3-turbo", "parakeet-tdt-0.6b-v3"],
        default="whisper-large-v3-turbo",
        help="Which transcription backend's output to read (default: whisper-large-v3-turbo)",
    )
    p.add_argument(
        "--model",
        choices=["llama", "haiku"],
        default="llama",
        help="LLM backend to use for extraction (default: llama)",
    )

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
    p.add_argument(
        "--backend", choices=["whisper-large-v3-turbo", "parakeet-tdt-0.6b-v3"], default="whisper-large-v3-turbo"
    )

    args = parser.parse_args()

    if getattr(args, "learn", False) and not getattr(args, "speakers", None):
        parser.error("--learn requires --speakers")
    if getattr(args, "learn", False) and args.command != "diarize" and not getattr(args, "diarize", False):
        parser.error("--learn requires --diarize")

    podcast = PODCASTS[args.podcast]
    backend = getattr(args, "transcriber", None) or getattr(args, "backend", "whisper-large-v3-turbo")
    transcript_dir, text_dir = dirs_for_backend(podcast, backend)

    raw_eps = load_episodes(podcast)
    episodes = [
        make_episode(i, ep, audio_dir=podcast.audio_dir, transcript_dir=transcript_dir, text_dir=text_dir)
        for i, ep in enumerate(raw_eps)
    ]

    match args.command:
        case "episodes":
            for ep in episodes:
                t = "t" if ep["text"].exists() else " "
                d = "d" if ep["diarized_text"].exists() else " "
                print(f"[{t}{d}] {ep['slug']}  {ep['title']}")
        case "sync":
            speakers = _parse_speakers(args.speakers)
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
                    speakers_path=podcast.speakers_path,
                )
        case "transcribe":
            if not 1 <= args.number <= len(episodes):
                sys.exit(f"Episode {args.number} not found.")
            ep = episodes[args.number - 1]
            if args.redo:
                for p in [ep["transcript"], ep["text"], ep["diarized_text"], *intermediate_paths(ep)]:
                    p.unlink(missing_ok=True)
            run_episode(
                ep,
                gap=args.gap,
                speakers=_parse_speakers(args.speakers),
                backend=backend,
                learn=args.learn,
                diarize=args.diarize,
                speakers_path=podcast.speakers_path,
            )
        case "extract":
            if not 1 <= args.number <= len(episodes):
                sys.exit(f"Episode {args.number} not found.")
            if not podcast.extraction_prompt:
                sys.exit(f"No extraction prompt configured for '{podcast.slug}'.")
            ep = episodes[args.number - 1]
            if not ep["text"].exists():
                sys.exit(f"No transcript at {ep['text']} — run 'transcribe' first.")
            raw = ep["text"].read_text(encoding="utf-8")
            cleaned = denoise(raw)
            saved = len(raw) - len(cleaned)
            print(f"{ep['slug']}: {len(raw)} → {len(cleaned)} chars ({saved / len(raw):.0%} removed by heuristics)")
            postprocessed = ep["text"].with_name(ep["text"].stem + ".postprocessed.txt")
            postprocessed.write_text(cleaned, encoding="utf-8")
            print(f"{ep['slug']}: postprocessed written to {postprocessed}")
            print(f"{ep['slug']}: extracting...")
            result = extract(cleaned, podcast, backend=args.model)
            out = ep["text"].with_name(ep["text"].stem + ".extracted.txt")
            out.write_text(result, encoding="utf-8")
            print(f"{ep['slug']}: written to {out}")
        case "diarize":
            if not 1 <= args.number <= len(episodes):
                sys.exit(f"Episode {args.number} not found.")
            ep = episodes[args.number - 1]
            speakers = _parse_speakers(args.speakers)
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


if __name__ == "__main__":
    main()
