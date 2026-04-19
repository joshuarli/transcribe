"""LLM-based extraction of high-signal content from denoised transcripts."""

import os

from transcribe.http import post_json
from transcribe.podcasts import Podcast

_DEFAULT_LLAMA_URL = "http://localhost:8080/v1/chat/completions"


def extract(text: str, podcast: Podcast, *, backend: str = "llama") -> str:
    if backend == "haiku":
        return _extract_haiku(text, podcast)
    return _extract_llama(text, podcast)


def _extract_llama(text: str, podcast: Podcast) -> str:
    url = os.environ.get("LLAMA_URL", _DEFAULT_LLAMA_URL)
    resp = post_json(url, {
        "messages": [
            {"role": "system", "content": podcast.extraction_prompt},
            {"role": "user", "content": text},
        ],
        "max_tokens": 8192,
    })
    return resp["choices"][0]["message"]["content"]  # type: ignore[index]


def _extract_haiku(text: str, podcast: Podcast) -> str:
    import anthropic
    import sys
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=8192,  # model maximum; extraction output can be dense
        system=podcast.extraction_prompt,
        messages=[{"role": "user", "content": text}],
    )
    if msg.stop_reason == "max_tokens":
        print("warning: extraction output was truncated at max_tokens limit", file=sys.stderr)
    return next(b.text for b in msg.content if b.type == "text")
