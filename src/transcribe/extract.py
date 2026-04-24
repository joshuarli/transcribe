"""LLM-based extraction of high-signal content from denoised transcripts."""

import os
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from transcribe.llama_serve import build_server_args, compute_context_length, detect_hardware, resolve_model

_DEFAULT_MLX_MODEL = "mlx-community/Qwen3.5-9B-8bit"
_DEFAULT_LLAMA_MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF"
_DEFAULT_LLAMA_QUANT = "UD-IQ4_NL"
_HIGH_MEM_LLAMA_MODEL = "unsloth/gemma-4-31B-it-GGUF"
_HIGH_MEM_LLAMA_QUANT = "UD-Q8_K_XL"
_HIGH_MEM_THRESHOLD_GB = 48.0
_DEFAULT_LLAMA_URL = "http://127.0.0.1:8080"


def _llama_model_env() -> str:
    if model := os.environ.get("LLAMA_MODEL"):
        return model
    total_mem_gb, _ = detect_hardware()
    return _HIGH_MEM_LLAMA_MODEL if total_mem_gb >= _HIGH_MEM_THRESHOLD_GB else _DEFAULT_LLAMA_MODEL


def _llama_quant() -> str:
    if quant := os.environ.get("LLAMA_QUANT"):
        return quant
    total_mem_gb, _ = detect_hardware()
    return _HIGH_MEM_LLAMA_QUANT if total_mem_gb >= _HIGH_MEM_THRESHOLD_GB else _DEFAULT_LLAMA_QUANT


def _mlx_model_id() -> str:
    """Resolve the MLX model ID, falling back to the default if the env var isn't a valid repo ID."""
    env = os.environ.get("MLX_MODEL", "")
    return env if "/" in env else _DEFAULT_MLX_MODEL


def model_slug() -> str:
    if os.environ.get("MLX_MODEL"):
        return _mlx_model_id().split("/")[-1].lower()
    llama = _llama_model_env()
    if llama.lower().endswith(".gguf"):
        return Path(llama).stem.lower()
    repo_name = llama.split("/")[-1].lower().removesuffix("-gguf")
    return f"{repo_name}-{_llama_quant().lower()}"


def extract(text: str, prompt: str) -> str:
    if os.environ.get("MLX_MODEL"):
        return _extract_mlx(text, prompt)
    return _extract_llama(text, prompt)


def _extract_mlx(text: str, prompt: str) -> str:
    import mlx_lm
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    model_id = _mlx_model_id()
    model, tokenizer, *_ = mlx_lm.load(model_id)

    local_path = Path(snapshot_download(model_id, local_files_only=True))
    model_size_gb = sum(f.stat().st_size for f in local_path.rglob("*.safetensors")) / (1024**3)
    max_tokens = compute_context_length(model_size_gb)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]
    # Qwen3 enables thinking by default; disable it for extraction tasks (same
    # rationale as the llama path: structured summarization doesn't benefit).
    is_qwen3 = "qwen3" in model_id.lower()
    chat_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if is_qwen3:
        chat_kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)

    chunks: list[str] = []
    for response in mlx_lm.stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        chunks.append(response.text)
        print(
            f"\r  {response.generation_tokens} tokens  {response.generation_tps:.1f} tok/s",
            end="",
            flush=True,
            file=sys.stderr,
        )
    print(file=sys.stderr)

    return "".join(chunks)


@contextmanager
def llama_server() -> Generator[str]:
    """Start llama-server, wait until ready, yield base_url, terminate on exit."""
    model_path = _resolve_llama_model()
    base_url = os.environ.get("LLAMA_URL", _DEFAULT_LLAMA_URL).rstrip("/")
    args = build_server_args(model_path, _llama_model_env())
    cmd = ["llama-server", *args]
    print(" ".join(cmd), file=sys.stderr)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        _wait_for_llama(f"{base_url}/health", proc)
        yield base_url
    finally:
        proc.terminate()
        proc.wait()


def extract_request(text: str, prompt: str, base_url: str) -> str:
    """Run one extraction against an already-running llama-server."""
    from transcribe.http import stream_post_json

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]
    n_prompt_tokens = _count_prompt_tokens(base_url, messages)

    stop_poll = threading.Event()
    poll_thread = threading.Thread(target=_poll_prompt_progress, args=(stop_poll, n_prompt_tokens), daemon=True)
    poll_thread.start()

    chunks: list[str] = []
    start: float | None = None
    try:
        for i, delta in enumerate(
            stream_post_json(
                f"{base_url}/v1/chat/completions",
                {
                    "messages": messages,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 64,
                    # Gemma 4 supports thinking but it adds overhead without
                    # improving structured summarization tasks
                    "thinking": False,
                },
            )
        ):
            if start is None:
                stop_poll.set()
                poll_thread.join()
                print(file=sys.stderr)
                start = time.monotonic()
            chunks.append(delta)
            elapsed = time.monotonic() - start
            tps = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(f"\r  {i + 1} tokens  {tps:.1f} tok/s", end="", flush=True, file=sys.stderr)
    finally:
        stop_poll.set()
    print(file=sys.stderr)
    return "".join(chunks)


def _extract_llama(text: str, prompt: str) -> str:
    with llama_server() as base_url:
        return extract_request(text, prompt, base_url)


def _resolve_llama_model() -> Path:
    return resolve_model(_llama_model_env(), _llama_quant())


def _count_prompt_tokens(base_url: str, messages: list[dict[str, str]]) -> int:
    """Token count for the prompt via /tokenize — no client-side tokenizer needed."""
    from transcribe.http import post_json

    content = "\n".join(m.get("content", "") for m in messages)
    try:
        resp = post_json(f"{base_url}/tokenize", {"content": content})
        return len(resp.get("tokens", []))
    except Exception:
        return 0


def _poll_prompt_progress(stop: threading.Event, n_tokens: int) -> None:
    start = time.monotonic()
    token_str = f"{n_tokens} tokens  " if n_tokens else ""
    while not stop.wait(0.5):
        elapsed = time.monotonic() - start
        print(f"\r  prefill: {token_str}{elapsed:.0f}s", end="", flush=True, file=sys.stderr)


def _wait_for_llama(health_url: str, proc: subprocess.Popen[bytes], timeout: int = 600) -> None:

    print("  waiting for llama-server...", file=sys.stderr)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(health_url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"llama-server did not become ready within {timeout}s")
