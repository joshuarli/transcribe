"""LLM-based extraction of high-signal content from denoised transcripts."""

import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from transcribe.podcasts import Podcast

_DEFAULT_MLX_MODEL = "mlx-community/LFM2-8B-A1B-6bit-MLX"
_DEFAULT_LLAMA_MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF"
_DEFAULT_LLAMA_URL = "http://127.0.0.1:8080"
_DEFAULT_LLAMA_QUANT = "UD-IQ4_NL"


def _llama_model_env() -> str:
    return os.environ.get("LLAMA_MODEL", _DEFAULT_LLAMA_MODEL)


def _llama_quant() -> str:
    return os.environ.get("LLAMA_QUANT", _DEFAULT_LLAMA_QUANT)


def model_slug() -> str:
    if mlx := os.environ.get("MLX_MODEL"):
        return mlx.split("/")[-1].lower()
    llama = _llama_model_env()
    if llama.lower().endswith(".gguf"):
        return Path(llama).stem.lower()
    repo_name = llama.split("/")[-1].lower().removesuffix("-gguf")
    return f"{repo_name}-{_llama_quant().lower()}"


def extract(text: str, podcast: Podcast) -> str:
    if os.environ.get("MLX_MODEL"):
        return _extract_mlx(text, podcast)
    return _extract_llama(text, podcast)


def _extract_mlx(text: str, podcast: Podcast) -> str:
    import mlx_lm
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    model_id = os.environ.get("MLX_MODEL", _DEFAULT_MLX_MODEL)
    model, tokenizer, *_ = mlx_lm.load(model_id)

    local_path = Path(snapshot_download(model_id, local_files_only=True))
    model_size_gb = sum(f.stat().st_size for f in local_path.rglob("*.safetensors")) / (1024**3)
    max_tokens = _compute_context_length(model_size_gb)

    messages = [
        {"role": "system", "content": podcast.extraction_prompt},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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


def _extract_llama(text: str, podcast: Podcast) -> str:
    model_path = _resolve_llama_model()
    base_url = os.environ.get("LLAMA_URL", _DEFAULT_LLAMA_URL).rstrip("/")
    args = _llama_server_args(model_path)

    cmd = ["llama-server", *args]
    print(" ".join(cmd), file=sys.stderr)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        _wait_for_llama(f"{base_url}/health", proc)

        from transcribe.http import stream_post_json

        messages = [
            {"role": "system", "content": podcast.extraction_prompt},
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
                    {"messages": messages},
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
    finally:
        proc.terminate()
        proc.wait()


def _resolve_llama_model() -> Path:
    """Return local GGUF path, downloading from HF if LLAMA_MODEL is a repo ID."""
    llama = _llama_model_env()
    if llama.lower().endswith(".gguf"):
        return Path(llama)

    from huggingface_hub import hf_hub_download, list_repo_files

    quant = _llama_quant()
    matches = [
        f for f in list_repo_files(llama) if f.lower().endswith(".gguf") and quant.lower() in f.lower() and "/" not in f
    ]
    if not matches:
        all_gguf = [f for f in list_repo_files(llama) if f.lower().endswith(".gguf")]
        raise ValueError(f"No GGUF matching '{quant}' in {llama}. Available: {all_gguf}")

    filename = matches[0]
    print(f"  model: {filename}", file=sys.stderr)
    return Path(hf_hub_download(repo_id=llama, filename=filename))


def _detect_hardware() -> tuple[float, int]:
    """Returns (total_memory_gb, physical_cores) via sysctl on macOS."""
    import subprocess

    def sysctl_int(name: str) -> int | None:
        try:
            r = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=2)
            return int(r.stdout.strip()) if r.returncode == 0 else None
        except Exception:
            return None

    mem_bytes = sysctl_int("hw.memsize")
    mem_gb = mem_bytes / (1024**3) if mem_bytes else 16.0
    cores = sysctl_int("hw.physicalcpu") or 4
    return mem_gb, cores


def _compute_context_length(model_size_gb: float) -> int:
    total_mem_gb, _ = _detect_hardware()
    available_gb = max(total_mem_gb - 4.0, 2.0)
    est_params_b = model_size_gb / 0.55
    # KV cache at q8_0: ~0.25 MB per 1k context per billion parameters
    kv_per_1k_ctx_mb = est_params_b * 0.25
    remaining_gb = max(available_gb - model_size_gb, 0.5)
    max_ctx = int((remaining_gb * 1024) / kv_per_1k_ctx_mb * 1024)
    return (max(4096, min(max_ctx, 131_072)) // 1024) * 1024


def _llama_server_args(model_path: Path) -> list[str]:
    total_mem_gb, physical_cores = _detect_hardware()
    model_size_gb = model_path.stat().st_size / (1024**3)
    context_length = _compute_context_length(model_size_gb)
    available_gb = max(total_mem_gb - 4.0, 2.0)
    remaining_gb = max(available_gb - model_size_gb, 0.5)

    # Metal can reliably address ~70% of unified memory; reserve the rest for
    # KV cache, activation buffers, and OS overhead. No partial offload on
    # Apple Silicon — CPU↔GPU round-trips hurt more than they help.
    metal_budget_gb = total_mem_gb * 0.70
    gpu_layers = 99 if model_size_gb < metal_budget_gb else 0

    gen_threads = max(physical_cores - 2, 1)
    batch_threads = physical_cores

    if remaining_gb > 6.0:
        batch_size = 4096
    elif remaining_gb > 3.0:
        batch_size = 2048
    else:
        batch_size = 1024
    # On Apple Silicon, unified memory means no VRAM spill penalty; match ubatch
    # to batch for maximum Metal throughput when memory allows
    ubatch_size = batch_size if remaining_gb > 6.0 else max(batch_size // 2, 256)

    parsed = urlparse(os.environ.get("LLAMA_URL", _DEFAULT_LLAMA_URL))
    port = parsed.port or 8080

    return [
        "--model",
        str(model_path),
        "--ctx-size",
        str(context_length),
        "-ngl",
        str(gpu_layers),
        "-fa",
        "on",
        "-b",
        str(batch_size),
        "-ub",
        str(ubatch_size),
        "-t",
        str(gen_threads),
        "-tb",
        str(batch_threads),
        "-np",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--jinja",
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
        "--no-context-shift",
        "--cache-reuse",
        "256",
        "--poll",
        "0",
        "--prio",
        "2",
    ]


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
