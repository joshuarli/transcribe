"""Hardware-aware llama-server launcher."""

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

_DEFAULT_URL = "http://127.0.0.1:8080"


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        print("usage: llama-serve <repo_id_or_gguf_path> [quant]", file=sys.stderr)
        sys.exit(1)

    repo_or_path = argv[0]
    quant = argv[1] if len(argv) > 1 else None

    model_path = resolve_model(repo_or_path, quant)
    args = build_server_args(model_path, repo_or_path)
    cmd = ["llama-server", *args]
    print(" ".join(cmd), file=sys.stderr)
    os.execvp("llama-server", cmd)


def resolve_model(repo_or_path: str, quant: str | None = None) -> Path:
    """Return a local GGUF path, downloading from HF if repo_or_path is a repo ID."""
    if repo_or_path.lower().endswith(".gguf"):
        return Path(repo_or_path)

    if quant is None:
        raise ValueError("quant is required when repo_or_path is a HuggingFace repo ID")

    from huggingface_hub import hf_hub_download, list_repo_files

    matches = [
        f
        for f in list_repo_files(repo_or_path)
        if f.lower().endswith(".gguf") and quant.lower() in f.lower() and "/" not in f
    ]
    if not matches:
        all_gguf = [f for f in list_repo_files(repo_or_path) if f.lower().endswith(".gguf")]
        raise ValueError(f"No GGUF matching '{quant}' in {repo_or_path}. Available: {all_gguf}")

    filename = matches[0]
    print(f"  model: {filename}", file=sys.stderr)
    return Path(hf_hub_download(repo_id=repo_or_path, filename=filename))


def detect_hardware() -> tuple[float, int]:
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


def compute_context_length(model_size_gb: float, kv_bits: int = 8, compute_reserve_gb: float = 2.0) -> int:
    total_mem_gb, _ = detect_hardware()
    # Metal's practical ceiling is ~70% of unified memory; the rest is OS/system reserved
    metal_budget_gb = total_mem_gb * 0.70
    available_gb = max(metal_budget_gb - compute_reserve_gb, 2.0)
    est_params_b = model_size_gb / 0.55
    # KV cache baseline: ~0.25 MB per 1k context per billion parameters at q8_0 (8-bit)
    kv_per_1k_ctx_mb = est_params_b * 0.25 * (kv_bits / 8)
    remaining_gb = max(available_gb - model_size_gb, 0.5)
    max_ctx = int((remaining_gb * 1024) / kv_per_1k_ctx_mb * 1024)
    return (max(4096, min(max_ctx, 131_072)) // 1024) * 1024


def build_server_args(model_path: Path, model_id: str = "") -> list[str]:
    total_mem_gb, physical_cores = detect_hardware()
    model_size_gb = model_path.stat().st_size / (1024**3)

    # Metal can reliably address ~70% of unified memory; reserve the rest for
    # KV cache, activation buffers, and OS overhead. No partial offload on
    # Apple Silicon — CPU↔GPU round-trips hurt more than they help.
    metal_budget_gb = total_mem_gb * 0.70
    gpu_layers = 99 if model_size_gb < metal_budget_gb else 0

    gen_threads = max(physical_cores - 2, 1)
    batch_threads = physical_cores

    # Size batch relative to Metal headroom after the model, not total system RAM.
    # Oversized batches inflate the compute buffer (which lives in GPU memory), so
    # this is the primary lever for avoiding OOM on memory-constrained systems.
    metal_remaining_gb = max(metal_budget_gb - model_size_gb, 0.5)
    if metal_remaining_gb > 8.0:
        batch_size = 4096
    elif metal_remaining_gb > 4.0:
        batch_size = 2048
    else:
        batch_size = 1024
    # On Apple Silicon, unified memory means no VRAM spill penalty; match ubatch
    # to batch for maximum Metal throughput when memory allows
    ubatch_size = batch_size if metal_remaining_gb > 8.0 else max(batch_size // 2, 256)

    # Estimate compute buffer from empirical observation (~4 GB at batch 4096 for a
    # 16 GB model), scaling linearly with batch size and capped at model size.
    compute_est_gb = (batch_size / 4096) * min(model_size_gb * 0.25, 4.0)

    # Use q4_0 KV cache when Metal headroom is tight — halves KV memory vs q8_0
    # with acceptable quality impact for most inference tasks.
    kv_quant = "q4_0" if metal_remaining_gb < 8.0 else "q8_0"
    kv_bits = 4 if metal_remaining_gb < 8.0 else 8

    context_length = compute_context_length(model_size_gb, kv_bits=kv_bits, compute_reserve_gb=compute_est_gb)

    parsed = urlparse(os.environ.get("LLAMA_URL", _DEFAULT_URL))
    port = parsed.port or 8080

    # Gemma 4 MoE uses hybrid (local/global) attention that regresses with Metal FA;
    # the dense Gemma 4 31B does not have this issue and benefits from FA normally.
    mid = model_id.lower()
    is_gemma_moe = ("gemma" in mid or "gemma" in model_path.name.lower()) and (
        "a1b" in mid or "a4b" in mid or "-a" in mid
    )

    args = [
        "--model",
        str(model_path),
        "--ctx-size",
        str(context_length),
        "-ngl",
        str(gpu_layers),
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
        kv_quant,
        "-ctv",
        kv_quant,
        "--no-context-shift",
        "--cache-reuse",
        "256",
        "--poll",
        "0",
        "--prio",
        "2",
    ]
    if not is_gemma_moe:
        args += ["-fa", "on"]
    return args
