from dataclasses import dataclass


@dataclass(frozen=True)
class MLXModel:
    id: str
    repo_id: str


# @dataclass(frozen=True)
# class HFSeq2SeqModel:
#     id: str
#     repo_id: str


@dataclass(frozen=True)
class TorchModel:
    id: str
    repo_id: str


@dataclass(frozen=True)
class LlamaModel:
    id: str
    repo_id: str
    quant: str
    flash_attn: bool = True


# Needs finetune to be somewhat useful
# FLAN_T5 = HFSeq2SeqModel(
#     id="flan-t5-base",
#     repo_id="google/flan-t5-base",
# large is too slow for very little gain
# )
PHI_4_MINI_INSTRUCT = TorchModel(
    id="phi-4-mini-instruct",
    repo_id="microsoft/Phi-4-mini-instruct",
)

QWEN3_9B = MLXModel(
    id="qwen3.5-9b-8bit",
    repo_id="mlx-community/Qwen3.5-9B-8bit",
)

GEMMA_4_26B = LlamaModel(
    id="gemma-4-26b-a4b-it-ud-iq4_nl",
    repo_id="unsloth/gemma-4-26B-A4B-it-GGUF",
    quant="UD-IQ4_NL",
    # Gemma 4 MoE hybrid attention regresses with Metal flash attention
    flash_attn=False,
)

GEMMA_4_31B = LlamaModel(
    id="gemma-4-31b-it-ud-q8_k_xl",
    repo_id="unsloth/gemma-4-31B-it-GGUF",
    quant="UD-Q8_K_XL",
    flash_attn=True,
)

_LLAMA_CATALOG: list[LlamaModel] = [GEMMA_4_26B, GEMMA_4_31B]

_HIGH_MEM_THRESHOLD_GB = 48.0


def default_llama_model(mem_gb: float) -> LlamaModel:
    return GEMMA_4_31B if mem_gb >= _HIGH_MEM_THRESHOLD_GB else GEMMA_4_26B


def lookup_llama_model(id_or_repo: str) -> LlamaModel | None:
    for m in _LLAMA_CATALOG:
        if m.id == id_or_repo or m.repo_id == id_or_repo:
            return m
    return None
