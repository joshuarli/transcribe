# Agents

## Structure

```
src/transcribe/
  __init__.py       — empty
  main.py           — CLI entry point; registers subcommands from cli/
  pipeline.py       — episode pipeline: download, transcribe, diarize, render; checkpoint logic
  transcribe.py     — transcription backends (mlx-whisper, parakeet); BACKENDS list
  diarize.py        — pyannote speaker diarization; assign_speakers, extract_cluster_embeddings
  speakers.py       — speaker embedding save/load/match (cosine similarity)
  episode.py        — Episode TypedDict construction
  feed.py           — RSS feed fetch and episode parsing
  podcasts.py       — Podcast registry and config
  models.py         — model registry: MLXModel, LlamaModel; PHI_4_MINI_INSTRUCT, QWEN3_9B, GEMMA_4_*
  extract.py        — LLM extraction: llama_server context manager, extract_request, MLX path
  llama_serve.py    — hardware-aware llama-server launcher: resolve_model, build_server_args
  denoise.py        — heuristic transcript cleanup (regex, number normalization, filler removal)
  denoise2.py       — high-quality NLP denoise pipeline (spaCy + sentence-transformers + llama ASR correction)
  http.py           — audio download with cancel support; post_json, stream_post_json
  types.py          — shared TypedDicts (Episode, Segment, RawSegment, StoredTranscript, …)
  cli/
    common.py       — shared argparse helpers (render args, speaker parsing)
    episodes.py     — list episodes with transcript status
    sync.py         — transcribe all episodes (fetch → download → transcribe → render)
    transcribe.py   — transcribe a single episode by number
    diarize.py      — diarize an already-transcribed episode
    extract.py      — extract culinary content from a transcript via LLM
    distill.py      — distill niche culinary knowledge from extracted transcript via LLM
    denoise.py      — heuristic denoise (denoise.py pipeline)
    denoise2.py     — high-quality NLP denoise (denoise2.py pipeline)
pyproject.toml      — deps, build (uv_build, module-root=src), console script (main → transcribe.main:main)
transcripts/        — committed output: transcripts/<backend>/<slug>.txt
cache/              — gitignored: feeds, audio, intermediate JSON
```

## Pipeline

HRN + Acast RSS → audio download → transcription backend → (optional) pyannote diarization → rendered text

Per-episode flow: `cache/audio/<slug>.opus` → `cache/<backend>/<slug>.raw.json` → `cache/<backend>/<slug>.json` → `transcripts/<backend>/<slug>.txt`

## Key details

- All heavy imports (`mlx_whisper`, `torch`, `pyannote`, `mlx_lm`) are lazy — inside function bodies — so `main episodes` starts instantly.
- `transcribe.py` exposes a `BACKENDS` list that drives all `--backend` choices throughout the CLI.
- Diarization is in `diarize.py`, re-exported from `transcribe.py` for backward compatibility.
- Diarization uses `exclusive_speaker_diarization` from `DiarizeOutput` (community-1 API), which has no overlapping turns — better for per-segment speaker assignment.
- Speaker assignment: for each segment, find the pyannote turn with greatest temporal overlap (two-pointer O(n+m) walk).
- Paragraphs split on ≥0.5s silence gaps. Format: `[M:SS | SPEAKER_N] text`.
- Feed slugs: `{NNN}-{title-slug}` (e.g. `042-episode-42-raw-vegan-diet`).
- Acast feed fetched with ETag conditional request; HRN always re-fetched.

## LLM models (`models.py`)

- `MLXModel` — loaded via `mlx_lm` (Apple Silicon native weights).
- `LlamaModel` — loaded via `llama-server` (GGUF); fields: `id`, `repo_id`, `quant`, `flash_attn`.
- `PHI_4_MINI_INSTRUCT` — used for ASR post-correction in `denoise2`.
- `QWEN3_9B` — MLX model; used for extraction/distillation on lower-memory machines.
- `GEMMA_4_26B` / `GEMMA_4_31B` — Llama models; `default_llama_model(mem_gb)` picks between them at 48 GB threshold.
- `lookup_llama_model(id_or_repo)` — resolve a model by id or repo_id string.

## denoise2 pipeline

Five-step pipeline in `denoise2.py`:
1. **Boilerplate gating** (`_prepass`) — strip timestamps, drop filler/bumper/sponsor paragraphs and pre-roll.
2. **Regex normalization** (`_normalize_para`) — disfluencies, stutters, numbers, units, URLs.
3. **ASR post-correction** (`_asr_correct`) — per-paragraph correction via `PHI_4_MINI_INSTRUCT` running under `llama-server`; falls back to original if output diverges > 15% or drops content words.
4. **NLP cleanup** (`_clean_filler`, `_should_keep`) — spaCy `en_core_web_trf`; removes filler phrases and short contentless sentences, with discourse-marker exemptions.
5. **Embedding deduplication** (`_deduplicate`) — `all-mpnet-base-v2` cosine similarity clustering; keeps longest sentence per cluster.

## Parakeet backend (`parakeet-tdt-0.6b-v3`)

Uses `parakeet-mlx` on Apple Silicon via MLX. Several non-obvious optimizations keep the GPU at 100%:

- **bfloat16 encoder**: mel is computed in float32 (required by `get_logmel`'s `mx.view` trick), then cast to bfloat16 before the encoder. Halves activation memory, allowing ~2x longer chunks.
- **Pre-computed mels**: all chunk mels are computed and `mx.eval`'d before the encode loop so the GPU never idles waiting for CPU STFT work between chunks.
- **Silence-based chunking**: splits land in genuine silence (≥300 ms) rather than at fixed intervals, so chunk boundaries never cut through speech.
- **Batched beam search** (`_patch_batched_beam`): monkey-patches `model.decode_beam` with a version that runs all beam hypotheses through the decoder LSTM and joint network in a single batched call per step, instead of one call per hypothesis. Also batches `log_softmax + argpartition` in one compute graph with one `mx.eval`.
- **mx.take for encoder frame lookup**: gathers all hypothesis encoder frames in one GPU gather op instead of N individual slices.
- **Parent-pointer token tree**: hypothesis token sequences are stored as a tree of `(AlignedToken, parent_idx)` nodes; extending a hypothesis is O(1) append. The winning sequence is reconstructed by walking the tree once at the end.
- **Incremental hash**: hypothesis identity uses `hash((parent_hash, tok_id))` — O(1) per step — instead of hashing the full token list.
- **Deferred hidden state eval**: `h_new`/`c_new` from each beam step are intentionally left lazy (not included in `mx.eval`) so MLX can fuse consecutive iterations' LSTM dispatches into one Metal submission.
- **vocab_texts lookup table**: `tok_decode` is pre-called for every token ID once before the loop; hot path does a direct list index instead of a function call per new token.
