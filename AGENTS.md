# Agents

## Structure

```
src/transcribe/
  __init__.py    — empty
  main.py        — pipeline CLI: feed fetch, audio download, transcribe, render
  pipeline.py    — episode pipeline: download, transcribe, diarize, render; checkpoint logic
  transcribe.py  — transcription backends (mlx-whisper, parakeet, cohere); standalone CLI
  diarize.py     — pyannote speaker diarization; assign_speakers, extract_cluster_embeddings
  speakers.py    — speaker embedding save/load/match (cosine similarity)
  episode.py     — Episode TypedDict construction
  feed.py        — RSS feed fetch and episode parsing
  podcasts.py    — Podcast registry and config
  extract.py     — LLM-based culinary content extraction
  denoise.py     — heuristic transcript cleanup before extraction
  http.py        — audio download with cancel support
  types.py       — shared TypedDicts (Episode, Segment, RawSegment, StoredTranscript, …)
pyproject.toml   — deps, build (uv_build, module-root=src), console script (main → transcribe.main:main)
transcripts/     — committed output: transcripts/<backend>/<slug>.txt
cache/           — gitignored: feeds, audio, intermediate JSON
```

## Pipeline

HRN + Acast RSS → audio download → transcription backend → (optional) pyannote diarization → rendered text

Per-episode flow: `cache/audio/<slug>.opus` → `cache/<backend>/<slug>.raw.json` → `cache/<backend>/<slug>.json` → `transcripts/<backend>/<slug>.txt`

## Key details

- All heavy imports (`mlx_whisper`, `torch`, `pyannote`, `transformers`) are lazy — inside function bodies — so `main episodes` starts instantly.
- `transcribe.py` exposes a `BACKENDS` list that drives all `--backend` choices in `main.py`.
- Diarization is in `diarize.py`, re-exported from `transcribe.py` for backward compatibility.
- Diarization uses `exclusive_speaker_diarization` from `DiarizeOutput` (community-1 API), which has no overlapping turns — better for per-segment speaker assignment.
- Speaker assignment: for each segment, find the pyannote turn with greatest temporal overlap (two-pointer O(n+m) walk).
- Paragraphs split on ≥0.5s silence gaps. Format: `[M:SS | SPEAKER_N] text`.
- Feed slugs: `{NNN}-{title-slug}` (e.g. `042-episode-42-raw-vegan-diet`).
- Acast feed fetched with ETag conditional request; HRN always re-fetched.

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

## Cohere backends (`cohere-transcribe-03-2026`, `cohere-transcribe-03-2026-mlx`)

Neither Cohere variant supports native timestamp output (both are CTC-style models — they output text only, no timing). Paragraph breaks are produced in two passes:

1. **Inference pass**: audio is split into large silence-bounded chunks (same ≥300 ms / `_silence_split_points` logic as parakeet). Each chunk is transcribed and the resulting text is sentence-split via regex (`(?<=[.!?])\s+(?=[A-Z])`), with timestamps distributed proportionally by character count within the chunk's time span. These timestamps are synthetic.
2. **Silence pass**: after all chunks are transcribed, `_silence_windows` scans the full audio for runs of silence ≥500 ms. `_apply_silence_breaks` then finds the segment boundary whose synthetic timestamp is nearest each silence midpoint and patches that boundary to span the actual silence window. `render`'s gap check then sees real silence durations instead of synthetic zeros.

Whisper avoids all of this because it is an encoder-decoder that explicitly predicts `<|timestamp|>` tokens — each segment comes with a genuine start/end from the model itself.

Both backends support checkpoint/resume: each chunk result is written to a `.ckpt.json` JSONL file immediately after transcription, so `^C` mid-episode resumes from the last completed chunk.

`cohere-transcribe-03-2026` uses HuggingFace Transformers on MPS (PyTorch GPU).  
`cohere-transcribe-03-2026-mlx` uses `mlx-speech`'s `CohereAsrModel` with the `mlx-int8` quantized weights from `mlx-community/cohere-transcribe-03-2026-mlx-8bit`. The weights are downloaded via `huggingface_hub.snapshot_download` on first use.
