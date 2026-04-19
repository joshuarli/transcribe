# Agents

## Structure

```
src/transcribe/
  __init__.py    — empty
  main.py        — pipeline CLI: feed fetch, audio download, transcribe, render
  transcribe.py  — mlx-whisper transcription + pyannote diarization; also a standalone CLI
pyproject.toml   — deps, build (uv_build, module-root=src), console script (run → transcribe.main:main)
transcripts/     — committed output: transcripts/mlx-diarize/<slug>.txt
cache/           — gitignored: feeds, audio, intermediate JSON
```

## Pipeline

HRN + Acast RSS → audio download → mlx-whisper (large-v3-turbo) → pyannote diarization (speaker-diarization-community-1) → rendered text

Per-episode flow: `cache/audio/<slug>.mp3` → `cache/mlx-diarize/<slug>.json` → `transcripts/mlx-diarize/<slug>.txt`

## Key details

- `transcribe.py` is called directly from `main.py` (no subprocess). All heavy imports (`mlx_whisper`, `torch`, `pyannote`) are lazy — inside function bodies — so `run episodes` starts instantly.
- Diarization uses `exclusive_speaker_diarization` from `DiarizeOutput` (community-1 API), which has no overlapping turns — better for per-segment speaker assignment.
- Speaker assignment: for each whisper segment, find the pyannote turn with greatest temporal overlap.
- Audio is loaded via `torchaudio` before passing to pyannote to avoid MP3 frame-boundary sample count mismatches.
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
