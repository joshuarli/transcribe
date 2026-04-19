# Cooking Issues Transcripts

Transcripts of the [Cooking Issues](https://heritageradionetwork.org/series/cooking-issues/) podcast. Transcribed locally on Apple Silicon with mlx-whisper + pyannote diarization.

## Setup

```sh
uv sync
export HUGGING_FACE_TOKEN=hf_...  # requires access to pyannote/speaker-diarization-community-1
```

## Usage

```sh
main episodes                          # list all episodes with transcription status
main transcribe 42                     # transcribe episode 42
main transcribe 42 --backend parakeet-tdt-0.6b-v3  # use parakeet instead of mlx-whisper
main --podcast cooking-issues episodes # explicitly select a podcast (default: cooking-issues)
main transcribe 42 --speakers Dave,Nastassia         # label speakers by first appearance
main transcribe 42 --speakers Dave,Nastassia --learn # label AND save embeddings for future episodes
main transcribe 43                     # auto-identifies Dave/Nastassia from saved embeddings
main transcribe 42 --gap 1.0           # paragraph break at ≥1s silence (default: 0.5)
main retranscribe 42                   # delete cached transcript and re-transcribe
main sync                              # transcribe all episodes (parallel downloads)
main sync --dry-run                    # preview what would be downloaded/transcribed
```

## Backends

### whisper-large-v3-turbo (default)
`mlx-community/whisper-large-v3-turbo` via mlx-whisper. Processes audio in 30 s windows internally.

Runs with `temperature=0.0` (greedy, no fallback ladder) and `condition_on_previous_text=False` (prevents hallucination cascades). Both settings improve determinism and accuracy.

### parakeet-tdt-0.6b-v3
`mlx-community/parakeet-tdt-0.6b-v3` via parakeet-mlx. Better at dense speech and fast talkers.

Long episodes are split into chunks before transcription — the model can't fit a full hour in one pass. Rather than cutting at a fixed timer (which would split mid-sentence), the chunker detects genuine silence gaps (≥300 ms of low energy) and snaps each boundary to the nearest one within ±10 s of the target. Sentences are never cut mid-word.

Chunk duration is sized automatically from the Metal GPU memory budget:

| RAM  | Metal budget (75%) | Target chunk |
|------|-------------------|--------------|
| 16 GB | ~12 GB           | 45 s         |
| 32 GB | ~24 GB           | 90 s         |
| 64 GB | ~48 GB           | 180 s        |

Runs with `dtype=float32` (eliminates bfloat16 GPU non-determinism) and `Beam` decoding (beam_size=5, higher accuracy than greedy).

Note: `max_buffer_length` (16 GB on all M1 chips regardless of RAM) is a per-buffer hardware cap and is intentionally ignored here — total unified memory is the right measure.

Progress is printed per chunk so you can see where transcription is in the episode:

```
042-episode-42-raw-vegan-diet: transcribing...
  parakeet: target chunk 90s (Metal budget 24 GB)
  [1/28] 0:00 – 1:32
  [2/28] 1:32 – 3:05
  ...
```

Output goes to `transcripts/parakeet-tdt-0.6b-v3/`, separate from `transcripts/whisper-large-v3-turbo/`.

## Speaker learning

After transcribing an episode with known speakers, `--learn` saves their voice embeddings to
`cache/speakers.json`. Subsequent transcriptions automatically match diarization clusters against
stored embeddings via cosine similarity (threshold 0.75) and apply the names without any flags.

To update or add speakers at any time, re-run with `--learn`.

## One-off transcription

```sh
 transcribe audio.mp3 out.json --diarize
 transcribe audio.mp3 out.json --diarize --backend parakeet-tdt-0.6b-v3
```

Transcripts are committed to `transcripts/whisper-large-v3-turbo/` and `transcripts/parakeet-tdt-0.6b-v3/`, one `.txt` per episode per backend.
