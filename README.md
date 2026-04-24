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
main transcribe 42 --redo              # delete cached transcript and re-transcribe from scratch
main sync                              # transcribe all episodes (parallel downloads)
main sync --dry-run                    # preview what would be downloaded/transcribed
main extract 42                        # extract culinary content from transcript via local Llama
main extract 42 --model haiku          # use Claude Haiku instead of local Llama
```

## Backends

### whisper-large-v3-mlx (default)
`mlx-community/whisper-large-v3-mlx` via mlx-whisper. Processes audio in 30 s windows internally.

Key settings:
- `temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` — fallback ladder; retries with higher temperature on low-confidence windows instead of hallucinating
- `condition_on_previous_text=False` — prevents hallucination cascades where one bad prediction poisons subsequent windows
- `no_speech_threshold=0.3` — more aggressive silence filtering (default is 0.6)
- `hallucination_silence_threshold=2.0` — suppress windows where the model produces text over a long silence
- `best_of=10` — sample 10 candidates and pick the best; improves recall of disfluencies and self-corrections
- `initial_prompt` — seeds domain vocabulary (rotovap, Searzall, hydrocolloids, etc.) to bias the decoder toward podcast-specific tokens

Output goes to `transcripts/whisper-large-v3-mlx/`.

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

Progress is printed per chunk so you can see where transcription is in the episode:

```
042-episode-42-raw-vegan-diet: transcribing...
  parakeet: target chunk 90s (Metal budget 24 GB)
  [1/28] 0:00 – 1:32
  [2/28] 1:32 – 3:05
  ...
```

Output goes to `transcripts/parakeet-tdt-0.6b-v3/`.

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

Transcripts are committed to `transcripts/<backend>/`, one `.txt` per episode per backend.
