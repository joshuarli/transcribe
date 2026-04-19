# TODO

## Contextual hinting for transcription accuracy

### mlx-whisper: `initial_prompt`

`mlx_whisper.transcribe()` accepts `initial_prompt` as a first-class parameter. Pass a prose string seeding domain vocabulary and names — e.g. *"Dave Arnold and Nastassia Lopez discuss rotovap, Searzall, and hydrocolloids on Cooking Issues."* — and Whisper's decoder treats it as prior context, biasing toward those tokens. One line of code in `_transcribe_mlx`.

Maintain the prompt in a top-level `prompt.txt` (or `config/prompt.txt`), loaded once in `_load_or_transcribe` and silently skipped if absent.


## Speaker attribution in rapid exchanges

Pyannote diarization fails on rapid back-and-forth (sub-second turns like "Yeah?" / "Uh-huh." / "Nope.") because Heritage Radio encodes both hosts into a shared mono mix (L/R correlation ~0.999). Longer monologues get attributed correctly; it's only dense dialogue that collapses to a single speaker.

### Approach: LLM re-attribution pass

After diarization, detect "suspect" regions — runs of segments all assigned the same speaker that contain dialogue markers (short sentences, question/answer pairs, rejoinders). Extract those segments with surrounding context and ask an LLM to attribute each sentence to a named speaker.

This works well for Cooking Issues specifically because Dave and Nastassia have very distinct text profiles:
- Dave: long meandering sentences, "you see", "obviously", "the thing is", food science tangents, questions to callers
- Nastassia: short responses, skeptical rejoinders ("That's gross", "I don't know"), straight-person role

Implementation sketch:
1. Heuristic to flag suspect regions: N consecutive segments, same speaker label, at least one sentence ≤ 6 words (a rejoinder) — these are the exchanges pyannote collapsed
2. Build a prompt with speaker descriptions + the flagged segment texts
3. Call Claude (claude-haiku-4-5 is fine, it's cheap and the task is easy) to return per-sentence speaker assignments
4. Patch the segment list and re-render

The LLM sees dialogue structure, verbal tics, topic expertise — things the acoustic model can't. Should recover most of the rapid-exchange attribution without any audio preprocessing.

### Parking lot: audio source separation

SpeechBrain's SepFormer (`speechbrain/sepformer-libri2mix`) can separate 2 speakers from a mono mix. Would give pyannote real acoustic contrast. Tradeoffs: meaningful extra compute per episode, bleed-through artifacts on short utterances, another large dependency. Worth revisiting if the LLM approach hits a ceiling on accuracy.
