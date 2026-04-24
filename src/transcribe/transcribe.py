#!/usr/bin/env python3

import argparse
import itertools
import json
import math
import os
import re
import sys
import time
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from transcribe.diarize import assign_speakers, diarize, extract_cluster_embeddings  # noqa: F401
from transcribe.types import (
    ParakeetCheckpoint,
    RawSegment,
    TranscriptResult,
)

DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"
PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

BACKENDS = [
    "parakeet-tdt-0.6b-v3",
    "whisper-large-v3-mlx",
]


def transcribe(
    audio_path: str,
    backend: str = "whisper-large-v3-mlx",
    model: str = DEFAULT_MODEL,
    checkpoint_path: Path | None = None,
) -> TranscriptResult:
    if backend == "parakeet-tdt-0.6b-v3":
        return _transcribe_parakeet(audio_path, checkpoint_path=checkpoint_path)
    return _transcribe_mlx(audio_path, model)


def _transcribe_mlx(audio_path: str, model: str = DEFAULT_MODEL) -> TranscriptResult:
    import mlx_whisper

    return cast(
        "TranscriptResult",
        mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model,
            word_timestamps=True,
            verbose=False,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # retry at higher temps if segment fails quality checks
            condition_on_previous_text=False,  # avoids hallucination snowball across segments
            no_speech_threshold=0.3,  # default 0.6 drops low-energy short segments (e.g. corrections after "uh")
            hallucination_silence_threshold=2.0,  # re-examine gaps >2s instead of seek-skipping over them
            # biases vocabulary toward domain terms
            initial_prompt=(
                "Cooking Issues podcast. Hosts: Dave Arnold, Nastassia Lopez."
                " Topics: food science, cocktails, culinary techniques, modernist cooking."
            ),
            language="en",  # skip language detection
            best_of=10,  # candidates sampled per temperature fallback step; default 5
        ),
    )


def _load_audio_16k(audio_path: str) -> tuple[Any, int]:
    """Load audio as float32 mono at 16 kHz."""
    import numpy as np
    import soundfile as sf

    target_sr = 16_000
    data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        new_len = int(len(data) * target_sr / sr)
        data = np.interp(np.linspace(0, len(data) - 1, new_len), np.arange(len(data)), data).astype(np.float32)
        sr = target_sr
    return data, sr


def _audio_chunks(data: Any, sr: int) -> list[tuple[int, int]]:  # noqa: ANN401
    splits = _silence_split_points(data, sr, _optimal_chunk_s())
    return list(itertools.pairwise([0, *splits, len(data)]))


def _load_checkpoint(checkpoint_path: Path | None, label: str, n: int) -> dict[int, ParakeetCheckpoint]:
    done: dict[int, ParakeetCheckpoint] = {}
    if checkpoint_path and checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = cast("ParakeetCheckpoint", json.loads(line))
                done[rec["i"]] = rec
            except Exception:
                pass
        if done:
            print(f"  {label}: resuming from chunk {max(done) + 2}/{n}")
    return done


def _run_chunks(
    chunks: list[tuple[int, int]],
    sr: int,
    infer: Callable[[int, int, int], tuple[str, list[RawSegment]]],
    checkpoint_path: Path | None,
    label: str,
    *,
    done: dict[int, ParakeetCheckpoint] | None = None,
) -> TranscriptResult:
    n = len(chunks)
    if done is None:
        done = _load_checkpoint(checkpoint_path, label, n)
    all_segments: list[RawSegment] = []
    texts: list[str] = []
    i = 0
    run_start = time.monotonic()
    try:
        for i, (s0, s1) in enumerate(chunks):
            if i in done:
                texts.append(done[i]["text"])
                all_segments.extend(done[i]["segments"])
                continue
            print(f"  [{i + 1}/{n}] {_fmt_ts(s0 / sr)} - {_fmt_ts(s1 / sr)}", flush=True)
            chunk_start = time.monotonic()
            text, segs = infer(i, s0, s1)
            elapsed = time.monotonic() - chunk_start
            audio_s = (s1 - s0) / sr
            print(
                f"    {elapsed:.1f}s ({audio_s / elapsed:.1f}x realtime) | {time.monotonic() - run_start:.0f}s total",
                flush=True,
            )
            texts.append(text)
            all_segments.extend(segs)
            if checkpoint_path:
                with checkpoint_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"i": i, "text": text, "segments": segs}) + "\n")
    except KeyboardInterrupt:
        print(f"\n  {label}: interrupted after chunk {i + 1}/{n} — checkpoint saved, re-run to resume")
        raise
    if checkpoint_path:
        checkpoint_path.unlink(missing_ok=True)
    return {"text": " ".join(t for t in texts if t), "language": "en", "segments": all_segments}


def _silence_windows(data: Any, sr: int, min_silence_s: float = 0.5) -> list[tuple[float, float]]:  # noqa: ANN401
    """Return (t_start, t_end) in seconds for every silence run ≥ min_silence_s."""
    import numpy as np

    frame_ms = 20
    frame_samples = sr * frame_ms // 1000
    n_frames = len(data) // frame_samples

    rms = np.sqrt(np.array([np.mean(data[i * frame_samples : (i + 1) * frame_samples] ** 2) for i in range(n_frames)]))
    nonzero = rms[rms > 0]
    # 0.15x rather than the 0.05x used in _silence_split_points: paragraph detection
    # needs to catch speech pauses in noisy podcast audio (no true silence floor),
    # not just dead-air between segments.
    threshold = float(np.percentile(nonzero, 75)) * 0.15 if len(nonzero) else 0.0
    is_silent = rms < threshold

    min_frames = max(1, int(min_silence_s * 1000 / frame_ms))
    windows: list[tuple[float, float]] = []
    run_start: int | None = None
    for i, silent in enumerate(is_silent):
        if silent and run_start is None:
            run_start = i
        elif not silent and run_start is not None:
            if i - run_start >= min_frames:
                windows.append((run_start * frame_ms / 1000, i * frame_ms / 1000))
            run_start = None
    if run_start is not None and n_frames - run_start >= min_frames:
        windows.append((run_start * frame_ms / 1000, n_frames * frame_ms / 1000))
    return windows


def _apply_silence_breaks(segments: list[RawSegment], silence_windows: list[tuple[float, float]]) -> list[RawSegment]:
    """Patch synthetic segment timestamps so render() breaks paragraphs at real audio silences.

    For each silence window, finds the segment boundary whose synthetic midpoint
    is nearest to the silence midpoint and widens the gap to the actual silence span.
    """
    if not segments or not silence_windows:
        return segments
    segs = [dict(s) for s in segments]
    # boundary i is between segs[i] and segs[i+1]; its synthetic time is segs[i]["end"]
    for t0, t1 in silence_windows:
        mid = (t0 + t1) / 2
        best = min(range(len(segs) - 1), key=lambda i: abs(segs[i]["end"] - mid))
        segs[best]["end"] = t0
        segs[best + 1]["start"] = t1
    return [cast("RawSegment", s) for s in segs]


def _metal_budget_gb() -> float:
    """Return the usable Metal memory budget in GB, used to size parakeet chunks.

    Reads total unified memory from MLX (or sysctl) and takes 75%.
    We deliberately skip max_buffer_length — that's a per-buffer hardware cap
    (16 GB on all M1 chips) unrelated to how much total memory is available.
    """
    try:
        import mlx.core as mx

        info = mx.device_info()
        # memory_size = total unified RAM; recommended_max_working_set_size is
        # also correct if present (it's ~75% of RAM set by the driver).
        for key in ("recommended_max_working_set_size", "memory_size"):
            if key in info:
                gb = int(info[key]) / (1024**3)
                # memory_size is raw RAM; apply 75% headroom
                if key == "memory_size":
                    gb *= 0.75
                return gb
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(out.strip()) / (1024**3) * 0.75
    except Exception:
        return 8.0  # conservative fallback


def _optimal_chunk_s() -> int:
    """Scale target chunk duration from Metal memory budget. 30 s is safe at 8 GB.

    bfloat16 halves activation memory vs float32, so we allow double the cap.
    """
    budget = _metal_budget_gb()
    return max(30, min(int(30 * budget / 8.0), 600))


def _silence_split_points(data: Any, sr: int, target_s: float) -> list[int]:  # noqa: ANN401
    """Return sample indices where the audio should be split.

    Splits land in the middle of genuine silence (≥300 ms) rather than mid-word,
    so parakeet chunk boundaries never cut through spoken audio.  For each
    target boundary we search ±10 s for the nearest silent region.
    """
    import numpy as np

    frame_ms = 20
    frame_samples = sr * frame_ms // 1000
    n_frames = len(data) // frame_samples

    rms = np.sqrt(np.array([np.mean(data[i * frame_samples : (i + 1) * frame_samples] ** 2) for i in range(n_frames)]))

    nonzero = rms[rms > 0]
    threshold = float(np.percentile(nonzero, 75)) * 0.05 if len(nonzero) else 0.0
    is_silent = rms < threshold

    # Collect midpoints of silence runs ≥ 300 ms
    min_frames = max(1, 300 // frame_ms)
    silence_mids: list[int] = []
    run_start: int | None = None
    for i, silent in enumerate(is_silent):
        if silent and run_start is None:
            run_start = i
        elif not silent and run_start is not None:
            if i - run_start >= min_frames:
                silence_mids.append((run_start + i) // 2)
            run_start = None
    if run_start is not None and n_frames - run_start >= min_frames:
        silence_mids.append((run_start + n_frames) // 2)

    target_frames = int(target_s * 1000 / frame_ms)
    search_frames = int(10_000 / frame_ms)  # ±10 s window

    splits: list[int] = []
    pos = 0
    while pos < n_frames:
        nxt = pos + target_frames
        if nxt >= n_frames:
            break
        best, best_dist = nxt, search_frames + 1
        for mid in silence_mids:
            d = abs(mid - nxt)
            if d < best_dist:
                best_dist, best = d, mid
        splits.append(best * frame_samples)
        pos = best
    return splits


def _fmt_ts(s: float) -> str:
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _patch_batched_beam(model: Any) -> None:  # noqa: ANN401
    """Replace decode_beam with a heavily optimised batched version.

    Key changes vs the original:
    - Decoder + joint called once for all beam hypotheses per step (beam_size x fewer kernels).
    - log_softmax + argpartition batched across hypotheses in one compute graph, one eval.
    - mx.take replaces N individual slice+stack for encoder frame gathering.
    - Parent-pointer tree: token append is O(1) instead of O(n); hash is incremental O(1).
    """
    from dataclasses import field as dc_field

    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from parakeet_mlx.alignment import AlignedToken
    from parakeet_mlx.parakeet import Beam, DecodingConfig
    from parakeet_mlx.tokenizer import decode as tok_decode

    @dataclass
    class _Hyp:
        score: float
        step: int
        last_token: int | None
        hidden_state: tuple[mx.array, mx.array] | None
        stuck: int
        node_idx: int  # index into nodes[]; -1 = start of sequence
        length: int  # non-blank tokens emitted so far
        _hash: int = dc_field(default=0)  # incremental O(1) hash of emitted token IDs

        def __hash__(self) -> int:
            return hash((self.step, self._hash))

    def decode_beam_batched(
        self: Any,  # noqa: ANN401
        features: mx.array,
        lengths: mx.array | None = None,
        last_token: list[int | None] | None = None,
        hidden_state: list[tuple[mx.array, mx.array] | None] | None = None,
        *,
        config: DecodingConfig | None = None,
    ) -> tuple[list[list[AlignedToken]], list[tuple[mx.array, mx.array] | None]]:
        if config is None:
            config = DecodingConfig()
        assert isinstance(config.decoding, Beam)
        beam_token = min(config.decoding.beam_size, len(self.vocabulary) + 1)
        beam_duration = min(config.decoding.beam_size, len(self.durations))
        max_candidates = round(config.decoding.beam_size * config.decoding.patience)
        vocab_size = len(self.vocabulary)
        duration_reward = config.decoding.duration_reward
        length_penalty = config.decoding.length_penalty

        time_ratio = self.time_ratio
        durations = self.durations
        max_symbols = self.max_symbols
        vocab_texts = [tok_decode([i], self.vocabulary) for i in range(vocab_size)]

        n_batch, seq_len, *_ = features.shape
        if lengths is None:
            lengths = mx.array([seq_len] * n_batch)
        if last_token is None:
            last_token = [None] * n_batch
        if hidden_state is None:
            hidden_state = [None] * n_batch

        results: list[list[AlignedToken]] = []
        results_hidden: list[tuple[mx.array, mx.array] | None] = []

        for b in range(n_batch):
            feature = features[b : b + 1]  # [1, S, d_model]
            length = int(lengths[b])

            # Shared token tree: list of (AlignedToken, parent_node_idx).
            # Appending is O(1); path is reconstructed once from the winner at the end.
            nodes: list[tuple[AlignedToken, int]] = []

            finished: list[_Hyp] = []
            beam: list[_Hyp] = [
                _Hyp(
                    score=0.0,
                    step=0,
                    last_token=last_token[b],
                    hidden_state=hidden_state[b],
                    stuck=0,
                    node_idx=-1,
                    length=0,
                )
            ]

            while len(finished) < max_candidates and beam:
                # --- batch decoder across all hypotheses ---
                # numpy→mx.array is faster than mx.array from a Python list of lists
                token_ids_np = np.array(
                    [[h.last_token if h.last_token is not None else 0] for h in beam], dtype=np.int32
                )
                embedded = self.decoder.prediction["embed"](mx.array(token_ids_np))
                none_mask = mx.array(
                    np.array([[[0.0 if h.last_token is None else 1.0]] for h in beam], dtype=np.float32)
                )
                embedded = embedded * none_mask

                ref = next((h.hidden_state[0] for h in beam if h.hidden_state is not None), None)
                if ref is None:
                    hc_in = None
                else:
                    hs = [h.hidden_state[0] if h.hidden_state else mx.zeros_like(ref) for h in beam]
                    cs = [h.hidden_state[1] if h.hidden_state else mx.zeros_like(ref) for h in beam]
                    hc_in = (mx.concatenate(hs, axis=1), mx.concatenate(cs, axis=1))

                dec_out, (h_new, c_new) = self.decoder.prediction["dec_rnn"](embedded, hc_in)
                dec_out = dec_out.astype(feature.dtype)
                h_new = h_new.astype(feature.dtype)
                c_new = c_new.astype(feature.dtype)

                # --- batch joint: mx.take gathers all encoder frames in one op ---
                enc_frames = mx.take(feature[0], mx.array(np.array([h.step for h in beam], dtype=np.int32)), axis=0)[
                    :, None, :
                ]
                joint_out = self.joint(enc_frames, dec_out)  # [n, 1, 1, vocab+durations]

                # Batch softmax + topk — all fused into one compute graph, one eval
                tok_lp_all = nn.log_softmax(joint_out[:, 0, 0, : vocab_size + 1], -1)
                dur_lp_all = nn.log_softmax(joint_out[:, 0, 0, vocab_size + 1 :], -1)
                tok_k_all = mx.argpartition(tok_lp_all, -beam_token, axis=-1)[:, -beam_token:]
                dur_k_all = mx.argpartition(dur_lp_all, -beam_duration, axis=-1)[:, -beam_duration:]

                # h_new/c_new stay lazy so MLX can fuse them into the next iteration's LSTM dispatch
                mx.eval(tok_lp_all, dur_lp_all, tok_k_all, dur_k_all)

                tok_lp_lists = cast("list[list[float]]", tok_lp_all.tolist())
                dur_lp_lists = cast("list[list[float]]", dur_lp_all.tolist())
                tok_k_lists = cast("list[list[int]]", tok_k_all.tolist())
                dur_k_lists = cast("list[list[int]]", dur_k_all.tolist())

                # --- expand candidates (pure Python from here) ---
                candidates: dict[int, _Hyp] = {}
                for j, hyp in enumerate(beam):
                    dec_hidden_j = (h_new[:, j : j + 1, :], c_new[:, j : j + 1, :])
                    tok_lp_list = tok_lp_lists[j]
                    dur_lp_list = dur_lp_lists[j]

                    for tok in tok_k_lists[j]:
                        is_blank = tok == vocab_size
                        for dec in dur_k_lists[j]:
                            dur = durations[dec]
                            stuck = 0 if dur != 0 else hyp.stuck + 1
                            if max_symbols is not None and stuck >= max_symbols:
                                step = hyp.step + 1
                                stuck = 0
                            else:
                                step = hyp.step + dur

                            if is_blank:
                                node_idx = hyp.node_idx
                                new_hash = hyp._hash
                                new_length = hyp.length
                            else:
                                node_idx = len(nodes)
                                nodes.append(
                                    (
                                        AlignedToken(
                                            id=tok,
                                            start=hyp.step * time_ratio,
                                            duration=dur * time_ratio,
                                            confidence=math.exp(tok_lp_list[tok] + dur_lp_list[dec]),
                                            text=vocab_texts[tok],
                                        ),
                                        hyp.node_idx,
                                    )
                                )
                                new_hash = hash((hyp._hash, tok))
                                new_length = hyp.length + 1

                            new_hyp = _Hyp(
                                score=(
                                    hyp.score
                                    + tok_lp_list[tok] * (1 - duration_reward)
                                    + dur_lp_list[dec] * duration_reward
                                ),
                                step=step,
                                last_token=hyp.last_token if is_blank else tok,
                                hidden_state=hyp.hidden_state if is_blank else dec_hidden_j,
                                stuck=stuck,
                                node_idx=node_idx,
                                length=new_length,
                                _hash=new_hash,
                            )
                            key = hash(new_hyp)
                            if key in candidates:
                                other = candidates[key]
                                mx_ = max(other.score, new_hyp.score)
                                merged = mx_ + math.log(math.exp(other.score - mx_) + math.exp(new_hyp.score - mx_))
                                if new_hyp.score > other.score:
                                    candidates[key] = new_hyp
                                candidates[key].score = merged
                            else:
                                candidates[key] = new_hyp

                finished.extend(h for h in candidates.values() if h.step >= length)
                beam = sorted(
                    (h for h in candidates.values() if h.step < length),
                    key=lambda x: x.score,
                    reverse=True,
                )[: config.decoding.beam_size]

            finished = finished + beam
            if not finished:
                results.append([])
                results_hidden.append(hidden_state[b])
            else:
                best = max(
                    finished,
                    key=lambda x: x.score / (max(1, x.length) ** length_penalty),
                )
                # Reconstruct token list by walking the parent-pointer tree
                tokens: list[AlignedToken] = []
                idx = best.node_idx
                while idx >= 0:
                    token, idx = nodes[idx]
                    tokens.append(token)
                tokens.reverse()
                results.append(tokens)
                results_hidden.append(best.hidden_state)

        return results, results_hidden

    model.decode_beam = types.MethodType(decode_beam_batched, model)


def _transcribe_parakeet(audio_path: str, checkpoint_path: Path | None = None) -> TranscriptResult:
    import mlx.core as mx
    from parakeet_mlx import from_pretrained
    from parakeet_mlx.alignment import sentences_to_result, tokens_to_sentences
    from parakeet_mlx.audio import get_logmel
    from parakeet_mlx.parakeet import Beam, DecodingConfig

    chunk_s = _optimal_chunk_s()
    print(f"  parakeet: target chunk {chunk_s}s (Metal budget {_metal_budget_gb():.0f} GB)")

    model = from_pretrained(PARAKEET_MODEL)
    _patch_batched_beam(model)
    decoding_config = DecodingConfig(decoding=Beam())

    data, sr = _load_audio_16k(audio_path)

    chunks = _audio_chunks(data, sr)
    done = _load_checkpoint(checkpoint_path, "parakeet", len(chunks))

    # Pre-compute all mels upfront so the GPU never idles waiting for CPU mel work.
    # get_logmel requires float32 (uses mx.view to split complex STFT output);
    # cast to bfloat16 afterward so the encoder runs in bfloat16.
    # Typical podcast: ~3 MB per chunk mel, well within spare RAM.
    print("  parakeet: pre-computing mel spectrograms...", flush=True)
    chunk_mels: list[mx.array | None] = []
    for i, (s0, s1) in enumerate(chunks):
        if i in done:
            chunk_mels.append(None)
        else:
            chunk_mels.append(get_logmel(mx.array(data[s0:s1]), model.preprocessor_config).astype(mx.bfloat16))
    mx.eval(*[m for m in chunk_mels if m is not None])

    def _run_chunk(i: int, s0: int, _s1: int) -> tuple[str, list[RawSegment]]:
        mel = chunk_mels[i]
        assert mel is not None
        features, out_lengths = model.encoder(mel)
        mx.eval(features, out_lengths)
        tokens, _ = model.decode(features, out_lengths, config=decoding_config)
        offset = s0 / sr
        aligned = sentences_to_result(tokens_to_sentences(tokens[0], decoding_config.sentence))
        segs: list[RawSegment] = [
            {"start": seg.start + offset, "end": seg.end + offset, "text": " " + seg.text.strip()}
            for seg in aligned.sentences
        ]
        return aligned.text, segs

    result = _run_chunks(chunks, sr, _run_chunk, checkpoint_path, "parakeet", done=done)
    result["segments"] = _apply_silence_breaks(result["segments"], _silence_windows(data, sr))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("output")
    parser.add_argument("--backend", choices=BACKENDS, default=BACKENDS[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--diarize", action="store_true")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    if not Path(args.audio).exists():
        sys.exit(f"Audio file not found: {args.audio}")

    print(f"Loading {args.backend} backend...", file=sys.stderr)
    if args.backend == "whisper-large-v3-mlx":
        import mlx_whisper  # noqa: F401 — triggers model cache check before progress message
    print(f"Transcribing {args.audio}...", file=sys.stderr)
    result = transcribe(args.audio, backend=args.backend, model=args.model)
    segments = result["segments"]

    if args.diarize:
        hf_token = args.hf_token or os.environ.get("HUGGING_FACE_TOKEN")
        if not hf_token:
            sys.exit("Set --hf-token or $HUGGING_FACE_TOKEN for diarization.")
        print("Loading pyannote pipeline...", file=sys.stderr)
        import torch  # noqa: F401

        print("Diarizing...", file=sys.stderr)
        annotation, _waveform, _sr, _pipeline = diarize(args.audio, hf_token)
        segments = assign_speakers(segments, annotation)

    Path(args.output).write_text(
        json.dumps(
            {
                "text": result["text"],
                "language": result.get("language"),
                "segments": segments,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done. Wrote {len(segments)} segments to {args.output}.", file=sys.stderr)


if __name__ == "__main__":
    main()
