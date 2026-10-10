# Pretranslate benchmark

Does translating a speculative transcription ahead of time shorten the wait
between finalize and the chatbox submission?

Measured with `tools/bench_pretranslate.py` on an RTX 5090 and a Ryzen 9
9950X3D, Windows 11. The GPU was not guaranteed idle: it showed 1% load and
3 GB in use before the run, and nothing else was started during it.

    python tools/bench_pretranslate.py --stt large-v3-turbo --mt nllb-600M-int8
    python tools/bench_pretranslate.py --stt large-v3-turbo --mt nllb-1.3B-int8

Each run feeds 15 LibriSpeech test-clean utterances of 6 s or less (61 s of
speech, evenly spaced picks via `bench_stt.load_utterances`) joined by random
0.6 to 1.5 s gaps, through the real threaded `Pipeline` at real-time pace.
Whisper large-v3-turbo runs on CUDA, the target is Japanese, and only the
chatbox and the audio source are fakes. The two arms use identical audio and
alternate order over 3 repeats (off/on, on/off, off/on). The 15 utterances
split into 16 finals, so each arm has 48 finals.

Latency is `SessionStats.latency_samples`, the pipeline's own finalize to
chatbox submission window. The bench swaps the tracker's clock for
`perf_counter`, because `time.monotonic` ticks at 15.6 ms on Windows.

## Results

| MT model | arm | finals | median ms | p90 ms | spec reuse | offered | hit | miss | wasted | MT calls/final |
|---|---|---|---|---|---|---|---|---|---|---|
| nllb-600M-int8 | off | 48 | 81.9 | 104.6 | 48/48 | 0 | 0 | 48 | 0 | 1.00 |
| nllb-600M-int8 | on | 48 | 0.1 | 0.1 | 48/48 | 57 | 48 | 0 | 9 | 1.19 |
| nllb-1.3B-int8 | off | 48 | 147.6 | 193.2 | 48/48 | 0 | 0 | 48 | 0 | 1.00 |
| nllb-1.3B-int8 | on | 48 | 0.1 | 63.9 | 48/48 | 57 | 48 | 0 | 9 | 1.19 |

Median latency per repeat was 82.2, 82.6 and 77.5 ms (600M, off) and 148.4,
146.3 and 145.8 ms (1.3B, off); every "on" repeat had a median of 0.1 ms.
The median MT call took 82 ms (600M) and 146 ms (1.3B) with the feature off.

## What this shows

- Every final reused its speculative (48/48) and every reuse found its
  prefetch (48 hits, 0 misses), so the hit rate equals the reuse rate here.
  That is the best case: clean read speech, gaps of at least 0.6 s, no noise.
  Real chat will reuse less often, and the saving scales with that.
- When the prefetch lands, the whole MT call leaves the post-finalize path:
  about 82 ms with the 600M model and about 148 ms with the 1.3B model.
- The cost is extra MT work during speech. 9 of 57 prefetches were wasted
  (the speculative text changed before the final, so its translation was
  discarded), giving 1.19 MT calls per final instead of 1.00.
- With the 1.3B model the "on" p90 is 63.9 ms and the maximum 169 ms, so a
  few finals still waited. This run did not establish why. The likeliest
  cause is the MT worker still busy with another utterance's job, but that
  was not measured.
- The figure is the caption pipeline only. It does not include VAD
  endpointing, the speculative STT call, or the chatbox rate limit.
