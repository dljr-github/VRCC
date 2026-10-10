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
| nllb-600M-int8 | off | 48 | 78.7 | 105.2 | 48/48 | 0 | 0 | 0 | 0 | 1.00 |
| nllb-600M-int8 | on | 48 | 0.1 | 0.5 | 48/48 | 57 | 48 | 0 | 9 | 1.19 |
| nllb-1.3B-int8 | off | 48 | 159.4 | 214.6 | 48/48 | 0 | 0 | 0 | 0 | 1.00 |
| nllb-1.3B-int8 | on | 48 | 0.1 | 37.7 | 48/48 | 57 | 48 | 0 | 9 | 1.19 |

Median latency per repeat was 79.4, 78.9 and 77.2 ms (600M, off) and 162.7,
160.0 and 157.9 ms (1.3B, off); every "on" repeat had a median of 0.1 ms.
The mean MT call took 77 to 79 ms (600M) and 158 to 163 ms (1.3B) with the
feature off. The prefetch is only offered into an empty MT queue.

## What this shows

- Every final reused its speculative (48/48) and every reuse found its
  prefetch (48 hits, 0 misses), so the hit rate equals the reuse rate here.
  That is the best case: clean read speech, gaps of at least 0.6 s, no noise.
  Real chat will reuse less often, and the saving scales with that.
- When the prefetch lands, the whole MT call leaves the post-finalize path:
  about 79 ms with the 600M model and about 160 ms with the 1.3B model.
- The cost is extra MT work during speech. 9 of 57 prefetches were wasted
  (the speculative text changed before the final, so its translation was
  discarded), giving 1.19 MT calls per final instead of 1.00.
- Some "on" finals still waited: the maximum was 62.6 ms (600M) and 150.1 ms
  (1.3B), and the 1.3B p90 was 37.7 ms. This run did not establish why. Two
  candidates. The MT worker may still be busy with another utterance's job.
  Or a prefetch of the same or another utterance is still running when the
  final arrives: a running MT call cannot be preempted, so the real job waits
  for it and still counts as a hit, up to one MT call. A hit does not show
  the prefetch had finished before finalize. The bench does not record
  whether a prefetch was running when the real job was enqueued.
- The figure is the caption pipeline only. It does not include VAD
  endpointing, the speculative STT call, or the chatbox rate limit.
