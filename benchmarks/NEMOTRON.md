# Nemotron 3.5 ASR Streaming

`tools/bench_nemotron.py` runs NVIDIA's Nemotron 3.5 ASR Streaming 0.6B
(40 language-locales, OpenMDW-1.1) through sherpa-onnx, using sherpa's
int8 export, on the same LibriSpeech utterances as `tools/bench_stt.py`.
It is a candidate STT backend, not an app model yet.

Each utterance is decoded twice:

- **offline**: the whole clip at once. Compare with the latency column in
  [README.md](README.md).
- **live**: the clip fed in 32 ms pieces, decoding as it goes, as a
  streaming engine in VRCC would see the microphone. **finalize** is the
  time from the last piece to the final text, which is the wait after you
  stop talking. **live RTF** is the total decode work over the clip.

```powershell
pip install sherpa-onnx
python tools/bench_nemotron.py --chunks 160,560,1120 --threads 4
python tools/bench_nemotron.py --chunks 560 --max-seconds 2.6 --utterances 400
python tools/bench_nemotron.py --chunks 560 --samples
```

`--samples` decodes the multilingual clips inside the model package, once
with the clip's language forced and once on auto. The PyPI sherpa-onnx
wheel is CPU-only.

## LibriSpeech, cloud container CPU

Run on 2026-11-01 in a 4-core shared cloud container (no GPU), 100
LibriSpeech test-clean utterances (whole clips, median about 7 s), English
forced, 4 threads. The baseline rows come from `tools/bench_stt.py
--device cpu` on the same utterances and machine.

| model | WER | median latency (whole clip) | RTF | finalize after last audio |
|---|---|---|---|---|
| Nemotron 560 ms | 3.6% | 1.969 s offline | 0.329 live | **0.309 s** median, 0.363 s p90 |
| Nemotron 1120 ms | 3.8% | 1.424 s offline | 0.230 live | 0.396 s median, 0.462 s p90 |
| parakeet-tdt-0.6b-v3 | 2.4% | 0.422 s | 0.067 | n/a |
| small | 3.9% | 1.542 s | 0.215 | n/a |
| large-v3-turbo | 1.9% | 4.778 s | 0.645 | n/a |

- Live decoding gives the same text as offline decoding (same WER).
- Once you stop talking, the 560 ms model has the caption in about 0.31 s,
  quicker than Parakeet's 0.42 s, and it stays flat as clips get longer,
  because the work was done while you spoke. But it spends about five
  times Parakeet's total CPU (RTF 0.33 against 0.067), and its English WER
  is worse (3.6% against 2.4%). Against Whisper `small`, the current
  default, it is close on WER with a far shorter wait.
- For English on CPU, Parakeet stays the better choice. Nemotron's case is
  the languages Parakeet lacks (Japanese, Korean, Chinese, Arabic, Hindi,
  Vietnamese and others) and the shorter wait; the next measurement should
  be WER on those languages.

## First look at the package's clips

Run on 2026-11-01 in a shared cloud container with 4 threads, so the
timings only compare against each other. LibriSpeech could not be
downloaded there, so this used the package's own clips.

- Streaming pays off: on 4 to 5 second French and Vietnamese clips,
  finalize took 0.33 to 0.51 s against 1.5 to 1.9 s for the offline
  decode of the same clip, with the same text.
- Forcing the language matters. On auto, the German clip lost its first
  half ("hat ein Ende, nur die Wurst hat zwei" became "Nur die Wurst hat
  zwei") and the Japanese clip started with a Korean syllable. Forced
  language fixed both.
- Korean came out with no spaces between words, and the Japanese clip had
  two wrong kanji even when forced.
- With 0.5 s of tail padding the 560 ms model cut the last word of the
  Vietnamese clip. Two chunks of padding (1.12 s) did not, so that is the
  default.
