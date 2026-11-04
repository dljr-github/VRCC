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
LibriSpeech test-clean utterances (whole clips, 7.5 s on average), English
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

## Short utterances, cloud container CPU

Chatbox lines are short, so this reruns English on the 200 shortest
LibriSpeech test-clean clips (2.6 s or less), 4 threads, on 2026-11-03 (a
different CPU from the table above; compare within this table).

| model | WER | wait after speech (median) | RTF |
|---|---|---|---|
| Nemotron 560 ms | 5.3% | 0.486 s finalize (p90 0.571 s) | 0.607 live |
| parakeet-tdt-0.6b-v3 | 5.4% | 0.315 s whole clip | 0.141 |
| small | 5.3% | 1.755 s whole clip | 0.785 |

On short lines the streaming advantage is gone: Parakeet transcribes the
whole clip faster than Nemotron finalizes its tail, at a quarter of the
CPU. The two-chunk tail padding (1.12 s of silence) is most of Nemotron's
work on a clip this short.

## Japanese, Korean and Chinese (FLEURS), cloud container CPU

`tools/bench_fleurs.py`, 60 FLEURS dev sentences per language (one
recording each, 11 to 13 s on average), language forced, 4 threads, quality
gates open, on 2026-11-03. CER ignores punctuation, spaces and case.
"Wait" is how long after the end of the audio the text is ready: whole
clip for the VRCC models, finalize for Nemotron.

| lang | model | CER | wait after speech (median) | RTF |
|---|---|---|---|---|
| ja | sense-voice-small | 6.9% | 0.559 s | 0.047 |
| ja | small | 14.3% | 2.750 s | 0.226 |
| ja | large-v3-turbo | 5.4% | 8.009 s | 0.631 |
| ja | Nemotron 560 ms | 13.2% | 0.433 s | 0.462 |
| ko | sense-voice-small | 7.9% | 0.652 s | 0.054 |
| ko | small | 8.6% | 2.821 s | 0.225 |
| ko | large-v3-turbo | 5.8% | 7.872 s | 0.632 |
| ko | Nemotron 560 ms | 9.6% | 0.427 s | 0.462 |
| zh | sense-voice-small | 7.2% | 0.559 s | 0.055 |
| zh | small | 13.1% | 2.756 s | 0.258 |
| zh | large-v3-turbo | 8.8% | 7.914 s | 0.737 |
| zh | Nemotron 560 ms | 20.1% | 0.476 s | 0.499 |

- SenseVoice Small, already in VRCC, beats Nemotron on all three
  languages, at about a tenth of the CPU and a wait only 0.1 s longer.
- Nemotron's Chinese is Simplified, so the 20% is not a script problem.
  Some of it is numbers spelled out ("三个世纪" against "3个世纪"), which
  CER counts, but most is wrong homophones and names ("独者" for "读者",
  "博克" for "博客").
- On CPU, Nemotron 3.5 Streaming does not earn a place: Parakeet is better
  for English and SenseVoice for CJK, both cheaper.

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
