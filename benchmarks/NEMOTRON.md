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

## First look (cloud container, not a real machine)

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
