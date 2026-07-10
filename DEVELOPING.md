# Developing VRCC

Building from source, tests, benchmarks and packaging. If you just want
captions in VRChat, the [README](README.md) covers everything.

## Install from source

Most users don't need this; use the
[packaged exe](README.md#download) instead. A source install needs
Python 3.10+ on Windows 10/11.

CPU-only:

```
python -m venv .venv
.venv\Scripts\pip install -e .
```

With NVIDIA GPU acceleration (bundles the CUDA 12 cuBLAS + cuDNN runtimes
as pip wheels, so no system CUDA install is needed):

```
.venv\Scripts\pip install -e .[cuda]
.venv\Scripts\pip install "onnxruntime-gpu>=1.21,<1.23"
```

The second command swaps in the GPU build of onnxruntime so the Parakeet
model also runs on the GPU (whisper models use CTranslate2's own CUDA
path). It must run **after** the first: `onnxruntime-gpu` installs the
same `onnxruntime` package as the CPU wheel pulled in by faster-whisper,
and the one installed last wins. Keep it below 1.23. Newer builds are
built against CUDA 13, `[cuda]` ships the CUDA 12 wheels that CTranslate2
needs, and with mismatched builds that model silently runs on the CPU
instead. Skipping the second command entirely is also fine; Parakeet
then runs on the CPU.

> **GPU note:** CUDA use requires an NVIDIA driver of version **570 or
> newer**. On older drivers the app detects this at startup and falls back
> to CPU automatically.

Run the app:

```
.venv\Scripts\vrcc              # or: .venv\Scripts\python -m vrcc.cli
```

## Speech-to-text benchmarks

Numbers from `tools/bench_stt.py`: 100
[LibriSpeech](https://www.openslr.org/12/) test-clean utterances run
through the same engine path the app uses, at default settings (Speed
mode). Machine: Windows 11, Ryzen 9 9950X3D, RTX 5090, driver 610.62.
WER is word error rate on English read speech after Whisper-style
normalization, so it says nothing about other languages. It is scored with
VRCC's quality gates open, so it measures what the model heard rather than
what the app chose to suppress. Latency is the median time to transcribe
one utterance. Whisper models run float16 on GPU and int8 on CPU, which is
why their two WER columns differ slightly; Parakeet is int8 either way.

| Model | Size | WER (GPU) | WER (CPU) | GPU latency | CPU latency |
| ----- | ---- | --------- | --------- | ----------- | ----------- |
| `tiny` | 75 MB | 7.4% | 7.9% | 0.03 s | 0.13 s |
| `base` | 145 MB | 5.7% | 5.9% | 0.04 s | 0.25 s |
| `small` | 484 MB | 3.7% | 3.7% | 0.09 s | 0.74 s |
| `medium` | 1.5 GB | 2.7% | 2.6% | 0.17 s | 2.41 s |
| `large-v3` | 3.1 GB | 1.7% | 1.8% | 0.24 s | 3.90 s |
| `large-v3-turbo` | 1.6 GB | 1.7% | 1.6% | 0.07 s | 2.81 s |
| `distil-large-v3.5` | 1.5 GB | 2.4% | 2.3% | 0.06 s | 2.78 s |
| `distil-small.en` | 332 MB | 4.0% | 4.0% | 0.04 s | 0.64 s |
| `parakeet-tdt-0.6b-v3` | 690 MB | 2.3% | 2.3% | 0.21 s | **0.13 s** |

One result is worth reading twice. Parakeet is **faster on the CPU than on
the GPU** (0.13 s vs 0.21 s; its int8 ONNX graph does not suit CUDA), which
is why VRCC runs it on the CPU when the device is left on Auto. On the CPU
it reaches 2.3% at 0.13 s, beating the `small` default (3.7% at 0.74 s) on
both accuracy and speed, which no other model near that latency manages.

Widening the beam (the Quality mode) is close to free on a GPU and buys
little: `base` improves from 5.7% to 4.7% for an extra 20 ms, `medium`
from 2.7% to 2.4% for 10 ms, while `large-v3-turbo` gets no better at all.
Speed is the right default, and VRCC now recommends the mode per model
from these measurements rather than leaving you to guess.

Numbers from other machines are collected in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md), including the per-model
Speed vs Quality table. If you want to add yours,
[benchmarks/README.md](benchmarks/README.md) has the two commands.

## Development

```
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\python -m pytest tests/ -v            # unit suite
.venv\Scripts\python -m pytest tests/ -v -m integration  # downloads models
```

To benchmark the STT models on your hardware (and contribute the numbers),
see [benchmarks/README.md](benchmarks/README.md).

## End-to-end smoke test

`scripts/smoke_e2e.py` runs the real VAD → STT → MT → chatbox pipeline over
a WAV file (no GUI, no UDP; the chatbox client prints to stdout) and
reports per-stage timings:

```
.venv\Scripts\python scripts\smoke_e2e.py path\to\speech.wav --target Japanese
```

The same path runs as an integration test:
`set VRCC_E2E_WAV=path\to\speech.wav` then
`.venv\Scripts\python -m pytest tests/ -v -m integration`.

## Building a standalone exe

A PyInstaller one-folder spec lives in `packaging/vrcc.spec`:

```
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\pyinstaller packaging\vrcc.spec --noconfirm --distpath dist
```

This produces `dist\VRCC\VRCC.exe` (windowed, no console). Models are not
bundled; the exe downloads them on first run exactly like the source
install, and shares the same default models directory. Distribute the whole
`dist\VRCC` folder; pair the exe with `--portable` if you want a fully
self-contained folder.

## Contributing a UI translation

Adding a UI translation is a single JSON file in `vrcc/i18n/`: copy the
keys from any existing catalog and translate the values.
`tests/test_i18n.py` checks that every catalog carries the exact set of
strings the app uses.

## Regenerating the walkthrough

The screenshots under `assets/images/` are rendered by
`tools/walkthrough_shots.py`, which builds the app's real widgets offscreen
and feeds them a staged caption session. Everything that shapes the pixels
(hardware verdict, config values, caption text, timestamps, render scale)
is pinned in the script; on one machine, two consecutive runs produce
byte-identical PNGs:

```
.venv\Scripts\python tools\walkthrough_shots.py
```

The video is a Remotion project in `tools/walkthrough`. Its public dir
points at `assets/images/`, so it reads those screenshots directly:

```
cd tools\walkthrough
npm install
npm run render:gif
```

`render:gif` overwrites `assets/walkthrough.gif`, the file the README
embeds. It renders at 800x450 and keeps every third frame so the GIF stays
small. `npm run render:mp4` writes a full-rate H.264 preview to
`tools/walkthrough/out/` (gitignored), and `npm run studio` opens the
live editor.
