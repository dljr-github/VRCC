# VRCC

Speak into your microphone and your words appear in the VRChat chatbox as
live captions, with translations into up to three languages underneath.
Everything runs locally: speech recognition via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) or NVIDIA's
[Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
(run as an ONNX export via [onnx-asr](https://github.com/istupakov/onnx-asr))
and machine translation via
[CTranslate2](https://github.com/OpenNMT/CTranslate2)
(NLLB / M2M100 / MADLAD models). No cloud services, no API keys.

```
You say:   "Hello, how are you today?"
Chatbox:   Hello, how are you today?
           こんにちは 今日はどうですか?
```

## Download (recommended)

From the [latest release](https://github.com/dljr-github/VRCC/releases/latest),
grab the zip that matches your hardware, unzip it anywhere, and run
`VRCC.exe`. No Python setup needed; the first-run wizard downloads the
models for you.

- **`VRCC-cuda-windows-x64.zip`** for PCs with an NVIDIA GPU (driver 570
  or newer). Near-instant captions, and it falls back to CPU by itself
  when no usable GPU is found.
- **`VRCC-windows-x64.zip`** is the CPU-only build, a much smaller
  download. Captions are identical, just a moment slower; the default
  models are sized to keep up on CPU.

GPU acceleration only supports NVIDIA cards at the moment (no AMD hardware
to test on). On AMD or Intel graphics, use the CPU build.

Installing from source (below) is only for developers who want to
contribute.

## Requirements

- Windows 10/11 (Python 3.10+ needed for source installs only)
- Optional: an NVIDIA GPU. CPU and GPU produce identical captions; the
  only difference is latency (GPU responds near-instantly, CPU a moment
  slower). The first-run wizard picks a sensible default for your machine.

## Install from source (developers)

Most users don't need this; use the
[packaged exe](#download-recommended) instead.

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

## First run

If the configured models aren't downloaded yet, a first-run wizard opens and
downloads them for you. Approximate download sizes:

| Kind | Model (default) | Size |
| ---- | --------------- | ---- |
| Speech-to-text | whisper `small` | ~480 MB |
| Translation | `nllb-600M-int8` | ~650 MB |

Other options range from whisper `tiny` (~75 MB) up to `large-v3` (~3 GB),
plus NVIDIA's `parakeet-tdt-0.6b-v3` (~690 MB, very accurate and fast),
limited to English + 24 other European languages (no Japanese/Korean/Chinese).
MT models range from `m2m100-418M-int8` (~480 MB) up to `madlad400-3b`
(~3.5 GB). Models can be added/removed later via the **Models** dialog.
See [the benchmarks below](#speech-to-text-benchmarks) for measured
accuracy and speed.

## Usage

1. Start VRChat and enable OSC: **Action menu → Options → OSC → Enabled**.
2. Start VRCC. Pick your microphone, source language and up to three
   target languages.
3. Talk. Utterances are segmented automatically (Silero VAD), transcribed,
   translated, and sent to the chatbox, throttled to stay inside VRChat's
   chatbox rate limit so continuous speech never triggers the in-game
   spam mute.
4. **Typed messages:** the text box at the bottom of the main window sends
   typed text through the same translate → chatbox path (useful when you'd
   rather not speak).
5. **Mute sync:** when enabled, muting yourself in VRChat pauses
   captioning (configurable to ignore or invert). This uses VRChat's
   OSCQuery discovery and works **only when VRChat runs on the same PC**
   (localhost); captioning itself works regardless.

### Interface language

The interface follows your Windows display language by default and can speak
18 languages (English, 日本語, 한국어, 简体中文, 繁體中文, Español, Français,
Deutsch, Italiano, Português (Brasil), Русский, Українська, Polski,
Nederlands, Türkçe, Bahasa Indonesia, Tiếng Việt, ไทย). Pick a different one
under **Settings → Simple → Language**; it applies after a restart. This
only affects VRCC's own interface; caption languages are chosen in the main
window, and adding a UI translation is a single JSON file in `vrcc/i18n/`
(copy the keys from any existing catalog).

### Performance modes

**Settings → Simple → Mode** switches between two presets:

- **Speed** (default): greedy decoding (beam 1) and short silence
  thresholds. Captions appear fastest.
- **Quality**: beam 5 STT / beam 3 MT and slightly longer silence
  thresholds. Noticeably better phrasing, a few hundred milliseconds
  slower.

Parakeet always decodes at full accuracy, so the Mode control is greyed
out while it is the active voice model. The individual
knobs (VAD timings, beam sizes, quality gates and so on) live in
**Settings → Advanced**.

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
it reaches 2.3% at 0.13 s, beating the `small` default (3.7% at 0.75 s) on
both accuracy and speed, which no other model near that latency manages.

Widening the beam (the Quality mode) is close to free on a GPU and buys
little: `base` improves from 5.7% to 4.7% for an extra 20 ms, `medium`
from 2.7% to 2.4% for 10 ms, while `large-v3-turbo` gets no better at all.
Speed is the right default, and VRCC now recommends the mode per model
from these measurements rather than leaving you to guess.

Numbers from other machines are collected in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md). If you want to add yours,
[benchmarks/README.md](benchmarks/README.md) has the two commands.

### Picking a model

A disclaimer before quoting numbers: every figure in this section comes
from the one machine benchmarked above. The accuracy numbers and
the relative speed ratios carry over to other machines; the absolute
latencies do not. On a slower CPU, expect every CPU time here to stretch
by roughly the same factor. [benchmarks/RESULTS.md](benchmarks/RESULTS.md)
collects numbers from other hardware.

On an NVIDIA GPU, keep the default `large-v3-turbo`. It matched
`large-v3` at 1.7% while being about 3.5x faster, and it handles every
language.

On CPU it depends on the language you speak:

- One of the 25 European languages: use `parakeet-tdt-0.6b-v3`. It reaches
  2.3% at 0.13 s, beating the `small` default (3.7% at 0.75 s) on accuracy
  *and* latency, and it is not close. It also detects the spoken language
  on its own within that set.
- Japanese, Korean, Chinese, or anything else outside that set: stay on
  `small`. Every whisper model that beats it needs seconds per caption on
  a CPU.

Parakeet is faster on the CPU than on the GPU (0.13 s vs 0.21 s), because
its int8 ONNX graph does not suit CUDA. So the CPU build is enough for it,
and if you play VRChat on the same PC it leaves the whole GPU to the game.
VRCC does this for you when the device is left on Auto.

The distil models lost to `large-v3-turbo` on GPU and to Parakeet and
`small` on CPU in these runs, so there's little reason to pick them.

The first-run wizard picks for your hardware and your spoken language,
which it takes from your Windows display language. On a CPU that means
Parakeet when you speak a language it covers, and `small` when you do not.
Set the spoken language to Auto and it stays with the whisper models: with
no language known ahead of time, a European-only model cannot be trusted
to cover it.

## Where things are stored

| What | Default location |
| ---- | ---------------- |
| Config | `%LOCALAPPDATA%\VRCC\VRCC\config.json` |
| Models | `%LOCALAPPDATA%\VRCC\VRCC\models\` |
| Logs | `%LOCALAPPDATA%\VRCC\VRCC\logs\vrcc-<date>-<time>.log` |

Every run writes its own log file at full debug detail, and the five newest
are kept. When reporting a problem, attach the newest file from that folder.

Run with `--portable` to keep config, models and logs in the application's
own directory instead (handy on a USB stick or for isolated installs).

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

## Model licenses

The **code** in this repository is separate from the **models** it
downloads; check that a model's license fits your use:

- **NLLB** models (default `nllb-600M-int8`, plus 1.3B/3.3B):
  **CC-BY-NC-4.0**, *non-commercial use only*.
- **M2M100** models (`m2m100-418M-int8`, `m2m100-1.2B-int8`): **MIT**, a
  good alternative if you need a permissive license.
- **MADLAD-400** (`madlad400-3b`): **Apache-2.0**.
- Whisper models: **MIT** (OpenAI weights, SYSTRAN CT2 conversions).
- **Parakeet** (`parakeet-tdt-0.6b-v3`): **CC-BY-4.0** (NVIDIA weights,
  istupakov ONNX export).

## Troubleshooting

- **"No GPU detected" / everything runs on CPU**: make sure you're running
  the CUDA build (`VRCC-cuda-windows-x64.zip`, or a source install with
  `pip install -e .[cuda]`) and that your NVIDIA driver is ≥ 570. GPU
  acceleration is NVIDIA-only for now (no AMD hardware to test on).
  CPU-only operation is normal otherwise: captions are identical, just a
  moment slower; the default models are sized to keep up on CPU.
- **GPU runs out of VRAM**: the engines fall back to CPU (int8)
  automatically for the session (same captions, higher latency); pick a
  smaller model to stay on GPU.
- **VRChat isn't showing captions**: enable OSC in VRChat (Action menu →
  Options → OSC), and check the OSC address in VRCC's settings matches
  where VRChat listens (default `127.0.0.1:9000`). If you use other OSC
  tools (e.g. a router like OSCRepeater), point VRCC at the router's port.
- **Mute sync does nothing**: it requires VRChat on the *same machine*
  (localhost mDNS/OSCQuery discovery), OSC enabled in-game, and an avatar
  that reports `MuteSelf`.
- **First transcription is slow**: model load and warm-up happen once at
  startup; captions flow at full speed afterwards.
- **Reporting a bug**: attach the newest file from the logs folder above.
  Each run writes one file at full debug detail and the five newest are
  kept, so the file with the latest timestamp is the run that went wrong.

## Development

```
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\python -m pytest tests/ -v            # unit suite
.venv\Scripts\python -m pytest tests/ -v -m integration  # downloads models
```

To benchmark the STT models on your hardware (and contribute the numbers),
see [benchmarks/README.md](benchmarks/README.md).
