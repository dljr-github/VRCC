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

VRCC is built to be plug and play. Run it and the first-run wizard sizes
up your machine, picks the models, and sets the performance mode for you;
every recommendation traces back to a measured benchmark run rather than
a guess. The tuning knobs are still there in Settings, but you should not
need them.

![VRCC walkthrough](assets/walkthrough.gif)

## Download

From the [latest release](https://github.com/dljr-github/VRCC/releases/latest),
grab the zip that matches your hardware, unzip it anywhere, and run
`VRCC.exe`. Windows 10/11; no Python setup needed. The first-run wizard
downloads the models for you.

- The CUDA zip (its name starts with `VRCC-cuda-windows-x64`) for PCs
  with an NVIDIA GPU (driver 570 or newer). Near-instant captions, and it
  falls back to CPU by itself when no usable GPU is found.
- The CPU zip (`VRCC-windows-x64`) is a much smaller download. Captions
  are identical, just a moment slower; the default models are sized to
  keep up on CPU.

GPU acceleration only supports NVIDIA cards at the moment (no AMD hardware
to test on). On AMD or Intel graphics, use the CPU build.

Installing from source is only for developers who want to contribute; see
[DEVELOPING.md](DEVELOPING.md).

## First run

If the configured models aren't downloaded yet, a first-run wizard opens,
picks models for your hardware and your spoken language (taken from your
Windows display language), and downloads them:

| Your machine | Speech-to-text | Translation | Download |
| ------------ | -------------- | ----------- | -------- |
| NVIDIA GPU with 16 GB+ VRAM | `large-v3-turbo` | `nllb-1.3B-int8` | ~3 GB |
| Otherwise, a language Parakeet covers | `parakeet-tdt-0.6b-v3` | `nllb-600M-int8` | ~1.3 GB |
| Otherwise | whisper `small` | `nllb-600M-int8` | ~1.1 GB |

The wizard shows what it picked and lets you switch the run device before
downloading.

![First-run wizard recommending models for an NVIDIA GPU](assets/images/firstrun.png)

Other options range from whisper `tiny` (~75 MB) up to `large-v3` (~3 GB),
plus NVIDIA's `parakeet-tdt-0.6b-v3` (~690 MB, very accurate and fast),
limited to English + 24 other European languages (no Japanese/Korean/Chinese).
MT models range from `m2m100-418M-int8` (~480 MB) up to `madlad400-3b`
(~3.5 GB). Models can be added/removed later via the **Models** dialog.
See [Picking a model](#picking-a-model) below for measured accuracy and
speed.

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

![Main window captioning with Japanese and Spanish translations](assets/images/main-window.png)

### Interface language

The interface follows your Windows display language by default and can speak
18 languages (English, 日本語, 한국어, 简体中文, 繁體中文, Español, Français,
Deutsch, Italiano, Português (Brasil), Русский, Українська, Polski,
Nederlands, Türkçe, Bahasa Indonesia, Tiếng Việt, ไทย). Pick a different one
under **Settings → Simple → Language**; it applies as soon as the Settings
window closes. This only affects VRCC's own interface; caption languages are
chosen in the main window.

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

## Picking a model

Every figure in this section comes from `tools/bench_stt.py`: 100
[LibriSpeech](https://www.openslr.org/12/) test-clean utterances run
through the same engine path the app uses, on one machine (Windows 11,
Ryzen 9 9950X3D, RTX 5090, driver 610.62). WER is word error rate on
English read speech; latency is the median time to transcribe one
utterance. The accuracy numbers and the relative speed ratios carry over
to other machines; the absolute latencies do not. On a slower CPU, expect
every CPU time here to stretch by roughly the same factor. The full
tables and methodology are in
[DEVELOPING.md](DEVELOPING.md#speech-to-text-benchmarks), and
[benchmarks/RESULTS.md](benchmarks/RESULTS.md) collects numbers from
other hardware.

On an NVIDIA GPU, keep the default `large-v3-turbo`. It matched
`large-v3` at 1.7% WER while being about 3.5x faster, and it handles every
language.

On CPU it depends on the language you speak:

- One of the 25 European languages: use `parakeet-tdt-0.6b-v3`. It reaches
  2.3% at 0.13 s, beating the `small` default (3.7% at 0.74 s) on accuracy
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
  the CUDA build (the `VRCC-cuda-windows-x64` zip, or a source install with
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

## Developing

Building from source, tests, benchmarks and packaging live in
[DEVELOPING.md](DEVELOPING.md).
