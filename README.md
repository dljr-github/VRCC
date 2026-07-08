# VRCC

Speak into your microphone and your words appear in the VRChat chatbox as
live captions — with translations into up to three languages underneath.
Everything runs locally: speech recognition via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) and machine
translation via [CTranslate2](https://github.com/OpenNMT/CTranslate2)
(NLLB / M2M100 / MADLAD models). No cloud services, no API keys.

```
You say:   "Hello, how are you today?"
Chatbox:   Hello, how are you today?
           こんにちは 今日はどうですか?
```

## Download (recommended)

From the [latest release](https://github.com/dljr-github/VRCC/releases/latest),
grab the zip that matches your hardware, unzip it anywhere, and run
`VRCC.exe`. No Python setup needed — the first-run wizard downloads the
models for you.

- **`VRCC-cuda-windows-x64.zip`** — for PCs with an NVIDIA GPU (driver 570
  or newer): near-instant captions, with automatic CPU fallback when no
  usable GPU is found.
- **`VRCC-windows-x64.zip`** — CPU-only build, much smaller download.
  Captions are identical, just a moment slower; the default models are
  sized to keep up on CPU.

GPU acceleration only supports NVIDIA cards at the moment (no AMD hardware
to test on) — on AMD or Intel graphics, use the CPU build.

Installing from source (below) is only for developers who want to
contribute.

## Requirements

- Windows 10/11 (Python 3.10+ needed for source installs only)
- Optional: an NVIDIA GPU. CPU and GPU produce identical captions — the
  only difference is latency (GPU responds near-instantly, CPU a moment
  slower). The first-run wizard picks a sensible default for your machine.

## Install from source (developers)

Most users don't need this — use the
[packaged exe](#download-recommended) instead.

CPU-only:

```
python -m venv .venv
.venv\Scripts\pip install -e .
```

With NVIDIA GPU acceleration (bundles the CUDA 12 cuBLAS runtime as a pip
wheel — no system CUDA install needed):

```
.venv\Scripts\pip install -e .[cuda]
```

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
and MT models from `m2m100-418M-int8` (~480 MB) up to `madlad400-3b`
(~3.5 GB). Models can be added/removed later via the **Models** dialog.

## Usage

1. Start VRChat and enable OSC: **Action menu → Options → OSC → Enabled**.
2. Start VRCC. Pick your microphone, source language and up to three
   target languages.
3. Talk. Utterances are segmented automatically (Silero VAD), transcribed,
   translated, and sent to the chatbox — throttled to stay inside VRChat's
   chatbox rate limit so continuous speech never triggers the in-game
   spam mute.
4. **Typed messages:** the text box at the bottom of the main window sends
   typed text through the same translate → chatbox path (useful when you'd
   rather not speak).
5. **Mute sync:** when enabled, muting yourself in VRChat pauses
   captioning (configurable to ignore or invert). This uses VRChat's
   OSCQuery discovery and works **only when VRChat runs on the same PC**
   (localhost); captioning itself works regardless.

### Performance profiles

The main window has a one-click **Latency / Quality** toggle:

- **Latency** (default): greedy decoding (beam 1), short silence
  thresholds — captions appear fastest.
- **Quality**: beam 5 STT / beam 3 MT and slightly longer silence
  thresholds — noticeably better phrasing at the cost of a few hundred
  extra milliseconds.

Individual knobs (VAD timings, beam sizes, quality gates…) live in
**Settings → Advanced**.

## Where things are stored

| What | Default location |
| ---- | ---------------- |
| Config | `%LOCALAPPDATA%\VRCC\VRCC\config.json` |
| Models | `%LOCALAPPDATA%\VRCC\VRCC\models\` |
| Logs | `%LOCALAPPDATA%\VRCC\VRCC\logs\vrcc.log` |

Run with `--portable` to keep config, models and logs in the application's
own directory instead (handy on a USB stick or for isolated installs).

## Building a standalone exe

A PyInstaller one-folder spec lives in `packaging/vrcc.spec`:

```
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\pyinstaller packaging\vrcc.spec --noconfirm --distpath dist
```

This produces `dist\VRCC\VRCC.exe` (windowed, no console). Models are not
bundled — the exe downloads them on first run exactly like the source
install, and shares the same default models directory. Distribute the whole
`dist\VRCC` folder; pair the exe with `--portable` if you want a fully
self-contained folder.

## End-to-end smoke test

`scripts/smoke_e2e.py` runs the real VAD → STT → MT → chatbox pipeline over
a WAV file (no GUI, no UDP — the chatbox client prints to stdout) and
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
  **CC-BY-NC-4.0** — *non-commercial use only*.
- **M2M100** models (`m2m100-418M-int8`, `m2m100-1.2B-int8`): **MIT** — a
  good alternative if you need a permissive license.
- **MADLAD-400** (`madlad400-3b`): **Apache-2.0**.
- Whisper models: **MIT** (OpenAI weights, SYSTRAN CT2 conversions).

## Troubleshooting

- **"No GPU detected" / everything runs on CPU** — make sure you're running
  the CUDA build (`VRCC-cuda-windows-x64.zip`, or a source install with
  `pip install -e .[cuda]`) and that your NVIDIA driver is ≥ 570. GPU
  acceleration is NVIDIA-only for now (no AMD hardware to test on).
  CPU-only operation is normal otherwise: captions are identical, just a
  moment slower; the default models are sized to keep up on CPU.
- **GPU runs out of VRAM** — the engines fall back to CPU (int8)
  automatically for the session (same captions, higher latency); pick a
  smaller model to stay on GPU.
- **VRChat isn't showing captions** — enable OSC in VRChat (Action menu →
  Options → OSC), and check the OSC address in VRCC's settings matches
  where VRChat listens (default `127.0.0.1:9000`). If you use other OSC
  tools (e.g. a router like OSCRepeater), point VRCC at the router's port.
- **Mute sync does nothing** — it requires VRChat on the *same machine*
  (localhost mDNS/OSCQuery discovery), OSC enabled in-game, and an avatar
  that reports `MuteSelf`.
- **First transcription is slow** — model load + warm-up happens once at
  startup; captions flow at full speed afterwards.

## Development

```
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\python -m pytest tests/ -v            # unit suite
.venv\Scripts\python -m pytest tests/ -v -m integration  # downloads models
```