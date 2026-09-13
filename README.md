# VRCC

Also available in [简体中文](README.zh-Hans.md), [日本語](README.ja.md) and
[한국어](README.ko.md).

Speak into your microphone and your words appear in the VRChat chatbox as
live captions, with translations into up to three languages underneath.
Everything runs on your own Windows 10 or 11 PC. No cloud services, no
API keys.

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

<a id="setup"></a>

## Get VRCC running

This takes about five minutes, plus a one-time model download. There is no
Python to install and no API keys to find; everything runs locally on
Windows 10 or 11.

<a id="download"></a>

1. Open the Releases page. The ready-to-run program is not on the repository
   page you land on first. It sits behind
   [Releases](https://github.com/dljr-github/VRCC/releases/latest). That link
   goes straight to the newest one. You will see a version number, the
   release notes, and at the bottom a list headed **Assets**. The files live
   in there.

   The last two entries under Assets, "Source code (zip)" and "Source code
   (tar.gz)", are VRCC's own code rather than the program. Skip both. The two
   you are choosing between are named `VRCC-cuda-windows-x64` and
   `VRCC-windows-x64`, each followed by the version number.

2. Take the zip that suits your PC. `VRCC-cuda-windows-x64` is the CUDA
   build, for an NVIDIA graphics card with 8 GB of VRAM or more on driver
   570 or newer. Captions are worked out on the card and appear almost at
   once. `VRCC-windows-x64` is the CPU build. It is a much smaller download
   and the one to take on an NVIDIA card under 8 GB, on AMD or Intel
   graphics, and on a PC with no graphics card at all. Captions are
   identical, just a moment slower, and the default models are sized to keep
   up on CPU.

   To see what you have, press Ctrl+Shift+Esc for Task Manager, open the
   **Performance** tab and click **GPU**. The card's name, its dedicated
   memory and the driver version are all on that page. If you would rather
   not look, take the CUDA zip. It is the bigger download, but it runs on the
   processor by itself when it finds no card it can use.

   GPU acceleration covers NVIDIA cards only for now, because there is no AMD
   hardware here to test on.

3. Unblock the zip, then unpack it. Right-click the file you downloaded and
   choose **Properties**, at the bottom of the menu. If there is an
   **Unblock** tickbox near the bottom of the General tab, tick it and click
   OK. Windows marks files that came from the internet, and clearing the mark
   here also saves you the warning in the next step.

   Then right-click the zip again and choose **Extract All**. Explorer shows
   a zip as though it were a folder, but starting the program from in there
   hands Windows one file out of the middle of it and it will not run. Pick
   somewhere like Documents and you get a VRCC folder with `VRCC.exe` inside.
   Keep that folder together and put it wherever you like. There is nothing
   to install.

4. Run `VRCC.exe`. Double-click it in the folder you just extracted. Windows
   may put up a blue box headed "Windows protected your PC". Click
   **More info**, the small link under the message, then the **Run anyway**
   button that appears. VRCC is not code signed, so Windows shows this for a
   program it has not seen much of. That box is worth clicking through for
   the file you took from the Releases page above, and it is not advice to
   click through it anywhere else. If you ticked Unblock a moment ago, it
   may not appear at all.

5. Let the first-run wizard download the models. The first time you launch
   VRCC, a wizard checks your machine and picks a speech-to-text model and a
   translation model for you (about 1 to 3 GB to fetch), then downloads them.
   You can switch the run device or the models before it starts. This only
   happens once. [First run](#first-run) has the table of what it picks and
   why.

6. Enable OSC in VRChat. Open **Action menu → Options → OSC → Enabled**.
   VRCC sends captions to VRChat over OSC (default address
   `127.0.0.1:9000`); without this, they won't reach the chatbox.

7. Choose your microphone and languages. In VRCC, pick your **microphone**,
   your **source language** (the one you speak), and up to **three target
   languages** to translate into.

8. Start talking. Your words appear in the VRChat chatbox as live captions,
   with the translations underneath. You can also type into the box at the
   bottom of the window to send text the same way.

Later, open **Settings** to change the interface language, swap models, or
switch between Speed and Quality modes. [Usage](#usage) covers the rest of
what VRCC does day to day, and [Troubleshooting](#troubleshooting) is where
to start if captions never turn up. Building from source is for developers
who want to contribute (see [DEVELOPING.md](DEVELOPING.md)).

## What runs on your machine

Your voice never leaves your PC. There is no cloud service behind VRCC and
no API key to paste in.

Speech recognition runs through
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), or NVIDIA's
[Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
(run as an ONNX export via [onnx-asr](https://github.com/istupakov/onnx-asr))
for European languages, or
[SenseVoice-Small](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) for
Chinese, Japanese, Korean and English, which is available but not
recommended. Machine translation runs through
[CTranslate2](https://github.com/OpenNMT/CTranslate2), with the NLLB, M2M100
and MADLAD models. The wizard picks from these on your behalf, and
[Picking a model](#picking-a-model) has the measurements behind the choice.

## First run

If the configured models aren't downloaded yet, a first-run wizard opens,
picks models for your hardware and your spoken language (taken from your
Windows display language), and downloads them:

| Your machine | Speech-to-text | Translation | Download |
| ------------ | -------------- | ----------- | -------- |
| NVIDIA GPU with 16 GB+ VRAM | `large-v3-turbo` | `nllb-1.3B-int8` | ~3 GB |
| NVIDIA GPU with 11 GB to 16 GB VRAM | `large-v3-turbo` | `nllb-600M-int8` | ~2.3 GB |
| No GPU, or an NVIDIA GPU under 11 GB, a language Parakeet covers | `parakeet-tdt-0.6b-v3` | `nllb-600M-int8` | ~1.3 GB |
| No GPU, or an NVIDIA GPU under 11 GB, any other language | whisper `small` | `nllb-600M-int8` | ~1.1 GB |

VRChat itself wants a lot of the card's video memory. VRCC reserves 8 GB for
it and sizes the voice model against whatever the card has left once the
translation model is accounted for, which is why `large-v3-turbo` needs
roughly 11 GB rather than 8: under that, it would leave VRChat too little
room to run alongside it.

Japanese, Korean and Chinese use the same models as everything else. Parakeet
does not cover them, so a machine with no GPU, or an NVIDIA GPU under 11 GB,
takes whisper `small` rather than Parakeet. `sense-voice-small` is still
available in the Models window and is much smaller (240 MB), but
faster-whisper transcribed real VRChat speech more accurately in testing, so
it is no longer recommended.

The wizard shows what it picked and lets you switch the run device before
downloading.

![First-run wizard recommending models for an NVIDIA GPU](assets/images/firstrun.png)

Other options range from whisper `tiny` (~75 MB) up to `large-v3` (~3 GB),
plus NVIDIA's `parakeet-tdt-0.6b-v3` (~690 MB, very accurate and fast),
limited to English + 24 other European languages (no Japanese/Korean/Chinese).
MT models range from `m2m100-418M-int8` (~480 MB) up to `madlad400-3b`
(~3.5 GB). NLLB leads the recommendations on measured caption quality; the
M2M100 models are the permissive-license alternative (see
[Model licenses](#model-licenses)). Models can be added/removed later via the **Models** dialog.
See [Picking a model](#picking-a-model) below for measured accuracy and
speed.

Once VRCC is ready to run, the main window opens beside a panel of setup
steps, with rows for turning captioning on, the voice model finishing its load,
your voice reaching VRCC, VRChat turning up on the network, and a caption going
out over OSC, plus an optional row for sound picked up from your speakers that
never blocks the rest. The VRChat row only proves its OSC service advertised
itself over mDNS, not that anything reached the game, so a network that blocks
mDNS can leave the row unticked while VRChat runs fine, and the panel goes on
opening at every launch for as long as that lasts. The caption row is evidence
in the other direction. It proves a caption was sent, not that VRChat picked it
up, because OSC never confirms that a message got through, and it drops out of
what the panel requires whenever sending to VRChat is switched off. The panel
stops opening on later launches once every row it still requires has passed.
Closing it before then dismisses it for that session only, and it is back on
the next start. Its own **Don't show this again** button dismisses it for
good instead, whether or not every row has passed. Settings, on the Simple
page, brings it back either way.

## Usage

1. Start VRChat and enable OSC: **Action menu → Options → OSC → Enabled**.
2. Start VRCC. Pick your microphone, source language and up to three
   target languages.
3. Talk. Your mic is cleaned of steady background noise first (on by
   default), then utterances are segmented automatically (Silero VAD),
   transcribed, translated, and sent to the chatbox, throttled to stay
   inside VRChat's chatbox rate limit so continuous speech never triggers
   the in-game spam mute. Turn it down or off under **Settings → Voice
   recognition → Reduce background noise**.
4. **Read other people:** press **Hear others** in the main window to
   transcribe and translate the voices coming out of your speakers, so you
   can follow someone speaking a language you don't read. Choose which
   device it listens to, and which language it shows, under **Settings →
   Simple → What other people say**. Two things to know.
   It captures the whole output device rather than VRChat's voice channel,
   because Windows offers no per-app voice tap, so game and world audio are
   transcribed too. And it is shown only in the VRCC window: other people's
   words are never sent to the chatbox, since the room already heard them and
   relaying it would republish their speech under your name. Without a
   graphics card it shares one voice model with your own captions, so both
   get slower.
5. **Typed messages:** the text box at the bottom of the main window sends
   typed text through the same translate → chatbox path (useful when you'd
   rather not speak).
6. **Mute sync:** when enabled, muting yourself in VRChat makes VRCC stop
   listening entirely (configurable to ignore or invert). Speech made
   while muted is never captured, so unmuting mid-sentence captions only
   what you say after the unmute. This uses VRChat's OSCQuery discovery
   and works **only when VRChat runs on the same PC** (localhost);
   captioning itself works regardless.

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

- **Speed** (default): greedy decoding (beam 1), and it finalises a caption
  after 600 ms of silence. Captions appear fastest.
- **Quality**: beam 5, and it waits 800 ms. Noticeably better phrasing, about
  200 ms slower per caption.

Mode changes captions only. Translation keeps its own search width under
**Settings → Translation → Advanced (fine-tuning)**, because the measurements
behind these presets cover speech recognition and say nothing about
translation.

Switching mode rewrites the pause timings on the Advanced page and Search
width on the Voice recognition page. If you have set any of those by hand,
VRCC names them and asks first; flipping between the two modes to compare
never prompts, because those values came from a preset rather than from you.

Parakeet and SenseVoice always decode at full accuracy, so the Mode control
is greyed out while either is the active voice model. The individual
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

## Updating

VRCC looks for a newer release when it starts, and puts up a notice if it
finds one. The notice has a button that opens the release page in your
browser. Nothing is downloaded or swapped for you, so updating is the same
few steps as the first time: take the new zip, unblock it, extract it, and
run `VRCC.exe` from the new folder.

Your settings and your downloaded models do not live in the application
folder (see the table above), so they carry over by themselves. The models
are not fetched a second time and the first-run wizard does not reappear.
You can keep the old folder until you are happy, then delete it.

If you would rather not be told, open **Settings** and go to the
**Advanced / Power users** page, then untick **Tell me when a new version
is available**.

A portable install works the other way round. Under `--portable` the
config, models and logs all sit in the application's own folder, so
replacing that folder throws them away. Copy `config.json` and the `models`
folder into the new one first, or unpack the new build over the top of the
old.

## Model licenses

The **code** in this repository is separate from the **models** it
downloads; check that a model's license fits your use:

- **NLLB** models (default `nllb-600M-int8`, plus 1.3B and 3.3B):
  **CC-BY-NC-4.0**, *non-commercial use only*. These are the recommended
  models, so if you use VRCC commercially (paid or sponsored streaming, or
  at work) pick an M2M100 model in the **Models** window instead.
- **M2M100** models (`m2m100-418M-int8`, plus 1.2B): **MIT**, so they carry
  no such restriction. In a blind comparison over 60 VRChat-style utterances
  into Japanese, Chinese and Korean they were less accurate than NLLB 600M,
  which is the trade for the permissive license.
- **MADLAD-400** (`madlad400-3b`): **Apache-2.0**.
- Whisper models: **MIT** (OpenAI weights, SYSTRAN CT2 conversions).
- **Parakeet** (`parakeet-tdt-0.6b-v3`): **CC-BY-4.0** (NVIDIA weights,
  istupakov ONNX export).
- **GTCRN** (the *Reduce background noise* denoiser, bundled not
  downloaded): **MIT** (Xiaobin Rong, https://github.com/Xiaobin-Rong/gtcrn).

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
- **A small screen appears when VRCC starts**: it names what VRCC is
  loading and closes once VRCC has something ready to show. The first
  launch after an update or a reboot is slower, because Windows is
  reading the program's files from disk for the first time. If a launch
  dies, the newest log file shows how far it got, which helps when
  reporting a problem.
- **First transcription is slow**: model load and warm-up continue
  inside the window after the startup screen closes; captions flow at
  full speed afterwards.
- **Reporting a bug**: attach the newest file from the logs folder above.
  Each run writes one file at full debug detail and the five newest are
  kept, so the file with the latest timestamp is the run that went wrong.
- **Starting VRCC again does nothing**: only one copy runs at a time, and
  launching it again asks the running copy to show itself. If it doesn't
  come forward, Windows may be holding focus for whatever you clicked
  last, so the taskbar button is where to look.
- **Setup steps don't reappear**: closing the panel dismisses it for that
  session, and it opens again next start unless the rows it requires have
  passed. Its own **Don't show this again** button dismisses it for good
  instead. Either way, open **Settings → Simple** and use **Bring back the
  setup steps** to get it back.

## Developing

Building from source, tests, benchmarks and packaging live in
[DEVELOPING.md](DEVELOPING.md).
