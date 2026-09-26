# Speech detection across microphones

Run the same labeled recordings through the previous duration policy and
the candidate defaults, using real streaming Silero probabilities:

```powershell
python -m tools.bench_segmentation recordings/manifest.json --output bench_results/segmentation.json
```

Install the project's `bench` extra for WAV/FLAC decoding. The tool runs
locally and downloads nothing. Manifest paths are relative to the manifest:

```json
[
  {"path": "headset-yes.wav", "kind": "speech", "microphone": "headset-usb", "language": "en"},
  {"path": "laptop-keyboard.wav", "kind": "nonspeech", "microphone": "laptop", "language": "none"}
]
```

Use one utterance per speech clip. Include short replies, quiet sentences,
pauses and ordinary conversation in each supported language. Include room
silence, breathing, keyboard sounds and handling noise as non-speech clips.
Keep consented recordings from headset, laptop, webcam, USB and Bluetooth
inputs in separate groups. Competing speakers are speech, so evaluate them
separately from non-speech noise; VAD cannot identify the intended speaker.

Every recording is replayed unchanged, at two lower gains, with clipping,
with a gain step, with reduced bandwidth, and with added white noise.
These are signal stress cases, not simulations of microphone hardware or
Bluetooth codecs. Report results by microphone, language and variation;
an aggregate can conceal a regression for quiet voices or one device class.

The JSON records source hashes, both configurations, start/final event times,
missing finals on speech, false finals on non-speech, and unfinished segments.
Times are relative to the original clip, excluding the added warm-up silence.
Both policies use the current duration accounting, excluding trailing silence
and uncertainty. The baseline uses immediate onset and a 500 ms minimum,
isolating the configuration change. The candidate confirms onset after 64 ms of
consecutive evidence and permits utterances of at least 96 ms. These are
initial policy choices, not universally optimal measured thresholds.

A detected utterance is not proof of a correct transcription. Before a public
release, also score the emitted audio through the supported STT engines against
human transcripts, count false captions per minute of non-speech, and measure
end-of-speech to final-caption latency. The existing `tools/bench_stt.py` covers
model WER and inference latency; this tool isolates segmentation. Synthetic
fixtures and gain changes alone cannot establish support for every microphone.

Implementation references: [Pipecat's onset state machine](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/audio/vad/vad_analyzer.py)
and [Silero's duration and padding order](https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py).
