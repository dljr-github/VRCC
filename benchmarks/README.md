# STT benchmark results

Speed and accuracy numbers for the speech-to-text models VRCC offers,
measured on real machines with `tools/bench_stt.py`. Each JSON file in
this directory is one machine; [RESULTS.md](RESULTS.md) has the rendered
tables for all of them.

## What is measured

The harness drives VRCC's own engine path (`create_stt_engine` with the
app-default settings: Speed mode, source language English) over
utterances from [LibriSpeech](https://www.openslr.org/12/) test-clean,
fed pre-segmented exactly as the app's VAD hands them over. The app's mic
capture path does not run here (the denoiser, on by default, and the
gain), so these numbers isolate the model and the denoiser default does
not change them. Per model and device it records:

- **WER**: word error rate against the reference transcripts, after
  Whisper-style English text normalization of both sides. Lower is better.
  All five quality gates are opened while measuring (whisper's avg_logprob,
  no_speech and compression_ratio; the parakeet/sensevoice avg_logprob
  gates those two onnx-asr backends check themselves), so WER reflects what
  the model recognized rather than what a gate suppressed; utterances the
  app's default gates would have dropped are counted separately as
  `gated`, against whichever threshold applies to the model's backend. A
  whisper repetition loop the compression_ratio gate would drop is not on
  this counter (that ratio isn't reported back), so it shows up only as
  scored text, never as `gated` or `empty`.
- **latency**: median wall-clock seconds to transcribe one utterance
  (roughly what you wait between finishing a sentence and seeing the
  caption).
- **RTF**: real-time factor, processing time / audio duration. 0.05 means
  20x faster than real time.
- **load / warm-up**: one-time startup cost.
- **beam**: 1 is the Speed mode default, 5 is Quality mode. Running both
  shows what Quality buys and what it costs on your hardware, which is
  what the app uses to recommend a mode per model.

The dataset is English read speech, so WER only ranks English accuracy;
speed numbers transfer to other languages.

## Noise conditions

Clean LibriSpeech is not what breaks a model in the app: a model that ranks
well on clean speech can rank worst once another person is talking in the
room. `--noise babble --snr N` overlays several corpus speakers (seeded,
so the same command always builds the same babble); `--noise white --snr N`
overlays seeded Gaussian noise instead. Either way `N` is the SNR in dB
against each utterance's own level. See `tools/bench_noise.py` for exactly
how the noise is built and mixed.

```
python tools/bench_stt.py --device cuda --noise babble --snr 10 --out bench_results/babble10
```

Point `--out` at a directory of its own per noise condition: a noisy run's
result files are stamped with the condition (in both the filename and the
recorded utterance set), so they never overwrite or silently blend with a
clean run in the same directory, but `--export` still refuses to bundle a
directory that mixes conditions. Export and report each condition the same
way as a clean run, naming the file after both the machine and the
condition:

```
python tools/bench_stt.py --export benchmarks/rtx-4070-ryzen-7700-babble10.json --out bench_results/babble10
python tools/bench_report.py --write
```

The utterance-set fingerprint covers the utterances and noise condition,
not which gates were open. A result file measured before a gate fix in
`bench_model` therefore still reads as "exists, skipping" to the resume
logic and keeps under-measuring the model's error rate; re-run it with
`--force` after any change to the gates the harness opens.

## Contributing your machine

1. Install from source with the benchmark extras
   (see [DEVELOPING.md](../DEVELOPING.md#install-from-source);
   add `.[bench]`, and on NVIDIA machines the CUDA steps):

   ```
   pip install -e .[dev,bench]
   ```

2. Fetch the dataset (~350 MB) and any missing models (~9.5 GB for all nine):

   ```
   python tools/bench_stt.py --download-only
   ```

3. Benchmark. Close other heavy programs first, and on laptops plug in
   and pick the performance power plan. Each device pass takes roughly
   10-60 minutes depending on hardware:

   ```
   python tools/bench_stt.py --device cpu
   python tools/bench_stt.py --device cuda
   ```

   Skip the cuda passes on machines without an NVIDIA GPU.

   Then measure Quality mode on the whisper models that are fast enough to
   use live on that device (the others already lag at beam 1, so their
   beam-5 numbers cannot change any advice):

   ```
   python tools/bench_stt.py --device cuda --beam 5
   python tools/bench_stt.py --device cpu --beam 5 --models small,base,tiny,distil-small.en
   ```

   Interrupted runs resume where they left off (one JSON per model, device
   and beam lands in `bench_results/`). Every record is stamped with the
   utterance set it measured, and the export refuses to bundle runs that
   used different sets.

4. Export your numbers (name the file after your hardware) and refresh
   the rendered tables:

   ```
   python tools/bench_stt.py --export benchmarks/rtx-4070-ryzen-7700.json
   python tools/bench_report.py --write
   ```

5. Open a PR adding your JSON and the regenerated `RESULTS.md`.

Partial contributions are welcome: a CPU-only machine, or a subset of
models via `--models` (e.g. laptops without disk space for all nine).
