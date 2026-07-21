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
  The quality gates are opened while measuring, so WER reflects what the
  model recognized rather than what the gates suppressed; utterances the
  app's default gates would have dropped are counted separately as
  `gated`.
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
