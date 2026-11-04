# Translation regression gate

`tools/bench_mt.py` drives the real `TranslateEngine` with the shipped
decoding config (beam 4, device and compute type resolved as the app
resolves them) and counts the failures a caption would show. It is a
gate, not a ranking. Quality averaged over target languages would weight
languages a given user may never use, so no model is scored overall.
Each arm is judged on the same counted failures.

## Measured

- DROPPED: a number in the source with no match in the output. Numbers
  are compared as an ordered sequence, so "4 4 4" coming back as "4" is
  two drops, and "1 2 3" coming back as "3 2 1" is a drop. A number
  written in words in the output counts as kept (column SPELLED) when
  the tool can read the word: 0 to 20, the tens, hundred and thousand in
  en, de, fr, es and ru, plus CJK numerals and Korean native and
  sino-Korean counting words. Anything outside those tables reads as
  DROPPED. The "merged" column is the subset of DROPPED where every
  digit survived but the runs were joined ("8 8 8 8" to "8888").
- ADDED: the output has a digit run the source did not.
- dash frag / dash full: output opens with a dash the source did not
  have. "frag" is the one-to-three word fragments, "full" is the
  complete sentences and the number-bearing sentences. The bench builds
  the engine with `strip_dash=False`, so this is the raw decoder output
  before `TranslateEngine.translate` removes an added "- ".
- loops: output where a four-character slice repeats four or more times
  in a row (`longest_run`, the check in
  `tests/integration/test_translate_quality.py`). This also fires on a
  faithful rendering of a repeated source ("No, no, no, no, no."), so it
  counts what the guard would flag, not only decoder failures.
- ms/call: wall time of one `translate()` call. The bench batches at
  most three targets per call, as the app does, and warms the engine
  first.

## Corpus

`tools/mt_corpus.json`: 364 source sentences in en (112), ja (47), zh (43), ko (40),
fr (41), es (41) and de (40), in six groups (number, repeat, spoken, interjection,
fragment, sentence). Each is translated into six targets (Japanese,
Chinese Simplified, Korean, French, Spanish, Russian; English is added
for non-English sources and the source language is removed), giving
2224 source/target pairs, 928 of them with a number in the source, 734
fragment pairs and 1063 full-sentence pairs. The "spoken" group spells
its digits in words ("one one two three five eight"), and the
instrument reads those as numbers only for that group.

## Results

Command:

```powershell
python -m tools.bench_mt --models nllb-600M-int8,nllb-1.3B-int8,m2m100-418M-int8,m2m100-1.2B-int8 --arms 1.0:3,1.1:3,1.1:0 --show 400 --json raw.json
```

Machine: AMD64 Family 26 Model 68 CPU, NVIDIA GeForce RTX 5090 (driver
617.42), models run as `cuda:int8_float16`. Run on 2026-10-10 as one
pass per arm, so the timings carry no variance estimate and differences
of a few milliseconds are noise.

rep is repetition_penalty, ngram is no_repeat_ngram_size. 1.1 with 3 is
the shipped setting.

| model | rep | ngram | digit/all pairs | KEPT | SPELLED | DROPPED | merged | ADDED | dash frag | dash full | loops | ms/call med | ms/call mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nllb-600M-int8 | 1.0 | 3 | 928/2224 | 681 | 108 | 130 | 24 | 9 | 83/734 (11.3%) | 0/1063 (0.0%) | 0 | 68 | 71 |
| nllb-600M-int8 | 1.1 | 3 | 928/2224 | 657 | 103 | 161 | 31 | 7 | 99/734 (13.5%) | 0/1063 (0.0%) | 0 | 68 | 71 |
| nllb-600M-int8 | 1.1 | 0 | 928/2224 | 682 | 103 | 136 | 24 | 7 | 99/734 (13.5%) | 0/1063 (0.0%) | 5 | 69 | 73 |
| nllb-1.3B-int8 | 1.0 | 3 | 928/2224 | 628 | 127 | 167 | 45 | 6 | 43/734 (5.9%) | 0/1063 (0.0%) | 0 | 125 | 131 |
| nllb-1.3B-int8 | 1.1 | 3 | 928/2224 | 604 | 117 | 200 | 52 | 7 | 57/734 (7.8%) | 0/1063 (0.0%) | 0 | 121 | 126 |
| nllb-1.3B-int8 | 1.1 | 0 | 928/2224 | 629 | 117 | 174 | 48 | 8 | 57/734 (7.8%) | 0/1063 (0.0%) | 1 | 127 | 132 |
| m2m100-418M-int8 | 1.0 | 3 | 928/2224 | 749 | 43 | 123 | 4 | 15 | 0/734 (0.0%) | 0/1063 (0.0%) | 0 | 62 | 66 |
| m2m100-418M-int8 | 1.1 | 3 | 928/2224 | 746 | 44 | 128 | 4 | 12 | 0/734 (0.0%) | 0/1063 (0.0%) | 1 | 58 | 64 |
| m2m100-418M-int8 | 1.1 | 0 | 928/2224 | 801 | 35 | 81 | 4 | 13 | 0/734 (0.0%) | 0/1063 (0.0%) | 18 | 61 | 65 |
| m2m100-1.2B-int8 | 1.0 | 3 | 928/2224 | 753 | 61 | 88 | 7 | 27 | 0/734 (0.0%) | 0/1063 (0.0%) | 0 | 108 | 117 |
| m2m100-1.2B-int8 | 1.1 | 3 | 928/2224 | 750 | 68 | 96 | 7 | 15 | 0/734 (0.0%) | 0/1063 (0.0%) | 0 | 111 | 119 |
| m2m100-1.2B-int8 | 1.1 | 0 | 928/2224 | 728 | 58 | 73 | 7 | 70 | 0/734 (0.0%) | 0/1063 (0.0%) | 22 | 111 | 117 |

KEPT, SPELLED, DROPPED and ADDED sum to the 928 digit pairs, plus any
ADDED on a pair whose source had no number (none to two per arm).
"digit/all pairs" is the number-bearing pairs over all pairs.

## Reading it

- On both NLLB models, repetition_penalty 1.1 adds drops against 1.0 at
  ngram 3 (600M 130 to 161, 1.3B 167 to 200) and adds dashed fragments
  (600M 11.3% to 13.5%, 1.3B 5.9% to 7.8%). Switching ngram off at 1.1
  takes back part of the drops (136 and 174) and lets loops back in (5
  and 1, every one a repeated-source interjection rendered faithfully).
- m2m100 never prefixes a dash. Its drop count moves little with the
  penalty and falls with ngram 0 (123/128 to 81 on 418M), but ngram 0
  brings 18 and 22 loop flags and, on 1.2B, 70 ADDED.
- NLLB-1.3B dashes fragments about half as often as 600M at the same
  setting, and takes about 1.8 times the time per call on this GPU.
- The penalty does not show in wall time. Neither does ngram.

Limits. The instrument cannot read number words outside its tables
(for example Russian oblique forms), so some DROPPED are spelled
numbers it cannot see. The reverse error is bounded by rules:
Korean 네, 세, 두, 한 count only before a counter or inside a list
that holds an unambiguous number, and a CJK numeral does not count
inside a listed word (一起, 十分). Digit pairs are weighted toward the sentence
shapes in the corpus, which is mostly synthetic VRChat-style speech and
not recorded captions. One pass, no confidence intervals.

## LLM arm: HY-MT1.5 1.8B through llama-server

`tools/bench_llm_mt.py` runs Tencent's HY-MT1.5 1.8B through a llama.cpp
`llama-server` child process and scores it with the same corpus, targets
and columns as above, so its rows sit next to the NLLB ones. The server
gets one slot per chatbox target (`--parallel 3`) and the three requests
of a call go out together. Prompts are Tencent's published templates.
The `beam` column holds the sampling arm instead: `greedy`, or
`recommended` (Tencent's top_k 20, top_p 0.6, temperature 0.7,
repetition penalty 1.05). Each arm also prints the server's load time and
the VRAM it took.

### First results (cloud container, CPU only)

Run on 2026-11-01 in a 4-core shared cloud container with no GPU, on
`--limit 120` (120 sentences, 732 pairs, 312 with a number). llama-server
could not be built there, so the HY-MT rows ran on llama-cpp-python's
server (0.3.36), which handles one request at a time: its ms/call is three
decodes back to back, not three in parallel. The "device" column still
says np=3 because that is the label the tool prints.

| model | rep | ngram | beam | device | digit/all pairs | KEPT | SPELLED | DROPPED | merged | ADDED | dash frag | dash full | loops | ms/call med | ms/call mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nllb-600M-int8 | 1.0 | 3 | 4 | cpu:int8 | 312/732 | 228 | 36 | 44 | 6 | 4 | 32/245 (13.1%) | 0/347 (0.0%) | 0 | 588 | 614 |
| HY-MT1.5-1.8B-Q4_K_M | 1.0 | 0 | greedy | llama-cpp-python, cpu, serial | 312/732 | 288 | 5 | 16 | 0 | 3 | 0/245 (0.0%) | 0/347 (0.0%) | 1 | 3303 | 3225 |
| HY-MT1.5-1.8B-Q8_0 | 1.0 | 0 | greedy | llama-cpp-python, cpu, serial | 312/732 | 283 | 5 | 20 | 1 | 4 | 0/245 (0.0%) | 0/347 (0.0%) | 1 | 3385 | 3257 |
| HY-MT1.5-1.8B-Q8_0 | 1.05 | 0 | recommended | llama-cpp-python, cpu, serial | 312/732 | 278 | 6 | 25 | 0 | 3 | 0/245 (0.0%) | 0/347 (0.0%) | 0 | 3327 | 3242 |

- HY-MT drops about a third as many numbers as NLLB-600M (16 against 44)
  and never dresses a fragment as dialogue (0% against 13.1%).
- Many of HY-MT's remaining DROPPED and ADDED are correct renderings the
  instrument cannot read: "12月12日" as "12 décembre", or a Spanish date
  written back as "12月12日". The real misses are digit runs read as
  something else: "6 6 6" became "6:06", "88 88" became "88", and "one two
  three four" became "once, twice, three times, four times" in Chinese.
- The one loop is "Hmm hmm hmm hmm." into Russian, which ran to the
  output cap as a single "Хмммм...". The app would need the same loop
  guard NLLB output gets.
- Q8_0 is no better than Q4_K_M here, and Tencent's sampling setting is
  slightly worse than greedy. Q4_K_M greedy is the arm to carry forward.
- On this CPU HY-MT took about 5.5 times NLLB-600M's time per call. With
  the decodes serial, that is roughly 1.1 s per target. The 5090 numbers,
  with llama-server's parallel slots, are what decide whether it is fast
  enough.

### Full corpus, CPU, all three targets batched

Run on 2026-11-03 in a 4-core shared cloud container with no GPU (a
different, slower CPU than the run above, so compare rows within this
table only), on the whole corpus (2224 pairs, 928 with a number), 4
threads each. HY-MT ran in-process through `--backend inproc`
(tools/llm_mt_inproc.py): one llama.cpp context holds a sequence per
target and every decode step advances all three together, the way NLLB's
`translate_batch` does. That cut its time per call by about a quarter
against decoding the targets one after another.

| model | rep | ngram | beam | device | digit/all pairs | KEPT | SPELLED | DROPPED | merged | ADDED | dash frag | dash full | loops | ms/call med | ms/call mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nllb-600M-int8 | 1.0 | 3 | 4 | cpu:int8 | 928/2224 | 676 | 108 | 134 | 22 | 10 | 82/734 (11.2%) | 0/1063 (0.0%) | 0 | 969 | 1012 |
| HY-MT1.5-1.8B-Q4_K_M | 1.0 | 0 | greedy | llama-cpp-python cpu t=4 batch=3 | 928/2224 | 867 | 16 | 39 | 0 | 6 | 0/734 (0.0%) | 0/1063 (0.0%) | 6 | 2014 | 2032 |

- The quality result holds on the full corpus: HY-MT keeps 93% of the
  numbers against NLLB's 73%, drops 39 against 134, and puts a dialogue
  dash on no fragment against 11%.
- It costs about twice NLLB's time per call on CPU (2.0 s against 1.0 s
  median for three targets). On a fast desktop CPU both would be quicker,
  but the ratio is what to expect.
- Six loops, all repeated interjections ("Oh oh oh oh oh oh oh." and
  "Hmm hmm hmm hmm." into Russian ran to the cap as one long word; the
  rest are faithful "No, no, no, no, no."). A loop guard like NLLB's is
  still needed.
- Many remaining DROPPED rows are correct date renderings ("12月12日" as
  "12 décembre"); the real misses are spoken digit runs like "Zero zero
  seven".

To run it, unzip a llama.cpp release
(https://github.com/ggml-org/llama.cpp/releases, the CUDA 12 build for an
NVIDIA GPU) and:

```powershell
python -m tools.bench_llm_mt --server C:\llama\llama-server.exe --quants Q4_K_M,Q8_0 --sampling greedy,recommended --show 40 --json llm.json
```

The GGUF downloads once from `tencent/HY-MT1.5-1.8B-GGUF`. `--gpu-layers 0`
runs it on the CPU. HY-MT1.5 is under the Tencent HY Community License,
which does not apply in the European Union, the United Kingdom or South
Korea.
