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
  complete sentences and the number-bearing sentences.
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
| nllb-600M-int8 | 1.0 | 3 | 928/2224 | 682 | 109 | 128 | 24 | 9 | 83/734 (11.3%) | 0/1063 (0.0%) | 0 | 70 | 73 |
| nllb-600M-int8 | 1.1 | 3 | 928/2224 | 658 | 104 | 159 | 31 | 7 | 99/734 (13.5%) | 0/1063 (0.0%) | 0 | 71 | 73 |
| nllb-600M-int8 | 1.1 | 0 | 928/2224 | 683 | 104 | 134 | 24 | 7 | 99/734 (13.5%) | 0/1063 (0.0%) | 5 | 72 | 76 |
| nllb-1.3B-int8 | 1.0 | 3 | 928/2224 | 629 | 128 | 165 | 45 | 6 | 43/734 (5.9%) | 0/1063 (0.0%) | 0 | 131 | 137 |
| nllb-1.3B-int8 | 1.1 | 3 | 928/2224 | 604 | 117 | 200 | 52 | 7 | 57/734 (7.8%) | 0/1063 (0.0%) | 0 | 131 | 137 |
| nllb-1.3B-int8 | 1.1 | 0 | 928/2224 | 629 | 118 | 173 | 48 | 8 | 57/734 (7.8%) | 0/1063 (0.0%) | 1 | 137 | 144 |
| m2m100-418M-int8 | 1.0 | 3 | 928/2224 | 745 | 43 | 123 | 4 | 19 | 0/734 (0.0%) | 0/1063 (0.0%) | 0 | 68 | 72 |
| m2m100-418M-int8 | 1.1 | 3 | 928/2224 | 742 | 44 | 128 | 4 | 16 | 0/734 (0.0%) | 0/1063 (0.0%) | 1 | 62 | 67 |
| m2m100-418M-int8 | 1.1 | 0 | 928/2224 | 797 | 35 | 81 | 4 | 17 | 0/734 (0.0%) | 0/1063 (0.0%) | 18 | 65 | 70 |
| m2m100-1.2B-int8 | 1.0 | 3 | 928/2224 | 743 | 58 | 88 | 7 | 40 | 0/734 (0.0%) | 0/1063 (0.0%) | 0 | 119 | 128 |
| m2m100-1.2B-int8 | 1.1 | 3 | 928/2224 | 738 | 66 | 96 | 7 | 29 | 0/734 (0.0%) | 0/1063 (0.0%) | 0 | 120 | 127 |
| m2m100-1.2B-int8 | 1.1 | 0 | 928/2224 | 722 | 58 | 73 | 7 | 76 | 0/734 (0.0%) | 0/1063 (0.0%) | 22 | 119 | 127 |

KEPT, SPELLED, DROPPED and ADDED sum to the 928 digit pairs, plus any
ADDED on a pair whose source had no number (none to two per arm).
"digit/all pairs" is the number-bearing pairs over all pairs.

## Reading it

- On both NLLB models, repetition_penalty 1.1 adds drops against 1.0 at
  ngram 3 (600M 128 to 159, 1.3B 165 to 200) and adds dashed fragments
  (600M 11.3% to 13.5%, 1.3B 5.9% to 7.8%). Switching ngram off at 1.1
  takes back part of the drops (134 and 173) and lets loops back in (5
  and 1, every one a repeated-source interjection rendered faithfully).
- m2m100 never prefixes a dash. Its drop count moves little with the
  penalty and falls with ngram 0 (123/128 to 81 on 418M), but ngram 0
  brings 18 and 22 loop flags and, on 1.2B, 76 ADDED.
- NLLB-1.3B dashes fragments about half as often as 600M at the same
  setting, and takes about 1.8 times the time per call on this GPU.
- The penalty does not show in wall time. Neither does ngram.

Limits. The instrument cannot read number words outside its tables
(for example Russian oblique forms), so some DROPPED are spelled
numbers it cannot see. Digit pairs are weighted toward the sentence
shapes in the corpus, which is mostly synthetic VRChat-style speech and
not recorded captions. One pass, no confidence intervals.
