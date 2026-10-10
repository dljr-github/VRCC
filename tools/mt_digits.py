"""Pure helpers for the translation regression gate (no models, no vrcc).

The digit instrument compares the ORDERED sequence of numbers in a source with
the ones in its translation. A set comparison hides the failures that matter:
"4 4 4" collapsing to "4", and "1 2 3" coming back as "3 2 1", both pass it.
A number written out in words in the output ("10" -> "diez") counts as
preserved when the tool can read that language's number words. That is a small
table (0 to 20, the tens, hundred, thousand) plus CJK numerals, not a general
spell-out parser, so an unreadable word form is reported as DROPPED.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

KEPT, SPELLED, DROPPED, ADDED = "KEPT", "SPELLED", "DROPPED", "ADDED"


def _split(spec: str) -> list[str]:
    return spec.split(",")


_UNITS = {
    "en": "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty",
    "de": "null eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf "
    "dreizehn vierzehn fünfzehn sechzehn siebzehn achtzehn neunzehn zwanzig",
    "es": "cero uno dos tres cuatro cinco seis siete ocho nueve diez once doce "
    "trece catorce quince dieciséis diecisiete dieciocho diecinueve veinte",
    "ru": "ноль один два три четыре пять шесть семь восемь девять десять "
    "одиннадцать двенадцать тринадцать четырнадцать пятнадцать шестнадцать "
    "семнадцать восемнадцать девятнадцать двадцать",
}
_TENS = {
    "en": "thirty forty fifty sixty seventy eighty ninety",
    "de": "dreißig vierzig fünfzig sechzig siebzig achtzig neunzig",
    "es": "treinta cuarenta cincuenta sesenta setenta ochenta noventa",
    "ru": "тридцать сорок пятьдесят шестьдесят семьдесят восемьдесят девяносто",
}
# Multi-word French entries are comma separated so the space inside survives.
_FR_UNITS = (
    "zéro,un,deux,trois,quatre,cinq,six,sept,huit,neuf,dix,onze,douze,treize,"
    "quatorze,quinze,seize,dix sept,dix huit,dix neuf,vingt"
)
_FR_TENS = (
    "trente,quarante,cinquante,soixante,soixante dix,quatre vingt,"
    "quatre vingt dix"
)
_BIG = {
    "en": {"hundred": 100, "thousand": 1000},
    "de": {"hundert": 100, "tausend": 1000},
    "fr": {"cent": 100, "mille": 1000},
    "es": {"cien": 100, "ciento": 100, "mil": 1000},
    "ru": {"сто": 100, "тысяча": 1000},
}
_EXTRA = {
    "de": {"ein": 1, "eine": 1, "dreissig": 30},
    "es": {"un": 1, "una": 1, "veintiuno": 21, "veintiún": 21, "veintidós": 22,
           "veintitrés": 23, "veinticuatro": 24, "veinticinco": 25,
           "veintiséis": 26, "veintisiete": 27, "veintiocho": 28,
           "veintinueve": 29},
    "ru": {"одна": 1, "одно": 1, "две": 2},
    "fr": {"quatre vingts": 80},
}
_JOINERS = {"y", "et"}


def _build() -> dict[str, dict[str, int]]:
    tables: dict[str, dict[str, int]] = {}
    for lang in (*_UNITS, "fr"):
        units = _split(_FR_UNITS) if lang == "fr" else _UNITS[lang].split()
        tens = _split(_FR_TENS) if lang == "fr" else _TENS[lang].split()
        table = {w: i for i, w in enumerate(units)}
        table.update({w: 30 + 10 * i for i, w in enumerate(tens)})
        table.update(_BIG[lang])
        table.update(_EXTRA.get(lang, {}))
        tables[lang] = table
    return tables


_WORDS = _build()

_KANJI = {"〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "兩": 2, "三": 3, "四": 4,
          "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_KANJI_MULT = {"十": 10, "百": 100, "千": 1000}
_KANJI_RUN = re.compile("[〇零一二两兩三四五六七八九十百千]+")
# Words that contain a numeral character without counting anything. A short
# list: an ideograph after a numeral is usually a counter (七章, 两只).
_KANJI_WORDS = re.compile(
    "十分|万一|萬一|不一|統一|统一|一起|一下|一样|一樣|一些|一直|一定|一般|一切"
    "|一緒|一生|一致|一同|一方|一部|一応|一旦|唯一"
)
_SINO = {"영": "〇", "공": "〇", "일": "一", "이": "二", "삼": "三", "사": "四",
         "오": "五", "육": "六", "칠": "七", "팔": "八", "구": "九", "십": "十",
         "백": "百", "천": "千"}
_NATIVE = {"하나": 1, "한": 1, "둘": 2, "두": 2, "셋": 3, "세": 3, "넷": 4,
           "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9,
           "열": 10, "스물": 20, "서른": 30, "마흔": 40, "쉰": 50}
_COUNTERS = ("시간", "마리", "번째", "분", "시", "개", "명", "번", "초", "살",
             "달", "에", "를", "은", "는", "가", "도", "만", "씩")
_PARTICLES = frozenset("에를은는가도만")
# 네 "yes", 세 "three"/"tax", 두 "two", 한 "one"/"Korea" are everyday words.
# They count as numbers only with a counter after them.
_AMBIGUOUS = frozenset("네세두한")

_LETTERS = re.compile(r"[^\W\d_]+")
_HANGUL = re.compile(r"[가-힣]+")
_DIGITS = re.compile(r"\d{1,3}(?:[,.  ]\d{3})+(?!\d)|\d+")


@dataclass(frozen=True)
class Num:
    """A number found in a text. ``digits`` is set when it was written in
    digits; ``value`` is None for digit strings that carry no value ("0042",
    where the leading zeros are the point)."""

    digits: str | None
    value: int | None


def _kanji_value(run: str) -> int:
    if not any(c in _KANJI_MULT for c in run):
        return int("".join(str(_KANJI[c]) for c in run))
    total, cur = 0, 0
    for c in run:
        if c in _KANJI:
            cur = _KANJI[c]
        else:
            total += (cur or 1) * _KANJI_MULT[c]
            cur = 0
    return total + cur


def _native_value(stem: str) -> int | None:
    if stem in _NATIVE:
        return _NATIVE[stem]
    for tens in ("열", "스물", "서른", "마흔", "쉰"):
        if stem.startswith(tens) and _NATIVE.get(stem[len(tens):], 10) < 10:
            return _NATIVE[tens] + _NATIVE[stem[len(tens):]]
    return None


def _hangul_value(token: str, next_token: str = "", in_list: bool = False
                  ) -> int | None:
    if token in _AMBIGUOUS:
        counters = tuple(c for c in _COUNTERS if c not in _PARTICLES)
        counted = next_token.startswith(counters) or in_list
        return _NATIVE[token] if counted else None
    if (value := _native_value(token)) is not None:
        return value
    for suffix in _COUNTERS:
        stem = token[: -len(suffix)]
        if token.endswith(suffix) and stem:
            if stem in _AMBIGUOUS and suffix in _PARTICLES:
                continue
            if (value := _native_value(stem)) is not None:
                return value
            if all(c in _SINO for c in stem):
                return _kanji_value("".join(_SINO[c] for c in stem))
    if len(token) > 1 and all(c in _SINO for c in token):
        return _kanji_value("".join(_SINO[c] for c in token))
    return None


_LIST_GAP = re.compile(r"[\s,、]*")


def _in_list(text: str, runs: list, plain: list[bool], k: int) -> bool:
    """True when run k sits in a comma or space separated list that holds an
    unambiguous Korean number ("세, 두, 하나"). A list of bare 네 does not."""
    for step in (1, -1):
        j = k
        while 0 <= j + step < len(runs):
            lo, hi = sorted((runs[j].end(), runs[j + step].start()))
            if _LIST_GAP.fullmatch(text[lo:hi]) is None:
                break
            j += step
            if plain[j]:
                return True
    return False


def _latin_words(text: str, table: dict[str, int]) -> list[tuple[int, Num]]:
    found: list[tuple[int, Num]] = []
    spans = [(m.start(), m.group()) for m in _LETTERS.finditer(text)]
    i = 0
    while i < len(spans):
        hit = None
        for n in (3, 2, 1):
            phrase = " ".join(w for _, w in spans[i : i + n])
            if len(spans[i : i + n]) == n and phrase in table:
                hit = (n, table[phrase])
                break
        if hit is None and "und" in spans[i][1]:
            lo, _, hi = spans[i][1].partition("und")
            if table.get(lo, 0) in range(1, 10) and table.get(hi, 0) >= 20:
                hit = (1, table[lo] + table[hi])
        if hit is None:
            i += 1
            continue
        n, value = hit
        j = i + n
        if value in range(20, 100) and value % 10 == 0:
            k = j + 1 if j < len(spans) and spans[j][1] in _JOINERS else j
            if k < len(spans) and table.get(spans[k][1], 0) in range(1, 10):
                value += table[spans[k][1]]
                j = k + 1
        found.append((spans[i][0], Num(None, value)))
        i = j
    return found


def numbers(text: str, lang: str = "en", words: bool = True) -> list[Num]:
    """Every number in ``text`` in reading order. ``words`` also reads number
    words in ``lang``; leave it off for a source where "one" is a pronoun."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        str(unicodedata.digit(c)) if c.isdigit() and not c.isascii() else c
        for c in text
    )
    found: list[tuple[int, Num]] = []
    for m in _DIGITS.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        keep = digits == "0" or not digits.startswith("0")
        found.append((m.start(), Num(digits, int(digits) if keep else None)))
    if words:
        lowered = text.lower()
        if lang in _WORDS:
            found += _latin_words(lowered, _WORDS[lang])
        if lang in ("ja", "zh"):
            words_at = [m.span() for m in _KANJI_WORDS.finditer(text)]
            for m in _KANJI_RUN.finditer(text):
                if any(a < m.end() and m.start() < b for a, b in words_at):
                    continue
                found.append((m.start(), Num(None, _kanji_value(m.group()))))
        if lang == "ko":
            runs = list(_HANGUL.finditer(text))
            plain = [
                _hangul_value(m.group(), runs[k + 1].group() if k + 1 < len(runs)
                              else "") is not None
                for k, m in enumerate(runs)
            ]
            for k, m in enumerate(runs):
                nxt = runs[k + 1].group() if k + 1 < len(runs) else ""
                value = _hangul_value(m.group(), nxt, _in_list(text, runs, plain, k))
                if value is not None:
                    found.append((m.start(), Num(None, value)))
    return [num for _, num in sorted(found, key=lambda p: p[0])]


def _same(a: Num, b: Num) -> str | None:
    if a.digits is not None and a.digits == b.digits:
        return KEPT
    if a.value is not None and a.value == b.value:
        return KEPT if (a.digits is None) == (b.digits is None) else SPELLED
    return None


def compare(
    src: str, out: str, src_lang: str = "en", out_lang: str = "en",
    spoken_source: bool = False,
) -> dict[str, int]:
    """Per-number tallies. Each source number is searched for, in order, in
    what is left of the output; a hit consumes everything up to it, so a
    reordering shows up as DROPPED plus ADDED."""
    want = numbers(src, src_lang, words=spoken_source)
    have = numbers(out, out_lang)
    tally = {KEPT: 0, SPELLED: 0, DROPPED: 0, ADDED: 0}
    pos = 0
    used: set[int] = set()
    for num in want:
        for j in range(pos, len(have)):
            verdict = _same(num, have[j])
            if verdict:
                tally[verdict] += 1
                used.add(j)
                pos = j + 1
                break
        else:
            tally[DROPPED] += 1
    tally[ADDED] = sum(
        1 for j, h in enumerate(have) if h.digits is not None and j not in used
    )
    return tally


def classify(
    src: str, out: str, src_lang: str = "en", out_lang: str = "en",
    spoken_source: bool = False,
) -> str:
    """One verdict per pair: DROPPED if any source number is missing or out of
    order, else ADDED if the output invented a digit run, else SPELLED if any
    number was written out in words, else KEPT. Digits regrouped across spaces
    ("555 1212" -> "5551212") count as KEPT."""
    t = compare(src, out, src_lang, out_lang, spoken_source)
    if t[DROPPED] or t[ADDED]:
        want = "".join(n.digits or "" for n in numbers(src, src_lang, False))
        have = "".join(n.digits or "" for n in numbers(out, out_lang, False))
        runs = [n.digits for n in numbers(src, src_lang, False)]
        if want and want == have and all(len(d) >= 3 for d in runs):
            return KEPT
    if t[DROPPED]:
        return DROPPED
    if t[ADDED]:
        return ADDED
    return SPELLED if t[SPELLED] else KEPT


def merged(src: str, out: str, src_lang: str = "en", out_lang: str = "en") -> bool:
    """True when every source digit is present, in order, but the runs were
    joined or split differently ("8 8 8 8" -> "8888"). classify() still calls
    that DROPPED, since a counted sequence has become one number; this lets a
    report separate it from a number that is simply gone."""
    want = "".join(n.digits or "" for n in numbers(src, src_lang, False))
    have = "".join(n.digits or "" for n in numbers(out, out_lang, False))
    return bool(want) and want == have


def longest_run(text: str, unit: int = 4) -> int:
    """How many times the most-repeated `unit`-character slice repeats back to
    back. A translator stuck in a loop scores high; ordinary text does not."""
    best = 1
    for start in range(max(0, len(text) - unit)):
        piece = text[start : start + unit]
        if not piece.strip():
            continue
        run = 1
        while text.startswith(piece * (run + 1), start):
            run += 1
        best = max(best, run)
    return best


_DASH = re.compile("^\\s*[-\u2010-\u2015]")


def leading_dash(src: str, out: str) -> bool:
    """True when the output opens with a dash the source did not have. NLLB
    prefixes "- " to one-to-three word fragments as if they were dialogue."""
    return bool(_DASH.match(out)) and not _DASH.match(src)
