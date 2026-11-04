"""HY-MT1.5 translation in-process through llama-cpp-python, all targets in one batch.

The CPU counterpart of :mod:`tools.llm_mt`'s llama-server engine, with the
same ``load / warm_up / unload / translate`` shape: one llama.cpp context
holds one sequence per target, and every decode step advances all of them
together in a single ``llama_decode`` call, the way NLLB's ``translate_batch``
decodes its targets together. Decoding is greedy (argmax), the arm that
scored best in benchmarks/MT.md. Bench-only; the app does not import it.
"""

from __future__ import annotations

import ctypes
import time
from pathlib import Path

import numpy as np

from tools.llm_mt import max_tokens_for, render_prompt, supports
from vrcc.core.languages import Language, get
from vrcc.translate.punctuation import normalize


class InProcessTranslator:
    def __init__(self, gguf: Path, *, threads: int = 4, parallel: int = 3,
                 ctx_per_seq: int = 512) -> None:
        self._gguf = Path(gguf)
        self._threads = threads
        self._parallel = parallel
        self._ctx_per_seq = ctx_per_seq
        self._llm = None
        self._ctx = None
        self._batch = None
        self._format = None
        self.load_s: float | None = None

    @property
    def device(self) -> str:
        return f"llama-cpp-python cpu t={self._threads} batch={self._parallel}"

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        import llama_cpp
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        t0 = time.perf_counter()
        # The Llama object is kept for its model, tokenizer and chat template;
        # its own one-sequence context is not used for decoding.
        self._llm = llama_cpp.Llama(str(self._gguf), n_ctx=256, n_threads=self._threads,
                                    n_gpu_layers=0, verbose=False)
        params = llama_cpp.llama_context_default_params()
        params.n_ctx = self._ctx_per_seq * self._parallel
        params.n_batch = self._ctx_per_seq * self._parallel
        params.n_ubatch = params.n_batch
        params.n_seq_max = self._parallel
        params.n_threads = self._threads
        params.n_threads_batch = self._threads
        params.kv_unified = False
        self._ctx = llama_cpp.llama_init_from_model(self._llm._model.model, params)
        if not self._ctx:
            raise RuntimeError("llama.cpp could not create the batched context")
        self._batch = llama_cpp.llama_batch_init(params.n_batch, 0, self._parallel)
        model = self._llm._model
        fmt = Jinja2ChatFormatter(
            template=self._llm.metadata["tokenizer.chat_template"],
            eos_token=model.token_get_text(self._llm.token_eos()),
            bos_token=model.token_get_text(self._llm.token_bos()),
        )
        self._format = lambda content: fmt(
            messages=[{"role": "user", "content": content}]).prompt
        self.load_s = time.perf_counter() - t0

    def warm_up(self) -> None:
        if not self.translate("hello", get("English"), [get("Japanese")]):
            raise RuntimeError("LLM warm-up produced no output")

    def unload(self) -> None:
        import llama_cpp

        if self._batch is not None:
            llama_cpp.llama_batch_free(self._batch)
            self._batch = None
        if self._ctx is not None:
            llama_cpp.llama_free(self._ctx)
            self._ctx = None
        if self._llm is not None:
            self._llm.close()
            self._llm = None

    # -- decoding ----------------------------------------------------------

    def _fill(self, items: list[tuple[int, int, int, bool]]) -> None:
        """Load (token, pos, seq, wants_logits) rows into the batch."""
        b = self._batch
        for i, (tok, pos, seq, logits) in enumerate(items):
            b.token[i] = tok
            b.pos[i] = pos
            b.n_seq_id[i] = 1
            b.seq_id[i][0] = seq
            b.logits[i] = 1 if logits else 0
        b.n_tokens = len(items)

    def _decode(self, items) -> None:
        import llama_cpp

        self._fill(items)
        if llama_cpp.llama_decode(self._ctx, self._batch) != 0:
            raise RuntimeError("llama_decode failed")

    def _argmax(self, row: int) -> int:
        import llama_cpp

        n_vocab = self._llm.n_vocab()
        ptr = llama_cpp.llama_get_logits_ith(self._ctx, row)
        logits = np.ctypeslib.as_array(
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), shape=(n_vocab,))
        return int(np.argmax(logits))

    def generate(self, prompts: list[str], max_tokens: list[int]) -> list[str]:
        """Greedy-decode every prompt together; one sequence id per prompt."""
        import llama_cpp

        mem = llama_cpp.llama_get_memory(self._ctx)
        llama_cpp.llama_memory_clear(mem, True)
        tokens = [self._llm.tokenize(self._format(p).encode("utf-8"),
                                     add_bos=False, special=True) for p in prompts]
        items, last_row = [], []
        for seq, toks in enumerate(tokens):
            for pos, tok in enumerate(toks):
                items.append((tok, pos, seq, pos == len(toks) - 1))
            last_row.append(len(items) - 1)
        self._decode(items)
        nxt = [self._argmax(r) for r in last_row]
        pos = [len(t) for t in tokens]
        out: list[list[int]] = [[] for _ in prompts]
        vocab = llama_cpp.llama_model_get_vocab(self._llm._model.model)
        live = set(range(len(prompts)))
        while live:
            for seq in sorted(live):
                if llama_cpp.llama_vocab_is_eog(vocab, nxt[seq]) \
                        or len(out[seq]) >= max_tokens[seq]:
                    live.discard(seq)
                else:
                    out[seq].append(nxt[seq])
            if not live:
                break
            order = sorted(live)
            self._decode([(nxt[s], pos[s], s, True) for s in order])
            for row, s in enumerate(order):
                pos[s] += 1
                nxt[s] = self._argmax(row)
        return [self._llm.detokenize(o).decode("utf-8", errors="replace").strip()
                for o in out]

    def translate(self, text: str, src: Language,
                  targets: list[Language]) -> list[tuple[str, str]]:
        if self._ctx is None:
            raise RuntimeError("translate() called before load()")
        wanted = [t for t in dict.fromkeys(targets) if t != src and supports(t)]
        if not wanted:
            return []
        outs = self.generate([render_prompt(text, src, t) for t in wanted],
                             [max_tokens_for(text)] * len(wanted))
        return [(t.display, normalize(o, t)) for t, o in zip(wanted, outs)]
