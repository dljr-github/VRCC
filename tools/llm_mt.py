"""HY-MT1.5 translation through a llama.cpp ``llama-server`` child process.

Bench-only prototype of the LLM translation engine proposed for VRCC: the
same ``load / warm_up / unload / translate(text, src, targets)`` shape as
:class:`vrcc.translate.engine.TranslateEngine`, so tools/bench_mt.py's
scoring runs on it unchanged. Not imported by the app.

The server runs with one slot per chatbox target (``--parallel 3``), and
``translate()`` sends one chat request per target at the same time, so the
three decodes share the GPU the way NLLB's single batched call does.

Prompts are the ones Tencent publishes for HY-MT1.5
(https://github.com/Tencent-Hunyuan/HY-MT): the Chinese instruction for any
pair with Chinese on either side, the English one otherwise, no system
prompt. Language names are the full names from that README's table.

Licence: HY-MT1.5 is under the Tencent HY Community License, which does not
apply in the European Union, the United Kingdom or South Korea. Running this
bench there is outside the licence.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vrcc.core.languages import Language, get
from vrcc.translate.punctuation import normalize

# VRCC display name -> (English name, Chinese name) as HY-MT1.5's README
# lists them. VRCC languages missing here (Swedish, Norwegian, Danish,
# Finnish, Greek, Romanian) are not among the model's 33.
HY_MT_LANGUAGES: dict[str, tuple[str, str]] = {
    "English": ("English", "英语"),
    "Japanese": ("Japanese", "日语"),
    "Korean": ("Korean", "韩语"),
    "Chinese Simplified": ("Chinese", "中文"),
    "Chinese Traditional": ("Traditional Chinese", "繁体中文"),
    "Spanish": ("Spanish", "西班牙语"),
    "French": ("French", "法语"),
    "German": ("German", "德语"),
    "Portuguese": ("Portuguese", "葡萄牙语"),
    "Russian": ("Russian", "俄语"),
    "Italian": ("Italian", "意大利语"),
    "Indonesian": ("Indonesian", "印尼语"),
    "Thai": ("Thai", "泰语"),
    "Vietnamese": ("Vietnamese", "越南语"),
    "Arabic": ("Arabic", "阿拉伯语"),
    "Hindi": ("Hindi", "印地语"),
    "Turkish": ("Turkish", "土耳其语"),
    "Polish": ("Polish", "波兰语"),
    "Dutch": ("Dutch", "荷兰语"),
    "Ukrainian": ("Ukrainian", "乌克兰语"),
    "Filipino": ("Filipino", "菲律宾语"),
    "Malay": ("Malay", "马来语"),
    "Czech": ("Czech", "捷克语"),
    "Hebrew": ("Hebrew", "希伯来语"),
}

_ZH_TEMPLATE = "将以下文本翻译为{target}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
_XX_TEMPLATE = (
    "Translate the following segment into {target}, without additional "
    "explanation.\n\n{text}"
)

# Sampling arms. "recommended" is Tencent's published setting; "greedy" is
# what a caption probably wants (the same input always gives the same line).
SAMPLING: dict[str, dict] = {
    "greedy": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "repeat_penalty": 1.0},
    "recommended": {
        "temperature": 0.7, "top_k": 20, "top_p": 0.6, "repeat_penalty": 1.05,
    },
}

_HEALTH_TIMEOUT_S = 300.0
_REQUEST_TIMEOUT_S = 60.0


def supports(lang: Language) -> bool:
    """True when HY-MT1.5 can translate into (or from) ``lang``."""
    return lang.display in HY_MT_LANGUAGES


def _is_chinese(lang: Language) -> bool:
    return lang.display.startswith("Chinese")


def render_prompt(text: str, src: Language, tgt: Language) -> str:
    """The HY-MT1.5 user message translating ``text`` into ``tgt``."""
    english, chinese = HY_MT_LANGUAGES[tgt.display]
    if _is_chinese(src) or _is_chinese(tgt):
        return _ZH_TEMPLATE.format(target=chinese, text=text)
    return _XX_TEMPLATE.format(target=english, text=text)


def max_tokens_for(text: str) -> int:
    """Output cap: generous for a faithful translation, short enough that a
    model answering instead of translating stops quickly."""
    return min(512, 32 + 3 * len(text))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _post_json(url: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class LlamaServerTranslator:
    """One ``llama-server`` process serving one GGUF translation model.

    ``post`` is injectable for tests: ``post(url, body, timeout) -> dict``.
    """

    def __init__(
        self,
        server: Path,
        gguf: Path,
        *,
        gpu_layers: int = 99,
        parallel: int = 3,
        ctx_per_slot: int = 1024,
        sampling: str = "greedy",
        extra_args: list[str] | None = None,
        log_path: Path | None = None,
        post=_post_json,
    ) -> None:
        self._server = Path(server)
        self._gguf = Path(gguf)
        self._gpu_layers = gpu_layers
        self._parallel = parallel
        self._ctx_per_slot = ctx_per_slot
        self._params = dict(SAMPLING[sampling])
        self._extra = list(extra_args or [])
        self._log_path = log_path
        self._post = post
        self._proc: subprocess.Popen | None = None
        self._log = None
        self._pool: ThreadPoolExecutor | None = None
        self.base_url: str | None = None
        self.load_s: float | None = None

    @property
    def device(self) -> str:
        where = "cpu" if self._gpu_layers == 0 else f"gpu ngl={self._gpu_layers}"
        return f"llama.cpp {where} np={self._parallel}"

    # -- lifecycle ---------------------------------------------------------

    def command(self, port: int) -> list[str]:
        return [
            str(self._server), "-m", str(self._gguf),
            "--host", "127.0.0.1", "--port", str(port),
            "--parallel", str(self._parallel),
            "--ctx-size", str(self._ctx_per_slot * self._parallel),
            "--n-gpu-layers", str(self._gpu_layers),
            *self._extra,
        ]

    def load(self) -> None:
        """Start the server and wait until it reports the model loaded."""
        port = free_port()
        self.base_url = f"http://127.0.0.1:{port}"
        self._log = (
            open(self._log_path, "w", encoding="utf-8") if self._log_path
            else subprocess.DEVNULL
        )
        t0 = time.perf_counter()
        self._proc = subprocess.Popen(
            self.command(port), stdout=self._log, stderr=subprocess.STDOUT
        )
        self._wait_healthy()
        self.load_s = time.perf_counter() - t0
        self._pool = ThreadPoolExecutor(max_workers=self._parallel)

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + _HEALTH_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {self._proc.returncode}"
                    + (f"; see {self._log_path}" if self._log_path else "")
                )
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass  # still starting, or 503 while the model loads
            time.sleep(0.25)
        self.unload()
        raise RuntimeError("llama-server did not become healthy in time")

    def warm_up(self) -> None:
        if not self.translate("hello", get("English"), [get("Japanese")]):
            raise RuntimeError("LLM warm-up produced no output")

    def unload(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._log not in (None, subprocess.DEVNULL):
            self._log.close()
        self._log = None

    # -- translation -------------------------------------------------------

    def _one(self, text: str, src: Language, tgt: Language) -> str:
        body = {
            "messages": [{"role": "user", "content": render_prompt(text, src, tgt)}],
            "max_tokens": max_tokens_for(text),
            "seed": 0,
            "stream": False,
            **self._params,
        }
        reply = self._post(
            f"{self.base_url}/v1/chat/completions", body, _REQUEST_TIMEOUT_S
        )
        return (reply["choices"][0]["message"].get("content") or "").strip()

    def translate(
        self, text: str, src: Language, targets: list[Language]
    ) -> list[tuple[str, str]]:
        """Translate into every supported target at once, in target order.
        Unsupported targets and the source language itself are left out."""
        if self.base_url is None:
            raise RuntimeError("translate() called before load()")
        wanted = [t for t in dict.fromkeys(targets) if t != src and supports(t)]
        if self._pool is None:
            outs = [self._one(text, src, t) for t in wanted]
        else:
            outs = list(self._pool.map(lambda t: self._one(text, src, t), wanted))
        return [(t.display, normalize(o, t)) for t, o in zip(wanted, outs)]
