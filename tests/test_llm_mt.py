"""tools/llm_mt.py and tools/bench_llm_mt.py: HY-MT prompts, target
filtering, concurrent requests and the llama-server lifecycle. No model is
loaded; a tiny HTTP server stands in for llama-server."""

import sys
import textwrap
import time

import pytest

from tools import bench_llm_mt, bench_mt, llm_mt
from vrcc.core.languages import get

EN, JA, ZH, ZH_T, KO, FR, SV = (
    get(n) for n in ("English", "Japanese", "Chinese Simplified",
                     "Chinese Traditional", "Korean", "French", "Swedish")
)


def test_english_template_for_pairs_without_chinese():
    prompt = llm_mt.render_prompt("It's on the house.", EN, JA)
    assert prompt == (
        "Translate the following segment into Japanese, without additional "
        "explanation.\n\nIt's on the house."
    )


@pytest.mark.parametrize("src,tgt,name", [(EN, ZH, "中文"), (ZH, KO, "韩语"),
                                          (EN, ZH_T, "繁体中文")])
def test_chinese_template_when_either_side_is_chinese(src, tgt, name):
    prompt = llm_mt.render_prompt("x", src, tgt)
    assert prompt.startswith(f"将以下文本翻译为{name}，")
    assert prompt.endswith("\n\nx")


def test_bench_default_targets_are_all_supported():
    assert all(llm_mt.supports(get(n)) for n in bench_mt.DEFAULT_TARGETS)
    assert not llm_mt.supports(SV)


def test_max_tokens_grows_with_input_and_is_capped():
    assert llm_mt.max_tokens_for("hi") < llm_mt.max_tokens_for("hello " * 10)
    assert llm_mt.max_tokens_for("x" * 10_000) == 512


def _fake_translator(sampling="greedy"):
    calls = []

    def post(url, body, timeout):
        calls.append(body)
        time.sleep(0.1)  # lets the three requests overlap
        content = body["messages"][0]["content"]
        return {"choices": [{"message": {"content": f" <{content[-5:]}> "}}]}

    t = llm_mt.LlamaServerTranslator("server", "m.gguf", sampling=sampling, post=post)
    t.base_url = "http://127.0.0.1:1"
    t._pool = llm_mt.ThreadPoolExecutor(max_workers=3)
    return t, calls


def test_translate_keeps_target_order_and_drops_unsupported_and_source():
    t, calls = _fake_translator()
    out = t.translate("hello", EN, [FR, SV, EN, JA, FR])
    assert [d for d, _ in out] == ["French", "Japanese"]
    assert all(text == "<hello>" for _, text in out)  # stripped
    assert len(calls) == 2
    assert calls[0]["temperature"] == 0.0 and calls[0]["top_k"] == 1


def test_translate_requests_run_concurrently():
    t, _ = _fake_translator()
    t0 = time.perf_counter()
    t.translate("hello", EN, [FR, JA, KO])
    assert time.perf_counter() - t0 < 0.25  # three 100 ms posts, not 300 ms


def test_recommended_sampling_is_tencents_setting():
    t, calls = _fake_translator("recommended")
    t.translate("hello", EN, [JA])
    body = calls[0]
    assert (body["temperature"], body["top_k"], body["top_p"],
            body["repeat_penalty"]) == (0.7, 20, 0.6, 1.05)


def test_translate_before_load_raises():
    t = llm_mt.LlamaServerTranslator("server", "m.gguf")
    with pytest.raises(RuntimeError):
        t.translate("hello", EN, [JA])


def test_command_carries_slots_context_and_offload():
    t = llm_mt.LlamaServerTranslator("srv", "m.gguf", gpu_layers=0, parallel=3,
                                     ctx_per_slot=512, extra_args=["--jinja"])
    cmd = t.command(8080)
    assert cmd[cmd.index("--parallel") + 1] == "3"
    assert cmd[cmd.index("--ctx-size") + 1] == "1536"
    assert cmd[cmd.index("--n-gpu-layers") + 1] == "0"
    assert cmd[-1] == "--jinja"
    assert t.device == "llama.cpp cpu np=3"


_FAKE_SERVER = textwrap.dedent("""
    import json, sys, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    port = int(sys.argv[sys.argv.index("--port") + 1])
    started = time.monotonic()
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            loading = time.monotonic() - started < 0.3
            self._send(503 if loading else 200, {"status": "ok"})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            text = body["messages"][0]["content"].rsplit("\\n", 1)[-1]
            self._send(200, {"choices": [{"message": {"content": text[::-1]}}]})
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
""")


class _ScriptServer(llm_mt.LlamaServerTranslator):
    def __init__(self, script, **kw):
        super().__init__("unused", "m.gguf", **kw)
        self._script = script

    def command(self, port):
        return [sys.executable, str(self._script), "--port", str(port)]


def test_lifecycle_against_a_stand_in_server(tmp_path):
    script = tmp_path / "fake_server.py"
    script.write_text(_FAKE_SERVER, encoding="utf-8")
    t = _ScriptServer(script, log_path=tmp_path / "server.log")
    t.load()
    try:
        assert t.load_s is not None and t.load_s >= 0.3  # waited out the 503s
        t.warm_up()
        assert t.translate("abc", EN, [FR, JA]) == [("French", "cba"), ("Japanese", "cba")]
    finally:
        proc = t._proc
        t.unload()
    assert proc.poll() is not None


def test_server_that_exits_fails_load(tmp_path):
    script = tmp_path / "dies.py"
    script.write_text("import sys; sys.exit(3)", encoding="utf-8")
    t = _ScriptServer(script)
    with pytest.raises(RuntimeError, match="exited with code 3"):
        t.load()


def test_pick_gguf_matches_quant_case_insensitively():
    files = ["README.md", "HY-MT1.5-1.8B-Q4_K_M.gguf", "HY-MT1.5-1.8B-Q8_0.gguf",
             "HY-MT1.5-1.8B-Q4_K_M-imatrix.gguf"]
    assert bench_llm_mt.pick_gguf(files, "q4_k_m") == "HY-MT1.5-1.8B-Q4_K_M.gguf"
    with pytest.raises(SystemExit, match="Q5_K_M"):
        bench_llm_mt.pick_gguf(files, "Q5_K_M")


def test_score_engine_skips_unsupported_targets():
    entries = [{"src": "en", "group": "sentence", "text": "I have 3 cats.",
                "spoken": False}]
    seen = []

    def translate(text, src, targets):
        seen.append([t.display for t in targets])
        return [(t.display, "3") for t in targets]

    rows, call_ms = bench_mt.score_engine(
        translate, entries, ["Japanese", "Swedish", "French"], llm_mt.supports)
    assert seen == [["Japanese", "French"]]
    assert [r["target"] for r in rows] == ["Japanese", "French"]
    assert len(call_ms) == 1
    assert bench_llm_mt.skipped_targets(entries, ["Japanese", "Swedish"]) == 1


def test_summarize_accepts_an_llm_arm():
    arm = {"model": "hy", "rep": 1.0, "ngram": 0, "beam": "greedy",
           "device": "llama.cpp", "rows": [], "call_ms": [1.0, 3.0]}
    s = bench_mt.summarize(arm)
    assert s["ms_median"] == 2.0
    assert "| hy | 1.0 | 0 | greedy |" in bench_mt.table([s])
