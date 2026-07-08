"""Tests for the OSCQuery server/protocol layer (real HTTP + OSC UDP over
ephemeral localhost sockets, fake Zeroconf) and ``fetch_initial_mute``.
"""

import http.client
import json
import time

from vrcc.osc.mutesync import OscQueryServer, fetch_initial_mute

MUTE_SELF = "/avatar/parameters/MuteSelf"


def _wait_until(predicate, timeout=2.0, interval=0.01):
    """Poll `predicate` until truthy or `timeout` real seconds elapse.

    Only used to synchronize with background daemon threads (HTTP/OSC
    servers, the initial-fetch thread) -- never as a stand-in for logic
    timing.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# -- fakes -----------------------------------------------------------------


class FakeZeroconf:
    """Records registered ServiceInfos; no network I/O."""

    def __init__(self) -> None:
        self.registered: list = []
        self.closed = False

    def register_service(self, info) -> None:
        self.registered.append(info)

    def close(self) -> None:
        self.closed = True


class FakeBrowser:
    """Injected in place of the real zeroconf-backed service browser.

    `find_first` returns a preset ``(ip, port)`` record (or None for "no
    VRChat found").
    """

    def __init__(self, record) -> None:
        self.record = record
        self.closed = False
        self.calls: list = []

    def find_first(self, service_type, name_prefix, timeout):
        self.calls.append((service_type, name_prefix, timeout))
        return self.record

    def close(self) -> None:
        self.closed = True


def _http_get(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


# -- OscQueryServer: real HTTP with zeroconf mocked out --------------------


def test_oscquery_http_host_info_and_namespace():
    fake_zc = FakeZeroconf()
    server = OscQueryServer(
        "VRCC-test", lambda v: None, zeroconf_factory=lambda: fake_zc
    )
    server.start()
    try:
        assert server.http_port
        assert server.osc_port

        status, body = _http_get(server.http_port, "/?HOST_INFO")
        assert status == 200
        hi = json.loads(body)
        assert hi["NAME"] == "VRCC-test"
        assert hi["OSC_IP"] == "127.0.0.1"
        assert hi["OSC_PORT"] == server.osc_port
        assert hi["OSC_TRANSPORT"] == "UDP"
        assert hi["EXTENSIONS"] == {"ACCESS": True, "VALUE": True}

        status, body = _http_get(server.http_port, "/")
        assert status == 200
        tree = json.loads(body)
        node = tree["CONTENTS"]["avatar"]["CONTENTS"]["parameters"]["CONTENTS"][
            "MuteSelf"
        ]
        assert node["FULL_PATH"] == MUTE_SELF
        assert node["ACCESS"] == 3
        assert node["TYPE"] == "T"

        status, body = _http_get(server.http_port, "/avatar/parameters")
        assert status == 200
        sub = json.loads(body)
        assert sub["FULL_PATH"] == "/avatar/parameters"
        assert sub["CONTENTS"]["MuteSelf"]["FULL_PATH"] == MUTE_SELF

        # both zeroconf services advertised
        assert len(fake_zc.registered) == 2
        types = {info.type for info in fake_zc.registered}
        assert types == {"_oscjson._tcp.local.", "_osc._udp.local."}
    finally:
        server.stop()
    assert fake_zc.closed is True


def test_oscquery_unknown_path_404():
    fake_zc = FakeZeroconf()
    server = OscQueryServer("VRCC-404", lambda v: None, zeroconf_factory=lambda: fake_zc)
    server.start()
    try:
        status, _ = _http_get(server.http_port, "/avatar/nope")
        assert status == 404
    finally:
        server.stop()


# -- OscQueryServer: real OSC UDP dispatch ---------------------------------


def test_oscquery_dispatches_mute_message():
    from pythonosc.udp_client import SimpleUDPClient

    got: list = []
    fake_zc = FakeZeroconf()
    server = OscQueryServer(
        "VRCC-osc", lambda v: got.append(v), zeroconf_factory=lambda: fake_zc
    )
    server.start()
    try:
        client = SimpleUDPClient("127.0.0.1", server.osc_port)
        client.send_message(MUTE_SELF, True)
        assert _wait_until(lambda: got == [True])
        client.send_message(MUTE_SELF, False)
        assert _wait_until(lambda: got == [True, False])
    finally:
        server.stop()


def test_oscquery_ignores_malformed_message():
    from pythonosc.udp_client import SimpleUDPClient

    got: list = []
    fake_zc = FakeZeroconf()
    server = OscQueryServer(
        "VRCC-mal", lambda v: got.append(v), zeroconf_factory=lambda: fake_zc
    )
    server.start()
    try:
        client = SimpleUDPClient("127.0.0.1", server.osc_port)
        client.send_message(MUTE_SELF, [])  # no args -> ignored
        client.send_message(MUTE_SELF, True)  # valid
        # only the valid message lands; the malformed one produced nothing
        assert _wait_until(lambda: got == [True])
    finally:
        server.stop()


def test_oscquery_zeroconf_failure_still_serves():
    class ExplodingZeroconf(FakeZeroconf):
        def register_service(self, info):
            super().register_service(info)
            if len(self.registered) == 2:  # fail on the second registration
                raise OSError("mDNS blocked by firewall")

    exploding = ExplodingZeroconf()
    server = OscQueryServer(
        "VRCC-nomdns", lambda v: None, zeroconf_factory=lambda: exploding
    )
    mdns_ok = server.start()
    try:
        assert mdns_ok is False  # advertising failed...
        # ...but the HTTP server is still up and answering.
        status, body = _http_get(server.http_port, "/?HOST_INFO")
        assert status == 200
        assert json.loads(body)["NAME"] == "VRCC-nomdns"
        # the partially-registered instance was closed (no leak).
        assert exploding.closed is True
    finally:
        server.stop()


def test_oscquery_stop_is_idempotent():
    fake_zc = FakeZeroconf()
    server = OscQueryServer("VRCC-idem", lambda v: None, zeroconf_factory=lambda: fake_zc)
    server.stop()  # before start -- no error
    server.start()
    server.stop()
    server.stop()  # twice -- no error


# -- fetch_initial_mute with injected fakes --------------------------------


def test_fetch_initial_mute_found_true():
    captured = {}

    def http_get(url):
        captured["url"] = url
        return json.dumps({"VALUE": [True]})

    browser = FakeBrowser(("127.0.0.1", 8080))
    result = fetch_initial_mute(
        zeroconf_factory=lambda: browser, http_get=http_get
    )
    assert result is True
    assert captured["url"] == "http://127.0.0.1:8080/avatar/parameters/MuteSelf"
    assert browser.closed is True


def test_fetch_initial_mute_found_false():
    result = fetch_initial_mute(
        zeroconf_factory=lambda: FakeBrowser(("127.0.0.1", 8080)),
        http_get=lambda url: json.dumps({"VALUE": [False]}),
    )
    assert result is False


def test_fetch_initial_mute_no_service():
    browser = FakeBrowser(None)
    result = fetch_initial_mute(
        timeout=0.1,
        zeroconf_factory=lambda: browser,
        http_get=lambda url: "",
    )
    assert result is None
    assert browser.closed is True


def test_fetch_initial_mute_garbage_json():
    result = fetch_initial_mute(
        zeroconf_factory=lambda: FakeBrowser(("127.0.0.1", 8080)),
        http_get=lambda url: "not json {{{",
    )
    assert result is None


def test_fetch_initial_mute_http_raises():
    def boom(url):
        raise OSError("connection refused")

    result = fetch_initial_mute(
        zeroconf_factory=lambda: FakeBrowser(("127.0.0.1", 8080)),
        http_get=boom,
    )
    assert result is None
