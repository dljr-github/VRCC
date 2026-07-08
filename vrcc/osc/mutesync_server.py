"""OSCQuery wire/discovery layer for mute sync.

VRChat only pushes ``MuteSelf`` to an OSC endpoint advertised over OSCQuery
(mDNS/zeroconf + a small HTTP service); :class:`OscQueryServer` runs just
enough of that server side to receive it. :func:`fetch_initial_mute` does the
reverse lookup (browse + HTTP GET) to seed the current value on startup.
Qt-free; imports ``pythonosc``/``zeroconf`` lazily. The higher-level state
machine (debounce, publish, ``should_caption``) lives in
:mod:`vrcc.osc.mutesync`, which composes these pieces.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from typing import Callable
from urllib.parse import urlparse

logger = logging.getLogger("vrcc.osc.mutesync_server")

MUTE_SELF_ADDRESS = "/avatar/parameters/MuteSelf"

# How long serve_forever() waits between shutdown-flag checks. Small enough
# that stop() returns promptly, large enough not to busy-spin.
_POLL_INTERVAL_S = 0.5
_JOIN_TIMEOUT_S = 2.0
# Timeout for the one-shot HTTP GET against VRChat's OSCQuery service.
_HTTP_GET_TIMEOUT_S = 2.0


# -- OSCQuery HTTP payloads -------------------------------------------------


def _namespace_tree() -> dict:
    """The full OSCQuery namespace we advertise: a single writable
    ``/avatar/parameters/MuteSelf`` boolean, nested under container nodes."""
    return {
        "FULL_PATH": "/",
        "ACCESS": 0,
        "CONTENTS": {
            "avatar": {
                "FULL_PATH": "/avatar",
                "ACCESS": 0,
                "CONTENTS": {
                    "parameters": {
                        "FULL_PATH": "/avatar/parameters",
                        "ACCESS": 0,
                        "CONTENTS": {
                            "MuteSelf": {
                                "FULL_PATH": MUTE_SELF_ADDRESS,
                                "ACCESS": 3,
                                "TYPE": "T",
                            }
                        },
                    }
                },
            }
        },
    }


def _node_for_path(path: str) -> dict | None:
    """Walk the namespace tree to the node addressed by `path`.

    ``"/"`` (or empty) returns the root; ``/avatar/parameters`` returns that
    container subtree; an unknown path returns None (-> HTTP 404).
    """
    node = _namespace_tree()
    parts = [p for p in path.split("/") if p]
    for part in parts:
        contents = node.get("CONTENTS")
        if not isinstance(contents, dict) or part not in contents:
            return None
        node = contents[part]
    return node


# -- OscQueryServer ---------------------------------------------------------


def _default_zeroconf():
    from zeroconf import Zeroconf

    return Zeroconf()


class OscQueryServer:
    """Runs the HTTP (OSCQuery) + OSC UDP servers and advertises them over mDNS.

    ``name`` is the service name VRChat sees; ``on_mute(bool)`` fires on each
    ``MuteSelf`` message; ``zeroconf_factory`` is injectable for tests. Both
    servers bind a random localhost port (exposed as :attr:`http_port`/
    :attr:`osc_port` after start); start()/stop() are thread-safe/idempotent.
    """

    def __init__(
        self,
        name: str,
        on_mute: Callable[[bool], None],
        zeroconf_factory: Callable[[], object] | None = None,
    ) -> None:
        self._name = name
        self._on_mute = on_mute
        self._zeroconf_factory = zeroconf_factory or _default_zeroconf

        self.http_port: int | None = None
        self.osc_port: int | None = None

        self._http_server = None
        self._osc_server = None
        self._http_thread: threading.Thread | None = None
        self._osc_thread: threading.Thread | None = None
        self._zeroconf = None
        self._started = False
        self._lock = threading.RLock()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> bool:
        """Bind + serve both servers, then advertise over zeroconf. Returns
        ``mdns_ok`` (False if registration failed -- servers still run, but
        VRChat may not discover us). No-op returning the prior result if
        already started.
        """
        with self._lock:
            if self._started:
                return self._zeroconf is not None

            self._osc_server = self._build_osc_server()
            self.osc_port = self._osc_server.server_address[1]

            self._http_server = self._build_http_server()
            self.http_port = self._http_server.server_address[1]

            self._osc_thread = self._serve(self._osc_server, "OscQueryOSC")
            self._http_thread = self._serve(self._http_server, "OscQueryHTTP")

            mdns_ok = self._register_zeroconf()
            self._started = True
            return mdns_ok

    def stop(self) -> None:
        """Unregister zeroconf and shut down both servers, joining their
        threads. Safe to call before :meth:`start` and more than once."""
        with self._lock:
            if not self._started:
                return
            self._started = False

            if self._zeroconf is not None:
                try:
                    self._zeroconf.close()
                except Exception:
                    logger.debug("zeroconf close failed", exc_info=True)
                self._zeroconf = None

            for server in (self._http_server, self._osc_server):
                if server is None:
                    continue
                try:
                    server.shutdown()
                    server.server_close()
                except Exception:
                    logger.debug("server shutdown failed", exc_info=True)

            for thread in (self._http_thread, self._osc_thread):
                if thread is not None:
                    thread.join(timeout=_JOIN_TIMEOUT_S)

            self._http_server = self._osc_server = None
            self._http_thread = self._osc_thread = None

    # -- OSC ---------------------------------------------------------------

    def _build_osc_server(self):
        from pythonosc.dispatcher import Dispatcher
        from pythonosc.osc_server import ThreadingOSCUDPServer

        dispatcher = Dispatcher()
        dispatcher.map(MUTE_SELF_ADDRESS, self._handle_mute)
        return ThreadingOSCUDPServer(("127.0.0.1", 0), dispatcher)

    def _handle_mute(self, addr, *osc_args) -> None:
        # pythonosc calls a no-fixed-arg handler as (address, *osc_args).
        if not osc_args:
            logger.debug("MuteSelf OSC message with no args; ignoring")
            return
        self._on_mute(bool(osc_args[0]))

    # -- HTTP --------------------------------------------------------------

    def _build_http_server(self):
        from http.server import ThreadingHTTPServer

        return ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())

    def _make_handler(self):
        from http.server import BaseHTTPRequestHandler

        server = self

        class _OscQueryHandler(BaseHTTPRequestHandler):
            # Suppress BaseHTTPRequestHandler's default stderr access logging.
            def log_message(self, *args, **kwargs) -> None:  # noqa: D401
                pass

            def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
                server._handle_get(self)

        return _OscQueryHandler

    def _handle_get(self, req) -> None:
        parsed = urlparse(req.path)
        if "HOST_INFO" in parsed.query:
            self._respond(req, 200, self._host_info())
            return
        node = _node_for_path(parsed.path or "/")
        if node is None:
            self._respond(req, 404, {"error": "not found"})
            return
        self._respond(req, 200, node)

    def _host_info(self) -> dict:
        return {
            "NAME": self._name,
            "OSC_IP": "127.0.0.1",
            "OSC_PORT": self.osc_port,
            "OSC_TRANSPORT": "UDP",
            "EXTENSIONS": {"ACCESS": True, "VALUE": True},
        }

    @staticmethod
    def _respond(req, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        req.send_response(status)
        req.send_header("Content-Type", "application/json")
        req.send_header("Content-Length", str(len(body)))
        req.end_headers()
        req.wfile.write(body)

    # -- threads + zeroconf ------------------------------------------------

    @staticmethod
    def _serve(server, name: str) -> threading.Thread:
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": _POLL_INTERVAL_S},
            name=name,
            daemon=True,
        )
        thread.start()
        return thread

    def _register_zeroconf(self) -> bool:
        zc = None
        try:
            from zeroconf import ServiceInfo

            addr = socket.inet_aton("127.0.0.1")
            oscjson = ServiceInfo(
                type_="_oscjson._tcp.local.",
                name=f"{self._name}._oscjson._tcp.local.",
                addresses=[addr],
                port=self.http_port,
                properties={},
            )
            osc = ServiceInfo(
                type_="_osc._udp.local.",
                name=f"{self._name}._osc._udp.local.",
                addresses=[addr],
                port=self.osc_port,
                properties={},
            )
            zc = self._zeroconf_factory()
            zc.register_service(oscjson)
            zc.register_service(osc)
            self._zeroconf = zc
            return True
        except Exception:
            logger.warning(
                "zeroconf registration failed; VRChat won't discover us "
                "(mute sync inactive until it can)",
                exc_info=True,
            )
            # A partially-created instance (e.g. one service registered
            # before the second raised) must be closed so we don't leak it.
            if zc is not None:
                try:
                    zc.close()
                except Exception:
                    logger.debug("zeroconf cleanup close failed", exc_info=True)
            return False


# -- fetch_initial_mute -----------------------------------------------------


class _RealZeroconfBrowser:
    """Default browser: uses real zeroconf to find the first service of a
    type whose name starts with a prefix, returning ``(ip, port)``."""

    def __init__(self) -> None:
        from zeroconf import Zeroconf

        self._zc = Zeroconf()

    def find_first(
        self, service_type: str, name_prefix: str, timeout: float
    ) -> tuple[str, int] | None:
        from zeroconf import ServiceBrowser, ServiceStateChange

        found: dict[str, tuple[str, int]] = {}
        event = threading.Event()

        def on_change(zeroconf, service_type, name, state_change) -> None:
            if state_change is not ServiceStateChange.Added:
                return
            if not name.startswith(name_prefix):
                return
            info = zeroconf.get_service_info(
                service_type, name, timeout=max(500, int(timeout * 1000))
            )
            if info and info.addresses:
                found["record"] = (socket.inet_ntoa(info.addresses[0]), info.port)
                event.set()

        browser = ServiceBrowser(self._zc, service_type, handlers=[on_change])
        try:
            if event.wait(timeout):
                return found.get("record")
            return None
        finally:
            browser.cancel()

    def close(self) -> None:
        self._zc.close()


def _default_http_get(url: str) -> str:
    from urllib.request import urlopen

    with urlopen(url, timeout=_HTTP_GET_TIMEOUT_S) as resp:
        return resp.read().decode("utf-8")


def fetch_initial_mute(
    timeout: float = 3.0,
    zeroconf_factory: Callable[[], object] | None = None,
    http_get: Callable[[str], str] | None = None,
) -> bool | None:
    """Best-effort one-shot read of VRChat's current ``MuteSelf`` value: browse
    for its OSCQuery service, HTTP GET the ``MuteSelf`` node, parse
    ``VALUE[0]``. Returns ``None`` on any failure (never raises); ``timeout``
    bounds only mDNS discovery. ``zeroconf_factory``/``http_get`` are
    injectable for tests.
    """
    make_browser = zeroconf_factory or _RealZeroconfBrowser
    get = http_get or _default_http_get

    browser = None
    try:
        browser = make_browser()
        record = browser.find_first(
            "_oscjson._tcp.local.", "VRChat-Client", timeout
        )
        if record is None:
            return None
        ip, port = record
        url = f"http://{ip}:{port}{MUTE_SELF_ADDRESS}"
        data = json.loads(get(url))
        return bool(data["VALUE"][0])
    except Exception:
        logger.debug("fetch_initial_mute failed", exc_info=True)
        return None
    finally:
        if browser is not None:
            try:
                browser.close()
            except Exception:
                logger.debug("browser close failed", exc_info=True)
