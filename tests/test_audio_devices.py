"""Tests for input-device enumeration and default-device lookup
(``vrcc.audio.devices``): host-api de-duplication preference and the
never-raise contract around PortAudio failures.
"""

from __future__ import annotations

import logging


class TestListInputDevices:
    def test_filters_output_only_devices(self, monkeypatch):
        from vrcc.audio import devices

        fake_devices = [
            {"index": 0, "name": "Mic A", "hostapi": 0, "max_input_channels": 2},
            {"index": 1, "name": "Speakers", "hostapi": 0, "max_input_channels": 0},
        ]
        fake_hostapis = [{"name": "MME"}]
        monkeypatch.setattr(devices.sd, "query_devices", lambda: fake_devices)
        monkeypatch.setattr(devices.sd, "query_hostapis", lambda: fake_hostapis)

        result = devices.list_input_devices()

        assert result == [(0, "Mic A")]

    def test_dedupes_by_name_preferring_wasapi(self, monkeypatch):
        from vrcc.audio import devices

        fake_devices = [
            {"index": 0, "name": "Logitech Mic", "hostapi": 0, "max_input_channels": 2},
            {"index": 1, "name": "Logitech Mic", "hostapi": 1, "max_input_channels": 2},
            {"index": 2, "name": "Logitech Mic", "hostapi": 2, "max_input_channels": 2},
        ]
        fake_hostapis = [
            {"name": "MME"},
            {"name": "Windows DirectSound"},
            {"name": "Windows WASAPI"},
        ]
        monkeypatch.setattr(devices.sd, "query_devices", lambda: fake_devices)
        monkeypatch.setattr(devices.sd, "query_hostapis", lambda: fake_hostapis)

        result = devices.list_input_devices()

        assert result == [(2, "Logitech Mic")]

    def test_dedupes_by_name_falls_back_to_first_seen_without_wasapi(self, monkeypatch):
        from vrcc.audio import devices

        fake_devices = [
            {"index": 0, "name": "Some Mic", "hostapi": 0, "max_input_channels": 2},
            {"index": 1, "name": "Some Mic", "hostapi": 1, "max_input_channels": 2},
        ]
        fake_hostapis = [{"name": "MME"}, {"name": "Windows DirectSound"}]
        monkeypatch.setattr(devices.sd, "query_devices", lambda: fake_devices)
        monkeypatch.setattr(devices.sd, "query_hostapis", lambda: fake_hostapis)

        result = devices.list_input_devices()

        assert result == [(0, "Some Mic")]

    def test_preserves_first_seen_order_across_distinct_names(self, monkeypatch):
        from vrcc.audio import devices

        fake_devices = [
            {"index": 0, "name": "Mic B", "hostapi": 0, "max_input_channels": 1},
            {"index": 1, "name": "Mic A", "hostapi": 0, "max_input_channels": 1},
        ]
        fake_hostapis = [{"name": "MME"}]
        monkeypatch.setattr(devices.sd, "query_devices", lambda: fake_devices)
        monkeypatch.setattr(devices.sd, "query_hostapis", lambda: fake_hostapis)

        result = devices.list_input_devices()

        assert result == [(0, "Mic B"), (1, "Mic A")]

    def test_never_raises_on_sounddevice_error(self, monkeypatch):
        from vrcc.audio import devices

        def boom():
            raise OSError("PortAudio not initialized")

        monkeypatch.setattr(devices.sd, "query_devices", boom)

        assert devices.list_input_devices() == []

    def test_never_raises_when_hostapi_lookup_fails(self, monkeypatch):
        from vrcc.audio import devices

        fake_devices = [
            {"index": 0, "name": "Mic A", "hostapi": 0, "max_input_channels": 2},
        ]
        monkeypatch.setattr(devices.sd, "query_devices", lambda: fake_devices)

        def boom():
            raise OSError("no hostapis")

        monkeypatch.setattr(devices.sd, "query_hostapis", boom)

        assert devices.list_input_devices() == []

    def test_wasapi_wins_over_mme_directsound_and_wdmks(self, monkeypatch):
        from vrcc.audio import devices

        fake_devices = [
            {"index": 0, "name": "Multi Mic", "hostapi": 0, "max_input_channels": 2},
            {"index": 1, "name": "Multi Mic", "hostapi": 1, "max_input_channels": 2},
            {"index": 2, "name": "Multi Mic", "hostapi": 2, "max_input_channels": 2},
            {"index": 3, "name": "Multi Mic", "hostapi": 3, "max_input_channels": 2},
        ]
        fake_hostapis = [
            {"name": "MME"},
            {"name": "Windows DirectSound"},
            {"name": "Windows WDM-KS"},
            {"name": "Windows WASAPI"},
        ]
        monkeypatch.setattr(devices.sd, "query_devices", lambda: fake_devices)
        monkeypatch.setattr(devices.sd, "query_hostapis", lambda: fake_hostapis)

        result = devices.list_input_devices()

        assert result == [(3, "Multi Mic")]

    def test_mme_used_over_directsound_when_no_wasapi_entry(self, monkeypatch):
        # DirectSound is listed first (lower index) so a first-seen fallback
        # would pick it; the MME entry must win on host-api preference alone.
        from vrcc.audio import devices

        fake_devices = [
            {"index": 0, "name": "Some Mic", "hostapi": 0, "max_input_channels": 2},
            {"index": 5, "name": "Some Mic", "hostapi": 1, "max_input_channels": 2},
        ]
        fake_hostapis = [{"name": "Windows DirectSound"}, {"name": "MME"}]
        monkeypatch.setattr(devices.sd, "query_devices", lambda: fake_devices)
        monkeypatch.setattr(devices.sd, "query_hostapis", lambda: fake_hostapis)

        result = devices.list_input_devices()

        assert result == [(5, "Some Mic")]

    def test_directsound_and_wdmks_only_device_is_dropped(self, monkeypatch):
        from vrcc.audio import devices

        fake_devices = [
            {"index": 0, "name": "Ghost Mic", "hostapi": 0, "max_input_channels": 2},
            {"index": 1, "name": "Ghost Mic", "hostapi": 1, "max_input_channels": 2},
            {"index": 2, "name": "Real Mic", "hostapi": 2, "max_input_channels": 2},
        ]
        fake_hostapis = [
            {"name": "Windows DirectSound"},
            {"name": "Windows WDM-KS"},
            {"name": "Windows WASAPI"},
        ]
        monkeypatch.setattr(devices.sd, "query_devices", lambda: fake_devices)
        monkeypatch.setattr(devices.sd, "query_hostapis", lambda: fake_hostapis)

        result = devices.list_input_devices()

        assert result == [(2, "Real Mic")]

    def test_two_distinct_mics_each_resolve_and_both_appear(self, monkeypatch):
        from vrcc.audio import devices

        fake_devices = [
            {"index": 0, "name": "Mic B", "hostapi": 0, "max_input_channels": 1},
            {"index": 1, "name": "Mic A", "hostapi": 1, "max_input_channels": 1},
            {"index": 2, "name": "Mic B", "hostapi": 2, "max_input_channels": 1},
        ]
        fake_hostapis = [
            {"name": "Windows DirectSound"},
            {"name": "Windows WASAPI"},
            {"name": "MME"},
        ]
        monkeypatch.setattr(devices.sd, "query_devices", lambda: fake_devices)
        monkeypatch.setattr(devices.sd, "query_hostapis", lambda: fake_hostapis)

        result = devices.list_input_devices()

        assert result == [(2, "Mic B"), (1, "Mic A")]

    def test_safety_net_falls_back_when_no_wasapi_or_mme_device_exists(
        self, monkeypatch, caplog
    ):
        from vrcc.audio import devices

        fake_devices = [
            {"index": 0, "name": "Only DS Mic", "hostapi": 0, "max_input_channels": 2},
            {"index": 1, "name": "Only DS Mic", "hostapi": 1, "max_input_channels": 2},
        ]
        fake_hostapis = [
            {"name": "Windows DirectSound"},
            {"name": "Windows WDM-KS"},
        ]
        monkeypatch.setattr(devices.sd, "query_devices", lambda: fake_devices)
        monkeypatch.setattr(devices.sd, "query_hostapis", lambda: fake_hostapis)

        with caplog.at_level(logging.DEBUG, logger="vrcc.audio"):
            result = devices.list_input_devices()

        assert result == [(0, "Only DS Mic")]
        assert any("safety net" in rec.message for rec in caplog.records)


class TestDefaultInputDevice:
    def test_returns_the_input_index(self, monkeypatch):
        from vrcc.audio import devices

        monkeypatch.setattr(devices.sd.default, "device", [3, 7])

        assert devices.default_input_device() == 3

    def test_returns_none_when_negative(self, monkeypatch):
        from vrcc.audio import devices

        monkeypatch.setattr(devices.sd.default, "device", [-1, -1])

        assert devices.default_input_device() is None

    def test_never_raises_on_sounddevice_error(self, monkeypatch):
        from vrcc.audio import devices

        class ExplodingDefault:
            @property
            def device(self):
                raise OSError("no default device")

        monkeypatch.setattr(devices, "sd", type("SD", (), {"default": ExplodingDefault()}))

        assert devices.default_input_device() is None
