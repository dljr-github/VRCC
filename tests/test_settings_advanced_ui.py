"""Offscreen GUI tests for the Advanced settings page: the device combo tells
the truth about a stored card this PC does not have, the precision combo offers
only what the selected device can run (and re-filters when it changes), and the
chatbox Separator shows its newline instead of looking empty.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from vrcc.core.config import ConfigStore, default_paths
from vrcc.gui import settings_advanced
from vrcc.gui.settings import SettingsDialog


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _store(tmp_path):
    return ConfigStore(default_paths(portable=True, app_dir=tmp_path).config_file)


def _items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def _index_of(combo, data):
    return next(i for i in range(combo.count()) if combo.itemData(i) == data)


def _pin_devices(monkeypatch, names):
    monkeypatch.setattr(settings_advanced, "device_names", lambda: names)
    # The real resolution consults the driver and the cuBLAS probe, which
    # differ between a GPU dev box and a CI runner; the filtering under test
    # does not care how the device was resolved, only what it resolved to.
    monkeypatch.setattr(
        settings_advanced,
        "resolved_device",
        lambda device, index=0, model_id=None: "cpu" if device == "auto" else device,
    )


def _pin_compute(monkeypatch, per_device):
    monkeypatch.setattr(
        settings_advanced,
        "_supported_compute_types",
        lambda device, index: list(per_device.get(device, [])),
    )


# -- a stored card this machine does not have --------------------------------


def test_missing_gpu_is_named_not_silently_shown_as_auto(qapp, tmp_path, monkeypatch):
    # A portable install carried to a PC with fewer GPUs: config says cuda:3,
    # this machine has none. Falling back to Auto would claim a setting the
    # config does not hold.
    _pin_devices(monkeypatch, [])
    store = _store(tmp_path)
    store.config.stt.device = "cuda"
    store.config.stt.device_index = 3
    dlg = SettingsDialog(store)
    try:
        combo = dlg._stt_device_combo
        assert combo.currentData() == ("cuda", 3)
        assert "3" in combo.currentText()
        assert combo.currentIndex() != 0
        # The Auto hint stays hidden because the section is not on Auto.
        assert dlg._stt_device_auto_label.isHidden()
    finally:
        dlg.close()
        dlg.deleteLater()


def test_picking_auto_after_a_missing_gpu_actually_applies(qapp, tmp_path, monkeypatch):
    # With the combo parked on index 0, choosing Auto changed no index and so
    # fired no handler: the config kept the absent card.
    _pin_devices(monkeypatch, [])
    store = _store(tmp_path)
    store.config.stt.device = "cuda"
    store.config.stt.device_index = 3
    dlg = SettingsDialog(store)
    try:
        combo = dlg._stt_device_combo
        combo.setCurrentIndex(_index_of(combo, ("auto", 0)))
        assert (store.config.stt.device, store.config.stt.device_index) == ("auto", 0)
        assert not dlg._stt_device_auto_label.isHidden()
        assert dlg._stt_device_auto_label.text()
    finally:
        dlg.close()
        dlg.deleteLater()


def test_present_gpu_still_selects_its_own_entry(qapp, tmp_path, monkeypatch):
    _pin_devices(monkeypatch, ["Fake GPU"])
    store = _store(tmp_path)
    store.config.translate.device = "cuda"
    store.config.translate.device_index = 0
    dlg = SettingsDialog(store)
    try:
        combo = dlg._mt_device_combo
        assert combo.currentData() == ("cuda", 0)
        assert combo.count() == 3  # Auto, CPU, the one GPU: no extra entry
    finally:
        dlg.close()
        dlg.deleteLater()


# -- precision follows the device --------------------------------------------


def test_compute_combo_offers_only_what_the_device_supports(qapp, tmp_path, monkeypatch):
    _pin_devices(monkeypatch, ["Fake GPU"])
    _pin_compute(
        monkeypatch,
        {"cpu": ["int8", "float32"], "cuda": ["float16", "int8", "int8_float16"]},
    )
    store = _store(tmp_path)
    store.config.stt.device = "cpu"
    dlg = SettingsDialog(store)
    try:
        combo = dlg._stt_compute_combo
        assert _items(combo) == ["auto", "int8", "float32"]
        assert "float16" not in _items(combo)  # CTranslate2 cannot do it on a CPU
    finally:
        dlg.close()
        dlg.deleteLater()


def test_compute_combo_refilters_when_the_device_changes(qapp, tmp_path, monkeypatch):
    _pin_devices(monkeypatch, ["Fake GPU"])
    _pin_compute(
        monkeypatch, {"cpu": ["int8", "float32"], "cuda": ["float16", "int8"]}
    )
    store = _store(tmp_path)
    store.config.stt.device = "cpu"
    dlg = SettingsDialog(store)
    try:
        device, compute = dlg._stt_device_combo, dlg._stt_compute_combo
        device.setCurrentIndex(_index_of(device, ("cuda", 0)))
        assert _items(compute) == ["auto", "float16", "int8"]
        compute.setCurrentText("float16")
        assert store.config.stt.compute_type == "float16"

        # Back to the processor: float16 is gone, and the setting with it.
        device.setCurrentIndex(_index_of(device, ("cpu", 0)))
        assert _items(compute) == ["auto", "int8", "float32"]
        assert store.config.stt.compute_type == "auto"
        assert compute.currentText() == "auto"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_compute_combo_keeps_a_supported_choice_across_a_device_change(
    qapp, tmp_path, monkeypatch
):
    _pin_devices(monkeypatch, ["Fake GPU"])
    _pin_compute(
        monkeypatch, {"cpu": ["int8", "float32"], "cuda": ["float16", "int8"]}
    )
    store = _store(tmp_path)
    store.config.translate.device = "cpu"
    store.config.translate.compute_type = "int8"
    dlg = SettingsDialog(store)
    try:
        device = dlg._mt_device_combo
        device.setCurrentIndex(_index_of(device, ("cuda", 0)))
        assert store.config.translate.compute_type == "int8"
        assert dlg._mt_compute_combo.currentText() == "int8"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_stored_precision_survives_merely_opening_settings(qapp, tmp_path, monkeypatch):
    # Opening the dialog is not a decision: a hand-edited value stays offered
    # and stays in config until the user moves the device themselves.
    _pin_devices(monkeypatch, ["Fake GPU"])
    _pin_compute(monkeypatch, {"cpu": ["int8", "float32"], "cuda": ["float16"]})
    store = _store(tmp_path)
    store.config.stt.device = "cpu"
    store.config.stt.compute_type = "float16"
    dlg = SettingsDialog(store)
    try:
        assert store.config.stt.compute_type == "float16"
        assert dlg._stt_compute_combo.currentText() == "float16"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_compute_values_fall_back_to_the_union_when_the_probe_is_empty(
    qapp, tmp_path, monkeypatch
):
    # No ctranslate2 (or a device it refuses to inspect): the control must not
    # collapse to Auto alone.
    _pin_devices(monkeypatch, [])
    _pin_compute(monkeypatch, {})
    store = _store(tmp_path)
    dlg = SettingsDialog(store)
    try:
        assert _items(dlg._stt_compute_combo) == ["auto"]
    finally:
        dlg.close()
        dlg.deleteLater()

    _pin_compute(monkeypatch, {"cpu": [], "cuda": ["float16"]})
    store2 = _store(tmp_path / "second")
    dlg2 = SettingsDialog(store2)
    try:
        assert _items(dlg2._stt_compute_combo) == ["auto", "float16"]
    finally:
        dlg2.close()
        dlg2.deleteLater()


# -- the chatbox separator ----------------------------------------------------


def test_separator_escapes_and_unescapes_round_trip():
    for raw in ("\n", " | ", "", "a\nb"):
        assert (
            settings_advanced.unescape_separator(
                settings_advanced.escape_separator(raw)
            )
            == raw
        )


def test_separator_field_shows_the_newline_and_writes_it_back(qapp, tmp_path):
    store = _store(tmp_path)
    assert store.config.osc.translation_separator == "\n"  # the default
    dlg = SettingsDialog(store)
    try:
        edit = dlg._separator_edit
        assert edit.text() == "\\n"  # not an empty-looking box
        assert edit.toolTip().strip()

        edit.setText(" | ")
        assert store.config.osc.translation_separator == " | "
        edit.setText("\\n")
        assert store.config.osc.translation_separator == "\n"
    finally:
        dlg.close()
        dlg.deleteLater()
