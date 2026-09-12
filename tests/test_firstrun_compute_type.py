"""The first-run wizard must size its recommendation against the compute type
the engines will actually run at, not the ``preset_for_choice`` default.

Kept apart from :mod:`tests.test_firstrun_device_ui`, which sits at the
500-line cap; reuses its ``_wizard``/``_teardown`` fixtures the way
:mod:`tests.test_firstrun_plan` reuses them for the same reason.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from tests.test_firstrun_device_ui import _teardown, _wizard
from vrcc.core import hardware, recommend


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


_FULL_COMPUTE_SUPPORT = {
    "int8_float16", "int8_bfloat16", "int8", "float16", "bfloat16",
    "int8_float32", "float32",
}

# A card whose driver/kernel build omits every int8 entry (rather than one
# keyed on compute capability, which best_compute_type no longer consults):
# the ladder in hardware.py then resolves to "float16".
_NO_INT8_COMPUTE_SUPPORT = {"float16", "bfloat16", "float32"}


def _pin_card(monkeypatch, vram_mb, supported=_FULL_COMPUTE_SUPPORT):
    """A usable CUDA card of ``vram_mb`` whose CTranslate2 ``supported`` set
    is as given. Patched on both ``recommend`` and ``hardware``: the wizard's
    own VRAM/tier reads go through the former, model_fit.vram_warning's
    graphics-card probe through the latter, and the two must agree for a test
    comparing them to mean anything. ``get_supported_compute_types`` is
    pinned too, so the result does not depend on whatever GPU (if any) runs
    the test."""
    import ctranslate2

    monkeypatch.setattr(
        ctranslate2, "get_supported_compute_types",
        lambda device, index: supported,
    )
    monkeypatch.setattr(recommend, "can_run_cuda", lambda: True)
    monkeypatch.setattr(recommend, "total_vram_bytes", lambda index=0: vram_mb * 1024**2)
    monkeypatch.setattr(hardware, "total_vram_bytes", lambda index=0: vram_mb * 1024**2)


def test_gpu_choice_on_a_card_with_no_int8_kernels_never_recommends_a_model_settings_would_flag(
    qapp, tmp_path, monkeypatch
):
    """A card whose supported compute types omit int8 entirely resolves to
    float16 (best_compute_type walks the ladder past every int8 entry), and
    the engines then pay the float16 VRAM peak, which runs 1.13x to 1.67x
    higher than int8. preset_for_choice must be sized against that same
    resolved compute type, or the wizard can recommend a model
    model_fit.vram_warning -- which does resolve it -- then flags as too
    large for the same card: the exact contradiction recommend.py promises
    never happens.
    """
    from vrcc.gui import model_fit
    from vrcc.stt.registry import WHISPER_MODELS

    _pin_card(monkeypatch, 6144, supported=_NO_INT8_COMPUTE_SUPPORT)
    wiz, store, _dm, bridge = _wizard(
        tmp_path, monkeypatch, tier="gpu_low", default_choice="gpu"
    )
    try:
        wiz._device_choice.set_value("GPU")
        cfg = store.config
        size_mb = WHISPER_MODELS[wiz.recommended_whisper].size_mb
        flagged = model_fit.vram_warning(
            size_mb, "cuda", wiz.recommended_whisper,
            cfg.stt.device_index, cfg.stt.compute_type,
        )
        assert flagged is None, (wiz.recommended_whisper, flagged)
    finally:
        _teardown(wiz, bridge)


def test_gpu_choice_on_an_int8_card_keeps_the_established_pick(
    qapp, tmp_path, monkeypatch
):
    """Paired with the no-int8-kernels test above: a card whose supported
    compute types include int8 resolves to an int8 compute type, and the pick
    for it is exactly what ranking against the int8 table gives."""
    _pin_card(monkeypatch, 6144)
    wiz, store, _dm, bridge = _wizard(
        tmp_path, monkeypatch, tier="gpu_low", default_choice="gpu"
    )
    try:
        wiz._device_choice.set_value("GPU")
        assert wiz._compute.startswith("int8")
        expected, _mt = recommend.preset_for_tier(
            "gpu_low", wiz._spoken_codes(), wiz._factor, wiz._vram_mb, "int8"
        )
        assert wiz.recommended_whisper == expected
    finally:
        _teardown(wiz, bridge)
