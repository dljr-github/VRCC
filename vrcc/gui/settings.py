"""Settings dialog: a tabbed editor bound live to the :class:`ConfigStore`.

Controls write their field immediately and call ``save_soon()`` (a ``_loading``
guard suppresses this at construction); startup-only fields show a restart
banner. Page bodies live in ``settings_pages`` / ``settings_advanced``.
"""

from __future__ import annotations

import logging

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from vrcc.core.config import ConfigStore, apply_profile
from vrcc.core.hardware import device_names
from vrcc.gui import model_fit, settings_advanced, settings_pages
from vrcc.gui.style import PALETTE, resolve_theme
from vrcc.i18n import tr
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS

logger = logging.getLogger("vrcc.gui.settings")

_AUTO = "auto"

# Fields read only at startup build time -> changing them needs a restart (drives
# the banner). Model fields + translate.enabled hotswap live, so are absent here.
_RESTART_FIELDS = (
    ("audio", "device"),
    ("stt", "device"),
    ("stt", "device_index"),
    ("stt", "compute_type"),
    ("stt", "cpu_threads"),
    ("stt", "num_workers"),
    ("translate", "device"),
    ("translate", "device_index"),
    ("translate", "compute_type"),
    ("translate", "inter_threads"),
    ("translate", "intra_threads"),
    ("translate", "max_queued_batches"),
    ("osc", "ip"),
    ("osc", "port"),
    # The token bucket snapshots these at ChatboxSender construction.
    ("osc", "min_interval_s"),
    ("osc", "burst"),
    ("mute_sync", "enabled"),
    ("gui", "theme"),
    ("gui", "font_scale"),
    ("gui", "ui_language"),
    ("vad", "speculative_silence_ms"),
    ("vad", "finalize_silence_ms"),
    ("vad", "min_utterance_ms"),
    ("vad", "pre_roll_ms"),
    ("vad", "max_utterance_s"),
)

class SettingsDialog(QDialog):
    def __init__(
        self,
        config_store: ConfigStore,
        parent=None,
        *,
        download_manager=None,
        on_model_change=None,
    ) -> None:
        super().__init__(parent)
        self._store = config_store
        self._cfg = config_store.config
        # Theme-resolved hint/warning styles so hints read correctly in light mode.
        # Inline px sizes scale with the text-size preset (QSS wins over setFont).
        p = PALETTE[resolve_theme(self._cfg.gui.theme)]
        scale = max(0.5, min(2.0, self._cfg.gui.font_scale))
        self._muted_style = f"color: {p['muted']}; font-size: {round(11 * scale)}px;"
        self._warn_style = (
            f"color: {p['warn']}; background: transparent; "
            f"border: 1px solid {p['warn']}; padding: 6px; border-radius: 4px;"
        )
        # Optional collaborators (None when headless): download_manager filters
        # combos to downloaded-only; on_model_change(kind) hotswaps a model change.
        self._download_manager = download_manager
        self._on_model_change = on_model_change
        self._loading = True

        self.setWindowTitle(tr("Settings"))
        self.resize(660, 600)

        self._initial_restart = self._snapshot_restart_fields()
        # Widget refs the profile-reset buttons / Mode control rewrite.
        self._vad_spins: dict[str, QSpinBox | QDoubleSpinBox] = {}
        self._stt_beam_spin: QSpinBox | None = None
        self._stt_temp_spin: QDoubleSpinBox | None = None
        self._mt_beam_spin: QSpinBox | None = None
        self._model_combo: QComboBox | None = None
        self._english_only_indices: list[int] = []
        # Currently-selected model id per combo, so a cancelled fit-warning
        # switch can revert the combo without re-firing its handler.
        self._voice_selected_id: str | None = None
        self._mt_selected_id: str | None = None
        self._mode: SegmentedControl | None = None
        self._mode_desc: QLabel | None = None

        self._build_ui()
        self._loading = False
        self._update_restart_banner()

    def _changed(self) -> None:
        if self._loading:
            return
        self._store.save_soon()
        self._update_restart_banner()

    def _snapshot_restart_fields(self) -> dict[tuple[str, str], object]:
        return {
            (sec, field): getattr(getattr(self._cfg, sec), field)
            for sec, field in _RESTART_FIELDS
        }

    def _update_restart_banner(self) -> None:
        changed = any(
            getattr(getattr(self._cfg, sec), field) != self._initial_restart[(sec, field)]
            for sec, field in _RESTART_FIELDS
        )
        self._restart_banner.setVisible(changed)

    def _downloaded_whisper_specs(self) -> list:
        """Whisper specs to offer: downloaded-only with a manager, else all."""
        specs = list(WHISPER_MODELS.values())
        if self._download_manager is None:
            return specs
        return [s for s in specs if self._download_manager.is_whisper_downloaded(s.id)]

    def _downloaded_mt_specs(self) -> list:
        """MT specs to offer: downloaded-only with a manager, else all."""
        specs = list(MT_MODELS.values())
        if self._download_manager is None:
            return specs
        return [s for s in specs if self._download_manager.is_mt_downloaded(s)]

    def _confirm_model_fit(self, model_id: str, registry: dict, device: str) -> bool:
        """Whether a switch to ``model_id`` proceeds; ``False`` only when a
        :func:`model_fit.vram_warning` prompt is declined (caller reverts)."""
        spec = registry.get(model_id)
        if spec is None:
            return True
        msg = model_fit.vram_warning(spec.size_mb, device)
        if not msg:
            return True
        answer = QMessageBox.question(
            self,
            tr("Large model"),
            msg + "\n\n" + tr("Switch anyway?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _revert_combo(self, combo: QComboBox, model_id) -> None:
        """Restore ``combo`` to ``model_id`` under a ``_loading`` guard (no re-fire)."""
        idx = combo.findData(model_id)
        if idx < 0:
            return
        self._loading = True
        try:
            combo.setCurrentIndex(idx)
        finally:
            self._loading = False

    def _remove_deleted_placeholder(self, combo: QComboBox) -> None:
        """Drop the disabled deleted-model placeholder (itemData None) after a
        real pick lands -- it has served its purpose. Guarded so the index
        shift from removeItem never re-fires the change handler."""
        for i in range(combo.count()):
            if combo.itemData(i) is None:
                self._loading = True
                try:
                    combo.removeItem(i)
                finally:
                    self._loading = False
                if combo is self._model_combo:
                    # Positional english-only greying: shift past the removal.
                    self._english_only_indices = [
                        j - 1 if j > i else j for j in self._english_only_indices
                    ]
                return

    def _on_voice_model_changed(self, _i: int) -> None:
        if self._loading:
            return
        new_id = self._model_combo.currentData()
        if new_id is None:
            return
        if not self._confirm_model_fit(new_id, WHISPER_MODELS, self._cfg.stt.device):
            self._revert_combo(self._model_combo, self._voice_selected_id)
            return
        self._voice_selected_id = new_id
        self._cfg.stt.model = new_id
        self._remove_deleted_placeholder(self._model_combo)
        self._changed()
        if self._on_model_change is not None:
            self._on_model_change("stt")

    def _on_mt_model_changed(self, _i: int) -> None:
        if self._loading:
            return
        new_id = self._translate_model_combo.currentData()
        if new_id is None:
            return
        if not self._confirm_model_fit(new_id, MT_MODELS, self._cfg.translate.device):
            self._revert_combo(self._translate_model_combo, self._mt_selected_id)
            return
        self._mt_selected_id = new_id
        self._cfg.translate.model = new_id
        self._remove_deleted_placeholder(self._translate_model_combo)
        self._changed()
        if self._on_model_change is not None:
            self._on_model_change("mt")

    def _on_translate_toggled(self, checked: bool) -> None:
        if self._loading:
            return
        self._cfg.translate.enabled = bool(checked)
        self._changed()
        if self._on_model_change is not None:
            self._on_model_change("mt")

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._scrolled(settings_pages.build_simple_page(self)), tr("Simple"))
        self._tabs.addTab(self._scrolled(settings_pages.build_voice_page(self)), tr("Voice recognition"))
        self._tabs.addTab(self._scrolled(settings_pages.build_translation_page(self)), tr("Translation"))
        self._tabs.addTab(self._scrolled(settings_advanced.build_vrchat_page(self)), tr("VRChat"))
        self._tabs.addTab(self._scrolled(settings_advanced.build_advanced_page(self)), tr("Advanced / Power users"))
        root.addWidget(self._tabs)

        self._restart_banner = QLabel(
            tr(
                "Some changes (device, precision, threads, connection, "
                "appearance) take effect after restarting VRCC."
            )
        )
        self._restart_banner.setWordWrap(True)
        self._restart_banner.setStyleSheet(self._warn_style)
        self._restart_banner.setVisible(False)
        root.addWidget(self._restart_banner)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        close_btn = QPushButton(tr("Close"))
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        root.addLayout(buttons)

    def _scrolled(self, page: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(page)
        return scroll

    def _spin(self, lo: int, hi: int, value: int) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(int(value))
        return s

    def _dspin(self, lo: float, hi: float, value: float, decimals: int, step: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSingleStep(step)
        s.setValue(float(value))
        return s

    def _anchored_slider(self, slider: QSlider, extra: QWidget | None = None):
        low, high = QLabel(tr("Low")), QLabel(tr("High"))
        low.setStyleSheet(self._muted_style)
        high.setStyleSheet(self._muted_style)
        row = QHBoxLayout()
        for w in (low, slider, high, extra):
            if w is not None:
                row.addWidget(w)
        holder = QWidget()
        holder.setLayout(row)
        return holder, low, high

    def _bind_int(self, spin: QSpinBox, section, field) -> None:
        def on_change(v):
            if self._loading:
                return
            setattr(section, field, int(v))
            self._changed()
        spin.valueChanged.connect(on_change)

    def _bind_float(self, spin: QDoubleSpinBox, section, field) -> None:
        def on_change(v):
            if self._loading:
                return
            setattr(section, field, float(v))
            self._changed()
        spin.valueChanged.connect(on_change)

    def _bind_checkbox(self, box: QCheckBox, section, field) -> None:
        def on_change(checked):
            if self._loading:
                return
            setattr(section, field, bool(checked))
            self._changed()
        box.toggled.connect(on_change)

    def _bind_line(self, edit: QLineEdit, section, field) -> None:
        def on_change(text):
            if self._loading:
                return
            setattr(section, field, text)
            self._changed()
        edit.textChanged.connect(on_change)

    def _bind_text_combo(self, combo: QComboBox, section, field) -> None:
        def on_change(_i):
            if self._loading:
                return
            setattr(section, field, combo.currentText())
            self._changed()
        combo.currentIndexChanged.connect(on_change)

    def _bind_data_combo(self, combo: QComboBox, section, field) -> None:
        """Bind a combo by ``currentData()`` so the friendly label ("Dark")
        never leaks into config (which keeps "dark")."""
        def on_change(_i):
            if self._loading:
                return
            setattr(section, field, combo.currentData())
            self._changed()
        combo.currentIndexChanged.connect(on_change)

    def _device_choices(self):
        choices = [(tr("Auto"), _AUTO, 0), (tr("CPU"), "cpu", 0)]
        try:
            names = device_names()
        except Exception:  # noqa: BLE001
            names = []
        for i, name in enumerate(names):
            choices.append((tr("GPU {index}: {name}", index=i, name=name), "cuda", i))
        return choices

    def _make_device_combo(self, section) -> QComboBox:
        combo = QComboBox()
        for label, device, index in self._device_choices():
            combo.addItem(label, (device, index))
        current = (section.device, section.device_index)
        for i in range(combo.count()):
            if combo.itemData(i) == current:
                combo.setCurrentIndex(i)
                break

        def on_change(_i):
            if self._loading:
                return
            device, index = combo.currentData()
            section.device = device
            section.device_index = index
            self._changed()
        combo.currentIndexChanged.connect(on_change)
        return combo

    def _supported_compute_types(self, device: str, index: int):
        try:
            import ctranslate2

            return sorted(ctranslate2.get_supported_compute_types(device, index))
        except Exception:  # noqa: BLE001
            return []

    def _make_compute_combo(self, section) -> QComboBox:
        values = [_AUTO]
        seen = {_AUTO}
        for device, index in (("cpu", 0), ("cuda", 0)):
            for ct in self._supported_compute_types(device, index):
                if ct not in seen:
                    seen.add(ct)
                    values.append(ct)
        combo = QComboBox()
        combo.addItems(values)
        if section.compute_type not in values:
            combo.addItem(section.compute_type)
        combo.setCurrentText(section.compute_type)
        self._bind_text_combo(combo, section, "compute_type")
        return combo

    def _make_input_device_combo(self) -> QComboBox:
        """The microphone picker reused on the Simple page."""
        combo = QComboBox()
        combo.addItem(tr("Auto (system default)"), _AUTO)
        try:
            from vrcc.audio.devices import list_input_devices

            for _index, name in list_input_devices():
                combo.addItem(name, name)
        except Exception:  # noqa: BLE001
            logger.debug("could not list input devices", exc_info=True)
        cur = self._cfg.audio.device
        idx = combo.findData(cur)
        if idx < 0:
            combo.addItem(cur, cur)
            idx = combo.findData(cur)
        combo.setCurrentIndex(idx)
        combo.setToolTip(tr("Which microphone to listen to."))

        def on_device(_i):
            if self._loading:
                return
            self._cfg.audio.device = combo.currentData()
            self._changed()
        combo.currentIndexChanged.connect(on_device)
        return combo

    def _on_mode_changed(self, value: str) -> None:
        """Mode control changed: apply the matching Speed/Quality profile."""
        if self._loading:
            return
        self._apply_profile("quality" if value == "Quality" else "latency")

    def _apply_profile(self, profile: str) -> None:
        """Apply a Speed/Quality preset via ``apply_profile``, then sync widgets."""
        self._loading = True
        try:
            apply_profile(self._cfg, profile)
            for field, spin in self._vad_spins.items():
                spin.setValue(getattr(self._cfg.vad, field))
            if self._stt_beam_spin is not None:
                self._stt_beam_spin.setValue(self._cfg.stt.beam_size)
            if self._stt_temp_spin is not None:
                self._stt_temp_spin.setValue(self._cfg.stt.temperature)
            if self._mt_beam_spin is not None:
                self._mt_beam_spin.setValue(self._cfg.translate.beam_size)
            if self._mode is not None:
                self._mode.set_value("Quality" if profile == "quality" else "Speed")
        finally:
            self._loading = False
        self._changed()

    @staticmethod
    def _set_combo_text(combo: QComboBox, text: str) -> None:
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _update_english_only_items(self) -> None:
        """Grey out english-only whisper models unless the source is English."""
        if self._model_combo is None:
            return
        allow_english_only = self._source_combo.currentText() == "English"
        model = self._model_combo.model()
        for i in self._english_only_indices:
            item = model.item(i)
            if item is not None:
                item.setEnabled(allow_english_only)
