"""Settings dialog: a tabbed editor bound live to the :class:`ConfigStore`.

Controls write their field immediately and call ``save_soon()`` (a ``_loading``
guard suppresses this at construction). Every edit is also pushed into the
running app so nothing needs a restart: an optional ``apply``
(:class:`~vrcc.core.live_apply.LiveApply`) handle applies audio/VAD/OSC/mute/
engine changes live, the text-size preset retints the QApplication, and a
language change rebuilds the window on dialog close. Page bodies live in
``settings_pages`` / ``settings_advanced``.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer
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

from vrcc.core import languages
from vrcc.core.config import ConfigStore, apply_profile
from vrcc.gui import model_fit, model_prompts, settings_advanced, settings_live, settings_pages
from vrcc.gui.style import PALETTE, apply_font_scale, apply_theme_guarded, resolve_theme
from vrcc.i18n import tr
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS

_AUTO = "auto"

# Coalesce window (ms) for the live-apply flush: valueChanged fires per keystroke
# and an engine rebuild is expensive, so an edit only restarts this timer -- the
# hooks run once when it settles (or immediately on dialog close).
_APPLY_DEBOUNCE_MS = 400


class SettingsDialog(QDialog):
    def __init__(
        self,
        config_store: ConfigStore,
        parent=None,
        *,
        download_manager=None,
        on_model_change=None,
        apply=None,
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
        # Qt-free live-apply handle (None when headless / in unit tests that
        # construct the dialog bare): present -> every edit takes effect in the
        # running app with no restart.
        self._apply = apply
        self._loading = True

        self.setWindowTitle(tr("Settings"))
        self.resize(660, 600)

        # Values already live in the running stack; the debounced flush diffs
        # against this so a change applies once and a no-op re-selection doesn't.
        self._applied = settings_live.snapshot(self._specs())
        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(_APPLY_DEBOUNCE_MS)
        self._apply_timer.timeout.connect(self._apply_live_changes)
        # Widget refs the profile-reset buttons / Mode control rewrite.
        self._vad_spins: dict[str, QSpinBox | QDoubleSpinBox] = {}
        self._stt_beam_spin: QSpinBox | None = None
        self._stt_temp_spin: QDoubleSpinBox | None = None
        self._mt_beam_spin: QSpinBox | None = None
        self._model_combo: QComboBox | None = None
        # (combo index, spec) for voice models that can't transcribe every
        # language (distil English-only, Parakeet's European set).
        self._limited_model_indices: list[tuple[int, object]] = []
        # Currently-selected model id per combo, so a cancelled fit-warning
        # switch can revert the combo without re-firing its handler.
        self._voice_selected_id: str | None = None
        self._mt_selected_id: str | None = None
        self._mode: SegmentedControl | None = None
        self._mode_desc: QLabel | None = None
        # Bound by build_simple_page; greys the Mode control for voice models
        # the Speed/Quality presets can't tune (greedy onnx_asr decoders).
        self._update_mode_for_model = lambda: None

        self._build_ui()
        self._loading = False

    def _changed(self) -> None:
        if self._loading:
            return
        self._store.save_soon()
        if self._apply is not None:
            self._apply_timer.start()  # (re)arm the coalesced live-apply flush

    def _specs(self):
        return settings_live.live_specs(self._cfg, self._apply, self._apply_text_size)

    def _apply_live_changes(self) -> None:
        """Push settled edits into the running stack (GUI thread): each field
        group whose values moved since the last flush runs its hook exactly
        once. Inert when constructed headless (``apply is None``)."""
        if self._apply is None:
            return
        settings_live.flush(self._applied, self._specs())

    def _apply_text_size(self) -> None:
        """Apply the text-size preset to the live QApplication: rebuild the QSS
        at the current font scale (its baked font-size wins over setFont) and
        scale the base font, as :func:`vrcc.app.run` does at startup. The
        palette is fixed, so colors never change."""
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            return
        apply_theme_guarded(app, self._cfg.gui.theme, self._cfg.gui.font_scale)
        apply_font_scale(app, self._cfg.gui.font_scale)

    def done(self, result: int) -> None:  # noqa: N802 -- Qt override
        # accept()/reject()/close all funnel through done(); flush any edit still
        # inside the debounce window before the dialog goes away.
        self._apply_timer.stop()
        self._apply_live_changes()
        super().done(result)

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
                    # Positional language-limited greying: shift past the removal.
                    self._limited_model_indices = [
                        (j - 1 if j > i else j, spec)
                        for j, spec in self._limited_model_indices
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
        model_prompts.maybe_prefer_cpu(self, new_id)
        self._update_mode_for_model()
        model_prompts.grey_unsupported_languages(
            self._source_combo, self._cfg.stt.model,
            translating=self._cfg.translate.enabled,
        )
        self._remove_deleted_placeholder(self._model_combo)
        self._changed()
        if self._on_model_change is not None:
            self._on_model_change("stt")
            # The swap rebuilds with every current stt engine field (config is
            # re-read at build), covering the CPU-offer device flip above; the
            # debounced flush must not force a second, identical rebuild.
            settings_live.rebaseline(self._applied, self._specs(), ("stt",))

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
        # The "auto" greying in both combos depends on this flag (the onnx-asr
        # backend cannot report its detection to the translator), so the popup
        # state must track the toggle live. Guarded like the other optional
        # widgets: the Simple page hosting the toggle builds before the Voice
        # page hosting the combos.
        source = getattr(self, "_source_combo", None)
        if source is not None:
            model_prompts.grey_unsupported_languages(
                source, self._cfg.stt.model, translating=self._cfg.translate.enabled
            )
            self._update_language_limited_items()
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

    def _update_language_limited_items(self) -> None:
        """Grey out voice models that can't serve the selected spoken
        language. "auto" keeps models that detect the language within their
        set enabled, but greys those that can't detect at all (distil, which
        would transcribe as English regardless) and, while translation is on,
        the onnx-asr models: they detect but tag every result "en", which
        would mislabel the translator's source."""
        if self._model_combo is None:
            return
        source = self._source_combo.currentText()
        model = self._model_combo.model()
        for i, spec in self._limited_model_indices:
            if source == _AUTO:
                enabled = spec.auto_language and not (
                    spec.backend == "onnx_asr" and self._cfg.translate.enabled
                )
            else:
                enabled = (
                    spec.languages is None
                    or languages.get(source).whisper in spec.languages
                )
            item = model.item(i)
            if item is not None:
                item.setEnabled(enabled)
