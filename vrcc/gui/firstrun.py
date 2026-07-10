"""First-run wizard: pick a hardware-appropriate model preset and download it.

Shown by app.run when the configured models are missing. Proposes the
recommend-tier STT+MT preset plus a "You speak"/"They read" language picker,
then downloads both models on a background thread. DownloadManager is injected
so tests drive the flow without a network.
"""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from vrcc.core import hardware, recommend
from vrcc.core.config import ConfigStore
from vrcc.core.languages import LANGUAGES
from vrcc.gui.bridge import BusBridge
from vrcc.gui.model_labels import fmt_size, mt_display_name, whisper_display_name
from vrcc.gui.models_dialog import ModelsDialog
from vrcc.gui.style import PALETTE, resolve_theme
from vrcc.gui.widgets import SegmentedControl, arrow_svg, icon_label, no_wheel
from vrcc.i18n import tr, tr_noop
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS

logger = logging.getLogger("vrcc.gui.firstrun")

_AUTO = "auto"

_DEVICE_TOOLTIP = tr_noop(
    "GPU gives near-instant captions but uses video memory (VRAM) that "
    "VRChat also needs. CPU is a little slower and leaves your graphics "
    "card alone."
)

_SIZE_TRADEOFF = tr_noop(
    "Bigger models caption more accurately, but respond more slowly and use "
    "more memory. The picks below balance that for your choice."
)


class FirstRunWizard(QDialog):
    """Propose + download a hardware-appropriate model preset on first run."""

    _download_done = Signal(bool, str)  # success, error

    def __init__(
        self,
        config_store: ConfigStore,
        download_manager,
        bridge: BusBridge,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._store = config_store
        self._dm = download_manager
        self._bridge = bridge
        self._downloading = False

        self.tier = recommend.detect_tier()
        # Default device: GPU only when the card has >=16 GB VRAM, else CPU
        # (user decision); _refresh_plan re-derives this pair on change.
        self._default_choice = recommend.default_device_choice()
        self.recommended_whisper, self.recommended_mt = recommend.preset_for_choice(
            self._default_choice, tier=self.tier, language=self._source_whisper_code()
        )
        # Resolved once at construction (theme + text size are restart-applied).
        self._p = PALETTE[resolve_theme(self._store.config.gui.theme)]
        self._scale = max(0.5, min(2.0, self._store.config.gui.font_scale))

        self.setWindowTitle(tr("Welcome to VRCC"))
        self.setModal(True)
        # Tall enough for the device row + explainer added to the download section.
        self.resize(560, 500)
        self._build_ui()
        # Config mirrors the visible default from the start, so the Models
        # dialog's tier badge never disagrees with the Run-on control.
        self._apply_device_choice()

        self._bridge.download_progress.connect(self._on_progress)
        self._download_done.connect(self._on_download_done)

    # -- construction ------------------------------------------------------

    def _translation_enabled(self) -> bool:
        return self._store.config.translate.enabled

    def _source_whisper_code(self) -> str | None:
        """Whisper code for the configured spoken language; ``None`` for
        "auto" (or an unknown name), keeping the recommendation language-blind."""
        lang = LANGUAGES.get(self._store.config.stt.source_language)
        return None if lang is None else lang.whisper

    def _total_mb(self) -> int:
        total = WHISPER_MODELS[self.recommended_whisper].size_mb
        if self._translation_enabled():
            total += MT_MODELS[self.recommended_mt].size_mb
        return total

    def _section_label(self, text: str) -> QLabel:
        # ~1.15em over the 14px body, with top spacing so section heads read as a step.
        label = QLabel(text)
        label.setStyleSheet(
            f"font-size: {round(16 * self._scale)}px; font-weight: 700; "
            f"margin-top: 10px; color: {self._p['text']};"
        )
        return label

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(14)

        self._headline = QLabel(tr("Welcome to VRCC. Let's get you captioning."))
        self._headline.setStyleSheet(  # ~1.4em headline, bold
            f"font-size: {round(20 * self._scale)}px; font-weight: 700; "
            f"color: {self._p['text']};"
        )
        self._headline.setWordWrap(True)
        root.addWidget(self._headline)

        subtitle = QLabel(
            tr(
                "Pick the languages you'll use, then download the voice and "
                "translation models VRCC needs to caption (and translate) your "
                "speech."
            )
        )
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        # -- language picker ("You speak" / "They read") -----------------------
        root.addWidget(self._section_label(tr("Pick your languages")))

        lang_row = QHBoxLayout()
        lang_row.setSpacing(8)
        lang_row.addWidget(QLabel(tr("You speak")))
        self._source_combo = no_wheel(QComboBox())
        self._source_combo.addItems([_AUTO, *LANGUAGES.keys()])
        self._set_combo_text(self._source_combo, self._store.config.stt.source_language)
        self._source_combo.currentTextChanged.connect(self._on_source_changed)
        lang_row.addWidget(self._source_combo)

        lang_row.addWidget(
            icon_label(arrow_svg(self._p["muted"]), 16, colors=self._p, fallback_text="->")
        )
        lang_row.addWidget(QLabel(tr("They read")))
        self._target_combo = no_wheel(QComboBox())
        self._target_combo.addItems(list(LANGUAGES.keys()))
        existing_targets = self._store.config.translate.targets
        self._set_combo_text(
            self._target_combo, existing_targets[0] if existing_targets else "Japanese"
        )
        self._target_combo.currentTextChanged.connect(self._on_target_changed)
        lang_row.addWidget(self._target_combo)
        lang_row.addStretch(1)
        root.addLayout(lang_row)

        # -- model download proposal -------------------------------------------
        root.addWidget(self._section_label(tr("Download the voice + translation models")))

        explainer = QLabel(tr(_SIZE_TRADEOFF))
        explainer.setWordWrap(True)
        explainer.setStyleSheet(f"color: {self._p['muted']};")
        root.addWidget(explainer)

        device_row = QHBoxLayout()
        device_row.setSpacing(8)
        run_on_label = QLabel(tr("Run on"))
        run_on_label.setToolTip(tr(_DEVICE_TOOLTIP))
        device_row.addWidget(run_on_label)
        self._device_choice = SegmentedControl(
            [("CPU", tr("CPU")), ("GPU", tr("GPU"))],
            "GPU" if self._default_choice == "gpu" else "CPU",
        )
        self._device_choice.setToolTip(tr(_DEVICE_TOOLTIP))
        if self.tier == "cpu":
            # Disabled even when a card is visible: the wizard only offers
            # what the recommender stands behind, and an expert can still
            # pin cuda in Settings.
            if hardware.cuda_device_count() > 0:
                tooltip = tr(
                    "This version of VRCC cannot use your graphics card. "
                    "The CUDA download can use it."
                )
            else:
                tooltip = tr("No graphics card detected.")
            self._device_choice.set_option_enabled("GPU", False, tooltip=tooltip)
        self._device_choice.changed.connect(self._on_device_changed)
        device_row.addWidget(self._device_choice)
        device_row.addStretch(1)
        root.addLayout(device_row)

        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        root.addWidget(self._summary_label)
        self._refresh_plan()

        if self._translation_enabled():
            mt = MT_MODELS[self.recommended_mt]
            # The single license mention lives here (the summary above never repeats it).
            note = QLabel(
                tr(
                    "Note: the translation model is licensed {license} "
                    "(free for personal, non-commercial use).",
                    license=mt.license,
                )
            )
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {self._p['muted']};")
            root.addWidget(note)

        # Both bars start hidden -- shown only once their download starts (in
        # _on_download_and_start), so the wizard never shows two empty bars up front.
        self._whisper_bar = QProgressBar()
        self._whisper_bar.setRange(0, 100)
        self._whisper_bar.setFormat(tr("Speech model: %p%"))
        self._whisper_bar.setVisible(False)
        root.addWidget(self._whisper_bar)

        self._mt_bar = QProgressBar()
        self._mt_bar.setRange(0, 100)
        self._mt_bar.setFormat(tr("Translation model: %p%"))
        self._mt_bar.setVisible(False)
        root.addWidget(self._mt_bar)

        root.addStretch(1)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self._download_btn = QPushButton(tr("Download && start"))
        self._download_btn.setDefault(True)
        self._download_btn.setProperty("buttonRole", "primary")
        self._download_btn.clicked.connect(self._on_download_and_start)
        buttons.addWidget(self._download_btn)

        self._manual_btn = QPushButton(tr("Choose existing models…"))
        self._manual_btn.clicked.connect(self._on_choose_manually)
        buttons.addWidget(self._manual_btn)

        buttons.addStretch(1)
        self._cancel_btn = QPushButton(tr("Cancel"))
        self._cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(self._cancel_btn)
        root.addLayout(buttons)

    # -- device choice + plan refresh ----------------------------------------

    def _cpu_chosen(self) -> bool:
        return self._device_choice.value() == "CPU"

    def _on_device_changed(self, _value: str) -> None:
        # Mirror the choice into config immediately (not just on accept) so
        # anything opened from the wizard -- e.g. the Models dialog's
        # "Recommended for your PC" badge via tier_for_config -- agrees with
        # the visible selection instead of falling back to the detected tier.
        self._apply_device_choice()
        self._refresh_plan()

    def _refresh_plan(self) -> None:
        """Recompute the recommended preset for the device choice + spoken
        language and rewrite the Detected/Speech/Translation/Total lines in
        place."""
        self.recommended_whisper, self.recommended_mt = recommend.preset_for_choice(
            "cpu" if self._cpu_chosen() else "gpu",
            tier=self.tier,
            language=self._source_whisper_code(),
        )
        whisper = WHISPER_MODELS[self.recommended_whisper]
        tier_label = {
            "gpu_high": tr("fast graphics card"),
            "gpu_low": tr("graphics card"),
            "cpu": self._cpu_tier_label(),
        }[self.tier]
        lines = [
            tr("Detected: {tier}", tier=tier_label), "",
            tr("Speech: {label} ({size})",
               label=whisper_display_name(self.recommended_whisper),
               size=fmt_size(whisper.size_mb)),
        ]
        if self._translation_enabled():
            mt = MT_MODELS[self.recommended_mt]
            lines.append(tr("Translation: {label} ({size})",
                            label=mt_display_name(mt.id), size=fmt_size(mt.size_mb)))
        lines.append("")
        lines.append(tr("Total download: {size}", size=fmt_size(self._total_mb())))
        self._summary_label.setText("\n".join(lines))

    @staticmethod
    def _cpu_tier_label() -> str:
        """Detected-line label for the "cpu" tier. That tier also covers a
        visible CUDA device this install cannot drive (no loadable cuBLAS),
        where "no graphics card" would be plainly false."""
        if hardware.cuda_device_count() > 0:
            return tr(
                "graphics card that this version cannot use, using your processor"
            )
        return tr("no graphics card, using your processor")

    # -- language picker -----------------------------------------------------

    @staticmethod
    def _set_combo_text(combo: QComboBox, text: str) -> None:
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _on_source_changed(self, text: str) -> None:
        # The combo's initial selection is set before this signal is connected in
        # _build_ui, so this only fires on a real user edit -- no _loading guard needed.
        self._store.config.stt.source_language = text
        self._store.save_soon()
        # The recommendation is language-aware (a restricted model can lead
        # only when it covers the spoken language), so re-plan on change.
        self._refresh_plan()

    def _on_target_changed(self, text: str) -> None:
        self._store.config.translate.targets = [text]
        self._store.save_soon()

    # -- config ------------------------------------------------------------

    def _apply_device_choice(self) -> None:
        """Write the Run-on choice to config and persist soon. Every accept
        path calls this -- the visible CPU/GPU selection is never ignored."""
        cfg = self._store.config
        cfg.stt.device = cfg.translate.device = "cpu" if self._cpu_chosen() else "auto"
        self._store.save_soon()

    def _apply_recommendation(self) -> None:
        """Point config at the recommended models + chosen device, persist soon."""
        cfg = self._store.config
        cfg.stt.model = self.recommended_whisper
        if self._translation_enabled():
            cfg.translate.model = self.recommended_mt
        self._apply_device_choice()

    def _configured_models_present(self) -> bool:
        cfg = self._store.config
        if not self._dm.is_whisper_downloaded(cfg.stt.model):
            return False
        if cfg.translate.enabled:
            spec = MT_MODELS.get(cfg.translate.model)
            if spec is None or not self._dm.is_mt_downloaded(spec):
                return False
        return True

    # -- download path -----------------------------------------------------

    def _download_body(self) -> None:
        """Download the recommended STT (and MT) models, sequentially. Runs on
        a worker thread in the GUI; called directly in tests."""
        self._dm.ensure_whisper(self.recommended_whisper)
        if self._translation_enabled():
            self._dm.ensure_mt(MT_MODELS[self.recommended_mt])

    def _on_download_and_start(self) -> None:
        if self._downloading:
            return
        self._apply_recommendation()
        self._downloading = True
        self._set_buttons_enabled(False)
        # Downloads start here (sequentially, on the worker thread below); reveal bars now.
        self._whisper_bar.setVisible(True)
        if self._translation_enabled():
            self._mt_bar.setVisible(True)
        # Whisper downloads emit no byte progress (only a terminal done event), so
        # show an indeterminate "busy" bar instead of a frozen 0%.
        self._whisper_bar.setRange(0, 0)

        def worker() -> None:
            error = ""
            success = True
            try:
                self._download_body()
            except Exception as exc:  # noqa: BLE001 -- surfaced via the signal
                success = False
                error = str(exc)
                logger.exception("first-run download failed")
            self._download_done.emit(success, error)

        threading.Thread(target=worker, name="FirstRunDownload", daemon=True).start()

    def _on_progress(self, event) -> None:
        if event.model_id == self.recommended_whisper:
            bar = self._whisper_bar
        elif event.model_id == self.recommended_mt:
            bar = self._mt_bar
        else:
            return
        if event.done:
            bar.setRange(0, 100)  # leave any indeterminate "busy" state
            bar.setValue(100)
        elif event.total > 0:
            bar.setValue(int(100 * event.downloaded / event.total))

    def _on_download_done(self, success: bool, error: str) -> None:
        self._downloading = False
        if success:
            self.accept()
            return
        self._set_buttons_enabled(True)
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.warning(
            self,
            tr("Download failed"),
            tr(
                "Could not download the recommended models:\n\n{error}\n\n"
                "You can try again or choose existing models.",
                error=error,
            ),
        )

    # -- manual path -------------------------------------------------------

    def _on_choose_manually(self) -> None:
        if self._downloading:
            return
        # Do NOT force the recommended preset here -- "choose manually" means the
        # user picks; the models dialog lets them download any model.
        ModelsDialog(
            self._dm, self._bridge, config_store=self._store, parent=self
        ).exec()
        # Invariant: never rewrite the MODEL config when the configured models
        # are already present -- respect the user's own pick and start. The
        # Run-on choice is this wizard's own control, so it still applies.
        if self._configured_models_present():
            self._apply_device_choice()
            self.accept()
            return
        # Configured models missing (e.g. user downloaded a different set): point
        # config at the best models on disk, or stay open with a hint if none usable.
        cfg = self._store.config
        whisper, mt = recommend.best_downloaded(
            self._dm, translate=cfg.translate.enabled,
            tier="cpu" if self._cpu_chosen() else self.tier,
            language=self._source_whisper_code(),
        )
        if whisper and (mt or not cfg.translate.enabled):
            cfg.stt.model = whisper
            if cfg.translate.enabled:
                cfg.translate.model = mt
            self._apply_device_choice()
            self.accept()
        else:
            self._warn_need_model(has_whisper=whisper is not None)

    def _warn_need_model(self, *, has_whisper: bool) -> None:
        from PySide6.QtWidgets import QMessageBox

        if not has_whisper:
            message = tr("Download at least a voice model to continue.")
        else:
            message = tr(
                "Download a translation model too, or turn off translation "
                "in Settings, to continue."
            )
        QMessageBox.information(self, tr("Almost there"), message)

    # -- helpers -----------------------------------------------------------

    def reject(self) -> None:  # noqa: N802 -- Qt override
        """Refuse to close while a download is running. Qt routes Esc and the
        titlebar X through reject() too, so this one guard keeps the daemon
        download thread from outliving the app (a mid-download exit could leave
        a partial model snapshot on disk)."""
        if self._downloading:
            logger.info("ignoring close request during first-run download")
            return
        super().reject()

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self._download_btn.setEnabled(enabled)
        self._manual_btn.setEnabled(enabled)
        self._cancel_btn.setEnabled(enabled)
