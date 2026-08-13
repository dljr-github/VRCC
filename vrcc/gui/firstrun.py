"""First-run wizard: pick a hardware-appropriate model preset and download it.

Shown by app.run when the configured models are missing. Asks which languages
the user speaks (:mod:`vrcc.gui.firstrun_languages`) and who reads the
translation, proposes the recommend-tier STT+MT preset for that answer
(:mod:`vrcc.gui.firstrun_plan`), then downloads both models on a background
thread (:mod:`vrcc.gui.firstrun_download`) or takes what is already on disk
(:mod:`vrcc.gui.firstrun_manual`). DownloadManager is injected so tests drive
the flow without a network.

This module is the widgets and the wiring between them; the four modules above
hold the logic, for the 500-line cap.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from vrcc.core import calibrate, hardware, recommend
from vrcc.core.config import ConfigStore
from vrcc.core.languages import LANGUAGES
from vrcc.gui import (
    firstrun_download,
    firstrun_languages,
    firstrun_manual,
    firstrun_plan,
    mt_prompts,
)
from vrcc.gui.bridge import BusBridge
from vrcc.gui.model_labels import mt_license_note
from vrcc.gui.style import PALETTE, resolve_theme
from vrcc.gui.widgets import (
    SegmentedControl,
    arrow_svg,
    icon_label,
    no_wheel,
    set_combo_text,
)
from vrcc.i18n import tr, tr_noop

logger = logging.getLogger("vrcc.gui.firstrun")

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

        index = self._store.config.stt.device_index
        self.tier = recommend.detect_tier(index)
        # Machine speed and VRAM are read before the first recommendation so
        # the plan reads right at once (~57 ms cold, then stored). Default
        # device is GPU only at >=16 GB (user decision); _refresh_plan re-derives.
        self._factor = calibrate.cached_factor(self._store.config)
        self._vram_mb = recommend.detected_vram_mb(index)
        self._default_choice = recommend.default_device_choice(index)
        self.recommended_whisper, self.recommended_mt = recommend.preset_for_choice(
            self._default_choice, self.tier, self._spoken_codes(),
            self._factor, self._vram_mb)
        # Resolved once at construction (theme + text size are restart-applied).
        self._p = PALETTE[resolve_theme(self._store.config.gui.theme)]
        self._scale = max(0.5, min(2.0, self._store.config.gui.font_scale))

        self.setWindowTitle(tr("Welcome to VRCC"))
        self.setModal(True)
        # Tall enough for the device row, the explainer and the translation
        # tick above the plan summary.
        self.resize(560, 540)
        self._build_ui()
        # Config mirrors the visible default from the start, so the Models
        # dialog's tier badge never disagrees with the Run-on control.
        self._apply_device_choice()

        self._bridge.download_progress.connect(self._on_progress)
        self._download_done.connect(self._on_download_done)

    # -- construction ------------------------------------------------------

    def _translation_enabled(self) -> bool:
        return self._store.config.translate.enabled

    def _spoken_codes(self) -> tuple[str, ...]:
        """Whisper codes for the stored spoken answer, or the OS-locale source
        language behind it. The picker still shows the question and a pick is
        required; the locale seed only pre-fills a default the user can change."""
        return recommend.spoken_whisper_codes(self._store.config)

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

        speak_label = QLabel(tr("You speak (tick every language you use)"))
        speak_label.setStyleSheet(f"color: {self._p['muted']};")
        root.addWidget(speak_label)

        self._spoken_list = firstrun_languages.build_spoken_picker(self)
        root.addWidget(self._spoken_list)

        speak_hint = QLabel(tr("Pick at least one language you speak to continue."))
        speak_hint.setStyleSheet(f"color: {self._p['muted']};")
        root.addWidget(speak_hint)

        # Translation is declined here or nowhere: app.run shows this wizard
        # modally before any window exists, so Settings cannot be reached until
        # it closes, and without a toggle the 483 MB translation model is the
        # price of finishing first run.
        self._translate_check = QCheckBox(tr("Translate my speech"))
        # Connected after the initial state, like the language picker: the
        # handler touches widgets built further down this method.
        self._translate_check.setChecked(self._translation_enabled())
        self._translate_check.toggled.connect(self._on_translate_toggled)
        root.addWidget(self._translate_check)

        lang_row = QHBoxLayout()
        lang_row.setSpacing(8)
        lang_row.addWidget(
            icon_label(arrow_svg(self._p["muted"]), 16, colors=self._p, fallback_text="->")
        )
        self._target_label = QLabel(tr("They read"))
        lang_row.addWidget(self._target_label)
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

        # The single license mention lives here (the summary above never
        # repeats it). model_labels owns the wording, including the
        # non-commercial rider only a CC-BY-NC model earns. Text and visibility
        # follow the plan: the MT model changes with the tier, and there is
        # nothing to license once translation is off.
        self._license_note = QLabel()
        self._license_note.setWordWrap(True)
        self._license_note.setStyleSheet(f"color: {self._p['muted']};")
        root.addWidget(self._license_note)

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
        self._sync_translation_widgets()
        self._refresh_plan()

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
        languages and rewrite the Detected/Speech/Translation/Total lines in
        place."""
        self.recommended_whisper, self.recommended_mt = recommend.preset_for_choice(
            "cpu" if self._cpu_chosen() else "gpu", self.tier,
            self._spoken_codes(), self._factor, self._vram_mb,
        )
        # A target the plan's MT model writes the same way as another is dropped
        # from config on the first main-window load, so it must not be offerable.
        # Every reachable preset is an NLLB, which renders both Chinese
        # scripts, so nothing is greyed today. Kept because it is derived from
        # the plan rather than hardcoded: a preset that ever collapses a pair
        # again must not be offerable here.
        mt_prompts.grey_collapsed_targets(self._target_combo, self.recommended_mt)
        self._retarget_off_source()
        self._summary_label.setText("\n".join(firstrun_plan.summary_lines(self)))
        note = (
            mt_license_note(self.recommended_mt)
            if self._translation_enabled() else ""
        )
        self._license_note.setText(note)
        self._license_note.setVisible(bool(note))
        self._update_proceed_enabled()

    def _on_translate_toggled(self, checked: bool) -> None:
        self._store.config.translate.enabled = checked
        self._sync_translation_widgets()
        self._refresh_plan()
        # The source language rule turns on whether the translator will be
        # handed one at all: a model that detects the language but cannot say
        # which it heard only breaks the translation.
        firstrun_languages.apply_source_language(self)
        self._store.save_soon()

    def _sync_translation_widgets(self, live: bool = True) -> None:
        """Grey the "They read" row while translation is off. Greyed rather
        than hidden, so the tick above it reads as the reason. ``live`` is
        False while a download owns the controls."""
        enabled = live and self._translation_enabled()
        self._target_combo.setEnabled(enabled)
        self._target_label.setEnabled(enabled)

    def _update_proceed_enabled(self) -> None:
        """Proceed needs a spoken-language pick and no download in flight. The
        first call is the last line of _build_ui, so the buttons always exist."""
        ready = bool(firstrun_languages.checked_spoken(self)) and not self._downloading
        self._download_btn.setEnabled(ready)
        self._manual_btn.setEnabled(ready)

    # -- language picker -----------------------------------------------------

    _set_combo_text = staticmethod(set_combo_text)

    def _on_target_changed(self, text: str) -> None:
        self._store.config.translate.targets = [text]
        self._store.save_soon()

    def _retarget_off_source(self) -> None:
        firstrun_languages.retarget_off_source(self)

    # -- config ------------------------------------------------------------

    def _apply_device_choice(self) -> None:
        """Write the Run-on choice to config and persist soon. Every accept
        path calls this -- the visible CPU/GPU selection is never ignored."""
        cfg = self._store.config
        cfg.stt.device = cfg.translate.device = "cpu" if self._cpu_chosen() else "auto"
        self._store.save_soon()

    def _apply_recommendation(self) -> None:
        """Point config at the recommended models + chosen device, persist soon.

        Also commits the spoken-language ticks, which a user who accepted the
        pre-ticked default never edited -- without this the answer the wizard
        visibly acted on would not be the one written down, and a later
        re-recommend would disagree with the models it just downloaded.
        """
        cfg = self._store.config
        cfg.stt.spoken_languages = firstrun_languages.checked_spoken(self)
        firstrun_languages.apply_source_language(self)
        cfg.stt.model = self.recommended_whisper
        if self._translation_enabled():
            cfg.translate.model = self.recommended_mt
        self._apply_device_choice()

    def _configured_models_present(self) -> bool:
        return firstrun_manual.configured_models_present(self)

    # -- download path -----------------------------------------------------

    def _download_body(self) -> None:
        # Kept as a method: the worker thread calls it and the tests patch it
        # here to stand in for a network.
        firstrun_download.download_body(self)

    def _on_download_and_start(self) -> None:
        firstrun_download.on_download_and_start(self)

    def _on_progress(self, event) -> None:
        firstrun_download.on_progress(self, event)

    def _on_download_done(self, success: bool, error: str) -> None:
        firstrun_download.on_download_done(self, success, error)

    # -- manual path -------------------------------------------------------

    def _on_choose_manually(self) -> None:
        firstrun_manual.on_choose_manually(self)

    def _warn_if_source_unservable(self, whisper_id: str) -> bool:
        return firstrun_languages.warn_if_source_unservable(self, whisper_id)

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
        # The plan inputs freeze too. _apply_recommendation writes the pair to
        # config before the worker starts, but the worker re-reads
        # recommended_whisper/recommended_mt between its two fetches, so moving
        # any of these mid-download fetches one model while config names the
        # other, and startup then points the engine at a directory with no
        # model.bin in it.
        self._device_choice.setEnabled(enabled)
        self._spoken_list.setEnabled(enabled)
        self._translate_check.setEnabled(enabled)
        # Through the sync, so handing the controls back after a failed
        # download does not re-enable a row translation is switched off.
        self._sync_translation_widgets(enabled)
