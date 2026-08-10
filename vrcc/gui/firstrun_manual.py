"""The first-run wizard's "Choose existing models" path.

Split out of :mod:`vrcc.gui.firstrun` for the 500-line cap, beside
:mod:`vrcc.gui.firstrun_download`: the two are the wizard's two ways of ending
up with models on disk. Functions take the wizard, the way
:mod:`vrcc.gui.firstrun_languages` does.
"""

from __future__ import annotations

from vrcc.core import recommend
from vrcc.gui import firstrun_languages
from vrcc.i18n import tr
from vrcc.translate.registry import MT_MODELS

__all__ = ["configured_models_present", "on_choose_manually", "warn_need_model"]


def configured_models_present(wizard) -> bool:
    cfg = wizard._store.config
    if not wizard._dm.is_whisper_downloaded(cfg.stt.model):
        return False
    if cfg.translate.enabled:
        spec = MT_MODELS.get(cfg.translate.model)
        if spec is None or not wizard._dm.is_mt_downloaded(spec):
            return False
    return True


def on_choose_manually(wizard) -> None:
    from vrcc.gui.models_dialog import ModelsDialog

    if wizard._downloading:
        return
    # Do NOT force the recommended preset -- "choose manually" means the user
    # picks. That dialog shares our store and offers the same language picker,
    # so re-read the answer rather than trust the wizard's own widgets.
    ModelsDialog(wizard._dm, wizard._bridge,
                 config_store=wizard._store, parent=wizard).exec()
    firstrun_languages.resync_spoken(wizard)
    # Invariant: never rewrite the MODEL config when the configured models are
    # already present -- respect the user's own pick and start. The Run-on
    # choice is the wizard's own control, so it still applies.
    if configured_models_present(wizard):
        if wizard._warn_if_source_unservable(wizard._store.config.stt.model):
            return
        wizard._apply_device_choice()
        wizard.accept()
        return
    # Configured models missing (e.g. user downloaded a different set): point
    # config at the best models on disk, or stay open with a hint if none usable.
    cfg = wizard._store.config
    whisper, mt = recommend.best_downloaded(
        wizard._dm, translate=cfg.translate.enabled, factor=wizard._factor,
        tier="cpu" if wizard._cpu_chosen() else wizard.tier,
        languages=wizard._spoken_codes(), vram_mb=wizard._vram_mb)
    if not whisper or not (mt or not cfg.translate.enabled):
        warn_need_model(wizard, has_whisper=whisper is not None)
        return
    cfg.stt.model = whisper
    if cfg.translate.enabled:
        cfg.translate.model = mt
    # The source language was derived against whatever model was recommended
    # earlier; this path installs a different one, and a source the installed
    # model cannot transcribe would caption in silence. The download path
    # re-derives via _apply_recommendation.
    firstrun_languages.resolve_source_language(
        cfg, firstrun_languages.checked_spoken(wizard), whisper,
        wizard._translation_enabled(),
    )
    # resolve_source_language leaves the stored value alone when NOTHING the
    # user ticked is servable, which is the one case that must not start:
    # captions would come out wrong in silence, and the main window's rescue
    # nudge has nothing better on disk to offer.
    if wizard._warn_if_source_unservable(whisper):
        return
    wizard._apply_device_choice()
    wizard.accept()


def warn_need_model(wizard, *, has_whisper: bool) -> None:
    from PySide6.QtWidgets import QMessageBox

    if not has_whisper:
        message = tr("Download at least a voice model to continue.")
    else:
        # Not "turn it off in Settings": Settings does not open until the
        # wizard closes, and the tick that does it is on this screen.
        # The label is substituted rather than spelled out, so it always
        # matches the checkbox this catalog actually renders. Quoted inline, it
        # named a control five translations do not have.
        message = tr(
            'Download a translation model too, or untick "{name}", to continue.',
            name=tr("Translate my speech"),
        )
    QMessageBox.information(wizard, tr("Almost there"), message)
