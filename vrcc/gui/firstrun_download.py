"""The first-run wizard's download step, and the disk check in front of it.

Split out of :mod:`vrcc.gui.firstrun` for the 500-line cap, beside
:mod:`vrcc.gui.firstrun_manual`. Functions take the wizard, the way
:mod:`vrcc.gui.firstrun_languages` does.

The worker thread reads ``wizard.recommended_whisper`` / ``recommended_mt`` at
call time, so :func:`on_download_and_start` freezes every control that can move
them before it starts, and :func:`on_download_done` hands them back.
"""

from __future__ import annotations

import logging
import threading

from vrcc.gui import firstrun_plan, model_fit
from vrcc.i18n import tr
from vrcc.stt.registry import WHISPER_MODELS
from vrcc.translate.registry import MT_MODELS

logger = logging.getLogger("vrcc.gui.firstrun")


def download_body(wizard) -> None:
    """Download the recommended STT (and MT) models, sequentially. Runs on
    a worker thread in the GUI; called directly in tests."""
    wizard._dm.ensure_whisper(wizard.recommended_whisper)
    if wizard._translation_enabled():
        wizard._dm.ensure_mt(MT_MODELS[wizard.recommended_mt])


def confirm_disk_space(wizard) -> bool:
    """Whether there is room for what the plan still has to fetch, or the user
    said to try anyway.

    The Models window refuses the same files one at a time; the wizard fetches
    a pair back to back, so the check has to cover both. Without it a 1.2 GB
    plan starts on a disk with 300 MB free and lands the user in a raw
    downloader traceback.
    """
    from PySide6.QtWidgets import QMessageBox

    message = model_fit.disk_warning(
        getattr(wizard._dm, "models_dir", None), firstrun_plan.pending_mb(wizard)
    )
    if not message:
        return True
    answer = QMessageBox.question(
        wizard,
        tr("Low disk space"),
        message + "\n\n" + tr("Download anyway?"),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes


def on_download_and_start(wizard) -> None:
    if wizard._downloading:
        return
    if not confirm_disk_space(wizard):
        return
    wizard._apply_recommendation()
    wizard._downloading = True
    wizard._set_buttons_enabled(False)
    # Downloads start here (sequentially, on the worker thread below); reveal bars now.
    wizard._whisper_bar.setVisible(True)
    if wizard._translation_enabled():
        wizard._mt_bar.setVisible(True)
    # faster-whisper downloads emit no byte progress (only a terminal done
    # event), so those get an indeterminate "busy" bar instead of a frozen 0%.
    # The onnx-asr models DO report bytes, and forcing min == max on them paints
    # a busy indicator that swallows a real percentage on a multi-hundred-MB
    # fetch. Same condition the Models window uses.
    reports_bytes = WHISPER_MODELS[wizard.recommended_whisper].backend != "whisper"
    wizard._whisper_bar.setRange(0, 100 if reports_bytes else 0)

    def worker() -> None:
        error = ""
        success = True
        try:
            # Through the wizard, not the module function: the tests patch the
            # method to stand in for a network.
            wizard._download_body()
        except Exception as exc:  # noqa: BLE001 -- surfaced via the signal
            success = False
            error = str(exc)
            logger.exception("first-run download failed")
        wizard._download_done.emit(success, error)

    threading.Thread(target=worker, name="FirstRunDownload", daemon=True).start()

def on_progress(wizard, event) -> None:
    if event.model_id == wizard.recommended_whisper:
        bar = wizard._whisper_bar
    elif event.model_id == wizard.recommended_mt:
        bar = wizard._mt_bar
    else:
        return
    if event.done:
        bar.setRange(0, 100)  # leave any indeterminate "busy" state
        bar.setValue(100)
    elif event.total > 0:
        bar.setValue(int(100 * event.downloaded / event.total))

def on_download_done(wizard, success: bool, error: str) -> None:
    wizard._downloading = False
    if success:
        wizard.accept()
        return
    # Everything back, not just Cancel: the plan inputs were frozen for the
    # download and a failed one must not leave the wizard unusable.
    # _update_proceed_enabled still re-gates the two proceed buttons on a
    # spoken-language pick.
    wizard._set_buttons_enabled(True)
    wizard._update_proceed_enabled()
    # Hide the bars and drop the indeterminate range, or the speech bar keeps
    # animating after the failure as though a download were still running.
    for bar in (wizard._whisper_bar, wizard._mt_bar):
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setVisible(False)
    from PySide6.QtWidgets import QMessageBox

    QMessageBox.warning(
        wizard,
        tr("Download failed"),
        tr(
            "Could not download the recommended models:\n\n{error}\n\n"
            "You can try again or choose existing models.",
            error=error,
        ),
    )

