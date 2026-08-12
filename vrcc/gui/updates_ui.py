"""Non-modal update notice + the manual check-result handling. Qt.

Notify only: an available update opens a modeless dialog with a button that
opens the GitHub release page in the browser. No download or swap happens here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QMessageBox

from vrcc.i18n import tr

if TYPE_CHECKING:
    from vrcc.gui.main_window import MainWindow


def show_update_available(window: "MainWindow", latest: str, url: str) -> None:
    """Modeless notice with an Open download page button."""
    box = QMessageBox(window)
    box.setWindowModality(Qt.WindowModality.NonModal)
    box.setIcon(QMessageBox.Icon.Information)
    box.setWindowTitle(tr("Update available"))
    box.setText(tr("VRCC {version} is available.", version=latest))
    open_btn = box.addButton(tr("Open download page"), QMessageBox.ButtonRole.AcceptRole)
    box.addButton(tr("Later"), QMessageBox.ButtonRole.RejectRole)
    box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

    def on_click(_btn):
        if box.clickedButton() is open_btn and url:
            QDesktopServices.openUrl(QUrl(url))
    box.buttonClicked.connect(on_click)
    box.show()


def handle_result(window: "MainWindow", event) -> None:
    """Route an UpdateCheckResult to the right non-modal feedback."""
    if event.available:
        show_update_available(window, event.latest, event.url)
    elif event.error:
        window._flash_status(tr("Could not check for updates."))
    else:
        window._flash_status(tr("VRCC is up to date."))


def show_about(window: "MainWindow") -> None:
    """The About box. Here rather than in main_window because this module
    already owns the window's message boxes, and that file is at its cap."""
    from vrcc import __version__

    blurb = tr("Live voice captioning and translation for the VRChat chatbox.")
    credit = tr('Created by <a href="https://github.com/dljr-github">dljr-github</a>')
    QMessageBox.about(
        window,
        tr("About VRCC"),
        f"<p><b>VRCC</b> v{__version__}</p><p>{blurb}</p><p>{credit}</p>",
    )
