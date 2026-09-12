"""Render assets/splash.svg to assets/splash.png for the bootloader splash.

Needs only PySide6 and the standard library. Takes no arguments and always
produces the same file from the same SVG source.

PyInstaller's bootloader shows this PNG before the interpreter starts, and
PyInstaller/building/splash.py defaults max_img_size to (760, 480): an image
larger than that is resized only if Pillow is installed, and Pillow is not a
dependency of this project. An oversized splash would fail the release
build, not just look wrong, so the render size is checked here rather than
left for that build to discover.
"""

import sys
from pathlib import Path

from PySide6.QtCore import QBuffer, QIODevice, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

REPO_ROOT = Path(__file__).resolve().parent.parent
SVG = REPO_ROOT / "assets" / "splash.svg"
OUT = REPO_ROOT / "assets" / "splash.png"
MAX_SIZE = (760, 480)


def render_png(svg: Path) -> tuple[bytes, int, int]:
    renderer = QSvgRenderer(str(svg))
    if not renderer.isValid():
        raise SystemExit(f"unreadable SVG: {svg}")
    # width and height must track the SVG's own declaration: the max_img_size
    # check below only guards anything real if it reflects the file that
    # actually ships.
    default = renderer.defaultSize()
    width, height = default.width(), default.height()
    if width > MAX_SIZE[0] or height > MAX_SIZE[1]:
        raise SystemExit(
            f"splash size {width}x{height} exceeds PyInstaller's "
            f"max_img_size {MAX_SIZE[0]}x{MAX_SIZE[1]}; Pillow is not a "
            "dependency of this project, so a release build would fail "
            "trying to resize it"
        )
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not image.save(buffer, "PNG"):
        raise SystemExit(f"could not encode {svg.name} at {width}x{height}")
    return bytes(buffer.data()), width, height


def main() -> None:
    QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    blob, width, height = render_png(SVG)
    OUT.write_bytes(blob)
    print(f"wrote {OUT.name}: {width}x{height}")


if __name__ == "__main__":
    main()
