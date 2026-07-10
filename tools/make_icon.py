"""Render the VRCC icon SVGs and pack them into vrcc/vrcc.ico.

Reads assets/icon/vrcc-icon.svg (32 px and up) and vrcc-icon-small.svg
(16 and 24 px, edges aligned to the pixel grid). Needs only PySide6 and
the standard library. Every ICO entry embeds the PNG bytes directly;
Windows Vista and later accept PNG-compressed entries at any size.
Takes no arguments and always produces the same file from the same SVG
sources. The ICO lives inside the package so source installs and the
frozen build resolve it identically (see vrcc.gui.style).
"""

import struct
import sys
from pathlib import Path

from PySide6.QtCore import QBuffer, QIODevice, Qt
from PySide6.QtGui import QGuiApplication, QIcon, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

REPO_ROOT = Path(__file__).resolve().parent.parent
MASTER = REPO_ROOT / "assets" / "icon" / "vrcc-icon.svg"
SMALL = REPO_ROOT / "assets" / "icon" / "vrcc-icon-small.svg"
OUT = REPO_ROOT / "vrcc" / "vrcc.ico"
SIZES = (16, 24, 32, 48, 64, 128, 256)
SMALL_MAX = 24


def render_png(svg: Path, size: int) -> bytes:
    renderer = QSvgRenderer(str(svg))
    if not renderer.isValid():
        raise SystemExit(f"unreadable SVG: {svg}")
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not image.save(buffer, "PNG"):
        raise SystemExit(f"could not encode {svg.name} at {size}px")
    return bytes(buffer.data())


def write_ico(entries: list[tuple[int, bytes]], out: Path) -> None:
    header = struct.pack("<HHH", 0, 1, len(entries))
    directory = b""
    offset = 6 + 16 * len(entries)
    for size, blob in entries:
        edge = 0 if size >= 256 else size
        directory += struct.pack(
            "<BBBBHHII", edge, edge, 0, 0, 1, 32, len(blob), offset
        )
        offset += len(blob)
    out.write_bytes(header + directory + b"".join(blob for _, blob in entries))


def verify(ico: Path) -> None:
    data = ico.read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", data, 0)
    if (reserved, kind) != (0, 1) or count != len(SIZES):
        raise SystemExit(f"bad ICO header in {ico}: {reserved} {kind} {count}")
    edges = []
    for i in range(count):
        w, h = struct.unpack_from("<BB", data, 6 + 16 * i)
        edges.append(256 if w == 0 else w)
        if w != h:
            raise SystemExit(f"non square entry {w}x{h} in {ico}")
    expected = sorted(SIZES)
    if sorted(edges) != expected:
        raise SystemExit(f"ICO sizes {sorted(edges)} != {expected}")
    icon = QIcon(str(ico))
    found = sorted(s.width() for s in icon.availableSizes())
    if found != expected:
        raise SystemExit(f"QIcon sees sizes {found} != {expected}")
    pixmap = icon.pixmap(32, 32)
    # pixmap() scales by the screen's device pixel ratio; compare the
    # logical size, not the physical one.
    logical = round(pixmap.width() / pixmap.devicePixelRatio())
    if pixmap.isNull() or logical != 32:
        raise SystemExit(f"QIcon could not produce a 32px pixmap from {ico}")
    print(f"verified {ico.name}: sizes {found}, 32px pixmap ok")


def main() -> None:
    QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    entries = []
    for size in SIZES:
        svg = SMALL if size <= SMALL_MAX and SMALL.exists() else MASTER
        entries.append((size, render_png(svg, size)))
    write_ico(entries, OUT)
    verify(OUT)


if __name__ == "__main__":
    main()
