"""Generates crisp multi-resolution assets/icon.png and assets/icon.ico via PyQt6."""

from __future__ import annotations

from pathlib import Path
import struct

from PyQt6.QtCore import QBuffer, QIODevice, QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor,
    QGuiApplication,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def render_icon_image(size: int) -> QImage:
    """Render an anti-aliased dark-slate rounded icon with glowing amber/blue emblem."""
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)

    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    s = float(size)
    margin = s * 0.06
    rect = QRectF(margin, margin, s - 2 * margin, s - 2 * margin)
    radius = s * 0.22

    # Dark slate gradient background
    bg_grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
    bg_grad.setColorAt(0.0, QColor("#181F2E"))
    bg_grad.setColorAt(1.0, QColor("#0E121B"))

    bg_path = QPainterPath()
    bg_path.addRoundedRect(rect, radius, radius)
    painter.fillPath(bg_path, bg_grad)

    # Subtle outer border
    border_pen = QPen(QColor("#33415C"), max(1.0, s * 0.03))
    painter.setPen(border_pen)
    painter.drawPath(bg_path)

    # Stylized "C" / Copilot arc in warm amber gold (vertically centered in [0.24..0.76])
    arc_rect = QRectF(s * 0.24, s * 0.24, s * 0.52, s * 0.52)
    arc_grad = QLinearGradient(arc_rect.topLeft(), arc_rect.bottomRight())
    arc_grad.setColorAt(0.0, QColor("#FBBF24"))
    arc_grad.setColorAt(1.0, QColor("#F59E0B"))
    arc_pen = QPen(arc_grad, max(1.8, s * 0.115), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    painter.setPen(arc_pen)
    # Draw open arc on the right (45 deg to 315 deg)
    painter.drawArc(arc_rect, 45 * 16, 270 * 16)

    # Glowing blue-cyan AI spark dot on the right opening / top accent
    dot_radius = max(1.3, s * 0.085)
    dot_center = QPointF(s * 0.67, s * 0.34)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#38BDF8"))
    painter.drawEllipse(dot_center, dot_radius, dot_radius)

    # Small inner audio bar accent in center-right
    bar_w = max(1.2, s * 0.075)
    bar_h = s * 0.16
    bar_rect = QRectF(s * 0.46, s * 0.44, bar_w, bar_h)
    painter.setBrush(QColor("#FDE68A"))
    painter.drawRoundedRect(bar_rect, bar_w * 0.5, bar_w * 0.5)

    painter.end()
    return img


def _image_to_png_bytes(img: QImage) -> bytes:
    buf = QBuffer()
    if not buf.open(QIODevice.OpenModeFlag.WriteOnly):
        raise RuntimeError("Failed to open in-memory QBuffer for PNG encoding")
    try:
        if not img.save(buf, "PNG"):
            raise RuntimeError("QImage.save(PNG) returned False")
        return bytes(buf.data())
    finally:
        buf.close()


def build_ico_bytes(png_frames: list[tuple[int, bytes]]) -> bytes:
    """Pack multiple PNG-encoded icon frames into a valid Windows .ico binary."""
    count = len(png_frames)
    header = struct.pack("<HHH", 0, 1, count)
    offset = 6 + 16 * count
    entries: list[bytes] = []
    payloads: list[bytes] = []

    for size, png_data in png_frames:
        dim = 0 if size >= 256 else size
        entry = struct.pack(
            "<BBBBHHII",
            dim,
            dim,
            0,  # color count
            0,  # reserved
            1,  # color planes
            32,  # bits per pixel
            len(png_data),
            offset,
        )
        entries.append(entry)
        payloads.append(png_data)
        offset += len(png_data)

    return header + b"".join(entries) + b"".join(payloads)


def render_checkmark_image(size: int = 16) -> QImage:
    """Render a crisp transparent dark-ink checkmark PNG for styled dark checkboxes."""
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)

    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    s = float(size)
    pen = QPen(
        QColor("#0E1117"),
        max(2.0, s * 0.16),
        Qt.PenStyle.SolidLine,
        Qt.PenCapStyle.RoundCap,
        Qt.PenJoinStyle.RoundJoin,
    )
    painter.setPen(pen)
    path = QPainterPath()
    path.moveTo(s * 0.22, s * 0.52)
    path.lineTo(s * 0.43, s * 0.73)
    path.lineTo(s * 0.80, s * 0.28)
    painter.drawPath(path)
    painter.end()
    return img


def generate_icons(assets_dir: Path | None = None) -> tuple[Path, Path]:
    """Generate assets/icon.png (256px), multi-size assets/icon.ico, and assets/check.png."""
    app = QGuiApplication.instance()
    owns_app = False
    if app is None:
        app = QGuiApplication([])
        owns_app = True

    target_dir = assets_dir or (Path(__file__).resolve().parent / "assets")
    target_dir.mkdir(parents=True, exist_ok=True)

    frames: list[tuple[int, bytes]] = []
    png_256 = b""
    for size in ICON_SIZES:
        img = render_icon_image(size)
        png_bytes = _image_to_png_bytes(img)
        frames.append((size, png_bytes))
        if size == 256:
            png_256 = png_bytes

    png_path = target_dir / "icon.png"
    ico_path = target_dir / "icon.ico"
    check_path = target_dir / "check.png"
    png_path.write_bytes(png_256)
    ico_path.write_bytes(build_ico_bytes(frames))
    check_path.write_bytes(_image_to_png_bytes(render_checkmark_image(16)))

    if owns_app:
        app.quit()

    return png_path, ico_path


if __name__ == "__main__":
    p_png, p_ico = generate_icons()
    print(f"Generated {p_png} ({p_png.stat().st_size} bytes) and {p_ico} ({p_ico.stat().st_size} bytes)")
