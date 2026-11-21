"""Fast PyQt6 screen capture utilities for live code/IDE vision."""

from __future__ import annotations

import base64
from dataclasses import dataclass

from PyQt6.QtCore import QBuffer, QIODevice, Qt
from PyQt6.QtGui import QGuiApplication


@dataclass(frozen=True)
class ScreenCaptureResult:
    """Captured and encoded screen image ready for multimodal LLM input."""

    image_bytes: bytes
    mime_type: str
    width: int
    height: int
    monitor_name: str = ""

    def to_data_url(self) -> str:
        """Encode image bytes as a base64 data URL."""
        b64 = base64.b64encode(self.image_bytes).decode("ascii")
        return f"data:{self.mime_type};base64,{b64}"


def list_screens() -> list[tuple[int, str]]:
    """Return available displays as (index, human_readable_label)."""
    if QGuiApplication.instance() is None:
        return []
    screens = list(QGuiApplication.screens())
    out: list[tuple[int, str]] = []
    for idx, scr in enumerate(screens):
        geom = scr.geometry()
        dpr = float(scr.devicePixelRatio() or 1.0)
        phys_w = int(round(geom.width() * dpr))
        phys_h = int(round(geom.height() * dpr))
        raw_name = (scr.name() or f"Display {idx + 1}").strip()
        label = f"Монітор {idx + 1}: {raw_name} ({phys_w}×{phys_h})"
        out.append((idx, label))
    return out


def _encode_jpeg(pixmap, quality: int) -> bytes:
    buf = QBuffer()
    if not buf.open(QIODevice.OpenModeFlag.WriteOnly):
        return b""
    try:
        ok = pixmap.save(buf, "JPEG", quality)
        if not ok:
            return b""
        return bytes(buf.data())
    finally:
        buf.close()


def capture_screen(
    screen_index: int = -1,
    max_width: int = 1600,
    max_height: int | None = None,
    jpeg_quality: int = 88,
    max_bytes: int = 1_500_000,
) -> ScreenCaptureResult | None:
    """Capture the selected monitor (or primary screen) as a crisp, byte-bounded JPEG image."""
    if QGuiApplication.instance() is None:
        return None
    screens = list(QGuiApplication.screens())
    if not screens:
        return None

    if 0 <= screen_index < len(screens):
        target = screens[screen_index]
        display_idx = screen_index
    else:
        target = QGuiApplication.primaryScreen() or screens[0]
        try:
            display_idx = screens.index(target)
        except ValueError:
            display_idx = 0

    if target is None:
        return None

    pixmap = target.grabWindow(0)
    if pixmap is None or pixmap.isNull():
        return None

    limit_long = max(640, int(max_width))
    if max_height is None:
        limit_w = limit_long
        limit_h = limit_long
    else:
        limit_w = limit_long
        limit_h = max(360, int(max_height))

    if pixmap.width() > limit_w or pixmap.height() > limit_h:
        pixmap = pixmap.scaled(
            limit_w,
            limit_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    base_q = max(40, min(96, int(jpeg_quality)))
    data = _encode_jpeg(pixmap, base_q)
    if not data:
        return None

    if len(data) > max_bytes:
        for q in (80, 72, 60):
            if q < base_q:
                candidate = _encode_jpeg(pixmap, q)
                if candidate:
                    data = candidate
                if len(data) <= max_bytes:
                    break

    cur_pixmap = pixmap
    for _ in range(3):
        if len(data) <= max_bytes:
            break
        nw = max(640, int(cur_pixmap.width() * 0.85))
        nh = max(360, int(cur_pixmap.height() * 0.85))
        if nw >= cur_pixmap.width() and nh >= cur_pixmap.height():
            break
        cur_pixmap = cur_pixmap.scaled(
            nw,
            nh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        candidate = _encode_jpeg(cur_pixmap, 72)
        if candidate:
            pixmap = cur_pixmap
            data = candidate

    raw_name = (target.name() or f"Display {display_idx + 1}").strip()
    return ScreenCaptureResult(
        image_bytes=data,
        mime_type="image/jpeg",
        width=pixmap.width(),
        height=pixmap.height(),
        monitor_name=f"Монітор {display_idx + 1}: {raw_name}",
    )
