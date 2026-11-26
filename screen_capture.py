"""Fast PyQt6 screen & region capture utilities (including stealth ScreenSnipOverlay)."""

from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import sys

from PyQt6.QtCore import QBuffer, QIODevice, QPoint, QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget

_WDA_EXCLUDEFROMCAPTURE = 0x00000011


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


def _apply_snip_stealth_affinity(hwnd: int) -> None:
    if sys.platform != "win32" or hwnd <= 0:
        return
    try:
        user32 = ctypes.windll.user32
        set_fn = user32.SetWindowDisplayAffinity
        set_fn.argtypes = [wintypes.HWND, wintypes.DWORD]
        set_fn.restype = wintypes.BOOL
        set_fn(wintypes.HWND(hwnd), wintypes.DWORD(_WDA_EXCLUDEFROMCAPTURE))
    except Exception:
        pass


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


def _encode_jpeg(pixmap: QPixmap, quality: int) -> bytes:
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


def encode_pixmap_to_result(
    pixmap: QPixmap,
    *,
    max_width: int = 1600,
    max_height: int | None = None,
    jpeg_quality: int = 88,
    max_bytes: int = 1_500_000,
    label: str = "Екран",
) -> ScreenCaptureResult | None:
    """Scale (if needed) and byte-bounded JPEG-encode a captured QPixmap."""
    if pixmap is None or pixmap.isNull():
        return None

    limit_long = max(640, int(max_width))
    if max_height is None:
        limit_w = limit_long
        limit_h = limit_long
    else:
        limit_w = limit_long
        limit_h = max(360, int(max_height))

    cur = pixmap
    if cur.width() > limit_w or cur.height() > limit_h:
        cur = cur.scaled(
            limit_w,
            limit_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    base_q = max(40, min(96, int(jpeg_quality)))
    data = _encode_jpeg(cur, base_q)
    if not data:
        return None

    if len(data) > max_bytes:
        for q in (80, 72, 60):
            if q < base_q:
                candidate = _encode_jpeg(cur, q)
                if candidate:
                    data = candidate
                if len(data) <= max_bytes:
                    break

    for _ in range(3):
        if len(data) <= max_bytes:
            break
        nw = max(640, int(cur.width() * 0.85))
        nh = max(360, int(cur.height() * 0.85))
        if nw >= cur.width() and nh >= cur.height():
            break
        cur = cur.scaled(
            nw,
            nh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        candidate = _encode_jpeg(cur, 72)
        if candidate:
            data = candidate

    return ScreenCaptureResult(
        image_bytes=data,
        mime_type="image/jpeg",
        width=cur.width(),
        height=cur.height(),
        monitor_name=label,
    )


def _grab_virtual_desktop_screenshot() -> tuple[QRect, QPixmap] | None:
    """Capture a composite screenshot across all screens in virtual desktop coordinates."""
    if QGuiApplication.instance() is None:
        return None
    screens = list(QGuiApplication.screens())
    if not screens:
        return None

    virtual_rect = QRect()
    for scr in screens:
        virtual_rect = virtual_rect.united(scr.geometry())
    if virtual_rect.width() <= 0 or virtual_rect.height() <= 0:
        return None

    composite = QPixmap(virtual_rect.size())
    composite.fill(QColor(0, 0, 0))
    painter = QPainter(composite)
    try:
        for scr in screens:
            geo = scr.geometry()
            shot = scr.grabWindow(0)
            if shot is None or shot.isNull():
                continue
            target_rect = QRect(
                geo.left() - virtual_rect.left(),
                geo.top() - virtual_rect.top(),
                geo.width(),
                geo.height(),
            )
            painter.drawPixmap(target_rect, shot)
    finally:
        painter.end()
    return virtual_rect, composite


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
    raw_name = (target.name() or f"Display {display_idx + 1}").strip()
    return encode_pixmap_to_result(
        pixmap,
        max_width=max_width,
        max_height=max_height,
        jpeg_quality=jpeg_quality,
        max_bytes=max_bytes,
        label=f"Монітор {display_idx + 1}: {raw_name}",
    )


def capture_screen_region(
    rect: QRect,
    *,
    max_width: int = 1920,
    jpeg_quality: int = 92,
    max_bytes: int = 1_500_000,
) -> ScreenCaptureResult | None:
    """Capture a specific virtual-desktop QRect as a crisp, high-detail JPEG."""
    norm = rect.normalized()
    if norm.width() < 8 or norm.height() < 8:
        return None
    if QGuiApplication.instance() is None:
        return None

    scr = QGuiApplication.screenAt(norm.center()) or QGuiApplication.primaryScreen()
    if scr is not None and scr.geometry().contains(norm):
        geo = scr.geometry()
        lx = norm.left() - geo.left()
        ly = norm.top() - geo.top()
        shot = scr.grabWindow(0, lx, ly, norm.width(), norm.height())
        if shot is not None and not shot.isNull():
            return encode_pixmap_to_result(
                shot,
                max_width=max_width,
                jpeg_quality=jpeg_quality,
                max_bytes=max_bytes,
                label=f"Область {norm.width()}×{norm.height()}",
            )

    grabbed = _grab_virtual_desktop_screenshot()
    if grabbed is None:
        return None
    vrect, composite = grabbed
    local_rect = QRect(
        norm.left() - vrect.left(),
        norm.top() - vrect.top(),
        norm.width(),
        norm.height(),
    ).intersected(QRect(0, 0, composite.width(), composite.height()))
    if local_rect.width() < 8 or local_rect.height() < 8:
        return None
    cropped = composite.copy(local_rect)
    return encode_pixmap_to_result(
        cropped,
        max_width=max_width,
        jpeg_quality=jpeg_quality,
        max_bytes=max_bytes,
        label=f"Область {local_rect.width()}×{local_rect.height()}",
    )


class ScreenSnipOverlay(QWidget):
    """Frameless stealth region-selection overlay (Snip-to-Ask) across all displays."""

    region_captured = pyqtSignal(object)
    cancelled = pyqtSignal()

    def __init__(self, max_width: int = 1920, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._max_width = max(800, int(max_width))
        self._origin: QPoint | None = None
        self._current: QPoint | None = None
        self._done = False

        self._virtual_rect = QRect(0, 0, 1280, 720)
        self._backdrop = QPixmap()
        grabbed = _grab_virtual_desktop_screenshot()
        if grabbed is not None:
            self._virtual_rect, self._backdrop = grabbed

        self.setWindowTitle("AI Copilot — Виділення області")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.setGeometry(self._virtual_rect)

    def _selection_rect(self) -> QRect:
        if self._origin is None or self._current is None:
            return QRect()
        return QRect(self._origin, self._current).normalized()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        hwnd = int(self.winId()) if self.winId() else 0
        _apply_snip_stealth_affinity(hwnd)
        self.raise_()
        self.activateWindow()
        QTimer.singleShot(0, lambda: _apply_snip_stealth_affinity(int(self.winId()) if self.winId() else 0))

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Escape:
            self._cancel()
            event.accept()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton:
            self._cancel()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            self._origin = pos
            self._current = pos
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._origin is not None:
            self._current = event.position().toPoint()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton and self._origin is not None:
            self._current = event.position().toPoint()
            sel = self._selection_rect()
            if sel.width() < 16 or sel.height() < 16:
                self._cancel()
                event.accept()
                return
            result = self._extract_selection(sel)
            if result is None:
                self._cancel()
            else:
                self._done = True
                self.close()
                self.region_captured.emit(result)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _extract_selection(self, local_rect: QRect) -> ScreenCaptureResult | None:
        if not self._backdrop.isNull():
            clipped = local_rect.intersected(QRect(0, 0, self._backdrop.width(), self._backdrop.height()))
            if clipped.width() >= 16 and clipped.height() >= 16:
                cropped = self._backdrop.copy(clipped)
                res = encode_pixmap_to_result(
                    cropped,
                    max_width=self._max_width,
                    jpeg_quality=92,
                    label=f"Область {clipped.width()}×{clipped.height()}",
                )
                if res is not None:
                    return res

        global_rect = QRect(
            self._virtual_rect.left() + local_rect.left(),
            self._virtual_rect.top() + local_rect.top(),
            local_rect.width(),
            local_rect.height(),
        )
        return capture_screen_region(global_rect, max_width=self._max_width)

    def _cancel(self) -> None:
        if self._done:
            return
        self._done = True
        self.close()
        self.cancelled.emit()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        try:
            if not self._backdrop.isNull():
                painter.drawPixmap(self.rect(), self._backdrop)
            painter.fillRect(self.rect(), QColor(8, 12, 20, 128))

            sel = self._selection_rect()
            if sel.width() >= 4 and sel.height() >= 4:
                if not self._backdrop.isNull():
                    painter.drawPixmap(sel, self._backdrop, sel)
                painter.setPen(QPen(QColor("#38BDF8"), 2))
                painter.drawRect(sel)

                dim_text = f"{sel.width()} × {sel.height()} px"
                painter.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
                badge_rect = QRect(sel.left(), max(8, sel.top() - 26), 118, 22)
                painter.fillRect(badge_rect, QColor(15, 23, 42, 220))
                painter.setPen(QColor("#E0F2FE"))
                painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, dim_text)

            hint_w = 500
            hint_rect = QRect(max(12, (self.width() - hint_w) // 2), 18, hint_w, 32)
            painter.fillRect(hint_rect, QColor(15, 23, 42, 225))
            painter.setPen(QPen(QColor("#334155"), 1))
            painter.drawRect(hint_rect)
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
            painter.setPen(QColor("#F8FAFC"))
            painter.drawText(
                hint_rect,
                Qt.AlignmentFlag.AlignCenter,
                "Виділіть область (графік / формула / код) · Esc або ПКМ — скасувати",
            )
        finally:
            painter.end()
