"""PyQt6 translucent stealth overlay window with Compact and Expanded modes."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import re
import sys
from typing import Iterable

from PyQt6.QtCore import QEvent, QPoint, QRect, QSize, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QDesktopServices,
    QGuiApplication,
    QIcon,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizeGrip,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from audio_capture import AudioDeviceInfo
from context_loader import build_file_dialog_filter

WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011

ALLOWED_LINK_SCHEMES = {"http", "https"}
_FENCE_OPEN_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$")

DARK_MARKDOWN_CSS = """
body {
    color: #F1F5F9;
    line-height: 1.45;
}
code {
    background-color: #1E293B;
    color: #FDE68A;
    padding: 1px 4px;
    border-radius: 4px;
    font-family: "Consolas", "Cascadia Code", monospace;
}
pre {
    background-color: #0D1119;
    color: #E2E8F0;
    border: 1px solid #263145;
    border-radius: 6px;
    padding: 8px 10px;
    font-family: "Consolas", "Cascadia Code", monospace;
    white-space: pre-wrap;
}
a {
    color: #60A5FA;
    text-decoration: none;
}
blockquote {
    border-left: 3px solid #3B82F6;
    margin: 6px 0;
    padding: 2px 10px;
    color: #CBD5E1;
    background-color: #131B29;
}
table {
    border-collapse: collapse;
    margin: 6px 0;
}
th, td {
    border: 1px solid #2D3748;
    padding: 4px 8px;
}
th {
    background-color: #1A2333;
    color: #F8FAFC;
}
hr {
    background-color: #2D3748;
    height: 1px;
    border: none;
}
"""

DARK_SCROLLBAR_QSS = """
QScrollBar:vertical {
    background: #10141D;
    width: 8px;
    margin: 2px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #2D384D;
    min-height: 24px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #425270;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}
QScrollBar:horizontal {
    background: #10141D;
    height: 8px;
    margin: 2px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: #2D384D;
    min-width: 24px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal:hover {
    background: #425270;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
}
"""


def balance_markdown_fences(markdown_text: str) -> str:
    """Close any trailing unclosed fenced code block so streamed Markdown renders cleanly."""
    if not markdown_text:
        return ""
    fence_char = ""
    fence_len = 0
    for line in markdown_text.splitlines():
        m = _FENCE_OPEN_RE.match(line)
        if not m:
            continue
        marker = m.group(1)
        rest = m.group(2).strip()
        if not fence_char:
            if marker[0] == "`" and "`" in rest:
                continue
            fence_char = marker[0]
            fence_len = len(marker)
        elif marker[0] == fence_char and len(marker) >= fence_len and not rest:
            fence_char = ""
            fence_len = 0
    if not fence_char:
        return markdown_text
    suffix = "" if markdown_text.endswith("\n") else "\n"
    return f"{markdown_text}{suffix}{fence_char * fence_len}"


def open_safe_external_link(url: QUrl) -> None:
    """Open clicked Markdown links only when they use a safe HTTP(S) scheme."""
    if url.isValid() and url.scheme().lower() in ALLOWED_LINK_SCHEMES:
        QDesktopServices.openUrl(url)


def configure_markdown_browser(browser: QTextBrowser) -> None:
    """Apply dark Markdown styling and safe external-link handling to a QTextBrowser."""
    browser.setOpenLinks(False)
    browser.setOpenExternalLinks(False)
    browser.document().setDefaultStyleSheet(DARK_MARKDOWN_CSS)
    browser.anchorClicked.connect(open_safe_external_link)


def set_browser_markdown_sticky(
    browser: QTextBrowser,
    markdown_text: str,
    *,
    force_scroll_top: bool = False,
) -> None:
    """Render Markdown while preserving scroll position unless already pinned at bottom."""
    sb = browser.verticalScrollBar()
    was_at_bottom = sb is None or sb.maximum() <= 0 or sb.value() >= sb.maximum() - 24
    prev_val = sb.value() if sb is not None else 0
    browser.setMarkdown(balance_markdown_fences(markdown_text))
    if sb is None:
        return
    if force_scroll_top:
        sb.setValue(0)
    elif was_at_bottom:
        sb.setValue(sb.maximum())
    else:
        sb.setValue(min(prev_val, sb.maximum()))


class ElidedLabel(QLabel):
    """QLabel that elides long text cleanly without pushing parent layout widths."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        hint = super().minimumSizeHint()
        return QSize(min(hint.width(), 40), hint.height())

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        rect = self.contentsRect()
        elided = self.fontMetrics().elidedText(
            self.text(), Qt.TextElideMode.ElideRight, max(1, rect.width())
        )
        flags = int(self.alignment().value) | int(Qt.TextFlag.TextSingleLine.value)
        painter.drawText(rect, flags, elided)

DARK_OVERLAY_QSS = """
QWidget#RootContainer {
    background-color: #11151E;
    border: 1px solid #2C3548;
    border-radius: 10px;
    color: #E8EEF7;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}
QWidget#TitleBar {
    background-color: #171D28;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    border-bottom: 1px solid #262E3E;
}
QLabel#StatusDot {
    min-width: 9px;
    max-width: 9px;
    min-height: 9px;
    max-height: 9px;
    border-radius: 4px;
    background-color: #64748B;
}
QLabel#StatusDot[state="listening"] {
    background-color: #22C55E;
}
QLabel#StatusDot[state="generating"] {
    background-color: #F59E0B;
}
QLabel#StatusDot[state="error"] {
    background-color: #EF4444;
}
QLabel#TitleLabel {
    color: #F2F6FC;
    font-size: 13px;
    font-weight: 600;
}
QLabel#QuoteLabel {
    color: #93A4BE;
    font-size: 12px;
    padding: 2px 4px 4px 4px;
    border-bottom: 1px solid #1F2634;
}
QLabel#SectionTitle {
    color: #93A4BE;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}
QLabel#ContextBadge {
    color: #9FB0C9;
    background-color: #1A202C;
    border: 1px solid #2A3242;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 12px;
}
QLabel#ContextBadge[hasContext="true"] {
    color: #9DE8C4;
    border-color: #2B6B4F;
    background-color: #152922;
}
QLabel#StatusLabel {
    color: #8A99B2;
    font-size: 11px;
}
QLabel#SourceBadge {
    color: #93C5FD;
    background-color: #142238;
    border: 1px solid #263E66;
    border-radius: 5px;
    padding: 1px 6px;
    font-size: 11px;
}
QPushButton {
    background-color: #1E2534;
    color: #E3EAF5;
    border: 1px solid #2F394D;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #283246;
    border-color: #41506B;
}
QPushButton:pressed {
    background-color: #171D28;
}
QPushButton:disabled {
    background-color: #151A24;
    border-color: #222938;
    color: #55627A;
}
QPushButton#HeaderIconBtn {
    background-color: transparent;
    border: 1px solid transparent;
    color: #A6B4CA;
    padding: 2px 3px;
    min-width: 18px;
    font-size: 12px;
    font-weight: 600;
}
QPushButton#HeaderIconBtn:hover {
    background-color: #252E40;
    border-color: #36435A;
    color: #F8FAFC;
}
QPushButton#HeaderIconBtn[generating="true"] {
    background-color: #7F1D1D;
    border-color: #EF4444;
    color: #FECACA;
}
QPushButton#NavIconBtn {
    background-color: #1A202B;
    border: 1px solid #2A3346;
    color: #CBD5E1;
    padding: 1px 5px;
    min-width: 18px;
    font-size: 11px;
    font-weight: 700;
}
QPushButton#NavIconBtn:hover {
    background-color: #263042;
    color: #F8FAFC;
}
QPushButton#NavIconBtn:disabled {
    background-color: #131822;
    border-color: #202736;
    color: #475569;
}
QPushButton#StealthButton {
    padding: 2px 4px;
    font-size: 11px;
    font-weight: 600;
}
QPushButton#StealthButton[stealth="on"],
QPushButton#StealthButton[stealth="on"]:hover,
QPushButton#StealthButton[stealth="on"]:pressed {
    background-color: #15352A;
    border-color: #2D8A66;
    color: #7BF0BE;
}
QPushButton#StealthButton[stealth="off"],
QPushButton#StealthButton[stealth="off"]:hover,
QPushButton#StealthButton[stealth="off"]:pressed {
    background-color: #451A1A;
    border-color: #DC2626;
    color: #FCA5A5;
}
QPushButton#WebSearchButton {
    padding: 2px 4px;
    font-size: 11px;
    font-weight: 600;
}
QPushButton#WebSearchButton:checked {
    background-color: #132C44;
    border-color: #2563EB;
    color: #93C5FD;
}
QPushButton#ScreenVisionButton {
    padding: 2px 4px;
    font-size: 11px;
    font-weight: 600;
}
QPushButton#ScreenVisionButton:checked {
    background-color: #261C45;
    border-color: #7C3AED;
    color: #C4B5FD;
}
QPushButton#CaptureButton[capturing="false"] {
    background-color: #18352B;
    border-color: #2A7D5B;
    color: #A4F5D1;
    font-weight: 600;
}
QPushButton#CaptureButton[capturing="false"]:hover {
    background-color: #1E4437;
}
QPushButton#CaptureButton[capturing="true"] {
    background-color: #441E25;
    border-color: #993949;
    color: #FFB4BE;
    font-weight: 600;
}
QPushButton#CaptureButton[capturing="true"]:hover {
    background-color: #56242D;
}
QPushButton#AnswerButton {
    background-color: #D97706;
    border-color: #F59E0B;
    color: #11151E;
    font-weight: 700;
    padding: 7px 14px;
}
QPushButton#AnswerButton:hover {
    background-color: #F59E0B;
}
QPushButton#AnswerButton[generating="true"] {
    background-color: #DC2626;
    border-color: #EF4444;
    color: #FFFFFF;
}
QPushButton#AnswerButton[generating="true"]:hover {
    background-color: #EF4444;
}
QPushButton#AnswerButton:disabled {
    background-color: #1E293B;
    border-color: #334155;
    color: #64748B;
}
QPushButton#IconButton {
    padding: 3px 8px;
    min-width: 26px;
}
QPushButton#CloseButton {
    background-color: transparent;
    border: none;
    color: #94A3B8;
    padding: 2px 6px;
    font-weight: 600;
}
QPushButton#CloseButton:hover {
    background-color: #882233;
    color: #FFFFFF;
}
QComboBox {
    background-color: #1A202B;
    border: 1px solid #2D3648;
    border-radius: 6px;
    padding: 4px 8px;
    color: #E6ECF5;
    min-height: 20px;
}
QComboBox:disabled {
    background-color: #131821;
    border-color: #202736;
    color: #64748B;
}
QComboBox QAbstractItemView {
    background-color: #1A202B;
    color: #E6ECF5;
    selection-background-color: #2563EB;
    border: 1px solid #2D3648;
}
QLineEdit {
    background-color: #171C26;
    border: 1px solid #2C3547;
    border-radius: 6px;
    padding: 7px 10px;
    color: #F1F5F9;
    font-size: 13px;
}
QLineEdit:focus {
    border-color: #F59E0B;
}
QLineEdit:disabled {
    background-color: #12161E;
    color: #64748B;
}
QTextEdit, QTextBrowser {
    background-color: #151A24;
    border: 1px solid #252E3E;
    border-radius: 8px;
    padding: 8px;
    color: #F1F5F9;
    selection-background-color: #2563EB;
}
QTextEdit#TranscriptView {
    font-size: 13px;
    line-height: 1.4;
}
QTextBrowser#AnswerView {
    line-height: 1.45;
}
QProgressBar#LevelBar {
    background-color: #191F2A;
    border: 1px solid #283142;
    border-radius: 4px;
}
QProgressBar#LevelBar::chunk {
    background-color: #22C55E;
    border-radius: 3px;
}
QSplitter::handle {
    background-color: #252E3E;
    width: 3px;
    margin: 2px 2px;
    border-radius: 1px;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #2C3547;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #F59E0B;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}
""" + DARK_SCROLLBAR_QSS


def get_window_display_affinity(hwnd: int) -> int | None:
    """Query current Windows DWM display affinity for a top-level HWND."""
    if sys.platform != "win32" or hwnd <= 0:
        return None
    try:
        user32 = ctypes.windll.user32
        get_fn = getattr(user32, "GetWindowDisplayAffinity", None)
        if get_fn is None:
            return None
        get_fn.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        get_fn.restype = wintypes.BOOL
        actual = wintypes.DWORD(0)
        if not bool(get_fn(wintypes.HWND(hwnd), ctypes.byref(actual))):
            return None
        return int(actual.value)
    except Exception:
        return None


def apply_window_display_affinity(hwnd: int, exclude_from_capture: bool) -> bool:
    """Toggle Windows WDA_EXCLUDEFROMCAPTURE on a top-level HWND and verify via DWM."""
    if sys.platform != "win32" or hwnd <= 0:
        return False

    try:
        user32 = ctypes.windll.user32
        set_fn = user32.SetWindowDisplayAffinity
        set_fn.argtypes = [wintypes.HWND, wintypes.DWORD]
        set_fn.restype = wintypes.BOOL
        affinity = WDA_EXCLUDEFROMCAPTURE if exclude_from_capture else WDA_NONE
        if not bool(set_fn(wintypes.HWND(hwnd), wintypes.DWORD(affinity))):
            return False
        get_fn = getattr(user32, "GetWindowDisplayAffinity", None)
        if get_fn is not None:
            get_fn.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
            get_fn.restype = wintypes.BOOL
            actual = wintypes.DWORD(0)
            if bool(get_fn(wintypes.HWND(hwnd), ctypes.byref(actual))):
                return int(actual.value) == affinity
        return True
    except Exception:
        return False


def _disable_button_focus(btn: QPushButton) -> None:
    """Prevent QPushButton from stealing keyboard focus or triggering on Space/Enter."""
    btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    btn.setAutoDefault(False)
    btn.setDefault(False)


class CopilotOverlayWindow(QWidget):
    """Translucent always-on-top stealth overlay with Compact Answer Card and Expanded Split modes."""

    start_capture_requested = pyqtSignal(int)
    stop_capture_requested = pyqtSignal()
    refresh_devices_requested = pyqtSignal()
    load_context_requested = pyqtSignal(list)
    clear_context_requested = pyqtSignal()
    clear_all_requested = pyqtSignal()
    answer_requested = pyqtSignal(str, str)
    cancel_answer_requested = pyqtSignal()
    deep_answer_requested = pyqtSignal()
    qa_nav_requested = pyqtSignal(int)
    device_selected = pyqtSignal(int)
    stealth_toggled = pyqtSignal(bool)
    opacity_changed = pyqtSignal(int)
    web_search_toggled = pyqtSignal(bool)
    screen_vision_toggled = pyqtSignal(bool)
    open_dashboard_requested = pyqtSignal()
    expanded_changed = pyqtSignal(bool)
    visibility_changed = pyqtSignal(bool)
    closing = pyqtSignal()

    _RESIZE_MARGIN = 6

    def __init__(
        self,
        hotkey_answer_label: str = "Ctrl+Space",
        hotkey_toggle_label: str = "Ctrl+Shift+H",
        stealth_enabled: bool = True,
        opacity_percent: int = 92,
        font_scale: float = 1.0,
        web_search_enabled: bool = False,
        screen_vision_enabled: bool = False,
        app_icon: QIcon | None = None,
        parent: QWidget | None = None,
        hotkey_screen_label: str = "Ctrl+Alt+S",
        hotkey_web_label: str = "Ctrl+Alt+W",
        hotkey_capture_label: str = "Ctrl+Alt+R",
    ) -> None:
        super().__init__(parent)
        self._hotkey_answer_label = hotkey_answer_label
        self._hotkey_toggle_label = hotkey_toggle_label
        self._hotkey_screen_label = hotkey_screen_label
        self._hotkey_web_label = hotkey_web_label
        self._hotkey_capture_label = hotkey_capture_label

        self._stealth_enabled = bool(stealth_enabled)
        self._stealth_applied = False
        self._web_search_enabled = bool(web_search_enabled)
        self._screen_vision_enabled = bool(screen_vision_enabled)
        self._is_capturing = False
        self._is_generating = False
        self._expanded = False
        self._font_scale = max(0.85, min(1.40, float(font_scale)))
        self._drag_offset: QPoint | None = None
        self._resize_edges = ""
        self._resize_start_global: QPoint | None = None
        self._resize_start_geo: QRect | None = None

        self._compact_size = QSize(480, 345)
        self._expanded_size = QSize(880, 540)

        self._answer_markdown = ""
        self._answer_dirty = False
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(40)
        self._render_timer.timeout.connect(self._flush_answer_render)

        self._copy_reset_timer = QTimer(self)
        self._copy_reset_timer.setSingleShot(True)
        self._copy_reset_timer.setInterval(1200)
        self._copy_reset_timer.timeout.connect(self._reset_copy_button_icon)

        self._stealth_guard_timer = QTimer(self)
        self._stealth_guard_timer.setInterval(2000)
        self._stealth_guard_timer.timeout.connect(self._verify_stealth_affinity)

        self.setWindowTitle("AI Copilot — Оверлей")
        if app_icon is not None and not app_icon.isNull():
            self.setWindowIcon(app_icon)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)

        self._root: QWidget | None = None
        self._title_bar: QWidget | None = None
        self._body: QWidget | None = None

        self._build_ui()
        self._setup_shortcuts()
        self.setStyleSheet(DARK_OVERLAY_QSS)
        self.set_font_scale(self._font_scale)
        self.set_opacity_percent(opacity_percent)
        self.set_stealth_mode(self._stealth_enabled)
        self.set_web_search_enabled(self._web_search_enabled)
        self.set_screen_vision_enabled(self._screen_vision_enabled)
        self.set_qa_history_position(0, 0)
        self.set_expanded(False)

    @property
    def stealth_enabled(self) -> bool:
        return self._stealth_enabled

    @property
    def stealth_applied(self) -> bool:
        return self._stealth_applied

    @property
    def web_search_enabled(self) -> bool:
        return self._web_search_enabled

    @property
    def screen_vision_enabled(self) -> bool:
        return self._screen_vision_enabled

    @property
    def is_expanded(self) -> bool:
        return self._expanded

    @property
    def is_generating(self) -> bool:
        return self._is_generating

    def _setup_shortcuts(self) -> None:
        esc_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        esc_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        esc_shortcut.activated.connect(self._on_escape_shortcut)

        prev_shortcut = QShortcut(QKeySequence("Alt+Left"), self)
        prev_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        prev_shortcut.activated.connect(lambda: self.qa_nav_requested.emit(-1))

        next_shortcut = QShortcut(QKeySequence("Alt+Right"), self)
        next_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        next_shortcut.activated.connect(lambda: self.qa_nav_requested.emit(1))

    def _on_escape_shortcut(self) -> None:
        if self._is_generating:
            self.cancel_answer_requested.emit()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self._root = QWidget(self)
        self._root.setObjectName("RootContainer")
        self._root.setMouseTracking(True)
        outer_layout.addWidget(self._root)

        main_layout = QVBoxLayout(self._root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Custom draggable compact header bar
        self._title_bar = QWidget(self._root)
        self._title_bar.setObjectName("TitleBar")
        self._title_bar.setFixedHeight(36)
        self._title_bar.setMouseTracking(True)
        self._title_layout = QHBoxLayout(self._title_bar)
        self._title_layout.setContentsMargins(8, 4, 6, 4)
        self._title_layout.setSpacing(4)

        self._status_dot = QLabel(self._title_bar)
        self._status_dot.setObjectName("StatusDot")
        self._status_dot.setProperty("state", "idle")
        self._title_layout.addWidget(self._status_dot)

        self._title_label = ElidedLabel("Очікування", self._title_bar)
        self._title_label.setObjectName("TitleLabel")
        self._title_label.setMinimumWidth(42)
        self._title_layout.addWidget(self._title_label, 1)

        self._stealth_btn = QPushButton("◈ Stealth", self._title_bar)
        self._stealth_btn.setObjectName("StealthButton")
        self._stealth_btn.setCheckable(False)
        self._stealth_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._stealth_btn.setProperty("stealth", "on" if self._stealth_enabled else "off")
        self._stealth_btn.setToolTip(
            "Stealth-режим завжди активний (невидимий на Zoom/Discord/OBS). Вимкнення — лише в Налаштуваннях → Оверлей."
        )
        self._title_layout.addWidget(self._stealth_btn)

        self._web_search_btn = QPushButton("🌐 Веб", self._title_bar)
        self._web_search_btn.setObjectName("WebSearchButton")
        self._web_search_btn.setCheckable(True)
        self._web_search_btn.setChecked(self._web_search_enabled)
        self._web_search_btn.setToolTip(
            "Веб-пошук та AI-агент: дозволити пошук актуальної інформації в інтернеті"
        )
        self._web_search_btn.toggled.connect(self._on_web_search_btn_toggled)
        self._title_layout.addWidget(self._web_search_btn)

        self._screen_vision_btn = QPushButton("👁 Екр.", self._title_bar)
        self._screen_vision_btn.setObjectName("ScreenVisionButton")
        self._screen_vision_btn.setCheckable(True)
        self._screen_vision_btn.setChecked(self._screen_vision_enabled)
        self._screen_vision_btn.setToolTip(
            "Зір екрана: прикріплювати знімок екрана (код/слайди), щоб ШІ бачив код"
        )
        self._screen_vision_btn.toggled.connect(self._on_screen_vision_btn_toggled)
        self._title_layout.addWidget(self._screen_vision_btn)

        self._title_layout.addStretch(1)

        self._quick_answer_btn = QPushButton("↵", self._title_bar)
        self._quick_answer_btn.setObjectName("HeaderIconBtn")
        self._quick_answer_btn.setToolTip(f"Відповісти зараз ({self._hotkey_answer_label})")
        self._quick_answer_btn.clicked.connect(self._on_quick_answer_clicked)
        self._title_layout.addWidget(self._quick_answer_btn)

        self._quick_clear_btn = QPushButton("⟳", self._title_bar)
        self._quick_clear_btn.setObjectName("HeaderIconBtn")
        self._quick_clear_btn.setToolTip("Очистити транскрибацію та відповідь")
        self._quick_clear_btn.clicked.connect(self._on_clear_clicked)
        self._title_layout.addWidget(self._quick_clear_btn)

        self._copy_answer_btn = QPushButton("📋", self._title_bar)
        self._copy_answer_btn.setObjectName("HeaderIconBtn")
        self._copy_answer_btn.setToolTip("Копіювати відповідь у буфер обміну")
        self._copy_answer_btn.clicked.connect(self._on_copy_answer_clicked)
        self._title_layout.addWidget(self._copy_answer_btn)

        self._open_dashboard_btn = QPushButton("⚙", self._title_bar)
        self._open_dashboard_btn.setObjectName("HeaderIconBtn")
        self._open_dashboard_btn.setToolTip("Відкрити панель керування та налаштування")
        self._open_dashboard_btn.clicked.connect(self.open_dashboard_requested.emit)
        self._title_layout.addWidget(self._open_dashboard_btn)

        self._expand_btn = QPushButton("⤢", self._title_bar)
        self._expand_btn.setObjectName("HeaderIconBtn")
        self._expand_btn.setToolTip("Розгорнути / згорнути оверлей")
        self._expand_btn.clicked.connect(lambda: self.set_expanded(not self._expanded))
        self._title_layout.addWidget(self._expand_btn)

        self._hide_btn = QPushButton("×", self._title_bar)
        self._hide_btn.setObjectName("CloseButton")
        self._hide_btn.setToolTip(f"Сховати оверлей ({self._hotkey_toggle_label})")
        self._hide_btn.clicked.connect(self.hide)
        self._title_layout.addWidget(self._hide_btn)

        main_layout.addWidget(self._title_bar)

        # Body container
        self._body = QWidget(self._root)
        self._body.setMouseTracking(True)
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(10, 8, 10, 6)
        body_layout.setSpacing(8)
        main_layout.addWidget(self._body, 1)

        # 2. Expanded-only toolbar
        self._toolbar_widget = QWidget(self._body)
        toolbar = QHBoxLayout(self._toolbar_widget)
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(6)

        self.device_combo = QComboBox(self._toolbar_widget)
        self.device_combo.setMinimumWidth(190)
        self.device_combo.setToolTip("Пристрій перехоплення системного звуку (WASAPI Loopback)")
        self.device_combo.currentIndexChanged.connect(self._on_device_combo_changed)
        toolbar.addWidget(self.device_combo, 2)

        self._refresh_devices_btn = QPushButton("↻", self._toolbar_widget)
        self._refresh_devices_btn.setObjectName("IconButton")
        self._refresh_devices_btn.setToolTip("Оновити список аудіопристроїв")
        self._refresh_devices_btn.clicked.connect(self.refresh_devices_requested.emit)
        toolbar.addWidget(self._refresh_devices_btn)

        self._capture_btn = QPushButton("▶ Старт", self._toolbar_widget)
        self._capture_btn.setObjectName("CaptureButton")
        self._capture_btn.setProperty("capturing", "false")
        self._capture_btn.setToolTip(
            f"Запуск / зупинка прослуховування звуку ({self._hotkey_capture_label})"
        )
        self._capture_btn.clicked.connect(self._on_capture_clicked)
        toolbar.addWidget(self._capture_btn)

        self.level_bar = QProgressBar(self._toolbar_widget)
        self.level_bar.setObjectName("LevelBar")
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedWidth(48)
        self.level_bar.setFixedHeight(14)
        self.level_bar.setToolTip("Рівень вхідного системного сигналу")
        toolbar.addWidget(self.level_bar)

        self._load_context_btn = QPushButton("Контекст", self._toolbar_widget)
        self._load_context_btn.clicked.connect(self._on_load_context_clicked)
        toolbar.addWidget(self._load_context_btn)

        self._context_badge = ElidedLabel("Не завантажено", self._toolbar_widget)
        self._context_badge.setObjectName("ContextBadge")
        self._context_badge.setProperty("hasContext", "false")
        toolbar.addWidget(self._context_badge, 1)

        self._clear_context_btn = QPushButton("×", self._toolbar_widget)
        self._clear_context_btn.setObjectName("IconButton")
        self._clear_context_btn.setToolTip("Очистити завантажений контекст")
        self._clear_context_btn.setVisible(False)
        self._clear_context_btn.clicked.connect(self.clear_context_requested.emit)
        toolbar.addWidget(self._clear_context_btn)

        opacity_caption = QLabel("Проз.:", self._toolbar_widget)
        opacity_caption.setObjectName("StatusLabel")
        toolbar.addWidget(opacity_caption)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal, self._toolbar_widget)
        self.opacity_slider.setRange(30, 100)
        self.opacity_slider.setValue(92)
        self.opacity_slider.setFixedWidth(74)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        toolbar.addWidget(self.opacity_slider)

        self._opacity_value_label = QLabel("92%", self._toolbar_widget)
        self._opacity_value_label.setObjectName("StatusLabel")
        self._opacity_value_label.setFixedWidth(32)
        toolbar.addWidget(self._opacity_value_label)

        body_layout.addWidget(self._toolbar_widget)

        # Compact-only latest spoken Ukrainian quote bar
        self._quote_label = QLabel(self._body)
        self._quote_label.setObjectName("QuoteLabel")
        self._quote_label.setWordWrap(True)
        self._quote_label.setMaximumHeight(46)
        self._quote_label.setVisible(False)
        body_layout.addWidget(self._quote_label)

        # 3. Main split view: Transcript (left, expanded only) & AI Answer (right)
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self._body)

        self._left_pane = QWidget(self.splitter)
        left_layout = QVBoxLayout(self._left_pane)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(4)

        left_header = QLabel(
            "Транскрибація (виділіть фрагмент або тисніть Відповісти)",
            self._left_pane,
        )
        left_header.setObjectName("SectionTitle")
        left_layout.addWidget(left_header)

        self.transcript_edit = QTextEdit(self._left_pane)
        self.transcript_edit.setObjectName("TranscriptView")
        self.transcript_edit.setReadOnly(True)
        self.transcript_edit.setPlaceholderText(
            "Натисніть «Старт» для прослуховування системного звуку Zoom..."
        )
        left_layout.addWidget(self.transcript_edit, 1)

        self._right_pane = QWidget(self.splitter)
        right_layout = QVBoxLayout(self._right_pane)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(4)

        self._right_header = QLabel("Відповідь ШІ (тезисно)", self._right_pane)
        self._right_header.setObjectName("SectionTitle")
        right_layout.addWidget(self._right_header)

        self.answer_browser = QTextBrowser(self._right_pane)
        self.answer_browser.setObjectName("AnswerView")
        configure_markdown_browser(self.answer_browser)
        self.answer_browser.setPlaceholderText(
            f"Натисніть {self._hotkey_answer_label} або ↵ для миттєвої тезисної відповіді..."
        )
        right_layout.addWidget(self.answer_browser, 1)

        self.splitter.setSizes([350, 500])
        body_layout.addWidget(self.splitter, 1)

        # 4. Manual question input + Answer / Stop button
        self._prompt_row_widget = QWidget(self._body)
        prompt_row = QHBoxLayout(self._prompt_row_widget)
        prompt_row.setContentsMargins(0, 0, 0, 0)
        prompt_row.setSpacing(8)

        self.manual_input = QLineEdit(self._prompt_row_widget)
        self.manual_input.setPlaceholderText(
            "Додаткове питання або уточнення (Enter для надсилання)..."
        )
        self.manual_input.returnPressed.connect(self._emit_answer_request)
        prompt_row.addWidget(self.manual_input, 1)

        self.answer_btn = QPushButton(
            f"Відповісти ({self._hotkey_answer_label})", self._prompt_row_widget
        )
        self.answer_btn.setObjectName("AnswerButton")
        self.answer_btn.setProperty("generating", "false")
        self.answer_btn.clicked.connect(self._on_answer_btn_clicked)
        prompt_row.addWidget(self.answer_btn)

        body_layout.addWidget(self._prompt_row_widget)

        # 5. Subtle footer status bar + source badges + Q&A history nav + resize grip
        status_row = QHBoxLayout()
        status_row.setContentsMargins(2, 0, 0, 0)
        status_row.setSpacing(6)

        self._provenance_label = ElidedLabel("Готово", self._body)
        self._provenance_label.setObjectName("StatusLabel")
        self._provenance_label.setToolTip("Стан відповіді ШІ")
        status_row.addWidget(self._provenance_label, 1)

        self._sources_badge = ElidedLabel("", self._body)
        self._sources_badge.setObjectName("SourceBadge")
        self._sources_badge.setVisible(False)
        self._sources_badge.setMaximumWidth(170)
        status_row.addWidget(self._sources_badge)

        self._status_sep = QFrame(self._body)
        self._status_sep.setFrameShape(QFrame.Shape.VLine)
        self._status_sep.setStyleSheet("color: #2A3242;")
        status_row.addWidget(self._status_sep)

        self._whisper_status_label = ElidedLabel("Whisper: ініціалізація...", self._body)
        self._whisper_status_label.setObjectName("StatusLabel")
        status_row.addWidget(self._whisper_status_label, 1)

        self._qa_nav_widget = QWidget(self._body)
        qa_nav_l = QHBoxLayout(self._qa_nav_widget)
        qa_nav_l.setContentsMargins(0, 0, 0, 0)
        qa_nav_l.setSpacing(2)

        self._qa_prev_btn = QPushButton("‹", self._qa_nav_widget)
        self._qa_prev_btn.setObjectName("NavIconBtn")
        self._qa_prev_btn.setToolTip("Попередня відповідь (Alt+Left)")
        self._qa_prev_btn.clicked.connect(lambda: self.qa_nav_requested.emit(-1))
        qa_nav_l.addWidget(self._qa_prev_btn)

        self._qa_counter_lbl = QLabel("0/0", self._qa_nav_widget)
        self._qa_counter_lbl.setObjectName("StatusLabel")
        self._qa_counter_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._qa_counter_lbl.setMinimumWidth(26)
        qa_nav_l.addWidget(self._qa_counter_lbl)

        self._qa_next_btn = QPushButton("›", self._qa_nav_widget)
        self._qa_next_btn.setObjectName("NavIconBtn")
        self._qa_next_btn.setToolTip("Наступна відповідь (Alt+Right)")
        self._qa_next_btn.clicked.connect(lambda: self.qa_nav_requested.emit(1))
        qa_nav_l.addWidget(self._qa_next_btn)

        status_row.addWidget(self._qa_nav_widget)

        self._deep_btn = QPushButton("Детальніше →", self._body)
        self._deep_btn.setObjectName("HeaderIconBtn")
        self._deep_btn.setStyleSheet("color: #FBBF24; font-size: 11px; padding: 2px 6px;")
        self._deep_btn.setToolTip("Згенерувати розгорнуту відповідь з поясненнями/кодом")
        self._deep_btn.clicked.connect(self.deep_answer_requested.emit)
        status_row.addWidget(self._deep_btn)

        self._hotkeys_label = QLabel(
            f"{self._hotkey_answer_label} — відповідь | {self._hotkey_toggle_label} — сховати",
            self._body,
        )
        self._hotkeys_label.setObjectName("StatusLabel")
        status_row.addWidget(self._hotkeys_label)

        size_grip = QSizeGrip(self._root)
        size_grip.setFixedSize(16, 16)
        status_row.addWidget(size_grip)

        body_layout.addLayout(status_row)

        for target in (self._root, self._title_bar, self._body):
            if target is not None:
                target.installEventFilter(self)
        for btn in self.findChildren(QPushButton):
            _disable_button_focus(btn)

    # --- Compact / Expanded layout ---

    def set_expanded(self, expanded: bool) -> None:
        """Switch between minimal Compact Answer Card and full Expanded Split View."""
        target = bool(expanded)
        if self.isVisible():
            if self._expanded:
                self._expanded_size = self.size()
            else:
                self._compact_size = self.size()

        self._expanded = target
        self._toolbar_widget.setVisible(target)
        self._left_pane.setVisible(target)
        self._right_header.setVisible(target)
        self._prompt_row_widget.setVisible(True)
        self._status_sep.setVisible(target)
        self._whisper_status_label.setVisible(target)
        self._hotkeys_label.setVisible(target)
        self._quote_label.setVisible((not target) and bool(self._quote_label.text().strip()))

        right_layout = self._right_pane.layout()
        if right_layout is not None:
            right_layout.setContentsMargins(4 if target else 0, 0, 0, 0)

        if target:
            self._title_layout.setContentsMargins(10, 4, 8, 4)
            self._title_layout.setSpacing(5)
            self._expand_btn.setText("▭")
            self._expand_btn.setToolTip("Згорнути в компактний режим")
            self._stealth_btn.setText("Stealth: ON" if self._stealth_enabled else "⚠ Stealth: OFF")
            self.manual_input.setPlaceholderText(
                "Додаткове питання або уточнення (Enter для надсилання)..."
            )
            self.setMinimumSize(580, 360)
            self.resize(self._expanded_size)
            self.splitter.setSizes([350, 500])
        else:
            self._title_layout.setContentsMargins(7, 4, 5, 4)
            self._title_layout.setSpacing(3)
            self._expand_btn.setText("⤢")
            self._expand_btn.setToolTip("Розгорнути повний інтерфейс")
            self._stealth_btn.setText("◈ Stealth" if self._stealth_enabled else "⚠ Викл.")
            self.manual_input.setPlaceholderText("Введіть питання (Enter)...")
            self.setMinimumSize(420, 260)
            self.resize(self._compact_size)

        self._apply_generating_ui(self._is_generating)
        self._refresh_web_btn_label()
        self._refresh_screen_btn_label()
        self.clamp_to_available_screen()
        self.set_stealth_mode(self._stealth_enabled)
        if self.isVisible():
            QTimer.singleShot(0, self._reapply_stealth_if_visible)
        self.expanded_changed.emit(self._expanded)

    def set_hotkey_labels(
        self,
        answer_label: str,
        toggle_label: str,
        screen_label: str | None = None,
        web_label: str | None = None,
        capture_label: str | None = None,
    ) -> None:
        self._hotkey_answer_label = answer_label
        self._hotkey_toggle_label = toggle_label
        if screen_label is not None:
            self._hotkey_screen_label = screen_label
        if web_label is not None:
            self._hotkey_web_label = web_label
        if capture_label is not None:
            self._hotkey_capture_label = capture_label
        self._hide_btn.setToolTip(f"Сховати оверлей ({toggle_label})")
        self._capture_btn.setToolTip(
            f"Запуск / зупинка прослуховування звуку ({self._hotkey_capture_label})"
        )
        self._hotkeys_label.setText(
            f"{answer_label} — відповідь | {toggle_label} — сховати | {self._hotkey_screen_label} — екран"
        )
        self.answer_browser.setPlaceholderText(
            f"Натисніть {answer_label} або ↵ для миттєвої тезисної відповіді..."
        )
        self.set_web_search_enabled(self._web_search_enabled)
        self.set_screen_vision_enabled(self._screen_vision_enabled)
        self._apply_generating_ui(self._is_generating)

    def set_font_scale(self, scale: float) -> None:
        """Adjust overlay readability font size (0.85 - 1.40) across all text views."""
        self._font_scale = max(0.85, min(1.40, float(scale)))
        ans_px = int(round(14 * self._font_scale))
        tr_px = int(round(13 * self._font_scale))
        quote_px = int(round(12 * self._font_scale))
        self.answer_browser.setStyleSheet(f"font-size: {ans_px}px; line-height: 1.45;")
        self.transcript_edit.setStyleSheet(f"font-size: {tr_px}px; line-height: 1.40;")
        self._quote_label.setStyleSheet(f"font-size: {quote_px}px;")

    def _apply_opacity(self, value: int) -> None:
        clamped = max(30, min(100, int(value)))
        self.setWindowOpacity(max(0.30, min(1.0, clamped / 100.0)))
        self._opacity_value_label.setText(f"{clamped}%")

    def set_opacity_percent(self, percent: int) -> None:
        """Set window opacity in percentage (30..100) without re-emitting user signal."""
        clamped = max(30, min(100, int(percent)))
        if self.opacity_slider.value() != clamped:
            self.opacity_slider.blockSignals(True)
            self.opacity_slider.setValue(clamped)
            self.opacity_slider.blockSignals(False)
        self._apply_opacity(clamped)

    def _on_opacity_changed(self, value: int) -> None:
        self._apply_opacity(value)
        self.opacity_changed.emit(max(30, min(100, int(value))))

    def clamp_to_available_screen(self) -> None:
        """Keep the overlay bounded and visible inside the nearest available monitor area."""
        geo = self.frameGeometry()
        screen = QGuiApplication.screenAt(geo.center()) or QGuiApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        if avail.width() <= 0 or avail.height() <= 0:
            return

        w = min(self.width(), avail.width())
        h = min(self.height(), avail.height())
        if w != self.width() or h != self.height():
            self.resize(w, h)

        x = max(avail.left(), min(self.x(), avail.right() - w + 1))
        y = max(avail.top(), min(self.y(), avail.bottom() - h + 1))
        if x != self.x() or y != self.y():
            self.move(x, y)

    # --- Window dragging, edge resizing & Stealth lifecycle ---

    def _reapply_stealth_if_visible(self) -> None:
        if self.isVisible():
            self.set_stealth_mode(self._stealth_enabled)

    def _verify_stealth_affinity(self) -> None:
        if not self._stealth_enabled or not self.isVisible():
            return
        hwnd = int(self.winId()) if self.winId() else 0
        if hwnd <= 0:
            return
        actual = get_window_display_affinity(hwnd)
        if actual is None or actual != WDA_EXCLUDEFROMCAPTURE:
            self.set_stealth_mode(True)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self.clamp_to_available_screen()
        self.set_stealth_mode(self._stealth_enabled)
        if self._stealth_enabled:
            self._stealth_guard_timer.start()
        QTimer.singleShot(0, self._reapply_stealth_if_visible)
        self.visibility_changed.emit(True)

    def changeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        super().changeEvent(event)
        watched = {
            QEvent.Type.WindowStateChange,
            QEvent.Type.ActivationChange,
            QEvent.Type.WinIdChange,
        }
        screen_ev = getattr(QEvent.Type, "ScreenChangeInternal", None)
        if screen_ev is not None:
            watched.add(screen_ev)
        if event.type() in watched and self.isVisible():
            QTimer.singleShot(0, self._reapply_stealth_if_visible)

    def hideEvent(self, event) -> None:  # type: ignore[override]
        self._stealth_guard_timer.stop()
        super().hideEvent(event)
        self.visibility_changed.emit(False)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stealth_guard_timer.stop()
        self._render_timer.stop()
        self._copy_reset_timer.stop()
        self.closing.emit()
        super().closeEvent(event)

    def _hit_test_edges(self, local_pos: QPoint) -> str:
        m = self._RESIZE_MARGIN
        w = self.width()
        h = self.height()
        if w <= m * 2 or h <= m * 2:
            return ""
        left = local_pos.x() <= m
        right = local_pos.x() >= w - m
        top = local_pos.y() <= m
        bottom = local_pos.y() >= h - m
        edges = ""
        if top:
            edges += "T"
        elif bottom:
            edges += "B"
        if left:
            edges += "L"
        elif right:
            edges += "R"
        return edges

    @staticmethod
    def _cursor_for_edges(edges: str) -> Qt.CursorShape | None:
        if edges in ("TL", "BR"):
            return Qt.CursorShape.SizeFDiagCursor
        if edges in ("TR", "BL"):
            return Qt.CursorShape.SizeBDiagCursor
        if edges in ("L", "R"):
            return Qt.CursorShape.SizeHorCursor
        if edges in ("T", "B"):
            return Qt.CursorShape.SizeVerCursor
        return None

    def _apply_edge_resize(self, global_pos: QPoint) -> None:
        if not self._resize_edges or self._resize_start_global is None or self._resize_start_geo is None:
            return
        delta = global_pos - self._resize_start_global
        geo = QRect(self._resize_start_geo)
        min_w = max(self.minimumWidth(), 320)
        min_h = max(self.minimumHeight(), 220)

        if "R" in self._resize_edges:
            geo.setRight(max(geo.left() + min_w - 1, self._resize_start_geo.right() + delta.x()))
        if "B" in self._resize_edges:
            geo.setBottom(max(geo.top() + min_h - 1, self._resize_start_geo.bottom() + delta.y()))
        if "L" in self._resize_edges:
            new_left = min(self._resize_start_geo.right() - min_w + 1, self._resize_start_geo.left() + delta.x())
            geo.setLeft(new_left)
        if "T" in self._resize_edges:
            new_top = min(self._resize_start_geo.bottom() - min_h + 1, self._resize_start_geo.top() + delta.y())
            geo.setTop(new_top)

        self.setGeometry(geo)

    def eventFilter(self, watched, event: QEvent) -> bool:  # type: ignore[override]
        ev_type = event.type()
        targets = (self._root, self._title_bar, self._body)
        if watched is not None and watched in targets:
            if ev_type == QEvent.Type.Leave and not self._resize_edges:
                self.unsetCursor()
            elif ev_type in (
                QEvent.Type.MouseMove,
                QEvent.Type.MouseButtonPress,
                QEvent.Type.MouseButtonRelease,
            ) and isinstance(event, QMouseEvent):
                global_pos = event.globalPosition().toPoint()
                local_pos = self.mapFromGlobal(global_pos)
                if ev_type == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                    edges = self._hit_test_edges(local_pos)
                    if edges:
                        self._resize_edges = edges
                        self._resize_start_global = global_pos
                        self._resize_start_geo = self.geometry()
                        event.accept()
                        return True
                elif ev_type == QEvent.Type.MouseMove:
                    if self._resize_edges and (event.buttons() & Qt.MouseButton.LeftButton):
                        self._apply_edge_resize(global_pos)
                        event.accept()
                        return True
                    edges = self._hit_test_edges(local_pos)
                    cursor = self._cursor_for_edges(edges)
                    if cursor is not None:
                        self.setCursor(cursor)
                    elif not self._resize_edges:
                        self.unsetCursor()
                elif ev_type == QEvent.Type.MouseButtonRelease and self._resize_edges:
                    self._resize_edges = ""
                    self._resize_start_global = None
                    self._resize_start_geo = None
                    self.clamp_to_available_screen()
                    self.unsetCursor()
                    event.accept()
                    return True
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            edges = self._hit_test_edges(pos)
            if edges:
                self._resize_edges = edges
                self._resize_start_global = event.globalPosition().toPoint()
                self._resize_start_geo = self.geometry()
                event.accept()
                return
            if self._title_bar.geometry().contains(pos):
                child = self.childAt(pos)
                if not isinstance(child, (QPushButton, QSlider)):
                    self._drag_offset = (
                        event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                    )
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._resize_edges and (event.buttons() & Qt.MouseButton.LeftButton):
            self._apply_edge_resize(event.globalPosition().toPoint())
            event.accept()
            return
        if self._drag_offset is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        was_active = self._drag_offset is not None or bool(self._resize_edges)
        self._drag_offset = None
        self._resize_edges = ""
        self._resize_start_global = None
        self._resize_start_geo = None
        self.unsetCursor()
        if was_active:
            self.clamp_to_available_screen()
        super().mouseReleaseEvent(event)

    # --- Public state & UI callbacks ---

    def set_stealth_mode(self, enabled: bool) -> bool:
        """Apply or remove Windows WDA_EXCLUDEFROMCAPTURE on this overlay window."""
        self._stealth_enabled = bool(enabled)
        self._stealth_btn.setProperty("stealth", "on" if self._stealth_enabled else "off")
        self._stealth_btn.style().unpolish(self._stealth_btn)
        self._stealth_btn.style().polish(self._stealth_btn)

        if self._expanded:
            self._stealth_btn.setText("Stealth: ON" if self._stealth_enabled else "⚠ Stealth: OFF")
        else:
            self._stealth_btn.setText("◈ Stealth" if self._stealth_enabled else "⚠ Викл.")

        if self._stealth_enabled:
            self._stealth_btn.setToolTip(
                "Stealth-режим завжди активний (невидимий на Zoom/Discord/OBS). Вимкнення — лише в Налаштуваннях → Оверлей."
            )
        else:
            self._stealth_btn.setToolTip(
                "УВАГА: Stealth-режим вимкнено в Налаштуваннях! Вікно видиме при демонстрації екрана."
            )

        hwnd = int(self.winId()) if self.winId() else 0
        ok = apply_window_display_affinity(hwnd, self._stealth_enabled)
        self._stealth_applied = ok
        if self._stealth_enabled and self.isVisible():
            if not self._stealth_guard_timer.isActive():
                self._stealth_guard_timer.start()
        else:
            self._stealth_guard_timer.stop()
        return ok

    def _refresh_web_btn_label(self) -> None:
        if self._expanded and self._web_search_enabled:
            self._web_search_btn.setText("🌐 Веб: ON")
        else:
            self._web_search_btn.setText("🌐 Веб")

    def _on_web_search_btn_toggled(self, checked: bool) -> None:
        self._web_search_enabled = bool(checked)
        self._refresh_web_btn_label()
        self.web_search_toggled.emit(self._web_search_enabled)

    def set_web_search_enabled(self, enabled: bool) -> None:
        """Synchronize the overlay Web Search toggle button without re-emitting signals."""
        self._web_search_enabled = bool(enabled)
        if self._web_search_btn.isChecked() != self._web_search_enabled:
            self._web_search_btn.blockSignals(True)
            self._web_search_btn.setChecked(self._web_search_enabled)
            self._web_search_btn.blockSignals(False)
        self._refresh_web_btn_label()
        state_desc = "увімкнено" if self._web_search_enabled else "вимкнено"
        self._web_search_btn.setToolTip(
            f"Веб-пошук та AI-агент ({state_desc}, {self._hotkey_web_label}): дозволити пошук в інтернеті"
        )

    def _refresh_screen_btn_label(self) -> None:
        if self._expanded and self._screen_vision_enabled:
            self._screen_vision_btn.setText("👁 Екран: ON")
        elif self._expanded:
            self._screen_vision_btn.setText("👁 Екран")
        else:
            self._screen_vision_btn.setText("👁 Екр.")

    def _on_screen_vision_btn_toggled(self, checked: bool) -> None:
        self._screen_vision_enabled = bool(checked)
        self._refresh_screen_btn_label()
        self.screen_vision_toggled.emit(self._screen_vision_enabled)

    def set_screen_vision_enabled(self, enabled: bool) -> None:
        """Synchronize the overlay Screen Vision toggle button without re-emitting signals."""
        self._screen_vision_enabled = bool(enabled)
        if self._screen_vision_btn.isChecked() != self._screen_vision_enabled:
            self._screen_vision_btn.blockSignals(True)
            self._screen_vision_btn.setChecked(self._screen_vision_enabled)
            self._screen_vision_btn.blockSignals(False)
        self._refresh_screen_btn_label()
        state_desc = "увімкнено" if self._screen_vision_enabled else "вимкнено"
        self._screen_vision_btn.setToolTip(
            f"Зір екрана ({state_desc}, {self._hotkey_screen_label}): агент бачить код і слайди на вашому екрані"
        )

    def toggle_visibility(self) -> None:
        """Show or hide the overlay window while preserving Stealth affinity."""
        if self.isVisible() and not self.isMinimized():
            self.hide()
            return

        self.showNormal()
        self.clamp_to_available_screen()
        self.raise_()
        self.activateWindow()
        self.set_stealth_mode(self._stealth_enabled)
        QTimer.singleShot(0, self._reapply_stealth_if_visible)

    def set_status_state(self, state: str, label: str) -> None:
        """Update header indicator dot ('idle'|'listening'|'generating'|'error') and text."""
        self._title_label.setText(label)
        self._status_dot.setProperty("state", state)
        self._status_dot.style().unpolish(self._status_dot)
        self._status_dot.style().polish(self._status_dot)

    def set_answer_provenance(self, text: str) -> None:
        clean = text.strip() or "Готово"
        self._provenance_label.setText(clean)
        self._provenance_label.setToolTip(clean)

    def set_answer_sources(self, sources: list[str]) -> None:
        """Display active document/web sources in a compact elided badge separate from audio status."""
        if not sources:
            self._sources_badge.setText("")
            self._sources_badge.setToolTip("")
            self._sources_badge.setVisible(False)
            return
        preview = ", ".join(sources[:2])
        if len(sources) > 2:
            preview += f" (+{len(sources) - 2})"
        self._sources_badge.setText(preview)
        self._sources_badge.setToolTip("Використані джерела:\n" + "\n".join(sources))
        self._sources_badge.setVisible(True)

    def set_qa_history_position(self, current: int, total: int) -> None:
        """Update Q&A history pagination controls ('‹ n/m ›')."""
        total = max(0, int(total))
        current = max(0, min(int(current), total))
        self._qa_counter_lbl.setText(f"{current}/{total}")
        self._qa_prev_btn.setEnabled(current > 1)
        self._qa_next_btn.setEnabled(0 < current < total)
        self._qa_nav_widget.setVisible(total > 1)

    def set_latest_quote(self, text: str) -> None:
        cleaned = text.strip()
        if cleaned:
            self._quote_label.setText(f"❝ {cleaned}")
            if not self._expanded:
                self._quote_label.setVisible(True)
        else:
            self._quote_label.setText("")
            self._quote_label.setVisible(False)

    def _reset_copy_button_icon(self) -> None:
        self._copy_answer_btn.setText("📋")

    def _on_copy_answer_clicked(self) -> None:
        text = self._answer_markdown.strip() or self.answer_browser.toPlainText().strip()
        if not text:
            return
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
            self._copy_answer_btn.setText("✓")
            self._copy_reset_timer.start()

    def _on_device_combo_changed(self, _idx: int) -> None:
        data = self.device_combo.currentData()
        if isinstance(data, int) and data >= 0:
            self.device_selected.emit(data)

    def set_selected_device_index(self, device_index: int) -> None:
        idx = self.device_combo.findData(device_index)
        if idx >= 0 and idx != self.device_combo.currentIndex():
            self.device_combo.blockSignals(True)
            self.device_combo.setCurrentIndex(idx)
            self.device_combo.blockSignals(False)

    def set_devices(self, devices: Iterable[AudioDeviceInfo]) -> None:
        """Populate the loopback audio device selector."""
        current_data = self.device_combo.currentData()
        self.device_combo.blockSignals(True)
        self.device_combo.clear()

        device_list = list(devices)
        if not device_list:
            self.device_combo.addItem("WASAPI Loopback не знайдено", -1)
            self._capture_btn.setEnabled(False)
            self.device_combo.blockSignals(False)
            return

        self._capture_btn.setEnabled(True)
        selected_idx = 0
        for i, dev in enumerate(device_list):
            prefix = "★ " if dev.is_default else ""
            label = f"{prefix}{dev.name} ({dev.sample_rate} Hz)"
            self.device_combo.addItem(label, dev.index)
            if current_data is not None and dev.index == current_data:
                selected_idx = i

        self.device_combo.setCurrentIndex(selected_idx)
        self.device_combo.blockSignals(False)

    def _on_capture_clicked(self) -> None:
        if self._is_capturing:
            self.stop_capture_requested.emit()
        else:
            dev_idx = self.device_combo.currentData()
            self.start_capture_requested.emit(int(dev_idx) if dev_idx is not None else -1)

    def set_capturing(self, active: bool) -> None:
        """Update start/stop toggle button, status dot, and device dropdown lock state."""
        self._is_capturing = bool(active)
        self.device_combo.setEnabled(not self._is_capturing)
        self._refresh_devices_btn.setEnabled(not self._is_capturing)
        self._capture_btn.setText("■ Стоп" if self._is_capturing else "▶ Старт")
        self._capture_btn.setProperty("capturing", "true" if self._is_capturing else "false")
        self._capture_btn.style().unpolish(self._capture_btn)
        self._capture_btn.style().polish(self._capture_btn)
        if not self._is_generating:
            if self._is_capturing:
                self.set_status_state("listening", "Слухаю…")
            else:
                self.set_status_state("idle", "Пауза")
        if not self._is_capturing:
            self.level_bar.setValue(0)

    def set_audio_level(self, rms: float) -> None:
        """Update the audio level meter from an RMS value."""
        scaled = int(min(100, max(0, rms * 1400)))
        self.level_bar.setValue(scaled)

    def _on_load_context_clicked(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Оберіть файли контексту (звіт, код, конспект)",
            "",
            build_file_dialog_filter(),
        )
        if paths:
            self.load_context_requested.emit(paths)

    def set_context_summary(self, summary_text: str, has_context: bool) -> None:
        """Update context badge text and clear button visibility."""
        self._context_badge.setText(summary_text)
        self._context_badge.setToolTip(summary_text)
        self._context_badge.setProperty("hasContext", "true" if has_context else "false")
        self._context_badge.style().unpolish(self._context_badge)
        self._context_badge.style().polish(self._context_badge)
        self._clear_context_btn.setVisible(has_context)

    def _on_clear_clicked(self) -> None:
        self.clear_transcript_and_answer()
        self.clear_all_requested.emit()

    def clear_transcript_and_answer(self) -> None:
        """Clear both the live transcript, latest quote, and AI answer panels."""
        self._render_timer.stop()
        self._answer_markdown = ""
        self._answer_dirty = False
        self._apply_generating_ui(False)
        self.transcript_edit.clear()
        self.answer_browser.clear()
        self.set_latest_quote("")
        self.set_answer_sources([])
        self.set_answer_provenance("Готово")

    def append_transcript(self, timestamp_str: str, text: str) -> None:
        """Append a timestamped Ukrainian transcript utterance with sticky autoscroll."""
        cleaned = text.strip()
        if not cleaned:
            return
        self.set_latest_quote(cleaned)
        line = f"[{timestamp_str}] {cleaned}"
        sb = self.transcript_edit.verticalScrollBar()
        was_at_bottom = sb is None or sb.maximum() <= 0 or sb.value() >= sb.maximum() - 24
        self.transcript_edit.append(line)
        if sb is not None and was_at_bottom:
            sb.setValue(sb.maximum())

    def get_selected_transcript(self) -> str:
        """Return currently selected text from the transcript box, if any."""
        cursor = self.transcript_edit.textCursor()
        return cursor.selectedText().replace("\u2029", "\n").strip()

    def _apply_generating_ui(self, generating: bool) -> None:
        self._is_generating = bool(generating)
        self.answer_btn.setProperty("generating", "true" if generating else "false")
        self.answer_btn.style().unpolish(self.answer_btn)
        self.answer_btn.style().polish(self.answer_btn)
        self._quick_answer_btn.setProperty("generating", "true" if generating else "false")
        self._quick_answer_btn.style().unpolish(self._quick_answer_btn)
        self._quick_answer_btn.style().polish(self._quick_answer_btn)

        if generating:
            self.answer_btn.setText("■ Стоп (Esc)" if self._expanded else "■ Стоп")
            self.answer_btn.setToolTip("Зупинити генерацію відповіді (Esc)")
            self._quick_answer_btn.setText("■")
            self._quick_answer_btn.setToolTip("Зупинити генерацію відповіді (Esc)")
        else:
            self.answer_btn.setText(
                f"Відповісти ({self._hotkey_answer_label})" if self._expanded else "Надіслати"
            )
            self.answer_btn.setToolTip(f"Отримати відповідь ШІ ({self._hotkey_answer_label})")
            self._quick_answer_btn.setText("↵")
            self._quick_answer_btn.setToolTip(f"Відповісти зараз ({self._hotkey_answer_label})")
        self._deep_btn.setEnabled(not generating)

    def _on_quick_answer_clicked(self) -> None:
        if self._is_generating:
            self.cancel_answer_requested.emit()
        else:
            self._emit_answer_request()

    def _on_answer_btn_clicked(self) -> None:
        if self._is_generating:
            self.cancel_answer_requested.emit()
        else:
            self._emit_answer_request()

    def _emit_answer_request(self) -> None:
        if self._is_generating:
            return
        manual = self.manual_input.text().strip()
        selected = self.get_selected_transcript()
        self.manual_input.clear()
        self.answer_requested.emit(manual, selected)

    def begin_answer_stream(self) -> None:
        """Prepare AI answer panel for a new streaming response."""
        self._answer_markdown = ""
        self._answer_dirty = False
        self.answer_browser.clear()
        self._apply_generating_ui(True)
        self.set_status_state("generating", "Генерую…")
        self._render_timer.start()

    def append_answer_chunk(self, chunk: str) -> None:
        """Buffer an incoming streamed Markdown chunk for coalesced rendering."""
        if not chunk:
            return
        self._answer_markdown += chunk
        self._answer_dirty = True

    def _flush_answer_render(self) -> None:
        if not self._answer_dirty:
            return
        self._answer_dirty = False
        set_browser_markdown_sticky(self.answer_browser, self._answer_markdown)

    def finish_answer_stream(self) -> None:
        """Flush final Markdown content when LLM streaming finishes."""
        self._render_timer.stop()
        self._answer_dirty = True
        self._flush_answer_render()
        self._apply_generating_ui(False)
        if self._is_capturing:
            self.set_status_state("listening", "Слухаю…")
        else:
            self.set_status_state("idle", "Готово")

    def cancel_answer_stream(self) -> None:
        """Finalize UI cleanly when the user cancels a streaming answer."""
        self._render_timer.stop()
        if self._answer_markdown:
            self._answer_dirty = True
            self._flush_answer_render()
        self._apply_generating_ui(False)
        if self._is_capturing:
            self.set_status_state("listening", "Слухаю…")
        else:
            self.set_status_state("idle", "Зупинено")

    def set_answer_markdown_history(self, markdown_text: str) -> None:
        """Render a stored Q&A answer from session history and scroll to top."""
        self._render_timer.stop()
        self._apply_generating_ui(False)
        self._answer_markdown = markdown_text or ""
        self._answer_dirty = False
        set_browser_markdown_sticky(
            self.answer_browser, self._answer_markdown, force_scroll_top=True
        )

    def show_answer_error(self, message: str) -> None:
        """Render an error notice clearly in the AI response pane."""
        self._render_timer.stop()
        self._apply_generating_ui(False)
        self._answer_markdown = f"**Помилка:**\n\n- {message}"
        self._answer_dirty = False
        set_browser_markdown_sticky(
            self.answer_browser, self._answer_markdown, force_scroll_top=True
        )
        self.set_status_state("error", "Помилка")

    def set_whisper_status(self, text: str) -> None:
        self._whisper_status_label.setText(text)
        self._whisper_status_label.setToolTip(text)

    def set_llm_status(self, text: str) -> None:
        self.set_answer_provenance(text)
