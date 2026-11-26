"""Intuitive PyQt6 Control Panel (Dashboard) inspired by Magerko/copilot."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from PyQt6.QtCore import QEvent, QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from audio_capture import AudioDeviceInfo
from config import (
    AppConfig,
    DEFAULT_SYSTEM_PROMPT,
    PromptProfile,
    get_provider_api_key,
    get_provider_model,
    get_resource_path,
    has_valid_llm_credentials,
)
from context_loader import LoadedDocument, build_file_dialog_filter
from llm_client import ApiKeyCheckResult, get_deprecated_model_replacement
from ui import (
    DARK_SCROLLBAR_QSS,
    WDA_EXCLUDEFROMCAPTURE,
    ElidedLabel,
    apply_window_display_affinity,
    configure_markdown_browser,
    get_ui_font_stack,
    get_window_display_affinity,
    set_browser_markdown_sticky,
)

PROVIDER_CHOICES: list[tuple[str, str]] = [
    ("openai", "OpenAI (GPT-6 Luna / Sol / Astra)"),
    ("groq", "Groq (Qwen 3.8 / GPT-OSS / Llama)"),
    ("cerebras", "Cerebras (Qwen 3.8 / GPT-OSS — надшвидкий)"),
    ("gemini", "Google Gemini (Gemini 3.8 Flash / Gemini 4)"),
    ("ollama", "Ollama (локально на ПК, без ключа)"),
    ("lmstudio", "LM Studio (локально на ПК, без ключа)"),
]

FALLBACK_PROVIDER_CHOICES: list[tuple[str, str]] = [
    ("auto", "Авто (будь-який інший налаштований провайдер)"),
    ("none", "Вимкнено (без перемикання провайдерів)"),
    *PROVIDER_CHOICES,
]

MODEL_PRESETS: dict[str, list[str]] = {
    "openai": ["gpt-6-luna", "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6.2-sol", "gpt-6.2-astra"],
    "groq": [
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "groq/compound-mini",
        "groq/compound",
        "llama-3.1-8b-instant",
    ],
    "cerebras": [
        "qwen-3.8-27b",
        "gpt-oss-120b",
    ],
    "gemini": [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-4-hydrogen",
        "gemini-4-argon",
        "gemini-4-carbon",
        "gemini-4-neon",
    ],
    "ollama": ["qwen2.5:14b", "qwen2.5:7b", "llama3.1:8b", "gemma3:12b"],
    "lmstudio": ["local-model"],
}

BASE_URL_PLACEHOLDERS: dict[str, str] = {
    "openai": "https://api.openai.com/v1 (стандартно)",
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "gemini": "Не використовується для Gemini",
}

STT_PROVIDER_CHOICES: list[tuple[str, str]] = [
    ("local", "Локальний Whisper (faster-whisper на GPU/CPU — безкоштовно)"),
    ("deepgram", "Deepgram Nova-3 (хмарний real-time WebSocket — макс. швидкість + якість)"),
    ("elevenlabs", "ElevenLabs Scribe (хмарний — найвища точність української ~3% WER)"),
    ("groq", "Groq Cloud Whisper (хмарний — працює з вашим Groq API-ключем, 0 VRAM)"),
]

STT_MODEL_PRESETS: dict[str, list[str]] = {
    "deepgram": ["nova-3", "nova-2"],
    "elevenlabs": ["scribe_v3", "scribe_v2", "scribe_v1"],
    "groq": ["whisper-large-v3", "whisper-large-v3-turbo"],
}

STT_PROVIDER_HINTS: dict[str, str] = {
    "local": (
        "Працює локально з нейромережевим Silero VAD (ріже репліки за реальними паузами). "
        "Для кращої української оберіть whisper-large-v3-turbo-uk або large-v3."
    ),
    "deepgram": (
        "Потоковий real-time WebSocket (wss://api.deepgram.com). Миттєво транскрибує під час мови "
        "з пунктуацією і підтримкою термінів словника (nova-3)."
    ),
    "elevenlabs": (
        "Найнижчий рівень помилок для української (~3–5% WER, модель scribe_v3 + keyterms). "
        "Автоматично підхоплює ELEVENLABS_API_KEY із .env, якщо поле ключа порожнє."
    ),
    "groq": (
        "Виконує whisper-large-v3 у хмарі Groq за ~0.3 с без навантаження на GPU. "
        "Ключ STT можна лишити порожнім — підхопиться ваш Groq API-ключ зі вкладки ШІ."
    ),
}

WHISPER_MODEL_CHOICES: list[tuple[str, str]] = [
    (
        "skypro1111/whisper-large-v3-turbo-ukrainian-ukraine-3percent-ct2",
        "whisper-large-v3-turbo-uk (fine-tuned UA — топ локальна для української, ~1.6 ГБ)",
    ),
    ("large-v3-turbo", "large-v3-turbo — швидка й якісна для української (рекомендовано)"),
    ("large-v3", "large-v3 — максимальна точність української (32-шаровий декодер, ~3 ГБ VRAM)"),
    ("medium", "medium — збалансована швидкість і якість (~1.5 ГБ)"),
    ("small", "small — легка, швидка, підходить для слабших GPU/CPU"),
    ("base", "base — мінімальна модель (~150 МБ, нижча якість UA)"),
]

WHISPER_LANG_CHOICES: list[tuple[str, str]] = [
    ("uk", "Українська (примусово uk — найкраща стабільність)"),
    ("en+uk", "Змішана EN + UA (розмовна англійська + українська)"),
    ("auto", "Автовизначення мови (auto)"),
    ("en", "English (en)"),
]

PROFILE_LANG_CHOICES: list[tuple[str, str]] = [
    ("uk", "Українська (uk)"),
    ("en+uk", "Змішана EN + UA (розмовна англійська)"),
    ("en", "English (en)"),
    ("auto", "Автовизначення (auto)"),
    ("", "Не змінювати мову STT"),
]

def _build_dashboard_qss(primary_family: str | None = None) -> str:
    """Build dark Dashboard stylesheet with bundled Inter font stack and dark-safe controls."""
    font_stack = get_ui_font_stack(primary_family)
    check_path = get_resource_path("assets/check.png")
    check_rule = (
        f'\n    image: url("{check_path.as_posix()}");'
        if check_path.is_file()
        else ""
    )
    qss = _DASHBOARD_QSS_TEMPLATE.replace("__UI_FONT_STACK__", font_stack)
    return qss.replace("__CHECK_IMAGE_RULE__", check_rule)


_DASHBOARD_QSS_TEMPLATE = """
QWidget#DashboardRoot {
    background-color: #0E1117;
    color: #E6EDF6;
    font-family: __UI_FONT_STACK__;
    font-size: 13px;
}
QLabel {
    color: #E2E8F0;
    background-color: transparent;
}
QCheckBox {
    color: #E2E8F0;
    spacing: 8px;
    background-color: transparent;
}
QCheckBox:disabled {
    color: #64748B;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #3B4963;
    background-color: #10141D;
}
QCheckBox::indicator:hover {
    border-color: #F59E0B;
}
QCheckBox::indicator:checked {
    background-color: #F59E0B;
    border-color: #FBBF24;__CHECK_IMAGE_RULE__
}
QCheckBox::indicator:disabled {
    background-color: #0D1119;
    border-color: #232B3B;
}
QWidget#Sidebar {
    background-color: #131722;
    border-right: 1px solid #222938;
}
QLabel#BrandTitle {
    color: #F8FAFC;
    font-size: 16px;
    font-weight: 700;
}
QLabel#BrandSub {
    color: #8292AB;
    font-size: 11px;
}
QPushButton#NavBtn {
    text-align: left;
    padding: 10px 14px;
    border: 1px solid transparent;
    border-radius: 8px;
    background-color: transparent;
    color: #A8B5CA;
    font-size: 13px;
    font-weight: 500;
}
QPushButton#NavBtn:hover {
    background-color: #1B2130;
    color: #F1F5F9;
}
QPushButton#NavBtn:checked {
    background-color: #20283B;
    border-color: #323F5B;
    color: #FFFFFF;
    font-weight: 600;
}
QFrame#Card {
    background-color: #141923;
    border: 1px solid #232B3B;
    border-radius: 12px;
}
QFrame#SetupBanner {
    background-color: #2B2010;
    border: 1px solid #8A5A16;
    border-radius: 10px;
}
QLabel#PageHeader {
    font-size: 20px;
    font-weight: 700;
    color: #F8FAFC;
}
QLabel#CardTitle {
    font-size: 15px;
    font-weight: 700;
    color: #F8FAFC;
}
QLabel#MutedText {
    color: #93A2B8;
    font-size: 12px;
}
QLabel#CheckDot {
    min-width: 9px;
    max-width: 9px;
    min-height: 9px;
    max-height: 9px;
    border-radius: 4px;
    background-color: #64748B;
}
QLabel#CheckDot[status="ok"] {
    background-color: #22C55E;
}
QLabel#CheckDot[status="warn"] {
    background-color: #F59E0B;
}
QLabel#CheckDot[status="error"] {
    background-color: #EF4444;
}
QLabel#KbdPill {
    background-color: #1B2230;
    border: 1px solid #323E54;
    border-radius: 5px;
    padding: 3px 8px;
    color: #E2E8F0;
    font-family: "Consolas", "Segoe UI", monospace;
    font-size: 12px;
    font-weight: 600;
}
QPushButton {
    background-color: #1E2534;
    color: #E3EAF5;
    border: 1px solid #303B52;
    border-radius: 8px;
    padding: 7px 14px;
    font-size: 13px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #283246;
    border-color: #455473;
}
QPushButton#PrimaryCTA {
    background-color: #F59E0B;
    border-color: #FBBF24;
    color: #0E1117;
    font-weight: 700;
    padding: 9px 18px;
}
QPushButton#PrimaryCTA:hover {
    background-color: #FBBF24;
}
QPushButton#PrimaryCTA[capturing="true"] {
    background-color: #DC2626;
    border-color: #EF4444;
    color: #FFFFFF;
}
QPushButton#PrimaryCTA[capturing="true"]:hover {
    background-color: #EF4444;
}
QPushButton#DangerBtn {
    background-color: #361920;
    border-color: #7A2B3B;
    color: #FCA5A5;
}
QPushButton#DangerBtn:hover {
    background-color: #4C2029;
}
QPushButton#LiveWebBtn:checked {
    background-color: #132C44;
    border-color: #2563EB;
    color: #93C5FD;
    font-weight: 600;
}
QPushButton#LiveScreenBtn:checked {
    background-color: #261C45;
    border-color: #7C3AED;
    color: #C4B5FD;
    font-weight: 600;
}
QLineEdit, QPlainTextEdit, QTextEdit, QTextBrowser, QListWidget, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #10141D;
    border: 1px solid #283244;
    border-radius: 8px;
    padding: 6px 10px;
    color: #F1F5F9;
}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #F59E0B;
}
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QComboBox QAbstractItemView {
    background-color: #141923;
    color: #F1F5F9;
    selection-background-color: #2563EB;
    border: 1px solid #2D3648;
}
QTabWidget::pane {
    border: 1px solid #232B3B;
    border-radius: 10px;
    background-color: #141923;
    padding: 10px;
}
QTabBar::tab {
    background-color: transparent;
    color: #93A2B8;
    padding: 8px 16px;
    border-bottom: 2px solid transparent;
    font-weight: 600;
}
QTabBar::tab:selected {
    color: #F8FAFC;
    border-bottom: 2px solid #F59E0B;
}
QProgressBar#HomeLevelBar {
    background-color: #0E1117;
    border: 1px solid #263042;
    border-radius: 4px;
}
QProgressBar#HomeLevelBar::chunk {
    background-color: #22C55E;
    border-radius: 3px;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #283244;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #F59E0B;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QScrollArea {
    background: transparent;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background: transparent;
}
QPushButton:disabled, QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    background-color: #0D1119;
    border-color: #1D2534;
    color: #55627A;
}
QPushButton#PrimaryCTA[generating="true"] {
    background-color: #DC2626;
    border-color: #EF4444;
    color: #FFFFFF;
}
QPushButton#PrimaryCTA[generating="true"]:hover {
    background-color: #EF4444;
}
QPushButton#NavIconBtn {
    background-color: #1A202B;
    border: 1px solid #2A3346;
    color: #CBD5E1;
    padding: 2px 7px;
    min-width: 22px;
    font-weight: 700;
}
QPushButton#NavIconBtn:hover {
    background-color: #263042;
    color: #F8FAFC;
}
""" + DARK_SCROLLBAR_QSS


DASHBOARD_QSS = _build_dashboard_qss("Inter")


class ContextDropListWidget(QListWidget):
    """QListWidget accepting file drops for context documents."""

    files_dropped = pyqtSignal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            paths = [
                url.toLocalFile()
                for url in event.mimeData().urls()
                if url.isLocalFile() and url.toLocalFile()
            ]
            if paths:
                self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class DashboardWindow(QWidget):
    """Full control panel with Home checklist, Live session, Documents, and GUI Settings."""

    start_capture_requested = pyqtSignal(int)
    stop_capture_requested = pyqtSignal()
    refresh_devices_requested = pyqtSignal()
    toggle_overlay_requested = pyqtSignal()
    load_context_requested = pyqtSignal(list)
    reload_context_requested = pyqtSignal()
    remove_context_file_requested = pyqtSignal(str)
    clear_context_requested = pyqtSignal()
    clear_all_requested = pyqtSignal()
    answer_requested = pyqtSignal(str, str)
    cancel_answer_requested = pyqtSignal()
    deep_answer_requested = pyqtSignal()
    qa_nav_requested = pyqtSignal(int)
    qa_jump_requested = pyqtSignal(int)
    snip_requested = pyqtSignal()
    device_selected = pyqtSignal(int)
    web_search_toggled = pyqtSignal(bool)
    screen_vision_toggled = pyqtSignal(bool)
    settings_saved = pyqtSignal(object)
    test_llm_requested = pyqtSignal(object)
    fetch_models_requested = pyqtSignal(object)
    test_stt_requested = pyqtSignal(object)
    test_web_search_requested = pyqtSignal(object)
    test_screen_capture_requested = pyqtSignal(int, int)
    profile_selected = pyqtSignal(str)
    profile_create_requested = pyqtSignal(str)
    profile_rename_requested = pyqtSignal(str, str)
    profile_delete_requested = pyqtSignal(str)
    profile_language_changed = pyqtSignal(str, str)
    profile_export_requested = pyqtSignal(str, bool)
    profile_import_requested = pyqtSignal(str)
    run_diagnostics_requested = pyqtSignal()
    quit_requested = pyqtSignal()
    minimized_to_tray = pyqtSignal()

    def __init__(
        self,
        config: AppConfig,
        app_icon: QIcon | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("DashboardRoot")
        self._config = replace(config)
        self._active_provider = str(config.llm_provider or "openai")
        self._last_check_result: ApiKeyCheckResult | None = None
        self._pending_settings_verify = False
        self._draft_keys: dict[str, str] = {}
        self._draft_models: dict[str, str] = {}
        self._is_capturing = False
        self._is_generating = False
        self._allow_close = False
        self._hide_to_tray = False
        self._overlay_visible = False
        self._stealth_applied = False
        self._settings_save_rejected = False
        self._last_answer_markdown = ""
        self._qa_history_entries: list[tuple[str, str, str, list[str]]] = []
        self._qa_history_nav_items: list[tuple[str, str]] = []
        self._qa_history_active_idx = -1

        self._copy_reset_timer = QTimer(self)
        self._copy_reset_timer.setSingleShot(True)
        self._copy_reset_timer.setInterval(1200)
        self._copy_reset_timer.timeout.connect(self._reset_live_copy_btn)

        self._stealth_guard_timer = QTimer(self)
        self._stealth_guard_timer.setInterval(2000)
        self._stealth_guard_timer.timeout.connect(self._verify_dashboard_stealth_affinity)

        self.setWindowTitle("AI Copilot • Панель керування")
        if app_icon is not None and not app_icon.isNull():
            self.setWindowIcon(app_icon)
        self.resize(1000, 670)
        self.setMinimumSize(840, 560)

        self._build_ui(app_icon)
        self._setup_shortcuts()
        self.setStyleSheet(_build_dashboard_qss())
        self._populate_settings_from_config(self._config)
        self.refresh_readiness()

    @property
    def stealth_applied(self) -> bool:
        return self._stealth_applied

    def _setup_shortcuts(self) -> None:
        esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        esc.setContext(Qt.ShortcutContext.WindowShortcut)
        esc.activated.connect(self._on_escape_shortcut)

        prev_sc = QShortcut(QKeySequence("Alt+Left"), self)
        prev_sc.setContext(Qt.ShortcutContext.WindowShortcut)
        prev_sc.activated.connect(lambda: self.qa_nav_requested.emit(-1))

        next_sc = QShortcut(QKeySequence("Alt+Right"), self)
        next_sc.setContext(Qt.ShortcutContext.WindowShortcut)
        next_sc.activated.connect(lambda: self.qa_nav_requested.emit(1))

    def _on_escape_shortcut(self) -> None:
        if self._is_generating and self._stack.currentIndex() == 1:
            self.cancel_answer_requested.emit()

    def set_hide_to_tray(self, enabled: bool) -> None:
        self._hide_to_tray = bool(enabled)

    def force_close(self) -> None:
        self._allow_close = True
        self.close()

    def _auto_save_settings_if_valid(self) -> None:
        if self._allow_close:
            return
        candidate = self._collect_config_from_ui()
        if candidate == self._config:
            return
        if (
            self._validate_hotkey_pair(
                candidate.hotkey_answer,
                candidate.hotkey_toggle,
                candidate.hotkey_screen,
                candidate.hotkey_web,
                candidate.hotkey_capture,
                candidate.hotkey_snip,
            )
            is not None
        ):
            return
        if candidate.llm_provider == self._config.llm_provider or has_valid_llm_credentials(
            candidate
        ):
            self._on_save_settings_clicked()

    def _on_quit_clicked(self) -> None:
        if self._stack.currentIndex() == 3:
            self._auto_save_settings_if_valid()
        self.quit_requested.emit()

    def _apply_dashboard_stealth(self) -> bool:
        hwnd = int(self.winId()) if self.winId() else 0
        ok = apply_window_display_affinity(hwnd, self._config.dashboard_stealth)
        self._stealth_applied = ok
        if self._config.dashboard_stealth and self.isVisible():
            if not self._stealth_guard_timer.isActive():
                self._stealth_guard_timer.start()
        else:
            self._stealth_guard_timer.stop()
        return ok

    def _verify_dashboard_stealth_affinity(self) -> None:
        if not self._config.dashboard_stealth or not self.isVisible():
            return
        hwnd = int(self.winId()) if self.winId() else 0
        if hwnd <= 0:
            return
        actual = get_window_display_affinity(hwnd)
        if actual is None or actual != WDA_EXCLUDEFROMCAPTURE:
            self._apply_dashboard_stealth()

    def _reapply_stealth_if_visible(self) -> None:
        if self.isVisible():
            self._apply_dashboard_stealth()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._apply_dashboard_stealth()
        QTimer.singleShot(0, self._reapply_stealth_if_visible)

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

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if not self._allow_close and self._stack.currentIndex() == 3:
            self._auto_save_settings_if_valid()
        if not self._allow_close and self._hide_to_tray:
            event.ignore()
            self._stealth_guard_timer.stop()
            self.hide()
            self.minimized_to_tray.emit()
            return
        self._stealth_guard_timer.stop()
        self._copy_reset_timer.stop()
        if not self._allow_close:
            self.quit_requested.emit()
        super().closeEvent(event)

    # --- UI construction ---

    def _build_ui(self, app_icon: QIcon | None) -> None:
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Left navigation sidebar
        sidebar = QWidget(self)
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(176)
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(14, 18, 14, 14)
        sb_layout.setSpacing(6)

        brand_row = QHBoxLayout()
        brand_row.setSpacing(10)
        if app_icon is not None and not app_icon.isNull():
            logo_lbl = QLabel(sidebar)
            dpr = max(1.0, float(self.devicePixelRatioF()))
            px = max(30, int(round(30 * dpr)))
            pm = app_icon.pixmap(px, px)
            pm.setDevicePixelRatio(dpr)
            logo_lbl.setFixedSize(30, 30)
            logo_lbl.setPixmap(pm)
            brand_row.addWidget(logo_lbl)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)
        b_title = QLabel("AI Copilot", sidebar)
        b_title.setObjectName("BrandTitle")
        b_sub = QLabel("Zoom • Українська (GPU)", sidebar)
        b_sub.setObjectName("BrandSub")
        brand_text.addWidget(b_title)
        brand_text.addWidget(b_sub)
        brand_row.addLayout(brand_text, 1)
        sb_layout.addLayout(brand_row)
        sb_layout.addSpacing(14)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        nav_items = [
            ("Головна", 0),
            ("Сесія (Live)", 1),
            ("Документи", 2),
            ("Налаштування", 3),
        ]
        self._nav_buttons: list[QPushButton] = []
        for label, idx in nav_items:
            btn = QPushButton(label, sidebar)
            btn.setObjectName("NavBtn")
            btn.setCheckable(True)
            if idx == 0:
                btn.setChecked(True)
            btn.clicked.connect(lambda _checked, i=idx: self._switch_page(i))
            self._nav_group.addButton(btn, idx)
            self._nav_buttons.append(btn)
            sb_layout.addWidget(btn)

        sb_layout.addStretch(1)

        self._sb_overlay_btn = QPushButton("Показати оверлей", sidebar)
        self._sb_overlay_btn.clicked.connect(self.toggle_overlay_requested.emit)
        sb_layout.addWidget(self._sb_overlay_btn)

        self._sb_quit_btn = QPushButton("Вихід", sidebar)
        self._sb_quit_btn.setObjectName("DangerBtn")
        self._sb_quit_btn.clicked.connect(self._on_quit_clicked)
        sb_layout.addWidget(self._sb_quit_btn)

        ver_lbl = QLabel("v2.0 • Stealth + CUDA", sidebar)
        ver_lbl.setObjectName("BrandSub")
        ver_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sb_layout.addWidget(ver_lbl)

        root_layout.addWidget(sidebar)

        # Right pages stack
        self._stack = QStackedWidget(self)
        self._stack.addWidget(self._build_home_page())
        self._stack.addWidget(self._build_live_page())
        self._stack.addWidget(self._build_docs_page())
        self._stack.addWidget(self._build_settings_page())
        root_layout.addWidget(self._stack, 1)

    def _switch_page(self, index: int) -> None:
        if self._stack.currentIndex() == 3 and index != 3:
            self._auto_save_settings_if_valid()
        self._stack.setCurrentIndex(index)
        if 0 <= index < len(self._nav_buttons):
            self._nav_buttons[index].setChecked(True)

    def _make_card(self, parent: QWidget) -> QFrame:
        card = QFrame(parent)
        card.setObjectName("Card")
        return card

    # --- Page 0: Home ---

    def _build_home_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        # First-run setup banner
        self._setup_banner = QFrame(page)
        self._setup_banner.setObjectName("SetupBanner")
        banner_l = QHBoxLayout(self._setup_banner)
        banner_l.setContentsMargins(12, 7, 12, 7)
        banner_txt = QLabel(
            "Потрібно налаштувати LLM: додайте API-ключ (OpenAI / Groq / Gemini) або оберіть локальну Ollama.",
            self._setup_banner,
        )
        banner_txt.setWordWrap(True)
        banner_txt.setStyleSheet("color: #FDE68A; font-weight: 500;")
        banner_l.addWidget(banner_txt, 1)

        banner_btn = QPushButton("Налаштувати LLM →", self._setup_banner)
        banner_btn.setObjectName("PrimaryCTA")
        banner_btn.clicked.connect(lambda: self._open_settings_tab(0))
        banner_l.addWidget(banner_btn)
        layout.addWidget(self._setup_banner)

        # Top split: Hero card + Readiness checklist
        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        hero = self._make_card(page)
        hero_l = QVBoxLayout(hero)
        hero_l.setContentsMargins(12, 12, 12, 12)
        hero_l.setSpacing(8)

        h_title = QLabel("Готові до захисту або зустрічі?", hero)
        h_title.setObjectName("PageHeader")
        h_title.setWordWrap(True)
        hero_l.addWidget(h_title)

        h_desc = QLabel(
            "Програма локально перехоплює голос співрозмовника з Zoom через WASAPI Loopback, "
            "розпізнає українську мову на GPU та показує стислі тези в Stealth-оверлеї.",
            hero,
        )
        h_desc.setObjectName("MutedText")
        h_desc.setWordWrap(True)
        hero_l.addWidget(h_desc)

        dev_row = QHBoxLayout()
        dev_row.setSpacing(6)
        self.home_device_combo = QComboBox(hero)
        self.home_device_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.home_device_combo.setMinimumContentsLength(16)
        self.home_device_combo.currentIndexChanged.connect(self._on_home_device_changed)
        dev_row.addWidget(self.home_device_combo, 1)

        self._home_refresh_btn = QPushButton("↻", hero)
        self._home_refresh_btn.setFixedWidth(36)
        self._home_refresh_btn.setToolTip("Оновити пристрої WASAPI Loopback")
        self._home_refresh_btn.clicked.connect(self.refresh_devices_requested.emit)
        dev_row.addWidget(self._home_refresh_btn)
        hero_l.addLayout(dev_row)

        hero_l.addStretch(1)

        cta_row = QHBoxLayout()
        cta_row.setSpacing(6)
        self._home_start_btn = QPushButton("▶ Почати сесію", hero)
        self._home_start_btn.setObjectName("PrimaryCTA")
        self._home_start_btn.setProperty("capturing", "false")
        self._home_start_btn.setStyleSheet("padding: 8px 12px;")
        self._home_start_btn.clicked.connect(self._on_toggle_capture_clicked)
        cta_row.addWidget(self._home_start_btn)

        self._home_overlay_btn = QPushButton("Показати оверлей", hero)
        self._home_overlay_btn.setStyleSheet("padding: 7px 10px;")
        self._home_overlay_btn.clicked.connect(self.toggle_overlay_requested.emit)
        cta_row.addWidget(self._home_overlay_btn)
        hero_l.addLayout(cta_row)

        top_row.addWidget(hero, 11)

        # Readiness Checklist card
        ready_card = self._make_card(page)
        ready_l = QVBoxLayout(ready_card)
        ready_l.setContentsMargins(14, 10, 14, 10)
        ready_l.setSpacing(5)

        r_title = QLabel("Стан готовності", ready_card)
        r_title.setObjectName("CardTitle")
        ready_l.addWidget(r_title)

        self._dot_audio, self._lbl_audio = self._add_check_row(ready_l, "Звук")
        self.home_level_bar = QProgressBar(ready_card)
        self.home_level_bar.setObjectName("HomeLevelBar")
        self.home_level_bar.setRange(0, 100)
        self.home_level_bar.setValue(0)
        self.home_level_bar.setTextVisible(False)
        self.home_level_bar.setFixedHeight(6)
        ready_l.addWidget(self.home_level_bar)

        self._dot_whisper, self._lbl_whisper = self._add_check_row(ready_l, "Мовлення (UA)")
        self._whisper_progress = QProgressBar(ready_card)
        self._whisper_progress.setObjectName("HomeLevelBar")
        self._whisper_progress.setRange(0, 0)
        self._whisper_progress.setTextVisible(False)
        self._whisper_progress.setFixedHeight(5)
        self._whisper_progress.setVisible(True)
        ready_l.addWidget(self._whisper_progress)

        self._dot_llm, self._lbl_llm = self._add_check_row(ready_l, "Модель відповідей")
        llm_actions_row = QHBoxLayout()
        llm_actions_row.setContentsMargins(90, 0, 0, 0)
        self._home_verify_btn = QPushButton("✓ Перевірити API-ключ", ready_card)
        self._home_verify_btn.setStyleSheet("padding: 3px 10px; font-size: 11px;")
        self._home_verify_btn.clicked.connect(self._on_home_verify_key_clicked)
        llm_actions_row.addWidget(self._home_verify_btn)
        llm_actions_row.addStretch(1)
        ready_l.addLayout(llm_actions_row)

        self._dot_docs, self._lbl_docs = self._add_check_row(ready_l, "Документи")
        self._dot_web, self._lbl_web = self._add_check_row(ready_l, "Веб-пошук")
        self._dot_screen, self._lbl_screen = self._add_check_row(ready_l, "Зір (Екран)")
        self._dot_stealth, self._lbl_stealth = self._add_check_row(ready_l, "Захист екрана")
        ready_l.addStretch(1)

        top_row.addWidget(ready_card, 10)
        layout.addLayout(top_row)

        # Bottom split: Privacy info + Hotkeys
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)

        priv_card = self._make_card(page)
        priv_l = QVBoxLayout(priv_card)
        priv_l.setContentsMargins(14, 9, 14, 9)
        priv_l.setSpacing(4)
        p_title = QLabel("Як це працює і приватність", priv_card)
        p_title.setObjectName("CardTitle")
        priv_l.addWidget(p_title)
        p_body = QLabel(
            "• Розпізнавання української мови відбувається локально (faster-whisper + CUDA).\n"
            "• Системний звук захоплюється напряму без віртуальних кабелів та ботів.\n"
            "• У LLM надсилається лише остання хвилина реплік + ваш контекст при тригері.\n"
            "• Stealth-режим (WDA_EXCLUDEFROMCAPTURE) ховає оверлей від Zoom/Discord/OBS.",
            priv_card,
        )
        p_body.setObjectName("MutedText")
        p_body.setWordWrap(True)
        priv_l.addWidget(p_body)
        priv_l.addStretch(1)
        bottom_row.addWidget(priv_card, 1)

        hk_card = self._make_card(page)
        hk_l = QVBoxLayout(hk_card)
        hk_l.setContentsMargins(14, 6, 14, 6)
        hk_l.setSpacing(2)
        hk_title = QLabel("Гарячі клавіші", hk_card)
        hk_title.setObjectName("CardTitle")
        hk_l.addWidget(hk_title)

        self._hk_answer_pill = self._add_hotkey_row(hk_l, "Тезисна відповідь зараз")
        self._hk_toggle_pill = self._add_hotkey_row(hk_l, "Показати / сховати оверлей")
        self._hk_capture_pill = self._add_hotkey_row(hk_l, "Старт / стоп сесії")
        self._hk_screen_pill = self._add_hotkey_row(hk_l, "Зір екрана (👁)")
        self._hk_web_pill = self._add_hotkey_row(hk_l, "Веб-пошук (🌐)")
        self._hk_snip_pill = self._add_hotkey_row(hk_l, "Область екрана (✂)")

        hk_tip = QLabel(
            "Гарячі клавіші працюють глобально незалежно від української чи англійської розкладки.",
            hk_card,
        )
        hk_tip.setObjectName("MutedText")
        hk_tip.setWordWrap(True)
        hk_l.addWidget(hk_tip)
        hk_l.addStretch(1)
        bottom_row.addWidget(hk_card, 1)

        layout.addLayout(bottom_row, 1)
        return page

    def _add_check_row(self, parent_layout: QVBoxLayout, name: str) -> tuple[QLabel, QLabel]:
        row = QHBoxLayout()
        row.setSpacing(6)
        dot = QLabel()
        dot.setObjectName("CheckDot")
        dot.setProperty("status", "idle")
        row.addWidget(dot)

        name_lbl = QLabel(name)
        name_lbl.setObjectName("MutedText")
        name_lbl.setFixedWidth(106)
        row.addWidget(name_lbl)

        val_lbl = ElidedLabel("—")
        val_lbl.setStyleSheet("color: #E2E8F0; font-size: 12px;")
        row.addWidget(val_lbl, 1)
        parent_layout.addLayout(row)
        return dot, val_lbl

    def _add_hotkey_row(self, parent_layout: QVBoxLayout, desc: str) -> QLabel:
        row = QHBoxLayout()
        desc_lbl = QLabel(desc)
        desc_lbl.setObjectName("MutedText")
        row.addWidget(desc_lbl, 1)
        pill = QLabel("—")
        pill.setObjectName("KbdPill")
        row.addWidget(pill)
        parent_layout.addLayout(row)
        return pill

    # --- Page 1: Live Session ---

    def _build_live_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(8)
        title = QLabel("Жива сесія", page)
        title.setObjectName("PageHeader")
        hdr_row.addWidget(title)
        hdr_row.addStretch(1)

        self._live_overlay_btn = QPushButton("Показати оверлей", page)
        self._live_overlay_btn.clicked.connect(self.toggle_overlay_requested.emit)
        hdr_row.addWidget(self._live_overlay_btn)
        layout.addLayout(hdr_row)

        top_bar = QHBoxLayout()
        top_bar.setSpacing(5)

        self._live_start_btn = QPushButton("▶ Почати сесію", page)
        self._live_start_btn.setObjectName("PrimaryCTA")
        self._live_start_btn.setProperty("capturing", "false")
        self._live_start_btn.setStyleSheet("padding: 6px 10px;")
        self._live_start_btn.clicked.connect(self._on_toggle_capture_clicked)
        top_bar.addWidget(self._live_start_btn)

        self._live_web_btn = QPushButton("🌐 Веб", page)
        self._live_web_btn.setObjectName("LiveWebBtn")
        self._live_web_btn.setCheckable(True)
        self._live_web_btn.setStyleSheet("padding: 5px 8px;")
        self._live_web_btn.setToolTip(
            "Увімкнути / вимкнути пошук в інтернеті та веб-агента під час відповідей"
        )
        self._live_web_btn.toggled.connect(self._on_live_web_toggled)
        top_bar.addWidget(self._live_web_btn)

        self._live_screen_btn = QPushButton("👁 Екран", page)
        self._live_screen_btn.setObjectName("LiveScreenBtn")
        self._live_screen_btn.setCheckable(True)
        self._live_screen_btn.setStyleSheet("padding: 5px 8px;")
        self._live_screen_btn.setToolTip(
            "Зір екрана: прикріплювати знімок екрана (код в IDE / слайди)"
        )
        self._live_screen_btn.toggled.connect(self._on_live_screen_toggled)
        top_bar.addWidget(self._live_screen_btn)

        self._live_snip_btn = QPushButton("✂ Обл.", page)
        self._live_snip_btn.setStyleSheet("padding: 5px 8px;")
        self._live_snip_btn.setToolTip(
            "Виділити область екрана (Ctrl+Alt+A) — графік, формула або фрагмент коду"
        )
        self._live_snip_btn.clicked.connect(self.snip_requested.emit)
        top_bar.addWidget(self._live_snip_btn)

        top_bar.addStretch(1)

        live_export_btn = QPushButton("Експорт (.md)", page)
        live_export_btn.setStyleSheet("padding: 5px 8px;")
        live_export_btn.setToolTip("Зберегти транскрипт та всю сесію ШІ у файл")
        live_export_btn.clicked.connect(self._export_session_markdown)
        top_bar.addWidget(live_export_btn)

        live_clear_btn = QPushButton("Очистити", page)
        live_clear_btn.setStyleSheet("padding: 5px 8px;")
        live_clear_btn.clicked.connect(self.clear_all_requested.emit)
        top_bar.addWidget(live_clear_btn)
        layout.addLayout(top_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal, page)
        left_w = QWidget(splitter)
        left_l = QVBoxLayout(left_w)
        left_l.setContentsMargins(0, 0, 6, 0)
        left_l.addWidget(QLabel("Розпізнана мова (українська)", left_w))
        self.live_transcript_edit = QTextEdit(left_w)
        self.live_transcript_edit.setReadOnly(True)
        self.live_transcript_edit.setPlaceholderText(
            "Натисніть «Почати сесію» — тут з'явиться жива транскрибація звуку з Zoom..."
        )
        left_l.addWidget(self.live_transcript_edit, 1)

        right_w = QWidget(splitter)
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(6, 0, 0, 0)
        ans_hdr = QHBoxLayout()
        ans_hdr.setSpacing(6)
        ans_hdr.addWidget(QLabel("Відповідь ШІ", right_w))

        self._live_qa_nav = QWidget(right_w)
        qa_l = QHBoxLayout(self._live_qa_nav)
        qa_l.setContentsMargins(0, 0, 0, 0)
        qa_l.setSpacing(2)
        self._live_qa_prev_btn = QPushButton("‹", self._live_qa_nav)
        self._live_qa_prev_btn.setObjectName("NavIconBtn")
        self._live_qa_prev_btn.setToolTip("Попередня відповідь (Alt+Left)")
        self._live_qa_prev_btn.clicked.connect(lambda: self.qa_nav_requested.emit(-1))
        qa_l.addWidget(self._live_qa_prev_btn)
        self._live_qa_counter_lbl = QPushButton("0/0", self._live_qa_nav)
        self._live_qa_counter_lbl.setObjectName("NavIconBtn")
        self._live_qa_counter_lbl.setToolTip("Швидкий перехід між картками відповідей")
        self._live_qa_counter_lbl.clicked.connect(self._on_live_qa_counter_clicked)
        qa_l.addWidget(self._live_qa_counter_lbl)
        self._live_qa_next_btn = QPushButton("›", self._live_qa_nav)
        self._live_qa_next_btn.setObjectName("NavIconBtn")
        self._live_qa_next_btn.setToolTip("Наступна відповідь (Alt+Right)")
        self._live_qa_next_btn.clicked.connect(lambda: self.qa_nav_requested.emit(1))
        qa_l.addWidget(self._live_qa_next_btn)
        self._live_qa_nav.setVisible(False)
        ans_hdr.addWidget(self._live_qa_nav)

        self._live_sources_lbl = ElidedLabel("", right_w)
        self._live_sources_lbl.setObjectName("MutedText")
        self._live_sources_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        ans_hdr.addWidget(self._live_sources_lbl, 1)

        self._live_copy_btn = QPushButton("📋 Копіювати", right_w)
        self._live_copy_btn.setStyleSheet("padding: 3px 9px; font-size: 11px;")
        self._live_copy_btn.setToolTip("Копіювати поточну відповідь ШІ")
        self._live_copy_btn.clicked.connect(self._on_live_copy_clicked)
        ans_hdr.addWidget(self._live_copy_btn)
        right_l.addLayout(ans_hdr)

        self.live_answer_browser = QTextBrowser(right_w)
        configure_markdown_browser(self.live_answer_browser)
        self.live_answer_browser.setPlaceholderText(
            "Виділіть питання зліва або натисніть «Відповісти»..."
        )
        right_l.addWidget(self.live_answer_browser, 1)

        splitter.setSizes([420, 440])
        layout.addWidget(splitter, 1)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)
        self.live_manual_input = QLineEdit(page)
        self.live_manual_input.setPlaceholderText("Введіть ручне запитання або уточнення...")
        self.live_manual_input.returnPressed.connect(self._emit_live_answer)
        bottom_row.addWidget(self.live_manual_input, 1)

        self._live_answer_btn = QPushButton("Відповісти", page)
        self._live_answer_btn.setObjectName("PrimaryCTA")
        self._live_answer_btn.setProperty("generating", "false")
        self._live_answer_btn.clicked.connect(self._on_live_answer_btn_clicked)
        bottom_row.addWidget(self._live_answer_btn)

        self._live_deep_btn = QPushButton("Детальніше", page)
        self._live_deep_btn.setToolTip("Розгорнута відповідь з прикладами чи кодом")
        self._live_deep_btn.clicked.connect(self.deep_answer_requested.emit)
        bottom_row.addWidget(self._live_deep_btn)
        layout.addLayout(bottom_row)

        for btn in page.findChildren(QPushButton):
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setAutoDefault(False)
            btn.setDefault(False)
        return page

    # --- Page 2: Documents & Prompt ---

    def _build_docs_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        title = QLabel("Документи та системний промпт", page)
        title.setObjectName("PageHeader")
        layout.addWidget(title)

        prof_card = self._make_card(page)
        prof_l = QVBoxLayout(prof_card)
        prof_l.setContentsMargins(12, 8, 12, 8)
        prof_l.setSpacing(6)

        prof_row = QHBoxLayout()
        prof_row.setSpacing(6)
        prof_lbl = QLabel("Профіль:", prof_card)
        prof_lbl.setObjectName("MutedText")
        prof_row.addWidget(prof_lbl)

        self.profile_combo = QComboBox(prof_card)
        self.profile_combo.setMinimumWidth(160)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_combo_changed)
        prof_row.addWidget(self.profile_combo, 1)

        self._new_profile_btn = QPushButton("+ Новий", prof_card)
        self._new_profile_btn.setToolTip("Створити новий профіль предмета")
        self._new_profile_btn.clicked.connect(self._on_new_profile_clicked)
        prof_row.addWidget(self._new_profile_btn)

        self._rename_profile_btn = QPushButton("✎ Перейменувати", prof_card)
        self._rename_profile_btn.setToolTip("Змінити назву активного профілю")
        self._rename_profile_btn.clicked.connect(self._on_rename_profile_clicked)
        prof_row.addWidget(self._rename_profile_btn)

        self._delete_profile_btn = QPushButton("✕", prof_card)
        self._delete_profile_btn.setObjectName("DangerBtn")
        self._delete_profile_btn.setToolTip("Видалити активний профіль")
        self._delete_profile_btn.setFixedWidth(34)
        self._delete_profile_btn.clicked.connect(self._on_delete_profile_clicked)
        prof_row.addWidget(self._delete_profile_btn)
        prof_l.addLayout(prof_row)

        prof_lang_row = QHBoxLayout()
        prof_lang_row.setSpacing(6)
        prof_lang_lbl = QLabel("Мова STT:", prof_card)
        prof_lang_lbl.setObjectName("MutedText")
        prof_lang_row.addWidget(prof_lang_lbl)

        self.profile_lang_combo = QComboBox(prof_card)
        for code, label in PROFILE_LANG_CHOICES:
            self.profile_lang_combo.addItem(label, code)
        self.profile_lang_combo.currentIndexChanged.connect(self._on_profile_lang_changed)
        prof_lang_row.addWidget(self.profile_lang_combo, 1)

        self._export_profile_btn = QPushButton("⬆ Експорт", prof_card)
        self._export_profile_btn.setToolTip("Експортувати профіль(і) у JSON без секретів і шляхів")
        self._export_profile_btn.clicked.connect(self._on_export_profile_clicked)
        prof_lang_row.addWidget(self._export_profile_btn)

        self._import_profile_btn = QPushButton("⬇ Імпорт", prof_card)
        self._import_profile_btn.setToolTip("Імпортувати профіль(і) з JSON-файлу")
        self._import_profile_btn.clicked.connect(self._on_import_profile_clicked)
        prof_lang_row.addWidget(self._import_profile_btn)

        prof_l.addLayout(prof_lang_row)
        layout.addWidget(prof_card)

        hdr = QHBoxLayout()
        hdr.setSpacing(8)

        add_btn = QPushButton("+ Додати файли", page)
        add_btn.setObjectName("PrimaryCTA")
        add_btn.clicked.connect(self._on_add_files_clicked)
        hdr.addWidget(add_btn)

        add_dir_btn = QPushButton("+ Додати папку", page)
        add_dir_btn.setToolTip("Завантажити весь проєкт чи папку з документами та кодом")
        add_dir_btn.clicked.connect(self._on_add_folder_clicked)
        hdr.addWidget(add_dir_btn)

        reload_btn = QPushButton("↻ Оновити", page)
        reload_btn.setToolTip("Перечитати змінені файли на диску")
        reload_btn.clicked.connect(self.reload_context_requested.emit)
        hdr.addWidget(reload_btn)

        hdr.addStretch(1)

        clear_btn = QPushButton("Очистити все", page)
        clear_btn.clicked.connect(self.clear_context_requested.emit)
        hdr.addWidget(clear_btn)
        layout.addLayout(hdr)

        hint = QLabel(
            "Перетягніть файли або папки сюди (PDF, DOCX, TXT, MD, Python, C#, JS/TS, Java тощо). "
            "Терміни з документів також використовуються для корекції розпізнавання слів.",
            page,
        )
        hint.setObjectName("MutedText")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._docs_status_lbl = QLabel("", page)
        self._docs_status_lbl.setWordWrap(True)
        self._docs_status_lbl.setVisible(False)
        layout.addWidget(self._docs_status_lbl)

        self.docs_list = ContextDropListWidget(page)
        self.docs_list.files_dropped.connect(self.load_context_requested.emit)
        layout.addWidget(self.docs_list, 1)

        vocab_row = QHBoxLayout()
        vocab_row.setSpacing(8)
        vocab_lbl = QLabel("Словник термінів:", page)
        vocab_lbl.setObjectName("MutedText")
        vocab_row.addWidget(vocab_lbl)
        self.vocab_edit = QLineEdit(page)
        self.vocab_edit.setPlaceholderText(
            "Через кому, напр.: PyQt6, WASAPI, FastAPI, PostgreSQL (для точної нормалізації Whisper)"
        )
        vocab_row.addWidget(self.vocab_edit, 1)
        vocab_save_btn = QPushButton("Застосувати словник", page)
        vocab_save_btn.clicked.connect(self._on_save_vocab_clicked)
        vocab_row.addWidget(vocab_save_btn)
        layout.addLayout(vocab_row)

        prompt_card = self._make_card(page)
        prompt_l = QVBoxLayout(prompt_card)
        prompt_l.setContentsMargins(14, 12, 14, 12)
        prompt_l.setSpacing(8)

        p_hdr = QHBoxLayout()
        p_lbl = QLabel("Системний промпт асистента", prompt_card)
        p_lbl.setObjectName("CardTitle")
        p_hdr.addWidget(p_lbl)
        p_hdr.addStretch(1)

        reset_prompt_btn = QPushButton("Скинути", prompt_card)
        reset_prompt_btn.setToolTip("Скинути системний промпт до стандартного")
        reset_prompt_btn.clicked.connect(self._reset_prompt_default)
        p_hdr.addWidget(reset_prompt_btn)

        save_prompt_btn = QPushButton("Зберегти промпт", prompt_card)
        save_prompt_btn.setObjectName("PrimaryCTA")
        save_prompt_btn.clicked.connect(self._on_save_prompt_clicked)
        p_hdr.addWidget(save_prompt_btn)
        prompt_l.addLayout(p_hdr)

        self.prompt_edit = QPlainTextEdit(prompt_card)
        self.prompt_edit.setFixedHeight(84)
        prompt_l.addWidget(self.prompt_edit)

        layout.addWidget(prompt_card)
        return page

    # --- Page 3: Settings ---

    def _build_settings_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        title = QLabel("Налаштування", page)
        title.setObjectName("PageHeader")
        layout.addWidget(title)

        self._settings_tabs = QTabWidget(page)
        self._settings_tabs.addTab(self._build_llm_tab(), "LLM Провайдер")
        self._settings_tabs.addTab(self._build_speech_tab(), "Мовлення та Аудіо")
        self._settings_tabs.addTab(self._build_overlay_tab(), "Оверлей і Гарячі клавіші")
        self._settings_tabs.addTab(self._build_web_tab(), "Веб і Екран (Агент)")
        layout.addWidget(self._settings_tabs, 1)

        footer = QHBoxLayout()
        self._save_status_lbl = QLabel("", page)
        self._save_status_lbl.setObjectName("MutedText")
        footer.addWidget(self._save_status_lbl, 1)

        self._diag_btn = QPushButton("🩺 Діагностика", page)
        self._diag_btn.setToolTip("Перевірити сумісність Windows, VC++, AVX2 та аудіопристроїв")
        self._diag_btn.clicked.connect(self.run_diagnostics_requested.emit)
        footer.addWidget(self._diag_btn)

        save_btn = QPushButton("Зберегти та застосувати", page)
        save_btn.setObjectName("PrimaryCTA")
        save_btn.clicked.connect(self._on_save_settings_clicked)
        footer.addWidget(save_btn)
        layout.addLayout(footer)
        return page

    def _build_llm_tab(self) -> QWidget:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        tab = QWidget(scroll)
        form = QFormLayout(tab)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(12)

        self._provider_combo = QComboBox(tab)
        for code, label in PROVIDER_CHOICES:
            self._provider_combo.addItem(label, code)
        self._provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        form.addRow("Провайдер:", self._provider_combo)

        key_row = QHBoxLayout()
        self._api_key_edit = QLineEdit(tab)
        self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_edit.setPlaceholderText("Вставте API-ключ...")
        self._api_key_edit.textChanged.connect(self._on_credentials_edited)
        key_row.addWidget(self._api_key_edit, 1)
        self._toggle_key_btn = QPushButton("Показати", tab)
        self._toggle_key_btn.setCheckable(True)
        self._toggle_key_btn.toggled.connect(self._on_toggle_api_key_visibility)
        key_row.addWidget(self._toggle_key_btn)
        self._verify_key_btn = QPushButton("✓ Перевірити ключ", tab)
        self._verify_key_btn.setToolTip("Перевірити дійсність API-ключа та активність обраної моделі")
        self._verify_key_btn.clicked.connect(self._on_test_connection_clicked)
        key_row.addWidget(self._verify_key_btn)
        form.addRow("API-ключ:", key_row)

        model_col = QVBoxLayout()
        model_col.setSpacing(4)
        model_row = QHBoxLayout()
        self._model_combo = QComboBox(tab)
        self._model_combo.setEditable(True)
        self._model_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._model_combo.setMinimumContentsLength(22)
        self._model_combo.currentTextChanged.connect(self._on_model_text_changed)
        model_row.addWidget(self._model_combo, 1)
        self._fetch_models_btn = QPushButton("↻ Завантажити моделі", tab)
        self._fetch_models_btn.setToolTip(
            "Отримати актуальний список доступних моделей за вказаним API-ключем"
        )
        self._fetch_models_btn.clicked.connect(self._on_fetch_models_clicked)
        model_row.addWidget(self._fetch_models_btn)
        model_col.addLayout(model_row)

        self._model_warn_lbl = QLabel("", tab)
        self._model_warn_lbl.setWordWrap(True)
        self._model_warn_lbl.setStyleSheet("color: #FBBF24; font-size: 11px;")
        self._model_warn_lbl.setVisible(False)
        model_col.addWidget(self._model_warn_lbl)
        form.addRow("Модель:", model_col)

        self._base_url_edit = QLineEdit(tab)
        self._base_url_edit.textChanged.connect(self._on_credentials_edited)
        form.addRow("Base URL (опційно):", self._base_url_edit)

        self._fallback_provider_combo = QComboBox(tab)
        for code, label in FALLBACK_PROVIDER_CHOICES:
            self._fallback_provider_combo.addItem(label, code)
        self._fallback_provider_combo.setToolTip(
            "Резервний провайдер, якщо основний повернув 429, таймаут або помилку сервера"
        )
        form.addRow("Авто-фолбек провайдер:", self._fallback_provider_combo)

        params_row = QHBoxLayout()
        self._temp_spin = QDoubleSpinBox(tab)
        self._temp_spin.setRange(0.0, 2.0)
        self._temp_spin.setSingleStep(0.1)
        params_row.addWidget(QLabel("Температура:", tab))
        params_row.addWidget(self._temp_spin)
        params_row.addSpacing(16)
        self._max_tokens_spin = QSpinBox(tab)
        self._max_tokens_spin.setRange(64, 8192)
        self._max_tokens_spin.setSingleStep(64)
        params_row.addWidget(QLabel("Макс. токенів:", tab))
        params_row.addWidget(self._max_tokens_spin)
        params_row.addStretch(1)
        form.addRow("Параметри:", params_row)

        self._test_result_lbl = QLabel("", tab)
        self._test_result_lbl.setWordWrap(True)
        form.addRow("Діагностика:", self._test_result_lbl)

        scroll.setWidget(tab)
        return scroll

    def _build_speech_tab(self) -> QWidget:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        tab = QWidget(scroll)
        form = QFormLayout(tab)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(12)
        self._speech_form = form

        self._stt_provider_combo = QComboBox(tab)
        for code, label in STT_PROVIDER_CHOICES:
            self._stt_provider_combo.addItem(label, code)
        self._stt_provider_combo.currentIndexChanged.connect(self._on_stt_provider_changed)
        form.addRow("Рушій розпізнавання (STT):", self._stt_provider_combo)

        self._stt_hint_lbl = QLabel("", tab)
        self._stt_hint_lbl.setObjectName("MutedText")
        self._stt_hint_lbl.setWordWrap(True)
        form.addRow("", self._stt_hint_lbl)

        self._stt_model_combo = QComboBox(tab)
        self._stt_model_combo.setEditable(True)
        form.addRow("Модель хмарного STT:", self._stt_model_combo)

        stt_key_row = QHBoxLayout()
        self._stt_api_key_edit = QLineEdit(tab)
        self._stt_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        stt_key_row.addWidget(self._stt_api_key_edit, 1)
        self._toggle_stt_key_btn = QPushButton("Показати", tab)
        self._toggle_stt_key_btn.setCheckable(True)
        self._toggle_stt_key_btn.toggled.connect(self._on_toggle_stt_key_visibility)
        stt_key_row.addWidget(self._toggle_stt_key_btn)
        self._stt_key_row = stt_key_row
        form.addRow("STT API-ключ:", stt_key_row)

        stt_diag_row = QHBoxLayout()
        self._test_stt_btn = QPushButton("Перевірити STT", tab)
        self._test_stt_btn.clicked.connect(self._on_test_stt_clicked)
        stt_diag_row.addWidget(self._test_stt_btn)
        self._stt_test_lbl = QLabel("", tab)
        self._stt_test_lbl.setWordWrap(True)
        stt_diag_row.addWidget(self._stt_test_lbl, 1)
        form.addRow("Діагностика STT:", stt_diag_row)

        self._whisper_model_combo = QComboBox(tab)
        for code, label in WHISPER_MODEL_CHOICES:
            self._whisper_model_combo.addItem(label, code)
        form.addRow("Модель Whisper:", self._whisper_model_combo)

        self._whisper_device_combo = QComboBox(tab)
        self._whisper_device_combo.addItem("CUDA — відеокарта NVIDIA (рекомендовано)", "cuda")
        self._whisper_device_combo.addItem("Auto — автовибір", "auto")
        self._whisper_device_combo.addItem("CPU — без відеокарти", "cpu")
        form.addRow("Пристрій:", self._whisper_device_combo)

        self._whisper_compute_combo = QComboBox(tab)
        self._whisper_compute_combo.addItem("float16 (швидко на RTX GPU)", "float16")
        self._whisper_compute_combo.addItem("int8 (економія пам'яті / CPU)", "int8")
        self._whisper_compute_combo.addItem("float32", "float32")
        form.addRow("Точність обчислень:", self._whisper_compute_combo)

        self._whisper_lang_combo = QComboBox(tab)
        for code, label in WHISPER_LANG_CHOICES:
            self._whisper_lang_combo.addItem(label, code)
        form.addRow("Мова розпізнавання:", self._whisper_lang_combo)

        self._beam_size_spin = QSpinBox(tab)
        self._beam_size_spin.setRange(1, 10)
        form.addRow("Якість пошуку (Beam size):", self._beam_size_spin)

        self._chunk_sec_spin = QDoubleSpinBox(tab)
        self._chunk_sec_spin.setRange(2.0, 15.0)
        self._chunk_sec_spin.setSingleStep(0.5)
        self._chunk_sec_spin.setSuffix(" с")
        form.addRow("Довжина чанка:", self._chunk_sec_spin)

        self._silence_sec_spin = QDoubleSpinBox(tab)
        self._silence_sec_spin.setRange(0.2, 5.0)
        self._silence_sec_spin.setSingleStep(0.1)
        self._silence_sec_spin.setSuffix(" с")
        form.addRow("Пауза кінця фрази:", self._silence_sec_spin)

        self._window_sec_spin = QSpinBox(tab)
        self._window_sec_spin.setRange(10, 300)
        self._window_sec_spin.setSingleStep(10)
        self._window_sec_spin.setSuffix(" с")
        form.addRow("Вікно контексту реплік:", self._window_sec_spin)

        self._silence_thresh_spin = QDoubleSpinBox(tab)
        self._silence_thresh_spin.setRange(0.001, 0.050)
        self._silence_thresh_spin.setDecimals(3)
        self._silence_thresh_spin.setSingleStep(0.001)
        self._silence_thresh_spin.setSuffix(" RMS")
        self._silence_thresh_spin.setToolTip(
            "Резервний поріг гучності мовлення, коли Silero VAD вимкнений"
        )
        form.addRow("Поріг тиші (RMS резерв):", self._silence_thresh_spin)

        self._rag_budget_spin = QSpinBox(tab)
        self._rag_budget_spin.setRange(2000, 40000)
        self._rag_budget_spin.setSingleStep(1000)
        self._rag_budget_spin.setSuffix(" симв.")
        self._rag_budget_spin.setToolTip(
            "Скільки релевантних символів із ваших документів передавати LLM (більше = повніший контекст)"
        )
        form.addRow("Бюджет RAG-контексту:", self._rag_budget_spin)

        scroll.setWidget(tab)
        return scroll

    def _build_overlay_tab(self) -> QWidget:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        tab = QWidget(scroll)
        form = QFormLayout(tab)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(12)

        self._ov_stealth_chk = QCheckBox(
            "Stealth-режим оверлею (невидимий при шарингу екрана)",
            tab,
        )
        self._ov_stealth_chk.setToolTip(
            "Використовує Windows WDA_EXCLUDEFROMCAPTURE для приховування оверлею на дзвінках та записах."
        )
        self._ov_stealth_chk.clicked.connect(self._on_ov_stealth_checkbox_clicked)
        form.addRow("Оверлей:", self._ov_stealth_chk)

        self._dash_stealth_chk = QCheckBox(
            "Stealth-режим панелі керування (ховати від запису екрана)",
            tab,
        )
        self._dash_stealth_chk.setToolTip(
            "Застосовує WDA_EXCLUDEFROMCAPTURE також до головного вікна панелі керування."
        )
        self._dash_stealth_chk.clicked.connect(self._on_dash_stealth_checkbox_clicked)
        form.addRow("Панель:", self._dash_stealth_chk)

        stealth_info = QLabel(
            "Stealth-режим завжди увімкнений при запуску. Вимкнути його можна лише тут через підтвердження.",
            tab,
        )
        stealth_info.setObjectName("MutedText")
        stealth_info.setWordWrap(True)
        form.addRow("", stealth_info)

        op_row = QHBoxLayout()
        self._ov_opacity_slider = QSlider(Qt.Orientation.Horizontal, tab)
        self._ov_opacity_slider.setRange(30, 100)
        self._ov_opacity_lbl = QLabel("92%", tab)
        self._ov_opacity_lbl.setFixedWidth(40)
        self._ov_opacity_slider.valueChanged.connect(
            lambda v: self._ov_opacity_lbl.setText(f"{v}%")
        )
        op_row.addWidget(self._ov_opacity_slider, 1)
        op_row.addWidget(self._ov_opacity_lbl)
        form.addRow("Прозорість оверлею:", op_row)

        fs_row = QHBoxLayout()
        self._ov_font_slider = QSlider(Qt.Orientation.Horizontal, tab)
        self._ov_font_slider.setRange(85, 140)
        self._ov_font_lbl = QLabel("100%", tab)
        self._ov_font_lbl.setFixedWidth(40)
        self._ov_font_slider.valueChanged.connect(
            lambda v: self._ov_font_lbl.setText(f"{v}%")
        )
        fs_row.addWidget(self._ov_font_slider, 1)
        fs_row.addWidget(self._ov_font_lbl)
        form.addRow("Масштаб шрифту:", fs_row)

        self._hk_answer_edit = QLineEdit(tab)
        self._hk_answer_edit.setPlaceholderText("ctrl+space")
        form.addRow("Клавіша відповіді:", self._hk_answer_edit)

        self._hk_toggle_edit = QLineEdit(tab)
        self._hk_toggle_edit.setPlaceholderText("ctrl+shift+h")
        form.addRow("Показати / сховати:", self._hk_toggle_edit)

        self._hk_capture_edit = QLineEdit(tab)
        self._hk_capture_edit.setPlaceholderText("ctrl+alt+r")
        form.addRow("Старт / стоп сесії:", self._hk_capture_edit)

        self._hk_screen_edit = QLineEdit(tab)
        self._hk_screen_edit.setPlaceholderText("ctrl+alt+s")
        form.addRow("Зір екрана (👁):", self._hk_screen_edit)

        self._hk_web_edit = QLineEdit(tab)
        self._hk_web_edit.setPlaceholderText("ctrl+alt+w")
        form.addRow("Веб-пошук (🌐):", self._hk_web_edit)

        self._hk_snip_edit = QLineEdit(tab)
        self._hk_snip_edit.setPlaceholderText("ctrl+alt+a")
        form.addRow("Область екрана (✂):", self._hk_snip_edit)

        scroll.setWidget(tab)
        return scroll

    def _build_web_tab(self) -> QWidget:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        tab = QWidget(scroll)
        form = QFormLayout(tab)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(12)

        self._web_enabled_chk = QCheckBox(
            "Увімкнути пошук в інтернеті та веб-агента",
            tab,
        )
        form.addRow("Доступ до мережі:", self._web_enabled_chk)

        self._web_agent_mode_combo = QComboBox(tab)
        self._web_agent_mode_combo.addItem(
            "AI-агент (сам вирішує, що шукати й які сторінки читати)",
            "agent",
        )
        self._web_agent_mode_combo.addItem(
            "Швидкий пошук (1 запит у пошукач + стрімінг, мінімальна затримка)",
            "fast",
        )
        form.addRow("Режим роботи:", self._web_agent_mode_combo)

        self._web_provider_combo = QComboBox(tab)
        self._web_provider_combo.addItem(
            "Авто (безкоштовний DDGS / Вікіпедія або Tavily/Brave за ключем)",
            "auto",
        )
        self._web_provider_combo.addItem(
            "DuckDuckGo / Metasearch (ddgs, без API-ключа)",
            "ddgs",
        )
        self._web_provider_combo.addItem(
            "Tavily Search API (потрібен ключ tvly-...)",
            "tavily",
        )
        self._web_provider_combo.addItem(
            "Brave Search API (потрібен ключ BSA...)",
            "brave",
        )
        self._web_provider_combo.addItem(
            "Вікіпедія API (українська / англійська, без ключа)",
            "wikipedia",
        )
        form.addRow("Пошуковий рушій:", self._web_provider_combo)

        web_key_row = QHBoxLayout()
        self._web_api_key_edit = QLineEdit(tab)
        self._web_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._web_api_key_edit.setPlaceholderText(
            "Опційно: ключ Tavily (tvly-...) або Brave (BSA...); для DDGS/Вікі не потрібен"
        )
        web_key_row.addWidget(self._web_api_key_edit, 1)
        self._toggle_web_key_btn = QPushButton("Показати", tab)
        self._toggle_web_key_btn.setCheckable(True)
        self._toggle_web_key_btn.toggled.connect(self._on_toggle_web_key_visibility)
        web_key_row.addWidget(self._toggle_web_key_btn)
        form.addRow("API-ключ пошуку:", web_key_row)

        self._web_max_results_spin = QSpinBox(tab)
        self._web_max_results_spin.setRange(1, 10)
        self._web_max_results_spin.setSuffix(" рез.")
        form.addRow("Макс. результатів:", self._web_max_results_spin)

        obscura_col = QVBoxLayout()
        obscura_col.setSpacing(4)
        obscura_row = QHBoxLayout()
        self._obscura_path_edit = QLineEdit(tab)
        self._obscura_path_edit.setPlaceholderText(
            "Опційно: шлях до obscura.exe (інакше швидкий HTTP+HTML парсер)"
        )
        obscura_row.addWidget(self._obscura_path_edit, 1)
        obscura_browse_btn = QPushButton("Огляд...", tab)
        obscura_browse_btn.clicked.connect(self._on_browse_obscura_clicked)
        obscura_row.addWidget(obscura_browse_btn)
        obscura_col.addLayout(obscura_row)

        obscura_hint = QLabel(
            "Якщо вказати obscura.exe, агент використовуватиме його для JS-важких сторінок "
            "(--dump markdown) з автоматичним HTTP-фолбеком.",
            tab,
        )
        obscura_hint.setObjectName("MutedText")
        obscura_hint.setWordWrap(True)
        obscura_col.addWidget(obscura_hint)
        form.addRow("Obscura CLI:", obscura_col)

        web_test_row = QHBoxLayout()
        self._test_web_btn = QPushButton("🔍 Перевірити веб-пошук", tab)
        self._test_web_btn.clicked.connect(self._on_test_web_clicked)
        web_test_row.addWidget(self._test_web_btn)
        self._web_test_lbl = QLabel("", tab)
        self._web_test_lbl.setWordWrap(True)
        web_test_row.addWidget(self._web_test_lbl, 1)
        form.addRow("Діагностика веб:", web_test_row)

        self._screen_enabled_chk = QCheckBox(
            "Дозволити агенту бачити екран (код в IDE / слайди на екрані)",
            tab,
        )
        form.addRow("Зір (Екран):", self._screen_enabled_chk)

        self._screen_monitor_combo = QComboBox(tab)
        self._screen_monitor_combo.addItem("Основний монітор (авто)", -1)
        form.addRow("Монітор для знімка:", self._screen_monitor_combo)

        self._screen_max_width_spin = QSpinBox(tab)
        self._screen_max_width_spin.setRange(800, 2560)
        self._screen_max_width_spin.setSingleStep(160)
        self._screen_max_width_spin.setSuffix(" px")
        form.addRow("Макс. ширина знімка:", self._screen_max_width_spin)

        screen_test_row = QHBoxLayout()
        self._test_screen_btn = QPushButton("👁 Тест знімка екрана", tab)
        self._test_screen_btn.clicked.connect(self._on_test_screen_clicked)
        screen_test_row.addWidget(self._test_screen_btn)
        self._screen_test_lbl = QLabel("", tab)
        self._screen_test_lbl.setWordWrap(True)
        screen_test_row.addWidget(self._screen_test_lbl, 1)
        form.addRow("Діагностика екрана:", screen_test_row)

        groq_tip = QLabel(
            "Підказка для Groq: швидкий AI-агент працює з будь-якою моделлю Groq "
            "(рекомендовано qwen/qwen3.8-27b — підтримує і веб-інструменти, і зір екрана, "
            "та мінімальну затримку reasoning_effort=none). Якщо увімкнути «👁 Екран» "
            "на текстовій моделі (напр. llama-3.1-8b-instant), аналіз знімка автоматично "
            "виконається через qwen/qwen3.8-27b.",
            tab,
        )
        groq_tip.setObjectName("MutedText")
        groq_tip.setWordWrap(True)
        form.addRow("Groq / Gemini:", groq_tip)

        scroll.setWidget(tab)
        return scroll

    # --- Settings binding ---

    def _on_toggle_api_key_visibility(self, checked: bool) -> None:
        self._api_key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        )
        self._toggle_key_btn.setText("Сховати" if checked else "Показати")

    def _on_toggle_web_key_visibility(self, checked: bool) -> None:
        self._web_api_key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        )
        self._toggle_web_key_btn.setText("Сховати" if checked else "Показати")

    def _on_toggle_stt_key_visibility(self, checked: bool) -> None:
        self._stt_api_key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        )
        self._toggle_stt_key_btn.setText("Сховати" if checked else "Показати")

    def _on_stt_provider_changed(self, _idx: int, *, clear_text: bool = True) -> None:
        prov = str(self._stt_provider_combo.currentData() or "local").lower()
        is_local = prov == "local"
        is_deepgram = prov == "deepgram"

        self._stt_hint_lbl.setText(STT_PROVIDER_HINTS.get(prov, ""))
        self._speech_form.setRowVisible(self._stt_model_combo, not is_local)
        self._speech_form.setRowVisible(self._stt_key_row, not is_local)
        self._speech_form.setRowVisible(self._whisper_model_combo, is_local)
        self._speech_form.setRowVisible(self._whisper_device_combo, is_local)
        self._speech_form.setRowVisible(self._whisper_compute_combo, is_local)
        self._speech_form.setRowVisible(self._beam_size_spin, is_local)
        self._speech_form.setRowVisible(self._chunk_sec_spin, not is_deepgram)
        self._speech_form.setRowVisible(self._silence_thresh_spin, not is_deepgram)

        if not is_local:
            presets = STT_MODEL_PRESETS.get(prov, [])
            cur = self._stt_model_combo.currentText().strip()
            self._stt_model_combo.blockSignals(True)
            self._stt_model_combo.clear()
            for m in presets:
                self._stt_model_combo.addItem(m)
            if not clear_text and cur:
                self._stt_model_combo.setCurrentText(cur)
            elif presets:
                self._stt_model_combo.setCurrentIndex(0)
            self._stt_model_combo.blockSignals(False)

            placeholders = {
                "deepgram": "Вставте Deepgram API Key...",
                "elevenlabs": "Опційно: порожньо = ELEVENLABS_API_KEY із .env",
                "groq": "Опційно: порожньо = Groq API-ключ зі вкладки ШІ",
            }
            self._stt_api_key_edit.setPlaceholderText(placeholders.get(prov, "Вставте STT API-ключ..."))

        self._stt_test_lbl.setText("")

    def _on_test_stt_clicked(self) -> None:
        temp_cfg = self._collect_config_from_ui()
        self._test_stt_btn.setEnabled(False)
        self._stt_test_lbl.setStyleSheet("color: #93A2B8;")
        self._stt_test_lbl.setText("Перевіряю STT...")
        self.test_stt_requested.emit(temp_cfg)

    def set_stt_test_result(self, ok: bool, message: str) -> None:
        self._test_stt_btn.setEnabled(True)
        color = "#4ADE80" if ok else "#F87171"
        self._stt_test_lbl.setStyleSheet(f"color: {color}; font-weight: 500;")
        self._stt_test_lbl.setText(message)

    def _open_settings_tab(self, tab_idx: int) -> None:
        self._switch_page(3)
        self._settings_tabs.setCurrentIndex(tab_idx)

    def _populate_settings_from_config(self, cfg: AppConfig) -> None:
        self._config = replace(cfg)
        self._active_provider = str(cfg.llm_provider or "openai")
        self._draft_keys.clear()
        self._draft_models.clear()

        p_idx = self._provider_combo.findData(cfg.llm_provider)
        self._provider_combo.blockSignals(True)
        self._provider_combo.setCurrentIndex(max(0, p_idx))
        self._provider_combo.blockSignals(False)
        self._on_provider_changed(self._provider_combo.currentIndex(), clear_stale_base_url=False)

        current_model = get_provider_model(cfg, cfg.llm_provider)
        self._model_combo.blockSignals(True)
        self._model_combo.setCurrentText(current_model)
        self._model_combo.blockSignals(False)
        self._update_model_warning(current_model)
        self._base_url_edit.setText("" if cfg.llm_provider == "gemini" else cfg.openai_base_url)
        self._select_combo_by_data(self._fallback_provider_combo, cfg.llm_fallback_provider)
        self._temp_spin.setValue(cfg.llm_temperature)
        self._max_tokens_spin.setValue(cfg.llm_max_tokens)

        self._select_combo_by_data(self._stt_provider_combo, cfg.stt_provider)
        if cfg.whisper_model and self._whisper_model_combo.findData(cfg.whisper_model) < 0:
            self._whisper_model_combo.addItem(cfg.whisper_model, cfg.whisper_model)
        self._select_combo_by_data(self._whisper_model_combo, cfg.whisper_model)
        self._select_combo_by_data(self._whisper_device_combo, cfg.whisper_device)
        self._select_combo_by_data(self._whisper_compute_combo, cfg.whisper_compute_type)
        self._select_combo_by_data(self._whisper_lang_combo, cfg.whisper_language)
        self._beam_size_spin.setValue(cfg.whisper_beam_size)
        self._chunk_sec_spin.setValue(cfg.audio_chunk_seconds)
        self._silence_sec_spin.setValue(cfg.audio_silence_seconds)
        self._window_sec_spin.setValue(cfg.transcript_window_seconds)
        self._silence_thresh_spin.setValue(cfg.audio_silence_threshold)
        self._rag_budget_spin.setValue(cfg.rag_context_budget_chars)
        self._stt_api_key_edit.setText(cfg.stt_api_key)
        self._on_stt_provider_changed(self._stt_provider_combo.currentIndex(), clear_text=True)
        if cfg.stt_model:
            self._stt_model_combo.setCurrentText(cfg.stt_model)

        self._ov_stealth_chk.blockSignals(True)
        self._ov_stealth_chk.setChecked(cfg.overlay_stealth)
        self._ov_stealth_chk.blockSignals(False)
        self._dash_stealth_chk.blockSignals(True)
        self._dash_stealth_chk.setChecked(cfg.dashboard_stealth)
        self._dash_stealth_chk.blockSignals(False)
        self._ov_opacity_slider.setValue(cfg.overlay_opacity)
        self._ov_font_slider.setValue(int(round(cfg.overlay_font_scale * 100)))
        self._hk_answer_edit.setText(cfg.hotkey_answer)
        self._hk_toggle_edit.setText(cfg.hotkey_toggle)
        self._hk_capture_edit.setText(cfg.hotkey_capture)
        self._hk_screen_edit.setText(cfg.hotkey_screen)
        self._hk_web_edit.setText(cfg.hotkey_web)
        self._hk_snip_edit.setText(cfg.hotkey_snip)
        self.vocab_edit.setText(cfg.custom_vocabulary)
        self.prompt_edit.setPlainText(cfg.system_prompt)

        self._web_enabled_chk.blockSignals(True)
        self._web_enabled_chk.setChecked(cfg.web_search_enabled)
        self._web_enabled_chk.blockSignals(False)
        self._select_combo_by_data(self._web_agent_mode_combo, cfg.web_agent_mode)
        self._select_combo_by_data(self._web_provider_combo, cfg.web_search_provider)
        self._web_api_key_edit.setText(cfg.web_search_api_key)
        self._web_max_results_spin.setValue(cfg.web_search_max_results)
        self._obscura_path_edit.setText(cfg.obscura_path)
        self._sync_live_web_btn(cfg.web_search_enabled)

        self._screen_enabled_chk.blockSignals(True)
        self._screen_enabled_chk.setChecked(cfg.screen_vision_enabled)
        self._screen_enabled_chk.blockSignals(False)
        self._select_combo_by_data(self._screen_monitor_combo, cfg.screen_capture_index)
        self._screen_max_width_spin.setValue(cfg.screen_capture_max_width)
        self._sync_live_screen_btn(cfg.screen_vision_enabled)

        self._hk_answer_pill.setText(self._fmt_hotkey(cfg.hotkey_answer))
        self._hk_toggle_pill.setText(self._fmt_hotkey(cfg.hotkey_toggle))
        self._hk_capture_pill.setText(self._fmt_hotkey(cfg.hotkey_capture))
        self._hk_screen_pill.setText(self._fmt_hotkey(cfg.hotkey_screen))
        self._hk_web_pill.setText(self._fmt_hotkey(cfg.hotkey_web))
        self._hk_snip_pill.setText(self._fmt_hotkey(cfg.hotkey_snip))
        self._refresh_live_toggle_tooltips(cfg)

    @staticmethod
    def _select_combo_by_data(combo: QComboBox, value: str) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setCurrentIndex(0)

    @staticmethod
    def _fmt_hotkey(spec: str) -> str:
        return " + ".join(p.strip().strip("<>").capitalize() for p in spec.split("+") if p.strip())

    def _on_provider_changed(self, _idx: int, *, clear_stale_base_url: bool = True) -> None:
        provider = str(self._provider_combo.currentData() or "openai")
        prev_provider = self._active_provider
        provider_switched = provider != prev_provider
        if provider_switched and prev_provider:
            if prev_provider not in {"ollama", "lmstudio"}:
                self._draft_keys[prev_provider] = self._api_key_edit.text().strip()
            prev_model_txt = self._model_combo.currentText().strip()
            if prev_model_txt:
                self._draft_models[prev_provider] = prev_model_txt

        self._active_provider = provider
        is_local = provider in {"ollama", "lmstudio"}
        is_gemini = provider == "gemini"

        self._api_key_edit.blockSignals(True)
        self._api_key_edit.setEnabled(not is_local)
        if is_local:
            self._api_key_edit.setText("")
            self._api_key_edit.setPlaceholderText("Для локальної моделі ключ не потрібен")
        else:
            key_placeholders = {
                "gemini": "Вставте GEMINI_API_KEY...",
                "groq": "Вставте Groq API Key (gsk_...)...",
                "cerebras": "Вставте Cerebras API Key (csk-...)...",
                "openai": "Вставте OPENAI_API_KEY...",
            }
            restored_key = self._draft_keys.get(provider, get_provider_api_key(self._config, provider))
            self._api_key_edit.setText(restored_key)
            self._api_key_edit.setPlaceholderText(
                key_placeholders.get(provider, "Вставте API-ключ...")
            )
        self._api_key_edit.blockSignals(False)

        self._base_url_edit.blockSignals(True)
        self._base_url_edit.setEnabled(not is_gemini)
        if is_gemini or (clear_stale_base_url and provider_switched):
            self._base_url_edit.setText("")
        self._base_url_edit.setPlaceholderText(BASE_URL_PLACEHOLDERS.get(provider, ""))
        self._base_url_edit.blockSignals(False)

        target_model = self._draft_models.get(provider) or get_provider_model(self._config, provider)
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        presets = MODEL_PRESETS.get(provider, [])
        for m in presets:
            self._model_combo.addItem(m)
        if target_model:
            self._model_combo.setCurrentText(target_model)
        elif presets:
            self._model_combo.setCurrentIndex(0)
        self._model_combo.blockSignals(False)
        self._update_model_warning(self._model_combo.currentText())
        self._test_result_lbl.setText("")
        if provider_switched:
            self._save_status_lbl.setStyleSheet("color: #FBBF24;")
            self._save_status_lbl.setText(
                "Є незбережені зміни — збережеться автоматично або натисніть «Зберегти та застосувати»."
            )

    def _on_credentials_edited(self, _text: str) -> None:
        self._last_check_result = None

    def _update_model_warning(self, text: str) -> None:
        repl = get_deprecated_model_replacement(text)
        if repl:
            self._model_warn_lbl.setText(
                f"Увага: модель «{text.strip()}» застаріла або вимкнена. "
                f"Рекомендована заміна: {repl} (або натисніть «↻ Завантажити моделі»)."
            )
            self._model_warn_lbl.setVisible(True)
        else:
            self._model_warn_lbl.setVisible(False)

    def _on_model_text_changed(self, text: str) -> None:
        self._last_check_result = None
        self._update_model_warning(text)

    def _collect_config_from_ui(self) -> AppConfig:
        provider = str(self._provider_combo.currentData() or "openai")
        model_text = self._model_combo.currentText().strip()
        key_text = self._api_key_edit.text().strip()

        openai_key = self._draft_keys.get("openai", self._config.openai_api_key)
        gemini_key = self._draft_keys.get("gemini", self._config.gemini_api_key)
        groq_key = self._draft_keys.get("groq", self._config.groq_api_key)
        cerebras_key = self._draft_keys.get("cerebras", self._config.cerebras_api_key)
        openai_model = self._draft_models.get("openai", self._config.openai_model)
        gemini_model = self._draft_models.get("gemini", self._config.gemini_model)
        groq_model = self._draft_models.get("groq", self._config.groq_model)
        cerebras_model = self._draft_models.get("cerebras", self._config.cerebras_model)

        if provider == "gemini":
            gemini_key = key_text
            if model_text:
                gemini_model = model_text
        elif provider == "groq":
            groq_key = key_text
            if model_text:
                groq_model = model_text
        elif provider == "cerebras":
            cerebras_key = key_text
            if model_text:
                cerebras_model = model_text
        else:
            if provider not in {"ollama", "lmstudio"}:
                openai_key = key_text
            if model_text:
                openai_model = model_text

        prompt_text = self.prompt_edit.toPlainText().strip() or DEFAULT_SYSTEM_PROMPT
        base_url = "" if provider == "gemini" else self._base_url_edit.text().strip()

        return replace(
            self._config,
            llm_provider=provider,
            openai_api_key=openai_key,
            openai_base_url=base_url,
            openai_model=openai_model,
            groq_api_key=groq_key,
            groq_model=groq_model,
            cerebras_api_key=cerebras_key,
            cerebras_model=cerebras_model,
            gemini_api_key=gemini_key,
            gemini_model=gemini_model,
            llm_fallback_provider=str(self._fallback_provider_combo.currentData() or "auto"),
            llm_temperature=float(self._temp_spin.value()),
            llm_max_tokens=int(self._max_tokens_spin.value()),
            whisper_model=str(self._whisper_model_combo.currentData() or "large-v3-turbo"),
            whisper_device=str(self._whisper_device_combo.currentData() or "cuda"),
            whisper_compute_type=str(self._whisper_compute_combo.currentData() or "float16"),
            whisper_language=str(self._whisper_lang_combo.currentData() or "uk"),
            whisper_beam_size=int(self._beam_size_spin.value()),
            stt_provider=str(self._stt_provider_combo.currentData() or "local"),
            stt_api_key=self._stt_api_key_edit.text().strip(),
            stt_model=(
                self._config.stt_model
                if str(self._stt_provider_combo.currentData() or "local") == "local"
                else self._stt_model_combo.currentText().strip()
            ),
            audio_chunk_seconds=float(self._chunk_sec_spin.value()),
            audio_silence_seconds=float(self._silence_sec_spin.value()),
            audio_silence_threshold=round(float(self._silence_thresh_spin.value()), 4),
            transcript_window_seconds=int(self._window_sec_spin.value()),
            rag_context_budget_chars=int(self._rag_budget_spin.value()),
            overlay_stealth=self._ov_stealth_chk.isChecked(),
            dashboard_stealth=self._dash_stealth_chk.isChecked(),
            overlay_opacity=int(self._ov_opacity_slider.value()),
            overlay_font_scale=round(self._ov_font_slider.value() / 100.0, 2),
            hotkey_answer=self._hk_answer_edit.text().strip().lower() or "ctrl+space",
            hotkey_toggle=self._hk_toggle_edit.text().strip().lower() or "ctrl+shift+h",
            hotkey_capture=self._hk_capture_edit.text().strip().lower() or "ctrl+alt+r",
            hotkey_screen=self._hk_screen_edit.text().strip().lower() or "ctrl+alt+s",
            hotkey_web=self._hk_web_edit.text().strip().lower() or "ctrl+alt+w",
            hotkey_snip=self._hk_snip_edit.text().strip().lower() or "ctrl+alt+a",
            custom_vocabulary=self.vocab_edit.text().strip(),
            system_prompt=prompt_text,
            web_search_enabled=self._web_enabled_chk.isChecked(),
            web_search_provider=str(self._web_provider_combo.currentData() or "auto"),
            web_search_api_key=self._web_api_key_edit.text().strip(),
            web_agent_mode=str(self._web_agent_mode_combo.currentData() or "agent"),
            web_search_max_results=int(self._web_max_results_spin.value()),
            obscura_path=self._obscura_path_edit.text().strip(),
            screen_vision_enabled=self._screen_enabled_chk.isChecked(),
            screen_capture_index=int(self._screen_monitor_combo.currentData() if self._screen_monitor_combo.currentData() is not None else -1),
            screen_capture_max_width=int(self._screen_max_width_spin.value()),
        )

    def _on_browse_obscura_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Оберіть виконуваний файл Obscura CLI",
            "",
            "Виконувані файли (*.exe);;Усі файли (*.*)",
        )
        if path_str:
            self._obscura_path_edit.setText(path_str)

    def _on_test_web_clicked(self) -> None:
        temp_cfg = self._collect_config_from_ui()
        self._test_web_btn.setEnabled(False)
        self._web_test_lbl.setStyleSheet("color: #93A2B8;")
        self._web_test_lbl.setText("Виконую тестовий пошуковий запит...")
        self.test_web_search_requested.emit(temp_cfg)

    def set_web_test_result(self, ok: bool, message: str) -> None:
        self._test_web_btn.setEnabled(True)
        color = "#4ADE80" if ok else "#F87171"
        self._web_test_lbl.setStyleSheet(f"color: {color}; font-weight: 500;")
        self._web_test_lbl.setText(message)

    def _sync_live_web_btn(self, enabled: bool) -> None:
        state = bool(enabled)
        self._live_web_btn.blockSignals(True)
        self._live_web_btn.setChecked(state)
        self._live_web_btn.setText("🌐 Веб: ON" if state else "🌐 Веб")
        self._live_web_btn.blockSignals(False)

    def _on_live_web_toggled(self, checked: bool) -> None:
        state = bool(checked)
        self._config.web_search_enabled = state
        self._live_web_btn.setText("🌐 Веб: ON" if state else "🌐 Веб")
        self._web_enabled_chk.blockSignals(True)
        self._web_enabled_chk.setChecked(state)
        self._web_enabled_chk.blockSignals(False)
        self.refresh_readiness()
        self.web_search_toggled.emit(state)

    def set_web_search_enabled(self, enabled: bool) -> None:
        """Synchronize Live tab, Settings checkbox, and Home readiness with web state."""
        state = bool(enabled)
        self._config.web_search_enabled = state
        self._sync_live_web_btn(state)
        if self._web_enabled_chk.isChecked() != state:
            self._web_enabled_chk.blockSignals(True)
            self._web_enabled_chk.setChecked(state)
            self._web_enabled_chk.blockSignals(False)
        self.refresh_readiness()

    def set_screens(self, screens: list[tuple[int, str]]) -> None:
        """Populate display selector while keeping auto primary monitor (-1) first."""
        current_data = self._screen_monitor_combo.currentData()
        target_data = (
            current_data
            if (isinstance(current_data, int) and current_data >= 0)
            else self._config.screen_capture_index
        )
        self._screen_monitor_combo.blockSignals(True)
        self._screen_monitor_combo.clear()
        self._screen_monitor_combo.addItem("Основний монітор (авто)", -1)
        for idx, label in screens:
            self._screen_monitor_combo.addItem(label, idx)
        self._select_combo_by_data(self._screen_monitor_combo, int(target_data))
        self._screen_monitor_combo.blockSignals(False)

    def _on_test_screen_clicked(self) -> None:
        data = self._screen_monitor_combo.currentData()
        idx = int(data) if data is not None else -1
        max_w = int(self._screen_max_width_spin.value())
        self._screen_test_lbl.setStyleSheet("color: #93A2B8;")
        self._screen_test_lbl.setText("Роблю тестовий знімок екрана...")
        self.test_screen_capture_requested.emit(idx, max_w)

    def set_screen_test_result(self, ok: bool, message: str) -> None:
        color = "#4ADE80" if ok else "#F87171"
        self._screen_test_lbl.setStyleSheet(f"color: {color}; font-weight: 500;")
        self._screen_test_lbl.setText(message)

    def _sync_live_screen_btn(self, enabled: bool) -> None:
        state = bool(enabled)
        self._live_screen_btn.blockSignals(True)
        self._live_screen_btn.setChecked(state)
        self._live_screen_btn.setText("👁 Екран: ON" if state else "👁 Екран")
        self._live_screen_btn.blockSignals(False)

    def _on_live_screen_toggled(self, checked: bool) -> None:
        state = bool(checked)
        self._config.screen_vision_enabled = state
        self._live_screen_btn.setText("👁 Екран: ON" if state else "👁 Екран")
        self._screen_enabled_chk.blockSignals(True)
        self._screen_enabled_chk.setChecked(state)
        self._screen_enabled_chk.blockSignals(False)
        self.refresh_readiness()
        self.screen_vision_toggled.emit(state)

    def set_screen_vision_enabled(self, enabled: bool) -> None:
        """Synchronize Live tab, Settings checkbox, and Home readiness with Screen Vision state."""
        state = bool(enabled)
        self._config.screen_vision_enabled = state
        self._sync_live_screen_btn(state)
        if self._screen_enabled_chk.isChecked() != state:
            self._screen_enabled_chk.blockSignals(True)
            self._screen_enabled_chk.setChecked(state)
            self._screen_enabled_chk.blockSignals(False)
        self.refresh_readiness()

    def _set_llm_check_buttons_enabled(self, enabled: bool) -> None:
        self._verify_key_btn.setEnabled(enabled)
        self._fetch_models_btn.setEnabled(enabled)
        self._home_verify_btn.setEnabled(enabled)

    def _on_home_verify_key_clicked(self) -> None:
        self._pending_settings_verify = False
        self._set_llm_check_buttons_enabled(False)
        self._set_dot(self._dot_llm, "warn")
        self._lbl_llm.setText("Перевірка API-ключа...")
        self._test_result_lbl.setStyleSheet("color: #93A2B8;")
        self._test_result_lbl.setText("Перевірка ключа та моделі...")
        self.test_llm_requested.emit(self._config)

    def _on_test_connection_clicked(self) -> None:
        self._pending_settings_verify = True
        temp_cfg = self._collect_config_from_ui()
        self._set_llm_check_buttons_enabled(False)
        self._test_result_lbl.setStyleSheet("color: #93A2B8;")
        self._test_result_lbl.setText("Перевірка API-ключа та моделі...")
        self.test_llm_requested.emit(temp_cfg)

    def _on_fetch_models_clicked(self) -> None:
        temp_cfg = self._collect_config_from_ui()
        self._set_llm_check_buttons_enabled(False)
        self._test_result_lbl.setStyleSheet("color: #93A2B8;")
        self._test_result_lbl.setText("Завантаження списку активних моделей...")
        self.fetch_models_requested.emit(temp_cfg)

    def _apply_live_models_to_combo(
        self, models: list[str], force_select_first_if_missing: bool = False
    ) -> None:
        if not models:
            return
        current = self._model_combo.currentText().strip()
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for m in models:
            self._model_combo.addItem(m)
        if current in models:
            self._model_combo.setCurrentText(current)
        elif force_select_first_if_missing:
            self._model_combo.setCurrentIndex(0)
        elif current:
            self._model_combo.setCurrentText(current)
        self._model_combo.blockSignals(False)
        self._update_model_warning(self._model_combo.currentText())

    def set_fetched_models(self, models: list[str], error: str = "") -> None:
        self._set_llm_check_buttons_enabled(True)
        if error:
            self._test_result_lbl.setStyleSheet("color: #F87171; font-weight: 500;")
            self._test_result_lbl.setText(error)
            return
        if not models:
            self._test_result_lbl.setStyleSheet("color: #FBBF24; font-weight: 500;")
            self._test_result_lbl.setText("Ключ дійсний, але список моделей порожній.")
            return

        current = self._model_combo.currentText().strip()
        is_deprecated = get_deprecated_model_replacement(current) is not None
        self._apply_live_models_to_combo(
            models, force_select_first_if_missing=(is_deprecated or current not in models)
        )
        self._test_result_lbl.setStyleSheet("color: #4ADE80; font-weight: 500;")
        self._test_result_lbl.setText(
            f"API-ключ дійсний ✓ Завантажено {len(models)} активних моделей."
        )

    def set_api_check_result(self, result: ApiKeyCheckResult) -> None:
        from_settings = self._pending_settings_verify
        self._pending_settings_verify = False
        self._set_llm_check_buttons_enabled(True)
        self._last_check_result = result

        if result.available_models:
            self._apply_live_models_to_combo(
                result.available_models,
                force_select_first_if_missing=(not result.model_valid),
            )

        msg = result.message
        if from_settings and result.ok:
            self._on_save_settings_clicked()
            if not self._settings_save_rejected and "Збережено" not in msg:
                msg = f"{msg} · Збережено та активовано ✓"

        color_map = {
            "ok": "#4ADE80",
            "warn": "#FBBF24",
            "error": "#F87171",
        }
        color = color_map.get(result.status, "#F87171")
        self._test_result_lbl.setStyleSheet(f"color: {color}; font-weight: 500;")
        self._test_result_lbl.setText(msg)
        self.refresh_readiness()

    def set_test_connection_result(self, ok: bool, message: str) -> None:
        self.set_api_check_result(
            ApiKeyCheckResult(
                ok=ok,
                key_valid=ok,
                model_valid=ok,
                status="ok" if ok else "error",
                message=message,
            )
        )

    def _refresh_live_toggle_tooltips(self, cfg: AppConfig | None = None) -> None:
        ref = cfg or self._config
        cap_tip = f"Запустити / зупинити прослуховування ({self._fmt_hotkey(ref.hotkey_capture)})"
        self._home_start_btn.setToolTip(cap_tip)
        self._live_start_btn.setToolTip(cap_tip)
        self._live_web_btn.setToolTip(
            f"Увімкнути / вимкнути пошук в інтернеті ({self._fmt_hotkey(ref.hotkey_web)})"
        )
        self._live_screen_btn.setToolTip(
            f"Зір екрана: прикріплювати знімок екрана ({self._fmt_hotkey(ref.hotkey_screen)})"
        )
        self._live_snip_btn.setToolTip(
            f"Виділити область екрана ({self._fmt_hotkey(ref.hotkey_snip)}) — графік, формула або фрагмент коду"
        )

    @staticmethod
    def _validate_hotkey_pair(
        hk_answer: str,
        hk_toggle: str,
        hk_screen: str | None = None,
        hk_web: str | None = None,
        hk_capture: str | None = None,
        hk_snip: str | None = None,
    ) -> str | None:
        modifiers = {"ctrl", "control", "alt", "shift", "win", "cmd", "super"}
        specs: list[tuple[str, str]] = [
            ("Клавіша відповіді", hk_answer),
            ("Показати / сховати", hk_toggle),
        ]
        if hk_capture is not None:
            specs.append(("Старт / стоп сесії", hk_capture))
        if hk_screen is not None:
            specs.append(("Зір екрана", hk_screen))
        if hk_web is not None:
            specs.append(("Веб-пошук", hk_web))
        if hk_snip is not None:
            specs.append(("Область екрана", hk_snip))

        seen: dict[str, str] = {}
        for label, raw in specs:
            spec = raw.strip().lower()
            if not spec or not spec.isascii():
                return f"{label}: використовуйте латинські клавіші, наприклад ctrl+space."
            parts = [p.strip().strip("<>") for p in spec.split("+") if p.strip()]
            non_mods = [p for p in parts if p not in modifiers]
            if len(parts) < 2 or len(non_mods) != 1:
                return f"{label}: вкажіть модифікатор + 1 клавішу (напр. ctrl+space)."
            norm = "+".join(parts)
            if norm in seen:
                return f"Гарячі клавіші «{seen[norm]}» та «{label}» не можуть збігатися."
            seen[norm] = label
        return None

    def _on_save_settings_clicked(self) -> None:
        new_cfg = self._collect_config_from_ui()
        hotkey_err = self._validate_hotkey_pair(
            new_cfg.hotkey_answer,
            new_cfg.hotkey_toggle,
            new_cfg.hotkey_screen,
            new_cfg.hotkey_web,
            new_cfg.hotkey_capture,
            new_cfg.hotkey_snip,
        )
        if hotkey_err:
            self.show_settings_error(hotkey_err)
            return

        self._settings_save_rejected = False
        self.settings_saved.emit(new_cfg)
        if self._settings_save_rejected:
            return

        self._config = replace(new_cfg)
        self._draft_keys.clear()
        self._draft_models.clear()
        self._sync_live_web_btn(new_cfg.web_search_enabled)
        self._sync_live_screen_btn(new_cfg.screen_vision_enabled)
        self._hk_answer_pill.setText(self._fmt_hotkey(new_cfg.hotkey_answer))
        self._hk_toggle_pill.setText(self._fmt_hotkey(new_cfg.hotkey_toggle))
        self._hk_capture_pill.setText(self._fmt_hotkey(new_cfg.hotkey_capture))
        self._hk_screen_pill.setText(self._fmt_hotkey(new_cfg.hotkey_screen))
        self._hk_web_pill.setText(self._fmt_hotkey(new_cfg.hotkey_web))
        self._hk_snip_pill.setText(self._fmt_hotkey(new_cfg.hotkey_snip))
        self._refresh_live_toggle_tooltips(new_cfg)
        self._save_status_lbl.setStyleSheet("color: #4ADE80;")
        self._save_status_lbl.setText("Налаштування збережено та застосовано.")
        self.refresh_readiness()

    def _on_save_vocab_clicked(self) -> None:
        vocab_text = self.vocab_edit.text().strip()
        updated_cfg = replace(self._config, custom_vocabulary=vocab_text)
        self._settings_save_rejected = False
        self.settings_saved.emit(updated_cfg)
        if self._settings_save_rejected:
            self.show_docs_status(self._save_status_lbl.text(), is_error=True)
            return
        self._config = updated_cfg
        self.show_docs_status("Словник термінів збережено та застосовано до Whisper і RAG.")

    def _on_save_prompt_clicked(self) -> None:
        prompt_text = self.prompt_edit.toPlainText().strip() or DEFAULT_SYSTEM_PROMPT
        updated_cfg = replace(self._config, system_prompt=prompt_text)
        self._settings_save_rejected = False
        self.settings_saved.emit(updated_cfg)
        if self._settings_save_rejected:
            self.show_docs_status(self._save_status_lbl.text(), is_error=True)
            return
        self._config = updated_cfg
        self.show_docs_status("Системний промпт збережено та застосовано.")

    def show_docs_status(self, message: str, is_error: bool = False) -> None:
        clean = message.strip()
        if not clean:
            self._docs_status_lbl.setVisible(False)
            self._docs_status_lbl.setText("")
            return
        color = "#F87171" if is_error else "#4ADE80"
        self._docs_status_lbl.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 500;")
        self._docs_status_lbl.setText(clean)
        self._docs_status_lbl.setVisible(True)

    def show_settings_error(self, message: str) -> None:
        self._settings_save_rejected = True
        self._save_status_lbl.setStyleSheet("color: #F87171;")
        self._save_status_lbl.setText(message)

    def _reset_prompt_default(self) -> None:
        self.prompt_edit.setPlainText(DEFAULT_SYSTEM_PROMPT)

    def _on_profile_combo_changed(self, _idx: int) -> None:
        pid = self.profile_combo.currentData()
        if isinstance(pid, str) and pid:
            self.profile_selected.emit(pid)

    def _on_profile_lang_changed(self, _idx: int) -> None:
        pid = self.profile_combo.currentData()
        lang_code = self.profile_lang_combo.currentData()
        if isinstance(pid, str) and pid and isinstance(lang_code, str):
            self.profile_language_changed.emit(pid, lang_code)

    def _on_new_profile_clicked(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            "Новий профіль",
            "Назва профілю (напр. БД, Python, Історія):",
        )
        if ok and name.strip():
            self.profile_create_requested.emit(name.strip())

    def _on_rename_profile_clicked(self) -> None:
        pid = self.profile_combo.currentData()
        if not isinstance(pid, str) or not pid:
            return
        cur_name = self.profile_combo.currentText().strip()
        new_name, ok = QInputDialog.getText(
            self,
            "Перейменувати профіль",
            "Нова назва профілю:",
            text=cur_name,
        )
        if ok and new_name.strip() and new_name.strip() != cur_name:
            self.profile_rename_requested.emit(pid, new_name.strip())

    def _on_delete_profile_clicked(self) -> None:
        pid = self.profile_combo.currentData()
        if not isinstance(pid, str) or not pid or self.profile_combo.count() <= 1:
            return
        cur_name = self.profile_combo.currentText().strip()
        reply = QMessageBox.question(
            self,
            "Видалити профіль",
            f"Видалити профіль «{cur_name}»?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.profile_delete_requested.emit(pid)

    def _on_export_profile_clicked(self) -> None:
        export_all = False
        if self.profile_combo.count() > 1:
            box = QMessageBox(self)
            box.setWindowTitle("Експорт профілів")
            box.setText("Експортувати лише поточний профіль чи всі профілі?\n(API-ключі та локальні шляхи файлів не включаються.)")
            box.addButton("Лише поточний", QMessageBox.ButtonRole.AcceptRole)
            btn_all = box.addButton("Усі профілі", QMessageBox.ButtonRole.ActionRole)
            btn_cancel = box.addButton("Скасувати", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            clicked = box.clickedButton()
            if clicked == btn_cancel or clicked is None:
                return
            export_all = clicked == btn_all

        default_name = (
            "ai_copilot_profiles.json"
            if export_all
            else f"profile_{self.profile_combo.currentText().strip() or 'subject'}.json"
        )
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Експортувати профіль(і)",
            default_name,
            "JSON профілі (*.json)",
        )
        if path_str:
            self.profile_export_requested.emit(path_str, export_all)

    def _on_import_profile_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Імпортувати профіль(і)",
            "",
            "JSON профілі (*.json)",
        )
        if path_str:
            self.profile_import_requested.emit(path_str)

    def sync_profile_state(
        self, cfg: AppConfig, profiles: list[PromptProfile], active_id: str
    ) -> None:
        """Synchronize Docs profile switcher, prompt/vocab fields, and Settings STT language."""
        self._config = replace(cfg)
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        for prof in profiles:
            self.profile_combo.addItem(prof.name, prof.id)
        self._select_combo_by_data(self.profile_combo, active_id)
        self.profile_combo.blockSignals(False)
        self._delete_profile_btn.setEnabled(len(profiles) > 1)

        active_prof = next((p for p in profiles if p.id == active_id), None)
        lang_val = active_prof.stt_language if active_prof is not None else cfg.whisper_language
        self.profile_lang_combo.blockSignals(True)
        self._select_combo_by_data(self.profile_lang_combo, lang_val)
        self.profile_lang_combo.blockSignals(False)

        self.vocab_edit.setText(cfg.custom_vocabulary)
        self.prompt_edit.setPlainText(cfg.system_prompt)
        self._select_combo_by_data(self._whisper_lang_combo, cfg.whisper_language)

    # --- Live state updates ---

    def _set_dot(self, dot: QLabel, status: str) -> None:
        dot.setProperty("status", status)
        dot.style().unpolish(dot)
        dot.style().polish(dot)

    def refresh_readiness(self) -> None:
        has_creds = has_valid_llm_credentials(self._config)
        self._setup_banner.setVisible(not has_creds)

        model_name = get_provider_model(self._config)
        repl = get_deprecated_model_replacement(model_name)

        if not has_creds:
            self._set_dot(self._dot_llm, "warn")
            self._lbl_llm.setText(f"{self._config.llm_provider} — потрібен API-ключ")
        elif self._last_check_result is not None:
            res = self._last_check_result
            if res.ok:
                self._set_dot(self._dot_llm, "ok")
                self._lbl_llm.setText(
                    f"{self._config.llm_provider} / {model_name} (ключ перевірено ✓)"
                )
            elif res.key_valid and not res.model_valid:
                self._set_dot(self._dot_llm, "warn")
                self._lbl_llm.setText(
                    f"{self._config.llm_provider} — ключ дійсний, оберіть активну модель"
                )
            else:
                self._set_dot(self._dot_llm, "error")
                self._lbl_llm.setText(
                    f"{self._config.llm_provider} — ключ не працює"
                )
        elif repl is not None:
            self._set_dot(self._dot_llm, "warn")
            self._lbl_llm.setText(
                f"{self._config.llm_provider} / {model_name} (застаріла → {repl})"
            )
        else:
            self._set_dot(self._dot_llm, "ok")
            self._lbl_llm.setText(f"{self._config.llm_provider} / {model_name}")

        if self._config.web_search_enabled:
            mode_lbl = "Агент" if self._config.web_agent_mode == "agent" else "Швидкий"
            self._set_dot(self._dot_web, "ok")
            self._lbl_web.setText(f"Увімкнено ({mode_lbl} · {self._config.web_search_provider})")
        else:
            self._set_dot(self._dot_web, "idle")
            self._lbl_web.setText("Вимкнено (перемикач 🌐 Веб)")

        if self._config.screen_vision_enabled:
            self._set_dot(self._dot_screen, "ok")
            self._lbl_screen.setText("Увімкнено (бачить код/екран)")
        else:
            self._set_dot(self._dot_screen, "idle")
            self._lbl_screen.setText("Вимкнено (перемикач 👁 Екран)")

        if self._config.overlay_stealth:
            self._set_dot(self._dot_stealth, "ok")
            self._lbl_stealth.setText("Увімкнено (невидимий на Screen Share)")
        else:
            self._set_dot(self._dot_stealth, "warn")
            self._lbl_stealth.setText("Вимкнено (видимий при записі екрана)")

    def set_devices(self, devices: Iterable[AudioDeviceInfo]) -> None:
        current_data = self.home_device_combo.currentData()
        self.home_device_combo.blockSignals(True)
        self.home_device_combo.clear()
        dev_list = list(devices)

        if not dev_list:
            self.home_device_combo.addItem("WASAPI Loopback не знайдено", -1)
            self._home_start_btn.setEnabled(False)
            self._live_start_btn.setEnabled(False)
            self._set_dot(self._dot_audio, "error")
            self._lbl_audio.setText("Пристрій не знайдено")
            self.home_device_combo.blockSignals(False)
            return

        self._home_start_btn.setEnabled(True)
        self._live_start_btn.setEnabled(True)
        selected_idx = 0
        for i, dev in enumerate(dev_list):
            prefix = "★ " if dev.is_default else ""
            self.home_device_combo.addItem(f"{prefix}{dev.name}", dev.index)
            if current_data is not None and dev.index == current_data:
                selected_idx = i
        self.home_device_combo.setCurrentIndex(selected_idx)
        self.home_device_combo.blockSignals(False)
        self._update_audio_status_label()

    def _on_home_device_changed(self, _idx: int) -> None:
        self._update_audio_status_label()
        data = self.home_device_combo.currentData()
        if isinstance(data, int) and data >= 0:
            self.device_selected.emit(data)

    def set_selected_device_index(self, device_index: int) -> None:
        idx = self.home_device_combo.findData(device_index)
        if idx >= 0 and idx != self.home_device_combo.currentIndex():
            self.home_device_combo.blockSignals(True)
            self.home_device_combo.setCurrentIndex(idx)
            self.home_device_combo.blockSignals(False)
            self._update_audio_status_label()

    def _update_audio_status_label(self) -> None:
        txt = self.home_device_combo.currentText().lstrip("★ ").strip()
        if self._is_capturing:
            self._set_dot(self._dot_audio, "ok")
            self._lbl_audio.setText(f"Активний: {txt}")
        else:
            self._set_dot(self._dot_audio, "idle")
            self._lbl_audio.setText(f"Готовий: {txt}")

    def _on_toggle_capture_clicked(self) -> None:
        if self._is_capturing:
            self.stop_capture_requested.emit()
        else:
            dev_idx = self.home_device_combo.currentData()
            self.start_capture_requested.emit(int(dev_idx) if dev_idx is not None else -1)

    def set_capturing(self, active: bool) -> None:
        self._is_capturing = bool(active)
        self.home_device_combo.setEnabled(not self._is_capturing)
        self._home_refresh_btn.setEnabled(not self._is_capturing)
        for btn in (self._home_start_btn, self._live_start_btn):
            btn.setText("■ Зупинити сесію" if self._is_capturing else "▶ Почати сесію")
            btn.setProperty("capturing", "true" if self._is_capturing else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        if not self._is_capturing:
            self.home_level_bar.setValue(0)
        self._update_audio_status_label()

    def set_audio_level(self, rms: float) -> None:
        self.home_level_bar.setValue(int(min(100, max(0, rms * 1400))))

    def set_overlay_visible(self, visible: bool) -> None:
        self._overlay_visible = bool(visible)
        text = "Сховати оверлей" if self._overlay_visible else "Показати оверлей"
        self._home_overlay_btn.setText(text)
        self._sb_overlay_btn.setText(text)
        self._live_overlay_btn.setText(text)

    def _confirm_disable_stealth(self, target_name: str) -> bool:
        reply = QMessageBox.warning(
            self,
            "Вимкнути Stealth-режим?",
            f"Увага! Якщо вимкнути Stealth-режим ({target_name}), вікно стане видимим "
            "під час демонстрації або запису екрана (Zoom, Discord, OBS, Teams).\n\n"
            "Після перезапуску програми Stealth-режим знову увімкнеться автоматично.\n\n"
            "Ви справді хочете вимкнути Stealth-режим зараз?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _on_ov_stealth_checkbox_clicked(self, checked: bool) -> None:
        if not checked and not self._confirm_disable_stealth("оверлей"):
            self._ov_stealth_chk.blockSignals(True)
            self._ov_stealth_chk.setChecked(True)
            self._ov_stealth_chk.blockSignals(False)
            return
        self._auto_save_settings_if_valid()

    def _on_dash_stealth_checkbox_clicked(self, checked: bool) -> None:
        if not checked and not self._confirm_disable_stealth("панель керування"):
            self._dash_stealth_chk.blockSignals(True)
            self._dash_stealth_chk.setChecked(True)
            self._dash_stealth_chk.blockSignals(False)
            return
        self._auto_save_settings_if_valid()

    def set_overlay_stealth(self, enabled: bool) -> None:
        """Synchronize overlay stealth state in dashboard config, settings UI, and readiness."""
        state = bool(enabled)
        self._config.overlay_stealth = state
        if self._ov_stealth_chk.isChecked() != state:
            self._ov_stealth_chk.blockSignals(True)
            self._ov_stealth_chk.setChecked(state)
            self._ov_stealth_chk.blockSignals(False)
        self.refresh_readiness()

    def set_dashboard_stealth(self, enabled: bool) -> None:
        """Synchronize dashboard stealth state, DWM affinity, and watchdog timer."""
        state = bool(enabled)
        self._config.dashboard_stealth = state
        if self._dash_stealth_chk.isChecked() != state:
            self._dash_stealth_chk.blockSignals(True)
            self._dash_stealth_chk.setChecked(state)
            self._dash_stealth_chk.blockSignals(False)
        self._apply_dashboard_stealth()
        self.refresh_readiness()

    def set_overlay_opacity(self, percent: int) -> None:
        """Synchronize overlay opacity slider without re-emitting user signals."""
        clamped = max(30, min(100, int(percent)))
        self._config.overlay_opacity = clamped
        if self._ov_opacity_slider.value() != clamped:
            self._ov_opacity_slider.blockSignals(True)
            self._ov_opacity_slider.setValue(clamped)
            self._ov_opacity_slider.blockSignals(False)
        self._ov_opacity_lbl.setText(f"{clamped}%")

    def set_whisper_status(self, text: str) -> None:
        clean = text.replace("Whisper:", "").replace("STT:", "").strip()
        self._lbl_whisper.setText(clean)
        low = text.lower()
        if "помилка" in low or "немає api-ключа" in low:
            self._set_dot(self._dot_whisper, "error")
            self._whisper_progress.setVisible(False)
        elif "завантаження" in low or "ініціалізація" in low or "підключення" in low or "перепідключення" in low:
            self._set_dot(self._dot_whisper, "warn")
            self._whisper_progress.setVisible(True)
        elif "cpu" in low:
            self._set_dot(self._dot_whisper, "warn")
            self._whisper_progress.setVisible(False)
        else:
            self._set_dot(self._dot_whisper, "ok")
            self._whisper_progress.setVisible(False)

    def set_documents(self, docs: list[LoadedDocument], summary_text: str) -> None:
        self.docs_list.setUpdatesEnabled(False)
        try:
            self.docs_list.clear()
            if not docs:
                self._set_dot(self._dot_docs, "idle")
                self._lbl_docs.setText("0 файлів (додайте звіт, код або конспект)")
                placeholder = QListWidgetItem(
                    "Немає завантажених файлів. Перетягніть файли/папку сюди або натисніть «+ Додати файли»."
                )
                placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
                self.docs_list.addItem(placeholder)
                return

            self._set_dot(self._dot_docs, "ok")
            self._lbl_docs.setText(summary_text)
            for doc in docs:
                item = QListWidgetItem(self.docs_list)
                row_w = QWidget(self.docs_list)
                row_l = QHBoxLayout(row_w)
                row_l.setContentsMargins(8, 6, 8, 6)

                display = getattr(doc, "label", doc.name)
                name_lbl = QLabel(f"📄  {display}", row_w)
                name_lbl.setStyleSheet("font-weight: 600; color: #F1F5F9;")
                tok_est = getattr(doc, "token_estimate", max(1, (doc.char_count + 3) // 4))
                page_cnt = getattr(doc, "page_count", 0)
                is_trunc = getattr(doc, "truncated", False)
                tip_parts = [doc.path, f"~{tok_est} токенів"]
                if page_cnt:
                    tip_parts.append(f"{page_cnt} стор.")
                if is_trunc:
                    tip_parts.append("обрізано за лімітом розміру")
                name_lbl.setToolTip("\n".join(tip_parts))
                row_l.addWidget(name_lbl, 1)

                size_lbl = QLabel(f"{doc.char_count:,} симв.".replace(",", " "), row_w)
                size_lbl.setObjectName("MutedText")
                row_l.addWidget(size_lbl)

                rm_btn = QPushButton("Видалити", row_w)
                rm_btn.setObjectName("DangerBtn")
                rm_btn.setStyleSheet("padding: 4px 10px;")
                rm_btn.clicked.connect(
                    lambda _c=False, p=doc.path: self.remove_context_file_requested.emit(p)
                )
                row_l.addWidget(rm_btn)

                item.setSizeHint(row_w.sizeHint())
                self.docs_list.addItem(item)
                self.docs_list.setItemWidget(item, row_w)
        finally:
            self.docs_list.setUpdatesEnabled(True)

    def _on_add_files_clicked(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Оберіть файли контексту (звіт, код, конспект)",
            "",
            build_file_dialog_filter(),
        )
        if paths:
            self.load_context_requested.emit(paths)

    def _on_add_folder_clicked(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Оберіть папку проєкту або матеріалів",
            "",
        )
        if folder:
            self.load_context_requested.emit([folder])

    def set_generating(self, active: bool) -> None:
        self._is_generating = bool(active)
        self._live_answer_btn.setProperty("generating", "true" if active else "false")
        self._live_answer_btn.style().unpolish(self._live_answer_btn)
        self._live_answer_btn.style().polish(self._live_answer_btn)
        self._live_answer_btn.setText("■ Стоп (Esc)" if active else "Відповісти")
        self._live_answer_btn.setToolTip(
            "Зупинити генерацію (Esc)" if active else "Отримати відповідь ШІ"
        )
        self._live_deep_btn.setEnabled(not active)

    def _on_live_answer_btn_clicked(self) -> None:
        if self._is_generating:
            self.cancel_answer_requested.emit()
        else:
            self._emit_live_answer()

    def _emit_live_answer(self) -> None:
        if self._is_generating:
            return
        manual = self.live_manual_input.text().strip()
        cursor = self.live_transcript_edit.textCursor()
        selected = cursor.selectedText().replace("\u2029", "\n").strip()
        self.live_manual_input.clear()
        self.answer_requested.emit(manual, selected)

    def set_qa_history_position(self, current: int, total: int) -> None:
        total = max(0, int(total))
        current = max(0, min(int(current), total))
        self._live_qa_counter_lbl.setText(f"{current}/{total} ▾" if total > 1 else f"{current}/{total}")
        self._live_qa_counter_lbl.setEnabled(total > 1)
        self._live_qa_prev_btn.setEnabled(current > 1)
        self._live_qa_next_btn.setEnabled(0 < current < total)
        self._live_qa_nav.setVisible(total > 1)

    def set_qa_history_items(self, items: list[tuple[str, str]], active_index: int) -> None:
        """Store (timestamp, question) tuples for quick-jump card selection."""
        self._qa_history_nav_items = list(items)
        self._qa_history_active_idx = int(active_index)

    def _on_live_qa_counter_clicked(self) -> None:
        if len(self._qa_history_nav_items) <= 1:
            return
        menu = QMenu(self)
        for idx, (ts, question) in enumerate(self._qa_history_nav_items):
            preview = " ".join((question or "").split())
            if len(preview) > 54:
                preview = preview[:51] + "…"
            act = menu.addAction(f"#{idx + 1} [{ts}] {preview or 'Відповідь'}")
            act.setCheckable(True)
            act.setChecked(idx == self._qa_history_active_idx)
            act.triggered.connect(lambda _chk=False, target=idx: self.qa_jump_requested.emit(target))
        menu.exec(self._live_qa_counter_lbl.mapToGlobal(QPoint(0, self._live_qa_counter_lbl.height())))

    def set_qa_history_entries(self, entries: list[tuple[str, str, str, list[str]]]) -> None:
        """Store formatted (timestamp, question, answer, sources) tuples for Markdown export."""
        self._qa_history_entries = list(entries)

    def _reset_live_copy_btn(self) -> None:
        self._live_copy_btn.setText("📋 Копіювати")

    def _on_live_copy_clicked(self) -> None:
        text = self._last_answer_markdown.strip() or self.live_answer_browser.toPlainText().strip()
        if not text:
            return
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
            self._live_copy_btn.setText("✓ Скопійовано")
            self._copy_reset_timer.start()

    def open_home(self) -> None:
        self._switch_page(0)

    def open_live(self) -> None:
        self._switch_page(1)

    def open_docs(self) -> None:
        self._switch_page(2)

    def open_settings(self, tab_idx: int = 0) -> None:
        self._open_settings_tab(tab_idx)

    def append_transcript(self, timestamp_str: str, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        sb = self.live_transcript_edit.verticalScrollBar()
        was_at_bottom = sb is None or sb.maximum() <= 0 or sb.value() >= sb.maximum() - 24
        self.live_transcript_edit.append(f"[{timestamp_str}] {cleaned}")
        if sb is not None and was_at_bottom:
            sb.setValue(sb.maximum())

    def set_live_sources(self, sources: list[str]) -> None:
        if not sources:
            self._live_sources_lbl.setText("")
            self._live_sources_lbl.setToolTip("")
            return
        preview = ", ".join(sources[:3])
        if len(sources) > 3:
            preview += f" (+{len(sources) - 3})"
        txt = f"Джерела: {preview}"
        self._live_sources_lbl.setText(txt)
        self._live_sources_lbl.setToolTip("Використані джерела:\n" + "\n".join(sources))

    def set_live_answer_markdown(
        self, markdown_text: str, *, force_scroll_top: bool = False
    ) -> None:
        self._last_answer_markdown = markdown_text
        set_browser_markdown_sticky(
            self.live_answer_browser, markdown_text, force_scroll_top=force_scroll_top
        )

    def clear_live_views(self) -> None:
        self._last_answer_markdown = ""
        self._qa_history_entries.clear()
        self.set_generating(False)
        self.set_qa_history_position(0, 0)
        self.set_live_sources([])
        self.live_transcript_edit.clear()
        self.live_answer_browser.clear()

    def _export_session_markdown(self) -> None:
        transcript = self.live_transcript_edit.toPlainText().strip()
        answer = self._last_answer_markdown.strip() or self.live_answer_browser.toPlainText().strip()
        if not transcript and not answer and not self._qa_history_entries:
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Зберегти матеріали сесії",
            "zoom_copilot_session.md",
            "Markdown (*.md);;Текст (*.txt)",
        )
        if not path_str:
            return
        parts = ["# Сесія AI Copilot (Zoom • UA)"]
        if transcript:
            parts.append(f"## Транскрибація\n\n{transcript}")
        if self._qa_history_entries:
            qa_sections: list[str] = []
            for idx, (ts, question, ans, sources) in enumerate(self._qa_history_entries, start=1):
                q_head = f"### {idx}. [{ts}] {question}".strip()
                block = [q_head]
                if sources:
                    block.append("**Джерела:** " + ", ".join(sources))
                block.append(ans.strip())
                qa_sections.append("\n\n".join(block))
            parts.append("## Відповіді ШІ\n\n" + "\n\n---\n\n".join(qa_sections))
        elif answer:
            parts.append(f"## Відповідь ШІ\n\n{answer}")
        try:
            Path(path_str).write_text("\n\n".join(parts) + "\n", encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Помилка збереження",
                f"Не вдалося зберегти файл сесії:\n{exc}",
            )
