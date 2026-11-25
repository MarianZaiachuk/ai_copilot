"""Entry point, dual-window orchestration, single-instance IPC, redacted logging, and hotkeys."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass, field, replace
from datetime import datetime
import io
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import queue
import sys
import threading
import time

from config import (
    AppConfig,
    DEFAULT_SYSTEM_PROMPT,
    ProfileStore,
    PromptProfile,
    get_app_dir,
    get_resource_path,
    load_config,
    parse_custom_vocabulary,
    register_cuda_dlls,
    save_config_to_env,
)
from security import (
    RedactingFormatter,
    SecretRedactingFilter,
    redact_secrets,
)

SINGLE_INSTANCE_SERVER_NAME = "ZoomAICopilot_SingleInstance_v1"
SINGLE_INSTANCE_MUTEX_NAME = "Local\\ZoomAICopilot_SingleInstance_Mutex_v1"
_ERROR_ALREADY_EXISTS = 183
_ERROR_ACCESS_DENIED = 5


class WindowsSingleInstanceGuard:
    """Exclusive named mutex guard preventing duplicate processes/tray icons on Windows."""

    def __init__(self, mutex_name: str = SINGLE_INSTANCE_MUTEX_NAME) -> None:
        self._mutex_name = mutex_name
        self._handle: int | None = None
        self._kernel32 = None
        if sys.platform == "win32":
            try:
                k32 = ctypes.WinDLL("kernel32", use_last_error=True)
                k32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
                k32.CreateMutexW.restype = ctypes.c_void_p
                k32.CloseHandle.argtypes = [ctypes.c_void_p]
                k32.CloseHandle.restype = ctypes.c_int
                self._kernel32 = k32
            except Exception:
                self._kernel32 = None

    def acquire(self) -> bool:
        """Acquire exclusive instance mutex; return False if another instance holds it."""
        if sys.platform != "win32" or self._kernel32 is None:
            return True
        if self._handle is not None:
            return True
        handle = self._kernel32.CreateMutexW(None, False, self._mutex_name)
        err = ctypes.get_last_error()
        if not handle:
            return err not in (_ERROR_ALREADY_EXISTS, _ERROR_ACCESS_DENIED)
        if err == _ERROR_ALREADY_EXISTS:
            self._kernel32.CloseHandle(handle)
            return False
        self._handle = int(handle)
        return True

    def release(self) -> None:
        """Release mutex handle cleanly on shutdown."""
        if self._handle is not None and self._kernel32 is not None:
            try:
                self._kernel32.CloseHandle(ctypes.c_void_p(self._handle))
            except Exception:
                pass
            self._handle = None


def _get_runtime_data_dir() -> Path:
    """Return writable directory for logs and runtime state (AppDir or %APPDATA%)."""
    app_dir = get_app_dir()
    if os.access(app_dir, os.W_OK):
        return app_dir
    appdata_dir = Path(os.getenv("APPDATA", str(Path.home()))) / "ZoomAICopilot"
    appdata_dir.mkdir(parents=True, exist_ok=True)
    return appdata_dir


class _LoggingStdStream(io.TextIOBase):
    """Redirect frozen None stdout/stderr writes into logger without holding a second file handle."""

    def __init__(self, target_logger: logging.Logger, level: int = logging.INFO) -> None:
        super().__init__()
        self._logger = target_logger
        self._level = level
        self._lock = threading.Lock()
        self._buf = ""

    @property
    def encoding(self) -> str:
        return "utf-8"

    @property
    def errors(self) -> str:
        return "replace"

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not text:
            return 0
        s = str(text)
        with self._lock:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                cleaned = line.rstrip("\r")
                if cleaned:
                    self._logger.log(self._level, "%s", redact_secrets(cleaned))
        return len(s)

    def flush(self) -> None:
        with self._lock:
            cleaned = self._buf.rstrip("\r\n")
            self._buf = ""
        if cleaned:
            self._logger.log(self._level, "%s", redact_secrets(cleaned))


def _setup_logging_and_streams() -> None:
    """Configure rotating secret-redacting logging and guard None stdout/stderr in .exe."""
    data_dir = _get_runtime_data_dir()
    log_path = data_dir / "copilot.log"

    had_stderr = sys.stderr is not None
    formatter = RedactingFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    secret_filter = SecretRedactingFilter()

    handlers: list[logging.Handler] = []
    for candidate in (
        log_path,
        Path(os.getenv("APPDATA", str(Path.home()))) / "ZoomAICopilot" / f"copilot_{os.getpid()}.log",
    ):
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                candidate,
                maxBytes=2_000_000,
                backupCount=3,
                encoding="utf-8",
                delay=True,
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(secret_filter)
            handlers.append(file_handler)
            break
        except OSError:
            continue

    if had_stderr and sys.stderr is not None:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(secret_filter)
        handlers.append(stream_handler)

    if not handlers:
        handlers.append(logging.NullHandler())

    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        force=True,
    )

    if sys.stdout is None:
        sys.stdout = _LoggingStdStream(logging.getLogger("stdio.out"), logging.INFO)  # type: ignore[assignment]
    if sys.stderr is None:
        sys.stderr = _LoggingStdStream(logging.getLogger("stdio.err"), logging.WARNING)  # type: ignore[assignment]


_setup_logging_and_streams()

# Register CUDA/cuDNN DLL paths BEFORE importing transcriber / faster_whisper.
register_cuda_dlls()

from PyQt6.QtCore import QCoreApplication, QObject, Qt, QThread, QTimer, pyqtSignal  # noqa: E402
from PyQt6.QtGui import QAction, QActionGroup, QGuiApplication, QIcon  # noqa: E402
from PyQt6.QtNetwork import QLocalServer, QLocalSocket  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon  # noqa: E402
from pynput import keyboard  # noqa: E402

from audio_capture import WasapiLoopbackCapturer, list_loopback_devices  # noqa: E402
from context_loader import ContextManager, expand_context_paths  # noqa: E402
from dashboard import DashboardWindow  # noqa: E402
from llm_client import ApiKeyCheckResult, LLMClient  # noqa: E402
from screen_capture import capture_screen, list_screens  # noqa: E402
from transcriber import (  # noqa: E402
    CloudSTTTranscriber,
    DeepgramRealtimeClient,
    WhisperTranscriber,
    verify_stt_provider,
)
from ui import CopilotOverlayWindow, apply_window_display_affinity  # noqa: E402
from web_search import search_web  # noqa: E402

logger = logging.getLogger(__name__)

_SPECIAL_VK_MAP: dict[str, int] = {
    "space": 0x20,
    "enter": 0x0D,
    "return": 0x0D,
    "tab": 0x09,
    "esc": 0x1B,
    "escape": 0x1B,
    "backspace": 0x08,
    "insert": 0x2D,
    "delete": 0x2E,
    "home": 0x24,
    "end": 0x23,
    "page_up": 0x21,
    "page_down": 0x22,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    **{f"f{i}": 0x6F + i for i in range(1, 13)},
}

_MODIFIER_ALIASES: dict[str, str] = {
    "ctrl": "ctrl",
    "control": "ctrl",
    "shift": "shift",
    "alt": "alt",
    "win": "win",
    "cmd": "win",
    "super": "win",
}

_WIN32_MODIFIER_VKS: dict[str, tuple[int, ...]] = {
    "ctrl": (0x11, 0xA2, 0xA3),
    "shift": (0x10, 0xA0, 0xA1),
    "alt": (0x12, 0xA4, 0xA5),
    "win": (0x5B, 0x5C),
}


def is_win32_vk_down(vk: int) -> bool:
    """Query physical key state on Windows to prune stuck modifiers after focus switches."""
    if sys.platform != "win32":
        return True
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)
    except Exception:
        return True


def parse_hotkey_spec(spec: str) -> tuple[frozenset[str], int]:
    """Parse a hotkey string (e.g. 'ctrl+shift+h') into (modifiers, Windows VK code)."""
    tokens = [tok.strip().strip("<>").lower() for tok in spec.split("+") if tok.strip()]
    if not tokens:
        raise ValueError("Порожнє визначення гарячої клавіші")

    modifiers: set[str] = set()
    trigger_vk: int | None = None

    for tok in tokens:
        if tok in _MODIFIER_ALIASES:
            modifiers.add(_MODIFIER_ALIASES[tok])
            continue
        if tok in _SPECIAL_VK_MAP:
            vk = _SPECIAL_VK_MAP[tok]
        elif len(tok) == 1 and tok.isascii() and tok.isalnum():
            vk = ord(tok.upper())
        else:
            raise ValueError(f"Непідтримувана клавіша '{tok}' у '{spec}'")

        if trigger_vk is not None:
            raise ValueError(f"Можна вказати лише одну основну клавішу у '{spec}'")
        trigger_vk = vk

    if trigger_vk is None:
        raise ValueError(f"Не вказано основну клавішу у '{spec}'")

    return frozenset(modifiers), trigger_vk


def format_hotkey_label(spec: str) -> str:
    """Format a hotkey spec cleanly for UI labels (e.g. 'Ctrl+Space')."""
    parts = [tok.strip().strip("<>").capitalize() for tok in spec.split("+") if tok.strip()]
    return "+".join(parts) if parts else spec


class GlobalHotkeyBridge(QObject):
    """Global keyboard listener matching Windows VK codes (works in UA/EN layouts)."""

    answer_hotkey_pressed = pyqtSignal()
    toggle_hotkey_pressed = pyqtSignal()

    def __init__(
        self,
        answer_spec: str,
        toggle_spec: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        ans = parse_hotkey_spec(answer_spec)
        tog = parse_hotkey_spec(toggle_spec)
        if ans == tog:
            raise ValueError("Гарячі клавіші відповіді та приховування не можуть збігатися")
        self._answer_combo = ans
        self._toggle_combo = tog

        self._lock = threading.Lock()
        self._pressed_modifiers: set[str] = set()
        self._fired_vks: set[int] = set()
        self._listener: keyboard.Listener | None = None

    def update_hotkeys(self, answer_spec: str, toggle_spec: str) -> None:
        new_ans = parse_hotkey_spec(answer_spec)
        new_tog = parse_hotkey_spec(toggle_spec)
        if new_ans == new_tog:
            raise ValueError("Гарячі клавіші відповіді та приховування не можуть збігатися")
        with self._lock:
            self._answer_combo = new_ans
            self._toggle_combo = new_tog
            self._fired_vks.clear()

    def start(self) -> None:
        if self._listener is not None:
            return
        try:
            self._listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self._listener.daemon = True
            self._listener.start()
        except Exception as exc:
            logger.warning("Failed to start global hotkey listener: %s", exc)
            self._listener = None

    def stop(self) -> None:
        listener = self._listener
        self._listener = None
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass
            try:
                listener.join(timeout=1.0)
            except Exception:
                pass

    @staticmethod
    def _classify_modifier(key) -> str | None:
        if not isinstance(key, keyboard.Key):
            return None
        if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            return "ctrl"
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            return "shift"
        if key in (
            keyboard.Key.alt,
            keyboard.Key.alt_l,
            keyboard.Key.alt_r,
            keyboard.Key.alt_gr,
        ):
            return "alt"
        if key in tuple(
            k
            for k in (
                getattr(keyboard.Key, "cmd", None),
                getattr(keyboard.Key, "cmd_l", None),
                getattr(keyboard.Key, "cmd_r", None),
            )
            if k is not None
        ):
            return "win"
        return None

    @staticmethod
    def _extract_vk(key) -> int | None:
        if isinstance(key, keyboard.Key):
            special = {
                keyboard.Key.space: 0x20,
                keyboard.Key.enter: 0x0D,
                keyboard.Key.tab: 0x09,
                keyboard.Key.esc: 0x1B,
                keyboard.Key.backspace: 0x08,
                keyboard.Key.left: 0x25,
                keyboard.Key.up: 0x26,
                keyboard.Key.right: 0x27,
                keyboard.Key.down: 0x28,
            }
            if key in special:
                return special[key]

        vk = getattr(key, "vk", None)
        if isinstance(vk, int):
            return vk

        val = getattr(key, "value", None)
        val_vk = getattr(val, "vk", None)
        if isinstance(val_vk, int):
            return val_vk

        char = getattr(key, "char", None)
        if isinstance(char, str) and len(char) == 1 and char.isascii() and char.isalnum():
            return ord(char.upper())

        return None

    def _prune_stale_modifiers_locked(self) -> None:
        if self._listener is None or sys.platform != "win32":
            return
        for mod in list(self._pressed_modifiers):
            vks = _WIN32_MODIFIER_VKS.get(mod)
            if vks and not any(is_win32_vk_down(vk) for vk in vks):
                self._pressed_modifiers.discard(mod)
        for mod, vks in _WIN32_MODIFIER_VKS.items():
            if mod in self._pressed_modifiers:
                continue
            if any(is_win32_vk_down(vk) for vk in vks):
                self._pressed_modifiers.add(mod)

    def _on_press(self, key) -> None:
        mod = self._classify_modifier(key)
        pending_signal = None
        with self._lock:
            if mod is not None:
                self._pressed_modifiers.add(mod)
                return

            self._prune_stale_modifiers_locked()
            vk = self._extract_vk(key)
            if vk is None or vk in self._fired_vks:
                return

            current_mods = frozenset(self._pressed_modifiers)
            if (current_mods, vk) == self._answer_combo:
                self._fired_vks.add(vk)
                pending_signal = self.answer_hotkey_pressed
            elif (current_mods, vk) == self._toggle_combo:
                self._fired_vks.add(vk)
                pending_signal = self.toggle_hotkey_pressed

        if pending_signal is not None:
            pending_signal.emit()

    def _on_release(self, key) -> None:
        mod = self._classify_modifier(key)
        with self._lock:
            if mod is not None:
                vks = _WIN32_MODIFIER_VKS.get(mod)
                still_down = (
                    self._listener is not None
                    and sys.platform == "win32"
                    and bool(vks)
                    and any(is_win32_vk_down(vk) for vk in vks)
                )
                if not still_down:
                    self._pressed_modifiers.discard(mod)
                    self._fired_vks.clear()
                return

            vk = self._extract_vk(key)
            if vk is not None:
                self._fired_vks.discard(vk)


class _QueueStopSentinel:
    """Marker pushed onto audio queues to stop a TranscriberThread."""


class AudioCaptureThread(QThread):
    """Background thread capturing WASAPI loopback chunks/PCM and emitting audio RMS level."""

    level_updated = pyqtSignal(float)
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    flush_completed = pyqtSignal(bool)

    def __init__(
        self,
        config: AppConfig,
        device_index: int | None,
        chunk_queue: queue.Queue,
        raw_pcm_queue: queue.Queue | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._device_index = device_index
        self._chunk_queue = chunk_queue
        self._raw_pcm_queue = raw_pcm_queue
        self._stop_event = threading.Event()
        self._flush_event = threading.Event()
        self._capturer: WasapiLoopbackCapturer | None = None

    def stop(self) -> None:
        self._stop_event.set()
        self._flush_event.set()

    def request_flush(self) -> bool:
        """Ask the capture loop to flush any pending speech tail immediately."""
        had_pending = bool(self._capturer is not None and self._capturer.has_pending_speech)
        self._flush_event.set()
        return had_pending

    def _enqueue_chunk_meta(self, chunk, has_overlap: bool) -> None:
        payload = (chunk, bool(has_overlap))
        try:
            self._chunk_queue.put_nowait(payload)
        except queue.Full:
            try:
                old = self._chunk_queue.get_nowait()
                if old is None or isinstance(old, _QueueStopSentinel):
                    self._chunk_queue.put_nowait(old)
                    return
                logger.warning("Audio chunk queue full; dropped oldest un-transcribed chunk")
            except queue.Empty:
                pass
            try:
                self._chunk_queue.put_nowait(payload)
            except queue.Full:
                pass

    def _enqueue_pcm16(self, chunk_16k) -> None:
        if self._raw_pcm_queue is None:
            return
        import numpy as np

        pcm16_bytes = (np.clip(chunk_16k, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        if not pcm16_bytes:
            return
        try:
            self._raw_pcm_queue.put_nowait(pcm16_bytes)
        except queue.Full:
            try:
                old = self._raw_pcm_queue.get_nowait()
                if old is not None and not isinstance(old, (bytes, bytearray)):
                    # Preserve non-audio control sentinel if shutdown raced.
                    self._raw_pcm_queue.put_nowait(old)
                    return
            except queue.Empty:
                pass
            try:
                self._raw_pcm_queue.put_nowait(pcm16_bytes)
            except queue.Full:
                pass

    def run(self) -> None:
        capturer = WasapiLoopbackCapturer(
            device_index=self._device_index,
            chunk_seconds=self._config.audio_chunk_seconds,
            overlap_seconds=self._config.audio_overlap_seconds,
            silence_seconds=self._config.audio_silence_seconds,
            silence_threshold=self._config.audio_silence_threshold,
        )
        use_raw_pcm = (
            (self._config.stt_provider or "local").strip().lower() == "deepgram"
            and self._raw_pcm_queue is not None
        )
        self._capturer = capturer
        try:
            if use_raw_pcm:
                capturer.run_loop(
                    stop_event=self._stop_event,
                    on_raw_16k=self._enqueue_pcm16,
                    on_level=self.level_updated.emit,
                    on_status=self.status_updated.emit,
                )
            else:
                capturer.run_loop(
                    stop_event=self._stop_event,
                    on_chunk_meta=self._enqueue_chunk_meta,
                    on_level=self.level_updated.emit,
                    on_status=self.status_updated.emit,
                    flush_event=self._flush_event,
                    on_flushed=lambda emitted: (
                        self.flush_completed.emit(bool(emitted))
                        if not self._stop_event.is_set()
                        else None
                    ),
                )
        except Exception as exc:
            logger.exception("Audio capture error")
            self.error_occurred.emit(f"Помилка аудіозахоплення: {exc}")
        finally:
            self._capturer = None
            if use_raw_pcm and self._raw_pcm_queue is not None:
                try:
                    self._raw_pcm_queue.put_nowait(None)
                except queue.Full:
                    try:
                        self._raw_pcm_queue.get_nowait()
                        self._raw_pcm_queue.put_nowait(None)
                    except Exception:
                        pass


class TranscriberThread(QThread):
    """Background worker orchestrating local Whisper, Deepgram WS streaming, or Cloud STT."""

    model_status = pyqtSignal(str)
    transcript_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    chunk_processed = pyqtSignal()

    def __init__(
        self,
        config: AppConfig,
        chunk_queue: queue.Queue,
        initial_vocab: list[str] | None = None,
        raw_pcm_queue: queue.Queue | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._chunk_queue = chunk_queue
        self._raw_pcm_queue = raw_pcm_queue
        self._stop_event = threading.Event()
        self._busy_event = threading.Event()
        self._stop_sentinel = _QueueStopSentinel()
        self._initial_vocab = list(initial_vocab or [])
        self._provider = (config.stt_provider or "local").strip().lower()
        self._transcriber = WhisperTranscriber(
            config=config,
            on_status=self.model_status.emit,
        )
        if self._initial_vocab:
            self._transcriber.set_vocabulary(self._initial_vocab)
        self._active_cloud: CloudSTTTranscriber | None = None
        self._active_deepgram: DeepgramRealtimeClient | None = None

    @property
    def is_busy(self) -> bool:
        """Return True while a chunk is actively being decoded."""
        return self._busy_event.is_set()

    def set_vocabulary(self, terms: list[str]) -> None:
        self._initial_vocab = list(terms)
        self._transcriber.set_vocabulary(terms)
        if self._active_cloud is not None:
            self._active_cloud.set_vocabulary(terms)
        if self._active_deepgram is not None:
            self._active_deepgram.set_vocabulary(terms)

    def update_runtime_config(self, new_config: AppConfig) -> None:
        self._config = new_config
        self._transcriber.update_config(new_config)
        if self._active_cloud is not None:
            self._active_cloud.update_config(new_config)
        if self._active_deepgram is not None:
            self._active_deepgram.update_config(new_config)

    def reset_history(self) -> None:
        self._transcriber.reset_history()
        if self._active_cloud is not None:
            self._active_cloud.reset_history()

    def _push_sentinel(self, target_queue: queue.Queue | None) -> None:
        if target_queue is None:
            return
        while True:
            try:
                target_queue.put_nowait(self._stop_sentinel)
                return
            except queue.Full:
                try:
                    target_queue.get_nowait()
                except queue.Empty:
                    return

    def stop(self) -> None:
        self._stop_event.set()
        if self._provider == "deepgram" and self._raw_pcm_queue is not None:
            self._push_sentinel(self._raw_pcm_queue)
        else:
            self._push_sentinel(self._chunk_queue)

    def _run_chunked_recognizer(self, recognizer) -> None:
        import numpy as np

        while not self._stop_event.is_set():
            try:
                item = self._chunk_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is self._stop_sentinel or item is None:
                break
            if isinstance(item, _QueueStopSentinel):
                continue

            if isinstance(item, tuple) and len(item) == 2:
                audio, has_overlap = item
            else:
                audio, has_overlap = item, True

            if not isinstance(audio, np.ndarray):
                logger.debug(
                    "Skipping non-ndarray item from chunk queue: %s",
                    type(audio).__name__,
                )
                continue

            self._busy_event.set()
            try:
                text = recognizer.transcribe(audio, has_overlap=bool(has_overlap))
                if text and not self._stop_event.is_set():
                    self.transcript_ready.emit(text)
            except Exception as exc:
                logger.exception("Transcription error")
                if not self._stop_event.is_set():
                    self.error_occurred.emit(f"Помилка транскрибації: {exc}")
            finally:
                self._busy_event.clear()
                if not self._stop_event.is_set():
                    self.chunk_processed.emit()

    def run(self) -> None:
        prov = self._provider
        if prov == "deepgram" and self._raw_pcm_queue is not None:
            dg = DeepgramRealtimeClient(self._config)
            if self._initial_vocab:
                dg.set_vocabulary(self._initial_vocab)
            self._active_deepgram = dg
            self.model_status.emit(dg.status_label())
            try:
                dg.run_session(
                    self._raw_pcm_queue,
                    self._stop_event,
                    on_transcript=lambda txt: (
                        self.transcript_ready.emit(txt)
                        if txt and not self._stop_event.is_set()
                        else None
                    ),
                    on_status=lambda msg: (
                        self.model_status.emit(msg)
                        if not self._stop_event.is_set()
                        else None
                    ),
                    on_error=lambda err: (
                        self.error_occurred.emit(err)
                        if not self._stop_event.is_set()
                        else None
                    ),
                    stop_sentinel=self._stop_sentinel,
                )
            finally:
                self._active_deepgram = None
            return

        if prov in {"groq", "elevenlabs"}:
            cloud = CloudSTTTranscriber(self._config)
            if self._initial_vocab:
                cloud.set_vocabulary(self._initial_vocab)
            self._active_cloud = cloud
            self.model_status.emit(cloud.status_label())
            try:
                self._run_chunked_recognizer(cloud)
            finally:
                self._active_cloud = None
                try:
                    cloud.unload_model()
                except Exception:
                    pass
            return

        try:
            try:
                device, compute = self._transcriber.load_model()
                self.model_status.emit(
                    f"Whisper: {self._config.whisper_model} ({device.upper()} {compute})"
                )
            except Exception as exc:
                logger.exception("Failed to initialize Whisper model")
                self.model_status.emit("Whisper: помилка завантаження")
                self.error_occurred.emit(f"Не вдалося завантажити Whisper: {exc}")
                return

            self._run_chunked_recognizer(self._transcriber)
        finally:
            try:
                self._transcriber.unload_model()
            except Exception:
                pass


class ContextLoaderThread(QThread):
    """Loads, expands, or reloads documents/folders in background to keep UI responsive."""

    completed = pyqtSignal(int, str, list, list)

    def __init__(
        self,
        context_manager: ContextManager,
        gen_id: int,
        mode: str,
        paths: list[str] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._context_manager = context_manager
        self._gen_id = gen_id
        self._mode = mode
        self._paths = list(paths or [])

    def run(self) -> None:
        try:
            if self._mode == "reload":
                reloaded, errors = self._context_manager.reload_modified_files()
                self.completed.emit(self._gen_id, self._mode, reloaded, errors)
            else:
                if self._mode == "replace":
                    self._context_manager.clear()
                if self._paths:
                    expanded, notices = expand_context_paths(self._paths)
                    loaded, errors = self._context_manager.add_files(expanded)
                    all_errors = list(notices) + list(errors)
                    self.completed.emit(self._gen_id, self._mode, loaded, all_errors)
                else:
                    self.completed.emit(self._gen_id, self._mode, [], [])
        except Exception as exc:
            logger.exception("Context loading failed")
            self.completed.emit(self._gen_id, self._mode, [], [f"Помилка читання файлів: {exc}"])


class LLMWorkerThread(QThread):
    """Streams LLM responses asynchronously without blocking the PyQt6 event loop."""

    chunk_received = pyqtSignal(int, str)
    status_updated = pyqtSignal(int, str)
    sources_updated = pyqtSignal(int, list)
    model_used = pyqtSignal(int, str, str)
    answer_finished = pyqtSignal(int, float)
    error_occurred = pyqtSignal(int, str)

    def __init__(
        self,
        request_id: int,
        llm_client: LLMClient,
        transcript_text: str,
        context_text: str,
        manual_question: str,
        detail_mode: str = "concise",
        history: tuple[tuple[str, str], ...] = (),
        context_revision: str = "",
        bypass_cache: bool = False,
        screen_image_bytes: bytes | None = None,
        screen_image_mime: str = "image/jpeg",
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.request_id = request_id
        self._llm_client = llm_client
        self._transcript_text = transcript_text
        self._context_text = context_text
        self._manual_question = manual_question
        self._detail_mode = detail_mode
        self._history = history
        self._context_revision = context_revision
        self._bypass_cache = bypass_cache
        self._screen_image_bytes = screen_image_bytes
        self._screen_image_mime = screen_image_mime
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def _emit_status(self, msg: str) -> None:
        if not self._cancel_event.is_set():
            self.status_updated.emit(self.request_id, msg)

    def _emit_sources(self, srcs: list[str]) -> None:
        if not self._cancel_event.is_set():
            self.sources_updated.emit(self.request_id, srcs)

    def _emit_model_used(self, actual_model: str, route_tag: str) -> None:
        if not self._cancel_event.is_set():
            self.model_used.emit(self.request_id, actual_model, route_tag)

    def run(self) -> None:
        t0 = time.monotonic()
        try:
            for chunk in self._llm_client.stream_answer(
                transcript_text=self._transcript_text,
                context_text=self._context_text,
                manual_question=self._manual_question,
                detail_mode=self._detail_mode,
                history=self._history,
                context_revision=self._context_revision,
                bypass_cache=self._bypass_cache,
                cancel_event=self._cancel_event,
                on_status=self._emit_status,
                on_sources=self._emit_sources,
                on_model_used=self._emit_model_used,
                screen_image_bytes=self._screen_image_bytes,
                screen_image_mime=self._screen_image_mime,
            ):
                if self._cancel_event.is_set():
                    break
                self.chunk_received.emit(self.request_id, chunk)
            if not self._cancel_event.is_set():
                self.answer_finished.emit(self.request_id, time.monotonic() - t0)
        except Exception as exc:
            if not self._cancel_event.is_set():
                logger.exception("LLM streaming error")
                self.error_occurred.emit(self.request_id, str(exc))


class WebSearchTestThread(QThread):
    """Background connectivity probe for the configured web search engine."""

    completed = pyqtSignal(bool, str)

    def __init__(self, config: AppConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._config = config

    def run(self) -> None:
        from urllib.parse import urlparse

        try:
            results = search_web(
                "Python programming language",
                provider=self._config.web_search_provider,
                api_key=self._config.web_search_api_key,
                max_results=3,
            )
            if not results:
                self.completed.emit(
                    False,
                    "Результатів не знайдено — перевірте інтернет-з'єднання або інший рушій.",
                )
                return
            domains: list[str] = []
            for r in results:
                host = (urlparse(r.url).hostname or "").removeprefix("www.")
                if host and host not in domains:
                    domains.append(host)
            self.completed.emit(
                True,
                f"Пошук працює ✓ Знайдено {len(results)} рез. ({', '.join(domains[:3])})",
            )
        except Exception as exc:
            msg = redact_secrets(
                f"Помилка пошуку: {exc}",
                extra_secrets=(self._config.web_search_api_key,),
            )
            self.completed.emit(False, msg)


class STTTestThread(QThread):
    """Background connectivity probe for configured local or cloud STT engine."""

    completed = pyqtSignal(bool, str)

    def __init__(self, config: AppConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._config = config

    def run(self) -> None:
        try:
            ok, msg = verify_stt_provider(self._config)
            self.completed.emit(ok, msg)
        except Exception as exc:
            safe_msg = redact_secrets(
                f"Помилка перевірки STT: {exc}",
                extra_secrets=(self._config.stt_api_key, self._config.openai_api_key),
            )
            self.completed.emit(False, safe_msg)


class LLMTestThread(QThread):
    """Asynchronous API key & model verification probe for Settings/Home."""

    completed = pyqtSignal(object)

    def __init__(self, config: AppConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._config = config

    def run(self) -> None:
        client = LLMClient(self._config)
        try:
            result = client.verify_api_key()
            self.completed.emit(result)
        except Exception as exc:
            logger.exception("API key verification thread error")
            self.completed.emit(
                ApiKeyCheckResult(
                    ok=False,
                    key_valid=False,
                    model_valid=False,
                    status="error",
                    message=client._redact(f"Помилка перевірки: {exc}"),
                )
            )


class LLMModelsThread(QThread):
    """Fetches active models for the entered API key/endpoint."""

    models_loaded = pyqtSignal(list, str)

    def __init__(self, config: AppConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._config = config

    def run(self) -> None:
        client = LLMClient(self._config)
        try:
            models = client.list_available_models()
            self.models_loaded.emit(models, "")
        except Exception as exc:
            self.models_loaded.emit([], client._redact(f"Не вдалося завантажити моделі: {exc}"))


def _safe_is_running(thread: QThread | None) -> bool:
    if thread is None:
        return False
    try:
        return thread.isRunning()
    except RuntimeError:
        return False


def compute_request_context_budget(
    config: AppConfig, detail_mode: str = "concise"
) -> tuple[int, int]:
    """Compute tight per-request (max_chars, full_context_threshold) for fast, low-TPM RAG."""
    stored_budget = max(2000, min(config.context_max_chars, config.rag_context_budget_chars))
    provider = config.llm_provider.lower()
    if provider == "groq":
        if detail_mode == "detailed":
            max_chars = min(stored_budget, 10000)
            threshold = min(max_chars, 7500)
        else:
            max_chars = min(stored_budget, 6500)
            threshold = min(max_chars, 5000)
        return max_chars, threshold

    if detail_mode == "detailed":
        max_chars = min(config.context_max_chars, max(stored_budget, 18000))
        threshold = min(max_chars, 12000)
    else:
        max_chars = min(stored_budget, 12000)
        threshold = min(max_chars, 9000)
    return max_chars, threshold


@dataclass
class QAEntry:
    """Single session Question/Answer record supporting navigation, follow-ups, and export."""

    timestamp: str
    question: str
    transcript_excerpt: str
    manual_question: str
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    provenance: str = ""
    detail_mode: str = "concise"


class CopilotController(QObject):
    """Coordinates Dashboard, Stealth Overlay, Tray Icon, Audio, Whisper STT, and LLM."""

    def __init__(
        self,
        config: AppConfig,
        dashboard: DashboardWindow,
        overlay: CopilotOverlayWindow,
        app_icon: QIcon | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.dashboard = dashboard
        self.overlay = overlay
        self.context_manager = ContextManager()

        legacy_paths = self._read_legacy_context_paths()
        self._profile_store = ProfileStore.load_or_init(
            self._profiles_file(), config, legacy_context_paths=legacy_paths
        )
        active_prof = self._profile_store.active_profile()
        effective_lang = active_prof.stt_language or config.whisper_language
        effective_cfg = replace(
            config,
            system_prompt=active_prof.system_prompt,
            custom_vocabulary=active_prof.custom_vocabulary,
            whisper_language=effective_lang,
        )
        if effective_cfg != config:
            save_config_to_env(effective_cfg)
        self.config = effective_cfg
        self.llm_client = LLMClient(effective_cfg)

        self._chunk_queue: queue.Queue = queue.Queue(maxsize=10)
        self._raw_pcm_queue: queue.Queue = queue.Queue(maxsize=120)
        self._transcript_history: list[tuple[float, str]] = []
        self._qa_history: list[QAEntry] = []
        self._qa_index = -1
        self._active_request_id = 0
        self._active_actual_model = ""
        self._active_route_tag = "primary"
        self._context_revision = 0
        self._context_gen_id = 0
        self._pending_context_job: tuple[str, list[str]] | None = None
        self._last_transcript_excerpt = ""
        self._last_manual_question = ""
        self._is_capturing = False
        self._shutting_down = False
        self._tray_notified = False

        self._live_answer_buffer = ""
        self._live_answer_dirty = False
        self._current_doc_sources: list[str] = []
        self._current_all_sources: list[str] = []
        self._live_render_timer = QTimer(self)
        self._live_render_timer.setInterval(60)
        self._live_render_timer.timeout.connect(self._flush_dashboard_answer)

        self._opacity_save_timer = QTimer(self)
        self._opacity_save_timer.setSingleShot(True)
        self._opacity_save_timer.setInterval(300)
        self._opacity_save_timer.timeout.connect(self._persist_overlay_opacity)

        self._pending_answer_req: tuple[int, str, str, str] | None = None
        self._pending_answer_seq = 0
        self._waiting_audio_flush = False
        self._chunks_since_trigger = 0
        self._flush_wait_timer = QTimer(self)
        self._flush_wait_timer.setSingleShot(True)
        self._flush_wait_timer.timeout.connect(self._on_flush_wait_timeout)

        self._audio_thread: AudioCaptureThread | None = None
        self._llm_thread: LLMWorkerThread | None = None
        self._retired_llm_threads: list[LLMWorkerThread] = []
        self._retired_transcriber_threads: list[TranscriberThread] = []
        self._context_thread: ContextLoaderThread | None = None
        self._test_thread: LLMTestThread | None = None
        self._models_thread: LLMModelsThread | None = None
        self._web_test_thread: WebSearchTestThread | None = None
        self._stt_test_thread: STTTestThread | None = None

        self._transcriber_thread = self._create_transcriber_thread(self.config)

        self._hotkey_bridge = GlobalHotkeyBridge(
            answer_spec=self.config.hotkey_answer,
            toggle_spec=self.config.hotkey_toggle,
            parent=self,
        )
        self._hotkey_bridge.answer_hotkey_pressed.connect(self._on_answer_hotkey)
        self._hotkey_bridge.toggle_hotkey_pressed.connect(self.overlay.toggle_visibility)

        self._tray_icon: QSystemTrayIcon | None = None
        self._tray_menu: QMenu | None = None
        self._tray_capture_action: QAction | None = None
        self._tray_profiles_menu: QMenu | None = None
        self._tray_profiles_group: QActionGroup | None = None
        self._tray_web_action: QAction | None = None
        self._tray_screen_action: QAction | None = None
        self._tray_stealth_action: QAction | None = None
        self._setup_tray(app_icon)

        self._connect_signals()
        self.dashboard.sync_profile_state(
            self.config, self._profile_store.profiles, active_prof.id
        )
        self.overlay.set_llm_status(self.llm_client.provider_display_name())
        self.dashboard.set_documents([], self.context_manager.summary())
        self.dashboard.set_screens(list_screens())
        self.refresh_devices()

    def _merged_vocabulary(self) -> list[str]:
        doc_terms = self.context_manager.extract_vocabulary_terms()
        user_terms = parse_custom_vocabulary(self.config.custom_vocabulary)
        seen: set[str] = set()
        merged: list[str] = []
        for term in user_terms + doc_terms:
            key = term.lower()
            if key not in seen:
                seen.add(key)
                merged.append(term)
        return merged

    def _create_transcriber_thread(self, cfg: AppConfig) -> TranscriberThread:
        thread = TranscriberThread(
            cfg,
            self._chunk_queue,
            initial_vocab=self._merged_vocabulary(),
            raw_pcm_queue=self._raw_pcm_queue,
            parent=self,
        )
        thread.model_status.connect(self._on_whisper_status)
        thread.transcript_ready.connect(self._on_transcript_ready)
        thread.error_occurred.connect(self._on_worker_error)
        thread.chunk_processed.connect(self._on_transcriber_chunk_processed)
        return thread

    def _setup_tray(self, app_icon: QIcon | None) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self.dashboard.set_hide_to_tray(False)
            return

        icon = (
            app_icon
            if (app_icon is not None and not app_icon.isNull())
            else self.dashboard.windowIcon()
        )
        tray = QSystemTrayIcon(icon, self)
        tray.setToolTip("AI Copilot • Zoom (UA)")

        menu = QMenu()
        act_dash = QAction("Відкрити панель керування", menu)
        act_dash.triggered.connect(self.show_dashboard)
        menu.addAction(act_dash)

        act_ov = QAction("Показати / сховати оверлей", menu)
        act_ov.triggered.connect(self.overlay.toggle_visibility)
        menu.addAction(act_ov)

        self._tray_capture_action = QAction("▶ Почати запис", menu)
        self._tray_capture_action.triggered.connect(self._toggle_capture_from_tray)
        menu.addAction(self._tray_capture_action)

        menu.addSeparator()

        self._tray_profiles_menu = QMenu("📚 Профіль", menu)
        menu.addMenu(self._tray_profiles_menu)
        self._rebuild_tray_profiles_menu()

        menu.addSeparator()

        self._tray_web_action = QAction("🌐 Веб-пошук", menu)
        self._tray_web_action.setCheckable(True)
        self._tray_web_action.setChecked(self.config.web_search_enabled)
        self._tray_web_action.toggled.connect(self.set_web_search_enabled)
        menu.addAction(self._tray_web_action)

        self._tray_screen_action = QAction("👁 Зір екрана", menu)
        self._tray_screen_action.setCheckable(True)
        self._tray_screen_action.setChecked(self.config.screen_vision_enabled)
        self._tray_screen_action.toggled.connect(self.set_screen_vision_enabled)
        menu.addAction(self._tray_screen_action)

        self._tray_stealth_action = QAction(menu)
        self._tray_stealth_action.setCheckable(False)
        self._tray_stealth_action.setEnabled(False)
        menu.addAction(self._tray_stealth_action)

        menu.addSeparator()
        act_quit = QAction("Вихід", menu)
        act_quit.triggered.connect(self.request_quit)
        menu.addAction(act_quit)

        self._tray_menu = menu
        tray.setContextMenu(menu)
        tray.activated.connect(self._on_tray_activated)
        tray.show()
        self._tray_icon = tray
        self.dashboard.set_hide_to_tray(True)
        self._sync_tray_check_states()

    def _sync_tray_check_states(self) -> None:
        pairs = (
            (self._tray_web_action, self.config.web_search_enabled),
            (self._tray_screen_action, self.config.screen_vision_enabled),
        )
        for action, checked in pairs:
            if action is not None and action.isChecked() != bool(checked):
                action.blockSignals(True)
                action.setChecked(bool(checked))
                action.blockSignals(False)
        if self._tray_stealth_action is not None:
            if self.config.overlay_stealth:
                self._tray_stealth_action.setText("◈ Stealth оверлею: УВІМКНЕНО")
            else:
                self._tray_stealth_action.setText("⚠ Stealth оверлею: ВИМКНЕНО (Налаштування)")

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.show_dashboard()

    def _toggle_capture_from_tray(self) -> None:
        if self._is_capturing:
            self.stop_capture()
        else:
            dev_idx = self.dashboard.home_device_combo.currentData()
            self.start_capture(int(dev_idx) if dev_idx is not None else -1)

    def _on_screens_changed(self, _screen=None) -> None:
        if self._shutting_down:
            return
        self.dashboard.set_screens(list_screens())
        self.overlay.clamp_to_available_screen()
        self.overlay.set_stealth_mode(self.config.overlay_stealth)
        self.dashboard.set_dashboard_stealth(self.config.dashboard_stealth)

    def _connect_signals(self) -> None:
        app = QApplication.instance()
        if isinstance(app, QGuiApplication):
            app.screenAdded.connect(self._on_screens_changed)
            app.screenRemoved.connect(self._on_screens_changed)

        # Overlay signals
        self.overlay.refresh_devices_requested.connect(self.refresh_devices)
        self.overlay.start_capture_requested.connect(self.start_capture)
        self.overlay.stop_capture_requested.connect(self.stop_capture)
        self.overlay.load_context_requested.connect(self.load_context_files)
        self.overlay.clear_context_requested.connect(self.clear_context)
        self.overlay.clear_all_requested.connect(self.clear_history)
        self.overlay.answer_requested.connect(self.trigger_answer)
        self.overlay.cancel_answer_requested.connect(self.cancel_answer)
        self.overlay.deep_answer_requested.connect(self.trigger_deep_answer)
        self.overlay.qa_nav_requested.connect(self.navigate_qa)
        self.overlay.device_selected.connect(self.dashboard.set_selected_device_index)
        self.overlay.opacity_changed.connect(self._on_overlay_opacity_changed)
        self.overlay.web_search_toggled.connect(self.set_web_search_enabled)
        self.overlay.screen_vision_toggled.connect(self.set_screen_vision_enabled)
        self.overlay.open_dashboard_requested.connect(self.show_dashboard)
        self.overlay.visibility_changed.connect(self.dashboard.set_overlay_visible)

        # Dashboard signals
        self.dashboard.refresh_devices_requested.connect(self.refresh_devices)
        self.dashboard.start_capture_requested.connect(self.start_capture)
        self.dashboard.stop_capture_requested.connect(self.stop_capture)
        self.dashboard.toggle_overlay_requested.connect(self.overlay.toggle_visibility)
        self.dashboard.load_context_requested.connect(self.load_context_files)
        self.dashboard.reload_context_requested.connect(self.reload_context_files)
        self.dashboard.remove_context_file_requested.connect(self.remove_context_file)
        self.dashboard.clear_context_requested.connect(self.clear_context)
        self.dashboard.clear_all_requested.connect(self.clear_history)
        self.dashboard.answer_requested.connect(self.trigger_answer)
        self.dashboard.cancel_answer_requested.connect(self.cancel_answer)
        self.dashboard.deep_answer_requested.connect(self.trigger_deep_answer)
        self.dashboard.qa_nav_requested.connect(self.navigate_qa)
        self.dashboard.device_selected.connect(self.overlay.set_selected_device_index)
        self.dashboard.web_search_toggled.connect(self.set_web_search_enabled)
        self.dashboard.screen_vision_toggled.connect(self.set_screen_vision_enabled)
        self.dashboard.settings_saved.connect(self.apply_new_settings)
        self.dashboard.test_llm_requested.connect(self.test_llm_connection)
        self.dashboard.fetch_models_requested.connect(self.fetch_llm_models)
        self.dashboard.test_stt_requested.connect(self.test_stt)
        self.dashboard.test_web_search_requested.connect(self.test_web_search)
        self.dashboard.test_screen_capture_requested.connect(self.test_screen_capture)
        self.dashboard.profile_selected.connect(self.switch_profile)
        self.dashboard.profile_create_requested.connect(self.create_profile)
        self.dashboard.profile_rename_requested.connect(self.rename_profile)
        self.dashboard.profile_delete_requested.connect(self.delete_profile)
        self.dashboard.profile_language_changed.connect(self.update_profile_language)
        self.dashboard.quit_requested.connect(self.request_quit)
        self.dashboard.minimized_to_tray.connect(self._on_dashboard_minimized)

    def _on_overlay_stealth_toggled(self, enabled: bool) -> None:
        state = bool(enabled)
        self.config.overlay_stealth = state
        self.overlay.set_stealth_mode(state)
        self.dashboard.set_overlay_stealth(state)
        self._sync_tray_check_states()
        save_config_to_env(self.config)

    def _on_overlay_opacity_changed(self, value: int) -> None:
        clamped = max(30, min(100, int(value)))
        self.config.overlay_opacity = clamped
        self.dashboard.set_overlay_opacity(clamped)
        self._opacity_save_timer.start()

    def _persist_overlay_opacity(self) -> None:
        if not self._shutting_down:
            save_config_to_env(self.config)

    def set_web_search_enabled(self, enabled: bool) -> None:
        """Toggle live web search across Overlay, Dashboard, Tray, LLMClient, and .env."""
        state = bool(enabled)
        self.config.web_search_enabled = state
        self.llm_client.update_config(self.config)
        save_config_to_env(self.config)
        self.overlay.set_web_search_enabled(state)
        self.dashboard.set_web_search_enabled(state)
        self._sync_tray_check_states()

    def set_screen_vision_enabled(self, enabled: bool) -> None:
        """Toggle live screen vision across Overlay, Dashboard, Tray, LLMClient, and .env."""
        state = bool(enabled)
        self.config.screen_vision_enabled = state
        self.llm_client.update_config(self.config)
        save_config_to_env(self.config)
        self.overlay.set_screen_vision_enabled(state)
        self.dashboard.set_screen_vision_enabled(state)
        self._sync_tray_check_states()

    def test_screen_capture(self, screen_index: int, max_width: int = -1) -> None:
        """Capture a test screenshot and display resolution, size, and latency."""
        effective_width = max_width if max_width >= 400 else self.config.screen_capture_max_width
        t0 = time.monotonic()
        shot = capture_screen(
            screen_index=screen_index,
            max_width=effective_width,
        )
        elapsed_ms = int(round((time.monotonic() - t0) * 1000))
        if shot is None or not shot.image_bytes:
            self.dashboard.set_screen_test_result(
                False,
                "Не вдалося захопити екран. Перевірте вибір монітора.",
            )
            return
        size_kb = max(1, len(shot.image_bytes) // 1024)
        self.dashboard.set_screen_test_result(
            True,
            f"Знімок ОК ✓ {shot.width}×{shot.height} · {size_kb} КБ · {elapsed_ms} мс",
        )

    def _on_dashboard_minimized(self) -> None:
        if self._tray_icon is not None and not self._tray_notified:
            self._tray_notified = True
            self._tray_icon.showMessage(
                "AI Copilot працює у фоні",
                f"Натисніть {format_hotkey_label(self.config.hotkey_toggle)}, щоб показати або сховати оверлей.",
                QSystemTrayIcon.MessageIcon.Information,
                2500,
            )

    def show_dashboard(self) -> None:
        state = (self.dashboard.windowState() & ~Qt.WindowState.WindowMinimized) | Qt.WindowState.WindowActive
        self.dashboard.setWindowState(state)
        self.dashboard.showNormal()
        self.dashboard.raise_()
        self.dashboard.activateWindow()
        hwnd = int(self.dashboard.winId()) if self.dashboard.winId() else 0
        apply_window_display_affinity(hwnd, self.config.dashboard_stealth)

    def start(self) -> None:
        """Start Whisper model worker, global keyboard listener, and restore saved documents."""
        self._transcriber_thread.start()
        self._hotkey_bridge.start()
        self._restore_saved_context_sources()

    def refresh_devices(self) -> None:
        devices = list_loopback_devices()
        self.overlay.set_devices(devices)
        self.dashboard.set_devices(devices)

    def start_capture(self, device_index: int) -> None:
        if self._shutting_down or _safe_is_running(self._audio_thread):
            return

        for q in (self._chunk_queue, self._raw_pcm_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        target_index = device_index if device_index >= 0 else None
        worker = AudioCaptureThread(
            config=self.config,
            device_index=target_index,
            chunk_queue=self._chunk_queue,
            raw_pcm_queue=self._raw_pcm_queue,
            parent=self,
        )
        worker.level_updated.connect(self._on_audio_level)
        worker.status_updated.connect(self._on_audio_status)
        worker.error_occurred.connect(self._on_audio_error)
        worker.flush_completed.connect(self._on_audio_flush_completed)
        worker.finished.connect(lambda w=worker: self._on_audio_finished(w))
        worker.finished.connect(worker.deleteLater)
        self._audio_thread = worker
        worker.start()

        self._set_capturing_state(True)
        if not self.overlay.isVisible():
            self.overlay.showNormal()
            self.overlay.raise_()
            self.overlay.set_stealth_mode(self.overlay.stealth_enabled)

    def stop_capture(self) -> None:
        if _safe_is_running(self._audio_thread):
            assert self._audio_thread is not None
            self._audio_thread.stop()
            self._audio_thread.wait(3000)
        self._set_capturing_state(False)

    def _set_capturing_state(self, active: bool) -> None:
        self._is_capturing = bool(active)
        self.overlay.set_capturing(self._is_capturing)
        self.dashboard.set_capturing(self._is_capturing)
        if self._tray_capture_action is not None:
            self._tray_capture_action.setText(
                "■ Зупинити запис" if self._is_capturing else "▶ Почати запис"
            )

    def _on_audio_level(self, rms: float) -> None:
        if not self._shutting_down:
            self.overlay.set_audio_level(rms)
            self.dashboard.set_audio_level(rms)

    def _on_audio_status(self, message: str) -> None:
        if not self._shutting_down:
            self.overlay.set_whisper_status(message)
            self.dashboard.set_whisper_status(message)

    def _on_audio_finished(self, worker: AudioCaptureThread) -> None:
        if self._audio_thread is worker:
            self._audio_thread = None
        if not self._shutting_down:
            self._set_capturing_state(False)

    def _on_audio_error(self, message: str) -> None:
        if self._shutting_down:
            return
        self._set_capturing_state(False)
        self.overlay.show_answer_error(message)
        self.dashboard.set_live_answer_markdown(f"**Помилка:**\n\n- {message}")

    def _on_whisper_status(self, text: str) -> None:
        if not self._shutting_down:
            self.overlay.set_whisper_status(text)
            self.dashboard.set_whisper_status(text)

    def _on_worker_error(self, message: str) -> None:
        if not self._shutting_down:
            self.overlay.show_answer_error(message)
            self.dashboard.set_live_answer_markdown(f"**Помилка:**\n\n- {message}")

    def _on_transcript_ready(self, text: str) -> None:
        if self._shutting_down:
            return
        now = time.monotonic()
        self._transcript_history.append((now, text))
        if len(self._transcript_history) > 500:
            self._transcript_history = self._transcript_history[-500:]
        ts_label = datetime.now().strftime("%H:%M:%S")
        self.overlay.append_transcript(ts_label, text)
        self.dashboard.append_transcript(ts_label, text)

    def _sync_context_ui(self) -> None:
        docs = self.context_manager.documents
        summary = self.context_manager.summary()
        self.overlay.set_context_summary(summary, bool(docs))
        self.dashboard.set_documents(docs, summary)
        self._transcriber_thread.set_vocabulary(self._merged_vocabulary())

    @staticmethod
    def _context_sources_file() -> Path:
        return _get_runtime_data_dir() / "context_sources.json"

    @staticmethod
    def _profiles_file() -> Path:
        return _get_runtime_data_dir() / "prompt_profiles.json"

    def _read_legacy_context_paths(self) -> list[str]:
        src_file = self._context_sources_file()
        if not src_file.is_file():
            return []
        try:
            data = json.loads(src_file.read_text(encoding="utf-8"))
            raw_paths = data.get("paths", []) if isinstance(data, dict) else []
            return [str(p) for p in raw_paths if isinstance(p, str) and p.strip()]
        except Exception as exc:
            logger.debug("Failed to read context_sources.json: %s", exc)
            return []

    def _rebuild_tray_profiles_menu(self) -> None:
        if self._tray_profiles_menu is None:
            return
        active = self._profile_store.active_profile()
        self._tray_profiles_menu.setTitle(f"📚 Профіль: {active.name}")
        self._tray_profiles_menu.clear()
        group = QActionGroup(self._tray_profiles_menu)
        group.setExclusive(True)
        for prof in self._profile_store.profiles:
            act = QAction(prof.name, self._tray_profiles_menu)
            act.setCheckable(True)
            act.setChecked(prof.id == active.id)
            act.triggered.connect(lambda _chk, pid=prof.id: self.switch_profile(pid))
            group.addAction(act)
            self._tray_profiles_menu.addAction(act)
        self._tray_profiles_group = group

    def _flush_active_profile_drafts(self) -> PromptProfile:
        cur = self._profile_store.active_profile()
        draft_prompt = self.dashboard.prompt_edit.toPlainText().strip() or DEFAULT_SYSTEM_PROMPT
        draft_vocab = self.dashboard.vocab_edit.text().strip()
        if draft_prompt != cur.system_prompt or draft_vocab != cur.custom_vocabulary:
            self._profile_store.update_profile(
                cur.id,
                system_prompt=draft_prompt,
                custom_vocabulary=draft_vocab,
            )
        return self._profile_store.active_profile()

    def switch_profile(self, profile_id: str) -> None:
        """Switch active subject profile, STT language, vocabulary, and workspace docs."""
        if self._shutting_down:
            return
        cur = self._profile_store.active_profile()
        if profile_id == cur.id:
            self._rebuild_tray_profiles_menu()
            return

        self._flush_active_profile_drafts()
        prof = self._profile_store.set_active(profile_id)
        if prof is None:
            return

        next_lang = prof.stt_language if prof.stt_language else self.config.whisper_language
        updated_cfg = replace(
            self.config,
            system_prompt=prof.system_prompt,
            custom_vocabulary=prof.custom_vocabulary,
            whisper_language=next_lang,
        )
        self.apply_new_settings(updated_cfg, sync_active_profile=False)
        self._transcriber_thread.reset_history()
        self.dashboard.sync_profile_state(
            self.config, self._profile_store.profiles, prof.id
        )
        self._rebuild_tray_profiles_menu()

        existing = [p for p in prof.context_paths if Path(p).exists()]
        self._start_context_worker("replace", existing)
        self.dashboard.show_docs_status(
            f"Активовано профіль «{prof.name}» (мова STT: {self.config.whisper_language or 'auto'})."
        )

    def create_profile(self, name: str) -> None:
        """Create a new empty profile starting from the user's preferred Ukrainian defaults."""
        if self._shutting_down:
            return
        self._flush_active_profile_drafts()
        prof = self._profile_store.create_profile(
            name=name,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            stt_language="uk",
            custom_vocabulary="",
            context_paths=[],
        )
        updated_cfg = replace(
            self.config,
            system_prompt=prof.system_prompt,
            custom_vocabulary=prof.custom_vocabulary,
            whisper_language=prof.stt_language or self.config.whisper_language,
        )
        self.apply_new_settings(updated_cfg, sync_active_profile=False)
        self._transcriber_thread.reset_history()
        self.dashboard.sync_profile_state(
            self.config, self._profile_store.profiles, prof.id
        )
        self._rebuild_tray_profiles_menu()
        self._start_context_worker("replace", [])
        self.dashboard.show_docs_status(f"Створено профіль «{prof.name}».")

    def rename_profile(self, profile_id: str, new_name: str) -> None:
        if self._shutting_down:
            return
        updated = self._profile_store.update_profile(profile_id, name=new_name)
        if updated is None:
            return
        active = self._profile_store.active_profile()
        self.dashboard.sync_profile_state(
            self.config, self._profile_store.profiles, active.id
        )
        self._rebuild_tray_profiles_menu()
        self.dashboard.show_docs_status(f"Профіль перейменовано на «{updated.name}».")

    def delete_profile(self, profile_id: str) -> None:
        if self._shutting_down:
            return
        fallback = self._profile_store.delete_profile(profile_id)
        if fallback is None:
            self.dashboard.show_docs_status("Не можна видалити останній профіль.", is_error=True)
            return
        next_lang = (
            fallback.stt_language if fallback.stt_language else self.config.whisper_language
        )
        updated_cfg = replace(
            self.config,
            system_prompt=fallback.system_prompt,
            custom_vocabulary=fallback.custom_vocabulary,
            whisper_language=next_lang,
        )
        self.apply_new_settings(updated_cfg, sync_active_profile=False)
        self._transcriber_thread.reset_history()
        self.dashboard.sync_profile_state(
            self.config, self._profile_store.profiles, fallback.id
        )
        self._rebuild_tray_profiles_menu()
        existing = [p for p in fallback.context_paths if Path(p).exists()]
        self._start_context_worker("replace", existing)
        self.dashboard.show_docs_status(f"Профіль видалено. Активний: «{fallback.name}».")

    def update_profile_language(self, profile_id: str, lang_code: str) -> None:
        if self._shutting_down:
            return
        prof = self._profile_store.update_profile(profile_id, stt_language=lang_code)
        if prof is None:
            return
        active = self._profile_store.active_profile()
        if prof.id == active.id and prof.stt_language:
            updated_cfg = replace(self.config, whisper_language=prof.stt_language)
            self.apply_new_settings(updated_cfg, sync_active_profile=False)
        self.dashboard.sync_profile_state(
            self.config, self._profile_store.profiles, active.id
        )
        self.dashboard.show_docs_status(
            f"Мову STT профілю «{prof.name}» оновлено: {lang_code or 'не змінювати'}."
        )

    def _save_context_sources(self, *, update_active_profile: bool = True) -> None:
        paths = [doc.path for doc in self.context_manager.documents]
        if update_active_profile:
            active = self._profile_store.active_profile()
            self._profile_store.update_profile(active.id, context_paths=paths)
        target = self._context_sources_file()
        tmp = target.with_suffix(".json.tmp")
        payload = {"paths": paths}
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, target)
        except OSError as exc:
            logger.debug("Failed to persist context_sources.json: %s", exc)

    def _restore_saved_context_sources(self) -> None:
        active = self._profile_store.active_profile()
        existing = [str(p) for p in active.context_paths if isinstance(p, str) and Path(p).exists()]
        if existing:
            self._start_context_worker("restore", existing)

    def _start_context_worker(self, mode: str, paths: list[str] | None = None) -> None:
        if self._shutting_down:
            return
        if _safe_is_running(self._context_thread):
            self._pending_context_job = (mode, list(paths or []))
            return
        self._context_gen_id += 1
        gen_id = self._context_gen_id
        self.dashboard.show_docs_status(
            "Оновлення змінених файлів..." if mode == "reload" else "Читання та індексація документів..."
        )
        worker = ContextLoaderThread(
            self.context_manager,
            gen_id=gen_id,
            mode=mode,
            paths=paths,
            parent=self,
        )
        worker.completed.connect(self._on_context_loaded)
        worker.finished.connect(lambda w=worker: self._cleanup_context_thread(w))
        worker.finished.connect(worker.deleteLater)
        self._context_thread = worker
        worker.start()

    def _cleanup_context_thread(self, worker: ContextLoaderThread) -> None:
        if self._context_thread is worker:
            self._context_thread = None
        if not self._shutting_down and self._pending_context_job is not None:
            mode, paths = self._pending_context_job
            self._pending_context_job = None
            self._start_context_worker(mode, paths)

    def _on_context_loaded(self, gen_id: int, mode: str, loaded: list, errors: list[str]) -> None:
        if self._shutting_down or gen_id != self._context_gen_id:
            return
        try:
            self._context_revision += 1
            self._sync_context_ui()
            self._save_context_sources(update_active_profile=(mode == "add"))

            if errors:
                preview = "; ".join(errors[:3])
                suffix = f" (+{len(errors) - 3})" if len(errors) > 3 else ""
                if loaded:
                    self.dashboard.show_docs_status(
                        f"Завантажено {len(loaded)} файл(ів), але з попередженнями: {preview}{suffix}",
                        is_error=True,
                    )
                elif mode in {"restore", "replace"}:
                    self.dashboard.show_docs_status("", is_error=False)
                else:
                    self.dashboard.show_docs_status(f"Помилка завантаження: {preview}{suffix}", is_error=True)
            else:
                if mode == "reload":
                    msg = (
                        f"Оновлено {len(loaded)} змінених файлів."
                        if loaded
                        else "Усі завантажені файли актуальні."
                    )
                    self.dashboard.show_docs_status(msg)
                elif loaded:
                    self.dashboard.show_docs_status(f"Готово: проіндексовано {len(loaded)} файл(ів).")
                elif mode != "replace":
                    self.dashboard.show_docs_status("Підтримуваних файлів не знайдено.", is_error=True)
        except Exception as exc:
            logger.exception("Failed to update context UI after loading files")
            self.dashboard.show_docs_status(f"Помилка оновлення списку документів: {exc}", is_error=True)

    def load_context_files(self, paths: list[str]) -> None:
        if paths and not self._shutting_down:
            self._start_context_worker("add", paths)

    def reload_context_files(self) -> None:
        if not self._shutting_down:
            self._start_context_worker("reload")

    def remove_context_file(self, path: str) -> None:
        self.context_manager.remove_file(path)
        self._context_revision += 1
        self._sync_context_ui()
        self._save_context_sources()

    def clear_context(self) -> None:
        self._pending_context_job = None
        self.context_manager.clear()
        self._context_revision += 1
        self._sync_context_ui()
        self._save_context_sources()
        self.dashboard.show_docs_status("Список документів очищено.")

    def _sync_qa_nav(self) -> None:
        total = len(self._qa_history)
        current = self._qa_index + 1 if 0 <= self._qa_index < total else 0
        self.overlay.set_qa_history_position(current, total)
        self.dashboard.set_qa_history_position(current, total)
        export_rows = [
            (entry.timestamp, entry.question, entry.answer, list(entry.sources))
            for entry in self._qa_history
            if entry.answer.strip()
        ]
        self.dashboard.set_qa_history_entries(export_rows)

    def navigate_qa(self, delta: int) -> None:
        total = len(self._qa_history)
        if total <= 1:
            return
        cur = self._qa_index if 0 <= self._qa_index < total else total - 1
        nxt = max(0, min(total - 1, cur + delta))
        if nxt == cur:
            return
        self._qa_index = nxt
        entry = self._qa_history[nxt]
        self.overlay.set_answer_markdown_history(entry.answer)
        self.overlay.set_answer_sources(entry.sources)
        self.overlay.set_answer_provenance(entry.provenance or f"Історія {nxt + 1}/{total}")
        self.dashboard.set_live_answer_markdown(entry.answer, force_scroll_top=True)
        self.dashboard.set_live_sources(entry.sources)
        self._sync_qa_nav()

    def clear_history(self) -> None:
        self._clear_pending_answer_flush()
        self._active_request_id += 1
        self._live_render_timer.stop()
        if _safe_is_running(self._llm_thread):
            assert self._llm_thread is not None
            self._llm_thread.cancel()
            self._retired_llm_threads.append(self._llm_thread)
        self._transcript_history.clear()
        self._qa_history.clear()
        self._qa_index = -1
        self._last_transcript_excerpt = ""
        self._last_manual_question = ""
        self._transcriber_thread.reset_history()
        self.overlay.clear_transcript_and_answer()
        self.dashboard.clear_live_views()
        self._sync_qa_nav()

    def cancel_answer(self) -> None:
        if self._shutting_down:
            return
        if self._pending_answer_req is not None:
            self._clear_pending_answer_flush()
            self.overlay.cancel_answer_stream()
            self.overlay.set_answer_provenance("Зупинено користувачем")
            self.dashboard.set_generating(False)
            return
        if not _safe_is_running(self._llm_thread):
            self.overlay.cancel_answer_stream()
            self.dashboard.set_generating(False)
            return

        assert self._llm_thread is not None
        self._active_request_id += 1
        self._live_render_timer.stop()
        self._llm_thread.cancel()
        self._retired_llm_threads.append(self._llm_thread)

        partial = self._live_answer_buffer.strip()
        if 0 <= self._qa_index < len(self._qa_history):
            entry = self._qa_history[self._qa_index]
            if partial:
                entry.answer = f"{partial}\n\n_(генерацію зупинено)_"
                entry.provenance = "Зупинено користувачем"
                self.dashboard.set_live_answer_markdown(entry.answer)
            elif not entry.answer:
                self._qa_history.pop(self._qa_index)
                self._qa_index = len(self._qa_history) - 1

        self.overlay.cancel_answer_stream()
        self.overlay.set_answer_provenance("Зупинено користувачем")
        self.dashboard.set_generating(False)
        self._sync_qa_nav()

    def _collect_recent_transcript(self, selected_text: str) -> str:
        if selected_text.strip():
            return selected_text.strip()

        if not self._transcript_history:
            return ""

        cutoff = time.monotonic() - self.config.transcript_window_seconds
        recent = [text for ts, text in self._transcript_history if ts >= cutoff]
        if not recent:
            recent = [text for _ts, text in self._transcript_history[-5:]]
        return "\n".join(recent)

    def _on_answer_hotkey(self) -> None:
        if self._shutting_down:
            return
        manual = self.overlay.manual_input.text().strip()
        selected = self.overlay.get_selected_transcript()
        self.overlay.manual_input.clear()
        was_hidden = not self.overlay.isVisible()
        self.trigger_answer(manual, selected)
        if was_hidden:
            self.overlay.showNormal()
            self.overlay.raise_()
            self.overlay.set_stealth_mode(self.overlay.stealth_enabled)

    def trigger_deep_answer(self) -> None:
        """Re-run the latest or selected question in detailed mode."""
        selected = self.overlay.get_selected_transcript() or self._last_transcript_excerpt
        manual = self.overlay.manual_input.text().strip() or self._last_manual_question
        self.overlay.manual_input.clear()
        self.trigger_answer(manual, selected, detail_mode="detailed")

    def _recent_qa_turns(self) -> tuple[tuple[str, str], ...]:
        completed = [
            (entry.question, entry.answer)
            for entry in self._qa_history
            if entry.question.strip() and entry.answer.strip()
        ]
        return tuple(completed[-2:])

    def _clear_pending_answer_flush(self) -> None:
        self._flush_wait_timer.stop()
        self._pending_answer_req = None
        self._waiting_audio_flush = False
        self._chunks_since_trigger = 0

    def _dispatch_pending_answer(self) -> None:
        pending = self._pending_answer_req
        self._clear_pending_answer_flush()
        if pending is None or self._shutting_down:
            return
        _seq, manual_q, selected_txt, detail_mode = pending
        self._execute_answer(manual_q, selected_txt, detail_mode=detail_mode)

    def _on_audio_flush_completed(self, emitted_chunk: bool) -> None:
        if self._pending_answer_req is None:
            return
        self._waiting_audio_flush = False
        if (
            self._chunk_queue.empty()
            and not self._transcriber_thread.is_busy
            and (not emitted_chunk or self._chunks_since_trigger > 0)
        ):
            self._dispatch_pending_answer()

    def _on_transcriber_chunk_processed(self) -> None:
        if self._pending_answer_req is None:
            return
        self._chunks_since_trigger += 1
        if (
            not self._waiting_audio_flush
            and self._chunk_queue.empty()
            and not self._transcriber_thread.is_busy
        ):
            self._dispatch_pending_answer()

    def _on_flush_wait_timeout(self) -> None:
        if self._pending_answer_req is not None:
            self._dispatch_pending_answer()

    def trigger_answer(
        self,
        manual_question: str,
        selected_transcript: str,
        detail_mode: str = "concise",
    ) -> None:
        if self._shutting_down:
            return
        self._clear_pending_answer_flush()

        prov = (self.config.stt_provider or "local").strip().lower()
        if (
            manual_question.strip()
            or selected_transcript.strip()
            or not self._is_capturing
            or prov == "deepgram"
            or not _safe_is_running(self._audio_thread)
        ):
            self._execute_answer(manual_question, selected_transcript, detail_mode=detail_mode)
            return

        assert self._audio_thread is not None
        had_buf = self._audio_thread.request_flush()
        queued_or_busy = (not self._chunk_queue.empty()) or self._transcriber_thread.is_busy
        if not had_buf and not queued_or_busy:
            self._execute_answer(manual_question, selected_transcript, detail_mode=detail_mode)
            return

        self._pending_answer_seq += 1
        self._pending_answer_req = (
            self._pending_answer_seq,
            manual_question,
            selected_transcript,
            detail_mode,
        )
        self._waiting_audio_flush = had_buf
        self._chunks_since_trigger = 0
        self.overlay.set_answer_provenance("⚡ Дорозпізнаю кінець фрази...")
        self._flush_wait_timer.start(1100)

    def _execute_answer(
        self,
        manual_question: str,
        selected_transcript: str,
        detail_mode: str = "concise",
    ) -> None:
        if self._shutting_down:
            return

        transcript_excerpt = self._collect_recent_transcript(selected_transcript)
        manual_clean = manual_question.strip()

        if not transcript_excerpt and not manual_clean and not self.config.screen_vision_enabled:
            hint = (
                "Немає мовлення чи запитання. Натисніть «Старт» для прослуховування, "
                "введіть питання вручну або увімкніть «👁 Екран»."
            )
            self.overlay.show_answer_error(hint)
            self.overlay.set_answer_provenance("Немає вхідного тексту")
            self.dashboard.set_live_answer_markdown(f"**Підказка:**\n\n- {hint}")
            return

        self._last_transcript_excerpt = transcript_excerpt
        self._last_manual_question = manual_clean

        if manual_clean and transcript_excerpt:
            rag_query = f"{manual_clean}\n{manual_clean}\n{transcript_excerpt}"
        else:
            rag_query = manual_clean or transcript_excerpt

        ctx_max_chars, ctx_full_threshold = compute_request_context_budget(
            self.config, detail_mode=detail_mode
        )
        context_text, sources = self.context_manager.retrieve_context(
            query=rag_query,
            max_chars=ctx_max_chars,
            full_context_threshold=ctx_full_threshold,
        )

        screen_bytes: bytes | None = None
        screen_mime = "image/jpeg"
        if self.config.screen_vision_enabled:
            shot = capture_screen(
                screen_index=self.config.screen_capture_index,
                max_width=self.config.screen_capture_max_width,
            )
            if shot is not None and shot.image_bytes:
                screen_bytes = shot.image_bytes
                screen_mime = shot.mime_type

        doc_sources = list(sources)
        if screen_bytes is not None and "👁 Екран" not in doc_sources:
            doc_sources.append("👁 Екран")
        self._current_doc_sources = doc_sources
        self._current_all_sources = list(doc_sources)
        self.dashboard.set_live_sources(doc_sources)
        self.overlay.set_answer_sources(doc_sources)

        prior_history = self._recent_qa_turns()

        if _safe_is_running(self._llm_thread):
            assert self._llm_thread is not None
            self._llm_thread.cancel()
            self._retired_llm_threads.append(self._llm_thread)

        self._active_request_id += 1
        req_id = self._active_request_id
        self._active_actual_model = ""
        self._active_route_tag = "primary"

        display_question = manual_clean or transcript_excerpt or "[Аналіз екрана]"
        if len(display_question) > 220:
            display_question = display_question[:217] + "…"
        entry = QAEntry(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            question=display_question,
            transcript_excerpt=transcript_excerpt,
            manual_question=manual_clean,
            sources=list(doc_sources),
            detail_mode=detail_mode,
        )
        self._qa_history.append(entry)
        if len(self._qa_history) > 50:
            self._qa_history = self._qa_history[-50:]
        self._qa_index = len(self._qa_history) - 1
        self._sync_qa_nav()

        self._live_answer_buffer = ""
        self._live_answer_dirty = False
        self._live_render_timer.start()
        self.dashboard.set_live_answer_markdown("")
        self.dashboard.set_generating(True)

        self.overlay.begin_answer_stream()
        mode_tag = "детально" if detail_mode == "detailed" else "тезисно"
        web_tag = " · 🌐 Веб" if self.config.web_search_enabled else ""
        screen_tag = " · 👁 Екран" if screen_bytes is not None else ""
        self.overlay.set_answer_provenance(
            f"Генерація ({mode_tag}, {self.llm_client.provider_display_name()}{web_tag}{screen_tag})..."
        )

        worker = LLMWorkerThread(
            request_id=req_id,
            llm_client=self.llm_client,
            transcript_text=transcript_excerpt,
            context_text=context_text,
            manual_question=manual_clean,
            detail_mode=detail_mode,
            history=prior_history,
            context_revision=f"r{self._context_revision}:{ctx_max_chars}",
            bypass_cache=(detail_mode == "detailed"),
            screen_image_bytes=screen_bytes,
            screen_image_mime=screen_mime,
            parent=self,
        )
        worker.chunk_received.connect(self._on_llm_chunk)
        worker.status_updated.connect(self._on_llm_status)
        worker.sources_updated.connect(self._on_llm_sources)
        worker.model_used.connect(self._on_llm_model_used)
        worker.answer_finished.connect(self._on_llm_finished)
        worker.error_occurred.connect(self._on_llm_error)
        worker.finished.connect(lambda w=worker: self._cleanup_llm_thread(w))
        worker.finished.connect(worker.deleteLater)
        self._llm_thread = worker
        worker.start()

    def _cleanup_llm_thread(self, worker: LLMWorkerThread) -> None:
        if worker in self._retired_llm_threads:
            self._retired_llm_threads.remove(worker)
        if self._llm_thread is worker:
            self._llm_thread = None

    def _on_llm_status(self, req_id: int, message: str) -> None:
        if self._shutting_down or req_id != self._active_request_id:
            return
        self.overlay.set_answer_provenance(message)

    def _on_llm_sources(self, req_id: int, web_sources: list[str]) -> None:
        if self._shutting_down or req_id != self._active_request_id:
            return
        merged: list[str] = list(self._current_doc_sources)
        for src in web_sources:
            if src and src not in merged:
                merged.append(src)
        self._current_all_sources = merged
        self.dashboard.set_live_sources(merged)
        self.overlay.set_answer_sources(merged)
        if 0 <= self._qa_index < len(self._qa_history):
            self._qa_history[self._qa_index].sources = list(merged)

    def _on_llm_model_used(self, req_id: int, actual_model: str, route_tag: str) -> None:
        if self._shutting_down or req_id != self._active_request_id:
            return
        self._active_actual_model = actual_model
        self._active_route_tag = route_tag

    def _flush_dashboard_answer(self) -> None:
        if not self._live_answer_dirty:
            return
        self._live_answer_dirty = False
        self.dashboard.set_live_answer_markdown(self._live_answer_buffer)

    def _on_llm_chunk(self, req_id: int, chunk: str) -> None:
        if self._shutting_down or req_id != self._active_request_id:
            return
        self.overlay.append_answer_chunk(chunk)
        self._live_answer_buffer += chunk
        self._live_answer_dirty = True

    def _on_llm_finished(self, req_id: int, elapsed_sec: float) -> None:
        if self._shutting_down or req_id != self._active_request_id:
            return
        self._live_render_timer.stop()
        self._live_answer_dirty = True
        self._flush_dashboard_answer()
        self.overlay.finish_answer_stream()
        self.dashboard.set_generating(False)

        model_label = self._active_actual_model or self.llm_client.provider_display_name()
        if self._active_route_tag == "cache":
            prov = f"⚡ Кеш ({model_label}) · {elapsed_sec:.1f} с"
        elif self._active_route_tag == "fallback":
            prov = f"↪ Фолбек {model_label} · {elapsed_sec:.1f} с"
        elif self._active_actual_model:
            prov = f"{self.config.llm_provider} / {self._active_actual_model} · {elapsed_sec:.1f} с"
        else:
            prov = f"{self.llm_client.provider_display_name()} · {elapsed_sec:.1f} с"

        self.overlay.set_answer_provenance(prov)
        if 0 <= self._qa_index < len(self._qa_history):
            entry = self._qa_history[self._qa_index]
            entry.answer = self._live_answer_buffer
            entry.sources = list(self._current_all_sources)
            entry.provenance = prov
        self._sync_qa_nav()

    def _on_llm_error(self, req_id: int, message: str) -> None:
        if self._shutting_down or req_id != self._active_request_id:
            return
        self._live_render_timer.stop()
        self.overlay.show_answer_error(message)
        self.overlay.set_answer_provenance(f"Помилка ({self.llm_client.provider_display_name()})")
        self.dashboard.set_generating(False)
        self.dashboard.set_live_answer_markdown(f"**Помилка:**\n\n- {message}")
        if 0 <= self._qa_index < len(self._qa_history) and not self._qa_history[self._qa_index].answer:
            self._qa_history.pop(self._qa_index)
            self._qa_index = len(self._qa_history) - 1
        self._sync_qa_nav()

    def _cleanup_probe_thread(self, attr_name: str, worker: QThread) -> None:
        if getattr(self, attr_name, None) is worker:
            setattr(self, attr_name, None)

    def test_llm_connection(self, temp_config: AppConfig) -> None:
        if self._shutting_down or _safe_is_running(self._test_thread):
            return
        worker = LLMTestThread(temp_config, parent=self)
        worker.completed.connect(lambda res, w=worker: self._on_test_llm_completed(res, w))
        worker.finished.connect(lambda w=worker: self._cleanup_probe_thread("_test_thread", w))
        worker.finished.connect(worker.deleteLater)
        self._test_thread = worker
        worker.start()

    def _on_test_llm_completed(self, result, worker: LLMTestThread) -> None:
        if self._shutting_down or self._test_thread is not worker:
            return
        self._test_thread = None
        self.dashboard.set_api_check_result(result)

    def fetch_llm_models(self, temp_config: AppConfig) -> None:
        if self._shutting_down or _safe_is_running(self._models_thread):
            return
        worker = LLMModelsThread(temp_config, parent=self)
        worker.models_loaded.connect(
            lambda models, err, w=worker: self._on_fetch_models_completed(models, err, w)
        )
        worker.finished.connect(lambda w=worker: self._cleanup_probe_thread("_models_thread", w))
        worker.finished.connect(worker.deleteLater)
        self._models_thread = worker
        worker.start()

    def _on_fetch_models_completed(
        self, models: list[str], error: str, worker: LLMModelsThread
    ) -> None:
        if self._shutting_down or self._models_thread is not worker:
            return
        self._models_thread = None
        self.dashboard.set_fetched_models(models, error)

    def test_web_search(self, temp_config: AppConfig) -> None:
        if self._shutting_down or _safe_is_running(self._web_test_thread):
            return
        worker = WebSearchTestThread(temp_config, parent=self)
        worker.completed.connect(lambda ok, msg, w=worker: self._on_test_web_completed(ok, msg, w))
        worker.finished.connect(lambda w=worker: self._cleanup_probe_thread("_web_test_thread", w))
        worker.finished.connect(worker.deleteLater)
        self._web_test_thread = worker
        worker.start()

    def _on_test_web_completed(self, ok: bool, message: str, worker: WebSearchTestThread) -> None:
        if self._shutting_down or self._web_test_thread is not worker:
            return
        self._web_test_thread = None
        self.dashboard.set_web_test_result(ok, message)

    def test_stt(self, temp_config: AppConfig) -> None:
        if self._shutting_down or _safe_is_running(self._stt_test_thread):
            return
        worker = STTTestThread(temp_config, parent=self)
        worker.completed.connect(lambda ok, msg, w=worker: self._on_test_stt_completed(ok, msg, w))
        worker.finished.connect(lambda w=worker: self._cleanup_probe_thread("_stt_test_thread", w))
        worker.finished.connect(worker.deleteLater)
        self._stt_test_thread = worker
        worker.start()

    def _on_test_stt_completed(self, ok: bool, message: str, worker: STTTestThread) -> None:
        if self._shutting_down or self._stt_test_thread is not worker:
            return
        self._stt_test_thread = None
        self.dashboard.set_stt_test_result(ok, message)

    def apply_new_settings(
        self, new_config: AppConfig, *, sync_active_profile: bool = True
    ) -> None:
        try:
            self._hotkey_bridge.update_hotkeys(
                new_config.hotkey_answer,
                new_config.hotkey_toggle,
            )
        except ValueError as exc:
            self.dashboard.show_settings_error(f"Помилка гарячої клавіші: {exc}")
            return

        save_config_to_env(new_config)

        if sync_active_profile:
            active = self._profile_store.active_profile()
            target_lang = (
                new_config.whisper_language if active.stt_language != "" else active.stt_language
            )
            self._profile_store.update_profile(
                active.id,
                system_prompt=new_config.system_prompt,
                custom_vocabulary=new_config.custom_vocabulary,
                stt_language=target_lang,
            )
            self.dashboard.sync_profile_state(
                new_config, self._profile_store.profiles, active.id
            )

        prev_prov = (self.config.stt_provider or "local").strip().lower()
        new_prov = (new_config.stt_provider or "local").strip().lower()
        whisper_reload_needed = (
            new_prov != prev_prov
            or new_config.stt_model != self.config.stt_model
            or new_config.stt_api_key != self.config.stt_api_key
            or new_config.whisper_model != self.config.whisper_model
            or new_config.whisper_device != self.config.whisper_device
            or new_config.whisper_compute_type != self.config.whisper_compute_type
            or (
                new_prov == "groq"
                and (
                    new_config.groq_api_key != self.config.groq_api_key
                    or new_config.openai_api_key != self.config.openai_api_key
                )
            )
            or (
                new_prov == "deepgram"
                and (
                    new_config.whisper_language != self.config.whisper_language
                    or new_config.audio_silence_seconds != self.config.audio_silence_seconds
                )
            )
        )
        self.config = new_config
        self.llm_client.update_config(new_config)

        self.overlay.set_stealth_mode(new_config.overlay_stealth)
        self.dashboard.set_overlay_stealth(new_config.overlay_stealth)
        self.overlay.set_opacity_percent(new_config.overlay_opacity)
        self.overlay.set_font_scale(new_config.overlay_font_scale)
        self.overlay.set_web_search_enabled(new_config.web_search_enabled)
        self.dashboard.set_web_search_enabled(new_config.web_search_enabled)
        self.overlay.set_screen_vision_enabled(new_config.screen_vision_enabled)
        self.dashboard.set_screen_vision_enabled(new_config.screen_vision_enabled)
        self._sync_tray_check_states()
        self.overlay.set_hotkey_labels(
            format_hotkey_label(new_config.hotkey_answer),
            format_hotkey_label(new_config.hotkey_toggle),
        )
        self.overlay.set_answer_provenance(self.llm_client.provider_display_name())

        self.dashboard.set_dashboard_stealth(new_config.dashboard_stealth)

        merged_vocab = self._merged_vocabulary()
        if whisper_reload_needed:
            was_capturing = self._is_capturing
            dev_idx = self.dashboard.home_device_combo.currentData()
            self._clear_pending_answer_flush()
            if was_capturing:
                self.stop_capture()

            old_thread = self._transcriber_thread
            for sig in (
                old_thread.model_status,
                old_thread.transcript_ready,
                old_thread.error_occurred,
                old_thread.chunk_processed,
            ):
                try:
                    sig.disconnect()
                except Exception:
                    pass
            if _safe_is_running(old_thread):
                old_thread.stop()
                if not old_thread.wait(150):
                    self._retired_transcriber_threads.append(old_thread)

                    def _retire_transcriber(t: TranscriberThread = old_thread) -> None:
                        if t in self._retired_transcriber_threads:
                            self._retired_transcriber_threads.remove(t)
                        t.deleteLater()

                    old_thread.finished.connect(_retire_transcriber)
                else:
                    old_thread.deleteLater()
            else:
                old_thread.deleteLater()

            self._chunk_queue = queue.Queue(maxsize=10)
            self._raw_pcm_queue = queue.Queue(maxsize=120)
            self._transcriber_thread = self._create_transcriber_thread(new_config)
            self._transcriber_thread.start()
            if was_capturing:
                self.start_capture(int(dev_idx) if dev_idx is not None else -1)
        else:
            self._transcriber_thread.update_runtime_config(new_config)
            self._transcriber_thread.set_vocabulary(merged_vocab)

    def request_quit(self) -> None:
        self.shutdown()
        self.dashboard.force_close()
        self.overlay.close()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def shutdown(self) -> None:
        """Cleanly terminate tray, hotkeys, audio, Whisper, context, and LLM threads in parallel."""
        if self._shutting_down:
            return
        self._shutting_down = True
        self._pending_context_job = None
        self._clear_pending_answer_flush()

        self._live_render_timer.stop()
        self._opacity_save_timer.stop()
        self._hotkey_bridge.stop()

        if self._tray_icon is not None:
            self._tray_icon.hide()

        # Phase 1: Signal every active worker to stop/cancel concurrently before waiting.
        if _safe_is_running(self._llm_thread):
            assert self._llm_thread is not None
            self._llm_thread.cancel()
        for retired in list(self._retired_llm_threads):
            if _safe_is_running(retired):
                retired.cancel()
        if _safe_is_running(self._audio_thread):
            assert self._audio_thread is not None
            self._audio_thread.stop()
        if _safe_is_running(self._transcriber_thread):
            self._transcriber_thread.stop()
        for retired_t in list(self._retired_transcriber_threads):
            if _safe_is_running(retired_t):
                retired_t.stop()

        # Phase 2: Wait within bounded per-worker budgets.
        if _safe_is_running(self._llm_thread):
            assert self._llm_thread is not None
            self._llm_thread.wait(1500)
        for retired in list(self._retired_llm_threads):
            if _safe_is_running(retired):
                retired.wait(600)
        for retired_t in list(self._retired_transcriber_threads):
            if _safe_is_running(retired_t):
                retired_t.wait(600)
        for probe in (
            self._context_thread,
            self._test_thread,
            self._models_thread,
            self._web_test_thread,
            self._stt_test_thread,
        ):
            if _safe_is_running(probe):
                assert probe is not None
                probe.wait(1000)
        if _safe_is_running(self._audio_thread):
            assert self._audio_thread is not None
            self._audio_thread.wait(2500)
        if _safe_is_running(self._transcriber_thread):
            self._transcriber_thread.wait(3500)


def _load_app_icon() -> QIcon:
    """Load crisp multi-resolution Windows .ico first, falling back to PNG."""
    for rel in ("assets/icon.ico", "assets/icon.png"):
        path = get_resource_path(rel)
        if path.is_file():
            icon = QIcon(str(path))
            if not icon.isNull():
                return icon
    return QIcon()


def _run_smoke_test(out_path: str | None) -> int:
    """Self-verification mode for both Python and compiled ZoomAICopilot.exe."""
    import io
    import ctranslate2
    import ddgs
    import faster_whisper
    import numpy as np
    from docx import Document as DocxDocument
    from elevenlabs import ElevenLabs
    from faster_whisper import WhisperModel
    import websockets.sync.client
    from audio_capture import StreamingSileroVAD
    import web_search

    report: dict[str, object] = {
        "frozen": bool(getattr(sys, "frozen", False)),
        "executable": sys.executable,
    }

    report["web_search_ok"] = hasattr(web_search, "search_web") and hasattr(ddgs, "DDGS")
    report["websockets_ok"] = callable(getattr(websockets.sync.client, "connect", None))
    el_smoke = ElevenLabs(api_key="smoke_test_dummy")
    report["elevenlabs_ok"] = callable(getattr(el_smoke.speech_to_text, "convert", None))

    dll_dirs = register_cuda_dlls()
    report["dll_dirs"] = dll_dirs
    cuda_count = ctranslate2.get_cuda_device_count()
    report["cuda_device_count"] = cuda_count

    silero_path = Path(faster_whisper.__file__).parent / "assets" / "silero_vad_v6.onnx"
    report["silero_vad_exists"] = silero_path.is_file()
    report["streaming_vad_ok"] = StreamingSileroVAD().available

    # Check python-docx template bundling
    buf = io.BytesIO()
    d = DocxDocument()
    d.add_paragraph("Тест DOCX українською")
    d.save(buf)
    report["docx_ok"] = buf.tell() > 0

    # Check WhisperModel CUDA forward pass
    model = WhisperModel("tiny", device="cuda", compute_type="float16")
    audio = (0.04 * np.random.default_rng(7).standard_normal(16000)).astype(np.float32)
    segs, info = model.transcribe(audio, language="uk", vad_filter=True, beam_size=5)
    _ = list(segs)
    report["whisper_cuda_ok"] = True
    report["whisper_lang"] = info.language

    devices = list_loopback_devices()
    report["wasapi_devices"] = len(devices)

    cfg = load_config()
    app = QApplication.instance() or QApplication(sys.argv)
    app_icon = _load_app_icon()

    dash = DashboardWindow(config=cfg, app_icon=app_icon)
    dash.set_devices(devices)
    ov = CopilotOverlayWindow(app_icon=app_icon)
    ov.set_devices(devices)

    dash.show()
    ov.show()

    def _inspect_and_exit() -> None:
        report["stealth_applied"] = ov.stealth_applied
        report["compact_manual_input_ok"] = (not ov.is_expanded) and ov.manual_input.isVisible()
        title_min_w = ov._title_bar.minimumSizeHint().width()
        dash_min = dash.minimumSizeHint()
        report["compact_title_width"] = title_min_w
        report["compact_title_fit_ok"] = title_min_w <= 430
        report["dashboard_min_size"] = [dash_min.width(), dash_min.height()]
        report["dashboard_min_size_ok"] = dash_min.width() <= 840 and dash_min.height() <= 560
        ov.set_expanded(True)
        report["expanded_true"] = ov.is_expanded
        ov.set_expanded(False)
        report["expanded_false"] = not ov.is_expanded
        ov.set_opacity_percent(75)
        report["opacity_75"] = abs(ov.windowOpacity() - 0.75) <= 0.02
        ov.set_web_search_enabled(True)
        report["web_toggle_ok"] = ov.web_search_enabled
        ov.set_screen_vision_enabled(True)
        report["screen_toggle_ok"] = ov.screen_vision_enabled
        shot = capture_screen()
        report["screen_capture_ok"] = shot is not None and len(shot.image_bytes) > 100
        ov.close()
        dash.force_close()
        app.quit()

    QTimer.singleShot(350, _inspect_and_exit)
    app.exec()

    passed = (
        report["cuda_device_count"] >= 1
        and report["silero_vad_exists"] is True
        and report["streaming_vad_ok"] is True
        and report["websockets_ok"] is True
        and report["docx_ok"] is True
        and report["whisper_cuda_ok"] is True
        and report["wasapi_devices"] >= 1
        and report["stealth_applied"] is True
        and report["compact_manual_input_ok"] is True
        and report["compact_title_fit_ok"] is True
        and report["dashboard_min_size_ok"] is True
        and report["expanded_true"] is True
        and report["expanded_false"] is True
        and report["opacity_75"] is True
        and report["web_search_ok"] is True
        and report["elevenlabs_ok"] is True
        and report["web_toggle_ok"] is True
        and report["screen_toggle_ok"] is True
        and report["screen_capture_ok"] is True
    )
    report["passed"] = passed

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    return 0 if passed else 1


def _signal_existing_instance(server_name: str) -> bool:
    """Notify an already-running instance via QLocalSocket; return True if connected."""
    probe = QLocalSocket()
    probe.connectToServer(server_name)
    if not probe.waitForConnected(250):
        return False
    probe.write(b"SHOW\n")
    probe.waitForBytesWritten(250)
    probe.disconnectFromServer()
    return True


def main() -> int:
    """Application entry point with race-safe single-instance IPC."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-test-out", default="")
    args, qt_args = parser.parse_known_args()

    if args.smoke_test:
        return _run_smoke_test(args.smoke_test_out or None)

    instance_guard = WindowsSingleInstanceGuard(SINGLE_INSTANCE_MUTEX_NAME)
    if not instance_guard.acquire():
        app_stub = QCoreApplication([sys.argv[0], *qt_args])
        for attempt in range(5):
            if _signal_existing_instance(SINGLE_INSTANCE_SERVER_NAME):
                break
            if attempt < 4:
                time.sleep(0.20)
        del app_stub
        return 0

    app = QApplication([sys.argv[0], *qt_args])

    local_server = QLocalServer(app)
    QLocalServer.removeServer(SINGLE_INSTANCE_SERVER_NAME)
    if not local_server.listen(SINGLE_INSTANCE_SERVER_NAME):
        logger.warning("Single-instance IPC server could not bind (%s)", local_server.errorString())

    app.setQuitOnLastWindowClosed(False)
    config = load_config()
    if not config.overlay_stealth or not config.dashboard_stealth:
        config.overlay_stealth = True
        config.dashboard_stealth = True
        save_config_to_env(config)

    app_icon = _load_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    dashboard = DashboardWindow(config=config, app_icon=app_icon)
    overlay = CopilotOverlayWindow(
        hotkey_answer_label=format_hotkey_label(config.hotkey_answer),
        hotkey_toggle_label=format_hotkey_label(config.hotkey_toggle),
        stealth_enabled=config.overlay_stealth,
        opacity_percent=config.overlay_opacity,
        font_scale=config.overlay_font_scale,
        web_search_enabled=config.web_search_enabled,
        screen_vision_enabled=config.screen_vision_enabled,
        app_icon=app_icon,
    )

    controller = CopilotController(
        config=config,
        dashboard=dashboard,
        overlay=overlay,
        app_icon=app_icon,
    )

    def _on_second_instance() -> None:
        while local_server.hasPendingConnections():
            conn = local_server.nextPendingConnection()
            if conn is not None:
                conn.waitForReadyRead(100)
                conn.close()
                conn.deleteLater()
        controller.show_dashboard()

    local_server.newConnection.connect(_on_second_instance)
    app.aboutToQuit.connect(local_server.close)
    app.aboutToQuit.connect(controller.shutdown)
    app.aboutToQuit.connect(instance_guard.release)

    dashboard.show()
    controller.start()
    try:
        return app.exec()
    finally:
        instance_guard.release()


if __name__ == "__main__":
    sys.exit(main())
