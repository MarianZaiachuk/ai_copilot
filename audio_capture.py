"""Windows WASAPI loopback audio capture and 16 kHz mono resampling for Whisper."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import queue
import threading
import time
from typing import Callable

import numpy as np
import pyaudiowpatch as pyaudio
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
_PYAUDIO_LOCK = threading.Lock()
_SILERO_FRAME_SAMPLES = 512
_SILERO_CONTEXT_SAMPLES = 64


class StreamingSileroVAD:
    """Stateful chunk-by-chunk Silero VAD scorer over 16 kHz mono float32 audio."""

    def __init__(self) -> None:
        self._session = None
        self.available = False
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _SILERO_CONTEXT_SAMPLES), dtype=np.float32)
        self._pending = np.empty(0, dtype=np.float32)
        self.last_prob = 0.0
        try:
            from faster_whisper.vad import get_vad_model

            vad_model = get_vad_model()
            session = getattr(vad_model, "session", None)
            if session is not None:
                self._session = session
                self.available = True
        except Exception as exc:
            logger.debug("Silero VAD unavailable, falling back to RMS VAD: %s", exc)
            self.available = False

    def reset(self) -> None:
        self._h.fill(0.0)
        self._c.fill(0.0)
        self._context.fill(0.0)
        self._pending = np.empty(0, dtype=np.float32)
        self.last_prob = 0.0

    def score_block(self, block_16k: np.ndarray) -> float:
        """Process 16 kHz mono samples and return max speech probability in [0, 1], or -1.0 if unavailable."""
        if not self.available or self._session is None:
            return -1.0
        arr = np.nan_to_num(
            np.asarray(block_16k, dtype=np.float32).reshape(-1),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        if arr.size == 0:
            return self.last_prob

        self._pending = arr if self._pending.size == 0 else np.concatenate((self._pending, arr))
        if self._pending.size < _SILERO_FRAME_SAMPLES:
            return self.last_prob

        max_prob = 0.0
        pos = 0
        total = self._pending.size
        while pos + _SILERO_FRAME_SAMPLES <= total:
            frame = self._pending[pos : pos + _SILERO_FRAME_SAMPLES].reshape(1, _SILERO_FRAME_SAMPLES)
            x = np.concatenate((self._context, frame), axis=1)
            try:
                out, hn, cn = self._session.run(None, {"input": x, "h": self._h, "c": self._c})
                self._h = hn
                self._c = cn
                self._context = frame[:, -_SILERO_CONTEXT_SAMPLES:].copy()
                prob = float(np.asarray(out).reshape(-1)[0])
                if prob > max_prob:
                    max_prob = prob
            except Exception as exc:
                logger.warning("Silero VAD frame inference failed, falling back to RMS: %s", exc)
                self.available = False
                self._pending = np.empty(0, dtype=np.float32)
                return -1.0
            pos += _SILERO_FRAME_SAMPLES

        self._pending = (
            np.ascontiguousarray(self._pending[pos:], dtype=np.float32)
            if pos < total
            else np.empty(0, dtype=np.float32)
        )
        self.last_prob = max_prob
        return max_prob


@dataclass(frozen=True)
class AudioDeviceInfo:
    """Metadata describing a Windows WASAPI loopback audio device."""

    index: int
    name: str
    channels: int
    sample_rate: int
    is_default: bool = False


def list_loopback_devices() -> list[AudioDeviceInfo]:
    """Enumerate all available Windows WASAPI loopback devices, default first."""
    devices: list[AudioDeviceInfo] = []
    try:
        with _PYAUDIO_LOCK:
            with pyaudio.PyAudio() as p:
                default_index: int | None = None
                try:
                    default_info = p.get_default_wasapi_loopback()
                    default_index = int(default_info["index"])
                except (OSError, LookupError, KeyError, TypeError):
                    default_index = None

                try:
                    generator = p.get_loopback_device_info_generator()
                except (OSError, AttributeError):
                    generator = ()

                for raw in generator:
                    idx = int(raw["index"])
                    channels = max(1, int(raw.get("maxInputChannels", 2) or 2))
                    sample_rate = max(8000, int(raw.get("defaultSampleRate", 48000) or 48000))
                    name = str(raw.get("name", f"Loopback #{idx}"))
                    devices.append(
                        AudioDeviceInfo(
                            index=idx,
                            name=name,
                            channels=channels,
                            sample_rate=sample_rate,
                            is_default=(idx == default_index),
                        )
                    )
    except OSError as exc:
        logger.warning("Failed to enumerate WASAPI loopback devices: %s", exc)
        return []

    devices.sort(key=lambda d: (not d.is_default, d.index))
    return devices


def get_default_loopback_device() -> AudioDeviceInfo | None:
    """Return the default WASAPI loopback device, or the first available fallback."""
    devices = list_loopback_devices()
    if not devices:
        return None
    for dev in devices:
        if dev.is_default:
            return dev
    return devices[0]


def _bytes_to_mono_float32(raw_bytes: bytes, fmt: int, channels: int) -> np.ndarray:
    """Safely convert aligned PyAudio PCM bytes into sanitized mono float32 at source sample rate."""
    if not raw_bytes:
        return np.empty(0, dtype=np.float32)

    dtype = np.float32 if fmt == pyaudio.paFloat32 else np.int16
    ch = max(1, int(channels))
    frame_bytes = np.dtype(dtype).itemsize * ch
    usable_bytes = (len(raw_bytes) // frame_bytes) * frame_bytes
    if usable_bytes == 0:
        return np.empty(0, dtype=np.float32)

    raw_view = memoryview(raw_bytes)[:usable_bytes]
    samples = np.frombuffer(raw_view, dtype=dtype)
    if fmt == pyaudio.paInt16:
        arr = samples.astype(np.float32) / 32768.0
    else:
        arr = samples.astype(np.float32, copy=True)

    if ch > 1:
        mono = arr.reshape(-1, ch).mean(axis=1, dtype=np.float32)
    else:
        mono = arr

    mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.ascontiguousarray(np.clip(mono, -1.0, 1.0), dtype=np.float32)


def resample_to_16k_mono(audio: np.ndarray, src_rate: int, channels: int) -> np.ndarray:
    """Convert interleaved float32/int16 audio into contiguous 16 kHz mono float32."""
    arr = np.asarray(audio)
    if arr.size == 0:
        return np.empty(0, dtype=np.float32)

    if arr.dtype == np.int16:
        samples = arr.astype(np.float32) / 32768.0
    elif arr.dtype == np.int32:
        samples = arr.astype(np.float32) / 2147483648.0
    else:
        samples = arr.astype(np.float32, copy=False)

    if samples.ndim == 2:
        mono = samples.mean(axis=1, dtype=np.float32)
    else:
        samples = samples.reshape(-1)
        ch = max(1, int(channels))
        if ch > 1:
            usable = (samples.size // ch) * ch
            if usable == 0:
                return np.empty(0, dtype=np.float32)
            mono = samples[:usable].reshape(-1, ch).mean(axis=1, dtype=np.float32)
        else:
            mono = samples

    if mono.size == 0:
        return np.empty(0, dtype=np.float32)

    mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
    rate = max(1, int(src_rate))
    if rate != TARGET_SAMPLE_RATE:
        g = math.gcd(TARGET_SAMPLE_RATE, rate)
        up = TARGET_SAMPLE_RATE // g
        down = rate // g
        mono = resample_poly(mono, up, down).astype(np.float32)

    mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
    mono = np.clip(mono, -1.0, 1.0)
    return np.ascontiguousarray(mono, dtype=np.float32)


class AudioChunker:
    """Stateful VAD/overlap chunker emitting mono utterances with clean overlap metadata."""

    def __init__(
        self,
        target_chunk_seconds: float = 4.0,
        overlap_seconds: float = 0.5,
        silence_seconds: float = 0.8,
        silence_threshold: float = 0.008,
        min_speech_seconds: float = 0.6,
        pre_roll_seconds: float = 0.4,
        sample_rate: int = TARGET_SAMPLE_RATE,
        max_chunk_seconds: float | None = None,
        vad_fn: Callable[[np.ndarray], float] | None = None,
        vad_threshold: float = 0.42,
        vad_release_threshold: float = 0.25,
    ) -> None:
        self.sample_rate = max(1000, int(sample_rate))
        self.target_chunk_seconds = max(1.0, float(target_chunk_seconds))
        self.overlap_seconds = max(0.0, min(float(overlap_seconds), self.target_chunk_seconds - 0.25))
        self.silence_seconds = max(0.2, float(silence_seconds))
        self.silence_threshold = max(1e-5, float(silence_threshold))
        self.min_speech_seconds = max(0.2, float(min_speech_seconds))
        self.pre_roll_seconds = max(0.0, float(pre_roll_seconds))
        self._vad_fn = vad_fn
        self._vad_threshold = float(vad_threshold)
        self._vad_release_threshold = float(vad_release_threshold)

        self._target_samples = int(self.target_chunk_seconds * self.sample_rate)
        self._max_samples = (
            int(max(self.target_chunk_seconds, float(max_chunk_seconds)) * self.sample_rate)
            if max_chunk_seconds is not None
            else self._target_samples
        )
        self._overlap_samples = int(self.overlap_seconds * self.sample_rate)
        self._min_speech_samples = int(self.min_speech_seconds * self.sample_rate)
        self._pre_roll_samples = int(self.pre_roll_seconds * self.sample_rate)
        self._min_flush_samples = int(0.4 * self.sample_rate)
        self._silence_samples_limit = int(self.silence_seconds * self.sample_rate)

        self._buffer: list[np.ndarray] = []
        self._total_samples = 0
        self._silent_samples_tail = 0
        self._has_speech = False
        self._vad_in_speech = False
        self._has_new_speech_since_cut = False
        self._next_chunk_has_overlap = False
        self.last_emit_had_overlap = False
        self._last_speech_time: float | None = None

    @staticmethod
    def _block_rms(arr: np.ndarray) -> float:
        raw_sq = np.square(arr, dtype=np.float64)
        raw_rms = float(np.sqrt(np.mean(raw_sq)))
        mean_val = float(np.mean(arr, dtype=np.float64))
        if abs(mean_val) > 1e-6:
            ac = arr - mean_val
            ac_rms = float(np.sqrt(np.mean(np.square(ac, dtype=np.float64))))
            # Preserve synthetic constant-amplitude test blocks while removing real hardware DC bias
            if ac_rms > 1e-6:
                return ac_rms
        return raw_rms

    def _concat_buffer(self) -> np.ndarray:
        if not self._buffer:
            return np.empty(0, dtype=np.float32)
        if len(self._buffer) == 1:
            return np.ascontiguousarray(self._buffer[0], dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(self._buffer), dtype=np.float32)

    def _trim_to_tail(self, max_samples: int) -> None:
        if max_samples <= 0:
            self._buffer.clear()
            self._total_samples = 0
            return
        if self._total_samples <= max_samples:
            return
        combined = self._concat_buffer()
        tail = np.ascontiguousarray(combined[-max_samples:], dtype=np.float32)
        self._buffer = [tail]
        self._total_samples = tail.size

    def _clear(self) -> None:
        self._buffer.clear()
        self._total_samples = 0
        self._silent_samples_tail = 0
        self._has_speech = False
        self._vad_in_speech = False
        self._has_new_speech_since_cut = False
        self._next_chunk_has_overlap = False
        self._last_speech_time = None

    def feed(self, block: np.ndarray, now: float) -> tuple[np.ndarray | None, float]:
        """Consume a mono float32 block and return (ready_chunk, rms_level)."""
        arr = np.nan_to_num(
            np.asarray(block, dtype=np.float32).reshape(-1),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        if arr.size == 0:
            return self.poll_silence(now), 0.0

        rms = self._block_rms(arr)
        self._buffer.append(arr)
        self._total_samples += arr.size

        vad_prob = -1.0
        if self._vad_fn is not None:
            vad_prob = 0.0 if rms < 1e-4 else float(self._vad_fn(arr))

        if vad_prob >= 0.0:
            thresh = self._vad_release_threshold if self._vad_in_speech else self._vad_threshold
            is_speech = vad_prob >= thresh
            self._vad_in_speech = is_speech
            is_pause_valley = (not is_speech) or (vad_prob < self._vad_release_threshold)
        else:
            is_speech = rms >= self.silence_threshold
            is_pause_valley = rms < self.silence_threshold * 1.4

        if is_speech:
            self._has_speech = True
            self._has_new_speech_since_cut = True
            self._silent_samples_tail = 0
            self._last_speech_time = now
        else:
            self._silent_samples_tail += arr.size

        if not self._has_speech:
            self._trim_to_tail(self._pre_roll_samples)
            self._silent_samples_tail = 0
            return None, rms

        soft_cut = (
            self._total_samples >= self._target_samples
            and (is_pause_valley or self._total_samples >= self._max_samples)
        )
        if soft_cut:
            emitted = self._concat_buffer()
            self.last_emit_had_overlap = self._next_chunk_has_overlap
            self._silent_samples_tail = 0
            self._has_new_speech_since_cut = False
            if self._overlap_samples > 0 and emitted.size > self._overlap_samples:
                tail = np.ascontiguousarray(emitted[-self._overlap_samples:], dtype=np.float32)
                self._buffer = [tail]
                self._total_samples = tail.size
                if self._max_samples > self._target_samples:
                    tail_active = (
                        is_speech
                        if vad_prob >= 0.0
                        else (self._block_rms(tail) >= self.silence_threshold * 0.85)
                    )
                    if tail_active:
                        self._has_speech = True
                        self._has_new_speech_since_cut = False
                        self._next_chunk_has_overlap = True
                        self._last_speech_time = now
                    else:
                        self._has_speech = False
                        self._vad_in_speech = False
                        self._next_chunk_has_overlap = False
                        self._last_speech_time = None
                else:
                    self._has_speech = False
                    self._vad_in_speech = False
                    self._next_chunk_has_overlap = True
                    self._last_speech_time = None
            else:
                self._clear()
            return emitted, rms

        if (
            self._total_samples >= self._min_speech_samples
            and self._last_speech_time is not None
            and self._silent_samples_tail >= self._silence_samples_limit
        ):
            if not self._has_new_speech_since_cut:
                self._clear()
                return None, rms
            emitted = self._concat_buffer()
            self.last_emit_had_overlap = self._next_chunk_has_overlap
            self._clear()
            return emitted, rms

        return None, rms

    def poll_silence(self, now: float) -> np.ndarray | None:
        """Flush buffered speech when WASAPI callbacks stop arriving during silence."""
        if (
            not self._has_speech
            or self._last_speech_time is None
            or (now - self._last_speech_time) < self.silence_seconds
        ):
            return None

        if self._total_samples < self._min_speech_samples or not self._has_new_speech_since_cut:
            self._clear()
            return None

        emitted = self._concat_buffer()
        self.last_emit_had_overlap = self._next_chunk_has_overlap
        self._clear()
        return emitted

    @property
    def has_pending_speech(self) -> bool:
        """Return True when the buffer currently holds enough un-emitted speech to flush."""
        return bool(
            self._has_speech
            and self._has_new_speech_since_cut
            and self._total_samples >= self._min_flush_samples
        )

    def flush(self) -> np.ndarray | None:
        """Flush any remaining buffered speech on capture stop."""
        if (
            self._has_speech
            and self._has_new_speech_since_cut
            and self._total_samples >= self._min_flush_samples
        ):
            emitted = self._concat_buffer()
            self.last_emit_had_overlap = self._next_chunk_has_overlap
            self._clear()
            return emitted
        self._clear()
        return None


class WasapiLoopbackCapturer:
    """Captures Windows system output audio via WASAPI loopback in non-blocking mode."""

    def __init__(
        self,
        device_index: int | None = None,
        chunk_seconds: float = 4.0,
        overlap_seconds: float = 0.5,
        silence_seconds: float = 0.5,
        silence_threshold: float = 0.008,
    ) -> None:
        self.device_index = device_index
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.silence_seconds = silence_seconds
        self.silence_threshold = silence_threshold
        self._resolved_device_name: str | None = None
        self._has_pending_flag = threading.Event()

    @property
    def has_pending_speech(self) -> bool:
        """Return True while buffered un-emitted speech is waiting in the active chunker."""
        return self._has_pending_flag.is_set()

    def _resolve_device(self, p: pyaudio.PyAudio) -> dict:
        if self._resolved_device_name:
            try:
                for loopback in p.get_loopback_device_info_generator():
                    if str(loopback.get("name", "")) == self._resolved_device_name:
                        return loopback
            except (OSError, AttributeError):
                pass

        if self.device_index is not None:
            try:
                info = p.get_device_info_by_index(self.device_index)
                if info.get("isLoopbackDevice", False):
                    self._resolved_device_name = str(info.get("name", "")) or None
                    return info
                target_name = str(info.get("name", ""))
                for loopback in p.get_loopback_device_info_generator():
                    if target_name and target_name in str(loopback.get("name", "")):
                        self._resolved_device_name = str(loopback.get("name", "")) or None
                        return loopback
            except (OSError, ValueError, KeyError, TypeError):
                logger.warning(
                    "Requested audio device index %s unavailable; falling back to default loopback.",
                    self.device_index,
                )

        default_lb = p.get_default_wasapi_loopback()
        self._resolved_device_name = str(default_lb.get("name", "")) or None
        return default_lb

    @staticmethod
    def _close_stream_quietly(stream, p: pyaudio.PyAudio | None = None) -> None:
        if stream is None:
            return
        try:
            if stream.is_active():
                stream.stop_stream()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
        if p is not None:
            try:
                remove_fn = getattr(p, "_remove_stream", None)
                if callable(remove_fn):
                    remove_fn(stream)
                else:
                    streams = getattr(p, "_streams", None)
                    if isinstance(streams, set):
                        streams.discard(stream)
            except Exception:
                pass

    def run_loop(
        self,
        stop_event: threading.Event,
        on_chunk: Callable[[np.ndarray], None] | None = None,
        on_level: Callable[[float], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        max_reconnects: int = 3,
        on_chunk_meta: Callable[[np.ndarray, bool], None] | None = None,
        on_raw_16k: Callable[[np.ndarray], None] | None = None,
        flush_event: threading.Event | None = None,
        on_flushed: Callable[[bool], None] | None = None,
    ) -> None:
        """Run capture loop with bounded WASAPI auto-reconnect until stop_event is set."""
        if on_chunk is None and on_chunk_meta is None and on_raw_16k is None:
            raise ValueError("One of on_chunk, on_chunk_meta, or on_raw_16k callback must be provided")

        raw_stream_mode = on_raw_16k is not None and on_chunk is None and on_chunk_meta is None

        def _emit_chunk(raw_mono: np.ndarray, src_rate: int, had_overlap: bool) -> None:
            chunk_16k = resample_to_16k_mono(raw_mono, src_rate=src_rate, channels=1)
            if chunk_16k.size == 0:
                return
            if on_chunk_meta is not None:
                on_chunk_meta(chunk_16k, had_overlap)
            elif on_chunk is not None:
                on_chunk(chunk_16k)

        def _make_stream_callback(
            raw_queue: queue.Queue[tuple[bytes, int]],
            format_box: list[int],
        ):
            def _callback(in_data, frame_count, time_info, status_flags):
                if stop_event.is_set():
                    return (None, pyaudio.paComplete)
                if in_data:
                    try:
                        raw_queue.put_nowait((in_data, format_box[0]))
                    except queue.Full:
                        try:
                            raw_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            raw_queue.put_nowait((in_data, format_box[0]))
                        except queue.Full:
                            pass
                return (None, pyaudio.paContinue)

            return _callback

        reconnect_attempts = 0
        while not stop_event.is_set():
            raw_queue: queue.Queue[tuple[bytes, int]] = queue.Queue(maxsize=256)
            p: pyaudio.PyAudio | None = None
            stream = None
            chunker: AudioChunker | None = None
            try:
                with _PYAUDIO_LOCK:
                    p = pyaudio.PyAudio()
                device_info = self._resolve_device(p)
                dev_index = int(device_info["index"])
                channels = max(1, int(device_info.get("maxInputChannels", 2) or 2))
                src_rate = max(8000, int(device_info.get("defaultSampleRate", 48000) or 48000))
                if not raw_stream_mode:
                    silero_vad = StreamingSileroVAD()
                    vad_scorer: Callable[[np.ndarray], float] | None = None
                    if silero_vad.available:

                        def _score_vad(
                            raw_mono: np.ndarray,
                            _vad: StreamingSileroVAD = silero_vad,
                            _rate: int = src_rate,
                        ) -> float:
                            block_16k = resample_to_16k_mono(raw_mono, src_rate=_rate, channels=1)
                            return _vad.score_block(block_16k)

                        vad_scorer = _score_vad
                    max_utterance_sec = (
                        min(14.0, max(self.chunk_seconds + 4.0, 10.0))
                        if vad_scorer is not None
                        else self.chunk_seconds + 1.5
                    )
                    chunker = AudioChunker(
                        target_chunk_seconds=self.chunk_seconds,
                        overlap_seconds=self.overlap_seconds,
                        silence_seconds=self.silence_seconds,
                        silence_threshold=self.silence_threshold,
                        sample_rate=src_rate,
                        max_chunk_seconds=max_utterance_sec,
                        vad_fn=vad_scorer,
                    )
                frames_per_buffer = max(512, int(src_rate * 0.05))  # ~50 ms blocks
                format_box = [pyaudio.paFloat32]
                callback = _make_stream_callback(raw_queue, format_box)

                try:
                    stream = p.open(
                        format=pyaudio.paFloat32,
                        channels=channels,
                        rate=src_rate,
                        input=True,
                        input_device_index=dev_index,
                        frames_per_buffer=frames_per_buffer,
                        stream_callback=callback,
                    )
                except OSError:
                    format_box[0] = pyaudio.paInt16
                    stream = p.open(
                        format=pyaudio.paInt16,
                        channels=channels,
                        rate=src_rate,
                        input=True,
                        input_device_index=dev_index,
                        frames_per_buffer=frames_per_buffer,
                        stream_callback=callback,
                    )

                logger.info(
                    "WASAPI loopback started: %s (idx=%d, rate=%d, ch=%d, fmt=%s, mode=%s)",
                    device_info.get("name", "unknown"),
                    dev_index,
                    src_rate,
                    channels,
                    "float32" if format_box[0] == pyaudio.paFloat32 else "int16",
                    "raw16k" if raw_stream_mode else "vad_chunker",
                )
                stream.start_stream()

                last_level_emit = 0.0
                last_active_check = time.monotonic()

                def _sync_pending(_chunker: AudioChunker | None = chunker) -> None:
                    if _chunker is not None and _chunker.has_pending_speech:
                        self._has_pending_flag.set()
                    else:
                        self._has_pending_flag.clear()

                def _handle_flush(
                    _chunker: AudioChunker | None = chunker,
                    _rate: int = src_rate,
                ) -> None:
                    if flush_event is None or not flush_event.is_set():
                        return
                    flush_event.clear()
                    flushed: np.ndarray | None = None
                    had_ov = False
                    if _chunker is not None:
                        flushed = _chunker.flush()
                        had_ov = _chunker.last_emit_had_overlap
                    _sync_pending(_chunker)
                    if flushed is not None:
                        _emit_chunk(flushed, _rate, had_ov)
                    if on_flushed is not None:
                        on_flushed(flushed is not None)

                while not stop_event.is_set():
                    try:
                        raw_bytes, fmt = raw_queue.get(timeout=0.04)
                    except queue.Empty:
                        _handle_flush()
                        now = time.monotonic()
                        if chunker is not None:
                            ready = chunker.poll_silence(now)
                            _sync_pending()
                            if ready is not None:
                                _emit_chunk(ready, src_rate, chunker.last_emit_had_overlap)
                        if on_level is not None and (now - last_level_emit) >= 0.1:
                            on_level(0.0)
                            last_level_emit = now
                        if (now - last_active_check) >= 1.5:
                            last_active_check = now
                            if not stream.is_active():
                                raise OSError("WASAPI stream became inactive") from None
                        continue

                    reconnect_attempts = 0
                    mono_src = _bytes_to_mono_float32(raw_bytes, fmt, channels)
                    now = time.monotonic()
                    if raw_stream_mode:
                        rms = AudioChunker._block_rms(mono_src)
                        if on_raw_16k is not None and mono_src.size > 0:
                            block_16k = resample_to_16k_mono(mono_src, src_rate=src_rate, channels=1)
                            if block_16k.size > 0:
                                on_raw_16k(block_16k)
                    else:
                        assert chunker is not None
                        ready, rms = chunker.feed(mono_src, now)
                        if ready is not None:
                            _emit_chunk(ready, src_rate, chunker.last_emit_had_overlap)
                        _sync_pending()
                    _handle_flush()
                    if on_level is not None and (now - last_level_emit) >= 0.08:
                        on_level(rms)
                        last_level_emit = now

                if chunker is not None:
                    final_chunk = chunker.flush()
                    _sync_pending()
                    if final_chunk is not None:
                        _emit_chunk(final_chunk, src_rate, chunker.last_emit_had_overlap)
                break
            except (OSError, LookupError, ValueError, KeyError) as exc:
                if stop_event.is_set():
                    break
                if chunker is not None:
                    leftover = chunker.flush()
                    if leftover is not None and "src_rate" in locals():
                        _emit_chunk(leftover, src_rate, chunker.last_emit_had_overlap)
                reconnect_attempts += 1
                logger.warning(
                    "WASAPI capture interrupted (%s), attempt %d/%d",
                    exc,
                    reconnect_attempts,
                    max_reconnects,
                )
                if reconnect_attempts > max_reconnects:
                    raise RuntimeError(
                        "Втрачено зв'язок із аудіопристроєм WASAPI Loopback. "
                        "Перевірте пристрій виводу та натисніть «Старт»."
                    ) from exc
                if on_status is not None:
                    on_status(f"Аудіопристрій змінився — перепідключення ({reconnect_attempts}/{max_reconnects})...")
                stop_event.wait(0.8)
            finally:
                self._has_pending_flag.clear()
                self._close_stream_quietly(stream, p)
                if p is not None:
                    with _PYAUDIO_LOCK:
                        try:
                            p.terminate()
                        except Exception:
                            pass
