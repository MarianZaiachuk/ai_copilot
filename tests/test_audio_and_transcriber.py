"""Tests for audio resampling, chunking, Ukrainian ASR filtering, and CUDA fault fallback."""

from __future__ import annotations

import numpy as np

import audio_capture
import config
import transcriber


def test_resample_48k_float32_and_44k_int16_to_16k_mono() -> None:
    src_48k = np.ones(48000 * 2, dtype=np.float32) * 0.25
    res_48k = audio_capture.resample_to_16k_mono(src_48k, src_rate=48000, channels=2)
    assert res_48k.dtype == np.float32 and res_48k.ndim == 1
    assert res_48k.size == 16000
    assert abs(float(np.mean(res_48k[1000:-1000])) - 0.25) < 0.01

    src_44k = (np.ones(44100 * 2, dtype=np.float32) * 16384).astype(np.int16)
    res_44k = audio_capture.resample_to_16k_mono(src_44k, src_rate=44100, channels=2)
    assert res_44k.dtype == np.float32 and res_44k.ndim == 1
    assert abs(res_44k.size - 16000) <= 5
    assert abs(float(np.mean(res_44k[1000:-1000])) - 0.5) < 0.02


def test_audio_chunker_overlap_poll_silence_and_flush() -> None:
    chunker = audio_capture.AudioChunker(
        target_chunk_seconds=2.0,
        overlap_seconds=0.5,
        silence_seconds=0.5,
        silence_threshold=0.01,
        min_speech_seconds=0.4,
        pre_roll_seconds=0.2,
    )
    speech_block = np.ones(16000, dtype=np.float32) * 0.1
    c1, rms1 = chunker.feed(speech_block, now=1.0)
    assert c1 is None and rms1 > 0.05
    c2, _ = chunker.feed(speech_block, now=2.0)
    assert c2 is not None and c2.size == 32000

    half_speech = np.ones(8000, dtype=np.float32) * 0.1
    c3, _ = chunker.feed(half_speech, now=2.5)
    assert c3 is None
    assert chunker.poll_silence(now=2.7) is None
    flushed = chunker.poll_silence(now=3.1)
    assert flushed is not None and flushed.size == 16000

    chunker.feed(half_speech, now=4.0)
    final = chunker.flush()
    assert final is not None and final.size == 8000


def test_hallucination_tags_dedup_and_vocabulary_preservation() -> None:
    assert transcriber.strip_non_speech_tags("[музика] Привіт (оплески) ♪♪") == "Привіт"
    assert transcriber.is_hallucination("[музика]")
    assert transcriber.is_hallucination("Дякую за перегляд!")
    assert transcriber.is_hallucination("так так так так так")
    assert not transcriber.is_hallucination("Яка складність алгоритму Дейкстри?")

    dedup = transcriber.deduplicate_overlap(
        "Які переваги використання Faster Whisper",
        "faster whisper на відеокарті NVIDIA",
    )
    assert dedup == "на відеокарті NVIDIA"

    vocab = ["FastAPI", "WASAPI", "PyQt6", "PostgreSQL", "Алгоритм"]
    norm_text = transcriber.apply_vocabulary(
        "Комп'ютер обробляє запит fastapi через wasapi в postgersql, алгоритму вистачає",
        vocab,
    )
    assert "Комп'ютер" in norm_text
    assert "FastAPI" in norm_text and "WASAPI" in norm_text and "PostgreSQL" in norm_text
    assert "алгоритму" in norm_text


def test_fault_injection_cuda_runtime_fallback_to_cpu(monkeypatch) -> None:
    """Fault injection: CUDA RuntimeError on transcribe triggers deterministic CPU int8 fallback."""
    calls: list[tuple[str, str]] = []

    class _FakeSeg:
        text = "Розкажіть про архітектуру системи."
        no_speech_prob = 0.05
        avg_logprob = -0.2
        compression_ratio = 1.2

    class _FakeModel:
        def __init__(self, model_name: str, device: str, compute_type: str) -> None:
            self.device = device
            self.compute_type = compute_type
            calls.append((device, compute_type))

        def transcribe(self, audio, **kwargs):
            if self.device == "cuda":
                raise RuntimeError("Library cudnn_ops64_9.dll is not found")
            return [_FakeSeg()], None

    monkeypatch.setattr(transcriber, "WhisperModel", _FakeModel)
    statuses: list[str] = []
    wt = transcriber.WhisperTranscriber(
        config.AppConfig(whisper_model="large-v3-turbo", whisper_device="cuda", whisper_compute_type="float16"),
        on_status=statuses.append,
    )
    dev, comp = wt.load_model()
    assert (dev, comp) == ("cuda", "float16")

    text = wt.transcribe(np.ones(16000, dtype=np.float32) * 0.05)
    assert text == "Розкажіть про архітектуру системи."
    assert wt.active_device == "cpu" and wt.active_compute_type == "int8"
    assert calls == [("cuda", "float16"), ("cpu", "int8")]
    assert any("CPU" in s for s in statuses)


def test_ukrainian_asr_enhancements_and_soft_chunking() -> None:
    from types import SimpleNamespace
    from unittest.mock import patch

    import numpy as np

    import audio_capture
    import config
    import transcriber

    # 1. Auto-gain normalization boosts quiet speech but leaves silence and loud speech intact
    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    quiet = 0.015 * np.sin(2 * np.pi * 220 * t)
    quiet_norm = transcriber.normalize_speech_audio(quiet)
    quiet_rms = float(np.sqrt(np.mean(np.square(quiet_norm))))
    assert 0.06 <= quiet_rms <= 0.09

    silence = np.full(sr, 0.0005, dtype=np.float32)
    assert np.allclose(transcriber.normalize_speech_audio(silence), silence)

    loud = 0.25 * np.sin(2 * np.pi * 220 * t)
    assert np.allclose(transcriber.normalize_speech_audio(loud), loud)

    # 2. Soft-cut AudioChunker waits during active speech and cuts on pause valley before max_chunk_seconds
    chunker = audio_capture.AudioChunker(
        sample_rate=16000,
        target_chunk_seconds=2.0,
        overlap_seconds=0.5,
        silence_seconds=0.8,
        silence_threshold=0.01,
        max_chunk_seconds=3.5,
    )
    active_block = np.full(8000, 0.05, dtype=np.float32)  # 0.5s
    valley_block = np.full(8000, 0.004, dtype=np.float32)  # 0.5s < 0.01 * 1.4
    out_chunks = []
    for i in range(4):  # 2.0s active -> holds instead of mid-word hard cut
        res, _ = chunker.feed(active_block, now=(i + 1) * 0.5)
        if res is not None:
            out_chunks.append(res)
    assert len(out_chunks) == 0
    res_valley, _ = chunker.feed(valley_block, now=2.5)  # 2.5s -> pause valley triggers clean cut!
    assert res_valley is not None
    assert len(res_valley) == int(2.5 * 16000)

    # Hard ceiling still cuts at max_chunk_seconds if speaker never pauses
    chunker_hard = audio_capture.AudioChunker(
        sample_rate=16000,
        target_chunk_seconds=2.0,
        overlap_seconds=0.5,
        silence_seconds=0.8,
        silence_threshold=0.01,
        max_chunk_seconds=3.0,
    )
    hard_out = []
    for i in range(6):
        res, _ = chunker_hard.feed(active_block, now=(i + 1) * 0.5)
        if res is not None:
            hard_out.append(res)
    assert len(hard_out) == 1
    assert len(hard_out[0]) == int(3.0 * 16000)

    # 3. Fuzzy overlap deduplication, vocab Cyrillic alias gating, and anchor echo filtering
    assert (
        transcriber.deduplicate_overlap(
            "Система працює на швидкому протоколі HTTP",
            "на швидкому протоколі HTTP версії 3",
        )
        == "версії 3"
    )
    assert (
        transcriber.apply_vocabulary("використовуємо васапі і фастапі", ["WASAPI"])
        == "використовуємо WASAPI і фастапі"
    )
    assert transcriber.is_hallucination("Усний захист і технічна розмова українською мовою.")

    # 4. WhisperTranscriber passes tuned Ukrainian ASR kwargs (hotwords, patience, vad_parameters, repetition_penalty)
    captured_kwargs: dict[str, object] = {}

    class _DummyModel:
        def transcribe(self, audio, **kwargs):
            captured_kwargs.update(kwargs)
            seg = SimpleNamespace(
                text="Перевірка васапі потоку.",
                no_speech_prob=0.05,
                avg_logprob=-0.25,
                compression_ratio=1.2,
            )
            return iter([seg]), SimpleNamespace()

    cfg = config.AppConfig(whisper_device="cuda", whisper_beam_size=5, whisper_language="uk")
    tr = transcriber.WhisperTranscriber(cfg)
    tr.set_vocabulary(["WASAPI", "PyQt6"])
    with patch.object(transcriber, "WhisperModel", return_value=_DummyModel()):
        text = tr.transcribe(quiet)
        assert text == "Перевірка WASAPI потоку."
        assert captured_kwargs["language"] == "uk"
        assert captured_kwargs["beam_size"] == 5
        assert captured_kwargs["patience"] == 1.2
        assert captured_kwargs["temperature"] == (0.0, 0.2)
        assert captured_kwargs["repetition_penalty"] == 1.1
        assert captured_kwargs["no_repeat_ngram_size"] == 3
        assert captured_kwargs["without_timestamps"] is True
        assert "WASAPI" in str(captured_kwargs["hotwords"])
        assert "українською мовою" in str(captured_kwargs["initial_prompt"])
        assert isinstance(captured_kwargs["vad_parameters"], dict)
        assert captured_kwargs["vad_parameters"]["threshold"] == 0.35

        # Second call carries trailing previous text inside initial_prompt
        tr.transcribe(quiet)
        assert "Перевірка WASAPI потоку." in str(captured_kwargs["initial_prompt"])


def test_audio_sanitization_chunker_overlap_flag_and_transcriber_cleanup() -> None:
    from types import SimpleNamespace
    from unittest.mock import patch

    # NaN / unaligned byte stream sanitization (both paFloat32 and paInt16)
    arr = np.array([0.2, np.nan, np.inf, -0.3], dtype=np.float32)
    raw = arr.tobytes() + b"\x00"  # odd trailing byte
    mono = audio_capture._bytes_to_mono_float32(
        raw, audio_capture.pyaudio.paFloat32, channels=2
    )
    assert mono.dtype == np.float32 and mono.size == 2
    assert np.all(np.isfinite(mono))

    raw_i16 = np.array([16384, -16384, 16384, 16384], dtype=np.int16).tobytes()
    mono_i16 = audio_capture._bytes_to_mono_float32(
        raw_i16, audio_capture.pyaudio.paInt16, channels=2
    )
    assert mono_i16.dtype == np.float32 and mono_i16.size == 2
    assert abs(float(mono_i16[0])) < 1e-4 and abs(float(mono_i16[1]) - 0.5) < 1e-3

    # AudioChunker distinguishes target split (has_overlap=True) vs silence flush (has_overlap=False)
    chunker = audio_capture.AudioChunker(
        target_chunk_seconds=2.0,
        overlap_seconds=0.5,
        silence_seconds=0.5,
        silence_threshold=0.01,
        min_speech_seconds=0.4,
        pre_roll_seconds=0.2,
    )
    speech_1s = np.ones(16000, dtype=np.float32) * 0.1
    silence_1s = np.zeros(16000, dtype=np.float32)

    c1, rms1 = chunker.feed(speech_1s, 1.0)
    assert c1 is None and rms1 > 0.05
    c2, _ = chunker.feed(speech_1s, 2.0)
    assert c2 is not None and c2.size == 32000
    assert chunker.last_emit_had_overlap is False  # First chunk has no prior overlap
    c3a, _ = chunker.feed(speech_1s, 3.0)
    assert c3a is None
    c3, _ = chunker.feed(speech_1s, 4.0)
    assert c3 is not None
    assert chunker.last_emit_had_overlap is True  # Subsequent split carries tail overlap

    # Brief click below min_speech_seconds is discarded after trailing silence
    chunker2 = audio_capture.AudioChunker(
        target_chunk_seconds=5.0,
        overlap_seconds=0.5,
        silence_seconds=0.5,
        silence_threshold=0.01,
        min_speech_seconds=0.4,
    )
    c_click, _ = chunker2.feed(np.ones(1600, dtype=np.float32) * 0.1, 1.0)
    assert c_click is None
    c_drop, _ = chunker2.feed(np.zeros(1600, dtype=np.float32), 1.1)
    assert c_drop is None
    assert chunker2.poll_silence(1.8) is None
    assert chunker2.flush() is None

    # Valid speech ended by silence sets last_emit_had_overlap = False
    c_sp, _ = chunker2.feed(speech_1s, 3.0)
    assert c_sp is None
    c_flushed, _ = chunker2.feed(silence_1s, 4.0)
    assert c_flushed is not None
    assert chunker2.last_emit_had_overlap is False

    # WhisperTranscriber: Node.js/C# vocab before dedup, hallucination drop per segment, unload_model
    class _SegModel:
        def transcribe(self, audio, **kwargs):
            return (
                iter(
                    [
                        SimpleNamespace(
                            text="дякую за перегляд",
                            no_speech_prob=0.1,
                            avg_logprob=-0.2,
                            compression_ratio=1.1,
                        ),
                        SimpleNamespace(
                            text="сервіс на node.js і c#",
                            no_speech_prob=0.1,
                            avg_logprob=-0.2,
                            compression_ratio=1.1,
                        ),
                    ]
                ),
                SimpleNamespace(),
            )

    tr = transcriber.WhisperTranscriber(config.AppConfig(whisper_device="cpu"))
    tr.set_vocabulary(["Node.js", "C#"])
    with patch.object(transcriber, "WhisperModel", return_value=_SegModel()):
        out1 = tr.transcribe(speech_1s * 0.3, has_overlap=False)
        assert "дякую" not in out1.lower()
        assert "Node.js" in out1 and "C#" in out1

        # has_overlap=False skips cross-chunk suffix dedup
        tr2 = transcriber.WhisperTranscriber(config.AppConfig(whisper_device="cpu"))

        class _RepeatModel:
            def transcribe(self, audio, **kwargs):
                return (
                    iter(
                        [
                            SimpleNamespace(
                                text="так",
                                no_speech_prob=0.05,
                                avg_logprob=-0.2,
                                compression_ratio=1.0,
                            )
                        ]
                    ),
                    SimpleNamespace(),
                )

        with patch.object(transcriber, "WhisperModel", return_value=_RepeatModel()):
            assert tr2.transcribe(speech_1s * 0.3, has_overlap=False) == "так"
            assert tr2.transcribe(speech_1s * 0.3, has_overlap=False) == "так"
        tr2.unload_model()
        assert tr2._model is None


def test_wasapi_run_loop_with_on_chunk_meta_only() -> None:
    import threading
    from unittest.mock import patch
    import pytest

    capturer = audio_capture.WasapiLoopbackCapturer(
        device_index=None,
        chunk_seconds=1.2,
        overlap_seconds=0.3,
        silence_seconds=0.4,
        silence_threshold=0.01,
    )

    with pytest.raises(ValueError):
        capturer.run_loop(stop_event=threading.Event())

    speech_block = (np.ones(16000 * 2, dtype=np.float32) * 0.2).tobytes()

    class _FakeStream:
        def __init__(self, callback) -> None:
            self._cb = callback
            self._active = True

        def start_stream(self) -> None:
            self._cb(speech_block, 16000, None, 0)
            self._cb(speech_block, 16000, None, 0)

        def is_active(self) -> bool:
            return self._active

        def stop_stream(self) -> None:
            self._active = False

        def close(self) -> None:
            self._active = False

    class _FakePyAudio:
        def get_default_wasapi_loopback(self) -> dict:
            return {
                "index": 1,
                "name": "Fake Loopback",
                "isLoopbackDevice": True,
                "maxInputChannels": 2,
                "defaultSampleRate": 16000,
            }

        def open(self, *, stream_callback, **_kwargs):
            return _FakeStream(stream_callback)

        def terminate(self) -> None:
            pass

    stop_event = threading.Event()
    safety_timer = threading.Timer(3.0, stop_event.set)
    safety_timer.start()
    received: list[tuple[np.ndarray, bool]] = []
    levels: list[float] = []

    def _on_chunk_meta(chunk_16k: np.ndarray, had_overlap: bool) -> None:
        received.append((chunk_16k, had_overlap))
        stop_event.set()

    try:
        with patch.object(audio_capture.pyaudio, "PyAudio", return_value=_FakePyAudio()):
            capturer.run_loop(
                stop_event=stop_event,
                on_chunk_meta=_on_chunk_meta,
                on_level=levels.append,
            )
    finally:
        safety_timer.cancel()

    assert len(received) == 1
    assert received[0][0].dtype == np.float32 and received[0][0].size >= 16000
    assert received[0][1] is False
    assert any(lvl > 0.1 for lvl in levels)
