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
        assert captured_kwargs["temperature"] == (0.0, 0.2, 0.4)
        assert captured_kwargs["repetition_penalty"] == 1.02
        assert captured_kwargs["no_repeat_ngram_size"] == 0
        assert captured_kwargs["without_timestamps"] is True
        assert "WASAPI" in str(captured_kwargs["hotwords"])
        assert "українською мовою" in str(captured_kwargs["initial_prompt"])
        assert isinstance(captured_kwargs["vad_parameters"], dict)
        assert captured_kwargs["vad_parameters"]["threshold"] == 0.30

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
        with (
            patch.object(audio_capture.pyaudio, "PyAudio", return_value=_FakePyAudio()),
            patch.object(
                audio_capture,
                "StreamingSileroVAD",
                return_value=audio_capture.StreamingSileroVAD.__new__(audio_capture.StreamingSileroVAD),
            ) as mock_vad_cls,
        ):
            mock_vad_cls.return_value.available = False
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


def test_silero_vad_raw_streaming_ct2_tokenizer_and_cloud_stt(tmp_path) -> None:
    import json
    import os
    import queue
    import threading
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    # 1. StreamingSileroVAD scores digital silence low and AudioChunker respects vad_fn hysteresis
    vad = audio_capture.StreamingSileroVAD()
    assert vad.available is True
    assert vad.score_block(np.zeros(1024, dtype=np.float32)) < 0.25

    vad_probs = [0.85, 0.80, 0.30, 0.10]  # drops below release threshold 0.25 only on block 4
    idx_box = [0]

    def _fake_vad(_blk):
        val = vad_probs[min(idx_box[0], len(vad_probs) - 1)]
        idx_box[0] += 1
        return val

    vad_chunker = audio_capture.AudioChunker(
        target_chunk_seconds=2.0,
        overlap_seconds=0.5,
        silence_seconds=0.5,
        max_chunk_seconds=6.0,
        vad_fn=_fake_vad,
    )
    blk_1s = np.full(16000, 0.08, dtype=np.float32)
    assert vad_chunker.feed(blk_1s, 1.0)[0] is None
    assert vad_chunker.feed(blk_1s, 2.0)[0] is None  # prob=0.80 still active speech -> holds past 2s
    assert vad_chunker.feed(blk_1s, 3.0)[0] is None  # prob=0.30 above release 0.25 -> holds
    emitted, _ = vad_chunker.feed(blk_1s, 4.0)  # prob=0.10 -> releases clean utterance at pause!
    assert emitted is not None and emitted.size == 4 * 16000

    # 2. WasapiLoopbackCapturer.run_loop supports on_raw_16k for real-time streaming
    raw_frames: list[np.ndarray] = []
    stop_ev = threading.Event()
    speech_bytes = (np.ones(16000, dtype=np.float32) * 0.2).tobytes()

    class _RawStream:
        def __init__(self, cb) -> None:
            self._cb = cb
            self._active = True

        def start_stream(self) -> None:
            self._cb(speech_bytes, 8000, None, 0)

        def is_active(self) -> bool:
            return self._active

        def stop_stream(self) -> None:
            self._active = False

        def close(self) -> None:
            self._active = False

    class _RawPyAudio:
        def get_default_wasapi_loopback(self) -> dict:
            return {
                "index": 2,
                "name": "Fake Raw",
                "isLoopbackDevice": True,
                "maxInputChannels": 2,
                "defaultSampleRate": 16000,
            }

        def open(self, *, stream_callback, **_kw):
            return _RawStream(stream_callback)

        def terminate(self) -> None:
            pass

    def _on_raw(blk: np.ndarray) -> None:
        raw_frames.append(blk)
        stop_ev.set()

    with patch.object(audio_capture.pyaudio, "PyAudio", return_value=_RawPyAudio()):
        audio_capture.WasapiLoopbackCapturer().run_loop(stop_event=stop_ev, on_raw_16k=_on_raw)
    assert len(raw_frames) == 1 and raw_frames[0].dtype == np.float32

    # 3. _ensure_ct2_tokenizer copies Whisper Turbo tokenizer when CT2 repo lacks tokenizer.json
    ct2_dir = tmp_path / "ct2_uk"
    ct2_dir.mkdir()
    (ct2_dir / "vocabulary.json").write_text(json.dumps(["a"] * 51866), encoding="utf-8")
    fake_tok = tmp_path / "cached_tok.json"
    fake_tok.write_text('{"version":"1.0"}', encoding="utf-8")
    with patch("huggingface_hub.hf_hub_download", return_value=str(fake_tok)):
        transcriber._ensure_ct2_tokenizer(str(ct2_dir))
    assert (ct2_dir / "tokenizer.json").is_file()

    # 4. CloudSTTTranscriber supports Groq (key reuse) and ElevenLabs (language_code='ukr')
    groq_cfg = config.AppConfig(
        stt_provider="groq",
        llm_provider="groq",
        openai_api_key="gsk_reuse_key_1234567890",
        whisper_language="uk",
    )
    assert transcriber.resolve_stt_api_key(groq_cfg) == "gsk_reuse_key_1234567890"
    fake_oai = MagicMock()
    fake_oai.audio.transcriptions.create.return_value = SimpleNamespace(text="Перевірка грок")
    cloud_groq = transcriber.CloudSTTTranscriber(groq_cfg)
    cloud_groq.set_vocabulary(["Groq"])
    with patch("openai.OpenAI", return_value=fake_oai):
        txt = cloud_groq.transcribe(np.full(16000, 0.08, dtype=np.float32), has_overlap=False)
    assert txt == "Перевірка Groq"
    assert fake_oai.audio.transcriptions.create.call_args.kwargs["language"] == "uk"

    el_cfg = config.AppConfig(
        stt_provider="elevenlabs",
        stt_api_key="el_secret_key_999888",
        whisper_language="uk",
    )
    fake_el_client = MagicMock()
    fake_el_client.speech_to_text.convert.return_value = SimpleNamespace(text="Текст через васапі")
    cloud_el = transcriber.CloudSTTTranscriber(el_cfg)
    with patch.dict(os.environ, {}, clear=False):
        real_el = cloud_el._get_elevenlabs_client("el_secret_key_999888")
        assert real_el._client_wrapper.get_headers().get("xi-api-key") == "el_secret_key_999888"
    cloud_el.unload_model()

    cloud_el.set_vocabulary(["WASAPI", "List<T>", "arr[0]", "a\\b"])
    with (
        patch.dict(os.environ, {}, clear=False),
        patch.object(transcriber, "ElevenLabs", return_value=fake_el_client) as el_cls,
    ):
        el_txt = cloud_el.transcribe(np.full(16000, 0.08, dtype=np.float32), has_overlap=False)
    assert el_txt == "Текст через WASAPI"
    assert el_cls.call_args.kwargs["api_key"] == "el_secret_key_999888"
    el_kwargs = fake_el_client.speech_to_text.convert.call_args.kwargs
    assert el_kwargs["model_id"] == "scribe_v3"
    assert el_kwargs["language_code"] == "ukr"
    assert el_kwargs["tag_audio_events"] is False
    assert el_kwargs["timestamps_granularity"] == "none"
    assert el_kwargs["keyterms"] == ["WASAPI"]

    formatted_err = transcriber._format_cloud_stt_error(
        "elevenlabs",
        SimpleNamespace(
            status_code=401,
            body={"detail": {"message": "Invalid API key el_secret_key_999888"}},
        ),
        "el_secret_key_999888",
    )
    assert "HTTP 401" in formatted_err
    assert "el_secret_key_999888" not in formatted_err
    assert "headers:" not in formatted_err

    el_v1_cfg = config.AppConfig(
        stt_provider="elevenlabs",
        stt_api_key="el_secret_key_999888",
        stt_model="scribe_v1",
        whisper_language="uk",
    )
    fake_el_v1 = MagicMock()
    fake_el_v1.speech_to_text.convert.return_value = SimpleNamespace(text="Тест")
    cloud_el_v1 = transcriber.CloudSTTTranscriber(el_v1_cfg)
    cloud_el_v1.set_vocabulary(["WASAPI"])
    with (
        patch.dict(os.environ, {}, clear=False),
        patch.object(transcriber, "ElevenLabs", return_value=fake_el_v1),
    ):
        cloud_el_v1.transcribe(np.full(16000, 0.08, dtype=np.float32), has_overlap=False)
    assert "keyterms" not in fake_el_v1.speech_to_text.convert.call_args.kwargs

    # 5. DeepgramRealtimeClient URL + streaming session + CloseStream + verify_stt_provider
    dg_cfg = config.AppConfig(
        stt_provider="deepgram",
        stt_api_key="dg_secret_key_112233",
        whisper_language="uk",
        audio_silence_seconds=0.8,
    )
    ws_url = transcriber.DeepgramRealtimeClient.build_ws_url(dg_cfg, ["PyQt6", "WASAPI"])
    assert "model=nova-3" in ws_url and "language=uk" in ws_url and "keyterm=PyQt6" in ws_url

    dg_client = transcriber.DeepgramRealtimeClient(dg_cfg)
    stop_sentinel = object()
    sent_payloads: list[object] = []

    class _FakeWS:
        def __init__(self) -> None:
            self._msgs = [
                json.dumps(
                    {
                        "type": "Results",
                        "is_final": True,
                        "channel": {"alternatives": [{"transcript": "Слава Україні, тестуємо васапі"}]},
                    }
                )
            ]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def send(self, payload) -> None:
            sent_payloads.append(payload)

        def __iter__(self):
            return iter(self._msgs)

    dg_client.set_vocabulary(["WASAPI"])
    dg_texts: list[str] = []
    dg_statuses: list[str] = []
    dg_errors: list[str] = []
    dg_stop = threading.Event()

    # Wait until first transcript then trigger stop
    q_single: queue.Queue = queue.Queue()
    q_single.put(b"\x01\x02" * 512)
    q_single.put(stop_sentinel)
    with patch.object(transcriber, "_ws_connect", return_value=_FakeWS()):
        dg_client.run_session(
            q_single,
            dg_stop,
            on_transcript=dg_texts.append,
            on_status=dg_statuses.append,
            on_error=dg_errors.append,
            stop_sentinel=stop_sentinel,
        )
    assert dg_texts == ["Слава Україні, тестуємо WASAPI"]
    assert not dg_errors
    assert any(isinstance(p, str) and "CloseStream" in p for p in sent_payloads)

    dg_check_resp = MagicMock(status_code=200)
    dg_check_resp.raise_for_status.return_value = None
    with patch.object(transcriber.requests, "get", return_value=dg_check_resp):
        ok, msg = transcriber.verify_stt_provider(dg_cfg)
    assert ok is True and "Deepgram" in msg


def test_wasapi_run_loop_mid_capture_flush_event() -> None:
    import threading
    from unittest.mock import patch

    capturer = audio_capture.WasapiLoopbackCapturer(
        device_index=None,
        chunk_seconds=5.0,
        overlap_seconds=0.5,
        silence_seconds=3.0,
        silence_threshold=0.01,
    )
    # Inject 0.6s (< target_chunk_seconds=5.0s and silence_seconds=3.0s) so only flush_event can emit it promptly
    short_speech = (np.ones(int(16000 * 0.6), dtype=np.float32) * 0.15).tobytes()
    flush_ev = threading.Event()
    stop_ev = threading.Event()
    chunks_out: list[np.ndarray] = []
    flushed_flags: list[bool] = []
    pending_seen: list[bool] = []

    class _FlushStream:
        def __init__(self, callback) -> None:
            self._cb = callback
            self._active = True

        def start_stream(self) -> None:
            self._cb(short_speech, 9600, None, 0)
            threading.Timer(0.06, flush_ev.set).start()

        def is_active(self) -> bool:
            return self._active

        def stop_stream(self) -> None:
            self._active = False

        def close(self) -> None:
            self._active = False

    class _FlushPyAudio:
        def get_default_wasapi_loopback(self) -> dict:
            return {
                "index": 1,
                "name": "Flush Loopback",
                "isLoopbackDevice": True,
                "maxInputChannels": 1,
                "defaultSampleRate": 16000,
            }

        def open(self, *, stream_callback, **_kw):
            return _FlushStream(stream_callback)

        def terminate(self) -> None:
            pass

    def _on_chunk_meta(chunk_16k: np.ndarray, _had_ov: bool) -> None:
        chunks_out.append(chunk_16k)

    def _on_flushed(emitted: bool) -> None:
        flushed_flags.append(emitted)
        stop_ev.set()

    safety = threading.Timer(2.5, stop_ev.set)
    safety.start()
    try:
        stub_vad = audio_capture.StreamingSileroVAD.__new__(audio_capture.StreamingSileroVAD)
        stub_vad.available = False
        with (
            patch.object(audio_capture.pyaudio, "PyAudio", return_value=_FlushPyAudio()),
            patch.object(audio_capture, "StreamingSileroVAD", return_value=stub_vad),
        ):
            capturer.run_loop(
                stop_event=stop_ev,
                on_chunk_meta=_on_chunk_meta,
                on_level=lambda _lvl: pending_seen.append(capturer.has_pending_speech),
                flush_event=flush_ev,
                on_flushed=_on_flushed,
            )
    finally:
        safety.cancel()

    assert flushed_flags == [True]
    assert len(chunks_out) == 1
    assert any(pending_seen)
    assert capturer.has_pending_speech is False


def test_transcriber_thread_ignores_stale_stop_sentinel_and_non_array_items() -> None:
    import queue
    from PyQt6.QtCore import QCoreApplication
    import main

    _app = QCoreApplication.instance() or QCoreApplication([])

    chunk_q: queue.Queue = queue.Queue()
    raw_q: queue.Queue = queue.Queue()
    dg_thread = main.TranscriberThread(
        config.AppConfig(stt_provider="deepgram"),
        chunk_queue=chunk_q,
        raw_pcm_queue=raw_q,
    )
    dg_thread.stop()
    assert chunk_q.empty() is True
    assert isinstance(raw_q.get_nowait(), main._QueueStopSentinel)

    el_chunk_q: queue.Queue = queue.Queue()
    el_raw_q: queue.Queue = queue.Queue()
    el_chunk_q.put(main._QueueStopSentinel())
    el_chunk_q.put(object())
    el_chunk_q.put((np.full(16000, 0.08, dtype=np.float32), False))
    el_chunk_q.put(None)

    el_thread = main.TranscriberThread(
        config.AppConfig(stt_provider="elevenlabs"),
        chunk_queue=el_chunk_q,
        raw_pcm_queue=el_raw_q,
    )
    transcripts: list[str] = []
    errors: list[str] = []
    el_thread.transcript_ready.connect(transcripts.append)
    el_thread.error_occurred.connect(errors.append)

    class _StubRecognizer:
        def transcribe(self, audio: np.ndarray, *, has_overlap: bool = True) -> str:
            assert isinstance(audio, np.ndarray)
            assert has_overlap is False
            return "Тест пройдено"

    el_thread._run_chunked_recognizer(_StubRecognizer())
    QCoreApplication.processEvents()
    assert errors == []
    assert transcripts == ["Тест пройдено"]


def test_mixed_en_uk_stt_language_routing() -> None:
    import os
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    # 1. ElevenLabs scribe_v3 sends primary 'eng' + secondary ['ukr']
    el_v3 = transcriber.CloudSTTTranscriber(
        config.AppConfig(
            stt_provider="elevenlabs",
            stt_api_key="el_mixed_key_123456",
            stt_model="scribe_v3",
            whisper_language="en+uk",
        )
    )
    fake_v3 = MagicMock()
    fake_v3.speech_to_text.convert.return_value = SimpleNamespace(text="Hello світ")
    with (
        patch.dict(os.environ, {}, clear=False),
        patch.object(transcriber, "ElevenLabs", return_value=fake_v3),
    ):
        txt = el_v3.transcribe(np.full(16000, 0.08, dtype=np.float32), has_overlap=False)
    assert txt == "Hello світ"
    v3_kwargs = fake_v3.speech_to_text.convert.call_args.kwargs
    assert v3_kwargs["language_code"] == "eng"
    assert v3_kwargs["secondary_language_codes"] == ["ukr"]

    # 2. ElevenLabs scribe_v2 omits both for automatic multilingual detection
    el_v2 = transcriber.CloudSTTTranscriber(
        config.AppConfig(
            stt_provider="elevenlabs",
            stt_api_key="el_mixed_key_123456",
            stt_model="scribe_v2",
            whisper_language="en+uk",
        )
    )
    fake_v2 = MagicMock()
    fake_v2.speech_to_text.convert.return_value = SimpleNamespace(text="Hello")
    with (
        patch.dict(os.environ, {}, clear=False),
        patch.object(transcriber, "ElevenLabs", return_value=fake_v2),
    ):
        el_v2.transcribe(np.full(16000, 0.08, dtype=np.float32), has_overlap=False)
    v2_kwargs = fake_v2.speech_to_text.convert.call_args.kwargs
    assert "language_code" not in v2_kwargs
    assert "secondary_language_codes" not in v2_kwargs

    # 3. Groq Whisper omits language kwarg for auto bilingual decoding
    groq_mixed = transcriber.CloudSTTTranscriber(
        config.AppConfig(
            stt_provider="groq",
            llm_provider="groq",
            openai_api_key="gsk_mixed_key_123456",
            whisper_language="en+uk",
        )
    )
    fake_groq = MagicMock()
    fake_groq.audio.transcriptions.create.return_value = SimpleNamespace(text="Mixed text")
    with patch("openai.OpenAI", return_value=fake_groq):
        groq_mixed.transcribe(np.full(16000, 0.08, dtype=np.float32), has_overlap=False)
    assert "language" not in fake_groq.audio.transcriptions.create.call_args.kwargs

    # 4. Deepgram uses language=multi for mixed mode
    dg_url = transcriber.DeepgramRealtimeClient.build_ws_url(
        config.AppConfig(stt_provider="deepgram", whisper_language="en+uk")
    )
    assert "language=multi" in dg_url
