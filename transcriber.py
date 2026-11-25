"""Local GPU/CPU Ukrainian speech transcription using faster-whisper."""

from __future__ import annotations

from functools import lru_cache
import gc
import io
import json
import logging
import os
import queue
import re
import shutil
import threading
import time
from typing import Any, Callable, Iterable
from urllib.parse import urlencode
import wave

from dotenv import load_dotenv
from elevenlabs import ElevenLabs
import numpy as np
import requests
from websockets.sync.client import connect as _ws_connect

from config import AppConfig, get_env_file_path, register_cuda_dlls
from security import redact_secrets

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Register pip-installed or bundled NVIDIA cuBLAS / cuDNN DLLs BEFORE importing faster_whisper.
register_cuda_dlls()

from faster_whisper import WhisperModel  # noqa: E402

UK_FINE_TUNED_WHISPER_REPO = (
    "skypro1111/whisper-large-v3-turbo-ukrainian-ukraine-3percent-ct2"
)
STT_DEFAULT_MODELS: dict[str, str] = {
    "deepgram": "nova-3",
    "groq": "whisper-large-v3",
    "elevenlabs": "scribe_v3",
}
_ISO639_1_TO_3: dict[str, str] = {
    "uk": "ukr",
    "en": "eng",
    "ru": "rus",
    "pl": "pol",
    "de": "deu",
}
_EL_DISALLOWED_KEYTERM_CHARS = frozenset("<>{}[]\\")


def _format_cloud_stt_error(prov: str, exc: Exception, api_key: str) -> str:
    """Format concise, redacted cloud STT error without dumping raw SDK HTTP headers."""
    status_code = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    detail_msg = ""
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict):
            detail_msg = str(detail.get("message") or detail.get("status") or "").strip()
        elif isinstance(detail, str):
            detail_msg = detail.strip()
        elif "message" in body:
            detail_msg = str(body.get("message") or "").strip()
    elif isinstance(body, str) and body.strip():
        detail_msg = body.strip()[:240]

    raw_msg = f"HTTP {status_code}: {detail_msg}" if (status_code is not None and detail_msg) else str(exc)
    safe_msg = redact_secrets(raw_msg, extra_secrets=(api_key,))
    return f"Помилка {prov} STT: {safe_msg}"


def _ensure_ct2_tokenizer(model_dir: str) -> None:
    """Populate missing tokenizer.json in community v3/v3-turbo CT2 repos from Whisper Turbo."""
    if not model_dir or not os.path.isdir(model_dir):
        return
    tok_path = os.path.join(model_dir, "tokenizer.json")
    if os.path.isfile(tok_path):
        return
    vocab_path = os.path.join(model_dir, "vocabulary.json")
    if not os.path.isfile(vocab_path):
        return
    try:
        with open(vocab_path, encoding="utf-8") as fh:
            vocab_data = json.load(fh)
        if not isinstance(vocab_data, (list, dict)) or len(vocab_data) < 51866:
            return
        from huggingface_hub import hf_hub_download

        src_tok = hf_hub_download("openai/whisper-large-v3-turbo", "tokenizer.json")
        if src_tok and os.path.isfile(src_tok):
            shutil.copy2(src_tok, tok_path)
    except Exception:
        pass


def _resolve_cached_whisper_model(model_name: str) -> str:
    """Return local CT2 directory when cached (or HF repo), avoiding redundant network checks."""
    if not model_name:
        return model_name
    if os.path.isdir(model_name):
        _ensure_ct2_tokenizer(model_name)
        return model_name
    try:
        from faster_whisper.utils import download_model

        cached_dir = download_model(model_name, local_files_only=True)
        if cached_dir and os.path.isdir(str(cached_dir)):
            _ensure_ct2_tokenizer(str(cached_dir))
            return str(cached_dir)
        if "/" in model_name:
            downloaded_dir = download_model(model_name, local_files_only=False)
            if downloaded_dir and os.path.isdir(str(downloaded_dir)):
                _ensure_ct2_tokenizer(str(downloaded_dir))
                return str(downloaded_dir)
    except Exception:
        pass
    return model_name

logger = logging.getLogger(__name__)

_PUNCT_STRIP_RE = re.compile(r"^[\s.,!?;:\"'«»„“()\[\]—–\-…]+|[\s.,!?;:\"'«»„“()\[\]—–\-…]+$")
_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яІіЇїЄєҐґ'’-]+")
_VOCAB_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яІіЇїЄєҐґ'’]+")
_NON_SPEECH_TAG_RE = re.compile(
    r"[\[({（【][^\]\)}）】]*[\]\)}）】]|\*[^*]+\*|[♪♫🎶🎵]+"
)

_UK_PROMPT_ANCHOR = "Усний захист і технічна розмова українською мовою."

_CYRILLIC_TECH_ALIASES: dict[str, tuple[str, ...]] = {
    "fastapi": ("фастапі", "фастейпіай"),
    "python": ("пайтон", "пітон"),
    "pyqt6": ("пайкюті", "пайкют6"),
    "wasapi": ("васапі",),
    "postgresql": ("постгрес", "постгресескюель"),
    "docker": ("докер",),
    "github": ("гітхаб",),
    "kubernetes": ("кубернетес",),
    "javascript": ("джаваскрипт",),
    "typescript": ("тайпскрипт",),
    "react": ("реакт",),
    "linux": ("лінукс",),
    "groq": ("грок", "гроук"),
    "whisper": ("віспер",),
    "c#": ("сішарп", "сі-шарп"),
    "dotnet": ("дотнет",),
    "sql": ("ескюель", "сиквел"),
    "json": ("джейсон",),
    "async": ("асінк",),
    "linq": ("лінк",),
}

HALLUCINATION_EXACT_PHRASES: frozenset[str] = frozenset(
    {
        "дякую за перегляд",
        "дякую за увагу",
        "дякую всім за перегляд",
        "дякуємо за перегляд",
        "дякую за прослуховування",
        "дякую за перегляд цього відео",
        "підписуйтесь на канал",
        "підписуйтесь на мій канал",
        "підпишіться на канал",
        "не забудьте підписатися",
        "ставте лайк",
        "ставте лайки",
        "продовження слідує",
        "далі буде",
        "до нових зустрічей",
        "субтитри зробив",
        "субтитри створив",
        "редактор субтитрів",
        "субтитри підготував",
        "переклад субтитрів",
        "субтитрування",
        "субтитри",
        "озвучено",
        "діма торжок",
        "музика",
        "оплески",
        "усний захист і технічна розмова українською мовою",
        "усний захист і технічна розмова українською",
        "thank you for watching",
        "thanks for watching",
        "subscribe to the channel",
        "please subscribe",
        "mbc",
    }
)

HALLUCINATION_SUBSTRINGS: tuple[str, ...] = (
    "субтитри зробив",
    "субтитри створив",
    "редактор субтитрів",
    "субтитри підготував",
    "переклад субтитрів",
    "субтитрування",
    "озвучено",
    "діма торжок",
    "підписуйтесь на канал",
    "підписуйтесь на мій канал",
    "не забудьте підписатися",
    "дякую за перегляд",
    "дякую за прослуховування",
    "дякую за перегляд цього відео",
    "усний захист і технічна розмова українською",
    "thank you for watching",
    "thanks for watching",
)


def normalize_speech_audio(audio: np.ndarray) -> np.ndarray:
    """Gently normalize quiet speech chunks without amplifying background silence or clipping."""
    arr = np.nan_to_num(
        np.asarray(audio, dtype=np.float32).reshape(-1),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )
    if arr.size == 0:
        return arr
    if float(np.ptp(arr)) > 1e-6:
        mean_val = float(np.mean(arr, dtype=np.float64))
        if abs(mean_val) > 1e-4:
            arr = arr - mean_val
    rms = float(np.sqrt(np.mean(np.square(arr, dtype=np.float64))))
    if rms < 0.002 or rms >= 0.065:
        return np.ascontiguousarray(arr, dtype=np.float32)
    peak = float(np.max(np.abs(arr)))
    max_safe_gain = (0.92 / peak) if peak > 1e-6 else 6.0
    gain = min(6.0, 0.075 / rms, max_safe_gain)
    boosted = np.clip(arr * gain, -0.95, 0.95)
    return np.ascontiguousarray(boosted, dtype=np.float32)


def strip_non_speech_tags(text: str) -> str:
    """Remove bracketed sound annotations such as '[музика]', '(оплески)', and notes."""
    if not text:
        return ""
    cleaned = _NON_SPEECH_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalize_phrase(text: str) -> str:
    cleaned = _PUNCT_STRIP_RE.sub("", text.strip().lower())
    return re.sub(r"\s+", " ", cleaned)


def _normalize_word(word: str) -> str:
    return _PUNCT_STRIP_RE.sub("", word.lower())


def _levenshtein(a: str, b: str) -> int:
    n = len(b)
    if not a:
        return n
    if n == 0:
        return len(a)
    dp = list(range(n + 1))
    for i, ca in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            tmp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if ca == cb else 1))
            prev = tmp
    return dp[n]


@lru_cache(maxsize=8)
def _compile_vocab(
    vocab_tuple: tuple[str, ...],
) -> tuple[re.Pattern[str] | None, dict[str, str], list[tuple[str, str, bool]], dict[str, str]]:
    terms: list[tuple[str, str, bool]] = []
    seen_keys: set[str] = set()
    cyrillic_map: dict[str, str] = {}
    special_phrases: dict[str, str] = {}

    for term in vocab_tuple:
        canon = term.strip()
        if not canon:
            continue
        if len(canon) >= 2 and re.search(r"[\s.+\-#]", canon):
            special_phrases.setdefault(canon.lower(), canon)
        key = re.sub(r"[^0-9A-Za-zА-Яа-яІіЇїЄєҐґ'’]+", "", canon).lower()
        if len(key) < 2 or key in seen_keys:
            continue
        seen_keys.add(key)
        terms.append((canon, key, key.isascii()))
        for alias in _CYRILLIC_TECH_ALIASES.get(key, ()):
            cyrillic_map.setdefault(alias, canon)

    special_re: re.Pattern[str] | None = None
    if special_phrases:
        ordered = sorted(special_phrases, key=len, reverse=True)
        joined = "|".join(re.escape(p) for p in ordered)
        special_re = re.compile(
            rf"(?<![0-9A-Za-zА-Яа-яІіЇїЄєҐґ_])(?:{joined})(?![0-9A-Za-zА-Яа-яІіЇїЄєҐґ_])",
            re.IGNORECASE,
        )

    return special_re, special_phrases, terms, cyrillic_map


def apply_vocabulary(text: str, vocab: Iterable[str]) -> str:
    """Normalize casing/spelling of recognized technical terms and Cyrillic aliases from context.

    Never inserts unspoken words. Fuzzy matching and Cyrillic IT aliases activate only
    for canonical terms present in `vocab`, keeping Ukrainian inflections intact.
    """
    if not text:
        return text

    vocab_tuple = tuple(t.strip() for t in vocab if t and t.strip())
    if not vocab_tuple:
        return text

    special_re, special_phrases, terms, cyrillic_map = _compile_vocab(vocab_tuple)
    updated = text
    if special_re is not None:
        updated = special_re.sub(
            lambda m: special_phrases.get(m.group(0).lower(), m.group(0)),
            updated,
        )

    if not terms and not cyrillic_map:
        return updated

    def _replace(match: re.Match[str]) -> str:
        word = match.group(0)
        key = word.lower()
        if len(key) < 2:
            return word
        alias_hit = cyrillic_map.get(key)
        if alias_hit is not None:
            return alias_hit
        for canon, term_key, is_ascii_term in terms:
            if key == term_key:
                return canon
            if (
                is_ascii_term
                and key.isascii()
                and len(term_key) >= 6
                and abs(len(key) - len(term_key)) <= 2
            ):
                dist = _levenshtein(key, term_key)
                if 0 < dist <= len(term_key) // 5:
                    return canon
        return word

    return _VOCAB_TOKEN_RE.sub(_replace, updated)


def is_hallucination(text: str) -> bool:
    """Detect empty outputs, non-speech tags, Whisper subtitle artifacts, and token loops."""
    if not text or not text.strip():
        return True

    stripped = strip_non_speech_tags(text)
    if not stripped:
        return True

    norm = _normalize_phrase(stripped)
    if not norm:
        return True

    words = _WORD_RE.findall(norm)
    if not words:
        return True

    if len(norm) <= 1 and not norm.isdigit():
        return True

    if norm in HALLUCINATION_EXACT_PHRASES:
        return True

    for phrase in HALLUCINATION_SUBSTRINGS:
        if phrase in norm:
            return True

    # Detect single-word repetition >= 4 times consecutively (e.g. "так так так так").
    run_len = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run_len += 1
            if run_len >= 4:
                return True
        else:
            run_len = 1

    # Detect repeated 2-word loops >= 3 times consecutively (e.g. "або так або так або так").
    if len(words) >= 6:
        for i in range(len(words) - 5):
            if (
                words[i] == words[i + 2] == words[i + 4]
                and words[i + 1] == words[i + 3] == words[i + 5]
            ):
                return True

    # Low lexical diversity across longer segments usually indicates a hallucination loop.
    if len(words) >= 6:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return True

    return False


def _words_similar(a: str, b: str, *, allow_fuzzy: bool = True) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    if not allow_fuzzy:
        return False
    if len(a) >= 5 and len(b) >= 5:
        if a[:5] == b[:5]:
            return True
        if abs(len(a) - len(b)) <= 1 and _levenshtein(a, b) <= 1:
            return True
    return False


def deduplicate_overlap(prev_text: str, new_text: str, max_words: int = 8) -> str:
    """Strip duplicated boundary words caused by overlapping audio chunks."""
    new_stripped = new_text.strip()
    if not prev_text or not new_stripped:
        return new_stripped

    prev_tokens = prev_text.strip().split()
    new_tokens = new_stripped.split()
    if not prev_tokens or not new_tokens:
        return new_stripped

    prev_words = [w for w in (_normalize_word(t) for t in prev_tokens) if w]
    new_pairs = [(idx, _normalize_word(t)) for idx, t in enumerate(new_tokens)]
    new_word_pairs = [(idx, w) for idx, w in new_pairs if w]
    if not prev_words or not new_word_pairs:
        return new_stripped

    limit = min(max_words, len(prev_words), len(new_word_pairs))
    for k in range(limit, 0, -1):
        head_pairs = new_word_pairs[:k]
        head_words = [w for _, w in head_pairs]
        allow_fuzzy = k >= 2
        candidate_windows = [prev_words[-k:]]
        if k >= 2 and len(prev_words) >= k + 1:
            candidate_windows.append(prev_words[-k - 1 : -1])

        for tail_words in candidate_windows:
            if all(
                _words_similar(a, b, allow_fuzzy=allow_fuzzy)
                for a, b in zip(tail_words, head_words, strict=True)
            ):
                cut_tok_idx = head_pairs[-1][0] + 1
                return " ".join(new_tokens[cut_tok_idx:]).strip()

    return new_stripped


def _looks_like_cuda_runtime_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "cuda",
            "cudnn",
            "cublas",
            "cufft",
            "curand",
            "nvrtc",
            "dll",
            "library",
            "out of memory",
            "oom",
            "gpu",
            "driver",
        )
    )


_CPU_UNSUPPORTED_COMPUTE_TYPES: frozenset[str] = frozenset(
    {"float16", "int8_float16", "bfloat16", "int8_bfloat16"}
)


class WhisperTranscriber:
    """Manages faster-whisper model lifecycle, CUDA->CPU fallback, and Ukrainian filtering."""

    def __init__(
        self,
        config: AppConfig,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self._on_status = on_status
        self._model: WhisperModel | None = None
        self.active_device: str = config.whisper_device
        self.active_compute_type: str = config.whisper_compute_type
        self._last_text: str = ""
        self._history_epoch: int = 0
        self._state_lock = threading.RLock()
        self._vocabulary: list[str] = []

    def set_vocabulary(self, terms: Iterable[str]) -> None:
        """Update domain vocabulary terms used for post-recognition spelling normalization."""
        cleaned = [t.strip() for t in terms if t and t.strip()]
        with self._state_lock:
            self._vocabulary = cleaned
        _compile_vocab(tuple(cleaned))

    def unload_model(self) -> None:
        """Explicitly release WhisperModel weights and free GPU VRAM."""
        with self._state_lock:
            model = self._model
            self._model = None
        if model is not None:
            try:
                inner = getattr(model, "model", None)
                unload_fn = getattr(inner, "unload_model", None)
                if callable(unload_fn):
                    unload_fn()
            except Exception:
                pass
            del model
            gc.collect()

    def update_config(self, new_config: AppConfig) -> bool:
        """Update transcriber settings and return True if the Whisper model must reload."""
        with self._state_lock:
            needs_reload = (
                new_config.whisper_model != self.config.whisper_model
                or new_config.whisper_device != self.config.whisper_device
                or new_config.whisper_compute_type != self.config.whisper_compute_type
            )
            self.config = new_config
        if needs_reload:
            self.unload_model()
        return needs_reload

    def _emit_status(self, message: str) -> None:
        if self._on_status is not None:
            self._on_status(message)

    def _detect_loaded_backend(self, default_device: str, default_compute: str) -> tuple[str, str]:
        inner = getattr(self._model, "model", None) or self._model
        dev = str(getattr(inner, "device", "") or default_device).lower()
        cmp_type = str(getattr(inner, "compute_type", "") or default_compute).lower()
        return dev, cmp_type

    def load_model(self) -> tuple[str, str]:
        """Load Whisper model on configured device with CUDA int8 / CPU int8 fallbacks."""
        register_cuda_dlls()
        with self._state_lock:
            cfg = self.config
        requested_device = cfg.whisper_device.lower()
        requested_compute = cfg.whisper_compute_type.lower()
        resolved_model = _resolve_cached_whisper_model(cfg.whisper_model)

        if requested_device == "cpu":
            compute = "int8" if requested_compute in _CPU_UNSUPPORTED_COMPUTE_TYPES else requested_compute
            self._emit_status(f"Завантаження {cfg.whisper_model} (CPU)...")
            model = WhisperModel(
                resolved_model,
                device="cpu",
                compute_type=compute,
            )
            with self._state_lock:
                self._model = model
                self.active_device, self.active_compute_type = self._detect_loaded_backend("cpu", compute)
                return self.active_device, self.active_compute_type

        try:
            self._emit_status(
                f"Завантаження {cfg.whisper_model} ({requested_device.upper()})..."
            )
            model = WhisperModel(
                resolved_model,
                device=requested_device,
                compute_type=requested_compute,
            )
            with self._state_lock:
                self._model = model
                default_dev = "cuda" if requested_device == "auto" else requested_device
                self.active_device, self.active_compute_type = self._detect_loaded_backend(
                    default_dev, requested_compute
                )
                return self.active_device, self.active_compute_type
        except Exception as exc:
            err_low = str(exc).lower()
            # On GPUs without fast FP16, try CUDA int8 first before falling back to CPU
            if (
                requested_device == "cuda"
                and requested_compute != "int8"
                and ("float16" in err_low or "compute type" in err_low or "unsupported" in err_low)
                and "dll" not in err_low
                and "not found" not in err_low
            ):
                try:
                    self._emit_status(f"Завантаження {cfg.whisper_model} (CUDA int8)...")
                    model = WhisperModel(
                        resolved_model,
                        device="cuda",
                        compute_type="int8",
                    )
                    with self._state_lock:
                        self._model = model
                        self.active_device, self.active_compute_type = self._detect_loaded_backend(
                            "cuda", "int8"
                        )
                        return self.active_device, self.active_compute_type
                except Exception:
                    pass

            logger.warning(
                "Failed to load WhisperModel on %s (%s): %s. Falling back to CPU int8.",
                requested_device,
                requested_compute,
                exc,
            )
            self._emit_status("CUDA недоступна — перехід на CPU (int8)...")
            model = WhisperModel(
                resolved_model,
                device="cpu",
                compute_type="int8",
            )
            with self._state_lock:
                self._model = model
                self.active_device, self.active_compute_type = self._detect_loaded_backend("cpu", "int8")
                return self.active_device, self.active_compute_type

    def _fallback_to_cpu(self) -> None:
        logger.warning("Runtime CUDA error during transcription; reloading model on CPU int8.")
        self._emit_status("CUDA помилка під час розпізнавання — перехід на CPU...")
        self.unload_model()
        with self._state_lock:
            model_name = self.config.whisper_model
        resolved_model = _resolve_cached_whisper_model(model_name)
        model = WhisperModel(
            resolved_model,
            device="cpu",
            compute_type="int8",
        )
        with self._state_lock:
            self._model = model
            self.active_device = "cpu"
            self.active_compute_type = "int8"

    def _run_transcribe(self, audio: np.ndarray) -> str:
        with self._state_lock:
            model = self._model
            cfg = self.config
            vocab_snapshot = list(self._vocabulary)
            last_text = self._last_text
            device = self.active_device

        assert model is not None
        primary_lang, secondary_langs = _parse_stt_language(cfg.whisper_language)
        language_arg = None if secondary_langs else primary_lang

        prepared = normalize_speech_audio(audio)
        hotwords_str = " ".join(vocab_snapshot[:12]).strip() or None
        initial_prompt = _build_uk_prompt(last_text, language_arg)

        if device == "cpu":
            beam_size = min(2, int(cfg.whisper_beam_size))
            patience = 1.0
        else:
            beam_size = int(cfg.whisper_beam_size)
            patience = 1.2

        segments, _info = model.transcribe(
            prepared,
            language=language_arg,
            beam_size=beam_size,
            patience=patience,
            temperature=(0.0, 0.2, 0.4),
            repetition_penalty=1.02,
            no_repeat_ngram_size=0,
            initial_prompt=initial_prompt,
            hotwords=hotwords_str,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.30,
                "min_speech_duration_ms": 120,
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 400,
            },
            condition_on_previous_text=False,
            no_speech_threshold=0.65,
            log_prob_threshold=-1.2,
            compression_ratio_threshold=2.35,
            without_timestamps=True,
        )

        kept_parts: list[str] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            no_speech_prob = getattr(seg, "no_speech_prob", 0.0)
            avg_logprob = getattr(seg, "avg_logprob", 0.0)
            compression_ratio = getattr(seg, "compression_ratio", 1.0)
            if no_speech_prob > 0.72 or avg_logprob < -1.25 or compression_ratio > 2.35:
                continue
            seg_clean = strip_non_speech_tags(text)
            if not seg_clean or is_hallucination(seg_clean):
                continue
            kept_parts.append(seg_clean)

        return re.sub(r"\s+", " ", " ".join(kept_parts)).strip()

    def transcribe(self, audio: np.ndarray, *, has_overlap: bool = True) -> str:
        """Transcribe 16 kHz mono float32 audio and return clean Ukrainian text."""
        with self._state_lock:
            need_load = self._model is None
            epoch_start = self._history_epoch
        if need_load:
            self.load_model()

        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return ""

        try:
            raw_text = self._run_transcribe(arr)
        except (RuntimeError, OSError) as exc:
            if self.active_device != "cpu" and _looks_like_cuda_runtime_error(exc):
                self._fallback_to_cpu()
                raw_text = self._run_transcribe(arr)
            else:
                raise

        with self._state_lock:
            vocab = list(self._vocabulary)
            prev_text = self._last_text if has_overlap else ""

        deduped = _postprocess_transcript_text(
            raw_text, vocab=vocab, prev_text=prev_text, has_overlap=has_overlap
        )
        if not deduped:
            return ""

        with self._state_lock:
            if self._history_epoch != epoch_start:
                return ""
            self._last_text = deduped
        return deduped

    def reset_history(self) -> None:
        """Clear last-utterance overlap state and invalidate any in-flight chunk."""
        with self._state_lock:
            self._history_epoch += 1
            self._last_text = ""


def _build_uk_prompt(last_text: str, language_arg: str | None) -> str | None:
    """Build trailing Ukrainian context prompt without injecting unspoken tokens."""
    if language_arg != "uk":
        return None
    raw_tail = last_text.strip()
    if not raw_tail:
        return _UK_PROMPT_ANCHOR
    tail = raw_tail[-90:].lstrip()
    if len(raw_tail) > 90 and " " in tail:
        tail = tail.split(" ", 1)[1]
    return f"{_UK_PROMPT_ANCHOR} {tail}".strip()


def _postprocess_transcript_text(
    raw_text: str,
    *,
    vocab: list[str],
    prev_text: str,
    has_overlap: bool,
) -> str:
    """Clean tags/hallucinations, normalize domain terms, and deduplicate overlap."""
    cleaned = strip_non_speech_tags(raw_text)
    if is_hallucination(cleaned):
        return ""
    normalized = apply_vocabulary(cleaned, vocab) if vocab else cleaned
    deduped = (
        deduplicate_overlap(prev_text, normalized)
        if (has_overlap and prev_text)
        else normalized
    )
    if is_hallucination(deduped):
        return ""
    return deduped.strip()


def _parse_stt_language(raw_lang: str) -> tuple[str | None, list[str]]:
    """Parse single or mixed STT language spec (e.g. 'uk', 'en+uk', 'auto')."""
    norm = (raw_lang or "").strip().lower()
    if norm in {"", "auto"}:
        return None, []
    if "+" in norm:
        parts = [part.strip() for part in norm.split("+") if part.strip()]
        if parts:
            return parts[0], parts[1:]
        return None, []
    return norm, []


def resolve_stt_model(provider: str, stt_model: str = "") -> str:
    """Return effective STT model ID for the given provider with sensible 2026 defaults."""
    prov = (provider or "local").strip().lower()
    cleaned = (stt_model or "").strip()
    if cleaned:
        return cleaned
    return STT_DEFAULT_MODELS.get(prov, "")


def _load_elevenlabs_env() -> None:
    env_file = get_env_file_path()
    if env_file.is_file():
        load_dotenv(env_file, override=False, interpolate=False)
    load_dotenv(".env", override=False, interpolate=False)


def resolve_stt_api_key(config: AppConfig) -> str:
    """Resolve explicit STT API key, ELEVENLABS_API_KEY from .env, or Groq LLM key reuse."""
    direct = (config.stt_api_key or "").strip()
    if direct:
        return direct
    prov = (config.stt_provider or "local").strip().lower()
    if prov == "elevenlabs":
        _load_elevenlabs_env()
        return os.getenv("ELEVENLABS_API_KEY", "").strip()
    if prov == "groq":
        groq_direct = (config.groq_api_key or "").strip()
        if groq_direct:
            return groq_direct
        candidate = (config.openai_api_key or "").strip()
        if (config.llm_provider or "").strip().lower() == "groq" or candidate.startswith("gsk_"):
            return candidate
    return ""


def encode_float32_to_wav_bytes(audio_16k: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert normalized mono float32 samples into in-memory 16-bit PCM WAV bytes."""
    arr = np.nan_to_num(
        np.asarray(audio_16k, dtype=np.float32).reshape(-1),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )
    pcm16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(max(8000, int(sample_rate)))
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


class CloudSTTTranscriber:
    """Chunked cloud speech recognizer supporting Groq Whisper and ElevenLabs Scribe."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._state_lock = threading.RLock()
        self._vocabulary: list[str] = []
        self._last_text = ""
        self._history_epoch = 0
        self._groq_client: Any = None
        self._groq_key = ""
        self._elevenlabs_client: ElevenLabs | None = None
        self._elevenlabs_key = ""

    def set_vocabulary(self, terms: Iterable[str]) -> None:
        cleaned = [t.strip() for t in terms if t and t.strip()]
        with self._state_lock:
            self._vocabulary = cleaned

    def update_config(self, new_config: AppConfig) -> None:
        with self._state_lock:
            self.config = new_config
            self._groq_client = None
            self._groq_key = ""
            self._elevenlabs_client = None
            self._elevenlabs_key = ""

    def unload_model(self) -> None:
        with self._state_lock:
            client = self._groq_client
            el_client = self._elevenlabs_client
            self._groq_client = None
            self._groq_key = ""
            self._elevenlabs_client = None
            self._elevenlabs_key = ""
        if client is not None:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        if el_client is not None:
            el_httpx = getattr(
                getattr(getattr(el_client, "_client_wrapper", None), "httpx_client", None),
                "httpx_client",
                None,
            )
            close_el_fn = getattr(el_httpx, "close", None)
            if callable(close_el_fn):
                try:
                    close_el_fn()
                except Exception:
                    pass

    def reset_history(self) -> None:
        with self._state_lock:
            self._history_epoch += 1
            self._last_text = ""

    def status_label(self) -> str:
        with self._state_lock:
            cfg = self.config
        prov = (cfg.stt_provider or "local").strip().lower()
        model = resolve_stt_model(prov, cfg.stt_model)
        key = resolve_stt_api_key(cfg)
        if not key:
            return f"STT ({prov}): немає API-ключа"
        if prov == "groq":
            return f"STT: Groq Cloud ({model})"
        if prov == "elevenlabs":
            return f"STT: ElevenLabs Scribe ({model})"
        return f"STT: {prov} ({model})"

    def _get_groq_client(self, api_key: str):
        with self._state_lock:
            if self._groq_client is not None and self._groq_key == api_key:
                return self._groq_client
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                timeout=20.0,
            )
            self._groq_client = client
            self._groq_key = api_key
            return client

    def _get_elevenlabs_client(self, api_key: str) -> ElevenLabs:
        with self._state_lock:
            _load_elevenlabs_env()
            effective_key = (api_key or os.getenv("ELEVENLABS_API_KEY", "")).strip()
            if effective_key and os.getenv("ELEVENLABS_API_KEY", "").strip() != effective_key:
                os.environ["ELEVENLABS_API_KEY"] = effective_key
            if self._elevenlabs_client is not None and self._elevenlabs_key == effective_key:
                return self._elevenlabs_client
            client = ElevenLabs(api_key=effective_key, timeout=25.0)
            self._elevenlabs_client = client
            self._elevenlabs_key = effective_key
            return client

    def transcribe(self, audio: np.ndarray, *, has_overlap: bool = True) -> str:
        with self._state_lock:
            cfg = self.config
            vocab = list(self._vocabulary)
            last_text = self._last_text
            epoch_start = self._history_epoch

        prov = (cfg.stt_provider or "").strip().lower()
        api_key = resolve_stt_api_key(cfg)
        if not api_key:
            raise RuntimeError(f"Вкажіть API-ключ для STT ({prov}) у Налаштуваннях.")

        prepared = normalize_speech_audio(audio)
        if prepared.size == 0:
            return ""
        wav_bytes = encode_float32_to_wav_bytes(prepared, 16000)
        primary_lang, secondary_langs = _parse_stt_language(cfg.whisper_language)
        model = resolve_stt_model(prov, cfg.stt_model)

        try:
            if prov == "groq":
                client = self._get_groq_client(api_key)
                groq_lang = None if secondary_langs else primary_lang
                prompt_text = _build_uk_prompt(last_text, groq_lang)
                kwargs: dict[str, Any] = {
                    "model": model,
                    "file": ("chunk.wav", wav_bytes, "audio/wav"),
                    "temperature": 0.0,
                }
                if groq_lang:
                    kwargs["language"] = groq_lang
                if prompt_text:
                    kwargs["prompt"] = prompt_text
                resp = client.audio.transcriptions.create(**kwargs)
                raw_text = str(getattr(resp, "text", "") or "")
            elif prov == "elevenlabs":
                el_client = self._get_elevenlabs_client(api_key)
                convert_kwargs: dict[str, Any] = {
                    "file": ("chunk.wav", wav_bytes, "audio/wav"),
                    "model_id": model,
                    "tag_audio_events": False,
                    "timestamps_granularity": "none",
                }
                if secondary_langs:
                    if "v3" in model.lower() and primary_lang:
                        convert_kwargs["language_code"] = _ISO639_1_TO_3.get(
                            primary_lang, primary_lang
                        )
                        convert_kwargs["secondary_language_codes"] = [
                            _ISO639_1_TO_3.get(code, code) for code in secondary_langs
                        ]
                elif primary_lang:
                    convert_kwargs["language_code"] = _ISO639_1_TO_3.get(
                        primary_lang, primary_lang
                    )
                keyterms = [
                    t.strip()
                    for t in vocab
                    if t.strip()
                    and len(t.strip()) <= 50
                    and len(t.strip().split()) <= 5
                    and not any(ch in _EL_DISALLOWED_KEYTERM_CHARS for ch in t)
                ][:100]
                if keyterms and "v1" not in model.lower():
                    convert_kwargs["keyterms"] = keyterms
                result = el_client.speech_to_text.convert(**convert_kwargs)
                raw_text = str(getattr(result, "text", "") or "")
            else:
                raise RuntimeError(f"Непідтримуваний хмарний провайдер STT: {prov}")
        except Exception as exc:
            raise RuntimeError(_format_cloud_stt_error(prov, exc, api_key)) from exc

        deduped = _postprocess_transcript_text(
            raw_text,
            vocab=vocab,
            prev_text=last_text if has_overlap else "",
            has_overlap=has_overlap,
        )
        if not deduped:
            return ""
        with self._state_lock:
            if self._history_epoch != epoch_start:
                return ""
            self._last_text = deduped
        return deduped


class DeepgramRealtimeClient:
    """Real-time bidirectional WebSocket streaming client for Deepgram Nova-3."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._state_lock = threading.RLock()
        self._vocabulary: list[str] = []

    def set_vocabulary(self, terms: Iterable[str]) -> None:
        cleaned = [t.strip() for t in terms if t and t.strip()]
        with self._state_lock:
            self._vocabulary = cleaned

    def update_config(self, new_config: AppConfig) -> None:
        with self._state_lock:
            self.config = new_config

    def status_label(self) -> str:
        with self._state_lock:
            cfg = self.config
        model = resolve_stt_model("deepgram", cfg.stt_model)
        key = resolve_stt_api_key(cfg)
        if not key:
            return "STT (deepgram): немає API-ключа"
        return f"STT: Deepgram {model} (real-time)"

    @staticmethod
    def build_ws_url(config: AppConfig, vocabulary: Iterable[str] = ()) -> str:
        """Build Deepgram /v1/listen WebSocket URL with Nova-3 and Ukrainian endpointing."""
        model = resolve_stt_model("deepgram", config.stt_model)
        lang_raw = (config.whisper_language or "uk").strip().lower()
        lang_param = "multi" if (lang_raw in {"", "auto"} or "+" in lang_raw) else lang_raw
        endpointing_ms = int(max(200, min(2000, round(config.audio_silence_seconds * 1000))))
        params: list[tuple[str, str]] = [
            ("model", model),
            ("language", lang_param),
            ("encoding", "linear16"),
            ("sample_rate", "16000"),
            ("channels", "1"),
            ("smart_format", "true"),
            ("punctuate", "true"),
            ("interim_results", "false"),
            ("endpointing", str(endpointing_ms)),
        ]
        if "nova-3" in model.lower():
            count = 0
            for raw_term in vocabulary:
                term = raw_term.strip()
                if not term or len(term) > 48:
                    continue
                params.append(("keyterm", term))
                count += 1
                if count >= 25:
                    break
        return f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"

    def run_session(
        self,
        pcm_queue: queue.Queue,
        stop_event: threading.Event,
        on_transcript: Callable[[str], None],
        on_status: Callable[[str], None],
        on_error: Callable[[str], None],
        *,
        stop_sentinel: object | None = None,
    ) -> None:
        """Stream PCM16 audio frames to Deepgram while capture is active."""
        with self._state_lock:
            cfg = self.config
        api_key = resolve_stt_api_key(cfg)
        if not api_key:
            on_status("STT (deepgram): немає API-ключа")
            return

        reconnect_attempts = 0
        sentinel_obj = stop_sentinel if stop_sentinel is not None else object()
        while not stop_event.is_set():
            # Wait lazily for real PCM bytes before connecting (no idle WS billing).
            first_item: Any = None
            while not stop_event.is_set():
                try:
                    item = pcm_queue.get(timeout=0.3)
                except queue.Empty:
                    continue
                if item is sentinel_obj:
                    return
                if isinstance(item, (bytes, bytearray)) and item:
                    first_item = bytes(item)
                    break

            if stop_event.is_set() or first_item is None:
                return

            with self._state_lock:
                cfg = self.config
                vocab = list(self._vocabulary)
            url = self.build_ws_url(cfg, vocab)
            model_name = resolve_stt_model("deepgram", cfg.stt_model)

            try:
                on_status(f"STT: підключення Deepgram {model_name}...")
                with _ws_connect(
                    url,
                    additional_headers={"Authorization": f"Token {api_key}"},
                    open_timeout=10.0,
                    ping_interval=15.0,
                    ping_timeout=15.0,
                    close_timeout=3.0,
                ) as ws:
                    reconnect_attempts = 0
                    on_status(f"STT: Deepgram {model_name} (онлайн ✓)")
                    rx_done = threading.Event()
                    rx_error: list[Exception] = []

                    def _recv_loop(
                        _ws=ws,
                        _rx_done: threading.Event = rx_done,
                        _rx_error: list[Exception] = rx_error,
                    ) -> None:
                        try:
                            for raw_msg in _ws:
                                if stop_event.is_set():
                                    break
                                if not isinstance(raw_msg, str):
                                    continue
                                try:
                                    msg = json.loads(raw_msg)
                                except ValueError:
                                    continue
                                if msg.get("type") != "Results" or not msg.get("is_final"):
                                    continue
                                channel = msg.get("channel") or {}
                                alts = channel.get("alternatives") or []
                                if not alts:
                                    continue
                                raw_text = str(alts[0].get("transcript", "") or "").strip()
                                if not raw_text:
                                    continue
                                with self._state_lock:
                                    cur_vocab = list(self._vocabulary)
                                clean_text = _postprocess_transcript_text(
                                    raw_text,
                                    vocab=cur_vocab,
                                    prev_text="",
                                    has_overlap=False,
                                )
                                if clean_text:
                                    on_transcript(clean_text)
                        except Exception as exc:
                            if not stop_event.is_set():
                                _rx_error.append(exc)
                        finally:
                            _rx_done.set()

                    rx_thread = threading.Thread(
                        target=_recv_loop, name="DeepgramRX", daemon=True
                    )
                    rx_thread.start()

                    ws.send(first_item)
                    last_tx_time = time.monotonic()
                    capture_ended = False
                    stopped_by_sentinel = False

                    while not stop_event.is_set() and not rx_done.is_set():
                        try:
                            nxt = pcm_queue.get(timeout=0.4)
                        except queue.Empty:
                            now = time.monotonic()
                            if (now - last_tx_time) >= 4.0:
                                ws.send(json.dumps({"type": "KeepAlive"}))
                                last_tx_time = now
                            continue

                        if nxt is sentinel_obj:
                            stopped_by_sentinel = True
                            capture_ended = True
                            break
                        if nxt is None:
                            capture_ended = True
                            break
                        if isinstance(nxt, (bytes, bytearray)) and nxt:
                            ws.send(bytes(nxt))
                            last_tx_time = time.monotonic()

                    try:
                        ws.send(json.dumps({"type": "CloseStream"}))
                    except Exception:
                        pass
                    rx_thread.join(timeout=1.5)

                    if stopped_by_sentinel or stop_event.is_set():
                        return
                    if capture_ended and not rx_error:
                        on_status(self.status_label())
                        continue
                    if rx_error:
                        raise rx_error[0]
                    raise RuntimeError("Deepgram WebSocket closed unexpectedly")

            except Exception as exc:
                if stop_event.is_set():
                    return
                safe_msg = redact_secrets(str(exc), extra_secrets=(api_key,))
                low = safe_msg.lower()
                if "401" in low or "403" in low or "unauthorized" in low or "forbidden" in low:
                    on_status("STT (deepgram): помилка ключа")
                    on_error("Deepgram відхилив API-ключ (401/403). Перевірте STT API-ключ.")
                    return
                reconnect_attempts += 1
                logger.warning("Deepgram WebSocket disconnected (%s), attempt %d/3", safe_msg, reconnect_attempts)
                if reconnect_attempts > 3:
                    on_status("STT (deepgram): помилка з'єднання")
                    on_error(f"Помилка Deepgram WebSocket: {safe_msg}")
                    return
                on_status(f"STT: Deepgram перепідключення ({reconnect_attempts}/3)...")
                stop_event.wait(0.6)


def verify_stt_provider(config: AppConfig) -> tuple[bool, str]:
    """Verify STT provider credentials, reachability, or local Whisper cache readiness."""
    prov = (config.stt_provider or "local").strip().lower()
    if prov == "local":
        try:
            import ctranslate2

            cuda_count = int(ctranslate2.get_cuda_device_count())
        except Exception:
            cuda_count = 0
        cached_msg = "у кеші"
        try:
            from faster_whisper.utils import download_model

            cached = download_model(config.whisper_model, local_files_only=True)
            if not cached:
                cached_msg = "буде завантажено при запуску"
        except Exception:
            cached_msg = "буде завантажено при запуску"
        hw = f"CUDA пристроїв: {cuda_count}" if cuda_count > 0 else "CPU режим"
        return True, f"Локальний Whisper готовий ✓ ({config.whisper_model}, {cached_msg}, {hw})"

    api_key = resolve_stt_api_key(config)
    if not api_key:
        return False, f"Вкажіть API-ключ для {prov} перед перевіркою."
    model = resolve_stt_model(prov, config.stt_model)

    try:
        if prov == "deepgram":
            resp = requests.get(
                "https://api.deepgram.com/v1/projects",
                headers={"Authorization": f"Token {api_key}"},
                timeout=8.0,
            )
            if resp.status_code == 401:
                return False, "Невірний Deepgram API-ключ (401)."
            if resp.status_code == 403:
                return False, "Ключ Deepgram відхилено (403 — перевірте права доступу)."
            resp.raise_for_status()
            return True, f"Deepgram підключено ✓ (модель: {model})"

        if prov == "elevenlabs":
            _load_elevenlabs_env()
            if api_key and os.getenv("ELEVENLABS_API_KEY", "").strip() != api_key:
                os.environ["ELEVENLABS_API_KEY"] = api_key
            _ = ElevenLabs(api_key=api_key, timeout=8.0)
            resp = requests.get(
                "https://api.elevenlabs.io/v1/user",
                headers={"xi-api-key": api_key},
                timeout=8.0,
            )
            if resp.status_code in (401, 403):
                return False, f"Невірний ElevenLabs API-ключ ({resp.status_code})."
            resp.raise_for_status()
            return True, f"ElevenLabs Scribe підключено ✓ (модель: {model})"

        if prov == "groq":
            resp = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=8.0,
            )
            if resp.status_code in (401, 403):
                return False, f"Невірний Groq API-ключ ({resp.status_code})."
            resp.raise_for_status()
            return True, f"Groq Cloud STT підключено ✓ (модель: {model})"

        return False, f"Невідомий провайдер STT: {prov}"
    except Exception as exc:
        safe_msg = redact_secrets(str(exc), extra_secrets=(api_key,))
        return False, f"Помилка перевірки {prov}: {safe_msg}"
