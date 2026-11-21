"""Local GPU/CPU Ukrainian speech transcription using faster-whisper."""

from __future__ import annotations

from functools import lru_cache
import gc
import logging
import os
import re
import threading
from typing import Callable, Iterable

import numpy as np

from config import AppConfig, register_cuda_dlls

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Register pip-installed or bundled NVIDIA cuBLAS / cuDNN DLLs BEFORE importing faster_whisper.
register_cuda_dlls()

from faster_whisper import WhisperModel  # noqa: E402


def _resolve_cached_whisper_model(model_name: str) -> str:
    """Return local HuggingFace cache directory if already downloaded, avoiding network checks."""
    if not model_name or os.path.isdir(model_name):
        return model_name
    try:
        from faster_whisper.utils import download_model

        cached_dir = download_model(model_name, local_files_only=True)
        if cached_dir and os.path.isdir(str(cached_dir)):
            return str(cached_dir)
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
        lang = cfg.whisper_language.strip().lower()
        language_arg = None if lang in {"", "auto"} else lang

        prepared = normalize_speech_audio(audio)
        hotwords_str = " ".join(vocab_snapshot[:30]).strip() or None

        initial_prompt: str | None = None
        if language_arg == "uk":
            raw_tail = last_text.strip()
            if raw_tail:
                tail = raw_tail[-90:].lstrip()
                if len(raw_tail) > 90 and " " in tail:
                    tail = tail.split(" ", 1)[1]
                initial_prompt = f"{_UK_PROMPT_ANCHOR} {tail}".strip()
            else:
                initial_prompt = _UK_PROMPT_ANCHOR

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
            temperature=(0.0, 0.2),
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            initial_prompt=initial_prompt,
            hotwords=hotwords_str,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.35,
                "min_speech_duration_ms": 150,
                "min_silence_duration_ms": 450,
                "speech_pad_ms": 380,
            },
            condition_on_previous_text=False,
            no_speech_threshold=0.55,
            log_prob_threshold=-1.0,
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
            if no_speech_prob > 0.70 or avg_logprob < -1.05 or compression_ratio > 2.35:
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

        cleaned = strip_non_speech_tags(raw_text)
        if is_hallucination(cleaned):
            return ""

        with self._state_lock:
            vocab = list(self._vocabulary)
            prev_text = self._last_text if has_overlap else ""

        normalized = apply_vocabulary(cleaned, vocab) if vocab else cleaned
        if has_overlap and prev_text:
            deduped = deduplicate_overlap(prev_text, normalized)
        else:
            deduped = normalized

        if is_hallucination(deduped):
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
