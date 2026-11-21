"""Application configuration, atomic .env persistence, and CUDA DLL registration."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import site
import sys
import tempfile
from typing import Any

from dotenv import load_dotenv

from security import register_secret

VALID_LLM_PROVIDERS: frozenset[str] = frozenset(
    {"openai", "groq", "gemini", "ollama", "lmstudio"}
)
VALID_WHISPER_DEVICES: frozenset[str] = frozenset({"cuda", "cpu", "auto"})
VALID_SEARCH_PROVIDERS: frozenset[str] = frozenset(
    {"auto", "ddgs", "tavily", "brave", "wikipedia"}
)
VALID_AGENT_MODES: frozenset[str] = frozenset({"agent", "fast"})

DEFAULT_SYSTEM_PROMPT = (
    'Ти — асистент студента на усному захисті роботи. Твоє завдання — давати '
    'максимально стислі, точні, тезисні відповіді українською мовою на питання '
    'викладача, спираючись на наданий контекст роботи. Без вступних слів '
    '("Звісно", "Ось відповідь"). Використовуй короткі буліт-поїнти, щоб '
    'відповідь можна було прочитати за 2 секунди і переказати своїми словами.'
)

_DLL_DIR_HANDLES: list[Any] = []
_REGISTERED_DLL_DIRS: set[str] = set()
_VOCAB_SPLIT_RE = re.compile(r"[,;\r\n]+")


def get_app_dir() -> Path:
    """Return the application directory (alongside the .exe when frozen, else source dir)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_resource_path(rel_path: str) -> Path:
    """Resolve bundled asset paths in both development mode and PyInstaller builds."""
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidate = Path(meipass) / rel_path
            if candidate.exists():
                return candidate
    return get_app_dir() / rel_path


def get_env_file_path() -> Path:
    """Return writable `.env` path next to app/exe, falling back to %APPDATA% if needed."""
    app_env = get_app_dir() / ".env"
    if app_env.is_file():
        if os.access(app_env, os.W_OK):
            return app_env
    elif os.access(get_app_dir(), os.W_OK):
        return app_env

    appdata_dir = Path(os.getenv("APPDATA", str(Path.home()))) / "ZoomAICopilot"
    appdata_dir.mkdir(parents=True, exist_ok=True)
    return appdata_dir / ".env"


def parse_custom_vocabulary(raw: str) -> list[str]:
    """Parse comma/semicolon/newline-separated custom vocabulary terms."""
    if not raw or not raw.strip():
        return []
    seen: set[str] = set()
    terms: list[str] = []
    for part in _VOCAB_SPLIT_RE.split(raw):
        cleaned = part.strip()
        if len(cleaned) < 2:
            continue
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            terms.append(cleaned)
    return terms


def _parse_float(
    raw: str | None,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Parse a float from string with default fallback and optional range clamping."""
    if raw is None or not raw.strip():
        value = default
    else:
        try:
            value = float(raw.strip())
            if not math.isfinite(value):
                value = default
        except (ValueError, OverflowError):
            value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _parse_int(
    raw: str | None,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Parse an integer from string with default fallback and optional range clamping."""
    if raw is None or not raw.strip():
        value = default
    else:
        try:
            parsed = float(raw.strip())
            if not math.isfinite(parsed):
                value = default
            else:
                value = int(parsed)
        except (ValueError, OverflowError):
            value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _parse_bool(raw: str | None, default: bool) -> bool:
    """Parse a boolean string ('1'/'true'/'yes'/'on') with default fallback."""
    if raw is None or not raw.strip():
        return default
    norm = raw.strip().lower()
    if norm in {"1", "true", "yes", "on"}:
        return True
    if norm in {"0", "false", "no", "off"}:
        return False
    return default


def register_cuda_dlls() -> list[str]:
    """Register NVIDIA CUDA/cuDNN bin folders on Windows before ctranslate2 import."""
    if sys.platform != "win32":
        return []

    candidate_roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate_roots.append(Path(meipass))

    exe_dir = Path(sys.executable).resolve().parent
    candidate_roots.append(exe_dir / "_internal")
    candidate_roots.append(exe_dir)

    try:
        for sp in site.getsitepackages():
            candidate_roots.append(Path(sp))
    except Exception:
        pass

    try:
        user_sp = site.getusersitepackages()
        if user_sp:
            candidate_roots.append(Path(user_sp))
    except Exception:
        pass

    candidate_roots.append(Path(sys.prefix) / "Lib" / "site-packages")

    seen_roots: set[str] = set()
    discovered_dirs: list[Path] = []
    priority_subpaths = (
        Path("nvidia") / "cublas" / "bin",
        Path("nvidia") / "cudnn" / "bin",
        Path("nvidia") / "cuda_runtime" / "bin",
    )

    for root in candidate_roots:
        try:
            resolved_root = str(root.resolve())
        except OSError:
            resolved_root = str(root)
        key = os.path.normcase(resolved_root)
        if key in seen_roots or not root.is_dir():
            continue
        seen_roots.add(key)

        for rel in priority_subpaths:
            bin_dir = root / rel
            if bin_dir.is_dir():
                discovered_dirs.append(bin_dir)

        nvidia_dir = root / "nvidia"
        if nvidia_dir.is_dir():
            try:
                for child in sorted(nvidia_dir.iterdir()):
                    bin_dir = child / "bin"
                    if bin_dir.is_dir() and bin_dir not in discovered_dirs:
                        discovered_dirs.append(bin_dir)
            except OSError:
                pass

        # In frozen PyInstaller bundles, some CUDA DLLs may also sit directly inside _internal.
        if meipass and (
            os.path.normcase(resolved_root) == os.path.normcase(str(Path(meipass).resolve()))
            or root.name == "_internal"
        ):
            if root not in discovered_dirs:
                discovered_dirs.append(root)

    registered: list[str] = []
    current_path_entries = [
        os.path.normcase(os.path.normpath(p))
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if p
    ]

    for bin_dir in discovered_dirs:
        dir_str = str(bin_dir.resolve())
        norm_key = os.path.normcase(dir_str)
        if norm_key not in _REGISTERED_DLL_DIRS:
            if hasattr(os, "add_dll_directory"):
                try:
                    handle = os.add_dll_directory(dir_str)
                    _DLL_DIR_HANDLES.append(handle)
                except OSError:
                    continue
            _REGISTERED_DLL_DIRS.add(norm_key)
        if norm_key not in current_path_entries:
            os.environ["PATH"] = dir_str + os.pathsep + os.environ.get("PATH", "")
            current_path_entries.insert(0, norm_key)
        registered.append(dir_str)

    return registered


@dataclass
class AppConfig:
    """Typed runtime settings loaded from environment variables / `.env`."""

    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-6-luna"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 512

    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    whisper_language: str = "uk"
    whisper_beam_size: int = 5

    audio_chunk_seconds: float = 5.0
    audio_overlap_seconds: float = 0.85
    audio_silence_seconds: float = 0.8
    audio_silence_threshold: float = 0.006

    transcript_window_seconds: int = 60
    context_max_chars: int = 120000
    rag_context_budget_chars: int = 12000
    custom_vocabulary: str = ""
    hotkey_answer: str = "ctrl+space"
    hotkey_toggle: str = "ctrl+shift+h"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    overlay_opacity: int = 92
    overlay_stealth: bool = True
    dashboard_stealth: bool = False
    overlay_font_scale: float = 1.0

    web_search_enabled: bool = False
    web_search_provider: str = "auto"
    web_search_api_key: str = ""
    web_agent_mode: str = "agent"
    web_search_max_results: int = 5
    obscura_path: str = ""

    screen_vision_enabled: bool = False
    screen_capture_index: int = -1
    screen_capture_max_width: int = 1600


def has_valid_llm_credentials(config: AppConfig) -> bool:
    """Return True if the selected LLM provider has required credentials (or is local)."""
    provider = config.llm_provider.lower()
    if provider not in VALID_LLM_PROVIDERS:
        return False
    if provider in {"ollama", "lmstudio"}:
        return True
    if provider == "gemini":
        return bool(config.gemini_api_key.strip())
    return bool(config.openai_api_key.strip())


def load_config(env_path: str | None = None, override: bool = False) -> AppConfig:
    """Load configuration from `.env` (if present) and environment variables."""
    if env_path:
        target_env = Path(env_path)
        if target_env.is_file():
            load_dotenv(dotenv_path=target_env, override=override, interpolate=False)
    else:
        target_env = get_env_file_path()
        if target_env.is_file():
            load_dotenv(dotenv_path=target_env, override=override, interpolate=False)
        else:
            cwd_env = Path.cwd() / ".env"
            if cwd_env.is_file():
                load_dotenv(dotenv_path=cwd_env, override=override, interpolate=False)
            else:
                load_dotenv(override=override, interpolate=False)

    def _env_str(key: str, default: str) -> str:
        val = os.getenv(key)
        if val is None:
            return default
        stripped = val.strip()
        return stripped if stripped else default

    chunk_seconds = _parse_float(
        os.getenv("AUDIO_CHUNK_SECONDS"), 5.0, minimum=2.0, maximum=15.0
    )
    overlap_seconds = _parse_float(
        os.getenv("AUDIO_OVERLAP_SECONDS"),
        0.85,
        minimum=0.0,
        maximum=max(0.0, chunk_seconds - 0.5),
    )

    raw_prompt = _env_str("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    system_prompt = (
        raw_prompt.replace("\\n", "\n")
        if ("\n" not in raw_prompt and "\\n" in raw_prompt)
        else raw_prompt
    )

    raw_llm_prov = _env_str("LLM_PROVIDER", "openai").lower()
    llm_provider = raw_llm_prov if raw_llm_prov in VALID_LLM_PROVIDERS else "openai"

    raw_whisper_dev = _env_str("WHISPER_DEVICE", "cuda").lower()
    whisper_device = raw_whisper_dev if raw_whisper_dev in VALID_WHISPER_DEVICES else "cuda"

    raw_web_prov = _env_str("WEB_SEARCH_PROVIDER", "auto").lower()
    web_provider = (
        raw_web_prov if raw_web_prov in VALID_SEARCH_PROVIDERS else "auto"
    )
    raw_agent_mode = _env_str("WEB_AGENT_MODE", "agent").lower()
    web_agent_mode = raw_agent_mode if raw_agent_mode in VALID_AGENT_MODES else "agent"

    cfg = AppConfig(
        llm_provider=llm_provider,
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "").strip(),
        openai_model=_env_str("OPENAI_MODEL", "gpt-6-luna"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=_env_str("GEMINI_MODEL", "gemini-3.8-flash"),
        llm_temperature=_parse_float(
            os.getenv("LLM_TEMPERATURE"), 0.2, minimum=0.0, maximum=2.0
        ),
        llm_max_tokens=_parse_int(
            os.getenv("LLM_MAX_TOKENS"), 512, minimum=64, maximum=8192
        ),
        whisper_model=_env_str("WHISPER_MODEL", "large-v3-turbo"),
        whisper_device=whisper_device,
        whisper_compute_type=_env_str("WHISPER_COMPUTE_TYPE", "float16"),
        whisper_language=_env_str("WHISPER_LANGUAGE", "uk"),
        whisper_beam_size=_parse_int(
            os.getenv("WHISPER_BEAM_SIZE"), 5, minimum=1, maximum=10
        ),
        audio_chunk_seconds=chunk_seconds,
        audio_overlap_seconds=overlap_seconds,
        audio_silence_seconds=_parse_float(
            os.getenv("AUDIO_SILENCE_SECONDS"), 0.8, minimum=0.2, maximum=5.0
        ),
        audio_silence_threshold=_parse_float(
            os.getenv("AUDIO_SILENCE_THRESHOLD"), 0.006, minimum=0.0005, maximum=0.5
        ),
        transcript_window_seconds=_parse_int(
            os.getenv("TRANSCRIPT_WINDOW_SECONDS"), 60, minimum=10, maximum=300
        ),
        context_max_chars=_parse_int(
            os.getenv("CONTEXT_MAX_CHARS"), 120000, minimum=2000, maximum=1000000
        ),
        rag_context_budget_chars=_parse_int(
            os.getenv("RAG_CONTEXT_BUDGET_CHARS"), 12000, minimum=2000, maximum=120000
        ),
        custom_vocabulary=os.getenv("CUSTOM_VOCABULARY", "").strip(),
        hotkey_answer=_env_str("HOTKEY_ANSWER", "ctrl+space").lower(),
        hotkey_toggle=_env_str("HOTKEY_TOGGLE", "ctrl+shift+h").lower(),
        system_prompt=system_prompt,
        overlay_opacity=_parse_int(
            os.getenv("OVERLAY_OPACITY"), 92, minimum=30, maximum=100
        ),
        overlay_stealth=_parse_bool(os.getenv("OVERLAY_STEALTH"), True),
        dashboard_stealth=_parse_bool(os.getenv("DASHBOARD_STEALTH"), False),
        overlay_font_scale=_parse_float(
            os.getenv("OVERLAY_FONT_SCALE"), 1.0, minimum=0.85, maximum=1.40
        ),
        web_search_enabled=_parse_bool(os.getenv("WEB_SEARCH_ENABLED"), False),
        web_search_provider=web_provider,
        web_search_api_key=os.getenv("WEB_SEARCH_API_KEY", "").strip(),
        web_agent_mode=web_agent_mode,
        web_search_max_results=_parse_int(
            os.getenv("WEB_SEARCH_MAX_RESULTS"), 5, minimum=1, maximum=10
        ),
        obscura_path=os.getenv("OBSCURA_PATH", "").strip(),
        screen_vision_enabled=_parse_bool(os.getenv("SCREEN_VISION_ENABLED"), False),
        screen_capture_index=_parse_int(
            os.getenv("SCREEN_CAPTURE_INDEX"), -1, minimum=-1, maximum=16
        ),
        screen_capture_max_width=_parse_int(
            os.getenv("SCREEN_CAPTURE_MAX_WIDTH"), 1600, minimum=800, maximum=2560
        ),
    )
    register_secret(cfg.openai_api_key)
    register_secret(cfg.gemini_api_key)
    register_secret(cfg.web_search_api_key)
    return cfg


def _format_env_value(value: str) -> str:
    if not value:
        return ""
    needs_quotes = any(ch in value for ch in (" ", "\t", "\n", "\r", "#", '"', "'", "=", ","))
    if not needs_quotes:
        return value
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    escaped = (
        normalized.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )
    return f'"{escaped}"'


def save_config_to_env(config: AppConfig, env_path: Path | None = None) -> Path:
    """Atomically persist AppConfig settings to `.env` and update process environment."""
    target = Path(env_path) if env_path else get_env_file_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    register_secret(config.openai_api_key)
    register_secret(config.gemini_api_key)
    register_secret(config.web_search_api_key)

    pairs: list[tuple[str, str]] = [
        ("LLM_PROVIDER", config.llm_provider.strip().lower()),
        ("OPENAI_API_KEY", config.openai_api_key.strip()),
        ("OPENAI_BASE_URL", config.openai_base_url.strip()),
        ("OPENAI_MODEL", config.openai_model.strip()),
        ("GEMINI_API_KEY", config.gemini_api_key.strip()),
        ("GEMINI_MODEL", config.gemini_model.strip()),
        ("LLM_TEMPERATURE", f"{config.llm_temperature:.2f}"),
        ("LLM_MAX_TOKENS", str(config.llm_max_tokens)),
        ("WHISPER_MODEL", config.whisper_model.strip()),
        ("WHISPER_DEVICE", config.whisper_device.strip().lower()),
        ("WHISPER_COMPUTE_TYPE", config.whisper_compute_type.strip()),
        ("WHISPER_LANGUAGE", config.whisper_language.strip()),
        ("WHISPER_BEAM_SIZE", str(config.whisper_beam_size)),
        ("AUDIO_CHUNK_SECONDS", f"{config.audio_chunk_seconds:.2f}"),
        ("AUDIO_OVERLAP_SECONDS", f"{config.audio_overlap_seconds:.2f}"),
        ("AUDIO_SILENCE_SECONDS", f"{config.audio_silence_seconds:.2f}"),
        ("AUDIO_SILENCE_THRESHOLD", f"{config.audio_silence_threshold:.4f}"),
        ("TRANSCRIPT_WINDOW_SECONDS", str(config.transcript_window_seconds)),
        ("CONTEXT_MAX_CHARS", str(config.context_max_chars)),
        ("RAG_CONTEXT_BUDGET_CHARS", str(config.rag_context_budget_chars)),
        ("CUSTOM_VOCABULARY", config.custom_vocabulary.strip()),
        ("HOTKEY_ANSWER", config.hotkey_answer.strip().lower()),
        ("HOTKEY_TOGGLE", config.hotkey_toggle.strip().lower()),
        ("OVERLAY_OPACITY", str(config.overlay_opacity)),
        ("OVERLAY_STEALTH", "true" if config.overlay_stealth else "false"),
        ("DASHBOARD_STEALTH", "true" if config.dashboard_stealth else "false"),
        ("OVERLAY_FONT_SCALE", f"{config.overlay_font_scale:.2f}"),
        ("WEB_SEARCH_ENABLED", "true" if config.web_search_enabled else "false"),
        ("WEB_SEARCH_PROVIDER", config.web_search_provider.strip().lower() or "auto"),
        ("WEB_SEARCH_API_KEY", config.web_search_api_key.strip()),
        ("WEB_AGENT_MODE", config.web_agent_mode.strip().lower() or "agent"),
        ("WEB_SEARCH_MAX_RESULTS", str(max(1, min(10, int(config.web_search_max_results))))),
        ("OBSCURA_PATH", config.obscura_path.strip()),
        ("SCREEN_VISION_ENABLED", "true" if config.screen_vision_enabled else "false"),
        ("SCREEN_CAPTURE_INDEX", str(max(-1, min(16, int(config.screen_capture_index))))),
        ("SCREEN_CAPTURE_MAX_WIDTH", str(max(800, min(2560, int(config.screen_capture_max_width))))),
        ("SYSTEM_PROMPT", config.system_prompt.strip() or DEFAULT_SYSTEM_PROMPT),
    ]

    lines = ["# AI Copilot for Zoom — User Configuration"]
    for key, val in pairs:
        os.environ[key] = val
        lines.append(f"{key}={_format_env_value(val)}")

    payload = "\n".join(lines) + "\n"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            prefix=".env.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise

    return target
