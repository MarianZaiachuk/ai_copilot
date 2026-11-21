"""Tests for config persistence, secret redaction, and logging filters."""

from __future__ import annotations

import logging
from pathlib import Path

import config
import security


def test_secret_redaction_patterns_and_registered_secrets() -> None:
    security.clear_registered_secrets()
    try:
        security.register_secret("my-custom-secret-token-987654")
        raw = (
            "Auth failed: Bearer tok_super_secret_123, "
            "google=AIzaSyA1234567890123456789012345, "
            "openai=sk-proj-abcdef1234567890, "
            "groq=gsk_1234567890abcdef1234, "
            "tavily=tvly-dev1234567890abcdef, "
            "brave=BSAToken1234567890abcdef, "
            "url=https://api.example.com/v1?key=secretval99&x=1, "
            "custom=my-custom-secret-token-987654"
        )
        redacted = security.redact_secrets(raw)
        assert "AIzaSyA123456" not in redacted
        assert "sk-proj-abcdef" not in redacted
        assert "gsk_1234567890" not in redacted
        assert "tvly-dev1234567890" not in redacted
        assert "BSAToken1234567890" not in redacted
        assert "tok_super_secret_123" not in redacted
        assert "secretval99" not in redacted
        assert "my-custom-secret-token-987654" not in redacted
        assert "REDACTED" in redacted
    finally:
        security.clear_registered_secrets()


def test_logging_filter_and_formatter_redact_secrets() -> None:
    security.clear_registered_secrets()
    try:
        security.register_secret("super-secret-api-value-112233")
        filt = security.SecretRedactingFilter()
        fmt = security.RedactingFormatter("%(levelname)s: %(message)s")

        try:
            raise RuntimeError("Upstream failed with sk-secretkey1234567890abc")
        except RuntimeError as exc:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="Bad key %s",
                args=("super-secret-api-value-112233",),
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            assert filt.filter(record) is True
            rendered = fmt.format(record)
            assert "super-secret-api-value-112233" not in rendered
            assert "sk-secretkey1234567890abc" not in rendered
            assert "***REDACTED***" in rendered
    finally:
        security.clear_registered_secrets()


def test_atomic_env_save_and_load_roundtrip(tmp_path: Path, monkeypatch) -> None:
    for key in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "CUSTOM_VOCABULARY",
        "RAG_CONTEXT_BUDGET_CHARS",
        "SYSTEM_PROMPT",
        "OVERLAY_OPACITY",
        "OVERLAY_STEALTH",
        "DASHBOARD_STEALTH",
        "OVERLAY_FONT_SCALE",
        "WEB_SEARCH_ENABLED",
        "WEB_SEARCH_PROVIDER",
        "WEB_SEARCH_API_KEY",
        "WEB_AGENT_MODE",
        "WEB_SEARCH_MAX_RESULTS",
        "OBSCURA_PATH",
        "SCREEN_VISION_ENABLED",
        "SCREEN_CAPTURE_INDEX",
        "SCREEN_CAPTURE_MAX_WIDTH",
    ):
        monkeypatch.delenv(key, raising=False)

    env_file = tmp_path / ".env"
    original = config.AppConfig(
        llm_provider="gemini",
        gemini_api_key="AIzaSyTestRoundtripKey000011112222",
        gemini_model="gemini-3.8-flash",
        custom_vocabulary="PyQt6, WASAPI, FastAPI",
        rag_context_budget_chars=18000,
        overlay_opacity=85,
        overlay_stealth=True,
        dashboard_stealth=True,
        overlay_font_scale=1.15,
        web_search_enabled=True,
        web_search_provider="tavily",
        web_search_api_key="tvly-roundtrip1234567890",
        web_agent_mode="fast",
        web_search_max_results=7,
        obscura_path="C:/tools/obscura.exe",
        screen_vision_enabled=True,
        screen_capture_index=1,
        screen_capture_max_width=1280,
        system_prompt='Рядок 1: "тест"\nРядок 2: #коментар',
    )
    saved = config.save_config_to_env(original, env_file)
    assert saved == env_file
    assert not list(tmp_path.glob("*.tmp"))

    loaded = config.load_config(str(env_file), override=True)
    assert loaded.llm_provider == "gemini"
    assert loaded.gemini_api_key == "AIzaSyTestRoundtripKey000011112222"
    assert loaded.gemini_model == "gemini-3.8-flash"
    assert loaded.custom_vocabulary == "PyQt6, WASAPI, FastAPI"
    assert loaded.rag_context_budget_chars == 18000
    assert loaded.overlay_opacity == 85
    assert loaded.overlay_stealth is True
    assert loaded.dashboard_stealth is True
    assert abs(loaded.overlay_font_scale - 1.15) < 0.01
    assert loaded.web_search_enabled is True
    assert loaded.web_search_provider == "tavily"
    assert loaded.web_search_api_key == "tvly-roundtrip1234567890"
    assert loaded.web_agent_mode == "fast"
    assert loaded.web_search_max_results == 7
    assert loaded.obscura_path == "C:/tools/obscura.exe"
    assert loaded.screen_vision_enabled is True
    assert loaded.screen_capture_index == 1
    assert loaded.screen_capture_max_width == 1280
    assert loaded.system_prompt == original.system_prompt
    assert config.parse_custom_vocabulary(loaded.custom_vocabulary) == ["PyQt6", "WASAPI", "FastAPI"]


def test_expanded_secret_redaction_stream_and_env_edge_cases(tmp_path: Path, monkeypatch) -> None:
    import io

    security.clear_registered_secrets()
    raw = (
        "Authorization: Bearer YWJjZGVmMTIzNDU2Nzg5MA== | "
        "Authorization: Basic dXNlcjpwYXNzMTIzNA== | "
        "x-goog-api-key: AIzaExpandedHeaderKey123456 | "
        '{"api_key": "json-secret-value-999"} | '
        "https://alice:s3cr3tp4ss@proxy.local:8443/v1?token=qtok987 | "
        "pplx-1234567890abcdef1234 hf_1234567890abcdef1234 ghp_1234567890abcdef123456"
    )
    cleaned = security.redact_secrets(raw)
    for leaked in (
        "YWJjZGVmMTIzNDU2Nzg5MA==",
        "dXNlcjpwYXNzMTIzNA==",
        "AIzaExpandedHeaderKey123456",
        "json-secret-value-999",
        "s3cr3tp4ss",
        "qtok987",
        "pplx-1234567890",
        "hf_1234567890",
        "ghp_1234567890",
    ):
        assert leaked not in cleaned

    sink = io.StringIO()
    stream = security.RedactingTextStream(sink)
    assert stream.encoding == "utf-8"
    assert stream.errors == "replace"
    stream.writelines(["Token: sk-proj-writelines1234567890\n"])
    assert "sk-proj-writelines" not in sink.getvalue()
    assert "***REDACTED***" in sink.getvalue()

    # Non-finite float, lone CR newline normalization, and no interpolation of $VAR
    bad_env = tmp_path / "bad.env"
    bad_env.write_text(
        "LLM_TEMPERATURE=nan\n"
        "AUDIO_CHUNK_SECONDS=inf\n"
        "SYSTEM_PROMPT=Виводь $HOME та ${USER} буквально\n",
        encoding="utf-8",
    )
    loaded_bad = config.load_config(str(bad_env), override=True)
    assert loaded_bad.llm_temperature == config.AppConfig().llm_temperature
    assert loaded_bad.audio_chunk_seconds == config.AppConfig().audio_chunk_seconds
    assert "$HOME" in loaded_bad.system_prompt and "${USER}" in loaded_bad.system_prompt

    # Explicit non-existent env_path must NOT silently read cwd/.env
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    (cwd_dir / ".env").write_text("OPENAI_API_KEY=sk-cwd-should-not-leak-12345\n", encoding="utf-8")
    monkeypatch.chdir(cwd_dir)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    missing_explicit = tmp_path / "does_not_exist.env"
    isolated = config.load_config(str(missing_explicit), override=True)
    assert isolated.openai_api_key == ""

    # Lone \r in system_prompt is normalized cleanly on save/load
    cr_env = tmp_path / "cr.env"
    cr_cfg = config.AppConfig(system_prompt="Рядок1\rРядок2\r\nРядок3")
    config.save_config_to_env(cr_cfg, cr_env)
    reloaded_cr = config.load_config(str(cr_env), override=True)
    assert reloaded_cr.system_prompt == "Рядок1\nРядок2\nРядок3"
