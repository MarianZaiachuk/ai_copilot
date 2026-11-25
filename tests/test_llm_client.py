"""Tests for LLM client: 2026 models, prompts, pooling, key checks, and fault injection."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import config
import llm_client


class _HttpErr(Exception):
    def __init__(self, msg: str, status_code: int) -> None:
        super().__init__(msg)
        self.status_code = status_code


def test_2026_defaults_and_xml_delimited_prompts() -> None:
    cfg = config.AppConfig()
    assert cfg.gemini_model == "gemini-3.8-flash"
    assert cfg.openai_model == "gpt-6-luna"
    assert llm_client.get_deprecated_model_replacement("gemini-2.5-flash") == "gemini-3.8-flash"

    concise = llm_client.build_user_prompt(
        transcript_text="Що таке WASAPI?",
        context_text="WASAPI перехоплює звук.",
        manual_question="Коротко",
        detail_mode="concise",
    )
    assert "<documents_context>" in concise
    assert "<live_transcript>" in concise
    assert "<user_question>" in concise
    assert concise.index("<documents_context>") < concise.index("<live_transcript>")
    assert "system prompt" in concise
    assert "українською" not in concise
    assert "3–5" not in concise

    detailed = llm_client.build_user_prompt("Що таке WASAPI?", detail_mode="detailed")
    assert "detailed" in detailed
    assert "<documents_context>" not in detailed
    assert "VERY LAST" in concise
    assert "visual graph" in llm_client._GROUNDING_BOUNDARY_RULE
    assert "графік" in llm_client._SCREEN_VISION_RULE


def test_list_available_models_filtering_for_gemini_and_groq() -> None:
    fake_gem = MagicMock()
    fake_gem.models.list.return_value = [
        SimpleNamespace(name="models/gemini-3.8-flash", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-4-hydrogen", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/text-embedding-004", supported_actions=["embedContent"]),
        SimpleNamespace(name="models/imagen-4.0-generate", supported_actions=["predict"]),
        SimpleNamespace(name="models/gemini-3.5-flash-lite", supported_actions=["generateContent"]),
    ]
    with patch.object(llm_client.genai, "Client", return_value=fake_gem):
        gem_models = llm_client.LLMClient(
            config.AppConfig(llm_provider="gemini", gemini_api_key="AIzaTest12345678901234567890")
        ).list_available_models()
        assert gem_models == ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-4-hydrogen"]

    fake_oai = MagicMock()
    fake_oai.models.list.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(id="whisper-large-v3"),
            SimpleNamespace(id="qwen/qwen3.8-27b"),
            SimpleNamespace(id="meta-llama/llama-prompt-guard-2-86m"),
            SimpleNamespace(id="openai/gpt-oss-20b"),
        ]
    )
    with patch.object(llm_client, "OpenAI", return_value=fake_oai):
        groq_models = llm_client.LLMClient(
            config.AppConfig(llm_provider="groq", openai_api_key="gsk_test12345678901234")
        ).list_available_models()
        assert groq_models == ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"]


def test_verify_api_key_all_four_outcomes() -> None:
    fake_gem = MagicMock()
    fake_gem.models.list.return_value = [
        SimpleNamespace(name="models/gemini-3.8-flash", supported_actions=["generateContent"])
    ]
    fake_gem.models.generate_content.return_value = SimpleNamespace(text="ОК")
    with patch.object(llm_client.genai, "Client", return_value=fake_gem):
        ok_res = llm_client.LLMClient(
            config.AppConfig(llm_provider="gemini", gemini_api_key="AIzaValid12345678901234567", gemini_model="gemini-3.8-flash")
        ).verify_api_key()
        assert ok_res.ok and ok_res.key_valid and ok_res.model_valid and ok_res.status == "ok"

    fake_gem.models.generate_content.side_effect = _HttpErr("models/gemini-2.5-flash not found", 404)
    with patch.object(llm_client.genai, "Client", return_value=fake_gem):
        dep_res = llm_client.LLMClient(
            config.AppConfig(llm_provider="gemini", gemini_api_key="AIzaValid12345678901234567", gemini_model="gemini-2.5-flash")
        ).verify_api_key()
        assert not dep_res.ok and dep_res.key_valid and not dep_res.model_valid and dep_res.status == "warn"

    fake_gem_bad = MagicMock()
    fake_gem_bad.models.list.side_effect = _HttpErr("API_KEY_INVALID", 401)
    with patch.object(llm_client.genai, "Client", return_value=fake_gem_bad):
        bad_res = llm_client.LLMClient(
            config.AppConfig(llm_provider="gemini", gemini_api_key="AIzaBad1234567890123456789")
        ).verify_api_key()
        assert not bad_res.ok and not bad_res.key_valid and bad_res.status == "error"

    fake_gem_quota = MagicMock()
    fake_gem_quota.models.list.return_value = [
        SimpleNamespace(name="models/gemini-3.8-flash", supported_actions=["generateContent"])
    ]
    fake_gem_quota.models.generate_content.side_effect = _HttpErr("RESOURCE_EXHAUSTED quota", 429)
    with patch.object(llm_client.genai, "Client", return_value=fake_gem_quota):
        q_res = llm_client.LLMClient(
            config.AppConfig(llm_provider="gemini", gemini_api_key="AIzaValid12345678901234567", gemini_model="gemini-3.8-flash")
        ).verify_api_key()
        assert not q_res.ok and q_res.key_valid and not q_res.model_valid and q_res.status == "warn"


def test_client_pooling_cancellation_and_error_redaction() -> None:
    secret_key = "sk-supersecretpoolkey12345678"
    fake_oai = MagicMock()
    fake_oai.chat.completions.create.return_value = iter(
        [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="- Теза"))])]
    )

    with patch.object(llm_client, "OpenAI", return_value=fake_oai) as ctor:
        client = llm_client.LLMClient(config.AppConfig(llm_provider="openai", openai_api_key=secret_key))
        assert list(client.stream_answer("Питання?")) == ["- Теза"]
        assert ctor.call_args.kwargs["max_retries"] == 0

        # Second call reuses pooled OpenAI client without reconstructing
        fake_oai.chat.completions.create.return_value = iter(
            [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="- Теза 2"))])]
        )
        assert list(client.stream_answer("Питання 2?")) == ["- Теза 2"]
        assert ctor.call_count == 1

        # Cancellation interrupts stream immediately
        cancel_ev = threading.Event()
        cancel_ev.set()
        fake_oai.chat.completions.create.return_value = iter(
            [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="- Не має вийти"))])]
        )
        assert list(client.stream_answer("Питання?", cancel_event=cancel_ev)) == []

        # update_config invalidates pooled client
        client.update_config(config.AppConfig(llm_provider="openai", openai_api_key="sk-newkey999999999999999"))
        fake_oai.chat.completions.create.side_effect = RuntimeError(
            f"Upstream rejected sk-newkey999999999999999 and {secret_key}"
        )
        with pytest.raises(RuntimeError) as exc_info:
            list(client.stream_answer("Питання?"))
        assert "sk-newkey" not in str(exc_info.value)
        assert "sk-supersecret" not in str(exc_info.value)
        assert "***REDACTED***" in str(exc_info.value)


def test_groq_429_instant_fallback_and_cooldown_skip() -> None:
    def _chunk(text: str):
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))]
        )

    fake_oai = MagicMock()
    # Call 1: qwen/qwen3.8-27b hits 429 -> instant fallback to openai/gpt-oss-20b (0s sleep)
    # Call 2: skips cooled-down qwen/qwen3.8-27b and calls openai/gpt-oss-20b directly
    # Call 3: openai/gpt-oss-20b returns 404 -> instant fallback to llama-3.1-8b-instant
    # Call 4: calls llama-3.1-8b-instant directly
    fake_oai.chat.completions.create.side_effect = [
        _HttpErr("Rate limit reached. Please try again in 21.0s", 429),
        iter([_chunk("- Швидка відповідь")]),
        iter([_chunk("- Друга без очікування")]),
        _HttpErr("model_not_found", 404),
        iter([_chunk("- З Llama 3.1")]),
        iter([_chunk("- Наступна Llama 3.1")]),
    ]

    cfg = config.AppConfig(
        llm_provider="groq",
        openai_api_key="gsk_fallback_test_1234567890",
        openai_model="qwen/qwen3.8-27b",
        web_search_enabled=True,
        web_agent_mode="agent",
    )
    statuses1: list[str] = []
    statuses2: list[str] = []
    with (
        patch.object(llm_client, "OpenAI", return_value=fake_oai) as ctor,
        patch.object(llm_client.time, "sleep") as sleep_mock,
    ):
        client = llm_client.LLMClient(cfg)

        chunks1 = list(
            client.stream_answer(
                "Поясни архітектуру",
                context_text="Клас OrderService обробляє замовлення.",
                on_status=statuses1.append,
            )
        )
        assert ctor.call_args.kwargs["max_retries"] == 0
        assert chunks1 == ["- Швидка відповідь"]
        assert sleep_mock.call_count == 0
        assert fake_oai.chat.completions.create.call_count == 2
        assert fake_oai.chat.completions.create.call_args_list[0].kwargs["model"] == "qwen/qwen3.8-27b"
        assert fake_oai.chat.completions.create.call_args_list[1].kwargs["model"] == "openai/gpt-oss-20b"
        assert any("openai/gpt-oss-20b" in s for s in statuses1)

        # Call 2 immediately skips cooling qwen/qwen3.8-27b -> only 1 create() call!
        chunks2 = list(client.stream_answer("Друге питання", on_status=statuses2.append))
        assert chunks2 == ["- Друга без очікування"]
        assert fake_oai.chat.completions.create.call_count == 3
        assert fake_oai.chat.completions.create.call_args_list[2].kwargs["model"] == "openai/gpt-oss-20b"
        assert any("openai/gpt-oss-20b" in s for s in statuses2)

        # Call 3: openai/gpt-oss-20b 404 -> falls back to llama-3.1-8b-instant
        chunks3 = list(client.stream_answer("Третє питання"))
        assert chunks3 == ["- З Llama 3.1"]
        assert fake_oai.chat.completions.create.call_args_list[3].kwargs["model"] == "openai/gpt-oss-20b"
        assert fake_oai.chat.completions.create.call_args_list[4].kwargs["model"] == "llama-3.1-8b-instant"

        # Call 4: directly uses llama-3.1-8b-instant (skips cooled qwen & unavailable gpt-oss-20b)
        chunks4 = list(client.stream_answer("Четверте питання"))
        assert chunks4 == ["- Наступна Llama 3.1"]
        assert fake_oai.chat.completions.create.call_count == 6
        assert fake_oai.chat.completions.create.call_args_list[5].kwargs["model"] == "llama-3.1-8b-instant"


def test_think_filter_error_classification_history_and_response_cache() -> None:
    # 1. ThinkTagFilter strips <think> blocks split arbitrarily across chunks
    f = llm_client.ThinkTagFilter()
    tokens = ["<thi", "nk>Прихо", "вані роздуми</th", "ink>", "- Фіналь", "на теза"]
    out = "".join(f.feed(tok) for tok in tokens) + f.finish()
    assert "Приховані" not in out
    assert out == "- Фінальна теза"

    # 2. _classify_api_error does NOT misclassify 400 / 4048 token limit as not_found
    assert llm_client._classify_api_error(RuntimeError("Error 400: max_tokens 4048 exceeded")) == "other"
    assert llm_client._classify_api_error(_HttpErr("model_not_found", 404)) == "not_found"
    assert llm_client._classify_api_error(_HttpErr("Rate limit reached 429", 429)) == "quota"

    # 3. 2-turn history, TTL cache, context_revision invalidation, bypass_cache, and on_model_used
    fake_oai = MagicMock()
    fake_oai.chat.completions.create.return_value = iter(
        [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="- Кешована теза"))])]
    )
    cfg = config.AppConfig(
        llm_provider="openai",
        openai_api_key="sk-cache-test-1234567890",
        openai_model="gpt-6-luna",
    )
    with patch.object(llm_client, "OpenAI", return_value=fake_oai):
        client = llm_client.LLMClient(cfg)
        used_routes: list[tuple[str, str]] = []

        # Call 1 -> populates cache, includes <recent_dialogue>
        c1 = list(
            client.stream_answer(
                "Поточне питання",
                history=(("Минуле питання?", "- Минула відповідь"),),
                context_revision="r1",
                on_model_used=lambda m, r: used_routes.append((m, r)),
            )
        )
        assert c1 == ["- Кешована теза"]
        assert used_routes[-1] == ("gpt-6-luna", "")
        sent_user_prompt = fake_oai.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
        assert "<recent_dialogue>" in sent_user_prompt
        assert "Минуле питання?" in sent_user_prompt

        # Call 2 with identical args & revision -> instant cache hit (no new create call)
        c2 = list(
            client.stream_answer(
                "Поточне питання",
                history=(("Минуле питання?", "- Минула відповідь"),),
                context_revision="r1",
                on_model_used=lambda m, r: used_routes.append((m, r)),
            )
        )
        assert c2 == ["- Кешована теза"]
        assert used_routes[-1][1] == "cache"
        assert fake_oai.chat.completions.create.call_count == 1

        # Call 3 with changed context_revision or bypass_cache=True -> misses cache
        fake_oai.chat.completions.create.return_value = iter(
            [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="- Оновлена теза"))])]
        )
        c3 = list(
            client.stream_answer(
                "Поточне питання",
                history=(("Минуле питання?", "- Минула відповідь"),),
                context_revision="r2",
                on_model_used=lambda m, r: used_routes.append((m, r)),
            )
        )
        assert c3 == ["- Оновлена теза"]
        assert used_routes[-1] == ("gpt-6-luna", "")
        assert fake_oai.chat.completions.create.call_count == 2


def test_cerebras_provider_settings_sorting_and_redaction() -> None:
    from security import redact_secrets

    assert "cerebras" in config.VALID_LLM_PROVIDERS

    cfg_with_key = config.AppConfig(
        llm_provider="cerebras",
        openai_api_key="csk-1234567890abcdef",
        openai_model="qwen-3.8-27b",
    )
    cfg_no_key = config.AppConfig(llm_provider="cerebras", openai_api_key="")

    assert config.has_valid_llm_credentials(cfg_with_key) is True
    assert config.has_valid_llm_credentials(cfg_no_key) is False
    assert llm_client.resolve_openai_settings(cfg_with_key) == (
        "csk-1234567890abcdef",
        "https://api.cerebras.ai/v1",
    )

    ordered = llm_client._sort_openai_models(
        ["llama-3.3-70b", "gpt-oss-120b", "qwen-3.8-27b"],
        "cerebras",
    )
    assert ordered == ["qwen-3.8-27b", "gpt-oss-120b", "llama-3.3-70b"]

    masked = redact_secrets("Authorization: csk-1234567890abcdef")
    assert "csk-1234567890abcdef" not in masked
    assert "***REDACTED***" in masked


def test_cerebras_reasoning_budget_vision_routing_and_empty_retry() -> None:
    # 1. qwen-3.8-27b in concise mode disables reasoning tokens; gpt-oss-120b uses low/hidden + boosted max_tokens
    assert llm_client._groq_reasoning_extra_body("cerebras", "qwen-3.8-27b", "concise") == {
        "reasoning_effort": "none"
    }
    oss_extra = llm_client._groq_reasoning_extra_body("cerebras", "gpt-oss-120b", "concise")
    assert oss_extra == {"reasoning_effort": "low", "reasoning_format": "hidden"}
    assert llm_client._effective_max_tokens(512, oss_extra) == 1200

    # 2. Cerebras text-only gpt-oss-120b auto-routes to qwen-3.8-27b when screen image is attached
    fake_oai = MagicMock()
    captured_models: list[str] = []
    captured_max_tokens: list[int] = []

    def _fake_create(**kwargs):
        captured_models.append(kwargs["model"])
        captured_max_tokens.append(kwargs["max_tokens"])
        return iter(
            [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="- Бачу код"))])]
        )

    fake_oai.chat.completions.create.side_effect = _fake_create
    cfg_oss = config.AppConfig(
        llm_provider="cerebras",
        cerebras_api_key="csk-test1234567890abcd",
        cerebras_model="gpt-oss-120b",
    )
    with patch.object(llm_client, "OpenAI", return_value=fake_oai):
        client = llm_client.LLMClient(cfg_oss)
        chunks = list(
            client.stream_answer(
                "",
                manual_question="Що на екрані?",
                screen_image_bytes=b"\xff\xd8\xff\xe0fakejpg",
                screen_image_mime="image/jpeg",
            )
        )
    assert chunks == ["- Бачу код"]
    assert captured_models == ["qwen-3.8-27b"]

    # 3. Empty stream triggers one retry with boosted max_tokens, then raises clear RuntimeError if still empty
    fake_oai_empty = MagicMock()
    retry_tokens: list[int] = []

    def _empty_create(**kwargs):
        retry_tokens.append(kwargs["max_tokens"])
        return iter(
            [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])]
        )

    fake_oai_empty.chat.completions.create.side_effect = _empty_create
    cfg_qwen = config.AppConfig(
        llm_provider="cerebras",
        cerebras_api_key="csk-test1234567890abcd",
        cerebras_model="qwen-3.8-27b",
        llm_max_tokens=512,
    )
    with patch.object(llm_client, "OpenAI", return_value=fake_oai_empty):
        client_q = llm_client.LLMClient(cfg_qwen)
        with pytest.raises(RuntimeError, match="порожню відповідь"):
            list(client_q.stream_answer("Що на екрані?"))
    assert retry_tokens == [512, 1536]
