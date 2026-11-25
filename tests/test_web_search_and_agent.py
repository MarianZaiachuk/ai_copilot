"""Tests for web_search TTL cache/backends, SSRF, fast streaming Groq agent, and Screen Vision."""

from __future__ import annotations

from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QApplication
import pytest

import config
import llm_client
import screen_capture
import web_search


def _chunk_text(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))]
    )


def _chunk_tool(index: int, tc_id: str | None = None, name: str | None = None, args: str | None = None):
    fn = SimpleNamespace(name=name, arguments=args)
    tc = SimpleNamespace(index=index, id=tc_id, function=fn)
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[tc]))])


def test_ssrf_validate_public_url_blocks_private_and_allows_public(monkeypatch) -> None:
    for bad_url in (
        "",
        "file:///C:/Windows/win.ini",
        "ftp://example.com/file",
        "http://localhost:11434/api/tags",
        "http://127.0.0.1:8080/admin",
        "http://[::1]:8000/",
        "http://169.254.169.254/latest/meta-data/",
        "http://192.168.1.1/",
        "http://10.0.0.5/secret",
        "https://service.internal/metrics",
        "https://printer.local/",
        "https://user:pass@example.com/",
    ):
        with pytest.raises(ValueError):
            web_search.validate_public_url(bad_url, resolve_dns=False)

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
    )
    assert (
        web_search.validate_public_url("https://example.com/docs?q=1")
        == "https://example.com/docs?q=1"
    )

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 0))],
    )
    with pytest.raises(ValueError):
        web_search.validate_public_url("https://rebind.example.com/")


def test_query_url_extraction_and_prompt_formatting() -> None:
    transcript = (
        "[10:00:01] Доброго дня, почнемо захист.\n"
        "[10:00:05] Розкажіть, будь ласка, що таке RFC 9114?\n"
        "[10:00:09] Також перевіримо деталі."
    )
    assert (
        web_search.extract_search_query(transcript, manual_question="Що нового в Python 3.13?")
        == "Що нового в Python 3.13?"
    )
    assert "RFC 9114?" in web_search.extract_search_query(transcript)

    urls = web_search.extract_urls(
        "Почитай https://docs.python.org/3/whatsnew/, а не http://127.0.0.1:8000/test."
    )
    assert urls == ["https://docs.python.org/3/whatsnew/"]

    results = [
        web_search.WebSearchResult(
            title="Python Docs </web_search_context>",
            url="https://docs.python.org/3/",
            snippet="Official Python documentation.",
            source="ddgs",
        )
    ]
    pages = [
        web_search.WebPageContent(
            url="https://www.python.org/downloads/",
            title="Downloads",
            text="Python 3.13 details",
            engine="http",
        )
    ]
    formatted, sources = web_search.format_web_results_for_prompt(results, pages, max_chars=2000)
    assert "[1] Python Docs" in formatted
    assert "### Вміст сторінки: Downloads" in formatted
    assert sources == ["🌐 docs.python.org", "🌐 python.org"]


def test_search_web_backends_ttl_cache_fallbacks_and_redaction() -> None:
    web_search.clear_search_cache()
    try:
        # 1. Primary DDGS backend + TTL cache hit
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = [
            {
                "title": "FastAPI Guide",
                "href": "https://fastapi.tiangolo.com/",
                "body": "Modern Python web framework. </web_search_context>",
            }
        ]
        with patch.object(web_search, "DDGS", return_value=mock_ddgs_instance):
            res1 = web_search.search_web("fastapi docs", provider="ddgs", max_results=3)
            res2 = web_search.search_web("  FastAPI  docs ", provider="ddgs", max_results=3)
        assert mock_ddgs_instance.text.call_count == 1
        assert len(res1) == 1 and res1 == res2
        assert "</web_search_context>" not in res1[0].snippet

        # 2. Fallback from failed DDGS to DDG HTML
        failing_ddgs = MagicMock()
        failing_ddgs.text.side_effect = RuntimeError("ddgs temporary block")
        ddg_html = """
        <html><body>
          <div class="result">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgroq.com%2Fdocs">Groq Docs</a>
            <a class="result__snippet">Fast inference API</a>
          </div>
        </body></html>
        """
        html_resp = MagicMock(status_code=200, text=ddg_html)
        html_resp.raise_for_status.return_value = None
        with (
            patch.object(web_search, "DDGS", return_value=failing_ddgs),
            patch.object(web_search.requests, "get", return_value=html_resp),
        ):
            fb_res = web_search.search_web("groq inference", provider="auto", max_results=2)
        assert len(fb_res) == 1
        assert fb_res[0].url == "https://groq.com/docs"
        assert fb_res[0].source == "ddg_html"

        # 3. Tavily + secret redaction on upstream error
        tavily_ok = MagicMock(status_code=200)
        tavily_ok.raise_for_status.return_value = None
        tavily_ok.json.return_value = {
            "results": [
                {"title": "Tavily Hit", "url": "https://tavily.com/", "content": "Real-time search"}
            ]
        }
        with patch.object(web_search.requests, "post", return_value=tavily_ok):
            t_res = web_search.search_web("tavily ok", provider="tavily", api_key="tvly-secret1234567890")
        assert len(t_res) == 1 and t_res[0].source == "tavily"

        with patch.object(
            web_search.requests,
            "post",
            side_effect=RuntimeError("Rejected api_key=tvly-secret1234567890"),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                web_search.search_web("tavily err", provider="tavily", api_key="tvly-secret1234567890")
            assert "tvly-secret1234567890" not in str(exc_info.value)
            assert "REDACTED" in str(exc_info.value)

        # 4. Brave empty key raises ValueError; Wikipedia uk -> en fallback works
        with pytest.raises(ValueError):
            web_search.search_web("test", provider="brave", api_key="")

        wiki_uk = MagicMock()
        wiki_uk.raise_for_status.return_value = None
        wiki_uk.json.return_value = {"query": {"search": []}}
        wiki_en = MagicMock()
        wiki_en.raise_for_status.return_value = None
        wiki_en.json.return_value = {
            "query": {"search": [{"title": "PyQt", "snippet": "<span>Python</span> bindings"}]}
        }
        with patch.object(web_search.requests, "get", side_effect=[wiki_uk, wiki_en]):
            w_res = web_search.search_web("PyQt", provider="wikipedia", max_results=2)
        assert len(w_res) == 1
        assert w_res[0].url == "https://en.wikipedia.org/wiki/PyQt"
        assert w_res[0].snippet == "Python bindings"
    finally:
        web_search.clear_search_cache()


def test_fetch_webpage_http_and_optional_obscura(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
    )

    sample_html = (
        b"<html><head><title>WASAPI Guide</title><script>evil()</script></head>"
        b"<body><nav>Home | Menu</nav><main><h1>Loopback</h1><p>"
        + (b"WASAPI records system audio at low latency. " * 6)
        + b"</p></main><footer>Copyright</footer></body></html>"
    )
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.url = "https://example.com/wasapi"
    mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
    mock_resp.encoding = "utf-8"
    mock_resp.iter_content.return_value = [sample_html]

    with patch.object(web_search.requests, "get", return_value=mock_resp):
        page = web_search.fetch_webpage("https://example.com/wasapi")
    assert page.engine == "http"
    assert page.title == "WASAPI Guide"
    assert "Loopback" in page.text
    assert "evil()" not in page.text
    assert "Copyright" not in page.text

    fake_obscura = tmp_path / "obscura.exe"
    fake_obscura.write_text("stub", encoding="utf-8")

    class _OkPopen:
        def __init__(self, *args, **kwargs) -> None:
            self.cmd = args[0] if args else []
            self.returncode = 0

        def poll(self):
            return 0

        def communicate(self, timeout=None):
            return "# Rendered MD\nJS content ok", ""

        def kill(self) -> None:
            pass

        def wait(self, timeout=None) -> int:
            return 0

    popen_holder: list[_OkPopen] = []

    def _ok_popen_factory(*args, **kwargs):
        proc = _OkPopen(*args, **kwargs)
        popen_holder.append(proc)
        return proc

    with patch.object(web_search.subprocess, "Popen", side_effect=_ok_popen_factory):
        obs_page = web_search.fetch_webpage(
            "https://example.com/spa", obscura_path=str(fake_obscura)
        )
    assert obs_page.engine == "obscura"
    assert "Rendered MD" in obs_page.text
    assert popen_holder[0].cmd == [
        str(fake_obscura),
        "fetch",
        "https://example.com/spa",
        "--dump",
        "markdown",
    ]

    class _FailPopen(_OkPopen):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.returncode = 1

        def communicate(self, timeout=None):
            return "", "crash"

    with (
        patch.object(web_search.subprocess, "Popen", side_effect=_FailPopen),
        patch.object(web_search.requests, "get", return_value=mock_resp),
    ):
        fb_page = web_search.fetch_webpage(
            "https://example.com/wasapi", obscura_path=str(fake_obscura)
        )
    assert fb_page.engine == "http"
    assert "Loopback" in fb_page.text


def test_groq_agent_streamed_tool_calling_and_optimizations() -> None:
    # 1. Concise mode: Round 0 streams fragmented tool_calls -> parallel search+fetch -> final stream
    round0_stream = iter(
        [
            _chunk_tool(0, tc_id="call_1", name="web_search", args='{"query": "Groq '),
            _chunk_tool(0, args='fast"}'),
            _chunk_tool(1, tc_id="call_2", name="fetch_webpage", args='{"url": "https://groq.com/docs"}'),
        ]
    )
    final_stream = iter([_chunk_text("- Groq "), _chunk_text("швидко")])

    fake_oai = MagicMock()
    fake_oai.chat.completions.create.side_effect = [round0_stream, final_stream]

    statuses: list[str] = []
    sources_seen: list[list[str]] = []

    cfg = config.AppConfig(
        llm_provider="groq",
        openai_api_key="gsk_agent_test_1234567890",
        openai_model="qwen/qwen3.8-27b",
        web_search_enabled=True,
        web_agent_mode="agent",
    )
    with (
        patch.object(llm_client, "OpenAI", return_value=fake_oai),
        patch.object(
            llm_client,
            "search_web",
            return_value=[
                web_search.WebSearchResult(
                    title="Groq Docs", url="https://groq.com/docs", snippet="Low latency"
                )
            ],
        ) as sw_mock,
        patch.object(
            llm_client,
            "fetch_webpage",
            return_value=web_search.WebPageContent(
                url="https://groq.com/docs", title="Groq Docs", text="Details here", engine="http"
            ),
        ) as fw_mock,
    ):
        chunks = list(
            llm_client.LLMClient(cfg).stream_answer(
                "Як прискорити Groq?",
                detail_mode="concise",
                on_status=statuses.append,
                on_sources=sources_seen.append,
            )
        )
    assert chunks == ["- Groq ", "швидко"]
    assert sw_mock.call_count == 1
    assert fw_mock.call_count == 1
    assert fake_oai.chat.completions.create.call_count == 2  # 1 tool round + final stream
    assert fake_oai.chat.completions.create.call_args_list[0].kwargs["stream"] is True
    assert fake_oai.chat.completions.create.call_args_list[0].kwargs["extra_body"] == {
        "reasoning_effort": "none"
    }
    assert any("🌐 Шукаю:" in s for s in statuses)
    assert sources_seen[-1] == ["🌐 groq.com"]

    # 2. Round 0 without tool calls streams directly with zero extra round-trip
    fake_direct = MagicMock()
    fake_direct.chat.completions.create.return_value = iter(
        [_chunk_text("- З документів "), _chunk_text("прямо")]
    )
    with (
        patch.object(llm_client, "OpenAI", return_value=fake_direct),
        patch.object(llm_client, "search_web") as no_search_mock,
    ):
        direct_chunks = list(llm_client.LLMClient(cfg).stream_answer("Що у звіті?"))
    assert direct_chunks == ["- З документів ", "прямо"]
    assert fake_direct.chat.completions.create.call_count == 1
    assert no_search_mock.call_count == 0

    # 3. Tool-calling 400 fallback -> fast direct search + stream
    fake_oai_fb = MagicMock()
    fast_stream = iter([_chunk_text("- Фолбек OK")])
    fake_oai_fb.chat.completions.create.side_effect = [
        RuntimeError("400 tool_use_failed"),
        fast_stream,
    ]
    fb_sources: list[list[str]] = []
    with (
        patch.object(llm_client, "OpenAI", return_value=fake_oai_fb),
        patch.object(
            llm_client,
            "search_web",
            return_value=[
                web_search.WebSearchResult(
                    title="Python Info", url="https://docs.python.org/3/", snippet="Python info"
                )
            ],
        ),
    ):
        fb_chunks = list(
            llm_client.LLMClient(cfg).stream_answer(
                "Що нового в Python?",
                on_sources=fb_sources.append,
            )
        )
    assert fb_chunks == ["- Фолбек OK"]
    assert fb_sources == [["🌐 docs.python.org"]]
    second_call_msgs = fake_oai_fb.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert "<web_search_context>" in second_call_msgs[1]["content"]

    # 4. Native groq/compound-mini streams directly when no image is attached
    fake_compound = MagicMock()
    fake_compound.chat.completions.create.return_value = iter([_chunk_text("- Compound OK")])
    cmp_cfg = config.AppConfig(
        llm_provider="groq",
        openai_api_key="gsk_agent_test_1234567890",
        openai_model="groq/compound-mini",
        web_search_enabled=True,
    )
    cmp_sources: list[list[str]] = []
    with (
        patch.object(llm_client, "OpenAI", return_value=fake_compound),
        patch.object(llm_client, "search_web") as no_local_sw,
    ):
        cmp_chunks = list(
            llm_client.LLMClient(cmp_cfg).stream_answer(
                "Останні новини ШІ?", on_sources=cmp_sources.append
            )
        )
    assert cmp_chunks == ["- Compound OK"]
    assert no_local_sw.call_count == 0
    assert cmp_sources == [["🌐 Groq Compound Web"]]

    # 5. Gemini native Google Search grounding attaches GoogleSearch tool and reads metadata
    fake_gem = MagicMock()
    grounding_chunk = SimpleNamespace(
        web=SimpleNamespace(title="wikipedia.org", uri="https://en.wikipedia.org/wiki/Python")
    )
    gem_chunk = SimpleNamespace(
        text="- З інтернету",
        candidates=[
            SimpleNamespace(grounding_metadata=SimpleNamespace(grounding_chunks=[grounding_chunk]))
        ],
    )
    fake_gem.models.generate_content_stream.return_value = iter([gem_chunk])
    gem_cfg = config.AppConfig(
        llm_provider="gemini",
        gemini_api_key="AIzaGemTestKey123456789012345",
        gemini_model="gemini-3.8-flash",
        web_search_enabled=True,
    )
    gem_sources: list[list[str]] = []
    with patch.object(llm_client.genai, "Client", return_value=fake_gem):
        gem_chunks = list(
            llm_client.LLMClient(gem_cfg).stream_answer("Курс євро?", on_sources=gem_sources.append)
        )
    assert gem_chunks == ["- З інтернету"]
    assert gem_sources == [["🌐 wikipedia.org"]]
    used_cfg = fake_gem.models.generate_content_stream.call_args.kwargs["config"]
    assert used_cfg.tools and len(used_cfg.tools) == 1


def test_screen_capture_and_multimodal_vision() -> None:
    _app = QApplication.instance() or QApplication([])
    screens = screen_capture.list_screens()
    assert len(screens) >= 1

    shot = screen_capture.capture_screen(max_width=1280, max_height=720)
    assert shot is not None
    assert shot.image_bytes[:3] == b"\xff\xd8\xff"
    assert shot.to_data_url().startswith("data:image/jpeg;base64,")

    # 1. Groq multimodal qwen/qwen3.8-27b receives image_url payload
    fake_qwen = MagicMock()
    fake_qwen.chat.completions.create.return_value = iter([_chunk_text("- Помилка у рядку 42")])
    vis_cfg = config.AppConfig(
        llm_provider="groq",
        openai_api_key="gsk_vision_test_1234567890",
        openai_model="qwen/qwen3.8-27b",
        web_search_enabled=False,
        screen_vision_enabled=True,
    )
    with patch.object(llm_client, "OpenAI", return_value=fake_qwen):
        vis_chunks = list(
            llm_client.LLMClient(vis_cfg).stream_answer(
                "Поясни цей код",
                screen_image_bytes=shot.image_bytes,
                screen_image_mime=shot.mime_type,
            )
        )
    assert vis_chunks == ["- Помилка у рядку 42"]
    sent_msgs = fake_qwen.chat.completions.create.call_args.kwargs["messages"]
    assert isinstance(sent_msgs[1]["content"], list)
    assert sent_msgs[1]["content"][1]["type"] == "image_url"
    assert sent_msgs[1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    # 2. Text-only Groq model auto-routes to qwen/qwen3.8-27b and allows empty text+transcript when screenshot is attached
    fake_routed = MagicMock()
    fake_routed.chat.completions.create.return_value = iter([_chunk_text("- Код коректний")])
    text_only_cfg = config.AppConfig(
        llm_provider="groq",
        openai_api_key="gsk_vision_test_1234567890",
        openai_model="llama-3.1-8b-instant",
        web_search_enabled=False,
        screen_vision_enabled=True,
    )
    statuses: list[str] = []
    with patch.object(llm_client, "OpenAI", return_value=fake_routed):
        routed_chunks = list(
            llm_client.LLMClient(text_only_cfg).stream_answer(
                "",
                manual_question="",
                on_status=statuses.append,
                screen_image_bytes=shot.image_bytes,
                screen_image_mime=shot.mime_type,
            )
        )
    assert routed_chunks == ["- Код коректний"]
    assert fake_routed.chat.completions.create.call_args.kwargs["model"] == "qwen/qwen3.8-27b"
    assert any("qwen/qwen3.8-27b" in s for s in statuses)

    # 3. Gemini Vision passes [prompt, Part]
    fake_gem = MagicMock()
    fake_gem.models.generate_content_stream.return_value = iter([SimpleNamespace(text="- Складність O(n)")])
    gem_vis_cfg = config.AppConfig(
        llm_provider="gemini",
        gemini_api_key="AIzaGemVision123456789012345",
        gemini_model="gemini-3.8-flash",
        web_search_enabled=False,
        screen_vision_enabled=True,
    )
    with patch.object(llm_client.genai, "Client", return_value=fake_gem):
        gem_vis_chunks = list(
            llm_client.LLMClient(gem_vis_cfg).stream_answer(
                "Яка складність функції?",
                screen_image_bytes=shot.image_bytes,
                screen_image_mime=shot.mime_type,
            )
        )
    assert gem_vis_chunks == ["- Складність O(n)"]
    gem_contents = fake_gem.models.generate_content_stream.call_args.kwargs["contents"]
    assert isinstance(gem_contents, list) and len(gem_contents) == 2


def test_docs_first_policy_across_agent_fast_and_gemini() -> None:
    # 1. Agent mode with loaded documents: checks <documents_context> first, answers directly without web search
    fake_agent = MagicMock()
    fake_agent.chat.completions.create.return_value = iter(
        [_chunk_text("- У роботі "), _chunk_text("використовується FAST-X")]
    )
    agent_cfg = config.AppConfig(
        llm_provider="groq",
        openai_api_key="gsk_docs_agent_1234567890",
        openai_model="qwen/qwen3.8-27b",
        web_search_enabled=True,
        web_agent_mode="agent",
    )
    agent_statuses: list[str] = []
    with (
        patch.object(llm_client, "OpenAI", return_value=fake_agent),
        patch.object(llm_client, "search_web") as agent_sw,
    ):
        agent_chunks = list(
            llm_client.LLMClient(agent_cfg).stream_answer(
                "Який протокол у моєму проєкті?",
                context_text="Мій проєкт використовує FAST-X протокол.",
                on_status=agent_statuses.append,
            )
        )
    assert agent_chunks == ["- У роботі ", "використовується FAST-X"]
    assert agent_sw.call_count == 0
    assert any("Аналізую ваші документи" in s for s in agent_statuses)
    sent_msgs = fake_agent.chat.completions.create.call_args.kwargs["messages"]
    assert "MANDATORY ORDER" in sent_msgs[0]["content"]

    # 2. Fast mode with loaded documents (answer exists in docs -> zero web calls)
    fake_fast_hit = MagicMock()
    fake_fast_hit.chat.completions.create.return_value = iter(
        [_chunk_text("- У документах"), _chunk_text(": PyQt6 + WASAPI.")]
    )
    fast_cfg = config.AppConfig(
        llm_provider="groq",
        openai_api_key="gsk_docs_fast_1234567890",
        openai_model="qwen/qwen3.8-27b",
        web_search_enabled=True,
        web_agent_mode="fast",
    )
    fast_statuses: list[str] = []
    with (
        patch.object(llm_client, "OpenAI", return_value=fake_fast_hit),
        patch.object(llm_client, "search_web") as fast_sw_hit,
    ):
        fast_hit_chunks = list(
            llm_client.LLMClient(fast_cfg).stream_answer(
                "На чому написаний інтерфейс?",
                context_text="Архітектура застосунку: PyQt6 + WASAPI.",
                on_status=fast_statuses.append,
            )
        )
    assert fast_hit_chunks == ["- У документах", ": PyQt6 + WASAPI."]
    assert fast_sw_hit.call_count == 0
    assert any("Аналізую ваші документи" in s for s in fast_statuses)

    # 3. Fast mode when answer is missing from docs -> catches fragmented [[NEED_WEB: ...]] marker and falls back to search
    fake_fast_miss = MagicMock()
    gate_stream = iter(
        [_chunk_text("  [[NEED"), _chunk_text("_WEB: курс євро НБУ 2026]]")]
    )
    web_stream = iter([_chunk_text("- Курс [веб: bank.gov.ua]")])
    fake_fast_miss.chat.completions.create.side_effect = [gate_stream, web_stream]
    miss_statuses: list[str] = []
    miss_sources: list[list[str]] = []
    with (
        patch.object(llm_client, "OpenAI", return_value=fake_fast_miss),
        patch.object(
            llm_client,
            "search_web",
            return_value=[
                web_search.WebSearchResult(
                    title="НБУ Курс",
                    url="https://bank.gov.ua/ua/markets/exchangerates",
                    snippet="Офіційний курс євро",
                )
            ],
        ) as fast_sw_miss,
    ):
        miss_chunks = list(
            llm_client.LLMClient(fast_cfg).stream_answer(
                "Який сьогодні курс євро?",
                context_text="Локальний конспект містить лише лекції з ОС.",
                on_status=miss_statuses.append,
                on_sources=miss_sources.append,
            )
        )
    assert miss_chunks == ["- Курс [веб: bank.gov.ua]"]
    assert all("NEED_WEB" not in c for c in miss_chunks)
    assert fast_sw_miss.call_count == 1
    called_query = fast_sw_miss.call_args.args[0] if fast_sw_miss.call_args.args else fast_sw_miss.call_args.kwargs.get("query", "")
    assert "курс євро" in called_query
    assert any("У файлах не знайдено" in s for s in miss_statuses)
    assert miss_sources[-1] == ["🌐 bank.gov.ua"]

    # 4. Gemini Docs-First: (a) answered from docs without GoogleSearch tool; (b) [[NEED_WEB]] escalates to GoogleSearch
    fake_gem_hit = MagicMock()
    fake_gem_hit.models.generate_content_stream.return_value = iter(
        [SimpleNamespace(text="- Відповідь з документа")]
    )
    gem_cfg = config.AppConfig(
        llm_provider="gemini",
        gemini_api_key="AIzaDocsFirstKey12345678901",
        gemini_model="gemini-3.8-flash",
        web_search_enabled=True,
    )
    with patch.object(llm_client.genai, "Client", return_value=fake_gem_hit):
        gem_hit_chunks = list(
            llm_client.LLMClient(gem_cfg).stream_answer(
                "Що таке модуль A?",
                context_text="Модуль A відповідає за захоплення звуку.",
            )
        )
    assert gem_hit_chunks == ["- Відповідь з документа"]
    assert fake_gem_hit.models.generate_content_stream.call_count == 1
    assert not fake_gem_hit.models.generate_content_stream.call_args.kwargs["config"].tools

    fake_gem_miss = MagicMock()
    gem_gate = iter([SimpleNamespace(text="[[NEED_WEB: RFC 9114 HTTP/3]]")])
    gem_grounding = SimpleNamespace(
        web=SimpleNamespace(title="rfc-editor.org", uri="https://www.rfc-editor.org/rfc/rfc9114")
    )
    gem_web = iter(
        [
            SimpleNamespace(
                text="- RFC 9114 визначає HTTP/3",
                candidates=[
                    SimpleNamespace(
                        grounding_metadata=SimpleNamespace(grounding_chunks=[gem_grounding])
                    )
                ],
            )
        ]
    )
    fake_gem_miss.models.generate_content_stream.side_effect = [gem_gate, gem_web]
    gem_miss_sources: list[list[str]] = []
    with patch.object(llm_client.genai, "Client", return_value=fake_gem_miss):
        gem_miss_chunks = list(
            llm_client.LLMClient(gem_cfg).stream_answer(
                "Який RFC описує HTTP/3?",
                context_text="Локальний звіт про UDP сокети.",
                on_sources=gem_miss_sources.append,
            )
        )
    assert gem_miss_chunks == ["- RFC 9114 визначає HTTP/3"]
    assert fake_gem_miss.models.generate_content_stream.call_count == 2
    assert not fake_gem_miss.models.generate_content_stream.call_args_list[0].kwargs["config"].tools
    assert len(fake_gem_miss.models.generate_content_stream.call_args_list[1].kwargs["config"].tools) == 1
    assert gem_miss_sources[-1] == ["🌐 rfc-editor.org"]


def test_ssrf_edge_cases_charset_cancellation_and_cache_isolation(monkeypatch) -> None:
    import threading

    # SSRF: IPv4-mapped IPv6, non-HTTP port, single-label host, fail-closed DNS
    for blocked in (
        "http://[::ffff:127.0.0.1]/admin",
        "https://example.com:22/ssh",
        "http://intranet/secret",
    ):
        with pytest.raises(ValueError):
            web_search.validate_public_url(blocked)

    monkeypatch.setattr(
        web_search.socket,
        "getaddrinfo",
        MagicMock(side_effect=socket.gaierror("DNS failed")),
    )
    with pytest.raises(ValueError):
        web_search.validate_public_url("https://unresolvable.example.com/")

    # Ukrainian charset detection: ISO-8859-1 header but <meta charset="utf-8"> Cyrillic body
    monkeypatch.setattr(
        web_search.socket,
        "getaddrinfo",
        lambda host, port, *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
    )
    html_utf8_bytes = (
        '<html><head><meta charset="utf-8"><title>Тест</title></head>'
        "<body><main><p>Привіт, українська енциклопедія програмування та алгоритмів "
        "для швидкого аналізу документації!</p></main></body></html>"
    ).encode("utf-8")

    fake_resp = MagicMock()
    fake_resp.raise_for_status.return_value = None
    fake_resp.status_code = 200
    fake_resp.headers = {"Content-Type": "text/html"}
    fake_resp.encoding = "ISO-8859-1"
    fake_resp.url = "https://example.com/ua"
    fake_resp.iter_content.return_value = [html_utf8_bytes]
    with patch.object(web_search.requests, "get", return_value=fake_resp):
        extracted = web_search.fetch_webpage("https://example.com/ua")
    assert "Привіт, українська" in extracted.text

    # cancel_event aborts search_web immediately
    cancel = threading.Event()
    cancel.set()
    web_search.clear_search_cache()
    try:
        with pytest.raises(InterruptedError):
            web_search.search_web("cancelled query", provider="wikipedia", cancel_event=cancel)

        # Cache isolates api_key hashes
        calls: list[str] = []

        def _fake_tavily(query: str, api_key: str, *args, **kwargs):
            calls.append(api_key)
            return [web_search.WebSearchResult(title=f"R-{api_key}", url="https://example.com", snippet="ok")]

        with patch.object(web_search, "_search_tavily", side_effect=_fake_tavily):
            r1 = web_search.search_web("same query", provider="tavily", api_key="tvly-key-one-11111")
            r2 = web_search.search_web("same query", provider="tavily", api_key="tvly-key-two-22222")
        assert calls == ["tvly-key-one-11111", "tvly-key-two-22222"]
        assert r1[0].title == "R-tvly-key-one-11111" and r2[0].title == "R-tvly-key-two-22222"
    finally:
        web_search.clear_search_cache()
