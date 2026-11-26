"""Tests for v1.3 features: LaTeX rendering, LLM failover, snip, card history, profile export/import, diagnostics & safe startup."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import QApplication, QTextBrowser
from pynput import keyboard
import pytest

import config
import dashboard
import diagnostics
import latex_renderer
import llm_client
import main as app_main
import screen_capture
import ui


def test_latex_renderer_and_qtextbrowser_integration() -> None:
    src = (
        "Ціна квитка $100 або $50 не є формулою.\n\n"
        "Період коливань $T = \\frac{2\\pi}{\\omega}$ і корінь $\\sqrt[3]{x^2 + y_1} \\le \\infty$.\n\n"
        "$$E = m c^2 \\Rightarrow \\Delta \\Omega \\neq 0$$\n\n"
        "`$keep_inline_code$`\n"
        "```python\nprice = '$not_math$'\n```"
    )
    masked, placeholders = latex_renderer.extract_latex_placeholders(src)
    assert len(placeholders) == 3
    assert "$100" in masked and "$50" in masked
    assert "`$keep_inline_code$`" in masked
    assert "price = '$not_math$'" in masked

    app = QApplication.instance() or QApplication([])
    assert app is not None
    ui.apply_dark_theme_and_fonts(app)
    browser = QTextBrowser()
    ui.configure_markdown_browser(browser)
    ui.set_browser_markdown_sticky(browser, src)

    plain = browser.toPlainText()
    rendered_html = browser.toHtml()
    assert "π" in plain and "ω" in plain
    assert "√" in plain and "≤" in plain and "∞" in plain
    assert "⇒" in plain and "Δ" in plain and "Ω" in plain and "≠" in plain
    assert "$100" in plain and "$not_math$" in plain
    assert "Cambria Math" in rendered_html or "Segoe UI Symbol" in rendered_html


def test_cross_provider_llm_failover_and_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = config.AppConfig(
        llm_provider="cerebras",
        cerebras_api_key="csk-primary-test-key",
        cerebras_model="qwen-3.8-27b",
        groq_api_key="gsk-backup-test-key",
        groq_model="qwen/qwen3.8-27b",
        llm_fallback_provider="auto",
    )
    client = llm_client.LLMClient(cfg)

    def _fake_stream(self, *, cfg, **kwargs):
        if cfg.llm_provider == "cerebras":
            raise TimeoutError("Cerebras timed out")
        yield "Відповідь "
        yield "через Groq"

    monkeypatch.setattr(llm_client.LLMClient, "_stream_with_active_config", _fake_stream)

    statuses: list[str] = []
    used: list[tuple[str, str]] = []
    chunks = list(
        client.stream_answer(
            transcript_text="Що таке амплітуда?",
            on_status=statuses.append,
            on_model_used=lambda m, tag: used.append((m, tag)),
            bypass_cache=True,
        )
    )
    assert "".join(chunks) == "Відповідь через Groq"
    assert any("авто-фолбек на groq" in s for s in statuses)
    assert used == [("groq / qwen/qwen3.8-27b", "fallback")]

    # Fallback disabled -> primary error propagates cleanly
    no_fb_cfg = config.AppConfig(
        llm_provider="cerebras",
        cerebras_api_key="csk-primary-test-key",
        groq_api_key="gsk-backup-test-key",
        llm_fallback_provider="none",
    )
    client_no_fb = llm_client.LLMClient(no_fb_cfg)
    with pytest.raises(RuntimeError):
        list(client_no_fb.stream_answer(transcript_text="Тест", bypass_cache=True))

    # Partial streamed output must never silently switch providers mid-answer
    def _partial_fail(self, *, cfg, **kwargs):
        if cfg.llm_provider == "cerebras":
            yield "Початок..."
            raise ConnectionError("network drop")
        yield "Не повинно виконатися"

    monkeypatch.setattr(llm_client.LLMClient, "_stream_with_active_config", _partial_fail)
    out: list[str] = []
    with pytest.raises(RuntimeError):
        for piece in client.stream_answer(transcript_text="Тест", bypass_cache=True):
            out.append(piece)
    assert out == ["Початок..."]


def test_profile_export_and_import_roundtrip(tmp_path: Path) -> None:
    store_path = tmp_path / "prompt_profiles.json"
    cfg = config.AppConfig(
        openai_api_key="sk-secret-do-not-export",
        custom_vocabulary="Dapper, Npgsql",
    )
    store = config.ProfileStore.load_or_init(
        store_path,
        cfg,
        legacy_context_paths=[r"C:\SecretLocalPath\secret_code.cs"],
    )
    export_file = tmp_path / "shared_profiles.json"
    count = store.export_to_file(export_file)
    assert count == 2

    raw_text = export_file.read_text(encoding="utf-8")
    assert "sk-secret-do-not-export" not in raw_text
    assert r"C:\SecretLocalPath" not in raw_text
    parsed = json.loads(raw_text)
    assert parsed["format"] == "zoom_ai_copilot_profiles"
    assert parsed["profiles"][0]["doc_names"] == ["secret_code.cs"]

    # Import back into same store -> deduplicated names, empty local context_paths
    imported = store.import_from_file(export_file)
    assert len(imported) == 2
    assert imported[0].name == "Захист C# (імпорт)"
    assert imported[0].context_paths == []
    assert store.active_profile_id == imported[0].id

    # Invalid file raises ValueError
    bad_file = tmp_path / "bad.json"
    bad_file.write_text('{"foo": 1}', encoding="utf-8")
    with pytest.raises(ValueError):
        store.import_from_file(bad_file)


def test_diagnostics_report_and_software_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_json = tmp_path / "diag.json"
    monkeypatch.setattr(diagnostics, "_diagnostics_data_dir", lambda: tmp_path)
    report = diagnostics.run_diagnostics(write_report=True, show_dialog=False, out_path=str(out_json))
    assert (tmp_path / "diagnostics_report.txt").is_file()
    assert out_json.is_file()
    assert "cpu_avx2" in report
    assert "vcredist_x64" in report
    assert "modules" in report
    assert "summary" in report

    # Software rendering helper defaults to True, supports --hardware opt-out,
    # and forces software mode back on when --software or crash sentinel is present.
    sentinel = tmp_path / ".startup_in_progress"
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("ZOOM_COPILOT_SOFTWARE_GL", raising=False)
    monkeypatch.delenv("ZOOM_COPILOT_HARDWARE_GL", raising=False)
    for var in ("QT_OPENGL", "QSG_RHI_BACKEND", "QT_QUICK_BACKEND"):
        monkeypatch.delenv(var, raising=False)

    assert app_main._apply_software_rendering_env_if_needed(argv=[], sentinel_path=sentinel) is True
    assert os.environ.get("QT_OPENGL") == "software"
    for var in ("QT_OPENGL", "QSG_RHI_BACKEND", "QT_QUICK_BACKEND"):
        monkeypatch.delenv(var, raising=False)

    assert app_main._apply_software_rendering_env_if_needed(argv=["--hardware"], sentinel_path=sentinel) is False
    assert os.environ.get("QT_OPENGL") is None

    monkeypatch.setenv("ZOOM_COPILOT_HARDWARE_GL", "1")
    assert app_main._apply_software_rendering_env_if_needed(argv=[], sentinel_path=sentinel) is False
    monkeypatch.delenv("ZOOM_COPILOT_HARDWARE_GL", raising=False)

    sentinel.write_text("1234", encoding="utf-8")
    assert app_main._apply_software_rendering_env_if_needed(argv=["--hardware"], sentinel_path=sentinel) is True
    assert os.environ.get("QT_OPENGL") == "software"


def test_program_files_routes_data_to_appdata_and_dark_theme_fonts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Forced Fusion dark theme + bundled Inter fonts (real assets/fonts before mocking get_app_dir)
    app = QApplication.instance() or QApplication([])
    loaded_family = ui.apply_dark_theme_and_fonts(app)
    assert "inter" in loaded_family.lower()
    assert app.property("copilot_theme_style") == "Fusion"
    pal = app.palette()
    assert pal.color(pal.ColorRole.Window).name().lower() == "#0e1117"
    assert pal.color(pal.ColorRole.WindowText).lightness() > pal.color(pal.ColorRole.Window).lightness()
    dash_qss = dashboard._build_dashboard_qss(loaded_family)
    assert "QCheckBox::indicator" in dash_qss
    assert "QLabel {" in dash_qss

    fake_pf = tmp_path / "Program Files"
    fake_app_dir = fake_pf / "ZoomAICopilot"
    fake_app_dir.mkdir(parents=True, exist_ok=True)
    fake_appdata = tmp_path / "AppDataRoaming"

    monkeypatch.setenv("ProgramFiles", str(fake_pf))
    monkeypatch.setenv("ProgramW6432", str(fake_pf))
    monkeypatch.setenv("APPDATA", str(fake_appdata))
    monkeypatch.setattr(config, "get_app_dir", lambda: fake_app_dir)

    assert config.is_under_program_files(fake_app_dir) is True
    assert config.is_under_program_files(tmp_path / "Portable") is False

    # Seed legacy runtime file inside Program Files and verify one-time copy into %APPDATA%
    (fake_app_dir / "prompt_profiles.json").write_text('{"profiles":[]}', encoding="utf-8")
    env_path = config.get_env_file_path()
    data_dir = config.get_runtime_data_dir()
    assert env_path == fake_appdata / "ZoomAICopilot" / ".env"
    assert data_dir == fake_appdata / "ZoomAICopilot"
    assert (data_dir / "prompt_profiles.json").is_file()


def test_snip_hotkey_card_history_and_ui_fit() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None
    ui.apply_dark_theme_and_fonts(app)

    assert app_main.parse_hotkey_spec("ctrl+alt+a") == (frozenset({"ctrl", "alt"}), 0x41)
    bridge = app_main.GlobalHotkeyBridge(
        "ctrl+space",
        "ctrl+shift+h",
        screen_spec="ctrl+alt+s",
        web_spec="ctrl+alt+w",
        capture_spec="ctrl+alt+r",
        snip_spec="ctrl+alt+a",
    )
    snip_calls: list[int] = []
    bridge.snip_hotkey_pressed.connect(lambda: snip_calls.append(1))
    uk_a_key = SimpleNamespace(vk=0x41, char="ф")
    bridge._on_press(keyboard.Key.ctrl_l)
    bridge._on_press(keyboard.Key.alt_l)
    bridge._on_press(uk_a_key)
    assert len(snip_calls) == 1

    # Pixmap region encoder & minimum size bounds
    pm = QPixmap(120, 80)
    pm.fill(QColor("#1E293B"))
    encoded = screen_capture.encode_pixmap_to_result(pm, label="Область 120×80")
    assert encoded is not None and encoded.image_bytes.startswith(b"\xff\xd8")
    assert screen_capture.capture_screen_region(QRect(0, 0, 4, 4)) is None

    ov = ui.CopilotOverlayWindow()
    dash = dashboard.DashboardWindow(
        config=config.AppConfig(web_search_enabled=True, screen_vision_enabled=True)
    )
    try:
        dash.show()
        app.processEvents()
        assert ov._title_bar.minimumSizeHint().width() <= 430
        dash_min = dash.minimumSizeHint()
        assert dash_min.width() <= 840 and dash_min.height() <= 560

        ov.set_qa_history_position(2, 3)
        ov.set_qa_history_items([("10:00", "Питання 1"), ("10:01", "Питання 2"), ("10:02", "Питання 3")], 1)
        assert ov._qa_counter_lbl.text() == "2/3 ▾"
        jumps: list[int] = []
        ov.qa_jump_requested.connect(jumps.append)
        ov.qa_jump_requested.emit(0)
        assert jumps == [0]

        dash._fallback_provider_combo.setCurrentIndex(
            dash._fallback_provider_combo.findData("groq")
        )
        dash._hk_snip_edit.setText("ctrl+alt+k")
        collected = dash._collect_config_from_ui()
        assert collected.llm_fallback_provider == "groq"
        assert collected.hotkey_snip == "ctrl+alt+k"
    finally:
        ov.close()
        dash.force_close()


def test_snip_arrow_cursor_and_honest_stealth_indicator(monkeypatch: pytest.MonkeyPatch) -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    # Snip overlay uses normal arrow cursor instead of telltale crosshair
    snip = screen_capture.ScreenSnipOverlay()
    try:
        assert snip.cursor().shape() == Qt.CursorShape.ArrowCursor
    finally:
        snip.close()

    # Honest Stealth indicator reflects real OS read-back, never faking green on failure
    monkeypatch.setattr(ui, "apply_window_display_affinity", lambda _hwnd, _on: False)
    monkeypatch.setattr(dashboard, "apply_window_display_affinity", lambda _hwnd, _on: False)

    ov = ui.CopilotOverlayWindow(stealth_enabled=True)
    dash = dashboard.DashboardWindow(config=config.AppConfig(overlay_stealth=True, dashboard_stealth=True))
    events: list[tuple[bool, bool]] = []
    ov.stealth_status_changed.connect(lambda en, ok: events.append((en, ok)))
    ov.stealth_status_changed.connect(dash.set_overlay_stealth)
    try:
        assert ov.set_stealth_mode(True) is False
        assert ov.stealth_applied is False
        assert ov._stealth_btn.property("stealth") == "off"
        assert "Збій" in ov._stealth_btn.text()
        assert events[-1] == (True, False)
        assert dash._dot_stealth.property("status") == "error"
        assert "ПОМИЛКА" in dash._lbl_stealth.text()

        # When Windows confirms affinity, both overlay and dashboard switch to verified green
        monkeypatch.setattr(ui, "apply_window_display_affinity", lambda _hwnd, _on: True)
        monkeypatch.setattr(dashboard, "apply_window_display_affinity", lambda _hwnd, _on: True)
        assert ov.set_stealth_mode(True) is True
        assert ov.stealth_applied is True
        assert ov._stealth_btn.property("stealth") == "on"
        assert dash._dot_stealth.property("status") == "ok"
        assert "Підтверджено Windows" in dash._lbl_stealth.text()

        # If dashboard affinity fails while dashboard is visible, indicator flags error
        dash.show()
        app.processEvents()
        monkeypatch.setattr(dashboard, "apply_window_display_affinity", lambda _hwnd, _on: False)
        dash.set_dashboard_stealth(True)
        assert dash._dot_stealth.property("status") == "error"
        assert "панель керування НЕ захищена" in dash._lbl_stealth.text()
    finally:
        ov.close()
        dash.force_close()
