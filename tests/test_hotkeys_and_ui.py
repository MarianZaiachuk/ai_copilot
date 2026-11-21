"""Tests for layout-independent hotkeys, Stealth Overlay, and Dashboard UI."""

from __future__ import annotations

from types import SimpleNamespace

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
from pynput import keyboard

import audio_capture
import config
import dashboard
import llm_client
import main as app_main
import ui


def test_parse_hotkey_spec_and_bridge_debounce() -> None:
    assert app_main.parse_hotkey_spec("ctrl+space") == (frozenset({"ctrl"}), 0x20)
    assert app_main.parse_hotkey_spec("ctrl+shift+h") == (frozenset({"ctrl", "shift"}), 0x48)

    bridge = app_main.GlobalHotkeyBridge("ctrl+space", "ctrl+shift+h")
    ans_calls = []
    tog_calls = []
    bridge.answer_hotkey_pressed.connect(lambda: ans_calls.append(1))
    bridge.toggle_hotkey_pressed.connect(lambda: tog_calls.append(1))

    # Simulated Ukrainian layout KeyCode for H key (vk=0x48, char='р')
    uk_h_key = SimpleNamespace(vk=0x48, char="р")
    bridge._on_press(keyboard.Key.ctrl_l)
    bridge._on_press(keyboard.Key.shift_l)
    bridge._on_press(uk_h_key)
    bridge._on_press(uk_h_key)  # repeat must be debounced
    assert len(tog_calls) == 1

    bridge._on_release(uk_h_key)
    bridge._on_release(keyboard.Key.shift_l)
    bridge._on_press(keyboard.Key.space)
    assert len(ans_calls) == 1


def test_overlay_and_dashboard_ui_integration() -> None:
    app = QApplication.instance() or QApplication([])
    cfg = config.AppConfig(
        llm_provider="gemini",
        gemini_api_key="AIzaTestKey12345678901234567",
        gemini_model="gemini-2.5-flash",
        custom_vocabulary="PyQt6, WASAPI",
    )
    dash = dashboard.DashboardWindow(config=cfg)
    ov = ui.CopilotOverlayWindow()

    devs = [
        audio_capture.AudioDeviceInfo(index=10, name="Speakers (Realtek)", channels=2, sample_rate=48000, is_default=True),
        audio_capture.AudioDeviceInfo(index=12, name="Headphones (USB)", channels=2, sample_rate=48000, is_default=False),
    ]
    dash.set_devices(devs)
    ov.set_devices(devs)

    # Sync device selection from Dashboard -> Overlay
    dash.device_selected.connect(ov.set_selected_device_index)
    dash.home_device_combo.setCurrentIndex(1)
    assert ov.device_combo.currentData() == 12

    # Bi-directional sync for Web Search & Screen Vision toggles between Overlay and Dashboard
    ov.web_search_toggled.connect(dash.set_web_search_enabled)
    dash.web_search_toggled.connect(ov.set_web_search_enabled)
    ov.screen_vision_toggled.connect(dash.set_screen_vision_enabled)
    dash.screen_vision_toggled.connect(ov.set_screen_vision_enabled)

    deep_triggered = []
    ov.deep_answer_requested.connect(lambda: deep_triggered.append(1))

    dash.show()
    ov.show()

    results = {}

    def _run_ui_assertions() -> None:
        results["stealth_applied"] = ov.stealth_applied
        results["compact_default"] = not ov.is_expanded
        results["compact_manual_visible"] = ov.manual_input.isVisible() and ov.answer_btn.isVisible()
        ov.set_expanded(True)
        results["expanded_true"] = ov.is_expanded
        ov.set_expanded(False)
        results["compact_manual_still_visible"] = ov.manual_input.isVisible()
        ov.set_opacity_percent(75)
        results["opacity_75"] = abs(ov.windowOpacity() - 0.75) <= 0.02

        ov._web_search_btn.click()
        results["web_sync_ov_to_dash"] = (
            ov.web_search_enabled
            and dash._live_web_btn.isChecked()
            and dash._web_enabled_chk.isChecked()
        )
        dash._live_web_btn.click()
        results["web_sync_dash_to_ov"] = (
            (not ov.web_search_enabled)
            and (not dash._live_web_btn.isChecked())
            and (not dash._web_enabled_chk.isChecked())
        )
        ov._screen_vision_btn.click()
        results["screen_sync_ov_to_dash"] = (
            ov.screen_vision_enabled
            and dash._live_screen_btn.isChecked()
            and dash._screen_enabled_chk.isChecked()
        )
        dash._live_screen_btn.click()
        results["screen_sync_dash_to_ov"] = (
            (not ov.screen_vision_enabled)
            and (not dash._live_screen_btn.isChecked())
            and (not dash._screen_enabled_chk.isChecked())
        )
        dash._settings_tabs.setCurrentIndex(3)
        dash._web_enabled_chk.setChecked(True)
        dash._screen_enabled_chk.setChecked(True)
        dash._screen_max_width_spin.setValue(1280)
        dash._on_save_settings_clicked()
        saved_cfg = dash._collect_config_from_ui()
        results["web_settings_saved"] = saved_cfg.web_search_enabled and dash._live_web_btn.isChecked()
        results["screen_settings_saved"] = (
            saved_cfg.screen_vision_enabled
            and saved_cfg.screen_capture_max_width == 1280
            and dash._live_screen_btn.isChecked()
        )

        ov.begin_answer_stream()
        ov.append_answer_chunk("- Теза відповіді")
        ov.finish_answer_stream()
        ov._on_copy_answer_clicked()
        clipboard = QApplication.clipboard()
        results["copied_text"] = clipboard is not None and "Теза відповіді" in clipboard.text()

        ov._deep_btn.click()
        results["deep_signal"] = len(deep_triggered) == 1

        results["home_deprecated_hint"] = "застаріла" in dash._lbl_llm.text()
        dash.open_settings(0)
        results["settings_warn_visible"] = dash._model_warn_lbl.isVisible()

        dash.set_api_check_result(
            llm_client.ApiKeyCheckResult(
                ok=True,
                key_valid=True,
                model_valid=True,
                status="ok",
                message="API-ключ працює",
                available_models=["gemini-3.8-flash", "gemini-4-hydrogen"],
                latency_sec=0.3,
            )
        )
        results["home_verified"] = "ключ перевірено" in dash._lbl_llm.text()
        dash.set_whisper_status("Whisper: large-v3-turbo (CUDA float16)")
        results["whisper_progress_hidden"] = not dash._whisper_progress.isVisible()
        results["vocab_loaded"] = dash.vocab_edit.text() == "PyQt6, WASAPI"

        ov.close()
        dash.force_close()
        app.quit()

    QTimer.singleShot(250, _run_ui_assertions)
    app.exec()

    assert all(results.values()), f"UI assertion failure: {results}"


def test_polish_ui_stop_nav_validation_markdown_and_icons() -> None:
    import pytest
    import icon_gen

    # Hotkey multi-trigger and identical-pair rejection
    with pytest.raises(ValueError):
        app_main.parse_hotkey_spec("ctrl+a+b")
    with pytest.raises(ValueError):
        app_main.GlobalHotkeyBridge("ctrl+space", "ctrl+space")

    # Markdown fence balancer closes unfinished code blocks while leaving complete blocks alone
    unclosed = "Текст:\n```python\nx = 1"
    balanced = ui.balance_markdown_fences(unclosed)
    assert balanced.endswith("\n```")
    closed = "```python\nx = 1\n```"
    assert ui.balance_markdown_fences(closed) == closed

    # Icon generation produces crisp QImage & valid PNG signature
    img = icon_gen.render_icon_image(64)
    assert img.width() == 64 and img.height() == 64
    png_bytes = icon_gen._image_to_png_bytes(img)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    app = QApplication.instance() or QApplication([])
    cfg = config.AppConfig(
        llm_provider="groq",
        openai_api_key="gsk_test_ui_polish_1234567890",
        openai_model="qwen/qwen3.8-27b",
    )
    dash = dashboard.DashboardWindow(config=cfg)
    ov = ui.CopilotOverlayWindow()

    # Compact title bar fits <= 430px & Dashboard fits <= 840x560
    assert ov._title_bar.minimumSizeHint().width() <= 430
    dash_min = dash.minimumSizeHint()
    assert dash_min.width() <= 840 and dash_min.height() <= 560

    # Stop toggle emits cancel_answer_requested rather than answer_requested while generating
    cancels: list[int] = []
    answers: list[tuple[str, str]] = []
    ov.cancel_answer_requested.connect(lambda: cancels.append(1))
    ov.answer_requested.connect(lambda m, s: answers.append((m, s)))
    ov.begin_answer_stream()
    assert ov.is_generating and "Стоп" in ov.answer_btn.text()
    ov.answer_btn.click()
    assert cancels == [1] and answers == []
    ov.cancel_answer_stream()
    assert not ov.is_generating

    # Q&A pagination visibility and signals
    nav_deltas: list[int] = []
    ov.qa_nav_requested.connect(nav_deltas.append)
    ov.set_qa_history_position(2, 3)
    assert not ov._qa_nav_widget.isHidden()
    ov._qa_prev_btn.click()
    ov._qa_next_btn.click()
    assert nav_deltas == [-1, 1]

    # Docs standalone vocab/prompt handlers & invalid hotkey validation in settings
    saved_cfgs: list[config.AppConfig] = []
    dash.settings_saved.connect(saved_cfgs.append)
    dash.vocab_edit.setText("C#, Dapper, PostgreSQL")
    dash._on_save_vocab_clicked()
    assert len(saved_cfgs) == 1 and saved_cfgs[-1].custom_vocabulary == "C#, Dapper, PostgreSQL"

    # Identical hotkeys in settings must show error without emitting settings_saved
    dash._hk_answer_edit.setText("ctrl+space")
    dash._hk_toggle_edit.setText("ctrl+space")
    dash._on_save_settings_clicked()
    assert len(saved_cfgs) == 1
    assert "не можуть збігатися" in dash._save_status_lbl.text()

    # test_screen_capture_requested emits (screen_index, max_width)
    screen_reqs: list[tuple[int, int]] = []
    dash.test_screen_capture_requested.connect(lambda i, w: screen_reqs.append((i, w)))
    dash._screen_max_width_spin.setValue(1440)
    dash._on_test_screen_clicked()
    assert screen_reqs == [(-1, 1440)]

    # Regression test: Dashboard.set_documents renders LoadedDocument objects from files & folders without AttributeError
    import context_loader
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "Program.cs").write_text("public class Program { public static void Main() {} }", encoding="utf-8")
        (root / "README.md").write_text("# Опис проєкту\nСистема аналізу звуку.", encoding="utf-8")
        expanded, _ = context_loader.expand_context_paths([str(root)])
        cm = context_loader.ContextManager()
        loaded, errs = cm.add_files(expanded)
        assert not errs and len(loaded) == 2
        dash.set_documents(cm.documents, cm.summary())
        assert dash.docs_list.count() == 2

    ov.close()
    dash.force_close()
    app.processEvents()
