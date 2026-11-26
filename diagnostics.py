"""Standalone Windows environment diagnostics (--diagnose & UI check)."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

from config import get_app_dir, get_env_file_path, is_dir_writable
from security import redact_secrets

_PF_AVX2_INSTRUCTIONS_AVAILABLE = 40


def check_cpu_avx2() -> bool | None:
    """Return True/False for CPU AVX2 support on Windows, or None on other platforms."""
    if sys.platform != "win32":
        return None
    try:
        fn = ctypes.windll.kernel32.IsProcessorFeaturePresent
        fn.argtypes = [ctypes.c_uint32]
        fn.restype = ctypes.c_bool
        return bool(fn(_PF_AVX2_INSTRUCTIONS_AVAILABLE))
    except Exception:
        return None


def check_vcredist_x64() -> dict[str, Any]:
    """Check whether Microsoft Visual C++ 2015-2022 Redistributable (x64) is installed."""
    if sys.platform != "win32":
        return {"installed": True, "version": "n/a", "major": 0}
    try:
        import winreg

        subkey = r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, subkey) as key:
            installed, _ = winreg.QueryValueEx(key, "Installed")
            major, _ = winreg.QueryValueEx(key, "Major")
            version, _ = winreg.QueryValueEx(key, "Version")
            return {
                "installed": bool(int(installed) == 1 and int(major) >= 14),
                "version": str(version),
                "major": int(major),
            }
    except Exception:
        return {"installed": False, "version": "", "major": 0}


def _diagnostics_data_dir() -> Path:
    app_dir = get_app_dir()
    if is_dir_writable(app_dir):
        return app_dir
    target = Path(os.getenv("APPDATA", str(Path.home()))) / "ZoomAICopilot"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _probe_module(mod_name: str, *, import_check: bool = True) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(mod_name)
    except Exception as exc:
        return {"present": False, "imported": False, "error": redact_secrets(str(exc))}

    if spec is None:
        return {"present": False, "imported": False, "error": "не встановлено (відсутній у збірці)"}

    if not import_check:
        return {"present": True, "imported": None, "error": ""}

    try:
        importlib.import_module(mod_name)
        return {"present": True, "imported": True, "error": ""}
    except Exception as exc:
        return {"present": True, "imported": False, "error": redact_secrets(str(exc))}


def format_diagnostics_text(report: dict[str, Any]) -> str:
    """Render diagnostics dictionary into a clear Ukrainian text summary."""
    lines = [
        "=== AI Copilot — Діагностика системи ===",
        f"ОС: {report.get('os', '?')} ({report.get('arch', '?')})",
        f"Python: {report.get('python_version', '?')} | Frozen: {report.get('frozen', False)}",
        f"CPU AVX2: {report.get('cpu_avx2')}",
        (
            f"VC++ Redistributable x64: "
            f"{'OK (' + report['vcredist_x64'].get('version', '') + ')' if report.get('vcredist_x64', {}).get('installed') else 'ВІДСУТНІЙ'}"
        ),
        f"Папка даних: {report.get('data_dir', '?')} (запис: {report.get('data_dir_writable', False)})",
        f"Файл .env існує: {report.get('env_exists', False)}",
        f"WASAPI Loopback пристроїв: {report.get('wasapi_devices', '?')}",
        "",
        "Модулі:",
    ]
    modules: dict[str, dict[str, Any]] = report.get("modules", {})
    for name, info in modules.items():
        if info.get("imported") is True:
            state = "OK"
        elif info.get("present") and info.get("imported") is None:
            state = "наявний"
        elif info.get("present"):
            state = f"ПОМИЛКА: {info.get('error', '?')}"
        else:
            state = "відсутній"
        lines.append(f"  - {name}: {state}")

    issues = report.get("issues", [])
    lines.append("")
    if issues:
        lines.append("Виявлені проблеми та рекомендації:")
        for item in issues:
            lines.append(f"  ! {item}")
    else:
        lines.append("Вердикт: критичних проблем запуску не виявлено.")
    if report.get("report_path"):
        lines.append(f"\nЗвіт збережено: {report['report_path']}")
    return "\n".join(lines)


def run_diagnostics(
    *,
    write_report: bool = True,
    show_dialog: bool = False,
    out_path: str | None = None,
) -> dict[str, Any]:
    """Run startup/system compatibility diagnostics without leaking any user secrets."""
    data_dir = _diagnostics_data_dir()
    env_path = get_env_file_path()
    avx2 = check_cpu_avx2()
    vcredist = check_vcredist_x64()

    modules: dict[str, dict[str, Any]] = {}
    for mod in (
        "PyQt6.QtWidgets",
        "pynput",
        "pyaudiowpatch",
        "numpy",
        "scipy",
        "openai",
        "google.genai",
        "elevenlabs",
        "ddgs",
        "latex_renderer",
    ):
        modules[mod] = _probe_module(mod, import_check=True)

    for opt_mod in ("faster_whisper", "ctranslate2", "onnxruntime"):
        # Avoid eagerly loading heavy CUDA/AVX2 DLLs if only inspecting presence, unless safe
        modules[opt_mod] = _probe_module(opt_mod, import_check=False)

    wasapi_count: int | None = None
    wasapi_error = ""
    try:
        from audio_capture import list_loopback_devices

        wasapi_count = len(list_loopback_devices())
    except Exception as exc:
        wasapi_error = redact_secrets(str(exc))

    issues: list[str] = []
    if sys.platform == "win32" and not vcredist.get("installed"):
        issues.append(
            "Не встановлено Microsoft Visual C++ Redistributable 2015–2022 (x64). "
            "Завантажте та встановіть з https://aka.ms/vs/17/release/vc_redist.x64.exe"
        )
    if avx2 is False:
        issues.append(
            "Процесор не підтримує інструкції AVX2 — використовуйте Lite-версію та хмарний STT "
            "(ElevenLabs / Deepgram / Groq)."
        )
    for mod in ("PyQt6.QtWidgets", "pynput", "pyaudiowpatch", "openai"):
        if modules.get(mod, {}).get("imported") is not True:
            issues.append(f"Модуль {mod} не завантажився: {modules[mod].get('error')}")
    if wasapi_error:
        issues.append(f"Помилка перевірки аудіопристроїв WASAPI: {wasapi_error}")
    elif wasapi_count == 0:
        issues.append("Не знайдено жодного пристрою відтворення WASAPI Loopback (підключіть динаміки/навушники).")

    report: dict[str, Any] = {
        "os": platform.platform(),
        "arch": platform.machine(),
        "python_version": platform.python_version(),
        "frozen": bool(getattr(sys, "frozen", False)),
        "executable": sys.executable,
        "data_dir": str(data_dir),
        "data_dir_writable": is_dir_writable(data_dir),
        "env_exists": env_path.is_file(),
        "cpu_avx2": avx2,
        "vcredist_x64": vcredist,
        "wasapi_devices": wasapi_count,
        "wasapi_error": wasapi_error,
        "modules": modules,
        "issues": issues,
        "summary": (
            f"Виявлено проблем: {len(issues)}" if issues else "Усі системні перевірки пройдено ✓"
        ),
        "report_path": "",
    }

    report_text = format_diagnostics_text(report)
    if write_report:
        try:
            report_file = data_dir / "diagnostics_report.txt"
            report_file.write_text(report_text + "\n", encoding="utf-8")
            report["report_path"] = str(report_file)
            report_text = format_diagnostics_text(report)
        except OSError:
            pass

    if out_path:
        try:
            out_file = Path(out_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    if show_dialog and sys.platform == "win32" and not os.getenv("PYTEST_CURRENT_TEST"):
        try:
            flags = 0x30 if issues else 0x40  # MB_ICONWARNING / MB_ICONINFORMATION
            ctypes.windll.user32.MessageBoxW(None, report_text, "AI Copilot — Діагностика", flags)
        except Exception:
            pass

    return report
