# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for ZoomAICopilot (--onedir with NVIDIA CUDA 12 / cuDNN DLLs)."""

import os
from pathlib import Path
import site
import sys

IS_LITE_BUILD = os.getenv("ZOOM_COPILOT_LITE", "").strip().lower() in {"1", "true", "yes"}

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

ROOT = Path(SPECPATH).resolve()

datas = [
    (str(ROOT / "assets" / "icon.png"), "assets"),
    (str(ROOT / "assets" / "icon.ico"), "assets"),
]
datas += collect_data_files("faster_whisper")
datas += collect_data_files("certifi")
datas += collect_data_files("docx")
datas += collect_data_files("ddgs")

binaries = []
binaries += collect_dynamic_libs("ctranslate2")
binaries += collect_dynamic_libs("onnxruntime")
binaries += collect_dynamic_libs("primp")

# Collect required NVIDIA cuBLAS / cuDNN 9 / CUDA NVRTC DLLs into _internal/nvidia/<pkg>/bin (Full build only)
if not IS_LITE_BUILD:
    NVIDIA_ALLOWED_PKGS = {"cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"}
    seen_sp = set()
    site_roots = []
    for sp in list(site.getsitepackages()) + [str(Path(sys.prefix) / "Lib" / "site-packages")]:
        p = Path(sp)
        key = str(p).lower()
        if key not in seen_sp and p.is_dir():
            seen_sp.add(key)
            site_roots.append(p)

    for sp_root in site_roots:
        nvidia_dir = sp_root / "nvidia"
        if not nvidia_dir.is_dir():
            continue
        for pkg_dir in sorted(nvidia_dir.iterdir()):
            if pkg_dir.name not in NVIDIA_ALLOWED_PKGS:
                continue
            bin_dir = pkg_dir / "bin"
            if not bin_dir.is_dir():
                continue
            dest_rel = f"nvidia/{pkg_dir.name}/bin"
            for dll in bin_dir.glob("*.dll"):
                low_name = dll.name.lower()
                if low_name.endswith(".alt.dll") or low_name.startswith("nvblas64_"):
                    continue
                binaries.append((str(dll), dest_rel))

hiddenimports = (
    [
        "pynput",
        "pynput.keyboard",
        "pynput.keyboard._base",
        "pynput.keyboard._win32",
        "pynput._util",
        "pynput._util.win32",
        "pynput._util.win32_vks",
        "pyaudiowpatch",
        "faster_whisper",
        "ctranslate2",
        "onnxruntime",
        "scipy.signal",
        "scipy.special._ufuncs_cxx",
        "google.genai",
        "openai",
        "elevenlabs",
        "httpx",
        "httpcore",
        "h11",
        "pypdf",
        "docx",
        "dotenv",
        "screen_capture",
        "latex_renderer",
        "diagnostics",
        "web_search",
        "ddgs",
        "primp",
        "lxml.html",
        "lxml.etree",
    ]
    + collect_submodules("ddgs")
    + collect_submodules("primp")
    + collect_submodules("websockets")
    + collect_submodules("elevenlabs")
)

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "pytest",
        "ruff",
        "IPython",
        "jupyter",
        "notebook",
        "PyQt6.QtQml",
        "PyQt6.QtQuick",
        "PyQt6.QtQuickWidgets",
        "PyQt6.QtQuick3D",
        "PyQt6.QtWebEngineCore",
        "PyQt6.QtWebEngineWidgets",
        "PyQt6.QtMultimedia",
        "PyQt6.QtMultimediaWidgets",
        "PyQt6.QtSql",
        "PyQt6.QtBluetooth",
        "PyQt6.QtNfc",
        "PyQt6.QtPositioning",
        "PyQt6.QtSensors",
        "PyQt6.QtSerialPort",
        "PyQt6.QtTest",
        "PyQt6.QtXml",
        "PyQt6.QtDesigner",
        "PyQt6.QtHelp",
        "PyQt6.QtPdf",
        "PyQt6.QtPdfWidgets",
        "PyQt6.QtSvg",
        "PyQt6.QtSvgWidgets",
        "PyQt6.QtOpenGL",
        "PyQt6.QtOpenGLWidgets",
        "PyQt6.Qt3DCore",
        "PyQt6.Qt3DRender",
        "PyQt6.QtCharts",
        "PyQt6.QtDataVisualization",
    ],
    noarchive=False,
    optimize=0,
)

# Prune unneeded PyQt6 locale .qm translations (keep English and Ukrainian only).
a.datas = [
    entry
    for entry in a.datas
    if not (
        "translations" in entry[0].lower().replace("\\", "/")
        and entry[0].lower().endswith(".qm")
        and not any(tag in entry[0].lower() for tag in ("_en.", "_uk.", "_uk_ua."))
    )
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ZoomAICopilot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ZoomAICopilot",
)
