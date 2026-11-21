"""Build standalone Windows distribution (dist/ZoomAICopilot/ZoomAICopilot.exe) + shortcut."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def _ps_quote(value: str) -> str:
    """Escape single quotes for PowerShell single-quoted string literals."""
    return value.replace("'", "''")


def _create_windows_shortcut(exe_path: Path, link_path: Path) -> None:
    if sys.platform != "win32":
        return

    ps_link = _ps_quote(str(link_path))
    ps_exe = _ps_quote(str(exe_path))
    ps_dir = _ps_quote(str(exe_path.parent))
    ps_script = (
        "$ws = New-Object -ComObject WScript.Shell; "
        f"$sc = $ws.CreateShortcut('{ps_link}'); "
        f"$sc.TargetPath = '{ps_exe}'; "
        f"$sc.WorkingDirectory = '{ps_dir}'; "
        f"$sc.IconLocation = '{ps_exe},0'; "
        "$sc.Description = 'AI Copilot для Zoom (Українська мова)'; "
        "$sc.Save()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        check=True,
    )


def _needs_icon_rebuild(root: Path, icon_png: Path, icon_ico: Path) -> bool:
    if not icon_png.is_file() or not icon_ico.is_file():
        return True
    gen_py = root / "icon_gen.py"
    if gen_py.is_file():
        gen_mtime = gen_py.stat().st_mtime
        if icon_png.stat().st_mtime < gen_mtime or icon_ico.stat().st_mtime < gen_mtime:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build ZoomAICopilot Windows executable.")
    parser.add_argument(
        "--copy-env",
        action="store_true",
        help="Copy workspace .env into dist/ZoomAICopilot/.env on a clean build.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run ZoomAICopilot.exe --smoke-test after building.",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent
    assets_dir = root / "assets"
    icon_png = assets_dir / "icon.png"
    icon_ico = assets_dir / "icon.ico"

    if _needs_icon_rebuild(root, icon_png, icon_ico):
        import icon_gen

        icon_gen.generate_icons(assets_dir)

    dist_dir = root / "dist" / "ZoomAICopilot"
    dist_env = dist_dir / ".env"
    preserved_env_bytes: bytes | None = None
    if dist_env.is_file():
        preserved_env_bytes = dist_env.read_bytes()

    spec_path = root / "ZoomAICopilot.spec"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        str(spec_path),
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=str(root), check=True)
    except subprocess.CalledProcessError as exc:
        print(
            "\nПомилка збірки PyInstaller. Якщо зараз запущено ZoomAICopilot.exe, "
            "закрийте його (через іконку в треї -> Вихід) і повторіть збірку.",
            file=sys.stderr,
        )
        return exc.returncode or 1

    exe_path = dist_dir / "ZoomAICopilot.exe"
    if not exe_path.is_file():
        raise FileNotFoundError(f"Expected built executable not found: {exe_path}")

    env_example = root / ".env.example"
    if env_example.is_file():
        shutil.copy2(env_example, dist_dir / ".env.example")

    if preserved_env_bytes is not None:
        dist_env.write_bytes(preserved_env_bytes)
        print("Preserved existing dist/ZoomAICopilot/.env")
    elif args.copy_env:
        local_env = root / ".env"
        if local_env.is_file() and not dist_env.is_file():
            shutil.copy2(local_env, dist_env)
            print("Copied workspace .env into dist/ZoomAICopilot/.env (--copy-env)")

    shortcut_path = root / "ZoomAICopilot.lnk"
    try:
        _create_windows_shortcut(exe_path, shortcut_path)
        print(f"Created shortcut: {shortcut_path}")
    except Exception as exc:
        print(f"Warning: could not create shortcut: {exc}")

    print(f"Build complete: {exe_path}")

    if args.verify:
        print("Verifying built executable via --smoke-test...")
        res = subprocess.run([str(exe_path), "--smoke-test"], cwd=str(dist_dir))
        if res.returncode != 0:
            print(f"Smoke test failed with exit code {res.returncode}", file=sys.stderr)
            return res.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
