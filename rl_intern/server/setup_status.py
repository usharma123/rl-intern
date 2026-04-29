from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODAL_APPS = ("rl-intern-generic", "rl-intern-sb3")


def setup_status() -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    env_path = PROJECT_ROOT / ".env"
    modal_installed = _python_imports("modal")
    modal_authenticated = _modal_authenticated() if modal_installed else False
    deployed_apps = _modal_deployed_apps() if modal_authenticated else []

    return {
        "envFile": env_path.exists(),
        "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
        "huggingface": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")),
        "uv": shutil.which("uv") is not None,
        "bun": shutil.which("bun") is not None,
        "modalInstalled": modal_installed,
        "modalAuthenticated": modal_authenticated,
        "modalAppsDeployed": all(app in deployed_apps for app in MODAL_APPS),
        "modalApps": {app: app in deployed_apps for app in MODAL_APPS},
    }


def _python_imports(module: str) -> bool:
    proc = _run([sys.executable, "-c", f"import {module}"], timeout=5)
    return proc is not None and proc.returncode == 0


def _modal_authenticated() -> bool:
    modal = shutil.which("modal")
    if modal is None:
        return False
    proc = _run([modal, "token", "list"], timeout=3)
    return proc is not None and proc.returncode == 0


def _modal_deployed_apps() -> list[str]:
    modal = shutil.which("modal")
    if modal is None:
        return []
    proc = _run([modal, "app", "list"], timeout=3)
    if proc is None or proc.returncode != 0:
        return []
    output = f"{proc.stdout}\n{proc.stderr}"
    return [app for app in MODAL_APPS if app in output]


def _run(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
