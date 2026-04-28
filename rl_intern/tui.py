import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    bun = shutil.which("bun")
    if not bun:
        print(
            "rl-intern-tui requires Bun because OpenTUI is currently Bun-first.\n"
            "Install Bun from https://bun.sh, then run `bun install` in ./tui.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    project_root = Path(__file__).resolve().parents[1]
    tui_dir = project_root / "tui"
    if not (tui_dir / "node_modules").exists():
        print(
            "TUI dependencies are not installed. Run `cd tui && bun install` first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    env = os.environ.copy()
    env["RL_INTERN_PROJECT_ROOT"] = str(project_root)
    raise SystemExit(
        subprocess.call([bun, "run", "start", "--", *sys.argv[1:]], cwd=tui_dir, env=env)
    )
