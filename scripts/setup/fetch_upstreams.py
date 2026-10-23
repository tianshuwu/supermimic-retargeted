"""Fetch pinned external robot descriptions and retargeting tools (no model weights)."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--with-reference", action="store_true")
    args = p.parse_args()
    specs = json.loads((ROOT / "configs/upstreams.json").read_text())
    names = ["soma-retargeter", "a3-sonic", "x2-sonic"]
    if args.with_reference:
        names += ["primary", "reference"]
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    for name in names:
        spec = specs[name]
        dest = ROOT / "vendor" / name
        if (dest / ".git").exists():
            remote = subprocess.check_output(
                ["git", "-C", str(dest), "remote", "get-url", "origin"], text=True
            ).strip()
            commit = subprocess.check_output(
                ["git", "-C", str(dest), "rev-parse", "HEAD"], text=True
            ).strip()
            if remote != spec["url"] or commit != spec["commit"]:
                raise SystemExit(
                    f"{dest} differs from pinned source; move it aside before fetching."
                )
        else:
            dest.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", str(dest)], env=env, check=True)
            subprocess.run(
                ["git", "-C", str(dest), "remote", "add", "origin", spec["url"]],
                env=env,
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(dest), "fetch", "--depth=1", "origin", spec["commit"]],
                env=env,
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(dest), "checkout", "--detach", "FETCH_HEAD"], env=env, check=True
            )
        if name == "soma-retargeter":
            shutil.copy2(ROOT / "requirements/soma-uv.lock", dest / "uv.lock")
        print(name, spec["commit"])
    print(
        "Next: uv sync --project vendor/soma-retargeter --frozen; bash scripts/run_local.sh scripts/setup/build_robot_assets.py"
    )


if __name__ == "__main__":
    main()
