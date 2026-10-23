"""Resolve retargeting paths and invoke the isolated SOMA environment."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve_config(robot, input_dir=None, output_dir=None, batch_size=None):
    """Read the repository preset; explicit paths are relative to the caller's cwd."""
    config = json.loads((ROOT / f"configs/retarget/{robot}.json").read_text())
    for key, override in [("import_folder", input_dir), ("export_folder", output_dir)]:
        path = Path(override) if override is not None else ROOT / config[key]
        config[key] = str(path.resolve())
    if batch_size is not None:
        config["batch_size"] = batch_size
    if config["batch_size"] < 1:
        raise ValueError("batch-size must be positive")
    config["extra_robot_paths"] = [
        str((ROOT / path).resolve()) for path in config.get("extra_robot_paths", [])
    ]
    return config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", choices=["g1", "a3", "x2_ultra"], required=True)
    p.add_argument("--input", type=Path, help="Override import_folder in the robot preset")
    p.add_argument("--output", type=Path, help="Override export_folder in the robot preset")
    p.add_argument("--batch-size", type=int, help="Override batch_size in the robot preset")
    p.add_argument("--print-config", action="store_true", help="Print resolved JSON and exit")
    a = p.parse_args()
    try:
        config = resolve_config(a.robot, a.input, a.output, a.batch_size)
    except ValueError as error:
        p.error(str(error))
    if a.print_config:
        print(json.dumps(config, indent=2))
        return
    source = Path(config["import_folder"])
    if not source.is_dir() or not next(source.rglob("*.bvh"), None):
        p.error(f"No input BVH files found in {source}; see docs/DATA.md")
    python = ROOT / "vendor/soma-retargeter/.venv/bin/python"
    if not python.exists():
        raise SystemExit("Install the retargeting environment first; see README.")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    with tempfile.TemporaryDirectory(prefix="supermimic-retarget-") as temp:
        path = Path(temp) / "config.json"
        path.write_text(json.dumps(config))
        subprocess.run(
            [
                str(python),
                str(ROOT / "vendor/soma-retargeter/app/bvh_to_csv_converter.py"),
                "--config",
                str(path),
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
