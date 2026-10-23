"""Robot selection and repository paths, without importing simulation libraries."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ROBOTS = ("g1", "a3", "x2_ultra")


def robot_name(name: str | None = None) -> str:
    name = name or os.environ.get("SM_ROBOT", "g1")
    if name not in ROBOTS:
        raise ValueError(f"Unknown robot {name!r}; choose one of {', '.join(ROBOTS)}")
    return name


def assets_dir(robot: str | None = None) -> Path:
    return ROOT / "assets/robots" / robot_name(robot)


def profile_path(robot: str | None = None) -> Path:
    return Path(os.environ.get("SM_ROBOT_SPEC", assets_dir(robot) / "profile.json"))


def lafan_npz_dir(robot: str | None = None) -> Path:
    """Default data path, relative to the repository working directory."""
    return Path("data/npz") / robot_name(robot) / "lafan1"


def select_robot(robot: str, *, check_assets: bool = False) -> None:
    """Choose the embodiment before modules capture their joint dimensions."""
    robot = robot_name(robot)
    dimensions = sys.modules.get("supermimic.robots.dimensions") or sys.modules.get(
        "supermimic.robot_dimensions"
    )
    if dimensions is not None and dimensions.ROBOT != robot:
        raise RuntimeError("Select the robot before importing training or simulation modules")
    previous = os.environ.get("SM_ROBOT", "g1")
    os.environ["SM_ROBOT"] = robot
    if robot == "g1":
        os.environ.pop("SM_ROBOT_SPEC", None)
        required = assets_dir(robot) / "robot.xml"
    else:
        if previous != robot or "SM_ROBOT_SPEC" not in os.environ:
            os.environ["SM_ROBOT_SPEC"] = str(assets_dir(robot) / "profile.json")
        required = profile_path(robot)
    if check_assets and not required.is_file():
        raise FileNotFoundError(
            f"Robot asset missing: {required}. Run "
            f".venv/bin/python scripts/setup/build_robot_assets.py --robot {robot}"
        )
