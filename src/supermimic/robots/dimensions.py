"""Robot dimensions selected before importing training modules.

G1 remains the upstream default. Other embodiments use a versioned model profile.
"""

import json
from pathlib import Path

from supermimic.robots.paths import assets_dir, profile_path, robot_name

ROBOT = robot_name()
PROFILE = {}
if ROBOT != "g1":
    _profile_path = profile_path(ROBOT)
    PROFILE = json.loads(_profile_path.read_text())
    if PROFILE["robot"] != ROBOT:
        raise ValueError(f"Profile {_profile_path} does not describe {ROBOT}")
    if ROBOT == "x2_ultra" and not {"left_wrist_roll_link", "right_wrist_roll_link"}.issubset(
        PROFILE.get("tracked_bodies", [])
    ):
        raise ValueError(
            f"Outdated X2 hand tracking in {_profile_path}: track both wrist_roll_link palms. "
            "Back up custom assets before rebuilding with "
            "scripts/setup/build_robot_assets.py --robot x2_ultra"
        )
    _xml = Path(PROFILE["xml_path"])
    if not _xml.is_absolute():
        PROFILE["xml_path"] = str((_profile_path.parent / _xml).resolve())
NJ = int(PROFILE.get("num_joints", 29))
NQ, NV = NJ + 7, NJ + 6
STATE_DIM = FEAT_DIM = 2 * NJ + 13
DELTA_DIM = 2 * NJ + 12
XI_DIM = NJ + 4
OBS_DIM = 3 * NJ + 6
GOAL_FEAT_DIM = 2 * NJ + 10
OBS_TERMS = (
    ("joint_pos", NJ),
    ("joint_vel", NJ),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("actions", NJ),
)


def robot_xml_path() -> Path:
    """Resolve the active robot's local model; never fall back to package assets."""
    if ROBOT == "g1":
        path = assets_dir("g1") / "robot.xml"
    else:
        path = Path(PROFILE["xml_path"])
    if not path.is_file():
        raise FileNotFoundError(
            f"Robot model not found: {path}. From the repository root, run "
            f".venv/bin/python scripts/setup/build_robot_assets.py --robot {ROBOT}"
        )
    return path
