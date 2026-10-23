"""Robot dimensions selected before importing training modules.

G1 remains the upstream default. Other embodiments use a versioned model profile.
"""
import json
import os
from pathlib import Path
ROBOT = os.environ.get("SM_ROBOT", "g1")
PROFILE = {}
if ROBOT != "g1":
    profile_path = Path(os.environ.get("SM_ROBOT_SPEC", Path(__file__).resolve().parents[2] / "assets" / "robots" / ROBOT / "profile.json"))
    PROFILE = json.loads(profile_path.read_text())
    assert PROFILE["robot"] == ROBOT
NJ = int(PROFILE.get("num_joints", 29))
NQ, NV = NJ + 7, NJ + 6
STATE_DIM = FEAT_DIM = 2 * NJ + 13
DELTA_DIM = 2 * NJ + 12
XI_DIM = NJ + 4
OBS_DIM = 3 * NJ + 6
GOAL_FEAT_DIM = 2 * NJ + 10
OBS_TERMS = (("joint_pos", NJ), ("joint_vel", NJ), ("base_ang_vel", 3),
             ("projected_gravity", 3), ("actions", NJ))
