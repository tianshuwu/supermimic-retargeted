"""Select an embodiment before importing Python training/evaluation modules.

Example: uv run python scripts/robot.py a3 -m supermimic.scripts.train --help
"""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if len(sys.argv)<3 or sys.argv[1] not in ['g1','x2_ultra','a3']:
 raise SystemExit('Usage: python scripts/robot.py {g1|x2_ultra|a3} SCRIPT_OR_-m_MODULE [ARGS...]')
robot=sys.argv[1];os.environ['SM_ROBOT']=robot
if robot!='g1':
 profile=ROOT/'assets/robots'/robot/'profile.json'
 if not profile.exists():raise SystemExit('Build robot assets first: uv run python scripts/build_robot_assets.py')
 os.environ['SM_ROBOT_SPEC']=str(profile)
os.execv(sys.executable,[sys.executable,*sys.argv[2:]])
