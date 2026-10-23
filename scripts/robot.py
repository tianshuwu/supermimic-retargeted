"""Select a robot before launching a Python script or module.

Example: python scripts/robot.py a3 -m supermimic.scripts.train --help
"""

import argparse
import os
import sys

from supermimic.robots.paths import ROBOTS, select_robot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", choices=ROBOTS)
    parser.add_argument("command", nargs=argparse.REMAINDER, metavar="SCRIPT_OR_-m_MODULE")
    args = parser.parse_args()
    if not args.command:
        parser.error("provide a Python script or -m MODULE after the robot name")
    try:
        select_robot(args.robot, check_assets=True)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    os.execv(sys.executable, [sys.executable, *args.command])


if __name__ == "__main__":
    main()
