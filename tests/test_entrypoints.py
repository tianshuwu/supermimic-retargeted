"""Entry points select an embodiment before importing dimensional training code."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("robot,joints", [("g1", 29), ("a3", 31), ("x2_ultra", 31)])
def test_direct_train_cli_selects_robot(robot, joints):
    if robot != "g1" and not (ROOT / f"assets/robots/{robot}/profile.json").exists():
        pytest.skip("Build the robot assets first")
    env = os.environ.copy()
    env.pop("SM_ROBOT", None)
    env.pop("SM_ROBOT_SPEC", None)
    result = subprocess.run(
        [sys.executable, "scripts/training/train.py", "--robot", robot, "--print-config"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    cfg = json.loads(result.stdout)
    assert cfg["robot"] == robot
    assert cfg["data"]["npz_dir"] == f"data/npz/{robot}/lafan1"
    assert cfg["policy"]["goal_dim"] == 10 * (2 * joints + 10)
    assert cfg["env"]["num_envs"] == 4096


def test_smoke_preset_and_legacy_cli_accept_overrides():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "supermimic.scripts.train",
            "--robot",
            "g1",
            "--preset",
            "smoke",
            "--policy-trainer.batch-size",
            "64",
            "--print-config",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    cfg = json.loads(result.stdout)
    assert cfg["schedule"]["total_iterations"] == 100
    assert cfg["env"]["num_envs"] == 128
    assert cfg["world_model"]["hidden"] == [2048, 2048, 2048]
    assert cfg["wm_trainer"]["compile"] is None
    assert cfg["policy_trainer"]["batch_size"] == 64


def test_relative_profile_xml_uses_profile_directory(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"robot": "a3", "num_joints": 31, "xml_path": "robot.xml"}))
    env = os.environ.copy()
    env.update(SM_ROBOT="a3", SM_ROBOT_SPEC=str(profile))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from supermimic.robots.dimensions import PROFILE; print(PROFILE['xml_path'])",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert Path(result.stdout.strip()) == tmp_path / "robot.xml"


@pytest.mark.parametrize(
    "filename",
    [
        "setup/build_robot_assets.py",
        "data/prepare_lafan.py",
        "setup/fetch_upstreams.py",
        "data/retarget.py",
        "data/prepare_lafan_bvh.py",
        "evaluation/render_policy.py",
        "robot.py",
        "data/align_lafan_bone_frames.py",
        "data/select_bones.py",
        "data/extract_bones_subset.py",
        "data/finalize_manifest.py",
        "training/train.py",
    ],
)
def test_operational_scripts_do_not_execute_on_import(filename, monkeypatch):
    monkeypatch.setattr(sys, "argv", [filename, "--invalid-argument"])
    spec = importlib.util.spec_from_file_location("script_under_test", ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
