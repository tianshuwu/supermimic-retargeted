"""Retarget presets and both G1 CSV conventions work without SOMA or datasets."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from supermimic.data.motion_convert import G1_JOINT_NAMES, load_csv

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/data/retarget.py"


@pytest.mark.parametrize(
    "robot,target", [("g1", "unitree_g1"), ("a3", "agibot-a3t3"), ("x2_ultra", "agibot_x2ultra")]
)
def test_retarget_config_can_be_previewed_outside_checkout(robot, target, tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--robot", robot, "--print-config"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    config = json.loads(result.stdout)
    assert config["retarget_target"] == target
    assert config["import_folder"] == str(ROOT / "data/lafan1-soma-aligned")
    assert config["export_folder"] == str(ROOT / f"data/lafan1-retarget-{robot}")


def test_retarget_passes_overrides_to_isolated_environment(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("retarget_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "configs/retarget").mkdir(parents=True)
    (tmp_path / "configs/retarget/g1.json").write_bytes(
        (ROOT / "configs/retarget/g1.json").read_bytes()
    )
    interpreter = tmp_path / "vendor/soma-retargeter/.venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    source = tmp_path / "input"
    source.mkdir()
    (source / "clip.bvh").touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--robot",
            "g1",
            "--input",
            "input",
            "--output",
            "output",
            "--batch-size",
            "2",
        ],
    )
    monkeypatch.setenv("PYTHONPATH", "unrelated-environment")
    monkeypatch.setenv("PYTHONHOME", "unrelated-python")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        config = json.loads(Path(command[-1]).read_text())
        assert command[0] == str(interpreter)
        assert config["retarget_target"] == "unitree_g1"
        assert config["import_folder"] == str(source)
        assert config["export_folder"] == str(tmp_path / "output")
        assert config["batch_size"] == 2
        assert "PYTHONPATH" not in kwargs["env"] and "PYTHONHOME" not in kwargs["env"]
        assert kwargs["cwd"] == tmp_path and kwargs["check"]

    monkeypatch.setattr(module.subprocess, "run", run)
    module.main()
    assert len(calls) == 1
    assert not Path(calls[0][-1]).exists()


@pytest.mark.parametrize("suffix", ["", "_dof"])
def test_g1_soma_and_bones_csv_units_and_joint_order(suffix, tmp_path):
    columns = [
        "Frame",
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
    ]
    columns += [name + suffix for name in G1_JOINT_NAMES]
    row = [0, 100, -50, 80, 10, 20, 30] + list(range(29))
    source = tmp_path / "motion.csv"
    frame = pd.DataFrame([row, row], columns=columns)
    frame.to_csv(source, index=False)
    result = load_csv(source)
    assert result.shape == (2, 36)
    np.testing.assert_allclose(result[0, :3], [1, -0.5, 0.8])
    np.testing.assert_allclose(
        result[0, 3:7], Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_quat()
    )
    np.testing.assert_allclose(result[0, 7:], np.deg2rad(np.arange(29)))
    frame.drop(columns=columns[-1]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="joint columns"):
        load_csv(source)
    reordered = columns.copy()
    reordered[-1], reordered[-2] = reordered[-2], reordered[-1]
    frame[reordered].to_csv(source, index=False)
    with pytest.raises(ValueError, match="joint columns"):
        load_csv(source)
