"""Resolve retargeting paths and invoke the isolated SOMA environment."""

import argparse
import copy
import json
import math
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
    if "soma_override" in config:
        override = config["soma_override"]
        override["manifest"] = str((ROOT / override["manifest"]).resolve())
    return config


def prepare_soma_config(config, directory):
    """Register a temporary target override through SOMA's public manifest mechanism.

    Patch only the requested IK matches. Robot geometry, calibration and contact
    processing keep using the pinned upstream assets; vendor files are read-only.
    """
    config = copy.deepcopy(config)
    override = config.pop("soma_override", None)
    if override is None:
        return config
    manifest_path = Path(override["manifest"])
    asset_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    target = next(
        (
            entry
            for entry in manifest.get("targets", [manifest])
            if entry["name"] == config["retarget_target"]
        ),
        None,
    )
    if target is None:
        raise ValueError(f"Target {config['retarget_target']} missing from {manifest_path}")
    source = config["retarget_source"]
    retarget = json.loads((asset_root / target["retarget_configs"][source]).read_text())
    for name, changes in override["ik_match_table"].items():
        if name not in retarget["ik_match_table"]:
            raise ValueError(f"Unknown SOMA IK match: {name}")
        if changes.keys() - {"t_body", "r_body", "t_weight", "r_weight"}:
            raise ValueError(f"Unsupported SOMA IK override: {changes}")
        retarget["ik_match_table"][name].update(changes)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # SOMA resolves these paths relative to the manifest, not the JSON file.
    scaler_path = (asset_root / retarget["human_robot_scaler_config"]).resolve()
    if rotations := override.get("scaler_frame_rotations"):
        scaler = json.loads(scaler_path.read_text())
        for name, rotation in rotations.items():
            # Effector = human_world * offset. Postmultiply the original offset
            # by the old-to-new robot frame rotation at the calibration pose.
            offset = scaler["human_joint_offsets"][name]
            axis = rotation["axis"]
            norm = math.sqrt(sum(v * v for v in axis))
            if len(axis) != 3 or norm == 0:
                raise ValueError(f"Invalid calibration rotation axis for {name}")
            half = rotation["angle_rad"] / 2
            x2, y2, z2 = [v / norm * math.sin(half) for v in axis]
            w2 = math.cos(half)
            x1, y1, z1, w1 = offset[1]  # SOMA uses xyzw.
            offset[1] = [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ]
        scaler_path = directory / "scaler.json"
        scaler_path.write_text(json.dumps(scaler, indent=2) + "\n")
    retarget["human_robot_scaler_config"] = str(scaler_path.resolve())
    if "post_processing" in retarget:
        post = retarget["post_processing"]
        post["robot_config"] = str((asset_root / post["robot_config"]).resolve())
    for key in ("urdf_path", "xml_path"):
        if key in target["desc"]:
            target["desc"][key] = str((asset_root / target["desc"][key]).resolve())
    (directory / "retargeter.json").write_text(json.dumps(retarget, indent=2) + "\n")
    target["retarget_configs"] = {source: "retargeter.json"}
    (directory / "manifest.json").write_text(json.dumps(target, indent=2) + "\n")
    config["extra_robot_paths"].append(str(directory.resolve()))
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
        config = prepare_soma_config(config, Path(temp) / "robot")
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
