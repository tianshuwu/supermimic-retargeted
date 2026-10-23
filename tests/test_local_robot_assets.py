"""Local G1 geometry must drive both training and motion conversion."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants

from supermimic.data.motion_convert import g1_xml_path, load_g1_model
from supermimic.robots import assets as robot_asset
from supermimic.robots import paths as robot_paths

ROOT = Path(__file__).resolve().parents[1]


def _builder():
    from supermimic.robots import build

    return build


def test_local_g1_preserves_training_configuration():
    from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg

    upstream_cfg = g1_constants.get_g1_robot_cfg()
    local_cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[], num_envs=1)).scene.entities["robot"]
    assert g1_xml_path() == ROOT / "assets/robots/g1/robot.xml"
    assert local_cfg.spec_fn is robot_asset.get_spec
    assert local_cfg.init_state == upstream_cfg.init_state
    assert local_cfg.articulation == upstream_cfg.articulation
    assert local_cfg.collisions == upstream_cfg.collisions

    # Building the entity also applies the actuator and collision configuration.
    upstream = upstream_cfg.build().spec.compile()
    local = local_cfg.build().spec.compile()
    assert (local.nq, local.nv, local.nu) == (36, 35, 29)
    for field in (
        "body_mass",
        "body_inertia",
        "body_pos",
        "body_quat",
        "jnt_range",
        "jnt_axis",
        "dof_armature",
        "geom_contype",
        "geom_conaffinity",
        "geom_condim",
        "geom_friction",
        "geom_priority",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_forcerange",
    ):
        np.testing.assert_array_equal(getattr(local, field), getattr(upstream, field))


def test_build_and_local_edits_are_independent_of_package_geometry(tmp_path, monkeypatch):
    builder = _builder()
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    builder.build_g1()
    local = tmp_path / "assets/robots/g1/robot.xml"
    upstream_xml = g1_constants.G1_XML
    assert local.read_bytes() == upstream_xml.read_bytes()
    assert not local.is_symlink()
    assert not local.samefile(upstream_xml)
    for mesh in (upstream_xml.parent / "assets").iterdir():
        copied = local.parent / "assets" / mesh.name
        assert copied.read_bytes() == mesh.read_bytes()
        assert not copied.is_symlink() and not copied.samefile(mesh)
    assert (local.parent / "source.json").is_file()
    assert (local.parent / "licenses/mjlab/LICENSE").is_file()

    tree = ET.parse(local)
    inertial = tree.find("./worldbody/body[@name='pelvis']/inertial")
    new_mass = float(inertial.get("mass")) * 1.1
    inertial.set("mass", str(new_mass))
    tree.write(local)
    edited = local.read_bytes()
    builder.build_g1()
    assert local.read_bytes() == edited  # Rebuild must preserve managed local edits.

    monkeypatch.setattr(robot_paths, "ROOT", tmp_path)
    monkeypatch.setattr(g1_constants, "G1_XML", tmp_path / "unavailable-package-g1.xml")
    for model in (load_g1_model(), robot_asset.get_robot_cfg().spec_fn().compile()):
        pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        assert model.body_mass[pelvis] == pytest.approx(new_mass)


def test_missing_local_g1_has_no_package_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(robot_paths, "ROOT", tmp_path)
    for loader in (load_g1_model, robot_asset.get_robot_cfg().spec_fn):
        with pytest.raises(FileNotFoundError, match="build_robot_assets.py --robot g1"):
            loader()
