"""X2 palm orientation must affect both data modes and differentiable training rewards."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_outdated_x2_profile_is_rejected(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "robot": "x2_ultra",
                "num_joints": 31,
                "tracked_bodies": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
                "xml_path": "robot.xml",
            }
        )
    )
    env = os.environ.copy()
    env.update(SM_ROBOT="x2_ultra", SM_ROBOT_SPEC=str(profile))
    result = subprocess.run(
        [sys.executable, "-c", "import supermimic.robots.dimensions"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "Outdated X2 hand tracking" in result.stderr


def test_fresh_x2_build_tracks_palms(tmp_path, monkeypatch):
    if not (
        ROOT / "vendor/soma-retargeter/soma_retargeter/assets/robotics/agibot/x2/desc"
    ).exists():
        pytest.skip("Requires fetched X2 sources and meshes")
    from supermimic.robots import build

    (tmp_path / "vendor").symlink_to(ROOT / "vendor", target_is_directory=True)
    monkeypatch.setattr(build, "ROOT", tmp_path)
    build.build_soma_robot("x2_ultra", "x2", "pelvis", "torso_link", "link")
    profile = json.loads((tmp_path / "assets/robots/x2_ultra/profile.json").read_text())
    assert len(profile["tracked_bodies"]) == 14
    assert [n for n in profile["tracked_bodies"] if "wrist" in n] == [
        "left_wrist_roll_link",
        "right_wrist_roll_link",
    ]
    # The configured calibration correction must equal the actual pitch-to-palm
    # transform, including its sign and axis, at SOMA's reference joint pose.
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation

    source = ROOT / "vendor/soma-retargeter/soma_retargeter/assets/robotics/agibot/x2/configs"
    scaler = json.loads((source / "soma_to_agibot_x2ultra_scaler_config.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "assets/robots/x2_ultra/robot.xml"))
    data = mujoco.MjData(model)
    data.qpos[:] = scaler["reference_joint_q"]
    data.qpos[3:7] = np.array(scaler["reference_joint_q"])[[6, 3, 4, 5]]
    mujoco.mj_forward(model, data)
    pitch = model.body("left_wrist_pitch_link").id
    palm = model.body("left_wrist_roll_link").id
    np.testing.assert_allclose(data.xpos[pitch], data.xpos[palm], atol=1e-12)
    delta = data.xmat[pitch].reshape(3, 3).T @ data.xmat[palm].reshape(3, 3)
    config = json.loads((ROOT / "configs/retarget/x2_ultra.json").read_text())
    rotation = config["soma_override"]["scaler_frame_rotations"]["LeftHand"]
    expected = Rotation.from_rotvec(np.array(rotation["axis"]) * rotation["angle_rad"])
    np.testing.assert_allclose(delta, expected.as_matrix(), atol=1e-10)


def test_x2_palm_rewards_and_library_modes(tmp_path):
    profile = ROOT / "assets/robots/x2_ultra/profile.json"
    if not profile.exists():
        pytest.skip("Build X2 robot assets first")
    env = os.environ.copy()
    env.update(SM_ROBOT="x2_ultra", SM_ROBOT_SPEC=str(profile))
    code = r"""
import sys
from pathlib import Path
import numpy as np
import torch
from supermimic.robots.dimensions import PROFILE, NQ
from supermimic.data.motion_convert import load_g1_model, convert_csv_to_npz
from supermimic.envs.tracking_env_cfg import TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame

path=Path(sys.argv[1]);m=load_g1_model()
q0=np.array([PROFILE['default_joint_pos'][n] for n in PROFILE['joint_names']])
raw=np.tile(np.r_[0,0,PROFILE['standing_height'],0,0,0,1,q0],(10,1))
np.savetxt(path/'motion.csv',raw,delimiter=',')
convert_csv_to_npz(path/'motion.csv',path/'motion.npz',50,50)
cfg=make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[str(path/'motion.npz')],num_envs=1))
assert cfg.commands['motion'].body_names==TRACKED_BODIES
assert len(TRACKED_BODIES)==14
frame=torch.zeros(1,dtype=torch.long)
libraries=[MotionLibrary([path/'motion.npz'],TRACKED_BODIES,q0,device='cpu',compact=c,
                        dtype=torch.float64,store_dtype=torch.float64) for c in [False,True]]
for key in ['pos_w','quat_w','lin_vel_w','ang_vel_w']:
    torch.testing.assert_close(getattr(libraries[0].bodies(frame),key),getattr(libraries[1].bodies(frame),key),atol=1e-6,rtol=1e-6)
limits=m.jnt_range[1:];mid=limits.mean(1);span=(limits[:,1]-limits[:,0])*0.45
heads=AnalyticHeads(G1Kinematics(m,'cpu',torch.float64),TRACKED_BODIES,ANCHOR_BODY,ROOT_BODY,
                    torch.tensor(np.stack([mid-span,mid+span],axis=1)),torch.tensor(q0))
for lib in libraries:
    b=lib.bodies(frame);r=lib.ref(frame)
    ref=ReferenceFrame(b.pos_w,b.quat_w,b.lin_vel_w,b.ang_vel_w,r.root_pos_w,r.root_quat_w)
    qp,qv=lib.ref_qpos_qvel(frame);state=torch.cat([qp,qv],dim=-1)
    # Exercise a nonsingular pose error for the upstream quaternion log derivative.
    state[0,3:7]=torch.tensor([np.cos(0.015),np.sin(0.015),0,0])
    actions=torch.zeros((1,31),dtype=torch.float64)
    baseline=heads.reward(state,actions,actions,ref)['total'].item()
    for side,sign in [('left',-1),('right',1)]:
        for axis,angle in [('roll',sign*np.pi/3),('pitch',np.pi/9)]:
            index=int(m.joint(f'{side}_wrist_{axis}_joint').qposadr[0])
            perturbed=state.clone();perturbed[0,index]+=angle;perturbed.requires_grad_()
            terms=heads.reward(perturbed,actions,actions,ref)
            assert terms['joint_limit'].item()==0
            assert terms['total'].item()<baseline-1e-4
            terms['total'].sum().backward()
            assert torch.isfinite(perturbed.grad).all()
            assert perturbed.grad[0,index].item()*angle<0
print('both palms: full/compact libraries, simulator config, reward and corrective gradients passed')
"""
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], cwd=ROOT, env=env, check=True, timeout=90
    )
