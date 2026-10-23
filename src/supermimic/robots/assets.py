"""mjlab entity configurations that load geometry from local robot assets."""

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from supermimic.robots.dimensions import PROFILE, ROBOT, robot_xml_path


def get_spec():
    return mujoco.MjSpec.from_file(str(robot_xml_path()))


def get_robot_cfg():
    if ROBOT == "g1":
        from mjlab.asset_zoo.robots import get_g1_robot_cfg

        # Preserve the pinned G1 actuator, initial-state and collision settings.
        # Only the geometry source changes to our independently managed copy.
        cfg = get_g1_robot_cfg()
        cfg.spec_fn = get_spec
        return cfg
    actuators = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(n,),
            stiffness=PROFILE["stiffness"][n],
            damping=PROFILE["damping"][n],
            effort_limit=PROFILE["effort_limit"][n],
            armature=PROFILE["armature"][n],
        )
        for n in PROFILE["joint_names"]
    )
    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0, 0, PROFILE["standing_height"]),
            joint_pos=PROFILE["default_joint_pos"],
            joint_vel={".*": 0.0},
        ),
        collisions=(
            CollisionCfg(
                geom_names_expr=(".*_collision",),
                contype=1,
                conaffinity=0,
                condim={"^foot_.*_collision$": 3, ".*_collision": 1},
                priority={"^foot_.*_collision$": 1, ".*": 0},
                friction={"^foot_.*_collision$": (1.0,)},
            ),
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=actuators, soft_joint_pos_limit_factor=0.9
        ),
    )
