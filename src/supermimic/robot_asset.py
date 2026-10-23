"""mjlab entity configuration using the active versioned robot model profile."""
import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.utils.spec_config import CollisionCfg
from supermimic.robot_dimensions import PROFILE

def get_spec():
    return mujoco.MjSpec.from_file(PROFILE["xml_path"])

def get_robot_cfg():
    actuators=tuple(BuiltinPositionActuatorCfg(target_names_expr=(n,),
        stiffness=PROFILE["stiffness"][n],damping=PROFILE["damping"][n],
        effort_limit=PROFILE["effort_limit"][n],armature=PROFILE["armature"][n]) for n in PROFILE["joint_names"])
    return EntityCfg(spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0,0,PROFILE["standing_height"]),
            joint_pos=PROFILE["default_joint_pos"],joint_vel={".*":0.}),
        collisions=(CollisionCfg(geom_names_expr=(".*_collision",),contype=1,conaffinity=0,
            condim={"^foot_.*_collision$":3,".*_collision":1},
            priority={"^foot_.*_collision$":1,".*":0},friction={"^foot_.*_collision$":(1.,)}),),
        articulation=EntityArticulationInfoCfg(actuators=actuators,soft_joint_pos_limit_factor=.9))
