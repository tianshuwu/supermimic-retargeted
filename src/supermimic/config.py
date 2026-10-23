"""All training configuration in one dataclass tree (tyro-friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.policy.nets import CriticCfg, PolicyCfg
from supermimic.robots.paths import lafan_npz_dir, robot_name
from supermimic.world_model.wm_trainer import WMTrainerCfg
from supermimic.world_model.world_model import WorldModelCfg


@dataclass
class DataCfg:
    npz_dir: Path = field(default_factory=lafan_npz_dir)
    clips: list[str] | None = None  # None = all npz in npz_dir
    limit: int | None = None
    manifest: Path | None = None  # csv with name,keep (filter_bones_seed.py output)
    library: Path | None = (
        None  # packed compact library (.pt from build_library.py); overrides npz_dir
    )


@dataclass
class EnvCfg:
    num_envs: int = 4096
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"
    replay_clip_list: Path | None = None
    replay_sampling_fraction: float = 0.0
    replay_capacity: int = 1000
    future_frames: int = 10
    compact_library: bool = False
    anchor_obs: bool = False


@dataclass
class ScheduleCfg:
    total_iterations: int = 320_000  # env steps (plan D12 reading A)
    updates_per_env_step: float = 0.4  # UTD
    policy_updates_per_wm_update: int = 1
    warmup_iterations: int = 40  # env steps before the first update (fills k-step windows)
    lr_final_ratio: float = 0.03
    lr_decay_fraction: float = 250_000 / 320_000
    log_interval: int = 50  # iterations
    checkpoint_interval: int = 4000  # iterations
    max_hours: float | None = None


@dataclass
class TrainConfig:
    robot: str = field(default_factory=robot_name)
    run_name: str = "debug"
    log_root: Path = Path("logs")
    seed: int = 0
    device: str = "cuda:0"
    data: DataCfg = field(default_factory=DataCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    resume: Path | None = None
    warmstart_world_model: Path | None = None


def training_config(preset: str = "default") -> TrainConfig:
    """Keep algorithm defaults and the bounded local validation preset together."""
    cfg = TrainConfig()
    if preset == "default":
        return cfg
    if preset != "smoke":
        raise ValueError(f"Unknown training preset: {preset}")
    cfg.run_name = f"{cfg.robot}_lafan_smoke"
    cfg.env.num_envs = 128
    cfg.env.replay_capacity = 256
    cfg.wm_trainer.batch_size = 256
    cfg.wm_trainer.batches_per_update = 2
    cfg.wm_trainer.compile = None
    cfg.policy_trainer.batch_size = 128
    cfg.policy_trainer.compile = None
    cfg.schedule.total_iterations = 100
    cfg.schedule.log_interval = 20
    cfg.schedule.checkpoint_interval = 100
    cfg.schedule.max_hours = 1 / 6
    return cfg
