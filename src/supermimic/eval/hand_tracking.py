"""Paired palm and wrist errors from deterministic physical rollouts."""

from __future__ import annotations

import math

import torch
from mjlab.utils.lab_api.math import quat_error_magnitude

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.rewards import reanchor_reference


class HandTrackingEvaluator:
    """Fixed, paired windows; terminated windows cannot pass acceptance checks.

    Measurements are taken at reward computation, before automatic reset and
    reference advancement. Link poses have mjlab's normal one-physics-substep
    lag. Yaw/xy alignment matches the training reward. No reference state is
    injected after initialization, and reset episodes are excluded.
    """

    def __init__(self, npz, config, starts=(0, 20, 40, 60, 80, 100, 111.46), seconds=20):
        self.starts = list(starts)
        self.steps = round(seconds * 50)
        self.env = make_env(
            SuperMimicEnvCfgArgs(
                npz_files=[str(npz)],
                num_envs=len(starts),
                future_frames=config["env"]["future_frames"],
                anchor_obs=config["env"].get("anchor_obs", False),
                episode_length_s=1e5,
                action_noise_std=0,
                obs_noise=False,
                reset_noise=False,
                domain_randomization=False,
                sampling_mode="start",
                record_transitions=False,
                seed=0,
            ),
            device="cuda:0",
        )
        self.cmd = self.env.command_manager.get_term("motion")
        self.palms = [
            self.cmd.cfg.body_names.index(f"{side}_wrist_roll_link") for side in ("left", "right")
        ]
        self.joints = [
            self.cmd.robot.joint_names.index(f"{side}_wrist_{axis}_joint")
            for side in ("left", "right")
            for axis in ("yaw", "pitch", "roll")
        ]
        self.start_frames = torch.tensor(
            [round(s * 50) for s in starts], device=self.env.device, dtype=torch.long
        )
        if bool((self.start_frames + self.steps > self.cmd.library.num_frames).any()):
            self.close()
            raise ValueError("Evaluation windows exceed the reference clip")
        self.original_compute = self.env.reward_manager.compute
        self.env.reward_manager.compute = self._compute_and_measure

    def _compute_and_measure(self, dt):
        result = self.original_compute(dt)
        cmd = self.cmd
        pos, quat = reanchor_reference(
            cmd.body_pos_w,
            cmd.body_quat_w,
            cmd.anchor_pos_w,
            cmd.anchor_quat_w,
            cmd.robot_anchor_pos_w,
            cmd.robot_anchor_quat_w,
        )
        ori = quat_error_magnitude(quat[:, self.palms], cmd.robot_body_quat_w[:, self.palms])
        position = (pos[:, self.palms] - cmd.robot_body_pos_w[:, self.palms]).norm(dim=-1)
        joint = (cmd.joint_pos[:, self.joints] - cmd.robot_joint_pos[:, self.joints]).abs()
        self.samples.append(torch.cat([ori, position, joint], dim=-1).detach().clone())
        self.masks.append(self.active.clone())
        return result

    @torch.no_grad()
    def evaluate(self, policy):
        was_training = policy.training
        policy.eval()
        self.samples, self.masks = [], []
        self.active = torch.ones(len(self.starts), device=self.env.device, dtype=torch.bool)
        self.cmd.forced_frame[:] = self.start_frames
        obs, _ = self.env.reset()
        failures = [None] * len(self.starts)
        returns = torch.zeros(len(self.starts), device=self.env.device)
        for step in range(self.steps):
            obs, reward, term, tout, _ = self.env.step(policy(obs["policy"], obs["goal"]))
            returns += reward * self.active
            ended = self.active & (term | tout)
            for i in ended.nonzero().flatten().tolist():
                failures[i] = {"step": step + 1, "terminated": bool(term[i])}
            self.active &= ~ended
            if not bool(self.active.any()):
                break
        policy.train(was_training)
        values = torch.stack(self.samples).cpu()
        mask = torch.stack(self.masks).cpu()

        def metrics(v):
            out = {}
            for side, i in (("left", 0), ("right", 1)):
                out[side] = {
                    "palm_orientation_mean_deg": float(v[:, i].mean()) * 180 / math.pi,
                    "palm_orientation_p95_deg": float(v[:, i].quantile(0.95)) * 180 / math.pi,
                    "palm_position_mean_cm": float(v[:, 2 + i].mean()) * 100,
                    "wrist_joint_mae_deg": (
                        v[:, 4 + i * 3 : 7 + i * 3].mean(0) * (180 / math.pi)
                    ).tolist(),
                }
            return out

        report = {
            "protocol": "deterministic physical windows, paired samples before autoreset",
            "alignment": "reference xy/yaw aligned to torso, as in the training reward",
            "joint_order": ["yaw", "pitch", "roll"],
            "requested_steps_per_window": self.steps,
            "all_windows_completed": all(f is None for f in failures),
            "paired_samples": int(mask.sum()),
            "aggregate": metrics(values[mask]),
            "windows": [
                {
                    "start_seconds": start,
                    "steps": int(mask[:, i].sum()),
                    "failure": failures[i],
                    "norm_return": float(returns[i]) / max(1, int(mask[:, i].sum())) / 0.1,
                    **metrics(values[:, i][mask[:, i]]),
                }
                for i, start in enumerate(self.starts)
            ],
        }
        return report

    def close(self):
        self.env.close()


def comparable_hands(report, baseline, ratio=1.2):
    """Require left improvement, bilateral parity and no material right regression."""
    a, b = report["aggregate"], baseline["aggregate"]
    left, right = a["left"], a["right"]
    key = "palm_orientation_mean_deg"
    checks = {
        "all_windows_completed": report["all_windows_completed"],
        "left_palm_within_ratio": left[key] <= ratio * right[key],
        "left_roll_within_ratio": left["wrist_joint_mae_deg"][2]
        <= max(ratio * right["wrist_joint_mae_deg"][2], 2.0),
        "left_improved": left[key] < b["left"][key],
        "right_palm_preserved": right[key] <= b["right"][key] * 1.1,
        "right_roll_preserved": right["wrist_joint_mae_deg"][2]
        <= max(b["right"]["wrist_joint_mae_deg"][2] * 1.1, 2.0),
    }
    return {"passed": all(checks.values()), "ratio": ratio, "checks": checks}
