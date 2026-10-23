"""Evaluate a checkpoint on every clip of a motion set (paper Appendix F protocol).

  uv run python scripts/eval_benchmark.py --ckpt logs/e2/ckpt_latest.pt --npz-dir data/npz/lafan1
"""

from __future__ import annotations

from pathlib import Path

import tyro

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint, run_benchmark, save_results, summarize


def main(ckpt: Path, npz_dir: Path = Path("data/npz/lafan1"), clips: list[str] | None = None, clip_list: Path | None = None, manifest: Path | None = None, num_envs: int = 1024, device: str = "cuda:0", out: Path | None = None) -> None:
    if clip_list is not None:
        clips = [l.strip() for l in open(clip_list) if l.strip()]
    files = [str(f) for f in discover_npz(npz_dir, names=clips, manifest=manifest)]
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    results = run_benchmark(policy, files, device=device, num_envs=num_envs, future_frames=cfg["env"]["future_frames"], anchor_obs=cfg["env"].get("anchor_obs", False))
    s = summarize(results)
    print(f"\n{s['clips']} clips | failed {s['failed']} | mean norm return {s['mean_norm_return']:.4f} | step-weighted {s['step_weighted_norm_return']:.4f}")
    if "terms" in s:
        print("per-step term means (max 0.02 each for the 1.0-weight terms, 0.01 for root terms):")
        for k, v in s["terms"].items():
            print(f"  {k:28s} {v:.4f}")
    worst = sorted(results, key=lambda r: r.norm_return)[:10]
    for r in worst:
        print(f"  {r.name:32s} ret {r.norm_return:.3f} steps {r.steps}/{r.frames} {'FAIL@' + str(r.fail_step) if r.failed else ''}")
    out = out or ckpt.parent / f"benchmark_{ckpt.stem}.json"
    save_results(results, out)
    print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)
