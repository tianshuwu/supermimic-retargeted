"""Measure X2 palm orientation and wrist joint errors on physical rollouts."""

import argparse
import hashlib
import json
from pathlib import Path

from supermimic.robots.paths import select_robot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="Optional baseline checkpoint to compare")
    parser.add_argument("--starts", nargs="+", type=float, default=[0, 20, 40, 60, 80, 100, 111.46])
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args()
    select_robot("x2_ultra", check_assets=True)
    import torch

    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.eval.hand_tracking import HandTrackingEvaluator, comparable_hands

    def digest(path):
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()

    policy, config = load_policy_from_checkpoint(args.ckpt, "cuda:0")
    evaluator = HandTrackingEvaluator(args.npz, config, args.starts, args.seconds)
    try:
        torch.manual_seed(0)
        report = evaluator.evaluate(policy)
        report.update(
            checkpoint=str(args.ckpt),
            checkpoint_sha256=digest(args.ckpt),
            reference=str(args.npz),
            reference_sha256=digest(args.npz),
        )
        if args.baseline:
            baseline_policy, baseline_config = load_policy_from_checkpoint(args.baseline, "cuda:0")
            for key in ("future_frames", "anchor_obs"):
                if config["env"].get(key) != baseline_config["env"].get(key):
                    raise ValueError(f"Baseline has incompatible {key}")
            torch.manual_seed(0)
            baseline = evaluator.evaluate(baseline_policy)
            baseline.update(checkpoint=str(args.baseline), checkpoint_sha256=digest(args.baseline))
            report["baseline"] = baseline
            report["acceptance"] = comparable_hands(report, baseline)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
