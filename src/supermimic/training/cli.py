"""Training CLI; select the robot and GPU before importing the learning code."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime

from supermimic.robots.paths import ROBOTS, robot_name, select_robot


def _n_gpus() -> int:
    try:
        return len(os.listdir("/proc/driver/nvidia/gpus"))
    except OSError:
        return 0


def _pin_distributed_gpu() -> None:
    # A preset CUDA_VISIBLE_DEVICES list (e.g. "0,2,3" to skip a bad card) is indexed by LOCAL_RANK.
    _preset = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    _avail = _preset if _preset else [str(i) for i in range(_n_gpus())]
    if (
        "LOCAL_RANK" in os.environ
        and int(os.environ.get("LOCAL_WORLD_SIZE", "1")) <= len(_avail)
        and (not _preset or len(_preset) > 1)
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = _avail[int(os.environ["LOCAL_RANK"])]
        os.environ["SM_GPU_PINNED"] = "1"
        # Give each rank its own slice of cores (the training loop is single-thread CPU bound;
        # sharing cores between ranks adds jitter that lockstep all-reduces turn into waiting).
        try:
            cores = sorted(os.sched_getaffinity(0))
            n_local = int(os.environ["LOCAL_WORLD_SIZE"])
            per = len(cores) // n_local
            if per >= 2:
                r = int(os.environ["LOCAL_RANK"])
                os.sched_setaffinity(0, set(cores[r * per : (r + 1) * per]))
        except (OSError, ValueError):
            pass


def main(argv=None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap.add_argument("--robot", choices=ROBOTS, default=robot_name())
    bootstrap.add_argument("--preset", choices=("default", "smoke"), default="default")
    bootstrap.add_argument("--print-config", action="store_true")
    selected, remaining = bootstrap.parse_known_args(arguments)
    select_robot(selected.robot)
    _pin_distributed_gpu()

    import tyro

    from supermimic.config import TrainConfig, training_config

    cfg = tyro.cli(
        TrainConfig,
        default=training_config(selected.preset),
        args=remaining,
        description=(
            "Headless SuperMimic training. Entry options: --robot {g1,a3,x2_ultra}, "
            "--preset {default,smoke}, --print-config. The smoke preset ends after "
            "100 environment steps; any config option below overrides its defaults."
        ),
    )
    if selected.print_config:
        print(json.dumps(asdict(cfg), indent=2, default=str))
        return
    select_robot(cfg.robot, check_assets=True)
    if not any(a == "--run-name" or a.startswith("--run-name=") for a in remaining):
        cfg.run_name = f"{cfg.robot}_lafan_{selected.preset}_{datetime.now():%Y%m%d_%H%M%S}"

    from supermimic.training.loop import Trainer

    trainer = Trainer(cfg)
    try:
        trainer.run()
    finally:
        trainer.env.close()
        if trainer.writer is not None:
            trainer.writer.close()


if __name__ == "__main__":
    main()
