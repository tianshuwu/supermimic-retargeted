# X2 palm tracking correction

X2's wrist chain is **yaw → pitch → roll**. Its palm mesh belongs to the final
`wrist_roll_link`. The previous configuration tracked `wrist_yaw_link`, so a
pitch/roll error could rotate the palm without changing any tracked body pose.
The pinned SOMA preset also mapped `LeftHand` to `left_wrist_pitch_link`, leaving
the final left roll outside that hand IK objective.

| Configuration | Previous | Corrected |
|---|---|---|
| Training left/right hand bodies | `*_wrist_yaw_link` | `*_wrist_roll_link` |
| SOMA LeftHand position/orientation | `left_wrist_pitch_link` | `left_wrist_roll_link` |
| SOMA RightHand position/orientation | `right_wrist_roll_link` | Same |

The asset builder writes the corrected 14-body list. Existing X2 profiles with
the old list are rejected when loaded, so they cannot silently retain the old
reward. Simulator commands, full/compact motion libraries and the world model's
analytical reward head all consume the same `TRACKED_BODIES` list. G1 and A3
tracking definitions are unchanged.

## SOMA calibration and persistence

The correction lives in `configs/retarget/x2_ultra.json`, in `soma_override`.
`scripts/data/retarget.py` merges these entries with the pinned SOMA configuration
and registers a temporary override through SOMA's `extra_robot_paths` mechanism.
It does not modify vendor files or require copying third-party meshes/configs
into Git. Other IK weights, geometry and contact-processing settings are retained.
Use this project entrypoint; directly invoking the vendor app with its own
default config does not apply the project override.

SOMA's calibration pose has left wrist roll **−0.2131177634 rad**. Moving the
effector from pitch to roll therefore requires postmultiplying the existing
`LeftHand` offset quaternion by this local X-axis rotation. The two body origins
coincide, so the offset translation is unchanged. The regression test checks this
rotation against FK at the pinned `reference_joint_q` pose. Merely changing the
link name would introduce a calibration orientation bias.

## Existing assets, data and checkpoints

Back up customized X2 assets before rebuilding:

```bash
bash scripts/run_local.sh scripts/setup/build_robot_assets.py --robot x2_ultra
bash scripts/run_local.sh scripts/data/retarget.py --robot x2_ultra --print-config
```

For a customized profile whose geometry is already correct, update only its two
`tracked_bodies` entries to the terminal palms. This preserves its XML and motor
settings. The current workstation profile was migrated this way, with the old
profile and model files retained under `logs/x2-wrist-fix/backup/`.

Regenerate references into a separate directory, then validate them:

```bash
bash scripts/run_local.sh scripts/data/retarget.py --robot x2_ultra \
  --input data/lafan1-soma-aligned --output data/lafan1-retarget-x2-palms --batch-size 2
bash scripts/run_local.sh scripts/data/prepare_lafan.py --robot x2_ultra \
  --source data/lafan1-retarget-x2-palms --dest data/npz/x2_ultra/lafan1-palms --input-fps 30
```

Existing NPZ files store all body poses, so selecting the corrected tracking
bodies does not by itself require reconversion. It also does not fix incorrect
reference joint trajectories. Regenerate those with the corrected SOMA path;
if using a packed motion library, rebuild it from the new trajectories too.

The network's joint/state/observation dimensions are unchanged. Existing X2
weights can load, but their policy and critic learned the old reward. Continue
training in a separate run with the corrected references, or use the existing
world-model-only warmstart option. Preserve the checkpoint's architecture,
`anchor_obs` and other relevant training options. On the training server, for a
checkpoint at iteration 44000, an example continuation is:

```bash
.venv/bin/python scripts/training/train.py --robot x2_ultra \
  --resume logs/policy/x2_ultra_lafan-qualified.pt --run-name x2_lafan_palms \
  --data.npz-dir data/npz/x2_ultra/lafan1-palms --env.anchor-obs \
  --env.num-envs 4096 --env.replay-capacity 512 --policy-trainer.batch-size 2048 \
  --schedule.total-iterations 54000
```

`total-iterations` is the final counter, not the number of additional steps.
A new data path also resets the checkpoint's old sampling statistics. The
archived metrics in `docs/results/` predate this correction; corrected tracking
quality requires new evaluation, including explicit palm orientation errors.

## Regression checks

```bash
bash scripts/run_local.sh -m pytest tests/test_x2_wrist_tracking.py tests/test_retarget.py -q
```

Tests cover fresh asset generation, rejection of old profiles, unchanged vendor
files, calibration transforms, full/compact reference consistency and corrective
reward gradients for both wrists. Asset-dependent cases skip when X2 sources or
generated assets are unavailable. Local probe artifacts belong under `logs/`;
a short continuation only validates execution, not a corrected final policy.

## Measuring a resumed policy

`scripts/evaluation/eval_hand_tracking.py` measures paired palm orientation,
palm position and wrist yaw/pitch/roll errors from deterministic physical
rollouts. The default windows cover `dance1_subject1`: seven 20-second windows
starting at 0, 20, 40, 60, 80, 100 and 111.46 seconds. The last two overlap;
these are independently initialized windows, not one continuous full-clip run.
For other clips, pass suitable `--starts` and `--seconds` values.

```bash
bash scripts/run_local.sh scripts/evaluation/eval_hand_tracking.py \
  --ckpt logs/x2_lafan_palms/ckpt_latest.pt \
  --baseline logs/policy/x2_ultra_lafan-qualified.pt \
  --npz data/npz/x2_ultra/lafan1-palms/dance1_subject1.npz \
  --out logs/x2_lafan_palms/hand_comparison.json
```

Palm orientation is the geodesic angle to the reference after the same torso
xy/yaw alignment used by the training reward. Measurements are captured before
automatic resets; subsequent reset episodes do not contribute. Wrist joint
errors are reported separately so a favorable palm average cannot hide a roll
error. Each window reports its own completion and errors.

The comparison requires left palm mean error and roll MAE within 1.2 times the
right hand (a 2-degree floor applies to roll), improved left palm error, and no
more than 10% right-hand regression from baseline. All windows must complete.
Validate continuous tracking separately with `--starts 0 --seconds 131.46` on
this clip; window completion alone does not establish continuous completion.

For a visual check, use `scripts/evaluation/render_policy.py --robot x2_ultra`
with the same checkpoint and corrected NPZ directory. Solid geometry is the
physical policy; cyan geometry is the kinematic reference. If system `ffmpeg`
is unavailable, the renderer can use an installed `imageio-ffmpeg` binary;
pass `--no-label` if that binary lacks the `drawtext` filter.
