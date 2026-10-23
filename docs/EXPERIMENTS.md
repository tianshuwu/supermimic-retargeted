# Experiment record and limits

These are selected simulation engineering runs, not a strict reproduction claim. Full numeric summaries and checkpoint SHA256 values are in `docs/results/summary.json`; per-motion rollouts are in `docs/results/*-lafan.json` and `docs/results/*-bones.json`. Weights and training logs are not included in this code release.

The archived X2 runs predate the [terminal-palm tracking correction](X2_WRIST_FIX.md).
They used wrist-yaw tracking bodies and the original SOMA left-hand mapping.
The corrected defaults change the hand reward/reference semantics; new runs need
fresh evaluation and must not inherit the archived scores.

The active observations include `anchor_obs`. X2/A3 use 31 DOFs including two head joints, with 14 tracked bodies. Robot-ground contact is enabled, self-collision disabled, and the models use serial-joint/rigid-sole approximations. The 14-body tracking reward does not directly supervise every head joint. No real hardware tests were performed.

Each embodiment trained on two RTX 6000D GPUs with 4096 environments per rank, replay capacity 512, and policy batch 2048. LAFAN baselines used all 40 clips (2.4506 hours). X2/A3 policies/critics were initialized afresh for corrected references, with a world-model warmstart from earlier adaptation attempts. G1 retained the upstream G1 reference. Checkpoint selection used full deterministic clip rollouts, no noise or domain randomization, starting from the first reference frame.

LAFAN baseline selected iterations: G1 42000, X2 44000, A3 32000. BONES expanded selected iterations: G1 55000, X2 57000, A3 49000. These counters include earlier resumed training and must not be interpreted as the number of updates from scratch in the final phase. Full-checkpoint resume loads world model, policy, critic, their optimizers, and compatible sampling statistics; replay is freshly warmed up. The code also supports a world-model-only warmstart.

BONES libraries included 1965 / 1976 / 9951 accepted originals for G1 / X2 / A3 plus 40 LAFAN clips. BONES source hours were 4.461 / 4.476 / 21.516, respectively. Each validation set contained 198 accepted clips from the selected 200. Train/validation capture groups were disjoint; validation participated in checkpoint selection.

A3's initial expansion regressed on LAFAN. Its resumed run used `--env.replay-clip-list manifests/lafan1.txt --env.replay-sampling-fraction 0.5` to assign half of reset probability to the LAFAN bins while retaining adaptive probabilities inside each group and keeping all accepted BONES motions. G1/X2 used the default return-based sampling without this extra group mixture. The mix is an engineering curriculum choice, not a new algorithm claim.

Normalized return divides by executed steps, which can overstate an early-failing clip. We also report `completion_fraction` and `full_clip_weighted_return` to expose this limitation. The full-clip-weighted metric divides aggregate rewarded steps by full reference lengths. Per-clip failures remain visible.

The observed scores do not establish superiority over SONIC or the paper: observations, robot models, subsets and validation selection differ. Data preparation and short demo clips are not substitutes for full motion rollouts or hardware validation.

公开命令可用于从头训练。表中数字还依赖未随代码发布的训练状态，包括早期世界模型 warmstart；不能保证直接运行示例得到完全相同的数值。

## Historical comparison

![Paired simulation results](bones-comparison.png)
