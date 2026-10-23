# Training and evaluation

所有命令从仓库根目录执行，使用 `.venv/`；安装和数据准备见 [README](../README.md) 与 [DATA.md](DATA.md)。

## 查看配置与本地验证

训练参数集中在 `src/supermimic/config.py`。`--robot` 在导入训练模型前生效；`--preset smoke` 提供有界的本地验证，命令行参数可继续覆盖预设。

```bash
# 只打印配置，不创建仿真或启动训练
bash scripts/run_local.sh scripts/training/train.py --robot g1 --print-config
bash scripts/run_local.sh scripts/training/train.py --robot g1 --preset smoke --print-config

# Headless 本地短训：100 个环境步，正常结束并保存检查点
bash scripts/run_local.sh scripts/training/train.py --robot g1 --preset smoke
```

| 参数 | 默认配置 | smoke 预设 |
|---|---:|---:|
| 并行环境 | 4096 | 128 |
| replay 容量 / 环境 | 1000 | 256 |
| 世界模型 batch / 每次更新批数 | 2048 / 10 | 256 / 2 |
| 策略 batch / 展开步数 | 4096 / 32 | 128 / 32 |
| PyTorch 编译 | 开启 | 关闭 |
| 最大环境步数 | 320000 | 100 |
| 时间上限 | 无 | 10 分钟 |
| 检查点间隔 | 4000 步 | 100 步 |

预设沿用默认网络、学习率 `2e-4`、噪声与随机化。时间上限从训练循环开始计时，在日志间隔处检查；smoke 通常先达到 100 步上限。训练入口默认 headless，没有额外的 `--headless` 参数。

未指定 `--run-name` 时自动生成唯一运行名称。配置、TensorBoard 事件、`ckpt_latest.pt` 和编号检查点位于 `logs/<run-name>/`。短训通过表示训练流程可执行，不表示策略已收敛。

## 服务器训练与评测

以下命令供正式训练机器使用。可把 `a3` 替换成 `g1` 或 `x2_ultra`，并按设备资源调整规模。

```bash
# 单卡 LAFAN，参数对应仓库原有实验入口
bash scripts/run_local.sh scripts/training/train.py --robot a3 \
  --run-name a3_lafan --env.num-envs 4096 --env.replay-capacity 512 \
  --env.anchor-obs --policy-trainer.batch-size 2048 --schedule.max-hours 3

# 两卡，每个进程在初始化 CUDA 前选择自己的 GPU
bash scripts/run_local.sh -m torch.distributed.run --standalone --nproc-per-node 2 \
  scripts/training/train.py --robot a3 --run-name a3_lafan_ddp \
  --env.num-envs 4096 --env.replay-capacity 512 --env.anchor-obs \
  --policy-trainer.batch-size 2048 --schedule.max-hours 3

# 完整恢复世界模型、策略、critic 与优化器
bash scripts/run_local.sh scripts/training/train.py --robot a3 \
  --resume logs/a3_lafan/ckpt_latest.pt --run-name a3_lafan_resume \
  --env.anchor-obs --env.replay-capacity 512 \
  --policy-trainer.batch-size 2048 --schedule.max-hours 3

# 确定性评测
bash scripts/run_local.sh scripts/robot.py a3 scripts/evaluation/eval_benchmark.py \
  --ckpt logs/a3_lafan/ckpt_latest.pt

# 实体策略与半透明参考在同一场景中的视频
bash scripts/run_local.sh scripts/evaluation/render_policy.py --robot a3 \
  --ckpt logs/a3_lafan/ckpt_latest.pt --clip dance1_subject1 \
  --record-start-seconds 75 --seconds 20 --out videos/a3_clap.mp4
```

恢复时仍需传入原训练的数据集与配置覆盖参数，`--resume` 不会自动替换当前配置。检查点路径是示例，请替换为实际训练产物。BONES 扩容、过滤和 LAFAN replay 混合见 [docs/DATA.md](DATA.md) 与 [docs/EXPERIMENTS.md](EXPERIMENTS.md)。

录像会先物理执行到指定开始时间，遇到终止条件即停止。视频旁的 JSON 记录检查点 SHA256、片段和终止信息；青色参考虚影仅用于可视化。
