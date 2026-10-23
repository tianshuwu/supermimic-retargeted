# supermimic-retargeted

Personal, AI-assisted robot adaptation of **SuperMimic-Reproduced** for **G1, A3 and X2 Ultra**.

本仓库主要通过 vibe coding，把社区已有的 SuperMimic 复现接入 A3、X2 Ultra，并整理了数据处理、训练、评测和参考虚影渲染流程。我们没有提出新的算法，也不是机器人厂商或 SuperMimic 作者的官方项目。

训练实现主要来自 [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)，交叉参考 [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)。请优先引用原论文和上游项目。固定版本与其他来源见 [UPSTREAM.md](UPSTREAM.md)。

## 支持范围

| 本体 | 关节数 | 本次验证 |
|---|---:|---|
| Unitree G1 | 29 | LAFAN1、BONES 子集 |
| A3 | 31 | LAFAN1、约一万条 BONES 动作 |
| X2 Ultra | 31 | LAFAN1、约两千条 BONES 动作 |

A3/X2 的适配包括关节与状态维度、可微 FK、机器人参数、SOMA 重定向、LAFAN 骨骼坐标系及骨盆支点修正，以及脚底接触质检。`scripts/evaluation/render_policy.py` 在同一个场景中显示实体策略与半透明参考虚影；虚影不参与物理碰撞，也不进行逐帧位置对齐。

X2 现已改为跟踪最终手掌，并同步修正 SOMA 左手目标和校准。已有资产、数据及检查点的迁移见 [X2 手腕修复说明](docs/X2_WRIST_FIX.md)；下方历史实验指标仍对应修复前配置。

## 代码结构

核心训练保持在 `src/supermimic/` 的既有模块中；机器人适配和数据准备有独立入口。目录职责、调用关系及旧入口映射见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

```text
src/supermimic/
├── config.py          # 训练配置树，以及本地 smoke 预设
├── training/          # CLI、完整训练循环、分布式和学习率调度
├── world_model/       # 动力学模型、预测损失和解析计算
├── policy/            # 策略和 critic 网络
├── imagination/       # 想象轨迹、BPTT 和 actor–critic 更新
├── envs/              # mjlab 场景、观测、奖励、终止和动作参考
├── kinematics/        # 可微 FK 与旋转运算
├── robots/            # 机器人选择、维度、资产加载和构建
└── data/              # CSV/NPZ 转换、接触修正、replay 和采样
scripts/               # setup / data / training / evaluation / dev
configs/retarget/      # G1/A3/X2 的 SOMA 重定向任务配置
assets/robots/         # 本地 G1/A3/X2 模型，由构建脚本生成
data/                  # 原始数据、中间结果、NPZ 和打包数据（不入 Git）
manifests/             # 公开的动作选择清单（入 Git）
docs/results/          # README 对应的历史评测指标（入 Git）
tests/                 # 可复用回归测试源码（入 Git）
logs/                  # 各次运行的配置、权重、事件和审计报告（不入 Git）
```

脚本用途及路径迁移见 [scripts/README.md](scripts/README.md)。`manifests/` 的文件只指定选用哪些动作；训练读取 `data/` 中实际的轨迹数组或打包库。

## 环境与机器人资产

需要 Linux、Python 3.12 和 NVIDIA GPU。主环境为 `.venv/`，SOMA 重定向使用 `vendor/soma-retargeter/.venv/`。渲染视频另需支持 `drawtext` 的 `ffmpeg`。

**已有本地环境**：使用 `bash scripts/run_local.sh ...`。它直接调用 `.venv/bin/python`，清除外部 Python 环境干扰，将临时文件写到项目 `.cache/`，不会重新解析依赖。当前工作站已验证 PyTorch CUDA 12.6 / Warp CUDA 12 组合；项目原始锁文件面向 CUDA 13，在当前工作站运行 `uv sync` / `uv run` 会尝试替换兼容包。

**在适配 CUDA 13 的新机器上从零安装**：

```bash
uv sync --frozen --group dev
# A3/X2 资产构建、任一本体的 SOMA 重定向还需要外部源文件
.venv/bin/python scripts/setup/fetch_upstreams.py
uv sync --project vendor/soma-retargeter --frozen
# 获取 A3/X2 的机器人描述和网格
git -C vendor/soma-retargeter lfs pull --include='soma_retargeter/assets/robotics/agibot/**' --exclude=''
.venv/bin/python scripts/setup/build_robot_assets.py
```

`fetch_upstreams.py` 获取固定 commit 的外部仓库并跳过大型 LFS 文件，上面的 `git lfs pull` 获取 A3/X2 所需资产。G1 的 SOMA 重定向另有资产准备命令，见 [docs/DATA.md](docs/DATA.md)。固定版本与归属见 [UPSTREAM.md](UPSTREAM.md)。

仅使用 G1 时，在安装主环境后执行：

```bash
bash scripts/run_local.sh scripts/setup/build_robot_assets.py --robot g1
```

G1 的 `robot.xml`、35 个网格文件、许可证和 `source.json` 放在 `assets/robots/g1/`；它们是 mjlab 资产的独立本地副本。训练、评测和数据转换读取本地 XML，控制参数与碰撞配置沿用固定版本 mjlab。重复构建会校验并保留已有 G1 目录；重新导入前先将旧目录移走备份。

不指定 `--robot` 时，构建脚本准备 G1、A3、X2 三套资产。生成的 `assets/robots/`、数据、日志和权重不提交 Git。

## 准备数据

标准训练目录是 `data/npz/<robot>/lafan1/`，其中 `<robot>` 为 `g1`、`a3` 或 `x2_ultra`。训练、评测和测试使用相同布局。完整数据流程见 [docs/DATA.md](docs/DATA.md)。

`data/` 不是只存放可直接训练的数据：它还保存下载的源仓库、原始 BVH、重定向 CSV、转换中间文件和本次生成的过滤清单。训练日志、模型检查点放在 `logs/`。

G1 使用已重定向的 LAFAN CSV，将 30 Hz 输入转换为 50 Hz NPZ：

```bash
bash scripts/run_local.sh scripts/data/prepare_lafan.py --robot g1 \
  --source data/lafan1-retargeted/g1 --input-fps 30
```

这会输出 `data/npz/g1/lafan1/`、过滤清单和转换审计。A3/X2 需要先做 BVH 坐标适配及 SOMA 重定向。LAFAN 与 BONES 的源帧率分别为本次使用的 30 Hz、120 Hz，不可混用。

`configs/retarget/g1.json` 也支持从 BVH 重新生成 G1 CSV；原先没有该配置，是因为 G1 基线直接使用 ModelScope 的成品 CSV。新重定向输出默认放在 `data/lafan1-retarget-g1/`，转换时显式指定该来源。配置含义见 [configs/retarget/README.md](configs/retarget/README.md)。

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
.venv/bin/python scripts/training/train.py --robot a3 \
  --run-name a3_lafan --env.num-envs 4096 --env.replay-capacity 512 \
  --env.anchor-obs --policy-trainer.batch-size 2048 --schedule.max-hours 3

# 两卡，每个进程在初始化 CUDA 前选择自己的 GPU
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node 2 \
  scripts/training/train.py --robot a3 --run-name a3_lafan_ddp \
  --env.num-envs 4096 --env.replay-capacity 512 --env.anchor-obs \
  --policy-trainer.batch-size 2048 --schedule.max-hours 3

# 完整恢复世界模型、策略、critic 与优化器
.venv/bin/python scripts/training/train.py --robot a3 \
  --resume logs/a3_lafan/ckpt_latest.pt --run-name a3_lafan_resume

# 确定性评测
bash scripts/run_local.sh scripts/robot.py a3 scripts/evaluation/eval_benchmark.py \
  --ckpt logs/a3_lafan/ckpt_latest.pt

# 实体策略与半透明参考在同一场景中的视频
bash scripts/run_local.sh scripts/evaluation/render_policy.py --robot a3 \
  --ckpt logs/a3_lafan/ckpt_latest.pt --clip dance1_subject1 \
  --record-start-seconds 75 --seconds 20 --out videos/a3_clap.mp4
```

恢复时仍需传入原训练的数据集与配置覆盖参数，`--resume` 不会自动替换当前配置。检查点路径是示例，请替换为实际训练产物。BONES 扩容、过滤和 LAFAN replay 混合见 [docs/DATA.md](docs/DATA.md) 与 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

录像会先物理执行到指定开始时间，遇到终止条件即停止。视频旁的 JSON 记录检查点 SHA256、片段和终止信息；青色参考虚影仅用于可视化。

## 本次实验记录

以下是工程配置下的实际仿真结果，包含额外的 `anchor_obs`，**不与论文观测口径等价**。

| 本体 | LAFAN 基线失败 / 40 | 实际 BONES 训练条数 | 扩容后 LAFAN 失败 / 40 | BONES 验证失败 / 198 |
|---|---:|---:|---:|---:|
| G1 | 0 | 1965 | 2 | 1 |
| X2 Ultra | 2 | 1976 | 2 | 0 |
| A3 | 0 | 9951 | 2 | 1 |

每个 BONES 训练库还包含原来的 40 条 LAFAN。BONES 验证按采集组隔离，但参与选模，因此不能作为未触碰的最终测试集。LAFAN 指标反映训练动作上的跟踪。聚合与逐片段结果保存在 [docs/results/](docs/results/README.md)，复现配置和限制见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。权重暂不随代码发布。

`docs/results/` 是这些历史实验的评测数据，不是当前训练的完整参数文件，也不是本地短训结果。默认训练参数在 `src/supermimic/config.py`；每次运行的实际参数在 `logs/<run-name>/config.json`。

![Paired results](docs/bones-comparison.png)

A3/X2 使用串联关节与刚性脚底模型，关闭自碰撞、保留地面接触；没有实机验证，也没有做与 SONIC 的公平比较。当前仍有失败动作，这是一份个人适配和实验记录。

## 测试与许可

`tests/` 是供维护者和使用者复用的回归测试；运行前提、CPU/GPU 分类与跳过原因见 [tests/README.md](tests/README.md)。测试输出和本机审计日志不随测试源码发布。

```bash
bash scripts/run_local.sh -m pytest tests -m "not slow" -q
# 发布前：先将要发布的源码加入 Git 索引，再从索引导出干净源码检查
bash scripts/run_local.sh scripts/dev/check_source_distribution.py
```

训练代码保留上游 MIT 许可，见 [LICENSE](LICENSE)。第三方工具、机器人资产和数据保留各自许可，见 [UPSTREAM.md](UPSTREAM.md)。
