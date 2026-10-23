# Architecture and entry points

本次整理将外围工程代码归位，保留 SuperMimic 核心算法的模块粒度。核心逻辑仍可沿着一个训练循环阅读。

## 核心训练从一个循环展开

`src/supermimic/training/loop.py` 的 `Trainer` 负责环境、模型、优化器和检查点的生命周期，以及每次迭代的完整顺序：

1. 策略驱动物理环境，状态和执行动作写入 replay。
2. 世界模型从 replay 的连续窗口学习动力学。
3. 策略在世界模型中展开想象轨迹，通过 BPTT 更新。
4. 更新 critic，记录指标并保存检查点。

| 模块 | 职责 |
|---|---|
| `config.py` | 完整训练配置树、默认配置和有界 `smoke` 预设 |
| `training/loop.py` | 完整训练主循环及检查点读写 |
| `world_model/` | 动力学网络、状态表示、解析头和预测损失 |
| `policy/` | 策略编码器、解码器和 critic 网络 |
| `imagination/` | 想象轨迹展开、目标函数和策略更新 |
| `envs/` | mjlab 物理环境、任务配置、状态与参考动作接口 |
| `kinematics/` | 可微正向运动学与旋转计算 |
| `data/replay_buffer.py`、`recorder.py`、`sampling.py` | 在线训练数据与采样 |

没有把优化步骤拆成新的服务、注册框架或大量小模块。网络结构、奖励、终止条件和检查点字段沿用原实现。

## 机器人适配集中在 robots

| 文件 | 职责 |
|---|---|
| `robots/paths.py` | 轻量机器人选择、资产路径、默认数据路径 |
| `robots/dimensions.py` | 所选机器人的关节、状态和观测维度，读取 A3/X2 profile |
| `robots/assets.py` | 从本地模型生成 mjlab 实体配置 |
| `robots/build.py` | 导入 G1 资产，构建 A3/X2 模型及 profile |

机器人选择必须早于训练模块导入，因为网络和状态布局会捕获关节维度。`training/cli.py` 与 `scripts/robot.py` 都使用同一个选择函数。试图在一个已经导入机器人维度的进程中切换机器人会报错；应启动新的 Python 进程。

G1 从 `assets/robots/g1/robot.xml` 读取几何模型，沿用固定版本 mjlab 的控制和碰撞配置。A3/X2 从各自 `profile.json` 读取配置；新生成 profile 的 XML 路径相对于 profile 目录解析，原有绝对路径仍受支持。

X2 训练跟踪左右手的末端 `wrist_roll_link`，旧的 `wrist_yaw_link` profile 会报错。环境、动作库和想象轨迹奖励共享这套跟踪点。SOMA 的左手目标及校准由仓库 JSON 覆盖，迁移步骤见 [X2_WRIST_FIX.md](X2_WRIST_FIX.md)。

## 离线数据准备与在线 replay 分工明确

`data/motion_convert.py` 提供 CSV 读取、重采样和 CPU FK。`data/prepare.py` 组织多进程转换、A3/X2 接触修正、过滤清单及审计输出。导入准备模块不会解析命令行或开始转换，每个 worker 显式初始化机器人与转换参数。

标准数据路径是 `data/npz/<robot>/lafan1/`。训练默认值、评测默认值和 G1 测试均使用这个布局，显式传入的其他数据路径仍然有效。

SOMA 相关 BVH 对齐继续由独立环境运行；其依赖边界没有并入训练环境。BONES 的下载、筛选、提取和打包工具继续保持各自独立命令。

## 命令入口与兼容性

| 命令 | 实现 |
|---|---|
| `scripts/training/train.py` / `sm-train` | `training/cli.py` |
| `python -m supermimic.scripts.train` | 保留的旧入口，转发到同一个训练 CLI |
| `scripts/setup/build_robot_assets.py` | `robots/build.py` |
| `scripts/data/prepare_lafan.py` | `data/prepare.py` |
| `scripts/robot.py ROBOT ...` | 在导入目标脚本前统一选择机器人 |
| `scripts/run_local.sh ...` | 直接使用现有 `.venv`，不执行依赖同步 |

旧的 `supermimic.robot_dimensions`、`supermimic.robot_asset` 导入仍可使用，新代码使用 `supermimic.robots`。原检查点只保存模型状态及配置；移动 CLI 和资产模块不改变其字段。

训练 CLI 先解析 `--robot`、`--preset`、`--print-config`，完成 GPU/机器人选择后才导入训练配置和模型。其余选项由原有 tyro dataclass 配置解析。`--print-config` 不实例化物理环境；`smoke` 预设在 100 个环境步后正常保存退出。

## 验证

```bash
bash scripts/run_local.sh -m pytest tests -m "not slow" -q
bash scripts/run_local.sh scripts/training/train.py --robot g1 --preset smoke
```

测试覆盖本地模型与原模型的一致性、配置选择、入口导入无执行副作用、spawn worker 与串行转换的逐数组一致性、G1 的 SOMA/BONES CSV 格式，以及数据、FK、环境和有限训练更新。worker 测试使用临时生成的动作，不依赖本机旧 NPZ。需要数据或 GPU 的测试分别标为 `dataset`、`gpu`；世界模型长时间拟合测试使用 `slow` 标记。运行说明见 [tests/README.md](../tests/README.md)。

`scripts/dev/check_source_distribution.py` 从 Git 索引导出干净源码，重新准备 G1 资产并验证导入、入口和基础测试。提交前需先把新增源码加入索引。数据、机器人资产和历史检查点继续由本地目录管理。
