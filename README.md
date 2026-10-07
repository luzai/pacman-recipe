# Pacman Recipe

项目目录和 GitHub 仓库统一命名为 `pacman-recipe`。公共配方包为
`pacman_recipe`，环境包为 `pacman_env`；旧的 `areal_pacman` / `maapacman`
imports 和模块入口作为兼容层保留。AReaL 训练路径保留；slime 的独立视觉
adapter 与 CPU 验收已实现，真实 GPU 训练尚未验收。详见
[slime 迁移](docs/slime-migration.md)。迁移和 prompt 版本说明见
[名称迁移与 prompt 审计](docs/preparation-migration.md)。

当前 C2 使用 `edward-option-code-v2` prompt，动作协议仍为
`edward-option-code-v1`。旧 prompt 数据不能直接用于新训练：请重新准备数据并
验证 manifest，不要修改历史文件的哈希来绕过检查。

当前 primitive (`live_state_v3`) 与 options prompt 共用 `SHARED_GAME_RULES`：
最新截图、结构化状态优先、幽灵与通行规则、过关条件及首次死亡即结束本局的规则。
两套 prompt 均说明每次移动通常消耗 16 个游戏逻辑 ticks；判断可食用时间时应留余量，
并考虑幽灵也会移动。
Primitive 优先生存，再在安全的开放方向中收集普通豆子；仅保留走过的出口和上一步动作，
在方向同样安全且有助于吃豆时参考历史，允许反向或重复出口。两套输出协议保持各自独立。
这些文本变更会更新 prompt 指纹；新运行需要重新验证数据合约及真实 processor 的 token budget。

未来 VLM 运行使用 `fixed-image-dynamic-v2` 布局：system 规则，随后 user 中的
固定字段说明 → 当前截图 → 当前状态 → 候选目标 → 输出。ASCII 使用对应的地图、
状态、候选目标、输出四个分区；两者状态字段、候选顺序、动作代码及风险回退规则
保持一致，候选从状态 JSON 中拆出、每行一项。prompt 指纹已更新，新数据需重新
生成并验证合约。
布局纳入 VLM prompt 版本和指纹，旧运行应使用其原始源码快照。
后续 visual-grounding SFT/评估可共用
`pacman_recipe.level1.vision_prompt.grounding_messages(system, question, answer_format=...)`，
返回 processor chat-template 输入；图片由 processor 的 `images` 参数传入。
默认 grounding prompt 只读当前图像的可见事实，使用从左上角0开始的[row,column]
网格坐标，问题和输出格式单独分区；整图、crop、cells等题仍使用各自指定的JSON
答案格式，不加入战术候选或隐藏状态标签。`fixed_text`可显式指定数据集图例及约定。
训练和评估必须使用同一固定说明及图像预处理配置。已有运行的 SFT 源码与数据不修改。
Radix Cache 的收益需实测；RL 权重更新必须使旧 KV cache 失效。

下方固定 SHA 和 release 分支对应历史发布。新的名称与 prompt 改动只有在包含
本次迁移的提交中才可用；复现旧发布时使用该提交自带的 README 和入口。

本仓库提供 Pacman Level 1 的 headless 游戏环境和 AReaL 强化学习配方：

- `pacman_env`：真实 pygame Level 1 环境；
- `pacman_recipe`：数据集、原生多模态 workflow、Reward v3、轨迹审计和评估工具；
- `configs/level1/` 与 `scripts/level1/`：训练配置和运行入口。

固定 revision 的 AReaL fork 提供分布式调度、vLLM rollout、FSDP actor/reference engine 和 checkpoint 管理。环境已内置在本仓库，无需独立的 Pacman checkout。

C2 可显式启用[单步风险回退](docs/edward-risk-fallback.md)：正常 options 全部为空时，
展示所有开放方向及风险提示，让模型选择一步后重新规划。默认仍保留原有行为。

Simulator 支持 [JSON save-state / restore](docs/simulator-save-state.md)：在
`reset/step` 边界保存完整游戏、RNG、渲染和 episode 计数，可在新 worker 中继续。
接口为 `save_state()`、`restore_state(saved)` 或 `reset(saved_state=saved)`。

[Adaptive Backplay 实验](docs/adaptive-backplay.md) 将成功教师轨迹的 restart states
接入 primitive-action GRPO：先固定状态 pilot，再按当前 policy 的成功率选择更早状态。
独立实验入口不会替代现有 C1/C2 配方；训练效果以实际实验报告为准。

日常开发和交付统一使用 `release/maapacman-v0.1.0`。`backup/2026-09-04/*` 仅用于保留历史。

当前交付状态是 **source/recipe-only**：新两阶段方案正在集成验证，尚未完成对应的分布式 GPU smoke、完整训练及最终 C2 权重的独立运行验收，不能据此宣称最终 agent 已能通关。

2026-09-08 开发验收补充：C1 GC ON/OFF 各两次更新及 OFF checkpoint 的实际加载验证已完成；C2 和最终两阶段训练验收仍未完成。当前双机 A/B 实验使用固定 AReaL `98028b2bb565896383b5da0f7bf1857165d359ad` 和本仓库已记录文件哈希的开发快照，不能将下方已发布源码快照与此次开发验证混为一谈。正式发布时须同步固定 revision 与运行证据。

原生视觉 rollout 对无附加元数据的 RGB PNG 复用已有 base64 payload，避免发送前再次 PNG 编码；非 RGB 或带元数据的输入保留原 RGB 归一化编码路径。此优化不改变训练 processor、action mask、reward 或训练配置，也不实现跨请求图像缓存。等价性测试见 `tests/test_image_transport.py`；设置 `PACMAN_TEST_PROCESSOR_PATH` 为本地模型目录可额外核对真实 processor 的输入张量。编码耗时改善不代表整体训练同等加速。

## 源码边界

当前开发 C1/C2 YAML 已启用实验性视觉兼容适配，需要 AReaL
`release/pacman-v0.1.0` @ `9f93d1deb59c1c99cd7019afbd0e83bb40b62fc0`，并显式保持
actor/ref 的 `fsdp.memory_efficient_load=false`。旧 AReaL 发布快照不包含该适配器，
不能直接搭配当前开发 YAML；下面的获取命令已固定为上述 AReaL SHA，并须使用
匹配的 Transformers 5.7.0 / vLLM 0.22.1 环境。启动时会记录实际加载的视觉适配信息。
详见[视觉兼容说明](docs/vision-position-compatibility.md)。语言侧 log-prob mismatch
仍在排查，代码发布不代表完整两阶段训练或全批概率一致性已经验收。

游戏依赖仍固定为下面的 `cbb97115e407abc86a44adc82a1b8f360b3e8da0`；
`local/maapacman-v0.1.0-source` 是单独保存的旧源码分支，缺少显式 ghost-mode
接口，不能替代该依赖。运行 manifest 会记录实际三仓库 revision 和 dirty 状态。

| 源码层                                                        | 作用                                       | 当前配方使用的版本                                                         |
| ------------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------- |
| 本仓库                                                        | `pacman_recipe` 配方与内置 `pacman_env` 环境 | `release/maapacman-v0.1.0`；运行时记录实际 SHA                             |
| [luzai/AReaL](https://github.com/luzai/AReaL)                 | 训练、rollout、FSDP 和 checkpoint          | `release/pacman-v0.1.0` @ `9f93d1deb59c1c99cd7019afbd0e83bb40b62fc0`        |
| [luzai/pacman-python](https://github.com/luzai/pacman-python) | 游戏规则、资源和 pygame renderer           | `release/maapacman-v0.1.0` 中的 `cbb97115e407abc86a44adc82a1b8f360b3e8da0` |

复现时以固定 SHA 为准，不能仅依赖会继续更新的分支名。更新依赖 revision 后，需要同步运行 manifest 并重新验证。`pacman_recipe.synthetic.*` 保留用于历史合成迷宫实验。

## 获取固定源码

在 Linux 服务器上执行，替换示例路径：

```bash
export WORKSPACE_ROOT=/path/to/pacman-release
mkdir -p "$WORKSPACE_ROOT"
cd "$WORKSPACE_ROOT"

git clone --branch release/pacman-v0.1.0 --single-branch \
  https://github.com/luzai/AReaL.git
git -C AReaL checkout 9f93d1deb59c1c99cd7019afbd0e83bb40b62fc0

git clone --branch release/maapacman-v0.1.0 --single-branch \
  https://github.com/luzai/pacman-recipe.git

git clone --branch release/maapacman-v0.1.0 --single-branch \
  https://github.com/luzai/pacman-python.git
git -C pacman-python checkout cbb97115e407abc86a44adc82a1b8f360b3e8da0
```

三个 checkout 可以放在任意可写磁盘；以下命令使用并列目录：

```text
${WORKSPACE_ROOT}/
  AReaL/
  pacman-recipe/
  pacman-python/
```

正式运行前保存实际 revision：

```bash
git -C "$WORKSPACE_ROOT/AReaL" rev-parse HEAD
git -C "$WORKSPACE_ROOT/pacman-recipe" rev-parse HEAD
git -C "$WORKSPACE_ROOT/pacman-python" rev-parse HEAD
```

正式复现使用 Git clone + editable install；GitHub `Download ZIP` 不含数据 manifest 所需的 `.git` provenance。

## 环境安装

历史 8×H100 gate 的关键版本为 Python `3.12.13`、PyTorch `2.11.0+cu130`、Transformers `5.7.0`、vLLM `0.22.1`、pygame `2.6.1` 和 torch-memory-saver `0.0.9`。这些是历史环境记录，GPU wheel 仍需与目标节点的 driver/CUDA 匹配。

创建环境并注册两个本地 Python 项目：

```bash
export AREAL_ROOT="$WORKSPACE_ROOT/AReaL"
export PACMAN_RECIPE_ROOT="$WORKSPACE_ROOT/pacman-recipe"
export PACMAN_PYTHON_ROOT="$WORKSPACE_ROOT/pacman-python"
export ENV_ROOT=/path/to/conda/envs/maapacman-rl

conda create --prefix "$ENV_ROOT" python=3.12.13 pip -y
conda activate "$ENV_ROOT"
export PYTHON="$ENV_ROOT/bin/python"

# 按固定 AReaL checkout 的安装说明准备 vLLM/FSDP GPU 依赖。
"$PYTHON" -m pip install -e "$AREAL_ROOT"
"$PYTHON" -m pip install -e "${PACMAN_RECIPE_ROOT}[dev,dataset,agent]"
```

上面的 editable install 不替代 GPU 依赖安装。AReaL 兼容要求见[必需补丁说明](patches/README.md)。如果复用安装过独立 `pacman_env` 的旧环境，先卸载旧 distribution，再安装本仓库。`pacman-python` 只作为固定、只读的源码 checkout 使用。

## 环境检查

以下命令检查导入和真实 headless 游戏环境，不启动 GPU 训练：

```bash
cd "$PACMAN_RECIPE_ROOT"
export PYTHONPATH="$AREAL_ROOT:$PACMAN_RECIPE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy

"$PYTHON" - <<'PY'
from pathlib import Path

import areal
import pacman_recipe
import pacman_env
from pacman_env.env import PygamePacmanEnv

recipe_package = Path(pacman_recipe.__file__).resolve().parent
environment_package = Path(pacman_env.__file__).resolve().parent
assert recipe_package.parent == environment_package.parent
print("areal:", Path(areal.__file__).resolve())
print("pacman_recipe:", recipe_package)
print("pacman_env:", environment_package)

with PygamePacmanEnv() as env:
    image, info = env.reset(seed=0)
    assert info["env_id"] == "pacman-python-level1-ghostdoor-v3"
    assert info["backend"] == "original-pygame"
    assert image.shape == (400, 336, 3)
    print(info["env_id"], info["backend"], image.shape)
PY

CUDA_VISIBLE_DEVICES='' "$PYTHON" -m pytest -q
```

## 两份正式训练配置

| 配置                                                      | 初始化模型                  | 默认完整训练            |
| --------------------------------------------------------- | --------------------------- | ----------------------- |
| [curriculum1.yaml](configs/level1/train/curriculum1.yaml) | `Qwen/Qwen3.5-9B`           | 100 次 optimizer update |
| [curriculum2.yaml](configs/level1/train/curriculum2.yaml) | `Qwen/Qwen3.5-9B`           | 50 次 optimizer update  |

两阶段使用相同关卡、画面尺寸及共同观察字段，但动作协议、实际 prompt 与训练目标不同：

| 项目                     | Curriculum 1       | Curriculum 2           |
| ------------------------ | ------------------ | ---------------------- |
| `environment.ghost_mode` | `disabled`，无幽灵 | `normal`，正常移动幽灵 |
| 输入中的 `ghosts`        | `[]`               | 正常幽灵状态列表       |
| 每局底层 `env.step` 上限 | 512 | 512 |
| 动作 / harness | 单 token `U/D/L/R`，只允许当前可通行方向；不构造或调用 Edward | 单 token advertised option code，经 Edward 映射并执行 `C*/A*/E*` 候选 |
| `action_protocol` | `direct-open-action-token-v1` | `edward-option-code-v1` |
| `prompt_version` | `live-state-direct-action-v3` | `edward-option-code-v2` |
| `edward_options` | `false` | `true` |
| `action_token_choice` / `open_action_mask` | `true` / `true` | `false` / `false`，使用有效 option 候选 mask |
| reward objective | `step_local_raw_v1` | `episode_return_group_v1` |
| reward normalization / clip | `null` / `.inf` | 同状态 12 局 group mean/sample std / `20.0` |
| 训练 / 验证行数 | 80 / 4 | 80 / 4 |
| 训练 / 验证 seeds | 28–107 / 108–111 | 28–107 / 108–111 |
| Epochs / updates | 5 / 100 | 5 / 100 |
| 学习率 / `nearest_pellet_alpha` | `5e-7` / `0.1` | `5e-7` / `0.1` |

两者默认均为 8 张 GPU（4 rollout + 4 actor）、train/validation batch size 4，每个初始状态采样 12 局，即每次 update 48 局、每 epoch 20 updates。训练 RNG seed=1，与 dataset seed=28 区分。
C1 聚焦无幽灵导航与吃豆，C2 学习正常幽灵下的 Edward option 选择；两阶段都保留能量豆和通关规则。地图与初始位置不变，seeds 不是不同地图；这些是发布默认值，不代表已验证最优超参数。
独立 ghost 开关不关闭水果、不改变奖励事件定义；C1 的能量豆仍得分，但没有幽灵易受攻击计时。
启动器从 YAML 读取数据规模与步数；环境模式同时写入 dataset、run manifest 和每个原子帧，规则 hash 按模式区分。配置与数据不匹配时拒绝训练。
当前 dataset、split bundle 与 audit 的产物合约标识仍为 v4，但正式 bundle 新增必需的 `recipe_contract`、其 hash、逐行 action/prompt protocol 及三仓源码身份。C1 anchor 仅执行真实一步合法方向、没有 Edward；C2 anchor 保留真实候选。anchor 是环境审计证据，不是模型 rollout 或训练样本。旧 bundle 即使标记 v4 也必须重新生成，不要手补字段、复用旧规则 hash 或覆盖旧数据。CLI 的行数、seed、horizon 覆盖值必须与 YAML 一致；训练拒绝不匹配的 manifest、源码 hash、JSONL/HF 内容及非规范 bundle 路径。

两阶段都使用 `live_state_v3` 观察、`temperature=0.7`、`top_p=1.0`、单 token、thinking disabled；C1 不输出 `S` 或 JSON，C2 不输出方向或 JSON。C1 关闭训练内 validation；C2 在训练开始前验证一次，并以 `evaluator.freq_epochs=1` 在每个 epoch 结束后验证一次。`saver.freq_steps=1` 仍每 update 保存，保存 checkpoint 不等于已评估。此设置不追溯修改正在运行的旧配置；启用调度仍不保证框架不初始化 eval worker，也不等于 RAM OOM 已解决。

C1 是无 critic、逐步奖励的 PPO-style 更新：每次方向动作使用自己的 shaped reward，不做 group/advantage normalization，不广播整局回报，不跨游戏步做 GAE；保留原有 loss reduction。`.inf` 只是不截断有限的单步任务奖励，NaN/Inf reward 仍拒绝，JSON metadata 使用字符串 `"inf"`。C2 在同一初始状态的 12 局中先做整局回报归一化，再裁剪到 ±20；每局所有策略决策共享该局任务信号，整局 loss 等权，`option_return` 仅记录审计、不重复累加。两者保留 `ppo_n_minibatches=1`、KL=0.01 和 reference model（初始化跟随 actor），`critic/teacher/adv_norm=null`。

Reward v3：普通豆/能量豆各 +1、幽灵 +5、水果 0、通关 +50、死亡 -100、每底层步 -0.05、撞墙 -0.5，另加 alpha=0.1 的 nearest-pellet shaping。`use_base_reward=false`；C2 safety refusal 整局仅扣一次 -100，开局尚无模型决策即拒绝时不造训练样本，选择 AVOID 本身不扣分。保持既有格式错误 fail-closed 整局目标 -1。完整系数和审计规则以 [rewards.py](pacman_recipe/level1/rewards.py) 与两份 YAML 为准。

## 启动训练

使用上面的路径与已激活环境，在仓库根目录设置一次：

```bash
cd "$PACMAN_RECIPE_ROOT"
export OWNER_ROOT=/path/to/writable/owner-root
export ENV_ROOT="$CONDA_PREFIX"
export PYTHON="$ENV_ROOT/bin/python"
export MODEL_PATH=/path/to/Qwen3.5-9B
# 使用选中 YAML 的数据默认值，避免上次运行的环境变量覆盖本次配置。
unset TRAIN_EPISODES VALIDATION_EPISODES DATASET_MAX_STEPS

# 让启动器为每次运行自动生成新的名称、产物和数据集目录。
unset RUN_ID ARTIFACT_ROOT DATASET_OUTPUT_ROOT
```

先对 Curriculum 1 做两次更新的 smoke test：

```bash
CONFIG=configs/level1/train/curriculum1.yaml \
bash scripts/level1/train/run_level1_training.sh --smoke-updates 2
```

从同一个 Qwen base 独立验证 C2 两次更新：

```bash
MODEL_PATH=/path/to/Qwen3.5-9B \
CONFIG=configs/level1/train/curriculum2.yaml \
bash scripts/level1/train/run_level1_training.sh --smoke-updates 2
```

确认两阶段 smoke 结果后，从 Qwen3.5-9B 开始完整训练：

```bash
CONFIG=configs/level1/train/curriculum1.yaml \
bash scripts/level1/train/run_level1_training.sh
```

正式 C2 同样从 Qwen base 冷启动，不继承 C1 optimizer 或模型权重：

```bash
export MODEL_PATH=/path/to/Qwen3.5-9B
CONFIG=configs/level1/train/curriculum2.yaml \
bash scripts/level1/train/run_level1_training.sh
```

`MODEL_PATH` 必须指向完整的 Qwen3.5-9B base checkpoint，包含模型权重、配置和所需 tokenizer/processor 文件；不能直接指向训练产物根目录或不完整的恢复目录。启动器会离线校验这些文件。

上述检查只是结构预检，不等于完整模型加载或真实图像推理。C2 必须先完成从同一 Qwen base 启动的两次更新 smoke，再从未训练的同一 base 新开正式 50-update run；smoke 权重及 optimizer 状态不得继承到正式 run，也不能冒充最终 agent。

Curriculum 2 从 Qwen base 独立初始化模型、optimizer 和 scheduler。两份配置首次运行均为 `recover.mode: disabled`。
注意：在当前 AReaL 中，`disabled` 同时禁止保存恢复状态；中断后才改为 `auto`，无法补回之前未保存的 optimizer/dataloader 状态。
需要完整中断恢复的任务应在首次启动时就显式启用 `auto`，并保留同一 run 的数据、名称与路径；当前发布启动器面向新 run，不会覆盖或重建旧数据。
只有确实存在完整恢复状态时，才应使用原始训练命令加 `recover.mode=auto` 恢复；只有模型 checkpoint 时属于权重初始化新 run。

Smoke 使用同一份 YAML，通过 `total_train_steps=2` 限制更新次数，不需要第三份配置。也可以直接向 `train_areal.py --config ...` 传入 `--smoke-updates 2`，但需要先准备对应的数据集和运行环境。

### C1 reward A/B 对照

当前 C1 默认 `actor.gradient_checkpointing=false`，已在匹配的 8×H800、batch 4×12、512 步负载下完成两次更新验收；这不是任意负载都不会 OOM 的保证。C2 仍为 `true`，必须独立验收后再决定是否关闭。phase offload 保持开启，FSDP CPU parameter offload 保持关闭。

| 实验 | nearest-pellet shaping | 更新预算 |
| --- | --- | --- |
| A | 固定 alpha=0.1，`nearest_pellet_scale_by_cleared_ratio=false` | 4 |
| B | alpha=0.1 × 已清除普通豆比例，开关为 `true` | 100；前 4 次与 A 比较后继续 |

A、B 使用独立目录内的 `curriculum1.yaml` 副本，除上述开关外保持相同配置。两者从同一固定 base 独立初始化，不继承对方或 smoke 的模型/optimizer；使用相同 seeds、batch、环境、offload、GC 和 constant LR `5e-7`、zero warmup。YAML 保留完整 100-update 配方结构，A 的停止预算由显式 CLI 记录：

```bash
# A：该副本仅把 nearest_pellet_scale_by_cleared_ratio 改为 false。
CONFIG=/path/to/ab/a/curriculum1.yaml \
ARTIFACT_ROOT=/path/to/new-a-artifacts \
bash scripts/level1/train/run_level1_training.sh \
  --reward-ablation fixed-distance --smoke-updates 4

# B：使用默认比例缩放，不传 smoke cap；不要把 A 权重作为初始化。
CONFIG=/path/to/ab/b/curriculum1.yaml \
ARTIFACT_ROOT=/path/to/new-b-artifacts \
bash scripts/level1/train/run_level1_training.sh
```

两条命令在各自已配置的运行环境执行；若同机运行，必须等待所需 GPU 空闲。`fixed-distance` 只接受 C1、alpha=0.1、比例缩放关闭和 4-update cap，默认正式训练约束不变。每个 update 使用相同 validation seeds 108–111、各 12 次采样，比较普通/全部豆清除率、通关率、游戏分、步数/结束原因，以及 PPO loss、grad norm、KL、entropy、时间和显存。不同定义的 shaped reward 不能直接判优；单次实验不宣称统计显著性。先完成这项 C1 对照，再推进 C2。

启动器会检查源码导入、模型文件、GPU 空闲状态、AReaL 补丁和 prompt budget，生成不可变数据集并执行配置 dry-run，然后启动训练。默认产物位于 `${OWNER_ROOT}/run_artifacts/maapacman-rl/<recipe>-<timestamp>/`，数据集保存在其中的 `dataset/`。指定自定义输出路径时应使用新目录。

2026-09-08 的固定 A/B 开发快照在两台 Linux 服务器完整回归均为 **562 passed、1 skipped**，包括新增 A/B 入口及真实 processor 图像等价性检查；当前 README 文档补充不改变该已测代码。C1 smoke 和实际 checkpoint 加载已有验证，最终 C2 训练、选模及独立测试仍待完成。历史 2026-09-04 的 Windows 定向检查为 53 passed、2 deselected，不替代此次 Linux 结果，也不沿用旧实验的通关结论。

## 评估与产物

[评估工具目录](scripts/level1/evaluate/) 提供单模型评估、checkpoint 对比和结果汇总工具。`evaluate_level1.py --config` 读取对应 recipe 的 ghost/harness/reward/prompt/decoding；批量入口支持任意非零 checkpoint 数，也可通过 `CHECKPOINT_LIST` 选择子集。具体示例见[脚本说明](scripts/README.md)。旧权重录像与 C1 OFF checkpoint 单局已使用实际推理服务验证；这不等于最终 C2 权重及同事下载后的全链路验收。

独立评估默认采用 C2、held-out seeds 112–131、每 seed 3 个生成 seeds；正式执行时应先确认并冻结测试规模。`--purpose validation` 仅接受验证 seeds 108–111，选模不能使用 held-out 报告。每次调用预先保存 manifest，逐次尝试追加保存到 `.attempts.jsonl`，默认对基础设施失败最多重试两次且保留原错误。报告分别给出 `evaluation_completed`、`full_completions`、`win_rate`（包括失败尝试的分母）及 `planned_trial_win_rate`，两种分母不得混淆。

正式对比 Base、C1、C2 时，必须在同一协议内固定环境 revision、ghost/harness、seed、512 步和 decoding；不能直接把 C1/C2 原生训练回报横向比较。每局报告完成状态、terminal reason、通关率、清豆率、回报和失败；评估流程结束不等于观察到通关。

## 最终权重交付与验收

从 base 复现训练可以只交付固定源码、配置、模型 revision 和运行记录；若同事不重训而直接运行最终 agent，则必须另交付**完整 C2 推理 checkpoint + tokenizer/processor + 精确 Edward harness/config**。建议同时提供实际用于 C2 初始化的 C1 checkpoint；optimizer/scheduler 恢复状态是另一类制品，不属于推理必需文件。

完整 C2 权重计划作为独立大文件制品交付，Git 保留模型说明、manifest、逐文件校验和及下载入口。交付位置/访问范围、实测推理后端与硬件仍待确定；不要求读取其他用户目录，也不依赖 Pacman 独立仓库。必须记录 C2 run/update、C1 父模型 hash、base revision、三仓身份和导出参数，且无指向训练机的外部 symlink。

发布 checkpoint 不自动等于最后一步：按预先声明的规则，仅用 validation 108–111 在**实际留存的已训练候选**中选模并保留 last，报告候选 update 列表与规则。当前 `keep_last=2`、`keep_best_metric` 使用训练 reward，且 AReaL 先保存后评估；每 update 保存不代表留存全部 100 份，也不能宣称选出全程 validation 最优。默认最后一步在最近两份中；若需保护全程 validation 最优，仍须另实现验证后保护或规划全量保留，不能默增磁盘预算。之后固定独立测试集；目前建议 seeds 112–131 × 3 个固定生成 RNG seeds，共 60 局，具体规模尚待确认，不能用测试集反复调参。

验收依次要求：新位置的结构检查及完整模型加载；normal ghosts + Edward + 512 步的真实图像端到端运行；完整多局表现报告；从同事可访问位置重新下载后的 hash/加载/整局复验。模型加载成功或文件检查通过不能代替后续 gates。

评估记录的 checkpoint manifest 和服务模型别名还不能证明服务实际加载了该权重，当前明确记录 `weight_identity_verified=false`；须补充服务启动/加载日志及真实加载证据，不能凭别名把模型身份 gate 判为通过。

**先报告实际通关率，不设最低门槛。** 通关以真实 `terminal_reason=all_normal_pellets`、普通豆为 0 及对应 `level_cleared` 事件为依据，死亡、截断、安全拒绝或格式错误不算通关。报告全部尝试与基础设施错误，不静默删除失败；有真实通关再提供对应权重/hash/seed 的通关录像，0% 也如实报告“本测试集未观察到通关”，不自行扩大训练预算。当前尚无新方案最终权重的通关或下载复验结果。

## 运行约定

- 启动前确认所需 GPU 空闲；保留其他训练、推理和服务进程。
- 每次运行保存三仓 SHA、模型 revision、dataset manifest、实际配置及 driver/CUDA 信息。
- 根据日志中的 update 标记、checkpoint 和评估产物确认结果，不能仅凭启动器退出或进程存在判断完成。

## 详细文档

- [配置说明](configs/README.md)
- [配方架构与历史验证](docs/architecture/AREAL_RECIPE_DESIGN.md)
- [AReaL 必需补丁](patches/README.md)
- [数据集、训练、评估和报告脚本](scripts/README.md)
- [运行产物与保留策略](RUN_ARTIFACTS.md)
- [第三方来源与署名](THIRD_PARTY_NOTICES.md)

本仓库目前尚未选择项目级开源许可证；第三方来源和署名说明见 `THIRD_PARTY_NOTICES.md`。
