# AReaL Pacman RL Recipe 设计

状态：新两阶段 source/recipe 集成验证中；最终 C2 推理和通关验收尚未完成。
下文注明日期的证据是历史记录，不是新方案已通过的 gate。
环境提供方：本仓库内置的 `pacman_env.PygamePacmanEnv` API `3.0`
环境 ID：`pacman-python-level1-ghostdoor-v3`
Dataset 合约：`maapacman-level1-dataset-v4`
正式 C1/C2 dataset 默认上限：`512` 次底层 `env.step`
支持的 episode 上限：`32`、`256`、`512`、`2000` 个 action
通用 `PygamePacmanEnvConfig` 默认值：`512` 个 action；实际仍以不可变数据行为准
远端 Linux backend：SDL dummy
代码合约同步日期：`2026-09-04`
历史远端证据最后验证日期：`2026-07-23`

headless 环境实现位于本仓库的 `pacman_env/`，不再需要独立的 Pacman
checkout。

示例 checkout 布局：

```text
${WORKSPACE_ROOT}/AReaL
${WORKSPACE_ROOT}/pacman-recipe
${WORKSPACE_ROOT}/pacman-python
```

## 0. 当前两阶段发布合约

| 项目 | C1 | C2 |
| --- | --- | --- |
| 初始化 | Qwen3.5-9B | 完整 C1 权重，重置 optimizer/scheduler |
| 幽灵 / harness | disabled；数据、训练、原生评估均不调用 Edward | normal；Edward options |
| 输出协议 | 单个合法 U/D/L/R，`direct-open-action-token-v1` | 单个 advertised code，`edward-option-code-v1`，映射 C*/A*/E* |
| prompt version | `live-state-direct-action-v3` | `edward-option-code-v1` |
| objective / norm / clip | `step_local_raw_v1` / 无 / `.inf` | `episode_return_group_v1` / group-12 mean/sample std / 20 |

两阶段均为 512 次底层步、batch 4、每状态 12 局。C1 为 80/4 train/validation rows、
seeds 28–107/108–111、5 epochs = 100 updates；C2 为 40/4 rows、seeds 28–67/68–71、
5 epochs = 50 updates；每 update 48 局。训练 RNG seed=1 与 dataset seed=28 区分。学习率统一 5e-7，shaping alpha=0.1；保留 reference
KL=0.01、`ppo_n_minibatches=1`，无 critic/teacher/advantage normalization。
增加 seeds 不等于增加地图，无幽灵初始状态可能重复。

C1 每个方向只使用自己的 shaped step reward，不广播整局回报、不做 group
normalization 或跨游戏步 GAE，保留原有 loss reduction；准确说法是无 critic 的
逐步奖励 PPO-style 更新。C2 按同初始状态 12 局的完整回报先归一化再裁剪，将整局
任务信号分配给该局模型决策，整局 loss 等权；option_return 只审计、不重复相加。
使用样本标准差、`mean_leave1out=false`、`std_unbiased=true`、`eps=1e-5`。
C1 `.inf` 只是裁剪阈值，JSON 写为 `"inf"`；仍拒绝非有限 reward，保留 PPO/梯度裁剪。

当前交付为 source/recipe-only。下载即用还必须提供选定的完整 C2 权重、
tokenizer/processor、精确 Edward harness/config 和经过实测的推理后端。制品记录
逐文件 hash、C1 父权重、base revision、run/update、三仓身份及导出参数，不带指向
训练机的外部 symlink。建议同时提供实际 C1 父权重；optimizer 恢复状态是独立制品。

先按预先声明的 validation-only 规则在实际留存的已训练候选中选模，记录 update
列表并保留 last，再冻结独立测试集。当前 `keep_last=2` 与按训练 reward 的
`keep_best_metric` 在评估之前运行；每 update 保存不等于保留全部 100 份，也不能
宣称保护了全程 validation 最优。验证后保护或全量保留仍需单独实现和磁盘规划。当前建议
seeds 112–131 × 3 个固定生成 RNG seeds，共 60 局；这不是已完成结果。按 normal
ghosts + Edward + 512 步测试，记录所有尝试和错误，**报告实测胜率，不设最低门槛**。
0% 也如实说明；实际观察到通关才给通关录像，加载成功不等于通关。
完整模型加载、真实截图端到端运行、多局整局报告、交付位置重新下载复验是四个
独立 gate，均不能省略。

2026-09-04 当前工作树定向 CPU 验证：`test_dataset_stage_contract.py`、
`test_level1_v3_audits.py`、`test_curriculum_ghost_modes.py` 共 53 passed、2 deselected。
两项 workflow parse-failure 测试因 Windows 缺 `uvloop` 未纳入通过数。
该结果不代表 Linux full suite、分布式 GPU、最终权重推理或通关已通过。

## 1. 职责归属

```text
AReaL fork
  负责分布式训练、rollout worker 和 checkpoint 编排

pacman-recipe 仓库
  maapacman 包负责外部进程 wrapper、帧边界 action 协议、
  RGB Surface 提取和稳定的环境 API
  pacman_recipe 包负责 episode dataset、prompt、模型调用、解析、
  reward shaping、trajectory 记录、AReaL 配置和评估

pacman-python
  负责原版游戏规则、资源和 pygame renderer
```

`pacman-python` 是 sibling dependency，源代码必须保持干净。它不得 import
`pacman_env` 或 AReaL。

### 1.1 生产环境 API

生产 Level 1 文件只 import `PygamePacmanEnv`：

- `pacman_recipe/level1/workflow.py`
- `pacman_recipe/level1/level1_dataset.py`
- `train_areal.py` 的生产 dry-run 路径
- `scripts/level1/dataset/write_level1_manifest.py`
- `tests/test_level1_recipe.py`

唯一被接受的环境 import 是
`from pacman_env.env import PygamePacmanEnv, PygamePacmanEnvConfig`。recipe 不定义
也不暴露另一套 Pacman 环境合约。

## 2. Episode 架构

```text
episode row
  -> PacmanNativeVisionWorkflow（训练）/ PacmanImageOnlyWorkflow（独立评估）
  -> PygamePacmanEnv.reset()
  -> 原版 pacman-python 进程和 pygame Surface
  -> RGB observation
  -> AReaL multimodal rollout endpoint
  -> VLM completion
  -> C1: 受限 U/D/L/R -> 一次 PygamePacmanEnv.step(action)
     C2: 受限 option code -> Edward option -> 一次或多次 env.step
  -> 原版 score delta 和状态指标
  -> recipe reward adapter
  -> C1 逐步任务奖励 / C2 整局任务目标
  -> 重复执行，直到 terminated 或 truncated
```

环境是一个本地 Python 对象。OpenAI-compatible HTTP API 只存在于 AReaL 和模型
服务器之间；它不是游戏协议。

reset 后以及每个 action transaction 完成后，pygame worker 都会暂停在已返回的
`display.flip()` 边界，同时 AReaL 等待模型响应。模型推理期间，Pacman、ghost、
timer 和 animation 都不会推进。因此，模型延迟只改变 rollout 的墙钟时间，不会
产生隐藏的游戏帧。

## 3. 必需安装和 mirror

生产训练节点需要固定下面三层源码。`pacman-recipe` checkout 同时包含两个
Python 包，因此不需要独立 Pacman 仓库或安装：

```text
${AREAL_ROOT}
${PACMAN_RECIPE_ROOT}
${PACMAN_PYTHON_ROOT}
```

每个节点都应选择由当前使用者控制的持久化 `OWNER_ROOT`。它可以是指向数据盘的
符号链接，但必须在本机检查 ownership 和目标，不能照搬另一台机器的物理路径。

AReaL fork checkout 放在 deployable recipe mirror 外，并与 robotics 开发明确
隔离：

```text
${AREAL_ROOT}
  发布分支：release/pacman-v0.1.0
  固定 revision：ee872bae29152f4b553349385aced59abd1651ba
  origin：https://github.com/luzai/AReaL.git

${UNRELATED_AREAL_ROOT}
  本地分支：<unrelated-development-branch>
  用途：保留不相关的开发工作及原有 dirty state
```

AReaL fork 的物理路径应是每个节点本地的 `${AREAL_ROOT}`。`maapacman-rl` 的
editable 绑定和生产 launcher 必须从该工作树解析 `areal`。launcher 会把
`AREAL_ROOT` 放在 `PYTHONPATH` 最前面，并拒绝解析到其他目录的 import。不相关的
AReaL 开发 worktree 不是 Pacman 训练依赖。

应使用项目专用 Conda prefix，而不是系统 Python 或不相关的既有环境：

```bash
OWNER_ROOT="${OWNER_ROOT:?set an owner-controlled project root}"
CODE_ROOT="${CODE_ROOT:-$OWNER_ROOT/maapacman-stack}"
AREAL_ROOT="${AREAL_ROOT:-$OWNER_ROOT/AReaL}"
PACMAN_RECIPE_ROOT="${PACMAN_RECIPE_ROOT:-$CODE_ROOT/pacman-recipe}"
PACMAN_PYTHON_ROOT="${PACMAN_PYTHON_ROOT:-$CODE_ROOT/pacman-python}"
ENV_ROOT="${ENV_ROOT:-$OWNER_ROOT/.conda/envs/maapacman-rl}"
PYTHON="${PYTHON:-$ENV_ROOT/bin/python}"
export MAAPACMAN_PACMAN_PYTHON_ROOT="$PACMAN_PYTHON_ROOT"
: "${BASE_ENV:?set BASE_ENV to a compatible source Conda prefix}"

conda create -y -p "$ENV_ROOT" --clone "$BASE_ENV"

"$PYTHON" -m pip install "pygame==2.6.1"
"$PYTHON" -m pip install \
  -e "$AREAL_ROOT" \
  -e "$PACMAN_RECIPE_ROOT"
```

训练前固定并记录 AReaL、pacman-recipe 和 pacman-python 三个 Git revision。
内置 `pacman_env` 与 pacman-recipe 共用同一个 revision。rollout 期间将
`pacman-python` mirror 视为只读。每个 worker 的副本、`agent_state.json`、
pygame 进程和 IPC 都是 `/tmp` 下的可丢弃内容。

历史 H100 验证采用 `/tmp + pip --target`，以便在不触碰持久环境的情况下删除。
该方法证明了运行时兼容性，但不是生产安装 recipe。

一个不相关的旧 `pacman_gym` Conda 环境曾经过审计但未被修改。它当时使用
Python `3.11.15`，且没有 pygame 和内置的 `pacman_env` 模块，因此不是被接受的
recipe 环境。

launcher 设置 `SDL_VIDEODRIVER=dummy` 和 `SDL_AUDIODRIVER=dummy`。Xvfb 不属于
recipe runtime 或部署依赖。注明日期的 API-v1 证据记录了 node1 和 node5 使用
SDL dummy 通过原版 pygame worker 和 oracle gate；这些结果确定了显示方案，但不
属于当前 API-v3 验收结果。

## 4. 环境构造

```python
import os

from pacman_env.env import PygamePacmanEnv, PygamePacmanEnvConfig

env = PygamePacmanEnv(
    PygamePacmanEnvConfig(
        pacman_python_root=os.environ["MAAPACMAN_PACMAN_PYTHON_ROOT"],
        level=1,
        max_steps=512,
        ghost_mode="normal",  # C2；C1 显式选择 "disabled"。
        video_driver="dummy",
        audio_driver="dummy",
    )
)
```

正式两阶段与数据行生成都默认 512 次底层步。通用/历史 API-v3 数据行仍允许
`32`、`256`、`512`、`2000`；workflow 采用不可变数据行中的值，正式 recipe 校验
拒绝不匹配的 horizon。512 不代表 C2 的模型调用或 option 选择次数。

workflow 在 rollout 前进行验证：

```python
if env.spec.api_version != "3.0":
    raise RuntimeError("unsupported original-pygame environment API")
if env.spec.env_id != "pacman-python-level1-ghostdoor-v3":
    raise RuntimeError("wrong Pacman environment")
if env.spec.action_tokens != ("U", "D", "L", "R", "S"):
    raise RuntimeError("incompatible action contract")
if env.config.max_steps not in {32, 256, 512, 2000}:
    raise RuntimeError("unsupported level-1 episode cap")
```

它只能 import 公开的 `pacman_env.env` API，不能 import worker module。

## 5. Dataset 合约

下面是一条构造完整原版游戏 episode 的精简数据行：

```json
{
  "id": "level1-normal-seed28-train-0001",
  "split": "train",
  "dataset_contract_version": "maapacman-level1-dataset-v4",
  "action_protocol": "edward-option-code-v1",
  "prompt_version": "edward-option-code-v1",
  "env": {
    "ghost_mode": "normal",
    "name": "pacman-python-level1-ghostdoor-v3",
    "api_version": "3.0",
    "backend": "original-pygame",
    "pacman_python_revision": "cbb97115e407abc86a44adc82a1b8f360b3e8da0",
    "level_revision": "36116c17c6c0805fdb1a07216357ac64c88d2c3108a0e37dce2a01b4ea2a8b97",
    "level": 1,
    "seed": 28,
    "max_steps": 512,
    "observation_mode": "rgb"
  }
}
```

不要直接复用标记为 `maapacman-level1-v1` 或 API `1.0` 的历史数据行，必须先
检查其 `env_id`：这些行描述的是旧版 AReaL 私有环境，不是当前原版 pygame
API-v3 wrapper。当前数据行必须使用 dataset 合约
`maapacman-level1-dataset-v4`、环境 API `3.0` 和环境 ID
`pacman-python-level1-ghostdoor-v3`。

Dataset v4 要求 `env.ghost_mode`。`disabled` 数据不暴露幽灵，`normal` 数据
暴露四只正常幽灵。该模式属于 ruleset revision，必须与所选训练 recipe、运行时
状态、audit anchor 和 trajectory evidence 一致。

上例仅作精简说明，不是可直接验收的完整数据行。正式 bundle 还绑定三仓 provenance、
recipe/prompt/reward metadata 与 hash、真实一步 anchor、JSONL/HF 内容及 manifest
校验 sidecar。C1 anchor 在当前合法 U/D/L/R 中选第一个方向，不构造 Edward，也
不声称使用 planner provenance；C2 保留候选及被选 option。anchor 不是模型 rollout
或训练样本。旧 v4 bundle 缺少新字段也必须重生成，不手补、不覆盖；保持规范路径，
CLI 的行数/seed/horizon 必须匹配 YAML。

生产 Level 1 dataset 验证只接受 `env.max_steps` 为 `32`、`256`、`512` 或
`2000`；数据行生成默认采用 `512`。所选上限属于不可变数据行合约，因此不兼容
horizon 不得混入同一训练/评估 split。若 terminal transition 恰好落在上限，
`terminated=True` 优先，该成功 action 不得同时报告为 `truncated=True`。

## 6. Observation 与 action 协议

每轮模型调用包含：

1. 一条固定 system prompt。
2. 恰好一张由当前 `(400,336,3)` RGB Surface 编码的 PNG。
3. 对应阶段的 live-state 上下文和单 token 指令；C2 还包含当次 Edward 候选与编码映射。

C1 只允许当前可通行的 `U/D/L/R`，不输出 S 或 JSON；C2 只允许当次 advertised
option code，不输出方向或 JSON，由 harness 执行映射的 option。两类 mask 都是
动态的。保留格式错误 fail-closed 整局目标 -1，不把任意文本发送给环境，也不能用
自动 planner 替模型选择来掩盖失败。下述环境通用 S 操作不是正式 C1 的模型输出。

一个环境 step 是一笔在原版 pygame 帧边界结束的 action transaction。方向输入
在已提交的 Level 1 游戏中移动一个网格；在该网格移动完成前，transaction 可能
运行多个内部帧。完成条件由游戏状态和对应的 `display.flip()` 检测，绝不依赖
固定 sleep。`S` 不发送方向事件，只推进一个原版帧。

reset 后以及 transaction 完成后，worker 都会阻塞在包装后的 `display.flip()`
内部。只有 AReaL 发送下一个 action 时才恢复。这一暂停是环境合约的一部分：
无论模型调用耗时 `50 ms` 还是 `5 s`，action 都会应用到同一个已返回状态。

每个 rollout worker 独占唯一的临时运行目录、原版脚本副本、资源链接/私有副本、
pygame 进程、IPC channel、状态文件和 Surface。worker 不拥有 X server 或
`DISPLAY`；pygame 通过 SDL dummy 渲染，Pacman 直接读取完成后的 Surface。

### 当前 IPC 合约与历史 timing 证据

当前实现通过 subprocess pipe 使用 newline-delimited JSON。action 和 state 使用
小消息。每张 RGB 帧从 pygame Surface 复制，使用 zlib level 1 压缩，编码成
base64 JSON，再由父进程解码。目前没有使用跨进程 shared memory。

暂停的 `flip()` 恢复后，未修改的原版循环仍会执行 `clock.tick(60)`。
`tick(60)` 测量距上次调用以来的墙钟时间，其中包括在 `flip()` 内暂停等待模型
的时间。只有该总时间小于约 `16.67 ms` 时，它才会等待；不会在较慢的模型调用
结束后额外增加一个 `16.67 ms`。

下面是历史 API-v1、287-action 环境使用真实 `L/R` 移动和当时 renderer 得到的
本地 profiling 结果：

```text
1 worker env.step:                p50 16.81 ms, p95 18.58 ms
16 worker env.step:               p50 40.52 ms, p95 61.80 ms
zlib+base64+JSON frame roundtrip: p50  1.38 ms, p95  2.07 ms
shared-memory two-copy estimate:  p50  0.024 ms, p95 0.037 ms
small pipe notification:          p50  0.053 ms, p95 0.091 ms
1 worker, simulated 50 ms model wait:
  action to observation:          p50 11.80 ms, p95 14.98 ms
  complete model+environment turn:p50 62.13 ms, p95 65.18 ms
16 workers, simulated 50 ms model wait:
  action to observation:          p50 14.52 ms, p95 21.09 ms
  complete model+environment turn:p50 64.88 ms, p95 71.49 ms
```

即时 action 的 `16.81 ms` 结果对应当时受限速器约束的脚本 agent 场景，不能直接
与 `50 ms` 模型调用相加。在模拟 `50 ms` 等待时，历史单 worker 每个完整 turn
的 p50 为 `62.13 ms`，该次 287-action 运行约需 `17.83 s`。对应的本地
16-worker p50 为 `64.88 ms`，每个 worker episode 约需 `18.62 s`。这些是历史
本地测量，不是当前 H100 保证值。

在该历史 profiling 中，RGB codec 在 287 帧中约占 `0.40 s` 串行 CPU 工作；
shared-memory payload 当时估计每个 worker episode 可节省约 `0.38 s`。仍然适用
的一般结论是：当模型延迟已经超过 `16.67 ms` 帧预算时，`clock.tick(60)` 几乎
没有可消除的 sleep；它只会成为更快脚本或低延迟 policy 的瓶颈。

因此，第一版可执行 recipe 同时保留 pipe IPC 和原版 clock 行为。由选定 rollout
并发度下使用真实模型的 H100 profiling 决定是否值得采用任一优化。未来可为快于
帧预算的 policy 在训练专用 runtime 中绕过 `clock.tick(60)`，但只有在脚本序列
和完整 287-action oracle 的 position、score、collectible count、terminal state、
`logic_frames` 和 RGB hash 全部一致时才能接受。

后续的混合协议可以保留 pipe 来传输 action、state、request ID 和 frame sequence，
同时将每个 worker 的 RGB buffer 放入 shared memory。仅当所选 rollout 并发度下
的 H100 profiling 表明帧序列化导致 CPU 饱和时才需要这样做；它不阻塞第一次
训练 gate。

## 7. Reward 归属

Pacman 返回原版游戏的 score delta：

| 原版事件 | Base reward |
|---|---:|
| 空移动、撞墙或 `S` | `0` |
| 普通豆 | `10` |
| 大力丸 | `100` |

共同 event-reward-v3 使用 `use_base_reward=false`：普通豆/能量豆各 +1、幽灵 +5、
水果 0、通关 +50、死亡 -100、实际步 -0.05、撞墙 -0.5，加上 alpha=0.1、threshold=1、
按清豆率缩放、吃豆时跳过的 nearest-pellet shaping。原始分数仍记录审计，不再加入
训练回报。Edward safety refusal 整局/最后一次决策仅扣一次 100；开局没有模型决策
就拒绝时不造样本，选择 AVOID 本身不扣分。C1 无幽灵和 Edward，相应事件自然不出现。

每条 step 记录同时保存 `base_reward` 和 `shaped_reward`。

当普通豆数量归零时，已提交的原版游戏进入 mode `6`；大力丸不属于其内部过关
计数器。因此，对该 revision 而言，terminal reason 是 `all_normal_pellets`。
recipe 不得将其重新解释为 `all_pellets`。

## 8. Trajectory 来源追踪

每条 trajectory 至少记录：

```text
env_api_version
env_id
backend
pacman_python_revision
level_revision
renderer_revision
seed and max_steps
ghost_mode and three-repository provenance
action_protocol, prompt_version, template and actual prompt hashes
reward objective, reward clipping/normalization contract
RGB frame hashes where requested
action and parse status
base_reward and shaped_reward
score and collectible counts
pygame_mode
terminated, truncated and terminal_reason
```

使用不同 `pacman_python_revision` 生成的训练输出不得假装使用同一个环境而合并。

## 9. 当前可执行与评测合约

生产 workflow 直接构造 `PygamePacmanEnv`，并强制验证
API `3.0`、环境 ID `pacman-python-level1-ghostdoor-v3` 和 dataset 合约
`maapacman-level1-dataset-v4`。episode 上限取自已经验证的数据行；数据行生成默认
采用 `512`，通用数据行支持 `32`、`256`、`512`、`2000`。环境类默认值不覆盖
不可变数据行和所选正式 recipe 合约。

旧的 `107` 项测试、`13` 项 subtest 和 287-action oracle 属于 API-v1 历史证据，
不是当前 API-v3 验收结果；第 10-12 节仅为历史留档而保留这些实验。

H100 recipe 固定使用 8 张卡：

```text
actor:   fsdp:d4p1t1  -> GPU 0-3 上四个 actor worker
rollout: vllm:d4p1t1  -> GPU 4-7 上四个 rollout worker
```

launcher 显式设置固定的 `MAAPACMAN_PACMAN_PYTHON_ROOT`，将
`maapacman-rl/bin` 放在 `PATH` 第一位；GPU 被占用时拒绝启动，不会抢占无关进程。

每个 rollout 都有随机 `trajectory_sample_id`，文件名是：

```text
<dataset-row-id>--sample-<trajectory-sample-id>.json
```

文件用 exclusive create 写入，所以同一 dataset row 的 rollout 样本不会互相
覆盖。此前 group-12 实验每个 row 只留下最后一个样本，因此旧的两个 validation
文件不能解释为 24-sample mean。

训练与评测 decoding 完全分开：

```text
训练 rollout: sampled12，temperature 0.7，top_p 1.0，单 token
每次 update 后 validation: freq_steps=1，sampled12，temperature 0.7，top_p 1.0
保存 checkpoint: 独立的 freq_steps=1
训练后评估: 显式选择 recipe/config，sampled 与 greedy 分别报告
```

`PacmanImageOnlyWorkflow` 默认 `enable_thinking=false`，并拒绝
`enable_thinking=true`；请求中会显式发送
`chat_template_kwargs.enable_thinking=false`。trajectory 记录 prompt style、
decoding 合约、请求 body、模型原始回答和 reasoning 内容。sampled 和 greedy
两套 validation 都强制关闭 thinking；只要发现 reasoning 内容，报告生成就失败。

评估必须读取所选 recipe 的 ghost/harness/reward/prompt。Base/C1/C2 公平比较要
固定同一评估协议，不能直接比较不同阶段原生回报。真实通关要求
`terminal_reason=all_normal_pellets`、普通豆为零与对应 clear event，不能用高奖励
替代；流程完成与游戏通关分别报告，不设最低胜率门槛。

两份 YAML 均通过 `--smoke-updates 2` 复用；C2 smoke 必须使用真实完整的 C1 smoke
checkpoint。正式预算仍是每阶段 100 updates。初次 `recover.mode=disabled` 在当前
AReaL 同时关闭恢复状态保存，中断后改 auto 不能找回未保存的 optimizer；如需完整
恢复，应从首次运行就启用。最终完整模型加载、GPU evaluation 产物及下载复验仍待
验证，源码交付不等于完整可直接运行的已训练 agent。

## 10. 历史修正后的 group-12 证据

历史实验：

```text
${ARTIFACT_ROOT}/level1-group12-20260723b
```

该 API-v1、287-action 实验早于当前 API-v3 合约。原 post-training test 没有关闭
thinking，in-training validation 也误用了
temperature `1.0`。修正后的 evaluator 为 base、epoch0、epoch1、epoch2、
epoch3 和 final 重建完整 VLM checkpoint；每个 checkpoint 跑一个真实 greedy
episode，并为 final 另跑 24 个匹配 sampled episode。全部使用 seed `0`、
287 steps、原版 pygame RGB，并关闭 thinking。

证据：

```text
.../corrected_eval_20260723/comparison.json
```

| Policy | Decode | Score | 普通豆清除率 | 撞墙 | Actions |
|---|---|---:|---:|---:|---|
| base | greedy | 0 | 0.00% | 287 | `U x287` |
| epoch0 | greedy | 20 | 1.02% | 283 | `L x287` |
| epoch1-3 | greedy | 20 | 1.02% | 283 | `L x287` |
| final | greedy | 20 | 1.02% | 283 | `L x287` |
| final | sampled，24 episodes | 平均 20 | 平均 1.02% | 平均 283 | `L x6888` |

全部修正后的 episode 都是零 reasoning turn。结论是：第一次 update 就发生
single-action collapse，后续 update 没有恢复。

## 11. 历史静态 image-only prompt A/B

A/B 比较两种 prompt：

- `minimal_v1`：原始简短 image-only prompt；
- `live_static_v2`：只借用 live demo 中静态视觉提示、屏幕绝对方向、优先吃附近豆
  和避免蓝墙的规则。

`live_static_v2` 不包含坐标、豆数、legal-action set、OPEN/BLOCKED 方向、路线
hint、cell history 或 live-controller action veto。

base 和 collapsed-final policy 对每种 prompt 分别跑一个 greedy episode和
12 个 sampled episode。证据：

```text
.../prompt_ab_20260723b/comparison.json
```

live prompt 让 base greedy 从 score `0` 变成 `20`，但训练所关心的 sampled
分布更差：

| Base sampled policy | 平均 score | 普通豆清除率 | 平均撞墙 |
|---|---:|---:|---:|
| `minimal_v1` | 389.17 | 15.65% | 130.50 |
| `live_static_v2` | 357.50 | 14.80% | 151.75 |

两种 prompt 下，collapsed final 都是每一步输出 `L`、score `20`。因此下一轮
训练冻结 `minimal_v1`；`live_static_v2` 保留为已否决的 ablation。

## 12. 历史 Anti-collapse gate 与条件分支

历史 API-v1、287-action 配置：

```text
configs/level1/archive/level1_image_anticollapse_4update_group12_8gpu.yaml
```

从原始 Qwen3.5-9B 重新开始：

```text
max_steps = 287
group size = 12
rollout + actor workers = 4 + 4
reference = BF16，与 actor 同卡并启用原生 FSDP CPU 参数 offload
optimizer updates = 严格 4 次
rollout temperature = 0.7
learning rate = 1.5e-6
KL coefficient = 0.01
actor 参数存储 = FP32 master weights
冻结 reference 参数存储 = BF16，常驻 GPU
prompt = minimal_v1
validation = 每次 update 后 greedy temperature 0
reward = 保持 score_delta - step_penalty - wall_penalty
```

这一 gate 不启用 progress shaping，也不提供 Oracle action。验收条件是 sampled
policy 保持探索性，最佳修正 greedy checkpoint 明显改善，并且没有 parse failure
或 reasoning 内容。

第一次 `20260723a` 启动完成了 rollout 和 PPO 计算，但在第一次 xccl 权重同步时
OOM：新启用的 reference engine 和 actor 只剩 `1.63 GiB`，FSDP full-tensor
all-gather 还需要 `3.79 GiB`。`20260723b` 随后测试 reference offload，
但 AReaL 嵌套 `stdbuf` wrapper 让 TMS preload 字符串失效；recipe 自带的
`sitecustomize.py` 已解决这个解析问题。`20260723c` 完成了第一批全部
`48` 条 rollout 和首次 reference offload，但 TMS 在恢复大型、同卡的
FSDP reference 时以 `CUDA error: invalid argument` 失败。

所以正式 gate 不再采用 TMS。`20260723d` 让同卡 reference 以 BF16
常驻后，已经完成全部 `48` 条 rollout、ref-logp、PPO、xccl 权重同步，
并写出了完整的 `17.9 GB` checkpoint；但 actor + reference 峰值仍约为
`79.75 / 81.56 GiB`，异步显存分配错误最终在 checkpoint 的
`torch.cuda.synchronize()` 暴露，validation 尚未开始。

隔离实验 `20260723e` 使用了 `3 rollout + 4 actor + 1 独立 reference`。
它证明了物理隔离有效，但当前 AReaL controller 只产出三个 dataset group，
随后一直等待第四个，停在 `36/48` trajectories；这里每个同步 consumer
item 实际需要一个 rollout worker。

`20260723f` 曾尝试保持 `4 + 4` 并让 reference target rollout，但 AReaL
在 rollout 之前初始化 reference，因此以 `WorkerNotFoundError` 拒绝；
这里不修改上游初始化顺序。

所以正式八卡拓扑保持 `4 rollout + 4 actor`。BF16 reference 仍与 actor
同卡，但启用 FSDP2 原生 `fsdp.offload_params: true`：reference 不执行
前向时，冻结参数 shard 留在 CPU；FSDP layer 执行时再流入 GPU。这不同于
AReaL 的 TMS engine `offload`，所以全局 `enable_offload` 和
`ref.offload` 都仍为 false。actor 保持 `optimizer_dtype: float32`，
reference 使用 `optimizer_dtype: bfloat16`。这样既保留四个同步 rollout
group 与四路 actor sharding，又在 PPO、xccl、checkpoint 同步阶段释放
reference 参数显存。TMS smoke 脚本和兼容层只保留为失败实验的诊断证据，
正式配置不会启用它们。

最终接受的拓扑已在下面这个 run 完整跑通：

```text
${ARTIFACT_ROOT}/level1-anticollapse-4update-20260723h
```

它完成 4/4 optimizer update，保存四个完整 `17.9 GB` checkpoint，并落盘
200 条唯一 trajectory：192 条 sampled train 加 8 条历史 greedy validation。
原生 FSDP reference offload 将 reference 静态显存从约 `19.9 GiB` 降到
`2.3 GiB`；后续 PPO 约稳定在 `72.22 / 79.19 GiB`，没有持续增长或 OOM。

但修正后的匹配评测没有通过质量 gate：

| Policy | Decode | 平均 score | 普通豆清除率 | 平均撞墙 |
|---|---|---:|---:|---:|
| base | greedy1 | 0 | 0.00% | 287.00 |
| update02 | greedy1 | 20 | 1.02% | 283.00 |
| base | sampled12，`0.7/0.95` | 248.33 | 9.99% | 165.42 |
| update02 | sampled12，`0.7/0.95` | 213.33 | 8.97% | 180.42 |

全部修正 episode 都关闭 thinking，reasoning turn 和 parse failure 都是 0。
greedy 之所以选 update02，只是因为它 287 步全部重复 `R`；在与训练一致的
sampled 分布下，update02 反而比 base 更差。因此 sparse recipe 不应直接扩大。

下一个隔离变量是：

```text
alpha * (
  nearest_normal_pellet_distance_before
  - nearest_normal_pellet_distance_after
)
alpha = 1
```

距离用内置 `pacman_env` 包维护的 level 表示做 BFS；每个距离和 reward term
都必须落盘并独立复算，`pacman-python` 仍然不修改。

这个 follow-up 已实现为
`configs/level1/archive/level1_image_progress_4update_group12_8gpu.yaml`。除可审计的
alpha-1 reward term 外，它明确采用双 validation 合约：

```text
训练中 sampled validation:
  n_samples = 12
  temperature = 0.7
  top_p = 0.95
  enable_thinking = false

训练后逐 checkpoint 报告:
  sampled12 = 同样的 0.7 / 0.95
  greedy1 = temperature 0 / top_p 1
  两者 enable_thinking 都是 false
```

checkpoint 主选择指标使用与训练分布一致的 sampled validation；greedy 保留为
独立的 collapse/确定性诊断。最终报告分别给出 `best_sampled_label` 和
`best_greedy_label`，不会把两套结果混成一个数字。

当时记录的 fallback 计划是使用 `64`、`128` 和当时完整的 `287` horizon 做 Oracle
SFT curriculum，再回到 RL。这些数值属于历史实验计划，不是当前 API-v3 的
supported-cap 合约。

## 13. 历史官方 main 迁移 gate

2026-07-23，两台远端节点都建立了干净的官方 AReaL 工作树，commit 为
`4d7ee11479d61ebe6c6f020e2bdcda5d76c6a76b`；原 checkout 及全部
Morgan/robotics 变更原样保留在 `robotics/morgan-vla`。

node5 的八卡诊断使用
`configs/level1/archive/level1_official_areal_smoke_3b_4gpu.yaml` 和 Qwen2.5-VL-3B。它证明
官方 scheduler、4 个 vLLM worker、4 个 actor worker，以及真实 SDL-dummy
`PygamePacmanEnv` 截图链路都能初始化并生成有效游戏轨迹。随后在第一次
optimizer update 之前失败：

```text
ref.compute_logp
  -> FSDPEngine._prepare_mb_list
  -> KeyError: 'mm_token_type_ids'
```

这是接口不匹配，不是 OOM，也不是 Pygame 失败。官方 OpenAI proxy 会保留
token ID、log-prob、version、mask 和 reward，但
`InteractionWithTokenLogpReward.to_tensor_dict()` 没有保留图像 tensor 和
Qwen-VL 的 `mm_token_type_ids`；官方 Qwen-VL FSDP 正确地要求这两项。日志中
缺少 Megatron 的 vLLM `awex_adapter` 警告只是可选 plugin 噪声，四个推理服务
均已 ready 并实际处理 RGB 请求。

不能修改官方 AReaL 工作树，也不能静默退化成 text-only 训练。生产迁移必须在
recipe 层改为原生 multimodal `RolloutWorkflow`，返回官方 tensor 合约：

```text
input_ids
attention_mask
loss_mask
logprobs
versions
rewards
mm_token_type_ids
multi_modal_input[pixel_values, image_grid_thw]
```

截至这份 2026-07-23 记录结束时，该原生 workflow 尚未完成真实 reference-logp、
actor update、checkpoint save/reload 和固定评估。因此，第 10-12 节只能视为
AReaL-VLA stack 的历史结果，不能作为当前 API-v3 已兼容训练的证明。

## 附录 A：已过期的 2026-07-22 快照

下面保留的是历史证据。它的 test count、六卡 topology、trajectory 文件覆盖行为
和早期结论已经被第 4、5、9 节的当前合约取代。第 10-13 节同样保留注明日期的
后续证据，而不是当前 API-v3 验收结果。

### 历史实现状态

在该已过期快照中，可执行 Level 1 recipe 直接构造了 `PygamePacmanEnv`，验证了
API `1.0` 和 `pacman-python-level1-pygame-v1`，记录了普通豆和大力丸数量，并使用
原版游戏的 `all_normal_pellets` terminal reason。当时本地 pytest 已通过 `98`
项测试和 `11` 项 subtest，其中包括一次脚本化的历史 287-step 过关。

当时的远端 dataset row 被要求记录该历史环境 ID 和源码 revision；当时记录的
GPU 训练前待办还包括 rollout-launcher 取消测试、可选 no-wait clock policy gate，
以及继承 AReaL 依赖的协调。这些不是当前 API-v3 待办清单。

### 历史验收 gate

#### 本地 gate —— 已完成

- `pacman-python` 源 checkout 当时在 commit `d258122e...` 上保持干净。
- Pacman wrapper 当时使用原版 pygame Surface。
- Windows 原生和 SDL dummy 的 reset、`L,L,L,S` hash 当时完全一致。
- `PygamePacmanEnv` 测试当时通过 `6/6`，其中包括四个并发 worker。
- 完整 Pacman unittest suite 当时通过 `23/23`。
- 完整 pacman-recipe pytest suite 当时通过 `98` 项测试和 `11` 项 subtest。

#### Linux 显示 gate —— 已在一台 8×H100 测试节点完成

- 实际 pygame driver：SDL dummy；pygame `2.6.1`；SDL `2.28.4`。
- 三次重复 `L,L,L,S` 均与 Windows 原始 RGB hash 完全一致。
- 4-worker 和 16-worker 并发测试通过，且 runtime ID 唯一。
- 只读源资源并发测试通过。
- 287-step 原版游戏 oracle 与 Windows 状态及 RGB hash 一致。
- 没有遗留 worker 进程或临时 worker 目录。

Xvfb 当时未安装、未使用，也不属于该历史 recipe。

#### 历史 Recipe CPU gate

- node5 mirror 和专用 Conda prefix 当时已经存在。
- 在该持久环境中，直接 PygamePacmanEnv 测试、RGB parity、4/16-worker 隔离
  和完整 oracle 均已通过。
- 该历史 node1 gate 使用旧的三个 editable package 布局。当前 release 已将
  `pacman_env` 合并到 `pacman-recipe` checkout；这里保留其直接 pygame gate 以及
  已部署 recipe suite 通过 `98` 项测试和 `11` 项 subtest 的历史记录。
- node1 当时使用 node-local AReaL 开发 checkout，base 为 `da645a37...`，有
  26 个 dirty entry；node5 使用自己的物理 checkout。这些路径当时有意保持
  节点本地化，而不是共享。

当时尚待完成或协调的事项记录为：

- 部署当时迁移后的 AReaL workflow，并为
  `pacman-python-level1-pygame-v1` 重新生成远端 dataset row。
- 在当时的数据行和 workflow 构造中强制要求 `max_steps=287`。
- 重放当时完整的 287-action oracle，要求 `terminated=True`、`truncated=False`、
  `terminal_reason=all_normal_pellets`。
- 验证当时的模型请求、action 解析、reward 映射和清理。
- 使用原版 clock 行为记录真实模型 1/4/8/16-worker H100 timing；只有当该 profile
  证明 no-wait policy 有必要时，才比较两种 clock policy 的精确 state/RGB parity。
- 协调或明确隔离当时继承自 `areal-vla` 的依赖冲突。node1 使用 torch `2.11.0`
  和 transformers `5.7.0`；AReaL 声明 torch `<2.11`、transformers `<=5.3.0`。
- 替换或明确快照化当时继承的 node-local editable AReaL 源码。它基于 commit
  `78d1e50f...`，当时有 `1112` 个 dirty entry，因此不能作为当时已接受的训练
  来源。

#### 历史 GPU gate —— 当时未完成

当时的计划是在 Linux 和 CPU gate 都通过后再：

1. 运行冻结的 no-training baseline。
2. 运行 two-epoch overfit 实验。
3. 构建完整 checkpoint。
4. 重新评估完全相同的冻结 suite。
5. 比较 completion rate、score、pellet progress 和 wall rate。
