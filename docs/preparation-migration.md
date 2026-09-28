# 名称迁移与 prompt 审计

## 新入口

仓库：`luzai/pacman-recipe`。本地目录：`life/pacman-recipe`。
Python distribution：`pacman-recipe`。

| 原入口 | 当前入口 |
| --- | --- |
| `areal_pacman` | `pacman_recipe` |
| `maapacman` | `pacman_env` |
| `AREAL_PACMAN_ROOT` | `PACMAN_RECIPE_ROOT` |
| `MAAPACMAN_PACMAN_ROOT` / `MAAPACMAN_PACMAN_PYTHON_ROOT` | `PACMAN_PYTHON_ROOT` |
| `MAAPACMAN_LOGP_RPC_CHUNK_SIZE` | `PACMAN_LOGP_RPC_CHUNK_SIZE` |
| `MAAPACMAN_DISABLE_CUDNN_SDPA` | `PACMAN_DISABLE_CUDNN_SDPA` |

旧 Python 模块转发到同一实现，保留类身份；`python -m` 入口继续可用。
新启动脚本通过 `scripts/pacman_paths.sh` 检查新旧路径变量的一致性；发生冲突
立即报错。AReaL 和游戏依赖仍消费的旧变量会由兼容层导出。
游戏内部 `MAAPACMAN_PAUSE_ON_START` 属于固定版本 pacman-python 的接口，继续保留。

当前不修改 CodeHub/GitLab remote。远端实验目录、历史 branch/tag、已保存的
checkpoint、轨迹、数据和报告不随项目显示名称改写。带固定时间戳的历史启动
脚本仍指向原快照。目录移动后需重新执行 editable install，避免旧安装路径。
升级已有 Python 环境时先卸载旧 `areal-pacman` distribution；否则 site-packages
中的旧实包可能抢先于新的 editable 兼容层被导入。

## 数据兼容边界

新的 recipe metadata 带有 `schema=pacman-recipe-contract-v1`、
`project=pacman-recipe` 和 `training_backend=areal`。
现有 dataset/trajectory v4、reward v3 和 save-state 的序列化字段保留，包括
`source_revisions["areal-pacman"]` 与 `maapacman_*` provenance 键；它们是历史
wire contract，不能当成目录名进行替换。目录定位已经使用新包的实际文件位置。
slime 的 backend provenance 扩展将在其 adapter 接入时实现。

新训练继续验证实际 source revision、dirty 状态、源码哈希、环境和 prompt。
旧数据能被读取不代表能直接用于当前运行。源码或 prompt 不匹配时重新生成数据；
不重写历史哈希。旧轨迹中的 v1 prompt 由原始 renderer 单独审计。

## C2 prompt v2

动作协议和候选 code 不变；prompt 版本从 `edward-option-code-v1` 升为
`edward-option-code-v2`。配置、renderer、template hash 和实际发送证据一起更新。

- 模型可见名称从 MaaPacman 改为 Pacman。
- 保留 COLLECT / AVOID / ELIMINATE 指导和 structured-state 优先规则。
- 更正通关条件：吃完普通豆即通关，能量豆不是通关必要条件。
  依据：`pacman-python/pacman/pacman.pyw` 在普通豆计数清零时产生 `level_cleared`。
- 当前状态明确携带 `life_mode`；system 说明 single_death 在首次死亡时结束。
- 每次请求重新构造 system/user 两条消息，只携带一张当前截图。
- 只输出当前候选集合中的一个大写选项 code，不输出 JSON 或移动方向。

原始 v1 文件保存在 `pacman_recipe/level1/legacy_prompts_v1.py`，用于历史审计；
新 rollout 默认使用 v2。C1 prompt 与动作协议保持原版本。
普通模式与 risk fallback 都继续执行 token-budget 检查，禁止截断。

## 验证

```bash
python -m pip uninstall -y areal-pacman
python -m pip install -e '.[dev,dataset,agent]'
python -m pytest tests/ -x
python -m scripts.level1.train.check_edward_prompt_budget --model-path /path/to/model --config configs/level1/train/curriculum2_single_death.yaml
```

完整 workflow 测试需要可导入的 Linux AReaL 环境。Windows 可独立执行环境、
命名兼容和 prompt 测试；不将它们描述为 GPU 或训练验证。
`tests/test_preparation_migration.py` 覆盖新旧 import 身份、环境变量冲突、v1/v2
证据审计、最新单图输入以及生命模式。实际模型 token 数须用运行时 processor
测量，不能以字符数或假 tokenizer 代替。

### 实测 prompt 预算

2026-09-28 单图语义修订：options system prompt 明确以结构化 ghost state
判断 vulnerable，不从单张截图推断闪烁；同时要求 edible_ticks 足以安全拦截。
此修订改变 prompt hash。既有冻结输入、审计记录和下表 token 数保留为修订前证据；
后续运行需生成新输入并重新测量预算，不覆盖正在使用的实验快照。

使用 Qwen/Qwen3.5-9B 的真实 processor/tokenizer（revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`），在一张 400×336 图片、
四只鬼、十个普通候选或四个风险候选的构造样例上测量：

| Processor 设置 | 普通候选 | 风险回退 |
| --- | ---: | ---: |
| 基础模型，minimum pixel area 65536 | 1017 tokens | 983 tokens |
| 将 minimum pixel area 设为 537600 | 1422 tokens | 1388 tokens |

基础设置通过当前 1024 输入预算，但余量较小；运行前仍需执行预算检查。
第二行只模拟 grounding 的图像预处理设置，超过 1024，后续接入应配置足够预算，
并使用实际 merged checkpoint 的 processor 复测。未加载模型权重或执行推理。
完整输入示例、版本和哈希见 [prompt-audit-v2.json](prompt-audit-v2.json)。
