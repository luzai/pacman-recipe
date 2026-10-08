# slime 迁移：实现与验收

## 下一轮固定前缀与布局对齐（2026-10-07，工作区修改）

ASCII 改用 `edward-ascii-option-code-v2+fixed-map-dynamic-v1`，固定字段、候选、导航和评分说明由 VLM 固定前缀派生，全部位于地图前；动态地图／状态／候选／输出顺序不变。没有添加缓存对齐文字，SFT 与 RL 共用该 formatter。旧数据及运行中的独立源码快照保持原身份；下一轮需重新准备数据、实测 tokenizer 边界并冻结身份。

固定缓存代码是显式启用的实验路径：只允许精确固定前缀节点，启动及每次权重变化后逐 engine 预热，实际决策校验缓存命中长度；长度、token SHA、补丁 SHA、启动参数、192 workers 与 32/64 并发写入 cache contract。确定性重放和真实更新 A/B 验收完成前不宣布采用。进展见[前缀缓存报告](../../reports/slime-migration-20260928/round2/rollout-throughput-prefix-cache/REPORT.md)。以下启动状态属于历史记录。

## ASCII动态bank v2（2026-10-07）

独立模块`slime_pacman.diverse_bank_v2`与`route_capture_v2`实现4开局＋4回练＋2早期＋6近期、16代表池、完整路线早期与死亡前快照。v1模块保持不变；v2 checkpoint明确拒绝v1身份。运行集成和验证命令见[实现记录](../../reports/slime-migration-20260928/round2/ascii-edward-next-run/V2_IMPLEMENTATION.md)。v2与新prompt的真实初始化可行性、GPU概率/TIS和原生恢复尚待验证；旧prompt的200-update进程已停止，未完成optimizer更新。

独立200-update长训入口及预算见[LONG_TRAINING](../../reports/slime-migration-20260928/round2/ascii-edward-next-run/LONG_TRAINING.md)：共享总游戏上限55,248，不设时间上限，不另跑两更新smoke。Clip-Cov固定`0.01,1,5`，128单线程workers／每engine16并发／radix cache开启。旧prompt任务启动后按用户要求停止，未完成optimizer更新；新prompt尚未部署。

## 新prompt与启动状态（2026-10-07）

源码提交`45a89dd`已推送：ASCII采用地图／状态／逐行候选／输出分区，VLM采用`fixed-image-dynamic-v2`。状态和候选证据、动作代码及risk fallback保持一致；dataset/YAML与Slime runner版本同步，runner从实际prompt契约读取版本。已有目标继续执行时不要求新目标选择前缀。

修复后18项快速CPU检查通过（12.04秒，含真实游戏episode）；全仓收集仍因本机缺AReaL阻塞。未跑GPU smoke，尚未验证新prompt真实2048-token上限。旧部署包绑定`3b52577`，不能直接重启；需新源码包/独立目录，重新冻结dataset/runtime identity、初始化bank和baseline。原始模型及游戏资产可复用。旧任务于15:22 PDT停止后8卡已核实释放；再次启动须重新现场检查资源。训练保持停止。

## 迁移历史边界（以下保留原验收记录）

本地 adapter、公共 episode runner、中立数据格式及 CPU 验收已实现。
真实 Megatron/SGLang GPU 训练尚未验收；不能据此声称训练已迁移成功。

固定 Docker 镜像已完成完整 Megatron forward 和 SGLang 图片请求。
原生 CPU fast processor 插件在 5 次真实请求中与冻结输入逐项一致；同输入下的
模型概率仍有差异，正在定位，尚未放行训练。见
[数值定位记录](../../reports/slime-migration-20260928/NUMERICAL-LOCALIZATION.md)。

临时 conda 环境已完成单层 GDN 与实际 checkpoint 的 HF 图片采集，详见
[探索结果](../../reports/slime-migration-20260928/CONDA-PROBE.md)。GDN 输出和跨样本梯度
检查未见串扰，但打包后部分梯度存在差异，零容差检查未通过；正式 Docker 验收仍待完成。

目录：

```text
life/
  AReaL/                 原训练框架，保持独立
  slime/                 固定 89bfada990a00663846e0ac804de1454685ceed3
  pacman-python/         游戏源码
  pacman-recipe/
    pacman_env/          游戏环境
    pacman_recipe/       公共配方与 AReaL adapter
    slime_pacman/        slime adapter
    configs/slime/      C2 配置和运行环境来源
    patches/            两行 metadata 传输补丁
```

## 已确认的阶段范围（2026-09-28）

- slime 后续训练统一 `single_death`：首次死亡即结束 episode；环境 runner、二值 reward
  校验及当前共享 prompt 使用相同规则。历史 AReaL 运行配置和已保存轨迹不改写。
- 当前不启用 score centering，也不安排本阶段的适配或对照实验；保留 PPO clipping + GRPO 基线。
- 当前不接入 `/v1/decisions`，不为该接口升级 SGLang。继续使用原生 vision `/generate` 和 custom logit processor。
- adapter 只统一合法 token 集合、temperature 和概率记录；推理、调度和采样复用 SGLang。
- GDN state isolation 与 train–inference mismatch 分开验收，均为正式训练的前置门槛。
  共享 masked softmax 定义不代表两端原始 logits 或 GPU kernel 已经一致。
- 每完成一个可独立验收的小步骤，记录结果并与用户讨论，再推进下一阶段。

## 每次决策的数据路径

1. 公共 `PacmanEpisodeRunner` 创建当前截图和 Edward options。
2. 构造新的 system/user 消息，仅一张当前截图，不附加历史消息。
3. 使用模型自带 processor 生成训练端的 input IDs、pixel values、image grid；禁止截断。
4. SGLang 收到完整文本和当前图片；每次独立 request ID，没有 session 续写，也不逐步 flush 全局 KV。
5. 保存单个动作 token、完整允许集合、行为 log-prob、图片 hash、weight version。
6. Megatron 每次决策对应一条独立 sequence；microbatch=1，TP=PP=CP=1。

这是原生 vision 路径。客户端隔离已测试；服务端 prefix cache、图像处理缓存及
Qwen3.5 recurrent state 隔离仍需 GPU 上的 A/B/A、并发和更新权重测试。

已从 H100_2_1 读取并核对实际 merged processor、tokenizer、chat template 的 hash。
十个候选、四只鬼的保守预算样例为 **1422 input tokens / 2048**；image grid 为
`[1, 50, 42]`，合并后 525 个视觉 tokens。该 checkpoint 的 chat template 与
当前 base 仓库模板不同，运行时必须从选定 checkpoint 加载，不能互换。

## 训练数学

- 12 个完整 episode 共享一个初始 seed；每次更新 4 组，共 48 episodes。
- 通关奖励 1，其余合法游戏终局为 0。接口、解析、权重版本等错误直接报错并写失败记录。
- 不丢弃全零/全一组，不补采样替换失败组。
- 使用固定 slime 的 sample std（correction=1）和 epsilon=1e-6，先按 episode 归一化，再展开 decisions。
- episode 内每个决策权重为 `1 / decision_count`；外层按 48 episodes 平均。
- SGLang custom logit processor 在 FP32 中限制到允许 token 并除以 0.7；请求 temperature=1。
  Megatron 使用同一集合与 0.7，避免全词表 log-prob 和采样分布不一致。
- PPO clip 两边均为 0.2；KL、entropy、额外 advantage normalization 均关闭。
- 初始安全拒绝保留为零奖励 episode。没有模型决策时不伪造 weight version。
- DP 补齐行 loss mask=0，复用已有 rollout ID；不增加 episode，不修改归一化分母。

## 数据与可复现性

新数据使用 `pacman-episode-v1`；轨迹使用 `pacman-trajectory-v1`。
显式记录实际训练 backend 和各源码的 commit、dirty 状态、source hash。
旧 AReaL 数据/轨迹仍由旧校验器读取；运行时兼容字段转换只在内存进行。
数据准备只创建新目录；历史 manifest、轨迹和实验结果保持原样。

从 repo 根目录执行，运行环境需有本仓库及固定 slime 的 Python 路径：

```bash
export SLIME_ROOT="$(cd ../slime && pwd)"
export PYTHONPATH="$PWD:$SLIME_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PACMAN_SLIME_CONFIG="$PWD/configs/slime/c2.yaml"
python -m slime_pacman.prepare --output /absolute/new-dataset-directory
python -m slime_pacman.preflight --dataset /absolute/new-dataset-directory
```

任何源码、prompt 或配置变化后生成新数据，不能修改旧 hash 绕过验证。
部署时应保留 source identity；简单复制部分文件而丢失 Git/来源信息会拒绝运行。

## 上游最小补丁

```bash
git -C "$SLIME_ROOT" apply /absolute/pacman-recipe/patches/slime-pacman-metadata.patch
```

只让 `Sample.train_metadata` 经过 DP 分发和 Megatron minibatch，供 custom loss
读取允许 token 集合。preflight 验证固定 SHA 和补丁已应用，不改变 AReaL。

## GPU 验收顺序

独立工具及证据格式见 [slime 验收工具](slime-acceptance.md)。当前已实现比较器和
真实 GDN 单层探针入口，并已在临时 conda 上执行探索测试；完整模型/backend
正式 GPU 验收尚未完成。固定 Docker 镜像现已恢复导入，真实 SGLang 视觉 adapter
已完成两张冻结图片的 A→B→A 和并发 HTTP 探针，同输入 log-prob 差为 0。
并发请求实际分别 prefill，联合模型 batch 和缓存开启仍待测。
同 Docker HF 对照及服务端像素记录已完成。默认 SGLang GPU 图像预处理与训练侧
CPU 像素 hash 不同；保持 fast processor 并仅强制 CPU 的探针已实现 5/5 输入一致。
不能直接使用 `--disable-fast-image-processor`，它还切换 processor 实现，hash 仍不同。
严格同输入的 HF–SGLang 最大概率差仍为 0.098514。
完整 Megatron 原生视觉 eval forward 已完成，同权重/像素的 Megatron–SGLang
最大概率差为 0.042795、最大绝对 Δlogp 为 0.911916；尚不能判定可接受。
下一步定位 position IDs、vision embeddings 和数值 kernel 差异，
并把统一 processor 实现和计算设备落实到正式入口。当前仅诊断 wrapper，尚未修改生产入口。
不能将这一探索对照当成 Megatron–SGLang 验收，也不据此放宽容差。

1. 在专属运行环境中按 `runtime-pins.json` 核验 SGLang、Megatron、TE、CUDA/H100 支持；
   记录容器 digest、实际包版本及源码 SHA。上游 Dockerfile 的来源 pin 不等同于实测兼容。
2. 核验 merged grounding 模型与 processor（保留原像素设置）；base 模型可单独指定，
   两者分别记录 hash，不继承旧 RL optimizer。
3. 独立验收 GDN state isolation：训练端比较 A 单独运行与 A/B 打包、B/A 打包时 A 的
   输出和梯度，覆盖 convolution 与 GDN 的样本边界、不同长度及 padding。
   当前训练保留 microbatch=1；不能仅凭 Megatron 名称或 `cu_seqlens` 参数判定通过。
   SGLang 比较 A→B→A、并发 A/B、不同 batch 组合及缓存开关，检查 recurrent state、
   prefix cache 和图像缓存。每回合保持独立请求，不以逐回合全局 flush 代替正确隔离。
4. 单独运行 train–inference mismatch 验收：固定同一份权重、图片、processor、prompt、
   options、temperature，比较 HF/Megatron/SGLang 所有合法动作的概率，而非只比较采样动作。
   同时核验 input IDs、causal alignment、image grid、动作 token 和训练梯度。
   记录 Δlogp 的 p50/p95/p99/max、KL、最大概率差、importance ratio，以及同权重 ratio
   超出 [0.8, 1.2] 的比例。该区间只是异常指标，不是通过阈值。
   先测 BF16 重复运行的噪声，再在训练前固定并记录更严格的容差；不得根据待验收结果放宽门槛。
   非法动作概率、支持集或输入不一致、非有限值、错误权重版本均直接判失败。
5. 前两项通过后运行 1-update smoke：完整 vision/text 参数可训练，有限 loss/gradient，
   实际参数变化、保存。同步新权重后，在固定输入上重跑隔离和概率对齐检查，
   确认权重转换、加载及缓存失效行为；保存更新前后 weight version 和模型来源。
6. 从该 checkpoint 恢复，完成第 2 次 update，验证 optimizer/RNG/data cursor、HF 导出与重新加载。
7. 以上通过后再接正式 50-update 入口、每 10 次验证、latest2+best 保留和完整运行报告。
   当前不自动清理 checkpoint，也不启动正式训练。

下面只输出可审阅的 smoke 命令，不启动 GPU：

```bash
python -m slime_pacman.command \
  --model /absolute/merged-model \
  --dataset /absolute/new-dataset-directory \
  --run-dir /absolute/new-smoke-run --updates 1
```

GPU 上额外执行 `preflight --model ... --runtime`。执行生成的命令前设置绝对
`PACMAN_RUN_DIR`，并按工作区规则核验设备、进程归属和服务端口；不要复用或停止其他人的 Ray/容器。
`--updates 2 --resume /absolute/first-run/checkpoints` 输出恢复命令，尚需真实恢复验收。

生成的命令通过 `slime_pacman.launch` 启动，显式设置 Ray job environment，
将 CLI 指定的 config、run directory、slime root 与 processor 注册变量传给 worker。
启动前的 `PYTHONPATH` 还需包含实际 Megatron checkout（固定镜像内为
`/root/Megatron-LM`）；launcher 保留这些依赖路径。已有初始化的 Ray driver 会直接拒绝。

`preflight --runtime` 需要
`SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=slime_pacman.sglang_processors`。
不要设置 `--disable-fast-image-processor`：插件保留原 fast processor，仅将图像预处理
放在 CPU。注册、像素一致性和 Ray worker 环境继承是不同检查项。
源码改动后重新生成数据及 manifest，不能继续使用旧源码身份的数据目录。

## 验证命令

```bash
python -m pytest tests/test_slime_pacman.py tests/test_slime_generation.py \
  tests/test_slime_upstream_contract.py tests/test_slime_preflight.py -q
python -m pytest tests/ -x
```

部分 CPU 测试执行固定 slime 源码的 causal slice/reducer 函数，并替代 TP/CP 拓扑查询；
它们能验证 loss 数学和数据边界，不能替代真实 Megatron distributed forward/backward。

## 当前下一步：有界的 PPO 影响评估

暂停逐层 kernel 对齐。冻结包含“每次移动通常 16 个逻辑 ticks、预留余量并考虑幽灵移动”
的新共享 prompt，采集目标 32 个、多 seed/多决策进度的真实状态。固定 RNG 从合法动作
选择后续状态，明确这是 scripted 状态覆盖，不声称是当前模型的 on-policy 状态分布。

固定 Docker/checkpoint、像素和 token/support 后比较完整动作分布：KL、TV、按 SGLang
概率加权的绝对 log-ratio、落在 PPO clipping 区间外的概率质量；另记录真实采样动作。
对正负 unit advantage 分别计算 PPO loss 和 action-logit 梯度偏移。这是局部敏感性检查，
不把 logit 梯度当成完整模型参数梯度，也不把假设 advantage 当成实际训练 advantage。

本轮以状态覆盖和完整影响报告为终点；输入合同失败直接停止。结合加权 clipping、loss
和梯度影响判断是否值得进入单次更新诊断，不以最大 ratio 或逐 bit 相等作为唯一依据。
1-update/保存/同步/恢复仍单独验收；本轮不自动放行长训练。

## 固定起点实验的补采样覆盖

使用 `slime_pacman.sampling.generate_rollout` 时，若静态数据集行数等于
`rollout_batch_size`，每次更新从数据集全部行选取起点，每个起点恰好一组。
补采样仍从同一起点重玩，并通过数据源分配新的 sample/group ID，但数据源游标
不能决定下一次更新的起点覆盖。否则跨 shuffle epoch 时可能重复或遗漏起点。
这种完整覆盖要求数据集的 episode record ID 唯一；重复 ID 会直接报错。
更大数据集保留原有轮换选择，动态 bank 使用独立 curriculum 入口。

单开局 overfit 可以用只含一条记录的数据集。该记录分配到四个独立组，
episode record ID 保持相同，sample/group ID 仍由 slime 分别生成。
分配请求按数据集长度分块，避免上游仅跨一次 epoch 时取不满四组；
任何取样不足立即报错。不要复制四条同 ID 的数据行来模拟此模式。

训练胜率含零方差组筛选和补采样，不代替每个固定起点的独立评估。
# Experimental PPO clip control

The default C2 loss still requires symmetric clip 0.2 and temperature 0.7.
For the declared AReaL comparison only, set `PACMAN_EXPERIMENTAL_PPO_CLIP=0.05`
and both `--eps-clip 0.05 --eps-clip-high 0.05`. The loss rejects a missing,
unsupported, or mismatched declaration. The dataset C2 template retains clip
0.2; record the effective training override in the experiment manifest.
This does not reproduce AReaL's proximal KL.

## Experimental legal-action regularization

Default training keeps both coefficients zero. Single-start controls may enable
one declared regularizer at a time:

- Entropy: `PACMAN_EXPERIMENTAL_ENTROPY_COEF=0.01`, `--entropy-coef 0.01`.
  The loss subtracts the entropy of the legal-action distribution.
- Frozen-reference KL: `PACMAN_EXPERIMENTAL_KL_COEF=0.01`, `--use-kl-loss`,
  `--kl-loss-coef 0.01 --kl-coef 0`, `--ref-load /model`, and
  `--custom-megatron-init-path slime_pacman.reference_policy.install`.
  `/model` must be the same clean initial model, with no reference update interval.
  The adapter computes exact `KL(actor || reference)` on the same legal support
  and temperature, with detached reference probabilities. Native full-vocabulary
  reference log probabilities are insufficient for this contract.
  The launcher passes an explicit reference actor subclass through slime's
  `actor_cls` factory argument before Ray captures the class. Worker initialization
  installs batch transport only; changing an imported actor class after Ray has
  serialized it does not reliably change the running actor's methods.

Both terms use the policy loss's episode reduction. Only static microbatch1 and
TP=PP=CP=1 are supported for the reference hook. The reference backup needs extra
CPU memory and a forward pass; verify the first real GPU update and frozen model
identity before accepting a run. Coefficient0.01 is an experimental value.

## Single-update decoupled PPO with TIS

The custom loss now separates the PPO ratio `current / old` from the detached
sampling correction `old / behavior`. Both actor probabilities use the same
legal support, FP32 normalization and temperature0.7; behavior probabilities
come from the actual SGLang rollout. The per-decision correction is
`exp(min(log_p_old - log_p_behavior, log(2)))`: upper truncation at2,
no lower clipping, no batch normalization and no MIS sample rejection.
This bounded estimator trades bias for reduced variance;2 is a starting choice,
not an experimentally established optimum.

Only `num_steps_per_rollout=1` and `global_batch_size=rollout_batch_size *
n_samples_per_prompt` are supported. All gradient accumulation happens before
the single optimizer step, so the loss reuses the same forward's detached
masked probabilities as old. It adds no separate old-model forward. Multiple
optimizer steps and dropout are rejected; supporting them requires a separately
cached old-policy forward, not repeated detachment of changing probabilities.

Keep `--use-rollout-logprobs` for behavior transport and to skip native
full-vocabulary recomputation. Do not enable upstream `--use-tis`: this custom
loss owns correction and rejects an additional native correction. Reference KL
and entropy stay separate from TIS. Metrics include `tis_weight_mean`,
`tis_truncate_fraction`, `ppo_ratio_mean`, and the old/behavior
`masked_logprob_abs_diff`, using the existing episode reducer.
CPU tests do not establish real GPU or learning-curve acceptance. Historical
experiments used frozen source; this change does not revise their results.
