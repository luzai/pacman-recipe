# slime 独立验收工具

## 当前实现范围

`slime_pacman.acceptance` 提供完整合法动作分布比较和 sequence isolation 探针。
CPU 测试包含故意忽略样本边界的 recurrent module，确认探针能检出输出及梯度串扰。
CPU 测试只验证验收工具。临时 conda 中的真实单层 GDN 探针已执行，见
[探索报告](../../reports/slime-migration-20260928/CONDA-PROBE.md)；完整模型、SGLang 缓存、
权重同步及正式 Docker 验收仍待完成。

## GDN layer probe

在已核验版本且 GPU 分配安全的运行环境中使用：

```bash
python -m slime_pacman.probe_gdn --model /absolute/merged-model \
  --atol "$GDN_ATOL" --rtol "$GDN_RTOL" --output /absolute/new-gdn-report.json
```

容差须先通过独立的重复运行标定；工具不提供默认“可接受误差”。
此命令使用固定 slime 的真实 Qwen3.5 GDN layer 和 FLA kernel，按模型 config 构建
随机初始化的 BF16 单层。它不加载 checkpoint 权重、不启动 optimizer 或推理服务。
因此通过只说明该层、输入及 runtime 下的结果，不能代表整个模型通过。

比较 A 单独、重复 A、A/B 和 B/A，使用仅依赖 A 的固定线性 loss，检查：

- A 的输出、输入梯度和所有可训练参数梯度。
- B 的输入梯度严格为零，避免仅用数值容差掩盖跨样本梯度。
- 三组不同长度覆盖 64-token 边界。完整模型的 padding、视觉路径仍需后续验收。

使用 `autograd.grad`，不更新参数，不写参数的 `.grad`。报告记录 runtime、GPU、
config/patch hash、seed 和各参数误差。输出文件不可覆盖；失败返回非零退出码。

## 完整合法动作分布比较

每个 backend 的采集器需保存独立 JSON，至少包含：

| 字段 | 含义 |
| --- | --- |
| `fixture_sha256` | 固定输入 fixture 的 SHA256 |
| `weights_sha256` | 实际加载的共同权重清单 hash；不是服务端 version 字符串 |
| `processor_sha256` | processor 文件 hash 清单的共同 hash |
| `image_sha256`, `image_grid_thw`, `input_ids` | 输入图像、单图 grid 和完整模型输入 IDs |
| `allowed_token_ids`, `temperature` | 顺序一致的合法动作 token 和实际策略温度 |
| `log_probs` | 对应全部合法动作的归一化 log-prob |
| `outside_support_probability` | 额外验证的支持集外概率，必须为零 |

采集器还应保存 backend、组件版本、request/cache 条件和 weight version。
不能只凭 HTTP 请求参数填写“实际加载权重”，也不能只凭观察到一个合法采样动作就
填写支持集外概率为零。这些采集及来源验证尚未完成，比较器本身不证明记录真实性。

容差文件包含 `max_abs_logp`、`max_kl`、`calibration_evidence`（先前标定证据位置）。
在测试前保存此文件，不根据待验收结果放宽限制。

```bash
python -m slime_pacman.compare_evidence \
  --reference /absolute/megatron-state-a.json \
  --candidate /absolute/sglang-state-a.json \
  --tolerances /absolute/frozen-tolerances.json \
  --output /absolute/new-comparison.json
```

工具拒绝输入/权重/支持集不一致、非有限值、概率未归一化或 ratio 溢出。
报告包含 Δlogp 分位数、KL(reference || candidate)、最大概率差、ratio 范围和超出
PPO [0.8, 1.2] 的比例。该比例是诊断项；通过标准由显式 logp/KL 容差决定。
结果绑定两个记录和容差文件的 SHA256，不能覆盖已有文件。

这是单个输入的一对比较。完整验收还需要多个状态、串行/并发及缓存组合、更新前后
权重同步的覆盖清单，以及完整模型梯度与 padding 检查，不能由一条 `passed` 替代。

score centering 与 `/v1/decisions` 不在当前阶段范围内。
