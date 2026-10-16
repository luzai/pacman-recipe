# 下一次 slime 训练 handoff — 2026-10-16

用户已指定下一次新训练使用以下默认值，不再额外跑一遍 GPU profiling／验收；
后续实际训练出现问题时再根据证据调整。此次只修改本地 recipe 默认值，
不部署、不重启、不修改正在运行的 control／bonus，也不更改它们的历史源码身份。

| 参数 | 下一次新 job 默认值 |
| --- | --- |
| micro-batch-size | 2，固定每个 microbatch 两条决策样本，不是固定 token 预算 |
| recompute-granularity | full |
| recompute-method | block |
| recompute-num-layers | 24：32层模型中的前24层重算，后8层保存 activation |
| optimizer CPU offload | 保持开启，保留 overlap 和 precision-aware optimizer |
| TP / PP / CP | 1 / 1 / 1 |

默认启动参数由 `slime_pacman.command.build_command` 生成。自定义 launch 脚本若
显式覆盖这些值，需要同步采用本表；历史运行脚本与冻结报告不作批量改写。
不要把 `uniform + num-layers=24` 当作上述设置，两者重算语义不同。

## Adapter 处理

`batching.convert_samples` 支持静态 microbatch 1／2，仍拒绝动态 batching 和
TP／PP／CP大于1。零 loss 的 schedule padding 按 `DP × microbatch` 补齐，
不丢真实决策、不增加 episode ID、保持原 per-episode reduction／组内 reward 语义。
旧 microbatch=1 调用保持兼容；可选 frozen reference 路径也允许 microbatch=2。

## 已有证据与边界

2026-10-16 独立 H800 80GiB ×8 profiling：真实每条974-token输入下，
microbatch=2＋block24 的 forward/backward 为4.89秒／每rank32条决策，
tensor峰值62.3GiB；1536-token合成压力输入峰值66.8GiB。
每卡额外保留8GiB后，两种长度和605-token输入均在8个rank完成，最长压力峰值
74.8GiB（包含额外8GiB）。这些是 ASCII Primitive 输入，不能当作图片模式的显存保证。
profiling 未采用完整 GRPO／SGLang colocate 路径，
初始化为只读 HF 权重，不是当前正式 checkpoint；不能据此保证任意实际输入无 OOM。

证据：[profiling 选择报告](../../reports/primitive-training-profile-20261016/RECOMPUTE_SELECTION_REPORT.txt)。
报告原先保守推荐全量重算；用户随后明确选择 block24，因此下一次默认值以本 handoff 为准。

本次不新增 GPU 工作、不宣称完成端到端 GPU 验收。新 job 应记录实际参数、源码
SHA 和新的运行身份，不要篡改旧 manifest 或把新源码冒充旧 checkpoint 快照。
若实际训练 OOM 或有数值／打包异常，保留日志和失败证据后调整；
本说明不授权自动重启、改变 reward／prompt／训练预算，或干预无关进程。

本地检查：2026-10-16 在独立 CPU 测试环境中运行 generation、preflight、
regularization 和 upstream contract 四个测试文件，37项通过，包括静态
microbatch 1／2 的真实 upstream 转换、零 loss padding、episode 权重和
可选 reference 输出数量校验；reference／KL 默认仍关闭。
测试环境使用 Python 3.13、CPU PyTorch 2.8，非正式 GPU runtime。
全仓 `python -m pytest tests/ -x` 仍因缺少 `areal` 在收集阶段阻塞。
没有发起 GPU 复测，不能宣称完整测试套件或端到端 GPU 验收通过。
