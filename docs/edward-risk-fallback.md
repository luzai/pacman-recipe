# Edward 单步风险回退

新 Edward 运行默认使用 `risk_ranked`，也可在 C2 配置中显式设置：

```yaml
edward_options: true
edward_fallback_mode: risk_ranked
objective_encoding: edward-option-code-v1
```

流程：每次决策先生成原有 COLLECT / AVOID / ELIMINATE 候选；只要有候选，
生成、排序和执行逻辑保持原样。三类候选全部为空时，展示所有物理可走的
U/D/L/R 方向，标记为 `RISK_FALLBACK`，让模型选择。每次只执行一个环境
move（16 logic frames），然后重新尝试正常 options。即使全部方向都预测
有碰撞，也不会因为这个预测提前终止；死亡、通关和原有 episode 限制仍由
环境/workflow 正常处理。没有物理可走方向时仍拒绝。

新模式替换旧的“自动选择一个已通过检查的 emergency A0”步骤；它的触发
条件是正常 C/A/E 候选为空，而不是要求旧 emergency 检查也失败。

## 风险提示

风险候选复用 A0..A3 和 J/K/M/N 单 token 协议，但 strategy 明确为
`RISK_FALLBACK`，不伪装成通过安全检查的 AVOID。风险字段单独存储；
原有 `safety_margin` / `future_safe_exits` 留空。

排序依次比较：

1. 下一步 ghost 运动估计：clear_estimate、unknown、collision_predicted；
2. 更大的 route margin、ghost clearance、safe next cells；
3. 尽量避开死胡同、反向移动；完全相同则按 U/D/L/R。

这是启发式排序，不是死亡概率或安全证明。运动信息缺失、非有限值或
无效速度会标成 unknown；已知 ghost 的碰撞预测优先于其他 ghost 的 unknown。
沿用的运动估计不模拟吃 power pellet 后的状态变化，可能高估危险。
safe_next_cells 只表示相邻格子的距离检查结果，不保证存在独立逃生路线。
排名最后的方向仍可被选择。

## 兼容性与训练

- 新 Edward planner、native runner、AReaL/slime 配置及数据合约默认 `risk_ranked`；primitive 仍使用 `refuse`。
- 显式 `edward_fallback_mode: refuse`（baseline CLI 为 `--fallback refuse`）保留旧 emergency/refusal 行为。
- 显式 `refuse` 的候选序列化、prompt 文本和 template 指纹保持不变。
- 旧轨迹/合约缺少 fallback 字段时仍按 `refuse` 解读；新运行明确记录 `risk_ranked`。
  底层冻结的 prompt renderer 保留 legacy 默认参数，生产入口显式传入解析后的模式。
- 新模式在 recipe harness、prompt 指纹、trajectory decoding 和 observation
  context 中留下证据；轨迹审计检查方向完整性、风险字段、单步结束及实际执行。
- 单 token 支持集合、rollout logprob 和 PPO support ledger 使用原有链路。
  新模式下 A0..A3 的语义有显式风险提示；旧 checkpoint 可加载，但适应新提示后的
  胜率需要单独评估。
- 训练启动器按配置检查正常候选和风险候选两种 prompt 的预算；超长时拒绝
  启动。默认预算保持不变，正常决策省去风险表，风险提示用紧凑的数组表示。
- 不直接改动已有实验 YAML/manifest。新实验需要使用匹配的新配置重新生成
  数据/manifest，训练和评估都启用相同模式。不要把两个 harness 的结果混算。

针对性测试：

```bash
python -m pytest tests/env/test_edward_planner.py tests/env/test_edward_risk_fallback.py tests/test_edward_risk_workflow.py tests/test_level1_recipe.py -x
```

用实际模型 processor 单独检查新模式的最坏 prompt：

```bash
python scripts/level1/train/check_edward_prompt_budget.py --model-path /path/to/Qwen3.5-9B --fallback-mode risk_ranked --max-input-tokens 1024
```
