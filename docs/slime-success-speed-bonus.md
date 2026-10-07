# 通关后的短路线 bonus

在 `PACMAN_SLIME_CONFIG` 指向的 YAML 中设置 `success_speed_bonus: 0.05`。
默认值为 0，保持原来的二元奖励；允许范围为 0–0.1。

失败奖励为 0；成功奖励为 `1 + coefficient * (1 - steps / horizon)`。
`steps` 是本次 episode 实际执行的环境步数，不是模型决策数或游戏 ticks。
中途存档的 `horizon` 使用恢复时的 `remaining_budget`，真开局使用 `max_steps`。
环境本身的奖励和存档身份不修改，bonus 仅在 slime adapter 的 episode 结束后计算。

不同速度的全赢组有不同 reward，可以保留并计算 GRPO advantage。
相同 reward 的组仍按现有重采样规则处理；该实验应显式声明奖励配置。
后续实验使用同一起始状态12局组内 center：`A_i = r_i - mean(r_group)`，不除标准差，也不做全局 center 或后续 whitening。相同 reward 精确返回0；不同速度的全赢组仍可能产生非零 advantage，不自动强制归零。历史冻结源码的标准差归一化配置保持不变。

`rollout/win_rate` 和 `eval/<name>/win_rate` 始终是二元通关率。
`mean_reward` 单独报告含 bonus 的平均奖励，`success_steps_median` 仅统计成功局。
episode artifact 与 sample metadata 记录 coefficient 和执行步数。
建议独立比较通关率和成功局步数；CPU 测试不代表 GPU 训练效果已验证。
