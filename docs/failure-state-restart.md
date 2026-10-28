# Failure-State Restart（失败前状态重启训练）

新 Primitive formal map 训练默认开启 `failure_state_restart`。显式关闭：

```json
{"failure_state_restart": false}
```

`null` 同样关闭；`true` 或省略采用默认值。完整默认配置见
`configs/slime/maps/failure-state-restart.json`。旧 checkpoint 恢复必须显式关闭
以保留旧身份；不能直接把旧 optimizer 恢复伪装成新策略身份。
改变策略的 fork 仍需独立目录、父 checkpoint 完整校验和显式阶段转换。

Edward 的规划器历史不适用本次 Primitive 快照恢复协议，保持原行为。
检测到 Edward catalog 时，省略该字段即关闭；显式开启会拒绝并说明原因。
现有 vision 模板显式设为 `false`。

当前 Tier 与低 Tier 回放分别按 slot 轮转，在连续五次更新内请求
80% 失败前状态、20% 正常开局；保留原来的 Tier 配额、每组12次。
先等概率选 4–7、8–15、16–32 步档，再从可用整数偏移均匀采样。
档内状态沿用试跑的权重：未见状态1，混合成败状态2，同结果状态0.1。
没有同 Tier 状态时回退正常开局并记录 `restart_fallback`，不凭空报告80%。

状态库仅从首轮正常训练轨迹、每组前3个 replica 更新，恢复完整环境/RNG/
生命/步数和 primitive 动作历史。先精确 native 重放原前缀，再保存死亡前状态；
不会重用旧动作作为优化数据。默认每 Tier 最多64状态，12次更新后过期。
从头训练先积累正常开局轨迹，随后自动启用重启状态。

原始奖励按绝对生命和总步数计算，不赠送生命或额外512步。
过滤与补采保持原来的最终 reward 条件和16轮上限。课程晋级只统计
首轮正常开局的当前 Tier；独立 validation 始终正常开局、不筛胜负。
课程窗口按原 `window_batches` 个完整当前 Tier 正常开局 batch 的样本量累计：
有80%重启时会跨更多更新收集正常样本，不把最近3次更新误当成3个完整正常 batch。
胜率、最少局数、布局数和连续通过门槛均不降低；无可用重启状态时沿用原正常
开局晋级节奏。窗口语义记录为 `failure_state_restart-v2-normal-window` 身份，
不会自动接受旧版窗口身份的 checkpoint；历史冻结实验不热更新。
该功能不自动改变 Tier 上限、验证档数或训练预算。

joint checkpoint 包含状态库及所有被引用状态的SHA。恢复时重新定位到新运行目录，
并验证每个快照。历史试跑路径和 Backplay 名称不改写；该移植不启动新训练。
