# Edward：两个可复现的历史版本

这里仅整理 **Original＋risk_ranked** 和 **Latest**。生命模式是配置，不是版本名。
本目录不注册新默认入口，不改变现有训练、评测或生产配置。

|版本|代码与身份|行为|
|---|---|---|
|Original＋risk_ranked|原始 `28e9133`，见 `original_risk/ORIGINAL_PIN.json`|保留原普通候选、safe emergency、目标与 option 续行；仅在原算法安全拒绝时提供风险排序的单步候选|
|Latest|生产 `pacman_env/latest_edward.py`，实现提交 `6705954`|保留当时的完整路线绑定、安全检查、到期判断和 escape 信息；使用 risk_ranked|

## Original＋risk_ranked

`original_risk/original_planner.py` 是逐字节保留的历史原件。
adapter 仅在其 `advertised_candidates` 抛出安全拒绝时调用历史 risk helper；
fallback 执行一步后重选。正常候选和原 safe emergency 优先，不由新版候选算法替代。

归档入口把原实验对生产 `pacman_env.planner` 的依赖改为同目录
`frozen_native_planner.py`；其 LF 字节 SHA 与已验收实验源完全相同。
这避免未来生产代码变化悄悄改变历史基线。源码和 adapter 校验见 `MANIFEST.json`。
该文件含有完整历史模块，但 adapter 只借用两个 risk 方法及其所需类型，
不继承该模块的普通候选、safe emergency 或 option 续行实现。

CPU 差异检查（仓库根目录）：

```powershell
python baselines/edward_versions/original_risk/verify.py
```

复用原生 episode 时，在**新进程、创建 episode 之前**将归档目录加入
`sys.path`，调用 `original_adapter.install_original()`，随后使用
`edward_harness="legacy"`、`edward_fallback_mode="risk_ranked"`。
仅设置 `legacy` 不会自动安装这个历史版本。
该入口沿用实验的进程内安装方式，不用于给运行中的 worker 热更新。
七项 CPU 检查和历史 256 对拒绝前轨迹审计见 `ACCEPTANCE.json`。

## Latest：同一个版本，可配置生命模式

Latest 实现已在 Git 中，不复制另一个生产模块。
`VERSIONS.json` 记录实现提交和当前归档对应的源码指纹。
两个 JSON 文件是已有配置的**覆盖片段**，不是完整训练配置或启动命令：

```json
{
  "edward_harness": "latest",
  "edward_fallback_mode": "risk_ranked",
  "edward_prompt_variant": "compact_v1",
  "episode_life_mode": "single_death",
  "ghost_mode": "normal"
}
```

单命使用 `single_death`；四命使用历史字段 `original_three_lives`
（一条当前生命＋三条备用生命）。同样的生命模式也适用于 Original＋risk_ranked。

## 已完成评估记录

`EVALUATION.json` 仅保留这两个 risk_ranked 版本的八格统计；
不加入其他 fallback 组。两个模型、两种生命模式，每格64个相同 seed
940000–940063，21列×25行、四个普通 ghost、最多512步。

|D5＋LoRA A|单命通关|四命通关|
|---|---:|---:|
|Original＋risk_ranked|15/64|43/64|
|Latest|27/64|62/64|

62/64＝96.9% 是特定四命评估结果，不是版本名称或所有配置下的保证。
这两个历史版本使用引擎辅助信息，不能当作正在开发的公共反馈 Harness 的成绩。
固定一张地图的多 seed 评估不代表跨地图泛化。

完整证据仍保存在工作区 `reports/vision-edward-original-risk-only-20261026/`：
`FINAL_SUMMARY.json`、`FINAL_ACCEPTANCE.json`、`PREFIX_AUDIT.json` 和原始轨迹。
本归档只保存小型索引、指纹、代码和汇总，不上传模型、轨迹或凭据。
当前公共反馈原型和早期独立 safety candidate 均不属于本归档。
