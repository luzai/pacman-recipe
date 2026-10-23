# Compact Vision Edward prompt: isolated experiment

Not enabled by default or applied to active training. This renderer wraps the production renderer and keeps planner bindings and safety logic unchanged. It removes internal option IDs, null optional fields and repeated route fields, preserves zero/false values and safety information, and exposes full actions when route summaries would otherwise be ambiguous.

Run focused checks from this directory with `python -m unittest test_compact -v`.

## Paired evidence (2026-10-22)

Same 11 saved suffix states, environment RNG, history and remaining budgets. HF bf16 SDPA greedy legal choices; frozen recipe source 52d536a. These are related failure-state regressions, not independent full games or SGLang training acceptance.

| Model | Prompt | Wins | Deaths | Timeouts | Steps | Agent calls | Backtracks |
|---|---|---:|---:|---:|---:|---:|---:|
| D5 + LoRA A | Original | 11/11 | 0 | 0 | 335 | 67 | 9 |
| D5 + LoRA A | Compact | 11/11 | 0 | 0 | 258 | 56 | 10 |
| checkpoint160 | Original | 11/11 | 0 | 0 | 281 | 63 | 10 |
| checkpoint160 | Compact | 11/11 | 0 | 0 | 254 | 53 | 11 |

Matched initial-state input tokens including images: 19147 -> 15270 (-20.25%). Total trajectory token reductions additionally reflect changed actions and call counts. Eight unit checks and 44 real image processor boundaries passed. Fallback coverage was limited (D5A 1->1, checkpoint160 1->0); full-game paired validation remains necessary before promotion. Do not infer that every prompt fits 2048 from the mean token count.

Full local evidence and preserved failed setup attempts: workspace reports/vision-edward-compact-prompt-20261022. No credentials or trajectory payloads are published here. Source/runtime identity must be regenerated when adopting this renderer. Prefix-cache token boundaries must be remeasured and numerical parity requalified; current prefix-cache diagnostics failed, see ../../docs/vision-sglang-cache-diagnostics.md.
