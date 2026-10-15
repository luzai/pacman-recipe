# Configurable success reward

The slime adapter supports terminal completion reward with optional life and speed bonuses:

`R = won * [1 + success_life_bonus * (1 - deaths / (L - 1)) + success_speed_bonus * (1 - steps / H)]`.

`L` is the starting total life budget (4 for `original_three_lives`, 1 for
`single_death`). The life term is omitted for a one-life start. `H` is the
environment action horizon, not response token length. Failed episodes always
receive zero. For restart episodes, deaths are counted after restore, `L` is the
remaining life budget, and `H` is `restart_state.remaining_budget`.

New slime training defaults are `success_life_bonus: 0.1` and
`success_speed_bonus: 0.05` (2026-10-14). Both must be finite values in `[0, 0.1]`.
Frozen historical source/configurations are unchanged. Explicit zero coefficients
retain binary reward behavior; the speed-only helper API remains compatible.
Record these fields explicitly in a new run's YAML and regenerate its identity:

```yaml
success_life_bonus: 0.1
success_speed_bonus: 0.05
```

Coefficients are preserved in episode results, sample metadata and ordinary
rollout artifacts. A batch mixing coefficients is rejected. Group advantages
remain episode reward minus group mean, without standard deviation scaling.
Existing explicit `zero_binary_extremes` sampling contracts still zero all-win
groups; those contracts therefore do not learn bonus differences in all-win groups.

Validation win rate continues to use the binary completion event, independently
of shaped training reward. These bonuses encourage survival and efficiency;
generalization must still be assessed on held-out layouts and environment seeds.
Use a new run identity and source manifest when deploying this change; frozen
historical experiments are not modified.
