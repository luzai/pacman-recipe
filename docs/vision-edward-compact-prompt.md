# Vision Edward compact_v1

`PacmanConfig(edward_prompt_variant="compact_v1", groups_per_update=16,
clip=0.05, max_input_tokens=4096)` explicitly selects the compact image prompt.
It requires `edward_harness="latest"`, image observations, Edward actions and
`episode_life_mode="original_three_lives"` (four total lives). Existing configs
default to `standard`, retaining their prompt text and identities.

The compact renderer removes model-visible internal option IDs/route hashes,
omits unavailable optional fields, and merges identical junction/turn records.
Portal and approach information, fallback risk, and bounded escape evidence
remain. A missing field never means safe; false/zero evidence is retained.
Identical route summaries require full bound actions for disambiguation, otherwise
rendering fails. Internal route binding and per-step safety checks are unchanged.

Dataset generation must pass the same `edward_prompt_variant` to
`make_episode_record` (or use `record_options(config)`). Runner, trajectory audit,
and prompt metadata distinguish compact from standard. A recipe contract's
`harness.edward_prompt_variant` must also match. Do not hot-patch a running job or
reuse records with another prompt fingerprint.

The initial paired check used 11 saved diagnostic suffixes, not independent full
games. D5A and checkpoint160 each retained 11/11 wins and zero deaths/timeouts;
matched initial-input tokens decreased 20.25%, including unchanged image tokens.
Calls and steps decreased, but immediate reversals increased by one for each
model. Fallback coverage was limited. These results do not prove broad safety or
generalization. Frozen experimental evidence is in the workspace report directory
`reports/vision-edward-compact-prompt-20261022`.
