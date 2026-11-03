# Explicit action prefix isolation

The opt-in mode `unique-extra-key-v1` gives every SGLang policy action request a
fresh `extra_key` tied to its unique `rid`. It preserves the image, prompt,
legal action support, sampling parameters and complete-support log-probability
checks. Every response must report integer `cached_tokens: 0`; absent, boolean,
string or nonzero values are rejected.

For a new run, set both the recipe YAML and its rollout process environment:

```yaml
action_prefix_isolation: unique-extra-key-v1
```

```sh
export PACMAN_ACTION_PREFIX_ISOLATION=unique-extra-key-v1
```

The configuration default remains `off`; missing environment also means `off`.
Policy rollout passes this explicit mode to `SGLangGenerator` and rejects a
config/environment mismatch. Direct callers can pass
`action_prefix_isolation="unique-extra-key-v1"` with the same environment.
Existing independent sensor/diagnostic callers that omit the argument keep
their original behavior; the implementation does not modify an initial-image
sensor cache or global server cache settings. Explicit isolation cannot be
combined with an active fixed-prefix cache experiment.

This implementation comes from the 2026-11-02 visual experiment's
`sg_wire.sg_action_body` and `prefix_contract.require_prefix_isolation` boundary.
It incorporates the per-request namespace and matching config/environment
guard into the maintained generator and policy-rollout configuration, rather
than shipping its machine-specific service launcher. No running job is patched.
The CPU tests verify wire semantics and rejection behavior; they are not a new
SGLang/GPU acceptance result. A future deployed run must verify backend support
for `extra_key` and its actual zero-hit responses.
