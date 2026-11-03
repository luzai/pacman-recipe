# Opt-in engine-assisted experiment components

`pacman_recipe.experimental.engine_assisted` collects reusable algorithms from
the 2026-10-29 language-LoRA and 2026-11-02/03 engine-assisted work sessions.
It is independent of the default AReaL/slime recipe and does not launch jobs.
Source paths and file hashes are recorded in
[`SOURCES.json`](../pacman_recipe/experimental/engine_assisted/SOURCES.json).

## Included behavior

- `training_contract`: bind original attempts to protocol/checkpoint/schedule
  identities; a technical fault has null reward and masks its entire original
  group without replacement. The original attempt count remains the loss
  denominator. Zero within-group advantage skips model/entropy loss. Accepted
  groups shard disjointly across ranks. Terminal reward checks the absolute
  native horizon including the restart prefix. A separate manifest gate requires
  an accepted native replay/input audit; computing a batch plan alone is not
  evidence that a real game passed that audit.
- `curriculum`: normal-start training outcomes alone fill a promotion window.
  Restart successes and faulted groups cannot inflate promotion statistics. The
  density-course gates are explicit constructor arguments; this class does not
  replace the ASCII T1–T7 controller.
- `evaluation`: explicit inherited-update/interval/final-update routing,
  checkpoint/domain/split deduplication keys, and checkpoint metadata validation
  under a caller-supplied root. This validates metadata only, not symlink targets
  or file integrity; deployment code must check those before reading/deleting.
- `module_contract`, `language_lora`: a closed Qwen3.5 hybrid language-decoder
  projection inventory plus explicit output-head LoRA. All base/vision weights
  freeze; rank16/alpha16/dropout0 adapters are FP32 with zero initial up
  projections. Qualification derives the actual target inventory; training
  installation requires the reviewed inventory SHA supplied by the caller.
  `frozen_perception()` disables adapters and restores flags/modes even after
  an exception. State reload validates the exact module inventory and tensors.
- `choice_loss`: finite normalized legal-support probabilities at the historical
  temperature0.7, clipped selected-action PPO surrogate, forward categorical KL
  and explicitly requested entropy bonus. Original behavior/reference logs are
  detached, never replaced by trainer probabilities.

## Explicit protocol and historical scope

`training_contract.validate_protocol()` requires group/sample counts, original
attempt denominator, native horizon, batch budget, reward coefficients, frozen
source identities and disjoint train/validation/test topology IDs. There is no
global historical inventory SHA, map count, training budget or machine path.

The historical density curriculum used 36/12/12 topology splits and
0.6/0.8/1.0 food densities. The later original-map experiment used one training
topology, full food and separate held-out splits. Both can be represented by
explicit inputs; neither is selected as the recipe default. For example:

```python
from pacman_recipe.experimental.engine_assisted.evaluation import periodic_updates

assert periodic_updates(inherited_updates=5, interval=25, final_updates=100) == (30, 55, 80)
```

The caller dispatches a final evaluation separately. No worker, queue or service
is created by this helper. The earlier twenty-update plan is not imposed on the
later hundred-update plan.

## Dependencies and checks

Contracts, curriculum, inventory validation and evaluation routing use Python's
standard library. `language_lora` and `choice_loss` require an independently
installed compatible PyTorch build; NumPy is used when hashing adapter tensors.
The optional dependency group is `pip install '.[experimental-training]'`;
select the appropriate CPU/CUDA PyTorch distribution for the target environment.
Importing the package or pure contracts does not import PyTorch. No model download
or GPU is needed for the synthetic CPU unit tests:

```sh
python -m pytest tests/experimental/test_engine_experiment_contracts.py tests/experimental/test_engine_module_contract.py -q
python -m pytest tests/experimental/test_engine_language_lora.py tests/experimental/test_engine_choice_loss.py -q
```

The PyTorch tests validate zero-delta initialization, frozen base/vision
parameters, gradients/Adam state reload and the numerical loss. A skip when torch
is absent is not numerical acceptance or full model qualification. CPU toy tests
also do not reproduce full training, native replay or serving-backend parity.

## Not integrated as a runnable pipeline

Historical `engine_train.py` additionally depends on `paired_runner`, private
corpus/native replay bindings, `pilot_public_data`, backend qualification and
serving adapter manifests. Coordinators, Docker/SSH launchers, GPU placement,
service ownership registries and checkpoint retention scripts retain deployment
paths and run-specific authority. They are not copied into the reusable package.
The running experiment's frozen code, configuration and manifests are unchanged.
