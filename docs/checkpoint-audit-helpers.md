# Loaded-state comparison and checkpoint retention plans

These optional helpers extract reusable parts of the historical ASCII FSR recovery
and retention tooling. They do not install loader hooks, modify training defaults,
write audit files, delete checkpoints, or interact with remote services.

## Verify a loaded state

```python
from slime_pacman.map_training.restore_audit import (
    assert_state_restored, assert_tensor_dict_restored,
)

model_proof = assert_tensor_dict_restored(
    checkpoint_model, loaded_model.state_dict(),
    required_parameters=[name for name, _ in loaded_model.named_parameters()],
)
scheduler_proof = assert_state_restored(
    checkpoint_scheduler, loaded_scheduler.state_dict(), label="scheduler",
)
```

Capture complete state immediately before/after the real loader. Tensor comparison
includes bytes, shape and dtype. The flat tensor dictionary helper rejects scalar
or mixed values; compare those with `assert_state_restored`. Python/NumPy state
trees are supported; Torch tensors and generators require Torch only when passed.
The exact normalized representation is new and is not a replacement hash for old
experiment receipts. CPU-normalized array values are compared, not memory strides.

`required_parameters` prevents two incomplete state dictionaries from both omitting
the same parameter. A caller may explicitly allow a named unexpected empty uint8
vector with `allowed_empty_extras`; no architecture-specific exceptions are baked
in. The helper will not ignore nonempty extras, different dtypes or shapes.

For RNG, provide every relevant channel, not just the CPU generator. For optimizer
recovery, include moments, counters and parameter-group fields. A successful hash
comparison of a subset cannot establish that omitted state was restored. This
module does not collect distributed state, check all ranks, configure a Megatron
loader, qualify GPU resume, or prove that a subsequent optimizer update succeeds.

## Select retained versions without deleting files

```python
from slime_pacman.map_training.checkpoint_retention import select_retained_updates

selection = select_retained_updates(
    available_updates=[335, 340, 430],
    scores={335: (78, 96), 340: (58, 96)},
    pending_updates=[430],
)
assert selection.keep_updates == {335, 430}
```

The caller supplies already verified complete payload labels. Use a single
comparable evaluation protocol; the utility does not infer compatibility from
counts. Rates use exact rational arithmetic, ties keep the earlier label, and
missing pending payloads fail closed. Labels are not automatically incremented:
zero-based checkpoint iterations and completed optimizer counts must be mapped by
the caller using original receipts.

Scores may include separately protected external ancestors. If one is globally
best, `best_update` names it, but the local keep set only contains available
payloads; protect the external ancestor separately. The complement of the keep
set is **not** automatically safe to delete. Active processes, service boot and
adapter leases, consumer pins, path/symlink checks, receipts, and live ownership
remain requirements of any separately authorized cleanup operator.
