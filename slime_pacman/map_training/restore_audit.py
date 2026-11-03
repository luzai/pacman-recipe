"""Exact loaded-state comparison helpers, independent of a training backend.

No hooks are installed and no files are written. Tensor comparisons are CPU byte
comparisons, not GPU recovery qualification. Torch is imported only for Torch
values; NumPy-only/standard-library callers do not require it.
"""
import hashlib
import json
import math


def _array(value):
    family = type(value).__module__.split('.')[0]
    if family == 'torch':
        import torch
        if isinstance(value, torch.Generator):
            return None
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided:
                raise ValueError('restore audit requires a strided tensor')
            v = value.detach().cpu().contiguous()
            raw = v.reshape(-1).view(torch.uint8).numpy().tobytes()
            return dict(kind='torch_tensor', dtype=str(v.dtype), shape=list(v.shape),
                        sha256=hashlib.sha256(raw).hexdigest())
    if family == 'numpy':
        import numpy as np
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError('object arrays cannot be fingerprinted exactly')
            if value.dtype.fields is not None:
                raise ValueError('structured arrays require an explicit state schema')
            return dict(kind='numpy_array', dtype=value.dtype.str, shape=list(value.shape),
                        sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
    return None


def normalize_state(value):
    """Return a JSON-safe, type-sensitive snapshot of supported state values."""
    array = _array(value)
    if array is not None:
        return array
    family = type(value).__module__.split('.')[0]
    if family == 'torch':
        import torch
        if isinstance(value, torch.Generator):
            return dict(kind='torch_generator', state=normalize_state(value.get_state()))
    if family == 'numpy':
        import numpy as np
        if isinstance(value, np.generic):
            if value.dtype.fields is not None or value.dtype.hasobject:
                raise ValueError('structured scalars require an explicit state schema')
            return dict(kind='numpy_scalar', dtype=value.dtype.str,
                        value=normalize_state(value.item()))
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('nonfinite scalar in restore state')
        return value
    if type(value) in (list, tuple):
        return dict(kind=type(value).__name__, items=[normalize_state(x) for x in value])
    if isinstance(value, dict):
        # Optimizer states commonly have integer keys. Retain key types rather
        # than silently colliding the integer 1 with the string "1".
        items = [(normalize_state(k), normalize_state(v)) for k, v in value.items()]
        items.sort(key=lambda pair: _json(pair[0]))
        return dict(kind='dict', items=items)
    raise TypeError(f'unsupported restore state type: {type(value).__name__}')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def state_fingerprint(value):
    return hashlib.sha256(_json(normalize_state(value)).encode('utf-8')).hexdigest()


def assert_state_restored(expected, actual, *, label='state'):
    """Compare full scheduler/RNG/optimizer state supplied by a caller.

    The caller must capture *all* required RNG channels and optimizer fields.
    Fingerprinting a subset cannot prove omitted fields were loaded. Returned
    evidence contains hashes only; mismatches raise without dumping state values.
    """
    before, after = state_fingerprint(expected), state_fingerprint(actual)
    if before != after:
        raise ValueError(f'{label} restore mismatch')
    return dict(passed=True, label=label, expected_sha256=before, actual_sha256=after)


def assert_tensor_dict_restored(expected, actual, *, required_parameters=(),
                                allowed_empty_extras=()):
    """Compare flat named tensor/array dictionaries, including dtype and shape.

    Required parameter names catch parameters omitted from both dictionaries.
    A caller-named *unexpected* extra may be ignored only when it is exactly an
    empty uint8 vector. No model-specific keys are implicitly allowed. Non-array
    values are rejected; use ``assert_state_restored`` for mixed state trees.
    """
    if not isinstance(expected, dict) or not isinstance(actual, dict) or not expected:
        raise ValueError('nonempty expected tensor dictionary required')
    if any(type(k) is not str for k in [*expected, *actual]):
        raise ValueError('tensor names must be strings')
    required = set(required_parameters)
    if not required <= expected.keys() or not required <= actual.keys():
        raise ValueError('checkpoint is missing required model parameters')
    left = {k: _array(v) for k, v in expected.items()}
    right = {k: _array(v) for k, v in actual.items()}
    if any(v is None for v in [*left.values(), *right.values()]):
        raise TypeError('tensor dictionary contains a non-array value')
    permitted = set(allowed_empty_extras)
    if permitted & required:
        raise ValueError('required parameters cannot be permitted extras')
    ignored = []
    for k in right.keys() - left.keys():
        v = right[k]
        empty_uint8 = (v['shape'] == [0]
                       and ((v['kind'] == 'torch_tensor' and v['dtype'] == 'torch.uint8')
                            or (v['kind'] == 'numpy_array' and v['dtype'] == '|u1')))
        if k in permitted and empty_uint8:
            ignored.append(k)
    for k in ignored:
        del right[k]
    if left != right:
        raise ValueError('model tensor restore mismatch')
    digest = hashlib.sha256(_json(left).encode('utf-8')).hexdigest()
    return dict(passed=True, tensor_count=len(left), expected_sha256=digest,
                actual_sha256=digest, ignored_empty_extras=sorted(ignored))
