import random

import numpy as np
import pytest

from slime_pacman.map_training.restore_audit import (
    assert_state_restored, assert_tensor_dict_restored, normalize_state, state_fingerprint,
)


def test_full_rng_and_scheduler_state_change_is_detected():
    rng = random.Random(42)
    expected = {'rng': rng.getstate(), 'scheduler': {'step': 12, 'lr': 5e-7},
                'numpy': np.random.RandomState(42).get_state()}
    actual = {'numpy': np.random.RandomState(42).get_state(),
              'scheduler': {'lr': 5e-7, 'step': 12}, 'rng': rng.getstate()}
    assert assert_state_restored(expected, actual, label='resume')['passed']
    rng.random()
    actual['rng'] = rng.getstate()
    with pytest.raises(ValueError, match='resume restore mismatch'):
        assert_state_restored(expected, actual, label='resume')


def test_typed_keys_and_container_types_do_not_collide():
    assert state_fingerprint({1: [2]}) != state_fingerprint({'1': [2]})
    assert state_fingerprint([1, 2]) != state_fingerprint((1, 2))
    with pytest.raises(TypeError, match='unsupported'):
        normalize_state(object())


def test_tensor_dict_detects_byte_dtype_shape_and_parameter_omissions():
    a = np.array([1, 2], dtype=np.int64)
    proof = assert_tensor_dict_restored({'weight': a}, {'weight': a.copy()},
                                      required_parameters=['weight'])
    assert proof['tensor_count'] == 1 and proof['expected_sha256'] == proof['actual_sha256']
    for changed in [np.array([1, 3], dtype=np.int64), a.view(np.uint64), a.reshape(1, 2)]:
        with pytest.raises(ValueError, match='tensor restore mismatch'):
            assert_tensor_dict_restored({'weight': a}, {'weight': changed})
    with pytest.raises(ValueError, match='required'):
        assert_tensor_dict_restored({'buffer': a}, {'buffer': a}, required_parameters=['weight'])


def test_empty_extra_allowlist_cannot_hide_parameter_or_nonempty_tensor():
    a = np.array([1.0], dtype=np.float32)
    actual = {'weight': a, 'extra': np.empty(0, dtype=np.uint8)}
    with pytest.raises(ValueError, match='tensor restore mismatch'):
        assert_tensor_dict_restored({'weight': a}, actual)
    p = assert_tensor_dict_restored({'weight': a}, actual, allowed_empty_extras=['extra'])
    assert p['ignored_empty_extras'] == ['extra']
    for extra in [np.zeros(1, dtype=np.uint8), np.empty(0, dtype=np.float32),
                  np.empty((0, 2), dtype=np.uint8)]:
        with pytest.raises(ValueError, match='tensor restore mismatch'):
            assert_tensor_dict_restored({'weight': a}, {'weight': a, 'extra': extra},
                                        allowed_empty_extras=['extra'])
    with pytest.raises(ValueError, match='required'):
        assert_tensor_dict_restored({'weight': a}, {'weight': a},
                                    required_parameters=['weight'], allowed_empty_extras=['weight'])


def test_array_order_normalization_retains_values_not_storage_strides():
    a = np.arange(12).reshape(3, 4).T
    assert not a.flags.c_contiguous
    assert assert_tensor_dict_restored({'a': a}, {'a': a.copy(order='C')})['passed']


def test_object_arrays_nonfinite_scalars_and_non_arrays_fail_closed():
    with pytest.raises(ValueError, match='object arrays'):
        state_fingerprint(np.array([object()], dtype=object))
    with pytest.raises(ValueError, match='nonfinite'):
        state_fingerprint({'lr': float('nan')})
    with pytest.raises(ValueError, match='structured'):
        state_fingerprint(np.zeros(1, dtype=[('moment', 'f4')]))
    with pytest.raises(TypeError, match='non-array'):
        assert_tensor_dict_restored({'weight': 1}, {'weight': 1})


def test_numpy_scalar_dtype_and_python_scalar_remain_distinct():
    assert state_fingerprint(np.int32(1)) != state_fingerprint(np.int64(1))
    assert state_fingerprint(np.int32(1)) != state_fingerprint(1)


def test_torch_bfloat16_and_generator_when_dependency_is_available():
    torch = pytest.importorskip('torch')
    a = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    assert assert_tensor_dict_restored({'weight': a}, {'weight': a.clone()})['passed']
    g = torch.Generator().manual_seed(11)
    h = torch.Generator().manual_seed(11)
    assert assert_state_restored(g, h)['passed']
    torch.rand(1, generator=h)
    with pytest.raises(ValueError, match='restore mismatch'):
        assert_state_restored(g, h)
