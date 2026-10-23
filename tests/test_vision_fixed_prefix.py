import copy
import pytest

from slime_pacman.fixed_prefix import declare_prefix, vision_cache_contract, validate_vision_request


def contract(tmp_path):
    rows = [list(range(524)) + [9999, x] for x in (800, 801)]
    prefix = declare_prefix(rows, fixed_boundary_lengths=[524, 524])
    patch = tmp_path / 'guard.py'
    patch.write_text('fixed node guard')
    return vision_cache_contract(prefix, patch_path=patch,
        startup_args=['--max-running-requests', '8', '--context-length', '8192',
                      '--max-total-tokens', '65536'],
        canary_ids=rows[0], vision_start_token_id=9999), rows


def test_vision_rederives_prefix_and_accepts_different_images(tmp_path):
    c, rows = contract(tmp_path)
    assert c['prefix']['length'] == 512
    assert validate_vision_request(c, rows[0]) == validate_vision_request(c, rows[1])


@pytest.mark.parametrize('kind', ['prefix', 'image_boundary', 'no_image', 'schema'])
def test_refuse_wrong_identity_or_image_inside_cache(tmp_path, kind):
    c, rows = contract(tmp_path)
    c = copy.deepcopy(c)
    ids = rows[0][:]
    if kind == 'prefix': ids[0] = 700
    if kind == 'image_boundary': ids[500] = 9999
    if kind == 'no_image': ids.remove(9999)
    if kind == 'schema': c['schema'] = 'pacman-fixed-cache-runtime-v1'
    with pytest.raises(ValueError): validate_vision_request(c, ids)


def test_canary_never_enters_image(tmp_path):
    c, rows = contract(tmp_path)
    assert len(c['prefix']['token_ids']) + 1 < rows[0].index(9999)
    patch = tmp_path / 'guard.py'
    with pytest.raises(ValueError, match='startup'):
        vision_cache_contract(c['prefix'], patch_path=patch,
            startup_args=['--max-running-requests', '32'],
            canary_ids=rows[0], vision_start_token_id=9999)
