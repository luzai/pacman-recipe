import copy
import pytest

from pacman_recipe.experimental.engine_assisted.training_contract import (
    batch_plan, canonical_sha, loss_episodes, sample_indices, terminal_reward,
    validate_manifest, validate_protocol,
)
from pacman_recipe.experimental.engine_assisted.curriculum import PromotionWindow, normal_start_counts
from pacman_recipe.experimental.engine_assisted.evaluation import evaluation_key, periodic_updates, validate_checkpoint


def protocol():
    return dict(harness_kind='engine_truth_assisted', training_life_mode='single_death', variant='engine_single',
                groups_per_batch=16, samples_per_group=12, attempts_per_batch=192,
                loss_denominator=192, horizon=512, max_batches=200,
                replacement_sampling=False, fault_reward=None,
                target_inventory_sha256='a'*64, paired_task_schedule_sha256='b'*64,
                parent_game_sha256='c'*64, frozen_latest_sha256='d'*64,
                train_topology_ids=['train-' + str(i) for i in range(16)],
                validation_topology_ids=['validation'], test_topology_ids=['test'],
                success_reward=1.1, success_speed_bonus=.05)


def episodes(p):
    rows = []
    for group in range(16):
        restart = group >= 12
        source = 8 if restart else 0
        for sample in range(12):
            won = sample % 2 == 0
            row = dict(episode_id=f'g{group}s{sample}', protocol_sha256=canonical_sha(p),
                       batch=0, weight_version='language0', adapter_sha256='e'*64,
                       life_mode='single_death', harness_kind='engine_truth_assisted', split='train', stage='train', training_variant='engine_single',
                       paired_task_schedule_sha256=p['paired_task_schedule_sha256'],
                       topology_id='train-' + str(group), group=group, sample=sample,
                       fault=False, start_class='restart' if restart else 'normal', source_step=source,
                       steps=100, normal_initial_state=not restart, failure_state_restart=restart,
                       fault_code=None, trainable=True, full_native_outcome=True,
                       game_won=won, deaths=0 if won else 1, source_state_sha256='f'*64,
                       tier=0, game_seed=group)
            row['reward'] = terminal_reward(won=won, deaths=row['deaths'], steps=100, source_step=source,
                                             horizon=512, success_reward=1.1, speed_bonus=.05)
            rows.append(row)
    return rows


def plan(rows, p):
    return batch_plan(rows, 0, canonical_sha(p), 'language0', 'e'*64, p)


def test_fault_masks_original_group_and_rank_shards_do_not_duplicate():
    p = protocol(); rows = episodes(p)
    rows[0].update(fault=True, fault_code='backend_timeout', reward=None, trainable=False,
                   full_native_outcome=False, game_won=False)
    result = plan(rows, p)
    assert result['loss_denominator'] == 192
    assert result['excluded_group_ids'] == [0]
    assert result['loss_applied_episodes'] == 180
    assert result['advantages_by_episode']['g0s0'] is None
    shards = [{r['episode_id'] for r in loss_episodes(rows, result, rank, 7)} for rank in range(7)]
    assert len(set.union(*shards)) == sum(map(len, shards)) == 180
    rows[0]['reward'] = 0
    with pytest.raises(ValueError, match='null reward'):
        plan(rows, p)


def test_zero_advantage_skips_model_and_entropy_loss():
    p = protocol(); rows = episodes(p)
    for row in rows: row.update(game_won=False, deaths=1, reward=0.)
    result = plan(rows, p)
    assert result['skip_before_model_loss'] and result['loss_applied_episodes'] == 0
    assert not any(g['entropy_applied'] for g in result['group_proofs'])
    assert loss_episodes(rows, result) == []


@pytest.mark.parametrize('field,value', [('topology_id', 'validation'), ('source_state_sha256', '0'*64), ('weight_version', 'language1')])
def test_heldout_mixed_restart_and_wrong_checkpoint_rejected(field, value):
    p = protocol(); rows = episodes(p); rows[0][field] = value
    with pytest.raises(ValueError): plan(rows, p)


def test_protocol_is_explicit_and_supports_original_map_without_new_global_default():
    p = protocol()
    p['train_topology_ids'] = ['original-map']
    rows = episodes(p)
    for row in rows: row['topology_id'] = 'original-map'
    assert plan(rows, p)['attempts'] == 192
    p['test_topology_ids'] = ['original-map']
    with pytest.raises(ValueError, match='disjoint'): validate_protocol(p)


def test_terminal_prefix_budget_and_manifest_binding():
    with pytest.raises(ValueError): terminal_reward(won=False, deaths=0, steps=505, source_step=8, horizon=512, success_reward=1.1, speed_bonus=.05)
    p = protocol(); result = plan(episodes(p), p)
    with pytest.raises(ValueError): validate_manifest({}, result)
    manifest = {key: result[key] for key in ('protocol_sha256', 'weight_version', 'adapter_sha256', 'paired_task_schedule_sha256', 'harness_kind', 'training_variant', 'eligible_group_ids', 'excluded_group_ids', 'fault_episode_ids')}
    manifest['complete_native_private_audit_passed'] = True
    assert validate_manifest(manifest, result)
    manifest['adapter_sha256'] = '0'*64
    with pytest.raises(ValueError): validate_manifest(manifest, result)
    assert sample_indices(100, 'episode', 3, limit=64) == sample_indices(100, 'episode', 3, limit=64)


def promotion_batch(normal=12):
    return [dict(group=g, sample=s, life_mode='single_death', split='train',
                 start_class='normal' if g < normal else 'restart', topology_id=str(g),
                 tier=0, fault=False, reward=1., native_complete=True, won=True)
            for g in range(16) for s in range(12)]


def promotion_window():
    return PromotionWindow(groups_per_batch=16, samples_per_group=12, window_batches=3,
                           minimum_games=384, minimum_maps=8, success_threshold=.8,
                           consecutive_passes=2, max_tier=2)


def test_restart_success_cannot_fill_normal_promotion_window():
    window = promotion_window()
    for _ in range(5): result = window.record(promotion_batch(normal=3))
    assert result['games'] == 108 and not result['ready'] and window.tier == 0
    window = promotion_window()
    for _ in range(3): result = window.record(promotion_batch())
    assert result['passed'] and not result['promoted']
    assert window.record(promotion_batch())['promoted']
    assert window.tier == 1 and window.window == []


def test_fault_group_excluded_from_promotion_without_zero_reward_substitution():
    rows = promotion_batch(); rows[0].update(fault=True, reward=None, native_complete=False)
    assert normal_start_counts(rows, groups_per_batch=16, samples_per_group=12)['games'] == 132
    rows[0]['reward'] = 0
    with pytest.raises(ValueError): normal_start_counts(rows, groups_per_batch=16, samples_per_group=12)


def test_long_evaluation_schedule_and_checkpoint_roots_are_explicit():
    assert periodic_updates(inherited_updates=5, interval=25, final_updates=100) == (30, 55, 80)
    assert periodic_updates(inherited_updates=0, interval=10, final_updates=20) == (10,)
    assert evaluation_key('a'*64, 'original', 'validation') != evaluation_key('a'*64, 'unseen', 'validation')
    cp = dict(checkpoint_path='/owned/checkpoints/cp9.pt', checkpoint_name='cp9.pt', optimizer_updates=9,
              protocol_sha256='p', weight_version='language9', checkpoint_sha256='a'*64, adapter_sha256='b'*64)
    assert validate_checkpoint(cp, root='/owned/checkpoints', protocol_sha256='p', maximum_updates=100)
    cp['checkpoint_path'] = '/owned/checkpoints/../other/cp9.pt'
    with pytest.raises(ValueError): validate_checkpoint(cp, root='/owned/checkpoints', protocol_sha256='p', maximum_updates=100)
