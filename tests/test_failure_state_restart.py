import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime_pacman.map_training.failure_state_restart import (
    configured_policy, make_sampler, RestartBank, FailureStateRestartSampler,
)
from slime_pacman.map_training.checkpoints import atomic_json, sha256
from test_map_training_rollout import catalog


def config():
    c = catalog()
    c += [dict(deepcopy(t), task_id='higher-'+t['task_id'], tier=1,
               layout_sha256=f'{100+i:064x}') for i,t in enumerate(c)]
    return dict(catalog=c, seed=20261009, min_tier=0, max_tier=1,
                policy=dict(threshold=.8,window_batches=3,min_games=384,consecutive_checks=2,min_maps=8))


def populate(sampler, tmp_path):
    entries=[]
    for tier in [0,1]:
        for n in range(4,33):
            t=next(t for t in sampler.catalog if t['tier']==tier)
            p=tmp_path/f'state-{tier}-{n}.json';p.write_text('{}')
            rec=deepcopy(t['episode_record']);rec['environment']['seed']=tier+90
            rec['restart']=dict(path=str(p),sha256=sha256(p),id=f'failure-{tier}-{n}')
            entries.append(dict(id=f'{tier}-{n}',task_id=t['task_id'],tier=tier,split='train',
                                lookback=n,bin_id=0 if n<8 else 1 if n<16 else 2,
                                admitted_update=0,record=rec))
    atomic_json(sampler.restart_bank.root/'latest.json',dict(entries=entries,outcomes={},policy=sampler.restart_bank.policy))


def test_default_and_explicit_off_identity(tmp_path):
    c=config();s=make_sampler(c,tmp_path)
    assert isinstance(s,FailureStateRestartSampler)
    assert configured_policy({})['fraction']==.8
    assert configured_policy({'failure_state_restart':False}) is None
    assert configured_policy({'failure_state_restart':None}) is None
    old=make_sampler(dict(c,failure_state_restart=False),tmp_path)
    assert old.controller.identity!=s.controller.identity
    assert not hasattr(old,'restart_bank')
    with pytest.raises(ValueError):configured_policy({'failure_state_restart':{'fraction':.5}})
    edward=config();edward['catalog'][0]['episode_record']['prompt']={'action_protocol':'edward-option-code-v1'}
    assert configured_policy(edward) is None
    with pytest.raises(ValueError,match='Primitive'):
        configured_policy(dict(edward,failure_state_restart=True))


def test_same_tier_ratios_determinism_and_retry(tmp_path):
    s=make_sampler(config(),tmp_path);populate(s,tmp_path)
    s.controller.state['tier']=1
    counts=dict(current=[0,0],replay=[0,0])
    offsets=set()
    for u in range(5):
        s.controller.state['last_update']=u-1
        p=s.plan(u);assert p==s.plan(u)
        assert sum(g['task']['tier']==1 for g in p['groups'])==13
        for g in p['groups']:
            counts[g['start_class']][0]+=g['start']=='restart';counts[g['start_class']][1]+=1
            if g['start']=='restart':
                offsets.add(g['lookback']);assert 4<=g['lookback']<=32
                assert g['environment_seed']==g['task']['episode_record']['environment']['seed']
                assert len(set(g['sampling_seeds']))==12
                q=s.replacement(g,1,p['groups'],update=u)
                assert q['task']['tier']==g['task']['tier'] and q['start']=='restart'
                assert set(q['sampling_seeds']).isdisjoint(g['sampling_seeds'])
    assert counts==dict(current=[52,65],replay=[12,15])
    assert any(n<8 for n in offsets) and any(n>=16 for n in offsets)


def test_empty_bank_fallback_and_no_validation_binding(tmp_path):
    from slime_pacman.formal_rollout import validate_binding
    s=make_sampler(config(),tmp_path)
    p=s.plan(0);assert all(g['start']=='initial' for g in p['groups'])
    assert any(g.get('restart_fallback')=='no_same_tier_state' for g in p['groups'])
    populate(s,tmp_path);p=s.plan(0)
    g=next(g for g in p['groups'] if g['start']=='restart')
    validate_binding(g['task'],allow_restart=True)
    with pytest.raises(ValueError):validate_binding(g['task'])
    t=deepcopy(g['task']);t['split']='validation'
    with pytest.raises(ValueError):validate_binding(t,allow_restart=True)


def test_promotion_excludes_restart_successes(tmp_path):
    s=make_sampler(config(),tmp_path);populate(s,tmp_path);p=s.plan(0)
    es=[dict(group=g['group'],replica=r,task_id=g['task']['task_id'],won=g['start']=='restart')
        for g in p['groups'] for r in range(12)]
    s.complete(0,es,plan=p)
    assert s.controller.state['window'][-1]['wins']==0
    assert s.controller.state['window'][-1]['games']==12*sum(g['start']=='initial' for g in p['groups'])


def episodes_for(plan, won):
    return [dict(group=g['group'], replica=r, task_id=g['task']['task_id'], won=won(g))
            for g in plan['groups'] for r in range(12)]


def test_restart_bank_does_not_make_promotion_unreachable(tmp_path):
    s=make_sampler(config(),tmp_path);populate(s,tmp_path)
    original_policy=s.controller.policy
    for u in range(30):
        p=s.plan(u)
        event=s.complete(u,episodes_for(p,lambda g:g['start']=='initial'),plan=p)
        assert s.controller.policy is original_policy
        if event['passed']:
            assert event['games']>=3*16*12 and event['maps']>=8
        if s.controller.state['tier']==1:
            break
    assert s.controller.state['tier']==1
    assert u>=15  # 80% restart successes/failures cannot substitute for normal games.


def test_current_tier_promotion_excludes_lower_tier_replay(tmp_path):
    c=config();c['max_tier']=2
    c['catalog'] += [dict(deepcopy(t), task_id='highest-'+t['task_id'], tier=2,
                          layout_sha256=f'{200+i:064x}')
                     for i,t in enumerate(c['catalog'][:16])]
    s=make_sampler(c,tmp_path);populate(s,tmp_path);s.controller.state['tier']=1
    for u in range(30):
        p=s.plan(u)
        event=s.complete(u,episodes_for(p,lambda g:g['start']=='initial' and g['task']['tier']==1),plan=p)
        if event['passed']:
            assert event['rate']==1 and event['games']>=3*13*12
        if s.controller.state['tier']==2:
            break
    assert s.controller.state['tier']==2


def test_empty_restart_bank_keeps_normal_promotion_timing(tmp_path):
    c=config();s=make_sampler(c,tmp_path/'restart')
    old=make_sampler(dict(c,failure_state_restart=False),tmp_path/'disabled')
    for u in range(4):
        p=s.plan(u);q=old.plan(u)
        assert s.complete(u,episodes_for(p,lambda g:True),plan=p)==old.complete(
            u,episodes_for(q,lambda g:True),plan=q)
    assert s.controller.state['tier']==old.controller.state['tier']==1


def test_long_normal_window_survives_controller_restore(tmp_path):
    s=make_sampler(config(),tmp_path/'first');populate(s,tmp_path)
    for u in range(8):
        p=s.plan(u);s.complete(u,episodes_for(p,lambda g:True),plan=p)
    assert len(s.controller.state['window'])>s.controller.policy.window_batches
    resumed=make_sampler(config(),tmp_path/'resumed');populate(resumed,tmp_path)
    resumed.restore(s.snapshot(),completed_update=7,bank_version=s.controller.state['bank_version'])
    for u in range(8,18):
        p=s.plan(u);q=resumed.plan(u)
        assert s.complete(u,episodes_for(p,lambda g:True),plan=p)==resumed.complete(
            u,episodes_for(q,lambda g:True),plan=q)
    assert resumed.snapshot()==s.snapshot()


def test_complete_default_plan_and_failed_batch_are_transactional(tmp_path):
    s=make_sampler(config(),tmp_path);populate(s,tmp_path)
    p=s.plan(0);before=s.snapshot();policy=s.controller.policy
    episodes=episodes_for(p,lambda g:True)
    normal_group=next(g['group'] for g in p['groups'] if g['start']=='initial')
    missing=next(e for e in episodes if e['group']==normal_group)
    with pytest.raises(ValueError,match='incomplete'):
        s.complete(0,[e for e in episodes if e is not missing])
    assert s.snapshot()==before and s.controller.policy is policy
    s.complete(0,episodes_for(p,lambda g:True))
    assert s.controller.state['last_update']==0


def test_normal_window_can_reach_larger_explicit_min_games(tmp_path):
    c=config();c['policy']['window_batches']=1
    s=make_sampler(c,tmp_path);populate(s,tmp_path)
    for u in range(30):
        p=s.plan(u);event=s.complete(u,episodes_for(p,lambda g:True),plan=p)
        if event['passed']:
            assert event['games']>=c['policy']['min_games']
        if s.controller.state['tier']==1:
            break
    assert s.controller.state['tier']==1


def test_checkpoint_bank_restores_to_new_directory_and_rejects_tampering(tmp_path):
    s=make_sampler(config(),tmp_path/'run');populate(s,tmp_path)
    root=tmp_path/'checkpoints';files=s.restart_bank.checkpoint(root,3)
    restored=RestartBank(tmp_path/'new-run',config(),configured_policy({}))
    restored.restore(root,3)
    assert len(restored.read()['entries'])==58
    assert all(Path(e['record']['restart']['path']).is_relative_to(restored.root) for e in restored.read()['entries'])
    files[-1].write_text('changed')
    with pytest.raises(ValueError,match='changed'):restored.restore(root,3)


def test_bank_expires_without_changing_normal_fallback(tmp_path):
    s=make_sampler(config(),tmp_path);populate(s,tmp_path)
    s.restart_bank.refresh([],13)
    assert s.restart_bank.read()['entries']==[]
    p=s.plan(0)
    assert all(g['start']=='initial' for g in p['groups'])


def test_native_extractor_rejects_validation_before_environment_import(tmp_path):
    from slime_pacman.map_training.failure_state_snapshots import extract
    p=tmp_path/'bad.json';p.write_text(json.dumps(dict(split='validation')))
    with pytest.raises(AssertionError):extract((str(p),config(),str(tmp_path),configured_policy({})))


def test_worker_reward_preserves_absolute_budget(monkeypatch):
    from slime_pacman.map_training import workers
    import sys
    payload=dict(won=True,terminal_reason='all_normal_pellets',normal_pellets_remaining=0,
                 episode_life_mode='original_three_lives',death_count=2,max_steps=512,steps=100,
                 restart_state=dict(source_step=300,death_count=1,remaining_budget=212))
    result=SimpleNamespace(trajectory=dict(episode=payload),reward=99.,success_speed_bonus=.05,success_life_bonus=.1)
    rollout=SimpleNamespace(current_sources=lambda:{},collect_episode_in_worker=lambda *a:result)
    monkeypatch.setitem(sys.modules,'slime_pacman.rollout',rollout)
    # Worker invalidates a cached source function before running the native collector.
    rollout.current_sources.cache_clear=lambda:None
    workers.collect_task(dict(restart=True,source_revisions={}),'.','endpoint','.','config','1',{})
    assert result.reward==pytest.approx(1+.1*(1-2/3)+.05*(1-400/512))
    assert payload['restart_state']['source_step']==300 and payload['steps']==100


def test_real_native_prefix_snapshots_and_suffix_parity(tmp_path, monkeypatch):
    import random
    from pacman_env.env.pygame_environment import PygamePacmanEnv, PygamePacmanEnvConfig
    from pacman_recipe.level1.contracts import make_episode_record
    from pacman_recipe.level1.ascii_observation_spaced import render_ascii_map_spaced
    from slime_pacman.map_training.maps import load_runtime_level
    from slime_pacman.map_training.failure_state_snapshots import extract
    from slime_pacman.map_training import workers
    repo=Path(__file__).resolve().parents[1];game=repo.parent/'pacman-python'
    # Only bypass Torch's CPU-thread setter; the real native environment is used throughout.
    def cpu_init(root, ghosts):
        monkeypatch.setenv('PACMAN_PYTHON_ROOT',str(root))
        monkeypatch.setenv('SMALL_MAZE_GHOST_COUNT',str(ghosts))
    monkeypatch.setattr(workers,'worker_init',cpu_init)
    cpu_init(game,4)
    record=make_episode_record(37,split='train',recipe_root=repo,game_root=game,
        backend_root=repo.parent/'slime',max_steps=512,observation_mode='ascii',
        ascii_map_format='spaced',action_mode='primitive',edward_fallback_mode='refuse',
        episode_life_mode='original_three_lives')
    level=load_runtime_level(1,game);ts=[];rng=random.Random(9)
    cfg=PygamePacmanEnvConfig(pacman_python_root=str(game),max_steps=512,episode_life_mode='original_three_lives')
    with PygamePacmanEnv(cfg) as env:
        env.reset(seed=37)
        for i in range(512):
            snap=env.snapshot();action=rng.choice(snap['open'])
            context=dict(pacman_position=[snap['row'],snap['col']],ghosts=snap['ghosts'],
                         open_actions=snap['open'],ascii_map=render_ascii_map_spaced(level,snap),
                         recent_actions=[t['action'] for t in ts[-4:]],
                         recent_positions=[t['observation_context']['pacman_position'] for t in ts[-4:]])
            _,_,term,trunc,info=env.step(action)
            ts.append(dict(action=action,observation_context=context,terminated=term,truncated=trunc,
                           **{k:info[k] for k in ['death','death_count','normal_pellets_remaining','score','logic_frame_events','atomic_substeps']}))
            if info['death'] and i>=32:break
            if term or trunc:break
    assert any(t['death'] for t in ts),'native fixture must include a death'
    task=dict(task_id='native',split='train',tier=1,episode_record=record)
    x=dict(split='train',task_id='native',tier=1,update=0,record=record,
           map_context=dict(game_root=str(game),ghosts=4),trajectory=dict(episode=dict(trajectory=ts)))
    p=tmp_path/'episode.json';p.write_text(json.dumps(x))
    entries=extract((str(p),dict(catalog=[task]),str(tmp_path/'bank'),configured_policy({})))
    assert entries and all(4<=e['lookback']<=32 for e in entries)
    # A death shortly after a previous death has fewer than 32 eligible offsets.
    # Never cross that previous death merely to fill a bucket.
    entry=max(entries,key=lambda e:e['lookback'])
    state=json.loads(Path(entry['record']['restart']['path']).read_text())
    assert state['primitive_history']['recent_actions']==ts[entry['source_step']]['observation_context']['recent_actions']
    with PygamePacmanEnv(cfg) as env:
        env.restore_state(state)
        assert env.save_state()['sha256']==state['sha256']
        for t in ts[entry['source_step']:]:
            _,_,term,trunc,info=env.step(t['action'])
            assert info['death_count']==t['death_count'] and info['atomic_substeps']==t['atomic_substeps']
            assert term==t['terminated'] and trunc==t['truncated']
