"""Native runner route hooks using real game reset/step; no model or GPU."""
import asyncio
from unittest.mock import patch

import pytest
from pacman_recipe.level1.episode import PacmanEpisodeRunner, ModelTurn
from pacman_recipe.level1.trajectories import audit_trajectory
from pacman_env.latest_edward import LatestEdwardPlanner
from pacman_recipe.level1.level1_dataset import make_episode_row
from pacman_env.env import PygamePacmanEnv


class FakeObjectiveTokenizer:
    def encode(self, text, **_): return [ord(c) for c in text]
    def decode(self, token_ids, **_): return ''.join(chr(i) for i in token_ids)


class TwoStepEnv(PygamePacmanEnv):
    def step(self, action):
        image,reward,_,_,info=super().step(action)
        done=int(info['step'])>=2
        info=dict(info,terminated=done,truncated=False,
                  terminal_reason='test_complete' if done else None)
        return image,reward,done,False,info


class ChoosingRunner(PacmanEpisodeRunner):
    async def _call_model(self, messages, **options):
        code = options['objective_constraint'].rendered_choices[0]
        self.calls.append(code)
        return ModelTurn(code, 'choice-'+str(len(self.calls)), messages)


class TracedPlanner(LatestEdwardPlanner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trace=[]
        TracedPlanner.instance=self
    def bind(self, option, state):
        self.trace.append(('bind', option.option_id))
        return super().bind(option,state)
    def before_action(self, action):
        self.trace.append(('before',action))
        return super().before_action(action)
    def record_action(self, action):
        self.trace.append(('record',action))
        return super().record_action(action)


def run(boundaries=None, **options):
    runner=ChoosingRunner(env_factory=TwoStepEnv, tokenizer=FakeObjectiveTokenizer(),
        edward_options=True, image_prompt_style='live_state_v3', **options)
    runner.calls=[]
    if boundaries is not None: runner.decision_boundary_sink=boundaries.append
    asyncio.run(runner.run(make_episode_row(1,split='test')))
    return runner.last_episode


def test_latest_default_native_binding_and_auditable_route_menu():
    boundaries=[]
    with patch('pacman_env.latest_edward.LatestEdwardPlanner',TracedPlanner):
        ep=run(boundaries=boundaries)
    assert boundaries[0]['context']['planner']['edward_harness']=='latest'
    assert boundaries[0]['context']['planner']['route_visits']
    assert ep['decoding']['edward_harness']=='latest'
    assert ep['decoding']['edward_shuffle_options'] is False
    assert ep['edward_route_audit']['registry']
    assert ep['edward_route_audit']['events']
    assert ep['edward_route_audit']['lure_enabled'] is False
    trace=TracedPlanner.instance.trace
    assert trace[0][0]=='bind'
    assert sum(t[0]=='before' for t in trace)==len(ep['trajectory'])==2
    assert sum(t[0]=='record' for t in trace)==2
    for i,t in enumerate(trace):
        if t[0]=='before': assert trace[i+1]==('record',t[1])
    for step in ep['trajectory']:
        ctx=step['observation_context']
        assert 'route_menu' in ctx
        for ghost in ctx['ghosts']:
            assert not {'path_remaining','path_target','currentPath','rng'} & ghost.keys()
        if step['model_called']: assert ctx['route_menu']
        else: assert ctx['route_menu']=={}
    audit_trajectory(ep)


def test_explicit_legacy_uses_original_planner_no_route_hooks():
    with patch('pacman_env.latest_edward.LatestEdwardPlanner',side_effect=AssertionError('latest instantiated')):
        ep=run(edward_harness='legacy')
    assert ep['decoding']['edward_harness']=='legacy'
    assert all('route_menu' not in t['observation_context'] for t in ep['trajectory'])
    audit_trajectory(ep)


def test_invalid_selector_and_old_contract_fail_before_rollout():
    with pytest.raises(ValueError,match='edward_harness'):
        run(edward_harness='typo')
    with pytest.raises(ValueError,match='edward_harness'):
        run(recipe_contract={'harness':{'edward_harness':'legacy'}})


def test_latest_restart_restores_only_actual_pellets_and_checks_boundary():
    from pacman_recipe.level1.episode import _latest_snapshot_remaining
    from pacman_env.env.state import Position
    from pacman_env.actions import Action
    env=PygamePacmanEnv()
    try:
        env.reset(seed=1)
        planner=LatestEdwardPlanner()
        first=env.snapshot()
        initial=_latest_snapshot_remaining(planner,first)
        # Real game advance; a saved-state restart must not restore eaten pellets.
        for _ in range(4):
            snap=env.snapshot()
            legal=snap['open']
            env.step(Action(legal[0]))
        snap=env.snapshot()
        actual={Position(*r) for f in ('normal_pellet_positions','power_pellet_positions') for r in snap[f]}
        assert _latest_snapshot_remaining(planner,snap)==actual
        assert actual <= initial
        assert _latest_snapshot_remaining(planner,snap,recorded=actual)==actual
        with pytest.raises(ValueError,match='differs from snapshot'):
            _latest_snapshot_remaining(planner,snap,recorded=actual ^ {next(iter(initial))})
        for field in ('normal_pellet_positions','power_pellet_positions'):
            missing=dict(snap);del missing[field]
            with pytest.raises(ValueError,match='requires snapshot'):
                _latest_snapshot_remaining(planner,missing)
    finally:
        env.close()


def test_latest_native_boundary_restores_exact_decision_suffix(tmp_path):
    import hashlib
    import json

    class TwelveStepEnv(PygamePacmanEnv):
        def step(self, action):
            image,reward,terminated,truncated,info=super().step(action)
            done=terminated or int(info['step'])>=12
            info=dict(info,terminated=done,truncated=truncated,
                      terminal_reason=info['terminal_reason'] if terminated else 'test_complete' if done else None)
            return image,reward,done,truncated,info

    # Inference replay uses the native runner's injectable row validator; the
    # training dataset validator intentionally restricts restart learning to primitive.
    from pacman_recipe.level1.level1_dataset import validate_episode_row
    def replay_row_validator(data):
        validate_episode_row({k:v for k,v in data.items() if not k.startswith('restart_state_')})
    row=make_episode_row(1,split='test',seed=28)
    def play(data, harness='latest'):
        boundaries=[]
        runner=ChoosingRunner(env_factory=TwelveStepEnv,tokenizer=FakeObjectiveTokenizer(),
            row_validator=replay_row_validator,
            edward_options=True,image_prompt_style='live_state_v3',edward_harness=harness)
        runner.calls=[]
        runner.decision_boundary_sink=boundaries.append
        asyncio.run(runner.run(data))
        return runner.last_episode,boundaries

    original,boundaries=play(row)
    assert len(boundaries)>=2
    k=len(boundaries)//2
    boundary=boundaries[k]
    assert boundary['env_state']['payload']['episode']['steps']>0
    saved=dict(boundary['env_state'],boundary_context=boundary['context'])
    path=tmp_path/'latest-boundary.json'
    path.write_text(json.dumps(saved))
    restart=dict(row,restart_state_path=str(path),restart_state_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 restart_state_id='latest-native-boundary')
    resumed,resumed_boundaries=play(restart)
    assert [b['context']['verify'] for b in resumed_boundaries]==[b['context']['verify'] for b in boundaries[k:]]
    offset=boundary['env_state']['payload']['episode']['steps']
    assert [s['action'] for s in resumed['trajectory']]==[s['action'] for s in original['trajectory'][offset:]]
    assert resumed['normal_pellets_remaining']==original['normal_pellets_remaining']
    assert resumed['terminal_reason']==original['terminal_reason']
    audit_trajectory(resumed)
    with pytest.raises(ValueError,match='same Edward harness'):
        play(restart,harness='legacy')


def test_shuffle_opt_out_retains_fixed_codes_and_audit_rejects_tamper():
    from copy import deepcopy
    from pacman_recipe.level1.token_constraints import OPTION_CODE_BY_ID
    ep=run(edward_shuffle_options=False)
    for step in ep['trajectory']:
        assert step['option_shuffle_key'] is None
        assert all(OPTION_CODE_BY_ID[oid]==code for code,oid in step['option_code_map'].items())
    audit_trajectory(ep)
    shuffled=run(edward_shuffle_options=True)
    bad=deepcopy(shuffled)
    bad['trajectory'][0]['option_shuffle_key']='tampered-key'
    with pytest.raises(ValueError,match='option code'):
        audit_trajectory(bad)
