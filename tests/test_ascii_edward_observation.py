from copy import deepcopy
from types import SimpleNamespace
import ast
import json
import subprocess

import pytest
from pacman_recipe.level1.ascii_observation import render_ascii_map
from pacman_recipe.level1 import prompts


def example():
    level = SimpleNamespace(height=2, width=6, tiles=((100,1,20,2,3,0),(0,0,0,0,0,0)))
    state = dict(height=2,width=6,row=1,col=0,normal_pellet_positions=[[0,3],[1,1]],
                 power_pellet_positions=[[1,1]],ghosts=[
                     dict(id=0,state='eyes',position=[1,1]),
                     dict(id=1,state='vulnerable',position=[1,1]),
                     dict(id=2,state='normal',position=[1,1]),
                     dict(id=3,state='normal',position=[1,0])])
    return level,state


def test_live_cells_overlap_order_and_trailing_spaces():
    level,state=example()
    assert render_ascii_map(level,state).splitlines()[1:] == ['  012345','0 #-=.  ','1 PG    ']
    state['ghosts'].reverse()
    assert render_ascii_map(level,state).splitlines()[-1] == '1 PG    '
    state['normal_pellet_positions']=[]
    assert render_ascii_map(level,state).splitlines()[2] == '0 #-=   '
    del state['power_pellet_positions']
    with pytest.raises(KeyError): render_ascii_map(level,state)


def test_future_and_hidden_state_cannot_change_map_or_json():
    level,state=example()
    before=render_ascii_map(level,state)
    for ghost in state['ghosts']:
        ghost.update(direction='R',path_remaining=[[99,99]],velocity=[99,99])
    state['fruit']={'position':[0,5], 'future_path':[[1,5]]}
    assert render_ascii_map(level,state)==before
    context={'ghosts':state['ghosts']}
    constraint=SimpleNamespace(rendered_choices=[])
    result=prompts.compact_ascii_edward_decision_prompt(context,[],constraint,before)
    assert 'path_remaining' not in result and 'velocity' not in result and 'direction' not in result
    assert before+'\n\n[CURRENT STATE]' in result
    state_line=result.split('[CURRENT STATE]\n',1)[1].splitlines()[0]
    assert json.loads(state_line)['ghosts']==[
        [g['id'],g['state'],g['position']] for g in state['ghosts']]
    assert '"c":' not in state_line


def test_only_approved_system_replacements_and_new_identity():
    system=prompts.ASCII_EDWARD_SYSTEM_PROMPT
    assert 'screenshot' not in system and 'Colors can vary' not in system
    assert 'Map row r and column c equal [r,c]' in system
    old=prompts.prompt_contract_metadata('live_state_v3',edward_options=True)
    new=prompts.prompt_contract_metadata('ascii_edward_v1',edward_options=True)
    assert new['action_protocol']==old['action_protocol']
    assert new['prompt_template_sha256']!=old['prompt_template_sha256']
    assert new['prompt_version']=='edward-ascii-option-code-v2+fixed-map-dynamic-v1'


def test_archived_prompt_constants_and_renderer_source_unchanged():
    from pathlib import Path
    source=Path(prompts.__file__)
    old=subprocess.check_output(['git','show','HEAD:pacman_recipe/level1/prompts.py'],cwd=source.parents[2],text=True)
    previous=ast.parse(old); current=ast.parse(source.read_text())
    names={'SHARED_GAME_RULES','EDWARD_OPTION_CODE_V2_SYSTEM_PROMPT','EDWARD_OPTION_CODE_V2_USER_TEMPLATE',
           'EDWARD_RISK_NOTICE','EDWARD_RISK_USER_SUFFIX','compact_edward_decision_prompt',
           'risk_ranked_edward_decision_prompt','edward_system_prompt','render_edward_decision_prompt'}
    def selected(tree):
        return {getattr(node,'name',None) or node.targets[0].id:ast.dump(node) for node in tree.body
                if (getattr(node,'name',None) in names or isinstance(node,ast.Assign) and
                    isinstance(node.targets[0],ast.Name) and node.targets[0].id in names)}
    assert selected(previous)==selected(current)


def test_real_pygame_reset_step_save_restore_ascii():
    from pacman_env.env import PygamePacmanEnv,PygamePacmanEnvConfig,load_bundled_level
    env=PygamePacmanEnv(PygamePacmanEnvConfig(max_steps=32))
    try:
        env.reset(seed=3)
        level=load_bundled_level()
        initial=env.snapshot()
        first=render_ascii_map(level,initial)
        assert len(first.splitlines())==level.height+2
        assert all(len(line)==3+level.width for line in first.splitlines()[2:])
        saved=env.save_state()
        env.step(initial['open'][0])
        after=env.snapshot()
        observed=render_ascii_map(level,after)
        env.restore_state(saved)
        assert render_ascii_map(level,env.snapshot())==first
        env.step(initial['open'][0])
        assert render_ascii_map(level,env.snapshot())==observed
    finally:
        env.close()


def test_risk_suffix_and_candidate_bytes_preserved():
    level,state=example()
    board=render_ascii_map(level,state)
    candidate=SimpleNamespace(option_id='A0',strategy='RISK_FALLBACK',target=(1,2),
        first_action='R',route_distance=1,commit_moves=1,safety_margin=0,
        future_safe_exits=1,entity_id=None,risk=dict(rank=1,motion='unknown',
        ghost_clearance=2,route_margin=0,safe_next_cells=1,dead_end=False,reverse=False))
    constraint=SimpleNamespace(rendered_choices=['A'],code_for_option=lambda _: 'A')
    old=prompts.render_edward_decision_prompt({},[candidate],constraint,fallback_mode='risk_ranked')
    new=prompts.compact_ascii_edward_decision_prompt({},[candidate],constraint,board,fallback_mode='risk_ranked')
    old_state=json.loads(old.splitlines()[2])
    new_state=json.loads(new.split('[CURRENT STATE]\n',1)[1].splitlines()[0])
    candidate_line=new.split('[CANDIDATE OBJECTIVES]\n',1)[1].splitlines()[0]
    assert new_state=={k:v for k,v in old_state.items() if k!='c'}
    assert json.loads(candidate_line)==old_state['c'][0]
    assert new[new.index(' RISK_FALLBACK is not safety-approved.'):]==old[
        old.index(' RISK_FALLBACK is not safety-approved.'):]
    assert 'motion=clear_estimate/unknown/collision_predicted' in new


def test_sections_multiple_candidate_rows_and_layout_fingerprint():
    level,state=example()
    board=render_ascii_map(level,state)
    context=dict(pacman_position=[1,0],facing='R',pellets_remaining=3,
        maze_size=[2,6],ghosts=state['ghosts'],edible_ticks=16,last_action='U')
    candidates=[SimpleNamespace(option_id=f'C{i}',strategy='COLLECT',target=(0,i),
        first_action='R',route_distance=i,commit_moves=1,safety_margin=2,
        future_safe_exits=3,entity_id=None) for i in (1,2)]
    constraint=SimpleNamespace(rendered_choices=['B','C'],
        code_for_option=lambda x: {'C1':'B','C2':'C'}[x])
    rendered=prompts.compact_ascii_edward_decision_prompt(context,candidates,constraint,board)
    labels=['[CURRENT MAP]','[CURRENT STATE]','[CANDIDATE OBJECTIVES]','[OUTPUT]']
    assert [rendered.index(label) for label in labels]==sorted(rendered.index(label) for label in labels)
    assert board+'\n\n[CURRENT STATE]' in rendered
    rows=rendered.split('[CANDIDATE OBJECTIVES]\n',1)[1].split('\n\n[OUTPUT]',1)[0].splitlines()
    old=prompts.compact_edward_decision_prompt(context,candidates,constraint)
    assert [json.loads(row) for row in rows]==json.loads(old.splitlines()[2])['c']
    assert rendered.endswith('code from [B,C]; nothing else.')
    metadata=prompts.prompt_contract_metadata('ascii_edward_v1',edward_options=True)
    assert metadata['user_prompt_template_sha256']==prompts.text_sha256(prompts.ASCII_EDWARD_USER_TEMPLATE)
    assert metadata['user_prompt_template_sha256']!=prompts.text_sha256(
        prompts.EDWARD_OPTION_CODE_V2_USER_TEMPLATE.replace('{decision_state}','{ascii_map}\n{decision_state}',1))


def test_ascii_and_vlm_place_identical_fixed_explanations_before_observation():
    assert prompts.ASCII_EDWARD_FIXED_PREFIX==prompts.VISION_EDWARD_FIXED_PREFIX.replace(
        'screenshot','map').replace('[CURRENT IMAGE]','[CURRENT MAP]')
    before,after=prompts.ASCII_EDWARD_USER_TEMPLATE.split('{ascii_map}',1)
    assert before==prompts.ASCII_EDWARD_FIXED_PREFIX
    for explanation in ('Keys: p=','Candidate row:','first_action=','Metrics: distance='):
        assert explanation in before and explanation not in after
        assert explanation in prompts.VISION_EDWARD_FIXED_PREFIX
    assert after.startswith('\n\n[CURRENT STATE]\n{decision_state}')
