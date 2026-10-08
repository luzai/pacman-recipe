"""Opt-in spaced ASCII map (ascii_edward_spaced_v1): one cell per token, otherwise ascii_edward_v1."""
import asyncio
from types import SimpleNamespace

import pytest

from pacman_recipe.level1 import prompts
from pacman_recipe.level1.ascii_observation import render_ascii_map
from pacman_recipe.level1.ascii_observation_spaced import (
    ASCII_SPACED_MAP_HEADER, render_ascii_map_spaced, space_ascii_map, unspace_ascii_map)
from test_ascii_edward_observation import example


def test_spaced_layout_keeps_every_cell_and_inverts_exactly():
    level, state = example()
    packed = render_ascii_map(level, state)
    spaced = render_ascii_map_spaced(level, state)
    assert spaced.splitlines() == [ASCII_SPACED_MAP_HEADER, '  0 1 2 3 4 5', '0 # - = . _ _', '1 P G _ _ _ _']
    assert unspace_ascii_map(spaced) == packed
    with pytest.raises(ValueError):
        space_ascii_map(packed.replace('#', '_', 1))      # `_` is reserved for empty
    with pytest.raises(ValueError):
        unspace_ascii_map(spaced.replace('# -', '#-', 1))  # not one space per cell


def test_spaced_prompt_differs_only_in_map_and_legend():
    level, state = example()
    context = dict(pacman_position=[1, 0], facing='R', pellets_remaining=3, maze_size=[2, 6],
                   ghosts=state['ghosts'], edible_ticks=16, last_action='U')
    candidates = [SimpleNamespace(option_id=f'C{i}', strategy='COLLECT', target=(0, i), first_action='R',
                                  route_distance=i, commit_moves=1, safety_margin=2, future_safe_exits=3, entity_id=None)
                  for i in (1, 2)]
    constraint = SimpleNamespace(rendered_choices=['B', 'C'], code_for_option=lambda x: {'C1': 'B', 'C2': 'C'}[x])
    for mode in ('refuse', 'risk_ranked'):
        packed = prompts.compact_ascii_edward_decision_prompt(
            context, candidates, constraint, render_ascii_map(level, state), fallback_mode=mode)
        spaced = prompts.ascii_decision_prompt_for(
            'ascii_edward_spaced_v1', context, candidates, constraint, render_ascii_map_spaced(level, state), fallback_mode=mode)
        assert spaced == packed.replace(render_ascii_map(level, state), render_ascii_map_spaced(level, state))
        old = prompts.ascii_system_prompt_for('ascii_edward_v1', mode)
        new = prompts.ascii_system_prompt_for('ascii_edward_spaced_v1', mode)
        assert new == old.replace('= tunnel/level door, space empty.',
                                  '= tunnel/level door, _ empty; map cells are separated by single spaces.')
        a = prompts.prompt_contract_metadata('ascii_edward_v1', edward_options=True, fallback_mode=mode)
        b = prompts.prompt_contract_metadata('ascii_edward_spaced_v1', edward_options=True, fallback_mode=mode)
        assert a['action_protocol'] == b['action_protocol'] and a['user_prompt_template_sha256'] == b['user_prompt_template_sha256']
        assert a['prompt_version'] != b['prompt_version'] and a['prompt_template_sha256'] != b['prompt_template_sha256']
        assert prompts.prompt_text('ascii_edward_spaced_v1') == (prompts.ASCII_SPACED_EDWARD_SYSTEM_PROMPT, prompts.ASCII_EDWARD_USER_TEMPLATE)


def test_config_selects_spaced_style_only_for_ascii():
    from slime_pacman.config import PacmanConfig, runner_options
    cfg = PacmanConfig(groups_per_update=16, clip=.05, observation_mode='ascii', ascii_map_format='spaced')
    options = runner_options(cfg)
    assert options['image_prompt_style'] == 'ascii_edward_spaced_v1'
    assert options['prompt_version'].startswith('edward-ascii-spaced-option-code-v2')
    assert runner_options(PacmanConfig(groups_per_update=16, clip=.05, observation_mode='ascii'))['image_prompt_style'] == 'ascii_edward_v1'
    with pytest.raises(ValueError):
        PacmanConfig(ascii_map_format='spaced')
    with pytest.raises(ValueError):
        PacmanConfig(groups_per_update=16, clip=.05, observation_mode='ascii', ascii_map_format='dense')


def test_actual_spaced_ascii_runner_sends_spaced_map():
    from pathlib import Path
    from pacman_recipe.level1.contracts import make_episode_record
    from slime_pacman.config import PacmanConfig
    from slime_pacman.rollout import EpisodeRunner
    from test_slime_pacman import Tokenizer as ActionTokenizer
    from slime_pacman.generation import Decision
    root = Path(__file__).resolve().parents[1]
    record = make_episode_record(0, split='test', recipe_root=root, game_root=root.parent/'pacman-python', backend_root=root.parent/'slime',
                                 max_steps=32, observation_mode='ascii', edward_fallback_mode='risk_ranked', ascii_map_format='spaced')
    cfg = PacmanConfig(max_steps=32, groups_per_update=16, clip=.05, observation_mode='ascii',
                       edward_fallback_mode='risk_ranked', ascii_map_format='spaced')
    sent = []

    async def generate(messages, constraint):
        sent.append((messages[0]['content'], messages[1]['content'][0]['text']))
        token = constraint.allowed_token_ids[0]
        return Decision('teacher', [], token, chr(token), constraint.allowed_token_ids, 0., 'scripted', {}, None, [])
    result = asyncio.run(EpisodeRunner(record, tokenizer=ActionTokenizer(), generate=generate, config=cfg).collect(empty_weight_version='scripted'))
    assert sent and all(system == prompts.ASCII_SPACED_EDWARD_SYSTEM_PROMPT for system, _ in sent)
    assert all(ASCII_SPACED_MAP_HEADER in user and '[CURRENT STATE]' in user for _, user in sent)
    rows = result.trajectory['episode']['trajectory']
    assert all(row['observation_mode'] == 'ascii' and row['observation_png_sha256'] is None for row in rows)
    called = [row for row in rows if row.get('model_called')]
    assert called and all(row['observation_ascii_map'].startswith(ASCII_SPACED_MAP_HEADER) for row in called)
    assert result.trajectory['episode']['image_prompt_style'] == 'ascii_edward_spaced_v1'
