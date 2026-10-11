import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from pacman_recipe.level1.contracts import make_episode_record
from pacman_recipe.level1.token_constraints import PrimitiveTokenConstraint, ObjectiveParseError
from slime_pacman.config import PacmanConfig
from slime_pacman.generation import Decision
from slime_pacman.rollout import EpisodeRunner
from slime_pacman.grouping import binary_reward
from slime_pacman.map_training.maps import generate, parse_grid, canonical

ROOT = Path(__file__).resolve().parents[1]

def test_small_catalog_capacity_and_protected_tiles():
    template = (ROOT / 'configs/slime/maps/template-7.txt').read_text()
    original = parse_grid(template)[1]
    rows = generate(template, seed=61011, counts=dict(train=32, validation=8, test=8),
                    swaps=1, small_floor_pellets=True)
    assert len(rows) == 48 and len({r['equivalence_sha256'] for r in rows}) == 48
    assert canonical(original) not in {r['equivalence_sha256'] for r in rows}
    for row in rows:
        grid = parse_grid(row['text'])[1]
        for r, cells in enumerate(original):
            for c, tile in enumerate(cells):
                if tile not in (0, 2) and tile < 100:
                    assert grid[r][c] == tile

def test_original_lives_game_over_binary_and_invalid_mode():
    payload = dict(episode_life_mode='original_three_lives', won=False,
                   terminal_reason='game_over', normal_pellets_remaining=3)
    assert binary_reward(payload) == 0.
    payload['episode_life_mode'] = 'single_death'
    with pytest.raises(ValueError):
        binary_reward(payload)

class Tokenizer:
    all_special_ids = [0]
    def encode(self, text, **kwargs):
        return [ord(text)]
    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))

def test_direct_support_generator_and_strict_parsing():
    constraint = PrimitiveTokenConstraint.build(Tokenizer(), iter('RL'))
    assert constraint.option_ids == ('L', 'R')
    assert constraint.allowed_token_ids == [76, 82]
    for text in ('B', 'U', ' R', 'RR', ''):
        with pytest.raises(ObjectiveParseError):
            constraint.option_for_completion(text)
    with pytest.raises(ObjectiveParseError):
        constraint.option_for_tokens([82, 82])
    for support in ([], ['S'], ['R', 'B']):
        with pytest.raises(ValueError):
            PrimitiveTokenConstraint.build(Tokenizer(), support)

@pytest.mark.parametrize('damage', ['multi', 'special', 'duplicate', 'decode'])
def test_bad_tokenizer_rejected(damage):
    class Bad(Tokenizer):
        def encode(self, text, **kwargs):
            return {'multi': [1, 2], 'special': [0], 'duplicate': [1]}.get(damage, [ord(text)])
        def decode(self, ids, **kwargs):
            return '?' if damage == 'decode' else super().decode(ids, **kwargs)
    with pytest.raises(ValueError):
        PrimitiveTokenConstraint.build(Bad(), ['U', 'D'])

@pytest.mark.parametrize('life', ['single_death', 'original_three_lives'])
def test_native_real_game_primitive_audited(life, monkeypatch):
    monkeypatch.setenv('SMALL_MAZE_GHOST_COUNT', '1')
    config = PacmanConfig(action_mode='primitive', observation_mode='ascii',
        ascii_map_format='spaced', groups_per_update=16, clip=.05,
        ghost_mode='disabled', episode_life_mode=life, max_steps=32)
    record = make_episode_record(28, split='test', recipe_root=ROOT,
        game_root=ROOT.parent / 'pacman-python', backend_root=ROOT.parent / 'slime',
        max_steps=32, observation_mode='ascii', ascii_map_format='spaced',
        action_mode='primitive', ghost_mode='disabled', episode_life_mode=life)
    async def generate(messages, constraint):
        user = messages[1]['content'][0]['text']
        assert 'screenshot' not in user.lower()
        assert 'Ghost mode: disabled' in user
        assert all(item['type'] == 'text' for item in messages[1]['content'])
        token = constraint.allowed_token_ids[0]
        return Decision('prompt', [1, 2], token, chr(token),
            constraint.allowed_token_ids, 0., 'v0', {}, None, [])
    runner = EpisodeRunner(record, tokenizer=Tokenizer(), generate=generate, config=config)
    result = asyncio.run(runner.collect(empty_weight_version='v0'))
    assert result.reward in (0., 1.) and result.trajectory is not None
    payload = result.trajectory['episode']
    assert payload['total_shaped_reward'] == binary_reward(payload)
    assert payload['episode_life_mode'] == life
    for step in payload['trajectory']:
        assert not step['ghosts']
        assert not step.get('death')
    with pytest.raises(ValueError, match='contract differs'):
        EpisodeRunner(record, tokenizer=Tokenizer(), generate=generate,
            config=replace(config, ghost_mode='normal'))
