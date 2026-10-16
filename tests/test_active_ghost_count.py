"""Reduced-ghost curricula must change the engine, not just its snapshot."""
from types import SimpleNamespace

import pytest

from pacman_env.env import PygamePacmanEnv, PygamePacmanEnvConfig
from pacman_env.env._pygame_worker import _PygameBridge


def raw_ghosts(saved):
    nodes = saved['payload']['worker']['graph']['nodes']
    result = {}
    for node in nodes:
        if node.get('kind') == 'object' and node.get('name') == 'ghost':
            attrs = dict(nodes[node['attrs']['ref']]['items'])
            if attrs['id'] < 4:
                result[attrs['id']] = attrs
    assert len(result) == 4
    return result


@pytest.mark.parametrize('count,mode', [(1, 'normal'), (2, 'normal'),
                                         (4, 'normal'), (1, 'disabled')])
def test_real_engine_count_across_steps_restore_and_reset(monkeypatch, count, mode):
    monkeypatch.setenv('SMALL_MAZE_GHOST_COUNT', str(count))
    monkeypatch.setenv('SDL_VIDEODRIVER', 'dummy')
    monkeypatch.setenv('SDL_AUDIODRIVER', 'dummy')
    expected = count if mode == 'normal' else 0
    config = PygamePacmanEnvConfig(max_steps=16, ghost_mode=mode,
                                  episode_life_mode='original_three_lives')
    with PygamePacmanEnv(config) as env:
        env.reset(seed=2050)
        saved = env.save_state()
        for phase in range(4):
            assert len(env.snapshot()['ghosts']) == expected
            ghosts = raw_ghosts(env.save_state())
            assert sum(g['state'] != 4 for g in ghosts.values()) == expected
            for i in range(expected, 4):
                g = ghosts[i]
                assert (g['state'], g['x'], g['y'], g['velX'], g['velY']) == (
                    4, -64, -64, 0, 0)
            if phase == 0:
                for _ in range(4):
                    env.step(env.snapshot()['open'][0])
            elif phase == 1:
                env.restore_state(saved)
            elif phase == 2:
                env.reset(seed=2051)


def test_bridge_rejects_active_ghost_hidden_from_snapshot(monkeypatch):
    monkeypatch.setenv('SMALL_MAZE_GHOST_COUNT', '1')
    bridge = SimpleNamespace(_ghost_mode='normal')
    namespace = dict(GHOST_MODE='normal', CURRICULUM_ID=2,
                     ghosts={1: SimpleNamespace(state=1)})
    with pytest.raises(RuntimeError, match='unreported ghost is active'):
        _PygameBridge._capture(bridge, namespace, include_frame=False)
