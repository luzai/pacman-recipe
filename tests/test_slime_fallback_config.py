import pytest
from slime_pacman.config import PacmanConfig, runner_options


def test_explicit_risk_ranked_reaches_runner():
    config=PacmanConfig(edward_fallback_mode='risk_ranked')
    assert runner_options(config)['edward_fallback_mode']=='risk_ranked'
    assert runner_options(PacmanConfig())['edward_fallback_mode']=='refuse'


def test_invalid_fallback_rejected():
    with pytest.raises(ValueError,match='fallback'):
        PacmanConfig(edward_fallback_mode='unknown')
