import pytest
from slime_pacman.config import PacmanConfig, runner_options


def test_explicit_risk_ranked_reaches_runner():
    config=PacmanConfig(edward_fallback_mode='risk_ranked')
    assert runner_options(config)['edward_fallback_mode']=='risk_ranked'
    assert runner_options(PacmanConfig())['edward_fallback_mode']=='refuse'


def test_invalid_fallback_rejected():
    with pytest.raises(ValueError,match='fallback'):
        PacmanConfig(edward_fallback_mode='unknown')


@pytest.mark.parametrize("edward", [False, True])
def test_visual_dataset_config_and_harness_versions_agree(edward):
    from pathlib import Path
    import yaml
    from pacman_recipe.level1.prompts import prompt_contract_metadata
    from pacman_recipe.level1.recipe import DIRECT_PROMPT_VERSION, EDWARD_PROMPT_VERSION
    expected = prompt_contract_metadata('live_state_v3', edward_options=edward)['prompt_version']
    assert (EDWARD_PROMPT_VERSION if edward else DIRECT_PROMPT_VERSION) == expected
    root = Path(__file__).resolve().parents[1]
    for path in (root / 'configs/level1').rglob('*.yaml'):
        value = yaml.safe_load(path.read_text(encoding='utf-8'))
        if (isinstance(value, dict) and value.get('image_prompt_style') == 'live_state_v3'
                and bool(value.get('edward_options', False)) == edward
                and 'prompt_version' in value):
            assert value['prompt_version'] == expected, path
    if edward:
        assert runner_options(PacmanConfig())['prompt_version'] == expected
