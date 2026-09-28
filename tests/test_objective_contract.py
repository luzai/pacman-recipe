from __future__ import annotations

from types import SimpleNamespace

import pytest

from train_areal import _validate_reward_objective_contract
from pacman_recipe.synthetic.configs import PacmanAgentConfig


def _config(
    *,
    reward_norm=None,
    adv_norm=None,
    contract="step_local_raw_v1",
    edward_options=True,
    n_samples=12,
    use_sapo_loss=False,
    use_cispo_loss=False,
    ppo_n_minibatches=1,
    critic=None,
    teacher=None,
):
    return SimpleNamespace(
        reward_objective_contract=contract,
        workflow="pacman_recipe.workflow.PacmanNativeVisionWorkflow",
        actor=SimpleNamespace(
            reward_norm=reward_norm,
            adv_norm=adv_norm,
            use_sapo_loss=use_sapo_loss,
            use_cispo_loss=use_cispo_loss,
            ppo_n_minibatches=ppo_n_minibatches,
        ),
        gconfig=SimpleNamespace(n_samples=n_samples),
        edward_options=edward_options,
        critic=critic,
        teacher=teacher,
    )


def _episode_reward_norm(*, group_size=12, mean_leave1out=False):
    return SimpleNamespace(
        mean_level="group",
        mean_leave1out=mean_leave1out,
        std_level="group",
        group_size=group_size,
        std_unbiased=True,
        eps=1e-5,
    )


def test_option_return_raw_is_the_default_contract() -> None:
    contract_field = PacmanAgentConfig.__dataclass_fields__[
        "reward_objective_contract"
    ]
    assert contract_field.default == "option_return_raw_v1"


def test_step_local_raw_contract_accepts_disabled_normalization() -> None:
    _validate_reward_objective_contract(_config())


@pytest.mark.parametrize("field", ["reward_norm", "adv_norm"])
def test_step_local_raw_contract_rejects_normalization(field: str) -> None:
    kwargs = {field: SimpleNamespace(mean_level="group")}
    with pytest.raises(ValueError, match=field):
        _validate_reward_objective_contract(_config(**kwargs))


def test_step_local_raw_contract_rejects_unknown_contract() -> None:
    with pytest.raises(ValueError, match="unsupported reward_objective_contract"):
        _validate_reward_objective_contract(_config(contract="mystery"))


def test_option_return_raw_contract_accepts_disabled_normalization() -> None:
    _validate_reward_objective_contract(
        _config(contract="option_return_raw_v1")
    )


@pytest.mark.parametrize("field", ["reward_norm", "adv_norm"])
def test_option_return_raw_contract_rejects_normalization(field: str) -> None:
    kwargs = {field: SimpleNamespace(mean_level="group")}
    with pytest.raises(ValueError, match=field):
        _validate_reward_objective_contract(
            _config(contract="option_return_raw_v1", **kwargs)
        )


def test_option_return_raw_contract_requires_edward_options() -> None:
    with pytest.raises(ValueError, match="edward_options=true"):
        _validate_reward_objective_contract(
            _config(contract="option_return_raw_v1", edward_options=False)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ppo_n_minibatches": 2},
        {"use_sapo_loss": True},
        {"use_cispo_loss": True},
    ],
)
def test_option_return_raw_contract_requires_single_equal_episode_update(
    kwargs,
) -> None:
    with pytest.raises(ValueError, match="option_return_raw_v1"):
        _validate_reward_objective_contract(
            _config(contract="option_return_raw_v1", **kwargs)
        )


def test_episode_return_group_contract_accepts_twelve_complete_episodes() -> None:
    _validate_reward_objective_contract(
        _config(
            contract="episode_return_group_v1",
            reward_norm=_episode_reward_norm(),
        )
    )


@pytest.mark.parametrize(
    ("reward_norm", "n_samples"),
    [
        (None, 12),
        (_episode_reward_norm(group_size=11), 12),
        (_episode_reward_norm(mean_leave1out=True), 12),
        (_episode_reward_norm(), 11),
    ],
)
def test_episode_return_group_contract_rejects_wrong_group_contract(
    reward_norm, n_samples
) -> None:
    with pytest.raises(ValueError, match="episode_return_group_v1"):
        _validate_reward_objective_contract(
            _config(
                contract="episode_return_group_v1",
                reward_norm=reward_norm,
                n_samples=n_samples,
            )
        )


def test_episode_return_group_contract_requires_edward_options() -> None:
    with pytest.raises(ValueError, match="edward_options=true"):
        _validate_reward_objective_contract(
            _config(
                contract="episode_return_group_v1",
                reward_norm=_episode_reward_norm(),
                edward_options=False,
            )
        )


@pytest.mark.parametrize("surrogate", ["use_sapo_loss", "use_cispo_loss"])
def test_episode_return_group_contract_requires_grpo_surrogate(
    surrogate: str,
) -> None:
    with pytest.raises(ValueError, match="PPO/GRPO surrogate"):
        _validate_reward_objective_contract(
            _config(
                contract="episode_return_group_v1",
                reward_norm=_episode_reward_norm(),
                **{surrogate: True},
            )
        )


@pytest.mark.parametrize("role", ["critic", "teacher"])
def test_episode_return_group_contract_rejects_auxiliary_role(role: str) -> None:
    with pytest.raises(ValueError, match="critic=null and teacher=null"):
        _validate_reward_objective_contract(
            _config(
                contract="episode_return_group_v1",
                reward_norm=_episode_reward_norm(),
                **{role: SimpleNamespace()},
            )
        )


def test_episode_return_group_contract_rejects_multiple_ppo_minibatches() -> None:
    with pytest.raises(ValueError, match="ppo_n_minibatches=1"):
        _validate_reward_objective_contract(
            _config(
                contract="episode_return_group_v1",
                reward_norm=_episode_reward_norm(),
                ppo_n_minibatches=2,
            )
        )
