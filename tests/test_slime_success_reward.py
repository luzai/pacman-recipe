from types import SimpleNamespace
import pytest
from slime_pacman.config import PacmanConfig
from slime_pacman.grouping import success_reward, success_speed_reward, post_process_rewards

def test_future_training_default_bonuses():
    config = PacmanConfig()
    assert config.success_life_bonus == 0.1
    assert config.success_speed_bonus == 0.05

def payload(**changes):
    return dict(dict(episode_life_mode='original_three_lives',terminal_reason='all_normal_pellets',
                     won=True,normal_pellets_remaining=0,death_count=0,steps=256,max_steps=512),**changes)

@pytest.mark.parametrize('deaths',range(4))
def test_four_life_formula(deaths):
    p=payload(death_count=deaths)
    assert success_reward(p,success_life_bonus=.1,success_speed_bonus=.05)==pytest.approx(1+.1*(1-deaths/3)+.025)

def test_failure_never_gets_survival_or_speed_bonus():
    assert success_reward(payload(won=False,terminal_reason='max_steps',normal_pellets_remaining=2),
                          success_life_bonus=.1,success_speed_bonus=.05)==0

def test_single_life_and_restart_budgets():
    assert success_reward(payload(episode_life_mode='single_death'),success_life_bonus=.1,success_speed_bonus=.05)==1.025
    p=payload(death_count=2,steps=64,restart_state=dict(death_count=1,remaining_budget=128))
    assert success_reward(p,success_life_bonus=.1,success_speed_bonus=.05)==pytest.approx(1.075)

def test_binary_default_and_old_speed_api():
    assert success_reward(payload())==1
    assert success_speed_reward(payload(),.05)==1.025

@pytest.mark.parametrize('coefficient',[-.01,.11,float('nan'),float('inf')])
def test_invalid_configuration(coefficient):
    with pytest.raises(ValueError):PacmanConfig(success_life_bonus=coefficient)
    with pytest.raises(ValueError):success_reward(payload(),success_life_bonus=coefficient)

@pytest.mark.parametrize('changes',[dict(death_count=4),dict(death_count=-1),dict(death_count=True),dict(steps=513)])
def test_invalid_evidence(changes):
    with pytest.raises(ValueError):success_reward(payload(**changes),success_life_bonus=.1,success_speed_bonus=.05)

def test_episode_metadata_preserves_joint_bonus_and_rejects_mixing():
    samples=[]
    rewards=[1.125]*6+[1.025]*6
    for i,r in enumerate(rewards):
        for step in range(i%2+1):
            samples.append(SimpleNamespace(rollout_id=i,reward=r,train_metadata=dict(
                episode_id=i,group_id=0,initial_state_id='same',decision_count=i%2+1,
                decision_index=step,weight_version='v1',success_speed_bonus=.05,success_life_bonus=.1)))
    raw,adv=post_process_rewards(SimpleNamespace(n_samples_per_prompt=12),samples)
    assert adv==pytest.approx([r-1.075 for r in raw])
    samples[-1].train_metadata['success_life_bonus']=0
    with pytest.raises(ValueError,match='reward configuration'):post_process_rewards(SimpleNamespace(n_samples_per_prompt=12),samples)
