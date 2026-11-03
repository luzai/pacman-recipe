import math
import torch

def support_log_probs(logits,support,temperature=.7):
    if temperature!=.7 or not support or len(set(support))!=len(support) or any(type(i) is not int or not 0<=i<logits.shape[-1] for i in support):
        raise ValueError('fixed temperature and unique actual support required')
    values=logits[...,support].float()/temperature
    if not torch.isfinite(values).all():raise ValueError('non-finite advertised logits')
    return values.log_softmax(dim=-1)

def clipped_choice_loss(new_logs,old_logs,reference_logs,selected_index,advantage,clip=.2,kl_beta=.01):
    if not math.isfinite(float(clip)) or not 0<clip<1 or not math.isfinite(float(kl_beta)) or kl_beta<0:
        raise ValueError('finite positive PPO clip below one and nonnegative KL coefficient required')
    if new_logs.ndim!=1 or old_logs.shape!=new_logs.shape or reference_logs.shape!=new_logs.shape or not 0<=selected_index<len(new_logs):
        raise ValueError('complete aligned support distributions required')
    if not all(torch.isfinite(x).all() for x in (new_logs,old_logs,reference_logs)) or not math.isfinite(float(advantage)):
        raise ValueError('finite loss inputs required')
    for logs in (new_logs,old_logs,reference_logs):
        if abs(float(logs.detach().exp().sum())-1)>2e-4:raise ValueError('normalized support required')
    ratio=(new_logs[selected_index]-old_logs[selected_index].detach()).exp()
    reward_loss=-torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage)
    categorical_kl=(new_logs.exp()*(new_logs-reference_logs.detach())).sum()
    return reward_loss+kl_beta*categorical_kl


def engine_choice_loss(new_logs, old_logs, reference_logs, selected_index, advantage,
                       *, clip, kl_beta, entropy_bonus):
    """Explicit engine-experiment objective; callers control all coefficients."""
    if not math.isfinite(float(entropy_bonus)) or entropy_bonus < 0:
        raise ValueError('finite nonnegative entropy coefficient required')
    objective = clipped_choice_loss(new_logs, old_logs, reference_logs, selected_index,
                                    advantage, clip=clip, kl_beta=kl_beta)
    entropy = -(new_logs.exp() * new_logs).sum()
    return objective - entropy_bonus * entropy
