"""Isolated freeze provider and per-decision acceptance evidence."""
import hashlib
import json
import os
from pathlib import Path

def append(kind, value):
    root = Path('/run/evidence')
    root.mkdir(exist_ok=True)
    with (root / f'{kind}-{os.getpid()}.jsonl').open('a') as out:
        out.write(json.dumps(value) + '\n')

def get_model_provider(args, config, vp_stage):
    from slime_plugins.models.qwen3_5_vl import get_qwen3_5_vl_model_provider
    provider = get_qwen3_5_vl_model_provider(args, config, vp_stage)
    def build(pre_process=True, post_process=True, vp_stage=None):
        model = provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        frozen, trainable = [], []
        for name, p in model.named_parameters():
            p.requires_grad_(not name.startswith('model.visual.'))
            (trainable if p.requires_grad else frozen).append([name, p.numel()])
        assert frozen and trainable
        assert any('merger' in name for name, _ in frozen)
        assert all(name.startswith('language_model.') for name, _ in trainable)
        append('freeze', dict(frozen=frozen, trainable=trainable,
            frozen_count=sum(n for _, n in frozen), trainable_count=sum(n for _, n in trainable)))
        return model
    return build

def custom_loss(args, batch, logits, reducer):
    import torch
    from slime.backends.megatron_utils.loss import get_responses
    from slime_pacman.probability import custom_loss as original, masked_log_prob
    result = original(args, batch, logits, reducer)
    # These are the actual original custom_loss return values, not a second TIS
    # implementation. Reducer mass makes the bounds valid for split episodes.
    from slime_pacman import probability as probability_module
    metrics = result[1]
    mass = reducer(torch.ones_like(torch.cat(batch['rollout_log_probs'])))
    actual = {name: float(metrics[name].detach()) for name in
              ('tis_weight_mean', 'tis_truncate_fraction', 'masked_logprob_abs_diff')}
    actual.update(loss=float(result[0].detach()), reducer_mass=float(mass.detach()),
                  source_sha256=hashlib.sha256(Path(probability_module.__file__).read_bytes()).hexdigest(),
                  origin='original_custom_loss_return_values', upper_bound=2.0)
    import math
    assert all(math.isfinite(actual[name]) for name in
               ('loss', 'reducer_mass', 'tis_weight_mean', 'tis_truncate_fraction', 'masked_logprob_abs_diff'))
    assert actual['reducer_mass'] >= 0
    assert -1e-6 <= actual['tis_weight_mean'] <= 2 * actual['reducer_mass'] + 1e-5
    assert -1e-6 <= actual['tis_truncate_fraction'] <= actual['reducer_mass'] + 1e-5
    append('tis-actual', actual)
    responses = get_responses(logits.float(), args=args,
        unconcat_tokens=batch['unconcat_tokens'], total_lengths=batch['total_lengths'],
        response_lengths=batch['response_lengths'], apply_temperature=False)
    behavior = torch.cat(batch['rollout_log_probs']).detach()
    for i, ((row, tokens), meta) in enumerate(zip(responses, batch['metadata'], strict=True)):
        if meta.get('empty_episode'):
            continue
        logp, entropy = masked_log_prob(row[0], meta['allowed_token_ids'], int(tokens[0]), .7)
        log_probs = torch.log_softmax(row[0, meta['allowed_token_ids']].float() / .7, dim=0)
        mass = torch.softmax(row[0].float(), dim=0)[meta['allowed_token_ids']].sum()
        append('decisions', dict(episode=meta['episode_id'], step=meta['decision_index'],
            support=meta['allowed_token_ids'], action=int(tokens[0]),
            megatron_logp=float(logp.detach()), sglang_logp=float(behavior[i]),
            abs_diff=float((logp.detach()-behavior[i]).abs()), entropy=float(entropy.detach()),
            legal_mass=float(mass.detach()), candidate_log_probs=log_probs.detach().cpu().tolist(),
            decision_count=meta['decision_count'],clip_cov_detach=meta.get('clip_cov_detach')))
    return result
