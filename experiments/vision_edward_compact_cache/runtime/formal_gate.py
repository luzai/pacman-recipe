"""First-update numerical gate for the authorized new 200-update curriculum run."""
import os
import json
import math
from pathlib import Path
import subprocess

def check():
    first=int(os.environ.get('PACMAN_FIRST_NEW_UPDATE','0'));completed=first+1
    summary=json.loads(subprocess.check_output(['python','/acceptance/summarize.py','/run'],text=True))
    Path('/acceptance/first-update-summary.json').write_text(json.dumps(summary,indent=2))
    assert summary['freeze_actors']==8, 'not all eight actors reported freeze contract'
    assert all(r==dict(frozen=456010480,trainable=8953803264) for r in summary['freeze_counts'])
    assert summary['failures']==0 and summary['audit_failures']==0
    diff=summary['episode_weighted_logprob_abs_diff']
    assert diff is not None and math.isfinite(diff), f'nonfinite logprob discrepancy: {diff}'
    policy=json.loads(Path('/acceptance/GATE_POLICY.json').read_text())
    assert policy['policy_id']=='legal-tis-cache-aware-v1'
    assert policy['legacy_logprob_threshold']==.05 and policy['legacy_threshold_enforced'] is False
    assert policy['record_legacy_threshold_result'] is True and policy['tis_upper_bound']==2.0
    entropy=summary['megatron_entropy']['mean']
    assert math.isfinite(entropy) and entropy>.01, f'near-zero entropy: {entropy}'
    subprocess.run(['python','/acceptance/verify_export.py','/run/hf-'+os.environ.get('PACMAN_FIRST_NEW_UPDATE','0'),'/acceptance/export-0.json','/parent-hf'],check=True)
    export=json.loads(Path('/acceptance/export-0.json').read_text())
    result=dict(numerical_gate_passed=True,gate_policy=policy["policy_id"],legacy_logprob_threshold=.05,legacy_logprob_gate_passed=diff<=.05,mean_logprob_diff=diff,entropy=entropy,
        language_changed_tensors=export['language_changed_tensors'],
        learning_signal_qualified=export['language_changed_tensors']>0,
        note='Zero reward/gradient is reported separately; this gate does not establish learning.')
    from optimizer_readback import inspect_optimizer_boundary
    optimizer=inspect_optimizer_boundary('/run/checkpoints',completed)
    assert optimizer['serialized_steps']==[completed] and optimizer['moments_present'] and optimizer['scheduler_present'] and optimizer['rng_present']
    Path(f'/acceptance/optimizer-step{completed}-readback.json').write_text(json.dumps(optimizer,indent=2))
    archive=json.loads(Path('/acceptance/checkpoint-archives/latest.json').read_text())
    assert archive['update']==first and archive['receipt']['update']==first
    assert (Path('/acceptance/checkpoint-archives')/archive['file']).stat().st_size==archive['size']
    values=[]
    for file in Path('/run/episodes').glob('formal-train-*.json'):
        episode=json.loads(file.read_text())
        if episode['update']!=first:continue
        for row in episode.get('decisions',[]):
            lp=row.get('candidate_log_probs')
            if lp and len(lp)>1:
                weights=[math.exp(x-max(lp)) for x in lp];total=sum(weights)
                probs=[w/total for w in weights]
                values.append(-sum(p*math.log(p) for p in probs if p>0))
    assert values and sum(values)/len(values)>.01, 'legal action entropy gate failed'
    result.update(adam_step=completed,archive_update=first,legal_multi_action_entropy=sum(values)/len(values),legal_near_zero_fraction=sum(x<.01 for x in values)/len(values))
    assert result['learning_signal_qualified'], 'clean update has no visible language update'
    receipts=[json.loads(p.read_text()) for p in Path('/run/clip-cov').glob('*.json')]
    assert receipts, 'first-update ClipCov receipt missing'
    cov=receipts[0]
    assert all(r==cov for r in receipts), 'conflicting ClipCov receipts'
    assert (cov['ratio'],cov['low'],cov['high'])==(.002,1.,5.)
    assert 0<=cov['selected']<=cov['eligible']<=cov['decisions']
    expected=min(max(int(.002*cov['decisions']),1),cov['eligible'])
    assert cov['selected']==expected
    evidence=[json.loads(line) for p in Path('/run/evidence').glob('decisions-*.jsonl') for line in p.read_text().splitlines()]
    assert len(evidence)==cov['decisions']
    assert all(type(e['clip_cov_detach']) is bool for e in evidence)
    assert sum(e['clip_cov_detach'] for e in evidence)==cov['selected']
    result['clip_cov']=dict(cov,actual_detach_fraction=cov['selected']/cov['decisions'])
    batch=json.loads(Path(f'/run/effective-batches/rollout-{first:04d}.json').read_text())
    assert len(batch)==192
    groups={row['group'] for row in batch};assert len(groups)==16
    for group in groups:
        rows=[r for r in batch if r['group']==group]
        assert len(rows)==12 and len({r['reward'] for r in rows})>1
    assert len({r['weight_version'] for r in batch})==1
    result['reward_filter']=dict(mode='exact_final_reward',retry_task='same_tier_new_task',effective_groups=16,episodes=192)
    from cache_tis_gate import check_evidence
    result['cache_tis_evidence']=check_evidence(batch, evidence)
    result['cache_tis_evidence']['current_run_post_adam1_sync_verified']=False
    result['cache_tis_evidence']['note']='Gate precedes actor transfer; subsequent post-weight-update boundary verifies warmup before producer admission.'
    result['numerical_gate_passed']=False
    result['pre_sync_checks_passed']=True
    Path('/acceptance/first-update-pre-sync-gate.json').write_text(json.dumps(result,indent=2))
    print('FIRST_UPDATE_PRE_SYNC_GATE',json.dumps(result),flush=True)
