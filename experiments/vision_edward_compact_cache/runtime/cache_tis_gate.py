"""Evidence checks for the explicitly new TIS/cache-aware first-update policy."""
import hashlib
import json
import math
import os
from pathlib import Path


def check_evidence(batch, decisions):
    root = Path('/run')
    source = Path('/workspace/pacman-recipe/slime_pacman/probability.py')
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    actual = [json.loads(line) for p in (root / 'evidence').glob('tis-actual-*.jsonl')
              for line in p.read_text().splitlines()]
    assert actual and decisions
    for row in actual:
        assert row['origin'] == 'original_custom_loss_return_values'
        assert row['source_sha256'] == source_sha and row['upper_bound'] == 2.0
        assert all(math.isfinite(row[key]) for key in
                   ('loss', 'reducer_mass', 'tis_weight_mean', 'tis_truncate_fraction', 'masked_logprob_abs_diff'))
        assert row['reducer_mass'] >= 0
        assert -1e-6 <= row['tis_weight_mean'] <= 2 * row['reducer_mass'] + 1e-5
        assert -1e-6 <= row['tis_truncate_fraction'] <= row['reducer_mass'] + 1e-5
    for row in decisions:
        support = row['support']
        assert support and len(set(support)) == len(support) and row['action'] in support
        logps = row['candidate_log_probs']
        assert len(logps) == len(support) and all(math.isfinite(x) for x in logps)
        assert abs(sum(math.exp(x) for x in logps) - 1) < 1e-5
        assert all(math.isfinite(row[key]) for key in ('megatron_logp', 'sglang_logp', 'abs_diff', 'entropy'))
        assert abs(logps[support.index(row['action'])] - row['megatron_logp']) < 1e-5
    versions = {str(row['weight_version']) for row in batch}
    assert len(versions) == 1
    version = next(iter(versions))
    contract = json.loads(Path(os.environ['PACMAN_CACHE_CONTRACT']).read_text())
    boundaries = [json.loads(p.read_text()) for p in (root / 'cache-boundaries').glob('*.json')]
    generation = [row for row in boundaries if row['boundary'] == 'generate' and str(row['served_version']) == version]
    assert generation, 'No all-engine cache prepare before effective-batch generation'
    for receipt in generation:
        assert len(receipt['engines']) == 8 and len(set(receipt['engines'])) == 8
        assert receipt['mode'] == contract['mode']
        if contract['mode'] != 'off':
            assert len(receipt['warmup']) == 8
            assert {row['endpoint'].rstrip('/') for row in receipt['warmup']} == set(receipt['engines'])
            assert all(str(row['version']) == version for row in receipt['warmup'])
    proof_path = Path('/acceptance/cache-weight-update-acceptance.json')
    proof = json.loads(proof_path.read_text())
    if contract['mode'] != 'off':
        assert proof['passed'] is True and proof['actual_weight_update'] is True
        assert proof['old_cache_invalidated'] is True and proof['rewarm_verified'] is True
    return dict(policy_source_sha256=source_sha, actual_loss_calls=len(actual),
                actual_tis_weight_numerator=sum(row['tis_weight_mean'] for row in actual),
                actual_tis_reducer_mass=sum(row['reducer_mass'] for row in actual),
                actual_tis_truncated_numerator=sum(row['tis_truncate_fraction'] for row in actual),
                legal_decisions=len(decisions), served_version=version,
                generation_boundary_receipts=len(generation),
                independent_weight_update_proof_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest())


def finalize_after_sync(receipt):
    previous = Path('/acceptance/first-update-pre-sync-gate.json')
    result = json.loads(previous.read_text())
    assert result['pre_sync_checks_passed'] is True and result['numerical_gate_passed'] is False
    assert receipt['boundary'] == 'post-weight-update'
    assert receipt['rollout_id'] == int(os.environ.get('PACMAN_FIRST_NEW_UPDATE', '0'))
    old_version = result['cache_tis_evidence']['served_version']
    new_version = receipt['served_version']
    assert int(new_version) > int(old_version), 'Adam1 weights were not advanced on engines'
    assert len(receipt['engines']) == 8 and len(set(receipt['engines'])) == 8
    if receipt['mode'] != 'off':
        assert len(receipt['warmup']) == 8
        assert {row['endpoint'].rstrip('/') for row in receipt['warmup']} == set(receipt['engines'])
        assert all(str(row['version']) == new_version for row in receipt['warmup'])
        assert all(type(row.get('prewarm_cached_tokens')) is int and row['prewarm_cached_tokens'] == 0
                   for row in receipt['warmup']), 'Post-update warmup reused preexisting KV cache'
    result['cache_tis_evidence'].update(current_run_post_adam1_sync_verified=True,
        post_sync_receipt=receipt, current_run_old_cache_invalidated=True,
        note='All eight engines report newer weights, zero initial prefix cache hits and completed rewarm before producer admission.')
    result['numerical_gate_passed'] = True
    output = Path('/acceptance/first-update-gate.json')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print('FIRST_UPDATE_GATE', json.dumps(result), flush=True)
    return result
