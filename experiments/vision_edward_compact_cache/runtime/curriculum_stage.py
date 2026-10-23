"""Authorized stage bridge: verify old joint, reset only curriculum statistics."""
import json
from pathlib import Path


def checkpoint_load_hook(args, rollout_id):
    from slime_pacman.formal_rollout import get_runtime, checkpoint_load_hook as normal
    from slime_pacman.map_training.checkpoints import restore,sha256,atomic_json
    from slime_pacman.map_training.stages import begin_stage
    runtime=get_runtime(args)
    config=json.loads(Path('/acceptance/curriculum.json').read_text())
    stage=config['curriculum_transition']
    if rollout_id!=stage['parent_update']:
        return normal(args,rollout_id)
    if Path(args.load).resolve()!=Path('/parent-checkpoints').resolve():
        raise ValueError('stage bridge requires explicit read-only parent checkpoint')
    receipt_path=Path(args.load)/'joint'/f'completed-{rollout_id:07d}.json'
    digest=sha256(receipt_path)
    receipt=restore(args.load,identity=stage['parent_identity'],update=rollout_id)
    result=begin_stage(receipt,runtime.sampler,parent_identity=stage['parent_identity'],
        parent_update=rollout_id,receipt_sha256=digest,
        expected_receipt_sha256=stage['parent_receipt_sha256'],optimizer_budget=200)
    assert result['starting_tier']==0 and result['next_rollout']==args.start_rollout_id
    result['parent_model_optimizer_files_verified']=len(receipt['files'])
    atomic_json(Path('/run/CURRICULUM_STAGE_LOADED.json'),result)
    print('CURRICULUM_STAGE_LOADED',json.dumps(result),flush=True)


def source_hook():
    return checkpoint_load_hook
