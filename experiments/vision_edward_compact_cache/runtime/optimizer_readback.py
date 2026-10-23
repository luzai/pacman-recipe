"""Read trusted own Megatron optimizer/scheduler/RNG metadata on CPU.
Proves serialized step and file extents, not all moment values or GPU resume.
"""
import io
from pathlib import Path


def inspect_optimizer_boundary(checkpoint_root, completed_updates):
    import torch
    from torch.distributed.checkpoint import FileSystemReader
    root=Path(checkpoint_root).resolve();index=completed_updates-1
    if completed_updates<=0 or int((root/'latest_checkpointed_iteration.txt').read_text().strip())!=index:
        raise ValueError('Latest checkpoint differs from completed update')
    directory=root/f'iter_{index:07d}'
    meta=FileSystemReader(directory).read_metadata();keys=sorted(meta.state_dict_metadata)
    opt=[k for k in keys if 'optim' in k.lower()]
    missing=[];steps=[]
    def collect(value):
        if isinstance(value,dict):
            for k,item in value.items():
                if str(k)=='step' and isinstance(item,(int,float)):steps.append(item)
                elif str(k)=='step' and isinstance(item,torch.Tensor) and item.numel()==1:steps.append(item.item())
                else:collect(item)
        elif isinstance(value,(list,tuple)):
            for item in value:collect(item)
    for storage in meta.storage_data.values():
        path=directory/storage.relative_path
        if not path.is_file() or path.stat().st_size<storage.offset+storage.length:missing.append(str(path))
    if missing:raise ValueError('Missing/truncated checkpoint shards')
    for key,storage in meta.storage_data.items():
        if 'optimizer/shard_' in key.fqn and not hasattr(meta.state_dict_metadata[key.fqn],'size'):
            with (directory/storage.relative_path).open('rb') as stream:
                stream.seek(storage.offset);raw=stream.read(storage.length)
            collect(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False))
    common=torch.load(directory/'common.pt',map_location='cpu',weights_only=False)
    if common.get('iteration')!=index:raise ValueError('Common checkpoint iteration differs')
    moments={name:sum(k.endswith('.'+name) for k in opt) for name in ['exp_avg','exp_avg_sq']}
    evidence=dict(checkpoint_root=str(root),iteration=index,completed_updates=completed_updates,
                  serialized_steps=sorted(set(steps)),moments_present=all(moments.values()),
                  moment_key_counts=moments,shard_extents_complete=True,
                  scheduler_present='opt_param_scheduler' in common,
                  rng_present=any('rng_state' in k for k in keys) or 'rng_state' in common,
                  no_save_optim=getattr(common.get('args'),'no_save_optim',None),
                  no_save_rng=getattr(common.get('args'),'no_save_rng',None),
                  scope='CPU serialized-step/metadata/extents only; not finite moments or distributed resume')
    if evidence['serialized_steps']!=[completed_updates] or not evidence['moments_present'] or evidence['no_save_optim'] is not False:
        raise ValueError('Actual optimizer readback incomplete or wrong step')
    return evidence


if __name__=='__main__':
    import json,sys
    print(json.dumps(inspect_optimizer_boundary(sys.argv[1],int(sys.argv[2])),indent=2))
