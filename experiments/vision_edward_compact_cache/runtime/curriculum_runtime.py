"""Publish a verified joint checkpoint and retain bounded task-owned payloads."""
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess

def checkpoint_saved_hook(args, rollout_id):
    from slime_pacman.formal_rollout import checkpoint_saved_hook as publish
    receipt = publish(args, rollout_id)
    root = Path(args.save).resolve()
    assert root == Path('/run/checkpoints').resolve()
    durable = Path('/acceptance/checkpoint-archives')
    durable.mkdir(exist_ok=True)
    final = durable / f'checkpoint-{rollout_id:07d}.tar.gz'
    partial = durable / f'checkpoint-{rollout_id:07d}.tar.gz.partial'
    assert not partial.exists() and not final.exists()
    names = [r['path'] for r in receipt['files']]
    names += ['latest_checkpointed_iteration.txt', 'joint/latest.json',
              f'joint/completed-{rollout_id:07d}.json']
    for name in names:
        assert (root / name).resolve().is_relative_to(root)
    listing = durable / f'files-{rollout_id:07d}.txt'
    listing.write_text('\n'.join(names)+'\n')
    from archive_reservation import ensure_reservation
    reservation = ensure_reservation(durable)
    reservation.rename(partial)
    with partial.open('r+b') as out:
        result = subprocess.run(['tar','-I','env LD_LIBRARY_PATH=/archive-libs /usr/bin/pigz -p 8 -1','-cf','-',
            '-C',str(root),'-T',str(listing)], stdout=out)
        if result.returncode:
            raise RuntimeError('checkpoint archive failed; original payload retained')
        out.truncate(out.tell())
        out.flush(); os.fsync(out.fileno())
    subprocess.run(['env','LD_LIBRARY_PATH=/archive-libs','/usr/bin/pigz','-t',str(partial)],check=True)
    from slime_pacman.map_training.checkpoints import atomic_json, sha256
    digest = sha256(partial)
    partial.rename(final)
    atomic_json(durable/'latest.json',dict(update=rollout_id,file=final.name,
        size=final.stat().st_size,sha256=digest,receipt=receipt))
    # Only older payloads from this job, after the new durable archive verifies.
    for path in durable.glob('checkpoint-*.tar.gz'):
        if path != final:
            assert path.resolve().is_relative_to(durable.resolve())
            path.unlink()
    for path in root.glob('iter_*'):
        if path.name != f'iter_{rollout_id:07d}':
            assert path.resolve().parent == root
            shutil.rmtree(path)
    exports = sorted(Path('/run').glob('hf-*'), key=lambda p:int(p.name[3:]))
    keep = {Path('/run/hf-'+os.environ.get('PACMAN_FIRST_NEW_UPDATE','0')), *exports[-1:]}
    for path in exports:
        if path not in keep:
            assert path.resolve().parent == Path('/run').resolve()
            shutil.rmtree(path)
    if rollout_id < args.num_rollout - 1:
        ensure_reservation(durable)
    print('DURABLE_CHECKPOINT',rollout_id,final.stat().st_size,flush=True)
    return receipt
