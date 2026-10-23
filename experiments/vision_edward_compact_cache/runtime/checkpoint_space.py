"""Task-owned checkpoint cache with capacity validation BEFORE deleting or updating."""
import json,shutil,os
from pathlib import Path
ROOT=Path('/run/checkpoints')
ARCHIVE=Path('/acceptance/checkpoint-archives')
GiB=1024**3

def prepare_checkpoint_space(args, rollout_id):
 root=Path(args.save).resolve();assert root==ROOT.resolve();root.mkdir(parents=True,exist_ok=True)
 if not (ARCHIVE/'latest.json').exists():
  assert rollout_id==int(os.environ.get('PACMAN_FIRST_NEW_UPDATE','0')) and not list(root.glob('iter_*'))
  assert shutil.disk_usage(root).free>=170*GiB, 'cold checkpoint capacity below 170GiB'
  from archive_reservation import ensure_reservation
  ensure_reservation(ARCHIVE)
  assert shutil.disk_usage(ARCHIVE).free>=96*GiB, 'archive rotation headroom below 96GiB'
  return
 a=json.loads((ARCHIVE/'latest.json').read_text());prior=int(a['update'])
 receipt=json.loads((root/f'joint/completed-{prior:07d}.json').read_text())
 assert a['receipt']==receipt and prior<rollout_id and len(a['sha256'])==64
 tar=(ARCHIVE/a['file']).resolve();assert tar.parent==ARCHIVE.resolve() and tar.stat().st_size==a['size']
 # Include an HF export and 30 GiB reserve, even when the raw cache is absent.
 raw=sum(x['size'] for x in receipt['files'])
 need=raw+18*GiB+30*GiB
 targets=list(root.glob('iter_*'))
 assert all(p.name==f'iter_{prior:07d}' and p.resolve().parent==root and not p.is_symlink() for p in targets), 'unarchived/unknown raw cache; retain and audit'
 free=shutil.disk_usage(root).free
 reclaim=sum(f.stat().st_size for p in targets for f in p.rglob('*') if f.is_file())
 assert free+reclaim>=need, f'checkpoint preflight: need {need}, available after verified cache reclaim {free+reclaim}; no cache deleted'
 # Two durable archives coexist during atomic rotation.
 from archive_reservation import ensure_reservation
 ensure_reservation(ARCHIVE)
 if free<need:
  for p in targets:shutil.rmtree(p)
  print('RAW_CACHE_RECLAIMED_AFTER_VERIFIED_ARCHIVE',prior,flush=True)
 assert shutil.disk_usage(root).free>=need
 print('CHECKPOINT_CAPACITY_PREFLIGHT',rollout_id,need,shutil.disk_usage(root).free,flush=True)

def install_checkpoint_space_guard():
 import slime.ray.placement_group as placement
 original=placement.create_training_models
 def create(args,*a,**kw):
  actor,critic=original(args,*a,**kw);save=actor.save_model;train=actor.async_train
  def guarded(rollout_id,*a,**kw):
   prepare_checkpoint_space(args,rollout_id)
   return save(rollout_id,*a,**kw)
  def train_guarded(rollout_id,*a,**kw):
   prepare_checkpoint_space(args,rollout_id)
   return train(rollout_id,*a,**kw)
  actor.async_train=train_guarded
  actor.save_model=guarded
  return actor,critic
 placement.create_training_models=create
