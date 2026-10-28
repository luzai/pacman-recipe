"""Exact native replay of ordinary training trajectories into restart snapshots."""
import hashlib,json,random
from pathlib import Path
from copy import deepcopy
from .identity import digest
from .checkpoints import atomic_json as atomic
def extract(job):
 path, cfg, bank, policy = job
 BANK=Path(bank);POLICY=policy
 x=json.loads(Path(path).read_text());assert x['split']=='train'
 template=next(t for t in cfg['catalog'] if t['task_id']==x['task_id'])
 if template['split']!='train' or x['record']['source_revisions']!=template['episode_record']['source_revisions']:
  raise ValueError('restart source is not a matching training task')
 if x['record'].get('restart'):return []
 ts=x['trajectory']['episode']['trajectory'];deaths=[];prev=-1
 for i,t in enumerate(ts):
  if t['death']:
   choices=[(i+1-back,back) for back in range(4,33) if i+1-back>prev and i+1-back>0]
   if choices:deaths.append((i,choices))
   prev=i
 if not deaths:return []
 rng=random.Random(digest([Path(path).stem,POLICY['seed']]));death,choices=rng.choice(deaths);wanted=dict(choices);saved_states={}
 from slime_pacman.map_training.workers import worker_init
 worker_init(x['map_context']['game_root'],x['map_context']['ghosts'])
 from pacman_env.env.pygame_environment import PygamePacmanEnv,PygamePacmanEnvConfig
 from slime_pacman.map_training.maps import load_runtime_level
 from pacman_recipe.level1.ascii_observation_spaced import render_ascii_map_spaced
 game=x['map_context']['game_root'];spec=x['record']['environment'];level=load_runtime_level(1,game)
 with PygamePacmanEnv(PygamePacmanEnvConfig(pacman_python_root=game,level=1,max_steps=spec['max_steps'],ghost_mode=spec['ghost_mode'],episode_life_mode=spec['episode_life_mode'])) as e:
  identity=e._saved_state_identity();e._saved_state_identity=lambda:identity
  assert e._maapacman_source_revision==spec['environment_source_sha256']
  assert e._pacman_source_revision==spec['pacman_python_source_sha256']
  e.reset(seed=spec['seed'])
  for j,z in enumerate(ts[:death+1]):
   snap=e.snapshot();c=z['observation_context']
   assert [snap['row'],snap['col']]==c['pacman_position'] and snap['ghosts']==c['ghosts'] and snap['open']==c['open_actions']
   assert render_ascii_map_spaced(level,snap)==c['ascii_map']
   if j in wanted:
    saved=e.save_state();e.restore_state(saved);assert e.save_state()['sha256']==saved['sha256']
    exits={}
    for prior in ts[:j]:
     cell=tuple(prior['observation_context']['pacman_position']);exits.setdefault(cell,[]).append(prior['action'])
    saved['primitive_history']=dict(cell_exit_history=[[r,col,actions] for (r,col),actions in exits.items()],recent_actions=c['recent_actions'],recent_positions=c['recent_positions'])
    assert saved['payload']['episode']['steps']==j;saved_states[j]=saved
   _,_,term,trunc,info=e.step(z['action'])
   for k in ['death','death_count','normal_pellets_remaining','score','logic_frame_events','atomic_substeps']:assert info[k]==z[k],(j,k)
   assert term==z['terminated'] and trunc==z['truncated']
 template=next(t for t in cfg['catalog'] if t['task_id']==x['task_id']);result=[]
 from pacman_recipe.level1.contracts import validate_episode_record
 source_sha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
 for i,saved in saved_states.items():
  back=wanted[i];bin_id=next(k for k,(lo,hi) in enumerate(POLICY['bins']) if lo<=back<=hi)
  sid=digest([Path(path).stem,i,saved['sha256']]);p=BANK/'states'/f'{sid}.json';atomic(p,saved)
  record=deepcopy(x['record']);record['id']='failure-'+sid;record['restart']=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),id=record['id']);record['source_revisions']=deepcopy(template['episode_record']['source_revisions']);validate_episode_record(record)
  result.append(dict(id=sid,task_id=x['task_id'],record=record,tier=x['tier'],update=x['update'],source_artifact=Path(path).stem,source_sha256=source_sha,source_step=i,death_step=death+1,lookback=back,bin_id=bin_id,parity_steps=death+1,full_state_sha256=saved['sha256'],split='train'))
 return result
