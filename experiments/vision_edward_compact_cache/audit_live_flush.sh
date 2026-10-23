python3 - <<'HOSTPY'
from pathlib import Path
import subprocess
cid=Path('/docker/pacman-d5la-vision-edward-compact-cache-tis-20261023-v2/container.id').read_text().strip()
code='''import json,hashlib,re,datetime
from pathlib import Path
logs=Path('/tmp/ray/session_latest/logs');R=Path('/run')
boundaries=sorted([json.loads(p.read_text()) for p in (R/'cache-boundaries').glob('*.json')],key=lambda x:x['timestamp'])
post=next((x for x in boundaries if x['served_version']=='2'),None)
pre=[x for x in boundaries if x['served_version']=='1']
lower=max((x['timestamp'] for x in pre),default=0)
upper=post['timestamp'] if post else float('inf')
def stamp(line):
 m=re.match(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]',line)
 return datetime.datetime.strptime(m[1],'%Y-%m-%d %H:%M:%S').replace(tzinfo=datetime.timezone.utc).timestamp() if m else None
engines=[]
for p in sorted(logs.glob('worker-*.out')):
 text=p.read_text(errors='replace')
 if ':actor_name:SGLangEngine' not in text[:300]:continue
 err=p.with_suffix('.err');et=err.read_text(errors='replace') if err.exists() else ''
 url=re.search(r'Uvicorn running on (http://[^ ]+)',et)
 all_events=[]
 for i,line in enumerate(text.splitlines(),1):
  if not any(x in line for x in ['GET /flush_cache','POST /update_weights_from_tensor']):continue
  t=stamp(line)
  if t is not None and lower<=t<=upper:all_events.append(dict(line=i,timestamp=t,text=line))
 flush=[x for x in all_events if 'GET /flush_cache' in x['text'] and '200 OK' in x['text']]
 updates=[x for x in all_events if 'POST /update_weights_from_tensor' in x['text'] and '200 OK' in x['text']]
 ordered=bool(flush and updates and flush[0]['line']<updates[0]['line'])
 engines.append(dict(endpoint=url[1] if url else None,pid=p.stem.split('-')[-1],out_file=str(p),out_sha256=hashlib.sha256(text.encode()).hexdigest(),err_file=str(err),err_sha256=hashlib.sha256(et.encode()).hexdigest(),flush_before_tensor_200=ordered,events=all_events))
expected=set(post['engines']) if post else set()
passed=bool(post and len(engines)==8 and {x['endpoint'] for x in engines}==expected and all(x['flush_before_tensor_200'] for x in engines) and len(post['warmup'])==8 and all(x['version']=='2' and x['cached_tokens']==512 for x in post['warmup']))
print(json.dumps(dict(passed=passed,status='PASS' if passed else ('PENDING_POST_ADAM1_VERSION2' if post is None else 'AUDIT_REQUIRED'),window_start=lower,window_end=post['timestamp'] if post else None,post_boundary=post,engines=engines,prewarm_zero_not_used_as_flush_proof=True),indent=2))'''
r=subprocess.run(['docker','exec','-i',cid,'python','-'],input=code,text=True,capture_output=True)
print(r.stdout);print(r.stderr);raise SystemExit(r.returncode)
HOSTPY
