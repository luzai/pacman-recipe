"""Quota sampler for the fresh-192 ASCII experiment; no I/O or GPU claims.

The clock is completed optimizer updates (zero based). Opening rotation is
independent of bank seed caps. Min-cost flow enforces the review quota jointly
with distinct identities, seed and source-route limits, avoiding greedy dead ends.
"""
from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json

CONTRACT = "ascii-diverse-first192-v1"


def tier(wins, episodes):
    if type(wins) is not int or type(episodes) is not int or episodes <= 0 or not 0 <= wins <= episodes:
        raise ValueError("Invalid unfiltered binary outcome")
    return "blocked" if wins == 0 else "very_easy" if wins == episodes else "mixed"


@dataclass
class Policy:
    identity: str
    training_seeds: list[int]
    completed: int = 0
    initial_groups: int = 4
    states: dict = field(default_factory=dict)
    outcomes: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    representatives: list = field(default_factory=list)
    recent_window: int = 4
    route_cap: int = 2

    def __post_init__(self):
        if not self.identity or self.training_seeds != list(range(28)):
            raise ValueError("Exact training seeds 0..27 required")
        if type(self.completed) is not int or not 0 <= self.completed <= 200 or self.initial_groups != 4:
            raise ValueError("Invalid sampler clock/opening quota")
        if self.recent_window != 4 or self.route_cap != 2:
            raise ValueError("Frozen acceptance sampler limits differ")

    def measure(self, state_id, seed, bucket, file_sha, wins, episodes, update, weight, *, eligible=True):
        if self.pending or update != self.completed or seed not in self.training_seeds or type(eligible) is not bool:
            raise ValueError("Measurement at wrong policy boundary")
        tier(wins, episodes)
        if not state_id or len(file_sha) != 64 or not bucket or not weight:
            raise ValueError("Missing state provenance")
        old = self.states.get(state_id, {})
        if old and (old['seed'], old['bucket'], old['file_sha']) != (seed, bucket, file_sha):
            raise ValueError("State identity mutated")
        qualified = eligible and episodes == 24 and 3 <= wins <= 21
        self.states[state_id] = dict(old, seed=seed, bucket=bucket, file_sha=file_sha,
            wins=wins, episodes=episodes, measured_update=update, weight=weight,
            visits=old.get('visits', 0), eligible=eligible,
            ever_qualified=old.get('ever_qualified', False) or qualified,
            qualified_update=update if qualified else old.get('qualified_update'))

    def annotate(self, manifest):
        for key, state in self.states.items():
            entry = manifest['states'][key]
            sources = entry.get('sources', [])
            routes = sorted({s['artifact_id'] for s in sources})
            if not routes or entry['seed'] != state['seed'] or entry['file_sha256'] != state['file_sha']:
                raise ValueError("Missing/mutated source route or state identity")
            state.update(added_update=entry['added_update'], route_id=routes[0],
                         source_routes=routes, env_step=entry['env_step'])

    def active(self):
        return {k:s for k,s in self.states.items() if s['ever_qualified'] and s['eligible']
                and 0 <= self.completed-s['measured_update'] <= 20
                and tier(s['wins'], s['episodes']) == 'mixed'}

    def freeze_representatives(self):
        if self.representatives or self.completed or self.pending:
            raise ValueError("Representatives frozen once before training")
        pool = self.active()
        selected = []
        for _ in range(4):
            if not pool:
                raise ValueError("Insufficient initial representatives")
            used = [self.states[k] for k in selected]
            key = min(pool, key=lambda k:(pool[k]['seed'] in {s['seed'] for s in used},
                pool[k]['route_id'] in {s['route_id'] for s in used},
                pool[k]['bucket'] in {s['bucket'] for s in used}, k))
            selected.append(key); del pool[key]
        self.representatives = selected

    def select(self):
        if self.pending:
            return json.loads(json.dumps(self.pending))
        if self.completed >= 200:
            raise ValueError("Training budget exhausted")
        pool = {k:s for k,s in self.states.items() if s['ever_qualified']}
        if len(pool) < 12:
            raise ValueError("Insufficient retained distinct bank states")
        if any('route_id' not in s or 'added_update' not in s for s in pool.values()):
            raise ValueError("Bank provenance must be annotated before selection")
        # Network: source -> seed (2) -> route (2) -> category -> sink (4 review + 8 other).
        # One unit-capacity edge per state; parallel edges preserve actual state identities.
        graph = {}
        def edge(a, b, capacity, cost, label=None):
            graph.setdefault(a, []); graph.setdefault(b, [])
            forward = [b, len(graph[b]), capacity, cost, label]
            reverse = [a, len(graph[a]), 0, -cost, None]
            graph[a].append(forward); graph[b].append(reverse)
            return forward
        source, sink = ('source',), ('sink',)
        review_node, other_node = ('category','review'), ('category','other')
        edge(review_node,sink,4,0); edge(other_node,sink,8,0)
        routes = {}
        for s in pool.values():
            if s['route_id'] in routes and routes[s['route_id']] != s['seed']:
                raise ValueError("One source route has multiple seeds")
            routes[s['route_id']] = s['seed']
        active = self.active()
        order = sorted(pool, key=lambda k:hashlib.sha256(f'{self.identity}:{self.completed}:{k}'.encode()).hexdigest())
        # One shared state edge prohibits double use across categories.
        for seed in sorted({s['seed'] for s in pool.values()}):
            edge(source,('seed',seed),1,0); edge(source,('seed',seed),1,1000)
        for route,seed in sorted(routes.items()):
            edge(('seed',seed),('route',route),self.route_cap,0)
        refs = []
        for index,key in enumerate(order):
            s = pool[key]; node = ('state',key)
            cost = (0 if key in active else 1000000) + min(s['visits'],200)*100 + index
            edge(('route',s['route_id']),node,1,cost)
            if s['visits'] > 0 or key in self.representatives:
                refs.append((key,'review',edge(node,review_node,1,0,key)))
            recent = 0 <= self.completed-s['added_update'] <= self.recent_window
            refs.append((key,'recent_priority',edge(node,other_node,1,0 if recent else 100000,key)))
        for _ in range(12):
            distance = {source:0}; parents = {}
            for _ in range(len(graph)-1):
                changed = False
                for a,edges in graph.items():
                    if a not in distance: continue
                    for i,e in enumerate(edges):
                        if e[2] and distance.get(e[0],float('inf')) > distance[a]+e[3]:
                            distance[e[0]]=distance[a]+e[3]; parents[e[0]]=(a,i); changed=True
                if not changed: break
            if sink not in parents:
                raise ValueError("Insufficient joint review/seed/route quota; bounded refresh required")
            node = sink
            while node != source:
                a,i = parents[node]; e = graph[a][i]; e[2]-=1; graph[node][e[1]][2]+=1; node=a
        chosen = []
        for key,category,e in refs:
            if e[2] != 0: continue
            s = pool[key]; recent = 0 <= self.completed-s['added_update'] <= self.recent_window
            fallback = []
            if key not in active: fallback.append('retained_previous_qualification')
            if category == 'recent_priority' and not recent: fallback.append('recent_pool_shortage')
            chosen.append(dict(kind='bank',state_id=key,seed=s['seed'],file_sha=s['file_sha'],
                route_id=s['route_id'],bucket=s['bucket'],target_category=category,
                actual_category=('history_review' if s['visits'] else 'initial_representative') if category=='review'
                    else ('recent' if recent else 'other_retained'), fallback=fallback,
                qualification_age=self.completed-s['qualified_update'],
                current_probe_mixed=key in active,added_update=s['added_update'],env_step=s['env_step']))
        chosen.sort(key=lambda s:(s['target_category'] != 'review',s['state_id']))
        openings = [dict(kind='initial',seed=(4*self.completed+j)%28,
                         state_id=f'diverse-initial-{(4*self.completed+j)%28}',env_step=0) for j in range(4)]
        starts = openings+chosen
        validate_starts(starts,self.completed,self.route_cap)
        self.pending = starts
        return json.loads(json.dumps(starts))

    def commit(self, first_groups):
        if len(self.pending) != 16 or len(first_groups) != 16:
            raise ValueError("Expected all sixteen first-sampling groups")
        for start,row in zip(self.pending,first_groups,strict=True):
            if row['start'] != start or row['episodes'] != 12:
                raise ValueError("First-sampling identity/count changed")
            tier(row['wins'],row['episodes'])
        for start,row in zip(self.pending,first_groups,strict=True):
            if start['kind']=='bank':
                s=self.states[start['state_id']];s['visits']+=1
                s.update(wins=row['wins'],episodes=12,measured_update=self.completed)
        self.completed += 1
        self.outcomes.append(dict(update=self.completed,groups=first_groups,
            extreme_groups=sum(r['wins'] in (0,12) for r in first_groups)))
        self.pending=[]

    def dumps(self):
        return json.dumps(dict(schema=CONTRACT,**asdict(self)),sort_keys=True,allow_nan=False)

    @classmethod
    def loads(cls, raw, identity):
        value=json.loads(raw)
        if value.pop('schema',None)!=CONTRACT or value['identity']!=identity:
            raise ValueError("Sampler contract/identity differs")
        obj=cls(**value)
        if obj.pending: validate_starts(obj.pending,obj.completed,obj.route_cap)
        return obj


def validate_starts(starts,completed,route_cap=2):
    if len(starts)!=16 or len({s['state_id'] for s in starts})!=16:
        raise ValueError("Four openings and twelve truly distinct bank states required")
    openings=[s for s in starts if s['kind']=='initial'];bank=[s for s in starts if s['kind']=='bank']
    if [s['seed'] for s in openings]!=[(4*completed+j)%28 for j in range(4)] or len(bank)!=12:
        raise ValueError("Opening rotation or bank quota differs")
    if any(s['seed'] not in range(28) for s in starts):raise ValueError("Held-out seed in training")
    if max(Counter(s['seed'] for s in bank).values())>2 or max(Counter(s['route_id'] for s in bank).values())>route_cap:
        raise ValueError("Bank seed/source-route cap violated")
    if sum(s['target_category']=='review' for s in bank)!=4:
        raise ValueError("Four protected review slots required")
