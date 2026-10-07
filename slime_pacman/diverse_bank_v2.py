"""V2 first-192 quotas. V1 checkpoints are deliberately incompatible."""
from collections import Counter
from dataclasses import dataclass, field, asdict
import json
from .diverse_bank import Policy as V1Policy, tier

CONTRACT = "ascii-diverse-first192-v2"
QUOTAS = {"review": 4, "early": 2, "recent_priority": 6}


class QuotaShortage(ValueError):
    pass


def early_source(source):
    n, i = source.get("route_decision_count"), source.get("decision_index")
    return (type(n) is int and type(i) is int and n > 0 and 0 < i < n and 3*i <= n)


def flow_assignment(pool, roles, preferences, completed, active, fallback):
    """Min-cost flow over seed, primary route, unique state, and role capacity."""
    graph = {}
    def edge(a,b,cap,cost):
        graph.setdefault(a,[]); graph.setdefault(b,[])
        e=[b,len(graph[b]),cap,cost]
        graph[a].append(e); graph[b].append([a,len(graph[a])-1,0,-cost])
        return e
    source,sink=("source",),("sink",)
    for role, count in QUOTAS.items(): edge(("role",role),sink,count,0)
    for seed in sorted({s["seed"] for s in pool.values()}):
        edge(source,("seed",seed),1,0); edge(source,("seed",seed),1,100)
    routes={}
    for s in pool.values():
        route=s["route_id"]
        if route in routes and routes[route]!=s["seed"]:
            raise ValueError("One source route has multiple seeds")
        routes[route]=s["seed"]
    for route,seed in sorted(routes.items()): edge(("seed",seed),("route",route),2,0)
    refs=[]
    for key in sorted(pool):
        rank=sorted(roles).index(key)
        s=pool[key]; node=("state",key)
        edge(("route",s["route_id"]),node,1,(0 if key in active else 10000)+min(s["visits"],200)*10+rank)
        for role in QUOTAS:
            matches=role in roles[key]
            if not matches and not fallback: continue
            cost=(0 if matches else 10000000)
            if role=="review" and matches: cost+=preferences[key]*100000
            refs.append((key,role,edge(node,("role",role),1,cost)))
    for _ in range(12):
        distance={source:0}; parents={}
        for _ in range(len(graph)-1):
            changed=False
            for a,edges in graph.items():
                if a not in distance: continue
                for index,e in enumerate(edges):
                    if e[2] and distance.get(e[0],float("inf"))>distance[a]+e[3]:
                        distance[e[0]]=distance[a]+e[3];parents[e[0]]=(a,index);changed=True
            if not changed: break
        if sink not in parents: return None
        node=sink
        while node!=source:
            a,index=parents[node];e=graph[a][index];e[2]-=1;graph[node][e[1]][2]+=1;node=a
    return [(key,role) for key,role,e in refs if e[2]==0]


@dataclass
class Policy(V1Policy):
    representative_cursor: int = 0
    fallback_ledger: list = field(default_factory=list)
    representative_coverage: dict = field(default_factory=dict)

    def measure(self, *args, **kwargs):
        super().measure(*args, **kwargs)
        key = args[0] if args else kwargs['state_id']
        self.states[key]['reprobe_priority'] = int(self.states[key]['wins'] == 0)

    def annotate(self, manifest):
        super().annotate(manifest)
        route_seeds={}
        for key,s in self.states.items():
            entry=manifest["states"][key]
            s["provenance"]=json.loads(json.dumps(entry["sources"]))
            s["early"]=any(early_source(x) for x in s["provenance"])
            for route in s["source_routes"]:
                if route in route_seeds and route_seeds[route]!=s["seed"]:
                    raise ValueError("One source route has multiple seeds")
                route_seeds[route]=s["seed"]

    def freeze_representatives(self):
        if self.representatives or self.completed or self.pending:
            raise ValueError("Representatives frozen once before training")
        pool=self.active(); selected=[]
        if len(pool)<16: raise ValueError("Insufficient sixteen initial representatives")
        while len(selected)<16:
            used=[self.states[k] for k in selected]
            seeds=Counter(s["seed"] for s in used)
            routes=Counter(r for s in used for r in s["source_routes"])
            buckets=Counter(s["bucket"] for s in used)
            key=min(pool,key=lambda k:(seeds[pool[k]["seed"]],sum(routes[r] for r in pool[k]["source_routes"]),buckets[pool[k]["bucket"]],k))
            selected.append(key);del pool[key]
        self.representatives=selected

    def select(self, *, allow_category_fallback=False, search_limit=4096):
        if self.pending: return json.loads(json.dumps(self.pending))
        if self.completed>=200: raise ValueError("Training budget exhausted")
        pool={k:s for k,s in self.states.items() if s["ever_qualified"]}
        if len(pool)<12: raise QuotaShortage("Insufficient retained distinct bank states")
        if any(not s.get("source_routes") or "early" not in s for s in pool.values()):
            raise ValueError("Bank provenance must be annotated before selection")
        if len(self.representatives)!=16 or len(set(self.representatives))!=16:
            raise ValueError("Sixteen persistent representatives required")
        rotation=self.representatives[self.representative_cursor:]+self.representatives[:self.representative_cursor]
        preferences={k:rotation.index(k) if k in rotation else 16 for k in pool}
        roles={k:(["review"] if k in self.representatives else [])+
                  (["early"] if s["early"] else [])+
                  (["recent_priority"] if 0<=self.completed-s["added_update"]<=4 else []) for k,s in pool.items()}
        active=self.active(); visited=set(); best=None; best_cost=None
        # Primary-route flow is a relaxation. Branch on violating merged-provenance
        # triples: every feasible solution must omit at least one of those states.
        queue=[frozenset()]
        while queue:
            omitted=queue.pop()
            if omitted in visited: continue
            if len(visited)>=search_limit:
                raise QuotaShortage("Insufficient bounded joint solver budget; no infeasibility claim")
            visited.add(omitted)
            subset={k:s for k,s in pool.items() if k not in omitted}
            result=flow_assignment(subset,roles,preferences,self.completed,active,allow_category_fallback)
            if result is None: continue
            cost=sum((0 if role in roles[k] else 10000000)+
                     (preferences[k]*100000 if role=="review" and role in roles[k] else 0)+
                     (0 if k in active else 10000)+min(pool[k]["visits"],200)*10+sorted(pool).index(k)
                     for k,role in result)
            cost += 100 * sum(max(0, n-1) for n in Counter(pool[k]['seed'] for k,_ in result).values())
            if best_cost is not None and cost>=best_cost: continue
            counts=Counter(r for k,_ in result for r in pool[k]["source_routes"])
            bad=next((r for r in sorted(counts) if counts[r]>2),None)
            if bad is not None:
                triple=[k for k,_ in result if bad in pool[k]["source_routes"]][:3]
                queue.extend(omitted|{k} for k in reversed(triple));continue
            best,best_cost=result,cost
        if best is None: raise QuotaShortage("Insufficient joint category/seed/source-route quota; bounded refresh required")
        chosen=[]
        for key,role in best:
            s=pool[key]; reasons=[]
            if role not in roles[key]: reasons.append(role+"_pool_shortage")
            if key not in active: reasons.append("retained_previous_qualification")
            chosen.append(dict(kind="bank",state_id=key,seed=s["seed"],file_sha=s["file_sha"],
                route_id=s["route_id"],source_routes=s["source_routes"],provenance=s["provenance"],
                bucket=s["bucket"],slot_role=role,target_category=role,
                actual_category=("history_review" if s["visits"] else "initial_representative") if role=="review" and role in roles[key]
                    else role if role in roles[key] else "other_retained",
                actual_categories=roles[key],fallback=reasons,qualification_age=self.completed-s["qualified_update"],
                current_probe_mixed=key in active,added_update=s["added_update"],env_step=s["env_step"]))
        chosen.sort(key=lambda s:(list(QUOTAS).index(s["slot_role"]),preferences[s["state_id"]],s["state_id"]))
        openings=[dict(kind="initial",seed=(4*self.completed+j)%28,state_id=f"diverse-initial-{(4*self.completed+j)%28}",env_step=0) for j in range(4)]
        starts=openings+chosen;validate_starts(starts,self.completed)
        self.pending=starts
        return json.loads(json.dumps(starts))

    def commit(self, first_groups):
        starts=json.loads(json.dumps(self.pending))
        super().commit(first_groups)
        review=[s["state_id"] for s in starts if s.get("slot_role")=="review"]
        rotation=self.representatives[self.representative_cursor:]+self.representatives[:self.representative_cursor]
        skipped=[dict(state_id=k,reason='joint_constraints_or_other_slot_role')
                 for k in rotation[:4] if k not in review]
        for key in review:
            if key in self.representatives:
                self.representative_coverage[key]=self.representative_coverage.get(key,0)+1
        # Advance past the last actually selected representative, respecting skipped IDs.
        ranks=[(self.representatives.index(k)-self.representative_cursor)%16 for k in review if k in self.representatives]
        if ranks: self.representative_cursor=(self.representative_cursor+max(ranks)+1)%16
        for start in starts:
            if start["kind"]=="bank":
                self.states[start["state_id"]]["reprobe_priority"]=int(self.states[start["state_id"]]["wins"]==0)
        self.fallback_ledger.append(dict(update=self.completed,deficits=dict(Counter(s["slot_role"] for s in starts if any(x.endswith("_pool_shortage") for x in s.get("fallback",[])))),
            slots=starts,representative_skipped=skipped,representative_cursor=self.representative_cursor))

    def reprobe_order(self):
        active=self.active()
        return sorted((k for k in self.states if k not in active),key=lambda k:(-self.states[k].get("reprobe_priority",int(self.states[k]["wins"]==0)),self.states[k]["measured_update"],k))

    def dumps(self):
        return json.dumps(dict(schema=CONTRACT,**asdict(self)),sort_keys=True,allow_nan=False)

    @classmethod
    def loads(cls,raw,identity):
        value=json.loads(raw)
        if value.pop("schema",None)!=CONTRACT or value["identity"]!=identity:
            raise ValueError("Sampler contract/identity differs")
        obj=cls(**value)
        if not 0<=obj.representative_cursor<16: raise ValueError("Invalid representative cursor")
        if len(obj.representatives)!=16 or len(set(obj.representatives))!=16:
            raise ValueError("Invalid representative pool")
        if obj.pending: validate_starts(obj.pending,obj.completed)
        return obj


def validate_starts(starts,completed,route_cap=2):
    if route_cap!=2 or len(starts)!=16 or len({s["state_id"] for s in starts})!=16:
        raise ValueError("Four openings and twelve truly distinct bank states required")
    openings=[s for s in starts if s["kind"]=="initial"];bank=[s for s in starts if s["kind"]=="bank"]
    if [s["seed"] for s in openings]!=[(4*completed+j)%28 for j in range(4)] or len(bank)!=12:
        raise ValueError("Opening rotation or bank quota differs")
    if any(s["seed"] not in range(28) for s in starts): raise ValueError("Held-out seed in training")
    if any(not s.get("source_routes") for s in bank): raise ValueError("Missing full source provenance")
    if max(Counter(s["seed"] for s in bank).values())>2 or max(Counter(r for s in bank for r in set(s["source_routes"])).values())>2:
        raise ValueError("Bank seed/source-route cap violated")
    if Counter(s["slot_role"] for s in bank)!=Counter(QUOTAS): raise ValueError("V2 role quotas differ")
    if any(s["target_category"]!=s["slot_role"] for s in bank): raise ValueError("Role metadata differs")
