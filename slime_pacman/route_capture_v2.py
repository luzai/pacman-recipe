"""Complete route capture, independent of the legacy death ring.

Indices are zero-based model decision boundaries relative to this source episode.
N is its final decision count; endpoint is N (after the last decision). Targets
are floor(N/6), floor(N/3), excluding index 0. Each retained source stores N and
index; full restart context is hashed independently from the source metadata.
"""
import json
import os
from pathlib import Path
from .dynamic_bank import state_identity

CANDIDATE_SCHEMA = "pacman-bank-route-candidate-v2"


def route_candidates(boundaries, terminal_reason):
    boundaries=list(boundaries);n=len(boundaries)
    positions={}
    for divisor in (6,3):
        i=n//divisor
        if 0<i<n: positions.setdefault(i,[]).append(f"early_1/{divisor}")
    if terminal_reason=="death":
        for distance in (4,8,16):
            i=n-1-distance
            if i>=0: positions.setdefault(i,[]).append(f"death_before_{distance}")
    candidates={}
    for i,origins in sorted(positions.items()):
        boundary=boundaries[i];key=state_identity(boundary)
        # Index 0 can be a legitimate pre-death restart, but never an early slot.
        source=dict(decision_index=i,route_decision_count=n,route_endpoint=n,
                    index_basis="source_episode_zero_based",rounding="floor",origins=origins,
                    distance=n-1-i)
        if key in candidates:
            candidates[key]["positions"].append(source)
        else:
            candidates[key]=dict(candidate_id=key,distance=n-1-i,boundary=boundary,positions=[source])
    return list(candidates.values())


def candidate_order(candidates):
    if len({c["candidate_id"] for c in candidates})!=len(candidates):
        raise ValueError("Duplicate candidate state identity")
    # Early candidates consume the same finite probe ledger; no extra budget.
    return sorted(candidates,key=lambda c:(not any(3*p["decision_index"]<=p["route_decision_count"] and p["decision_index"]>0 for p in c["positions"]),-c["distance"],c["candidate_id"]))


def write_route_candidates(path,boundaries,episode,source):
    candidates=route_candidates(boundaries,episode.terminal_reason)
    if not candidates: return None
    last=boundaries[-1]
    payload=dict(schema=CANDIDATE_SCHEMA,source=dict(source,terminal_reason=episode.terminal_reason,
        reward=episode.reward,weight_version=episode.weight_version,decision_count=len(boundaries)),
        seed=last["env_state"]["payload"]["episode"]["seed"],
        last_choice_position=last["features"]["pacman_position"],candidates=candidates)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise ValueError("Route capture cannot overwrite existing provenance")
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload,allow_nan=False),encoding="utf-8")
    os.replace(temporary,path)
    return path
