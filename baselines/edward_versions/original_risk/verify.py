"""Focused differential CPU tests, independent of model choice."""
from pathlib import Path
import sys,json,dataclasses
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parents[2]))
from original_adapter import make_planner
Risk,original,identity=make_planner()
native=sys.modules['pacman_env._archived_original_risk_native']
base=original.EdwardPlanner();risk=Risk();start=base.level.pacman_start
state=dict(pacman_position=[start.row,start.col],ghosts=[],open=['U','D','L','R'])
def encode(cs):return [dataclasses.asdict(c) for c in cs]
# Genuine original candidate path; no fallback needed.
assert encode(base.advertised_candidates(state))==encode(risk.advertised_candidates(state))
assert risk.risk_only_refusal_count==0
for candidate in base.advertised_candidates(state):
 assert base.continue_option(candidate,state)==risk.continue_option(candidate,state)
# Exercise original safe-emergency path. Both instances have no ordinary goals;
# original emergency algorithm remains the first attempt, not replaced by risk.
base.candidates=lambda s:();risk.candidates=lambda s:()
safe=base.advertised_candidates(state)
assert encode(safe)==encode(risk.advertised_candidates(state))
assert safe[0].strategy=='AVOID' and risk.risk_only_refusal_count==0
# Force the original refusal boundary and compare only the added risk method
# against its frozen native source, including all physically-open actions.
def refuse(*a,**k):raise original.EdwardSafetyRefusal('controlled test refusal')
risk._emergency_candidate=refuse
added=risk.advertised_candidates(state)
control=native.EdwardPlanner(level=risk.level,fallback_mode='risk_ranked')
expected=control._risk_fallback_candidates(state,start,())
assert encode(added)==encode(expected)
assert all(c.strategy=='RISK_FALLBACK' and c.commit_moves==1 and c.risk for c in added)
assert {c.first_action for c in added}=={c.first_action for c in expected}
assert all(risk.continue_option(c,state)==(None,'max_commit') for c in added)
assert risk.risk_only_refusal_count==1
# No physical cardinal action remains a refusal; never invent a legal move.
try:risk.advertised_candidates(dict(state,open=[]))
except native.EdwardSafetyRefusal:pass
else:raise AssertionError('Physical legality refusal was swallowed')
assert original.EdwardPlanner.advertised_candidates is Risk.__mro__[1].advertised_candidates
print(json.dumps(dict(passed=True,checks=['immutable_original_sha','normal_candidates_identical','ordinary_continuation_identical','safe_emergency_identical','refusal_only_risk_equals_native','risk_single_step','physical_legality_refusal'],identity=identity)))
