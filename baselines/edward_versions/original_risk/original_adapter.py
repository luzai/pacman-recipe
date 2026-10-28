"""Original 28e9133 policy; add risk-ranked actions only on native refusal.

The frozen original module is never edited. In particular, original ordinary
candidate filtering, safe emergency, and non-fallback continuation stay intact.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys

def make_planner(exception_class=None):
    # Archive-only change: freeze the historical risk helper dependency.
    folder = Path(__file__).resolve().parent
    manifest = json.loads((folder / 'MANIFEST.json').read_text())
    for filename, expected in manifest['files'].items():
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == expected, filename
    native_name = 'pacman_env._archived_original_risk_native'
    native = sys.modules.get(native_name)
    if native is None:
        native_spec = importlib.util.spec_from_file_location(native_name, folder / 'frozen_native_planner.py')
        native = importlib.util.module_from_spec(native_spec)
        sys.modules[native_name] = native
        native_spec.loader.exec_module(native)
    folder=Path(__file__).resolve().parent
    pin=json.loads((folder/'ORIGINAL_PIN.json').read_text())
    source=folder/'original_planner.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==pin['sha256']
    name='pacman_env._historical_risk_only_original'
    assert name not in sys.modules, 'Install once in a fresh evaluator process'
    spec=importlib.util.spec_from_file_location(name,source)
    original=importlib.util.module_from_spec(spec);sys.modules[name]=original;spec.loader.exec_module(original)
    if exception_class is not None:original.EdwardSafetyRefusal=exception_class

    class OriginalRiskOnlyPlanner(original.EdwardPlanner):
        def __init__(self,level=None,*,fallback_mode='risk_ranked'):
            assert fallback_mode=='risk_ranked'
            super().__init__(level=level)
            self.fallback_mode='risk_ranked'
            self.risk_only_refusal_count=0

        # Exact frozen native helper/methods; no latest escape information or
        # public-motion predictor is installed on this subclass.
        _motion_risk=native.EdwardPlanner._motion_risk
        _risk_fallback_candidates=native.EdwardPlanner._risk_fallback_candidates

        def advertised_candidates(self,state):
            try:
                return super().advertised_candidates(state)
            except original.EdwardSafetyRefusal:
                self.risk_only_refusal_count+=1
                player=native._position(state.get('pacman_position')) or native._position((state.get('row'),state.get('col')))
                lethal=tuple(native._GhostThreat(g.get('id'),pos) for g in state.get('ghosts') or ()
                    if isinstance(g,dict) and g.get('state') in native._LETHAL_STATES
                    and (pos:=native._position(g.get('position'))) is not None)
                return self._risk_fallback_candidates(state,player,lethal)

        def continue_option(self,option,state):
            if option.strategy=='RISK_FALLBACK':return None,'max_commit'
            return super().continue_option(option,state)

    identity=dict(pin,adapter='original unchanged; catch native refusal only; exact frozen native risk methods',
        fallback_mode='risk_ranked',normal_and_safe_emergency_preserved=True,
        native_fallback_module_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest())
    return OriginalRiskOnlyPlanner,original,identity

def install_original():
    import pacman_recipe.level1.episode as episode
    cls,original,identity=make_planner(episode.EdwardSafetyRefusal)
    episode.EdwardPlanner=cls
    return identity
