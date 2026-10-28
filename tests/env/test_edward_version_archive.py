"""Archive identity, runnable original intervention, and configurable life modes."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'baselines/edward_versions'


def test_frozen_sources_and_latest_identity():
    folder = ARCHIVE / 'original_risk'
    manifest = json.loads((folder / 'MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
    versions = json.loads((ARCHIVE / 'VERSIONS.json').read_text())
    latest = versions['versions']['latest']
    source = (ROOT / latest['source']).read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(source).hexdigest() == latest['canonical_lf_sha256']


def test_original_intervention_in_fresh_process():
    result = subprocess.run([sys.executable, str(ARCHIVE / 'original_risk/verify.py')],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    proof = json.loads(result.stdout)
    assert proof['passed'] and len(proof['checks']) == 7
    assert proof['identity']['normal_and_safe_emergency_preserved']


def test_life_overrides_and_risk_only_results():
    for life in ('single_death', 'original_three_lives'):
        override = json.loads((ARCHIVE / f'latest-{life}.json').read_text())
        assert override['edward_harness'] == 'latest'
        assert override['edward_fallback_mode'] == 'risk_ranked'
        assert override['episode_life_mode'] == life
    cells = json.loads((ARCHIVE / 'EVALUATION.json').read_text())['cells']
    assert len(cells) == 8
    assert {x['arm'] for x in cells} == {'original_risk', 'latest'}
    assert all(x['n'] == 64 for x in cells)
