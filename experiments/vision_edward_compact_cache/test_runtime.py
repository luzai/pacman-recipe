import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).parent / 'runtime'


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FinalizeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.module = load('cache_tis_gate')
        self.previous = dict(pre_sync_checks_passed=True, numerical_gate_passed=False,
                             cache_tis_evidence=dict(served_version='1'))
        (self.root / 'first-update-pre-sync-gate.json').write_text(json.dumps(self.previous))
        self.receipt = dict(boundary='post-weight-update', rollout_id=0, served_version='2',
                            engines=[f'http://engine-{i}' for i in range(8)], mode='fixed')
        self.receipt['warmup'] = [dict(endpoint=x, version='2', prewarm_cached_tokens=0) for x in self.receipt['engines']]

    def tearDown(self):
        self.temp.cleanup()

    def run_finalize(self):
        with patch.object(self.module, 'Path', side_effect=lambda p: self.root / Path(p).name):
            with patch.dict('os.environ', {'PACMAN_FIRST_NEW_UPDATE': '0'}):
                return self.module.finalize_after_sync(self.receipt)

    def test_only_publish_after_new_version_all_engines(self):
        result = self.run_finalize()
        self.assertTrue(result['numerical_gate_passed'])
        self.assertTrue(result['cache_tis_evidence']['current_run_post_adam1_sync_verified'])
        self.assertTrue((self.root / 'first-update-gate.json').is_file())

    def test_same_version_rejected_without_marker(self):
        self.receipt['served_version'] = '1'
        with self.assertRaises(AssertionError):
            self.run_finalize()
        self.assertFalse((self.root / 'first-update-gate.json').exists())

    def test_missing_engine_rejected(self):
        self.receipt['warmup'].pop()
        with self.assertRaises(AssertionError):
            self.run_finalize()

    def test_mixed_version_rejected(self):
        self.receipt['warmup'][0]['version'] = '1'
        with self.assertRaises(AssertionError):
            self.run_finalize()

    def test_cannot_replace_completed_gate(self):
        self.run_finalize()
        with self.assertRaises(FileExistsError):
            self.run_finalize()

    def test_preexisting_kv_rejected_after_update(self):
        self.receipt['warmup'][0]['prewarm_cached_tokens'] = 512
        with self.assertRaises(AssertionError):
            self.run_finalize()

    def test_missing_invalidation_receipt_rejected(self):
        del self.receipt['warmup'][0]['prewarm_cached_tokens']
        with self.assertRaises(AssertionError):
            self.run_finalize()

    def test_order_kv_warm_finalize_before_resume(self):
        source = (ROOT / 'formal_train.py').read_text()
        tail = source[source.index('        actor_model.update_weights()'):]
        self.assertLess(tail.index('onload_kv.remote'), tail.index('prepare_cache.remote'))
        self.assertLess(tail.index('prepare_cache.remote'), tail.index('finalize_after_sync(cache_receipt)'))
        self.assertLess(tail.index('finalize_after_sync(cache_receipt)'), tail.index('resume_rollout_admission.remote'))


class BoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, versions, calls=1):
        module = load('cache_boundary')
        endpoints = [f'http://engine-{i}' for i in range(8)]
        warmer = AsyncMock(return_value=[dict(endpoint=x, version='1') for x in endpoints])
        class Response:
            def __init__(self, value): self.value = value
            def raise_for_status(self): pass
            def json(self): return self.value
        class Client:
            def __init__(self, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def get(self, url):
                if url.endswith('/workers'):
                    return Response(dict(workers=[dict(url=x) for x in endpoints]))
                return Response(dict(weight_version=versions[endpoints.index(url.removesuffix('/get_model_info'))]))
        stubs = {'httpx': SimpleNamespace(AsyncClient=Client),
                 'slime.rollout.sglang_rollout': SimpleNamespace(get_model_url=lambda *args: 'http://router/generate'),
                 'slime.utils.http_utils': SimpleNamespace(get_rollout_num_engines=lambda args: 8),
                 'slime_pacman.prefix_warmup': SimpleNamespace(warm_router=warmer)}
        with tempfile.TemporaryDirectory() as directory:
            contract = Path(directory) / 'contract.json'
            contract.write_text(json.dumps(dict(mode='fixed')))
            with patch.dict('sys.modules', stubs), patch.dict('os.environ', {
                'PACMAN_CACHE_CONTRACT': str(contract), 'PACMAN_RUN_DIR': directory}):
                for _ in range(calls):
                    await module.prepare(SimpleNamespace(), 'generate', 0)
        return warmer

    async def test_same_version_still_warms_again_after_boundary(self):
        warmer = await self.exercise(['1'] * 8, calls=2)
        self.assertEqual(warmer.await_count, 2)

    async def test_mixed_served_versions_fail_before_warm(self):
        with self.assertRaises(ValueError):
            await self.exercise(['1'] * 7 + ['2'])


if __name__ == '__main__':
    unittest.main()
