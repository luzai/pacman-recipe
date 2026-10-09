"""Validate an exported Hugging Face checkpoint before serving it (SGLang) or training from it.

Catches failures seen in practice: float32 leaking into config.json from fp32 master weights
(SGLang then picks fp16 and crashes), a base model.safetensors.index.json pointing at shards that
do not exist, a stale preprocessor resolution (65536 instead of 537600, one visual token per
board cell), non-bf16 tensors, and missing tokenizer/template files. Reads only JSON and
safetensors headers; no torch needed.
usage: python scripts/level1/report/check_hf_export.py DIR [DIR ...] [--min-pixels 537600]
"""
import json, struct, sys
from pathlib import Path

REQUIRED = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'preprocessor_config.json')


def _dtype_fields(x, path=''):
    if isinstance(x, dict):
        for k, v in x.items():
            p = f'{path}.{k}' if path else k
            if k in ('dtype', 'torch_dtype'):
                yield p, v
            yield from _dtype_fields(v, p)


def check(ck, min_pixels=537600):
    ck = Path(ck); problems = []
    for f in REQUIRED:
        if not (ck/f).exists(): problems.append(f'missing {f}')
    if (ck/'config.json').exists():
        cfg = json.loads((ck/'config.json').read_text())
        problems += [f'config {p}={v} (expected bfloat16)' for p, v in _dtype_fields(cfg) if v != 'bfloat16']
    shards = sorted(ck.glob('model*.safetensors'))
    if not shards: problems.append('no model*.safetensors')
    if (ck/'model.safetensors.index.json').exists():
        idx = json.loads((ck/'model.safetensors.index.json').read_text())
        missing = sorted({v for v in idx.get('weight_map', {}).values() if not (ck/v).exists()})
        if missing: problems.append(f'index.json points at missing shards {missing[:3]}')
    for s in shards:
        with s.open('rb') as fh:
            n = struct.unpack('<Q', fh.read(8))[0]; header = json.loads(fh.read(n))
        bad = sorted({v['dtype'] for k, v in header.items() if k != '__metadata__' and v['dtype'] != 'BF16'})
        if bad: problems.append(f'{s.name}: tensor dtypes {bad} (expected BF16)')
    if (ck/'preprocessor_config.json').exists():
        pre = json.loads((ck/'preprocessor_config.json').read_text())
        edge = (pre.get('size') or {}).get('shortest_edge', pre.get('min_pixels'))
        if edge != min_pixels: problems.append(f'preprocessor_config shortest_edge={edge} (expected {min_pixels})')
    if (ck/'processor_config.json').exists():
        ip = json.loads((ck/'processor_config.json').read_text()).get('image_processor') or {}
        edge = (ip.get('size') or {}).get('shortest_edge')
        if edge is not None and edge != min_pixels: problems.append(f'processor_config image shortest_edge={edge} (expected {min_pixels})')
    return problems


def main(argv):
    mp = 537600
    if '--min-pixels' in argv:
        i = argv.index('--min-pixels'); mp = int(argv[i+1]); argv = argv[:i] + argv[i+2:]
    bad = 0
    for d in argv:
        p = check(d, mp); bad += bool(p)
        print(('EXPORT_OK ' if not p else 'EXPORT_BAD ') + d + ('' if not p else ' :: ' + '; '.join(p)), flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
