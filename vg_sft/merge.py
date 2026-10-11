"""Stage 3: merge the language LoRA into the stage-1 checkpoint and export full bf16 HF weights.

Tokenizer, chat template and the cell-contract processor files are copied from the stage-1
checkpoint; the result must pass check_hf_export before it is served or used to start RL.

usage: python -m vg_sft.merge --base STAGE1 --adapter RUN/update-1000 --out MERGED
"""
import argparse, json, shutil
from pathlib import Path

from vg_sft.common import MIN_PIXELS


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--base', required=True, type=Path); a.add_argument('--adapter', required=True); a.add_argument('--out', required=True, type=Path)
    args = a.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText
    from scripts.level1.report.check_hf_export import check
    if args.out.exists(): raise FileExistsError(args.out)
    model = AutoModelForImageTextToText.from_pretrained(args.base, local_files_only=True, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    for f in list(args.base.glob('*.json')) + list(args.base.glob('*.jinja')):
        if f.name not in ('config.json', 'model.safetensors.index.json') and not (args.out/f.name).exists():
            shutil.copy(f, args.out/f.name)
    problems = check(args.out, min_pixels=MIN_PIXELS)
    if problems: raise RuntimeError(f'export {args.out} failed checks: {problems}')
    pre = json.loads((args.out/'preprocessor_config.json').read_text())
    print('MERGED ' + json.dumps(dict(out=str(args.out), contract=pre.get('pacman_image_contract'))), flush=True)


if __name__ == '__main__':
    main()
