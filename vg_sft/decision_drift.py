"""Optional: how much an SFT stage moved the option-code policy on real image RL decision prompts.

collect: run the real slime EpisodeRunner (image observation, cell contract, Edward risk_ranked,
         level 1) with a scripted teacher and store every decision prompt (messages, the 2x
         screenshot as PNG, the allowed option-code token ids).
drift:   one model's next-token distribution on each prompt: probability mass on the allowed
         codes, support-normalized log-probs, argmax.
compare: argmax agreement and KL(ref || other) on the support.

usage: python -m vg_sft.decision_drift collect --game PACMAN_PYTHON --slime SLIME --model M --out DIR [--seeds 94000 94001]
       python -m vg_sft.decision_drift drift --model M --out DIR --name NAME
       python -m vg_sft.decision_drift compare --out DIR --ref NAME --other NAME
"""
import argparse, base64, json, math, os
from pathlib import Path

from vg_sft.common import CONTRACT, load_model, load_processor


def read(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def collect(args):
    import asyncio
    from pacman_recipe.level1.contracts import make_episode_record
    from pacman_recipe.level1.vision_prompt import VISION_CONTRACT_ENV
    from slime_pacman.config import PacmanConfig
    from slime_pacman.generation import Decision
    from slime_pacman.rollout import EpisodeRunner
    from transformers import AutoTokenizer
    os.environ[VISION_CONTRACT_ENV] = CONTRACT
    recipe = Path(__file__).resolve().parents[1]
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    out = Path(args.out); (out/'images').mkdir(parents=True); rows = []
    for seed in args.seeds:
        rec = make_episode_record(seed, split='test', recipe_root=recipe, game_root=Path(args.game), backend_root=Path(args.slime),
                                  max_steps=512, observation_mode='image', edward_fallback_mode='risk_ranked')
        cfg = PacmanConfig(observation_mode='image', edward_fallback_mode='risk_ranked', vision_image_contract=CONTRACT)

        async def gen(messages, constraint, seed=seed):
            k = len(rows); content = []
            for part in messages[1]['content']:
                if part.get('type') == 'image_url':
                    name = f'images/d{k:05d}.png'; (out/name).write_bytes(base64.b64decode(part['image_url']['url'].split(',', 1)[1]))
                    content.append({'type': 'image', 'path': name})
                else:
                    content.append({'type': 'text', 'text': part['text']})
            rows.append(dict(id=k, seed=seed, system=messages[0]['content'], content=content, allowed=list(constraint.allowed_token_ids)))
            t = constraint.allowed_token_ids[0]
            return Decision('teacher', [], t, tok.decode([t]), constraint.allowed_token_ids, 0., 'scripted', {}, None, [])
        asyncio.run(EpisodeRunner(rec, tokenizer=tok, generate=gen, config=cfg).collect(empty_weight_version='scripted'))
        print(json.dumps(dict(seed=seed, decisions=len(rows))), flush=True)
    (out/'decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


def drift(args):
    import torch
    from PIL import Image
    out = Path(args.out); processor = load_processor(args.model); model = load_model(args.model).eval(); res = []
    with torch.inference_mode():
        for d in read(out/'decisions.jsonl'):
            image = None; content = []
            for p in d['content']:
                if p['type'] == 'image': image = Image.open(out/p['path']).convert('RGB'); content.append({'type': 'image'})
                else: content.append(p)
            text = processor.apply_chat_template([{'role': 'system', 'content': d['system']}, {'role': 'user', 'content': content}],
                                                 tokenize=False, add_generation_prompt=True, enable_thinking=False)
            batch = processor(text=[text], images=[image], return_tensors='pt').to('cuda:0')   # screenshot is already 2x
            w, h = image.size; assert batch['image_grid_thw'].tolist() == [[1, h//16, w//16]]
            logits = model(**batch, use_cache=False).logits[0, -1].float()
            allowed = torch.tensor(d['allowed'], device='cuda:0'); sub = torch.log_softmax(logits[allowed], -1)
            res.append(dict(id=d['id'], mass=float(torch.log_softmax(logits, -1)[allowed].exp().sum()), logp=sub.tolist(),
                            argmax=int(d['allowed'][int(sub.argmax())])))
    n = len(res)
    summ = dict(name=args.name, decisions=n, allowed_mass_mean=sum(r['mass'] for r in res)/n,
                entropy_mean=sum(-sum(math.exp(v)*v for v in r['logp']) for r in res)/n)
    (out/f'drift-{args.name}.json').write_text(json.dumps(dict(summary=summ, rows=res))); print('DRIFT ' + json.dumps(summ), flush=True)


def compare(args):
    out = Path(args.out); a = {r['id']: r for r in json.loads((out/f'drift-{args.ref}.json').read_text())['rows']}
    b = {r['id']: r for r in json.loads((out/f'drift-{args.other}.json').read_text())['rows']}; ids = sorted(a)
    summ = dict(ref=args.ref, other=args.other, decisions=len(ids), argmax_agreement=sum(a[i]['argmax'] == b[i]['argmax'] for i in ids)/len(ids),
                kl_ref_other=sum(sum(math.exp(p)*(p-q) for p, q in zip(a[i]['logp'], b[i]['logp'])) for i in ids)/len(ids))
    (out/f'compare-{args.ref}-{args.other}.json').write_text(json.dumps(summ, indent=1)); print('COMPARE ' + json.dumps(summ), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0]); p.add_argument('mode', choices=['collect', 'drift', 'compare'])
    p.add_argument('--game'); p.add_argument('--slime'); p.add_argument('--model'); p.add_argument('--out', required=True)
    p.add_argument('--seeds', type=int, nargs='*', default=[94000, 94001, 94002, 94003]); p.add_argument('--name')
    p.add_argument('--ref'); p.add_argument('--other')
    args = p.parse_args(); {'collect': collect, 'drift': drift, 'compare': compare}[args.mode](args)


if __name__ == '__main__':
    main()
