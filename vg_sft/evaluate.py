"""Greedy evaluation on a frozen test file.

Map questions are decoded with a schema-only grammar (exactly R rows of C space-separated
symbols, R x C taken from the row's own board; it never reads the image or the target) and,
with --raw, also without any constraint to measure format validity. Other tasks are decoded raw.
Outputs one JSONL per decoding mode; score with vg_sft.score.

usage: python -m vg_sft.evaluate --model M --dataset DIR --sets test-size --out EVALDIR [--raw]
"""
import argparse, json, time
from pathlib import Path

from vg_sft.common import CONTRACT, images_for, load_model, load_processor, move_batch, prompt_text, read_rows, write_json
from vg_sft.labels import SYMBOLS


class GridGrammar:
    """Tokens whose text uses only template characters; per-board templates on top."""
    def __init__(self, tokenizer):
        self.tokenizer, self.eos = tokenizer, tokenizer.eos_token_id
        valid = set('{}[]",: rows') | SYMBOLS; special = set(tokenizer.all_special_ids)
        ids = sorted(set(tokenizer.get_vocab().values()) - special); self.fragments = {}
        for start in range(0, len(ids), 4096):
            group = ids[start:start+4096]
            for i, t in zip(group, tokenizer.batch_decode([[i] for i in group], skip_special_tokens=False, clean_up_tokenization_spaces=False)):
                if t and set(t) <= valid: self.fragments[i] = t

    def constraint(self, prompt_length, rows, cols):
        return GridConstraint(self, prompt_length, rows, cols)


class GridConstraint:
    """prefix_allowed_tokens_fn for {"rows":[R strings of C symbols separated by single spaces]}."""
    def __init__(self, grammar, prompt_length, rows, cols):
        self.g, self.prompt_length = grammar, prompt_length; self.cache = {}
        self.template = json.dumps({'rows': [' '.join(['\x00']*cols)]*rows}, separators=(',', ':')).replace('\\u0000', '\x00')

    def advance(self, position, fragment):
        if not fragment or position+len(fragment) > len(self.template): return None
        for char in fragment:
            expected = self.template[position]
            if (char not in SYMBOLS) if expected == '\x00' else (char != expected): return None
            position += 1
        return position

    def allowed_at(self, position):
        if position == len(self.template): return [self.g.eos]
        if position not in self.cache:
            self.cache[position] = [i for i, t in self.g.fragments.items() if self.advance(position, t) is not None]
            if not self.cache[position]: raise ValueError(f'tokenizer cannot represent grammar position {position}')
        return self.cache[position]

    def __call__(self, batch_id, input_ids):
        completion = input_ids.tolist()[self.prompt_length:]
        if any(i not in self.g.fragments for i in completion): raise ValueError('invalid or special token inside grid')
        decoded = ''.join(self.g.fragments[i] for i in completion)
        if decoded != self.g.tokenizer.decode(completion, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            raise ValueError('tokenizer decoding is context dependent; unsupported')
        if completion and self.advance(0, decoded) != len(decoded): raise ValueError('invalid generated grid prefix')
        return self.allowed_at(len(decoded))

    def validate(self, text):
        if self.advance(0, text) != len(self.template): raise ValueError('incomplete or invalid grid')


def generate(model, processor, grammar, root, rows, path, mode, max_new_tokens):
    import torch
    path = Path(path); assert not path.exists()
    processor.tokenizer.padding_side = 'left'; model.eval(); started = time.monotonic()
    with path.open('w') as stream, torch.inference_mode():
        for k, r in enumerate(rows, 1):
            batch = move_batch(processor(text=[prompt_text(processor, r['question'])], images=images_for(root, [r]), return_tensors='pt'))
            n = batch['input_ids'].shape[1]; kw = {}; error = None
            if mode == 'fixed':
                c = grammar.constraint(n, len(r['grid']), len(r['grid'][0])); kw['prefix_allowed_tokens_fn'] = c
            try:
                ids = model.generate(**batch, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
                                     pad_token_id=processor.tokenizer.pad_token_id, eos_token_id=processor.tokenizer.eos_token_id, **kw)[0, n:]
                raw = processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                raw = raw if mode == 'fixed' else raw.strip()
                if mode == 'fixed': c.validate(raw)
            except ValueError as e:
                raw, error = '', str(e)
            stream.write(json.dumps(dict(id=r['id'], raw_text=raw, mode=mode, error=error, image_grid_thw=batch['image_grid_thw'].tolist()))+'\n')
            stream.flush()
            if k % 16 == 0 or k == len(rows):
                print(json.dumps(dict(event='eval_progress', file=path.name, done=k, total=len(rows), seconds=round(time.monotonic()-started, 1))), flush=True)


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--model', required=True); a.add_argument('--dataset', required=True, type=Path); a.add_argument('--out', required=True, type=Path)
    a.add_argument('--sets', nargs='+', required=True, help='test file stems, e.g. test-family test-size test-stream')
    a.add_argument('--raw', action='store_true', help='also decode maps without the grammar'); a.add_argument('--limit', type=int, default=0)
    a.add_argument('--map-tokens', type=int, default=1024)
    args = a.parse_args(); args.out.mkdir(parents=True)
    model = load_model(args.model); processor = load_processor(args.model); grammar = GridGrammar(processor.tokenizer)
    write_json(args.out/'manifest.json', dict(args={k: str(v) for k, v in vars(args).items()}, greedy=True, image_contract=CONTRACT))
    for name in args.sets:
        rows = read_rows(args.dataset/f'{name}.jsonl', args.limit)
        maps = [r for r in rows if r['task'] == 'map']; other = [r for r in rows if r['task'] != 'map']
        generate(model, processor, grammar, args.dataset, maps, args.out/f'{name}-maps-fixed.jsonl', 'fixed', args.map_tokens)
        if other: generate(model, processor, grammar, args.dataset, other, args.out/f'{name}-other.jsonl', 'raw', 768)
        if args.raw: generate(model, processor, grammar, args.dataset, maps, args.out/f'{name}-maps-raw.jsonl', 'raw', args.map_tokens)
    print('EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
