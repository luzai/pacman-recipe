"""Stage 2: language LoRA on top of the stage-1 checkpoint, vision and merger frozen.

LoRA r16 / alpha 32 / dropout 0.05 on every linear layer, then every LoRA tensor outside the
language model is frozen (their lora_B stays zero, so they contribute nothing; checked at the
end). Single GPU, batch = the plan's 16 draws as microbatches of 1, AdamW, 10% warmup, cosine to
0.1x, clip 1. Adapters are saved every --validate-every updates; use the final one.

usage: python -m vg_sft.train_lora --model STAGE1 --dataset DIR --plan plan-lora.json --out RUN
"""
import argparse, hashlib, json, random, time
from pathlib import Path

from vg_sft.common import completion_loss, load_model, load_processor, lr_factor, make_train_batch, move_batch, read_rows, write_json


def is_language(name): return '.language_model.' in name


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--model', required=True); a.add_argument('--dataset', required=True); a.add_argument('--plan', required=True)
    a.add_argument('--out', required=True); a.add_argument('--validation-file', default='validation.jsonl')
    a.add_argument('--lr', type=float, default=1e-4); a.add_argument('--rank', type=int, default=16)
    a.add_argument('--seed', type=int, default=20261008); a.add_argument('--validate-every', type=int, default=250)
    a.add_argument('--wall-minutes', type=float, default=480)
    args = a.parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    root, out = Path(args.dataset), Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out/'history.jsonl').exists(): raise FileExistsError(out)
    plan = json.loads(Path(args.plan).read_text()); rows = {r['id']: r for r in read_rows(root/plan['train_file'])}
    updates = plan['updates']; total = len(updates); val = read_rows(root/args.validation_file)
    processor = load_processor(args.model); model = load_model(args.model)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2*args.rank, lora_dropout=0.05, bias='none',
                                             task_type='CAUSAL_LM', target_modules='all-linear'))
    frozen = {}
    for n, p in model.named_parameters():
        if 'lora_' not in n: assert not p.requires_grad; continue
        p.requires_grad_(is_language(n))
        if not p.requires_grad: frozen[n] = p.detach().cpu().clone()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads(); model.config.use_cache = False
    params = [p for p in model.parameters() if p.requires_grad]; assert params
    write_json(out/'manifest.json', dict(args=vars(args), trainable_parameters=sum(p.numel() for p in params),
               plan_sha256=hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(), total_updates=total))
    print('MODEL_READY ' + json.dumps(dict(trainable_parameters=sum(p.numel() for p in params))), flush=True)
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    started = time.monotonic(); history = (out/'history.jsonl').open('w')

    def validate(update):
        model.eval(); total_loss = tokens = 0.0
        with torch.inference_mode():
            for r in val:
                b, l, k, _ = make_train_batch(processor, root, [r]); c = int((l != -100).sum())
                total_loss += float(completion_loss(model, move_batch(b), l.to('cuda:0'), k))*c; tokens += c
        ck = out/f'update-{update:04d}'
        if update: model.save_pretrained(ck); processor.save_pretrained(ck)
        rec = dict(event='validation', update=update, validation_loss=total_loss/tokens, checkpoint=str(ck) if update else None,
                   seconds=round(time.monotonic()-started, 1))
        history.write(json.dumps(rec)+'\n'); history.flush(); print('VALIDATION '+json.dumps(rec), flush=True)

    validate(0); done = 0
    for update, ids in enumerate(updates, 1):
        if time.monotonic()-started > args.wall_minutes*60: print('WALL_BUDGET_EXHAUSTED', flush=True); break
        model.train(); factor = lr_factor(update, total)
        for g in optimizer.param_groups: g['lr'] = args.lr*factor
        optimizer.zero_grad(set_to_none=True); mean = 0.0
        for i in ids:
            b, l, k, _ = make_train_batch(processor, root, [rows[i]])
            loss = completion_loss(model, move_batch(b), l.to('cuda:0'), k); assert torch.isfinite(loss)
            (loss/len(ids)).backward(); mean += float(loss.detach())/len(ids)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step(); done = update
        rec = dict(update=update, loss=mean, lr=args.lr*factor, grad_norm=float(gn), seconds=round(time.monotonic()-started, 1),
                   peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
        history.write(json.dumps(rec)+'\n'); history.flush(); print('TRAIN '+json.dumps(rec), flush=True)
        if update % args.validate_every == 0 or update == total: validate(update)
    if done != total and done % args.validate_every: validate(done)
    now = dict(model.named_parameters())
    assert all(torch.equal(t, now[n].detach().cpu()) for n, t in frozen.items()), 'a frozen non-language LoRA tensor changed'
    print('TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
