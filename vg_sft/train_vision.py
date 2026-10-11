"""Stage 1: full-parameter SFT of the vision encoder + merger, language frozen (FSDP2).

torchrun --nproc_per_node=W: each update's plan draws are split W ways; every rank backprops
loss*W/batch per example and FSDP averages over ranks, so the update equals the batch mean.
fp32 sharded master weights and AdamW, bf16 compute, fp32 gradient reduce; 10% warmup, cosine to
0.1x, clip 1, weight decay 0.01. Checkpoints are exported every --export-every updates as full
bf16 HF weights carrying the cell-contract image config, and must pass check_hf_export.

usage: torchrun --nproc_per_node 4 -m vg_sft.train_vision --model BASE --dataset DIR --plan plan.json --out RUN
"""
import argparse, hashlib, json, os, random, shutil, time
from pathlib import Path

from vg_sft.common import CONTRACT, MAX_PIXELS, MIN_PIXELS, completion_loss, load_processor, lr_factor, make_train_batch, move_batch, read_rows, write_json


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--model', required=True); a.add_argument('--dataset', required=True); a.add_argument('--plan', required=True)
    a.add_argument('--out', required=True); a.add_argument('--validation-file', default='validation.jsonl')
    a.add_argument('--lr', type=float, default=1e-5); a.add_argument('--seed', type=int, default=20261008)
    a.add_argument('--validate-every', type=int, default=100); a.add_argument('--export-every', type=int, default=1000)
    a.add_argument('--wall-minutes', type=float, default=600)
    args = a.parse_args()
    import torch, torch.distributed as dist
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
    from transformers import AutoModelForImageTextToText
    from scripts.level1.report.check_hf_export import check
    dist.init_process_group('nccl'); rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(local); dev = f'cuda:{local}'
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.enable_cudnn_sdp(False); torch.set_num_threads(4)
    root, out = Path(args.dataset), Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        if (out/'history.jsonl').exists(): raise FileExistsError(out)
    dist.barrier()
    plan = json.loads(Path(args.plan).read_text()); rows = {r['id']: r for r in read_rows(root/plan['train_file'])}
    updates = plan['updates']; total = len(updates); val = read_rows(root/args.validation_file)
    assert len(updates[0]) % world == 0
    processor = load_processor(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, local_files_only=True, dtype=torch.float32, attn_implementation='sdpa')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); model.config.use_cache = False
    trainable = 0
    for n, p in model.named_parameters():   # vision encoder + merger only; frozen before sharding
        p.requires_grad_('.visual.' in f'.{n}'); trainable += p.numel() if p.requires_grad else 0
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for blk in model.model.visual.blocks: fully_shard(blk, mp_policy=mp)
    for layer in model.model.language_model.layers: fully_shard(layer, mp_policy=mp)
    fully_shard(model, mp_policy=mp); model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]; assert params
    if rank == 0:
        write_json(out/'manifest.json', dict(args=vars(args), world=world, trainable_parameters=trainable, image_contract=CONTRACT,
                   plan_sha256=hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(), total_updates=total))
        print('MODEL_READY ' + json.dumps(dict(world=world, trainable_parameters=trainable)), flush=True)
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    started = time.monotonic(); history = (out/'history.jsonl').open('w') if rank == 0 else None

    def log(rec, tag):
        if rank == 0: history.write(json.dumps(rec)+'\n'); history.flush(); print(tag+' '+json.dumps(rec), flush=True)

    def val_loss():
        model.eval(); s = torch.zeros(2, device=dev, dtype=torch.float64)
        with torch.no_grad():
            for r in val[rank::world]:
                b, l, k, _ = make_train_batch(processor, root, [r]); c = int((l != -100).sum())
                s[0] += float(completion_loss(model, move_batch(b, dev), l.to(dev), k))*c; s[1] += c
        dist.all_reduce(s); return float(s[0]/s[1])

    def save(update):
        sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)); ck = None
        if rank == 0:
            ck = out/f'update-{update:04d}'; ck.mkdir()
            model.save_pretrained(ck, state_dict={k: t.to(torch.bfloat16) for k, t in sd.items()}, safe_serialization=True)
            # Weights are bf16: the fp32 master dtype must not leak into config.json.
            fix = lambda x: {k: ('bfloat16' if k in ('dtype', 'torch_dtype') and v == 'float32' else fix(v) if isinstance(v, dict) else v) for k, v in x.items()}
            (ck/'config.json').write_text(json.dumps(fix(json.loads((ck/'config.json').read_text())), indent=2))
            processor.save_pretrained(ck)
            for f in list(Path(args.model).glob('*.json')) + list(Path(args.model).glob('*.jinja')):
                if f.name not in ('model.safetensors.index.json', 'preprocessor_config.json') and not (ck/f.name).exists():
                    shutil.copy(f, ck/f.name)
            # processor.save_pretrained writes only processor_config.json; the image config SGLang reads is
            # the base preprocessor_config.json with the cell-contract pixel bounds.
            pre = json.loads((Path(args.model)/'preprocessor_config.json').read_text())
            pre['size'] = dict(pre.get('size') or {}, shortest_edge=MIN_PIXELS, longest_edge=MAX_PIXELS)
            pre['pacman_image_contract'] = CONTRACT
            (ck/'preprocessor_config.json').write_text(json.dumps(pre, indent=2) + '\n')
            problems = check(ck, min_pixels=MIN_PIXELS)
            if problems: raise RuntimeError(f'export {ck} failed checks: {problems}')
        del sd; dist.barrier(); return ck

    def validate(update):
        v = val_loss(); ck = save(update) if update and (update % args.export_every == 0 or update == total) else None
        log(dict(event='validation', update=update, validation_loss=v, checkpoint=str(ck) if ck else None,
                 seconds=round(time.monotonic()-started, 1)), 'VALIDATION')

    validate(0); done = 0
    for update, ids in enumerate(updates, 1):
        if time.monotonic()-started > args.wall_minutes*60: log(dict(event='wall_budget_exhausted', update=done), 'STOP'); break
        model.train(); factor = lr_factor(update, total)
        for g in optimizer.param_groups: g['lr'] = args.lr*factor
        optimizer.zero_grad(set_to_none=True); acc = torch.zeros(1, device=dev)
        for i in ids[rank::world]:
            b, l, k, _ = make_train_batch(processor, root, [rows[i]])
            loss = completion_loss(model, move_batch(b, dev), l.to(dev), k); assert torch.isfinite(loss)
            (loss*world/len(ids)).backward(); acc += loss.detach()/len(ids)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0); gn = gn.full_tensor() if hasattr(gn, 'full_tensor') else gn
        assert torch.isfinite(gn) and float(gn) > 0
        optimizer.step(); dist.all_reduce(acc); done = update
        log(dict(update=update, loss=float(acc), lr=args.lr*factor, grad_norm=float(gn), examples=len(ids),
                 seconds=round(time.monotonic()-started, 1), peak_memory_gib=torch.cuda.max_memory_allocated()/2**30), 'TRAIN')
        if update % args.validate_every == 0 or update == total: validate(update)
    if done != total and done % args.export_every: validate(done)
    if rank == 0: print('TRAINING_COMPLETE', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
