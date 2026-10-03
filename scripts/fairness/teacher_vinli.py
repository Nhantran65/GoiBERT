#!/usr/bin/env python3
"""Full-train CafeBERT ViNLI teacher, matching the existing fold recipe."""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts/bench8_kd')]
import common as C
from goibert import train as T
import scripts.bench8_kd.run as K


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--soft-out', type=Path, default=None)
    a = p.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    T.seed_everything(42)
    device = T.resolve_device('cuda:0')
    precision = T.resolve_precision('bf16', device)
    data = C.load('vinli')
    labels = tuple(data.label_names)
    tok = AutoTokenizer.from_pretrained(C.TOK, revision=C.TOK_REV, local_files_only=True, use_fast=True)
    model, spec = K.model_for('teacher', 'vinli', len(labels))
    model.to(device)
    optimizer = T.make_optimizer(model, 2e-5)
    train, ts = C.make_custom_loader(data.train, 'vinli', tok, labels, 32, 42, True, 256, a.limit)
    valid, vs = C.make_custom_loader(data.validation, 'vinli', tok, labels, 32, 42, False, 256)
    steps = len(train) * a.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, round(steps * .1), steps)
    config = {'task': 'vinli', 'seed': 42, 'learning_rate': 2e-5, 'epochs': a.epochs,
              'batch_size': 32, 'max_length': 256, 'precision': precision,
              'optimizer': 'E14 AdamW', 'include_test': False, 'model_spec': spec,
              'length_stats': {'train': ts, 'validation': vs}}
    C.write(out / 'resolved_config.json', config)
    best, best_epoch, curves = -math.inf, 0, []
    for epoch in range(1, a.epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, count, epoch_start = 0., 0, time.time()
        for batch in train:
            x = T._move_batch(batch, device)
            with T.make_autocast(precision, device):
                logits = model(x['input_ids'], x['attention_mask'], token_type_ids=x.get('token_type_ids'))
                loss = T.compute_loss(logits, x['labels'], num_aspects=0)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite teacher loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(batch['ids']); count += len(batch['ids'])
        vm, rows = C.validation(model, valid, 'vinli', labels, device, precision)
        score = float(vm['selection_f1'])
        curves.append({'epoch': epoch, 'train_loss': total/count, 'selection_f1': score,
                       'validation': vm, 'epoch_seconds': time.time()-epoch_start})
        C.write(out / 'epoch_curves.json', {'epochs': curves})
        if score > best:
            best, best_epoch = score, epoch
            C.write(out / 'best_validation_metrics.json', vm)
            C.write(out / 'validation_predictions.json', {'epoch': epoch, 'rows': rows})
            torch.save({'model': model.state_dict(), 'epoch': epoch}, out / 'best_checkpoint.pt')
        print(f'teacher/vinli epoch {epoch} score {score} best {best}@{best_epoch}', flush=True)
    del model
    torch.cuda.empty_cache()
    model, _ = K.model_for('teacher', 'vinli', len(labels))
    model.load_state_dict(torch.load(out / 'best_checkpoint.pt', map_location='cpu', weights_only=True)['model'])
    model.to(device).eval()
    rm, rr = C.validation(model, valid, 'vinli', labels, device, precision)
    saved = C.read(out / 'validation_predictions.json')['rows']
    from_file=C.evaluate('vinli',labels,[r['gold'] for r in saved],
                         [r['prediction'] for r in saved])
    checks = {'reload_matches_best': abs(rm['selection_f1']-best)<1e-9,
              'reload_predictions_identical': rr == saved,
              'prediction_file_matches_best': abs(from_file['selection_f1']-best)<1e-9,
              'finite_train_loss': all(math.isfinite(e['train_loss']) for e in curves)}
    if not all(checks.values()):
        raise RuntimeError(f'Teacher postrun FAIL: {checks}')
    loader, _ = C.make_custom_loader(data.train, 'vinli', tok, labels, 32, 42, False, 256, a.limit)
    soft = K.extract_oof(model, loader, 'vinli', device, precision)
    assert list(map(str, soft['ids'])) == list(map(str, data.train.ids[:a.limit]))
    target = a.soft_out or ROOT / 'artifacts/fairness/soft_insample/vinli.npz'
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **soft, label_names=np.asarray(labels))
    (out / 'best_checkpoint.pt').unlink()
    C.write(out / 'metadata.json', {**config, 'status': 'completed', 'best_epoch': best_epoch,
                                    'best_selection_f1': best, 'postrun_checks': checks,
                                    'elapsed_seconds': time.time()-started})
    C.write(out / 'status.json', {'status': 'completed', 'best_epoch': best_epoch})


if __name__ == '__main__':
    main()
