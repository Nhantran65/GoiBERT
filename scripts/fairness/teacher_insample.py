#!/usr/bin/env python3
"""Full-train CafeBERT teacher with the frozen fold recipe; validation only."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts' / 'cafe_oof_kd')]
import common as C
from scripts.sealion_benchmark import common as SB
from scripts.sealion_benchmark import train as SBT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=('vsfc_sentiment', 'absa_restaurant'), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', type=int, default=0)
    p.add_argument('--soft-out', type=Path, default=None)
    a = p.parse_args()
    base = SB.task_contract(a.task)
    contract = dict(base)
    SB.task_contract = lambda task_id, include_test=False: contract
    meta = SBT.train_one('cafebert', a.task, 2e-5, device_name='cuda:0',
                         micro_batch_size=32, smoke=a.smoke, out_dir=a.out)
    if meta['status'] != 'completed':
        raise RuntimeError(f"Teacher checks failed: {meta['postrun_checks']}")
    tokenizer, _ = SB.load_cafe_tokenizer()
    model, _ = SB.build_model('cafebert', contract)
    SB.load_model_checkpoint(a.out / 'best_checkpoint.pt', model)
    device = torch.device('cuda:0')
    model.to(device).eval()
    split = contract['train']
    bundle = contract['bundle']
    loader = C.e1_train._make_loader(tokenizer, split, max_length=SB.MAX_LENGTH,
                                     batch_size=64, shuffle=False, seed=42,
                                     labels=bundle.labels_for_model('train'), gold_labels=split.labels)
    ids, values = [], []
    with torch.inference_mode():
        for batch in loader:
            x = C.e1_train._move_batch(batch, device)
            with C.e1_train.make_autocast('bf16', device):
                logits = model(x['input_ids'], x['attention_mask'], token_type_ids=x.get('token_type_ids'))
            values.append(C.flat_logits(logits.float(), len(batch['ids'])).cpu().numpy())
            ids.extend(batch['ids'])
    if list(map(str, ids)) != list(map(str, split.ids)):
        raise AssertionError('Teacher train ID order differs')
    target = a.soft_out or ROOT / 'artifacts/fairness/soft_insample' / f'{a.task}.npz'
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, ids=np.asarray(ids), logits=np.concatenate(values),
                        aspect_class_counts=np.asarray(C.label_widths(bundle)))
    (a.out / 'best_checkpoint.pt').unlink()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
