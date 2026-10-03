#!/usr/bin/env python3
"""Cafe seed override for the frozen SEA-LION runner; validation only."""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.sealion_benchmark import common, train  # noqa: E402

ART = ROOT / 'artifacts/label_ceiling_check'

def original(task):
    return (ROOT / 'artifacts/cdbk_pilot/runs/T_absa_restaurant_seed42' if task == 'absa_restaurant'
            else ROOT / 'artifacts/sealion_benchmark/cafe_runs' / task / 'lr_2e-05/seed_42')

def config_diff(a, b, prefix=''):
    result = []
    for key in sorted(set(a) | set(b)):
        path = f'{prefix}.{key}' if prefix else key
        if key not in a or key not in b:
            result.append({'field': path, 'old': a.get(key), 'new': b.get(key)})
        elif isinstance(a[key], dict) and isinstance(b[key], dict):
            result.extend(config_diff(a[key], b[key], path))
        elif a[key] != b[key]:
            result.append({'field': path, 'old': a[key], 'new': b[key]})
    return result

def blocked_differences(differences, allowed):
    # Allowed fields also cover their nested paths (e.g. gpu_snapshot_at_start.0.free_mib).
    # The tokenizer snapshot dir later gained XLM-R-large weights (model.safetensors) from the
    # xlmr_large_baseline download; that file is not a tokenizer file, so an added entry is benign.
    def ok(d):
        field = d['field']
        if any(field == a or field.startswith(a + '.') for a in allowed):
            return True
        if field == 'tokenizer.files.model.safetensors' and d['old'] is None:
            return True
        # Restaurant seed 42 came from cdbk_pilot (arm T = plain CafeBERT, no KD); the sealion runner
        # records the same model as model_kind=cafebert. Accept exactly these schema-equivalent values.
        return (field, d['old'], d['new']) in [('arm', 'T', None), ('init', 'cafebert', None),
                                                ('model_kind', None, 'cafebert')] or \
            (field == 'kd' and d['old'] == {'enabled': False} and d['new'] is None)
    return [d for d in differences if not ok(d)]

def dry_config(task, seed, out):
    # Frozen seed-42 resolved config supplies model/tokenizer/data fingerprints.
    # This path neither builds a model nor calls CUDA.
    base = json.loads((original(task) / 'resolved_config.json').read_text())
    config = copy.deepcopy(base)
    config['seed'] = seed
    config['protocol'] = common.PROTOCOL
    config['selection'] = 'validation task-specific selection_f1, best epoch'
    config['pooling'] = 'first token'
    config['head'] = f"first token -> dropout0.1 -> dense{config['model']['head_hidden_size']} -> tanh -> dropout0.1 -> task logits"
    config['model']['benchmark_adapter'] = 'scripts/cdbk_pilot/common.py via sealion_benchmark'
    config['tokenizer']['benchmark_adapter'] = 'scripts/cdbk_pilot/common.py via sealion_benchmark'
    config['source_hash'] = common.source_hash()
    config['argv'] = [str(Path(__file__).relative_to(ROOT)), '--task', task, '--seed', str(seed), '--out-dir', str(out)]
    config['dry_run_config'] = True
    config['dry_run_source'] = str(original(task).relative_to(ROOT))
    config['dry_run_note'] = 'Model/tokenizer/data fingerprints and runtime snapshot inherited from seed-42; no model, tokenizer, CUDA or GPU snapshot loaded in dry-run.'
    return config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=common.TASKS, required=True)
    ap.add_argument('--seed', type=int, choices=(43, 44), required=True)
    ap.add_argument('--out-dir', type=Path)
    ap.add_argument('--dry-run-config', action='store_true')
    args = ap.parse_args()
    out = args.out_dir or ART / 'cafe_seeds' / args.task / f'seed_{args.seed}'
    if args.dry_run_config:
        print(json.dumps(dry_config(args.task, args.seed, out), ensure_ascii=False, indent=2))
        return
    common.SEED = args.seed  # used by seed_everything and both DataLoaders
    metadata = train.train_one('cafebert', args.task, 2e-5, device_name='cuda:0', micro_batch_size=32, out_dir=out)
    if metadata['status'] != 'completed' or not all(metadata['postrun_checks'].values()):
        raise RuntimeError('postrun checks failed')
    old = json.loads((original(args.task) / 'resolved_config.json').read_text())
    new = json.loads((out / 'resolved_config.json').read_text())
    allowed = {'seed', 'started_at', 'started_at_unix', 'argv', 'pid',
               'runtime.cuda_visible_devices', 'runtime.hostname', 'runtime.device',
               'runtime.device_name', 'gpu_snapshot_at_start', 'source_hash'}
    if args.task == 'absa_restaurant':
        allowed |= {'protocol', 'selection', 'head', 'pooling',
                    'model.benchmark_adapter', 'tokenizer.benchmark_adapter'}
    differences = config_diff(old, new)
    blocked = blocked_differences(differences, allowed)
    expected_head = f"first token -> dropout0.1 -> dense{new['model']['head_hidden_size']} -> tanh -> dropout0.1 -> task logits"
    if new['head'] != expected_head or new['selection'] != 'validation task-specific selection_f1, best epoch':
        blocked.append({'field': 'head/selection contract', 'new': [new['head'], new['selection']]})
    (out / 'recipe_check.json').write_text(json.dumps({'differences': differences, 'blocked': blocked, 'pass': not blocked}, indent=2) + '\n')
    contract = common.task_contract(args.task, include_test=False)
    val = common.e1_train.split_field(contract['bundle'], 'validation')
    saved = common.read_json(out / 'validation_predictions.json')['rows']
    checks = {'ids_stable_complete': [r['id'] for r in saved] == list(common.e1_train.split_ids(val)),
              'gold_identical': [r['gold'] for r in saved] == json.loads(json.dumps(list(common.e1_train.split_labels(val))))}
    (out / 'seed_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    if blocked or not all(checks.values()):
        raise RuntimeError(f'recipe or ID mismatch: {blocked}, {checks}')

if __name__ == '__main__':
    main()
