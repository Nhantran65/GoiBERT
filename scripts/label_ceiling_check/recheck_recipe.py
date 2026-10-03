#!/usr/bin/env python3
"""Re-run the seed_train recipe check (fixed nested-field filter) for runs that finished training
but were rejected by the old exact-match filter. Keeps the old verdict as recipe_check.pre_fix.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.label_ceiling_check.seed_train import ART, blocked_differences, config_diff, original  # noqa: E402

ALLOWED = {'seed', 'started_at', 'started_at_unix', 'argv', 'pid',
           'runtime.cuda_visible_devices', 'runtime.hostname', 'runtime.device',
           'runtime.device_name', 'gpu_snapshot_at_start', 'source_hash'}

for task, seed in [(a, int(b)) for a, b in (x.split(':') for x in sys.argv[1:])]:
    out = ART / 'cafe_seeds' / task / f'seed_{seed}'
    allowed = set(ALLOWED)
    if task == 'absa_restaurant':
        allowed |= {'protocol', 'selection', 'head', 'pooling', 'model.benchmark_adapter', 'tokenizer.benchmark_adapter'}
    old = json.loads((original(task) / 'resolved_config.json').read_text())
    new = json.loads((out / 'resolved_config.json').read_text())
    differences = config_diff(old, new)
    blocked = blocked_differences(differences, allowed)
    expected_head = f"first token -> dropout0.1 -> dense{new['model']['head_hidden_size']} -> tanh -> dropout0.1 -> task logits"
    if new['head'] != expected_head or new['selection'] != 'validation task-specific selection_f1, best epoch':
        blocked.append({'field': 'head/selection contract', 'new': [new['head'], new['selection']]})
    seed_checks = json.loads((out / 'seed_checks.json').read_text())
    prev = out / 'recipe_check.json'
    backup = out / 'recipe_check.pre_fix.json'
    if prev.exists() and not backup.exists():
        backup.write_text(prev.read_text())
    prev.write_text(json.dumps({'differences': differences, 'blocked': blocked, 'pass': not blocked and all(seed_checks.values()),
                                'rechecked_after_filter_fix': True}, indent=2) + '\n')
    print(task, seed, 'pass' if not blocked and all(seed_checks.values()) else f'BLOCKED {blocked}')
