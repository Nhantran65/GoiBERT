#!/usr/bin/env python3
"""Conservative GPU-hour forecast from completed smoke and reference runs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts/fairness'
TASKS = ('vsfc_sentiment', 'vsfc_topic', 'visfd', 'absa_restaurant', 'absa_hotel',
         'vispam', 'vinli', 'phoner')
BAMI = {'vsfc_sentiment': 2e-5, 'vsfc_topic': 1e-5, 'visfd': 5e-5,
        'absa_restaurant': 5e-5, 'absa_hotel': 5e-5, 'vispam': 2e-5,
        'vinli': 1e-5, 'phoner': 5e-5}
STUDENT = {'vsfc_sentiment': 1e-5, 'vsfc_topic': 1e-5, 'visfd': 1e-4,
           'absa_restaurant': 1e-4, 'absa_hotel': 1e-4, 'vispam': 5e-5,
           'vinli': 1e-4, 'phoner': 1e-4}


def read(path):
    return json.loads(path.read_text())


def main():
    smoke = {name: read(ART / 'smoke' / name / 'metadata.json') for name in
             ('bami_sentence', 'bami_absa', 'bami_ner_vram', 'm_ce', 's_prime_ce',
              's_prime_ce_new', 'teacher_old', 'teacher_absa', 'teacher_vinli',
              'student_insample', 'student_insample_vinli')}
    assert all(v['status'] == 'completed' and
               all(x for x in v['postrun_checks'].values() if isinstance(x,bool))
               for v in smoke.values())
    bami_seconds = {}
    student_seconds = {}
    for task in TASKS:
        if task in TASKS[:5]:
            ref = ROOT / f'artifacts/e14/trials/{task}/lr_{BAMI[task]:.0e}/seed_42/metadata.json'
        else:
            ref = ROOT / f'artifacts/bench8/runs/{task}/bami/lr_{BAMI[task]:.0e}/seed_42/metadata.json'
        bami_seconds[task] = float(read(ref)['elapsed_seconds'])
        ref_s = ROOT / f'artifacts/family_sm/finetune/s_prime/{task}/lr_{STUDENT[task]:.0e}/seed_42/metadata.json'
        student_seconds[task] = float(read(ref_s)['elapsed_seconds'])
    # E14 is slower than the E7 batch-32 path on the three new tasks; use its
    # measured runtime anyway. A 25% margin also covers concurrent contention.
    bami_hours = 1.25 * 8 * sum(bami_seconds.values()) / 3600
    student_hours = 1.25 * 6 * sum(student_seconds.values()) / 3600
    teacher_hours = 10.0  # plan ceiling for all three teachers and nine student runs
    test_hours = 2.0
    preflight_hours = 2.0  # three parity runs plus smoke and optimizer audits
    total = bami_hours + student_hours + teacher_hours + test_hours + preflight_hours
    result = {'bami_e7_ce_kd_gpu_hours': bami_hours, 'student_ce_gpu_hours': student_hours,
              'insample_gpu_hours': teacher_hours, 'test_gpu_hours': test_hours,
              'preflight_gpu_hours_reserve': preflight_hours,
              'total_gpu_hours': total, 'limit_gpu_hours': 65,
              'method': 'E14 and S-prime 30-epoch measured runtimes, 25% contention margin; #3 and test use locked plan allowance',
              'smoke_elapsed_seconds': {k: v['elapsed_seconds'] for k, v in smoke.items()},
              'reference_bami_seconds': bami_seconds, 'reference_student_seconds': student_seconds,
              'passed': total <= 65}
    ART.mkdir(parents=True, exist_ok=True)
    (ART / 'forecast.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if total > 65:
        raise SystemExit('STOP: forecast exceeds 65 GPU-hours')


if __name__ == '__main__':
    main()
