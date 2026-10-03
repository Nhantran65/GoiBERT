#!/usr/bin/env python3
"""Render the final Vietnamese report from frozen JSON artifacts."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/fairness'
TEST=ART/'test'
DOC=ROOT/'docs/FAIRNESS_ABLATION_RESULT.md'


def read(p):return json.loads(p.read_text())


def main():
    parity=read(ART/'parity_results.json')
    forecast=read(ART/'forecast.json')
    decisions=read(ART/'decisions.json')
    validation=read(ART/'validation_summary.json')
    confidence=read(ART/'confidence.json')
    launcher=read(ART/'launcher_summary.json')
    manifest=read(TEST/'locked_manifest.json')
    access=read(TEST/'test_access_log.json')
    table=read(TEST/'summary.json')
    bootstrap=read(TEST/'bootstrap.json')
    if not validation['complete'] or launcher['status']!='validation_complete':
        raise RuntimeError('Validation is incomplete')
    if any(set(validation['insample_comparison'].get(task,{}))!={'oof','insample'}
           for task in ('vsfc_sentiment','absa_restaurant','vinli')):
        raise RuntimeError('#3 validation comparison is incomplete')
    if set(confidence)!={'vsfc_sentiment','absa_restaurant','vinli'}:
        raise RuntimeError('#3 confidence statistics are incomplete')
    if manifest['checkpoint_count']!=96 or not manifest['validation_passed']:
        raise RuntimeError('Manifest gate incomplete')
    if any(e['status']!='completed' for e in access['entries']) or len(access['entries'])!=96:
        raise RuntimeError('Test access log incomplete')
    lines=['# Kết quả ablation phương pháp GoiBERT', '',
           '## Protocol và kiểm tra trước chạy', '',
           'Thiết kế, task, seed, LR và thứ tự test theo `docs/FAIRNESS_ABLATION_PLAN.md` ngày 2026-09-30. '+
           'Mỗi arm mới dùng 30 epoch, batch 32 và bf16; chọn LR/epoch trên validation.', '',
           '| Kiểm tra parity | PASS | Δ best F1 (điểm) | max Δ F1 epoch 1–5 (thang 0–1) | max Δ train loss epoch 1–5 |',
           '|---|---:|---:|---:|---:|']
    for name,v in parity.items():
        lines.append(f"| {name} | {v['passed']} | {v['best_f1_point_delta']} | {max(v['f1_abs_differences'][:5])} | {v['train_loss_first_five_max_abs']} |")
    lines += ['', 'Đường cong 30 epoch: `artifacts/fairness/parity_curves.csv`; sai khác từng epoch: `artifacts/fairness/parity_results.json`.',
              'Optimizer Bami E7: tất cả tham số nằm đúng một nhóm; LR embedding, lớp 0, lớp 11 và head '+
              'đã đối chiếu trong `artifacts/fairness/optimizer_contract.json`.', '',
              '| Smoke | Trạng thái | Giây | VRAM cấp phát đỉnh (byte) |',
              '|---|---|---:|---:|']
    for name in ('bami_sentence','bami_absa','bami_ner_vram','m_ce','s_prime_ce','s_prime_ce_new',
                 'teacher_old','teacher_absa','teacher_vinli','student_insample','student_insample_vinli'):
        meta=read(ART/'smoke'/name/'metadata.json')
        lines.append(f"| {name} | {meta['status']} | {meta['elapsed_seconds']} | {meta.get('peak_gpu_memory_allocated_bytes','—')} |")
    lines += ['',f"Dự báo trước chạy: {forecast['total_gpu_hours']} GPU-giờ; trần 65 GPU-giờ. "+
              f"GPU-giờ preflight và validation thực tế: {launcher['used_gpu_hours']}. "+
              f"GPU-giờ test: {read(TEST/'test_results.json')['elapsed_seconds']/3600}. "+
              f"Tổng: {launcher['used_gpu_hours']+read(TEST/'test_results.json')['elapsed_seconds']/3600} GPU-giờ.", '',
              '## Chọn LR Bami E7 trên validation', '',
              '| Arm/task | LR thử và F1 seed 42 | Best epoch seed 42 | LR chọn |', '|---|---|---|---:|']
    for key,v in decisions['bami_lr'].items():
        lines.append(f"| {key} | {json.dumps(v['candidates'],ensure_ascii=False)} | {json.dumps(v['candidate_best_epochs'],ensure_ascii=False)} | {v['chosen_lr']} |")
    lines += ['', '## Validation: 8 dòng recipe × 15 chỉ số', '',
              'Các ô là mean [min, max] qua seed 42/43/44, đơn vị điểm phần trăm. '+
              'CafeBERT là trung bình 5 fold trên validation để tham khảo.', '',
              '| Recipe | ' + ' | '.join(table['metric_order']) + ' |',
              '|---|'+'---:|'*15]
    for model,values in validation['validation_table'].items():
        cells=[]
        for metric in table['metric_order']:
            v=values.get(metric)
            cells.append(f"{v['mean']} [{v['range'][0]}, {v['range'][1]}]" if v else '—')
        lines.append('| '+model+' | '+' | '.join(cells)+' |')
    valmeans={model:sum(v['mean'] for v in values.values())/15
              for model,values in validation['validation_table'].items() if len(values)==15}
    lines += ['', 'Chênh lệch trung bình 15 chỉ số trên validation (điểm): '
              f"S′-KD − S′-CE = {valmeans['s_prime_kd']-valmeans['s_prime_ce']}; "
              f"M-KD − M-CE = {valmeans['m_kd']-valmeans['m_ce']}; "
              f"Bami-E7-KD − Bami-E7-CE = {valmeans['bami_e7_kd']-valmeans['bami_e7_ce']}; "
              f"Bami-E7-CE − Bami-E14 = {valmeans['bami_e7_ce']-valmeans['bami_e14']}."]
    lines += ['', '## #3: OOF so với in-sample, chỉ trên validation', '',
              '| Task | Chỉ số | OOF mean [min, max] | In-sample mean [min, max] |',
              '|---|---|---:|---:|']
    for task,v in validation['insample_comparison'].items():
        for metric in v.get('oof',{}):
            a=v['oof'][metric];b=v['insample'][metric]
            lines.append(f"| {task} | {metric} | {a['mean']} [{a['range'][0]}, {a['range'][1]}] | {b['mean']} [{b['range'][0]}, {b['range'][1]}] |")
    lines += ['', 'Độ tự tin của nhãn mềm tính softmax ở nhiệt độ 1, thống kê theo slot:', '',
              '| Task | Nguồn | Entropy trung bình (nat) | P(max > 0,99) | P(argmax khác gold) |',
              '|---|---|---:|---:|---:|']
    for task,arms in confidence.items():
        for source,v in arms.items():
            lines.append(f"| {task} | {source} | {v['entropy_mean_nats']} | {v['max_probability_gt_0_99_fraction']} | {v['argmax_disagrees_with_gold_fraction']} |")
    lines += ['', '## Khóa manifest và test một lần', '',
              f"Manifest gồm {manifest['checkpoint_count']} checkpoint, khóa lúc {manifest['locked_at']}; "+
              f"sha256 manifest trong access log: `{access['locked_manifest_sha256']}`. "+
              f"Truy cập test đầu tiên (hash file) lúc {access['test_file_hash_started_at']}; "+
              'toàn bộ 96 checkpoint đã tái tính validation từ checkpoint với tolerance 1e-6 trước khi khóa manifest. '+
              'Các prediction đã lưu được dùng cho mọi phân tích tiếp theo.', '',
              '## Test: 7 dòng × 15 chỉ số', '',
              'Các ô là mean [min, max] qua ba seed, đơn vị điểm phần trăm.', '',
              '| Recipe | ' + ' | '.join(table['metric_order']) + ' | Trung bình 15 | Trung bình 13 bỏ ViNLI | Tham số |',
              '|---|'+'---:|'*18]
    for model,values in table['table'].items():
        cells=[f"{values[m]['mean']} [{values[m]['range'][0]}, {values[m]['range'][1]}]" for m in table['metric_order']]
        params=values['parameters']
        lines.append('| '+model+' | '+' | '.join(cells)+f" | {values['mean_15']} | {values['mean_13_without_vinli']} | {params['min']}–{params['max']} |")
    testmeans={model:v['mean_15'] for model,v in table['table'].items()}
    lines += ['', 'Chênh lệch trung bình 15 chỉ số trên test (điểm): '
              f"S′-KD − S′-CE = {testmeans['s_prime_kd']-testmeans['s_prime_ce']}; "
              f"M-KD − M-CE = {testmeans['m_kd']-testmeans['m_ce']}; "
              f"Bami-E7-KD − Bami-E7-CE = {testmeans['bami_e7_kd']-testmeans['bami_e7_ce']}; "
              f"Bami-E7-CE − Bami-E14 = {testmeans['bami_e7_ce']-testmeans['bami_e14']}."]
    lines += ['', '## Paired bootstrap', '',
              f"{bootstrap['replicates']} mẫu, seed {bootstrap['seed']}; lấy lại theo câu và lấy trung bình ba seed trong mỗi mẫu. "+
              'Δ là hiệu điểm phần trăm, CI 95% percentile; p-value hai phía theo tỷ lệ mẫu vượt qua 0 với hiệu chỉnh +1.', '',
              '| Cặp (A − B) | Chỉ số | Δ | CI 95% | p-value |',
              '|---|---|---:|---:|---:|']
    for pair,values in bootstrap['metric_results'].items():
        for metric,v in values.items():
            lines.append(f"| {pair} | {metric} | {v['delta']} | [{v['ci_95'][0]}, {v['ci_95'][1]}] | {v['p_value_two_sided']} |")
        p=bootstrap['pairs'][pair]
        lines.append(f"| **{pair}** | **Δ > 0 và CI không chứa 0** | **{p['positive_ci_excludes_zero']}/{p['total_metrics']}** | **{p['fraction']}** | — |")
    lines += ['', '## Diễn giải', '',
              f"Trên test, trung bình 15 chỉ số của Bami-E7-KD là {testmeans['bami_e7_kd']}, "
              f"M-KD là {testmeans['m_kd']}, Bami-E14 là {testmeans['bami_e14']}. "
              'Paired bootstrap cho thấy KD cải thiện so với CE của cùng student ở '
              f"{bootstrap['pairs']['m_kd - m_ce']['positive_ci_excludes_zero']}/15 chỉ số M và "
              f"{bootstrap['pairs']['s_prime_kd - s_prime_ce']['positive_ci_excludes_zero']}/15 chỉ số S′ "
              'với Δ dương và CI 95% không chứa 0.', '',
              'Ở #3, OOF cao hơn in-sample ở cả sáu chỉ số validation của ba task; '
              'logits in-sample có entropy thấp hơn và tỷ lệ max-prob > 0,99 cao hơn. '
              'Đây là bằng chứng validation cho lựa chọn OOF trong ba task đã đo; #3 không có test.', '']
    lines += ['', '## Sự cố và giới hạn', '',
              'Lần smoke PhoNER đầu lỗi thiếu tham số `fold`; code mới đã sửa, '+
              'run dở chuyển vào `artifacts/fairness/aborted/` và smoke chạy lại PASS. '+
              'Run parity Bami `vsfc_topic` đầu dùng nhầm micro batch 32; đã lưu trong `aborted/`, '+
              'sửa chế độ E14 về micro batch 16 với accumulation 2 và chạy lại. '+
              'Lần chạy thử `summarize.py` đầu vướng metadata E14 cũ không có `postrun_checks`; '+
              'đã sửa cách đọc baseline và chạy lại PASS. '+
              'Các lỗi khác, nếu có, được ghi trong `artifacts/fairness/launcher_summary.json` và `artifacts/fairness/logs/`.', '',
              'LR #1 giữ nguyên arm KD; Bami E7 chỉ quét hai LR; #3 chỉ gồm ba task và không test. '+
              'So sánh Bami E7 với E14 gồm cả thay optimizer và chọn LR theo lưới hai giá trị trên seed 42. '+
              'Test cũ đã được dùng ở hai lượt trước, các model mới chỉ test một lần trong lượt này.', '',
              '## Lệnh tái lập', '',
              '```bash',
              'PYTHONPATH=src .venv-e0-cuda/bin/python scripts/fairness/optimizer_contract.py',
              'PYTHONPATH=src .venv-e0-cuda/bin/python scripts/fairness/parity.py',
              'PYTHONPATH=src .venv-e0-cuda/bin/python scripts/fairness/forecast.py',
              'setsid nohup env PYTHONPATH=src HF_HUB_OFFLINE=1 FAIR_GPUS=0,1,2,3 .venv-e0-cuda/bin/python scripts/fairness/launcher.py > artifacts/fairness/launcher.log 2>&1 < /dev/null &',
              'PYTHONPATH=src .venv-e0-cuda/bin/python scripts/fairness/confidence.py',
              'PYTHONPATH=src .venv-e0-cuda/bin/python scripts/fairness/summarize.py',
              'PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 .venv-e0-cuda/bin/python scripts/fairness/test_once.py',
              'PYTHONPATH=src .venv-e0-cuda/bin/python scripts/fairness/test_summary.py',
              'PYTHONPATH=src .venv-e0-cuda/bin/python scripts/fairness/bootstrap.py',
              'PYTHONPATH=src .venv-e0-cuda/bin/python scripts/fairness/render_report.py',
              '```', '']
    DOC.write_text('\n'.join(lines))
    created=[p.relative_to(ROOT).as_posix() for base in (ROOT/'scripts/fairness',ART)
             for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    created.append(DOC.relative_to(ROOT).as_posix())
    created.append((ART/'files_created.txt').relative_to(ROOT).as_posix())
    (ART/'files_created.txt').write_text('\n'.join(sorted(set(created)))+'\n')
    print(DOC)


if __name__=='__main__':main()
